## qwen3支持

> 课程项目已经支持的模型列表：qwen2.5,llama3.2, llama3.1, llama2

Qwen-3是由阿里巴巴通义千问团队在2025年4月28日发布的大语言模型，以下是它的优点：

1. 具备思考模式和非思考模式，支持多语言对话，推理能力显著提升。
2. 它采用混合专家架构，优化了推理效率，显存占用低。
3. Qwen-3适用于复杂任务处理、实时交互、多语言对话和智能体开发等多种场景。

基于以上以及大家对qwen3模型的认可，所以我们也是第一时间在课程项目中支持了qwen3大模型，为大家的秋招助力，下面我将整理我支持的过程。

## 模型的下载

https://huggingface.co/fushenshen/lession_model/blob/main/qwen3-0.6b.zip

解压完成后得到：

1. tokenizer.json
2. qwen0.6.bin2

我们将它放到models文件夹中，并重新编译qwen3_infer，在编译之前需要额外安装：

1. https://github.com/nlohmann/json json库
2. https://github.com/abseil/abseil-cpp abseil库

```Shell
mkdir build
cd build 
cmake .. -DQWEN3_SUPPORT=ON 
# 需要额外安装json库和abseil库
make -j16
./qwen3_infer qwen0.6.bin2 tokenizer.json
```

## qwen3模型的修改

在模型结构上，主要是添加了q_norm和k_norm层，**其实这就是RMSNorm层**，它在模型中的位置如下。可以看出无论在prefill还是decode阶段，**它都是在attention模块之前，求query, key, value三个矩阵（向量）之后的一个归一化过程。**知道了该模块的位置，我们再来看看它的输入、输出大小，均为[batch, seq_len, num_head, head_dim]，这几个维度大家应该都很熟悉了，分别是：

1. 输入的批次大小
2. 一个批次中词元的数量
3. 多头自注意力机制的头数
4. 每个头负责的维度空间大小

![img](https://tvle9mq8jh.feishu.cn/space/api/box/stream/download/asynccode/?code=OWY0MTM4YTQ0NjJjM2VkMzNlZmFkMzU3Nzk1NjcyYTdfbTZ1RlRZT0pscGV4OHdEazRVNmhsVTRTQUVJTUwycHRfVG9rZW46Qjd0UWJndjZ0b1A2eEh4aXZ3aWNBT3JkbnlmXzE3NzcyNTA3MTQ6MTc3NzI1NDMxNF9WNA)

## 对RMSNorm Cuda算子的修改

### 实现

有了以上的信息，我们就要考虑是否需要对以往实现的算子进行重写。在我们之前的推理框架中，RMSNorm层和对应的cuda算子已经有了，但是我们以往的输入是只支持一维的，也就是只支持输入为[dim]的情况。而现在的输入变成了[num_head, head_dim]，所以现在就需要独立对4个[num_head]个输入求RMSNorm归一化。

暂时无法在飞书文档外展示此内容

那么我们就要对计算的主要流程进行变化，可以看出这里的num_head个输入是互相独立的，**具体的更新见rmsnorm.cu中的row_rmsnorm_f32_dim**。在上文中我们说过，现在需要独立地对这里的4个输入求解rmsnorm，所以我们可以设置cuda中的block数量等于4，每个block负责一行输入的并行归约计算。

![img](https://tvle9mq8jh.feishu.cn/space/api/box/stream/download/asynccode/?code=MjQyOWZhMzJhMDEzYTA3MDkyYWFiNjgzNGIzMzhkMjNfdGs3dzBlT0VDTFhOSlRPVlVlN0xhRVFOV0paYVBTcEhfVG9rZW46QkFnd2JKWUVzbzhkUVR4R25BN2M3NDJjbnViXzE3NzcyNTA3MTQ6MTc3NzI1NDMxNF9WNA)

为了索引到输入的每一行，有下方的代码，这里的size等于head_dim。

```C++
float* block_in = in + bid * size
float* block_out = out + bid * size;
```

定位到某一行之后的计算流程就和之前课程中讲的保持一致了，大家可以自行复习。**简单来说就是使用同一个block中****所有的thread****，对一行的值进行归约计算(BlockReduce)，计算出需要的sum值和scale值**，多个不同的block就是计算多行。

[第5次课程-RMSNorm算子的CUDA实现](https://l0kzvikuq0w.feishu.cn/docx/BXtyd0xGHoYWFgxrPDkcw2HXnxb) row_rmsnorm_f32

### 启动

```C++
void rmsnorm_kernel_cu_dim(const tensor::Tensor& input, const tensor::Tensor& weight,
                           const tensor::Tensor& output, int32_t dim, void* stream) {
  ...
  ...

  const float eps = 1e-6f;
  const int32_t total_size = static_cast<int32_t>(input.size());
  const int32_t size = input.get_dim(input.dims_size() - 1); // size就是head_dim
  const int32_t dim_size = total_size / size; // 这里的dim-size就是num_head

  float* in_ptr = const_cast<float*>(input.ptr<float>());
  float* wei_ptr = const_cast<float*>(weight.ptr<float>());
  float* out_ptr = const_cast<float*>(output.ptr<float>());
  constexpr int threads_num = 128;
  if (stream) {
    cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
    row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size,
                                                               size, eps);
  } else {
    row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
  }
}
```

我们重点来看其中的第9行，我们计算出两个重要的变量：

1. dim_size：等于cuda block的数量，也就是num_head的数量
2. size：一行的元素数量

### 接入

有了上述的实现，我们就要将它接入到我们的模型forward流程。

```C++
void Qwen3Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  auto [key, val] = slice_kv_cache(layer_idx, pos);

  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));  // rmsnorm_output输入，query是输出

  // query norm
  auto query_norm = qwen_layers_->rmsnorm_layers_.at(layer_idx + 2 * config_->layer_num_ + 1);
  query.reshape({(int32_t)query.size() / config_->head_size_, config_->head_size_});
  query_norm->forward(query, query);
  query.reshape({(int32_t)query.size()});
```

和上图的流程一样，**qnorm的计算在求query矩阵之后，在attention计算之前**，对query矩阵进行一次归一化(q_norm)。作为比较，这是以下是《自制大模型推理框架》课程对qwen2.5的推理流程，可以看出就只是少了一次q_norm的过程。在模型当中添加了一个流程，把qnorm接入到query矩阵计算之后。

```C++
void Qwen2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(qwen_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  // query
  const auto& query_layer = qwen_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
```

## 对qwen3的调用

和之前一样，我们需要写一个demo工程来处理对promt的输入预处理和编码，以及调用qwen3的推理流程。

### 编码器的支持

对编码器我们可以复用qwen2.5, llama3的bpe编码方法，bpe方法见课件[自制大模型推理框架-第20次课程-LLama3.2模型的支持](https://l0kzvikuq0w.feishu.cn/docx/GflIdgyjoo3p7HxbnHEc3bVInde)。我们只要添加对qwen3的编译选项就可以，这里的编译选项就是在CMake编译的时候添加`-DQWEN3_SUPPORT=ON`的时候，推理框架会编译下方代码（也就是BPE编码器）。

```C++
#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT) || defined (QWEN3_SUPPORT)
class BpeEncodeLayer : public EncodeLayerBase {
   ...
   ...
}
```

有了这个编译选项，就可以在main_qwen3.cpp中开启编码为BPE的选项，有如下的代码：

```C++
int main(int argc, char* argv[]) {
  if (argc != 3) {
    LOG(INFO) << "Usage: ./demo checkpoint path tokenizer path";
    return -1;
  }
  const char* checkpoint_path = argv[1];  // e.g. out/model.bin
  const char* tokenizer_path = argv[2];

  model::Qwen3Model model(base::TokenizerType::kEncodeBpe, tokenizer_path, checkpoint_path, false);
  auto init_status = model.init(base::DeviceType::kDeviceCUDA);
  if (!init_status) {
    LOG(FATAL) << "The model init failed, the error code is: " << init_status.get_err_code();
  }
```

model的init函数中会初始化刚才说到的`BpeEncoderLayer`得到qwen3对应的编码器，这点因为在之前的课程中已经有了讲解，所以在这里不再对流程进行重复叙述。BPE是在课程：[自制大模型推理框架-第20次课程-LLama3.2模型的支持](https://l0kzvikuq0w.feishu.cn/docx/GflIdgyjoo3p7HxbnHEc3bVInde) 讲解的。

### 调用模型推理

**假设我们现在有一个输入**，`What is AI ? `，这是我提出的一个问题。先是将该输入prompt填入到一个结构化的模板之中，这是因为Qwen 等对话模型通常要求输入遵循固定的**角色标记格式**，例如：

- 明确区分 `用户输入` 和 `助手回复` 的段落
- 使用特殊符号（如 `<|im_start|>`, `<|im_end|>`）标记对话边界。

不难看出在这里第二个<|im_start>没有给出配套的<|im_end>，这是因为模型会从这个位置开始**继续生成文本**，直到主动输出一个 `<|im_end|>` 表示回复结束。**如果不需要模型有思维的过程，**还可以再模板中加入额外的空白`<think>`标签。

```C++
<|im_start|>user
What is AI? 
<|im_end|>

<|im_start|>assistant
```

以上的输入**（包括用于填充的模板）**随后就会被bpe编码器编码为一串词元序列，这个步骤和[自制大模型推理框架-第20次课程-LLama3.2模型的支持](https://l0kzvikuq0w.feishu.cn/docx/GflIdgyjoo3p7HxbnHEc3bVInde)是同样的，随后就是调用`init`初始化之后的`model`进行推理。

### 带think过程的输出

最后的模型输出如下，可以看到<think>标签内的是它的思考过程，随后的是经过思考后的输出，它对'what is ai?'这个问题给出了一个非常良好且切题的回答。

```Plain
What is AI?
<think>
Okay, the user is asking what AI is. Let me start by explaining the basic definition. AI refers to artificial intelligence, which is a technology that allows machines to perform tasks that typically require human intelligence. I should mention key areas like problem-solving, learning, and decision-making.

I need to make sure the explanation is clear and covers the main points. Also, I should include examples to help illustrate the concept. Maybe mention applications in different fields like healthcare, finance, and education. It's important to highlight that AI is a subset of computer science and has various uses.

Wait, should I add anything about the difference between AI and machine learning? That might be useful. Also, maybe touch on the ethical considerations, but since the user didn't ask about that, perhaps keep it simple. Let me check if there's any technical jargon I should avoid. Keep the language straightforward and accessible.
</think>

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines. It involves creating systems that can perform tasks typically requiring human intelligence, such as learning, problem-solving, reasoning, and decision-making. AI systems can be designed to:

1. **Learn from data**: Using algorithms to analyze patterns and improve over time.
2. **Make decisions**: Solving problems or making choices based on data.
3. **Interact with humans**: Understanding natural language, recognizing patterns, and adapting to user inputs.

AI is a subset of computer science and has applications in various fields, including healthcare, finance, education, and autonomous systems. It's continuously evolving, with advancements in machine learning and deep learning enabling more complex AI capabilities.
```

## 结语

谢谢qwen3团队的贡献！