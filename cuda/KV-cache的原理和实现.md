## KV-cache的原理和实现

**KV Cache已成为当前大型模型推理框架的标准配置**，其主要功能是通过牺牲一定的空间复杂度来换取时间效率的提升，从而加快大型模型的推理预测速度。

由于GPT、LLama等模型采用自回归的方式，逐个步骤进行推理，上一步的预测结果将被纳入下一步的计算中作为输入。假设初始输入序列（input token）的长度（seqlen）为1，其维度为1×dim，在每次自回归的计算过程中，我们都会将前一次预测得到的单词添加到输入序列中。

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=MGMzYjE1ZTc5NTZjNjQ1M2UxMTE5NDA5OTAyMTc3NDRfZXI5a05JenBtZ2JSR0Q1ZnVDeUVOd0IwSmlDbVYwUllfVG9rZW46SHFOZmJ4Uk5tbzhXdjR4bEl0QmNzSmlzbmZoXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

**第1步计算的时候**

​       “<s>”->编码-->32，**32去查一个词表**，词表对应32位置记录了一个向量，得到开始词对应的向量，它的维度是1×dim。

**当步长等于1的时候，我们有一个维度为*****1×dim*****的输入Token****，1是输入token的个数**，通过矩阵wq和矩阵wk（**wq和wk矩阵的维度均为*****dim×dim***）将输入序列(input token)映射得到Q和K矩阵，随后就是将Q矩阵和K^T进行矩阵相乘得到V矩阵，在步长等于1的时候Q的维度为1×dim，K的维度同样为1×dim，**当Q矩阵乘以K的转置时得到一个1x1的分数矩阵**，随后该分数矩阵再对1×dim的V矩阵进行加权，得到最终的注意力输出。

input token 矩阵乘 wv矩阵维度同样是dim×dim，是固有的权重，input token×Wv矩阵 = V矩阵。

$$Att_1(Q,K,V)=softmax(Q_1K_1^{T})V_1$$

**第2步计算的时候**

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTc1MjUyOTcxZDg5NzE0ZjgzMjNkM2VhZjEwY2NiNzVfSVJKRDNIS3Z2UnQyUWIyaFJZdWlZZzJ4bUlyMkFMc1ZfVG9rZW46UUR0d2J3QzVYb0lSb2l4Q0c2UWNBM2dMbmFkXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

**在第二步的计算过程中，我们处理了两个输入Token，维度为2×dim**。根据上一步的输入，得到的预测单词为"遥"。

因此，当这些Token与Wq和Wk矩阵相乘进行映射时，我们得到的Q和K矩阵的维度为2×dim。**2×dim，Wq和Wk权重矩阵同样为dim×dim大小，v矩阵也是2×dim。**但是第2步在Q矩阵与K^T矩阵相乘时，我们必须注意一个关键点：**Q矩阵的第1行与K^T矩阵的第2列是不会相乘的**，这一机制就是Transformer Decoder结构中的Causal Mask机制。

该机制的目的在于，在计算注意力(Attention)的过程中，将这些Token从注意力机制中屏蔽掉，确保模型在预测时仅能关注过去和当前的token，从而使得模型基于每个时间步骤可用的信息进行预测。对于Q矩阵的第一行而言，K矩阵的第二、三列代表的就是未来的数据。

**第3步计算的时候**

第三步计算同理，如下图所示：**我们处理3个输入Token，维度为3×dim 乘以 dim×dim的wq和wk权重矩阵分别得到Q和K，第三个Token来自于第2步中的预测。**

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2QzNzY4ZjQ0ZTYxZjk5MTFhNzAwOTAzYjhkZjBjMzRfanVPSXNHV3NVOEJLQ0FCTHhSUEU1ZGdteXRYS2JGWW9fVG9rZW46TDhxdGJCTmwzb3Jnbnh4WU1hemNmTTZVbkZnXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

从图中就可以看到，我们在这3步中积攒了大量的重复计算（第一、第二行），这些信息明明在前两步中已经被计算过了，却还是要在第3步中做重复的计算。

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=MTU2MDE1N2JmNWYxZTU5ZTNhYmYxNTIwYzEwZDQyYjhfamZkeTRUem92MTlkbnBQRThuZHdqM2l3Y3hnU3JDV2pfVG9rZW46QnJEa2J2ZFVIb0t4VEd4Vlh0VGNJQjNCbndmXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

因此，在第k次计算时，输入序列的长度将变为k，维度则为k×dim。这表明，随着计算次数的增加，输入序列的维度会持续增长，从而可能引起重复计算，为了剖析计算过程中的这种冗余，**我们可以将原本k×dim维度的Query矩阵拆分为两部分：**

1. 第一部分是包含第0行至第k-1行的Query1矩阵，其维度为(k-1)×dim；
2. **第二部分是仅包含第k行的Query2矩阵，维度为1×dim**。在执行第k次自回归计算时，我们只需要关注Query矩阵的第k行(也就是Query2矩阵)和Key矩阵第0到k列进行矩阵相乘就可以。

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=ODRjZTFjN2UyZjcxNGNiZTdlZmE0MWI2MTgxZWQzZmFfMzV5dlBVQWxCWXNvUmJVUm1ha0lJeUxSODV0UDdIVU9fVG9rZW46SmZFSWIyNGdRb1BWd3Z4cEdqVGNpSHVFbnhmXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

## **简化Attention的计算**

在上一段中我们说过了，在执行第k步自回归计算时，我们只需要关注**Query2**矩阵也就是Query矩阵的最后一

行，和Key矩阵所有列相乘得到结果就可以了。第一步当中，q1是空的，q2是第一行。第二步当中，q2是第2行，q1是第一行。

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=Y2M2YzA1YmMyMjMwNzIyN2UzMDk4ODU0OTIzNTg2ZGNfcFpaTzRLVU5YSDc2d1VsODg2VkJtdG1taGE5cXR1dnhfVG9rZW46Q2VraWJWYUFubzdIUjN4R1o4WmNyTWozbjRkXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

## KV-Cache

### K-Cache

**KVCache顾名思义就是缓存一部分K矩阵和V矩阵**，就像我们上文所说的K**矩阵同样可以分为Key1矩阵和Key2矩阵**，Key1矩阵是在前k-1步计算中得到的，Key2是在当前步骤中得到的。我们看图：

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=ODU5NmQzYzhlODIwOGI4ZjUyMmU0NzA2YTRmMjFhMGJfaU11bDQxTkNoSGtWUTVyeDRXekJpUWM4UjVWQ01ZMFpfVG9rZW46WWw3NGJtRlZub3l3R2h4TVZRYmNGMVJnbkpjXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

下一步的Key矩阵中的前k-1列和上一步中Key矩阵的前k'列是相同的，**例如在第3步中Key矩阵的第1和第2两列和第2步中的Key矩阵中是相同的。为什么会相同呢，在第2步中Key矩阵是这么得到的：**

[ input token 1, input token 2] × Wk(dim×dim的权重矩阵) = K矩阵

**而在第3步中则有：**

[ input token 1, input token 2, **input token 3 当前步骤的新词**] × Wk = K矩阵

因此第3步中得到的Key矩阵的前两列和第2步的K矩阵是相同的，既然是相同的，我们为什么不在计算Key矩阵的时候保存前k-1列，在第k步的时候只计算第k列（当前的input token×Wk矩阵）随后再拼接成一个完整的Key矩阵。这种记录Key矩阵前k-1列的方法就是所谓的KV-Cache。

### 如何改良

**原来的步骤**--[ input token 1, input token 2, **input token 3 当前步骤的新词**] × Wk = K矩阵

改良后

**K1列，K2列缓存下来**

现有的步骤**input token 3 当前步骤的新词 × Wk 得到K3 列**

**在计算的时候需要完整的K矩阵拼接起来，**

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=MTRhMThlODg2ZjRjYTJiMTI2OGM0MWExYTBjYzRlYzVfSGlYd2hxakNBSDFEQVhleXIzWHpWNW84SDhCY3M2VGRfVG9rZW46SGVubGI1dWRIb3RsQVR4M3VKVWNTUUIwbjVlXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

### V-Cache

对于V矩阵同样的，V矩阵也是由输入的多个token和Wv权重矩阵相乘得到的，在第k步和第k-1步得到的矩阵Value1和Value2有一部分是重叠的：

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=NDE1ZjczZmYyZGQ5ZmZjYjRlMTAxYTQxNTNiYTFlZTBfMkNxNjd5TkFIeU9QWDQ5M1ZxZHJodGhhSWJrSVRXbDBfVG9rZW46WFlJTmJwWm5ib2x3dnJ4WDFGYWNJVzhzbldiXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

所以我们需要将前一步计算出来的Value1矩阵放在一块Value-Cache区域中，**等到当前步的时候我们只需要将当前的输入token和Wv矩阵进行相乘得到Value2矩阵，再将它们拼接起来就可以得到完整的Value矩阵并开始注意力的计算。**

原来的步骤：[ input token 1, input token 2, **input token 3 当前步骤的新词**] × Wv = V矩阵

现有的步骤：V1行和V2行都被缓存了，**input token 3 × Wv得到V3行，****然后再拼接起来得到V矩阵****。**

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=Mzc0NTAyYTczNjRmOGVjMDVkNTQzYTJhZmNkMzQ1YzJfbldXWUVzY2NFR2ZGQkdoc0xMMDcwTFVzanp6TUJVa3BfVG9rZW46VllqQ2JWSlpRb1dEUGl4a0pTY2NKRTB2bkFlXzE3NzcyNTAyNDY6MTc3NzI1Mzg0Nl9WNA)

## 为KV-Cache申请空间

```C++
// kv cache
tensor::Tensor key_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_, config_->kv_dim_, true, alloc);

tensor::Tensor value_cache(base::DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,  config_->kv_dim_, true, alloc);
```

从上文的分析中，我们可以得知，为了给执行了**k**步计算的Transformer块准备KV Cache的空间，我们所需申请的空间大小应为k×dim。如果整个模型包含了**N**个这样的Transformer块，那么总共需要申请的显存空间将是**N**×**k**×**dim×sizeof(float)**。

暂时无法在飞书文档外展示此内容

由于在每一步自回归预测中，我们都会增加一个输入单词，因此单词的总数最多不会超过最大序列长度（max_seq_len）。因此，所需申请的显存空间大小可以表示为max_seq_len×N×dim×sizeof(float)，所以我们在**LLama2Model::init_mem()** 方法中申请了这块空间用于后续使用。

## 对KV-Cache空间的拆分

```C++
std::pair<tensor::Tensor, tensor::Tensor> 
    LLama2Model::slice_kv_cache(int32_t layer_idx,int32_t token_pos) const {
    // (N,max_selen,dim)
    // 索引到第几个transformer块,layer_idx个transformer块，第token_pos个位置的
    int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
    int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;
    简单来说，对于第layer_idx个transformer块和第token pos步，对应的存放位置为cache_offet
    它的索引就是(layer_index, token pos, :) kv_cache[layer_index,token_pos,:]
  
  
    // 把原指针做一个封装
    float* key_cache_ptr =
        const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
    float* val_cache_ptr =
        const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

    auto key_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr,key_cache_ptr, true);
    auto val_cache = std::make_shared<base::Buffer>(config_->kv_dim_ * sizeof(float), nullptr, val_cache_ptr, true);
    
   
    
    key_cache->set_device_type(device_type_);
    val_cache->set_device_type(device_type_);
    tensor::Tensor key(base::DataType::kDataTypeFp32, config_->kv_dim_);
    tensor::Tensor val(base::DataType::kDataTypeFp32, config_->kv_dim_);
    key.assign(key_cache);
    val.assign(val_cache);
    return {key, val}; // 表示的是
}
```

在上一节中，我们已经了解到KV-Cache的总空间大小为max_seqlen×layer_num×dim。因此，在第token_pos步时，我们可以通过索引位置(token_pos, layer_idx)来获取第layer_idx层的KV-Cache，具体代码见第3至第4行。接下来，我们将这个索引位置对应的区域封装为两个Tensor，并将这两个Tensor指定特定类型后返回，以便用于存储缓存数据。

![image-20260427083937625](https://raw.githubusercontent.com/wangbanjin1/pictures/main/image-20260427083937625.png)  

### 举例来说

Inputs = "我正在上课"

[32,64,95,77,123]

[[512维][512维] ]

Inputs = "我"

Token id = 32

通过32在词表中查到了一个512维度的向量

Input embeddings来表示

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=NDkzOWQ3NzMxNWU5ZTIyNDVkZDUwYjdiNTMyZWJlODZfTjNKeDc4VTVXeWRZRjA4RXFTeWJCQTVUU0UwbHhBUkFfVG9rZW46QVBKb2JleWVIb3VuaVB4ZnlmSmM1cjhZbndmXzE3NzcyNTAzMDc6MTc3NzI1MzkwN19WNA)

首先，让我们看一下LLama模型的架构图。该模型的输入可以是单段或多段文本。通过Tokenizer工具，例如我们课程项目中使用的sentencepiece，这些文本将被转换成一个词表索引序列。接下来，我们使用嵌入词表，将这些索引序列转换为一组输入向量，我们称之为输入嵌入（input embedding）。然后，我们对这些输入嵌入（input embedding）执行rmsnorm操作。

在此之后，将经过rmsnorm处理的输入嵌入分别与wq、wk和wv这三个矩阵进行矩阵乘法运算，从而得到Q、K、V三个矩阵。我们回归一下上节课中讲到了什么，在上节课中我们讲到了Attention的计算方式以及KV Cache为什么能节省内存，在本节课的一开始，我们先来回顾一下这两点主要内容：

Rmsnorm output @ wq  = Q

Rmsnorm output @ wk = K

Rmsnorm output @ wv = V

softmax(Q@K.T)@V  

## Attention的计算回顾

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjU0Mjc4MzYyN2MzNTAzYzBmOGM1ZDZlNzg3MzI2ZmFfUVJsU241b3huRTdLR0hkMG1aM2V4OUFsRGVMdjFvVnRfVG9rZW46THR5aGJpT2Fvb1BTcm14dzBtWWNESENubkFjXzE3NzcyNTAzMDc6MTc3NzI1MzkwN19WNA)

让我们以第一步的计算为例。在第一步中，特殊符号被编码为数字32。我们使用这个数字作为索引，在词表中查找对应的项。**在词表的第32个位置，记录了一个向量，这个向量代表了开始词的嵌入，其维度为1×dim**。当步长为1时，我们有一个维度为1×dim的input embedding。这里的1表示输入Token的数量，通过将input embedding与wq和wk矩阵（这两个矩阵的维度均为dim×dim）相乘，分别得到Q和K矩阵。接下来，我们将Q矩阵与K的转置矩阵相乘得到一个注意力权重矩阵。

在步长为1的情况下，Q矩阵的维度为1×dim，K矩阵的维度也是1×dim。当Q矩阵乘以K的转置时，我们得到一个1×1的权重矩阵。然后，这个权重矩阵用于对1×dim的V矩阵进行加权，从而得到最终的注意力输出。同时，input embedding与wv矩阵（维度同样为dim×dim）的乘积结果就是V矩阵。

$$Att_1(Q,K,V)=softmax(Q_1K_1^{T})V_1$$

这里就引申出了三个问题？

1. 怎么实现input embedding乘以wq、wk和wv矩阵；
2. wq、wk和wv矩阵怎么从权重模型文件中提取得到；
3. input embedding和wk 矩阵的乘积k，怎么放入到kv cache中。

目的是求出Rmsnorm output @ wk = K

但是实际上K的前N-1列已经被缓存下来了，我们当前步需要计算的只是第N列。

## 问题1

> 怎么实现input embedding矩阵乘wq、wk和wv矩阵?

在attention_qkv函数中，我们首先是通过slice_kv_cache函数切分出本步骤中需要切分得到下图中的k和v张量用来存放新一个步骤中的query和value矩阵，上节课已经讲过了这里的步骤，请同学们自行回顾。

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=MWQ5OWZkMWMyMWM2MjUwMDk3NDUzYWVkYjZiOTlmYmJfNHJIelNxa1FPNTBrSnBFR3U3QUswVkkwQzZDbm1KMjhfVG9rZW46S0RzNmI3UzhKb1YzTEZ4V2ltRmNVT3BQbkNmXzE3NzcyNTAzMDc6MTc3NzI1MzkwN19WNA)

另外我们从下图中就可以看出wq@input和wk@input中的输入存放于RMSNorm的输出之中，

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=MmFkZmVjNjUwNmRkNDM4MjNjZGVhNWIyNzZkNGJkZWVfazd4bngxQTd0ckw3aW52VU1pN0FIMTVYTnhMaDhtdGhfVG9rZW46SWhZamJCb1pvb2poZ3d4b0E0dWN6a0pSbnFoXzE3NzcyNTAzMDc6MTc3NzI1MzkwN19WNA)

因此，在代码的第12行至第15行，我们看到了`query_layer->forward(rmsnorm_output, query)`的调用。在这个算子调用过程中，我们将输入的嵌入向量（input embedding）与query layer自身的权重相乘并将计算结果存储在`query`变量中。同理，我们也采用了相同的方法来获取键（key）矩阵。最后，在第22行至第24行，我们对值矩阵进行了相应的计算，就像上文说的那样input embedding来自于RMSNorm层的输出。

$$w_q \times input \ embedding$$

$$w_k \times input \ embedding$$

$$w_v \times input \ embedding$$

```C++
void LLama2Model::attention_qkv(int32_t layer_idx, const tensor::Tensor& pos_tensor) const {
  CHECK(llama_layers_ != nullptr);
  // kv cache
  tensor::Tensor query = this->get_buffer(ModelBufferType::kQuery);
  int32_t pos = pos_tensor.index<int32_t>(0);
  // wq wk wv @ input 此处的key和val就是途中用于存放新一个step中的input token @ wk 
  // 和 input token@ kv
  const auto& [key, val] = slice_kv_cache(layer_idx, pos);
  
  
  // query
  // rmsnorm_output @ wq = query
  const auto& query_layer = llama_layers_->wq_layers_.at(layer_idx);
  CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";
  auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
  STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

  // key
  // rmsnorm_output @ wk = key，就是当前的最后一列K矩阵
  const auto& key_layer = llama_layers_->wk_layers_.at(layer_idx);
  CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
  STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
  
  // value
  // rmsnorm_output @ wv = value，就是当前的最后一行的V矩阵
  const auto& value_layer = llama_layers_->wv_layers_.at(layer_idx);
  CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
  STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

  // rope
  CHECK_NE(llama_layers_->rope_layer_, nullptr)
      << "The RoPE layer in the attention block is null pointer.";
  STATUS_CHECK(llama_layers_->rope_layer_->forward(
      query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
      get_buffer(ModelBufferType::kCosCache), tensor::Tensor{}));
}
```

## 问题2

> wq、wk和wv矩阵的参数怎么从权重模型文件中提取得到？

我们知道模型权重文件是由于`tools/export.py`文件导出的，其中out_file是我们打开的模型权重参数文件路径，我们依次写入token_embedding以及每个transformer块中attention_norm，wq，wk和wv等参数，所以我们在C++端使用MMap打开权重文件后也要用相同的顺序从中读取参数：

```Python
# next write out the embedding weights
serialize_fp32(out_file, model.tok_embeddings.weight)

# now all the layers
# attention weights
for layer in model.layers:
    serialize_fp32(out_file, layer.attention_norm.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wq.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wk.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wv.weight)
...
```

其中N是Transformer块的个数，当输出权重模型文件后权重有如下的排布：

```Plain
---------------
token embedding   1 × dim × vocab size
---------------
attention norm    N × dim
---------------
weight query      N × dim × dim <==== pos
---------------
weight key        N × dim × dim
---------------
weight value      N × dim × dim
---------------
void LLama2Model::create_param_layers() {
    // 读取embedding layer层的权重
    // ...
    // ...

    int32_t dim = config_->dim_;    
     // 逐层读取query layer，开始的偏移是dim × vocab size + N × dim
    size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
    // create weight matrix for query
    for (int32_t i = 0; i < config_->layer_num_; ++i) {
        auto wq = std::make_shared<op::MatmulLayer>(device_type_, dim, dim); // 创建一个新的matmul层
        wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
        llama_layers_->wq_layers_.push_back(wq); // 每个wq layer的维度是dim×dim
        pos += dim * dim; // pos指向下一个wq layer权重的开始
    }
```

当在推理开始之前实例化一个wq算子后，接下来就需要从模型权重文件中读取对应的权重，并将其赋值给这个算子，最后在权重赋值完毕后将该算子放到wq_layers_数组中以供推理过程中使用。

## 问题3

input emebdding和wk矩阵的矩阵乘积k，怎么放入到kv cache中，在上一节课程中我们知道在每次步骤的迭代中，也就是step逐次加1，k矩阵的列数也会加1，而前面的N-1列都是保存在Cache中，只有k矩阵的最后一列是通过**新增加的输入input token**和wk矩阵进行矩阵相乘得到的，我们要做的只需要进行拼接就可以。

![img](https://l0kzvikuq0w.feishu.cn/space/api/box/stream/download/asynccode/?code=NDFiZDUzNmZmYzZkOGU5MTEzYTZkZjI3NjE2ZGI0ZWVfd3VJRElrMDJSZTR5MzdlOUhNM0tIMmNWR04wYVBIWFVfVG9rZW46RE1ub2JLNlFob2x6ZWV4Z2hjVmNhSmw4bllnXzE3NzcyNTAzMDc6MTc3NzI1MzkwN19WNA)

所以到调用slice_kv_cache的时候，我们会切出当前transformer块第pos个列作为存放结构的列，对于step3来说，slice_kv_cache切出的列就是第三列，在切出之后就是要将本周新产生的input token，也就是第3个input token与wk矩阵相乘的结果会放到slice_kv_cache新切出的列中。