# **KV Cache是什么？它能减少多少计算量？**
KV Cache 是在自回归生成时的一种缓存技巧。每一步生成时，当前 token 的 Query 需要和历史所有 token 的 Key、Value 做 attention，而历史 token 的 Key 和 Value 其实之前已经算过了。如果不做缓存，每一步都要重新计算所有历史 token 的 Key 和 Value，总复杂度是 O(L³)。有了 KV Cache，每一步只计算当前 token 的 Key 和 Value，然后拼到缓存里，总复杂度就降到 O(L²)。

它的显存开销大致是：2 乘以 batch size 乘以 head 数量乘以 head 维度乘以序列长度，再乘上每个参数占的字节数。对于 7B 的 LLaMA，序列长度 2048 时，KV Cache 大约占 1GB 的显存。

为了进一步压缩，业界提出了 MQA 和 GQA。MQA 是所有 head 共享同一份 KV，显存直接降到原来的 1/h；GQA 是折中方案，把 head 分成几组，每组共享一份 KV。

# **大模型推理时，KV Cache 的作用是什么？它的内存占用如何计算？有哪些方法可以压缩或优化 KV Cache？**

**KV Cache 的作用**：自回归生成时，每生成一个新 token，它的 query 需要和之前所有 token 的 key、value 做 attention。KV Cache 把这些历史 key、value 存下来，避免每一步都重新算。没有它复杂度 O(n³)，有它降到 O(n²)。

**内存占用怎么算**：大致是 `2 × batch_size × num_heads × head_dim × seq_len × 2字节（fp16）`。举个例子，LLaMA-7B，batch=1，长度 2048，每层 KV Cache 约 32MB，32 层就是 1GB。

**怎么压缩优化**：第一，用 GQA 或 MQA，让多个 query 头共享一份 KV，显存降为原来的 1/组数 或 1/头数。第二，量化 KV Cache，比如用 INT8 存储，直接减半。第三，用 PagedAttention（vLLM 里的技术），把 KV Cache 分页管理，减少显存碎片。第四，用滑动窗口，只保留最近一段窗口的 KV，放弃更早的。第五，推理时用早停或投机解码，减少需要生成的长度。