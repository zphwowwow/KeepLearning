# GQA 和 MQA 的区别
这两个都是对标准 Multi‑Head Attention 的压缩，目的是减少 KV Cache 的显存占用。

MQA 是多查询注意力，它让所有注意力头共享同一份 Key 和 Value。也就是说，无论有多少个 query 头，都只用一份 KV。这样 KV Cache 的大小直接降到了原来的 1/h，h 是头数。但是精度损失比较大，因为每个头都用同样的 KV 信息。

GQA 是分组查询注意力，它是 MQA 和标准 MHA 的折中。先把 query 头分成若干组，每组内部共享一份 KV。比如 32 个 query 头，分成 4 组，每组 8 个头，那么就有 4 份 KV，显存降为原来的 4/32=1/8。GQA 的精度比 MQA 好，压缩比也够用，所以现在很多模型比如 LLaMA 2 和 3 都在用 GQA。

简单记忆：MHA 是一对一，MQA 是全部共享，GQA 是分组共享。