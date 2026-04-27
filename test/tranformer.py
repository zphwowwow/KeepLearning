import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleTransformerDecoder(nn.Module):
    """
    一个简单的 Transformer 解码器，用于自回归生成。
    包含 token 嵌入、位置嵌入、多个解码器层，以及最终的输出投影。
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 可学习的位置嵌入
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 堆叠多个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)  # 最后加一层 LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 输出词表 logits
        # 权重共享（可选）
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len) 输入 token 索引
        mask: (batch_size, seq_len) 或 (batch_size, 1, seq_len, seq_len) 的注意力掩码
        """
        seq_len = x.size(1)
        # 生成位置索引 [0, 1, ..., seq_len-1] 并扩展到 batch
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # (1, seq_len)

        # 嵌入并相加
        token_emb = self.token_embedding(x)  # (batch, seq_len, d_model)
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
        x = self.dropout(token_emb + pos_emb)

        # 因果掩码（如果未提供则自动生成）
        if mask is None:
            # 创建一个上三角矩阵，屏蔽未来 token
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

        # 通过每个解码器层
        for layer in self.layers:
            x = layer(x, mask)

        x = self.ln_final(x)          # (batch, seq_len, d_model)
        logits = self.lm_head(x)      # (batch, seq_len, vocab_size)
        return logits


class DecoderLayer(nn.Module):
    """单个 Transformer 解码器层"""
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self.feed_forward(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制，手动实现"""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # 线性投影并分割成多头
        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch, nhead, seq_len, head_dim)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (batch, nhead, seq_len, seq_len)

        if mask is not None:
            scores = scores + mask  # mask 应为与 scores 形状兼容的矩阵，填充 -inf 的位置被屏蔽

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)  # (batch, nhead, seq_len, head_dim)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # 合并头

        output = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    """前馈网络，两层线性 + ReLU"""
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# 使用示例
if __name__ == "__main__":
    # 超参数
    vocab_size = 10000
    d_model = 256
    nhead = 4
    num_layers = 3
    dim_feedforward = 1024
    max_len = 128
    batch_size = 2
    seq_len = 10

    model = SimpleTransformerDecoder(vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len=max_len)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))  # 随机输入

    # 前向传播
    logits = model(x)
    print(logits.shape)  # (2, 10, 10000)