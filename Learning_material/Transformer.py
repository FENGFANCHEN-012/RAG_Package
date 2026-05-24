import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 定义三个线性层，将输入映射为 Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        
        # 1. 计算 Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, dim)
        K = self.W_k(x)  # (batch, seq_len, dim)
        V = self.W_v(x)  # (batch, seq_len, dim)
        
        # 2. 计算注意力分数矩阵
        # 矩阵乘法: Q * K^T，需要转置 K 的最后两维
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)
        
        # 3. 缩放 (除以 sqrt(d_k))
        d_k = self.embed_dim
        scores = scores / (d_k ** 0.5)
        
        # 4. Softmax 归一化 (通常只在最后维度做，且可选择加上 mask)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        
        # 5. 加权求和: 权重 * V
        output = torch.matmul(attn_weights, V)  # (batch, seq_len, dim)
        
        return output

# --- 测试示例 ---
batch_size = 2
seq_len = 5
embed_dim = 8

x = torch.randn(batch_size, seq_len, embed_dim)
sa = SimpleSelfAttention(embed_dim)
out = sa(x)

print("输入形状:", x.shape)    # [2, 5, 8]
print("输出形状:", out.shape)  # [2, 5, 8] (形状不变，语义变了)