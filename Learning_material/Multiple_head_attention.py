class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=8, num_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # 4
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape  # (1, 5, 8)
        
        Q = self.W_q(x)  # (1, 5, 8)
        K = self.W_k(x)  # (1, 5, 8)
        V = self.W_v(x)  # (1, 5, 8)
        
        # 关键步骤：切分成多头
        Q = Q.view(B, T, self.num_heads, self.head_dim)  # (1, 5, 2, 4)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)
        
        # 转置，把 head 维度提前
        Q = Q.transpose(1, 2)  # (1, 2, 5, 4)  ← 2个头，每个头 5x4
        K = K.transpose(1, 2)  # (1, 2, 5, 4)
        V = V.transpose(1, 2)  # (1, 2, 5, 4)
        
        # 每个头独立计算注意力
        scores = Q @ K.transpose(-2, -1)  # (1, 2, 5, 5)  ← 每个头一个 5x5 矩阵
        attn = softmax(scores / sqrt(4), dim=-1)
        out = attn @ V  # (1, 2, 5, 4)
        
        # 合并多头
        out = out.transpose(1, 2).reshape(B, T, C)  # (1, 5, 8)
        return self.out_proj(out)
    



import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """多头注意力层"""
    # d_model dimension of the model like input should be 512.
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        # Q、K、V的权重矩阵（包含所有头）
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        # 最后的线性变换
        self.W_O = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        将最后一维分成(num_heads, d_k)
        x: (batch, seq_len, d_model)
        输出: (batch, num_heads, seq_len, d_k)
        """
        # -1 means auto determine the size of this dimension based on the other dimensions and the total number of elements
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V: (batch, seq_len, d_model)
        mask: (batch, seq_len, seq_len) 可选
        """
        batch_size = Q.shape[0]
        # 1. 线性变换
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)
        # 2. 分成多头
        Q = self.split_heads(Q, batch_size)  
        
        # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)
        # 3. 计算attention
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, num_heads, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_k)  # 缩放
        # 4. 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # 5. Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        # 6. 加权求和
        output = torch.matmul(attention_weights, V)  # (batch, num_heads, seq_len, d_k)
        # 7. 合并多头
        output = output.transpose(1, 2).contiguous()  # (batch, seq_len, num_heads, d_k)
        output = output.view(batch_size, -1, self.d_model)  # (batch, seq_len, d_model)
        # 8. 最后的线性变换
        output = self.W_O(output)
        return output, attention_weights

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.linear2(self.relu(self.linear1(x)))

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        # 多头注意力
        self.attention = MultiHeadAttention(d_model, num_heads)
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff)
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len, seq_len)
        """
        # 1. 多头注意力 + 残差连接 + LayerNorm
        attn_output, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        # 2. 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attention_weights
    
