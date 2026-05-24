import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=5000):
        """
        d_model: 词向量的维度 (比如 512)
        max_len: 模型能处理的最长句子长度 (预留 5000 个位置)
        """
        super().__init__()
        
        # 1. 创建一张空白的“位置编码表”，大小是：[最多5000个词, 每个词512维]
        pe = torch.zeros(max_len, d_model)
        
        # 2. 生成绝对位置索引：[0, 1, 2, 3, ..., 4999]
        # unsqueeze(1) 会把形状从 (5000) 变成 (5000, 1)，为了后面能做矩阵乘法
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 计算频率除数 (这是论文里公式的等价代码实现)
        # 用对数和指数技巧来计算 1 / (10000^(2i/d_model))，防止小数太小溢出
        #决定 波浪图案的频率，d_model越大，频率越密集
        # 频率越密集，模型就能更细粒度地捕捉位置关系，但也可能增加计算复杂度。
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. 关键操作：偶数列和奇数列分别用 sin 和 cos 填充
        # 0::2 表示从索引0开始，步长为2（即偶数维度：0, 2, 4...）
        pe[:, 0::2] = torch.sin(position * div_term)
        # sin(1*div_term) 和 cos(1*div_term) 的频率是一样的，但相位不同，这样模型就能区分不同位置的词了。

        # 1::2 表示从索引1开始，步长为2（即奇数维度：1, 3, 5...）
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. 增加一个 batch 维度，形状变成 (1, max_len, d_model)
        # 1个批次，5000个位置，每个512维
        pe = pe.unsqueeze(0)
        
        # 6. 把 pe 存到模型的 buffer 里。
        # 为什么不用 Parameter？因为位置编码是固定的数学公式算出来的，不需要大模型去学习和更新！
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: 就是上一层的输出 (通常是 Embedding 词嵌入矩阵)
        形状为 (batch_size, 句子实际长度 seq_len, d_model)
        """
        # 7. 把词嵌入(x) 和 截取到实际长度的位置编码(pe) 直接相加！
        x = x + self.pe[:, :x.size(1), :]
        return x