
from email.mime import text
import re

#示例：文档预处理代码
'''def preprocess_document(doc):
    # 1. 移除多余的空格和换行
    doc = re.sub(r'\s+', ' ', doc)
    # 2. 提取纯文本（从PDF、HTML等）
    if doc_type == 'pdf':
        text = extract_text_from_pdf(doc)
    # 3. 规范化格式
    text = text.strip().lower()
    # 4. 去除无用信息（页眉、页脚等）
    text = remove_headers_footers(text)
return text
'''  

text = '''

阿司匹林是一种常用的解热镇痛药。它的主要作用包括：
解热：降低发烧体温
镇痛：缓解轻到中度疼痛
抗血小板：预防血栓形成

但是，阿司匹林也有副作用。常见的副作用包括：
胃肠道反应：胃痛、恶心
出血风险：特别是长期服用


'''
#每300个字符一块
chunk_size = 300
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

#按句号、问号、感叹号分割
import nltk
sentences = nltk.sent_tokenize(text)
#段落分块（保留逻辑结构）
#按换行符或段落标记分割
chunks = text.split('\n\n')
#滑动窗口分块（带重叠，避免信息丢失）

chunk_size = 300
overlap = 50  # 重叠50字符
chunks = []

for i in range(0, len(text), chunk_size - overlap):
    chunks.append(text[i:i+chunk_size])


print(chunks)
    

# 将文本转化为向量

#使用OpenAI的Embedding模型
from openai import OpenAI
client = OpenAI()

text = "阿司匹林是一种解热镇痛药"
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)
vector = response.data[0].embedding
print(f"向量维度: {len(vector)}")  # 输出: 1536
print(f"前5个值: {vector[:5]}")    # 输出: [0.023, -0.014, 0.089, ...]

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 结果： 每个相乘再相加 / 两个的每个向量的平方再平方根相乘
vec1 = np.array([0.1, 0.3, 0.5])
vec2 = np.array([0.12, 0.29, 0.51])
similarity = cosine_similarity([vec1], [vec2])[0][0]
print(f"相似度: {similarity:.3f}")  # 输出: 0.999（非常相似）

