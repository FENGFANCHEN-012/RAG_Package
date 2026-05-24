# 替代检索和重排序方案文档

本文档记录了未启用的替代检索和重排序方案，包括其意义、介绍和启用方法。

---

## 1. Cohere Rerank

### 意义
商业级重排序服务，提供高质量的文档重排序能力，适合对精度要求极高的场景。

### 介绍
- **提供商**：Cohere
- **类型**：商业 API 服务
- **特点**：
  - 多语言支持（包括中文）
  - 商业级精度
  - 无需本地部署
  - 自动更新优化
- **优势**：
  - 精度高，经过大规模数据训练
  - 易于集成（REST API）
  - 无需管理模型
- **劣势**：
  - 需要付费（有免费额度限制）
  - 依赖网络 API，延迟可能较高
  - 数据隐私顾虑（数据发送到外部）
  - 需要管理 API Key

### 如何启用

**步骤 1：获取 API Key**
1. 访问 [Cohere 官网](https://cohere.com/)
2. 注册账号并获取 API Key

**步骤 2：安装依赖**
```bash
pip install cohere
```

**步骤 3：修改代码**
在 `Embedding.py` 中添加 Cohere Reranker 实现：

```python
import cohere

class CohereReranker:
    def __init__(self, api_key: str):
        self.client = cohere.Client(api_key)
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5):
        results = self.client.rerank(
            model="rerank-english-v2.0",  # 或使用多语言模型
            query=query,
            documents=[doc["content"] for doc in documents],
            top_n=top_k
        )
        return results
```

**步骤 4：集成到现有系统**
在 `data_server.py` 中替换现有的 CrossEncoderReranker：

```python
# 替换 CrossEncoderReranker 为 CohereReranker
reranker = CohereReranker(api_key="your-api-key")
```

---

## 2. ColBERT

### 意义
Token-level 检索模型，通过 MaxSim 操作符实现细粒度匹配，对复杂查询和长文档效果更好。

### 介绍
- **开发者**：Stanford University
- **类型**：开源检索模型
- **特点**：
  - Token-level 嵌入
  - MaxSim 操作符匹配
  - 细粒度语义理解
  - 对长文档效果好
- **优势**：
  - 检索精度高
  - 能捕捉细粒度语义
  - 对复杂查询友好
- **劣势**：
  - 计算成本高（10-100倍）
  - 存储成本高
  - 与 Pinecone 不兼容
  - 需要重构架构

### 如何启用

**步骤 1：安装依赖**
```bash
pip install ragatouille
```

**步骤 2：创建索引**
```python
from ragatouille import RAGPretrainedModel

# 使用预训练模型
RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
RAG.index(
    collection=["文档1内容", "文档2内容", ...],
    index_name="my_index"
)
```

**步骤 3：检索**
```python
results = RAG.search(query="查询内容", k=10)
```

**步骤 4：集成到现有系统**
- 需要替换现有的 Pinecone 向量检索
- 使用本地 FAISS 索引替代 Pinecone
- 修改 `data_server.py` 中的检索逻辑

**注意**：由于 Pinecone 不支持 ColBERT 的 token-level 索引，需要完全重构检索架构。

---

## 3. RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

### 意义
递归摘要检索方法，通过构建文档树结构实现多层级检索，适合长文档和需要全局概览的场景。

### 介绍
- **开发者**：Stanford University
- **类型**：开源检索架构
- **特点**：
  - 递归聚类和摘要生成
  - 构建多层树结构
  - 多层级检索能力
  - 保留全局上下文
- **优势**：
  - 对长文档效果好
  - 保留全局上下文
  - LLM 生成的摘要质量高
- **劣势**：
  - 复杂度极高
  - 计算成本高（多次 LLM 调用）
  - 不适合 Pinecone
  - 维护成本高

### 如何启用

**步骤 1：安装依赖**
```bash
pip install raptor
```

**步骤 2：构建文档树**
```python
from raptor import RAPTOR

# 初始化 RAPTOR
raptor = RAPTOR(
    embedding_model="BAAI/bge-large-zh-v1.5",
    llm_model="gpt-4"  # 或其他 LLM
)

# 构建文档树
tree = raptor.build_tree(documents)
```

**步骤 3：检索**
```python
results = raptor.search(query, tree, k=10)
```

**步骤 4：集成到现有系统**
- 替换现有的摘要检索逻辑
- 实现树结构存储（需要自定义索引）
- 修改 `data_server.py` 中的检索流程

**注意**：当前系统已实现类似效果（`get_docs_by_summary_search`），RAPTOR 的复杂度可能不值得。

---

## 4. TM2C2 (Text Mining for Conceptual Clustering)

### 意义
基于概念聚类的文本挖掘方法，通过概念级别的聚类提升检索相关性。

### 介绍
- **类型**：概念聚类方法
- **特点**：
  - 概念级别聚类
  - 语义主题提取
  - 改善检索相关性
  - 适合知识密集型文档
- **优势**：
  - 概念级别理解
  - 改善主题相关性
  - 对专业领域效果好
- **劣势**：
  - 实现复杂
  - 需要领域知识
  - 计算成本高
  - 社区支持有限

### 如何启用

**步骤 1：安装依赖**
```bash
pip install tm2c2
```

**步骤 2：概念聚类**
```python
from tm2c2 import ConceptClusterer

clusterer = ConceptClusterer(
    embedding_model="BAAI/bge-large-zh-v1.5"
)

# 聚类文档
clusters = clusterer.cluster(documents, n_clusters=10)
```

**步骤 3：基于概念检索**
```python
results = clusterer.search_by_concept(query, clusters)
```

**步骤 4：集成到现有系统**
- 在文档上传时进行概念聚类
- 存储概念标签到 Pinecone metadata
- 修改检索逻辑以支持概念过滤

**注意**：TM2C2 社区支持有限，建议优先考虑其他成熟方案。

---

## 5. ColBERT Reranker v2

### 意义
ColBERT 的改进版本，支持 centroid 聚合，在保持精度的同时降低计算成本。

### 介绍
- **开发者**：Stanford University
- **类型**：开源重排序模型
- **特点**：
  - Centroid 聚合优化
  - Token-level 匹配
  - 计算成本降低
  - 兼容向量数据库
- **优势**：
  - 比完整 ColBERT 成本低
  - 精度接近完整 ColBERT
  - 可以与 Pinecone 兼容
- **劣势**：
  - 仍需重新索引
  - 精度略低于完整 ColBERT
  - 实现复杂

### 如何启用

**步骤 1：安装依赖**
```bash
pip install colbert-ai
```

**步骤 2：使用 Centroid 模式**
```python
from colbert import Indexer, Searcher

# 使用 centroid 模式索引
indexer = Indexer(
    checkpoint="colbertv2.0",
    collection=documents,
    centroid_mode=True  # 启用 centroid 聚合
)
indexer.index("my_index")

# 检索
searcher = Searcher(index="my_index", centroid_mode=True)
results = searcher.search(query, k=10)
```

**步骤 3：集成到现有系统**
- 替换现有的 CrossEncoderReranker
- 或替换向量检索流程
- 修改 `data_server.py` 和 `Embedding.py`

**注意**：需要重新索引所有文档，迁移成本较高。

---

## 6. Jina AI Rerank

### 意义
Jina AI 提供的高性能重排序服务，支持多种模型和部署方式。

### 介绍
- **提供商**：Jina AI
- **类型**：商业 API + 开源模型
- **特点**：
  - 多语言支持
  - 高性能
  - 提供 API 和本地部署
  - 多种模型选择
- **优势**：
  - 精度高
  - 部署灵活（API 或本地）
  - 中文支持良好
  - 社区活跃
- **劣势**：
  - API 需要付费
  - 本地部署需要资源
  - 模型较大

### 如何启用

### 方案 A：使用 API

**步骤 1：获取 API Key**
1. 访问 [Jina AI 官网](https://jina.ai/)
2. 注册账号并获取 API Key

**步骤 2：安装依赖**
```bash
pip install jina
```

**步骤 3：使用 API**
```python
from jina import Client

client = Client(host="https://api.jina.ai/v1/rerank")

results = client.post(
    inputs={
        "query": "查询内容",
        "documents": ["文档1", "文档2", ...],
        "model": "jina-reranker-v1-base-en"
    }
)
```

### 方案 B：本地部署

**步骤 1：安装依赖**
```bash
pip install jina-reranker
```

**步骤 2：下载模型**
```python
from jina_reranker import Reranker

reranker = Reranker(model_name="jina-reranker-v1-base-en")
```

**步骤 3：集成到现有系统**
在 `Embedding.py` 中替换 CrossEncoderReranker：

```python
from jina_reranker import Reranker

class JinaReranker:
    def __init__(self, model_name: str = "jina-reranker-v1-base-en"):
        self.reranker = Reranker(model_name)
    
    def rerank(self, query: str, documents: List[dict], top_k: int = 5):
        # 实现重排序逻辑
        pass
```

---

## 总结与建议

### 当前方案
- **Embedding**: BAAI/bge-large-zh-v1.5（中文优化）
- **Reranker**: BAAI/bge-reranker-large（中文最强开源）
- **检索**: Pinecone + BM25 + 摘要检索

### 何时考虑替代方案

| 方案 | 适用场景 | 优先级 |
|------|----------|--------|
| Cohere Rerank | 有预算、对精度要求极高、无隐私顾虑 | 中 |
| ColBERT | 超长文档、复杂查询、有充足资源 | 低 |
| RAPTOR | 超长文档、需要多层级概览 | 低 |
| TM2C2 | 专业领域、概念级别聚类 | 低 |
| ColBERT Reranker v2 | 需要 ColBERT 精度但成本敏感 | 中 |
| Jina AI Rerank | 需要 API 服务或本地部署 | 中 |

### 推荐顺序
1. 先测试当前优化方案的效果
2. 如果不足，考虑 Jina AI Rerank（部署灵活）
3. 最后考虑 ColBERT 相关方案（成本高）

### 不建议的方案
- RAPTOR：当前已实现类似效果，复杂度过高
- TM2C2：社区支持有限，风险高
