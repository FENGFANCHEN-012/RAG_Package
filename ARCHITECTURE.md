# 系统架构文档

## 整体架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端 (Frontend)                          │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  frontend/js/app.js                                      │   │
│  │  - 用户界面管理                                           │   │
│  │  - 消息发送和显示                                         │   │
│  │  - 查询类型检测（关键词匹配）                             │   │
│  │  - API 路由                                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP Request
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    后端服务器 (Node.js)                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  backend/server.js                                        │   │
│  │  - Express API 服务器                                     │   │
│  │  - Python 脚本调用                                        │   │
│  │  - MySQL 数据库连接                                       │   │
│  │  - 请求路由                                               │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │ Spawn Python
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Python API 处理器 (api_handler.py)              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  请求路由分发                                             │   │
│  │  - handle_rag()          → RAG 查询                       │   │
│  │  - handle_query()        → 普通 LLM 查询                  │   │
│  │  - handle_database_query() → 数据库查询处理               │   │
│  │  - handle_classify_intent() → 意图分类                    │   │
│  │  - handle_generate_sql()  → SQL 生成                      │   │
│  └──────────────────────────────────────────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  RAG 处理器   │   │  AI 调用器    │   │ SQL 生成器    │
│               │   │               │   │               │
│ - 向量检索    │   │ - Ollama API  │   │ - LangChain    │
│ - 查询扩展    │   │ - 提示词生成  │   │ - 自然语言转SQL│
│ - 多阶段回退  │   │ - 对话历史    │   │ - 数据库查询  │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ Pinecone      │   │ Ollama        │   │ MySQL         │
│ 向量数据库    │   │ LLM 服务      │   │ 数据库        │
└───────────────┘   └───────────────┘   └───────────────┘
```

## 核心组件说明

### 1. 前端 (frontend/js/app.js)

**功能：**
- 管理聊天界面用户交互
- 处理消息发送和显示
- 使用关键词匹配检测数据库查询意图
- 根据查询类型路由到不同 API

**主要函数：**
- `isDatabaseQuery(message)`: 检测是否为数据库查询（关键词匹配）
- `sendMessage()`: 发送消息并路由到相应 API
- `handleDatabaseQuery()`: 处理数据库查询
- `handleRAGQuery()`: 处理 RAG 查询
- `handleNormalQuery()`: 处理普通查询

### 2. 后端服务器 (backend/server.js)

**功能：**
- Node.js Express 服务器
- 提供 RESTful API
- 调用 Python 脚本处理请求
- 管理 MySQL 数据库连接

**主要 API 端点：**
- `GET /api/health`: 健康检查
- `POST /api/query`: 普通 LLM 查询
- `POST /api/rag`: RAG 检索增强生成查询
- `POST /api/database-query`: 数据库查询
- `POST /api/classify-intent`: 意图分类

### 3. Python API 处理器 (api_handler.py)

**功能：**
- Python 后端主入口
- 路由请求到不同处理器
- 集成向量检索、AI 调用、数据库查询

**主要处理器：**
- `handle_rag()`: RAG 检索增强生成，支持多阶段回退
- `handle_query()`: 普通 LLM 查询
- `handle_database_query()`: 数据库查询结果处理
- `handle_classify_intent()`: 意图分类
- `handle_generate_sql()`: SQL 生成

### 4. AI 调用器 (call_AI.py)

**功能：**
- 封装对 LLM（Ollama/DeepSeek）的调用
- 支持对话历史传递
- 生成结构化提示词

**主要类：**
- `CallAI`: AI 调用封装类
  - `generate_prompt()`: 生成提示词
  - `call_ollama()`: 调用 Ollama API

### 5. 查询处理器 (query_processor.py)

**功能：**
- 处理和扩展用户查询
- 判断查询类型
- 使用 AI 生成多个相关查询

**主要函数：**
- `is_definition_query()`: 判断是否为定义类查询
- `is_complex_query()`: 判断是否为复杂查询
- `generate_multi_queries_ai()`: 使用 AI 生成多个相关查询

### 6. 意图分类器 (intent_classifier.py)

**功能：**
- 使用 TF-IDF + SVM 判断用户查询意图
- 将查询分类为：database_query、rag_query、normal_query

**主要类：**
- `IntentClassifier`: 意图分类器类
  - `train()`: 训练分类器
  - `predict()`: 预测查询意图

### 7. SQL 生成器 (sql_generator.py)

**功能：**
- 使用 LangChain 将自然语言转换为 SQL
- 执行 SQL 查询并返回结果
- 支持回退到关键词匹配

**主要类：**
- `SQLGenerator`: SQL 生成器类
  - `generate_sql()`: 将自然语言转换为 SQL 并执行

## 数据流程

### RAG 查询流程

```
用户输入 "糖尿病注意事项"
    ↓
前端检测查询类型
    ↓
发送到 /api/rag
    ↓
Node.js 调用 Python api_handler.py
    ↓
handle_rag() 处理
    ↓
query_processor 扩展查询（生成多个相关查询）
    ↓
data_server 从 Pinecone 检索文档
    ↓
call_AI 生成答案
    ↓
如果文档不相关，触发回退：
  1. AI 直接生成答案
  2. 使用 AI 生成的答案重新检索
  3. 网络搜索（DuckDuckGo）
    ↓
返回答案给前端
```

### 数据库查询流程

```
用户输入 "有多少员工"
    ↓
前端关键词匹配检测到数据库查询
    ↓
发送到 /api/database-query
    ↓
Node.js 调用 Python handle_generate_sql()
    ↓
sql_generator 生成 SQL
    ↓
执行 SQL 查询 MySQL
    ↓
返回查询结果
    ↓
call_AI 将结果转换为自然语言
    ↓
返回答案给前端
```

## 外部依赖

- **Ollama**: 本地 LLM 服务（http://localhost:11434）
- **Pinecone**: 向量数据库
- **MySQL**: 关系型数据库
- **DuckDuckGo**: 网络搜索 API

## 环境变量 (.env)

```
PORT=3000
DB_HOST=127.0.0.1
DB_PORT=3305
DB_DATABASE=company_info
DB_USERNAME=rootUser
DB_PASSWORD=123456
PINECONE_API_KEY=your_pinecone_api_key
```

## 技术栈

- **前端**: HTML, CSS, JavaScript
- **后端**: Node.js, Express
- **Python**: Python 3.x
- **AI**: Ollama (DeepSeek)
- **向量数据库**: Pinecone
- **关系数据库**: MySQL
- **机器学习**: scikit-learn, jieba
- **SQL 生成**: LangChain
