"# LLM Assistant - Frontend & Backend Setup

## 项目结构

```
Transformer/
├── backend/               # Node.js 后端
│   ├── server.js         # Express 服务器
│   ├── package.json      # Node.js 依赖
│   └── node_modules/
├── frontend/             # 前端界面
│   ├── public/
│   │   └── index.html    # 主页面
│   ├── css/
│   │   └── styles.css    # 样式文件
│   └── js/
│       └── app.js        # 前端逻辑
├── api_handler.py        # Python API 处理器
├── call_AI.py            # LLM 调用模块
└── RAG_development.py    # RAG 相关代码
```

## 快速启动

### 1. 确保 Ollama 已启动
```powershell
ollama serve
```

### 2. 启动后端服务器
```powershell
cd backend
npm start
```

### 3. 打开浏览器访问
```
http://localhost:3000
```

## API 接口说明

### POST /api/query
普通 LLM 查询接口

**请求体:**
```json
{
    "question": "你的问题",
    "model": "kimi-k2:1t-cloud",
    "max_tokens": 256
}
```

### POST /api/multi-query
生成多个相关查询

**请求体:**
```json
{
    "query": "原始查询",
    "model": "kimi-k2:1t-cloud"
}
```

### POST /api/rag
RAG 增强查询接口

**请求体:**
```json
{
    "question": "你的问题",
    "docs": [
        {"content": "参考文档1内容"},
        {"content": "参考文档2内容"}
    ],
    "model": "kimi-k2:1t-cloud",
    "max_tokens": 256
}
```

### GET /api/health
健康检查接口

## 功能特性

1. **普通查询**: 直接向 LLM 提问
2. **多查询生成**: 自动生成多个相关问题
3. **RAG 查询**: 基于参考文档回答问题
4. **模型选择**: 支持多种 Ollama 模型
5. **Token 控制**: 可调整最大生成 Token 数

## 开发模式

使用 nodemon 自动重启:
```powershell
cd backend
npm run dev
```

## 常见问题

### Q: 无法连接到服务器?
A: 确保后端服务器已启动 (`npm start` in backend folder)

### Q: LLM 响应为空?
A: 检查 Ollama 是否正在运行，模型是否已下载:
```powershell
ollama list
ollama pull kimi-k2:1t-cloud
```

### Q: Python 脚本报错?
A: 确保已安装 Python 依赖:
```powershell
pip install openai
```
"