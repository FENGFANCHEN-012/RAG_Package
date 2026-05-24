# Transformer RAG System

Transformer is a Retrieval-Augmented Generation (RAG) assistant that combines a simple web UI, a Node.js API server, and Python-based processing modules. It supports LLM chat, retrieval, intent classification, and natural-language-to-SQL flows.

## What this project does

- LLM chat and RAG-based answers
- Query routing based on intent classification
- Retrieval with vector search and hybrid search helpers
- Natural-language-to-SQL for database-backed questions

## Requirements

- Python 3.8+ and pip
- Node.js 16+ and npm
- Ollama or another LLM endpoint (optional)
- Pinecone or another vector DB (optional)
- MySQL or PostgreSQL (for SQL features)

## Project structure

- [backend/](backend/) — Node.js Express API server
- [frontend/](frontend/) — static UI (HTML/CSS/JS)
- [api_handler.py](api_handler.py) — Python request router (RAG, SQL, intent)
- [call_AI.py](call_AI.py) — LLM wrapper and prompt helpers
- [query_processor.py](query_processor.py) — query expansion and routing helpers
- [intent_classifier.py](intent_classifier.py) — intent classification logic
- [sql_generator.py](sql_generator.py) — natural-language-to-SQL flow
- [data_server.py](data_server.py), [retrieve.py](retrieve.py), [hybrid_search.py](hybrid_search.py) — retrieval helpers
- [Embedding.py](Embedding.py) — embedding utilities
- [core/](core/) — shared Python utilities
- [docs/](docs/) — documentation
- [intent_model/](intent_model/) — intent model artifacts
- [train_pure_text/](train_pure_text/) — training data or experiments
- [Learning_material/](Learning_material/) — reference materials
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design and flow diagrams
- [README_FRONTEND_BACKEND.md](README_FRONTEND_BACKEND.md) — frontend/backend quick start

## Environment

Create a local `.env` (do not commit). Example keys:

```
PORT=3000
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=your_db
DB_USERNAME=root
DB_PASSWORD=secret
DB_DRIVER=mysql
# For PostgreSQL
PG_HOST=127.0.0.1
PG_PORT=5432
PG_DATABASE=your_db
PG_USERNAME=postgres
PG_PASSWORD=secret
PINECONE_API_KEY=your_pinecone_key
OLLAMA_URL=http://localhost:11434
```

## Quickstart

1) Python setup:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
source .venv/bin/activate
pip install -r requirements.txt
```

2) Backend setup:

```powershell
cd backend
npm install
npm start
```

3) Frontend:

Open [frontend/public/index.html](frontend/public/index.html) in a browser, or serve it with any static server.

4) Optional services:

- Start Ollama if using local LLMs
- Configure Pinecone for retrieval
- Configure MySQL or PostgreSQL for SQL features

## API endpoints (backend)

- `GET /api/health`
- `POST /api/query`
- `POST /api/rag`
- `POST /api/database-query`
- `POST /api/classify-intent`

## SQL: MySQL and PostgreSQL

The system supports SQL generation against MySQL and PostgreSQL. Choose the driver and credentials in `.env`, then the SQL flow will use the configured database. See [sql_generator.py](sql_generator.py) for the NL-to-SQL pipeline.

## Pinecone upload (vector ingestion)

Document ingestion follows the typical RAG pattern:

1) Split documents into chunks
2) Generate embeddings
3) Upsert vectors into Pinecone

Embedding and retrieval helpers live in [Embedding.py](Embedding.py), [data_server.py](data_server.py), and [retrieve.py](retrieve.py). If you want, I can add a dedicated ingestion script and a `requirements.txt`.

## Web search fallback

When retrieval does not find relevant results, the pipeline can fall back to web search (DuckDuckGo) before answering. The flow and fallback order are described in [ARCHITECTURE.md](ARCHITECTURE.md).

## RAG strategy used

The RAG flow uses query expansion, retrieval, and answer synthesis with fallback:

1) Intent detection and query routing
2) Multi-query expansion (improve recall)
3) Vector or hybrid retrieval
4) LLM synthesis from retrieved context
5) Fallback to direct LLM answer or web search if context is weak

## Notes

- `.gitignore` excludes `node_modules/`, `vendor/`, and `.env`
- Use Git LFS for large binaries

