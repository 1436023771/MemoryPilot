# LangChain Starter (Python)

A minimal project framework for building LangChain applications with clear structure.

## Project Structure

```text
.
├── .env.example
├── .gitignore
├── docs
│   ├── IMPLEMENTATION_DETAILS.md
│   ├── TIME_AWARE_SEARCH_IMPROVEMENTS.md
│   └── WEB_SEARCH_DISPLAY.md
├── README.md
├── requirements.txt
├── app
│   ├── __init__.py
│   ├── agents
│   │   ├── __init__.py
│   │   ├── chains.py
│   │   └── tools.py
│   ├── cli
│   │   ├── __init__.py
│   │   └── main.py
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── prompts.py
│   ├── memory
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── read_only_memory.py
│   │   ├── sqlite_memory.py
│   │   ├── write_memory.py
│   │   └── fallback
│   │       ├── __init__.py
│   │       └── memory_extraction.py
│   └── ui
│       ├── __init__.py
│       └── gui_chat.py
├── memory
│   ├── long_term_memory.db
│   └── long_term_memory.txt
└── tests
    ├── test_config.py
    ├── test_embeddings.py
    ├── test_memory.py
    ├── test_read_only_memory.py
    ├── test_smoke.py
    ├── test_sqlite_memory.py
    └── test_write_memory.py
```

## Quick Start

1. Create and activate a virtual environment:

```bash
python3 -m venv agent
source agent/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
# then set provider and API key in .env
```

DeepSeek example:

```env
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat
TEMPERATURE=0.2
```

Docker sandbox (optional, for high-risk operations in container):

```env
DOCKER_SANDBOX_ENABLED=true
DOCKER_SANDBOX_IMAGE=python:3.11-slim
DOCKER_EXEC_TIMEOUT=30
DOCKER_MEMORY_LIMIT=512m
DOCKER_CPU_LIMIT=1
DOCKER_PIDS_LIMIT=128
DOCKER_NETWORK_MODE=bridge
# Optional persistent workspace mount
DOCKER_SANDBOX_WORKDIR=~/agent_sandbox_workspace
```

Notes:
- `run_python_code` now executes inside Docker sandbox and does not fall back to host execution.
- New tool `run_docker_command` is available for shell/file/network operations in the container.
- If `DOCKER_SANDBOX_ENABLED=false`, sandbox tools return an explicit error instead of running on host.

MCP server (initial implementation, for future decoupling):

```bash
python -m app.mcp.docker_sandbox_server
```

This server exposes `run_docker_command` and `run_python_in_docker` over MCP, backed by the shared sandbox core in `app/sandbox/docker_runner.py`.

Reading companion MCP server:

```bash
python -m app.mcp.reading_companion_server --transport streamable-http --host 127.0.0.1 --port 8767 --path /mcp
```

This server exposes `retrieve_reading_context` and is backed by the same pgvector retrieval pipeline used by `retrieve_pg_knowledge`.

To route agent docker tools through MCP instead of local direct execution:

```env
DOCKER_MCP_ENABLED=true
DOCKER_MCP_COMMAND=python -m app.mcp.docker_sandbox_server
DOCKER_MCP_TIMEOUT=30
```

When `DOCKER_MCP_ENABLED=true`, `run_docker_command` and `run_python_code` will call the MCP server path.

To route reading-companion retrieval through MCP first (with local fallback on MCP failure):

```env
READING_COMPANION_MCP_ENABLED=true
READING_COMPANION_MCP_SERVER_URL=http://127.0.0.1:8767/mcp
READING_COMPANION_MCP_TIMEOUT=30
```

One-command local stack launcher (main app + multiple MCP servers):

```bash
# Start LLMLingua MCP + Docker MCP, then run main app in foreground
./scripts/dev_stack.sh up

# Or run main app in background too
./scripts/dev_stack.sh up --detach-app

# Custom app command
./scripts/dev_stack.sh up --app-cmd "python -m app.ui.gui_chat"

# Stop all managed processes
./scripts/dev_stack.sh down

# Inspect status/logs
./scripts/dev_stack.sh status
./scripts/dev_stack.sh logs all
```

Lightweight alias mode (all + per-service up/down):

```bash
source ./scripts/dev_aliases.sh

# all
ds-all-up
ds-all-down
ds-force-down
ds-force-down-now
ds-all-cli-up

# single service
ds-llm-up && ds-llm-down
ds-docker-up && ds-docker-down
ds-app-up && ds-app-down
ds-app-cli-up

# inspect
ds-status
ds-logs
```

4. Run the demo chain:

```bash
python -m app.cli.main "Explain retrieval-augmented generation in simple terms"
```


Run read-only RAG demo (retrieve from local long-term memory file):

```bash
python -m app.cli.main --use-rag --session-id demo-rag "请根据我的偏好给出建议"
```

Run writable long-term memory demo (SQLite backend):

```bash
python -m app.cli.main --memory-backend sqlite --use-rag --write-memory --show-memory-write "我叫小李，我喜欢简洁回答"
python -m app.cli.main --memory-backend sqlite --use-rag "我是谁？"
```

Start multi-turn interactive chat:

```bash
python -m app.cli.main --interactive --session-id demo-chat
```

The conversation pipeline is LangGraph-only by default.

Enable LangSmith tracing for LangGraph node-level monitoring:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=agent-langgraph
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

When tracing is enabled, LangGraph runs are reported with:
- graph run name: `langgraph_qa_chain`
- node run names: `lg_plan`, `lg_retrieve_knowledge`, `lg_build_prompt`, `lg_assistant`, `lg_tools`, `lg_finalize`

Assistant persona behavior:

- Default persona is a general execution-oriented assistant.
- Reading companion behavior is extracted as a skill and activated on knowledge-route requests.
- Skills are standardized under `.github/skills/<skill-name>/`.
- Required file: `.github/skills/<skill-name>/skill.md`.
- Optional directories: `.github/skills/<skill-name>/scripts/` and `.github/skills/<skill-name>/knowledge/`.
- Current examples: `.github/skills/general/skill.md` and `.github/skills/reading-companion/skill.md`.

Start desktop chat window (GUI, defaults to short-term + long-term memory):

```bash
python -m app.ui.gui_chat
```

Ingest docs into PostgreSQL + pgvector knowledge base:

```bash
export PGVECTOR_DSN="postgresql://user:password@localhost:5432/agent_db"
python -m app.cli.ingest_pg_knowledge --input-path docs --table knowledge_chunks --reset
```

Query PostgreSQL + pgvector knowledge base:

```bash
python -m app.cli.query_pg_knowledge "这个项目的记忆机制是什么？" --table knowledge_chunks --top-k 5
```

Sync bookshelf folders to PostgreSQL + pgvector (series-aware):

```bash
# First run: set bookshelf path and DSN (saved to memory/bookshelf_sync.json)
python -m app.cli.sync_bookshelf --bookshelf-path ~/Bookshelf --pg-dsn "$PGVECTOR_DSN" --table knowledge_chunks

# Later: one command to sync current bookshelf state
python -m app.cli.sync_bookshelf
```

You can also keep both bookshelf path and DB port config in `.env`:

```env
BOOKSHELF_PATH=~/Bookshelf
PGVECTOR_HOST=localhost
PGVECTOR_PORT=5432
PGVECTOR_DBNAME=agent_db
PGVECTOR_USER=postgres
PGVECTOR_PASSWORD=postgres

# Optional knowledge pipeline tuning
KNOWLEDGE_CHAPTER_ANALYSIS_CONCURRENCY=4
KNOWLEDGE_SYNC_INCREMENTAL=true
KNOWLEDGE_SYNC_AUTO_DELETE_REMOVED=true
KNOWLEDGE_SYNC_HASH_CHECK=true
KNOWLEDGE_SYNC_SHOW_INCREMENTAL_STATS=true
KNOWLEDGE_SYNC_STATE_FILE=memory/bookshelf_sync_state.json
KNOWLEDGE_TOP_K_DEFAULT=5
KNOWLEDGE_CONTEXT_WINDOW_DEFAULT=3
KNOWLEDGE_RERANK_CANDIDATES_DEFAULT=14
KNOWLEDGE_LOCAL_RERANK_WEIGHT_SEMANTIC=0.5
KNOWLEDGE_LOCAL_RERANK_WEIGHT_CHARACTER=0.35
KNOWLEDGE_LOCAL_RERANK_WEIGHT_TIMELINE=0.15
KNOWLEDGE_BLEND_WEIGHT_LLM=0.6
KNOWLEDGE_BLEND_WEIGHT_LOCAL=0.4
```

Then sync with one command:

```bash
python -m app.cli.sync_bookshelf
```


5. Run tests:

```bash
pytest
```

## PostgreSQL + pgvector Knowledge Base

- Ingestion CLI: `python -m app.cli.ingest_pg_knowledge`
- Query CLI: `python -m app.cli.query_pg_knowledge`
- Detailed setup: `docs/POSTGRES_PGVECTOR_KNOWLEDGE_SETUP.md`

## Bookshelf Sync (Series -> Books)

- Bookshelf path convention:
    - `<bookshelf_root>/<series>/<book>/<chapter files>`
    - `<bookshelf_root>/<series>/<chapter files>` (book defaults to file name)
- During sync, each chunk gets structured metadata:
    - `series`, `book_name`, `book_id`, `chapter`, `section`, `relative_path`
- Useful flags:
    - `--dry-run`: only scan and print stats, no DB write
    - `--reset`: truncate target table before sync
    - `--config-file`: custom config file path (default: `memory/bookshelf_sync.json`)

## Next Extensions

- Add retrieval (`langchain-community` + vector store)
- Add tool calling and agents
- Add tracing with LangSmith
- Add FastAPI endpoint for serving

## Short-Term Memory Notes

- Memory is in-process and in-memory only.
- It is scoped by `session_id`.
- Restarting the Python process clears memory.
- Two separate CLI runs do not share memory, even with the same `session_id`.

## Read-Only Long-Term Memory (RAG Demo)

- Memory source can be SQLite or text file (SQLite is default).
- This mode is read-only: it retrieves memory before answering, but does not write back.
- CLI flags:
    - `--use-rag`: enable retrieval
    - `--memory-backend`: choose `sqlite` (default) or `txt`
    - `--memory-db`: custom sqlite db path
    - `--memory-file`: custom text memory path (when backend is `txt`)
    - `--top-k`: number of retrieved memory chunks

## Writable Long-Term Memory (Rule-Based Demo)

- This mode extracts stable facts from user input and appends them to memory file.
- Current extraction patterns include:
    - `我叫X` / `我是X`
    - `我喜欢X` / `我不喜欢X`
    - `我的目标是X`
- Fact format is normalized to structured lines:
    - `用户姓名：X。`
    - `用户喜欢：X。`
    - `用户不喜欢：X。`
    - `用户目标：X。`
- Update strategy:
    - `用户姓名` and `用户目标` are treated as single-value fields; new value replaces old value.
    - `用户喜欢` and `用户不喜欢` are multi-value fields; deduplicated append.
- CLI flags:
    - `--write-memory`: enable write-back from current user turn
    - `--show-memory-write`: print written facts for debugging

Conflict update example:

```bash
python -m app.cli.main --memory-backend sqlite --memory-db memory/long_term_memory_demo.db --use-rag --write-memory --show-memory-write "我叫小李"
python -m app.cli.main --memory-backend sqlite --memory-db memory/long_term_memory_demo.db --use-rag --write-memory --show-memory-write "我叫李华"
python -m app.cli.main --memory-backend sqlite --memory-db memory/long_term_memory_demo.db --use-rag "我叫什么？"
```
