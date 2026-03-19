# LangChain Starter (Python)

A minimal project framework for building LangChain applications with clear structure.

## Project Structure

```text
.
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app
│   ├── __init__.py
│   ├── chains.py
│   ├── config.py
│   ├── main.py
│   └── prompts.py
├── memory
│   └── long_term_memory.txt
└── tests
    └── test_smoke.py
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

4. Run the demo chain:

```bash
python -m app.main "Explain retrieval-augmented generation in simple terms"
```

Run read-only RAG demo (retrieve from local long-term memory file):

```bash
python -m app.main --use-rag --session-id demo-rag "请根据我的偏好给出建议"
```

Run writable long-term memory demo:

```bash
python -m app.main --use-rag --write-memory --show-memory-write "我叫小李，我喜欢简洁回答"
python -m app.main --use-rag "我是谁？"
```

Start multi-turn interactive chat:

```bash
python -m app.main --interactive --session-id demo-chat
```

5. Run tests:

```bash
pytest
```

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

- Memory source file: `memory/long_term_memory.txt`
- This mode is read-only: it retrieves memory before answering, but does not write back.
- CLI flags:
    - `--use-rag`: enable retrieval
    - `--memory-file`: custom memory file path
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
python -m app.main --use-rag --write-memory --show-memory-write --memory-file memory/long_term_memory_demo.txt "我叫小李"
python -m app.main --use-rag --write-memory --show-memory-write --memory-file memory/long_term_memory_demo.txt "我叫李华"
python -m app.main --use-rag --memory-file memory/long_term_memory_demo.txt "我叫什么？"
```
