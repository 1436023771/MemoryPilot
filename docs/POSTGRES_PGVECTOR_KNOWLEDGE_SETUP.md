# PostgreSQL + pgvector 文档知识库接入

这份文档说明如何把本地文档（如 docs 目录）写入 PostgreSQL + pgvector，并进行向量检索。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果你只想安装新增依赖：

```bash
pip install "psycopg[binary]" pgvector sentence-transformers pypdf ebooklib beautifulsoup4
```

## 2. 准备 PostgreSQL 与扩展

确保 PostgreSQL 可连接，并具备创建扩展权限。首次运行会自动执行：

- CREATE EXTENSION IF NOT EXISTS vector
- CREATE TABLE IF NOT EXISTS knowledge_chunks (...)

## 3. 配置连接串

可通过环境变量设置：

```bash
export PGVECTOR_DSN="postgresql://user:password@localhost:5432/agent_db"
```

也可以每次命令行用 --pg-dsn 显式传入。

## 4. 文档入库

示例：把 docs 目录全部写入 knowledge_chunks。

```bash
python -m app.cli.ingest_pg_knowledge --input-path docs --table knowledge_chunks --reset
```

常用参数：

- --chunk-size: 分块大小（默认 800）
- --chunk-overlap: 重叠大小（默认 120）
- --embedding-model: 向量模型（默认 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）

## 5. 检索验证

```bash
python -m app.cli.query_pg_knowledge "这个项目的记忆机制是什么？" --table knowledge_chunks --top-k 5
```

## 6. 当前实现说明

- 支持文件类型：txt / md / rst / py / pdf / epub
- 元数据包含：source, chunk_index
- 入库策略：document_id + chunk_id 冲突时 upsert
- 相似度：使用 pgvector 距离计算并返回 score

## 7. 与现有 Agent 的关系

当前步骤实现的是“知识库入库与检索基础能力”。

下一步你可以把 query_pg_knowledge 的检索逻辑封装为工具，接入 Agent 的工具列表，让 Agent 在回答前自动检索知识库。
