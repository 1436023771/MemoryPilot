from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class KnowledgeChunk:
    document_id: str
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    book_id: str = ""
    chapter: str = ""
    section: str = ""


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_identifier(name: str) -> str:
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"Invalid SQL identifier: {name}")
    return name


class PgVectorKnowledgeStore:
    """Store and retrieve document chunks using PostgreSQL + pgvector."""

    def __init__(self, dsn: str, table_name: str = "knowledge_chunks", embedding_dim: int = 384):
        if not dsn.strip():
            raise ValueError("dsn is required")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")

        self.dsn = dsn
        self.table_name = _validate_identifier(table_name)
        self.embedding_dim = embedding_dim

    def _connect(self):
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "psycopg and pgvector are required. Install with: pip install 'psycopg[binary]' pgvector"
            ) from exc

        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def init_schema(self) -> None:
        create_sql = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id BIGSERIAL PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            book_id TEXT NOT NULL DEFAULT '',
            chapter TEXT NOT NULL DEFAULT '',
            section TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding VECTOR({self.embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(document_id, chunk_id)
        );
        """
        alter_sql = [
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS book_id TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS chapter TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS section TEXT NOT NULL DEFAULT '';",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_book_id ON {self.table_name}(book_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_chapter ON {self.table_name}(chapter);",
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                for sql in alter_sql:
                    cur.execute(sql)
            conn.commit()

    def clear_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
            conn.commit()

    def upsert_chunks(self, chunks: list[KnowledgeChunk], embeddings: list[list[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have same length")
        if not chunks:
            return 0

        try:
            from psycopg.types.json import Json
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError("psycopg is required") from exc

        sql = f"""
        INSERT INTO {self.table_name}
            (document_id, chunk_id, book_id, chapter, section, content, metadata, embedding, updated_at)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (document_id, chunk_id)
        DO UPDATE SET
            book_id = EXCLUDED.book_id,
            chapter = EXCLUDED.chapter,
            section = EXCLUDED.section,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            embedding = EXCLUDED.embedding,
            updated_at = NOW();
        """

        params = []
        for chunk, emb in zip(chunks, embeddings):
            if len(emb) != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(emb)}"
                )
            params.append(
                (
                    chunk.document_id,
                    chunk.chunk_id,
                    chunk.book_id,
                    chunk.chapter,
                    chunk.section,
                    chunk.content,
                    Json(chunk.metadata or {}),
                    emb,
                )
            )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params)
            conn.commit()

        return len(params)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
        book_id: str | None = None,
        chapter: str | None = None,
    ) -> list[dict[str, Any]]:
        if not query_embedding:
            return []
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {len(query_embedding)}"
            )
        if top_k <= 0:
            return []

        try:
            from pgvector import Vector
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError("pgvector is required") from exc

        query_vec = Vector(query_embedding)

        where_parts: list[str] = []
        params: list[Any] = []
        if document_id:
            where_parts.append("document_id = %s")
            params.append(document_id)
        if book_id:
            where_parts.append("book_id = %s")
            params.append(book_id)
        if chapter:
            where_parts.append("chapter = %s")
            params.append(chapter)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        sql = f"""
        SELECT
            document_id,
            chunk_id,
            book_id,
            chapter,
            section,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM {self.table_name}
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        params = [*params, query_vec, query_vec, top_k]

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "document_id": row[0],
                    "chunk_id": row[1],
                    "book_id": row[2],
                    "chapter": row[3],
                    "section": row[4],
                    "content": row[5],
                    "metadata": row[6],
                    "score": float(row[7]),
                }
            )
        return results

    def get_chunk_with_context(
        self,
        document_id: str,
        chunk_id: str,
        window: int = 2,
        max_context_chars: int = 2200,
    ) -> dict[str, Any]:
        """Return matched chunk and nearby chunks for longer reading context."""
        if not document_id.strip() or not chunk_id.strip():
            return {
                "matched_chunk_id": chunk_id,
                "matched_content": "",
                "context_chunk_ids": [],
                "context_content": "",
            }

        safe_window = max(0, int(window))

        sql = f"""
        SELECT chunk_id, content
        FROM {self.table_name}
        WHERE document_id = %s
        ORDER BY chunk_id ASC;
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [document_id])
                rows = cur.fetchall()

        if not rows:
            return {
                "matched_chunk_id": chunk_id,
                "matched_content": "",
                "context_chunk_ids": [],
                "context_content": "",
            }

        ids = [str(row[0]) for row in rows]
        contents = [str(row[1]) for row in rows]

        try:
            center = ids.index(chunk_id)
        except ValueError:
            return {
                "matched_chunk_id": chunk_id,
                "matched_content": "",
                "context_chunk_ids": [],
                "context_content": "",
            }

        start = max(0, center - safe_window)
        end = min(len(rows), center + safe_window + 1)

        context_ids = ids[start:end]
        context_blocks = [c.strip() for c in contents[start:end] if c and c.strip()]
        merged_context = "\n\n".join(context_blocks).strip()
        if len(merged_context) > max_context_chars:
            merged_context = merged_context[:max_context_chars].rstrip() + "..."

        return {
            "matched_chunk_id": ids[center],
            "matched_content": contents[center].strip(),
            "context_chunk_ids": context_ids,
            "context_content": merged_context,
        }
