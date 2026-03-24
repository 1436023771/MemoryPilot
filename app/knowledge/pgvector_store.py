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
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding VECTOR({self.embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(document_id, chunk_id)
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
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
            (document_id, chunk_id, content, metadata, embedding, updated_at)
        VALUES
            (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (document_id, chunk_id)
        DO UPDATE SET
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

        where_clause = ""
        params: list[Any] = []
        if document_id:
            where_clause = "WHERE document_id = %s"
            params.append(document_id)

        sql = f"""
        SELECT
            document_id,
            chunk_id,
            content,
            metadata,
            1 - (embedding <=> %s::vector) AS score
        FROM {self.table_name}
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        params = [*params, query_vec, query_vec, top_k]
        # If where clause exists, reorder params to match SQL positions.
        if document_id:
            params = [document_id, query_vec, query_vec, top_k]

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
                    "content": row[2],
                    "metadata": row[3],
                    "score": float(row[4]),
                }
            )
        return results
