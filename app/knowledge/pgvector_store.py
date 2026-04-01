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
    chunk_order: int = 0
    timeline_order: int = 0
    scene_id: str = ""
    event_id: str = ""
    narrative_context: str = ""
    time_markers: list[str] | None = None
    character_mentions: list[str] | None = None
    relationship_edges: list[dict[str, Any]] | None = None


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
            chunk_order BIGINT NOT NULL DEFAULT 0,
            book_id TEXT NOT NULL DEFAULT '',
            chapter TEXT NOT NULL DEFAULT '',
            section TEXT NOT NULL DEFAULT '',
            scene_id TEXT NOT NULL DEFAULT '',
            event_id TEXT NOT NULL DEFAULT '',
            timeline_order INT NOT NULL DEFAULT 0,
            narrative_context TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            time_markers JSONB NOT NULL DEFAULT '[]'::jsonb,
            character_mentions JSONB NOT NULL DEFAULT '[]'::jsonb,
            relationship_edges JSONB NOT NULL DEFAULT '[]'::jsonb,
            embedding VECTOR({self.embedding_dim}) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(document_id, chunk_id)
        );
        """
        alter_sql = [
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS chunk_order BIGINT NOT NULL DEFAULT 0;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS book_id TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS chapter TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS section TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS scene_id TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS event_id TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS timeline_order INT NOT NULL DEFAULT 0;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS narrative_context TEXT NOT NULL DEFAULT '';",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS time_markers JSONB NOT NULL DEFAULT '[]'::jsonb;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS character_mentions JSONB NOT NULL DEFAULT '[]'::jsonb;",
            f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS relationship_edges JSONB NOT NULL DEFAULT '[]'::jsonb;",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_book_id ON {self.table_name}(book_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_chapter ON {self.table_name}(chapter);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_doc_chunk_order ON {self.table_name}(document_id, chunk_order);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_book_timeline ON {self.table_name}(book_id, timeline_order);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_book_scene ON {self.table_name}(book_id, scene_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_book_event ON {self.table_name}(book_id, event_id);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_narrative_context ON {self.table_name}(narrative_context);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_metadata_gin ON {self.table_name} USING GIN (metadata);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_character_mentions_gin ON {self.table_name} USING GIN (character_mentions);",
            f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_relationship_edges_gin ON {self.table_name} USING GIN (relationship_edges);",
        ]
        graph_schema_sql = [
            """
            CREATE TABLE IF NOT EXISTS character_nodes (
                id BIGSERIAL PRIMARY KEY,
                character_id TEXT NOT NULL UNIQUE,
                display_name TEXT NOT NULL,
                aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
                attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS event_nodes (
                id BIGSERIAL PRIMARY KEY,
                event_id TEXT NOT NULL UNIQUE,
                book_id TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                summary TEXT NOT NULL DEFAULT '',
                timeline_order INT NOT NULL DEFAULT 0,
                scene_id TEXT NOT NULL DEFAULT '',
                chapter TEXT NOT NULL DEFAULT '',
                attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS character_event_edges (
                id BIGSERIAL PRIMARY KEY,
                character_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                role_type TEXT NOT NULL DEFAULT '',
                confidence REAL NOT NULL DEFAULT 0.0,
                evidence_chunk_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(character_id, event_id, role_type)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS character_character_edges (
                id BIGSERIAL PRIMARY KEY,
                src_character_id TEXT NOT NULL,
                dst_character_id TEXT NOT NULL,
                relation_type TEXT NOT NULL DEFAULT '',
                polarity SMALLINT NOT NULL DEFAULT 0,
                strength REAL NOT NULL DEFAULT 0.0,
                first_timeline_order INT NOT NULL DEFAULT 0,
                last_timeline_order INT NOT NULL DEFAULT 0,
                evidence_chunk_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
                attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(src_character_id, dst_character_id, relation_type)
            );
            """,
            "CREATE INDEX IF NOT EXISTS idx_character_event_edges_character ON character_event_edges(character_id);",
            "CREATE INDEX IF NOT EXISTS idx_character_event_edges_event ON character_event_edges(event_id);",
            "CREATE INDEX IF NOT EXISTS idx_character_character_edges_src ON character_character_edges(src_character_id);",
            "CREATE INDEX IF NOT EXISTS idx_character_character_edges_dst ON character_character_edges(dst_character_id);",
            "CREATE INDEX IF NOT EXISTS idx_event_nodes_book_timeline ON event_nodes(book_id, timeline_order);",
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
                for sql in alter_sql:
                    cur.execute(sql)
                for sql in graph_schema_sql:
                    cur.execute(sql)
            conn.commit()

    def clear_table(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.table_name};")
            conn.commit()

    def delete_by_document_id(self, document_id: str) -> int:
        clean = (document_id or "").strip()
        if not clean:
            return 0
        sql = f"DELETE FROM {self.table_name} WHERE document_id = %s;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [clean])
                deleted = int(getattr(cur, "rowcount", 0) or 0)
            conn.commit()
        return deleted

    def count_by_document_id(self, document_id: str) -> int:
        clean = (document_id or "").strip()
        if not clean:
            return 0
        sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE document_id = %s;"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, [clean])
                row = cur.fetchone()
        if not row:
            return 0
        return int(row[0] or 0)

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
            (
                document_id,
                chunk_id,
                chunk_order,
                book_id,
                chapter,
                section,
                scene_id,
                event_id,
                timeline_order,
                narrative_context,
                content,
                metadata,
                time_markers,
                character_mentions,
                relationship_edges,
                embedding,
                updated_at
            )
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (document_id, chunk_id)
        DO UPDATE SET
            chunk_order = EXCLUDED.chunk_order,
            book_id = EXCLUDED.book_id,
            chapter = EXCLUDED.chapter,
            section = EXCLUDED.section,
            scene_id = EXCLUDED.scene_id,
            event_id = EXCLUDED.event_id,
            timeline_order = EXCLUDED.timeline_order,
            narrative_context = EXCLUDED.narrative_context,
            content = EXCLUDED.content,
            metadata = EXCLUDED.metadata,
            time_markers = EXCLUDED.time_markers,
            character_mentions = EXCLUDED.character_mentions,
            relationship_edges = EXCLUDED.relationship_edges,
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
                    int(chunk.chunk_order),
                    chunk.book_id,
                    chunk.chapter,
                    chunk.section,
                    chunk.scene_id,
                    chunk.event_id,
                    int(chunk.timeline_order),
                    chunk.narrative_context,
                    chunk.content,
                    Json(chunk.metadata or {}),
                    Json(chunk.time_markers or []),
                    Json(chunk.character_mentions or []),
                    Json(chunk.relationship_edges or []),
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
        timeline_order_min: int | None = None,
        timeline_order_max: int | None = None,
        scene_id: str | None = None,
        event_id: str | None = None,
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
        filter_params: list[Any] = []
        if document_id:
            where_parts.append("document_id = %s")
            filter_params.append(document_id)
        if book_id:
            where_parts.append("book_id = %s")
            filter_params.append(book_id)
        if chapter:
            where_parts.append("chapter = %s")
            filter_params.append(chapter)
        if timeline_order_min is not None:
            where_parts.append("timeline_order >= %s")
            filter_params.append(int(timeline_order_min))
        if timeline_order_max is not None:
            where_parts.append("timeline_order <= %s")
            filter_params.append(int(timeline_order_max))
        if scene_id:
            where_parts.append("scene_id = %s")
            filter_params.append(scene_id)
        if event_id:
            where_parts.append("event_id = %s")
            filter_params.append(event_id)

        where_clause = ""
        if where_parts:
            where_clause = "WHERE " + " AND ".join(where_parts)

        sql = f"""
        SELECT
            document_id,
            chunk_id,
            chunk_order,
            book_id,
            chapter,
            section,
            scene_id,
            event_id,
            timeline_order,
            narrative_context,
            content,
            metadata,
            time_markers,
            character_mentions,
            relationship_edges,
            1 - (embedding <=> %s::vector) AS score
        FROM {self.table_name}
        {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
        """

        params = [query_vec, *filter_params, query_vec, top_k]

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
                    "chunk_order": int(row[2]),
                    "book_id": row[3],
                    "chapter": row[4],
                    "section": row[5],
                    "scene_id": row[6],
                    "event_id": row[7],
                    "timeline_order": int(row[8]),
                    "narrative_context": row[9],
                    "content": row[10],
                    "metadata": row[11],
                    "time_markers": row[12],
                    "character_mentions": row[13],
                    "relationship_edges": row[14],
                    "score": float(row[15]),
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
        ORDER BY
            CASE WHEN chunk_order > 0 THEN 0 ELSE 1 END ASC,
            chunk_order ASC,
            CASE WHEN chunk_id ~ '^[0-9]+$' THEN chunk_id::BIGINT ELSE 9223372036854775807 END ASC,
            chunk_id ASC;
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

    def get_character_candidates(self, book_id: str | None = None, limit: int = 200) -> list[str]:
        """Return distinct character candidate names for query understanding.

        Candidates are merged from:
        - normalized graph table `character_nodes.display_name`
        - chunk-level JSONB field `character_mentions`
        """
        safe_limit = max(1, int(limit))
        names: list[str] = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                # Source 1: character_nodes (if available)
                try:
                    cur.execute(
                        "SELECT to_regclass('public.character_nodes') IS NOT NULL;"
                    )
                    exists_row = cur.fetchone()
                    exists = bool(exists_row[0]) if exists_row else False
                except Exception:  # noqa: BLE001
                    exists = False

                if exists:
                    cur.execute(
                        """
                        SELECT display_name
                        FROM character_nodes
                        WHERE COALESCE(display_name, '') <> ''
                        ORDER BY updated_at DESC, id DESC
                        LIMIT %s;
                        """,
                        [safe_limit],
                    )
                    for row in cur.fetchall():
                        names.append(str(row[0]).strip())

                # Source 2: chunk metadata field character_mentions
                if book_id:
                    cur.execute(
                        f"""
                        SELECT DISTINCT elem
                        FROM {self.table_name} t,
                             LATERAL jsonb_array_elements_text(COALESCE(t.character_mentions, '[]'::jsonb)) AS elem
                        WHERE t.book_id = %s
                          AND COALESCE(elem, '') <> ''
                        LIMIT %s;
                        """,
                        [book_id, safe_limit],
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT DISTINCT elem
                        FROM {self.table_name} t,
                             LATERAL jsonb_array_elements_text(COALESCE(t.character_mentions, '[]'::jsonb)) AS elem
                        WHERE COALESCE(elem, '') <> ''
                        LIMIT %s;
                        """,
                        [safe_limit],
                    )
                for row in cur.fetchall():
                    names.append(str(row[0]).strip())

        deduped: list[str] = []
        seen: set[str] = set()
        for name in names:
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(name)
            if len(deduped) >= safe_limit:
                break
        return deduped
