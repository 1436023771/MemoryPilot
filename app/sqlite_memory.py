from __future__ import annotations

import sqlite3
from pathlib import Path
import pickle
import re

from app.embeddings import EmbeddingManager
from app.read_only_memory import MemoryChunk


def _connect(db_path: Path) -> sqlite3.Connection:
    """连接 SQLite 数据库并返回连接对象。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_memory_db(db_path: Path) -> None:
    """初始化长期记忆表结构。"""
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(fact_key, fact_value)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL UNIQUE,
                embedding_blob BLOB NOT NULL,
                vector_dim INTEGER NOT NULL,
                vector_model TEXT NOT NULL DEFAULT 'tfidf-jieba-v1',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(fact_id) REFERENCES memory_facts(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embedding_meta (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                vectorizer_blob BLOB NOT NULL,
                vector_model TEXT NOT NULL DEFAULT 'tfidf-jieba-v1',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _rebuild_embeddings(conn: sqlite3.Connection) -> None:
    """基于当前 memory_facts 全量重建 TF-IDF 向量并写入 memory_embeddings。"""
    rows = conn.execute(
        """
        SELECT id, fact_text
        FROM memory_facts
        ORDER BY id ASC
        """
    ).fetchall()

    # 清空旧向量，保证词表变化时维度一致。
    conn.execute("DELETE FROM memory_embeddings")

    if not rows:
        return

    fact_ids = [row[0] for row in rows]
    texts = [row[1] for row in rows]

    # SQLite 里可能只有 1 条事实；此时 max_df 需为 1.0 避免阈值冲突。
    manager = EmbeddingManager(max_df=1.0)
    manager.fit(texts)

    conn.execute(
        """
        INSERT INTO memory_embedding_meta (id, vectorizer_blob, vector_model, updated_at)
        VALUES (1, ?, 'tfidf-jieba-v1', CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET
            vectorizer_blob = excluded.vectorizer_blob,
            vector_model = excluded.vector_model,
            updated_at = CURRENT_TIMESTAMP
        """,
        (manager.dumps(),),
    )

    for fact_id, text in zip(fact_ids, texts):
        vector = manager.encode(text)
        blob = pickle.dumps(vector)
        conn.execute(
            """
            INSERT INTO memory_embeddings (fact_id, embedding_blob, vector_dim, vector_model, updated_at)
            VALUES (?, ?, ?, 'tfidf-jieba-v1', CURRENT_TIMESTAMP)
            """,
            (fact_id, blob, int(vector.shape[0])),
        )


def load_memory_chunks_from_sqlite(db_path: Path) -> list[MemoryChunk]:
    """从 SQLite 读取长期记忆，返回可检索片段列表。"""
    init_memory_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT fact_text
            FROM memory_facts
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()

    return [MemoryChunk(text=row[0]) for row in rows]


def write_facts_to_sqlite(db_path: Path, facts: list) -> list[str]:
    """将事实写入 SQLite，并在单值字段冲突时执行覆盖更新。
    
    接收 list[MemoryFact]（结构化对象），直接写入而不需要文本解析。
    """
    if not facts:
        return []

    init_memory_db(db_path)
    written: list[str] = []
    single_value_keys = {"name", "goal"}

    with _connect(db_path) as conn:
        for fact in facts:
            # 直接从 MemoryFact 对象提取
            key = getattr(fact, "key", None)
            value = getattr(fact, "value", None)
            text = getattr(fact, "text", None)
            
            if not key or not value or not text:
                continue

            if key in single_value_keys:
                existing = conn.execute(
                    "SELECT fact_value FROM memory_facts WHERE fact_key = ? ORDER BY id DESC LIMIT 1",
                    (key,),
                ).fetchone()
                if existing and existing[0] == value:
                    continue

                conn.execute("DELETE FROM memory_facts WHERE fact_key = ?", (key,))
                conn.execute(
                    """
                    INSERT INTO memory_facts (fact_key, fact_value, fact_text, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (key, value, text),
                )
                written.append(text)
                continue

            existing_multi = conn.execute(
                "SELECT 1 FROM memory_facts WHERE fact_key = ? AND fact_value = ?",
                (key, value),
            ).fetchone()
            if existing_multi:
                continue

            conn.execute(
                """
                INSERT INTO memory_facts (fact_key, fact_value, fact_text, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (key, value, text),
            )
            written.append(text)

        # 第二阶段：写入完成后同步构建向量记忆。
        _rebuild_embeddings(conn)
        conn.commit()

    return written


def load_embeddings_count(db_path: Path) -> int:
    """返回当前已存储的向量条数，用于测试与健康检查。"""
    init_memory_db(db_path)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()
    return int(row[0]) if row else 0


def _tokenize_for_keyword(text: str) -> set[str]:
    """关键词召回分词：英文词 + 单字中文。"""
    normalized = text.lower()
    word_tokens = set(re.findall(r"[a-z0-9]+", normalized))
    cjk_tokens = {ch for ch in normalized if "\u4e00" <= ch <= "\u9fff"}
    return word_tokens | cjk_tokens


def retrieve_memory_context_hybrid_from_sqlite(
    db_path: Path,
    query: str,
    top_k: int = 3,
    vector_weight: float = 0.7,
) -> str:
    """混合检索：关键词分数 + 向量相似度分数加权排序。"""
    init_memory_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT f.id, f.fact_text, e.embedding_blob
            FROM memory_facts f
            LEFT JOIN memory_embeddings e ON e.fact_id = f.id
            ORDER BY f.updated_at DESC, f.id DESC
            """
        ).fetchall()
        meta = conn.execute(
            "SELECT vectorizer_blob FROM memory_embedding_meta WHERE id = 1"
        ).fetchone()

    if not rows:
        return ""

    # 关键词得分
    query_tokens = _tokenize_for_keyword(query)
    keyword_scores: dict[int, float] = {}
    for fact_id, fact_text, _ in rows:
        overlap = len(query_tokens & _tokenize_for_keyword(fact_text))
        keyword_scores[fact_id] = float(overlap)

    # 向量得分
    vector_scores: dict[int, float] = {fact_id: 0.0 for fact_id, _, _ in rows}
    if meta and meta[0]:
        manager = EmbeddingManager(max_df=1.0)
        manager.loads(meta[0])
        query_vec = manager.encode(query)

        for fact_id, _, embedding_blob in rows:
            if not embedding_blob:
                continue
            vec = pickle.loads(embedding_blob)
            vector_scores[fact_id] = max(0.0, manager.similarity(query_vec, vec))

    # 归一化，避免某一分量量纲主导
    max_keyword = max(keyword_scores.values()) if keyword_scores else 0.0
    max_vector = max(vector_scores.values()) if vector_scores else 0.0

    scored: list[tuple[float, str]] = []
    for fact_id, fact_text, _ in rows:
        k = keyword_scores.get(fact_id, 0.0)
        v = vector_scores.get(fact_id, 0.0)
        k_norm = (k / max_keyword) if max_keyword > 0 else 0.0
        v_norm = (v / max_vector) if max_vector > 0 else 0.0
        hybrid = vector_weight * v_norm + (1.0 - vector_weight) * k_norm
        if hybrid > 0:
            scored.append((hybrid, fact_text))

    if not scored:
        return ""

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = scored[: max(1, top_k)]
    return "\n".join(f"- {text}" for _, text in selected)
