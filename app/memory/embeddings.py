"""向量化模块：使用 sentence-transformers 生成语义向量。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_ID = "sentence-transformers-multilingual-minilm-v1"


class EmbeddingManager:
    """基于 sentence-transformers 的语义向量管理器。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        **_unused: object,
    ):
        self.model_name = model_name
        self._model = self._load_model(model_name)
        self.is_fitted = False
        self._cached_vectors: dict[str, np.ndarray] = {}
        self._vector_dim = int(self._model.get_sentence_embedding_dimension())

    @staticmethod
    def _load_model(model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # noqa: BLE001
            raise RuntimeError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            ) from exc
        return SentenceTransformer(model_name)

    def fit(self, texts: list[str]):
        """保留旧接口。sentence-transformers 无需拟合，但要求输入非空。"""
        if not texts:
            raise ValueError("Cannot fit with empty text list")

        self.is_fitted = True
        self._cached_vectors.clear()

    def encode(self, text: str) -> np.ndarray:
        """将文本编码为归一化语义向量。"""
        if not self.is_fitted:
            raise RuntimeError("Embedding model not initialized. Call fit() first.")

        if text in self._cached_vectors:
            return self._cached_vectors[text]

        vec = self._model.encode(str(text or ""), normalize_embeddings=True)
        vec = np.asarray(vec, dtype=np.float32)
        self._vector_dim = int(vec.shape[0]) if vec.ndim == 1 else self._vector_dim

        self._cached_vectors[text] = vec
        return vec

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算归一化向量的余弦相似度。"""
        left = np.asarray(vec1, dtype=np.float32)
        right = np.asarray(vec2, dtype=np.float32)
        return float(np.dot(left, right))

    def bulk_similarity(
        self,
        query_vec: np.ndarray,
        candidate_vecs: list[np.ndarray],
    ) -> list[float]:
        """
        快速批量计算query与多个候选向量的相似度。
        
        Args:
            query_vec: 查询向量
            candidate_vecs: 候选向量列表
            
        Returns:
            相似度得分列表
        """
        similarities = []
        for vec in candidate_vecs:
            sim = self.similarity(query_vec, vec)
            similarities.append(sim)
        return similarities

    def save(self, path: Path):
        """保存 embedding 配置到文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.dumps())

    def load(self, path: Path):
        """从文件加载 embedding 配置。"""
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")

        with open(path, "rb") as f:
            self.loads(f.read())

    def dumps(self) -> bytes:
        """将 embedding 配置序列化为字节。"""
        if not self.is_fitted:
            raise RuntimeError("Embedding model not initialized. Call fit() first.")
        payload = {
            "model_name": self.model_name,
            "model_id": EMBEDDING_MODEL_ID,
            "vector_dim": self.get_vector_dim(),
        }
        return json.dumps(payload, ensure_ascii=True).encode("utf-8")

    def loads(self, blob: bytes):
        """从内存字节反序列化 embedding 配置。"""
        data = json.loads(bytes(blob).decode("utf-8"))
        loaded_model_name = str(data.get("model_name") or DEFAULT_EMBEDDING_MODEL)
        self.model_name = loaded_model_name
        self._model = self._load_model(self.model_name)
        self._vector_dim = int(self._model.get_sentence_embedding_dimension())
        self.is_fitted = True
        self._cached_vectors.clear()

    def get_vocab_size(self) -> int:
        """兼容旧接口：返回向量维度。"""
        if not self.is_fitted:
            return 0
        return self.get_vector_dim()

    def get_vector_dim(self) -> int:
        """获取向量维度。"""
        if not self.is_fitted:
            return 0
        return int(self._vector_dim)


def create_and_fit_embedding_manager(corpus: list[str]) -> EmbeddingManager:
    """
    便捷函数：创建并拟合一个EmbeddingManager。
    
    Args:
        corpus: 用于拟合的文本语料库
        
    Returns:
        拟合后的EmbeddingManager实例
    """
    if not corpus:
        raise ValueError("Corpus cannot be empty")
    
    manager = EmbeddingManager()
    manager.fit(corpus)
    return manager
