"""
向量化模块：使用TF-IDF实现轻量级的中文文本向量化和相似度计算。
无需PyTorch依赖，适合Python 3.13环境。
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class EmbeddingManager:
    """
    基于TF-IDF的轻量级embedding管理器。
    优点：
    - 无PyTorch/transformers依赖，轻量级
    - 支持中文分词（jieba）
    - 快速向量化和相似度计算
    
    缺点：
    - 相比预训练NLP模型，语义理解能力较弱
    - 需要建立全局词汇表
    """

    def __init__(
        self,
        max_features: int = 5000,
        min_df: int = 1,
        max_df: float = 0.95,
    ):
        """
        初始化EmbeddingManager。
        
        Args:
            max_features: 最多保留的词汇数
            min_df: 最少文档频率（词最少要出现在N个文档中）
            max_df: 最大文档频率（词最多出现在该比例的文档中）
        """
        self.vectorizer = TfidfVectorizer(
            analyzer=self._tokenize,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            norm="l2",  # L2归一化，便于余弦相似度计算
        )
        self.is_fitted = False
        self._cached_vectors: dict[str, np.ndarray] = {}

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        将文本分词，支持中英文混合。
        
        Args:
            text: 输入文本
            
        Returns:
            分词结果列表
        """
        # 使用 jieba 分词，保持中文词汇完整
        tokens = jieba.cut(text.lower())
        
        # 过滤掉单字符中文和特殊符号（保留有意义的词）
        filtered = []
        for token in tokens:
            token = token.strip()
            if not token or len(token) == 0:
                continue
            # 过滤掉纯符号或单个汉字（除非是常见单字如"是"、"的"）
            if len(token) == 1 and ord(token) >= 0x4e00 and ord(token) <= 0x9fff:
                # 单个汉字 - 保留某些常见的
                if token not in {"的", "是", "在", "了", "个"}:
                    continue
            filtered.append(token)
        
        return filtered

    def fit(self, texts: list[str]):
        """
        基于输入文本拟合TF-IDF词汇表。
        必须在encode前调用一次。
        
        Args:
            texts: 用于拟合的文本列表
        """
        if not texts:
            raise ValueError("Cannot fit with empty text list")
        
        self.vectorizer.fit(texts)
        self.is_fitted = True
        self._cached_vectors.clear()

    def encode(self, text: str) -> np.ndarray:
        """
        将文本编码为TF-IDF向量。
        
        Args:
            text: 输入文本
            
        Returns:
            稀疏矩阵转换的稠密向量 (float32)
        """
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        
        # 检查缓存
        if text in self._cached_vectors:
            return self._cached_vectors[text]
        
        # 向量化
        vec = self.vectorizer.transform([text]).toarray()[0].astype(np.float32)
        
        # 缓存
        self._cached_vectors[text] = vec
        return vec

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度。
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            相似度得分 [0, 1]
        """
        # L2 归一化下的余弦相似度等于向量点积
        return float(np.dot(vec1, vec2))

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
        """保存向量化器状态到文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    def load(self, path: Path):
        """从文件加载向量化器状态。"""
        if not path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {path}")
        
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.is_fitted = True

    def dumps(self) -> bytes:
        """将向量化器序列化为内存字节。"""
        if not self.is_fitted:
            raise RuntimeError("Vectorizer not fitted. Call fit() first.")
        return pickle.dumps(self.vectorizer)

    def loads(self, blob: bytes):
        """从内存字节反序列化向量化器。"""
        self.vectorizer = pickle.loads(blob)
        self.is_fitted = True
        self._cached_vectors.clear()

    def get_vocab_size(self) -> int:
        """获取当前词汇表大小。"""
        if not self.is_fitted:
            return 0
        return len(self.vectorizer.get_feature_names_out())

    def get_vector_dim(self) -> int:
        """获取向量维度。"""
        if not self.is_fitted:
            return 0
        # TF-IDF向量维度 = 词汇表大小
        return len(self.vectorizer.get_feature_names_out())


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
