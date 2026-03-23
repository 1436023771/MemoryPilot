"""
向量化embedding模块的单元测试。
"""

from pathlib import Path

import pytest
from app.memory.embeddings import EmbeddingManager, create_and_fit_embedding_manager


class TestEmbeddingManager:
    """测试EmbeddingManager的基础功能。"""

    def test_initialization(self):
        """测试初始化。"""
        manager = EmbeddingManager()
        assert manager.is_fitted is False

    def test_fit_and_encode(self):
        """测试拟合和编码。"""
        corpus = [
            "我叫小李",
            "我喜欢编程",
            "我的目标是成为AI工程师",
        ]
        
        manager = EmbeddingManager()
        manager.fit(corpus)
        
        assert manager.is_fitted is True
        assert manager.get_vocab_size() > 0
        
        # 编码文本
        vec = manager.encode("我叫小李")
        assert vec is not None
        assert vec.shape[0] == manager.get_vector_dim()

    def test_similarity_calculation(self):
        """测试相似度计算。"""
        corpus = [
            "我叫小李，我是小李",
            "我喜欢编程，编程很有趣",
            "今天天气不错，适合出去玩",
        ]
        
        manager = EmbeddingManager()
        manager.fit(corpus)
        
        # 从训练集中直接编码，应该得到比较稳定的向量
        vec1 = manager.encode("我叫小李")
        vec2 = manager.encode("我是小李")
        similarity_same_topic = manager.similarity(vec1, vec2)
        
        # 不同主题的向量相似度应该较低
        vec3 = manager.encode("天气很好")
        similarity_diff_topic = manager.similarity(vec1, vec3)
        
        # 从训练集中的完整句子应该有高自相似度
        vec_full = manager.encode("我叫小李，我是小李")
        vec_full_again = manager.encode("我叫小李，我是小李")
        similarity_same_full = manager.similarity(vec_full, vec_full_again)
        # 验证相似度大小关系
        assert similarity_same_topic >= similarity_diff_topic
        # 相同文本的自相似度应该接近1.0
        assert similarity_same_full >= 0.99

    def test_bulk_similarity(self):
        """测试批量相似度计算。"""
        corpus = [
            "我叫小李",
            "我是小李",
            "我喜欢编程",
            "编程很有趣",
        ]
        
        manager = EmbeddingManager()
        manager.fit(corpus)
        
        query_vec = manager.encode("我是谁")
        candidates = [manager.encode(text) for text in corpus]
        
        similarities = manager.bulk_similarity(query_vec, candidates)
        
        assert len(similarities) == len(corpus)
        assert all(0 <= sim <= 1 for sim in similarities)

    def test_cache(self):
        """测试向量缓存。"""
        corpus = ["我叫小李", "我喜欢编程"]
        manager = EmbeddingManager()
        manager.fit(corpus)
        
        # 第一次编码
        vec1 = manager.encode("我叫小李")
        assert "我叫小李" in manager._cached_vectors
        
        # 第二次应该返回同一个缓存对象
        vec2 = manager.encode("我叫小李")
        assert vec1 is vec2

    def test_create_and_fit_helper(self):
        """测试便捷函数。"""
        corpus = ["我叫小李", "我喜欢编程"]
        manager = create_and_fit_embedding_manager(corpus)
        
        assert manager.is_fitted is True
        vec = manager.encode("测试")
        assert vec is not None

    def test_save_and_load(self, tmp_path: Path):
        """测试保存和加载向量化器。"""
        corpus = ["我叫小李", "我喜欢编程"]
        manager = create_and_fit_embedding_manager(corpus)
        
        # 保存
        save_path = tmp_path / "embedding_model.pkl"
        manager.save(save_path)
        assert save_path.exists()
        
        # 加载
        manager2 = EmbeddingManager()
        manager2.load(save_path)
        
        # 验证加载后的向量化器工作正常
        vec1 = manager.encode("我叫小李")
        vec2 = manager2.encode("我叫小李")
        
        # 应该是相同的向量（或非常接近）
        import numpy as np
        assert np.allclose(vec1, vec2)
