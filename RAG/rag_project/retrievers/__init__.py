"""
检索器模块

提供各种文档检索方法
"""

from .bm25_retriever import BM25Retriever

# 未来可以添加：
# from .dense_retriever import DenseRetriever
# from .hybrid_retriever import HybridRetriever

__all__ = [
    'BM25Retriever',
    # 'DenseRetriever',
    # 'HybridRetriever',
]