# retrievers/bm25_retriever.py

"""
BM25检索器模块 - retrievers/bm25_retriever.py

实现基于BM25算法的文档检索
"""

from typing import List, Dict
import bm25s
from config import BM25_CONFIG
# 导入我们自己编写的 TextTokenizer
from TextTokenizer import TextTokenizer


class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self, documents: List[Dict]):
        """
        初始化BM25检索器
        
        Args:
            documents: 文档列表，每个文档包含id和text
        """
        print("\n🔍 正在初始化BM25检索器...")
        
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # 1. 初始化我们自定义的分词器
        # 注意: 我们从 BM25_CONFIG 中获取语言，而不是硬编码
        self.tokenizer = TextTokenizer(language=BM25_CONFIG["stemmer_language"])
        
        # 准备文档文本
        corpus_texts = [doc['text'] for doc in documents]
        
        # 2. 使用我们自己的分词器对文档进行分词
        print("   正在使用自定义分词器对文档进行分词...")
        corpus_tokens = self.tokenizer.process(corpus_texts)
        
        # 3. 创建BM25索引
        print("   正在创建BM25索引...")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        
        print(f"✅ BM25索引创建完成！索引了 {len(documents)} 个文档")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回前K个最相关的文档
        
        Returns:
            检索到的文档列表，包含id、text和score
        """
        # 使用自定义分词器对查询进行分词
        query_tokens = self.tokenizer.process(query)
        
        # 检索
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        
        # 整理结果
        retrieved_docs = []
        for idx, score in zip(results[0], scores[0]):
            retrieved_docs.append({
                'id': self.doc_ids[idx],
                'text': self.documents[idx]['text'],
                'score': float(score)
            })
        
        return retrieved_docs
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict]]:
        """
        批量检索
        
        Args:
            queries: 查询列表
            top_k: 每个查询返回前K个文档
        
        Returns:
            检索结果列表的列表
        """
        # 使用自定义分词器进行批量分词
        query_tokens = self.tokenizer.process(queries)
        
        # 批量检索
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        
        # 整理所有查询的结果
        all_results = []
        for i in range(len(queries)):
            retrieved_docs = []
            for idx, score in zip(results[i], scores[i]):
                retrieved_docs.append({
                    'id': self.doc_ids[idx],
                    'text': self.documents[idx]['text'],
                    'score': float(score)
                })
            all_results.append(retrieved_docs)
            
        return all_results
    
    def get_name(self) -> str:
        """返回检索器名称"""
        return "BM25"