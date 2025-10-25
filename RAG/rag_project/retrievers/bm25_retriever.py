"""
BM25检索器 - 使用自定义TextTokenizer

修改说明：
1. 移除了内部的分词逻辑
2. 使用外部的TextTokenizer进行预处理
3. 更清晰的职责分离
"""

from typing import List, Dict
import bm25s
from TextTokenizer import TextTokenizer
from config import BM25_CONFIG


class BM25Retriever:
    """BM25检索器（使用自定义分词器）"""
    
    def __init__(self, documents: List[Dict]):
        """
        初始化BM25检索器
        
        Args:
            documents: 文档列表，每个文档包含id和text
        """
        print("\n🔍 正在初始化BM25检索器...")
        
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # 初始化自定义分词器
        self.tokenizer = TextTokenizer(
            language=BM25_CONFIG["stemmer_language"]
        )
        
        # 准备文档文本
        corpus_texts = [doc['text'] for doc in documents]
        
        # 使用自定义分词器处理文档
        print("   正在对文档进行预处理和分词...")
        corpus_tokens = self.tokenizer.process(corpus_texts)
        
        # 创建BM25索引
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
        # 使用自定义分词器处理查询
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
        return [self.search(query, top_k) for query in queries]
    
    def get_name(self) -> str:
        """返回检索器名称"""
        return "BM25"