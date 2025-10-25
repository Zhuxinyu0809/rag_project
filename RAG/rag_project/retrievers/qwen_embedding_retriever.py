"""
Qwen3-Embedding检索器 - retrievers/qwen_embedding_retriever.py

使用Qwen3-Embedding模型进行密集检索
"""

from typing import List, Dict
import numpy as np
from openai import OpenAI
from config import DENSE_RETRIEVER_CONFIG, SILICONFLOW_CONFIG


class QwenEmbeddingRetriever:
    """Qwen3-Embedding密集检索器（使用API）"""
    
    def __init__(self, documents: List[Dict]):
        """
        初始化Qwen3-Embedding检索器
        
        Args:
            documents: 文档列表，每个文档包含id和text
        """
        print("\n🔍 正在初始化Qwen3-Embedding检索器...")
        
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        # 初始化API客户端
        api_key = DENSE_RETRIEVER_CONFIG.get("api_key") or SILICONFLOW_CONFIG["api_key"]
        base_url = DENSE_RETRIEVER_CONFIG.get("api_base_url") or SILICONFLOW_CONFIG["base_url"]
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = DENSE_RETRIEVER_CONFIG["api_model"]
        
        # 测试模型是否可用
        print(f"   测试模型: {self.model_name}")
        self._test_embedding()
        
        # 为所有文档生成embeddings
        print(f"   正在为 {len(documents)} 个文档生成embeddings...")
        self.doc_embeddings = self._embed_documents(documents)
        
        print(f"✅ Qwen3-Embedding检索器初始化完成！")
    
    def _test_embedding(self):
        """测试embedding模型是否可用"""
        try:
            test_response = self.client.embeddings.create(
                model=self.model_name,
                input="test",
                encoding_format="float"
            )
            print(f"   ✅ 模型可用")
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ 模型测试失败: {error_msg[:100]}")
            
            # 尝试其他可能的模型名称
            alternative_names = [
                "Qwen/Qwen3-Embedding",
                "qwen3-embedding",
                "Qwen3-Embedding"
            ]
            
            for alt_name in alternative_names:
                if alt_name == self.model_name:
                    continue
                    
                try:
                    print(f"   尝试备用名称: {alt_name}")
                    test_response = self.client.embeddings.create(
                        model=alt_name,
                        input="test",
                        encoding_format="float"
                    )
                    self.model_name = alt_name
                    print(f"   ✅ 使用模型: {alt_name}")
                    return
                except:
                    continue
            
            raise ValueError(f"Qwen3-Embedding模型不可用。请检查API是否支持此模型。")
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        为文本列表生成embeddings
        
        Args:
            texts: 文本列表
        
        Returns:
            embeddings数组 (n_texts, embedding_dim)
        """
        embeddings = []
        batch_size = 100  # 每批处理100个文本
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch,
                    encoding_format="float"
                )
                
                # 提取embeddings
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % 1000 == 0:
                    print(f"      已处理 {min(i + batch_size, len(texts))}/{len(texts)} 个文本")
                
            except Exception as e:
                print(f"   ❌ 批次 {i//batch_size + 1} 失败: {str(e)[:50]}")
                # 失败的批次用零向量
                dim = DENSE_RETRIEVER_CONFIG["dimension"]
                embeddings.extend([[0.0] * dim] * len(batch))
        
        return np.array(embeddings)
    
    def _embed_documents(self, documents: List[Dict]) -> np.ndarray:
        """为所有文档生成embeddings"""
        texts = [doc['text'] for doc in documents]
        return self._embed_texts(texts)
    
    def _cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        计算查询embedding与所有文档embeddings的余弦相似度
        
        Args:
            query_embedding: 查询的embedding向量
        
        Returns:
            相似度分数数组
        """
        # 归一化
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = self.doc_embeddings / (np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-8)
        
        # 计算余弦相似度
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        检索最相关的文档
        
        Args:
            query: 查询文本
            top_k: 返回前K个最相关的文档
        
        Returns:
            检索到的文档列表，包含id、text和score
        """
        # 为查询生成embedding
        query_embedding = self._embed_texts([query])[0]
        
        # 计算相似度
        similarities = self._cosine_similarity(query_embedding)
        
        # 获取top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 整理结果
        retrieved_docs = []
        for idx in top_indices:
            retrieved_docs.append({
                'id': self.doc_ids[idx],
                'text': self.documents[idx]['text'],
                'score': float(similarities[idx])
            })
        
        return retrieved_docs
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[Dict]]:
        """批量检索"""
        return [self.search(query, top_k) for query in queries]
    
    def get_name(self) -> str:
        """返回检索器名称"""
        return "Qwen3-Embedding"
    
    def save_embeddings(self, filepath: str):
        """
        保存embeddings到文件（可选功能）
        
        Args:
            filepath: 保存路径
        """
        np.save(filepath, self.doc_embeddings)
        print(f"✅ Embeddings已保存到: {filepath}")
    
    def load_embeddings(self, filepath: str):
        """
        从文件加载embeddings（可选功能）
        
        Args:
            filepath: 文件路径
        """
        self.doc_embeddings = np.load(filepath)
        print(f"✅ Embeddings已从 {filepath} 加载")