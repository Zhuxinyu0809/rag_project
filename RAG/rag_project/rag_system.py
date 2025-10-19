"""
RAG系统模块 - rag_system.py

整合检索器和生成器，实现完整的RAG流程
"""

from typing import List, Dict, Optional
from retrievers.bm25_retriever import BM25Retriever
from generators.api_generator import APIGenerator


class RAGSystem:
    """检索增强生成(RAG)系统"""
    
    def __init__(
        self,
        retriever: BM25Retriever,
        generator: APIGenerator
    ):
        """
        初始化RAG系统
        
        Args:
            retriever: 检索器实例
            generator: 生成器实例
        """
        print("\n🎯 正在初始化RAG系统...")
        
        self.retriever = retriever
        self.generator = generator
        
        print(f"✅ RAG系统初始化完成")
        print(f"   - 检索器: {self.retriever.get_name()}")
        print(f"   - 生成器: {self.generator.get_name()}")
    
    def answer_question(
        self,
        query: str,
        top_k: int = 10,
        model_name: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        回答问题（完整的RAG流程）
        
        Args:
            query: 用户问题
            top_k: 检索文档数量
            model_name: 要使用的模型显示名称（None则使用当前模型）
            verbose: 是否打印详细信息
        
        Returns:
            包含问题、答案、检索文档等信息的字典
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"❓ 问题: {query}")
            print(f"{'='*70}")
        
        # 步骤1: 检索相关文档
        if verbose:
            print(f"🔍 检索前 {top_k} 个相关文档...")
        
        retrieved_docs = self.retriever.search(query, top_k=top_k)
        
        if verbose:
            print(f"✅ 检索完成，找到 {len(retrieved_docs)} 个文档")
        
        # 步骤2: 生成答案
        answer = self.generator.generate(query, retrieved_docs, model_name)
        
        # 返回完整结果
        return {
            'question': query,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'model_used': self.generator.current_model,
            'retriever_used': self.retriever.get_name()
        }
    
    def batch_answer(
        self,
        questions: List[str],
        top_k: int = 10,
        model_name: Optional[str] = None
    ) -> List[Dict]:
        """
        批量回答问题
        
        Args:
            questions: 问题列表
            top_k: 检索文档数量
            model_name: 要使用的模型
        
        Returns:
            结果列表
        """
        results = []
        total = len(questions)
        
        print(f"\n📄 批量处理 {total} 个问题...")
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{total}] 处理中...")
            result = self.answer_question(
                question,
                top_k=top_k,
                model_name=model_name,
                verbose=False
            )
            results.append(result)
            print(f"✅ 完成")
        
        return results
    
    def evaluate(
        self,
        test_samples: List[Dict],
        top_k: int = 10,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        在测试集上评估系统性能
        
        Args:
            test_samples: 测试样本列表
            top_k: 检索文档数量
            model_name: 要使用的模型
        
        Returns:
            评估结果
        """
        results = []
        
        for sample in test_samples:
            result = self.answer_question(
                sample['question'],
                top_k=top_k,
                model_name=model_name,
                verbose=False
            )
            
            # 添加标准答案（如果有）
            if 'answer' in sample:
                result['ground_truth'] = sample['answer']
            
            results.append(result)
        
        return {
            'results': results,
            'total': len(results)
        }