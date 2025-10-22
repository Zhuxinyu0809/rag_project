# evaluate_rag.py

"""
端到端 RAG 系统性能评估模块

负责计算 RAG 系统在生成答案上的性能指标，主要是 EM (Exact Match) 和 F1-score。
这与只评估检索器的 evaluate.py 不同。
"""

import re
import collections
from tqdm import tqdm
from typing import List, Dict

# 导入RAG系统的所有组件
from data_loader import DataLoader
from retrievers.bm25_retriever import BM25Retriever
from generators.api_generator import APIGenerator
from rag_system import RAGSystem

# ==============================================================================
# 辅助函数: 用于计算 EM 和 F1 分数
# ==============================================================================

def normalize_text(s: str) -> str:
    """
    文本标准化：小写、移除标点、移除冠词。
    """
    s = s.lower()
    s = re.sub(r'[^\w\s]', '', s) # 移除标点
    s = re.sub(r'\b(a|an|the)\b', ' ', s) # 移除冠词
    return ' '.join(s.split())

def calculate_f1(prediction: str, ground_truth: str) -> float:
    """
    计算两个字符串之间的 F1 分数。
    """
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    
    if not prediction_tokens or not ground_truth_tokens:
        return 1.0 if prediction_tokens == ground_truth_tokens else 0.0

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def calculate_em(prediction: str, ground_truth: str) -> float:
    """
    计算两个字符串是否完全匹配 (Exact Match)。
    """
    return 1.0 if normalize_text(prediction) == normalize_text(ground_truth) else 0.0

# ==============================================================================
# 主评估函数
# ==============================================================================

def evaluate_rag_system(rag_system: RAGSystem, data_loader: DataLoader, top_k_for_retrieval: int = 10):
    """
    在整个验证集上评估 RAG 系统的端到端性能。

    Args:
        rag_system (RAGSystem): 一个已经初始化好的 RAG 系统实例。
        data_loader (DataLoader): 数据加载器实例。
        top_k_for_retrieval (int): 在RAG流程中，检索器应该返回多少个文档。
    """
    print(f"\n" + "="*70)
    print(f"📊 开始端到端评估 RAG 系统")
    print(f"   评估指标: Exact Match (EM), F1-Score")
    print(f"   检索 Top-K: {top_k_for_retrieval}")
    print("="*70)

    eval_samples = data_loader.get_validation_samples()
    if not eval_samples:
        print("❌ 无法评估：验证集为空。")
        return

    total_f1 = 0.0
    total_em = 0.0

    # 遍历验证集中的每一个问题
    for sample in tqdm(eval_samples, desc="Evaluating RAG System"):
        question = sample['question']
        ground_truth_answer = sample['answer']
        
        # 使用 RAG 系统生成答案
        # 注意：这里我们不使用 verbose=True，以加快批量处理速度
        result = rag_system.answer_question(query=question, top_k=top_k_for_retrieval)
        predicted_answer = result['answer']
        
        # 计算 F1 和 EM
        total_f1 += calculate_f1(predicted_answer, ground_truth_answer)
        total_em += calculate_em(predicted_answer, ground_truth_answer)

    # 计算平均分数
    num_samples = len(eval_samples)
    avg_f1 = (total_f1 / num_samples) * 100
    avg_em = (total_em / num_samples) * 100

    # 打印评估结果报告
    print(f"\n" + "="*70)
    print(f"✅ 评估完成！在 {num_samples} 个验证样本上的端到端表现:")
    print("="*70)
    print(f"  - F1-Score: {avg_f1:.2f}%")
    print(f"  - Exact Match (EM): {avg_em:.2f}%")
    print("="*70)


if __name__ == "__main__":
    # --- 脚本主入口 ---
    # 1. 加载数据
    print("【步骤 1/5】加载数据...")
    data_loader = DataLoader()
    documents = data_loader.get_documents()
    
    # 2. 初始化检索器
    print("\n【步骤 2/5】初始化 BM25 检索器...")
    bm25_retriever = BM25Retriever(documents)

    # 3. 初始化生成器
    print("\n【步骤 3/5】初始化生成器...")
    generator = APIGenerator()

    # 4. 创建 RAG 系统
    print("\n【步骤 4/5】创建 RAG 系统...")
    rag_system = RAGSystem(
        retriever=bm25_retriever,
        generator=generator
    )

    # 5. 运行评估
    print("\n【步骤 5/5】开始端到端评估流程...")
    # 在这里，你可以调整 top_k_for_retrieval 来观察不同数量的检索文档对最终答案质量的影响
    evaluate_rag_system(rag_system, data_loader, top_k_for_retrieval=10)