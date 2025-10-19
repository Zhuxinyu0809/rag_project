"""
主程序 - main.py

程序入口，负责初始化各个模块并启动系统
"""

from data_loader import DataLoader
from retrievers.bm25_retriever import BM25Retriever
from generators.api_generator import APIGenerator
from rag_system import RAGSystem
from ui import launch_interface


def main():
    """主函数"""
    
    print("="*70)
    print("🚀 RAG问答系统启动")
    print("="*70)
    
    # 步骤1: 加载数据
    print("\n【步骤1/4】加载数据集")
    data_loader = DataLoader()
    documents = data_loader.get_documents()
    
    # 步骤2: 初始化检索器
    print("\n【步骤2/4】初始化检索器")
    retriever = BM25Retriever(documents)
    
    # 步骤3: 初始化生成器
    print("\n【步骤3/4】初始化生成器")
    generator = APIGenerator()
    
    # 步骤4: 创建RAG系统
    print("\n【步骤4/4】创建RAG系统")
    rag_system = RAGSystem(
        retriever=retriever,
        generator=generator
    )
    
    # 测试系统
    print("\n" + "="*70)
    print("🧪 测试系统")
    print("="*70)
    
    test_question = "Where was Barack Obama born?"
    test_result = rag_system.answer_question(
        query=test_question,
        top_k=10,
        verbose=True
    )
    
    print(f"\n💡 测试答案: {test_result['answer'][:100]}...")
    
    # 启动Web界面
    launch_interface(rag_system)


if __name__ == "__main__":
    main()