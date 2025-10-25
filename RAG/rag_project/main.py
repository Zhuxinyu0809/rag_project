"""
主程序 - main.py

程序入口，负责初始化各个模块并启动系统
"""

import sys
from data_loader import DataLoader
from retrievers.bm25_retriever import BM25Retriever
from retrievers.qwen_embedding_retriever import QwenEmbeddingRetriever
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
    
    # 步骤2: 选择并初始化检索器
    print("\n【步骤2/4】初始化检索器")
    print("\n请选择检索方法:")
    print("1. BM25 (稀疏检索 - 快速)")
    print("2. Qwen3-Embedding (密集检索 - 语义理解)")
    print("3. 两者都初始化 (可在UI中切换)")
    
    choice = input("\n请输入选项 (1/2/3，默认1): ").strip() or "1"
    
    retrievers = {}
    
    if choice in ["1", "3"]:
        print("\n初始化 BM25 检索器...")
        retrievers["BM25"] = BM25Retriever(documents)
    
    if choice in ["2", "3"]:
        print("\n初始化 Qwen3-Embedding 检索器...")
        print("⚠️  注意：这可能需要5-10分钟为所有文档生成embeddings")
        confirm = input("是否继续? (y/n，默认y): ").strip().lower() or "y"
        
        if confirm == "y":
            try:
                retrievers["Qwen3-Embedding"] = QwenEmbeddingRetriever(documents)
            except Exception as e:
                print(f"\n❌ Qwen3-Embedding初始化失败: {e}")
                print("将只使用BM25检索器")
                if "BM25" not in retrievers:
                    retrievers["BM25"] = BM25Retriever(documents)
    
    if not retrievers:
        print("\n❌ 没有可用的检索器")
        sys.exit(1)
    
    # 使用第一个检索器作为默认
    default_retriever = list(retrievers.values())[0]
    
    # 步骤3: 初始化生成器
    print("\n【步骤3/4】初始化生成器")
    generator = APIGenerator()
    
    # 步骤4: 创建RAG系统
    print("\n【步骤4/4】创建RAG系统")
    rag_system = RAGSystem(
        retriever=default_retriever,
        generator=generator
    )
    
    # 将所有检索器保存到系统中（供UI切换使用）
    rag_system.retrievers = retrievers
    
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
    
    print(f"\n💡 测试答案: {test_result['answer']}")
    
    # 启动Web界面
    launch_interface(rag_system)


if __name__ == "__main__":
    main()