"""
主程序 - main.py

程序入口，负责：
1. 加载原始数据。
2. 初始化检索器（Retriever），它会在内部完成文本处理和索引构建。
3. 初始化生成器（Generator）。
4. 将检索器和生成器组装成一个完整的RAG系统。
5. 启动用户界面。
"""

from data_loader import DataLoader
from retrievers.bm25_retriever import BM25Retriever
from generators.api_generator import APIGenerator
from rag_system import RAGSystem
from ui import launch_interface


def main():
    """主函数，编排RAG系统的启动流程"""
    
    # 打印启动横幅
    print("="*70)
    print("🚀 开始启动 RAG 问答系统...")
    print("="*70)
    
    # --- 步骤 1: 加载数据 ---
    # DataLoader 负责从源头（如Hugging Face）获取原始的、未经处理的文档和问答对。
    # 它不关心这些数据后续如何被使用。
    print("\n【步骤 1/4】正在加载数据集...")
    data_loader = DataLoader()
    documents = data_loader.get_documents()
    print(f"   -> 成功加载 {len(documents)} 篇文档。")
    
    # --- 步骤 2: 初始化检索器 ---
    # BM25Retriever 接收原始文档。
    # 在其 __init__ 方法内部，它会自动：
    #   a. 创建一个 TextTokenizer 实例。
    #   b. 使用 TextTokenizer 来清洗、分词和词干化所有文档。
    #   c. 使用处理后的词元（tokens）构建 BM25 索引。
    # main.py 不需要知道这个过程，实现了完美的封装。
    print("\n【步骤 2/4】正在初始化检索器...")
    retriever = BM25Retriever(documents)
    print(f"   -> {retriever.get_name()} 检索器初始化完成。")
    
    # --- 步骤 3: 初始化生成器 ---
    # APIGenerator 负责与外部的 LLM API 进行交互，以生成最终答案。
    print("\n【步骤 3/4】正在初始化生成器...")
    generator = APIGenerator()
    print(f"   -> {generator.get_name()} 生成器初始化完成。")
    
    # --- 步骤 4: 创建并组装 RAG 系统 ---
    # RAGSystem 将检索器和生成器组合在一起，形成一个完整的问答流程。
    print("\n【步骤 4/4】正在创建 RAG 系统...")
    rag_system = RAGSystem(
        retriever=retriever,
        generator=generator
    )
    print("   -> RAG 系统已准备就绪！")
    
    # --- 系统测试 ---
    # 在启动Web界面之前，运行一个简单的测试用例，确保系统工作正常。
    print("\n" + "="*70)
    print("🧪 正在对系统进行快速测试...")
    print("="*70)
    
    test_question = "Where was Barack Obama born?"
    # 调用 RAG 系统的核心方法来回答问题
    # `verbose=True` 会打印出检索过程的中间步骤，便于调试
    test_result = rag_system.answer_question(
        query=test_question,
        top_k=5,  # 检索5个最相关的文档
        verbose=True
    )
    
    # 打印最终生成的答案的摘要
    print(f"\n💡 测试答案 (摘要): {test_result['answer'][:150]}...")
    
    # --- 启动 Web 界面 ---
    # launch_interface 函数会启动一个 Gradio 或 Streamlit 应用，
    # 将 RAG 系统暴露给最终用户。
    print("\n" + "="*70)
    print("🖥️  所有组件初始化成功，正在启动 Web 用户界面...")
    print("="*70)
    launch_interface(rag_system)


if __name__ == "__main__":
    # 当脚本被直接执行时，运行主函数
    main()