"""
主程序入口 - main.py

整合所有模块，启动RAG系统
"""

from data_loader import DataLoader
from rag_system import RAGSystem
from ui import create_interface

# ============================================================================
# 主程序
# ============================================================================
def main():
    print("\n" + "="*70)
    print("🚀 RAG系统启动（可选择模型版本）")
    print("="*70)
    
    # 加载数据
    data_loader = DataLoader()
    documents = data_loader.get_documents()
    
    # 创建系统（会自动测试所有模型）
    rag_system = RAGSystem(documents)
    
    # 测试
    print("\n" + "="*70)
    print("🧪 测试系统")
    print("="*70)
    
    test_result = rag_system.answer_question("Where was Barack Obama born?", top_k=10)
    print(f"\n💡 答案: {test_result['answer']}")
    
    # 启动界面
    print("\n" + "="*70)
    print("🌐 启动Web界面")
    print("="*70)
    print(f"可用模型: {', '.join(rag_system.generator.get_available_model_names())}")
    
    demo = create_interface(rag_system)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()