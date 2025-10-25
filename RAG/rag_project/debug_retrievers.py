"""
诊断脚本 - 检查检索器配置

运行: python debug_retrievers.py
"""

from data_loader import DataLoader
from retrievers.bm25_retriever import BM25Retriever
from generators.api_generator import APIGenerator
from rag_system import RAGSystem


def diagnose():
    """诊断检索器配置"""
    
    print("\n" + "="*70)
    print("🔍 检索器配置诊断")
    print("="*70)
    
    # 加载少量数据用于测试
    print("\n1️⃣ 加载数据...")
    data_loader = DataLoader()
    documents = data_loader.get_documents()[:1000]  # 只用1000个文档测试
    print(f"✅ 加载了 {len(documents)} 个文档")
    
    # 创建检索器字典
    print("\n2️⃣ 初始化检索器...")
    retrievers = {}
    
    # BM25
    print("   初始化 BM25...")
    retrievers["BM25"] = BM25Retriever(documents)
    print(f"   ✅ BM25 类型: {type(retrievers['BM25'])}")
    print(f"   ✅ BM25 名称: {retrievers['BM25'].get_name()}")
    
    # 打印检索器字典
    print(f"\n3️⃣ 检索器字典内容:")
    print(f"   键(keys): {list(retrievers.keys())}")
    print(f"   数量: {len(retrievers)}")
    
    # 初始化生成器
    print("\n4️⃣ 初始化生成器...")
    generator = APIGenerator()
    
    # 创建RAG系统
    print("\n5️⃣ 创建RAG系统...")
    rag_system = RAGSystem(
        retriever=retrievers["BM25"],
        generator=generator
    )
    
    # 关键：保存检索器字典
    print("\n6️⃣ 保存检索器字典到系统...")
    rag_system.retrievers = retrievers
    
    # 检查
    print("\n7️⃣ 检查系统状态:")
    print(f"   rag_system 有 retrievers 属性? {hasattr(rag_system, 'retrievers')}")
    
    if hasattr(rag_system, 'retrievers'):
        print(f"   ✅ retrievers 类型: {type(rag_system.retrievers)}")
        print(f"   ✅ retrievers 内容: {rag_system.retrievers}")
        print(f"   ✅ retrievers 键: {list(rag_system.retrievers.keys())}")
        print(f"   ✅ retrievers 长度: {len(rag_system.retrievers)}")
        
        # 测试访问
        for name, retriever in rag_system.retrievers.items():
            print(f"\n   检索器 '{name}':")
            print(f"      类型: {type(retriever)}")
            print(f"      名称方法: {retriever.get_name()}")
    else:
        print("   ❌ 没有 retrievers 属性！")
    
    print(f"\n   当前检索器: {rag_system.retriever.get_name()}")
    
    # 模拟UI中的检索器获取
    print("\n8️⃣ 模拟UI获取检索器列表:")
    
    if hasattr(rag_system, 'retrievers') and rag_system.retrievers:
        available = list(rag_system.retrievers.keys())
        print(f"   ✅ 多检索器模式")
        print(f"   ✅ 可用检索器: {available}")
        print(f"   ✅ 数量: {len(available)}")
    else:
        available = [rag_system.retriever.get_name()]
        print(f"   ⚠️  单检索器模式")
        print(f"   ⚠️  检索器: {available}")
    
    # 总结
    print("\n" + "="*70)
    print("📊 诊断总结")
    print("="*70)
    
    if len(available) > 1:
        print("✅ 系统正确配置了多个检索器")
        print(f"✅ 可用检索器: {', '.join(available)}")
        print("✅ UI应该可以显示检索器选择下拉菜单")
    else:
        print("⚠️  系统只有一个检索器")
        print(f"⚠️  当前检索器: {available[0]}")
        print("⚠️  UI将不显示检索器选择功能")
        print("\n💡 解决方案:")
        print("   1. 在 main.py 中选择选项 3 (两者都初始化)")
        print("   2. 确保 Qwen3-Embedding 初始化成功")
        print("   3. 检查是否正确执行了 rag_system.retrievers = retrievers")
    
    print("="*70)


if __name__ == "__main__":
    diagnose()