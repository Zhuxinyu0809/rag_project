# 创建文件 check_import.py
import sys
print("Python路径:")
for p in sys.path:
    print(f"  - {p}")

print("\\n检查 rag_system.py 文件:")
try:
    import rag_system
    print(f"✅ 成功导入 rag_system 模块")
    print(f"模块路径: {rag_system.__file__}")
    print(f"\\n模块中的内容:")
    print(dir(rag_system))
    
    # 检查是否有 RAGSystem 类
    if hasattr(rag_system, 'RAGSystem'):
        print("✅ 找到 RAGSystem 类")
    else:
        print("❌ 没有找到 RAGSystem 类")
        
except Exception as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()