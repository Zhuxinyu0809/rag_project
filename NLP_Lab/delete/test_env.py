print("测试环境...")

# 测试1：datasets
try:
    from datasets import load_dataset
    print("✅ datasets 安装成功")
except:
    print("❌ datasets 安装失败")

# 测试2：bm25s
try:
    import bm25s
    print("✅ bm25s 安装成功")
except:
    print("❌ bm25s 安装失败")

# 测试3：PyStemmer
try:
    import Stemmer
    print("✅ PyStemmer 安装成功")
except:
    print("❌ PyStemmer 安装失败")

print("\n如果都显示✅，你就可以开始了！")