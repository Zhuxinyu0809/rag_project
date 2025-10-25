"""
检测API是否支持Embedding模型

运行: python check_embedding_api.py
"""

from openai import OpenAI
from config import SILICONFLOW_CONFIG


def check_embedding_support():
    """检测SiliconFlow是否支持Embedding"""
    
    print("="*70)
    print("🔍 检测API对Embedding的支持")
    print("="*70)
    
    # 初始化客户端
    client = OpenAI(
        api_key=SILICONFLOW_CONFIG["api_key"],
        base_url=SILICONFLOW_CONFIG["base_url"]
    )
    
    # 尝试获取模型列表
    print("\n1️⃣ 尝试获取所有可用模型...")
    try:
        models = client.models.list()
        print(f"✅ 成功获取模型列表，共 {len(models.data)} 个模型")
        
        # 查找Embedding相关模型
        embedding_models = []
        for model in models.data:
            model_id = model.id.lower()
            if 'embed' in model_id or 'embedding' in model_id:
                embedding_models.append(model.id)
        
        if embedding_models:
            print(f"\n✅ 找到 {len(embedding_models)} 个Embedding模型:")
            for i, model in enumerate(embedding_models, 1):
                print(f"   {i}. {model}")
        else:
            print("\n⚠️  没有找到Embedding相关模型")
            print("   SiliconFlow可能不支持Embedding API")
        
    except Exception as e:
        print(f"❌ 无法获取模型列表: {e}")
        print("   API可能不支持 models.list() 方法")
    
    # 尝试直接调用Embedding API
    print("\n2️⃣ 测试Embedding API调用...")
    
    test_models = [
        "Qwen/Qwen3-Embedding",
        "Qwen3-Embedding",
        "qwen3-embedding",
        "text-embedding-3-small",  # OpenAI格式
        "bge-large-zh-v1.5",  # BGE格式
    ]
    
    working_model = None
    
    for model_name in test_models:
        try:
            print(f"\n   测试模型: {model_name}")
            response = client.embeddings.create(
                model=model_name,
                input="test",
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            print(f"   ✅ 成功！")
            print(f"      - 向量维度: {len(embedding)}")
            print(f"      - 前5个值: {embedding[:5]}")
            
            working_model = model_name
            break
            
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "not found" in error_msg:
                print(f"   ❌ 模型不存在")
            elif "not supported" in error_msg:
                print(f"   ❌ 不支持Embedding")
            else:
                print(f"   ❌ 错误: {error_msg[:80]}")
    
    # 总结
    print("\n" + "="*70)
    print("📊 检测结果")
    print("="*70)
    
    if working_model:
        print(f"✅ SiliconFlow 支持 Embedding！")
        print(f"✅ 可用模型: {working_model}")
        print(f"\n💡 建议配置:")
        print(f'   在 config.py 中设置:')
        print(f'   DENSE_RETRIEVER_CONFIG["api_model"] = "{working_model}"')
    else:
        print("❌ SiliconFlow 不支持 Embedding API")
        print("\n💡 解决方案:")
        print("   方案1: 使用其他API提供商（如OpenRouter、OpenAI）")
        print("   方案2: 暂时只使用BM25检索器")
        print("   方案3: 使用本地Embedding模型（需要安装sentence-transformers）")
    
    print("="*70)
    
    return working_model


if __name__ == "__main__":
    check_embedding_support()