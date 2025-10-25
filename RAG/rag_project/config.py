"""
配置文件 - config.py

所有的配置都在这里集中管理
"""

# ============================================================================
# API配置
# ============================================================================

# SiliconFlow API配置
SILICONFLOW_CONFIG = {
    "api_key": "sk-uiunawvmaprjvukpszbuponajpulqjvjwxwemgftljnqbdzc",
    "base_url": "https://api.siliconflow.cn/v1",
}


# ============================================================================
# 模型配置
# ============================================================================

# 可用的Qwen模型列表
AVAILABLE_QWEN_MODELS = {
    # 显示名称: [可能的API名称列表]
    "Qwen2.5-7B (推荐)": [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct"
    ],
    "Qwen2-7B (旧版本)": [
        "Qwen/Qwen2-7B-Instruct",
        "Qwen2-7B-Instruct",
        "qwen2-7b-instruct"
    ]
}


# ============================================================================
# 数据集配置
# ============================================================================

DATASET_CONFIG = {
    "name": "izhx/COMP5423-25Fall-HQ-small",
    "cache_dir": None,  # 使用默认缓存目录
}


# ============================================================================
# 检索器配置
# ============================================================================

BM25_CONFIG = {
    "stemmer_language": "english",
    "use_stopwords": True,
}

# 未来可以添加其他检索器的配置
DENSE_RETRIEVER_CONFIG = {
    # Qwen3-Embedding配置（使用API）
    "use_api": True,  # True=使用API, False=本地模型
    "api_model": "Qwen/Qwen3-Embedding",  # API模型名
    "api_key": None,  # 使用主API配置
    "api_base_url": None,  # 使用主API配置
    
    # 本地模型配置（如果use_api=False）
    "local_model": "Qwen/Qwen3-Embedding",
    "device": "cpu",
    "batch_size": 32,
    
    # 向量索引配置
    "index_type": "flat",  # "flat" 或 "ivf"
    "dimension": 1024,  # Qwen3-Embedding的维度
}


# ============================================================================
# 生成器配置
# ============================================================================

GENERATION_CONFIG = {
    "max_tokens": 200,
    "temperature": 0.7,
    "top_p": 0.9,
}


# ============================================================================
# UI配置
# ============================================================================

UI_CONFIG = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "share": False,
    "show_error": True,
    "theme": "soft",  # 可选: "soft", "default", "monochrome"
}


# ============================================================================
# 系统配置
# ============================================================================

SYSTEM_CONFIG = {
    # 检索文档数量范围
    "top_k_min": 1,
    "top_k_max": 20,
    "top_k_default": 10,
    
    # 提示词中使用的文档数量
    "docs_for_prompt": 5,
    
    # 每个文档的最大字符数
    "max_doc_chars": 400,
}