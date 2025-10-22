# rag_project
# RAG问答系统 - 模块化版本

COMP5423 Natural Language Processing - Group Project

## 📁 项目结构

```
rag_project/
├── config.py                  # 配置文件
├── data_loader.py             # 数据加载模块
├── rag_system.py              # RAG系统主逻辑
├── ui.py                      # Gradio用户界面
├── main.py                    # 主程序入口
│
├── retrievers/                # 检索器模块目录
│   ├── __init__.py
│   ├── bm25_retriever.py      # BM25检索器
│   ├── dense_retriever.py     # 密集检索器（未来添加）
│   └── hybrid_retriever.py    # 混合检索器（未来添加）
│
├── generators/                # 生成器模块目录
│   ├── __init__.py
│   ├── api_generator.py       # API调用生成器
│   └── local_generator.py     # 本地模型生成器（未来添加）
│
├── utils/                     # 工具函数目录
│   ├── __init__.py
│   ├── evaluation.py          # 评估工具（未来添加）
│   └── export.py              # 结果导出工具（未来添加）
│
├── requirements.txt           # 依赖列表
└── README.md                  # 项目说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置API密钥

编辑 `config.py` 文件，填入你的API密钥：

```python
SILICONFLOW_CONFIG = {
    "api_key": "sk-你的密钥",  # ← 修改这里
    "base_url": "https://api.siliconflow.cn/v1",
}
```

### 3. 运行程序

```bash
python main.py
```

程序会自动：
1. 加载数据集
2. 初始化检索器
3. 测试API连接
4. 启动Web界面（http://localhost:7860）

## 🎯 模块说明

### 📝 config.py
- 集中管理所有配置
- API密钥、模型列表、系统参数等
- **修改配置时只需改这个文件**

### 📚 data_loader.py
- 加载HQ-small数据集
- 提供数据访问接口
- 支持训练集、验证集、测试集

### 🔍 retrievers/bm25_retriever.py
- BM25检索算法实现
- 文档索引和检索
- 可替换为其他检索器

### 🤖 generators/api_generator.py
- 通过API调用LLM
- 支持多模型切换
- 自动测试模型可用性

### 🎯 rag_system.py
- 整合检索和生成
- 实现完整RAG流程
- 提供批量处理和评估功能

### 🌐 ui.py
- Gradio Web界面
- 模型选择、参数调整
- 结果展示

### ▶️ main.py
- 程序入口
- 初始化所有模块
- 启动系统

## ➕ 如何添加新功能

### 添加新的检索器

1. 在 `retrievers/` 目录创建新文件，例如 `dense_retriever.py`
2. 实现检索器类：

```python
from typing import List, Dict

class DenseRetriever:
    def __init__(self, documents: List[Dict]):
        # 初始化代码
        pass
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        # 检索逻辑
        pass
    
    def get_name(self) -> str:
        return "Dense Retriever"
```

3. 在 `main.py` 中使用：

```python
from retrievers.dense_retriever import DenseRetriever

# 替换这行
retriever = DenseRetriever(documents)
```

### 添加新的LLM模型

1. 在 `config.py` 的 `AVAILABLE_QWEN_MODELS` 中添加：

```python
AVAILABLE_QWEN_MODELS = {
    # 现有模型...
    
    "新模型名称": [
        "api-model-name-1",
        "api-model-name-2"
    ]
}
```

2. 系统会自动测试并在UI中显示

### 添加新的API提供商

1. 在 `config.py` 中添加配置：

```python
NEW_API_CONFIG = {
    "api_key": "your-api-key",
    "base_url": "https://api.example.com/v1"
}
```

2. 在 `generators/api_generator.py` 中使用：

```python
generator = APIGenerator(
    api_key=NEW_API_CONFIG["api_key"],
    base_url=NEW_API_CONFIG["base_url"]
)
```

## 📊 运行评估

```python
from data_loader import DataLoader

# 加载验证集
data_loader = DataLoader()
val_samples = data_loader.get_validation_samples(n=100)

# 运行评估
results = rag_system.evaluate(
    test_samples=val_samples,
    top_k=10
)

print(f"评估完成，共 {results['total']} 个样本")
```

## 🛠️ 开发建议

### 代码风格
- 遵循PEP 8规范
- 每个函数添加类型注解
- 重要函数添加文档字符串

### 模块原则
- 单一职责：每个模块只做一件事
- 低耦合：模块之间依赖最小化
- 高内聚：相关功能放在同一模块

### 配置管理
- 所有配置写在 `config.py`
- 不要在代码中硬编码参数
- 使用常量命名（全大写+下划线）

## 🎓 项目报告建议

### 系统设计方法部分可以这样写：

```
我们的系统采用模块化设计，主要包含以下组件：

1. 数据加载模块：负责从HuggingFace加载HQ-small数据集
2. 检索模块：实现BM25算法，建立文档索引
3. 生成模块：通过SiliconFlow API调用Qwen模型
4. RAG系统：整合检索和生成，实现完整流程
5. 用户界面：基于Gradio实现Web交互

这种设计的优势：
- 易于测试和调试
- 便于添加新功能
- 代码复用性高
```

## 📝 待办事项

- [ ] 添加密集检索器（BGE/E5）
- [ ] 实现混合检索
- [ ] 添加评估指标计算
- [ ] 实现结果导出功能
- [ ] 添加多轮对话支持
- [ ] 实现Agent工作流

## 🤝 贡献者

COMP5423 - Group X

## 📄 许可证

本项目仅用于学术目的
