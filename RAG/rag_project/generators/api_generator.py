"""
API生成器模块 - generators/api_generator.py

通过API调用大语言模型(LLM)生成答案
"""

from typing import List, Dict, Optional
from openai import OpenAI
from config import (
    SILICONFLOW_CONFIG,
    AVAILABLE_QWEN_MODELS,
    GENERATION_CONFIG,
    SYSTEM_CONFIG
)


class APIGenerator:
    """API调用生成器（支持多模型切换）"""
    
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model_config: Dict = None
    ):
        """
        初始化API生成器
        
        Args:
            api_key: API密钥（None则使用配置文件）
            base_url: API基础URL（None则使用配置文件）
            model_config: 模型配置字典（None则使用默认配置）
        """
        print(f"\n🤖 正在初始化API生成器...")
        
        # 使用传入的配置或默认配置
        self.api_key = api_key or SILICONFLOW_CONFIG["api_key"]
        self.base_url = base_url or SILICONFLOW_CONFIG["base_url"]
        self.model_config = model_config or AVAILABLE_QWEN_MODELS
        
        # 检查API密钥
        if self.api_key.startswith("sk-your") or self.api_key.startswith("sk-在"):
            raise ValueError("请设置有效的API密钥！")
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        # 测试并缓存所有可用模型
        self.available_models = self._test_all_models()
        
        # 设置默认模型
        if self.available_models:
            self.current_model = list(self.available_models.values())[0]
            print(f"✅ 默认模型: {self.current_model}")
        else:
            raise ValueError("没有可用的模型！")
    
    def _test_all_models(self) -> Dict[str, str]:
        """测试所有模型，返回可用的模型映射"""
        print("\n🔍 正在测试模型可用性...")
        available = {}
        
        for display_name, possible_names in self.model_config.items():
            print(f"\n测试 {display_name}:")
            
            for model_name in possible_names:
                try:
                    print(f"   尝试: {model_name}")
                    
                    # 测试请求
                    response = self.client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": "Hi"}],
                        max_tokens=5,
                        timeout=10
                    )
                    
                    # 成功！
                    available[display_name] = model_name
                    print(f"   ✅ 可用")
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "does not exist" in error_str or "not found" in error_str:
                        print(f"   ❌ 不存在")
                    else:
                        print(f"   ❌ 错误")
                    continue
            
            if display_name not in available:
                print(f"   ⚠️  不可用")
        
        print(f"\n✅ 找到 {len(available)} 个可用模型")
        return available
    
    def get_available_model_names(self) -> List[str]:
        """获取可用模型的显示名称列表"""
        return list(self.available_models.keys())
    
    def switch_model(self, display_name: str) -> str:
        """
        切换到指定模型
        
        Args:
            display_name: 模型的显示名称
        
        Returns:
            实际使用的模型名称
        """
        if display_name in self.available_models:
            self.current_model = self.available_models[display_name]
            print(f"🔄 切换到模型: {self.current_model}")
            return self.current_model
        else:
            print(f"⚠️  模型 {display_name} 不可用，保持当前模型")
            return self.current_model
    
    def create_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        创建提示词
        
        Args:
            query: 用户问题
            documents: 检索到的文档列表
        
        Returns:
            完整的提示词
        """
        # 格式化文档上下文
        context = ""
        max_docs = SYSTEM_CONFIG["docs_for_prompt"]
        max_chars = SYSTEM_CONFIG["max_doc_chars"]
        
        for i, doc in enumerate(documents[:max_docs], 1):
            text = doc['text'][:max_chars]
            context += f"[Document {i}]\n{text}\n\n"
        
        # 构建完整提示词
        prompt = f"""Answer the question based on the provided documents.

Documents:
{context}

Question: {query}

Provide a concise answer. If the documents don't contain enough information, say "Cannot answer based on provided documents."

Answer:"""
        
        return prompt
    
    def generate(
        self,
        query: str,
        documents: List[Dict],
        model_name: Optional[str] = None
    ) -> str:
        """
        生成答案
        
        Args:
            query: 用户问题
            documents: 检索到的文档列表
            model_name: 要使用的模型显示名称（None则使用当前模型）
        
        Returns:
            生成的答案
        """
        # 如果指定了模型，先切换
        if model_name:
            self.switch_model(model_name)
        
        # 创建提示词
        prompt = self.create_prompt(query, documents)
        
        try:
            print(f"   🤖 使用模型 {self.current_model} 生成答案...")
            
            # 调用API
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=GENERATION_CONFIG["max_tokens"],
                temperature=GENERATION_CONFIG["temperature"]
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"   ✅ 生成成功")
            
            return answer
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ 生成失败: {error_msg[:100]}")
            return f"生成失败: {error_msg}"
    
    def get_name(self) -> str:
        """返回生成器名称"""
        return f"API Generator ({self.current_model})"