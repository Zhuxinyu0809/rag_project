"""
支持在UI界面选择Qwen模型的RAG系统

功能：
1. 在界面下拉菜单选择不同的Qwen模型
2. 实时切换模型
3. 显示当前使用的模型
"""

import json
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import bm25s
import Stemmer
import gradio as gr
from openai import OpenAI

# ============================================================================
# 配置
# ============================================================================
API_KEY = "sk-uiunawvmaprjvukpszbuponajpulqjvjwxwemgftljnqbdzc"  # ⚠️ 填入你的密钥
API_BASE_URL = "https://api.siliconflow.cn/v1"

# 定义所有可能的Qwen模型（按性能排序）
AVAILABLE_MODELS = {
    # 显示名称: 实际的API模型名
    "Qwen2.5-7B (推荐)": [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-7B-Instruct",
        "qwen2.5-7b-instruct"
    ],
    "Qwen2-7B ": [
        "Qwen/Qwen2-7B-Instruct",
        "Qwen2-7B-Instruct",
        "qwen2-7b-instruct"
    ]
}


# ============================================================================
# 数据加载
# ============================================================================
class DataLoader:
    def __init__(self):
        print("📚 正在加载数据集...")
        self.dataset = load_dataset("izhx/COMP5423-25Fall-HQ-small")
        self.collection = self.dataset['collection']
        print(f"✅ 数据加载完成！文档库: {len(self.collection)} 条")
    
    def get_documents(self) -> List[Dict]:
        return [{'id': item['id'], 'text': item['text']} for item in self.collection]


# ============================================================================
# BM25检索器
# ============================================================================
class BM25Retriever:
    def __init__(self, documents: List[Dict]):
        print("\n🔍 正在初始化BM25检索器...")
        self.documents = documents
        self.doc_ids = [doc['id'] for doc in documents]
        
        corpus_texts = [doc['text'] for doc in documents]
        stemmer = Stemmer.Stemmer("english")
        
        print("   正在建立索引...")
        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en", stemmer=stemmer)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        self.stemmer = stemmer
        print(f"✅ BM25索引创建完成！")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        results, scores = self.retriever.retrieve(query_tokens, k=top_k)
        return [
            {'id': self.doc_ids[idx], 'text': self.documents[idx]['text'], 'score': float(score)}
            for idx, score in zip(results[0], scores[0])
        ]


# ============================================================================
# 可切换模型的生成器
# ============================================================================
class FlexibleAPIGenerator:
    """支持动态切换模型的生成器"""
    
    def __init__(self):
        print(f"\n🤖 正在初始化API客户端...")
        
        # 检查API密钥
        if API_KEY.startswith("sk-你的"):
            print("\n" + "="*70)
            print("❌ 错误：请先设置API密钥！")
            print("="*70)
            raise ValueError("未设置API密钥")
        
        # 创建客户端
        self.client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
        
        # 测试并缓存所有可用的模型
        self.available_models = self._test_all_models()
        
        # 设置默认模型
        if self.available_models:
            self.current_model = list(self.available_models.values())[0]
            print(f"✅ 默认模型: {self.current_model}")
        else:
            raise ValueError("没有可用的模型")
    
    def _test_all_models(self) -> Dict[str, str]:
        """测试所有模型，返回可用的模型映射"""
        print("\n🔍 正在测试所有模型的可用性...")
        available = {}
        
        for display_name, possible_names in AVAILABLE_MODELS.items():
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
                    
                    # 成功！记录这个模型
                    available[display_name] = model_name
                    print(f"   ✅ 可用！使用: {model_name}")
                    break  # 找到一个就够了
                    
                except Exception as e:
                    error_str = str(e)
                    if "does not exist" in error_str:
                        print(f"   ❌ 不存在")
                    else:
                        print(f"   ❌ 错误: {error_str[:50]}")
                    continue
            
            if display_name not in available:
                print(f"   ⚠️  {display_name} 不可用")
        
        print(f"\n✅ 找到 {len(available)} 个可用模型")
        return available
    
    def get_available_model_names(self) -> List[str]:
        """获取可用模型的显示名称列表（供UI使用）"""
        return list(self.available_models.keys())
    
    def switch_model(self, display_name: str) -> str:
        """切换到指定的模型"""
        if display_name in self.available_models:
            self.current_model = self.available_models[display_name]
            print(f"🔄 切换到模型: {self.current_model}")
            return self.current_model
        else:
            print(f"⚠️  模型 {display_name} 不可用")
            return self.current_model
    
    def create_prompt(self, query: str, documents: List[Dict]) -> str:
        """创建提示词"""
        context = ""
        for i, doc in enumerate(documents[:5], 1):
            text = doc['text'][:400]
            context += f"[Document {i}]\n{text}\n\n"
        
        prompt = f"""Answer the question based on the provided documents.

Documents:
{context}

Question: {query}

Provide a concise answer. If the documents don't contain enough information, say "Cannot answer based on provided documents."

Answer:"""
        return prompt
    
    def generate(self, query: str, documents: List[Dict]) -> str:
        """生成答案"""
        prompt = self.create_prompt(query, documents)
        
        try:
            print(f"   🤖 使用模型 {self.current_model} 生成答案...")
            
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"   ✅ 生成成功")
            return answer
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ 生成失败: {error_msg[:100]}")
            return f"生成失败: {error_msg}"


# ============================================================================
# RAG系统
# ============================================================================
class RAGSystem:
    def __init__(self, documents: List[Dict]):
        self.retriever = BM25Retriever(documents)
        self.generator = FlexibleAPIGenerator()
    
    def answer_question(
        self, 
        query: str, 
        top_k: int = 10,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        回答问题
        
        Args:
            query: 问题
            top_k: 检索文档数量
            model_name: 模型显示名称（如果要切换模型）
        """
        # 如果指定了模型，先切换
        if model_name:
            self.generator.switch_model(model_name)
        
        print(f"\n{'='*70}")
        print(f"❓ 问题: {query}")
        print(f"🤖 模型: {self.generator.current_model}")
        print(f"{'='*70}")
        
        # 检索
        print(f"🔍 检索前 {top_k} 个相关文档...")
        retrieved_docs = self.retriever.search(query, top_k=top_k)
        print(f"✅ 检索完成")
        
        # 生成
        answer = self.generator.generate(query, retrieved_docs)
        
        return {
            'question': query,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'model_used': self.generator.current_model
        }


# ============================================================================
# Gradio界面（带模型选择器）
# ============================================================================
def create_interface(rag_system: RAGSystem):
    """创建带模型选择功能的界面"""
    
    # 获取可用模型列表
    available_models = rag_system.generator.get_available_model_names()
    
    if not available_models:
        raise ValueError("没有可用的模型！")
    
    def format_docs(docs: List[Dict]) -> str:
        result = "### 📚 检索到的文档\n\n"
        for i, doc in enumerate(docs, 1):
            result += f"**{i}. 文档 {doc['id']}** (相关度: {doc['score']:.4f})\n\n"
            result += f"{doc['text'][:200]}...\n\n---\n\n"
        return result
    
    def query_handler(question: str, top_k: int, model_choice: str) -> Tuple[str, str, str]:
        """处理查询并返回结果"""
        if not question.strip():
            return "请输入问题", "", "未使用模型"
        
        # 执行RAG，传入选择的模型
        result = rag_system.answer_question(question, top_k, model_choice)
        
        answer = result['answer']
        docs = format_docs(result['retrieved_docs'])
        model_info = f"**当前使用模型**: `{result['model_used']}`"
        
        return answer, docs, model_info
    
    # 创建界面
    with gr.Blocks(title="RAG系统 - 可选择模型", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(
            """
            # 🤖 RAG问答系统（可选择模型版本）
            
            在下方选择不同的Qwen模型，体验不同性能和速度的平衡
            """
        )
        
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入区域")
                
                question_input = gr.Textbox(
                    label="输入问题",
                    placeholder="例如: Where was Barack Obama born?",
                    lines=3
                )
                
                # 模型选择下拉菜单（重点！）
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],  # 默认选第一个
                    label="🤖 选择Qwen模型",
                    info="不同模型有不同的性能和速度"
                )
                
                # 检索参数
                top_k = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    label="📚 检索文档数量"
                )
                
                submit_btn = gr.Button("🚀 提交查询", variant="primary", size="lg")
            
            # 右侧：输出区域
            with gr.Column(scale=2):
                gr.Markdown("### 💡 输出区域")
                
                # 显示当前使用的模型
                model_info_display = gr.Markdown(
                    value=f"**当前模型**: `{rag_system.generator.current_model}`",
                )
                
                answer_output = gr.Textbox(
                    label="📝 生成的答案",
                    lines=6,
                    show_copy_button=True
                )
                
                docs_output = gr.Markdown(
                    label="📚 检索到的文档"
                )
        
        # 绑定事件
        submit_btn.click(
            fn=query_handler,
            inputs=[question_input, top_k, model_selector],
            outputs=[answer_output, docs_output, model_info_display]
        )
        
        # 示例问题
        gr.Examples(
            examples=[
                ["Where was Barack Obama born?"],
                ["What is the capital of France?"],
            ],
            inputs=question_input
        )
        
        # 底部说明
        gr.Markdown(
            """
            ---
            ### 📊 系统信息
            
            **可用模型数量**: {num_models}  
            **检索方法**: BM25  
            **文档库大小**: 144,718 个文档
            """.format(num_models=len(available_models))
        )
    
    return demo


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