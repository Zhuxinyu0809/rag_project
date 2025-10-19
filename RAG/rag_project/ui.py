"""
用户界面模块 - ui.py

使用Gradio创建Web界面
"""

from typing import Tuple
import gradio as gr
from rag_system import RAGSystem
from config import UI_CONFIG, SYSTEM_CONFIG


def format_retrieved_docs(docs: list) -> str:
    """
    格式化检索到的文档用于显示
    
    Args:
        docs: 文档列表
    
    Returns:
        格式化的Markdown文本
    """
    if not docs:
        return "未检索到相关文档"
    
    result = "### 📚 检索到的文档\n\n"
    
    for i, doc in enumerate(docs, 1):
        doc_id = doc.get('id', 'N/A')
        score = doc.get('score', 0.0)
        text = doc.get('text', '')[:200]  # 只显示前200字符
        
        result += f"**{i}. 文档 {doc_id}** (相关度: `{score:.4f}`)\n\n"
        result += f"{text}...\n\n"
        result += "---\n\n"
    
    return result


def create_gradio_interface(rag_system: RAGSystem):
    """
    创建Gradio界面
    
    Args:
        rag_system: RAG系统实例
    
    Returns:
        Gradio应用实例
    """
    # 获取可用模型列表
    available_models = rag_system.generator.get_available_model_names()
    
    if not available_models:
        raise ValueError("没有可用的模型！")
    
    def query_handler(
        question: str,
        top_k: int,
        model_choice: str
    ) -> Tuple[str, str, str]:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            model_choice: 选择的模型
        
        Returns:
            (答案, 检索文档, 模型信息)
        """
        if not question.strip():
            return "请输入问题", "", "未使用模型"
        
        # 执行RAG
        result = rag_system.answer_question(
            query=question,
            top_k=top_k,
            model_name=model_choice,
            verbose=True
        )
        
        # 格式化输出
        answer = result['answer']
        docs = format_retrieved_docs(result['retrieved_docs'])
        model_info = f"**当前使用模型**: `{result['model_used']}`"
        
        return answer, docs, model_info
    
    # 创建Gradio界面
    with gr.Blocks(
        title="RAG问答系统",
        theme=gr.themes.Soft() if UI_CONFIG["theme"] == "soft" else None
    ) as demo:
        
        # 标题
        gr.Markdown(
            """
            # 🤖 RAG问答系统
            ### 检索增强生成 (Retrieval-Augmented Generation)
            
            基于BM25检索 + Qwen大语言模型
            """
        )
        
        with gr.Row():
            # 左侧：输入区域
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入区域")
                
                question_input = gr.Textbox(
                    label="输入您的问题",
                    placeholder="例如: Where was Barack Obama born?",
                    lines=3
                )
                
                # 模型选择器
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="🤖 选择Qwen模型",
                    info="不同模型有不同的性能和速度"
                )
                
                # 检索参数
                top_k_slider = gr.Slider(
                    minimum=SYSTEM_CONFIG["top_k_min"],
                    maximum=SYSTEM_CONFIG["top_k_max"],
                    value=SYSTEM_CONFIG["top_k_default"],
                    step=1,
                    label="📚 检索文档数量"
                )
                
                submit_btn = gr.Button(
                    "🚀 提交查询",
                    variant="primary",
                    size="lg"
                )
                
                # 说明文本
                gr.Markdown(
                    """
                    #### 💡 使用说明
                    
                    1. 输入您的问题
                    2. 选择合适的模型
                    3. 调整检索数量（可选）
                    4. 点击提交查询
                    """
                )
            
            # 右侧：输出区域
            with gr.Column(scale=2):
                gr.Markdown("### 💡 输出区域")
                
                # 模型信息
                model_info_display = gr.Markdown(
                    value=f"**当前模型**: `{rag_system.generator.current_model}`"
                )
                
                # 答案输出
                answer_output = gr.Textbox(
                    label="📝 生成的答案",
                    lines=6,
                    show_copy_button=True
                )
                
                # 检索文档输出
                docs_output = gr.Markdown(
                    label="📚 检索到的文档"
                )
        
        # 绑定事件
        submit_btn.click(
            fn=query_handler,
            inputs=[question_input, top_k_slider, model_selector],
            outputs=[answer_output, docs_output, model_info_display]
        )
        
        # 示例问题
        gr.Examples(
            examples=[
                ["Where was Barack Obama born?", 10, available_models[0]],
                ["What is the capital of France?", 10, available_models[0]],
                ["Who wrote Romeo and Juliet?", 10, available_models[0]],
            ],
            inputs=[question_input, top_k_slider, model_selector],
            label="📌 示例问题"
        )
        
        # 底部信息
        gr.Markdown(
            f"""
            ---
            ### 📊 系统信息
            
            - **可用模型**: {len(available_models)} 个
            - **检索方法**: BM25
            - **文档库**: 144,718 个文档
            """
        )
    
    return demo


def launch_interface(rag_system: RAGSystem):
    """
    启动Gradio界面
    
    Args:
        rag_system: RAG系统实例
    """
    print("\n" + "="*70)
    print("🌐 启动Web界面")
    print("="*70)
    
    demo = create_gradio_interface(rag_system)
    
    demo.launch(
        server_name=UI_CONFIG["server_name"],
        server_port=UI_CONFIG["server_port"],
        share=UI_CONFIG["share"],
        show_error=UI_CONFIG["show_error"]
    )