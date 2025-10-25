"""
用户界面模块 - ui.py (支持检索器选择版本)

使用Gradio创建Web界面，支持：
1. 选择不同的Qwen模型
2. 选择不同的检索方法
3. 实时切换配置
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
        text = doc.get('text', '')[:200]
        
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
    
    # 获取可用检索器列表
    available_retrievers = []
    if hasattr(rag_system, 'retrievers') and rag_system.retrievers:
        # 如果有多个检索器
        available_retrievers = list(rag_system.retrievers.keys())
    else:
        # 如果只有一个检索器
        available_retrievers = [rag_system.retriever.get_name()]
    
    # 判断是否有多个检索器
    has_multiple_retrievers = len(available_retrievers) > 1
    
    def query_handler(
        question: str,
        top_k: int,
        model_choice: str,
        retriever_choice: str
    ) -> Tuple[str, str, str]:
        """
        处理用户查询
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            model_choice: 选择的模型
            retriever_choice: 选择的检索器
        
        Returns:
            (答案, 检索文档, 系统信息)
        """
        if not question.strip():
            return "⚠️ 请输入问题", "", "未执行查询"
        
        # 切换检索器（如果需要）
        if has_multiple_retrievers and retriever_choice:
            if retriever_choice in rag_system.retrievers:
                rag_system.retriever = rag_system.retrievers[retriever_choice]
                print(f"🔄 切换检索器到: {retriever_choice}")
        
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
        
        # 系统信息
        system_info = f"""### 📊 当前配置

- **检索方法**: `{result.get('retriever_used', 'Unknown')}`
- **生成模型**: `{result['model_used']}`
- **检索文档数**: {len(result['retrieved_docs'])} 个
"""
        
        return answer, docs, system_info
    
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
            
            **支持功能**:
            - 🔍 多种检索方法（BM25稀疏检索、Qwen3-Embedding密集检索）
            - 🤖 多种Qwen大语言模型
            - ⚡ 实时切换配置
            """
        )
        
        with gr.Row():
            # ========== 左侧：输入和配置区域 ==========
            with gr.Column(scale=1):
                gr.Markdown("### 📝 输入区域")
                
                # 问题输入框
                question_input = gr.Textbox(
                    label="💬 输入您的问题",
                    placeholder="例如: Where was Barack Obama born?",
                    lines=3
                )
                
                gr.Markdown("### ⚙️ 系统配置")
                
                # 检索器选择（重点！）
                retriever_selector = gr.Dropdown(
                    choices=available_retrievers,
                    value=available_retrievers[0],
                    label="🔍 选择检索方法",
                    info="选择不同的文档检索算法",
                    interactive=has_multiple_retrievers  # 只有多个时才可选
                )
                
                # 如果有多个检索器，显示说明
                if has_multiple_retrievers:
                    gr.Markdown(
                        """
                        **检索方法说明**:
                        - **BM25**: 关键词匹配，速度快
                        - **Qwen3-Embedding**: 语义理解，更智能
                        """
                    )
                
                # 模型选择
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="🤖 选择Qwen模型",
                    info="选择不同大小的语言模型"
                )
                
                # 检索参数
                top_k_slider = gr.Slider(
                    minimum=SYSTEM_CONFIG["top_k_min"],
                    maximum=SYSTEM_CONFIG["top_k_max"],
                    value=SYSTEM_CONFIG["top_k_default"],
                    step=1,
                    label="📚 检索文档数量",
                    info="检索更多文档可能提高答案质量"
                )
                
                # 提交按钮
                submit_btn = gr.Button(
                    "🚀 提交查询",
                    variant="primary",
                    size="lg"
                )
                
                # 使用说明
                with gr.Accordion("💡 使用说明", open=False):
                    gr.Markdown(
                        """
                        ### 如何使用
                        
                        1. **输入问题**: 在上方文本框输入您的问题
                        2. **选择检索方法**: 
                           - BM25: 适合精确关键词匹配
                           - Qwen3-Embedding: 适合语义理解
                        3. **选择模型**: 
                           - 7B: 质量最好，速度较慢
                           - 3B/1.5B: 平衡质量和速度
                        4. **调整参数**: 可选调整检索文档数量
                        5. **提交查询**: 点击按钮获取答案
                        
                        ### 性能对比
                        
                        | 检索方法 | 速度 | 语义理解 | 适用场景 |
                        |---------|------|---------|---------|
                        | BM25 | ⚡⚡⚡ | ⭐⭐ | 精确匹配 |
                        | Qwen3-Embedding | ⚡⚡ | ⭐⭐⭐⭐⭐ | 语义查询 |
                        """
                    )
            
            # ========== 右侧：输出区域 ==========
            with gr.Column(scale=2):
                gr.Markdown("### 💡 输出区域")
                
                # 系统信息显示
                system_info_display = gr.Markdown(
                    value=f"""### 📊 当前配置

- **检索方法**: `{rag_system.retriever.get_name()}`
- **生成模型**: `{rag_system.generator.current_model}`
- **检索文档数**: {SYSTEM_CONFIG['top_k_default']} 个
"""
                )
                
                # 答案输出
                answer_output = gr.Textbox(
                    label="📝 生成的答案",
                    lines=8,
                    show_copy_button=True,
                    placeholder="答案将在这里显示..."
                )
                
                # 检索文档输出（可折叠）
                with gr.Accordion("📚 检索到的文档", open=True):
                    docs_output = gr.Markdown(
                        value="检索结果将在这里显示..."
                    )
        
        # ========== 绑定事件 ==========
        submit_btn.click(
            fn=query_handler,
            inputs=[
                question_input,
                top_k_slider,
                model_selector,
                retriever_selector
            ],
            outputs=[
                answer_output,
                docs_output,
                system_info_display
            ]
        )
        
        # 支持回车提交
        question_input.submit(
            fn=query_handler,
            inputs=[
                question_input,
                top_k_slider,
                model_selector,
                retriever_selector
            ],
            outputs=[
                answer_output,
                docs_output,
                system_info_display
            ]
        )
        
        # ========== 示例问题 ==========
        gr.Examples(
            examples=[
                ["Where was Barack Obama born?"],
                ["What is the capital of France?"],
                ["Who wrote Romeo and Juliet?"],
                ["What are the causes of climate change?"],
            ],
            inputs=[question_input],
            label="📌 示例问题（点击自动填充）"
        )
        
        # ========== 底部信息 ==========
        gr.Markdown(
            f"""
            ---
            ### 📊 系统信息
            
            - **可用检索器**: {len(available_retrievers)} 个 ({', '.join(available_retrievers)})
            - **可用模型**: {len(available_models)} 个
            - **文档库大小**: 144,718 个文档
            
            ---
            
            COMP5423 Natural Language Processing - Group Project
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
    
    # 显示可用的检索器
    if hasattr(rag_system, 'retrievers') and rag_system.retrievers:
        print(f"可用检索器: {', '.join(rag_system.retrievers.keys())}")
    else:
        print(f"当前检索器: {rag_system.retriever.get_name()}")
    
    # 显示可用的模型
    print(f"可用模型: {', '.join(rag_system.generator.get_available_model_names())}")
    
    demo = create_gradio_interface(rag_system)
    
    demo.launch(
        server_name=UI_CONFIG["server_name"],
        server_port=UI_CONFIG["server_port"],
        share=UI_CONFIG["share"],
        show_error=UI_CONFIG["show_error"]
    )