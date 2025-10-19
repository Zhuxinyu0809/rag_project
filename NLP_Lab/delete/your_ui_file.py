import gradio as gr
import json
from typing import List, Dict, Tuple, Optional

# ============================================================================
# 这里需要导入你的RAG系统组件
# ============================================================================
# from your_retrieval_module import RetrieverSystem
# from your_generation_module import GeneratorSystem

class RAGInterface:
    """RAG系统的Gradio界面封装类"""
    
    def __init__(self):
        # 初始化你的检索器和生成器
        # self.retriever = RetrieverSystem()
        # self.generator = GeneratorSystem()
        
        # 对话历史（用于多轮对话）
        self.conversation_history = []
        
    def format_retrieved_docs(self, docs: List[Dict]) -> str:
        """格式化检索到的文档用于显示"""
        if not docs:
            return "未检索到相关文档"
        
        formatted = "### 📚 检索到的文档\n\n"
        for i, doc in enumerate(docs, 1):
            doc_id = doc.get('id', 'N/A')
            score = doc.get('score', 0.0)
            text = doc.get('text', '')[:200]  # 只显示前200字符
            formatted += f"**文档 {i}** (ID: `{doc_id}`, 相关度: `{score:.4f}`)\n"
            formatted += f"> {text}...\n\n"
        
        return formatted
    
    def format_intermediate_steps(self, steps: List[Dict]) -> str:
        """格式化中间推理步骤（Feature B的加分项）"""
        if not steps:
            return ""
        
        formatted = "### 🔍 中间推理过程\n\n"
        for i, step in enumerate(steps, 1):
            step_type = step.get('type', 'unknown')
            content = step.get('content', '')
            
            if step_type == 'query_rewrite':
                formatted += f"**步骤 {i}: 查询重写**\n"
                formatted += f"- 原始查询: `{step.get('original', '')}`\n"
                formatted += f"- 重写后: `{content}`\n\n"
            
            elif step_type == 'sub_query':
                formatted += f"**步骤 {i}: 子查询分解**\n"
                for j, sub_q in enumerate(content, 1):
                    formatted += f"  {j}. {sub_q}\n"
                formatted += "\n"
            
            elif step_type == 'self_check':
                formatted += f"**步骤 {i}: 自我检查**\n"
                formatted += f"- 验证结果: {content}\n\n"
            
            elif step_type == 'reasoning':
                formatted += f"**步骤 {i}: 推理过程**\n"
                formatted += f"{content}\n\n"
        
        return formatted
    
    def single_turn_rag(
        self,
        query: str,
        retrieval_method: str,
        top_k: int,
        generation_model: str,
        show_intermediate: bool
    ) -> Tuple[str, str, str]:
        """
        单轮RAG查询
        
        Returns:
            (answer, retrieved_docs, intermediate_steps)
        """
        if not query.strip():
            return "请输入查询问题", "", ""
        
        # ===== 这里调用你的实际实现 =====
        # 1. 检索文档
        # retrieved_docs = self.retriever.search(
        #     query=query,
        #     method=retrieval_method,
        #     top_k=top_k
        # )
        
        # 2. 生成答案
        # result = self.generator.generate(
        #     query=query,
        #     documents=retrieved_docs,
        #     model=generation_model,
        #     track_steps=show_intermediate
        # )
        
        # ===== 示例返回（替换为你的实际结果） =====
        # 模拟检索结果
        retrieved_docs = [
            {
                'id': 'doc_001',
                'score': 0.95,
                'text': '这是第一个相关文档的内容...'
            },
            {
                'id': 'doc_002',
                'score': 0.87,
                'text': '这是第二个相关文档的内容...'
            }
        ]
        
        # 模拟生成的答案
        answer = f"根据检索到的文档，针对您的问题「{query}」的答案是：[这里是生成的答案]\n\n使用的检索方法: {retrieval_method}\n使用的生成模型: {generation_model}"
        
        # 模拟中间步骤（如果启用）
        intermediate_steps = []
        if show_intermediate:
            intermediate_steps = [
                {
                    'type': 'query_rewrite',
                    'original': query,
                    'content': f'{query} (优化后的查询)'
                },
                {
                    'type': 'reasoning',
                    'content': '分析查询意图 → 检索相关文档 → 综合信息生成答案'
                }
            ]
        
        # 格式化输出
        formatted_docs = self.format_retrieved_docs(retrieved_docs)
        formatted_steps = self.format_intermediate_steps(intermediate_steps) if show_intermediate else ""
        
        return answer, formatted_docs, formatted_steps
    
    def multi_turn_rag(
        self,
        query: str,
        chat_history: List[Tuple[str, str]],
        retrieval_method: str,
        top_k: int,
        generation_model: str
    ) -> Tuple[List[Tuple[str, str]], str, str]:
        """
        多轮对话RAG（Feature A）
        
        Returns:
            (updated_chat_history, retrieved_docs, context_info)
        """
        if not query.strip():
            return chat_history, "", ""
        
        # ===== Feature A实现：上下文感知检索 =====
        # 1. 重构查询（融合对话历史）
        # context = self._build_context(chat_history)
        # reformulated_query = self.generator.reformulate_query(query, context)
        
        # 2. 检索
        # retrieved_docs = self.retriever.search(
        #     query=reformulated_query,
        #     method=retrieval_method,
        #     top_k=top_k
        # )
        
        # 3. 生成答案（包含对话历史）
        # answer = self.generator.generate_with_history(
        #     query=query,
        #     documents=retrieved_docs,
        #     history=chat_history,
        #     model=generation_model
        # )
        
        # ===== 示例返回 =====
        reformulated_query = f"{query} (基于上下文重构)"
        
        retrieved_docs = [
            {
                'id': 'doc_multi_001',
                'score': 0.92,
                'text': '这是多轮对话检索到的相关文档...'
            }
        ]
        
        answer = f"根据对话上下文，{query}的答案是：[生成的答案]"
        
        # 更新对话历史
        chat_history.append((query, answer))
        
        # 格式化输出
        formatted_docs = self.format_retrieved_docs(retrieved_docs)
        context_info = f"### 🔄 上下文处理\n\n**原始查询**: {query}\n\n**重构后查询**: {reformulated_query}\n\n**对话轮次**: {len(chat_history)}"
        
        return chat_history, formatted_docs, context_info
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        return [], "", ""


def create_rag_interface():
    """创建Gradio界面"""
    
    rag_system = RAGInterface()
    
    # 自定义CSS样式
    custom_css = """
    .container {max-width: 1200px; margin: auto;}
    .output-box {border: 1px solid #ddd; border-radius: 8px; padding: 15px; background-color: #f9f9f9;}
    .highlight {background-color: #fff3cd; padding: 2px 5px; border-radius: 3px;}
    """
    
    with gr.Blocks(css=custom_css, title="RAG系统 - COMP5423项目") as demo:
        
        gr.Markdown(
            """
            # 🤖 检索增强生成(RAG)系统
            ### COMP5423 Natural Language Processing - Group Project
            
            本系统支持多种检索方法和生成模型，可进行单轮或多轮对话式问答。
            """
        )
        
        # ========== Tab 1: 单轮RAG ==========
        with gr.Tab("📝 单轮问答"):
            gr.Markdown("### 基础RAG功能：输入问题，检索相关文档并生成答案")
            
            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(
                        label="输入您的问题",
                        placeholder="例如：Where was Barack Obama born?",
                        lines=3
                    )
                    
                    with gr.Accordion("⚙️ 系统配置", open=False):
                        retrieval_method = gr.Dropdown(
                            choices=[
                                "BM25 (稀疏检索)",
                                "TF-IDF (稀疏检索)",
                                "E5 (密集检索)",
                                "BGE (密集检索)",
                                "GTE (密集检索)",
                                "ColBERT (多向量检索)",
                                "Hybrid (混合检索)"
                            ],
                            value="BM25 (稀疏检索)",
                            label="检索方法"
                        )
                        
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="检索文档数量 (Top-K)"
                        )
                        
                        generation_model = gr.Dropdown(
                            choices=[
                                "Qwen2.5-0.5B-Instruct",
                                "Qwen2.5-1.5B-Instruct",
                                "Qwen2.5-3B-Instruct",
                                "Qwen2.5-7B-Instruct"
                            ],
                            value="Qwen2.5-1.5B-Instruct",
                            label="生成模型"
                        )
                        
                        show_intermediate = gr.Checkbox(
                            label="显示中间推理过程 (Feature B)",
                            value=False
                        )
                    
                    submit_btn = gr.Button("🚀 提交查询", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    answer_output = gr.Textbox(
                        label="💡 生成的答案",
                        lines=6,
                        elem_classes="output-box"
                    )
                    
                    retrieved_docs_output = gr.Markdown(
                        label="📚 检索到的文档"
                    )
                    
                    intermediate_output = gr.Markdown(
                        label="🔍 中间推理过程",
                        visible=True
                    )
            
            # 绑定事件
            submit_btn.click(
                fn=rag_system.single_turn_rag,
                inputs=[
                    query_input,
                    retrieval_method,
                    top_k,
                    generation_model,
                    show_intermediate
                ],
                outputs=[
                    answer_output,
                    retrieved_docs_output,
                    intermediate_output
                ]
            )
            
            # 示例问题
            gr.Examples(
                examples=[
                    ["Where was Barack Obama born?"],
                    ["What is the capital of France?"],
                    ["Who won the 2024 US Presidential Election?"]
                ],
                inputs=query_input,
                label="📌 示例问题"
            )
        
        # ========== Tab 2: 多轮对话RAG ==========
        with gr.Tab("💬 多轮对话 (Feature A)"):
            gr.Markdown("### 支持上下文感知的多轮对话问答")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=400
                    )
                    
                    with gr.Row():
                        multi_query_input = gr.Textbox(
                            label="输入您的问题",
                            placeholder="继续提问...",
                            scale=4
                        )
                        multi_submit_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    clear_btn = gr.Button("🗑️ 清空对话", variant="stop")
                
                with gr.Column(scale=1):
                    with gr.Accordion("⚙️ 系统配置", open=False):
                        multi_retrieval = gr.Dropdown(
                            choices=[
                                "BM25 (稀疏检索)",
                                "E5 (密集检索)",
                                "BGE (密集检索)",
                                "Hybrid (混合检索)"
                            ],
                            value="E5 (密集检索)",
                            label="检索方法"
                        )
                        
                        multi_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=10,
                            step=1,
                            label="检索文档数量"
                        )
                        
                        multi_model = gr.Dropdown(
                            choices=[
                                "Qwen2.5-1.5B-Instruct",
                                "Qwen2.5-3B-Instruct",
                                "Qwen2.5-7B-Instruct"
                            ],
                            value="Qwen2.5-3B-Instruct",
                            label="生成模型"
                        )
                    
                    multi_docs_output = gr.Markdown(
                        label="📚 当前检索文档"
                    )
                    
                    context_output = gr.Markdown(
                        label="🔄 上下文处理信息"
                    )
            
            # 绑定事件
            multi_submit_btn.click(
                fn=rag_system.multi_turn_rag,
                inputs=[
                    multi_query_input,
                    chatbot,
                    multi_retrieval,
                    multi_top_k,
                    multi_model
                ],
                outputs=[
                    chatbot,
                    multi_docs_output,
                    context_output
                ]
            ).then(
                fn=lambda: "",
                outputs=multi_query_input
            )
            
            multi_query_input.submit(
                fn=rag_system.multi_turn_rag,
                inputs=[
                    multi_query_input,
                    chatbot,
                    multi_retrieval,
                    multi_top_k,
                    multi_model
                ],
                outputs=[
                    chatbot,
                    multi_docs_output,
                    context_output
                ]
            ).then(
                fn=lambda: "",
                outputs=multi_query_input
            )
            
            clear_btn.click(
                fn=rag_system.clear_history,
                outputs=[chatbot, multi_docs_output, context_output]
            )
        
        # ========== Tab 3: 系统信息 ==========
        with gr.Tab("ℹ️ 系统信息"):
            gr.Markdown(
                """
                ## 系统功能说明
                
                ### 📝 单轮问答
                - 支持多种检索方法：稀疏检索（BM25, TF-IDF）、密集检索（E5, BGE, GTE）、多向量检索（ColBERT）
                - 可选择不同的Qwen2.5生成模型
                - **Feature B**: 启用"显示中间推理过程"可查看查询重写、推理步骤等
                
                ### 💬 多轮对话
                - **Feature A**: 支持上下文感知的多轮对话
                - 自动追踪对话历史和实体
                - 将后续问题重构为独立查询
                
                ### 🎯 评分项覆盖
                
                #### 基础要求 ✅
                - ✅ 用户输入查询
                - ✅ 显示检索文档
                - ✅ 展示生成答案
                - ✅ 清晰易读的布局
                
                #### 加分项 ⭐
                - ⭐ 显示中间推理步骤（查询重写、子问题分解、自我检查）
                - ⭐ 支持多轮对话交互
                - ⭐ 上下文处理信息展示
                
                ## 技术栈
                - **前端框架**: Gradio
                - **检索方法**: BM25, TF-IDF, E5, BGE, GTE, ColBERT
                - **生成模型**: Qwen2.5 系列
                
                ## 开发团队
                COMP5423 - Group X
                """
            )
    
    return demo


# ============================================================================
# 启动界面
# ============================================================================
if __name__ == "__main__":
    demo = create_rag_interface()
    demo.launch(
        share=False,  # 设置为True可生成公开链接
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )