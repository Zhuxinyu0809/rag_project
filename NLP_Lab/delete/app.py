import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any

# 页面配置
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .document-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .reasoning-step {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""
if 'show_reasoning' not in st.session_state:
    st.session_state.show_reasoning = False

# ============= 辅助函数 =============

def display_retrieved_documents(docs: List[Dict[str, Any]]):
    """显示检索到的文档"""
    st.subheader("📚 Retrieved Documents")
    
    for idx, doc in enumerate(docs, 1):
        with st.expander(f"Document {idx} - Score: {doc.get('score', 0):.4f}"):
            st.markdown(f"**Document ID:** `{doc.get('id', 'N/A')}`")
            st.markdown(f"**Content:**")
            st.write(doc.get('text', 'No content available'))

def display_reasoning_steps(steps: List[Dict[str, Any]]):
    """显示推理步骤（智能体工作流）"""
    st.subheader("🧠 Reasoning Process")
    
    for idx, step in enumerate(steps, 1):
        st.markdown(f"""
        <div class="reasoning-step">
            <strong>Step {idx}: {step.get('type', 'Unknown')}</strong><br>
            {step.get('content', 'No content')}
        </div>
        """, unsafe_allow_html=True)

def display_answer(answer: str):
    """显示答案"""
    st.markdown(f"""
    <div class="answer-box">
        <h3 style="color: #28a745; margin-top: 0;">✅ Answer</h3>
        <p style="font-size: 1.1rem; margin-bottom: 0;">{answer}</p>
    </div>
    """, unsafe_allow_html=True)

def display_chat_history():
    """显示对话历史"""
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry['question'])
        with st.chat_message("assistant"):
            st.write(entry['answer'])

# ============= 主界面 =============

# 标题
st.markdown('<h1 class="main-header">🤖 RAG Question Answering System</h1>', unsafe_allow_html=True)
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # 检索方法选择
    st.subheader("Retrieval Method")
    retrieval_method = st.selectbox(
        "Select retrieval method:",
        ["BM25", "Dense (E5)", "Dense (BGE)", "ColBERT", "Hybrid"],
        help="Choose the retrieval algorithm"
    )
    
    # 生成模型选择
    st.subheader("Generation Model")
    generation_model = st.selectbox(
        "Select model:",
        ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", 
         "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"],
        index=2
    )
    
    # 高级功能
    st.subheader("Advanced Features")
    enable_multiturn = st.checkbox("Enable Multi-turn Dialogue", value=False)
    enable_agentic = st.checkbox("Enable Agentic Workflow", value=False)
    show_reasoning = st.checkbox("Show Reasoning Steps", value=False)
    
    # 检索参数
    st.subheader("Retrieval Parameters")
    top_k = st.slider("Top K documents:", 1, 20, 10)
    
    # 生成参数
    st.subheader("Generation Parameters")
    temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max tokens:", 128, 1024, 512, 64)
    
    st.markdown("---")
    
    # 统计信息
    st.subheader("📊 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", len(st.session_state.chat_history))
    with col2:
        st.metric("Model", generation_model.split('-')[1])
    
    # 清除历史按钮
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# 主内容区域
tab1, tab2, tab3 = st.tabs(["💬 Query", "📜 History", "ℹ️ About"])

with tab1:
    # 查询输入区域
    st.subheader("Enter Your Question")
    
    query_input = st.text_area(
        "Question:",
        placeholder="Enter your question here...",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit_button = st.button("🔍 Search & Answer", type="primary", use_container_width=True)
    with col2:
        if enable_multiturn:
            st.info("Multi-turn: ON")
        else:
            st.text("Multi-turn: OFF")
    with col3:
        if enable_agentic:
            st.info("Agentic: ON")
        else:
            st.text("Agentic: OFF")
    
    # 处理查询
    if submit_button and query_input:
        with st.spinner("🔄 Processing your query..."):
            # ======= 这里接入你的RAG系统 =======
            # 示例数据 - 替换为实际的系统调用
            
            # 模拟检索结果
            retrieved_docs = [
                {
                    "id": "doc_001",
                    "text": "Barack Obama was born in Honolulu, Hawaii on August 4, 1961. He served as the 44th president of the United States from 2009 to 2017.",
                    "score": 0.9523
                },
                {
                    "id": "doc_045",
                    "text": "Michelle Obama was born Michelle LaVaughn Robinson on January 17, 1964, in Chicago, Illinois. She became the First Lady of the United States in 2009.",
                    "score": 0.8834
                },
                {
                    "id": "doc_102",
                    "text": "Hawaii is a U.S. state located in the Pacific Ocean. It became the 50th state in 1959.",
                    "score": 0.7291
                }
            ]
            
            # 模拟推理步骤（如果启用了智能体工作流）
            reasoning_steps = []
            if enable_agentic:
                reasoning_steps = [
                    {
                        "type": "Query Analysis",
                        "content": f"Analyzing query: '{query_input}'. Identified as a factual question requiring biographical information."
                    },
                    {
                        "type": "Query Decomposition",
                        "content": "Decomposed into sub-queries: 1) Who is Barack Obama? 2) Where was he born?"
                    },
                    {
                        "type": "Retrieval",
                        "content": f"Retrieved {len(retrieved_docs)} relevant documents using {retrieval_method} method."
                    },
                    {
                        "type": "Answer Synthesis",
                        "content": "Synthesizing answer from retrieved evidence..."
                    },
                    {
                        "type": "Self-Check",
                        "content": "Verified answer against retrieved documents. Confidence: High."
                    }
                ]
            
            # 模拟生成的答案
            answer = "Barack Obama was born in Honolulu, Hawaii, on August 4, 1961."
            
            # ======= 实际系统调用示例 =======
            # from your_rag_system import RAGSystem
            # rag_system = RAGSystem(
            #     retrieval_method=retrieval_method,
            #     generation_model=generation_model,
            #     top_k=top_k,
            #     temperature=temperature,
            #     max_tokens=max_tokens
            # )
            # result = rag_system.query(
            #     query_input,
            #     enable_multiturn=enable_multiturn,
            #     enable_agentic=enable_agentic
            # )
            # retrieved_docs = result['retrieved_docs']
            # answer = result['answer']
            # reasoning_steps = result.get('reasoning_steps', [])
            
        # 显示结果
        st.success("✨ Query processed successfully!")
        
        # 显示答案
        display_answer(answer)
        
        # 显示检索到的文档
        display_retrieved_documents(retrieved_docs[:top_k])
        
        # 显示推理步骤（如果启用）
        if enable_agentic and show_reasoning and reasoning_steps:
            display_reasoning_steps(reasoning_steps)
        
        # 保存到历史记录
        st.session_state.chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": query_input,
            "answer": answer,
            "retrieved_docs": retrieved_docs,
            "reasoning_steps": reasoning_steps if enable_agentic else [],
            "config": {
                "retrieval_method": retrieval_method,
                "model": generation_model,
                "top_k": top_k
            }
        })
        
        # 导出选项
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            # 导出当前结果
            result_json = json.dumps({
                "id": f"query_{len(st.session_state.chat_history)}",
                "question": query_input,
                "answer": answer,
                "retrieved_docs": [[doc['id'], doc['score']] for doc in retrieved_docs[:top_k]]
            }, indent=2)
            
            st.download_button(
                label="📥 Download Result (JSON)",
                data=result_json,
                file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # 复制答案到剪贴板
            st.code(answer, language=None)

with tab2:
    st.subheader("📜 Query History")
    
    if st.session_state.chat_history:
        for idx, entry in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Query {len(st.session_state.chat_history) - idx + 1} - {entry['timestamp']}"):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Answer:** {entry['answer']}")
                st.markdown(f"**Configuration:** {entry['config']['retrieval_method']} + {entry['config']['model']}")
                
                if st.button(f"View Details", key=f"details_{idx}"):
                    st.json(entry)
    else:
        st.info("No query history yet. Start by asking a question in the Query tab!")
    
    # 批量导出
    if st.session_state.chat_history:
        st.markdown("---")
        all_results = []
        for idx, entry in enumerate(st.session_state.chat_history, 1):
            all_results.append({
                "id": f"query_{idx}",
                "question": entry['question'],
                "answer": entry['answer'],
                "retrieved_docs": [[doc['id'], doc['score']] for doc in entry['retrieved_docs'][:10]]
            })
        
        all_results_json = "\n".join([json.dumps(r) for r in all_results])
        
        st.download_button(
            label="📥 Export All Results (JSONL)",
            data=all_results_json,
            file_name=f"test_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="application/json"
        )

with tab3:
    st.subheader("ℹ️ About This System")
    
    st.markdown("""
    ### COMP5423 RAG Question Answering System
    
    This system implements a **Retrieval-Augmented Generation (RAG)** pipeline for multi-hop question answering
    based on the HotpotQA dataset.
    
    #### Features:
    - 🔍 **Multiple Retrieval Methods**: BM25, Dense Retrieval, ColBERT, Hybrid
    - 🤖 **Flexible Generation**: Qwen2.5 models (0.5B to 7B)
    - 💬 **Multi-turn Dialogue**: Context-aware conversation
    - 🧠 **Agentic Workflow**: Query decomposition, self-checking, reasoning
    - 📊 **Comprehensive Evaluation**: EM and nDCG@10 metrics
    
    #### System Architecture:
    ```
    User Query → Retrieval Module → Generation Module → Answer
                      ↓                    ↓
                 Vector Store        LLM (Qwen2.5)
    ```
    
    #### Usage:
    1. Configure retrieval and generation settings in the sidebar
    2. Enter your question in the Query tab
    3. View retrieved documents and generated answer
    4. Export results for evaluation
    
    #### Developed by:
    Group X - COMP5423 Natural Language Processing
    
    📧 Contact: {runyang.you, xin404.zhang}@connect.polyu.hk
    """)
    
    # 系统信息
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Version", "1.0.0")
    with col2:
        st.metric("Dataset", "HQ-small")
    with col3:
        st.metric("Documents", "144,718")

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "COMP5423 Group Project - RAG System | PolyU 2025"
    "</div>",
    unsafe_allow_html=True
)