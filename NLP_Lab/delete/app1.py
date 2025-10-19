import streamlit as st
import json
from datetime import datetime
from typing import List, Dict, Any

# Page configuration
st.set_page_config(
    page_title="RAG Question Answering System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .doc-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
    .workflow-step {
        background-color: #e8f4f8;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #2ecc71;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = []

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ System Configuration")
    
    # Retrieval method selection
    st.markdown("### 🔍 Retrieval Method")
    retrieval_method = st.selectbox(
        "Select Method",
        ["BM25 (Sparse)", "Dense - E5", "Dense - BGE", "Dense - GTE", 
         "LLM-based - Qwen3", "Multi-vector - ColBERT", "Hybrid Retrieval"],
        help="Choose the retrieval method for document search"
    )
    
    top_k = st.slider("Top-K Documents", min_value=1, max_value=20, value=10)
    
    # Generation model selection
    st.markdown("### 🤖 Generation Model")
    generation_model = st.selectbox(
        "Select Model",
        ["Qwen2.5-0.5B-Instruct", "Qwen2.5-1.5B-Instruct", 
         "Qwen2.5-3B-Instruct", "Qwen2.5-7B-Instruct"]
    )
    
    # Advanced features
    st.markdown("### 🚀 Advanced Features")
    enable_multi_turn = st.checkbox("Enable Multi-Turn Conversation", value=False)
    enable_agentic = st.checkbox("Enable Agentic Workflow", value=False)
    
    if enable_agentic:
        st.markdown("**Agentic Options:**")
        enable_query_rewrite = st.checkbox("Query Rewriting", value=True)
        enable_self_check = st.checkbox("Self-Checking", value=True)
        enable_reasoning = st.checkbox("Reasoning Steps", value=True)
    
    # Temperature control
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.conversation_context = []
        st.rerun()

# Main content area
st.markdown('<div class="main-header">🤖 RAG Question Answering System</div>', unsafe_allow_html=True)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["💬 Chat Interface", "📊 System Insights", "ℹ️ About"])

with tab1:
    # Query input area
    st.markdown("### Ask Your Question")
    
    col1, col2 = st.columns([5, 1])
    with col1:
        user_query = st.text_input(
            "Enter your question:",
            placeholder="e.g., Who won the 2024 Nobel Prize in Physics?",
            label_visibility="collapsed"
        )
    with col2:
        submit_button = st.button("🔍 Search", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and user_query:
        with st.spinner("Processing your query..."):
            # This is where you would call your RAG system
            # For demonstration, I'll show the structure
            
            # Store query in context if multi-turn is enabled
            if enable_multi_turn:
                st.session_state.conversation_context.append({
                    "role": "user",
                    "content": user_query,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # Simulated workflow steps (replace with actual system calls)
            workflow_steps = []
            
            if enable_agentic:
                st.markdown("### 🔄 Workflow Steps")
                
                if enable_query_rewrite:
                    with st.expander("📝 Step 1: Query Rewriting", expanded=True):
                        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
                        st.write("**Original Query:**", user_query)
                        # Simulated rewritten query
                        rewritten = f"Reformulated: {user_query}"
                        st.write("**Rewritten Query:**", rewritten)
                        st.markdown('</div>', unsafe_allow_html=True)
                        workflow_steps.append(f"Query rewritten: {rewritten}")
                
                if enable_reasoning:
                    with st.expander("🧠 Step 2: Query Planning", expanded=True):
                        st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
                        st.write("**Sub-queries generated:**")
                        st.write("1. Sub-query 1 (simulated)")
                        st.write("2. Sub-query 2 (simulated)")
                        st.markdown('</div>', unsafe_allow_html=True)
                        workflow_steps.append("Generated 2 sub-queries")
            
            # Retrieved Documents Section
            st.markdown("### 📚 Retrieved Documents")
            
            # Simulated retrieved documents (replace with actual retrieval results)
            retrieved_docs = [
                {
                    "id": "doc_001",
                    "score": 0.95,
                    "text": "This is a sample retrieved document that contains relevant information about the query. In a real system, this would come from your retrieval module.",
                    "rank": 1
                },
                {
                    "id": "doc_002",
                    "score": 0.87,
                    "text": "Another relevant document with supporting evidence. The retrieval system ranks documents based on similarity scores.",
                    "rank": 2
                },
                {
                    "id": "doc_003",
                    "score": 0.82,
                    "text": "Third document providing additional context. Multiple documents help the LLM generate comprehensive answers.",
                    "rank": 3
                }
            ]
            
            for doc in retrieved_docs[:3]:  # Show top 3
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**Rank #{doc['rank']} - Document ID:** `{doc['id']}`")
                    with col2:
                        st.markdown(f'<span class="score-badge">Score: {doc["score"]:.3f}</span>', unsafe_allow_html=True)
                    st.markdown(f'<div class="doc-card">{doc["text"]}</div>', unsafe_allow_html=True)
            
            with st.expander(f"View all {len(retrieved_docs)} documents"):
                for doc in retrieved_docs:
                    st.markdown(f"**{doc['rank']}. {doc['id']}** (Score: {doc['score']:.3f})")
                    st.text(doc['text'][:100] + "...")
                    st.divider()
            
            # Self-checking step (if enabled)
            if enable_agentic and enable_self_check:
                with st.expander("✅ Step 3: Answer Verification", expanded=False):
                    st.markdown('<div class="workflow-step">', unsafe_allow_html=True)
                    st.write("**Verification Status:** ✅ Passed")
                    st.write("**Evidence Found:** 3/3 documents support the answer")
                    st.write("**Hallucination Check:** No inconsistencies detected")
                    st.markdown('</div>', unsafe_allow_html=True)
                    workflow_steps.append("Answer verified successfully")
            
            # Generated Answer Section
            st.markdown("### 💡 Generated Answer")
            answer_container = st.container()
            with answer_container:
                # Simulated answer (replace with actual generation)
                generated_answer = f"Based on the retrieved documents, here is the answer to your question: '{user_query}'. This is a simulated response. In your actual system, this would be generated by the Qwen2.5 model using the retrieved context."
                
                st.success(generated_answer)
                
                # Supporting evidence
                st.markdown("**Supporting Evidence:**")
                st.write("- Document doc_001 (Score: 0.95)")
                st.write("- Document doc_002 (Score: 0.87)")
                st.write("- Document doc_003 (Score: 0.82)")
            
            # Store in chat history
            st.session_state.chat_history.append({
                "query": user_query,
                "answer": generated_answer,
                "retrieved_docs": retrieved_docs,
                "workflow_steps": workflow_steps if enable_agentic else [],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "retrieval": retrieval_method,
                    "model": generation_model,
                    "top_k": top_k
                }
            })
    
    # Chat History
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 Conversation History")
        
        for i, entry in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q{len(st.session_state.chat_history)-i}: {entry['query'][:50]}... - {entry['timestamp']}", expanded=False):
                st.markdown(f"**Question:** {entry['query']}")
                st.markdown(f"**Answer:** {entry['answer']}")
                st.markdown(f"**Retrieved Documents:** {len(entry['retrieved_docs'])}")
                st.markdown(f"**Configuration:** {entry['config']['retrieval']} + {entry['config']['model']}")

with tab2:
    st.markdown("### 📊 System Performance Insights")
    
    if st.session_state.chat_history:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", len(st.session_state.chat_history))
        
        with col2:
            avg_docs = sum(len(entry['retrieved_docs']) for entry in st.session_state.chat_history) / len(st.session_state.chat_history)
            st.metric("Avg Retrieved Docs", f"{avg_docs:.1f}")
        
        with col3:
            if st.session_state.chat_history:
                avg_score = sum(doc['score'] for entry in st.session_state.chat_history for doc in entry['retrieved_docs'][:3]) / (len(st.session_state.chat_history) * 3)
                st.metric("Avg Retrieval Score", f"{avg_score:.3f}")
        
        st.markdown("---")
        st.markdown("### Recent Query Statistics")
        
        # Display recent queries in a table
        if st.session_state.chat_history:
            recent_data = []
            for entry in st.session_state.chat_history[-10:]:
                recent_data.append({
                    "Timestamp": entry['timestamp'],
                    "Query Length": len(entry['query']),
                    "Docs Retrieved": len(entry['retrieved_docs']),
                    "Retrieval Method": entry['config']['retrieval']
                })
            
            st.dataframe(recent_data, use_container_width=True)
    else:
        st.info("No queries yet. Start asking questions to see insights!")

with tab3:
    st.markdown("### ℹ️ About This System")
    st.markdown("""
    This is a **Retrieval-Augmented Generation (RAG)** system designed for the COMP5423 course project.
    
    **Key Features:**
    - 🔍 Multiple retrieval methods (Sparse, Dense, Multi-vector)
    - 🤖 Qwen2.5 model series for generation
    - 💬 Multi-turn conversation support
    - 🚀 Agentic workflow with query rewriting and self-checking
    - 📊 Real-time performance monitoring
    
    **Dataset:** HotpotQA subset (HQ-small)
    
    **Team Members:**
    - Member 1: Role & Contribution %
    - Member 2: Role & Contribution %
    - Member 3: Role & Contribution %
    - Member 4: Role & Contribution %
    
    **Project Repository:** [GitHub Link]
    
    **Evaluation Metrics:**
    - Answer Accuracy: Exact Match (EM)
    - Retrieval Quality: nDCG@10
    """)
    
    st.markdown("---")
    st.markdown("### 🔧 System Architecture")
    st.markdown("""
    ```
    User Query
        ↓
    [Query Processing & Rewriting]
        ↓
    [Retrieval Module] → Document Collection
        ↓
    [Retrieved Documents (Top-K)]
        ↓
    [Generation Module] → LLM (Qwen2.5)
        ↓
    [Self-Checking & Verification]
        ↓
    Final Answer
    ```
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "COMP5423 Natural Language Processing | RAG System Project | Fall 2025"
    "</div>",
    unsafe_allow_html=True
)