"""
ScholarMind Dashboard v2.0 — Premium Research Assistant with Battle Mode
A Streamlit-based interface with RAG chatbot + Model Arena + Evolution tracking.

Tabs:
    1. 💬 Chat — RAG-powered Q&A with Phi-3
    2. ⚔️ Model Arena — Side-by-side ScholarFormer vs Phi-3
    3. 📈 Evolution — Distillation progress & training metrics
    4. 📊 Comparison — Head-to-head evaluation results
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="ScholarMind | AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Dark theme with gradient accents */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0a0b0;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat container */
    .chat-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Message styling */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 1rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        padding: 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 1rem 0;
        max-width: 85%;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Citation styling */
    .citation-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 3px solid #667eea;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.9rem;
    }
    
    .citation-title {
        color: #667eea;
        font-weight: 600;
    }
    
    .citation-authors {
        color: #888;
        font-size: 0.8rem;
    }
    
    /* Stats cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-3px);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Arena model cards */
    .model-card-sf {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.15) 0%, rgba(255, 87, 34, 0.1) 100%);
        border: 1px solid rgba(255, 152, 0, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    .model-card-phi {
        background: linear-gradient(135deg, rgba(106, 27, 154, 0.15) 0%, rgba(156, 39, 176, 0.1) 100%);
        border: 1px solid rgba(156, 39, 176, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }
    
    /* Winner badge */
    .winner-badge {
        background: linear-gradient(90deg, #ffd700, #ffaa00);
        color: #000;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        padding: 1rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 15, 26, 0.95);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==================== SESSION STATE ====================
if 'messages' not in st.session_state:
    st.session_state.messages = []


# ==================== HELPER FUNCTIONS ====================

def format_citations(citations: list) -> str:
    """Format citations as HTML."""
    if not citations:
        return ""
    
    html = "<div style='margin-top: 1.5rem;'><h4 style='color: #667eea;'>📚 Sources</h4>"
    
    for cite in citations:
        title = cite.get('title', 'Unknown')
        authors = cite.get('authors', 'Unknown')
        url = cite.get('url', '')
        score = cite.get('score', 0)
        
        html += f"""
        <div class="citation-box">
            <span class="citation-title">[{cite['id']}] {title}</span><br>
            <span class="citation-authors">👤 {authors}</span>
            {f'<br><a href="{url}" target="_blank" style="color: #667eea;">🔗 View Paper</a>' if url else ''}
            <span style="float: right; color: #888;">Relevance: {score:.0%}</span>
        </div>
        """
    
    html += "</div>"
    return html


def load_comparison_reports() -> list:
    """Load all comparison report JSONs."""
    report_dir = os.path.join(CONFIG.get('logs_dir', './logs'), 'comparisons')
    if not os.path.exists(report_dir):
        return []
    
    reports = []
    for f in sorted(glob.glob(os.path.join(report_dir, 'comparison_*.json')), reverse=True):
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                reports.append(json.load(fh))
        except Exception:
            continue
    return reports


def load_distillation_results() -> dict:
    """Load pipeline results with distillation cycle history."""
    results_path = os.path.join(CONFIG.get('scholarformer_dir', './models/scholarformer'), 'pipeline_results.json')
    if os.path.exists(results_path):
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Model selection
    model_dirs = []
    models_path = CONFIG.get('model_dir', './models')
    if os.path.exists(models_path):
        model_dirs = [d for d in os.listdir(models_path) 
                     if os.path.isdir(os.path.join(models_path, d)) and d.startswith('fine_tuned')]
    
    if model_dirs:
        selected_model = st.selectbox(
            "Phi-3 Adapter",
            options=model_dirs,
            index=len(model_dirs) - 1
        )
        model_path = os.path.join(models_path, selected_model)
    else:
        st.info("No fine-tuned models found.")
        model_path = None
    
    st.markdown("---")
    
    # Vector store stats
    st.markdown("## 📊 Knowledge Base")
    try:
        from agents.vector_store import VectorStoreAgent
        vector_store = VectorStoreAgent(CONFIG)
        stats = vector_store.get_stats()
        st.metric("Documents", stats.get('total_documents', 0))
    except Exception:
        vector_store = None
        st.warning("Vector store unavailable")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## 🚀 Quick Actions")
    
    if st.button("🔄 Refresh"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Project info
    st.markdown("## 🧠 ScholarMind v2.0")
    st.caption("Self-Evolving Research Assistant")
    st.caption("Phi-3 (3.8B) + ScholarFormer (122M)")


# ==================== MAIN CONTENT ====================
st.markdown('<h1 class="main-header">🧠 ScholarMind</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Self-Evolving LLM Research Assistant | RAG + Custom Transformer + Knowledge Distillation</p>', unsafe_allow_html=True)

# ==================== TAB LAYOUT ====================
tab_chat, tab_arena, tab_evolution, tab_comparison = st.tabs([
    "💬 Chat", "⚔️ Model Arena", "📈 Evolution", "📊 Comparison"
])


# ==================== TAB 1: CHAT ====================
with tab_chat:
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">RAG</div>
            <div class="stat-label">Retrieval System</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        doc_count = 0
        try:
            if vector_store:
                doc_count = vector_store.get_stats().get('total_documents', 0)
        except Exception:
            pass
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{doc_count}</div>
            <div class="stat-label">Paper Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">Phi-3</div>
            <div class="stat-label">Base Model</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-number">4-bit</div>
            <div class="stat-label">Quantization</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💬 Ask Your Research Question")

    # Display chat messages
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            # Render citations separately so inner HTML is not escaped
            if msg.get("citations_html"):
                st.markdown(msg["citations_html"], unsafe_allow_html=True)

    # Input
    with st.form(key="query_form", clear_on_submit=True):
        query_input = st.text_input(
            "Type your question...",
            placeholder="e.g., What are the latest advances in parameter-efficient fine-tuning?",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button("🔍 Search", use_container_width=True)

    # Example queries
    st.markdown("**💡 Try:**")
    example_cols = st.columns(3)
    examples = [
        "What is LoRA?",
        "Explain transformer attention",
        "How does QLoRA work?"
    ]
    for i, ex in enumerate(examples):
        with example_cols[i]:
            if st.button(ex, key=f"ex_{i}"):
                st.session_state.pending_query = ex

    # Handle query
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
    elif submit_button and query_input:
        query = query_input
    else:
        query = None

    if query:
        st.session_state.messages.append({'role': 'user', 'content': query})
        
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                if model_path and os.path.exists(model_path):
                    from agents.rag_pipeline import RAGPipeline
                    rag_pipeline = RAGPipeline(CONFIG, model_path)
                    
                    if rag_pipeline and rag_pipeline.is_loaded:
                        result = rag_pipeline.query(query, top_k=5)
                        response = result.get('answer', 'No answer generated.')
                        citations = result.get('citations', [])
                        citations_html = format_citations(citations)
                    else:
                        raise Exception("Model not loaded")
                elif vector_store:
                    results = vector_store.search(query, top_k=5)
                    if results:
                        context = "\n".join([f"[{i+1}] {r.get('content', '')[:300]}" for i, r in enumerate(results)])
                        response = f"**Retrieval mode:**\n\n{context[:1000]}"
                        citations_html = ""
                    else:
                        response = "No relevant documents found."
                        citations_html = ""
                else:
                    response = "Knowledge base not loaded."
                    citations_html = ""
                
                st.session_state.messages.append({
                    'role': 'assistant', 'content': response, 'citations_html': citations_html
                })
            except Exception as e:
                st.session_state.messages.append({
                    'role': 'assistant', 'content': f"⚠️ Error: {e}", 'citations_html': ''
                })
        
        st.rerun()


# ==================== TAB 2: MODEL ARENA ====================
with tab_arena:
    st.markdown("### ⚔️ Model Arena: ScholarFormer vs Phi-3")
    st.markdown("*Compare responses from both models on the same query*")
    
    reports = load_comparison_reports()
    
    if reports:
        latest = reports[0]
        
        # Show latest comparison results
        sf_data = latest.get('scholarformer', {})
        phi3_data = latest.get('phi3', {})
        
        sf_examples = sf_data.get('examples', [])
        phi3_examples = phi3_data.get('examples', [])
        
        if sf_examples and phi3_examples:
            # Query selector
            queries = [ex.get('query', f'Query {i+1}') for i, ex in enumerate(sf_examples)]
            selected_idx = st.selectbox("Select a query:", range(len(queries)),
                                        format_func=lambda i: queries[i])
            
            col_sf, col_phi = st.columns(2)
            
            with col_sf:
                st.markdown("""
                <div class="model-card-sf">
                    <h4>🔶 ScholarFormer (122M)</h4>
                    <p style="color: #ff9800; font-size: 0.85rem;">Custom Transformer • Retrieval-Fused • Section-Aware</p>
                </div>
                """, unsafe_allow_html=True)
                
                sf_ex = sf_examples[selected_idx]
                st.markdown(f"**Response:**")
                st.markdown(f"> {sf_ex.get('response', 'N/A')[:500]}")
                st.metric("Latency", f"{sf_ex.get('latency_ms', 0):.0f} ms")
            
            with col_phi:
                st.markdown("""
                <div class="model-card-phi">
                    <h4>🟣 Phi-3 (3.8B)</h4>
                    <p style="color: #9c27b0; font-size: 0.85rem;">QLoRA Fine-tuned • 4-bit Quantized • 56 Training Cycles</p>
                </div>
                """, unsafe_allow_html=True)
                
                phi3_ex = phi3_examples[selected_idx]
                st.markdown(f"**Response:**")
                st.markdown(f"> {phi3_ex.get('response', 'N/A')[:500]}")
                st.metric("Latency", f"{phi3_ex.get('latency_ms', 0):.0f} ms")
            
            # Speed comparison
            sf_lat = sf_ex.get('latency_ms', 1)
            phi3_lat = phi3_ex.get('latency_ms', 1)
            speedup = phi3_lat / sf_lat if sf_lat > 0 else 0
            
            st.markdown("---")
            
            speed_col1, speed_col2, speed_col3 = st.columns(3)
            with speed_col1:
                st.metric("🔶 SF Latency", f"{sf_lat:.0f} ms")
            with speed_col2:
                st.metric("🟣 Phi-3 Latency", f"{phi3_lat:.0f} ms")
            with speed_col3:
                st.metric("⚡ Speedup", f"{speedup:.1f}x", 
                          delta=f"ScholarFormer is {'faster' if speedup > 1 else 'slower'}")
        else:
            st.info("No side-by-side examples available. Run `python run_comparison.py` on the server first.")
    else:
        st.info("🔍 No comparison reports found yet.")
        st.markdown("""
        **To generate comparison data, run on the server:**
        ```bash
        cd ~/Major_Project
        python run_comparison.py
        ```
        This will evaluate both models and create a comparison report.
        """)


# ==================== TAB 3: EVOLUTION ====================
with tab_evolution:
    st.markdown("### 📈 Training & Distillation Evolution")
    
    pipeline_results = load_distillation_results()
    
    if pipeline_results and 'distillation' in pipeline_results:
        distill = pipeline_results['distillation']
        cycle_history = distill.get('cycle_history', [])
        
        if cycle_history:
            # Summary cards
            card_cols = st.columns(4)
            with card_cols[0]:
                st.metric("Total Cycles", len(cycle_history))
            with card_cols[1]:
                st.metric("Total Q&A Pairs", distill.get('total_qa_pairs', 0))
            with card_cols[2]:
                st.metric("Total Time", distill.get('total_time_human', 'N/A'))
            with card_cols[3]:
                final_loss = cycle_history[-1].get('training_loss', 0)
                initial_loss = cycle_history[0].get('training_loss', 0)
                improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                st.metric("Loss Improvement", f"{improvement:.1f}%", 
                          delta=f"{initial_loss:.2f} → {final_loss:.2f}")
            
            st.markdown("---")
            
            # Training loss chart
            st.markdown("#### 📉 Distillation Loss Over Cycles")
            
            import pandas as pd
            
            df = pd.DataFrame(cycle_history)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.line_chart(
                    df.set_index('cycle')['training_loss'],
                    use_container_width=True
                )
                st.caption("Training loss decreasing = student learning from teacher")
            
            with chart_col2:
                st.bar_chart(
                    df.set_index('cycle')['num_qa_pairs'],
                    use_container_width=True
                )
                st.caption("Q&A pairs generated per cycle")
            
            # Cumulative Q&A pairs
            st.markdown("#### 📊 Cumulative Knowledge Absorption")
            st.line_chart(
                df.set_index('cycle')['total_qa_accumulated'],
                use_container_width=True
            )
            st.caption("Total Q&A pairs trained on (cumulative)")
            
            # Cycle details table
            st.markdown("#### 📋 Cycle Details")
            display_df = df[['cycle', 'num_qa_pairs', 'total_qa_accumulated', 'training_loss', 'time_seconds']].copy()
            display_df['time_seconds'] = display_df['time_seconds'].apply(lambda x: f"{x/60:.1f} min")
            display_df.columns = ['Cycle', 'New Q&A Pairs', 'Total Pairs', 'Training Loss', 'Duration']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No distillation cycle data available.")
    else:
        st.info("No pipeline results found. Run training first.")
    
    # Pipeline timing
    if pipeline_results:
        st.markdown("---")
        st.markdown("#### ⏱️ Pipeline Timeline")
        
        time_col1, time_col2, time_col3 = st.columns(3)
        
        pretrain = pipeline_results.get('pretrain', {})
        finetune = pipeline_results.get('finetune', {})
        distill_info = pipeline_results.get('distillation', {})
        
        with time_col1:
            status = pretrain.get('status', 'N/A')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">Stage 1</div>
                <div class="stat-label">Pretraining: {status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with time_col2:
            status = finetune.get('status', 'N/A')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">Stage 2</div>
                <div class="stat-label">Fine-tuning: {status}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with time_col3:
            time_human = distill_info.get('total_time_human', 'N/A')
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">Stage 3</div>
                <div class="stat-label">Distillation: {time_human}</div>
            </div>
            """, unsafe_allow_html=True)


# ==================== TAB 4: COMPARISON ====================
with tab_comparison:
    st.markdown("### 📊 Head-to-Head Evaluation Results")
    
    reports = load_comparison_reports()
    
    if reports:
        latest = reports[0]
        
        sf = latest.get('scholarformer', {})
        phi3 = latest.get('phi3', {})
        summary = latest.get('summary', {})
        
        st.markdown(f"*Last comparison: {latest.get('timestamp', 'Unknown')[:19]}*")
        
        # Metrics comparison
        st.markdown("#### 🎯 Metric Comparison")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            sf_ppl = sf.get('perplexity', 0)
            phi3_ppl = phi3.get('perplexity', 0)
            winner = "🔶" if sf_ppl < phi3_ppl else "🟣"
            st.metric(f"{winner} Perplexity ↓", 
                      f"SF: {sf_ppl:.1f}", 
                      delta=f"Phi-3: {phi3_ppl:.1f}")
        
        with metric_cols[1]:
            sf_bleu = sf.get('bleu', 0)
            phi3_bleu = phi3.get('bleu', 0)
            winner = "🔶" if sf_bleu > phi3_bleu else "🟣"
            st.metric(f"{winner} BLEU ↑", 
                      f"SF: {sf_bleu:.4f}", 
                      delta=f"Phi-3: {phi3_bleu:.4f}")
        
        with metric_cols[2]:
            sf_rouge = sf.get('rouge_l', 0)
            phi3_rouge = phi3.get('rouge_l', 0)
            winner = "🔶" if sf_rouge > phi3_rouge else "🟣"
            st.metric(f"{winner} ROUGE-L ↑", 
                      f"SF: {sf_rouge:.4f}", 
                      delta=f"Phi-3: {phi3_rouge:.4f}")
        
        with metric_cols[3]:
            speedup = summary.get('speedup', 0)
            st.metric("⚡ Speedup", 
                      f"{speedup:.1f}x",
                      delta="ScholarFormer faster" if speedup > 1 else "Phi-3 faster")
        
        st.markdown("---")
        
        # Architecture comparison
        st.markdown("#### 🏗️ Architecture Comparison")
        
        arch_col1, arch_col2 = st.columns(2)
        
        with arch_col1:
            st.markdown("""
            <div class="model-card-sf">
                <h4>🔶 ScholarFormer</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            | Spec | Value |
            |------|-------|
            | **Parameters** | {sf.get('parameters', 'N/A'):,} |
            | **Architecture** | Custom Decoder-Only |
            | **Innovations** | Section-Aware PE + Retrieval Cross-Attn |
            | **Training** | WikiText → Papers → Distillation |
            | **VRAM (eval)** | {sf.get('vram_mb', 0):.0f} MB |
            | **Avg Latency** | {sf.get('avg_latency_ms', 0):.0f} ms |
            """)
        
        with arch_col2:
            st.markdown("""
            <div class="model-card-phi">
                <h4>🟣 Phi-3</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            | Spec | Value |
            |------|-------|
            | **Parameters** | ~3.8B (4-bit quantized) |
            | **Architecture** | Dense Decoder-Only |
            | **Fine-tuning** | QLoRA (r=32, α=64) |
            | **Training** | 56 cycles on 1,588 papers |
            | **VRAM (eval)** | {phi3.get('vram_mb', 0):.0f} MB |
            | **Avg Latency** | {phi3.get('avg_latency_ms', 0):.0f} ms |
            """)
        
        # Historical comparisons
        if len(reports) > 1:
            st.markdown("---")
            st.markdown("#### 📅 Comparison History")
            
            history_data = []
            for r in reports:
                history_data.append({
                    'Date': r.get('timestamp', '')[:10],
                    'SF Perplexity': r.get('scholarformer', {}).get('perplexity', 0),
                    'Phi-3 Perplexity': r.get('phi3', {}).get('perplexity', 0),
                    'SF BLEU': r.get('scholarformer', {}).get('bleu', 0),
                    'Phi-3 BLEU': r.get('phi3', {}).get('bleu', 0),
                    'Speedup': r.get('summary', {}).get('speedup', 0),
                })
            
            import pandas as pd
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.info("🔍 No comparison reports yet.")
        
        st.markdown("""
        **To run the evaluation, SSH into the server and execute:**
        ```bash
        cd ~/Major_Project
        python run_comparison.py
        ```
        
        **What it evaluates:**
        - 📊 **Perplexity** — Language modeling quality (lower = better)
        - 📝 **BLEU** — Text generation quality (higher = better)
        - 📖 **ROUGE-L** — Text overlap quality (higher = better)
        - ⚡ **Latency** — Inference speed (lower = better)
        
        **Expected runtime:** ~5-10 minutes on RTX 3080 Ti
        """)


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>ScholarMind v2.0 | Self-Evolving LLM Research Assistant</p>
    <p>🧠 Phi-3 (3.8B) + 🔶 ScholarFormer (122M) + 🔍 FAISS + ⚡ QLoRA</p>
</div>
""", unsafe_allow_html=True)