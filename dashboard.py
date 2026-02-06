"""
ScholarMind Dashboard - Premium Research Assistant Interface
A Streamlit-based chatbot powered by RAG for LLM research questions.
"""

import os
import sys
import logging
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from agents.rag_pipeline import RAGPipeline
from agents.vector_store import VectorStoreAgent

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

if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None


# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def load_vector_store():
    """Load the vector store (cached)."""
    try:
        return VectorStoreAgent(CONFIG)
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        return None


@st.cache_resource
def load_rag_pipeline(model_path: str):
    """Load the RAG pipeline with model (cached)."""
    try:
        pipeline = RAGPipeline(CONFIG, model_path)
        return pipeline
    except Exception as e:
        logger.error(f"Failed to load RAG pipeline: {e}")
        return None


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
            "Select Model",
            options=model_dirs,
            index=len(model_dirs) - 1  # Default to latest
        )
        model_path = os.path.join(models_path, selected_model)
    else:
        st.info("No fine-tuned models found. Using base model.")
        model_path = None
    
    st.markdown("---")
    
    # Vector store stats
    st.markdown("## 📊 Knowledge Base")
    vector_store = load_vector_store()
    if vector_store:
        stats = vector_store.get_stats()
        st.metric("Documents", stats.get('total_documents', 0))
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## 🚀 Quick Actions")
    
    if st.button("🔄 Refresh Model"):
        st.cache_resource.clear()
        st.rerun()
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Example queries
    st.markdown("## 💡 Example Queries")
    examples = [
        "What is LoRA and how does it work?",
        "Explain QLoRA and its advantages",
        "Latest advances in LLM fine-tuning",
        "What are transformer architectures?",
        "Compare different parameter-efficient methods"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{example[:20]}"):
            st.session_state.pending_query = example


# ==================== MAIN CONTENT ====================
st.markdown('<h1 class="main-header">🧠 ScholarMind</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Self-Evolving LLM Research Assistant | Powered by RAG</p>', unsafe_allow_html=True)

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
    doc_count = vector_store.get_stats().get('total_documents', 0) if vector_store else 0
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

# Chat interface
st.markdown("### 💬 Ask Your Research Question")

# Display chat messages
for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f'<div class="user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message">🤖 {msg["content"]}{msg.get("citations_html", "")}</div>', unsafe_allow_html=True)

# Input area with form for Enter key to work
with st.form(key="query_form", clear_on_submit=True):
    query_input = st.text_input(
        "Type your question...",
        placeholder="e.g., What are the latest advances in parameter-efficient fine-tuning?",
        label_visibility="collapsed"
    )
    submit_button = st.form_submit_button("🔍 Search", use_container_width=True)

# Handle pending query from sidebar examples OR form submission
if 'pending_query' in st.session_state and st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None
elif submit_button and query_input:
    query = query_input
else:
    query = None

# Process query only if it's new
if query:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': query
    })
    
    with st.spinner("🔍 Searching knowledge base and generating response..."):
        try:
            # Try to use RAG pipeline with model for better answers
            if model_path and os.path.exists(model_path):
                # Load RAG pipeline with model (cached)
                rag_pipeline = load_rag_pipeline(model_path)
                
                if rag_pipeline and rag_pipeline.is_loaded:
                    # Use full RAG with model generation
                    result = rag_pipeline.query(query, top_k=5)
                    response = result.get('answer', 'No answer generated.')
                    citations = result.get('citations', [])
                    citations_html = format_citations(citations)
                else:
                    # Fall back to retrieval-only mode
                    st.warning("⚠️ Model not loaded. Using retrieval-only mode.")
                    raise Exception("Model not loaded")
            
            # Fall back to vector search only
            elif vector_store and vector_store.collection.count() > 0:
                results = vector_store.search(query, top_k=5)
                
                if results:
                    context_parts = []
                    citations = []
                    
                    for i, doc in enumerate(results, 1):
                        content = doc.get('content', '')[:500]
                        metadata = doc.get('metadata', {})
                        context_parts.append(f"[{i}] {content}")
                        citations.append({
                            'id': i,
                            'title': metadata.get('title', 'Unknown'),
                            'authors': metadata.get('authors', 'Unknown'),
                            'url': metadata.get('url', ''),
                            'score': doc.get('score', 0)
                        })
                    
                    context = "\n\n".join(context_parts)
                    response = f"""**Retrieved Context (retrieval-only mode):**

{context[:1500]}...

*Load a fine-tuned model for synthesized answers.*"""
                    citations_html = format_citations(citations)
                else:
                    response = "No relevant documents found."
                    citations_html = ""
            else:
                response = "Knowledge base empty. Run training first."
                citations_html = ""
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'citations_html': citations_html
            })
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            # Fall back to simple retrieval
            if vector_store:
                results = vector_store.search(query, top_k=5)
                if results:
                    context = "\n".join([f"[{i+1}] {r.get('content', '')[:300]}" for i, r in enumerate(results)])
                    response = f"**Retrieval mode (model loading failed):**\n\n{context[:1000]}..."
                    citations_html = ""
                else:
                    response = f"⚠️ Error: {str(e)}"
                    citations_html = ""
            else:
                response = f"⚠️ Error: {str(e)}"
                citations_html = ""
            
            st.session_state.messages.append({
                'role': 'assistant',
                'content': response,
                'citations_html': citations_html
            })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>ScholarMind v2.0 | Self-Evolving LLM Research Assistant</p>
    <p>Built with 🧠 Phi-3 + 🔍 ChromaDB + ⚡ QLoRA</p>
</div>
""", unsafe_allow_html=True)