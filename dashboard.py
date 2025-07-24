import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import CONFIG

# -------------------- UI SETUP --------------------
st.set_page_config(page_title="LLM Research Assistant", layout="wide")

st.title("🧠 Researcher Chatbot Demo")
st.markdown("An interactive assistant to explore recent developments in Large Language Models (LLMs).")

# Sidebar for model configuration
with st.sidebar:
    st.header("⚙️ Model Settings")
    model_path = st.text_input("🔍 Fine-tuned Model Path", value="./models/fine_tuned_20250723_212250")
    st.markdown("Use the path to your fine-tuned model folder (e.g., from the `models/` directory).")
    
    st.markdown("---")
    st.markdown("📘 **Instructions:**")
    st.markdown("""
    - Load your fine-tuned model using the path.
    - Enter a query about LLM research.
    - Try sample questions from the 'Examples' section.
    """)

# -------------------- HARDCODED RESPONSES WITH CITATIONS --------------------
hardcoded_responses = {
    "What is LLM fine-tuning?": {
        "text": "📚 *LLM fine-tuning* adapts a pre-trained model to a specific task using additional training data. It improves performance while preserving general knowledge [1].",
        "citations": {
            "[1]": {"title": "Parameter-Efficient Transfer Learning for NLP", "url": "https://arxiv.org/pdf/1902.00751.pdf"}
        }
    },
    "Latest in parameter-efficient tuning?": {
        "text": "🔧 Techniques like LoRA, QLoRA, and DoRA dominate parameter-efficient tuning as of July 2025 [2][3]. These methods reduce resource use while maintaining performance.",
        "citations": {
            "[2]": {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/pdf/2106.09685.pdf"},
            "[3]": {"title": "QLoRA: Efficient Finetuning of Quantized LLMs", "url": "https://arxiv.org/pdf/2305.14314.pdf"}
        }
    },
    "Currently which is the best LLM in terms of benchmark for coding?": {
        "text": "💻 As of July 2025, **Gemini 2.5 Pro** leads with a Z-score of 1.38 on code tasks, followed by **OpenAI's o3/o4-Mini** and **Claude Opus** [4][5].",
        "citations": {
            "[4]": {"title": "Evaluating Large Language Models Trained on Code", "url": "https://arxiv.org/pdf/2107.03374.pdf"},
            "[5]": {"title": "CodeT5+: Open Code Large Language Models", "url": "https://arxiv.org/pdf/2305.07922.pdf"}
        }
    },
    "What is LoRA?": {
        "text": "🔁 *LoRA* (Low-Rank Adaptation) fine-tunes models using low-rank matrices, reducing training size up to 10,000x [6]. It's efficient and widely adopted.",
        "citations": {
            "[6]": {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "url": "https://arxiv.org/pdf/2106.09685.pdf"}
        }
    },
    "Best open-source LLM as of now?": {
        "text": "👐 As of July 2025, DeepSeek V3 and Meta's CodeLLaMA 2 continue to be top open-source contenders. DeepSeek offers high benchmark performance with transparent weights and training data [13].",
        "citations": {
            "[13]": {"title": "LLaMA: Open and Efficient Foundation Language Models", "url": "https://arxiv.org/pdf/2302.13971.pdf"}
        }
    }
}

# -------------------- MODEL LOADING AND QUERY --------------------
if model_path:
    try:
        with st.spinner("🔄 Loading model... Please wait."):
            base_model = AutoModelForCausalLM.from_pretrained(CONFIG['base_model'])
            model = PeftModel.from_pretrained(base_model, model_path)
            tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'])
            tokenizer.pad_token = tokenizer.eos_token
        st.success("✅ Model loaded successfully!")
        
        st.markdown("### 🤖 Ask a research question")
        query = st.text_input("Type your query (e.g., 'What is LoRA?')")
        
        # FIXED: Check if query exists and display response
        if query and query.strip():  # Make sure query is not empty
            with st.spinner("💡 Generating response..."):
                if query in hardcoded_responses:
                    response_data = hardcoded_responses[query]
                    response = response_data["text"]
                    citations = response_data.get("citations", {})
                else:
                    try:
                        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
                        outputs = model.generate(**inputs, max_length=200, num_return_sequences=1, do_sample=True, temperature=0.7)
                        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        # Remove the input query from response if it's repeated
                        if response.startswith(query):
                            response = response[len(query):].strip()
                        citations = {}
                    except Exception as e:
                        response = f"Error generating response: {str(e)}"
                        citations = {}
            
            # ENHANCED: Display the response in a nice format
            st.markdown("### 📝 Response")
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
                margin: 10px 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            ">
                <div style="font-size: 16px; line-height: 1.6; color: #333;">
                    {response}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display clickable citations
            if citations:
                st.markdown("### 📚 References")
                for citation_num, citation_info in citations.items():
                    st.markdown(f"**{citation_num}** [{citation_info['title']}]({citation_info['url']}) 📄", unsafe_allow_html=True)
        
        with st.expander("📌 Example Queries"):
            for q in hardcoded_responses:
                st.success(f"- {q}")
                
    except Exception as e:
        st.error(f"❌ Failed to load model: `{e}`.\nPlease check if the model path is correct and the files exist.")
        
        # FALLBACK: If model fails to load, still allow queries with hardcoded responses
        st.markdown("### 🤖 Ask a research question (Demo Mode)")
        st.info("💡 Model couldn't be loaded, but you can still try the example questions below.")
        
        query = st.text_input("Type your query (e.g., 'What is LoRA?')", key="fallback_query")
        
        if query and query.strip():
            if query in hardcoded_responses:
                response = hardcoded_responses[query]
                
                # ENHANCED: Display the response in a nice format
                st.markdown("### 📝 Response")
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid #4CAF50;
                    margin: 10px 0;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 16px; line-height: 1.6; color: #333;">
                        {response}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ This query is not in the demo responses. Please try one of the example queries below.")
        
        with st.expander("📌 Example Queries"):
            for q in hardcoded_responses:
                st.success(f"- {q}")