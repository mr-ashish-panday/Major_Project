"""
ScholarMind Terminal Chatbot - Smart Router v2
Routes questions:
  SIMPLE ("what is X?") → base model answers directly (clean, accurate)
  RESEARCH ("latest findings in X?") → trained model + RAG + verify pipeline
Features: conversation memory, response cache, question expansion, smart prefixes
"""
# === SUPPRESS ALL WARNINGS ===
import warnings
warnings.filterwarnings("ignore")
import os, re
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
import logging
logging.disable(logging.WARNING)

import sys, glob, time, threading, io, torch, hashlib
from collections import OrderedDict

_stderr = sys.stderr
sys.stderr = io.StringIO()
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
sys.stderr = _stderr

from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIG
from agents.vector_store import VectorStoreAgent

SYS = "<" + "|system|" + ">"
END = "<" + "|end|" + ">"
USR = "<" + "|user|" + ">"
AST = "<" + "|assistant|" + ">"

# Keywords that indicate a RESEARCH question (needs trained model + RAG)
RESEARCH_KEYWORDS = [
    "latest", "recent", "new", "advancement", "finding", "discover",
    "state of the art", "sota", "breakthrough", "cutting edge",
    "trend", "emerging", "novel", "current research", "2024", "2025",
    "2026", "published", "paper", "study", "compare", "benchmark",
    "outperform", "improve upon", "better than", "challenge",
]

# Synonyms for question expansion (better RAG retrieval)
EXPANSION_MAP = {
    "llm": "large language model GPT BERT transformer",
    "lora": "low-rank adaptation parameter efficient fine-tuning PEFT",
    "rag": "retrieval augmented generation knowledge base vector search",
    "nlp": "natural language processing text understanding sentiment",
    "transformer": "self-attention encoder decoder BERT GPT architecture",
    "fine-tuning": "transfer learning adaptation domain-specific training",
    "distillation": "knowledge distillation teacher student model compression",
    "attention": "self-attention multi-head attention mechanism transformer",
    "embedding": "vector representation word2vec semantic space",
    "quantization": "model compression 4-bit 8-bit inference optimization",
    "rlhf": "reinforcement learning human feedback alignment reward model",
    "diffusion": "diffusion model denoising generative image synthesis",
    "gan": "generative adversarial network discriminator generator",
    "cnn": "convolutional neural network image classification vision",
    "rnn": "recurrent neural network LSTM GRU sequence modeling",
}


def thinking_animation(stop_event):
    frames = ["  Thinking.  ", "  Thinking.. ", "  Thinking..."]
    i = 0
    while not stop_event.is_set():
        print(f"\r{frames[i % 3]}", end="", flush=True)
        i += 1
        time.sleep(0.4)
    print("\r" + " " * 20 + "\r", end="", flush=True)


def is_research_question(question):
    """Route: is this a research question or a simple definition?"""
    q_lower = question.lower()
    for kw in RESEARCH_KEYWORDS:
        if kw in q_lower:
            return True
    return False


def load_system():
    print()
    print("=" * 60)
    print("  ScholarMind - AI Research Assistant")
    print("  Fine-tuned Phi-3 + 20,818 Research Papers")
    print("=" * 60)

    print("\n  Loading knowledge base...", end=" ", flush=True)
    vs = VectorStoreAgent(CONFIG)
    stats = vs.get_stats()
    print(f"{stats.get('total_documents', 0)} documents")

    print("  Loading model...", end=" ", flush=True)
    t0 = time.time()
    old = sys.stderr
    sys.stderr = io.StringIO()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb, device_map="auto",
        trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
        attn_implementation="eager",
    )
    tok = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True, token=os.environ.get("HF_TOKEN"),
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapters = sorted(glob.glob("models/fine_tuned_*"))
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[-1])

    sys.stderr = old
    print(f"done ({time.time() - t0:.1f}s)")
    print()
    print("  Ask any AI/ML research question.")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    return model, tok, vs


def _gen(model, tok, system, user, max_tokens=150, use_adapter=True, prefix=""):
    """Generate text with optional output prefix to guide the start."""
    prompt = (SYS + "\n" + system + END + "\n"
              + USR + "\n" + user + END + "\n"
              + AST + "\n" + prefix)
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(model.device)

    old = sys.stderr
    sys.stderr = io.StringIO()

    # Stop sequences to prevent rambling
    stop_ids = tok.encode("\n\n", add_special_tokens=False)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        no_repeat_ngram_size=4,
        pad_token_id=tok.pad_token_id,
    )

    with torch.no_grad():
        if not use_adapter and hasattr(model, 'disable_adapter'):
            with model.disable_adapter():
                out = model.generate(**gen_kwargs)
        else:
            out = model.generate(**gen_kwargs)

    sys.stderr = old

    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = tok.decode(gen, skip_special_tokens=True).strip()
    # Prepend the prefix back (it was in the prompt but decoded tokens start after)
    if prefix and not ans.startswith(prefix.strip()):
        ans = prefix.strip() + " " + ans
    return _clean(ans)


def _clean(text):
    """Strip meta-commentary, fake citations, and fix stuck words."""
    if not text:
        return text
    # Remove code fences
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = text.replace('```', '')
    # Remove meta-commentary
    for p in [
        r"Here'?t?\s*be\s+any\s+spelled.*?(?:here you go\s*:?\s*)",
        r"(?:I will|Let me|I have)\s+(?:proofread|check|review|correct).*?(?::\s*|\.\.\.+\s*)",
        r"(?:Here (?:is|are)|Below is).*?(?:corrected|revised|polished|fixed).*?(?::\s*|\.\.\.+\s*)",
        r"Let me know if.*$", r"Thankyou\s*!*\s*$", r"Thank you\s*!*\s*$",
        r"Please note that.*$", r"\[Text remains unchanged\]",
        r"^.*?here you go\s*:?\s*", r"I hope (?:this|that).*$",
    ]:
        text = re.sub(p, '', text, flags=re.IGNORECASE | re.DOTALL)
    # Remove fake URLs/citations
    text = re.sub(r'\(https?://[^)]*\)', '', text)
    text = re.sub(r'https?://\S+', '', text)
    # Fix stuck-together words (camelCase boundaries): "effectivenessthis" → "effectiveness this"
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Fix lowercase stuck words: "effectivenessthis" pattern
    # Split words that have 15+ chars with no space (likely stuck)
    text = re.sub(r'([a-z]{6,})([a-z]{6,})', lambda m: m.group(1) + ' ' + m.group(2) if len(m.group(0)) > 14 else m.group(0), text)
    # Remove emoji/non-ascii
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Trim at last sentence
    if text and text[-1] not in ".!?":
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > 20:
            text = text[:last + 1]
    return text.strip()


def _validate_answer(text, question):
    """Check answer quality. Returns (is_ok, reason)."""
    if not text or len(text) < 30:
        return False, "too_short"
    # Check for AI/ML relevance (at least one keyword should appear)
    aiml_terms = ["model", "learning", "neural", "train", "data", "language",
                  "transformer", "attention", "parameter", "llm", "nlp",
                  "fine-tun", "inference", "network", "ai", "ml", "deep",
                  "bert", "gpt", "lora", "adapter", "embedding"]
    text_lower = text.lower()
    has_aiml = any(t in text_lower for t in aiml_terms)
    q_lower = question.lower()
    is_aiml_question = any(t in q_lower for t in aiml_terms) or is_research_question(question)
    if is_aiml_question and not has_aiml:
        return False, "off_topic"
    # Check for ACTUAL garbling: lowercase words stuck together without space
    # Pattern: two runs of 5+ lowercase letters with no space between them
    # e.g. "effectivenessthis", "languagelearning" but NOT "implementation"
    stuck = re.findall(r'[a-z]{5,}[a-z]{5,}', text.replace(' ', ''))
    # Count instances where a 15+ char lowercase run exists in original text (no space)
    real_stuck = re.findall(r'(?<!\w)[a-z]{15,}(?!\w)', text)
    if len(real_stuck) > 2:
        return False, "garbled"
    return True, "ok"


# ============================================================
# SIMPLE PATH: Base model answers directly (adapter OFF)
# Few-shot examples + output prefix for accurate definitions
# ============================================================
def answer_simple(model, tok, question):
    """For simple definition questions - clean base model answer."""
    sys_msg = (
        "You are ScholarMind, an expert AI research assistant. "
        "Give accurate, concise definitions of AI/ML concepts. "
        "If the question is not about AI/ML, say: I specialize in AI and machine learning topics only. "
        "Structure: First define the term, then explain how it works, then give one application. "
        "Write exactly 2-3 grammatically perfect sentences."
    )
    # Few-shot examples to show exact expected format
    user = (
        "Example question: What is attention mechanism?\n"
        "Example answer: The attention mechanism is a component in neural networks that allows "
        "models to focus on relevant parts of the input sequence when producing an output. "
        "It computes weighted scores over input tokens, enabling the model to selectively "
        "attend to important information. Attention is the core building block of Transformer "
        "architectures used in models like BERT and GPT.\n\n"
        f"Now answer this question: {question}"
    )
    # Output prefix forces correct start
    prefix = _get_prefix(question)
    return _gen(model, tok, sys_msg, user,
                max_tokens=120, use_adapter=False, prefix=prefix)


def _get_prefix(question):
    """Generate a correct output prefix to guide the model's start."""
    q = question.lower().strip().rstrip('?').strip()
    prefixes = {
        # Core concepts
        "what is llm": "A Large Language Model (LLM) is",
        "what is an llm": "A Large Language Model (LLM) is",
        "what is lora": "Low-Rank Adaptation (LoRA) is",
        "what is qlora": "QLoRA (Quantized Low-Rank Adaptation) is",
        "what is peft": "Parameter-Efficient Fine-Tuning (PEFT) is",
        # Architectures
        "explain transformers": "Transformers are",
        "what is transformer": "A Transformer is",
        "what are transformers": "Transformers are",
        "what is attention mechanism": "The attention mechanism is",
        "what is attention": "The attention mechanism is",
        "what is self-attention": "Self-attention is",
        "what is multi-head attention": "Multi-head attention is",
        # Models
        "what is bert": "BERT (Bidirectional Encoder Representations from Transformers) is",
        "what is gpt": "GPT (Generative Pre-trained Transformer) is",
        "what is gpt-4": "GPT-4 is",
        "what is llama": "LLaMA (Large Language Model Meta AI) is",
        "what is mistral": "Mistral is",
        "what is phi": "Phi is",
        "what is gemini": "Gemini is",
        "what is claude": "Claude is",
        "what is t5": "T5 (Text-to-Text Transfer Transformer) is",
        # Techniques
        "what is fine-tuning": "Fine-tuning is",
        "what is fine tuning": "Fine-tuning is",
        "what is transfer learning": "Transfer learning is",
        "what is knowledge distillation": "Knowledge distillation is",
        "what is rag": "Retrieval-Augmented Generation (RAG) is",
        "what is rlhf": "Reinforcement Learning from Human Feedback (RLHF) is",
        "what is dpo": "Direct Preference Optimization (DPO) is",
        "what is quantization": "Quantization is",
        "what is pruning": "Pruning is",
        # Fundamentals
        "what is nlp": "Natural Language Processing (NLP) is",
        "what is deep learning": "Deep learning is",
        "what is machine learning": "Machine learning is",
        "what is neural network": "A neural network is",
        "what is embedding": "An embedding is",
        "what is tokenization": "Tokenization is",
        "what is backpropagation": "Backpropagation is",
        "what is gradient descent": "Gradient descent is",
        "what is batch normalization": "Batch normalization is",
        "what is dropout": "Dropout is",
        "what is overfitting": "Overfitting is",
        "what is underfitting": "Underfitting is",
        "what is regularization": "Regularization is",
        # Vision & generative
        "what is cnn": "A Convolutional Neural Network (CNN) is",
        "what is gan": "A Generative Adversarial Network (GAN) is",
        "what is diffusion model": "A diffusion model is",
        "what is stable diffusion": "Stable Diffusion is",
        "what is vae": "A Variational Autoencoder (VAE) is",
        "what is autoencoder": "An autoencoder is",
        # Sequence models
        "what is rnn": "A Recurrent Neural Network (RNN) is",
        "what is lstm": "Long Short-Term Memory (LSTM) is",
        # Training
        "what is catastrophic forgetting": "Catastrophic forgetting is",
        "what is continual learning": "Continual learning is",
        "what is curriculum learning": "Curriculum learning is",
        "what is few-shot learning": "Few-shot learning is",
        "what is zero-shot learning": "Zero-shot learning is",
        "what is prompt engineering": "Prompt engineering is",
        "what is in-context learning": "In-context learning is",
        "what is chain of thought": "Chain-of-thought prompting is",
        # Data
        "what is data augmentation": "Data augmentation is",
        "what is synthetic data": "Synthetic data is",
        "what is vector database": "A vector database is",
    }
    for key, val in prefixes.items():
        if key in q:
            return val
    return ""


def _expand_query(question):
    """Expand question with synonyms for better RAG retrieval."""
    q_lower = question.lower()
    expansions = [question]
    for term, synonyms in EXPANSION_MAP.items():
        if term in q_lower:
            expansions.append(synonyms)
    return " ".join(expansions)


# ============================================================
# RESEARCH PATH: Trained model + RAG + combine
# ============================================================
def research_draft(model, tok, question):
    """Trained model generates research-informed draft."""
    sys_msg = (
        "You are an AI research expert trained on the latest scientific papers. "
        "Answer about recent findings, developments, and research in AI/ML. "
        "Structure: First state the research area. Then describe 1-2 specific recent methods or findings. "
        "Then mention the impact or improvement. Be specific and technical."
    )
    return _gen(model, tok, sys_msg, question,
                max_tokens=100, use_adapter=True)


def search_papers(vs, question):
    """RAG search with question expansion for better retrieval."""
    expanded = _expand_query(question)
    results = vs.search(expanded, top_k=3)
    papers, evidence = [], []
    seen = set()
    for doc in results:
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        if title in seen or doc.get("score", 0) < 0.40:
            continue
        seen.add(title)
        idx = len(papers) + 1
        score = doc.get("score", 0)
        content = doc.get("content", "")[:200].replace("\n", " ").strip()
        papers.append({"id": idx, "title": title,
                        "authors": meta.get("authors", "Unknown"),
                        "score": round(score * 100)})
        evidence.append(f"[{idx}] {title}: {content}")
    return papers, "\n".join(evidence)


def research_combine(model, tok, question, draft, evidence):
    """Combine draft + paper evidence into final research answer."""
    sys_msg = (
        "You received a research draft and paper excerpts about AI/ML. "
        "Write a polished answer combining the key findings. "
        "Structure: First state the research topic. Then describe 1-2 key findings from the papers. "
        "Then mention the significance. "
        "Include citations [1], [2] when referencing specific papers. "
        "Fix any factual errors. Write 3-4 grammatically perfect sentences."
    )
    user = (
        f"Question: {question}\n\n"
        f"Research draft (may have errors):\n{draft}\n\n"
        f"Paper excerpts:\n{evidence}\n\n"
        f"Write a polished answer:"
    )
    return _gen(model, tok, sys_msg, user,
                max_tokens=150, use_adapter=False)


def wrap_text(text, width=68, indent=4):
    prefix = " " * indent
    words = text.split()
    lines, line = [], prefix
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = prefix + w
        else:
            line += (" " + w) if line.strip() else (prefix + w)
    if line.strip():
        lines.append(line)
    return "\n".join(lines)


# ============================================================
# Response cache for instant repeats
# ============================================================
class ResponseCache:
    """LRU cache for chatbot responses."""
    def __init__(self, maxsize=50):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def _key(self, question):
        q = question.lower().strip().rstrip('?!. ')
        return hashlib.md5(q.encode()).hexdigest()

    def get(self, question):
        k = self._key(question)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        return None

    def put(self, question, answer, papers, route):
        k = self._key(question)
        self._cache[k] = (answer, papers, route)
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


# ============================================================
# Conversation memory for follow-up questions
# ============================================================
class ConversationMemory:
    """Keeps last N turns for context."""
    def __init__(self, max_turns=5):
        self.history = []  # list of (question, answer)
        self.max_turns = max_turns

    def add(self, question, answer):
        self.history.append((question, answer))
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get_context(self):
        """Return conversation context string."""
        if not self.history:
            return ""
        lines = []
        for q, a in self.history[-3:]:  # last 3 turns max
            lines.append(f"User asked: {q}")
            lines.append(f"You answered: {a[:100]}...")
        return "\n".join(lines)

    def is_followup(self, question):
        """Detect if question is a follow-up."""
        followup_words = ["tell me more", "elaborate", "explain more",
                          "what about", "how about", "and ", "also",
                          "why", "can you explain", "more detail",
                          "what else", "continue", "go on"]
        q_lower = question.lower()
        return any(fw in q_lower for fw in followup_words) and len(self.history) > 0

    def expand_followup(self, question):
        """Expand a follow-up question with context from last turn."""
        if not self.history:
            return question
        last_q, last_a = self.history[-1]
        return f"{question} (context: previously discussed {last_q})"


def main():
    model, tok, vs = load_system()
    cache = ResponseCache()
    memory = ConversationMemory()

    while True:
        print()
        try:
            question = input("  You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!\n")
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\n  Goodbye!\n")
            break

        # Check cache first
        cached = cache.get(question)
        if cached:
            final, papers, route = cached
            print(f"  ScholarMind (cached) [{route}]:\n")
            print(wrap_text(final))
            if papers:
                print()
                print("  References:")
                for p in papers:
                    score_str = f" ({p['score']}%)"
                    print(f"    [{p['id']}] {p['title']}{score_str}")
                    print(f"        {p['authors']}")
            print()
            print("  " + "-" * 56)
            memory.add(question, final)
            continue

        # Expand follow-up questions with context
        effective_question = question
        if memory.is_followup(question):
            effective_question = memory.expand_followup(question)

        stop = threading.Event()
        anim = threading.Thread(target=thinking_animation, args=(stop,), daemon=True)
        anim.start()
        t0 = time.time()

        is_research = is_research_question(effective_question)
        papers = []

        # Generate answer (with retry if validation fails)
        for attempt in range(2):
            if is_research:
                # RESEARCH PATH: trained model draft → RAG → combine
                draft = research_draft(model, tok, effective_question)
                papers, evidence = search_papers(vs, effective_question)
                final = research_combine(model, tok, effective_question, draft, evidence)
            else:
                # SIMPLE PATH: base model answers directly
                final = answer_simple(model, tok, effective_question)

            # Validate answer
            is_ok, reason = _validate_answer(final, effective_question)
            if is_ok:
                break
            # If failed and first attempt, retry with stricter prompt
            if attempt == 0 and reason in ("off_topic", "garbled", "too_short"):
                if not is_research:
                    is_research = True
                    continue
                else:
                    continue

        stop.set()
        anim.join()
        latency = time.time() - t0

        route = "Research" if is_research else "Direct"
        print(f"  ScholarMind ({latency:.1f}s) [{route}]:\n")
        print(wrap_text(final))

        if papers:
            print()
            print("  References:")
            for p in papers:
                score_str = f" ({p['score']}%)"
                print(f"    [{p['id']}] {p['title']}{score_str}")
                print(f"        {p['authors']}")

        print()
        print("  " + "-" * 56)

        # Save to cache and memory
        cache.put(question, final, papers, route)
        memory.add(question, final)


if __name__ == "__main__":
    main()
