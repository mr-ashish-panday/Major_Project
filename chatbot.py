"""
ScholarMind Terminal Chatbot - Smart Router
Routes questions:
  SIMPLE ("what is X?") → base model answers directly (clean, accurate)
  RESEARCH ("latest findings in X?") → trained model + RAG + verify pipeline
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

import sys, glob, time, threading, io, torch

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


def _gen(model, tok, system, user, max_tokens=150, use_adapter=True):
    """Generate text. use_adapter=False tries context manager to disable adapter."""
    prompt = (SYS + "\n" + system + END + "\n"
              + USR + "\n" + user + END + "\n"
              + AST + "\n")
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(model.device)

    old = sys.stderr
    sys.stderr = io.StringIO()

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
            # Use context manager to cleanly disable adapter
            with model.disable_adapter():
                out = model.generate(**gen_kwargs)
        else:
            out = model.generate(**gen_kwargs)

    sys.stderr = old

    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = tok.decode(gen, skip_special_tokens=True).strip()
    return _clean(ans)


def _clean(text):
    """Strip meta-commentary and artifacts."""
    if not text:
        return text
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = text.replace('```', '')
    for p in [
        r"Here'?t?\s*be\s+any\s+spelled.*?(?:here you go\s*:?\s*)",
        r"(?:I will|Let me|I have)\s+(?:proofread|check|review|correct).*?(?::\s*|\.\.\.+\s*)",
        r"(?:Here (?:is|are)|Below is).*?(?:corrected|revised|polished|fixed).*?(?::\s*|\.\.\.+\s*)",
        r"Let me know if.*$", r"Thankyou\s*!*\s*$", r"Thank you\s*!*\s*$",
        r"Please note that.*$", r"\[Text remains unchanged\]",
        r"^.*?here you go\s*:?\s*", r"I hope (?:this|that).*$",
    ]:
        text = re.sub(p, '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in ".!?":
        last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
        if last > 20:
            text = text[:last + 1]
    return text.strip()


# ============================================================
# SIMPLE PATH: Base model answers directly (adapter OFF)
# ============================================================
def answer_simple(model, tok, question):
    """For simple definition questions - clean base model answer."""
    sys_msg = (
        "You are ScholarMind, an AI research assistant. "
        "Give a clear, accurate, and concise answer about AI/ML topics. "
        "If the question is not about AI/ML, politely decline. "
        "Write 2-3 sentences with perfect grammar."
    )
    return _gen(model, tok, sys_msg, question,
                max_tokens=120, use_adapter=False)


# ============================================================
# RESEARCH PATH: Trained model + RAG + combine
# ============================================================
def research_draft(model, tok, question):
    """Trained model generates research-informed draft."""
    sys_msg = (
        "You are an AI research expert trained on scientific papers. "
        "Answer about latest findings, developments, and research in AI/ML. "
        "Be specific and mention relevant techniques or methods."
    )
    return _gen(model, tok, sys_msg, question,
                max_tokens=100, use_adapter=True)


def search_papers(vs, question):
    """RAG search."""
    results = vs.search(question, top_k=3)
    papers, evidence = [], []
    seen = set()
    for doc in results:
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        if title in seen or doc.get("score", 0) < 0.40:
            continue
        seen.add(title)
        idx = len(papers) + 1
        content = doc.get("content", "")[:200].replace("\n", " ").strip()
        papers.append({"id": idx, "title": title,
                        "authors": meta.get("authors", "Unknown"),
                        "score": doc.get("score", 0)})
        evidence.append(f"[{idx}] {title}: {content}")
    return papers, "\n".join(evidence)


def research_combine(model, tok, question, draft, evidence):
    """Combine draft + paper evidence into final research answer."""
    sys_msg = (
        "You received a research draft and paper excerpts. "
        "Write a clean answer combining the key findings. "
        "Include citations [1], [2] when referencing papers. "
        "Fix any errors in the draft. Write 3-4 clear sentences."
    )
    user = (
        f"Question: {question}\n\n"
        f"Research draft:\n{draft}\n\n"
        f"Papers:\n{evidence}\n\n"
        f"Combined answer with citations:"
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


def main():
    model, tok, vs = load_system()

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

        stop = threading.Event()
        anim = threading.Thread(target=thinking_animation, args=(stop,), daemon=True)
        anim.start()
        t0 = time.time()

        is_research = is_research_question(question)

        if is_research:
            # RESEARCH PATH: trained model draft → RAG → combine
            draft = research_draft(model, tok, question)
            papers, evidence = search_papers(vs, question)
            final = research_combine(model, tok, question, draft, evidence)
        else:
            # SIMPLE PATH: base model answers directly
            final = answer_simple(model, tok, question)
            papers = []

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
                print(f"    [{p['id']}] {p['title']}")
                print(f"        {p['authors']}")

        print()
        print("  " + "-" * 56)


if __name__ == "__main__":
    main()
