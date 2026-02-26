"""
ScholarMind Terminal Chatbot - Smart Pipeline
Step 1: Fine-tuned model generates domain answer (has research knowledge)
Step 2: RAG retrieves latest paper evidence
Step 3: Base model acts as SMART JUDGE - decides what info to use,
        whether citations are needed, and produces final polished answer
"""
# === SUPPRESS ALL WARNINGS ===
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONWARNINGS"] = "ignore"
import logging
logging.disable(logging.WARNING)

import sys
import glob
import time
import threading
import io
import torch

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

# Phi-3 tokens
SYS = "<" + "|system|" + ">"
END = "<" + "|end|" + ">"
USR = "<" + "|user|" + ">"
AST = "<" + "|assistant|" + ">"


def thinking_animation(stop_event):
    """Animated thinking dots."""
    frames = ["  Thinking.  ", "  Thinking.. ", "  Thinking..."]
    i = 0
    while not stop_event.is_set():
        print(f"\r{frames[i % 3]}", end="", flush=True)
        i += 1
        time.sleep(0.4)
    print("\r" + " " * 20 + "\r", end="", flush=True)


def load_system():
    """Load all components."""
    print()
    print("=" * 60)
    print("  ScholarMind - AI Research Assistant")
    print("  Fine-tuned Phi-3 + 20,818 Research Papers")
    print("=" * 60)

    print("\n  Loading knowledge base...", end=" ", flush=True)
    vector_store = VectorStoreAgent(CONFIG)
    stats = vector_store.get_stats()
    print(f"{stats.get('total_documents', 0)} documents")

    print("  Loading model...", end=" ", flush=True)
    t0 = time.time()

    old_err = sys.stderr
    sys.stderr = io.StringIO()

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapters = sorted(glob.glob("models/fine_tuned_*"))
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[-1])

    sys.stderr = old_err
    print(f"done ({time.time() - t0:.1f}s)")
    print()
    print("  Ask any AI/ML research question.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    return model, tokenizer, vector_store


def _gen(model, tokenizer, system, user, max_tokens=120):
    """Core generation - stderr suppressed."""
    prompt = (
        SYS + "\n" + system + END + "\n"
        + USR + "\n" + user + END + "\n"
        + AST + "\n"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    old = sys.stderr
    sys.stderr = io.StringIO()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            top_p=0.8,
            top_k=30,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    sys.stderr = old

    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = tokenizer.decode(gen, skip_special_tokens=True).strip()

    # Trim at last complete sentence
    if ans and ans[-1] not in ".!?":
        last = max(ans.rfind("."), ans.rfind("!"), ans.rfind("?"))
        if last > 20:
            ans = ans[:last + 1]
    return ans


# ============================================================
# STEP 1: Fine-tuned model (adapter ON) - domain answer
# ============================================================
def step1_finetuned_answer(model, tokenizer, question):
    """Fine-tuned model generates answer using its domain knowledge."""
    system = (
        "You are an AI research expert trained on scientific papers. "
        "Answer the question about AI/ML/NLP/LLMs with technical accuracy. "
        "If not about AI/ML, say: I only answer AI/ML research questions."
    )
    return _gen(model, tokenizer, system, question, max_tokens=120)


# ============================================================
# STEP 2: RAG search - get paper evidence
# ============================================================
def step2_search_papers(vector_store, question):
    """Search knowledge base for relevant papers."""
    results = vector_store.search(question, top_k=3)
    papers = []
    evidence = []
    seen = set()

    for doc in results:
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        if title in seen:
            continue
        seen.add(title)
        score = doc.get("score", 0)
        if score < 0.40:
            continue

        idx = len(papers) + 1
        content = doc.get("content", "")[:250].replace("\n", " ").strip()
        papers.append({
            "id": idx,
            "title": title,
            "authors": meta.get("authors", "Unknown"),
            "score": score,
        })
        evidence.append(f"[{idx}] {title} by {meta.get('authors','Unknown')}: {content}")

    return papers, "\n".join(evidence)


# ============================================================
# STEP 3: Base model (adapter OFF) - SMART JUDGE
# ============================================================
def step3_smart_judge(model, tokenizer, question, finetuned_answer, paper_evidence, has_papers):
    """Base model decides what to use and produces final answer."""
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()

    if has_papers:
        system = (
            "You are a smart research assistant. You are given:\n"
            "1) A draft answer from a domain expert\n"
            "2) Evidence from research papers [1], [2], [3]\n\n"
            "Your job: Decide the best way to answer.\n"
            "- If the question is SIMPLE (like 'what is X?'), write a clean answer "
            "using the draft. Do NOT add citations for basic definitions.\n"
            "- If the question asks about LATEST findings, new developments, or specific research, "
            "combine the draft with paper evidence and ADD citations like [1], [2].\n\n"
            "Write one polished paragraph with perfect grammar. "
            "Only cite when the question specifically needs research evidence."
        )
        user = (
            f"Question: {question}\n\n"
            f"Draft answer:\n{finetuned_answer}\n\n"
            f"Paper evidence:\n{paper_evidence}\n\n"
            f"Write the best final answer:"
        )
    else:
        system = (
            "You are a professional scientific writer. "
            "Rewrite this draft with perfect grammar and clear structure. "
            "Keep the same meaning. Write 3-4 polished sentences."
        )
        user = f"Question: {question}\n\nDraft:\n{finetuned_answer}"

    result = _gen(model, tokenizer, system, user, max_tokens=180)

    if hasattr(model, 'enable_adapter_layers'):
        model.enable_adapter_layers()

    return result


def wrap_text(text, width=68, indent=4):
    """Word-wrap for terminal."""
    prefix = " " * indent
    words = text.split()
    lines = []
    line = prefix
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
    model, tokenizer, vector_store = load_system()

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

        # Thinking animation
        stop = threading.Event()
        anim = threading.Thread(target=thinking_animation, args=(stop,), daemon=True)
        anim.start()

        t0 = time.time()

        # STEP 1: Fine-tuned model domain answer
        draft = step1_finetuned_answer(model, tokenizer, question)

        # STEP 2: RAG search for papers
        papers, evidence = step2_search_papers(vector_store, question)

        # STEP 3: Base model smart judge
        final = step3_smart_judge(
            model, tokenizer, question, draft, evidence, len(papers) > 0
        )

        stop.set()
        anim.join()
        latency = time.time() - t0

        # Display final answer
        print(f"  ScholarMind ({latency:.1f}s):\n")
        print(wrap_text(final))

        # Show sources only if papers were found
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
