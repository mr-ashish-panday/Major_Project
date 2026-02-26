"""
ScholarMind Terminal Chatbot
Fine-tuned Phi-3 answers directly + papers shown as references.
"""
# === AGGRESSIVE WARNING SUPPRESSION ===
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
import contextlib
import io
import torch

# Redirect stderr during imports to catch C-level warnings
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
    """Show animated thinking dots."""
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

    # Vector store
    print("\n  Loading knowledge base...", end=" ", flush=True)
    vector_store = VectorStoreAgent(CONFIG)
    stats = vector_store.get_stats()
    print(f"{stats.get('total_documents', 0)} documents")

    # Load model with stderr suppressed
    print("  Loading model...", end=" ", flush=True)
    t0 = time.time()

    old_stderr = sys.stderr
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

    # Load fine-tuned adapter
    adapters = sorted(glob.glob("models/fine_tuned_*"))
    if adapters:
        model = PeftModel.from_pretrained(model, adapters[-1])

    sys.stderr = old_stderr
    print(f"done ({time.time() - t0:.1f}s)")
    print()
    print("  Ask any AI/ML research question.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    return model, tokenizer, vector_store


def _generate(model, tokenizer, system, user, max_tokens=120):
    """Core generation function."""
    prompt = (
        SYS + "\n" + system + END + "\n"
        + USR + "\n" + user + END + "\n"
        + AST + "\n"
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=1024
    ).to(model.device)

    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    with torch.no_grad():
        outputs = model.generate(
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
    sys.stderr = old_stderr

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Trim at last complete sentence
    if answer and answer[-1] not in ".!?":
        last_end = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
        if last_end > 20:
            answer = answer[:last_end + 1]
    return answer


def step1_domain_answer(model, tokenizer, question):
    """STEP 1: Fine-tuned model generates domain answer."""
    system = (
        "You are an AI research expert. Answer questions about AI, "
        "machine learning, deep learning, NLP, transformers, and LLMs. "
        "If the question is not about AI/ML, say: I only answer AI/ML questions. "
        "Give a technical, informative answer."
    )
    return _generate(model, tokenizer, system, question, max_tokens=120)


def step2_polish(model, tokenizer, question, draft):
    """STEP 2: Base model (adapter OFF) reformats with proper grammar."""
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()

    system = (
        "You are a professional scientific writer. "
        "Rewrite the following draft answer with perfect grammar, "
        "clear structure, and proper punctuation. "
        "Keep the same meaning and technical content. "
        "Write 3-4 polished sentences. Do not add new information."
    )
    user = f"Question: {question}\n\nDraft answer to rewrite:\n{draft}"
    result = _generate(model, tokenizer, system, user, max_tokens=150)

    if hasattr(model, 'enable_adapter_layers'):
        model.enable_adapter_layers()

    return result


def search_papers(vector_store, question):
    """Find related papers (shown as references, NOT fed to model)."""
    results = vector_store.search(question, top_k=3)
    papers = []
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
        papers.append({
            "id": len(papers) + 1,
            "title": title,
            "authors": meta.get("authors", "Unknown"),
            "score": score,
        })
    return papers


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

        # Start thinking animation
        stop = threading.Event()
        anim = threading.Thread(target=thinking_animation, args=(stop,), daemon=True)
        anim.start()

        t0 = time.time()

        # STEP 1: Fine-tuned model generates domain answer
        draft = step1_domain_answer(model, tokenizer, question)

        # STEP 2: Base model (adapter OFF) polishes with proper grammar
        answer = step2_polish(model, tokenizer, question, draft)

        # Find related papers (shown as references only)
        papers = search_papers(vector_store, question)

        # Stop animation
        stop.set()
        anim.join()
        latency = time.time() - t0

        # Display
        print(f"  ScholarMind ({latency:.1f}s):\n")
        print(wrap_text(answer))

        if papers:
            print()
            print("  Related Research:")
            for p in papers:
                print(f"    [{p['id']}] {p['title']}")
                print(f"        {p['authors']}")

        print()
        print("  " + "-" * 56)


if __name__ == "__main__":
    main()
