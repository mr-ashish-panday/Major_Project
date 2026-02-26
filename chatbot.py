"""
ScholarMind Terminal Chatbot - Multi-Pass Refinement
Step 1: Fine-tuned model generates domain draft
Step 2: RAG retrieves papers
Step 3: Base model REWRITES the draft into a proper answer
Step 4: Base model RECHECKS and polishes for grammar/quality
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

SYS = "<" + "|system|" + ">"
END = "<" + "|end|" + ">"
USR = "<" + "|user|" + ">"
AST = "<" + "|assistant|" + ">"


def thinking_animation(stop_event):
    frames = ["  Thinking.  ", "  Thinking.. ", "  Thinking..."]
    i = 0
    while not stop_event.is_set():
        print(f"\r{frames[i % 3]}", end="", flush=True)
        i += 1
        time.sleep(0.4)
    print("\r" + " " * 20 + "\r", end="", flush=True)


def load_system():
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

    sys.stderr = old
    print(f"done ({time.time() - t0:.1f}s)")
    print()
    print("  Ask any AI/ML research question.")
    print("  Type 'quit' to exit.")
    print("=" * 60)
    return model, tokenizer, vector_store


def _gen(model, tokenizer, system, user, max_tokens=150):
    """Core generation."""
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
            temperature=0.15,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    sys.stderr = old

    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = tokenizer.decode(gen, skip_special_tokens=True).strip()
    if ans and ans[-1] not in ".!?":
        last = max(ans.rfind("."), ans.rfind("!"), ans.rfind("?"))
        if last > 20:
            ans = ans[:last + 1]
    return ans


def step1_finetuned(model, tokenizer, question):
    """STEP 1: Fine-tuned model (adapter ON) generates domain draft."""
    sys_msg = (
        "You are an AI research expert. Answer the question about "
        "AI, machine learning, deep learning, NLP, or LLMs. "
        "Be specific and technical. Keep it short."
    )
    return _gen(model, tokenizer, sys_msg, question, max_tokens=100)


def step2_papers(vector_store, question):
    """STEP 2: Search knowledge base."""
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
        if doc.get("score", 0) < 0.40:
            continue
        idx = len(papers) + 1
        content = doc.get("content", "")[:200].replace("\n", " ").strip()
        papers.append({"id": idx, "title": title,
                        "authors": meta.get("authors", "Unknown"),
                        "score": doc.get("score", 0)})
        evidence.append(f"[{idx}] {title}: {content}")
    return papers, "\n".join(evidence)


def step3_rewrite(model, tokenizer, question, draft, evidence, has_papers):
    """STEP 3: Base model (adapter OFF) answers INDEPENDENTLY."""
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()

    if has_papers:
        sys_msg = (
            "You are ScholarMind, an expert AI research assistant. "
            "Answer this question using YOUR OWN knowledge about AI and machine learning. "
            "You are also given a rough draft from another model and some paper titles - "
            "use these ONLY as topic hints. Do NOT copy or trust the draft's definitions - "
            "it may contain errors. Write YOUR OWN accurate answer. "
            "For simple questions (what is X): give a clear, correct definition. No citations needed. "
            "For research questions (latest findings): mention relevant papers with [1], [2] citations. "
            "Write 3-4 clear sentences with perfect grammar."
        )
        user = (
            f"Question: {question}\n\n"
            f"[Topic hint from domain model - may contain errors, verify before using]:\n{draft}\n\n"
            f"[Research papers for citation if needed]:\n{evidence}\n\n"
            f"Now write YOUR OWN accurate answer:"
        )
    else:
        sys_msg = (
            "You are ScholarMind, an expert AI research assistant. "
            "Answer this question using YOUR OWN knowledge about AI and ML. "
            "Write 3-4 clear sentences with perfect grammar."
        )
        user = f"Question: {question}\n\nWrite your answer:"

    result = _gen(model, tokenizer, sys_msg, user, max_tokens=180)

    if hasattr(model, 'enable_adapter_layers'):
        model.enable_adapter_layers()
    return result


def step4_polish(model, tokenizer, question, answer):
    """STEP 4: Base model (adapter OFF) RECHECKS and polishes."""
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()

    sys_msg = (
        "You are a grammar checker. Review the following answer for: "
        "- Spelling errors "
        "- Grammar mistakes "
        "- Incomplete words or sentences "
        "- Nonsensical phrases "
        "Rewrite it with corrections. Keep the same meaning and length. "
        "If the answer is already correct, return it as-is."
    )
    user = f"Question: {question}\n\nAnswer to check:\n{answer}\n\nCorrected version:"

    result = _gen(model, tokenizer, sys_msg, user, max_tokens=180)

    if hasattr(model, 'enable_adapter_layers'):
        model.enable_adapter_layers()
    return result


def wrap_text(text, width=68, indent=4):
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

        stop = threading.Event()
        anim = threading.Thread(target=thinking_animation, args=(stop,), daemon=True)
        anim.start()

        t0 = time.time()

        # STEP 1: Fine-tuned draft
        draft = step1_finetuned(model, tokenizer, question)

        # STEP 2: RAG papers
        papers, evidence = step2_papers(vector_store, question)

        # STEP 3: Base model rewrites completely
        rewritten = step3_rewrite(model, tokenizer, question, draft, evidence, len(papers) > 0)

        # STEP 4: Base model rechecks grammar
        final = step4_polish(model, tokenizer, question, rewritten)

        stop.set()
        anim.join()
        latency = time.time() - t0

        print(f"  ScholarMind ({latency:.1f}s):\n")
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
