"""
ScholarMind Terminal Chatbot
Pipeline:
  Step 1: Trained model generates domain answer (has latest research knowledge)
  Step 2: RAG retrieves relevant paper evidence
  Step 3: Model checks the draft, corrects errors, combines with paper knowledge
  Step 4: Model does final verification pass on the combined answer
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


def _gen(model, tok, system, user, max_tokens=150):
    """Deterministic generation - no sampling, consistent output."""
    prompt = (SYS + "\n" + system + END + "\n"
              + USR + "\n" + user + END + "\n"
              + AST + "\n")
    inputs = tok(prompt, return_tensors="pt", truncation=True,
                 max_length=2048).to(model.device)

    old = sys.stderr
    sys.stderr = io.StringIO()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,           # DETERMINISTIC - no randomness
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tok.pad_token_id,
        )
    sys.stderr = old

    gen = out[0][inputs["input_ids"].shape[1]:]
    ans = tok.decode(gen, skip_special_tokens=True).strip()
    return _clean(ans)


def _clean(text):
    """Strip meta-commentary and artifacts."""
    if not text:
        return text
    # Remove code fences
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = text.replace('```', '')
    # Remove proofreading meta-commentary
    for p in [
        r"Here'?t?\s*be\s+any\s+spelled.*?(?:here you go\s*:?\s*)",
        r"(?:I will|Let me|I have)\s+(?:proofread|check|review|correct).*?(?::\s*|\.\.\.+\s*)",
        r"(?:Here (?:is|are)|Below is).*?(?:corrected|revised|polished|fixed).*?(?::\s*|\.\.\.+\s*)",
        r"Let me know if.*$", r"Thankyou\s*!*\s*$", r"Thank you\s*!*\s*$",
        r"Please note that.*$", r"\[Text remains unchanged\]",
        r"^.*?here you go\s*:?\s*", r"I hope (?:this|that).*$",
    ]:
        text = re.sub(p, '', text, flags=re.IGNORECASE | re.DOTALL)
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


# ============================================================
# STEP 1: Trained model generates domain draft
# ============================================================
def step1_domain_draft(model, tok, question):
    return _gen(model, tok,
        "You are an AI research expert. Answer concisely about AI, ML, "
        "deep learning, NLP, transformers, LLMs. If not AI/ML topic, "
        "say: I only answer AI/ML research questions.",
        question, max_tokens=100)


# ============================================================
# STEP 2: RAG search
# ============================================================
def step2_papers(vs, question):
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


# ============================================================
# STEP 3: Check draft, correct errors, combine with papers
# ============================================================
def step3_correct_and_combine(model, tok, question, draft, evidence, has_papers):
    if has_papers:
        sys_msg = (
            "You received a draft answer and research paper excerpts. "
            "Your job: check the draft for factual errors, fix any wrong terms, "
            "and write a correct, combined answer. Use paper information if the "
            "question asks about latest findings or developments. "
            "For simple definitions, just give the correct answer. "
            "Write 3-4 clean sentences. No meta-commentary."
        )
        user = (
            f"Question: {question}\n\n"
            f"Draft (may have errors - fix them):\n{draft}\n\n"
            f"Papers:\n{evidence}\n\n"
            f"Corrected answer:"
        )
    else:
        sys_msg = (
            "Check this draft for errors and rewrite correctly. "
            "Write 3-4 clean sentences. No meta-commentary."
        )
        user = f"Question: {question}\n\nDraft:\n{draft}\n\nCorrected answer:"
    return _gen(model, tok, sys_msg, user, max_tokens=150)


# ============================================================
# STEP 4: Final verification - ensure quality
# ============================================================
def step4_verify(model, tok, question, answer):
    sys_msg = (
        "Verify this answer is factually correct and grammatically clean. "
        "Fix any remaining errors. Output ONLY the final answer text. "
        "No commentary, no notes, no 'here is the corrected version'. "
        "Just the answer itself."
    )
    return _gen(model, tok, sys_msg,
        f"Question: {question}\n\nAnswer: {answer}\n\nVerified answer:",
        max_tokens=150)


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

        # STEP 1: Domain draft from trained model
        draft = step1_domain_draft(model, tok, question)

        # STEP 2: Find papers
        papers, evidence = step2_papers(vs, question)

        # STEP 3: Correct errors + combine with paper knowledge
        combined = step3_correct_and_combine(
            model, tok, question, draft, evidence, len(papers) > 0)

        # STEP 4: Final verification pass
        final = step4_verify(model, tok, question, combined)

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
