"""
ScholarMind Terminal Chatbot
Clean research assistant powered by fine-tuned Phi-3 + 20,818 papers.
"""
# Suppress ALL warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging
for name in ["transformers", "accelerate", "bitsandbytes", "peft", "torch",
             "transformers.generation", "transformers.modeling_utils",
             "transformers.tokenization_utils_base"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

import sys
import glob
import time
import threading
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
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
        time.sleep(0.5)
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

    # Phi-3
    print("  Loading model...", end=" ", flush=True)
    t0 = time.time()

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
        adapter_path = adapters[-1]
        print(f"\n  Loading fine-tuned adapter ({os.path.basename(adapter_path)})...", end=" ", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)

    print("done")
    print(f"  Ready in {time.time() - t0:.1f}s")
    print()
    print("  Ask any AI/ML research question.")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    return model, tokenizer, vector_store


def generate_answer(model, tokenizer, question, paper_context=""):
    """Generate a clean answer using retrieved paper context."""
    if paper_context:
        system = (
            "You are ScholarMind, an expert AI research assistant. "
            "Answer the question using the provided research paper excerpts. "
            "Write a clear, well-structured answer with proper grammar. "
            "Cite sources as [1], [2] when using information from papers. "
            "If the question is not about AI/ML, politely decline. "
            "Keep your answer to 3-4 sentences maximum."
        )
        user = (
            f"Research papers:\n{paper_context}\n\n"
            f"Question: {question}\n\n"
            f"Answer clearly and cite sources:"
        )
    else:
        system = (
            "You are ScholarMind, an expert AI research assistant. "
            "Give a clear, well-structured answer about AI/ML topics. "
            "If the question is not about AI/ML, politely decline. "
            "Keep your answer to 3-4 sentences maximum."
        )
        user = question

    prompt = (
        SYS + "\n" + system + END + "\n"
        + USR + "\n" + user + END + "\n"
        + AST + "\n"
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            top_p=0.8,
            top_k=30,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Trim at last complete sentence
    if answer and answer[-1] not in ".!?":
        last_end = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
        if last_end > 20:
            answer = answer[:last_end + 1]

    return answer


def search_papers(vector_store, question):
    """Search knowledge base for relevant papers."""
    results = vector_store.search(question, top_k=3)
    papers = []
    context_parts = []
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
        context_parts.append(f"[{idx}] {title}: {content}")

    return papers, "\n".join(context_parts)


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
        anim = threading.Thread(target=thinking_animation, args=(stop,))
        anim.start()

        t0 = time.time()

        # Search papers
        papers, paper_context = search_papers(vector_store, question)

        # Generate answer
        answer = generate_answer(model, tokenizer, question, paper_context)

        # Stop animation
        stop.set()
        anim.join()

        latency = time.time() - t0

        # Display answer
        print(f"  ScholarMind ({latency:.1f}s):\n")
        print(wrap_text(answer))

        # Show sources
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
