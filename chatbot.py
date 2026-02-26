"""
ScholarMind Terminal Chatbot
Two-pass approach: Model answer + Model-formatted research evidence.
"""
# Suppress all noisy warnings before imports
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)

import sys
import glob
import time
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


def load_system():
    """Load all components."""
    print()
    print("=" * 60)
    print("  ScholarMind - AI Research Assistant")
    print("  Powered by Phi-3 + 20,818 Research Papers")
    print("=" * 60)

    # Vector store
    print("\n  Loading knowledge base...", end=" ", flush=True)
    vector_store = VectorStoreAgent(CONFIG)
    stats = vector_store.get_stats()
    doc_count = stats.get("total_documents", 0)
    print(f"{doc_count} documents")

    # Phi-3
    print("  Loading Phi-3 (4-bit quantized)...", end=" ", flush=True)
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
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("done")

    # Load fine-tuned LoRA adapter
    adapters = sorted(glob.glob("models/fine_tuned_*"))
    if adapters:
        adapter_path = adapters[-1]
        print(f"  Loading LoRA adapter: {os.path.basename(adapter_path)}...", end=" ", flush=True)
        model = PeftModel.from_pretrained(model, adapter_path)
        print("done")

    print(f"  Ready in {time.time() - t0:.1f}s")
    print()
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    return model, tokenizer, vector_store


def generate(model, tokenizer, system_msg, user_msg, max_tokens=120):
    """Generate response with given system/user messages."""
    prompt = (
        SYS + "\n" + system_msg + END + "\n"
        + USR + "\n" + user_msg + END + "\n"
        + AST + "\n"
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

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

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Trim at last complete sentence
    if answer and answer[-1] not in ".!?":
        last_end = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
        if last_end > 20:
            answer = answer[:last_end + 1]

    return answer


def pass1_conceptual_answer(model, tokenizer, question):
    """Pass 1: Get a clean conceptual answer."""
    system = (
        "You are ScholarMind, an AI research assistant that ONLY answers questions "
        "about artificial intelligence, machine learning, deep learning, NLP, "
        "transformers, LLMs, and related computer science research topics. "
        "If the question is unrelated to AI/ML, politely decline. "
        "Give a clear, concise explanation in 2-3 sentences maximum."
    )
    return generate(model, tokenizer, system, question, max_tokens=100)


def pass2_format_evidence(model, tokenizer, question, paper_snippets):
    """Pass 2: Format paper excerpts into clean research evidence."""
    system = (
        "You are a research paper summarizer. "
        "Given paper excerpts related to a question, write 2-3 clean bullet points "
        "summarizing what the research says. Start each point with the paper number like [1], [2]. "
        "Be concise - one sentence per point. Only state facts from the excerpts."
    )

    user = (
        f"Question: {question}\n\n"
        f"Paper excerpts:\n{paper_snippets}\n\n"
        "Summarize the key research findings in 2-3 bullet points:"
    )
    return generate(model, tokenizer, system, user, max_tokens=120)


def search_papers(vector_store, question, top_k=3):
    """Search knowledge base and return papers + snippets."""
    results = vector_store.search(question, top_k=top_k)
    papers = []
    snippets = []
    seen = set()

    for doc in results:
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        if title in seen:
            continue
        seen.add(title)
        score = doc.get("score", 0)
        if score < 0.35:
            continue

        idx = len(papers) + 1
        content = doc.get("content", "")[:300].replace("\n", " ").strip()
        papers.append({
            "id": idx,
            "title": title,
            "authors": meta.get("authors", "Unknown"),
            "score": score,
        })
        snippets.append(f"[{idx}] From '{title}': {content}")

    return papers, "\n\n".join(snippets)


def wrap_text(text, width=70, indent=4):
    """Word-wrap text for terminal display."""
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

        # === PASS 1: Conceptual Answer ===
        print()
        print("  Thinking...", end="\r", flush=True)
        t0 = time.time()
        answer = pass1_conceptual_answer(model, tokenizer, question)
        t1 = time.time()

        print(f"  ScholarMind ({t1 - t0:.1f}s):")
        print()
        print(wrap_text(answer))

        # === SEARCH: Find related papers ===
        papers, snippets = search_papers(vector_store, question)

        if papers and snippets:
            # === PASS 2: Format paper evidence ===
            print()
            print("  Checking research papers...", end="\r", flush=True)
            evidence = pass2_format_evidence(model, tokenizer, question, snippets)
            t2 = time.time()

            print(f"  Research Evidence ({t2 - t1:.1f}s):")
            print()
            print(wrap_text(evidence))

            # Show source papers
            print()
            print("  Sources:")
            for p in papers:
                print(f"    [{p['id']}] {p['title']}")
                print(f"        by {p['authors']} ({p['score']:.0%} match)")

        print()
        print("  " + "-" * 56)


if __name__ == "__main__":
    main()
