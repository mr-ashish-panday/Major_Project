"""
ScholarMind Terminal Chatbot
Interactive research assistant powered by Phi-3 + FAISS knowledge base.
"""
import os
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

    print(f"done ({time.time() - t0:.1f}s)")
    print()
    print("  Type your question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    return model, tokenizer, vector_store


def generate_answer(model, tokenizer, question):
    """Generate a clean answer from Phi-3."""
    prompt = (
        SYS + "\n"
        "You are ScholarMind, an AI research assistant that ONLY answers questions "
        "about artificial intelligence, machine learning, deep learning, NLP, "
        "transformers, LLMs, and related computer science research topics. "
        "If the user asks about anything unrelated to AI/ML research, politely decline "
        "and say you can only help with AI and ML research questions. "
        "Keep answers concise, accurate, and under 100 words. "
        "End with a complete sentence." + END + "\n"
        + USR + "\n" + question + END + "\n"
        + AST + "\n"
    )

    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
            top_p=0.8,
            top_k=30,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - t0

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

    # Trim at last complete sentence to avoid cut-off gibberish
    if answer and answer[-1] not in ".!?":
        last_end = max(answer.rfind("."), answer.rfind("!"), answer.rfind("?"))
        if last_end > 20:  # Keep at least 20 chars
            answer = answer[:last_end + 1]

    return answer, latency


def find_related_papers(vector_store, question, top_k=3):
    """Find related papers from the knowledge base."""
    results = vector_store.search(question, top_k=top_k)
    papers = []
    seen_titles = set()
    for doc in results:
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        papers.append({
            "title": title,
            "authors": meta.get("authors", "Unknown"),
            "score": doc.get("score", 0),
        })
    return papers


def wrap_text(text, width=70, indent=2):
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

        # Generate answer
        print()
        print("  Thinking...", end="\r", flush=True)
        answer, latency = generate_answer(model, tokenizer, question)

        print(f"  ScholarMind ({latency:.1f}s):")
        print()
        print(wrap_text(answer))

        # Find related papers (only show if relevance > 35%)
        papers = find_related_papers(vector_store, question, top_k=3)
        relevant = [p for p in papers if p["score"] > 0.35]
        if relevant:
            print()
            print("  Related Papers:")
            for i, p in enumerate(relevant, 1):
                print(f"    [{i}] {p['title']}")
                print(f"        by {p['authors']} ({p['score']:.0%} match)")

        print()
        print("  " + "-" * 56)


if __name__ == "__main__":
    main()
