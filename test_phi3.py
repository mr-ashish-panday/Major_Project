"""Quick terminal test for Phi-3 model with RAG citations."""
import os
import glob
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIG
from agents.vector_store import VectorStoreAgent

print("=" * 60)
print("  Phi-3 Terminal Test (with RAG Citations)")
print("=" * 60)

# Load vector store
print("\nLoading knowledge base...")
vector_store = VectorStoreAgent(CONFIG)
stats = vector_store.get_stats()
print(f"Knowledge base: {stats.get('total_documents', 0)} documents loaded")

# Load model
print("\nLoading Phi-3 (4-bit quantized)...")
t0 = time.time()

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)

# Load fine-tuned LoRA adapter
adapters = sorted(glob.glob("models/fine_tuned_*"))
if adapters:
    adapter_path = adapters[-1]
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
else:
    print("No LoRA adapter found, using base Phi-3")
    model = base_model

tok = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Model loaded in {time.time() - t0:.1f}s\n")

# Phi-3 chat template tokens
SYS = "<" + "|system|" + ">"
END = "<" + "|end|" + ">"
USR = "<" + "|user|" + ">"
AST = "<" + "|assistant|" + ">"


def generate_answer(mdl, prompt, max_tokens=200):
    """Generate with tight parameters to avoid rambling."""
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(mdl.device)
    t1 = time.time()
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,          # Low temp = more focused
            top_p=0.85,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.3,   # Strong penalty to stop loops
            no_repeat_ngram_size=4,   # Block repeating 4-grams
            pad_token_id=tok.pad_token_id,
        )
    latency = time.time() - t1
    generated = out[0][inputs["input_ids"].shape[1]:]
    answer = tok.decode(generated, skip_special_tokens=True).strip()
    return answer, latency


def build_rag_prompt(question, context):
    """Build a clean RAG prompt."""
    return (
        SYS + "\n"
        "You are ScholarMind, an expert AI research assistant. "
        "Answer the question using ONLY the provided research context. "
        "Be concise and direct. Cite sources as [1], [2], [3]." + END + "\n"
        + USR + "\n"
        "Research context:\n" + context + "\n\n"
        "Question: " + question + END + "\n"
        + AST + "\n"
    )


# Test questions
questions = [
    "What is LoRA and how does it work?",
    "Explain the attention mechanism in transformers.",
    "What is knowledge distillation and why is it useful?",
]

for i, q in enumerate(questions, 1):
    print("=" * 60)
    print(f"  Question {i}: {q}")
    print("=" * 60)

    # RAG: Search knowledge base
    retrieved = vector_store.search(q, top_k=3)

    context_parts = []
    citations = []
    for j, doc in enumerate(retrieved, 1):
        content = doc.get("content", "")[:400]  # Shorter chunks = cleaner
        # Clean up the content
        content = content.replace("\n", " ").strip()
        meta = doc.get("metadata", {})
        title = meta.get("title", "Unknown")
        authors = meta.get("authors", "Unknown")
        score = doc.get("score", 0)
        context_parts.append(f"[{j}] From '{title}': {content}")
        citations.append({"id": j, "title": title, "authors": authors, "score": score})

    context = "\n\n".join(context_parts)
    prompt = build_rag_prompt(q, context)

    # Generate with fine-tuned adapter
    answer, latency = generate_answer(model, prompt)

    print(f"\n  Answer ({latency:.1f}s):\n")
    # Word wrap for terminal
    words = answer.split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 80:
            print(line)
            line = "  " + w
        else:
            line += " " + w if line.strip() else "  " + w
    if line.strip():
        print(line)

    # Print citations
    print(f"\n  Sources:")
    for c in citations:
        print(f"    [{c['id']}] {c['title']}")
        print(f"        by {c['authors']} | Relevance: {c['score']:.0%}")
    print()

print("=" * 60)
print("  Done!")
print("=" * 60)
