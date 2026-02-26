"""
ScholarMind Terminal Chatbot - Three-Pass Pipeline
Step 1: Fine-tuned model generates domain answer
Step 2: RAG retrieves latest paper evidence
Step 3: Base model combines both into polished final answer
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
             "transformers.generation", "transformers.modeling_utils"]:
    logging.getLogger(name).setLevel(logging.CRITICAL)

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
    print("  Powered by Fine-tuned Phi-3 + 20,818 Research Papers")
    print("=" * 60)

    # Vector store
    print("\n  Loading knowledge base...", end=" ", flush=True)
    vector_store = VectorStoreAgent(CONFIG)
    stats = vector_store.get_stats()
    print(f"{stats.get('total_documents', 0)} documents")

    # Phi-3 base
    print("  Loading Phi-3 (4-bit)...", end=" ", flush=True)
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
        attn_implementation="eager",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("done")

    # Load fine-tuned adapter
    model = base_model
    adapters = sorted(glob.glob("models/fine_tuned_*"))
    if adapters:
        adapter_path = adapters[-1]
        print(f"  Loading LoRA adapter: {os.path.basename(adapter_path)}...", end=" ", flush=True)
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print("done")

    print(f"  Ready in {time.time() - t0:.1f}s")
    print()
    print("  Ask any AI/ML research question. Type 'quit' to exit.")
    print("=" * 60)

    return model, tokenizer, vector_store


def generate(model, tokenizer, system_msg, user_msg, max_tokens=120):
    """Generate response from model."""
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


def step1_domain_answer(model, tokenizer, question):
    """STEP 1: Fine-tuned model generates domain-specific answer."""
    system = (
        "You are an AI research expert. Answer the question about AI, ML, "
        "deep learning, NLP, or LLMs. Be specific and technical. "
        "If the question is not about AI/ML, say: 'I only answer AI/ML research questions.'"
    )
    return generate(model, tokenizer, system, question, max_tokens=100)


def step2_rag_search(vector_store, question):
    """STEP 2: Search knowledge base for latest paper evidence."""
    results = vector_store.search(question, top_k=3)
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
        snippets.append(f"[{idx}] {title}: {content}")

    return papers, "\n".join(snippets)


def step3_combine_and_polish(model, tokenizer, question, domain_answer, paper_evidence):
    """STEP 3: Base model combines domain answer + paper evidence into polished output."""
    # Disable adapter to use base model (better grammar/instruction following)
    if hasattr(model, 'disable_adapter_layers'):
        model.disable_adapter_layers()

    system = (
        "You are a professional research writer. You will be given a draft answer "
        "and evidence from research papers. Combine them into ONE clear, well-written "
        "paragraph with perfect grammar. Cite papers as [1], [2], [3]. "
        "Do NOT add information beyond what is provided. "
        "Write exactly one polished paragraph, no bullet points."
    )

    user = (
        f"Question: {question}\n\n"
        f"Draft answer: {domain_answer}\n\n"
        f"Research evidence:\n{paper_evidence}\n\n"
        f"Write one clear, polished paragraph combining the above:"
    )

    result = generate(model, tokenizer, system, user, max_tokens=150)

    # Re-enable adapter
    if hasattr(model, 'enable_adapter_layers'):
        model.enable_adapter_layers()

    return result


def wrap_text(text, width=70, indent=4):
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

        t_start = time.time()

        # === STEP 1: Fine-tuned domain answer ===
        print("\n  [1/3] Generating domain answer...", end="\r", flush=True)
        domain_answer = step1_domain_answer(model, tokenizer, question)

        # === STEP 2: RAG search ===
        print("  [2/3] Searching 20,818 papers...   ", end="\r", flush=True)
        papers, paper_evidence = step2_rag_search(vector_store, question)

        # === STEP 3: Combine & polish ===
        if papers and paper_evidence:
            print("  [3/3] Combining & polishing...     ", end="\r", flush=True)
            final_answer = step3_combine_and_polish(
                model, tokenizer, question, domain_answer, paper_evidence
            )
        else:
            final_answer = domain_answer

        total_time = time.time() - t_start

        # Display final answer
        print(f"  ScholarMind ({total_time:.1f}s):              ")
        print()
        print(wrap_text(final_answer))

        # Show sources
        if papers:
            print()
            print("  Sources:")
            for p in papers:
                print(f"    [{p['id']}] {p['title']}")
                print(f"        by {p['authors']} ({p['score']:.0%} match)")

        print()
        print("  " + "-" * 56)


if __name__ == "__main__":
    main()
