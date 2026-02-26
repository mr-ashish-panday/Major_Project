"""Quick terminal test for Phi-3 model output."""
import os
import glob
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("  Phi-3 Terminal Test")
print("=" * 60)

# Load model
print("\nLoading Phi-3 (4-bit quantized)...")
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

# Load fine-tuned LoRA adapter
adapters = sorted(glob.glob("models/fine_tuned_*"))
if adapters:
    adapter_path = adapters[-1]
    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
else:
    print("No LoRA adapter found, using base Phi-3")

tok = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

print(f"Model loaded in {time.time() - t0:.1f}s\n")

# Phi-3 special tokens
SYS = "<" + "|system|" + ">"
END = "<" + "|end|" + ">"
USR = "<" + "|user|" + ">"
AST = "<" + "|assistant|" + ">"

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

    prompt = (
        SYS + "\n"
        "You are a helpful AI research assistant specializing in NLP and LLMs. "
        "Give clear, concise answers." + END + "\n"
        + USR + "\n"
        + q + END + "\n"
        + AST + "\n"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.15,
            pad_token_id=tok.pad_token_id,
        )
    latency = time.time() - t1

    # Extract only generated tokens
    generated = out[0][inputs["input_ids"].shape[1]:]
    answer = tok.decode(generated, skip_special_tokens=True).strip()

    print(f"\n  Answer ({latency:.1f}s):\n")
    print(f"  {answer}\n")

print("=" * 60)
print("  Done!")
print("=" * 60)
