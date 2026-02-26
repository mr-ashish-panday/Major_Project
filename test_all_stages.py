"""Test ALL 3 model stages side by side."""
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

print("=" * 60)
print("  All 3 Stages: Base vs Fine-tuned vs ScholarFormer")
print("=" * 60)

# Phi-3 tokens
SYS = "<" + "|system|" + ">"
END = "<" + "|end|" + ">"
USR = "<" + "|user|" + ">"
AST = "<" + "|assistant|" + ">"

questions = [
    "What is LoRA and how does it work?",
    "Explain the attention mechanism in transformers.",
    "What is knowledge distillation?",
]


def wrap_print(text, indent=6):
    """Word-wrap for terminal."""
    words = text.split()
    line = " " * indent
    for w in words:
        if len(line) + len(w) + 1 > 78:
            print(line)
            line = " " * indent + w
        else:
            line += (" " + w) if line.strip() else (" " * indent + w)
    if line.strip():
        print(line)


# ============================================================
# STAGE 1: Base Phi-3
# ============================================================
print("\n[1/3] Loading BASE Phi-3 (4-bit)...")
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

tok_phi = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
if tok_phi.pad_token is None:
    tok_phi.pad_token = tok_phi.eos_token

# Load fine-tuned adapter
ft_model = None
adapters = sorted(glob.glob("models/fine_tuned_*"))
if adapters:
    print(f"[2/3] Loading FINE-TUNED adapter: {adapters[-1]}")
    ft_model = PeftModel.from_pretrained(base_model, adapters[-1])
else:
    print("[2/3] No adapter found, skipping fine-tuned test")

print(f"Phi-3 loaded in {time.time() - t0:.1f}s")


def gen_phi3(mdl, question, max_tokens=200):
    prompt = (
        SYS + "\nYou are a helpful AI research assistant. "
        "Give a clear, concise answer." + END + "\n"
        + USR + "\n" + question + END + "\n"
        + AST + "\n"
    )
    inputs = tok_phi(prompt, return_tensors="pt", truncation=True, max_length=2048).to(mdl.device)
    t1 = time.time()
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            top_p=0.85,
            top_k=40,
            do_sample=True,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=tok_phi.pad_token_id,
        )
    latency = time.time() - t1
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tok_phi.decode(generated, skip_special_tokens=True).strip(), latency


# ============================================================
# STAGE 3: ScholarFormer
# ============================================================
sf_model = None
sf_tok = None
sf_checkpoint = None

# Find ScholarFormer checkpoint
for path in ["./models/scholarformer/checkpoints/best",
             "./models/scholarformer/checkpoints/latest"]:
    if os.path.exists(path):
        sf_checkpoint = path
        break

if sf_checkpoint:
    print(f"[3/3] Loading SCHOLARFORMER from {sf_checkpoint}...")
    t0 = time.time()
    try:
        from scholarformer.model import ScholarFormer
        from scholarformer.tokenizer import ScholarFormerTokenizer

        sf_tok = ScholarFormerTokenizer()

        # Load config
        import json
        config_path = os.path.join(sf_checkpoint, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                sf_config = json.load(f)
        else:
            sf_config = {"vocab_size": 32018, "d_model": 512, "n_heads": 8,
                         "n_layers": 6, "d_ff": 2048, "max_seq_len": 512,
                         "dropout": 0.1, "num_sections": 7}

        sf_model = ScholarFormer(
            vocab_size=sf_config.get("vocab_size", 32018),
            d_model=sf_config.get("d_model", 512),
            n_heads=sf_config.get("n_heads", 8),
            n_layers=sf_config.get("n_layers", 6),
            d_ff=sf_config.get("d_ff", 2048),
            max_seq_len=sf_config.get("max_seq_len", 512),
            dropout=sf_config.get("dropout", 0.1),
            num_sections=sf_config.get("num_sections", 7),
        )

        # Load weights
        weights_path = os.path.join(sf_checkpoint, "model.pt")
        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location="cpu")
            if "model_state_dict" in state:
                sf_model.load_state_dict(state["model_state_dict"], strict=False)
            else:
                sf_model.load_state_dict(state, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sf_model = sf_model.to(device).eval()
        print(f"ScholarFormer loaded in {time.time() - t0:.1f}s")

    except Exception as e:
        print(f"Failed to load ScholarFormer: {e}")
        sf_model = None
else:
    print("[3/3] No ScholarFormer checkpoint found, skipping")


def gen_scholarformer(question, max_tokens=200):
    prompt = f"Question: {question}\nAnswer:"
    inputs = sf_tok.tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=256).to(sf_model.device if hasattr(sf_model, 'device') else 'cpu')
    input_ids = inputs["input_ids"]

    t1 = time.time()
    with torch.no_grad():
        generated = input_ids.clone()
        for _ in range(max_tokens):
            outputs = sf_model(generated)
            next_logits = outputs[:, -1, :] / 0.5
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == sf_tok.tokenizer.eos_token_id:
                break
    latency = time.time() - t1

    new_tokens = generated[0][input_ids.shape[1]:]
    return sf_tok.tokenizer.decode(new_tokens, skip_special_tokens=True).strip(), latency


# ============================================================
# RUN ALL TESTS
# ============================================================
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)

for i, q in enumerate(questions, 1):
    print(f"\n{'='*60}")
    print(f"  Question {i}: {q}")
    print(f"{'='*60}")

    # Stage 1: Base Phi-3
    if ft_model is not None:
        ft_model.disable_adapter_layers()
        ans, lat = gen_phi3(ft_model, q)
        ft_model.enable_adapter_layers()
    else:
        ans, lat = gen_phi3(base_model, q)
    print(f"\n  [STAGE 1] BASE Phi-3 ({lat:.1f}s):")
    wrap_print(ans)

    # Stage 2: Fine-tuned Phi-3
    if ft_model is not None:
        ans, lat = gen_phi3(ft_model, q)
        print(f"\n  [STAGE 2] FINE-TUNED Phi-3 ({lat:.1f}s):")
        wrap_print(ans)

    # Stage 3: ScholarFormer
    if sf_model is not None:
        try:
            ans, lat = gen_scholarformer(q)
            print(f"\n  [STAGE 3] SCHOLARFORMER ({lat:.1f}s):")
            wrap_print(ans)
        except Exception as e:
            print(f"\n  [STAGE 3] SCHOLARFORMER: Error - {e}")

print(f"\n{'='*60}")
print("  Done! Compare and decide what to keep.")
print("=" * 60)
