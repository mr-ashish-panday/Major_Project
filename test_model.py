
import os
import glob
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import CONFIG
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def get_latest_model_path():
    model_dir = CONFIG['model_dir']
    if not os.path.exists(model_dir):
         return None
         
    # Check for fine_tuned_* directories
    dirs = glob.glob(os.path.join(model_dir, "fine_tuned_*"))
    # Also check for checkpoints
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    
    all_dirs = dirs + checkpoints
    if not all_dirs:
        return None
    
    # Sort by modification time
    latest_dir = max(all_dirs, key=os.path.getmtime)
    return latest_dir

def main():
    load_dotenv()
    
    print("\n" + "="*50)
    print("🤖 ScholarMind Model Tester")
    print("="*50)
    
    # 1. Select Model
    latest_path = get_latest_model_path()
    
    if not latest_path:
        print("❌ No fine-tuned models or checkpoints found in ./models")
        return
        
    print(f"\n📂 Found latest model: {latest_path}")
    choice = input("Press Enter to load this model, or type 'list' to see all: ")
    
    model_path = latest_path
    
    if choice.strip().lower() == 'list':
        all_dirs = glob.glob(os.path.join(CONFIG['model_dir'], "*"))
        all_dirs = [d for d in all_dirs if os.path.isdir(d) and ('fine_tuned' in d or 'checkpoint' in d)]
        all_dirs.sort(key=os.path.getmtime, reverse=True)
        
        print("\nAvailable Models:")
        for i, d in enumerate(all_dirs):
            print(f"{i+1}. {os.path.basename(d)}")
            
        idx = input("\nSelect model number: ")
        try:
            model_path = all_dirs[int(idx)-1]
        except:
            print("Invalid selection, using latest.")
            model_path = latest_path

    print(f"\n⏳ Loading model from: {model_path}...")
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model'], trust_remote_code=True)
    
    # 3. Load Base Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG['base_model'],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 4. Load Adapter
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    print("✅ Model loaded successfully!")
    print("\n" + "-"*50)
    print("Type your prompt below. Type 'exit' or 'quit' to stop.")
    print("-"*50)
    
    while True:
        prompt = input("\n👤 User: ")
        if prompt.lower() in ['exit', 'quit']:
            break
            
        full_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        print("\n🤖 Assistant: ", end="", flush=True)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the assistant part
        try:
            response = response.split("<|assistant|>")[-1].strip()
        except:
            pass
            
        print(response)

if __name__ == "__main__":
    main()
