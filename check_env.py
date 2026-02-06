import sys
import os

def check_environment():
    print("="*50)
    print("ENVIRONMENT CHECK")
    print("="*50)
    
    print(f"Python Version: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        cuda_avail = torch.cuda.is_available()
        print(f"✅ CUDA Available: {cuda_avail}")
        
        if cuda_avail:
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("❌ WARNING: CUDA not available! Training will be extremely slow.")
            
    except ImportError as e:
        print(f"❌ PyTorch Import Error: {e}")

    try:
        import triton
        print(f"✅ Triton: {triton.__version__}")
    except ImportError as e:
        print(f"❌ Triton Import Error: {e}")
        print("   -> 'triton' is required for optimizers. Install with: pip install triton")

    try:
        import transformers
        print(f"✅ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers Import Error: {e}")

    try:
        import sentence_transformers
        print(f"✅ SentenceTransformers: {sentence_transformers.__version__}")
    except ImportError as e:
        print(f"❌ SentenceTransformers Import Error: {e}")
        
    try:
        import bitsandbytes
        print(f"✅ BitsAndBytes: {bitsandbytes.__version__}")
    except ImportError as e:
        print(f"❌ BitsAndBytes Import Error: {e}")

    print("-" * 50)
    print("Testing Agent Imports...")
    try:
        from agents.validator import ValidatorAgent
        print("✅ agents.validator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import agents.validator: {e}")
    except Exception as e:
        print(f"❌ Crash during import: {e}")

    print("="*50)
    print("Check complete.")

if __name__ == "__main__":
    check_environment()
