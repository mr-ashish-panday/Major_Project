CONFIG = {
    # Model Configuration
    'base_model': 'microsoft/Phi-3-mini-4k-instruct',  # 3.8B params, fits in 12GB with 4-bit
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'quantization': '4bit',
    
    # Paths
    'data_dir': './data',
    'model_dir': './models',
    'logs_dir': './logs',
    'vector_db_path': './vector_store',
    
    # Data Extraction - MAXIMUM COVERAGE
    'arxiv_keywords': [
        'large language models',
        'LLM fine-tuning',
        'transformer architecture',
        'natural language processing',
        'parameter efficient fine-tuning',
        'LoRA',
        'QLoRA',
        'instruction tuning',
        'retrieval augmented generation',
        'attention mechanism',
        'BERT',
        'GPT',
        'prompt engineering',
        'chain of thought',
        'in-context learning',
        'language model pretraining',
        'neural machine translation',
        'text generation',
        'question answering',
        'knowledge distillation'
    ],
    'min_papers_threshold': 5,
    'arxiv_max_results': 150,      # Fetch 300 papers per run
    'arxiv_start_offset': 3800,     # Skip first 200 (already seen), get next 300
    
    # ===========================================
    # TRAINING CONFIG - MAXIMIZED FOR 12GB VRAM
    # ===========================================
    'gpu_memory_limit': '12GB',
    'training_epochs': 3,  # Start with 3 epochs to test
    'batch_size': 1,  # Reduced to fit more in memory
    'learning_rate': 1e-4,  # Lower LR for more epochs = finer tuning
    'gradient_accumulation_steps': 16,  # Effective batch = 16
    'warmup_ratio': 0.05,  # Shorter warmup with more epochs
    
    # LoRA Configuration - MAXIMUM CAPACITY for 12GB
    'lora_r': 32,  # Reduced for 12GB VRAM
    'lora_alpha': 64,  # 2x rank is best practice
    'lora_dropout': 0.1,  # Slightly higher for regularization
    'lora_target_modules': [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',  # Attention
        'gate_proj', 'up_proj', 'down_proj',  # MLP
        'lm_head'  # Output layer too!
    ],
    
    # Inference Configuration
    'max_context_length': 512,  # Reduced for training (OOM fix)
    'max_new_tokens': 1024,  # Longer responses
    'retrieval_top_k': 7,  # More context
    'temperature': 0.7,
    'top_p': 0.9,
    
    # Evaluation
    'perplexity_threshold': 0.85,
    'eval_sample_size': 100,  # More thorough evaluation
    
    # Performance Targets (for self-improvement)
    'min_bleu_target': 0.05,         # Minimum acceptable BLEU
    'min_rouge_target': 0.15,        # Minimum acceptable ROUGE-L
    'perplexity_improvement_target': 0.1,  # 10% improvement target
}