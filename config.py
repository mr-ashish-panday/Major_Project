CONFIG = {
    'base_model': 'gpt2',
    'data_dir': './data',
    'model_dir': './models',
    'logs_dir': './logs',
    'arxiv_keywords': ['large language models', 'LLM fine-tuning'],
    'min_papers_threshold': 50,  # Reduced for faster testing
    'gpu_memory_limit': '12GB',
    'training_epochs': 3,  # Increased for better convergence on GPU
    'batch_size': 8,  # Increased for GPU parallelism; adjust down if OOM error occurs
    'learning_rate': 1e-4,
    'perplexity_threshold': 0.85,
}