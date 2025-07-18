CONFIG = {
    'base_model': 'meta-llama/Llama-2-7b-hf',
    'data_dir': './data',
    'model_dir': './models',
    'logs_dir': './logs',
    'arxiv_keywords': ['large language models', 'LLM fine-tuning'],
    'min_papers_threshold': 50,
    'gpu_memory_limit': '12GB',
    'training_epochs': 3,
    'batch_size': 4,
    'learning_rate': 1e-4,
    'perplexity_threshold': 0.85,  # Relative improvement factor
}