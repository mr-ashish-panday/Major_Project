"""
ScholarMind - Self-Evolving LLM Research Assistant
Main entry point for the training and improvement pipeline.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from config import CONFIG
from agents.orchestrator import OrchestratorAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def check_environment():
    """Verify required environment variables and dependencies."""
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        logger.warning("HF_TOKEN not set. Some models may not be accessible.")
        logger.info("Set your HuggingFace token: export HF_TOKEN=your_token")
    else:
        logger.info("HuggingFace token found.")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.warning("No GPU detected. Training will be slow on CPU.")
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")


def run_pipeline():
    """Run the full training and improvement pipeline."""
    logger.info("=" * 60)
    logger.info("🧠 ScholarMind - Self-Evolving LLM Research Assistant")
    logger.info("=" * 60)
    
    check_environment()
    
    # Ensure directories exist
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['logs_dir'], exist_ok=True)
    os.makedirs(CONFIG['vector_db_path'], exist_ok=True)
    
    orchestrator = OrchestratorAgent(CONFIG)
    orchestrator.orchestrate()
    
    logger.info("Pipeline completed successfully!")


if __name__ == '__main__':
    run_pipeline()