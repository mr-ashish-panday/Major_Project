import os
import logging
from datetime import datetime
from config import CONFIG
from agents.orchestrator import OrchestratorAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment variables
HUGGINGFACE_TOKEN = "REMOVED_FOR_SECURITY"

def run_pipeline():
    orchestrator = OrchestratorAgent(CONFIG)
    orchestrator.orchestrate()

if __name__ == '__main__':
    run_pipeline()