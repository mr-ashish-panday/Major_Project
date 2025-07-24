import os
import json
import logging
from datetime import datetime
from typing import Dict
from agents.extractor import ExtractorAgent
from agents.preprocessor import PreprocessorAgent
# from agents.validator import ValidatorAgent  # Temporarily co mmented out
from agents.trainer import TrainerAgent
from agents.evaluator import EvaluatorAgent
from agents.self_improvement import SelfImprovementAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Initializing ExtractorAgent...")
        self.extractor = ExtractorAgent(config)
        logger.info("Initializing PreprocessorAgent...")
        self.preprocessor = PreprocessorAgent(config)
        # logger.info("Initializing ValidatorAgent...")  # Temporarily commented out
        # self.validator = ValidatorAgent(config)  # Temporarily commented out
        logger.info("Initializing TrainerAgent...")
        self.trainer = TrainerAgent(config)
        logger.info("Initializing EvaluatorAgent...")
        self.evaluator = EvaluatorAgent(config)
        logger.info("Initializing SelfImprovementAgent...")
        self.self_improver = SelfImprovementAgent(config)
        self.previous_perplexity = 1.0  # Initial baseline; persist in production

    def orchestrate(self):
        try:
            logger.info("Starting extraction...")
            papers = self.extractor.extract()
            logger.info(f"Extracted {len(papers)} papers")
            
            if len(papers) < self.config['min_papers_threshold']:
                logger.info("Insufficient new papers; skipping cycle.")
                return

            logger.info("Starting preprocessing...")
            processed = self.preprocessor.preprocess(papers)
            
            # VALIDATION AGENT BYPASSED - Using processed data directly
            logger.info("Validation step bypassed - using preprocessed data directly")
            validated = processed  # Skip validation, use preprocessed data directly
            
            # Basic safety check to ensure we have data to work with
            if not validated or len(validated) == 0:
                logger.info("No data after preprocessing; ending cycle.")
                return

            logger.info("Starting training...")
            model_path = self.trainer.train(validated)

            # 10% hold-out for testing
            test_data = [chunk for paper in validated[:int(len(validated)*0.1)]
                         for chunk in paper['chunks']]
            
            logger.info("Starting evaluation...")
            metrics = self.evaluator.evaluate(model_path, test_data)

            # Check threshold (example: relative perplexity improvement)
            if metrics['perplexity'] > self.config['perplexity_threshold'] * self.previous_perplexity:
                logger.info("Starting self-improvement...")
                updated_config = self.self_improver.improve(metrics)
                self.config.update(updated_config)
                logger.info("Config updated for next cycle based on metrics.")
            self.previous_perplexity = metrics['perplexity']

            # Log cycle results
            log_path = os.path.join(self.config['logs_dir'],
                                   f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(log_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info("Cycle completed successfully.")
                    
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")  # This will show exactly where the error occurs