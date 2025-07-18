import json
import logging
from typing import Dict
from agents.extractor import ExtractorAgent
from agents.preprocessor import PreprocessorAgent
from agents.validator import ValidatorAgent
from agents.trainer import TrainerAgent
from agents.evaluator import EvaluatorAgent
from agents.self_improvement import SelfImprovementAgent

logger = logging.getLogger(__name__)

class OrchestratorAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.extractor = ExtractorAgent(config)
        self.preprocessor = PreprocessorAgent(config)
        self.validator = ValidatorAgent(config)
        self.trainer = TrainerAgent(config)
        self.evaluator = EvaluatorAgent(config)
        self.self_improver = SelfImprovementAgent(config)
        self.previous_perplexity = 1.0  # Initial baseline; persist in production

    def orchestrate(self):
        try:
            papers = self.extractor.extract()
            if len(papers) < self.config['min_papers_threshold']:
                logger.info("Insufficient new papers; skipping cycle.")
                return

            processed = self.preprocessor.preprocess(papers)
            validated = self.validator.validate(processed)
            if not validated:
                logger.info("No valid data after validation; ending cycle.")
                return

            model_path = self.trainer.train(validated)

            test_data = [chunk for paper in validated[:int(len(validated)*0.1)] for chunk in paper['chunks']]  # 10% hold-out
            metrics = self.evaluator.evaluate(model_path, test_data)

            # Check threshold (example: relative perplexity improvement)
            if metrics['perplexity'] > self.config['perplexity_threshold'] * self.previous_perplexity:
                updated_config = self.self_improver.improve(metrics)
                self.config.update(updated_config)
                logger.info("Config updated for next cycle based on metrics.")
            self.previous_perplexity = metrics['perplexity']

            # Log cycle results
            log_path = os.path.join(self.config['logs_dir'], f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(log_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info("Cycle completed successfully.")
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")