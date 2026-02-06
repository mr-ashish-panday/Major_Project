"""
OrchestratorAgent - Coordinates all agents in the ScholarMind pipeline.
Main controller for the self-improving LLM research assistant.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional

from agents.extractor import ExtractorAgent
from agents.preprocessor import PreprocessorAgent
from agents.validator import ValidatorAgent
from agents.vector_store import VectorStoreAgent
from agents.trainer import TrainerAgent
from agents.evaluator import EvaluatorAgent
from agents.self_improvement import SelfImprovementAgent

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Orchestrates the complete ScholarMind pipeline:
    1. Extract papers from arXiv
    2. Preprocess PDFs into text chunks
    3. Validate and filter papers
    4. Store in vector database for RAG
    5. Fine-tune the model
    6. Evaluate performance
    7. Self-improve hyperparameters
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.previous_perplexity = 1.0
        
        logger.info("=" * 50)
        logger.info("Initializing ScholarMind Agents...")
        logger.info("=" * 50)
        
        logger.info("Initializing ExtractorAgent...")
        self.extractor = ExtractorAgent(config)
        
        logger.info("Initializing PreprocessorAgent...")
        self.preprocessor = PreprocessorAgent(config)
        
        logger.info("Initializing ValidatorAgent...")
        self.validator = ValidatorAgent(config)
        
        logger.info("Initializing VectorStoreAgent...")
        self.vector_store = VectorStoreAgent(config)
        
        logger.info("Initializing TrainerAgent...")
        self.trainer = TrainerAgent(config)
        
        logger.info("Initializing EvaluatorAgent...")
        self.evaluator = EvaluatorAgent(config)
        
        logger.info("Initializing SelfImprovementAgent...")
        self.self_improver = SelfImprovementAgent(config)
        
        logger.info("All agents initialized successfully!")
    
    def orchestrate(self, skip_training: bool = False) -> Optional[str]:
        """
        Run the complete pipeline.
        
        Args:
            skip_training: If True, skip the training step (useful for just ingesting papers)
            
        Returns:
            Path to the fine-tuned model, or None if training was skipped
        """
        try:
            # Step 1: Extract papers from arXiv
            logger.info("\n" + "=" * 50)
            logger.info("STEP 1: Extracting papers from arXiv...")
            logger.info("=" * 50)
            papers = self.extractor.extract()
            logger.info(f"Extracted {len(papers)} papers")
            
            if len(papers) < self.config.get('min_papers_threshold', 10):
                logger.warning(f"Insufficient papers ({len(papers)}). Minimum required: {self.config.get('min_papers_threshold', 10)}")
                logger.info("Continuing with available papers...")
            
            if not papers:
                logger.error("No papers extracted. Ending cycle.")
                return None
            
            # Step 2: Preprocess papers
            logger.info("\n" + "=" * 50)
            logger.info("STEP 2: Preprocessing papers...")
            logger.info("=" * 50)
            processed = self.preprocessor.preprocess(papers)
            logger.info(f"Preprocessed {len(processed)} papers")
            
            if not processed:
                logger.error("No papers after preprocessing. Ending cycle.")
                return None
            
            # Step 3: Validate papers
            logger.info("\n" + "=" * 50)
            logger.info("STEP 3: Validating papers...")
            logger.info("=" * 50)
            validated = self.validator.validate(processed)
            logger.info(f"Validated {len(validated)} papers")
            
            if not validated:
                logger.error("No papers passed validation. Ending cycle.")
                return None
            
            # Step 4: Store in vector database for RAG
            logger.info("\n" + "=" * 50)
            logger.info("STEP 4: Storing papers in vector database...")
            logger.info("=" * 50)
            chunks_added = self.vector_store.add_papers(validated)
            logger.info(f"Added {chunks_added} chunks to vector store")
            logger.info(f"Vector store now has {self.vector_store.collection.count()} total documents")
            
            if skip_training:
                logger.info("Training skipped as requested.")
                return None
            
            # Step 5: Train the model
            logger.info("\n" + "=" * 50)
            logger.info("STEP 5: Fine-tuning the model...")
            logger.info("=" * 50)
            model_path = self.trainer.train(validated)
            logger.info(f"Model saved at: {model_path}")
            
            # Step 6: Evaluate the model
            logger.info("\n" + "=" * 50)
            logger.info("STEP 6: Evaluating the model...")
            logger.info("=" * 50)
            
            # Use 10% of data for evaluation
            test_data = []
            for paper in validated[:max(1, int(len(validated) * 0.1))]:
                test_data.extend(paper.get('chunks', [])[:5])  # Take up to 5 chunks per paper
            
            metrics = self.evaluator.evaluate(model_path, test_data)
            logger.info(f"Evaluation metrics: {metrics}")
            
            # Step 7: Self-improvement check
            logger.info("\n" + "=" * 50)
            logger.info("STEP 7: Checking for self-improvement...")
            logger.info("=" * 50)
            
            if metrics.get('perplexity', 0) > self.config.get('perplexity_threshold', 0.85) * self.previous_perplexity:
                logger.info("Performance degradation detected. Triggering self-improvement...")
                updated_config = self.self_improver.improve(metrics)
                self.config.update(updated_config)
                logger.info(f"Config updated: {updated_config}")
            else:
                logger.info("Performance is satisfactory. No adjustment needed.")
            
            self.previous_perplexity = metrics.get('perplexity', self.previous_perplexity)
            
            # Log cycle results
            self._log_cycle_results(metrics, model_path)
            
            logger.info("\n" + "=" * 50)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 50)
            
            return model_path
            
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            return None
    
    def ingest_only(self) -> int:
        """
        Only extract, preprocess, validate, and store papers without training.
        Useful for building up the knowledge base.
        
        Returns:
            Number of chunks added to vector store
        """
        logger.info("Running ingestion-only mode...")
        
        papers = self.extractor.extract()
        if not papers:
            return 0
        
        processed = self.preprocessor.preprocess(papers)
        if not processed:
            return 0
        
        validated = self.validator.validate(processed)
        if not validated:
            return 0
        
        chunks_added = self.vector_store.add_papers(validated)
        logger.info(f"Ingestion complete. Added {chunks_added} chunks.")
        return chunks_added
    
    def _log_cycle_results(self, metrics: Dict, model_path: str) -> None:
        """Save cycle results to a log file."""
        log_path = os.path.join(
            self.config.get('logs_dir', './logs'),
            f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'metrics': metrics,
            'config': {k: v for k, v in self.config.items() if not callable(v)},
            'vector_store_stats': self.vector_store.get_stats()
        }
        
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        logger.info(f"Cycle results saved to {log_path}")
    
    def get_status(self) -> Dict:
        """Get current status of all agents."""
        return {
            'vector_store': self.vector_store.get_stats(),
            'trainer': self.trainer.get_model_info(),
            'config': {k: v for k, v in self.config.items() if not callable(v)}
        }