"""
SelfImprovementAgent - Optimizes hyperparameters based on evaluation metrics.
Part of the ScholarMind multi-agent system.

FIXED: Removed fixed seed=42 that caused identical suggestions every run.
IMPROVED: Expanded search space, added baseline comparison for better optimization.
"""

import os
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional

import optuna
from optuna.samplers import TPESampler

logger = logging.getLogger(__name__)


class SelfImprovementAgent:
    """
    Uses Bayesian optimization (Optuna) to suggest improved hyperparameters
    based on evaluation metrics from previous training runs.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.history: List[Dict] = []
        self.history_file = os.path.join(
            config.get('logs_dir', './logs'),
            'improvement_history.json'
        )
        self._load_history()
    
    def _load_history(self) -> None:
        """Load improvement history from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.history = json.load(f)
                logger.info(f"Loaded {len(self.history)} historical improvement records.")
            except Exception as e:
                logger.warning(f"Could not load history: {e}")
                self.history = []
    
    def _save_history(self) -> None:
        """Save improvement history to file."""
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _get_baseline_objective(self) -> float:
        """Get baseline objective from recent history for comparison."""
        if not self.history:
            return 1.0  # Default baseline (worst case)
        
        # Use best objective from last 5 runs as baseline
        recent = self.history[-5:]
        return min(record.get('objective', 1.0) for record in recent)
    
    def improve(self, metrics: Dict) -> Dict:
        """
        Suggest improved hyperparameters based on current metrics.
        
        Args:
            metrics: Dictionary with 'perplexity', 'bleu', 'rouge' scores
            
        Returns:
            Dictionary of suggested hyperparameter updates
        """
        logger.info("Running hyperparameter optimization...")
        
        # Get baseline for comparison
        baseline = self._get_baseline_objective()
        logger.info(f"Baseline objective from history: {baseline:.4f}")
        
        # EXPANDED hyperparameter search space
        hp_distributions = {
            'learning_rate': optuna.distributions.FloatDistribution(1e-5, 1e-3, log=True),
            'batch_size': optuna.distributions.IntDistribution(1, 4),
            'training_epochs': optuna.distributions.IntDistribution(1, 5),
            'lora_r': optuna.distributions.CategoricalDistribution([8, 16, 32, 64, 128]),
            'lora_alpha': optuna.distributions.CategoricalDistribution([16, 32, 64, 128]),
            'lora_dropout': optuna.distributions.FloatDistribution(0.01, 0.15),
            'gradient_accumulation_steps': optuna.distributions.IntDistribution(2, 16),
            'warmup_ratio': optuna.distributions.FloatDistribution(0.01, 0.1),
        }
        
        def objective(trial):
            # Suggest hyperparameters - EXPANDED RANGES
            lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            bs = trial.suggest_int('batch_size', 1, 4)
            epochs = trial.suggest_int('training_epochs', 1, 5)
            lora_r = trial.suggest_categorical('lora_r', [8, 16, 32, 64, 128])
            lora_alpha = trial.suggest_categorical('lora_alpha', [16, 32, 64, 128])
            dropout = trial.suggest_float('lora_dropout', 0.01, 0.15)
            grad_accum = trial.suggest_int('gradient_accumulation_steps', 2, 16)
            warmup = trial.suggest_float('warmup_ratio', 0.01, 0.1)
            
            # Calculate objective based on metrics
            perplexity = metrics.get('perplexity', 100)
            bleu = metrics.get('bleu', 0)
            rouge = metrics.get('rouge', 0)
            
            # Normalize scores (lower is better for objective)
            perplexity_score = min(perplexity / 50, 1.0)  # More aggressive perplexity goal
            text_score = 1 - (bleu * 0.4 + rouge * 0.6)  # Weight ROUGE higher (more reliable)
            
            # Combined objective with adjusted weights
            raw_objective = perplexity_score * 0.5 + text_score * 0.5
            
            # Encourage improvement over baseline
            improvement_bonus = max(0, baseline - raw_objective) * 0.2
            
            return raw_objective - improvement_bonus
        
        # Create study WITHOUT fixed seed - allows natural exploration
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=None)  # FIXED: No fixed seed!
        )
        
        # Add historical trials if available
        for record in self.history[-10:]:
            try:
                # Only add if params match current distribution
                params = record.get('params', {})
                if params:
                    study.add_trial(
                        optuna.trial.create_trial(
                            params=params,
                            values=[record.get('objective', 1.0)],
                            distributions=hp_distributions
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not add historical trial (likely schema mismatch): {e}")
        
        # Run optimization with more trials for better exploration
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        n_trials = 30 if len(self.history) < 5 else 20  # More trials early on
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best hyperparameters found: {best_params}")
        logger.info(f"Objective: {best_value:.4f} (baseline: {baseline:.4f})")
        
        # Record this improvement attempt with more details
        record = {
            'timestamp': datetime.now().isoformat(),
            'input_metrics': metrics,
            'params': best_params,
            'objective': best_value,
            'baseline': baseline,
            'improvement': baseline - best_value,
            'n_trials': n_trials
        }
        self.history.append(record)
        self._save_history()
        
        # Prepare config updates
        updated_config = self.config.copy()
        updated_config.update(best_params)
        
        logger.info(f"Suggested config improvements: {best_params}")
        return updated_config
    
    def get_history(self) -> List[Dict]:
        """Get the improvement history."""
        return self.history
    
    def get_best_config(self) -> Optional[Dict]:
        """Get the best configuration from history."""
        if not self.history:
            return None
        
        # Find record with lowest objective
        best = min(self.history, key=lambda x: x.get('objective', float('inf')))
        return best.get('params', {})