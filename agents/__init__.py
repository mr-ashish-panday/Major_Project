"""
ScholarMind Agents Package
A multi-agent system for self-evolving LLM research assistance.
"""

from agents.extractor import ExtractorAgent
from agents.preprocessor import PreprocessorAgent
from agents.validator import ValidatorAgent
from agents.vector_store import VectorStoreAgent
from agents.trainer import TrainerAgent
from agents.evaluator import EvaluatorAgent
from agents.self_improvement import SelfImprovementAgent
from agents.orchestrator import OrchestratorAgent
from agents.rag_pipeline import RAGPipeline
from agents.custom_trainer import CustomTrainerAgent
from agents.distillation import DistillationAgent
from agents.comparison import ComparisonAgent

__all__ = [
    'ExtractorAgent',
    'PreprocessorAgent',
    'ValidatorAgent',
    'VectorStoreAgent',
    'TrainerAgent',
    'EvaluatorAgent',
    'SelfImprovementAgent',
    'OrchestratorAgent',
    'RAGPipeline',
    'CustomTrainerAgent',
    'DistillationAgent',
    'ComparisonAgent',
]
