"""
ScholarFormer — A custom transformer architecture for academic paper understanding.

Novel components:
1. Section-Aware Positional Encoding
2. Retrieval-Fused Cross-Attention (Flamingo-style)

Built from scratch as part of the ScholarMind project.
"""

from scholarformer.config import ScholarFormerConfig
from scholarformer.model import ScholarFormerModel

# Tokenizer uses HuggingFace transformers — import separately to avoid
# version conflicts when only the model is needed
def get_tokenizer(*args, **kwargs):
    """Lazy import for ScholarFormerTokenizer."""
    from scholarformer.tokenizer import ScholarFormerTokenizer
    return ScholarFormerTokenizer(*args, **kwargs)

__all__ = [
    'ScholarFormerConfig',
    'ScholarFormerModel',
    'get_tokenizer',
]

__version__ = '1.0.0'
