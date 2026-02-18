"""
ScholarFormerConfig — Configuration for the ScholarFormer architecture.

Defines all hyperparameters for the ~200M parameter model:
- 12 decoder layers, 768 hidden dim, 12 attention heads
- SwiGLU FFN (768 → 2048 → 768)
- 4 cross-attention heads per layer for retrieval fusion
- 7 section types for section-aware positional encoding
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScholarFormerConfig:
    """Configuration for ScholarFormer model."""
    
    # ==========================================
    # Core Architecture
    # ==========================================
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12          # Self-attention heads
    head_dim: int = 64           # hidden_dim // num_heads
    
    # ==========================================
    # Vocabulary & Sequence
    # ==========================================
    vocab_size: int = 32_000     # BPE vocabulary (shared with Phi-3 tokenizer)
    max_seq_len: int = 1024      # Maximum sequence length
    
    # ==========================================
    # SwiGLU Feed-Forward Network
    # ==========================================
    ffn_intermediate_dim: int = 2048  # SwiGLU intermediate dimension
    
    # ==========================================
    # Retrieval-Fused Cross-Attention (NOVEL)
    # ==========================================
    num_cross_attn_heads: int = 4     # Cross-attention heads per layer
    retrieval_dim: int = 384          # MiniLM-L6-v2 output dimension
    max_retrieval_tokens: int = 64    # Max retrieved chunks to attend over
    cross_attn_dropout: float = 0.1
    
    # ==========================================
    # Section-Aware Positional Encoding (NOVEL)
    # ==========================================
    num_sections: int = 7             # Number of paper section types
    section_types: List[str] = field(default_factory=lambda: [
        'abstract',
        'introduction', 
        'methods',
        'results',
        'discussion',
        'conclusion',
        'other'
    ])
    
    # ==========================================
    # Regularization
    # ==========================================
    dropout: float = 0.1
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.1
    
    # ==========================================
    # Normalization
    # ==========================================
    rms_norm_eps: float = 1e-6
    
    # ==========================================
    # Initialization
    # ==========================================
    initializer_range: float = 0.02
    
    # ==========================================
    # Training
    # ==========================================
    tie_word_embeddings: bool = True  # Share input/output embeddings
    use_gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        self.head_dim = self.hidden_dim // self.num_heads
    
    @property
    def num_parameters_estimate(self) -> int:
        """Estimate total parameter count."""
        # Embeddings
        embed = self.vocab_size * self.hidden_dim  # token embeddings
        embed += self.max_seq_len * self.hidden_dim  # positional
        embed += self.num_sections * self.hidden_dim  # section embeddings
        
        # Per-layer
        per_layer = 0
        # Self-attention: Q, K, V, O projections
        per_layer += 4 * self.hidden_dim * self.hidden_dim
        # Cross-attention: Q, K, V, O projections (smaller)
        cross_dim = self.num_cross_attn_heads * self.head_dim
        per_layer += 2 * self.hidden_dim * cross_dim  # Q, O from hidden
        per_layer += 2 * self.hidden_dim * cross_dim  # K, V from retrieval (projected)
        # Retrieval projection (384 → hidden)
        per_layer += self.retrieval_dim * self.hidden_dim
        # SwiGLU FFN: gate + up + down
        per_layer += 3 * self.hidden_dim * self.ffn_intermediate_dim
        # Layer norms (3 per layer)
        per_layer += 3 * self.hidden_dim
        
        total_layers = per_layer * self.num_layers
        
        # Final norm + LM head (tied = 0 extra)
        final = self.hidden_dim
        if not self.tie_word_embeddings:
            final += self.vocab_size * self.hidden_dim
        
        return embed + total_layers + final
    
    def summary(self) -> str:
        """Print a human-readable summary."""
        est = self.num_parameters_estimate
        return (
            f"ScholarFormer Config:\n"
            f"  Layers: {self.num_layers}\n"
            f"  Hidden: {self.hidden_dim}\n"
            f"  Heads: {self.num_heads} (self-attn) + {self.num_cross_attn_heads} (cross-attn)\n"
            f"  FFN: SwiGLU {self.hidden_dim} → {self.ffn_intermediate_dim}\n"
            f"  Vocab: {self.vocab_size:,}\n"
            f"  Max Seq: {self.max_seq_len}\n"
            f"  Sections: {self.num_sections}\n"
            f"  Est. Params: ~{est / 1e6:.1f}M\n"
        )
