"""
ScholarFormer Model — Custom transformer for academic paper understanding.

Architecture (per decoder block):
    Input → RMSNorm → Self-Attention → Residual
          → RMSNorm → Retrieval-Fused Cross-Attention → Residual
          → RMSNorm → SwiGLU FFN → Residual

Novel components:
    1. SectionAwarePositionalEncoding: position + learned section embeddings
    2. RetrievalFusedCrossAttention: cross-attention over projected FAISS vectors

~200M parameters | 12 layers | 768 hidden | 12 SA heads + 4 CA heads per layer
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from scholarformer.config import ScholarFormerConfig

logger = logging.getLogger(__name__)


# ==============================================================================
# Building Blocks
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).
    
    More stable and efficient than LayerNorm for transformers.
    Used in LLaMA, Phi-3, and other modern architectures.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al., 2021.
    
    Encodes relative position information directly into attention scores.
    Used by most modern LLMs (LLaMA, Phi-3, Mistral).
    """
    
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute sin/cos cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build sin/cos cache for given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, 
                          cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors."""
    def rotate_half(x):
        """Rotate half of the hidden dimensions."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    # Reshape cos/sin for broadcasting: (seq_len, dim) → (1, 1, seq_len, dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ==============================================================================
# Novel Component 1: Section-Aware Positional Encoding
# ==============================================================================

class SectionAwarePositionalEncoding(nn.Module):
    """
    NOVEL: Combines RoPE with learned section embeddings.
    
    Research papers have strong structural patterns:
        Abstract → Introduction → Methods → Results → Discussion → Conclusion
    
    Standard positional encoding only knows token position.
    Section-aware encoding additionally tells the model WHERE in the paper
    structure the current tokens come from.
    
    Section types:
        0 = abstract, 1 = introduction, 2 = methods,
        3 = results, 4 = discussion, 5 = conclusion, 6 = other
    """
    
    def __init__(self, config: ScholarFormerConfig):
        super().__init__()
        self.config = config
        
        # RoPE for token positions (applied in attention)
        self.rotary_emb = RotaryPositionalEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len
        )
        
        # Learned section embeddings — added to hidden states
        self.section_embeddings = nn.Embedding(config.num_sections, config.hidden_dim)
        
        # Learnable scaling factor for section contribution
        self.section_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Initialize section embeddings with small values
        nn.init.normal_(self.section_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, hidden_states: torch.Tensor, 
                section_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add section-aware encoding to hidden states.
        
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            section_ids: (batch, seq_len) — section type for each token
                         If None, defaults to 'other' (section_id=6)
        
        Returns:
            hidden_states with section information added
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        if section_ids is None:
            # Default to 'other' section
            section_ids = torch.full(
                (batch_size, seq_len), 
                fill_value=self.config.num_sections - 1,
                dtype=torch.long,
                device=hidden_states.device
            )
        
        # Look up section embeddings and add with learned scaling
        section_emb = self.section_embeddings(section_ids)  # (batch, seq, hidden)
        hidden_states = hidden_states + self.section_scale * section_emb
        
        return hidden_states
    
    def get_rotary_emb(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get RoPE cos/sin for attention layers."""
        return self.rotary_emb(None, seq_len)


# ==============================================================================
# Standard Multi-Head Self-Attention
# ==============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE and causal masking.
    
    Standard transformer self-attention with:
    - Rotary positional embeddings (applied to Q, K)
    - Causal mask (autoregressive — can only attend to past tokens)
    - Pre-norm architecture (norm applied before attention)
    """
    
    def __init__(self, config: ScholarFormerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_dim = config.hidden_dim
        
        # Q, K, V, O projections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        
        self.attn_dropout = nn.Dropout(config.attention_dropout)
    
    def forward(self, hidden_states: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            cos, sin: RoPE embeddings from positional encoding
            attention_mask: (batch, 1, seq_len, seq_len) causal mask
        
        Returns:
            (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape: (batch, seq, hidden) → (batch, heads, seq, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back: (batch, heads, seq, head_dim) → (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection
        return self.o_proj(attn_output)


# ==============================================================================
# Novel Component 2: Retrieval-Fused Cross-Attention
# ==============================================================================

class RetrievalFusedCrossAttention(nn.Module):
    """
    NOVEL: Cross-attention over projected FAISS retrieval vectors.
    
    Instead of RAG-style prompt stuffing (pasting retrieved documents as text),
    this layer lets the model attend DIRECTLY to retrieval embeddings through
    dedicated cross-attention heads.
    
    Inspired by Flamingo (Alayrac et al., 2022) which used cross-attention
    to fuse visual features into a frozen LLM. We apply the same principle
    to fuse retrieval features for domain-specific question answering.
    
    Architecture:
        - Query comes from the decoder hidden states
        - Key/Value come from projected FAISS embeddings (384 → 768)
        - Uses fewer heads (4) than self-attention (12) for efficiency
        - Gated residual: output = hidden + gate * cross_attn(hidden, retrieval)
    """
    
    def __init__(self, config: ScholarFormerConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_cross_attn_heads
        self.head_dim = config.head_dim
        self.cross_dim = self.num_heads * self.head_dim  # 4 * 64 = 256
        
        # Project retrieval embeddings: 384 (MiniLM) → hidden_dim
        self.retrieval_proj = nn.Linear(config.retrieval_dim, config.hidden_dim, bias=False)
        
        # Cross-attention projections
        self.q_proj = nn.Linear(config.hidden_dim, self.cross_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, self.cross_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, self.cross_dim, bias=False)
        self.o_proj = nn.Linear(self.cross_dim, config.hidden_dim, bias=False)
        
        # Learned gating — controls how much retrieval signal flows in
        # Initialized near zero so the model starts close to a standard LM
        self.gate = nn.Parameter(torch.zeros(1))
        
        self.attn_dropout = nn.Dropout(config.cross_attn_dropout)
    
    def forward(self, hidden_states: torch.Tensor,
                retrieval_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) — from self-attention
            retrieval_embeddings: (batch, num_retrieved, retrieval_dim)
                                  Raw FAISS vectors (384-dim from MiniLM)
                                  If None, cross-attention is skipped (gated out)
        
        Returns:
            (batch, seq_len, hidden_dim) — hidden states with retrieval info fused in
        """
        # If no retrieval context, skip entirely (multiply by 0 gate)
        if retrieval_embeddings is None:
            return hidden_states
        
        batch_size, seq_len, _ = hidden_states.shape
        _, num_retrieved, _ = retrieval_embeddings.shape
        
        # Project retrieval embeddings to hidden dimension
        retrieval_hidden = self.retrieval_proj(retrieval_embeddings)  # (batch, num_ret, hidden)
        
        # Cross-attention: Q from hidden, K/V from retrieval
        q = self.q_proj(hidden_states)       # (batch, seq, cross_dim)
        k = self.k_proj(retrieval_hidden)    # (batch, num_ret, cross_dim)
        v = self.v_proj(retrieval_hidden)    # (batch, num_ret, cross_dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_retrieved, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_retrieved, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention (no causal mask — can attend to all retrieved docs)
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Weighted sum
        cross_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        cross_output = cross_output.transpose(1, 2).contiguous()
        cross_output = cross_output.view(batch_size, seq_len, self.cross_dim)
        
        # Output projection
        cross_output = self.o_proj(cross_output)
        
        # Gated residual — gate starts near 0, model learns when to use retrieval
        return hidden_states + torch.tanh(self.gate) * cross_output


# ==============================================================================
# SwiGLU Feed-Forward Network
# ==============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).
    
    More efficient than standard MLP. Used in LLaMA, PaLM, and Phi-3.
    
    SwiGLU(x) = (Swish(W_gate · x) ⊙ W_up · x) · W_down
    
    Where ⊙ is element-wise multiplication and Swish(x) = x · sigmoid(x).
    """
    
    def __init__(self, config: ScholarFormerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_intermediate_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


# ==============================================================================
# ScholarFormer Decoder Block
# ==============================================================================

class ScholarFormerBlock(nn.Module):
    """
    Single ScholarFormer decoder block.
    
    Architecture (pre-norm):
        Input
          → RMSNorm → Self-Attention → + Residual
          → RMSNorm → Retrieval Cross-Attention → + Residual (gated)
          → RMSNorm → SwiGLU FFN → + Residual
        Output
    """
    
    def __init__(self, config: ScholarFormerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Pre-norms
        self.self_attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.cross_attn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        
        # Attention layers
        self.self_attention = MultiHeadSelfAttention(config)
        self.cross_attention = RetrievalFusedCrossAttention(config)
        
        # Feed-forward
        self.ffn = SwiGLU(config)
        
        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, hidden_states: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                retrieval_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            cos, sin: RoPE embeddings
            attention_mask: causal mask
            retrieval_embeddings: (batch, num_retrieved, retrieval_dim) — FAISS vectors
        
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # 1. Self-Attention with residual
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attention(hidden_states, cos, sin, attention_mask)
        hidden_states = self.dropout(hidden_states) + residual
        
        # 2. Retrieval-Fused Cross-Attention with gated residual
        # RetrievalFusedCrossAttention returns: input + tanh(gate) * cross_output
        # With pre-norm pattern, we norm first, then extract the delta:
        #   normed = norm(hidden_states)
        #   ca_output = normed + tanh(gate) * cross(normed, retrieval)
        #   hidden_states = hidden_states + (ca_output - normed)  
        #                 = hidden_states + tanh(gate) * cross(normed, retrieval)
        residual = hidden_states
        normed = self.cross_attn_norm(hidden_states)
        ca_output = self.cross_attention(normed, retrieval_embeddings)
        hidden_states = residual + (ca_output - normed)
        
        # 3. SwiGLU FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        
        return hidden_states


# ==============================================================================
# Full ScholarFormer Model
# ==============================================================================

class ScholarFormerModel(nn.Module):
    """
    ScholarFormer — Custom transformer for academic paper understanding.
    
    A ~200M parameter causal language model with two novel innovations:
    
    1. Section-Aware Positional Encoding:
       Combines RoPE with learned section embeddings so the model knows
       which part of a paper (Abstract, Methods, Results, etc.) it's reading.
    
    2. Retrieval-Fused Cross-Attention:
       Dedicated cross-attention layers that attend directly to FAISS vectors,
       architecturally integrating retrieval instead of prompt-stuffing.
    
    Architecture:
        Token Embedding + Section Encoding
        → 12 × (Self-Attn → Cross-Attn → SwiGLU)
        → RMSNorm → LM Head
    
    Usage:
        config = ScholarFormerConfig()
        model = ScholarFormerModel(config)
        
        # Standard forward pass (language modeling)
        logits = model(input_ids, section_ids=section_ids)
        
        # With retrieval (RAG-style but architectural)
        logits = model(input_ids, retrieval_embeddings=faiss_vectors)
    """
    
    def __init__(self, config: ScholarFormerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Section-Aware Positional Encoding (NOVEL)
        self.positional_encoding = SectionAwarePositionalEncoding(config)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # Transformer decoder blocks
        self.layers = nn.ModuleList([
            ScholarFormerBlock(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.hidden_dim, config.rms_norm_eps)
        
        # Language model head
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Tie embeddings (share weights between input embedding and LM head)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.use_gradient_checkpointing
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ScholarFormer initialized: {total_params:,} total params, "
                    f"{trainable_params:,} trainable ({100*trainable_params/total_params:.1f}%)")
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
    
    def resize_token_embeddings(self, new_vocab_size: int):
        """
        Resize token embedding and LM head to match a larger vocabulary.
        
        Needed when the tokenizer adds special tokens (e.g., section markers)
        beyond the original vocab_size in the config/checkpoint.
        
        New embedding rows are initialized with small random values.
        """
        old_vocab_size = self.token_embedding.num_embeddings
        if new_vocab_size == old_vocab_size:
            return  # Nothing to do
        
        logger.info(f"Resizing embeddings: {old_vocab_size} → {new_vocab_size}")
        
        # Create new embedding
        new_embedding = nn.Embedding(new_vocab_size, self.config.hidden_dim)
        new_embedding.weight.data[:old_vocab_size] = self.token_embedding.weight.data
        # Initialize new rows with small random values
        if new_vocab_size > old_vocab_size:
            nn.init.normal_(
                new_embedding.weight.data[old_vocab_size:], 
                mean=0.0, std=self.config.initializer_range
            )
        
        self.token_embedding = new_embedding
        
        # Resize LM head
        if self.config.tie_word_embeddings:
            self.lm_head = nn.Linear(self.config.hidden_dim, new_vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        else:
            old_lm_head = self.lm_head
            self.lm_head = nn.Linear(self.config.hidden_dim, new_vocab_size, bias=False)
            self.lm_head.weight.data[:old_vocab_size] = old_lm_head.weight.data
        
        # Update config
        self.config.vocab_size = new_vocab_size
        logger.info(f"Embeddings resized to {new_vocab_size}")
    
    def _make_causal_mask(self, seq_len: int, device: torch.device, 
                           dtype: torch.dtype) -> torch.Tensor:
        """Create causal attention mask (lower triangular)."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)  # Upper triangle = -inf
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)
    
    def forward(self, 
                input_ids: torch.Tensor,
                section_ids: Optional[torch.Tensor] = None,
                retrieval_embeddings: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> dict:
        """
        Forward pass.
        
        Args:
            input_ids: (batch, seq_len) — token IDs
            section_ids: (batch, seq_len) — section type per token (0-6)
                         Optional: defaults to 'other' if not provided
            retrieval_embeddings: (batch, num_retrieved, 384) — FAISS vectors
                                  Optional: cross-attention skipped if not provided
            labels: (batch, seq_len) — target token IDs for loss computation
                    Typically input_ids shifted right by 1
        
        Returns:
            dict with 'logits', and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add section-aware encoding
        hidden_states = self.positional_encoding(hidden_states, section_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Get RoPE embeddings
        cos, sin = self.positional_encoding.get_rotary_emb(seq_len)
        cos = cos.to(device)
        sin = sin.to(device)
        
        # Create causal mask
        causal_mask = self._make_causal_mask(seq_len, device, hidden_states.dtype)
        
        # Pass through decoder blocks
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    layer,
                    hidden_states, cos, sin, causal_mask, retrieval_embeddings,
                    use_reentrant=False
                )
            else:
                hidden_states = layer(
                    hidden_states, cos, sin, causal_mask, retrieval_embeddings
                )
        
        # Final norm and LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        result = {'logits': logits}
        
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100  # Ignore padding
            )
            result['loss'] = loss
        
        return result
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 256,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 section_ids: Optional[torch.Tensor] = None,
                 retrieval_embeddings: Optional[torch.Tensor] = None,
                 eos_token_id: Optional[int] = None) -> torch.Tensor:
        """
        Autoregressive generation with top-k/top-p sampling.
        
        Args:
            input_ids: (batch, prefix_len) — prompt tokens
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (lower = more deterministic)
            top_p: nucleus sampling threshold
            top_k: top-k sampling threshold
            section_ids: section encoding for prompt tokens
            retrieval_embeddings: FAISS vectors for cross-attention
            eos_token_id: stop generation when this token is produced
        
        Returns:
            (batch, prefix_len + generated_len) — full token sequence
        """
        self.eval()
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Truncate to max sequence length if needed
            current_input = generated
            if current_input.shape[1] > self.config.max_seq_len:
                current_input = current_input[:, -self.config.max_seq_len:]
            
            # Prepare section_ids for current length
            current_section_ids = None
            if section_ids is not None:
                # Extend section_ids with 'other' for generated tokens
                gen_len = current_input.shape[1] - section_ids.shape[1]
                if gen_len > 0:
                    other_ids = torch.full(
                        (section_ids.shape[0], gen_len),
                        fill_value=self.config.num_sections - 1,
                        dtype=torch.long,
                        device=section_ids.device
                    )
                    current_section_ids = torch.cat([section_ids, other_ids], dim=1)
                else:
                    current_section_ids = section_ids[:, :current_input.shape[1]]
            
            # Forward pass
            outputs = self.forward(
                current_input, 
                section_ids=current_section_ids,
                retrieval_embeddings=retrieval_embeddings
            )
            
            # Get logits for the last token
            next_logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float('-inf')
                
                # Scatter back to original ordering
                next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {
            'token_embedding': sum(p.numel() for p in self.token_embedding.parameters()),
            'positional_encoding': sum(p.numel() for p in self.positional_encoding.parameters()),
            'self_attention': 0,
            'cross_attention': 0,
            'ffn': 0,
            'norms': 0,
            'lm_head': 0 if self.config.tie_word_embeddings else sum(p.numel() for p in self.lm_head.parameters()),
        }
        
        for layer in self.layers:
            counts['self_attention'] += sum(p.numel() for p in layer.self_attention.parameters())
            counts['cross_attention'] += sum(p.numel() for p in layer.cross_attention.parameters())
            counts['ffn'] += sum(p.numel() for p in layer.ffn.parameters())
            counts['norms'] += sum(p.numel() for p in layer.self_attn_norm.parameters())
            counts['norms'] += sum(p.numel() for p in layer.cross_attn_norm.parameters())
            counts['norms'] += sum(p.numel() for p in layer.ffn_norm.parameters())
        
        counts['final_norm'] = sum(p.numel() for p in self.final_norm.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts
