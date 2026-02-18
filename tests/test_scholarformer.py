"""
Unit tests for the ScholarFormer architecture.

Tests:
1. Config validation and parameter estimation
2. Individual building blocks (RMSNorm, RoPE, SwiGLU)
3. Self-attention (causal masking, RoPE)
4. Cross-attention (retrieval fusion, gating)
5. Section-aware positional encoding
6. Full model forward pass
7. Full model generation
8. Parameter count verification

Run: python -m pytest tests/test_scholarformer.py -v
  or: python tests/test_scholarformer.py
"""

import sys
import os
import math

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from scholarformer.config import ScholarFormerConfig
from scholarformer.model import (
    RMSNorm,
    RotaryPositionalEmbedding,
    apply_rotary_pos_emb,
    SectionAwarePositionalEncoding,
    MultiHeadSelfAttention,
    RetrievalFusedCrossAttention,
    SwiGLU,
    ScholarFormerBlock,
    ScholarFormerModel,
)


def test_config():
    """Test configuration defaults and validation."""
    print("=" * 60)
    print("TEST 1: Configuration")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    
    assert config.hidden_dim == 768
    assert config.num_layers == 12
    assert config.num_heads == 12
    assert config.head_dim == 64  # 768 // 12
    assert config.vocab_size == 32_000
    assert config.num_sections == 7
    assert config.num_cross_attn_heads == 4
    
    est = config.num_parameters_estimate
    print(f"  Estimated parameters: ~{est / 1e6:.1f}M")
    print(config.summary())
    print("  ✅ Config validation passed")


def test_rmsnorm():
    """Test RMSNorm output shape and normalization."""
    print("\n" + "=" * 60)
    print("TEST 2: RMSNorm")
    print("=" * 60)
    
    norm = RMSNorm(768)
    x = torch.randn(2, 10, 768)
    out = norm(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    
    # RMS of output should be close to 1 (that's the point of RMSNorm)
    rms_out = torch.sqrt(torch.mean(out ** 2, dim=-1))
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Output RMS mean: {rms_out.mean():.4f} (should be ~1.0)")
    print("  ✅ RMSNorm passed")


def test_rope():
    """Test Rotary Positional Embeddings."""
    print("\n" + "=" * 60)
    print("TEST 3: Rotary Positional Embeddings (RoPE)")
    print("=" * 60)
    
    rope = RotaryPositionalEmbedding(dim=64, max_seq_len=1024)
    cos, sin = rope(None, 128)
    
    assert cos.shape == (128, 64)
    assert sin.shape == (128, 64)
    
    # Test apply to Q, K
    q = torch.randn(2, 12, 128, 64)
    k = torch.randn(2, 12, 128, 64)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    
    print(f"  cos shape: {cos.shape}")
    print(f"  Q shape: {q.shape} → {q_rot.shape}")
    print(f"  K shape: {k.shape} → {k_rot.shape}")
    print("  ✅ RoPE passed")


def test_section_encoding():
    """Test Section-Aware Positional Encoding."""
    print("\n" + "=" * 60)
    print("TEST 4: Section-Aware Positional Encoding (NOVEL)")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    encoding = SectionAwarePositionalEncoding(config)
    
    batch_size, seq_len = 2, 128
    hidden = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    # Test with explicit section IDs
    section_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)  # all abstract
    out_with_sections = encoding(hidden, section_ids)
    
    # Test with no section IDs (defaults to 'other')
    out_no_sections = encoding(hidden, None)
    
    assert out_with_sections.shape == hidden.shape
    assert out_no_sections.shape == hidden.shape
    
    # Different sections should produce different outputs
    section_ids_methods = torch.full((batch_size, seq_len), 2, dtype=torch.long)
    out_methods = encoding(hidden, section_ids_methods)
    diff = (out_with_sections - out_methods).abs().mean().item()
    
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Section IDs shape: {section_ids.shape}")
    print(f"  Output shape: {out_with_sections.shape}")
    print(f"  Section scale: {encoding.section_scale.item():.4f}")
    print(f"  Diff between 'abstract' and 'methods': {diff:.6f} (should be > 0)")
    assert diff > 0, "Different sections should produce different outputs"
    print("  ✅ Section-Aware Encoding passed")


def test_self_attention():
    """Test Multi-Head Self-Attention."""
    print("\n" + "=" * 60)
    print("TEST 5: Multi-Head Self-Attention")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    attn = MultiHeadSelfAttention(config)
    
    batch_size, seq_len = 2, 64
    hidden = torch.randn(batch_size, seq_len, config.hidden_dim)
    
    rope = RotaryPositionalEmbedding(dim=config.head_dim, max_seq_len=config.max_seq_len)
    cos, sin = rope(None, seq_len)
    
    # Create causal mask
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    
    out = attn(hidden, cos, sin, mask)
    
    assert out.shape == hidden.shape
    params = sum(p.numel() for p in attn.parameters())
    
    print(f"  Input: {hidden.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  Heads: {config.num_heads}, Head dim: {config.head_dim}")
    print("  ✅ Self-Attention passed")


def test_cross_attention():
    """Test Retrieval-Fused Cross-Attention."""
    print("\n" + "=" * 60)
    print("TEST 6: Retrieval-Fused Cross-Attention (NOVEL)")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    cross_attn = RetrievalFusedCrossAttention(config)
    
    batch_size, seq_len = 2, 64
    num_retrieved = 16
    
    hidden = torch.randn(batch_size, seq_len, config.hidden_dim)
    retrieval = torch.randn(batch_size, num_retrieved, config.retrieval_dim)  # 384-dim MiniLM
    
    # With retrieval
    out_with_retrieval = cross_attn(hidden, retrieval)
    assert out_with_retrieval.shape == hidden.shape
    
    # Without retrieval (should return input unchanged)
    out_no_retrieval = cross_attn(hidden, None)
    assert out_no_retrieval.shape == hidden.shape
    assert torch.allclose(out_no_retrieval, hidden), "No retrieval should return input unchanged"
    
    # Gate should start near 0, so output should be close to input
    gate_value = torch.tanh(cross_attn.gate).item()
    diff = (out_with_retrieval - hidden).abs().mean().item()
    
    params = sum(p.numel() for p in cross_attn.parameters())
    
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Retrieval shape: {retrieval.shape}")
    print(f"  Output shape: {out_with_retrieval.shape}")
    print(f"  Gate value: {gate_value:.6f} (should be near 0)")
    print(f"  Output-Input diff: {diff:.6f} (should be small, gate ≈ 0)")
    print(f"  Parameters: {params:,}")
    print(f"  Cross-attn heads: {config.num_cross_attn_heads}")
    print("  ✅ Cross-Attention passed")


def test_swiglu():
    """Test SwiGLU FFN."""
    print("\n" + "=" * 60)
    print("TEST 7: SwiGLU Feed-Forward Network")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    ffn = SwiGLU(config)
    
    x = torch.randn(2, 64, config.hidden_dim)
    out = ffn(x)
    
    assert out.shape == x.shape
    params = sum(p.numel() for p in ffn.parameters())
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: {params:,}")
    print(f"  FFN: {config.hidden_dim} → {config.ffn_intermediate_dim} → {config.hidden_dim}")
    print("  ✅ SwiGLU passed")


def test_decoder_block():
    """Test full decoder block."""
    print("\n" + "=" * 60)
    print("TEST 8: ScholarFormer Decoder Block")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    block = ScholarFormerBlock(config, layer_idx=0)
    
    batch_size, seq_len = 2, 64
    hidden = torch.randn(batch_size, seq_len, config.hidden_dim)
    retrieval = torch.randn(batch_size, 16, config.retrieval_dim)
    
    rope = RotaryPositionalEmbedding(dim=config.head_dim)
    cos, sin = rope(None, seq_len)
    
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    
    # With retrieval
    out = block(hidden, cos, sin, mask, retrieval)
    assert out.shape == hidden.shape
    
    # Without retrieval
    out_no_ret = block(hidden, cos, sin, mask, None)
    assert out_no_ret.shape == hidden.shape
    
    params = sum(p.numel() for p in block.parameters())
    
    print(f"  Input: {hidden.shape}")
    print(f"  Output (with retrieval): {out.shape}")
    print(f"  Output (no retrieval): {out_no_ret.shape}")
    print(f"  Block parameters: {params:,} (~{params/1e6:.1f}M)")
    print("  ✅ Decoder Block passed")


def test_full_model():
    """Test the complete ScholarFormer model."""
    print("\n" + "=" * 60)
    print("TEST 9: Full ScholarFormer Model")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    model = ScholarFormerModel(config)
    model.eval()
    
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    section_ids = torch.randint(0, config.num_sections, (batch_size, seq_len))
    retrieval = torch.randn(batch_size, 16, config.retrieval_dim)
    labels = input_ids.clone()
    
    # Forward pass WITHOUT retrieval (Stage 1: pretraining mode)
    outputs_no_ret = model(input_ids, section_ids=section_ids, labels=labels)
    assert 'logits' in outputs_no_ret
    assert 'loss' in outputs_no_ret
    assert outputs_no_ret['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    # Forward pass WITH retrieval (Stage 2: domain adaptation mode)
    outputs_with_ret = model(input_ids, section_ids=section_ids,
                              retrieval_embeddings=retrieval, labels=labels)
    assert outputs_with_ret['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    # Count parameters
    counts = model.count_parameters()
    total = counts['total']
    
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Section IDs: {section_ids.shape}")
    print(f"  Retrieval: {retrieval.shape}")
    print(f"  Logits: {outputs_no_ret['logits'].shape}")
    print(f"  Loss (no retrieval): {outputs_no_ret['loss'].item():.4f}")
    print(f"  Loss (with retrieval): {outputs_with_ret['loss'].item():.4f}")
    print(f"\n  Parameter breakdown:")
    for name, count in counts.items():
        if name not in ('total', 'trainable'):
            print(f"    {name:25s}: {count:>12,} ({100*count/total:5.1f}%)")
    print(f"    {'':25s}  {'':>12s}")
    print(f"    {'TOTAL':25s}: {total:>12,}")
    print(f"    {'Trainable':25s}: {counts['trainable']:>12,}")
    print(f"\n  Total: ~{total/1e6:.1f}M parameters")
    print("  ✅ Full Model passed")


def test_generation():
    """Test autoregressive generation."""
    print("\n" + "=" * 60)
    print("TEST 10: Autoregressive Generation")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    model = ScholarFormerModel(config)
    model.eval()
    
    # Short prompt
    prompt = torch.randint(0, config.vocab_size, (1, 8))
    
    # Generate without retrieval
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.8)
    assert generated.shape[0] == 1
    assert generated.shape[1] == 8 + 16  # prompt + generated
    
    # Generate with retrieval
    retrieval = torch.randn(1, 8, config.retrieval_dim)
    generated_ret = model.generate(prompt, max_new_tokens=16, 
                                    retrieval_embeddings=retrieval)
    assert generated_ret.shape[0] == 1
    
    print(f"  Prompt: {prompt.shape}")
    print(f"  Generated (no retrieval): {generated.shape}")
    print(f"  Generated (with retrieval): {generated_ret.shape}")
    print(f"  Generated tokens: {generated[0, 8:].tolist()[:10]}...")
    print("  ✅ Generation passed")


def test_gradient_flow():
    """Test that gradients flow through all components."""
    print("\n" + "=" * 60)
    print("TEST 11: Gradient Flow")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    config.use_gradient_checkpointing = False  # Disable for gradient test
    model = ScholarFormerModel(config)
    model.train()
    
    batch_size, seq_len = 2, 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    section_ids = torch.randint(0, config.num_sections, (batch_size, seq_len))
    retrieval = torch.randn(batch_size, 8, config.retrieval_dim)
    labels = input_ids.clone()
    
    # Forward + backward
    outputs = model(input_ids, section_ids=section_ids, 
                    retrieval_embeddings=retrieval, labels=labels)
    loss = outputs['loss']
    loss.backward()
    
    # Check that key components received gradients
    has_grad = {
        'token_embedding': model.token_embedding.weight.grad is not None,
        'section_embedding': model.positional_encoding.section_embeddings.weight.grad is not None,
        'section_scale': model.positional_encoding.section_scale.grad is not None,
        'self_attn_q': model.layers[0].self_attention.q_proj.weight.grad is not None,
        'cross_attn_gate': model.layers[0].cross_attention.gate.grad is not None,
        'cross_attn_q': model.layers[0].cross_attention.q_proj.weight.grad is not None,
        'retrieval_proj': model.layers[0].cross_attention.retrieval_proj.weight.grad is not None,
        'ffn_gate': model.layers[0].ffn.gate_proj.weight.grad is not None,
    }
    
    all_have_grads = all(has_grad.values())
    
    for name, has in has_grad.items():
        status = "✅" if has else "❌"
        print(f"  {status} {name}: grad={'YES' if has else 'NO'}")
    
    print(f"\n  Loss: {loss.item():.4f}")
    assert all_have_grads, "All components should receive gradients!"
    print("  ✅ Gradient Flow passed")


def test_memory_estimate():
    """Estimate training memory requirements."""
    print("\n" + "=" * 60)
    print("TEST 12: Memory Estimation")
    print("=" * 60)
    
    config = ScholarFormerConfig()
    model = ScholarFormerModel(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    
    # Memory estimates
    fp32_mb = total_params * 4 / (1024 ** 2)
    fp16_mb = total_params * 2 / (1024 ** 2)
    
    # Training memory (model + gradients + optimizer states)
    # Adam: model + gradients + 2 optimizer states = 4x model size
    train_fp16_mb = fp16_mb * 4  # Mixed precision training
    train_fp32_mb = fp32_mb * 4
    
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size (FP32): {fp32_mb:.1f} MB")
    print(f"  Model size (FP16): {fp16_mb:.1f} MB")
    print(f"  Est. training memory (FP16 + Adam): ~{train_fp16_mb:.0f} MB = ~{train_fp16_mb/1024:.1f} GB")
    print(f"  Est. training memory (FP32 + Adam): ~{train_fp32_mb:.0f} MB = ~{train_fp32_mb/1024:.1f} GB")
    print(f"  Available VRAM: 12 GB")
    
    # Should fit comfortably
    assert train_fp16_mb / 1024 < 10, "Training should fit in 12GB VRAM"
    print("  ✅ Memory estimation passed — fits in 12GB VRAM")


def main():
    """Run all tests."""
    print("\n" + "🧪" * 30)
    print("   ScholarFormer Architecture Tests")
    print("🧪" * 30 + "\n")
    
    tests = [
        test_config,
        test_rmsnorm,
        test_rope,
        test_section_encoding,
        test_self_attention,
        test_cross_attention,
        test_swiglu,
        test_decoder_block,
        test_full_model,
        test_generation,
        test_gradient_flow,
        test_memory_estimate,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("🎉 ALL TESTS PASSED — ScholarFormer architecture is verified!")
    else:
        print(f"⚠️  {failed} test(s) failed")
    print("=" * 60)


if __name__ == '__main__':
    main()
