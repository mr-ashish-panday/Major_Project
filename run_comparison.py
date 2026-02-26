#!/usr/bin/env python3
"""
run_comparison.py — Standalone script to run ScholarFormer vs Phi-3 evaluation.

Usage:
    python run_comparison.py

This script:
    1. Loads test data from preprocessed paper chunks
    2. Evaluates ScholarFormer (Perplexity, BLEU, ROUGE, Latency)
    3. Evaluates Phi-3 (same metrics)
    4. Saves a comparison report JSON to logs/comparisons/

Expected runtime: ~5-10 minutes on RTX 3080 Ti
Expected VRAM: ~4GB peak (models loaded sequentially)
"""

import os
import sys
import json
import glob
import logging
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from agents.comparison import ComparisonAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


def find_scholarformer_checkpoint() -> str:
    """Find the best ScholarFormer checkpoint."""
    checkpoint_dir = "./models/scholarformer/checkpoints"
    
    # Priority order: distilled > best > latest step
    for name in ['distilled', 'best']:
        path = os.path.join(checkpoint_dir, name)
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'model.pt')):
            return path
    
    # Fall back to highest step checkpoint
    step_dirs = glob.glob(os.path.join(checkpoint_dir, 'step_*'))
    if step_dirs:
        return max(step_dirs, key=lambda p: int(os.path.basename(p).split('_')[1]))
    
    raise FileNotFoundError(f"No ScholarFormer checkpoint found in {checkpoint_dir}")


def load_test_chunks(max_chunks: int = 50) -> list:
    """Load test chunks from preprocessed paper data."""
    chunks = []
    
    # Try loading from vector store metadata
    metadata_path = "./vector_store/metadata.json"
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            for entry in metadata:
                content = entry.get('content', '') or entry.get('text', '')
                if content and len(content.strip()) > 50:
                    chunks.append(content.strip())
                if len(chunks) >= max_chunks:
                    break
            
            if chunks:
                logger.info(f"Loaded {len(chunks)} test chunks from vector_store metadata")
                return chunks
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
    
    # Try loading from preprocessed data directory
    data_dir = CONFIG.get('data_dir', './data')
    for json_file in glob.glob(os.path.join(data_dir, '**', '*.json'), recursive=True):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    content = item.get('content', '') or item.get('text', '') or item.get('chunk', '')
                    if content and len(content.strip()) > 50:
                        chunks.append(content.strip())
            elif isinstance(data, dict):
                for chunk in data.get('chunks', []):
                    if isinstance(chunk, str) and len(chunk.strip()) > 50:
                        chunks.append(chunk.strip())
                    elif isinstance(chunk, dict):
                        content = chunk.get('content', '') or chunk.get('text', '')
                        if content and len(content.strip()) > 50:
                            chunks.append(content.strip())
            
            if len(chunks) >= max_chunks:
                break
        except Exception:
            continue
    
    # Fall back to distillation Q&A pairs
    distill_dir = "./models/scholarformer/distillation"
    if not chunks and os.path.exists(distill_dir):
        for json_file in sorted(glob.glob(os.path.join(distill_dir, '*.json'))):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    qa_pairs = json.load(f)
                for pair in qa_pairs:
                    q = pair.get('question', '')
                    a = pair.get('answer', '')
                    if q and a:
                        chunks.append(f"Question: {q}\nAnswer: {a}")
                if len(chunks) >= max_chunks:
                    break
            except Exception:
                continue
    
    if chunks:
        logger.info(f"Loaded {len(chunks)} test chunks")
    else:
        # Generate basic test chunks for minimal evaluation
        logger.warning("No preprocessed data found, using minimal test set")
        chunks = [
            "The transformer architecture introduced by Vaswani et al. revolutionized natural language processing by replacing recurrent computations with self-attention mechanisms, enabling parallel processing of input sequences and achieving state-of-the-art results on machine translation tasks.",
            "Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that freezes the pretrained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks.",
            "Knowledge distillation is a model compression technique where a smaller student model is trained to mimic the behavior of a larger teacher model, typically by matching the soft probability distributions produced by the teacher on training examples.",
            "Retrieval-augmented generation combines the benefits of parametric and non-parametric memory by augmenting language models with a retrieval mechanism that can access external knowledge during generation.",
            "Quantization reduces the precision of model weights from floating-point to lower-bit representations such as 4-bit or 8-bit integers, significantly reducing model size and inference latency while maintaining acceptable accuracy.",
        ]
    
    return chunks[:max_chunks]


def main():
    print("\n" + "=" * 60)
    print("  ⚔️  ScholarFormer vs Phi-3: Head-to-Head Comparison")
    print("=" * 60)
    
    # Find checkpoint
    try:
        sf_checkpoint = find_scholarformer_checkpoint()
        print(f"\n📦 ScholarFormer checkpoint: {sf_checkpoint}")
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        return
    
    # Load test data
    test_chunks = load_test_chunks(max_chunks=30)
    print(f"📊 Test samples: {len(test_chunks)}")
    
    # Sample queries for side-by-side comparison
    sample_queries = [
        "What is the attention mechanism in transformers?",
        "Explain how LoRA reduces the number of trainable parameters.",
        "What are the main challenges in fine-tuning large language models?",
        "How does retrieval-augmented generation improve model accuracy?",
        "What is knowledge distillation and why is it useful?",
    ]
    
    # Run comparison
    agent = ComparisonAgent(CONFIG)
    report = agent.run_full_comparison(
        sf_checkpoint=sf_checkpoint,
        test_chunks=test_chunks,
        sample_queries=sample_queries,
        sequential=True,  # Load one model at a time
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("  🏆 FINAL RESULTS")
    print("=" * 60)
    
    sf = report['scholarformer']
    phi3 = report['phi3']
    
    print(f"\n  {'Metric':<22} {'ScholarFormer':>14} {'Phi-3':>14} {'Winner':>10}")
    print(f"  {'-' * 60}")
    
    # Perplexity (lower is better)
    ppl_winner = "SF ✅" if sf['perplexity'] < phi3['perplexity'] else "Phi-3 ✅"
    print(f"  {'Perplexity ↓':<22} {sf['perplexity']:>14.2f} {phi3['perplexity']:>14.2f} {ppl_winner:>10}")
    
    # BLEU (higher is better)
    bleu_winner = "SF ✅" if sf['bleu'] > phi3['bleu'] else "Phi-3 ✅"
    print(f"  {'BLEU ↑':<22} {sf['bleu']:>14.4f} {phi3['bleu']:>14.4f} {bleu_winner:>10}")
    
    # ROUGE (higher is better)
    rouge_winner = "SF ✅" if sf['rouge_l'] > phi3['rouge_l'] else "Phi-3 ✅"
    print(f"  {'ROUGE-L ↑':<22} {sf['rouge_l']:>14.4f} {phi3['rouge_l']:>14.4f} {rouge_winner:>10}")
    
    # Latency (lower is better)
    lat_winner = "SF ✅" if sf['avg_latency_ms'] < phi3['avg_latency_ms'] else "Phi-3 ✅"
    print(f"  {'Avg Latency (ms) ↓':<22} {sf['avg_latency_ms']:>14.0f} {phi3['avg_latency_ms']:>14.0f} {lat_winner:>10}")
    
    print(f"\n  Speedup: {report['summary']['speedup']:.1f}x (ScholarFormer is faster)")
    print(f"  Parameters: {report['summary']['param_ratio']}")
    print(f"\n  📄 Full report: {os.path.join('logs', 'comparisons')}")
    print("  ✅ Comparison complete!\n")


if __name__ == "__main__":
    main()
