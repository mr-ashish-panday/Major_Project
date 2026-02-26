"""
ComparisonAgent — Side-by-side evaluation of ScholarFormer vs Phi-3.

Runs both models on identical test data, collecting:
    - Perplexity (language modeling quality)
    - BLEU (text generation quality)
    - ROUGE-L (text overlap quality)
    - Inference latency
    - Response quality examples

Produces JSON comparison reports for the dashboard.

VRAM Strategy:
    Models are loaded sequentially (one at a time) to fit within 12GB.
    ScholarFormer (~0.5GB) and Phi-3 4-bit (~3.5GB) each fit easily,
    but loading both simultaneously would use ~4GB, which is fine too
    if there's enough headroom.
"""

import os
import json
import math
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from scholarformer.config import ScholarFormerConfig
from scholarformer.model import ScholarFormerModel

logger = logging.getLogger(__name__)


class ComparisonAgent:
    """
    Compare ScholarFormer vs Phi-3 on identical tasks.
    
    Evaluates both models on:
        1. Perplexity on held-out test chunks
        2. BLEU/ROUGE on text continuation tasks
        3. Inference latency
        4. Side-by-side generation examples
    
    Usage:
        comp = ComparisonAgent(config)
        report = comp.run_full_comparison(
            sf_checkpoint="./models/scholarformer/checkpoints/distilled",
            test_chunks=["chunk1...", "chunk2...", ...],
            sample_queries=["What is attention?", ...]
        )
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.output_dir = os.path.join(config.get('logs_dir', './logs'), 'comparisons')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Lazy-loaded metric calculators
        self._bleu = None
        self._rouge = None
    
    @property
    def bleu(self):
        if self._bleu is None:
            import evaluate
            self._bleu = evaluate.load('bleu')
        return self._bleu
    
    @property
    def rouge(self):
        if self._rouge is None:
            import evaluate
            self._rouge = evaluate.load('rouge')
        return self._rouge
    
    # ===================================================================
    # Model Loading
    # ===================================================================
    
    def _load_scholarformer(self, checkpoint_path: str):
        """Load ScholarFormer from checkpoint. Returns (model, tokenizer)."""
        logger.info(f"Loading ScholarFormer from {checkpoint_path}...")
        
        model_file = os.path.join(checkpoint_path, 'model.pt')
        checkpoint = torch.load(model_file, map_location='cpu')
        
        config_dict = checkpoint.get('config', {})
        model_config = ScholarFormerConfig(**{
            k: v for k, v in config_dict.items()
            if k in ScholarFormerConfig.__dataclass_fields__
        })
        
        model = ScholarFormerModel(model_config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)
        model.eval()
        
        # Load tokenizer
        try:
            from scholarformer.tokenizer import ScholarFormerTokenizer
            tokenizer = ScholarFormerTokenizer()
        except Exception:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.get('base_model', 'microsoft/Phi-3-mini-4k-instruct'),
                trust_remote_code=True,
                token=os.environ.get('HF_TOKEN')
            )
        
        params = sum(p.numel() for p in model.parameters())
        vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"ScholarFormer loaded: {params:,} params, {vram_mb:.0f}MB VRAM")
        
        return model, tokenizer
    
    def _load_phi3(self):
        """Load fine-tuned Phi-3 with 4-bit quantization. Returns (model, tokenizer)."""
        logger.info("Loading Phi-3 (4-bit quantized)...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        base_model = self.config.get('base_model', 'microsoft/Phi-3-mini-4k-instruct')
        
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            ),
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN'),
            torch_dtype=torch.float16,
        )
        
        # Load fine-tuned adapter
        model_dir = self.config.get('model_dir', './models')
        adapter_dirs = [d for d in os.listdir(model_dir)
                       if d.startswith('fine_tuned_') and os.path.isdir(os.path.join(model_dir, d))]
        
        if adapter_dirs:
            latest = sorted(adapter_dirs)[-1]
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(
                    model, os.path.join(model_dir, latest)
                )
                logger.info(f"Loaded Phi-3 adapter: {latest}")
            except Exception as e:
                logger.warning(f"No adapter loaded: {e}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        vram_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        logger.info(f"Phi-3 loaded, {vram_mb:.0f}MB VRAM")
        
        return model, tokenizer
    
    @staticmethod
    def _unload_model(model):
        """Free VRAM by deleting a model."""
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===================================================================
    # Metric Calculations
    # ===================================================================
    
    def _calculate_perplexity(self, model, tokenizer, texts: List[str],
                              is_scholarformer: bool = False) -> float:
        """Calculate perplexity on a set of texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        tok = tokenizer.tokenizer if (hasattr(tokenizer, 'tokenizer') and is_scholarformer) else tokenizer
        
        with torch.no_grad():
            for text in texts:
                if not text or len(text.strip()) < 10:
                    continue
                try:
                    inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
                    input_ids = inputs['input_ids'].to(self.device)
                    
                    if is_scholarformer:
                        outputs = model(input_ids)
                        logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                        # Manual cross-entropy
                        shift_logits = logits[:, :-1, :].contiguous()
                        shift_labels = input_ids[:, 1:].contiguous()
                        loss = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                    else:
                        outputs = model(input_ids=input_ids, labels=input_ids)
                        loss = outputs.loss
                    
                    if loss is not None:
                        num_tokens = input_ids.numel()
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                except Exception as e:
                    logger.warning(f"Perplexity error: {e}")
                    continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        return math.exp(min(avg_loss, 100))  # Cap to avoid overflow
    
    def _generate_continuations(self, model, tokenizer, texts: List[str],
                                 max_samples: int = 20,
                                 is_scholarformer: bool = False) -> List[Tuple[str, str]]:
        """Generate text continuations for BLEU/ROUGE evaluation."""
        model.eval()
        results = []
        
        tok = tokenizer.tokenizer if (hasattr(tokenizer, 'tokenizer') and is_scholarformer) else tokenizer
        
        for text in texts[:max_samples]:
            if not text or len(text.strip()) < 50:
                continue
            try:
                # Split: 30% prompt, 70% expected
                split_point = int(len(text) * 0.3)
                prompt = text[:split_point].strip()
                expected = text[split_point:].strip()
                
                if len(prompt) < 20 or len(expected) < 20:
                    continue
                
                inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=256)
                input_ids = inputs['input_ids'].to(self.device)
                
                with torch.no_grad():
                    if is_scholarformer:
                        output = model.generate(
                            input_ids, max_new_tokens=min(100, len(expected) // 4),
                            temperature=0.7, top_k=50
                        )
                    else:
                        output = model.generate(
                            input_ids=input_ids,
                            max_new_tokens=min(100, len(expected) // 4),
                            do_sample=True, temperature=0.7, top_p=0.9,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        )
                
                generated = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
                if generated.strip():
                    results.append((generated.strip(), expected[:500]))
            except Exception as e:
                logger.warning(f"Generation error: {e}")
                continue
        
        return results
    
    def _compute_bleu_rouge(self, pairs: List[Tuple[str, str]]) -> Dict:
        """Calculate BLEU and ROUGE from (prediction, reference) pairs."""
        if not pairs:
            return {'bleu': 0.0, 'rouge_l': 0.0, 'num_pairs': 0}
        
        preds, refs = zip(*pairs)
        
        try:
            bleu_result = self.bleu.compute(
                predictions=list(preds),
                references=[[r] for r in refs]
            )
            bleu_score = bleu_result.get('bleu', 0.0)
        except Exception as e:
            logger.warning(f"BLEU error: {e}")
            bleu_score = 0.0
        
        try:
            rouge_result = self.rouge.compute(
                predictions=list(preds), references=list(refs)
            )
            rouge_score = rouge_result.get('rougeL', 0.0)
        except Exception as e:
            logger.warning(f"ROUGE error: {e}")
            rouge_score = 0.0
        
        return {'bleu': bleu_score, 'rouge_l': rouge_score, 'num_pairs': len(pairs)}
    
    def _measure_latency(self, model, tokenizer, query: str,
                         is_scholarformer: bool = False) -> Tuple[str, float]:
        """Generate a response and measure latency in ms."""
        tok = tokenizer.tokenizer if (hasattr(tokenizer, 'tokenizer') and is_scholarformer) else tokenizer
        
        inputs = tok(query, return_tensors='pt', max_length=256, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        
        # Warmup
        with torch.no_grad():
            if is_scholarformer:
                model.generate(input_ids, max_new_tokens=5, temperature=0.7, top_k=50)
            else:
                model.generate(input_ids=input_ids, max_new_tokens=5,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            if is_scholarformer:
                output = model.generate(input_ids, max_new_tokens=128, temperature=0.7, top_k=50)
            else:
                output = model.generate(
                    input_ids=input_ids, max_new_tokens=128,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        
        response = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip(), latency
    
    # ===================================================================
    # Main Comparison Entry Point
    # ===================================================================
    
    def run_full_comparison(self, sf_checkpoint: str,
                            test_chunks: List[str],
                            sample_queries: Optional[List[str]] = None,
                            sequential: bool = True) -> Dict:
        """
        Run complete head-to-head comparison.
        
        Args:
            sf_checkpoint: Path to ScholarFormer checkpoint directory
            test_chunks:   List of text chunks for Perplexity/BLEU/ROUGE
            sample_queries: Optional queries for side-by-side generation
            sequential:    If True, load models one at a time (saves VRAM)
        
        Returns:
            Full comparison report dict (also saved to JSON)
        """
        if sample_queries is None:
            sample_queries = [
                "What is the attention mechanism in transformers?",
                "Explain how LoRA reduces the number of trainable parameters.",
                "What are the main challenges in fine-tuning large language models?",
                "How does retrieval-augmented generation improve model accuracy?",
                "What is knowledge distillation and why is it useful?",
            ]
        
        logger.info("=" * 60)
        logger.info("  ⚔️  MODEL COMPARISON: ScholarFormer vs Phi-3")
        logger.info("=" * 60)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_samples': len(test_chunks),
            'sample_queries': len(sample_queries),
        }
        
        # ── ScholarFormer Evaluation ──
        logger.info("\n📦 Phase 1: Evaluating ScholarFormer...")
        sf_model, sf_tok = self._load_scholarformer(sf_checkpoint)
        
        logger.info("  Computing perplexity...")
        sf_ppl = self._calculate_perplexity(sf_model, sf_tok, test_chunks, is_scholarformer=True)
        logger.info(f"  ScholarFormer Perplexity: {sf_ppl:.2f}")
        
        logger.info("  Generating continuations for BLEU/ROUGE...")
        sf_pairs = self._generate_continuations(sf_model, sf_tok, test_chunks, is_scholarformer=True)
        sf_scores = self._compute_bleu_rouge(sf_pairs)
        logger.info(f"  ScholarFormer BLEU: {sf_scores['bleu']:.4f}, ROUGE-L: {sf_scores['rouge_l']:.4f}")
        
        logger.info("  Measuring inference latency...")
        sf_examples = []
        sf_latencies = []
        for q in sample_queries:
            resp, lat = self._measure_latency(sf_model, sf_tok, q, is_scholarformer=True)
            sf_examples.append({'query': q, 'response': resp, 'latency_ms': lat})
            sf_latencies.append(lat)
            logger.info(f"    [{lat:.0f}ms] {q[:50]}...")
        
        sf_params = sum(p.numel() for p in sf_model.parameters())
        sf_vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        report['scholarformer'] = {
            'perplexity': sf_ppl,
            'bleu': sf_scores['bleu'],
            'rouge_l': sf_scores['rouge_l'],
            'eval_pairs': sf_scores['num_pairs'],
            'avg_latency_ms': sum(sf_latencies) / len(sf_latencies) if sf_latencies else 0,
            'min_latency_ms': min(sf_latencies) if sf_latencies else 0,
            'max_latency_ms': max(sf_latencies) if sf_latencies else 0,
            'parameters': sf_params,
            'vram_mb': sf_vram,
            'examples': sf_examples,
        }
        
        if sequential:
            self._unload_model(sf_model)
            logger.info("  ScholarFormer unloaded from VRAM")
        
        # ── Phi-3 Evaluation ──
        logger.info("\n🔮 Phase 2: Evaluating Phi-3...")
        phi3_model, phi3_tok = self._load_phi3()
        
        logger.info("  Computing perplexity...")
        phi3_ppl = self._calculate_perplexity(phi3_model, phi3_tok, test_chunks, is_scholarformer=False)
        logger.info(f"  Phi-3 Perplexity: {phi3_ppl:.2f}")
        
        logger.info("  Generating continuations for BLEU/ROUGE...")
        phi3_pairs = self._generate_continuations(phi3_model, phi3_tok, test_chunks, is_scholarformer=False)
        phi3_scores = self._compute_bleu_rouge(phi3_pairs)
        logger.info(f"  Phi-3 BLEU: {phi3_scores['bleu']:.4f}, ROUGE-L: {phi3_scores['rouge_l']:.4f}")
        
        logger.info("  Measuring inference latency...")
        phi3_examples = []
        phi3_latencies = []
        for q in sample_queries:
            resp, lat = self._measure_latency(phi3_model, phi3_tok, q, is_scholarformer=False)
            phi3_examples.append({'query': q, 'response': resp, 'latency_ms': lat})
            phi3_latencies.append(lat)
            logger.info(f"    [{lat:.0f}ms] {q[:50]}...")
        
        phi3_vram = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        report['phi3'] = {
            'perplexity': phi3_ppl,
            'bleu': phi3_scores['bleu'],
            'rouge_l': phi3_scores['rouge_l'],
            'eval_pairs': phi3_scores['num_pairs'],
            'avg_latency_ms': sum(phi3_latencies) / len(phi3_latencies) if phi3_latencies else 0,
            'min_latency_ms': min(phi3_latencies) if phi3_latencies else 0,
            'max_latency_ms': max(phi3_latencies) if phi3_latencies else 0,
            'parameters': '3,821,079,552 (4-bit quantized)',
            'vram_mb': phi3_vram,
            'examples': phi3_examples,
        }
        
        self._unload_model(phi3_model)
        
        # ── Summary ──
        avg_sf_lat = report['scholarformer']['avg_latency_ms']
        avg_phi3_lat = report['phi3']['avg_latency_ms']
        
        report['summary'] = {
            'perplexity_ratio': sf_ppl / phi3_ppl if phi3_ppl > 0 else float('inf'),
            'bleu_ratio': sf_scores['bleu'] / phi3_scores['bleu'] if phi3_scores['bleu'] > 0 else 0,
            'rouge_ratio': sf_scores['rouge_l'] / phi3_scores['rouge_l'] if phi3_scores['rouge_l'] > 0 else 0,
            'speedup': avg_phi3_lat / avg_sf_lat if avg_sf_lat > 0 else 0,
            'param_ratio': f"{sf_params:,} vs 3.8B (31x smaller)",
            'vram_savings': f"{phi3_vram - sf_vram:.0f}MB less",
        }
        
        # ── Save Report ──
        report_path = os.path.join(
            self.output_dir,
            f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # ── Print Summary ──
        logger.info("\n" + "=" * 60)
        logger.info("  📊 COMPARISON RESULTS")
        logger.info("=" * 60)
        logger.info(f"  {'Metric':<20} {'ScholarFormer':>15} {'Phi-3':>15}")
        logger.info(f"  {'-'*50}")
        logger.info(f"  {'Perplexity':<20} {sf_ppl:>15.2f} {phi3_ppl:>15.2f}")
        logger.info(f"  {'BLEU':<20} {sf_scores['bleu']:>15.4f} {phi3_scores['bleu']:>15.4f}")
        logger.info(f"  {'ROUGE-L':<20} {sf_scores['rouge_l']:>15.4f} {phi3_scores['rouge_l']:>15.4f}")
        logger.info(f"  {'Avg Latency (ms)':<20} {avg_sf_lat:>15.0f} {avg_phi3_lat:>15.0f}")
        logger.info(f"  {'Parameters':<20} {sf_params:>15,} {'~3.8B':>15}")
        logger.info(f"\n  Speedup: {report['summary']['speedup']:.1f}x faster (ScholarFormer)")
        logger.info(f"  Report saved: {report_path}")
        
        return report
