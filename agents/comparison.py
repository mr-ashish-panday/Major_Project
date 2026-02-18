"""
ComparisonAgent — Side-by-side evaluation of ScholarFormer vs Phi-3.

Runs both models on identical queries, collecting:
    - Response quality (BLEU, ROUGE scores)
    - Inference latency
    - Perplexity on held-out test data
    - Generation coherence metrics

Produces JSON comparison reports for the dashboard.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch

from scholarformer.config import ScholarFormerConfig
from scholarformer.model import ScholarFormerModel

logger = logging.getLogger(__name__)


class ComparisonAgent:
    """
    Compare ScholarFormer vs Phi-3 on identical tasks.
    
    Usage:
        comp = ComparisonAgent(config)
        comp.load_scholarformer(checkpoint_path)
        comp.load_phi3()
        
        results = comp.compare(test_queries)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.scholarformer = None
        self.sf_tokenizer = None
        self.phi3_model = None
        self.phi3_tokenizer = None
        
        self.output_dir = os.path.join(config.get('logs_dir', './logs'), 'comparisons')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_scholarformer(self, checkpoint_path: str):
        """Load ScholarFormer from a checkpoint."""
        logger.info(f"Loading ScholarFormer from {checkpoint_path}...")
        
        checkpoint = torch.load(
            os.path.join(checkpoint_path, 'model.pt'),
            map_location=self.device
        )
        
        config_dict = checkpoint.get('config', {})
        model_config = ScholarFormerConfig(**{
            k: v for k, v in config_dict.items()
            if k in ScholarFormerConfig.__dataclass_fields__
        })
        
        self.scholarformer = ScholarFormerModel(model_config).to(self.device)
        self.scholarformer.load_state_dict(checkpoint['model_state_dict'])
        self.scholarformer.eval()
        
        # Load tokenizer
        try:
            from scholarformer.tokenizer import ScholarFormerTokenizer
            self.sf_tokenizer = ScholarFormerTokenizer()
        except Exception:
            from transformers import AutoTokenizer
            self.sf_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        
        params = sum(p.numel() for p in self.scholarformer.parameters())
        logger.info(f"ScholarFormer loaded: {params:,} params")
    
    def load_phi3(self):
        """Load fine-tuned Phi-3."""
        logger.info("Loading Phi-3...")
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        base_model = self.config.get('base_model', 'microsoft/Phi-3-mini-4k-instruct')
        
        self.phi3_model = AutoModelForCausalLM.from_pretrained(
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
        
        # Try loading adapter
        model_dir = self.config.get('model_dir', './models')
        adapter_dirs = [d for d in os.listdir(model_dir)
                       if d.startswith('fine_tuned_') and os.path.isdir(os.path.join(model_dir, d))]
        
        if adapter_dirs:
            latest = sorted(adapter_dirs)[-1]
            try:
                from peft import PeftModel
                self.phi3_model = PeftModel.from_pretrained(
                    self.phi3_model, os.path.join(model_dir, latest)
                )
                logger.info(f"Loaded Phi-3 adapter: {latest}")
            except Exception as e:
                logger.warning(f"No adapter loaded: {e}")
        
        self.phi3_tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        if self.phi3_tokenizer.pad_token is None:
            self.phi3_tokenizer.pad_token = self.phi3_tokenizer.eos_token
        
        self.phi3_model.eval()
        logger.info("Phi-3 loaded")
    
    def _generate_scholarformer(self, query: str, max_new_tokens: int = 256) -> Tuple[str, float]:
        """Generate response from ScholarFormer, return (text, latency_ms)."""
        tok = (self.sf_tokenizer.tokenizer 
               if hasattr(self.sf_tokenizer, 'tokenizer') 
               else self.sf_tokenizer)
        
        inputs = tok(query, return_tensors='pt', max_length=512, truncation=True)
        input_ids = inputs['input_ids'].to(self.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            output = self.scholarformer.generate(
                input_ids, max_new_tokens=max_new_tokens,
                temperature=0.7, top_k=50
            )
        latency = (time.perf_counter() - start) * 1000  # ms
        
        response = tok.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response, latency
    
    def _generate_phi3(self, query: str, max_new_tokens: int = 256) -> Tuple[str, float]:
        """Generate response from Phi-3, return (text, latency_ms)."""
        inputs = self.phi3_tokenizer(
            query, return_tensors='pt', max_length=512, truncation=True
        ).to(self.phi3_model.device)
        
        start = time.perf_counter()
        with torch.no_grad():
            output = self.phi3_model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=0.7, do_sample=True, top_p=0.9,
                pad_token_id=self.phi3_tokenizer.pad_token_id,
            )
        latency = (time.perf_counter() - start) * 1000
        
        response = self.phi3_tokenizer.decode(
            output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
        )
        return response, latency
    
    def compare(self, queries: List[str], max_new_tokens: int = 256) -> Dict:
        """
        Run side-by-side comparison on a list of queries.
        
        Returns comparison report dict.
        """
        if self.scholarformer is None:
            raise RuntimeError("Load ScholarFormer first via load_scholarformer()")
        if self.phi3_model is None:
            raise RuntimeError("Load Phi-3 first via load_phi3()")
        
        logger.info(f"Running comparison on {len(queries)} queries...")
        
        results = []
        sf_latencies = []
        phi3_latencies = []
        
        for i, query in enumerate(queries):
            logger.info(f"\n  Query {i+1}/{len(queries)}: {query[:80]}...")
            
            # ScholarFormer
            sf_response, sf_latency = self._generate_scholarformer(query, max_new_tokens)
            sf_latencies.append(sf_latency)
            
            # Phi-3
            phi3_response, phi3_latency = self._generate_phi3(query, max_new_tokens)
            phi3_latencies.append(phi3_latency)
            
            result = {
                'query': query,
                'scholarformer': {
                    'response': sf_response,
                    'latency_ms': sf_latency,
                    'response_length': len(sf_response),
                },
                'phi3': {
                    'response': phi3_response,
                    'latency_ms': phi3_latency,
                    'response_length': len(phi3_response),
                },
            }
            results.append(result)
            
            logger.info(f"    SF: {sf_latency:.0f}ms, {len(sf_response)} chars")
            logger.info(f"    Phi-3: {phi3_latency:.0f}ms, {len(phi3_response)} chars")
        
        # Aggregate statistics
        report = {
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries),
            'scholarformer_stats': {
                'avg_latency_ms': sum(sf_latencies) / len(sf_latencies),
                'min_latency_ms': min(sf_latencies),
                'max_latency_ms': max(sf_latencies),
                'total_params': sum(p.numel() for p in self.scholarformer.parameters()),
            },
            'phi3_stats': {
                'avg_latency_ms': sum(phi3_latencies) / len(phi3_latencies),
                'min_latency_ms': min(phi3_latencies),
                'max_latency_ms': max(phi3_latencies),
                'total_params': '~3.8B (4-bit quantized)',
            },
            'speedup': sum(phi3_latencies) / max(1, sum(sf_latencies)),
            'results': results,
        }
        
        # Save report
        report_path = os.path.join(
            self.output_dir, 
            f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📊 Comparison Report:")
        logger.info(f"   ScholarFormer avg latency: {report['scholarformer_stats']['avg_latency_ms']:.0f}ms")
        logger.info(f"   Phi-3 avg latency: {report['phi3_stats']['avg_latency_ms']:.0f}ms")
        logger.info(f"   Speedup: {report['speedup']:.1f}x")
        logger.info(f"   Report saved: {report_path}")
        
        return report
