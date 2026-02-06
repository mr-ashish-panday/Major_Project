"""
EvaluatorAgent - Evaluates fine-tuned model performance.
Part of the ScholarMind multi-agent system.
"""

import os
import logging
import math
from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import evaluate

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    """
    Evaluates model performance using multiple metrics:
    - Perplexity (language modeling quality)
    - BLEU (text similarity)
    - ROUGE (summarization quality)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.bleu = evaluate.load('bleu')
        self.rouge = evaluate.load('rouge')
    
    def _load_model(self, model_path: str):
        """Load the fine-tuned model for evaluation."""
        logger.info(f"Loading model from {model_path} for evaluation...")
        
        base_model_name = self.config.get('base_model', 'microsoft/Phi-3-mini-4k-instruct')
        
        # Quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        
        # Load LoRA weights if they exist
        if os.path.exists(model_path):
            model = PeftModel.from_pretrained(base_model, model_path)
        else:
            model = base_model
        
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _calculate_perplexity(self, model, tokenizer, texts: List[str]) -> float:
        """Calculate perplexity on a set of texts."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in texts:
                if not text or len(text.strip()) < 10:
                    continue
                
                try:
                    inputs = tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512
                    ).to(model.device)
                    
                    outputs = model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss
                    
                    if loss is not None:
                        num_tokens = inputs['input_ids'].numel()
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                
                except Exception as e:
                    logger.warning(f"Error calculating loss for text: {e}")
                    continue
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def _generate_responses(self, model, tokenizer, texts: List[str], max_samples: int = 20) -> List[tuple]:
        """
        Generate responses for evaluation using proper continuation methodology.
        
        FIXED: Previously compared output against input (wrong!).
        Now: Split text into prompt (30%) and expected continuation (70%),
        then compare model's continuation against expected continuation.
        
        Returns:
            List of (prediction, expected) tuples
        """
        model.eval()
        results = []
        
        for text in texts[:max_samples]:
            if not text or len(text.strip()) < 50:  # Need enough text to split
                continue
            
            try:
                # FIXED: Split text into prompt and expected continuation
                # Use first 30% as prompt, remaining 70% as expected output
                split_point = int(len(text) * 0.3)
                prompt = text[:split_point].strip()
                expected = text[split_point:].strip()
                
                # Skip if either part is too short
                if len(prompt) < 20 or len(expected) < 20:
                    continue
                
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(model.device)
                
                # Generate continuation
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=min(150, len(expected) // 2),  # Match expected length roughly
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                    )
                
                # Get only the generated part (exclude prompt)
                generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the beginning
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                
                if generated:
                    results.append((generated, expected))
                
            except Exception as e:
                logger.warning(f"Error generating response: {e}")
                continue
        
        return results
    
    def evaluate(self, model_path: str, test_data: List[str]) -> Dict:
        """
        Evaluate the fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
            test_data: List of text samples for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Starting model evaluation...")
        
        if not test_data:
            logger.warning("No test data provided.")
            return {
                'perplexity': float('inf'),
                'bleu': 0.0,
                'rouge': 0.0,
                'num_samples': 0
            }
        
        # Load model
        model, tokenizer = self._load_model(model_path)
        
        # Calculate perplexity
        logger.info("Calculating perplexity...")
        sample_size = min(len(test_data), self.config.get('eval_sample_size', 50))
        perplexity = self._calculate_perplexity(model, tokenizer, test_data[:sample_size])
        logger.info(f"Perplexity: {perplexity:.4f}")
        
        # Generate responses for BLEU/ROUGE using proper continuation evaluation
        logger.info("Generating responses for BLEU/ROUGE evaluation...")
        results = self._generate_responses(model, tokenizer, test_data, max_samples=20)
        
        if results:
            # Unpack (prediction, expected) tuples - now properly compared!
            preds, refs = zip(*results)
            refs_formatted = [[r] for r in refs]  # BLEU expects list of references
            
            logger.info(f"Evaluating {len(preds)} valid prediction/reference pairs")
            
            # Calculate BLEU
            try:
                bleu_result = self.bleu.compute(predictions=list(preds), references=refs_formatted)
                bleu_score = bleu_result.get('bleu', 0.0)
            except Exception as e:
                logger.warning(f"BLEU calculation failed: {e}")
                bleu_score = 0.0
            
            # Calculate ROUGE
            try:
                rouge_result = self.rouge.compute(predictions=list(preds), references=list(refs))
                rouge_score = rouge_result.get('rougeL', 0.0)
            except Exception as e:
                logger.warning(f"ROUGE calculation failed: {e}")
                rouge_score = 0.0
        else:
            bleu_score = 0.0
            rouge_score = 0.0
            logger.warning("No valid prediction/reference pairs generated")
        
        logger.info(f"BLEU: {bleu_score:.4f}, ROUGE-L: {rouge_score:.4f}")
        
        metrics = {
            'perplexity': perplexity,
            'bleu': bleu_score,
            'rouge': rouge_score,
            'num_samples': len(test_data),
            'eval_pairs': len(results) if results else 0
        }
        
        logger.info(f"Evaluation complete: {metrics}")
        return metrics