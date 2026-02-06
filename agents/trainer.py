"""
TrainerAgent - Fine-tunes LLMs using QLoRA for memory efficiency.
Part of the ScholarMind multi-agent system.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import SFTTrainer

logger = logging.getLogger(__name__)


class TrainerAgent:
    """
    Fine-tunes LLMs on research papers using QLoRA.
    Optimized for 12GB VRAM with 4-bit quantization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_base_model()
    
    def _load_base_model(self) -> None:
        """Load the base model with 4-bit quantization for QLoRA."""
        logger.info("Loading base model for training...")
        
        # Clear CUDA cache to free up memory from previous agents (embedding models)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        base_model_name = self.config.get('base_model', 'microsoft/Phi-3-mini-4k-instruct')
        
        # 4-bit quantization config for QLoRA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN'),
            torch_dtype=torch.float16
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            token=os.environ.get('HF_TOKEN')
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.get('lora_r', 16),
            lora_alpha=self.config.get('lora_alpha', 32),
            lora_dropout=self.config.get('lora_dropout', 0.05),
            target_modules=self.config.get('lora_target_modules', 
                ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']),
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    def _prepare_dataset(self, validated: List[Dict]) -> Dataset:
        """Prepare training dataset from processed papers."""
        texts = []
        
        for paper in validated:
            chunks = paper.get('chunks', [])
            metadata = paper.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            
            for chunk in chunks:
                if len(chunk.strip()) < 100:  # Skip very short chunks
                    continue
                
                # Format as instruction-following data
                text = f"### Research Content from: {title}\n\n{chunk}\n\n### End of Content"
                texts.append(text)
        
        if not texts:
            raise ValueError("No valid training data found.")
        
        logger.info(f"Prepared {len(texts)} training samples")
        return Dataset.from_dict({'text': texts})
    
    def train(self, validated: List[Dict]) -> str:
        """
        Fine-tune the model on preprocessed papers.
        
        Args:
            validated: List of validated/preprocessed papers
            
        Returns:
            Path to saved fine-tuned model
        """
        logger.info("Preparing training data...")
        dataset = self._prepare_dataset(validated)
        
        # Training arguments optimized for 12GB VRAM
        output_dir = self.config.get('model_dir', './models')
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get('training_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 2),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            learning_rate=self.config.get('learning_rate', 2e-4),
            warmup_ratio=self.config.get('warmup_ratio', 0.1),
            fp16=True,
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            report_to="none",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
        )
        
        # Use SFTTrainer for supervised fine-tuning
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            dataset_text_field="text",
            max_seq_length=self.config.get('max_context_length', 2048),
            packing=False,  # Disabled to prevent OOM on 12GB VRAM
        )
        
        logger.info("Starting training...")
        trainer.train()
        
        # Save the fine-tuned model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(output_dir, f"fine_tuned_{timestamp}")
        
        logger.info(f"Saving model to {model_path}")
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model fine-tuned and saved at {model_path}")
        return model_path
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        
        return {
            "status": "loaded",
            "base_model": self.config.get('base_model', 'unknown'),
            "trainable_params": trainable,
            "total_params": total,
            "trainable_percent": 100 * trainable / total
        }