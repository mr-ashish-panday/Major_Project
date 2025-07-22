from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import Dataset
import torch
import logging
import os
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

class TrainerAgent:
    def __init__(self, config: Dict):
        self.config = config
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(self.config['base_model'], token=os.environ.get('HF_TOKEN'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'], token=os.environ.get('HF_TOKEN'))
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set padding token for GPT-2

    def train(self, validated: List[Dict]) -> str:
        texts = [chunk for paper in validated for chunk in paper['chunks']]
        if not texts:
            raise ValueError("No data for training.")
        dataset = Dataset.from_dict({'text': texts})
        
        def tokenize_function(x):
            tokenized = self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512)
            tokenized['labels'] = tokenized['input_ids'].copy()
            return tokenized
        
        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=['text'])

        # LoRA config adjusted for GPT-2
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=['c_attn', 'c_proj'], lora_dropout=0.05, bias='none')
        self.model = get_peft_model(self.model, lora_config)

        training_args = TrainingArguments(
            output_dir=self.config['model_dir'],
            num_train_epochs=self.config['training_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            fp16=True,  # Enable for GPU acceleration; automatically detects CUDA
            save_steps=500,
            logging_steps=100,
            eval_strategy="no"
        )

        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized)
        trainer.train()

        model_path = os.path.join(self.config['model_dir'], f"fine_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        trainer.save_model(model_path)
        logger.info(f"Model fine-tuned and saved at {model_path}.")
        return model_path