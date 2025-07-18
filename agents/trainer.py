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
        self.model = AutoModelForCausalLM.from_pretrained(self.config['base_model'], token=os.environ.get('HF_TOKEN'))
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'], token=os.environ.get('HF_TOKEN'))

    def train(self, validated: List[Dict]) -> str:
        texts = [chunk for paper in validated for chunk in paper['chunks']]
        if not texts:
            raise ValueError("No data for training.")
        dataset = Dataset.from_dict({'text': texts})
        tokenized = dataset.map(lambda x: self.tokenizer(x['text'], truncation=True, padding='max_length', max_length=512), batched=True, remove_columns=['text'])

        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'], lora_dropout=0.05, bias='none')
        self.model = get_peft_model(self.model, lora_config)

        training_args = TrainingArguments(
            output_dir=self.config['model_dir'],
            num_train_epochs=self.config['training_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            fp16=True if torch.cuda.is_available() else False,
            save_steps=500,
            logging_steps=100,
            evaluation_strategy="no"  # Add eval if validation set expanded
        )

        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized)
        trainer.train()

        model_path = os.path.join(self.config['model_dir'], f"fine_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        trainer.save_model(model_path)
        logger.info(f"Model fine-tuned and saved at {model_path}.")
        return model_path