"""
CustomTrainerAgent — Two-stage training pipeline for ScholarFormer.

Stage 1: General Pretraining on WikiText-103 (~100M tokens)
    - Trains ALL weights from random initialization
    - Model learns English grammar, syntax, and general knowledge
    - Uses standard causal language modeling objective

Stage 2: Domain Adaptation on research papers (~1,588 papers)
    - Fine-tunes on preprocessed paper chunks
    - Enables section-aware encoding
    - Cross-attention can optionally attend to FAISS retrieval vectors
    - Model specializes in academic paper understanding

Training features:
    - Mixed precision (FP16) for memory efficiency
    - Gradient checkpointing for larger batch sizes
    - Cosine learning rate schedule with warmup
    - Checkpoint save/resume
    - Training metrics logging
"""

import os
import json
import time
import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as cuda_autocast

from scholarformer.config import ScholarFormerConfig
from scholarformer.model import ScholarFormerModel

logger = logging.getLogger(__name__)


# ==============================================================================
# Datasets
# ==============================================================================

class WikiTextDataset(Dataset):
    """Dataset for WikiText-103 pretraining (Stage 1).
    
    Downloads and tokenizes WikiText-103 from HuggingFace datasets.
    Creates fixed-length chunks for efficient causal LM training.
    """
    
    def __init__(self, tokenizer, max_length: int = 1024, split: str = 'train'):
        """
        Args:
            tokenizer: ScholarFormerTokenizer or any HF-compatible tokenizer
            max_length: Sequence length for training chunks
            split: 'train', 'validation', or 'test'
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Check for cached tokenized chunks first
        cache_dir = './data/wikitext_cache'
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'wikitext103_chunks_{max_length}.pt')
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached tokenized chunks from {cache_path}...")
            self.chunks = torch.load(cache_path, weights_only=True)
            logger.info(f"Loaded {len(self.chunks):,} cached chunks of {max_length} tokens "
                         f"({len(self.chunks) * max_length:,} total tokens)")
            return
        
        logger.info(f"Loading WikiText-103 ({split} split)...")
        
        # Try loading from HuggingFace datasets
        try:
            from datasets import load_dataset
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split=split)
        except Exception as e:
            logger.warning(f"Failed to load from HuggingFace: {e}")
            logger.info("Trying offline cache...")
            from datasets import load_dataset
            dataset = load_dataset(
                'wikitext', 'wikitext-103-raw-v1', split=split,
                cache_dir=cache_dir
            )
        
        # Get the raw tokenizer handle
        tok = tokenizer.tokenizer if hasattr(tokenizer, 'tokenizer') else tokenizer
        
        # Filter out short/empty lines, cap at 500K for speed
        # 500K lines ≈ 60M tokens, more than enough for 122M param model
        MAX_LINES = 500_000
        lines = [t for t in dataset['text'] if len(t.strip()) > 10]
        if len(lines) > MAX_LINES:
            logger.info(f"Capping dataset from {len(lines):,} to {MAX_LINES:,} lines")
            lines = lines[:MAX_LINES]
        
        logger.info(f"Tokenizing {len(lines):,} lines using fast batch mode...")
        
        # Tokenize in batches using list-of-strings (Rust fast tokenizer)
        all_token_ids = []
        batch_size = 5000  # Large batches for fast tokenizer parallelism
        
        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            # Pass list of strings → fast tokenizer uses Rust parallelism
            encoded = tok(batch, truncation=False, return_attention_mask=False,
                         add_special_tokens=False)
            
            for ids in encoded['input_ids']:
                all_token_ids.extend(ids)
            
            progress = min(i + batch_size, len(lines))
            if (i // batch_size) % 10 == 0:
                logger.info(f"  Tokenized {progress:,}/{len(lines):,} lines "
                           f"({len(all_token_ids):,} tokens so far)")
        
        logger.info(f"Tokenization complete: {len(all_token_ids):,} total tokens")
        all_tokens = torch.tensor(all_token_ids, dtype=torch.long)
        
        # Create fixed-length chunks (non-overlapping)
        num_chunks = len(all_tokens) // max_length
        all_tokens = all_tokens[:num_chunks * max_length]
        self.chunks = all_tokens.view(num_chunks, max_length)
        
        # Cache to disk for future runs
        torch.save(self.chunks, cache_path)
        logger.info(f"Cached {num_chunks:,} chunks to {cache_path} (future runs load in ~5s)")
        
        logger.info(f"WikiText-103 ready: {len(all_tokens):,} tokens → "
                     f"{num_chunks:,} chunks of {max_length} tokens")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        input_ids = self.chunks[idx]
        # For causal LM, labels = input_ids (shifted internally by the model)
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
        }


class PaperChunkDataset(Dataset):
    """Dataset for research paper fine-tuning (Stage 2).
    
    Uses the same preprocessed paper chunks from the existing pipeline,
    with section-aware encoding for the novel positional embedding.
    """
    
    def __init__(self, validated_papers: List[Dict], tokenizer, 
                 max_length: int = 1024):
        """
        Args:
            validated_papers: List of validated paper dicts with 'chunks' and 'metadata'
            tokenizer: ScholarFormerTokenizer
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.samples = []
        
        section_map = {
            'abstract': 0, 'introduction': 1, 'methods': 2,
            'results': 3, 'discussion': 4, 'conclusion': 5, 'other': 6
        }
        
        for paper in validated_papers:
            chunks = paper.get('chunks', [])
            metadata = paper.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            
            for chunk in chunks:
                text = chunk if isinstance(chunk, str) else chunk.get('text', '')
                if len(text.strip()) < 50:
                    continue
                
                # Detect section from chunk content
                section_name = self._detect_section(text)
                section_id = section_map.get(section_name, 6)
                
                # Format with title context
                formatted = f"Research: {title}\n\n{text}"
                
                # Tokenize
                if hasattr(tokenizer, 'tokenizer'):
                    encoded = tokenizer.tokenizer(
                        formatted, max_length=max_length,
                        truncation=True, return_tensors='pt'
                    )
                else:
                    encoded = tokenizer(
                        formatted, max_length=max_length,
                        truncation=True, return_tensors='pt'
                    )
                
                input_ids = encoded['input_ids'].squeeze(0)
                seq_len = input_ids.shape[0]
                
                # Create section IDs tensor
                section_ids = torch.full((seq_len,), section_id, dtype=torch.long)
                
                self.samples.append({
                    'input_ids': input_ids,
                    'labels': input_ids.clone(),
                    'section_ids': section_ids,
                })
        
        logger.info(f"Paper dataset: {len(self.samples)} samples from "
                     f"{len(validated_papers)} papers")
    
    def _detect_section(self, text: str) -> str:
        """Simple regex-based section detection."""
        import re
        text_lower = text[:300].lower()
        
        patterns = {
            'abstract': r'\b(abstract)\b',
            'introduction': r'\b(introduction)\b',
            'methods': r'\b(method|methodology|approach|experimental setup)\b',
            'results': r'\b(results?|experiments?|findings)\b',
            'discussion': r'\b(discussion|analysis)\b',
            'conclusion': r'\b(conclusion|summary|future work)\b',
        }
        
        for section, pattern in patterns.items():
            if re.search(pattern, text_lower):
                return section
        return 'other'
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function that handles variable-length sequences."""
    # Find max length in this batch
    max_len = max(item['input_ids'].shape[0] for item in batch)
    
    input_ids_padded = []
    labels_padded = []
    section_ids_padded = []
    attention_masks = []
    has_sections = 'section_ids' in batch[0]
    
    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        # Pad input_ids with 0 (will be masked)
        input_ids_padded.append(
            torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        )
        
        # Pad labels with -100 (ignored in loss)
        labels_padded.append(
            torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        
        # Attention mask
        attention_masks.append(
            torch.cat([torch.ones(seq_len, dtype=torch.long), 
                       torch.zeros(pad_len, dtype=torch.long)])
        )
        
        # Section IDs (if present)
        if has_sections:
            section_ids_padded.append(
                torch.cat([item['section_ids'], 
                          torch.full((pad_len,), 6, dtype=torch.long)])  # 6 = 'other'
            )
    
    result = {
        'input_ids': torch.stack(input_ids_padded),
        'labels': torch.stack(labels_padded),
        'attention_mask': torch.stack(attention_masks),
    }
    
    if has_sections:
        result['section_ids'] = torch.stack(section_ids_padded)
    
    return result


# ==============================================================================
# Training Pipeline
# ==============================================================================

class CustomTrainerAgent:
    """
    Two-stage training pipeline for ScholarFormer.
    
    Stage 1: Pretrain on WikiText-103 (learn English)
    Stage 2: Fine-tune on research papers (learn domain)
    
    Usage:
        trainer = CustomTrainerAgent(config)
        
        # Stage 1: Pretrain
        trainer.pretrain(num_epochs=5)
        
        # Stage 2: Fine-tune
        trainer.finetune(validated_papers, num_epochs=10)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.model_config = ScholarFormerConfig(
            hidden_dim=config.get('sf_hidden_dim', 768),
            num_layers=config.get('sf_num_layers', 12),
            num_heads=config.get('sf_num_heads', 12),
            num_cross_attn_heads=config.get('sf_cross_attn_heads', 4),
            vocab_size=config.get('sf_vocab_size', 32000),
            max_seq_len=config.get('sf_max_seq_len', 1024),
            ffn_intermediate_dim=config.get('sf_ffn_dim', 2048),
            dropout=config.get('sf_dropout', 0.1),
            use_gradient_checkpointing=True,
        )
        
        # Initialize model
        logger.info(f"\n{self.model_config.summary()}")
        self.model = ScholarFormerModel(self.model_config).to(self.device)
        
        # Initialize tokenizer (lazy import to avoid HF version issues)
        self._tokenizer = None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
        
        # Output directories
        self.model_dir = os.path.join(config.get('model_dir', './models'), 'scholarformer')
        self.checkpoint_dir = os.path.join(self.model_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger.info(f"CustomTrainerAgent initialized on {self.device}")
        logger.info(f"Model dir: {self.model_dir}")
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            try:
                from scholarformer.tokenizer import ScholarFormerTokenizer
                self._tokenizer = ScholarFormerTokenizer()
            except Exception as e:
                logger.warning(f"ScholarFormerTokenizer failed: {e}, using GPT-2 fallback")
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained('gpt2')
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer
    
    def _create_optimizer(self, lr: float, weight_decay: float = 0.01) -> torch.optim.Optimizer:
        """Create AdamW optimizer with weight decay on non-bias/norm params."""
        # Separate parameters: decay vs no-decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'norm' in name or 'bias' in name or 'scale' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ], lr=lr, betas=(0.9, 0.95), eps=1e-8)
        
        return optimizer
    
    def _create_scheduler(self, optimizer, num_training_steps: int, 
                           warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        """Cosine learning rate schedule with linear warmup."""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # Min 10% of peak LR
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _training_loop(self, dataloader: DataLoader, optimizer, scheduler,
                        scaler: GradScaler, num_epochs: int, 
                        gradient_accumulation_steps: int = 4,
                        stage_name: str = "Training",
                        log_interval: int = 50,
                        save_interval: int = 500,
                        retrieval_embeddings: Optional[torch.Tensor] = None) -> List[Dict]:
        """
        Core training loop shared between pretraining and fine-tuning.
        
        Returns list of training metrics per epoch.
        """
        self.model.train()
        metrics_history = []
        
        total_steps_per_epoch = len(dataloader)
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_tokens = 0
            step_losses = []
            
            logger.info(f"\n{'='*60}")
            logger.info(f"[{stage_name}] Epoch {epoch + 1}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                section_ids = batch.get('section_ids')
                if section_ids is not None:
                    section_ids = section_ids.to(self.device)
                
                # Count real tokens (not padding)
                num_tokens = (labels != -100).sum().item()
                
                # Forward pass with mixed precision
                with cuda_autocast(dtype=torch.float16):
                    outputs = self.model(
                        input_ids,
                        section_ids=section_ids,
                        retrieval_embeddings=retrieval_embeddings,
                        labels=labels
                    )
                    loss = outputs['loss'] / gradient_accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Accumulate
                step_loss = loss.item() * gradient_accumulation_steps
                epoch_loss += step_loss
                epoch_tokens += num_tokens
                step_losses.append(step_loss)
                
                # Optimizer step (every N accumulation steps)
                if (step + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1
                
                # Logging
                if (step + 1) % log_interval == 0:
                    avg_loss = sum(step_losses[-log_interval:]) / len(step_losses[-log_interval:])
                    lr = scheduler.get_last_lr()[0]
                    tokens_per_sec = epoch_tokens / (time.time() - epoch_start)
                    
                    logger.info(
                        f"  Step {step + 1}/{total_steps_per_epoch} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Tokens/s: {tokens_per_sec:.0f} | "
                        f"Global Step: {self.global_step}"
                    )
                
                # Save checkpoint
                if self.global_step > 0 and self.global_step % save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}", stage_name)
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / max(1, len(dataloader))
            epoch_time = time.time() - epoch_start
            
            epoch_metrics = {
                'epoch': epoch + 1,
                'stage': stage_name,
                'avg_loss': avg_epoch_loss,
                'tokens_processed': epoch_tokens,
                'time_seconds': epoch_time,
                'learning_rate': scheduler.get_last_lr()[0],
                'global_step': self.global_step,
            }
            metrics_history.append(epoch_metrics)
            
            logger.info(f"\n  📊 Epoch {epoch + 1} complete:")
            logger.info(f"     Avg Loss: {avg_epoch_loss:.4f}")
            logger.info(f"     Tokens: {epoch_tokens:,}")
            logger.info(f"     Time: {epoch_time:.1f}s")
            logger.info(f"     Tokens/s: {epoch_tokens / epoch_time:.0f}")
            
            # Save best model
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self._save_checkpoint("best", stage_name)
                logger.info(f"     🏆 New best loss: {avg_epoch_loss:.4f}")
        
        return metrics_history
    
    # ==========================================================================
    # Stage 1: General Pretraining
    # ==========================================================================
    
    def pretrain(self, num_epochs: int = 3, batch_size: int = 4,
                 learning_rate: float = 3e-4, 
                 gradient_accumulation_steps: int = 8) -> Dict:
        """
        Stage 1: Pretrain ScholarFormer on WikiText-103.
        
        This teaches the model English language, grammar, and general knowledge
        from scratch (random initialization → language model).
        
        Args:
            num_epochs: Number of passes over WikiText-103
            batch_size: Per-device batch size
            learning_rate: Peak learning rate
            gradient_accumulation_steps: Accumulation steps for effective batch size
        
        Returns:
            Dict with training metrics
        """
        logger.info("=" * 60)
        logger.info("🚀 STAGE 1: General Pretraining on WikiText-103")
        logger.info("=" * 60)
        
        # Clear VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load WikiText-103 dataset
        dataset = WikiTextDataset(
            self.tokenizer, 
            max_length=self.model_config.max_seq_len,
            split='train'
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Parallel data loading to keep GPU fed
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        
        # Setup training
        total_steps = (len(dataloader) // gradient_accumulation_steps) * num_epochs
        warmup_steps = int(0.05 * total_steps)
        
        optimizer = self._create_optimizer(lr=learning_rate)
        scheduler = self._create_scheduler(optimizer, total_steps, warmup_steps)
        scaler = GradScaler()
        
        logger.info(f"Training samples: {len(dataset):,}")
        logger.info(f"Batch size: {batch_size} × {gradient_accumulation_steps} accum = {batch_size * gradient_accumulation_steps} effective")
        logger.info(f"Total steps: {total_steps:,}")
        logger.info(f"Warmup steps: {warmup_steps:,}")
        logger.info(f"Learning rate: {learning_rate}")
        
        # Train!
        start_time = time.time()
        metrics = self._training_loop(
            dataloader, optimizer, scheduler, scaler,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            stage_name="Pretrain",
            log_interval=100,
            save_interval=1000,
        )
        
        total_time = time.time() - start_time
        
        # Save final pretrained model
        self._save_checkpoint("pretrained_final", "Pretrain")
        
        result = {
            'stage': 'pretrain',
            'epochs': num_epochs,
            'total_time_seconds': total_time,
            'total_time_human': f"{total_time / 3600:.1f} hours",
            'final_loss': metrics[-1]['avg_loss'] if metrics else None,
            'metrics_history': metrics,
            'model_path': os.path.join(self.checkpoint_dir, 'pretrained_final'),
        }
        
        self.training_history.append(result)
        self._save_training_log()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Stage 1 COMPLETE: Pretrained in {total_time / 3600:.1f} hours")
        logger.info(f"   Final loss: {result['final_loss']:.4f}")
        logger.info(f"{'='*60}")
        
        return result
    
    # ==========================================================================
    # Stage 2: Domain Adaptation
    # ==========================================================================
    
    def finetune(self, validated_papers: List[Dict], num_epochs: int = 10,
                 batch_size: int = 4, learning_rate: float = 1e-4,
                 gradient_accumulation_steps: int = 4) -> Dict:
        """
        Stage 2: Fine-tune ScholarFormer on research papers.
        
        This adapts the pretrained model to the academic domain.
        The novel components (cross-attention, section encoding) are most
        useful here as they capture paper-specific structure.
        
        Args:
            validated_papers: List of validated paper dicts from the pipeline
            num_epochs: Number of passes over the paper dataset
            batch_size: Per-device batch size
            learning_rate: Peak learning rate (lower than pretraining)
            gradient_accumulation_steps: Accumulation steps
        
        Returns:
            Dict with training metrics
        """
        logger.info("=" * 60)
        logger.info("🎓 STAGE 2: Domain Adaptation on Research Papers")
        logger.info("=" * 60)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Build paper dataset with section awareness
        dataset = PaperChunkDataset(
            validated_papers, self.tokenizer,
            max_length=self.model_config.max_seq_len
        )
        
        if len(dataset) == 0:
            logger.error("No valid training samples found in papers!")
            return {'stage': 'finetune', 'error': 'No valid samples'}
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        
        # Setup training (lower LR for fine-tuning)
        total_steps = (len(dataloader) // gradient_accumulation_steps) * num_epochs
        warmup_steps = int(0.1 * total_steps)
        
        optimizer = self._create_optimizer(lr=learning_rate, weight_decay=0.01)
        scheduler = self._create_scheduler(optimizer, total_steps, warmup_steps)
        scaler = GradScaler()
        
        logger.info(f"Training samples: {len(dataset):,}")
        logger.info(f"Batch size: {batch_size} × {gradient_accumulation_steps} accum = {batch_size * gradient_accumulation_steps} effective")
        logger.info(f"Total steps: {total_steps:,}")
        logger.info(f"Learning rate: {learning_rate}")
        
        # Train!
        start_time = time.time()
        metrics = self._training_loop(
            dataloader, optimizer, scheduler, scaler,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            stage_name="Finetune",
            log_interval=50,
            save_interval=500,
        )
        
        total_time = time.time() - start_time
        
        # Save final fine-tuned model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_name = f"finetuned_{timestamp}"
        self._save_checkpoint(final_name, "Finetune")
        
        result = {
            'stage': 'finetune',
            'epochs': num_epochs,
            'num_papers': len(validated_papers),
            'num_samples': len(dataset),
            'total_time_seconds': total_time,
            'total_time_human': f"{total_time / 3600:.1f} hours",
            'final_loss': metrics[-1]['avg_loss'] if metrics else None,
            'metrics_history': metrics,
            'model_path': os.path.join(self.checkpoint_dir, final_name),
        }
        
        self.training_history.append(result)
        self._save_training_log()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Stage 2 COMPLETE: Fine-tuned in {total_time / 3600:.1f} hours")
        logger.info(f"   Final loss: {result['final_loss']:.4f}")
        logger.info(f"   Papers: {len(validated_papers)}, Samples: {len(dataset)}")
        logger.info(f"{'='*60}")
        
        return result
    
    # ==========================================================================
    # Checkpoint Management
    # ==========================================================================
    
    def _save_checkpoint(self, name: str, stage: str):
        """Save model checkpoint."""
        path = os.path.join(self.checkpoint_dir, name)
        os.makedirs(path, exist_ok=True)
        
        # Save model weights
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.model_config.__dict__,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
        }, os.path.join(path, 'model.pt'))
        
        # Save config separately for easy loading
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(self.model_config.__dict__, f, indent=2, default=str)
        
        logger.info(f"💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        checkpoint_file = os.path.join(path, 'model.pt')
        
        if not os.path.exists(checkpoint_file):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
        
        logger.info(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        
        # Rebuild model if config differs
        saved_config = checkpoint.get('config', {})
        self.model_config = ScholarFormerConfig(**{
            k: v for k, v in saved_config.items() 
            if k in ScholarFormerConfig.__dataclass_fields__
        })
        self.model = ScholarFormerModel(self.model_config).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        # Resize embeddings to match actual tokenizer vocab size
        # (checkpoint may have been saved with vocab_size=32000 but
        #  tokenizer adds section markers making it 32018)
        actual_vocab_size = self.tokenizer.vocab_size
        if actual_vocab_size != self.model.config.vocab_size:
            self.model.resize_token_embeddings(actual_vocab_size)
            self.model = self.model.to(self.device)
        
        logger.info(f"✅ Checkpoint loaded: step {self.global_step}, "
                     f"best_loss {self.best_loss:.4f}, "
                     f"stage: {checkpoint.get('stage', 'unknown')}")
    
    def _save_training_log(self):
        """Save training history to JSON."""
        log_path = os.path.join(self.model_dir, 'training_history.json')
        with open(log_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        logger.info(f"Training log saved to {log_path}")
    
    # ==========================================================================
    # Full Pipeline
    # ==========================================================================
    
    def train_full_pipeline(self, validated_papers: Optional[List[Dict]] = None,
                             pretrain_epochs: int = 3,
                             finetune_epochs: int = 10,
                             resume_from: Optional[str] = None) -> Dict:
        """
        Run the complete two-stage training pipeline.
        
        Args:
            validated_papers: Papers for Stage 2 (if None, only runs Stage 1)
            pretrain_epochs: Epochs for WikiText-103 pretraining
            finetune_epochs: Epochs for paper fine-tuning
            resume_from: Path to checkpoint to resume from (skips Stage 1)
        
        Returns:
            Dict with results from both stages
        """
        results = {}
        
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.load_checkpoint(resume_from)
        else:
            # Stage 1: Pretrain
            results['pretrain'] = self.pretrain(
                num_epochs=pretrain_epochs,
                batch_size=self.config.get('sf_pretrain_batch_size', 4),
                learning_rate=self.config.get('sf_pretrain_lr', 3e-4),
                gradient_accumulation_steps=self.config.get('sf_pretrain_grad_accum', 8),
            )
        
        # Stage 2: Fine-tune on papers
        if validated_papers:
            results['finetune'] = self.finetune(
                validated_papers,
                num_epochs=finetune_epochs,
                batch_size=self.config.get('sf_finetune_batch_size', 4),
                learning_rate=self.config.get('sf_finetune_lr', 1e-4),
                gradient_accumulation_steps=self.config.get('sf_finetune_grad_accum', 4),
            )
        
        logger.info("\n" + "🎉" * 20)
        logger.info("ScholarFormer training pipeline COMPLETE!")
        logger.info("🎉" * 20)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the current model."""
        counts = self.model.count_parameters()
        return {
            'status': 'loaded',
            'model_name': 'ScholarFormer',
            'total_params': counts['total'],
            'trainable_params': counts['trainable'],
            'device': str(self.device),
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'config': self.model_config.__dict__,
        }
