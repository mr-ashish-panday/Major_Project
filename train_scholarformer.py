"""
🚀 ScholarFormer — All-in-One Training Script
===============================================
Fire-and-forget script that runs the complete training pipeline:

    Stage 1: Pretrain on WikiText-103         (~2-3 hrs)
    Stage 2: Fine-tune on research papers     (~1-2 hrs)
    Stage 3: Cross-model distillation ×5      (~2-3 hrs)
                                        Total: ~6-8 hrs

Usage (on server):
    cd ~/Major_Project
    git pull origin main
    nohup python train_scholarformer.py > scholarformer_full.log 2>&1 &
    tail -f scholarformer_full.log

Requirements:
    - GPU with 12GB+ VRAM
    - PyTorch, datasets, transformers installed
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

# ==============================================================================
# Setup logging
# ==============================================================================
log_file = f"scholarformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('ScholarFormer')

# ==============================================================================
# Configuration
# ==============================================================================
CONFIG = {
    # Paths
    'model_dir': './models',
    'logs_dir': './logs',
    'data_dir': './data',
    'vector_db_path': './vector_store',
    
    # Existing model (for distillation teacher)
    'base_model': 'microsoft/Phi-3-mini-4k-instruct',
    
    # ScholarFormer architecture
    'sf_hidden_dim': 768,
    'sf_num_layers': 12,
    'sf_num_heads': 12,
    'sf_cross_attn_heads': 4,
    'sf_vocab_size': 32000,
    'sf_max_seq_len': 1024,
    'sf_ffn_dim': 2048,
    'sf_dropout': 0.1,
    
    # Stage 1: Pretraining
    'sf_pretrain_batch_size': 4,
    'sf_pretrain_lr': 3e-4,
    'sf_pretrain_grad_accum': 8,
    'sf_pretrain_epochs': 3,
    
    # Stage 2: Fine-tuning
    'sf_finetune_batch_size': 4,
    'sf_finetune_lr': 1e-4,
    'sf_finetune_grad_accum': 4,
    'sf_finetune_epochs': 10,
    
    # Stage 3: Distillation
    'sf_distill_cycles': 5,
    'sf_distill_qa_per_chunk': 3,
    'sf_distill_student_questions': 30,
}


def print_banner(text: str):
    """Print a visible banner."""
    logger.info("\n" + "=" * 70)
    logger.info(f"  {text}")
    logger.info("=" * 70 + "\n")


def get_paper_chunks():
    """Load preprocessed paper chunks from the existing pipeline data."""
    data_dir = CONFIG['data_dir']
    chunks = []
    
    # Try loading from preprocessed JSON files
    processed_dir = os.path.join(data_dir, 'processed')
    if os.path.exists(processed_dir):
        for fname in os.listdir(processed_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(processed_dir, fname), 'r', encoding='utf-8') as f:
                        paper = json.load(f)
                    paper_chunks = paper.get('chunks', [])
                    for c in paper_chunks:
                        text = c if isinstance(c, str) else c.get('text', '')
                        if len(text.strip()) > 50:
                            chunks.append(text)
                except Exception:
                    continue
    
    # Fallback: try loading from validated data
    if not chunks:
        validated_path = os.path.join(data_dir, 'validated_papers.json')
        if os.path.exists(validated_path):
            with open(validated_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
            for paper in papers:
                for c in paper.get('chunks', []):
                    text = c if isinstance(c, str) else c.get('text', '')
                    if len(text.strip()) > 50:
                        chunks.append(text)
    
    # Fallback: scan for any text files in data directory
    if not chunks:
        for root, dirs, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith('.txt'):
                    try:
                        with open(os.path.join(root, fname), 'r', encoding='utf-8') as f:
                            text = f.read()
                        if len(text) > 100:
                            # Split into ~1000 char chunks
                            for i in range(0, len(text), 1000):
                                chunk = text[i:i+1000]
                                if len(chunk) > 50:
                                    chunks.append(chunk)
                    except Exception:
                        continue
    
    logger.info(f"Loaded {len(chunks)} paper chunks for training")
    return chunks


def get_validated_papers():
    """Load validated papers in the format expected by PaperChunkDataset."""
    data_dir = CONFIG['data_dir']
    papers = []
    
    # Try processed directory
    processed_dir = os.path.join(data_dir, 'processed')
    if os.path.exists(processed_dir):
        for fname in os.listdir(processed_dir):
            if fname.endswith('.json'):
                try:
                    with open(os.path.join(processed_dir, fname), 'r', encoding='utf-8') as f:
                        paper = json.load(f)
                    if paper.get('chunks'):
                        papers.append(paper)
                except Exception:
                    continue
    
    # Fallback
    if not papers:
        validated_path = os.path.join(data_dir, 'validated_papers.json')
        if os.path.exists(validated_path):
            with open(validated_path, 'r', encoding='utf-8') as f:
                papers = json.load(f)
    
    logger.info(f"Loaded {len(papers)} validated papers")
    return papers


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    import torch
    
    pipeline_start = time.time()
    
    print_banner("🚀 ScholarFormer Full Training Pipeline")
    
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Device: {'CUDA — ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (WARNING: very slow)'}")
    
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        logger.info(f"VRAM: {vram:.1f} GB")
    
    results = {}
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 1: Pretrain on WikiText-103
    # ══════════════════════════════════════════════════════════════════════
    
    print_banner("📚 STAGE 1/3: Pretraining on WikiText-103")
    
    try:
        from agents.custom_trainer import CustomTrainerAgent
        
        trainer = CustomTrainerAgent(CONFIG)
        
        results['stage1'] = trainer.pretrain(
            num_epochs=CONFIG['sf_pretrain_epochs'],
            batch_size=CONFIG['sf_pretrain_batch_size'],
            learning_rate=CONFIG['sf_pretrain_lr'],
            gradient_accumulation_steps=CONFIG['sf_pretrain_grad_accum'],
        )
        
        logger.info(f"✅ Stage 1 complete — Loss: {results['stage1'].get('final_loss', 'N/A')}")
        
    except Exception as e:
        logger.error(f"❌ Stage 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['stage1'] = {'error': str(e)}
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 2: Fine-tune on Research Papers
    # ══════════════════════════════════════════════════════════════════════
    
    print_banner("🎓 STAGE 2/3: Fine-tuning on Research Papers")
    
    try:
        validated_papers = get_validated_papers()
        
        if validated_papers:
            results['stage2'] = trainer.finetune(
                validated_papers,
                num_epochs=CONFIG['sf_finetune_epochs'],
                batch_size=CONFIG['sf_finetune_batch_size'],
                learning_rate=CONFIG['sf_finetune_lr'],
                gradient_accumulation_steps=CONFIG['sf_finetune_grad_accum'],
            )
            logger.info(f"✅ Stage 2 complete — Loss: {results['stage2'].get('final_loss', 'N/A')}")
        else:
            logger.warning("⚠️ No papers found — skipping Stage 2")
            results['stage2'] = {'skipped': True, 'reason': 'No validated papers found'}
            
    except Exception as e:
        logger.error(f"❌ Stage 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['stage2'] = {'error': str(e)}
    
    # ══════════════════════════════════════════════════════════════════════
    # STAGE 3: Cross-Model Distillation
    # ══════════════════════════════════════════════════════════════════════
    
    print_banner("🧬 STAGE 3/3: Cross-Model Self-Distillation")
    
    try:
        paper_chunks = get_paper_chunks()
        
        if paper_chunks:
            from agents.distillation import DistillationAgent
            
            distiller = DistillationAgent(
                config=CONFIG,
                student_model=trainer.model,
                student_tokenizer=trainer.tokenizer,
            )
            
            results['stage3'] = distiller.distill(
                paper_chunks=paper_chunks,
                num_cycles=CONFIG['sf_distill_cycles'],
                qa_per_chunk=CONFIG['sf_distill_qa_per_chunk'],
                student_questions_per_cycle=CONFIG['sf_distill_student_questions'],
            )
            logger.info(f"✅ Stage 3 complete — {results['stage3'].get('total_qa_pairs', 0)} Q&A pairs generated")
        else:
            logger.warning("⚠️ No paper chunks found — skipping distillation")
            results['stage3'] = {'skipped': True, 'reason': 'No paper chunks found'}
            
    except Exception as e:
        logger.error(f"❌ Stage 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results['stage3'] = {'error': str(e)}
    
    # ══════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    
    total_time = time.time() - pipeline_start
    end_time = datetime.now()
    
    print_banner("🎉 TRAINING PIPELINE COMPLETE!")
    
    logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {total_time / 3600:.1f} hours ({total_time:.0f} seconds)")
    logger.info(f"\nResults Summary:")
    
    for stage, result in results.items():
        if 'error' in result:
            logger.info(f"  {stage}: ❌ FAILED — {result['error'][:100]}")
        elif result.get('skipped'):
            logger.info(f"  {stage}: ⏭️ Skipped — {result.get('reason', '')}")
        else:
            loss = result.get('final_loss', 'N/A')
            logger.info(f"  {stage}: ✅ Loss={loss}")
    
    # Save final results
    results['total_time_seconds'] = total_time
    results['total_time_human'] = f"{total_time / 3600:.1f} hours"
    results['end_time'] = end_time.isoformat()
    
    results_path = os.path.join(CONFIG['model_dir'], 'scholarformer', 'pipeline_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nFull results saved to: {results_path}")
    logger.info(f"Checkpoints at: {CONFIG['model_dir']}/scholarformer/checkpoints/")
    logger.info(f"Training log: {log_file}")


if __name__ == '__main__':
    main()
