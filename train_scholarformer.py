"""
🚀 ScholarFormer — Full Training Pipeline (Fire & Forget)

Run this script overnight on your GPU server:
    nohup python train_scholarformer.py > scholarformer_full.log 2>&1 &
    tail -f scholarformer_full.log

Stages:
    1. WikiText-103 Pretraining  (~2-3 hrs)
    2. Paper Fine-tuning         (~1-2 hrs)
    3. Cross-Model Distillation  (~2-3 hrs, 5 cycles)
                                 ─────────
                        Total:   ~6-8 hrs

Requirements:
    - GPU with 12GB+ VRAM
    - ~15GB free disk space
    - Internet (for WikiText-103 download in Stage 1)
"""

import os
import sys

# MUST be set before importing transformers/tokenizers to prevent fork deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import time
import logging
import traceback
from datetime import datetime

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────
log_file = f"scholarformer_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger('ScholarFormerPipeline')

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
CONFIG = {
    # Paths
    'model_dir': './models',
    'logs_dir': './logs',
    'data_dir': './data',
    'vector_db_path': './vector_store',

    # Phi-3 teacher model (for distillation)
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

    # Distillation
    'distillation_cycles': 5,
    'distillation_qa_per_chunk': 3,
    'distillation_student_questions': 30,
}


def print_banner(text: str):
    logger.info("\n" + "=" * 60)
    logger.info(f"  {text}")
    logger.info("=" * 60)


def load_validated_papers():
    """Load papers from the existing pipeline's data directory.
    
    Checks for preprocessed JSON first. If not found, extracts text
    directly from PDFs using PyMuPDF.
    """
    data_dir = CONFIG['data_dir']
    papers = []

    # 1. Check for preprocessed JSON files
    processed_dir = os.path.join(data_dir, 'processed')
    if os.path.exists(processed_dir):
        for f in os.listdir(processed_dir):
            if f.endswith('.json'):
                try:
                    with open(os.path.join(processed_dir, f), 'r', encoding='utf-8') as fh:
                        paper = json.load(fh)
                        papers.append(paper)
                except Exception:
                    continue

    # 2. Check for a single combined file
    combined_path = os.path.join(data_dir, 'validated_papers.json')
    if os.path.exists(combined_path) and not papers:
        with open(combined_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)

    # 3. Fallback: extract text from PDFs using PyMuPDF
    if not papers:
        logger.info("No preprocessed JSON found — extracting text from PDFs...")
        try:
            import fitz  # PyMuPDF
        except ImportError:
            logger.error("PyMuPDF not installed! Run: pip install pymupdf")
            return []
        
        pdf_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pdf')])
        logger.info(f"Found {len(pdf_files)} PDFs in {data_dir}")
        
        for i, pdf_file in enumerate(pdf_files):
            try:
                pdf_path = os.path.join(data_dir, pdf_file)
                doc = fitz.open(pdf_path)
                
                # Extract text from all pages
                full_text = ""
                for page in doc:
                    full_text += page.get_text() + "\n"
                doc.close()
                
                if len(full_text.strip()) < 200:
                    continue
                
                # Split into chunks of ~500 words
                words = full_text.split()
                chunk_size = 500
                chunks = []
                for j in range(0, len(words), chunk_size):
                    chunk_text = " ".join(words[j:j + chunk_size])
                    if len(chunk_text.strip()) > 100:
                        chunks.append(chunk_text)
                
                paper = {
                    'metadata': {
                        'title': pdf_file.replace('.pdf', ''),
                        'source': pdf_path,
                    },
                    'chunks': chunks,
                }
                papers.append(paper)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  Extracted {i + 1}/{len(pdf_files)} PDFs...")
                    
            except Exception as e:
                logger.warning(f"  Failed to extract {pdf_file}: {e}")
                continue
        
        # Save extracted papers for future runs
        if papers:
            save_path = os.path.join(data_dir, 'validated_papers.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(papers, f, ensure_ascii=False)
            logger.info(f"Saved {len(papers)} extracted papers to {save_path}")

    logger.info(f"Loaded {len(papers)} papers for fine-tuning")
    return papers


def extract_chunks(papers):
    """Extract text chunks from papers for distillation."""
    chunks = []
    for paper in papers:
        paper_chunks = paper.get('chunks', [])
        for chunk in paper_chunks:
            text = chunk if isinstance(chunk, str) else chunk.get('text', '')
            if len(text.strip()) > 100:
                chunks.append(text)
    logger.info(f"Extracted {len(chunks)} text chunks for distillation")
    return chunks


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def main():
    pipeline_start = time.time()
    results = {}

    print_banner("🚀 ScholarFormer Full Training Pipeline")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")

    # Import here to catch import errors early
    from agents.custom_trainer import CustomTrainerAgent

    trainer = CustomTrainerAgent(CONFIG)
    model_info = trainer.get_model_info()
    logger.info(f"Model: {model_info['total_params']:,} parameters on {model_info['device']}")

    # ──────────────────────────────────────────────────────────
    # STAGE 1: Load Pretrained Checkpoint (skip retraining)
    # ──────────────────────────────────────────────────────────
    checkpoint_path = os.path.join(
        CONFIG['model_dir'], 'scholarformer', 'checkpoints', 'step_1000'
    )
    
    if os.path.exists(checkpoint_path):
        print_banner("📚 STAGE 1: Loading Pretrained Checkpoint")
        try:
            trainer.load_checkpoint(checkpoint_path)
            results['pretrain'] = {
                'stage': 'pretrain',
                'status': 'loaded_from_checkpoint',
                'checkpoint': checkpoint_path,
                'global_step': trainer.global_step,
                'best_loss': trainer.best_loss,
            }
            logger.info(f"✅ Loaded pretrained model from {checkpoint_path}")
            logger.info(f"   Global step: {trainer.global_step}, Best loss: {trainer.best_loss:.4f}")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            traceback.print_exc()
            logger.info("Falling back to training from scratch...")
            pretrain_result = trainer.pretrain(
                num_epochs=CONFIG['sf_pretrain_epochs'],
                batch_size=CONFIG['sf_pretrain_batch_size'],
                learning_rate=CONFIG['sf_pretrain_lr'],
                gradient_accumulation_steps=CONFIG['sf_pretrain_grad_accum'],
            )
            results['pretrain'] = pretrain_result
    else:
        print_banner("📚 STAGE 1: WikiText-103 Pretraining (no checkpoint found)")
        try:
            pretrain_result = trainer.pretrain(
                num_epochs=CONFIG['sf_pretrain_epochs'],
                batch_size=CONFIG['sf_pretrain_batch_size'],
                learning_rate=CONFIG['sf_pretrain_lr'],
                gradient_accumulation_steps=CONFIG['sf_pretrain_grad_accum'],
            )
            results['pretrain'] = pretrain_result
            logger.info(f"✅ Stage 1 done in {pretrain_result['total_time_human']}")
            logger.info(f"   Final loss: {pretrain_result['final_loss']:.4f}")
        except Exception as e:
            logger.error(f"❌ Stage 1 FAILED: {e}")
            traceback.print_exc()
            results['pretrain'] = {'error': str(e)}

    # ──────────────────────────────────────────────────────────
    # STAGE 2: Paper Fine-tuning
    # ──────────────────────────────────────────────────────────
    print_banner("🎓 STAGE 2: Research Paper Fine-tuning")

    # Check for existing finetuned checkpoint (skip if already done)
    checkpoint_base = os.path.join(CONFIG['model_dir'], 'scholarformer', 'checkpoints')
    finetuned_checkpoints = sorted([
        d for d in os.listdir(checkpoint_base)
        if (d.startswith('finetuned_') or d == 'best') and os.path.isdir(os.path.join(checkpoint_base, d))
    ]) if os.path.exists(checkpoint_base) else []
    
    # Prefer 'finetuned_*' over 'best'
    ft_pick = next(
        (d for d in reversed(finetuned_checkpoints) if d.startswith('finetuned_')),
        finetuned_checkpoints[-1] if finetuned_checkpoints else None
    )

    if ft_pick:
        # Already finetuned — skip Stage 2 and load the latest checkpoint
        ft_path = os.path.join(checkpoint_base, ft_pick)
        logger.info(f"✅ Found finetuned checkpoint: {ft_pick}")
        logger.info(f"   Skipping Stage 2, loading from {ft_path}")
        try:
            trainer.load_checkpoint(ft_path)
            results['finetune'] = {
                'stage': 'finetune',
                'status': 'loaded_from_checkpoint',
                'checkpoint': ft_path,
            }
        except Exception as e:
            logger.error(f"❌ Failed to load finetuned checkpoint: {e}")
            results['finetune'] = {'error': str(e)}
    else:
        papers = load_validated_papers()

        if papers:
            try:
                finetune_result = trainer.finetune(
                    validated_papers=papers,
                    num_epochs=CONFIG['sf_finetune_epochs'],
                    batch_size=CONFIG['sf_finetune_batch_size'],
                    learning_rate=CONFIG['sf_finetune_lr'],
                    gradient_accumulation_steps=CONFIG['sf_finetune_grad_accum'],
                )
                results['finetune'] = finetune_result
                logger.info(f"✅ Stage 2 done in {finetune_result['total_time_human']}")
                logger.info(f"   Final loss: {finetune_result['final_loss']:.4f}")
            except Exception as e:
                logger.error(f"❌ Stage 2 FAILED: {e}")
                traceback.print_exc()
                results['finetune'] = {'error': str(e)}
        else:
            logger.warning("⚠️  No papers found — skipping Stage 2")
            logger.warning("   Place papers in ./data/processed/ as JSON files")
            results['finetune'] = {'skipped': 'no papers found'}

    # ──────────────────────────────────────────────────────────
    # STAGE 3: Cross-Model Distillation
    # ──────────────────────────────────────────────────────────
    print_banner("🧬 STAGE 3: Cross-Model Self-Distillation")

    if papers:
        try:
            from agents.distillation import DistillationAgent

            chunks = extract_chunks(papers)

            distiller = DistillationAgent(
                config=CONFIG,
                student_model=trainer.model,
                student_tokenizer=trainer.tokenizer,
            )

            distill_result = distiller.distill(
                paper_chunks=chunks,
                num_cycles=CONFIG['distillation_cycles'],
                qa_per_chunk=CONFIG['distillation_qa_per_chunk'],
                student_questions_per_cycle=CONFIG['distillation_student_questions'],
            )
            results['distillation'] = distill_result
            logger.info(f"✅ Stage 3 done in {distill_result['total_time_human']}")
            logger.info(f"   Total Q&A pairs: {distill_result['total_qa_pairs']}")
        except Exception as e:
            logger.error(f"❌ Stage 3 FAILED: {e}")
            traceback.print_exc()
            results['distillation'] = {'error': str(e)}
    else:
        logger.warning("⚠️  No papers — skipping distillation")
        results['distillation'] = {'skipped': 'no papers found'}

    # ──────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ──────────────────────────────────────────────────────────
    total_time = time.time() - pipeline_start

    print_banner("🎉 PIPELINE COMPLETE")
    logger.info(f"Total time: {total_time / 3600:.1f} hours")
    logger.info(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for stage, result in results.items():
        if 'error' in result:
            logger.info(f"  ❌ {stage}: FAILED — {result['error'][:100]}")
        elif 'skipped' in result:
            logger.info(f"  ⚠️  {stage}: SKIPPED — {result['skipped']}")
        else:
            loss = result.get('final_loss', 'N/A')
            time_str = result.get('total_time_human', 'N/A')
            logger.info(f"  ✅ {stage}: loss={loss}, time={time_str}")

    # Save pipeline results
    results['total_time_seconds'] = total_time
    results['total_time_human'] = f"{total_time / 3600:.1f} hours"
    results['completed_at'] = datetime.now().isoformat()

    results_path = os.path.join(CONFIG['model_dir'], 'scholarformer', 'pipeline_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Training log: {log_file}")


if __name__ == '__main__':
    main()
