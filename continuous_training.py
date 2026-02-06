"""
Continuous Training Loop - Automated Training Pipeline
Automatically updates arxiv offset and training log numbers between runs.

Usage:
    python continuous_training.py                    # Interactive mode (Ctrl+C to stop)
    python continuous_training.py --runs 5           # Run exactly 5 training cycles
    python continuous_training.py --cooldown 300     # 5-minute cooldown between runs

Transfer to college GPU:
    1. Copy this file along with training_state.json
    2. Run: python continuous_training.py
    3. Training will resume from where it left off
"""

import os
import sys
import json
import time
import signal
import logging
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configuration
STATE_FILE = './training_state.json'
DEFAULT_COOLDOWN = 300  # 5 minutes between runs
ARXIV_INCREMENT = 150   # Papers to skip per run (matches arxiv_max_results)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global shutdown_requested
    print("\n⏸️  Shutdown requested. Will stop after current cycle completes...")
    print("   (Press Ctrl+C again to force quit)")
    shutdown_requested = True


def load_state():
    """Load training state from JSON file."""
    default_state = {
        "arxiv_offset": 800,
        "log_number": 11,
        "last_run": None,
        "total_runs": 0,
        "history": []
    }
    
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in default_state.items():
                    if key not in state:
                        state[key] = value
                return state
        except json.JSONDecodeError:
            print(f"⚠️  Corrupted state file. Starting fresh.")
            return default_state
    return default_state


def save_state(state):
    """Save training state to JSON file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def setup_logging(log_number):
    """Configure logging to both console and numbered log file."""
    log_file = f'training{log_number}.log'
    
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up fresh logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return log_file


def run_training_cycle(state):
    """Execute one training cycle with the current state."""
    # Import here to avoid loading heavy modules at startup
    from config import CONFIG
    from agents.orchestrator import OrchestratorAgent
    
    # Override config with current offset
    CONFIG['arxiv_start_offset'] = state['arxiv_offset']
    
    # Ensure directories exist
    os.makedirs(CONFIG['data_dir'], exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)
    os.makedirs(CONFIG['logs_dir'], exist_ok=True)
    os.makedirs(CONFIG['vector_db_path'], exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"🚀 CONTINUOUS TRAINING - Run #{state['total_runs'] + 1}")
    logger.info(f"   arXiv offset: {state['arxiv_offset']}")
    logger.info(f"   Log file: training{state['log_number']}.log")
    logger.info("=" * 60)
    
    # Run the pipeline - let exceptions propagate!
    orchestrator = OrchestratorAgent(CONFIG)
    result = orchestrator.orchestrate()
    
    # Check if orchestration actually succeeded
    if result is None:
        logger.warning("⚠️  Full pipeline failed (likely arXiv rate limit)")
        logger.info("🔄 Falling back to training on existing data...")
        
        # Try train_only as fallback
        try:
            import subprocess
            fallback_result = subprocess.run(
                [sys.executable, 'train_only.py'],
                cwd=os.getcwd(),
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout
            )
            if fallback_result.returncode != 0:
                logger.error(f"Fallback training also failed: {fallback_result.stderr}")
                raise RuntimeError("Both full pipeline and fallback training failed")
            logger.info("✅ Fallback training completed!")
            return True
        except Exception as e:
            raise RuntimeError(f"Fallback training failed: {e}")
    
    logger.info("✅ Training cycle completed successfully!")
    return True


def main():
    global shutdown_requested
    
    parser = argparse.ArgumentParser(description='Continuous Training Loop')
    parser.add_argument('--runs', type=int, default=0, 
                        help='Number of runs (0 = infinite)')
    parser.add_argument('--cooldown', type=int, default=DEFAULT_COOLDOWN,
                        help=f'Cooldown seconds between runs (default: {DEFAULT_COOLDOWN})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would happen without training')
    args = parser.parse_args()
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)
    
    print("=" * 60)
    print("🔄 CONTINUOUS TRAINING SYSTEM")
    print("=" * 60)
    print(f"   Cooldown: {args.cooldown} seconds between runs")
    print(f"   Max runs: {'Infinite' if args.runs == 0 else args.runs}")
    print("   Press Ctrl+C to stop gracefully")
    print("=" * 60)
    
    # Load initial state
    state = load_state()
    print(f"\n📊 Current State:")
    print(f"   arXiv offset: {state['arxiv_offset']}")
    print(f"   Next log: training{state['log_number']}.log")
    print(f"   Total runs completed: {state['total_runs']}")
    print()
    
    run_count = 0
    
    while True:
        if shutdown_requested:
            print("\n🛑 Shutting down. State saved.")
            break
        
        if args.runs > 0 and run_count >= args.runs:
            print(f"\n✅ Completed {args.runs} runs as requested.")
            break
        
        # Reload state (in case it was modified externally)
        state = load_state()
        
        # Set up logging for this run
        log_file = setup_logging(state['log_number'])
        logger = logging.getLogger(__name__)
        
        if args.dry_run:
            print(f"\n[DRY RUN] Would start training run #{state['total_runs'] + 1}")
            print(f"          arXiv offset: {state['arxiv_offset']}")
            print(f"          Log file: {log_file}")
            run_count += 1
            time.sleep(2)
            continue
        
        try:
            # Run training
            success = run_training_cycle(state)
            
            if success:
                # Update state after successful run
                run_record = {
                    "run_number": state['total_runs'] + 1,
                    "arxiv_offset": state['arxiv_offset'],
                    "log_number": state['log_number'],
                    "timestamp": datetime.now().isoformat()
                }
                
                state['arxiv_offset'] += ARXIV_INCREMENT
                state['log_number'] += 1
                state['total_runs'] += 1
                state['last_run'] = datetime.now().isoformat()
                state['history'].append(run_record)
                
                # Keep only last 20 history entries
                if len(state['history']) > 20:
                    state['history'] = state['history'][-20:]
                
                save_state(state)
                
                logger.info(f"📈 State updated:")
                logger.info(f"   Next arXiv offset: {state['arxiv_offset']}")
                logger.info(f"   Next log file: training{state['log_number']}.log")
                
                run_count += 1
                
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            logger.info("   Will retry after cooldown...")
        
        # Check for shutdown before cooldown
        if shutdown_requested:
            print("\n🛑 Shutting down. State saved.")
            break
        
        # Cooldown between runs
        if args.runs == 0 or run_count < args.runs:
            print(f"\n⏳ Cooldown: {args.cooldown} seconds before next run...")
            for i in range(args.cooldown):
                if shutdown_requested:
                    break
                time.sleep(1)
                # Show countdown every 30 seconds
                remaining = args.cooldown - i
                if remaining % 30 == 0 and remaining > 0:
                    print(f"   {remaining} seconds remaining...")
    
    print("\n" + "=" * 60)
    print("📊 FINAL STATE:")
    state = load_state()
    print(f"   Total runs completed: {state['total_runs']}")
    print(f"   Next arXiv offset: {state['arxiv_offset']}")
    print(f"   Next log file: training{state['log_number']}.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
