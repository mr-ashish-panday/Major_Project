"""Log analysis and plotting script

Produces PNG figures under `reports/figures/`:
 - metrics_time_series.png (perplexity, bleu, rouge over time)
 - metrics_histograms.png (distributions)
 - step_durations.png (per-run step durations stacked)
 - checkpoint_sizes.png (model dir sizes over time)
 - error_counts.png (error/warning counts per run)
 - improvement_objective.png (objective over time from improvement_history.json)

Run: python scripts/log_analysis.py
"""

import os
import re
import json
import math
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import humanize

sns.set(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Helpers
_ts = lambda s: datetime.fromisoformat(s) if isinstance(s, str) else s

# 1. Parse improvement_history.json
impr_file = LOGS / "improvement_history.json"
impr = []
if impr_file.exists():
    with open(impr_file, "r") as f:
        impr = json.load(f)
impr_df = pd.DataFrame([{"timestamp": _ts(x["timestamp"]),
                         "perplexity": x["input_metrics"]["perplexity"],
                         "bleu": x["input_metrics"]["bleu"],
                         "rouge": x["input_metrics"]["rouge"],
                         "num_samples": x["input_metrics"]["num_samples"],
                         "objective": x.get("objective")}
                        for x in impr])

# 2. Parse cycle_*.json files (evaluation snapshots)
cycles = []
for p in LOGS.glob('cycle_*.json'):
    try:
        j = json.load(open(p, 'r'))
        cycles.append({
            'file': p.name,
            'timestamp': _ts(j.get('timestamp')),
            'perplexity': j.get('metrics', {}).get('perplexity'),
            'bleu': j.get('metrics', {}).get('bleu'),
            'rouge': j.get('metrics', {}).get('rouge'),
            'model_path': j.get('model_path'),
            'total_documents': j.get('vector_store_stats', {}).get('total_documents')
        })
    except Exception:
        continue
cycles_df = pd.DataFrame(cycles)

# 3. Parse training_*.log for evaluation metrics, steps, errors
training_runs = []
step_re = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}).*STEP (?P<step>\d+): (?P<desc>.*)$")
eval_re = re.compile(r"Evaluation (?:complete|metrics):\s*(?P<json>\{.*\})")
eval_simple_re = re.compile(r"Perplexity:\s*(?P<p>\d+\.\d+).*BLEU:\s*(?P<b>\d+\.\d+),?\s*ROUGE-L:\s*(?P<r>\d+\.\d+)")
err_re = re.compile(r"\b(ERROR|WARNING)\b")

for p in LOGS.glob('training_*.log'):
    with open(p, 'r') as f:
        lines = f.readlines()
    steps = []
    errors = 0
    warnings = 0
    evals = []
    start_ts = None
    for ln in lines:
        m = step_re.search(ln)
        if m:
            ts = datetime.strptime(m.group('ts'), "%Y-%m-%d %H:%M:%S,%f")
            steps.append({'ts': ts, 'step': int(m.group('step')), 'desc': m.group('desc').strip()})
            if start_ts is None:
                start_ts = ts
        me = eval_re.search(ln)
        if me:
            try:
                j = json.loads(me.group('json').replace("'", '"'))
                evals.append({'ts': None, 'perplexity': j.get('perplexity'), 'bleu': j.get('bleu'), 'rouge': j.get('rouge'), 'num_samples': j.get('num_samples')})
            except Exception:
                pass
        if 'Perplexity:' in ln and 'BLEU' in ln:
            m2 = re.search(r"Perplexity:\s*(?P<p>\d+\.?\d+)", ln)
            m3 = re.search(r"BLEU:\s*(?P<b>\d+\.?\d+)", ln)
            m4 = re.search(r"ROUGE-L:\s*(?P<r>\d+\.?\d+)", ln)
            if m2:
                evals.append({'ts': None, 'perplexity': float(m2.group('p')), 'bleu': float(m3.group('b')) if m3 else None, 'rouge': float(m4.group('r')) if m4 else None})
        if err_re.search(ln):
            if 'ERROR' in ln:
                errors += 1
            if 'WARNING' in ln:
                warnings += 1
    # associate evals timestamp with STEP 6 if possible
    eval_ts = None
    for s in steps:
        if s['step'] == 6:
            eval_ts = s['ts']
    # pick most recent eval metrics
    last_eval = None
    for e in reversed(evals):
        if e.get('perplexity') is not None:
            last_eval = e
            break
    training_runs.append({'file': p.name, 'start_ts': start_ts, 'steps': steps, 'errors': errors, 'warnings': warnings, 'eval': last_eval, 'eval_ts': eval_ts})

# training DataFrame with evaluation
tr_list = []
for r in training_runs:
    ev = r['eval'] or {}
    tr_list.append({'file': r['file'], 'start_ts': r['start_ts'], 'eval_ts': r['eval_ts'], 'perplexity': ev.get('perplexity'), 'bleu': ev.get('bleu'), 'rouge': ev.get('rouge'), 'errors': r['errors'], 'warnings': r['warnings'], 'num_steps': len(r['steps'])})
tr_df = pd.DataFrame(tr_list).sort_values('start_ts')

# 4. Compute step durations per run
step_durations = []
for r in training_runs:
    if not r['steps']:
        continue
    s = sorted(r['steps'], key=lambda x: x['ts'])
    for i in range(len(s)):
        start = s[i]['ts']
        end = s[i+1]['ts'] if i+1 < len(s) else None
        duration = (end - start).total_seconds() if end else None
        step_durations.append({'run': r['file'], 'step': s[i]['step'], 'desc': s[i]['desc'], 'start': start, 'duration_s': duration})
steps_df = pd.DataFrame(step_durations)

# 5. Check model directories and sizes
mdirs = []
for d in MODELS.iterdir():
    if d.is_dir():
        size = sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
        mtime = datetime.fromtimestamp(d.stat().st_mtime)
        mdirs.append({'dir': d.name, 'size_bytes': size, 'mtime': mtime})
mdirs_df = pd.DataFrame(mdirs).sort_values('mtime')

# PLOTTING
# Metrics time series (combine improvement history + training evals)
metric_df = impr_df.copy()
metric_df = metric_df.sort_values('timestamp')

# add training evals using pd.concat (pandas 2.0+ compatible)
new_rows = []
for _, row in tr_df.iterrows():
    if pd.notnull(row['perplexity']):
        new_rows.append({
            'timestamp': row['eval_ts'] or row['start_ts'],
            'perplexity': row['perplexity'],
            'bleu': row['bleu'],
            'rouge': row['rouge']
        })
if new_rows:
    metric_df = pd.concat([metric_df, pd.DataFrame(new_rows)], ignore_index=True)
metric_df = metric_df.sort_values('timestamp')

plt.figure(figsize=(10,5))
for col in ['perplexity','bleu','rouge']:
    if col in metric_df.columns:
        plt.plot(metric_df['timestamp'], metric_df[col], marker='o', label=col)
plt.legend()
plt.title('Evaluation metrics over time')
plt.xlabel('timestamp')
plt.tight_layout()
plt.savefig(OUT/"metrics_time_series.png")
plt.close()

# Histograms
plt.figure(figsize=(10,6))
plt.subplot(3,1,1)
sns.histplot(metric_df['perplexity'].dropna(), kde=False)
plt.title('Perplexity distribution')
plt.subplot(3,1,2)
sns.histplot(metric_df['bleu'].dropna(), kde=False)
plt.title('BLEU distribution')
plt.subplot(3,1,3)
sns.histplot(metric_df['rouge'].dropna(), kde=False)
plt.title('ROUGE distribution')
plt.tight_layout()
plt.savefig(OUT/"metrics_histograms.png")
plt.close()

# Step durations grouped bar per run (stacked)
if not steps_df.empty:
    pd_pivot = steps_df.pivot_table(index='run', columns='step', values='duration_s')
    pd_pivot.plot(kind='bar', stacked=True, figsize=(12,6))
    plt.ylabel('seconds')
    plt.title('Step durations per run (stacked by step number)')
    plt.tight_layout()
    plt.savefig(OUT/"step_durations.png")
    plt.close()

# Checkpoint/model sizes over time
if not mdirs_df.empty:
    plt.figure(figsize=(10,5))
    plt.plot(mdirs_df['mtime'], mdirs_df['size_bytes']/1024/1024, marker='o')
    for i,r in mdirs_df.iterrows():
        plt.text(r['mtime'], r['size_bytes']/1024/1024, r['dir'])
    plt.ylabel('Size (MB)')
    plt.title('Model directory sizes over time')
    plt.tight_layout()
    plt.savefig(OUT/"checkpoint_sizes.png")
    plt.close()

# Error/warning counts per run
if not tr_df.empty:
    plt.figure(figsize=(10,5))
    plt.bar(tr_df['start_ts'].astype(str), tr_df['errors'], label='errors')
    plt.bar(tr_df['start_ts'].astype(str), tr_df['warnings'], bottom=tr_df['errors'], label='warnings')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title('Errors and warnings per run')
    plt.tight_layout()
    plt.savefig(OUT/"error_counts.png")
    plt.close()

# Improvement objective over time
if 'objective' in impr_df.columns and not impr_df.empty:
    plt.figure(figsize=(8,4))
    plt.plot(impr_df['timestamp'], impr_df['objective'], marker='o')
    plt.title('Improvement objective over time')
    plt.tight_layout()
    plt.savefig(OUT/"improvement_objective.png")
    plt.close()

print('Saved figures to', OUT)