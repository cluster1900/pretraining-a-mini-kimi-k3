"""Run a bounded checkpoint continuation, then resume the full run if it passes.

Execute on v100 with the command in TRAINING_PLAN.md. All runtime output stays
under /data/mini-k3. A failed gate stops the controller; there are no retries.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone

import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/data/mini-k3')
CHECKPOINTS = BASE / 'checkpoints'
MANIFESTS = BASE / 'data/prepared-v2-supplement-v2/manifests'
METRICS = re.compile(
    r'\[Step (\d+)/38147\] loss (\S+) \| lm (\S+).*?'
    r'\| step (\S+)s \| tok/s ([\d,]+).*?'
    r'mem max alloc (\S+) GiB / reserved (\S+) GiB \| dead (\S+)% \| imb (\S+)'
)


def validate_gate(log: str, before: dict, after: dict) -> dict:
    start = int(before['step']) + 1
    final = start + 199
    if f'Resumed successfully at Step {start} ' not in log:
        raise ValueError('Gate did not resume the selected checkpoint')
    if 'Bounded continuation complete.' not in log:
        raise ValueError('Gate did not complete normally')
    if re.search(r'Traceback|out of memory|non.?finite|SKIPPED|Aborting|WARNING: resumed from OLDER', log, re.I):
        raise ValueError('Gate contains an error, skipped step or checkpoint fallback')
    rows = [[float(x.replace(',', '')) for x in match] for match in METRICS.findall(log)]
    if len(rows) < 20 or rows[0][0] != start or rows[-1][0] != final:
        raise ValueError('Missing gate telemetry or final step')
    if not all(math.isfinite(x) for row in rows for x in row):
        raise ValueError('Non-finite gate telemetry')
    if after['step'] != final or after['world_size'] != 3:
        raise ValueError('Final checkpoint step/world size mismatch')
    if before['run_signature'] != after['run_signature']:
        raise ValueError('Gate changed the full run signature')
    if after['total_tokens_seen'] != before['total_tokens_seen'] + 200 * 264192:
        raise ValueError('Unexpected token progress')
    val_lm = float(after['val_lm_loss'])
    if not math.isfinite(val_lm):
        raise ValueError('Non-finite validation loss')
    first_lm = statistics.mean(row[2] for row in rows[:5])
    last_lm = statistics.mean(row[2] for row in rows[-5:])
    if last_lm > first_lm * 1.1:
        raise ValueError('Mean LM loss increased by more than 10%')
    peak_reserved = max(row[6] for row in rows)
    if peak_reserved >= 30 or max(row[7] for row in rows) > 5 or max(row[8] for row in rows) > 10:
        raise ValueError('Memory headroom or routing gate failed')
    return dict(status='passed', start_step=start, final_step=final,
                first_mean_lm=first_lm, last_mean_lm=last_lm, val_lm=val_lm,
                peak_reserved_gib=peak_reserved,
                mean_step_seconds=statistics.mean(row[3] for row in rows))


def load_complete(path: Path) -> dict:
    names = ['COMPLETE', 'model.pt', 'optimizer.pt', 'scaler.pt', 'meta.pt']
    names += [f'{kind}_rank{rank}.pt' for rank in range(3) for kind in ('loader', 'rng')]
    if any(not (path / name).is_file() or (path / name).stat().st_size == 0 for name in names):
        raise ValueError(f'Incomplete checkpoint: {path}')
    return torch.load(path / 'meta.pt', map_location='cpu', weights_only=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_relative_to(BASE / 'logs'):
        parser.error('--run-dir must be under /data/mini-k3/logs')
    os.chdir(ROOT)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    locks = []
    for name in ('pretrain-10b.lock', 'pretrain-10b-3gpu.lock', 'pretrain-short-3gpu.lock'):
        handle = (BASE / 'logs' / name).open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    run_dir.mkdir(parents=True, exist_ok=False)
    status = dict(controller_pid=os.getpid(), cuda_visible_devices='0,1,2')

    def record(stage, **extra):
        status.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
        temp = run_dir / 'status.tmp'
        temp.write_text(json.dumps(status, indent=2) + '\n')
        temp.replace(run_dir / 'status.json')
        print(json.dumps(status), flush=True)

    def run(command, log_name):
        record(status['stage'], command=command, log=str(run_dir / log_name))
        with (run_dir / log_name).open('x') as output:
            subprocess.run(command, stdin=subprocess.DEVNULL, stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    try:
        record('preflight')
        candidates = sorted(path.parent for path in CHECKPOINTS.glob('step_*/COMPLETE'))
        if not candidates:
            raise ValueError('No complete training checkpoint; refusing fresh training')
        source = candidates[-1]
        before = load_complete(source)
        sig = before['run_signature']
        if before['world_size'] != 3 or sig['world_size'] != 3 or sig['gradient_accumulation_steps'] != 43:
            raise ValueError('Checkpoint has not been migrated to the three-GPU configuration')
        if sig['total_steps'] != 38147 or sig['schedule_total_steps'] != 38147 or before['step'] + 200 >= 38147:
            raise ValueError('Checkpoint is outside this continuation gate scope')
        gpu_processes = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
        if gpu_processes.strip():
            raise ValueError('GPU processes already running; refusing a duplicate launch')
        run([sys.executable, 'train/data/check_training_readiness.py',
             '--manifest', str(MANIFESTS / 'pretrain_stable.json'),
             '--validation-manifest', str(MANIFESTS / 'validation.json'),
             '--expected-training-tokens', '10078132224'], 'readiness.json')
        command = [str(BASE / 'venv/bin/torchrun'), '--standalone', '--nproc_per_node=3',
                   'train/train.py', '--resume', '--data_manifest', str(MANIFESTS / 'pretrain_stable.json'),
                   '--validation_manifest', str(MANIFESTS / 'validation.json'),
                   '--checkpoint_dir', str(CHECKPOINTS), '--total_steps', '38147', '--log_interval', '10']
        record('continuation_gate', source_checkpoint=str(source), source_step=before['step'])
        run(command + ['--stop-after-steps', '200', '--save_interval', '50'], 'gate.log')
        after_path = CHECKPOINTS / f"step_{before['step'] + 200:06d}"
        after = load_complete(after_path)
        report = validate_gate((run_dir / 'gate.log').read_text(), before, after)
        report['run_signature'] = after['run_signature']
        (run_dir / 'gate.json').write_text(json.dumps(report, indent=2) + '\n')
        record('formal_training', gate=report, source_checkpoint=str(after_path))
        run(command + ['--save_interval', '1000'], 'formal.log')
        record('complete')
    except Exception as error:
        record('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
