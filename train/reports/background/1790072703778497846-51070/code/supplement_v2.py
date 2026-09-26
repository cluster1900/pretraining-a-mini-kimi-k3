"""Execute Option C: Supplement CodeSearchNet Python to Mini-K3 in an isolated run root (supplement-v2).

Run root: /data/mini-k3/data/prepared-v2-supplement-v2
Reports: /data/mini-k3/data/reports/supplement-v2
Logs: /data/mini-k3/logs/prepared-v2-supplement-v2

Rebuilds code-python canonical and cleaned stages with CodeSearchNet included,
clones base prepared-v2-supplement-v1 using hardlinks,
rebinds paths, runs continue_v2 controller and assesses training coverage.
Never launches model training.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from canonical_v2 import atomic_json, digest_file


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Run:
    def __init__(self, work):
        self.work = Path(work)
        self.base = self.work / 'data'
        self.raw = self.base / 'raw'
        self.source_root = self.base / 'prepared-v2-supplement-v1'
        self.root = self.base / 'prepared-v2-supplement-v2'
        self.reports = self.base / 'reports/supplement-v2'
        self.logs = self.work / 'logs/prepared-v2-supplement-v2'
        self.state = self.reports / 'PIPELINE_STATUS.json'
        self.reports.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.status = {'status': 'running', 'stage': 'init', 'started_at': now(), 'root': str(self.root)}
        atomic_json(self.state, self.status)

    def update(self, stage, status=None, **kw):
        self.status.update(stage=stage, updated_at=now(), **kw)
        if status:
            self.status['status'] = status
        atomic_json(self.state, self.status)

    def report(self, name, data):
        data = dict(data, generated_at=now())
        atomic_json(self.reports / name, data)
        self.update(self.status['stage'], report=str(self.reports / name), report_sha256=digest_file(self.reports / name))

    def run(self, cmd, logname):
        log = self.logs / logname
        print(f'[{now()}] Running command: {" ".join(cmd)} (log: {logname})', flush=True)
        with log.open('ab') as handle:
            child = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            code = child.wait()
        if code:
            raise RuntimeError(f'command failed exit={code}: {log}')
        print(f'[{now()}] Command completed successfully: {logname}', flush=True)


def rebind_cloned_root(root, source_root):
    root = Path(root)
    old_prefix = str(source_root) + "/"
    new_prefix = str(root) + "/"
    sources = [
        "openassistant", "openhermes", "openr1", "ultrafeedback",
        "chinese-fineweb-edu", "cosmopedia", "finemath", "open-web-math", "dolma-body", "fineweb-edu"
    ]
    for s in sources:
        can_path = root / "canonical" / s / "COMPLETE.json"
        if can_path.is_file():
            can_data = json.loads(can_path.read_text())
            for part in can_data.get("parts", []):
                if part["path"].startswith(old_prefix):
                    part["path"] = part["path"].replace(old_prefix, new_prefix)
            atomic_json(can_path, can_data)
            can_hash = digest_file(can_path)

            cln_path = root / "cleaned" / s / "COMPLETE.json"
            if cln_path.is_file():
                cln_data = json.loads(cln_path.read_text())
                for part in cln_data.get("parts", []):
                    if part["path"].startswith(old_prefix):
                        part["path"] = part["path"].replace(old_prefix, new_prefix)
                cln_data["upstream_report_sha256"] = can_hash
                atomic_json(cln_path, cln_data)
                cln_hash = digest_file(cln_path)

                if s == "openhermes":
                    sr_path = root / "SOURCE_REVIEW.json"
                    if sr_path.is_file():
                        sr_data = json.loads(sr_path.read_text())
                        sr_data["sources"]["openhermes"]["stage_report_sha256"]["canonical"] = can_hash
                        sr_data["sources"]["openhermes"]["stage_report_sha256"]["cleaned"] = cln_hash
                        atomic_json(sr_path, sr_data)


def clone_root(run):
    if not run.root.exists():
        print(f'Cloning {run.source_root} -> {run.root} with hardlinks...', flush=True)
        subprocess.run(['cp', '-al', str(run.source_root), str(run.root)], check=True)
        for stage in ('canonical', 'cleaned'):
            src = run.root / stage / 'code-python'
            if src.exists():
                shutil.move(str(src), str(run.root / stage / 'code-python-before-supplement'))
        for stage in ('deduped', 'near-deduped', 'decontaminated', 'tokenized'):
            src = run.root / stage
            if src.exists():
                shutil.move(str(src), str(run.root / (stage + '-before-supplement')))
            (run.root / stage).mkdir(parents=True, exist_ok=True)
        for p in run.root.glob('*sqlite*'):
            p.rename(p.with_name(p.name + '-before-supplement'))
        for p in run.root.glob('*COMPLETE.json'):
            p.rename(p.with_name(p.name.replace('.json', '-before-supplement.json')))
    rebind_cloned_root(run.root, run.source_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='/data/mini-k3')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    run = Run(args.work)
    scripts = run.work / 'project/train/data'
    py = run.work / 'venv/bin/python'

    try:
        run.update('cloning_base')
        clone_root(run)

        # Rebuild canonical and cleaned for code-python
        can_marker = run.root / 'canonical/code-python/COMPLETE.json'
        if not can_marker.exists() or json.loads(can_marker.read_text()).get('status') != 'complete':
            run.update('canonical_code-python')
            run.run(
                [str(py), '-u', str(scripts / 'canonical_v2.py'), '--source', 'code-python',
                 '--root', str(run.raw), '--output', str(run.root / 'canonical')],
                'canonical-code-python.log'
            )

        cln_marker = run.root / 'cleaned/code-python/COMPLETE.json'
        if not cln_marker.exists() or json.loads(cln_marker.read_text()).get('status') != 'complete':
            run.update('cleaned_code-python')
            run.run(
                [str(py), '-u', str(scripts / 'clean_v2.py'), '--source', 'code-python',
                 '--root', str(run.root)],
                'cleaned-code-python.log'
            )

        # Drive downstream pipeline stages: dedup -> near-dedup -> decontaminate -> tokenize -> finalize -> smoke
        run.update('pipeline')
        run.run(
            [str(py), '-u', str(scripts / 'continue_v2.py'), '--root', str(run.root),
             '--work', str(run.work), '--workers', str(args.workers), '--log-dir', str(run.logs)],
            'controller.log'
        )

        # Final coverage evaluation against 10B WSD mixture
        run.update('coverage')
        run.run(
            [str(py), str(scripts / 'assess_training_coverage.py'),
             '--root', str(run.root), '--report', str(run.reports / 'coverage.json')],
            'coverage.log'
        )

        coverage = json.loads((run.reports / 'coverage.json').read_text())
        is_sufficient = coverage['status'] == 'sufficient_fixed_mix'
        run.update(
            'complete' if is_sufficient else 'needs_supplement',
            status='complete' if is_sufficient else 'needs_supplement',
            coverage=coverage
        )
        print(f'Supplement pipeline finished with status: {coverage["status"]}', flush=True)

    except Exception as exc:
        run.update(run.status.get('stage', 'failed'), status='failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
