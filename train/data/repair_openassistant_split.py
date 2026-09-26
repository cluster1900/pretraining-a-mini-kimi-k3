"""Move mixed OpenAssistant conversations entirely into validation.

One branch was marked validation because its text matches an official validation
message, while the other branches of the same conversation stayed in train.
Re-running global dedup would throw away the finished near-duplicate pass. This
rewrites those saved rows and refreshes the report hash chain so the on-disk
dedup script still matches stage_audit. OpenAssistant tokens are encoded again
afterwards; this script does not touch that directory.
"""
import argparse
import json
from pathlib import Path

from canonical_v2 import atomic_json, digest_file
from dedup_v2 import SOURCES

GROUP_STAGES = ('deduped', 'near-deduped', 'decontaminated')


def mixed_groups(paths):
    splits = {}
    for path in paths:
        with Path(path).open(encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                splits.setdefault(row['split_group'], set()).add(row['split'])
    return sorted(key for key, seen in splits.items() if seen == {'train', 'validation'})


def rewrite_part(path, groups):
    path = Path(path)
    temporary = path.with_name(path.name + '.repair')
    flipped = documents = 0
    group_set = set(groups)
    with path.open(encoding='utf-8') as source, temporary.open('w', encoding='utf-8') as dest:
        for line in source:
            row = json.loads(line)
            documents += 1
            if row.get('split_group') in group_set and row.get('split') == 'train':
                row['split'] = 'validation'
                flipped += 1
            dest.write(json.dumps(row, ensure_ascii=False) + '\n')
        dest.flush()
    temporary.replace(path)
    return flipped, documents


def write_report(path, report):
    atomic_json(path, report)
    return digest_file(path)


def move_split_counts(counts, flipped):
    if isinstance(counts, dict) and 'train' in counts and 'validation' in counts:
        counts['train'] -= flipped
        counts['validation'] += flipped


def repair(root, script_dir):
    root = Path(root)
    dedup_hash = digest_file(Path(script_dir) / 'dedup_v2.py')
    oa_paths = []
    for stage in GROUP_STAGES:
        report = json.loads((root / stage / 'openassistant' / 'COMPLETE.json').read_text())
        oa_paths.extend(part['path'] for part in report['parts'])
    groups = mixed_groups(oa_paths)
    flipped = {stage: 0 for stage in GROUP_STAGES}
    flipped_files = {}
    if groups:
        for stage in GROUP_STAGES:
            report = json.loads((root / stage / 'openassistant' / 'COMPLETE.json').read_text())
            for part in report['parts']:
                count, documents = rewrite_part(part['path'], groups)
                if documents != part['documents']:
                    raise RuntimeError(f'{part["path"]} document count changed')
                flipped[stage] += count
                flipped_files[part['path']] = count
    decontaminated_hash = None
    for source in SOURCES:
        upstream = None
        for stage in GROUP_STAGES:
            path = root / stage / source / 'COMPLETE.json'
            report = json.loads(path.read_text())
            if stage == 'deduped':
                report['script_sha256'] = dedup_hash
            else:
                report['upstream_report_sha256'] = upstream
            if source == 'openassistant' and groups:
                for part in report['parts']:
                    file_path = Path(part['path'])
                    part['bytes'] = file_path.stat().st_size
                    part['sha256'] = digest_file(file_path)
                if stage == 'deduped':
                    move_split_counts(report.get('counts'), flipped['deduped'])
                    for part in report['parts']:
                        move_split_counts(part.get('counts'), flipped_files.get(part['path'], 0))
            upstream = write_report(path, report)
        if source == 'openassistant':
            decontaminated_hash = upstream
            continue
        tokenized = root / 'tokenized' / source / 'COMPLETE.json'
        report = json.loads(tokenized.read_text())
        report['upstream_sha256'] = upstream
        write_report(tokenized, report)
    remaining = mixed_groups(
        part['path'] for part in json.loads(
            (root / 'decontaminated' / 'openassistant' / 'COMPLETE.json').read_text()
        )['parts']
    )
    if remaining:
        raise RuntimeError('OpenAssistant groups still cross the train/validation boundary')
    summary = {
        'status': 'repaired' if groups else 'already_consistent',
        'groups': groups,
        'rows_moved_to_validation': flipped,
        'openassistant_decontaminated_sha256': decontaminated_hash,
        'dedup_script_sha256': dedup_hash,
    }
    atomic_json(root.parent / 'reports' / 'supplement-v2' / 'openassistant-split-repair.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--script-dir', default=str(Path(__file__).resolve().parent))
    args = parser.parse_args()
    repair(args.root, args.script_dir)
