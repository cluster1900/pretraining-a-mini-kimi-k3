"""Deterministically inspect the provenance-filtered OpenHermes candidate."""
import argparse
from collections import Counter
import json
import random
import statistics
from pathlib import Path

from canonical_v2 import atomic_json, digest_file, json_array
from sample_openhermes_missing_source import inspect_row, excerpt


def summarize(values):
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': min(values),
            'median': statistics.median(values), 'mean': round(statistics.mean(values), 2),
            'max': max(values)}


def run(input_path, report_path, sample_size, show_excerpts):
    source = Path(input_path)
    rng = random.Random(20260919)
    reservoir = []
    total = 0
    source_counts = Counter()
    issue_counts = Counter()
    with source.open(encoding='utf-8') as handle:
        for row_index, row in enumerate(json_array(handle)):
            total += 1
            label = row.get('source')
            if label not in ('glaive-code-assist', 'metamath'):
                continue
            source_counts[label] += 1
            detail = inspect_row(row)
            for issue in detail['issues']:
                issue_counts[issue] += 1
            if len(reservoir) < sample_size:
                reservoir.append((row_index, row, detail))
            else:
                slot = rng.randrange(sum(source_counts.values()))
                if slot < sample_size:
                    reservoir[slot] = (row_index, row, detail)
    sample_rows = []
    for row_index, row, detail in sorted(reservoir):
        sample_rows.append({
            'row': row_index, 'source': row.get('source'),
            'content_sha256': detail['content_sha256'], 'issues': detail['issues'],
            'turns': detail['turns'], 'chars': detail['chars'],
            'roles': detail['roles'], 'email_hits': detail['email_hits'],
            'secret_hits': detail['secret_hits'],
        })
    report = {
        'status': 'complete',
        'scope': 'Read-only deterministic sample of the provenance-filtered OpenHermes candidate',
        'input_path': str(source), 'input_sha256': digest_file(source),
        'script_sha256': digest_file(__file__), 'records': total,
        'source_counts': dict(source_counts), 'sample_seed': 20260919,
        'sample_size': len(sample_rows), 'sample_records': sample_rows,
        'sample_eligible_by_structure_and_safety_checks': sum(not x['issues'] for x in sample_rows),
        'all_retained_issue_counts': dict(sorted(issue_counts.items())),
        'sample_turns': summarize([x['turns'] for x in sample_rows]),
        'sample_chars': summarize([x['chars'] for x in sample_rows]),
        'sample_email_records': sum(x['email_hits'] > 0 for x in sample_rows),
        'sample_secret_records': sum(x['secret_hits'] > 0 for x in sample_rows),
        'sample_duplicate_content_hashes': len(sample_rows) - len({x['content_sha256'] for x in sample_rows}),
        'limitations': [
            'This checks structure and obvious patterns; it does not certify factual correctness.',
            'It does not replace per-source license evidence or human review.',
        ],
    }
    atomic_json(report_path, report)
    print(json.dumps({k: report[k] for k in ('records', 'source_counts', 'sample_size',
                                              'sample_eligible_by_structure_and_safety_checks',
                                              'all_retained_issue_counts')}, ensure_ascii=False))
    if show_excerpts:
        for row_index, row, detail in sorted(reservoir)[:show_excerpts]:
            print(json.dumps({'row': row_index, 'source': row.get('source'),
                              'roles': detail['roles'], 'issues': detail['issues'],
                              'excerpt': excerpt(row)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--sample-size', type=int, default=256)
    parser.add_argument('--show-excerpts', type=int, default=0)
    args = parser.parse_args()
    if args.sample_size < 1 or args.show_excerpts < 0:
        raise SystemExit('sample-size must be positive and show-excerpts non-negative')
    run(args.input, args.report, args.sample_size, args.show_excerpts)


if __name__ == '__main__':
    main()
