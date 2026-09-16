"""Inventory raw subset labels without changing any production stage output."""
import argparse
from collections import Counter
import json
from pathlib import Path

from canonical_v2 import atomic_json, digest_file, json_array


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True)
    p.add_argument('--report', required=True)
    a = p.parse_args()
    source = Path(a.input)
    labels = Counter()
    missing = 0
    total = 0
    missing_schemas = Counter()
    missing_examples = {}
    alternate_fields = Counter()
    with source.open(encoding='utf-8') as handle:
        for row in json_array(handle):
            total += 1
            label = row.get('source')
            if not isinstance(label, str) or not label.strip():
                missing += 1
                schema = '|'.join(sorted(row))
                missing_schemas[schema] += 1
                missing_examples.setdefault(schema, total - 1)
                for key in ('dataset', 'dataset_name', 'subset', 'origin', 'source_name'):
                    if isinstance(row.get(key), str) and row[key].strip():
                        alternate_fields[key] += 1
            else:
                labels[label] += 1
    report = {
        'status': 'inventory_complete', 'license_review_status': 'pending',
        'input_path': str(source), 'input_sha256': digest_file(source),
        'script_sha256': digest_file(__file__), 'records': total,
        'missing_source_label': missing, 'source_counts': dict(sorted(labels.items())),
        'missing_source_schemas': dict(sorted(missing_schemas.items())),
        'missing_source_schema_first_row': missing_examples,
        'missing_source_alternate_label_fields': dict(alternate_fields),
        'scope': 'Raw source label inventory only; does not approve any subset license',
        'publisher_license_statement': 'https://huggingface.co/datasets/teknium/OpenHermes-2.5/discussions/9',
    }
    atomic_json(a.report, report)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
