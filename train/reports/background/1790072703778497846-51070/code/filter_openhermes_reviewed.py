"""Create a provenance-filtered OpenHermes candidate without changing raw data."""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

from canonical_v2 import atomic_json, digest_file, json_array


# These are the only OpenHermes sub-sources with a directly identifiable,
# permissive upstream dataset license recorded in the review evidence.
VERIFIED_SUBSOURCES = {
    'glaive-code-assist': {
        'upstream': 'glaiveai/glaive-code-assistant-v2',
        'license': 'Apache-2.0',
        'evidence': 'https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v2',
    },
    'metamath': {
        'upstream': 'meta-math/MetaMathQA',
        'license': 'MIT',
        'evidence': 'https://huggingface.co/datasets/meta-math/MetaMathQA',
    },
}
EMAIL = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)')


def redact_emails(row):
    """Redact email-shaped strings in retained conversation content."""
    hits = 0
    for turn in row.get('conversations', []):
        if not isinstance(turn, dict):
            continue
        key = 'content' if isinstance(turn.get('content'), str) else 'value'
        text = turn.get(key)
        if isinstance(text, str):
            hits += len(EMAIL.findall(text))
            turn[key] = EMAIL.sub('[EMAIL]', text)
    return hits


def filter_rows(input_path, output_path, report_path):
    source = Path(input_path)
    output = Path(output_path)
    if output.exists() or output.with_name(output.name + '.incomplete').exists():
        raise ValueError('Filtered output already exists; choose a new output path')
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    kept = Counter()
    temp = output.with_name(output.name + '.incomplete')
    first = True
    try:
        with source.open(encoding='utf-8') as handle, temp.open('w', encoding='utf-8') as out:
            out.write('[')
            for row_index, row in enumerate(json_array(handle)):
                counts['records'] += 1
                label = row.get('source')
                if not isinstance(label, str) or not label.strip():
                    counts['excluded_missing_source'] += 1
                    continue
                if label not in VERIFIED_SUBSOURCES:
                    counts['excluded_unverified_subsource'] += 1
                    continue
                if not isinstance(row.get('conversations'), list) or not row['conversations']:
                    counts['excluded_invalid_conversations'] += 1
                    continue
                counts['redacted_email_occurrences'] += redact_emails(row)
                if not first:
                    out.write(',')
                json.dump(row, out, ensure_ascii=False, separators=(',', ':'))
                first = False
                counts['retained'] += 1
                kept[label] += 1
            out.write(']')
            out.flush()
        temp.replace(output)
    finally:
        if temp.exists():
            temp.unlink()
    result = {
        'status': 'complete', 'scope': 'Retain only OpenHermes rows with an exact verified subsource label',
        'input_path': str(source), 'input_sha256': digest_file(source),
        'output_path': str(output), 'output_sha256': digest_file(output),
        'script_sha256': digest_file(__file__), 'counts': dict(counts),
        'retained_subsources': {name: dict(VERIFIED_SUBSOURCES[name], records=kept[name])
                               for name in sorted(kept)},
        'filter_rule': 'Keep source exactly equal to glaive-code-assist or metamath; redact email-shaped strings; exclude missing and all other labels.',
        'excluded_records_are_preserved_in_original_input': True,
    }
    atomic_json(report_path, result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    filter_rows(args.input, args.output, args.report)


if __name__ == '__main__':
    main()
