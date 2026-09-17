"""Deterministically inspect unlabeled OpenHermes records without changing data."""
import argparse
from collections import Counter
import hashlib
import json
import random
import re
import statistics
from pathlib import Path

from canonical_v2 import atomic_json, digest_file, json_array


EMAIL = re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)')
SECRET = re.compile(
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|'
    r'\bAKIA[0-9A-Z]{16}\b|\bgh[pousr]_[A-Za-z0-9]{30,}\b'
)
ROLE_ALIASES = {'human': 'user', 'gpt': 'assistant', 'prompter': 'user'}
VALID_ROLES = {'system', 'user', 'assistant', 'tool'}


def inspect_row(row):
    """Return structural and safety findings for one raw row."""
    issues = []
    conversations = row.get('conversations')
    if not isinstance(conversations, list) or not conversations:
        return {'issues': ['missing_or_empty_conversations'], 'turns': 0,
                'chars': 0, 'roles': [], 'email_hits': 0, 'secret_hits': 0,
                'content_sha256': hashlib.sha256(b'').hexdigest()}
    roles = []
    texts = []
    for turn in conversations:
        if not isinstance(turn, dict):
            issues.append('turn_not_object')
            continue
        role = ROLE_ALIASES.get(turn.get('role', turn.get('from')),
                                turn.get('role', turn.get('from')))
        text = turn.get('content', turn.get('value'))
        roles.append(role)
        if role not in VALID_ROLES:
            issues.append('invalid_role')
        if not isinstance(text, str):
            issues.append('non_string_content')
            continue
        texts.append(text)
        if not text.strip():
            issues.append('empty_turn')
        if '\x00' in text:
            issues.append('nul_byte')
        if text.count('\ufffd') > max(2, len(text) // 100):
            issues.append('encoding_damage')
        if SECRET.search(text):
            issues.append('secret_pattern')
        if EMAIL.search(text):
            issues.append('email_pattern')
    if not any(role == 'user' for role in roles):
        issues.append('no_user_prompt')
    if roles and roles[-1] != 'assistant':
        issues.append('no_final_assistant')
    if any(a == b and a in ('user', 'assistant') for a, b in zip(roles, roles[1:])):
        issues.append('repeated_role')
    payload = json.dumps(conversations, sort_keys=True, ensure_ascii=False).encode()
    return {
        'issues': sorted(set(issues)), 'turns': len(conversations),
        'chars': sum(len(text) for text in texts), 'roles': roles,
        'email_hits': sum(len(EMAIL.findall(text)) for text in texts),
        'secret_hits': sum(len(SECRET.findall(text)) for text in texts),
        'content_sha256': hashlib.sha256(payload).hexdigest(),
    }


def excerpt(row, limit=180, issue=None):
    """Create a terminal-only, redacted excerpt; reports store no raw text."""
    turns = row.get('conversations', [])
    if issue == 'email_pattern':
        turns = [turn for turn in turns if isinstance(turn, dict) and
                 isinstance(turn.get('content', turn.get('value')), str) and
                 EMAIL.search(turn.get('content', turn.get('value')))] or turns
    elif issue == 'secret_pattern':
        turns = [turn for turn in turns if isinstance(turn, dict) and
                 isinstance(turn.get('content', turn.get('value')), str) and
                 SECRET.search(turn.get('content', turn.get('value')))] or turns
    elif issue == 'encoding_damage':
        turns = [turn for turn in turns if isinstance(turn, dict) and
                 isinstance(turn.get('content', turn.get('value')), str) and
                 '\ufffd' in turn.get('content', turn.get('value'))] or turns
    for turn in turns:
        text = turn.get('content', turn.get('value')) if isinstance(turn, dict) else ''
        if isinstance(text, str) and text.strip():
            text = EMAIL.sub('[EMAIL]', text)
            text = SECRET.sub('[SECRET]', text)
            return ' '.join(text.split())[:limit]
    return ''


def summarize(values):
    if not values:
        return {'count': 0}
    return {
        'count': len(values), 'min': min(values),
        'median': statistics.median(values), 'mean': round(statistics.mean(values), 2),
        'max': max(values),
    }


def run(input_path, report_path, sample_size, show_excerpts):
    source = Path(input_path)
    rng = random.Random(20260917)
    reservoir = []
    total = 0
    missing = 0
    labeled = 0
    issue_counts = Counter()
    issue_examples = {}
    role_sequences = Counter()
    sampled_details = {}
    with source.open(encoding='utf-8') as handle:
        for row_index, row in enumerate(json_array(handle)):
            total += 1
            label = row.get('source')
            if isinstance(label, str) and label.strip():
                labeled += 1
                continue
            missing += 1
            detail = inspect_row(row)
            for issue in detail['issues']:
                issue_counts[issue] += 1
                issue_examples.setdefault(issue, (row_index, row, detail))
            if len(reservoir) < sample_size:
                reservoir.append((row_index, row, detail))
            else:
                slot = rng.randrange(missing)
                if slot < sample_size:
                    reservoir[slot] = (row_index, row, detail)
    sample_rows = []
    for row_index, row, detail in sorted(reservoir):
        role_sequences['>'.join(detail['roles'])] += 1
        sample_rows.append({
            'row': row_index, 'content_sha256': detail['content_sha256'],
            'issues': detail['issues'], 'turns': detail['turns'],
            'chars': detail['chars'], 'roles': detail['roles'],
            'email_hits': detail['email_hits'], 'secret_hits': detail['secret_hits'],
        })
    eligible = sum(not item['issues'] for item in sample_rows)
    duplicate_hashes = len(sample_rows) - len({item['content_sha256'] for item in sample_rows})
    report = {
        'status': 'complete', 'scope': 'Read-only deterministic reservoir sample of raw records without a non-empty source label',
        'input_path': str(source), 'input_sha256': digest_file(source),
        'script_sha256': digest_file(__file__), 'records': total,
        'missing_source_label': missing, 'labeled_records': labeled,
        'sample_seed': 20260917, 'sample_size': len(sample_rows),
        'sample_eligible_by_structure_and_safety_checks': eligible,
        'sample_duplicate_content_hashes': duplicate_hashes,
        'all_missing_issue_counts': dict(sorted(issue_counts.items())),
        'sample_role_sequences': dict(role_sequences.most_common()),
        'sample_turns': summarize([item['turns'] for item in sample_rows]),
        'sample_chars': summarize([item['chars'] for item in sample_rows]),
        'sample_email_records': sum(item['email_hits'] > 0 for item in sample_rows),
        'sample_secret_records': sum(item['secret_hits'] > 0 for item in sample_rows),
        'sample_records': sample_rows,
        'limitations': [
            'Content sampling can assess structure, quality and obvious safety patterns only.',
            'It cannot establish upstream provenance or license for records without a source label.',
            'The current cleaner checks explicit secret patterns but does not prove complete PII removal.',
        ],
    }
    atomic_json(report_path, report)
    print(json.dumps({k: report[k] for k in (
        'records', 'missing_source_label', 'sample_size',
        'sample_eligible_by_structure_and_safety_checks', 'all_missing_issue_counts',
        'sample_email_records', 'sample_secret_records', 'sample_duplicate_content_hashes')},
        ensure_ascii=False))
    if show_excerpts:
        for row_index, row, detail in sorted(reservoir)[:show_excerpts]:
            print(json.dumps({'row': row_index, 'roles': detail['roles'],
                              'issues': detail['issues'], 'excerpt': excerpt(row)},
                             ensure_ascii=False))
    if show_excerpts:
        for issue in sorted(issue_examples):
            row_index, row, detail = issue_examples[issue]
            print(json.dumps({'issue_example': issue, 'row': row_index,
                              'roles': detail['roles'], 'excerpt': excerpt(row, issue=issue)},
                             ensure_ascii=False))


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
