"""Check report hash links and document conservation across the v2 pipeline.

This checks metadata and file sizes. The finalizer separately hashes and scans
all token outputs; this module does not claim to re-scan the raw corpus.
"""
import argparse
import json
from pathlib import Path

from canonical_v2 import SPECS, atomic_json, digest_file

STAGES = ('canonical', 'cleaned', 'deduped', 'near-deduped',
          'decontaminated', 'tokenized')
SCRIPTS = ('canonical_v2.py', 'clean_v2.py', 'dedup_v2.py',
           'near_dedup_v2.py', 'contamination_v2.py', 'tokenize_v2.py')
BENCHMARKS = {'mmlu', 'arc', 'hellaswag', 'winogrande', 'piqa', 'gsm8k', 'humaneval'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_benchmark(root):
    path = Path(root) / 'benchmarks/13grams.json.gz'
    report = json.loads(path.with_suffix('.report.json').read_text())
    require(report.get('status') == 'complete' and report.get('n') == 13,
            'Benchmark index is not a completed 13-gram index')
    require(set(report.get('rows', {})) == BENCHMARKS,
            'Benchmark index must contain the seven required evaluations')
    require(all(0 < report.get('covered_rows', {}).get(b, 0) <= report['rows'][b]
                for b in BENCHMARKS), 'Missing benchmark coverage')
    require(all(report.get('ngrams', {}).get(m, 0) > 0 for m in ('prose', 'code')),
            'Empty benchmark matching mode')
    require(digest_file(path) == report.get('index_sha256'), 'Benchmark index hash mismatch')
    return report


def verify_source_review(root):
    """Require an explicit, per-source provenance/license approval record."""
    path = Path(root) / 'SOURCE_REVIEW.json'
    require(path.is_file(), 'Missing SOURCE_REVIEW.json')
    review = json.loads(path.read_text())
    require(review.get('status') == 'passed', 'Source review is not passed')
    sources = review.get('sources', {})
    require(set(sources) == set(SPECS), 'Source review does not cover all sources')
    for source, item in sources.items():
        require(item.get('training_approved') is True,
                source + ': source is not approved for training')
        require(item.get('license_review_status') == 'passed',
                source + ': license review is not passed')
        require(item.get('evidence'), source + ': missing provenance/license evidence')
    return review


def verify_stage_chain(root, source, through='tokenized', benchmark=None):
    root = Path(root).resolve()
    require(source in SPECS, 'Unexpected source: ' + source)
    require(through in STAGES, 'Unexpected stage: ' + through)
    previous_hash = None
    previous_documents = None
    evidence = {}
    for stage, script in zip(STAGES, SCRIPTS):
        path = root / stage / source / 'COMPLETE.json'
        report = json.loads(path.read_text())
        label = source + '/' + stage
        require(report.get('status') == 'complete' and report.get('source') == source,
                label + ': wrong status/source')
        require(report.get('script_sha256') == digest_file(Path(__file__).parent / script),
                label + ': processing script differs from recorded hash')
        if previous_hash is not None:
            field = 'upstream_sha256' if stage == 'tokenized' else 'upstream_report_sha256'
            require(report.get(field) == previous_hash, label + ': upstream report changed')
        counts = report['counts']
        if stage == 'canonical':
            require(report.get('repo') == SPECS[source][1], label + ': repository mismatch')
            documents = counts['output_records']
        elif stage == 'tokenized':
            documents = counts['documents']
            require(documents == previous_documents, label + ': lost/extra encoded documents')
            require(documents == counts.get('train_documents', 0) + counts.get('validation_documents', 0),
                    label + ': split counts do not balance')
            kind = 'pretrain' if SPECS[source][2] in ('pretrain', 'code') else SPECS[source][2]
            require(report.get('kind') == kind, label + ': wrong output kind')
            require(report.get('dtype') == ('<u4' if kind == 'pretrain' else 'jsonl'),
                    label + ': wrong output dtype')
        else:
            documents = counts['kept']
            require(counts['input'] == previous_documents, label + ': upstream document count mismatch')
            rejected = (sum(n for k, n in counts.items() if k.startswith('reject_'))
                        if stage == 'cleaned' else counts.get(
                            {'deduped': 'duplicates', 'near-deduped': 'near_duplicates',
                             'decontaminated': 'removed'}[stage], 0))
            require(counts['input'] == documents + rejected, label + ': counters do not balance')
        require(isinstance(documents, int) and documents > 0, label + ': empty/invalid document count')
        if stage != 'tokenized':
            parts = report['parts']
            require(sum(p['documents'] for p in parts) == documents,
                    label + ': part document counts do not balance')
            paths = [Path(p['path']).resolve() for p in parts]
            require(len(paths) == len(set(paths)), label + ': duplicate part paths')
            for p, item in zip(paths, parts):
                require(p.parent == path.parent and not p.name.endswith('.incomplete'),
                        label + ': part outside expected source directory')
                require(p.stat().st_size == item['bytes'], label + ': part size mismatch')
                require(isinstance(item.get('sha256'), str) and len(item['sha256']) == 64,
                        label + ': missing part hash')
        if stage == 'decontaminated':
            if benchmark is None:
                benchmark = verify_benchmark(root)
            require(report.get('index_sha256') == benchmark['index_sha256']
                    and report.get('index_ngrams') == benchmark['ngrams'],
                    label + ': different benchmark index')
        previous_hash = digest_file(path)
        previous_documents = documents
        evidence[stage] = {'report_sha256': previous_hash, 'documents': documents}
        if stage == through:
            break
    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--through', choices=STAGES, default='tokenized')
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    result = {'status': 'passed', 'through': args.through, 'sources': {}, 'errors': {},
              'scope': 'Report hash chain, script identity, counts and part sizes; not a full corpus re-scan',
              'script_sha256': digest_file(__file__)}
    for source in SPECS:
        try:
            result['sources'][source] = verify_stage_chain(args.root, source, args.through)
        except (ValueError, KeyError, OSError) as exc:
            result['status'] = 'failed'
            result['errors'][source] = str(exc)
    atomic_json(args.report, result)
    print(json.dumps({'status': result['status'], 'sources': len(result['sources']),
                      'errors': result['errors']}))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
