"""Apply the user's explicit 2026-09-17 approval to the exact reviewed subset.

This is a one-time registration of approval already given in the project task,
not an automatic license or content-quality assessment.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from canonical_v2 import atomic_json, digest_file
from stage_audit import require, verify_stage_chain

APPROVAL_TEXT = '批准这两个子集进入 SFT，接下来自动执行就可以了，我们不用实时监控着'
ORIGINAL_SHA = 'abe573d17eade4161aac321028027dd5ba614a6d9516d51bde9299d5353e1609'
FILTERED_SHA = '3a6a1610d8ce9a62788aa9d0b5a170bf0b265587b2f89215c3179d0959b7d25d'
SUBSETS = {'glaive-code-assist': 182240, 'metamath': 56448}


def approve(work):
    work = Path(work).resolve()
    root = work / 'data/prepared-v2'
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        review_path = root / 'SOURCE_REVIEW.json'
        previous = json.loads(review_path.read_text())
        filter_path = work / 'data/reports/openhermes-reviewed-filter-20260917-v2.json'
        sample_path = work / 'data/reports/openhermes-reviewed-sample-20260917-v2.json'
        filtered = json.loads(filter_path.read_text())
        sample = json.loads(sample_path.read_text())
        require(filtered['status'] == sample['status'] == 'complete', 'Review reports incomplete')
        require(filtered['input_sha256'] == ORIGINAL_SHA, 'Unexpected original OpenHermes version')
        require(filtered['output_sha256'] == sample['input_sha256'] == FILTERED_SHA,
                'Unexpected reviewed OpenHermes version')
        require({s: v['records'] for s, v in filtered['retained_subsources'].items()} == SUBSETS,
                'Approved subset counts differ')
        require(sample['source_counts'] == SUBSETS, 'Sample report scope differs')
        require(filtered['counts'] == {
            'records': 1001551, 'retained': 238688, 'excluded_missing_source': 496743,
            'excluded_unverified_subsource': 266120, 'redacted_email_occurrences': 3489},
            'Filter counters differ from the approved candidate')
        paths = [filter_path, sample_path, Path(filtered['input_path']), Path(filtered['output_path']),
                 Path(__file__).with_name('filter_openhermes_reviewed.py'),
                 Path(__file__).with_name('sample_openhermes_reviewed.py')]
        bindings = [{'path': str(p), 'sha256': digest_file(p)} for p in paths]
        require(bindings[2]['sha256'] == ORIGINAL_SHA and bindings[3]['sha256'] == FILTERED_SHA,
                'Raw/filtered input hash mismatch')
        require(bindings[4]['sha256'] == filtered['script_sha256']
                and bindings[5]['sha256'] == sample['script_sha256'], 'Review script hash mismatch')
        selected = work / 'data/raw/openhermes/openhermes2_5.json'
        require(selected.resolve() == Path(filtered['output_path']).resolve(), 'Selected raw input changed')
        chain = verify_stage_chain(root, 'openhermes', through='cleaned')
        canonical = json.loads((root / 'canonical/openhermes/COMPLETE.json').read_text())
        cleaned = json.loads((root / 'cleaned/openhermes/COMPLETE.json').read_text())
        require(len(canonical['inputs']) == 1 and canonical['inputs'][0]['sha256'] == FILTERED_SHA,
                'Canonical source is not the approved filtered input')
        require(chain['canonical']['documents'] == 238688 and chain['cleaned']['documents'] == 238672,
                'Approved candidate stage counts differ')
        require(cleaned['counts'].get('reject_encoding_damage') == 5
                and cleaned['counts'].get('reject_secret_pattern') == 11, 'Cleaning counters differ')
        evidence = [str(filter_path), str(sample_path),
                    'https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v2',
                    'https://huggingface.co/datasets/meta-math/MetaMathQA']
        item = previous['sources']['openhermes']
        item.update(training_approved=True, license_review_status='passed',
                    approval={'kind': 'explicit_user_approval', 'date': '2026-09-17',
                              'text': APPROVAL_TEXT, 'scope': 'SFT for these two filtered subsets only'},
                    license='Apache-2.0 (Glaive publisher declaration); MIT (MetaMathQA publisher declaration)',
                    evidence=evidence, file_bindings=bindings, retained_subsources=SUBSETS,
                    excluded_records={'missing_source': 496743, 'other_subsources': 266120},
                    quality_review_status='basic_checks_and_limited_sampling_only',
                    stage_report_sha256={s: e['report_sha256'] for s, e in chain.items()},
                    filtered_input_sha256=FILTERED_SHA, cleaned_documents=238672,
                    filter_rule=filtered['filter_rule'],
                    limitations=['Approval is limited to this candidate, not the original compilation.',
                                 'Sampling does not prove all answers correct; Glaive version mapping is based on publisher labels, not upstream row matching.'])
        item.pop('reason', None)
        item.pop('next_action', None)
        previous.update(schema_version=2, status='passed',
                        generated_at=datetime.now(timezone.utc).isoformat(),
                        scope='Selected sources; OpenHermes limited to the user-approved, hash-bound Glaive/MetaMath SFT candidate')
        backup = root / 'SOURCE_REVIEW.before-explicit-approval-20260917.json'
        if not backup.exists():
            atomic_json(backup, json.loads(review_path.read_text()))
        atomic_json(review_path, previous)
        print(json.dumps({'status': 'approval_recorded', 'review_sha256': digest_file(review_path),
                          'cleaned_documents': 238672, 'retained_subsources': SUBSETS}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', default='/data/mini-k3')
    approve(parser.parse_args().work)
