"""Reject incomplete or stale provenance chains before final manifest creation."""
import json
from pathlib import Path
import tempfile
import unittest

from canonical_v2 import SPECS, atomic_json, digest_file
from stage_audit import SCRIPTS, STAGES, verify_source_review, verify_stage_chain


class StageAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = 'fineweb-edu'
        self.benchmark = {'index_sha256': 'a' * 64, 'ngrams': {'prose': 1, 'code': 1}}
        previous = None
        for stage, script in zip(STAGES, SCRIPTS):
            directory = self.root / stage / self.source
            directory.mkdir(parents=True)
            part = directory / 'part-00000.jsonl'
            part.write_text('{}\n{}\n')
            report = {'status': 'complete', 'source': self.source,
                      'script_sha256': digest_file(Path(__file__).with_name(script)),
                      'counts': {'input': 2, 'kept': 2},
                      'parts': [{'path': str(part), 'bytes': part.stat().st_size,
                                 'sha256': digest_file(part), 'documents': 2}]}
            if stage == 'canonical':
                report.update(repo=SPECS[self.source][1], counts={'output_records': 2})
            else:
                report['upstream_sha256' if stage == 'tokenized' else 'upstream_report_sha256'] = previous
            if stage == 'decontaminated':
                report.update(index_sha256=self.benchmark['index_sha256'], index_ngrams=self.benchmark['ngrams'])
            if stage == 'tokenized':
                report.update(kind='pretrain', dtype='<u4',
                              counts={'documents': 2, 'train_documents': 1, 'validation_documents': 1})
            marker = directory / 'COMPLETE.json'
            atomic_json(marker, report)
            previous = digest_file(marker)

    def edit(self, stage, **fields):
        path = self.root / stage / self.source / 'COMPLETE.json'
        report = json.loads(path.read_text())
        report.update(fields)
        atomic_json(path, report)

    def verify(self):
        return verify_stage_chain(self.root, self.source, benchmark=self.benchmark)

    def test_complete_chain(self):
        self.assertEqual(len(self.verify()), 6)

    def test_changed_upstream_report(self):
        self.edit('canonical', extra='modified after cleaning')
        with self.assertRaisesRegex(ValueError, 'upstream report changed'):
            self.verify()

    def test_omitted_encoded_document(self):
        self.edit('tokenized', counts={'documents': 1, 'train_documents': 1})
        with self.assertRaisesRegex(ValueError, 'lost/extra encoded'):
            self.verify()

    def test_wrong_source(self):
        self.edit('tokenized', source='cosmopedia')
        with self.assertRaisesRegex(ValueError, 'wrong status/source'):
            self.verify()

    def test_different_processing_script(self):
        self.edit('tokenized', script_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'script differs'):
            self.verify()

    def test_changed_benchmark(self):
        self.benchmark['index_sha256'] = 'b' * 64
        with self.assertRaisesRegex(ValueError, 'different benchmark index'):
            self.verify()

    def test_truncated_part(self):
        (self.root / 'cleaned' / self.source / 'part-00000.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'part size mismatch'):
            self.verify()


class SourceReviewTests(unittest.TestCase):
    def test_unapproved_or_missing_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'Missing SOURCE_REVIEW'):
                verify_source_review(root)
            good = {'status': 'passed', 'sources': {source: {
                'training_approved': True, 'license_review_status': 'passed',
                'evidence': ['test-only evidence']} for source in SPECS}}
            for key, value, message in (
                ('training_approved', False, 'not approved'),
                ('license_review_status', 'pending', 'not passed'),
                ('evidence', [], 'missing provenance')):
                with self.subTest(key=key):
                    review = json.loads(json.dumps(good))
                    review['sources']['openhermes'][key] = value
                    atomic_json(root / 'SOURCE_REVIEW.json', review)
                    with self.assertRaisesRegex(ValueError, message):
                        verify_source_review(root)
            atomic_json(root / 'SOURCE_REVIEW.json', good)
            self.assertEqual(verify_source_review(root), good)


if __name__ == '__main__':
    unittest.main()
