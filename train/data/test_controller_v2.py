"""A duplicate controller must not corrupt status; absent approval stops work."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ControllerTests(unittest.TestCase):
    def run_controller(self, root):
        return subprocess.run([sys.executable, str(Path(__file__).with_name('continue_v2.py')),
                               '--root', str(root), '--work', str(root)],
                              capture_output=True, text=True, timeout=15)

    def test_duplicate_cannot_overwrite_live_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            status = root / 'PIPELINE_STATUS.json'
            status.write_text('{"status":"running","stage":"model_smoke","pid":123}')
            expected = status.read_bytes()
            with (root / 'controller.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = self.run_controller(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(status.read_bytes(), expected)

    def test_absent_review_is_reported_before_any_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_controller(root)
            self.assertNotEqual(result.returncode, 0)
            status = json.loads((root / 'PIPELINE_STATUS.json').read_text())
            self.assertEqual((status['status'], status['stage']), ('failed', 'source_review'))
            self.assertIn('Missing SOURCE_REVIEW', status['error'])
            self.assertFalse((root / 'dedup-index-v2.sqlite').exists())


if __name__ == '__main__':
    unittest.main()
