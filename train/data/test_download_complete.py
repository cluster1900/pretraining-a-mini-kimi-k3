"""Regression tests for full downloads: paging, checksums and completion state."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import importlib.util
spec = importlib.util.spec_from_file_location("download_complete", Path(__file__).with_name("download_complete.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)

class DownloadTests(unittest.TestCase):
    def test_pagination_includes_second_page(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'raw'
            calls=[({'sha':'abc','cardData':{'license':'test'}},None),
                   ([{'type':'file','path':'a','size':1,'oid':'a'}],d.MIRROR+'/next'),
                   ([{'type':'file','path':'b','size':2,'oid':'b'}],None)]
            with patch.object(d,'api_get',side_effect=calls):
                d.inventory({'name':'x','directory':'x','repo':'a/b','backend':'hf-mirror'},root)
            inv=json.loads((root/'x'/'INVENTORY.json').read_text())
            self.assertEqual(inv['total_bytes'],3)
            self.assertEqual([f['path'] for f in inv['files']],['a','b'])
    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'a'; p.write_bytes(b'data')
            with self.assertRaises(ValueError):
                d.checksum(p,{'path':'a','sha256':'0'*64})
            git=hashlib.sha1(b'blob 4\0data').hexdigest()
            self.assertEqual(d.checksum(p,{'path':'a','git_oid':git}),hashlib.sha256(b'data').hexdigest())
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError): d.safe_path('/tmp/x','../outside')
    def test_interrupted_download_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'raw'; folder=root/'x';folder.mkdir(parents=True)
            source={'name':'x','directory':'x','repo':'a/b','backend':'hf-mirror'}
            d.atomic(folder/'INVENTORY.json',dict(source,revision='abc',license='test',
                content_manifest_sha256='hash',total_bytes=1,files=[{'path':'a','size':1}]))
            with patch.object(d,'fetch',side_effect=OSError('network failure')),patch.object(d.time,'sleep'):
                d.download(source,root,0)
            self.assertFalse((folder/'FULL_DOWNLOAD_COMPLETE.json').exists())
            self.assertEqual(json.loads((root.parent/'reports/downloads/x.json').read_text())['status'],'failed')

if __name__=='__main__': unittest.main()
