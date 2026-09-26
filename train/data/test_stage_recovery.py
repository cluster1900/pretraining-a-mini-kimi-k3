"""Regression tests for crash recovery, split grouping and streaming shards."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from canonical_v2 import atomic_json,digest_file
from dedup_v2 import content_key,deduplicate,group_key,assigned_split
from tokenize_v2 import TokenWriter,encode_messages,encode_preference,encode_source

class FakeTokenizer:
    eos_token_id=255;vocab_size=300;fingerprint='fixture';asset_hashes={}
    def __init__(self,*args):pass
    def encode(self,text,append_eos=True):return [ord(c)%100+1 for c in text]+([255] if append_eos else [])

class RecoveryTests(unittest.TestCase):
    def fixture(self,root,source,texts):
        out=root/'cleaned'/source;out.mkdir(parents=True)
        rows=[]
        for i,text in enumerate(texts):
            row={'id':source+str(i),'source':source,'repo':'fixture','kind':'text','text':text}
            row['content_sha256']=content_key(row);rows.append(row)
        path=out/'part-00000.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        part={'path':str(path),'bytes':path.stat().st_size,'sha256':digest_file(path),'documents':len(rows)}
        atomic_json(out/'COMPLETE.json',{'parts':[part]})
    def test_crash_does_not_delete_records_on_resume(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root,'a',['first','second']);self.fixture(root,'b',['first','third'])
            def crash(*args):raise RuntimeError('injected crash before SQLite commit')
            with self.assertRaises(RuntimeError):deduplicate(root,['a','b'],after_part=crash)
            deduplicate(root,['a','b'])
            a=json.loads((root/'deduped/a/COMPLETE.json').read_text());b=json.loads((root/'deduped/b/COMPLETE.json').read_text())
            self.assertEqual(a['counts']['kept'],2);self.assertEqual(b['counts']['kept'],1)
            deduplicate(root,['a','b'])
            self.assertEqual(json.loads((root/'deduped/a/COMPLETE.json').read_text())['counts']['kept'],2)
    def test_changed_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root,'a',['one']);deduplicate(root,['a'])
            p=root/'cleaned/a/COMPLETE.json';data=json.loads(p.read_text());data['changed']=True;atomic_json(p,data)
            with self.assertRaises(ValueError):deduplicate(root,['a'])
    def test_same_prompt_different_sft_answers_share_split(self):
        a={'kind':'sft','messages':[{'role':'user','content':'question'},{'role':'assistant','content':'one'}]}
        b={'kind':'sft','messages':[{'role':'user','content':'question'},{'role':'assistant','content':'two'}]}
        self.assertEqual(group_key(a),group_key(b));self.assertEqual(assigned_split(group_key(a)),assigned_split(group_key(b)))
    def test_validation_text_promotes_the_whole_conversation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);out=root/'cleaned'/'openassistant';out.mkdir(parents=True)
            rows=[]
            for i,text,tree,official in (
                (0,'other branch','tree-train','train'),
                (1,'held out wording','tree-train','train'),
                (2,'held out wording','tree-val','validation'),
            ):
                row={'id':f'id{i}','source':'openassistant','repo':'OpenAssistant/oasst1','kind':'sft',
                     'messages':[{'role':'user','content':'q'},{'role':'assistant','content':text}],
                     'group_id':tree,'official_split':official}
                row['content_sha256']=content_key(row);rows.append(row)
            path=out/'part-00000.jsonl'
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
            part={'path':str(path),'bytes':path.stat().st_size,'sha256':digest_file(path),'documents':len(rows)}
            atomic_json(out/'COMPLETE.json',{'parts':[part]})
            deduplicate(root,['openassistant'])
            kept=[json.loads(line) for line in (root/'deduped'/'openassistant'/'part-00000.jsonl').read_text().splitlines()]
            promoted=[row for row in kept if row['group_id']=='tree-train']
            self.assertEqual(len(promoted),2)
            self.assertEqual({row['split'] for row in promoted},{'validation'})
    def test_uint32_cross_shard_boundary(self):
        with tempfile.TemporaryDirectory() as td:
            w=TokenWriter(td,3);w.write([1,2,3,4,255],255);w.write([9,255],255);w.close()
            self.assertEqual([x['tokens'] for x in w.parts],[3,3,1]);self.assertEqual(sum(x['eos'] for x in w.parts),2)
            self.assertEqual(sum(Path(x['path']).stat().st_size for x in w.parts),28)
    def test_prompt_labels_are_masked(self):
        t=FakeTokenizer();ids,labels=encode_messages([{'role':'user','content':'prompt'},{'role':'assistant','content':'answer'}],t)
        self.assertEqual(len(ids),len(labels));self.assertEqual(labels[-1],255)
        self.assertTrue(all(x==-100 for x in labels[:len(t.encode('<|user|>\nprompt\n',False))]))
        p=encode_preference({'prompt':[{'role':'user','content':'prompt'}],'chosen':'yes','rejected':'no'},t)
        self.assertEqual(p['chosen_ids'][:p['prompt_len']],p['rejected_ids'][:p['prompt_len']])
    def test_first_bin_is_not_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);src=root/'decontaminated/fineweb-edu';src.mkdir(parents=True);atomic_json(src/'COMPLETE.json',{'parts':[]})
            out=root/'tokenized/fineweb-edu';out.mkdir(parents=True);(out/'partial.bin').write_bytes(b'1234')
            with patch('tokenize_v2.K3Tokenizer',FakeTokenizer):
                with self.assertRaises(ValueError):encode_source('fineweb-edu',root,'unused')

if __name__=='__main__':unittest.main()
