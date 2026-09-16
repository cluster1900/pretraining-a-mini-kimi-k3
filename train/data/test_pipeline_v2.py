import importlib.util
import io
from pathlib import Path
import unittest
from collections import Counter

spec=importlib.util.spec_from_file_location('canonical_v2',Path(__file__).with_name('canonical_v2.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class SchemaTests(unittest.TestCase):
    def test_json_array_stream(self):
        self.assertEqual(list(m.json_array(io.StringIO('[{"x":1}, {"x":2}]'))),[{'x':1},{'x':2}])
        with self.assertRaises(ValueError): list(m.json_array(io.StringIO('[{}]garbage')))
        with self.assertRaises(ValueError): list(m.json_array(io.StringIO('[{},]')))
    def test_code_keeps_indentation_and_metadata(self):
        code='def f():\n    return 1\n'
        row={'repo_path':'a/b','files':[{'content':code,'language':'Python','file_path':'f.py','license_type':'MIT','is_vendor':False}]}
        r=list(m.adapt(row,'code-python',{'file':'x','row':0},Counter()))[0]
        self.assertEqual(r['text'],code);self.assertEqual(r['license_type'],'MIT');self.assertEqual(r['file_path'],'f.py')
    def test_sft_does_not_discard_answer(self):
        r=list(m.adapt({'problem':'p','messages':[{'role':'user','content':'p'},{'role':'assistant','content':'solution'}]},'openr1',{},Counter()))[0]
        self.assertEqual(r['messages'][1]['content'],'solution')
    def test_preference_keeps_both_responses(self):
        prompt={'role':'user','content':'question'}
        r=list(m.adapt({'chosen':[prompt,{'role':'assistant','content':'yes'}],'rejected':[prompt,{'role':'assistant','content':'no'}]},'ultrafeedback',{},Counter()))[0]
        self.assertEqual(r['chosen'],'yes');self.assertEqual(r['rejected'],'no');self.assertEqual(r['prompt'],[prompt])
    def test_bad_pair_rejected(self):
        with self.assertRaises(ValueError):list(m.adapt({'chosen':[],'rejected':[]},'ultrafeedback',{},Counter()))

class CleaningTests(unittest.TestCase):
    def test_python_whitespace_survives(self):
        import clean_v2
        source={'source':'code-python','kind':'code','license_type':'MIT','text':'def f():\n    return 1\n'}
        cleaned,reason=clean_v2.clean_record(source)
        self.assertIsNone(reason)
        self.assertEqual(cleaned['text'],source['text'])
        compile(cleaned['text'],'sample','exec')
    def test_generic_token_assignment_is_not_a_secret(self):
        import clean_v2
        text='def count_tokens():\n    token = "example"\n    return token\n'
        cleaned,reason=clean_v2.clean_record({'source':'code-python','kind':'code','license_type':'MIT','text':text})
        self.assertIsNone(reason)
    def test_unlicensed_code_is_rejected(self):
        import clean_v2
        cleaned,reason=clean_v2.clean_record({'source':'code-python','kind':'code','text':'print("hello")'*10,'license_type':'no_license'})
        self.assertIsNone(cleaned);self.assertEqual(reason,'code_license_not_allowlisted')

class ContaminationTests(unittest.TestCase):
    def test_empty_index_rejected(self):
        import contamination_v2 as c
        with self.assertRaises(ValueError):c.build_index([])
    def test_exact_injected_hit_and_clean_negative(self):
        import contamination_v2 as c
        text='one two three four five six seven eight nine ten eleven twelve thirteen fourteen'
        index,_=c.build_index([{'benchmark':'sample','segments':[text]}])
        self.assertTrue(c.match('prefix '+text+' suffix',index))
        self.assertFalse(c.match('entirely unrelated training document remains intact',index))
    def test_rolling_hash_matches_each_window(self):
        import contamination_v2 as c
        terms=[str(i) for i in range(80)]
        for i,h in c.windows(terms):self.assertEqual(h,next(c.windows(terms[i:i+13]))[1])
    def test_chinese_without_spaces(self):
        import contamination_v2 as c
        text='这是用来验证中文评测去污染的一个足够长的独立句子'
        index,_=c.build_index([{'benchmark':'sample','segments':[text]}])
        self.assertTrue(c.match(text,index))

if __name__=='__main__':unittest.main()
