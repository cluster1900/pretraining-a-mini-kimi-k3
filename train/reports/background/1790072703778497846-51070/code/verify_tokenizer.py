"""Compare local wrapper with the two inspected upstream encoding methods."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
from typing import Iterator, List

spec=importlib.util.spec_from_file_location('k3_tokenizer',Path(__file__).with_name('tokenizer.py'))
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def verify(directory):
    tok=mod.K3Tokenizer(directory)
    tree=ast.parse((Path(directory)/'tokenization_kimi.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TikTokenTokenizer')
    methods=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in
             ('_encode_text_piece','_split_whitespaces_or_nonwhitespaces')]
    if len(methods)!=2: raise AssertionError('Reference encoding methods missing')
    # No upstream module imports/model/chat code are executed.
    ref_class=ast.ClassDef(name='Reference',bases=[],keywords=[],body=methods,decorator_list=[])
    namespace={'List':List,'Iterator':Iterator}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[ref_class],type_ignores=[])),
                 '<inspected-upstream-encoding>','exec'),namespace)
    ref=namespace['Reference'](); ref.model=tok.enc
    samples=['', 'hello world!', '中文测试 English MIX café e\u0301',
        'def f(x):\n    return x + 1\n', '\\frac{1}{2} + \\alpha_{i}',
        '[EOS] [BOS] <|im_start|> <|end_of_msg|> [PAD]',
        '\t \r\n  multiple\n\nlines', '😀🌍𠀀', 'a'*60_000, ' '*60_000,
        ('中英 hello\n'*8000), ('abc def\n'*51000)]
    for index,text in enumerate(samples):
        ids=tok.encode(text,append_eos=False)
        assert ids==ref._encode_text_piece(text,allow_special_tokens=False),(index,'encoding differs')
        assert tok.decode(ids)==text,(index,'roundtrip differs')
        assert all(0<=x<tok.BOS_TOKEN_ID for x in ids),(index,'unexpected special token')
        assert tok.encode(text)==ids+[tok.eos_token_id]
    assert len(set(tok.special_tokens.values()))==256
    assert '<|im_start|>' not in tok.special_tokens
    assert tok.encode_batch(samples[:8])==[tok.encode(x) for x in samples[:8]]
    return {'status':'passed','cases':len(samples),'vocab_size':tok.vocab_size,
            'tokenizer_fingerprint':tok.fingerprint,'asset_hashes':tok.asset_hashes,
            'checks':['upstream ordinary IDs','UTF8 roundtrip','literal specials','document EOS','batch equivalence']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();result=verify(a.tokenizer)
    Path(a.report).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report).write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
