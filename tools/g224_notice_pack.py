"""Notice-preserving plain-token packaging of frozen G224; no native changes."""
import argparse
import ast
import io
import json
from pathlib import Path
import tempfile
import tokenize

import g_expert_g224_chunk_phased as generator
from g224_reference_pack import consolidate
from g_expert_g224_prefix import sha
from test_g_expert_g224_prefix import capture

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='0d84d0e34303f2a154a914ace131cbae661cc61def18f7bfcbc20ed3f218cf76'


def tokens(text,kind):
    return [token.string for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type==kind]


def notice_tokens(text):
    start=text.index('#                                  Apache License')
    end=text.index('def _release_private_scratch',start)
    low=text[:start].count('\n')+1
    high=text[:end].count('\n')+1
    return [token.string for token in tokenize.generate_tokens(io.StringIO(text).readline) if token.type==tokenize.COMMENT and (token.start[0]<=17 or low<=token.start[0]<high)]


def verify(text,packed):
    assert ast.dump(ast.parse(text),include_attributes=False)==ast.dump(ast.parse(packed),include_attributes=False)
    assert tokens(packed,tokenize.STRING)==tokens(text,tokenize.STRING)
    assert tokens(packed,tokenize.COMMENT)==notice_tokens(text)
    assert len(notice_tokens(text))==253
    assert all(term in packed for term in ('Portions Copyright vjeux','Submitted by alygg77','Copyright 2026 team keell0renz','Permission is hereby granted','THE SOFTWARE IS PROVIDED'))
    assert '\r' not in packed


def pack(text):
    start=text.index('#                                  Apache License')
    end=text.index('def _release_private_scratch',start)
    low=text[:start].count('\n')+1
    high=text[:end].count('\n')+1
    level=0
    items=[]
    removed=set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type==tokenize.COMMENT and not (token.start[0]<=17 or low<=token.start[0]<high):
            removed.add(token.start[0])
            continue
        if token.type==tokenize.NL and (not token.line.strip() or token.start[0] in removed):
            continue
        if token.type==tokenize.INDENT:
            level+=1
            items.append((token.type,' '*level))
        elif token.type==tokenize.DEDENT:
            level-=1
            items.append((token.type,token.string))
        else:
            items.append((token.type,token.string))
    packed=tokenize.untokenize(items)
    verify(text,packed)
    return packed


def calls(text):
    imports=[node for node in ast.parse(text).body if isinstance(node,ast.Import) and any(alias.asname=='_g224_chunk_hash' for alias in node.names)]
    assert len(imports)==1
    lines=text.splitlines(keepends=True)
    prefix=''.join(lines[:imports[0].lineno-1])
    suffix=''.join(lines[imports[0].lineno-1:])
    pending,env=capture(prefix,True)
    exec(suffix,env)
    records=[]
    for size in (1,8,16,32,128,256,384):
        pending.clear()
        env['_MA_EXTS']={}
        env['_G224_EXTS']={}
        state=dict(ma_sfx=('-DOKC_PRMTSC=1',),ma_sc6=True,ma_ext={})
        env['_fork_native_moe'](size,state)
        assert len(pending)==1
        records.append(dict(tokens=size,cuda_sha256=sha(pending[0]['cuda_sources']),cpp_sha256=sha(pending[0]['cpp_sources']),flags=pending[0]['extra_cuda_cflags']))
    assert sha(env['_G224_CUDA'])=='53d09576f86e93b9baaa0145dcbdf1b60d8736743677d0a4de60e89864d18492'
    return records


def run(out):
    with tempfile.TemporaryDirectory() as directory:
        frozen,native,_=generator.generate(ROOT,Path(directory))
    assert sha(frozen)=='d67dc8d1c32ad512048c2998229448a97aee3d7d530c3bb2bd6b4accfdc9dd05'
    consolidated=consolidate(frozen)
    packed=pack(consolidated)
    actual=calls(packed)
    assert actual==calls(frozen)
    assert len(packed.encode())==999492 and sha(packed)==EXPECTED
    out.mkdir(parents=True,exist_ok=True)
    (out/'variant.py').write_text(packed,encoding='utf8',newline='\n')
    result=dict(decision='NOTICE_PACKAGE_SOURCE_CPU_PASS_TARGET_NOT_RUN',payload_bytes=len(packed.encode()),payload_sha256=sha(packed),native_sha256=sha(native),frozen_python_sha256=sha(frozen),notices_exact=253,strings_exact=True,ast_equal_after_proven_binding=True,all_seven_calls_equal=True,calls=actual,gpu=False,judge=False)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)

