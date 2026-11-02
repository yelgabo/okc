"""Plain Python comments/blank formatting removal; native string values untouched."""
import argparse
import ast
import io
import json
from pathlib import Path
import tempfile
import tokenize

import g_expert_g224_chunk_phased as frozen
from g_expert_g224_prefix import sha
from test_g_expert_g224_chunk_phased import capture_treatment, NATIVE
from test_g_expert_g224_prefix import ROOT
from test_g_expert_g224_prefix import capture


def pack(text):
    lines=text.splitlines(keepends=True)
    protected=set()
    comments=[]
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type==tokenize.STRING:
            protected.update(range(token.start[0],token.end[0]+1))
        if token.type==tokenize.COMMENT and 'okc: problem=' not in token.string:
            comments.append(token)
    for token in reversed(comments):
        row,column=token.start
        endrow,endcolumn=token.end
        assert row==endrow
        lines[row-1]=lines[row-1][:column]+lines[row-1][endcolumn:]
    result=[]
    for row,line in enumerate(lines,1):
        if row not in protected:
            line=line.rstrip()+'\n'
            if not line.strip(): continue
        result.append(line)
    packed=''.join(result)
    assert ast.dump(ast.parse(packed),include_attributes=False)==ast.dump(ast.parse(text),include_attributes=False)
    return packed


def consolidate(text):
    tree=ast.parse(text)
    nodes=[node for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id=="_G224_CHUNK_OLD"]
    assert len(nodes)==1
    node=nodes[0]
    assert ast.literal_eval(node.value)==frozen.OLD_BODY
    old=ast.get_source_segment(text,node)
    replacement="_G224_CHUNK_OLD = _G224_CUDA[_G224_CUDA.index('__device__ void wgs_gather('):_G224_CUDA.index('\\n}\\n', _G224_CUDA.index('__device__ void wgs_gather(')) + 2]"
    updated=text.replace(old,replacement,1)
    env={"_G224_CUDA":frozen.parent.transform_native(frozen.parent.literal((ROOT/frozen.parent.BASE).read_text(),"MOE_CUDA_SRC"))}
    exec(replacement,env)
    assert env["_G224_CHUNK_OLD"]==frozen.OLD_BODY
    newtree=ast.parse(updated)
    newnode=next(item for item in newtree.body if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name) and item.targets[0].id=="_G224_CHUNK_OLD")
    newnode.value=node.value
    assert ast.dump(tree,include_attributes=False)==ast.dump(newtree,include_attributes=False)
    return updated


def real_calls(text):
    prefix,suffix=text.rsplit("\nimport hashlib as _g224_chunk_hash\n",1)
    calls,env=capture(prefix,True)
    calls.clear()
    exec("import hashlib as _g224_chunk_hash\n"+suffix,env)
    records=[]
    for tokens in (1,8,16,32,128,256,384):
        calls.clear()
        env["_MA_EXTS"]={}
        env["_G224_EXTS"]={}
        state=dict(ma_sfx=("-DOKC_PRMTSC=1",),ma_sc6=True,ma_ext={})
        env["_fork_native_moe"](tokens,state)
        assert len(calls)==1
        records.append(dict(tokens=tokens,flags=calls[0]["extra_cuda_cflags"],cuda_sha256=sha(calls[0]["cuda_sources"]),cpp_sha256=sha(calls[0]["cpp_sources"])))
    assert env["_G224_CUDA"]==NATIVE
    return records


def run(out):
    with tempfile.TemporaryDirectory() as directory:
        text,native,parent_manifest=frozen.generate(ROOT,Path(directory))
    assert sha(text)=='d67dc8d1c32ad512048c2998229448a97aee3d7d530c3bb2bd6b4accfdc9dd05'
    comment_only=pack(text)
    packed=consolidate(comment_only)
    calls,env=capture_treatment(text,True)
    assert env['_G224_CUDA']==NATIVE==native
    assert not any(token in text for token in ('inspect.getsource','linecache','sys._getframe'))
    actual=real_calls(packed)
    assert actual==real_calls(text)
    result=dict(scope='Plain comment/blank removal plus one redundant old-gather literal bound transparently to identical existing parent native substring; all seven actual factory calls equal.',
                original_bytes=len(text.encode()),original_sha256=sha(text),payload_bytes=len(packed.encode()),payload_sha256=sha(packed),
                native_sha256=sha(native),comment_only_bytes=len(comment_only.encode()),ast_equal_except_proven_redundant_binding=True,real_entry_calls=actual,
                treatment_compile=False,host_nvcc=False,gpu=False,judge=False)
    result['decision']='PACKED_SOURCE_CPU_ONLY' if len(packed.encode())<=1000000 else 'STOP_PLAIN_PACK_OVER_BUDGET'
    out.mkdir(parents=True,exist_ok=True)
    (out/'package-result.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['decision']=='PACKED_SOURCE_CPU_ONLY':
        (out/'variant.py').write_text(packed,encoding='utf8',newline='\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out)
