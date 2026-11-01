"""One transparent inactive-unit deletion: owned ONE_DETM diagnostic finalizer.
No other native specialization, runtime I/O, compression, or production change.
"""
import argparse,ast,hashlib,importlib.util,json,sys
from pathlib import Path

def remove_detm_helper(cuda):
    assert '#define ONE_DETM 0' in cuda
    needle='#if ONE_DETM\n// Diagnostic only:'
    # Anchor by the named helper, then match its immediately enclosing conditional.
    at=cuda.index('__device__ void finalize_det(')
    start=cuda.rfind('#if ONE_DETM\n',0,at)
    assert start>=0
    lines=cuda[start:].splitlines(keepends=True);depth=0;end=None
    for i,line in enumerate(lines):
        s=line.strip()
        if s.startswith(('#if ','#ifdef ','#ifndef ')):depth+=1
        elif s=='#endif':
            depth-=1
            if depth==0:end=start+sum(map(len,lines[:i+1]));break
    assert end is not None
    removed=cuda[start:end]
    assert removed.count('finalize_det(')==1 and '#else' not in removed
    revised=cuda[:start]+cuda[end:]
    assert revised+'' != cuda and cuda==revised[:start]+removed+revised[start:]
    return revised,removed

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.root/'tools'))
    from dl_owned_slice import build
    source=(a.root/'muse_decode_layer/cand/layer_lt_boundary.py').read_text(encoding='utf-8')
    old,manifest=build(source);tree=ast.parse(old)
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_ONE_SRC')
    cuda=ast.literal_eval(node.value);new,removed=remove_detm_helper(cuda)
    lines=old.splitlines(keepends=True);span=''.join(lines[node.lineno-1:node.end_lineno]);assert span.count(cuda)==1
    lines[node.lineno-1:node.end_lineno]=[span.replace(cuda,new)]
    output=''.join(lines)
    after=ast.parse(output)
    next(n for n in after.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_ONE_SRC').value=ast.Constant(cuda)
    assert ast.dump(after)==ast.dump(tree),'outside ONE_SRC changed'
    loader=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_one_load')
    assert ast.get_source_segment(old,loader).count('ONE_DETM')==0
    a.out.mkdir(parents=True,exist_ok=True)
    for name,value in (('owned_original.py',old),('owned_reduced.py',output),('original.cu',cuda),('reduced.cu',new),('removed-finalize-det.cu.txt',removed)):
        (a.out/name).write_text(value,encoding='utf-8',newline='\n')
    bench=(a.root/'tools/dl_make_bench_pub.py').read_text(encoding='utf-8');bt=ast.parse(bench)
    body=next(ast.literal_eval(n.value) for n in bt.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BODY')
    report=dict(original_owned_bytes=len(old.encode()),reduced_owned_bytes=len(output.encode()),saved_bytes=len(removed.encode()),removed_unit='ONE_DETM-off finalize_det helper only',original_device_sha256=hashlib.sha256(cuda.encode()).hexdigest(),reduced_device_sha256=hashlib.sha256(new.encode()).hexdigest(),unchanged_registration_and_python_AST=True,existing_harness_BODY_bytes=len(body.encode()),G_four_literal_floor=520047,combined_lower_bound_with_existing_harness=520047+len(output.encode())+len(body.encode()),G_checkpoint_plain_bytes=857866,combined_plain_checkpoint_bodies_and_existing_harness=857866+len(output.encode())+len(body.encode()),gpu_execution=False,H200_ready=False)
    (a.out/'budget.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
