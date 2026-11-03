"""CPU-only checkpoint: transparent G-family arm slice, not a public entrant.

Pure stage pin295ca590. Prune ONLY dead GLE-O branch under invariant PDL=False;
retain normal prepare and EVERY embedded native source literal byte-for-byte.
The selected arm supports verify32/sliding/page1; unsupported geometry raises.
This prototype exceeds the per-arm budget and MUST NOT be dispatched.
The four required literal lower bound applies only to whole-literal preservation,
not to every possible deeper native-code specialization/factoring.
"""
import argparse,ast,hashlib,json
from pathlib import Path
PIN='295ca590a9d07a2491d600e1c96ec5583d29ae3220f619bc9e99adf62ef48491'
GUARD='''
# PRIVATE comparator-arm slice only. Never submit this restricted diagnostic arm.
_g_slice_kernel=kernel_fn

def kernel_fn(data,state):
    c=data['config']
    if (c['batch'],c['q_len'],c['window'],c['page_size'])!=(32,8,2048,1):
        raise RuntimeError('G slice supports only verify32 sliding/page1')
    return _g_slice_kernel(data,state)
'''

def literals(tree):
    out={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000:
            out[ast.unparse(n.targets[0])]=n.value.value
    return out

def build(source,other_bytes=504940,harness_bytes=0):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN
    tree=ast.parse(source)
    writes=[n for n in ast.walk(tree) if isinstance(n,ast.Name) and n.id=='PDL' and isinstance(n.ctx,ast.Store)]
    assert len(writes)==1
    pdl=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PDL' for t in n.targets))
    assert ast.literal_eval(pdl.value) is False
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_pdl')
    assert ast.unparse(fn.body[0])=='return PDL and T <= PDL_MAX_T'
    after=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_after_attention'][-1]
    assert len(after.body)==2 and isinstance(after.body[0],ast.If) and isinstance(after.body[1],ast.Return)
    test=after.body[0].test
    assert isinstance(test,ast.BoolOp) and isinstance(test.op,ast.And)
    assert any(isinstance(v,ast.Call) and isinstance(v.func,ast.Name) and v.func.id=='_pdl' for v in test.values)
    assert not after.body[0].orelse
    removed=[];edits=[(after.body[0].lineno-1,after.body[0].end_lineno,'')]
    for name in ('_gle_after_attention','_gle_o_choice'):
        n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        loads=[x for x in ast.walk(tree) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load) and x.id==name]
        assert len(loads)==1 and after.body[0].lineno<=loads[0].lineno<=after.body[0].end_lineno
        assert not any(isinstance(x,ast.Constant) and isinstance(x.value,str) and name in x.value for x in ast.walk(tree))
        edits.append((n.lineno-1,n.end_lineno,''));removed.append(name)
    lines=source.splitlines(keepends=True)
    for a,z,replacement in sorted(edits,reverse=True):lines[a:z]=[replacement]
    result=''.join(lines)+GUARD;new=ast.parse(result)
    assert literals(tree)==literals(new),'native literal changed'
    # Original return call survives; every prepare function/alias is unchanged.
    def prepares(t):return [ast.dump(n,include_attributes=False) for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='prepare']
    assert prepares(tree)==prepares(new)
    native=literals(tree)
    required=('ATT7_CUDA_SRC','MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']",'bfp_SK2_SRC')
    floor={k:len(native[k].encode()) for k in required}
    budget=999999-other_bytes-harness_bytes
    report={'source_pin':PIN,'source_bytes':len(source.encode()),'prototype_bytes':len(result.encode()),
        'prototype_sha256':hashlib.sha256(result.encode()).hexdigest(),'other_arm_bytes':other_bytes,'harness_bytes':harness_bytes,
        'G_budget_bytes':budget,'fits':len(result.encode())<=budget,'removed_dead_functions':removed,
        'native_literal_values_unchanged':True,'normal_prepare_AST_unchanged':True,
        'whole_literal_floor':floor,'four_literal_bytes':sum(floor.values()),
        'whole_literal_floor_excess':sum(floor.values())-budget,'selected_geometry':{'batch':32,'q_len':8,'window':2048,'page_size':1},
        'gpu_execution':False,'actual_state_policy_verified':False,
        'reason':'normal prepare builds all5 MoE keys and T1..256 shared-RNG/redraw tuning; cannot delete low-T source as proven irrelevant',
        'release':'full single-arm G stage qualification separate; no paired gain inferred'}
    return result,report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--other-bytes',type=int,default=504940);p.add_argument('--harness-bytes',type=int,default=0);a=p.parse_args()
    s,r=build(a.source.read_text(encoding='utf8'),a.other_bytes,a.harness_bytes);a.output.write_bytes(s.encode());print(json.dumps(r,indent=2))
