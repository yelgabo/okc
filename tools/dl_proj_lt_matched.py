"""Combined Lt+owner-boundary full-score diagnostic under one prepared policy.

Uses root-owned boundary duplication. Shared extensions, policy and mutable scratch;
private arm inputs/outputs and synchronized sequential replays. Never submit probe.
"""
import argparse
import ast
import hashlib
from pathlib import Path
import sys
import dl_proj_lt_port as port
import dl_make_bench_pub as bench

RUNTIME=r'''
_combo_prepare=prepare
_timeline=lambda state:None
_AB_DEQ={}

def _combo_reset(data,base):
    data['stream'].copy_(base['stream'])
    data['attn_input'].copy_(base['attn_input'])

def _combo_decide(rows):
    import math
    scores=[]
    for si in (0,1):
        loads=[]
        for load in range(6):
            s=next(r[4] for r in rows if r[:3]==(load,'sliding',si))
            f=next(r[4] for r in rows if r[:3]==(load,'full',si))
            loads.append([(46*a+16*b)/62 for a,b in zip(s,f)])
        logs=[sum(math.log(v[a]/v[b]) for v in loads)/6 for a,b in ((0,2),(0,3),(1,2),(1,3))]
        nulls=[sum(math.log(v[a]/v[b]) for v in loads)/6 for a,b in ((1,0),(3,2))]
        pooled=sum(logs)/4
        keep=all(x>0 for x in logs) and pooled>2*max(abs(x) for x in nulls)
        scores.append(keep)
        print('COMBO_SCORE',si,'cross',[round(math.exp(x),7) for x in logs],
              'pooled',round(math.exp(pooled),7),'null',[round(math.exp(x),7) for x in nulls],
              'KEEP',keep,flush=True)
    return all(scores)

def prepare(weights):
    import random
    import statistics
    started=time.monotonic()
    state=_combo_prepare(weights)
    assert state['bnd_ours'] and len(state['dlplt'])==3
    control=dict(state,dlplt={})
    flush=torch.empty(256<<20,dtype=torch.uint8,device='cuda')
    rng=random.Random(1101747)
    # Explicit ABI gate of the affected boundary before whole-layer timing.
    for m in (1,8):
        for seed in (1101749,1101750):
            gen=torch.Generator(device='cuda').manual_seed(seed)
            stream=torch.randn((m,HIDDEN),device='cuda',generator=gen)
            f=torch.randn((m,UNPOOL_SLICE),device='cuda',generator=gen).to(torch.bfloat16)
            old=_bnd(stream.clone(),f,state);new=_bnd_owner(stream.clone(),f,state)
            assert all(torch.equal(a,b) for a,b in zip(old,new)),('boundary ABI',m,seed)
    print('COMBO_BOUNDARY bitwise T1/T8 twofreshinputs PASS',flush=True)
    shapes=((1,1),(16,1),(32,1),(1,8),(16,8),(32,8))
    rows=[];checks=0
    saved_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    try:
        for load,(batch,length) in enumerate(shapes):
            for layer in ('sliding','full'):
                for si in (0,1):
                    if time.monotonic()-started>275:raise RuntimeError('combined comparison incomplete275s')
                    seed=1101801+si*10000+load*100+(layer=='full')
                    base=_ab_inputs(batch,length,layer,seed)
                    front=_ab_ref_front(base,weights)
                    ids=front['biased'].topk(TOP_K).indices
                    start=LOCAL_EXPERTS*int(torch.bincount((ids//LOCAL_EXPERTS).flatten(),minlength=8).argmax())
                    base['local_start'].fill_(start);front['local_start']=start
                    front8=_ab_ref_front(base,weights,FP8);front8['local_start']=start
                    arms=[]
                    for changed in (False,False,True,True):
                        data={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in base.items()}
                        fn=_owner_layer if changed else kernel_fn
                        st=state if changed else control
                        fn(data,st)
                        _combo_reset(data,base)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):out=fn(data,st)
                        graph.replay();torch.cuda.synchronize()
                        ok,msg=_ab_validate(base,out,data,weights,front,front8)
                        assert ok,(load,layer,si,changed,msg)
                        checks+=1
                        arms.append((graph,data,out))
                    samples=[[],[],[],[]]
                    for rep in range(40):
                        for idx in rng.sample(range(4),4):
                            graph,data,out=arms[idx]
                            _combo_reset(data,base);flush.zero_()
                            a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            a.record();graph.replay();z.record();z.synchronize()
                            samples[idx].append(a.elapsed_time(z)*1000)
                    raw=[statistics.median(v) for v in samples]
                    rows.append((load,layer,si,seed,raw))
                    print('COMBO_CASE',load,layer,si,seed,[round(v,3) for v in raw],flush=True)
                    del arms,front,front8,base,ids
        assert checks==96 and len(rows)==24
        for line in state['dlplt_report']:print('COMBO_GATE',line,flush=True)
        print('COMBO_CORE checks96/96 routes3 boundaryPASS policyCOMMON elapsed',round(time.monotonic()-started,2),flush=True)
        keep=_combo_decide(rows)
        print('COMBO_KEEP',keep,flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32=saved_tf32
        _AB_DEQ.clear()
    return state
'''

def build(base,projection,root):
    sys.path.insert(0,str(root/'tools'))
    import dl_make_matched_short_boundary as shortbench
    import dl_short_rms_owner as short
    pure=port.build(short.candidate(base),projection)
    dual=shortbench.build(base)
    assert dual.endswith('\n'+shortbench.PROBE)
    dual=dual[:-len('\n'+shortbench.PROBE)]
    body=ast.parse(bench.BODY)
    names=('_ab_deq','_ab_ref_back','_ab_local','_ab_check_routing','_ab_compare','_ab_validate')
    helpers='\n\n'.join(ast.get_source_segment(bench.BODY,n) for n in body.body
                        if isinstance(n,ast.FunctionDef) and n.name in names)
    probe=port.build(dual,projection)+'\n'+helpers+'\n'+RUNTIME
    ast.parse(probe);assert len(probe.encode())<1_000_000
    return pure,probe

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('pure',type=Path);p.add_argument('probe',type=Path)
    a=p.parse_args()
    base=(a.root/'muse_decode_layer/cand/layer_attn31s.py').read_text(encoding='utf-8')
    projection=(port.ROOT/'muse_bf16_proj/submission.py').read_text(encoding='utf-8')
    for path,source in zip((a.pure,a.probe),build(base,projection,a.root)):
        path.write_text(source,encoding='utf-8',newline='\n')
        print(path,len(source.encode()),hashlib.sha256(source.encode()).hexdigest())
