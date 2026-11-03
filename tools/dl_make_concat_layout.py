"""Build a diagnostic for the physical layout of the concatenated pooler/router weight."""
from pathlib import Path
import sys

from dl_make_layer_timeline import extract_helpers

PROBE = r'''
_concat_prepare = prepare
_timeline = lambda state: None
_REF_STREAM_SEED, _REF_STREAM_MEAN = 5354, 0.7
_REF_STREAM_SHARED, _REF_STREAM_RANK, _REF_DRAFT_SHARE = 0.9, 4, 0.6

def prepare(weights):
    import random
    import statistics
    base = _concat_prepare(weights)
    original = base['pool_router']
    alternate = original.t().contiguous().t()
    assert torch.equal(original, alternate)
    states = [dict(base, pool_router=w) for w in (original,original,alternate,alternate)]
    flush = torch.empty(256 << 20, dtype=torch.uint8, device='cuda')
    rng = random.Random(931)
    rows = []
    for batch in (16,32):
        assert _front_variant(batch*8,base)=='concat-r'
        for layer in ('sliding','full'):
            for seed in (711371,711372):
                inputs = _ab_inputs(batch,8,layer,seed)
                flag = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    ids = _ab_ref_front(inputs,weights)['biased'].topk(TOP_K).indices
                    inputs['local_start'].fill_(32*int(torch.bincount((ids//32).flatten(),minlength=8).argmax()))
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = flag
                for scope in ('gemm','layer'):
                    def run(data,st):
                        if scope=='gemm':
                            return (torch.mm(data['attn_input'],st['pool_router'].t(),out_dtype=torch.float32),)
                        return kernel_fn(data,st)
                    if scope=='gemm':
                        ref = inputs['attn_input'].double() @ original.t().double()
                    arms = []
                    for st in states:
                        data={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
                        run(data,st)
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            out=run(data,st)
                        data['stream'].copy_(inputs['stream'])
                        data['attn_input'].copy_(inputs['attn_input'])
                        graph.replay()
                        torch.cuda.synchronize()
                        snap=[o.clone() for o in out]
                        if scope=='gemm':
                            for part in (slice(0,D_EXPERT),slice(D_EXPERT,D_EXPERT+N_EXPERTS),slice(D_EXPERT+N_EXPERTS,None)):
                                expected=ref[:,part]
                                assert float((snap[0][:,part].double()-expected).norm()/expected.norm().clamp_min(1e-12))<1e-4
                        elif arms:
                            for a,b in zip(snap,arms[0][3]):
                                if a.is_floating_point():
                                    assert float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<0.001
                                else:
                                    assert torch.equal(a.sort(-1).values,b.sort(-1).values)
                        arms.append((graph,data,out,snap))
                    samples=[[],[],[],[]]
                    for rep in range(24):
                        for i in rng.sample(range(4),4):
                            graph,data,_,_=arms[i]
                            data['stream'].copy_(inputs['stream'])
                            data['attn_input'].copy_(inputs['attn_input'])
                            flush.zero_()
                            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                            start.record()
                            graph.replay()
                            end.record()
                            end.synchronize()
                            samples[i].append(start.elapsed_time(end)*1000)
                    rows.append((scope,batch,layer,seed,[round(statistics.median(x),3) for x in samples]))
    for row in rows:
        print('CONCAT_LAYOUT',*row,flush=True)
    return base
'''

if __name__=='__main__':
    path=Path('muse_decode_layer/cand/layer_attn31s.py')
    source=path.read_text(encoding='utf-8')+'\n'+extract_helpers()+'\n'+PROBE
    compile(source,'<layout>','exec')
    assert len(source.encode())<1_000_000
    Path(sys.argv[1]).write_text(source,encoding='utf-8',newline='\n')
