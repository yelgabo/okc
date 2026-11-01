"""Build a prepare-only large-M diagnostic on the unchanged projection path.

Never submit the generated benchmark publicly. Prefill integration needs its own gate.
"""
from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = r'''
import random as _plrandom
import statistics as _plstats
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl as _plbase
_PLLOG=[]
def _plprint(s):_PLLOG.append(s);print(s,flush=True)

def _pldiag(weights,started):
    lt=_build_lt();dev=weights['qkv_gate'].device
    gen=torch.Generator(device=dev).manual_seed(91101)
    flush=torch.empty(256<<20,dtype=torch.uint8,device=dev)
    def timed(fns,x,reps,seed):
        graphs={};outputs={}
        for tag,fn in fns.items():
            fn(x);g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn(x)
            graphs[tag]=g;outputs[tag]=out
        times={tag:[] for tag in fns};rng=_plrandom.Random(seed)
        for rd in range(reps):
            order=list(fns);rng.shuffle(order)
            for tag in order:
                flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();graphs[tag].replay();b.record();b.synchronize()
                times[tag].append(1000*a.elapsed_time(b))
        return times
    def runner(p,i,warg,m,n,ws):
        out=torch.empty((m,n),device=dev,dtype=torch.bfloat16)
        def run(x):
            rc=lt.lt_run(p,i,warg,x,out,ws)
            if rc:raise RuntimeError('largeM lt_run status '+str(rc))
            return out
        return run
    def validate(tag,fn,x,w):
        out=fn(x);base=_plbase(x,w,None)
        err=float(torch.linalg.vector_norm(out.float()-base.float())/torch.linalg.vector_norm(base.float()))
        rows=torch.tensor([0,1,17,127,511,x.shape[0]//2,x.shape[0]-2,x.shape[0]-1],device=dev)
        ref=x[rows].double()@w.double().T
        er=float(torch.linalg.vector_norm(out[rows].double()-ref)/torch.linalg.vector_norm(ref))
        assert err<.005 and er<.005,(tag,err,er)
        _plprint(f'PLCHECK {tag} m={len(x)} baselineL2={err:.7f} fp64L2={er:.7f}')
    for name in ('o','qkv_gate','pooler','unpooler'):
        if time.time()-started>240:
            _plprint(f'PLSKIP {name} deadline240s elapsed={time.time()-started:.1f}');continue
        w=weights[name];n,k=w.shape;m=16384
        x=torch.randn((m,k),device=dev,generator=gen).to(torch.bfloat16)
        base=lambda x,w=w:_plbase(x,w,None)
        # Named isolated profile resolves the three-call aggregate in the layer receipt.
        base(x);torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:base(x)
        torch.cuda.synchronize()
        kernels=[(e.name,e.device_time_total) for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
        _plprint(f'PLPROFILE {name} m={m} n={n} k={k} kernels={kernels}')
        if name=='o':
            t=timed({'base0':base,'base1':base},x,12,91101)
            _plprint(f'PLBASE o '+str({s:_plstats.median(v) for s,v in t.items()}));continue
        wt=w.T.contiguous();ws=torch.empty(32<<20,device=dev,dtype=torch.uint8)
        proposals=[]
        for layout in range(4):
            p=lt.lt_plan_new(m,n,k,layout,ws.numel());assert p>=0
            for cap in (0,32<<20):lt.lt_heuristic(p,cap,12)
            # At most 24 accepted configs per layout, 96 total; no expansion or sweeps.
            for i in range(min(24,lt.lt_count(p))):
                proposals.append((p,i,layout))
        assert len(proposals)<=96
        fs={str(j):runner(p,i,wt if layout%2 else w,m,n,ws) for j,(p,i,layout) in enumerate(proposals)}
        screen=timed(fs,x,3,91102)
        ranked=sorted(range(len(proposals)),key=lambda j:_plstats.median(screen[str(j)]))[:2]
        configs=[(proposals[j][2],list(lt.lt_info(proposals[j][0],proposals[j][1]))[:9]) for j in ranked]
        _plprint(f'PLSELECT {name} candidates={len(proposals)} configs={configs} elapsed={time.time()-started:.1f}')
        # Independent plan handles and workspaces for each twin. Never share output buffers.
        arms={'base0':base,'base1':base};keepers=[]
        for rank,(layout,cfg) in enumerate(configs):
            for twin in ('a','b'):
                p=lt.lt_plan_new(m,n,k,layout,32<<20);idx=lt.lt_add_config(p,cfg)
                assert lt.lt_count(p)==1,(name,rank,twin,idx)
                privatews=torch.empty(32<<20,device=dev,dtype=torch.uint8);keepers.append(privatews)
                arms[f'p{rank}{twin}']=runner(p,0,wt if layout%2 else w,m,n,privatews)
        for inp in range(2):
            if time.time()-started>240:
                _plprint(f'PLSKIP {name} heldout={inp} deadline240s');break
            held=torch.randn((m,k),device=dev,generator=gen).to(torch.bfloat16)
            for tag,fn in arms.items():validate(name+' '+tag,fn,held,w)
            ts=timed(arms,held,30,91200+inp)
            null=[a-b for a,b in zip(ts['base1'],ts['base0'])]
            mednull=_plstats.median(null);mad=_plstats.median(abs(d-mednull) for d in null)
            threshold=max(.02*_plstats.median(ts['base0']),2*mad)
            for tag,v in ts.items():
                ds=[a-b for a,b in zip(v,ts['base0'])];delta=_plstats.median(ds)
                _plprint(f'PLTIME {name} input={inp} {tag} raw={_plstats.median(v):.3f} delta={delta:+.3f} faster={sum(d<0 for d in ds)}/30 threshold={threshold:.3f} accept={delta < -threshold}')
        # Adjacent token length and exact power-of-two probes, same selected configs.
        for mm,probe in ((16383,False),(16384,True)):
            held=torch.randn((mm,k),device=dev,generator=gen).to(torch.bfloat16)
            if probe:held.zero_();held[:,17]=2
            for rank,(layout,cfg) in enumerate(configs):
                p=lt.lt_plan_new(mm,n,k,layout,32<<20);lt.lt_add_config(p,cfg)
                if lt.lt_count(p)!=1:raise RuntimeError(f'nearby config rejected {name} {rank} m={mm}')
                validate(name+f' p{rank} probe={probe}',runner(p,0,wt if layout%2 else w,mm,n,ws),held,w)
        _plprint(f'PLDONE {name} elapsed={time.time()-started:.1f}')
    for line in _PLLOG:print(line,flush=True)

_pl_original_prepare=prepare
def prepare(weights):
    started=time.time();state=_pl_original_prepare(weights);_pldiag(weights,started);return state
'''

def generate(source):
    ast.parse(source)
    result=source+RUN
    ast.parse(result)
    return result

if __name__=='__main__':
    source=(ROOT/'muse_bf16_proj/submission.py').read_text()
    Path(sys.argv[1]).write_text(generate(source))
    print('Prepared large-M diagnostic; judged production unchanged')
