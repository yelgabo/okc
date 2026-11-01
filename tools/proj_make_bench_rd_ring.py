"""One bounded rd16 async ring diagnostic; judged path is unchanged C3."""
from pathlib import Path
import ast
import sys
from proj_rd_ring_source import generate, cpu_gate, ROOT

RUN=r'''
import random as _rrrandom
import statistics as _rrstats
_RRLOG=[]
def _rrprint(s):_RRLOG.append(s);print(s,flush=True)

@@PACK@@

def _rrbench(weights,state,started):
    mods={}
    for tag in ('a','b'):
        mods[tag]=load_inline(name='okc_rd16_ring_'+tag,cpp_sources='#include <torch/extension.h>\nstd::vector<int64_t> rr_info(bool,bool);void rr_fwd(at::Tensor,at::Tensor,at::Tensor,at::Tensor,int64_t,bool,bool);',
            cuda_sources=@@CUDA@@,functions=['rr_info','rr_fwd'],no_implicit_headers=True,
            extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)
    for tag,mod in mods.items():
        old=mod.rr_info(False,False);new=mod.rr_info(True,False)
        _rrprint(f'RRRES {tag} old={old} ring={new} fields=regs,local,static,dynamic,blocks')
        for ring in (False,True):_rrprint(f'RRPROBERES {tag} ring={ring} {mod.rr_info(ring,True)}')
        if new[1] or new[4]<old[4]:
            _rrprint(f'RRSKIP resource gate failed {tag}: zero-local and blocks>=old required; CLOSED as built')
            return
    if time.time()-started>280:_rrprint('RRSKIP deadline280s before correctness; no winner');return
    w=weights['pooler'];dev=w.device;packs={tag:_rr_pack(w) for tag in mods}
    assert all(pk is not None for pk in packs.values()),'rd16 escape pack unavailable'
    gen=torch.Generator(device=dev).manual_seed(101031)
    arms={'prod':state['pick']['pooler',16]};probes={}
    for tag,mod in mods.items():
        pk=packs[tag]
        for ring in (False,True):
            def make(mod=mod,pk=pk,ring=ring,memory=False):
                def run(x):
                    out=torch.empty((len(x),4096),device=dev,dtype=torch.bfloat16)
                    mod.rr_fwd(pk['stream'],pk['esc'],x,out,pk['base4'],ring,memory);return out
                return run
            label=tag+('_ring' if ring else '_old')
            arms[label]=make();probes[label+'_probe']=make(memory=True)
    rows=torch.tensor([0,1,17,63,512,1023,2047,4095],device=dev)
    def check(tag,fn,x):
        out=fn(x);base=torch.nn.functional.linear(x,w)
        er=float(torch.linalg.vector_norm(out.float()-base.float())/torch.linalg.vector_norm(base.float()))
        ref=x.double()@w[rows].double().T
        e64=float(torch.linalg.vector_norm(out[:,rows].double()-ref)/torch.linalg.vector_norm(ref))
        assert er<.005 and e64<.005,(tag,len(x),er,e64)
        return er,e64
    for m in (1,7,15,16):
        for probe in (False,True):
            x=torch.randn((m,8192),device=dev,generator=gen).to(torch.bfloat16)
            if probe:x.zero_();x[:,17]=2
            errors={tag:check(tag,fn,x) for tag,fn in arms.items() if tag!='prod' or m==16}
            _rrprint(f'RRCHECK m={m} probe={probe} PASS {errors}')
    # Capture each independent output and replay fresh values into the captured input.
    for tag,fn in arms.items():
        x=torch.randn((16,8192),device=dev,generator=gen).to(torch.bfloat16);fn(x)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn(x)
        for seed in range(2):
            x.copy_(torch.randn(x.shape,device=dev,generator=gen).to(torch.bfloat16));graph.replay()
            ref=x.double()@w[rows].double().T
            err=float(torch.linalg.vector_norm(out[:,rows].double()-ref)/torch.linalg.vector_norm(ref))
            assert err<.005,(tag,seed,err)
        _rrprint(f'RRREPLAY {tag} two fresh captured values PASS')
    flush=torch.empty(256<<20,device=dev,dtype=torch.uint8)
    allarms={**arms,**probes};complete=True
    for inp in range(2):
        if time.time()-started>280:_rrprint(f'RRSKIP heldout{inp} deadline280s; no winner');complete=False;break
        x=torch.randn((16,8192),device=dev,generator=gen).to(torch.bfloat16)
        for tag,fn in arms.items():check(tag,fn,x)
        graphs={};outputs={}
        for tag,fn in allarms.items():
            fn(x);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn(x)
            graphs[tag]=graph;outputs[tag]=out
        assert len({out.data_ptr() for out in outputs.values()})==len(outputs)
        times={tag:[] for tag in allarms};rng=_rrrandom.Random(101031+inp)
        for rd in range(100):
            order=list(allarms);rng.shuffle(order)
            for tag in order:
                flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();graphs[tag].replay();b.record();b.synchronize()
                times[tag].append(1000*a.elapsed_time(b))
        null=[a-b for a,b in zip(times['b_old'],times['a_old'])]
        mn=_rrstats.median(null);mad=_rrstats.median(abs(d-mn) for d in null)
        threshold=max(.3,2*mad)
        _rrprint(f'RRNULL input={inp} old2-old={mn:+.4f} MAD={mad:.4f} threshold={threshold:.4f}')
        for tag,v in times.items():
            ds=[a-b for a,b in zip(v,times['prod'])];d=_rrstats.median(ds)
            accept=(tag.endswith('_ring') and d < -threshold and sum(z<0 for z in ds)>=90)
            _rrprint(f'RRTIME input={inp} {tag} raw={_rrstats.median(v):.4f} delta={d:+.4f} faster={sum(z<0 for z in ds)}/100 accept={accept}')
    _rrprint(f'RRDONE computing_checks_complete={complete} elapsed={time.time()-started:.1f}; probe arms never scored')

_rr_original_prepare=prepare
def prepare(weights):
    started=time.time();state=_rr_original_prepare(weights);_rrbench(weights,state,started)
    for line in _RRLOG:print(line,flush=True)
    return state
'''

def make(source):
    cuda=generate(source)
    fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='rd_pack')
    pack=ast.get_source_segment(source,fn).replace('def rd_pack(w, geom_name):','def _rr_pack(w):')
    assert pack.count('mt, wr, wk, ku, _ = RD_GEOM[geom_name]')==1
    pack=pack.replace('mt, wr, wk, ku, _ = RD_GEOM[geom_name]','mt, wr, wk, ku, _ = (2,2,16,8,1)')
    result=source+RUN.replace('@@PACK@@',pack).replace('@@CUDA@@',repr(cuda))
    ast.parse(result)
    return result

if __name__=='__main__':
    cpu_gate();source=(ROOT/'muse_bf16_proj/submission.py').read_text()
    Path(sys.argv[1]).write_text(make(source))
    print('CPU source scaffold parse PASS; no mutation of production RD_GEOM')
