"""Owned qkv128 isolated WGMMA depth2 experiment; never submit this bench."""
from pathlib import Path
import ast
import sys
ROOT=Path(__file__).resolve().parents[1]
BIT=8388608

RESOURCE=r'''
#include <vector>
template<int V>std::vector<int64_t> qdepth_info_one(){
 using C=WdCfg<1,128,3,1,false,8,V>;
 auto k=wd_kernel<1,128,3,1,false,8,__nv_bfloat16,0,true,V>;
 C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));
 cudaLaunchConfig_t lc={};lc.gridDim=dim3(108,1);lc.blockDim=dim3(C::THREADS);lc.dynamicSmemBytes=C::SMEM;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;
 attr.val.clusterDim.x=3;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;lc.attrs=&attr;lc.numAttrs=1;
 int clusters=0,blocks=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters,k,&lc));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,C::SMEM));
 return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,C::SMEM,blocks,clusters};
}
std::vector<int64_t> qdepth_info(bool deep){return deep?qdepth_info_one<8388608>():qdepth_info_one<0>();}
'''
RUN=r'''
import random as _qdrandom
import statistics as _qdstats
_QDLOG=[]
def _qdprint(s):_QDLOG.append(s);print(s,flush=True)

def _qdbuild(name):
    keys=('SK_CONFIGS','SK_INDEX','WD_GEOM','WD_VAR','WD_NAMES','WD_INDEX','WD_ABL','WD_POL','_WD_SRC')
    scope=globals();snapshot={key:scope[key] for key in keys}
    try:
        scope['SK_CONFIGS']=[];scope['SK_INDEX']={}
        scope['WD_GEOM']={'qd1':(1,128,3,1,False,8),'qd2':(1,128,3,1,False,8)}
        scope['WD_VAR']={'qd2':8388608};scope['WD_NAMES']=['qd1','qd2'];scope['WD_INDEX']={'qd1':0,'qd2':1}
        scope['WD_ABL']={};scope['WD_POL']={'qd1','qd2'};scope['_WD_SRC']=@@SOURCE@@
        cuda=sk_source()+wd_source()+@@RESOURCE@@
    finally:scope.update(snapshot)
    return load_inline(name=name,cpp_sources='#include <torch/extension.h>\n'+_SK_CPP+_WD_CPP+'std::vector<int64_t> qdepth_info(bool);',
        cuda_sources=cuda,functions=['wd_fwd','qdepth_info'],no_implicit_headers=True,
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],verbose=False)

def _qdbench(weights,state,started):
    mods={tag:_qdbuild('okc_qkv_depth_'+tag) for tag in ('a','b')}
    for tag,mod in mods.items():
        old=mod.qdepth_info(False);new=mod.qdepth_info(True)
        _qdprint(f'QDRES {tag} depth1={old} depth2={new} fields=regs,local,static,dynamic,blocks,clusters')
        if new[1] or new[4]<old[4] or new[5]<36:
            _qdprint('QDSKIP depth2 resource gate failed; no performance winner');return
    w=weights['qkv_gate'];dev=w.device;gen=torch.Generator(device=dev).manual_seed(111031)
    arms={'prod':state['pick']['qkv_gate',128]};diags=[]
    for tag,mod in mods.items():
        for deep in (False,True):
            diag=torch.zeros(4,dtype=torch.int32,device=dev);diags.append(diag)
            def make(mod=mod,deep=deep,diag=diag):
                def run(x):
                    out=torch.empty((len(x),2304),dtype=torch.bfloat16,device=dev)
                    mod.wd_fwd(int(deep),w,x,out,2304,8192,diag);return out
                return run
            arms[tag+('_d2' if deep else '_d1')]=make()
    def check(fn,x):
        y=fn(x);ref=x.double()@w.double().T
        err=float(torch.linalg.vector_norm(y.double()-ref)/torch.linalg.vector_norm(ref))
        assert bool(torch.isfinite(y).all()) and err<.005,err
        assert all(not int(d.abs().max()) for d in diags),'qkv depth watchdog'
        return err
    for m in (100,127,128,129):
        for probe in (False,True):
            x=torch.randn((m,8192),device=dev,generator=gen).to(torch.bfloat16)
            if probe:x.zero_();x[:,17]=2
            errors={tag:check(fn,x) for tag,fn in arms.items() if tag!='prod' or m==128}
            _qdprint(f'QDCHECK m={m} probe={probe} PASS {errors}')
    for tag,fn in arms.items():
        x=torch.randn((128,8192),device=dev,generator=gen).to(torch.bfloat16);fn(x)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn(x)
        for fresh in range(3):
            x.copy_(torch.randn(x.shape,device=dev,generator=gen).to(torch.bfloat16));g.replay()
            ref=x.double()@w.double().T
            assert float(torch.linalg.vector_norm(out.double()-ref)/torch.linalg.vector_norm(ref))<.005
        _qdprint(f'QDREPLAY {tag} 3fresh captured inputs PASS')
    flush=torch.empty(256<<20,dtype=torch.uint8,device=dev)
    for inp in range(2):
        if time.time()-started>280:raise RuntimeError('qkv-depth comparison incomplete280s')
        x=torch.randn((128,8192),device=dev,generator=gen).to(torch.bfloat16)
        graphs={};outs={};times={tag:[] for tag in arms}
        for tag,fn in arms.items():
            check(fn,x);g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn(x)
            graphs[tag]=g;outs[tag]=out
        assert len({o.data_ptr() for o in outs.values()})==len(outs)
        rng=_qdrandom.Random(111031+inp)
        for rd in range(100):
            order=list(arms);rng.shuffle(order)
            for tag in order:
                flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
                a.record();graphs[tag].replay();b.record();b.synchronize();times[tag].append(1000*a.elapsed_time(b))
        null=[a-b for a,b in zip(times['b_d1'],times['a_d1'])];mn=_qdstats.median(null)
        mad=_qdstats.median(abs(z-mn) for z in null);threshold=max(.2,2*mad)
        _qdprint(f'QDNULL input={inp} b_d1-a_d1={mn:+.4f} MAD={mad:.4f} threshold={threshold:.4f}')
        for tag,ts in times.items():
            ds=[a-b for a,b in zip(ts,times['prod'])];delta=_qdstats.median(ds)
            _qdprint(f'QDTIME input={inp} {tag} raw={_qdstats.median(ts):.4f} delta={delta:+.4f} faster={sum(z<0 for z in ds)}/100')
        for tag in ('a','b'):
            ds=[a-b for a,b in zip(times[tag+'_d2'],times[tag+'_d1'])]
            _qdprint(f'QDCAUSAL input={inp} {tag} d2-d1={_qdstats.median(ds):+.4f} faster={sum(z<0 for z in ds)}/100')
    assert all(not int(d.abs().max()) for d in diags),'qkv depth watchdog'
    _qdprint(f'QDDONE computing gates complete elapsed={time.time()-started:.1f}')

_qd_original_prepare=prepare
def prepare(weights):
    started=time.time();state=_qd_original_prepare(weights);_qdbench(weights,state,started)
    for line in _QDLOG:print(line,flush=True)
    return state
'''

def patch(src):
    edits={
      'const int g = ((VAR & WV_UDESC)': 'constexpr int QDEPTH=(VAR & 8388608)?2:1;\n        static_assert((VAR & 8388608)==0 || (WG==1 && BN==128 && CLK==3 && CLR==1 && !HL && NST==8),"bounded qkv128 depth2");\n        const int g = ((VAR & WV_UDESC)',
      'wg_wait<1>();':'wg_wait<QDEPTH>();',
      'if (i > 0 && ((VAR & WV_E1)': 'if (i >= QDEPTH && ((VAR & WV_E1)',
      '((i - 1) % NST)':'((i - QDEPTH) % NST)',
    }
    for old,new in edits.items():
        assert src.count(old)==1,(old,src.count(old));src=src.replace(old,new)
    return src

def cpu_gate(source):
    wd=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='_WD_SRC' for t in n.targets))
    new=patch(wd)
    for depth in (1,2):
        for steps in (42,43):
            loaded=set(range(8));completed=set();released=set()
            for i in range(steps):
                assert i in loaded
                # After commit i and wait D, every group through i-D has completed.
                completed.update(range(max(0,i-depth+1)))
                if i>=depth:
                    done=i-depth;assert done in completed;released.add(done)
                    if done+8<steps:
                        assert done in released;loaded.add(done+8)
            assert loaded==set(range(steps))
    assert wd.count('wg_wait<0>();')==new.count('wg_wait<0>();')
    assert wd[wd.index('        if constexpr (!PUSH) {',wd.index('wg_wait<1>();')):]==new[new.index('        if constexpr (!PUSH) {',new.index('wg_wait<QDEPTH>();')):]
    return new

def make(source):
    wd=cpu_gate(source)
    out=source+RUN.replace('@@SOURCE@@',repr(wd)).replace('@@RESOURCE@@',repr(RESOURCE))
    ast.parse(out);return out

def host_gate(source):
    # Run actual builder with fake source emitters/compiler; prove all globals restored.
    ns={};keys=('SK_CONFIGS','SK_INDEX','WD_GEOM','WD_VAR','WD_NAMES','WD_INDEX','WD_ABL','WD_POL','_WD_SRC')
    saved={k:object() for k in keys};ns.update(saved)
    ns.update(_SK_CPP='',_WD_CPP='',sk_source=lambda:'sk',wd_source=lambda:'wd',load_inline=lambda **kw:kw)
    tree=ast.parse(make(source));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_qdbuild')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual builder>','exec'),ns)
    result=ns['_qdbuild']('cpu_gate')
    assert result['cuda_sources'].startswith('skwd') and all(ns[k] is saved[k] for k in keys)

if __name__=='__main__':
    source=(ROOT/'muse_bf16_proj/submission.py').read_text(encoding='utf-8')
    host_gate(source);Path(sys.argv[1]).write_text(make(source),encoding='utf-8')
    print('CPU stage lifetime, unchanged epilogue, host restoration and parse gates PASS')
