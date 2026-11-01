"""Bounded qkv256 low-smem large-tile DSMEM test, actual cluster residency gate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
RESOURCE=r'''
#include <vector>
template<int S,int A=0,int V=0> std::vector<int64_t> dense_info_one() {
 using C=WdCfg<2,128,4,1,false,S,V>;
 auto k=wd_kernel<2,128,4,1,false,S,__nv_bfloat16,A,false,V>;
 C10_CUDA_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,C::SMEM));
 cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));
 cudaLaunchConfig_t lc={};lc.gridDim=V?dim3(144,1):dim3(72,2);lc.blockDim=dim3(C::THREADS);lc.dynamicSmemBytes=C::SMEM;
 cudaLaunchAttribute at[1];at[0].id=cudaLaunchAttributeClusterDimension;
 at[0].val.clusterDim.x=4;at[0].val.clusterDim.y=1;at[0].val.clusterDim.z=1;lc.attrs=at;lc.numAttrs=1;
 int clusters=0,blocks=0;C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters,k,&lc));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,C::THREADS,C::SMEM));
 return {clusters,blocks,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,C::SMEM};
}
std::vector<int64_t> dense_info(int64_t s,int64_t a,int64_t v){
 if(v)return a==3?dense_info_one<3,3,8388608>():dense_info_one<3,0,8388608>();
 if(a==1)return dense_info_one<3,1>();if(a==2)return dense_info_one<3,2>();if(a==3)return dense_info_one<3,3>();
 return s==3?dense_info_one<3>():dense_info_one<4>();}
'''
RUN=r'''
import random as _dsrandom
import statistics as _dsstats
_DS_OUT=[]
_DS_ABLATE=@@ABLATE@@
_DS_TOKFAST=@@TOKFAST@@
def _dsprint(s):_DS_OUT.append(s);print(s,flush=True)
def _dsbuild(name):
    names=('SK_CONFIGS','SK_INDEX','WD_GEOM','WD_VAR','WD_NAMES','WD_INDEX','WD_ABL','_WD_SRC')
    g=globals();old={n:g[n] for n in names}
    try:
        g['SK_CONFIGS']=[c for c in old['SK_CONFIGS'] if c[0] in ROUTER_GEOM]
        g['SK_INDEX']={c[0]:i for i,c in enumerate(g['SK_CONFIGS'])}
        g['WD_GEOM']={'ds3':(2,128,4,1,False,3),'ds4':(2,128,4,1,False,4)}
        g['WD_VAR']={};g['WD_NAMES']=['ds3','ds4'];g['WD_ABL']={}
        if _DS_ABLATE:
            g['WD_ABL']={f'ds3_a{a}':('ds3',a) for a in (1,2,3)}
            g['WD_NAMES']+=list(g['WD_ABL'])
        if _DS_TOKFAST:
            g['WD_GEOM']['ds3tf']=(2,128,4,1,False,3)
            g['WD_VAR']={'ds3tf':8388608,'ds3tf_a3':8388608}
            g['WD_ABL']={'ds3_a3':('ds3',3),'ds3tf_a3':('ds3tf',3)}
            g['WD_NAMES']=['ds3','ds4','ds3_a3','ds3tf','ds3tf_a3']
            changes=[
                ('const int rowbase = RKORD ?', 'static_assert((VAR & 8388608) == 0 || (!C::GRED && CLR == 1), "tokenfast cluster form");\n    const int rowbase = (VAR & 8388608) ? (int)(blockIdx.x / (CL * ntt_o)) * C::BM : RKORD ?'),
                ('const int tok0 = RKORD ?', 'const int tok0 = (VAR & 8388608) ? (int)((blockIdx.x / CL) % ntt_o) * BN : RKORD ?'),
                ('lc.gridDim = (VAR & WV_RKORD) ?', 'lc.gridDim = (VAR & 8388608) ? dim3((unsigned)(N / C::BM * C::CL * ((M + BN - 1) / BN)), 1u) : (VAR & WV_RKORD) ?')]
            for old,new in changes:
                assert g['_WD_SRC'].count(old)==1,old
                g['_WD_SRC']=g['_WD_SRC'].replace(old,new)
        g['WD_INDEX']={n:i for i,n in enumerate(g['WD_NAMES'])}
        source=sk_source()+wd_source()+@@RESOURCE@@
    finally:g.update(old)
    return load_inline(name=name,cpp_sources='#include <torch/extension.h>\n'+_SK_CPP+_WD_CPP+'std::vector<int64_t> dense_info(int64_t,int64_t,int64_t);',cuda_sources=source,functions=['sk_fwd','wd_fwd','dense_info'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],no_implicit_headers=True,verbose=False)
def _dsbench(weights,state):
    dev=weights['qkv_gate'].device;w=weights['qkv_gate'];gen=torch.Generator(device=dev).manual_seed(71031)
    diag=torch.zeros(4,dtype=torch.int32,device=dev);arms={'prod':state['pick']['qkv_gate',256]}
    for label in ('a','b'):
        mod=_dsbuild('okc_dense_dsm_'+label)
        cases=[(3,0,0,0),(4,0,1,0)]+([(3,a,a+1,0) for a in (1,2,3)] if _DS_ABLATE else [])
        if _DS_TOKFAST:cases=[(3,0,0,0),(3,3,2,0),(3,0,3,8388608),(3,3,4,8388608)]
        for st,abl,cid,var in cases:
            tag=label+'s'+str(st)+('_tf' if var else '')+(f'_a{abl}_probe' if abl else '')
            info=mod.dense_info(st,abl,var)
            _dsprint(f'DRES {tag} clusters/blocks/regs/localbytes/staticshared/dynamicshared={info}')
            if info[0]<36:
                _dsprint(f'DSKIP {label}s{st} active clusters below36 for144CTA grid');continue
            def run(x,mod=mod,cid=cid):
                out=torch.empty((x.shape[0],2304),dtype=torch.bfloat16,device=dev)
                mod.wd_fwd(cid,w,x,out,2304,8192,diag);return out
            if abl:
                _dsprint(f'DPROBE {tag}: wrong/unguaranteed output, never scored or correctness evidence')
                arms[tag]=run;continue
            good=True
            for m in (129,200,256):
                x,probes=_gen_x(m,8192,gen,dev);out=run(x)
                ok,err,pe=_check(out,x.double()@w.double().T,probes,SELFTEST_TOL['bf16'])
                _dsprint(f'DCHECK {label}s{st} m={m} {ok} err={err:.3g}/{pe:.3g}')
                if not ok:good=False
            if good:arms[tag]=run
    x,_=_gen_x(256,8192,gen,dev);graphs={};outs={}
    for nm,fn in arms.items():
        fn(x);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn(x)
        graphs[nm]=g;outs[nm]=out
    for rep in range(2):
        xx,probes=_gen_x(256,8192,gen,dev);x.copy_(xx)
        ref=xx.double()@w.double().T
        for nm,g in list(graphs.items()):
            if nm.endswith('_probe'):continue
            g.replay();torch.cuda.synchronize();ok,err,pe=_check(outs[nm],ref,probes,SELFTEST_TOL['bf16'])
            _dsprint(f'DREPLAY {nm} rep={rep} {ok} err={err:.3g}/{pe:.3g}')
            if not ok:graphs.pop(nm)
    flush=torch.empty(256*1024*1024,dtype=torch.uint8,device=dev)
    times={nm:[] for nm in graphs};rng=_dsrandom.Random(71031)
    for rd in range(100):
        order=list(graphs);rng.shuffle(order)
        for nm in order:
            flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
            a.record();graphs[nm].replay();z.record();z.synchronize();times[nm].append(1000*a.elapsed_time(z))
    for nm,t in times.items():
        delta=[a-b for a,b in zip(t,times['prod'])]
        _dsprint(f'DTIME {nm} median_raw_us={_dsstats.median(t):.4f} delta={_dsstats.median(delta):+.4f} faster={sum(v<0 for v in delta)}/100')
_ds_original_prepare=prepare
def prepare(weights):
    state=_ds_original_prepare(weights)
    _dsbench(weights,state)
    for line in _DS_OUT:print(line,flush=True)
    return state
'''
if __name__=='__main__':
    if '--tokenfast' in sys.argv:
        for m in (129,200,256):
            nt=(m+127)//128;seen=set();writers=set()
            for block in range(18*nt*4):
                row=block//(4*nt);tok=(block//4)%nt;rank=block%4
                assert (row,tok,rank) not in seen;seen.add((row,tok,rank))
                assert len({((b//4)%nt,b//(4*nt)) for b in range(block//4*4,block//4*4+4)})==1
                for t in range(tok*128+rank*32,min(tok*128+(rank+1)*32,m)):
                    assert (row,t) not in writers;writers.add((row,t))
            assert len(seen)==18*nt*4 and len(writers)==18*m
            assert [k for r in range(4) for k in range(r*128//4,(r+1)*128//4)]==list(range(128))
        print('CPU tokenfast cluster mapping/unique-writer gate PASS')
    Path(sys.argv[1]).write_text((ROOT/'muse_bf16_proj/submission.py').read_text()+RUN.replace('@@RESOURCE@@',repr(RESOURCE)).replace('@@ABLATE@@',repr('--ablate' in sys.argv)).replace('@@TOKFAST@@',repr('--tokenfast' in sys.argv)))
    print('wrote',sys.argv[1])
