"""Generate private four-arm owned cluster-merge check, never submit."""
import ast
from pathlib import Path
import sys
import tempfile
import hashlib
import attn_make_bench_ab as ab
from attn_cluster_merge_patch import patch_cuda, once

RESOURCE=r'''
std::vector<int64_t> attn_one_cluster_resources() {
    auto k=attn_one_kernel<8,3>; Instance<8,3>::prepare();
    cudaFuncAttributes a{}; int blocks=0,clusters=0;
    TORCH_CHECK(cudaFuncGetAttributes(&a,k)==cudaSuccess,"cluster attrs");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,256,Instance<8,3>::BYTES)==cudaSuccess,"cluster blocks");
    TORCH_CHECK(cudaFuncSetAttribute(k,cudaFuncAttributeNonPortableClusterSizeAllowed,1)==cudaSuccess,"cluster opt-in");
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(32);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=Instance<8,3>::BYTES;
    cudaLaunchAttribute attrs[2]{};attrs[0].id=cudaLaunchAttributeCooperative;attrs[0].val.cooperative=1;
    attrs[1].id=cudaLaunchAttributeClusterDimension;attrs[1].val.clusterDim.x=16;attrs[1].val.clusterDim.y=1;attrs[1].val.clusterDim.z=1;
    cfg.attrs=attrs;cfg.numAttrs=2;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&clusters,k,&cfg)==cudaSuccess,"cluster occupancy");
    return {a.numRegs,(int64_t)a.localSizeBytes,Instance<8,3>::BYTES,blocks,clusters};
}
'''
SUPPORT=r'''
def _cm_native(data):
    c=data['config']
    return c['batch']==1 and c['q_len']==8 and c['window']==2048 and c['page_size']==1 and c['kv_scale']==1.0

_CM_RES={}
def _cm_resources():
    base=_ONE.attn_one_prepare(); rows=[base[i:i+5] for i in range(0,len(base),5)]
    baseline=next(r for r in rows if r[:2]==[8,3])
    assert set(_AB_MODS)=={'ctrl2','cm1','cm2'}
    for name in ('cm1','cm2'):
        if name not in _CM_RES:
            v=_AB_MODS[name].attn_one_cluster_resources();_CM_RES[name]=v
            assert v[1]==0 and v[2]==baseline[3] and v[3]>=baseline[4] and v[4]>=2,(name,v,baseline)
        print('CMRESOURCE',name,_CM_RES[name],'base8mode3',baseline,flush=True)

_cm_original_via=_ab_via
def _ab_via(name):
    if not name.startswith('cm'):return _cm_original_via(name)
    def call(data):
        if not _cm_native(data):return _one(data)
        with _ab_ctx(name):return _one(data,splits=32,nc=1,mode=3)
    return call

_cm_original_plan=_ab_plan
def _ab_plan(name,data):
    if name.startswith('cm'):
        return 'owned32CTA/2x16cluster/64keytile' if _cm_native(data) else 'held primary control'
    return _cm_original_plan(name,data)
'''

FRESH=r'''
def _cm_rel(a,b):
    return (a.float()-b.float()).norm()/b.float().norm().clamp(min=1e-12)

def _cm_fresh():
    data=_ob_inputs(1,80000,'tp8','sliding',8,'fixed','soft',1,1.0,seed=311179)
    dev=data['qkv'].device
    past=torch.arange(80000-2047,80000,device=dev)
    physical=data['page_table'][0,past].long()
    kp=data['k_cache'].view(-1,2,64)[physical].float()
    vp=data['v_cache'].view(-1,2,64)[physical].float()
    positions=80000+torch.arange(8,device=dev)
    keypositions=torch.arange(80000-2047,80008,device=dev)
    mask=(keypositions[None,:]<=positions[:,None]) & (keypositions[None,:]>=positions[:,None]-2047)
    functions=_ab_fns();graphs={}
    for name,fn in functions.items():
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):fn(data)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):out=fn(data)
        torch.cuda.synchronize();graphs[name]=(graph,out)
    gen=torch.Generator(device=dev).manual_seed(311179)
    bank=torch.randn((64,*data['qkv'].shape),generator=gen,device=dev).to(torch.bfloat16)
    verdicts=[];negatives=[];old=None
    saved_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        for i in range(64):
            data['qkv'].copy_(bank[i])
            rawq,rawk,rawv,gate=data['qkv'].split([1024,128,128,1024],-1)
            q=_ob_rope(_ob_rms(rawq.view(1,8,16,64))*5.4730064863875,positions[None,:])[0]
            qb=q.to(torch.bfloat16).float().view(8,2,8,64)
            q8=(q/5.4730064863875).to(FP8).float().mul(5.4730064863875).view(8,2,8,64)
            expected_k=_ob_rope(_ob_rms(rawk.view(1,8,2,64)),positions[None,:])[0].to(FP8).float()
            expected_v=rawv.view(8,2,64).to(FP8).float()
            for name,(graph,out) in graphs.items():
                graph.replay()
                k=data['k_cache'].view(-1,2,64)[data['cache_loc']].float()
                v=data['v_cache'].view(-1,2,64)[data['cache_loc']].float()
                keys=torch.cat((kp,k));values=torch.cat((vp,v))
                def reference(query):
                    score=torch.einsum('lhgd,thd->lhgt',query,keys)*0.125
                    prob=score.masked_fill(~mask[:,None,None,:],float('-inf')).softmax(-1)
                    a=torch.einsum('lhgt,thd->lhgd',prob,values).reshape(1,8,16,64)
                    return (_ob_rms(a)*torch.sigmoid(gate.float()).view(1,8,16,64)).to(torch.bfloat16).view(1,8,1024)
                ref=reference(qb);ref8=reference(q8)
                bound=(2.5*_cm_rel(ref8,ref)+0.015).clamp(0.03,0.15)
                rownorm=ref.float().view(-1,64).norm(dim=-1)
                floor=(0.1*rownorm.mean()).clamp(min=1e-12)
                def valid(output):
                    worst=((output.float()-ref.float()).view(-1,64).norm(dim=-1)/torch.maximum(rownorm,floor)).max()
                    return torch.isfinite(output).all() & (_cm_rel(output,ref)<=bound) & (worst<=5*bound)
                verdicts.append(valid(out) & (_cm_rel(k,expected_k)<=0.08) & (_cm_rel(v,expected_v)<=0.08))
                if name=='ctrl' and i==1:
                    negatives += [~valid(torch.zeros_like(out)),~valid(out+10),~valid(old)]
                if name=='ctrl':old=out.clone()
    finally:torch.backends.cuda.matmul.allow_tf32=saved_tf32
    # Only verdict drain after all256 same-stream fresh replays/reference checks.
    assert bool(torch.stack(verdicts+negatives).all()),'fresh official math/cache/row or negative gate'
    print('CMFRESH inputs64 arms4 checked256 negatives3 pass; no inter-replay hostsync',flush=True)
    for name in functions:
        with _ab_ctx(name):
            ok,msg=_ob_ws_check(dev)
            assert ok,(name,msg)
    print('CMFRESH all private workspace reset scans pass',flush=True)
'''


def generate(control,output):
    text=control.read_text(encoding='utf-8');tree=ast.parse(text)
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ONE_SRC' for t in n.targets))
    patched=patch_cuda(src)+RESOURCE
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'muse_attention_decode').mkdir();(root/'muse_attention_decode/submission.py').write_text(text,encoding='utf-8')
        oldroot,oldargv=ab.ROOT,sys.argv
        try:
            ab.ROOT=root;sys.argv=['bench',str(output),'--null','--shapes','all','--inputs','3','--rounds','120','--seed','311171','cm1@own=-DONE_CLUSTER_MERGE=1','cm2@own=-DONE_CLUSTER_MERGE=1,-DCM_NULL_BUILD=1'];ab.main()
        finally:ab.ROOT,sys.argv=oldroot,oldargv
    s=output.read_text(encoding='utf-8')
    s=once(s,'_AB_MODS = {}','_CM_SRC = '+repr(patched)+'\n_CM_CPP = _ONE_CPP + "\\nstd::vector<int64_t> attn_one_cluster_resources();\\n"\n_AB_MODS = {}')
    s=once(s,'            cpp_sources=_ONE_CPP,\n            cuda_sources=_ONE_SRC,','            cpp_sources=_CM_CPP if _name.startswith("cm") else _ONE_CPP,\n            cuda_sources=_CM_SRC if _name.startswith("cm") else _ONE_SRC,')
    s=once(s,'            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],','            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"] + (["attn_one_cluster_resources"] if _name.startswith("cm") else []),')
    s=once(s,'def _ab_check():',SUPPORT+FRESH+'\n\ndef _ab_check():\n    global _CM_CHECK_FAILED\n    _CM_CHECK_FAILED=True\n    _cm_resources()')
    s=once(s,'                    ok, msg = rel < 2e-2, ""','                    old_tf32=torch.backends.cuda.matmul.allow_tf32\n                    try:\n                        torch.backends.cuda.matmul.allow_tf32=False\n                        ok,msg=_ob_check(data,out)\n                    finally:torch.backends.cuda.matmul.allow_tf32=old_tf32\n                    ok = ok and rel < 2e-2')
    s=once(s,'    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)','    assert fails==0,"cluster official-math/row/cache gate"\n    _cm_fresh()\n    _CM_CHECK_FAILED=False\n    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)')
    s=once(s,'def _ab_bench(warm_calls=16, warm_reps=25):','def _ab_bench(warm_calls=16, warm_reps=25):\n    assert not _CM_CHECK_FAILED, "no timing after failed gate"')
    s += '\nassert not _CM_CHECK_FAILED, "cluster gate failed"\n'
    s=once(s,'            _ab_stsum()','            _cm_resources()\n            _ab_stsum()')
    compile(s,str(output),'exec');output.write_text(s,encoding='utf-8');print('CLUSTER_BENCH_SHA',hashlib.sha256(s.encode()).hexdigest())


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args();generate(a.control,a.output)
