"""Owned complete-head cluster A/B harness; emits private CHECK ONLY artifact."""
import ast
import hashlib
from pathlib import Path
import sys
import tempfile
import attn_make_bench_ab as ab
from attn_cluster_merge_patch import once
from attn_headcluster_patch import patch_headcluster
from attn_make_cluster_merge import FRESH as OLD_FRESH

RESOURCE=r'''
std::vector<int64_t> attn_one_head_resources(){
    auto k=attn_one_kernel<8,3>;Instance<8,3>::prepare();cudaFuncAttributes a{};int blocks=0,clusters=0;
    TORCH_CHECK(cudaFuncGetAttributes(&a,k)==cudaSuccess,"head resources");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,k,128,Instance<8,3>::BYTES)==cudaSuccess,"head blocks");
    cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(32);cfg.blockDim=dim3(128);cfg.dynamicSmemBytes=Instance<8,3>::BYTES;
    cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim.x=16;attr.val.clusterDim.y=1;attr.val.clusterDim.z=1;cfg.attrs=&attr;cfg.numAttrs=1;
    TORCH_CHECK(cudaOccupancyMaxActiveClusters(&clusters,k,&cfg)==cudaSuccess,"head clusters");
    return {a.numRegs,(int64_t)a.localSizeBytes,Instance<8,3>::BYTES,128,blocks,clusters};
}
'''
SUPPORT=r'''
def _hc_native(data):
    c=data['config']
    return (c['batch']==1 and c['q_len']==8 and c['window']==2048 and c['page_size']==1 and c['kv_scale']==1.0
            and (c['num_q_heads'],c['num_kv_heads'],c['head_dim'])==(16,2,64)
            and data['qkv'].dtype==torch.bfloat16 and data['qkv'].is_contiguous()
            and data['k_cache'].dtype==FP8 and data['v_cache'].dtype==FP8
            and data['k_cache'].is_contiguous() and data['v_cache'].is_contiguous()
            and max(data['k_cache'].numel(),data['v_cache'].numel())<(1<<32)
            and data['page_table'].dtype==torch.int32 and data['page_table'].stride(1)==1
            and data['seq_lens'].dtype==torch.int32 and data['cache_loc'].dtype==torch.int64)

_HC_RES={}
_HC_FRESH_STATS={}
def _hc_resources():
    lim=_ONE.attn_one_prepare();rows=[lim[i:i+5] for i in range(0,len(lim),5)]
    base=next(r for r in rows if r[:2]==[8,3])
    assert set(_AB_MODS)=={'ctrl2','hc1','hc2'},'missing independent head builds'
    for name in ('hc1','hc2'):
        if name not in _HC_RES:
            v=_AB_MODS[name].attn_one_head_resources();_HC_RES[name]=v
            assert v[1]==0 and v[2]==base[3] and v[3]==128 and v[4]>=2 and v[5]>=2,(name,v,base)
        print('HCRESOURCE',name,_HC_RES[name],'base8mode3',base,flush=True)
    if _HC_FRESH_STATS:print('HCFRESH_STATS',_HC_FRESH_STATS,flush=True)

_hc_original_via=_ab_via
def _ab_via(name):
    if not name.startswith('hc'):return _hc_original_via(name)
    def call(data):
        if not _hc_native(data):return _one(data)
        with _ab_ctx(name):return _one(data,splits=32,nc=1,mode=3)
    return call

_hc_original_plan=_ab_plan
def _ab_plan(name,data):
    if name.startswith('hc'):return '32CTA/2complete-head-clusters/128threads/direct-output' if _hc_native(data) else 'held primary control'
    return _hc_original_plan(name,data)

_hc_original_check=_ob_check
def _ob_check(data,out):
    previous=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return _hc_original_check(data,out)
    finally:torch.backends.cuda.matmul.allow_tf32=previous
'''
FRESH=OLD_FRESH.replace('_cm','_hc').replace('CMFRESH','HCFRESH')
FRESH=once(FRESH,'    functions=_ab_fns();graphs={}',"""    functions=_ab_fns();graphs={}
    armdata={name:{k:(v.clone() if isinstance(v,torch.Tensor) else dict(v) if isinstance(v,dict) else v)
                   for k,v in data.items()} for name in functions}
    for key in ('qkv','k_cache','v_cache'):
        assert len({d[key].data_ptr() for d in armdata.values()})==len(functions)""")
FRESH=once(FRESH,'    for name,fn in functions.items():','    for name,fn in functions.items():\n        data=armdata[name]')
FRESH=once(FRESH,"            data['qkv'].copy_(bank[i])","            for d in armdata.values():d['qkv'].copy_(bank[i])")
FRESH=once(FRESH,'            for name,(graph,out) in graphs.items():','            for name,(graph,out) in graphs.items():\n                data=armdata[name]')
FRESH=once(FRESH,"a=torch.einsum('lhgt,thd->lhgd',prob,values).reshape(1,8,16,64)","a=torch.einsum('lhgt,thd->lhgd',prob,values).reshape(1,8,16,64).to(torch.bfloat16).float()")
FRESH=once(FRESH,'    verdicts=[];negatives=[];old=None','    global _HC_FRESH_STATS\n    verdicts=[];negatives=[];old=None;maxima=[]')
FRESH=once(FRESH,'                verdicts.append(valid(out) & (_hc_rel(k,expected_k)<=0.08) & (_hc_rel(v,expected_v)<=0.08))','''                worst=((out.float()-ref.float()).view(-1,64).norm(dim=-1)/torch.maximum(rownorm,floor)).max()
                maxima.append(torch.stack((_hc_rel(out,ref),worst,_hc_rel(k,expected_k),_hc_rel(v,expected_v),bound)))
                verdicts.append(valid(out) & (_hc_rel(k,expected_k)<=0.08) & (_hc_rel(v,expected_v)<=0.08))''')
FRESH=once(FRESH,"    assert bool(torch.stack(verdicts+negatives).all()),'fresh official math/cache/row or negative gate'",'''    summary=torch.cat((torch.stack(verdicts+negatives).all().float().view(1),torch.stack(maxima).amax(0))).tolist()
    assert summary[0]==1,'fresh official math/cache/row or negative gate'
    _HC_FRESH_STATS={'distinct_qkv':64,'arms':4,'checked':256,'negatives_rejected':3,'max_error':summary[1],'max_row_error':summary[2],'max_k_error':summary[3],'max_v_error':summary[4],'max_bound':summary[5]}''')


def generate(control,output):
    text=control.read_text(encoding='utf-8');tree=ast.parse(text)
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_ONE_SRC' for t in n.targets))
    patched=patch_headcluster(src)+RESOURCE
    patched=once(patched,'#if ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)','#if defined(ONE_HEAD_ONLY) && ONE_HEAD_ONLY\n#define ONE_INSTANCES(X) X(8,3)\n#elif ONE_DETM\n#define ONE_INSTANCES(X) X(1, 49)')
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'muse_attention_decode').mkdir();(root/'muse_attention_decode/submission.py').write_text(text,encoding='utf-8')
        oldroot,oldargv=ab.ROOT,sys.argv
        try:
            ab.ROOT=root;sys.argv=['bench',str(output),'--null','--shapes','all','--inputs','3','--rounds','120','--seed','311201','hc1@own=-DONE_HEADCLUSTER=1,-DONE_HEAD_ONLY=1','hc2@own=-DONE_HEADCLUSTER=1,-DONE_HEAD_ONLY=1,-DHC_NULL_BUILD=1'];ab.main()
        finally:ab.ROOT,sys.argv=oldroot,oldargv
    s=output.read_text(encoding='utf-8')
    s=once(s,'_AB_MODS = {}','_HC_SRC = '+repr(patched)+'\n_HC_CPP = _ONE_CPP + "\\nstd::vector<int64_t> attn_one_head_resources();\\n"\n_AB_MODS = {}')
    s=once(s,'            cpp_sources=_ONE_CPP,\n            cuda_sources=_ONE_SRC,','            cpp_sources=_HC_CPP if _name.startswith("hc") else _ONE_CPP,\n            cuda_sources=_HC_SRC if _name.startswith("hc") else _ONE_SRC,')
    s=once(s,'            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],','            functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"] + (["attn_one_head_resources"] if _name.startswith("hc") else []),')
    s=once(s,'def _ab_check():',SUPPORT+FRESH+'\n\ndef _ab_check():\n    global _HC_CHECK_FAILED\n    _HC_CHECK_FAILED=True\n    _hc_resources()')
    s=once(s,'    for shape in shapes:','    shapes += [(1,0,"tp8","sliding",8,"fixed","normal",1,1.0),(1,10,"tp8","sliding",8,"fixed","normal",1,1.0)]\n    for shape_index,shape in enumerate(shapes):')
    s=once(s,'            data = _ob_inputs(*shape)','            data = _ob_inputs(*shape,seed=311201+shape_index*997)')
    s=once(s,'                    ok, msg = rel < 2e-2, ""','                    ok,msg=_ob_check(data,out)\n                    ok = ok and rel < 2e-2')
    s=once(s,'    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)','    assert fails==0,"head official math/cache/row gate"\n    _hc_fresh()\n    _HC_CHECK_FAILED=False\n    print(f"ABCHECK done: {fails} failing {_ob_elapsed()}", flush=True)')
    s=once(s,'def _ab_bench(warm_calls=16, warm_reps=25):','def _ab_bench(warm_calls=16, warm_reps=25):\n    assert not _HC_CHECK_FAILED,"no timing after failed head gate"')
    s=once(s,'            _ab_stsum()','            _hc_resources()\n            _ab_stsum()')
    s += '\nassert not _HC_CHECK_FAILED,"head check failed"\n'
    compile(s,str(output),'exec');output.write_bytes(s.encode('utf-8'));print('HEADCLUSTER_BENCH_SHA',hashlib.sha256(s.encode()).hexdigest())


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('control',type=Path);p.add_argument('output',type=Path);a=p.parse_args();generate(a.control,a.output)
