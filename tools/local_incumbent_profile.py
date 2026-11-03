"""RTX 5070 profile of portable layer-incumbent stages and its cuBLAS fallback.

This does not execute or estimate the Hopper whole-layer path.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PIN = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=Path('muse_decode_layer/cand/g1b4_stage_release.py'))
    p.add_argument('--out', type=Path, default=Path('.build/local-incumbent-profile'))
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    source_bytes = a.source.read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(source_bytes).hexdigest() == PIN
    spec = importlib.util.spec_from_file_location('local_layer_incumbent', a.source)
    inc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inc)
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20261103)
    prop = torch.cuda.get_device_properties(0)
    result = dict(
        started_at=datetime.now(timezone.utc).isoformat(), source=str(a.source),
        source_lf_sha256=PIN, torch=torch.__version__, cuda=torch.version.cuda,
        gpu=prop.name, capability=list(torch.cuda.get_device_capability()),
        total_memory=prop.total_memory, multiprocessors=prop.multi_processor_count,
        shared_memory_per_block=prop.shared_memory_per_block,
        shared_memory_per_block_optin=prop.shared_memory_per_block_optin,
        scope='Exact portable route/finish kernels; explicit existing cuBLAS dense fallback. NOT whole layer or H200.',
        method=dict(cold_flush_bytes=256*1024*1024, cold_rounds=31,
                    warm_calls_per_graph=64, warm_rounds=11, timing='CUDA events',
                    inputs='Synthetic fresh seeded BF16/FP32, no challenge weights or complete-layer data',
                    clock_lock=False, desktop_gpu=True), rows=[])
    rows = []
    keep = []
    def rand(*shape, dtype=torch.float32):
        return torch.randn(shape, device='cuda', dtype=dtype)
    def errors(actual, ref):
        d = actual.float()-ref.float()
        rel = float(d.norm()/ref.float().norm().clamp_min(1e-12))
        maximum = float(d.abs().max())
        assert torch.isfinite(actual).all() and rel < .016, (rel, maximum)
        return dict(relative_l2=rel, max_abs=maximum)
    for t in (1, 8):
        for name, n, k in [('qkv',2304,8192),('o',8192,1024),
                             ('pool',4096,8192),('un',1024,4096)]:
            x = rand(t,k,dtype=torch.bfloat16)
            w = (rand(n,k)/k**.5).to(torch.bfloat16)
            # Select the incumbent's existing cuBLAS fallback explicitly.
            state = {'mm_pick': {(t,name): True}}
            fn = lambda x=x,w=w,t=t,state=state,name=name: inc.dense(x,w,t,state,name)[0]
            actual = fn().view(t,n)
            check = errors(actual, x.float()@w.float().T)
            rows.append(dict(name=name+'_dense_fallback',tokens=t,scope='existing cuBLAS fallback; H200 selected SK2 NOT executed',
                             shape=[t,n,k], minimum_weight_bytes=w.numel()*w.element_size(),validation=check,fn=fn))
            keep.extend([x,w,state])
        pooled = rand(t,4096,dtype=torch.bfloat16)
        logits = rand(t,256)
        bias = .01*rand(256)
        gpre = (.1*rand(4096)).to(torch.bfloat16)
        ids = torch.empty((t,8),device='cuda',dtype=torch.int32)
        lids = torch.empty_like(ids)
        wts = torch.empty((t,8),device='cuda')
        xm = torch.empty_like(pooled)
        ls = torch.tensor([96],device='cuda',dtype=torch.int32)
        nw, parts, preload = inc.SL_TILE_CONFIG[t]
        def route(pooled=pooled,logits=logits,bias=bias,gpre=gpre,ids=ids,lids=lids,wts=wts,xm=xm,ls=ls,t=t,nw=nw,parts=parts,preload=preload):
            return inc._sl_tile_route[(t,parts+1)](pooled,logits,logits,bias,gpre,ids,wts,xm,ls,lids,
                SPLIT=True,EARLY=inc.SL_ROUTE_CONFIG[t][2],MODE=inc.SL_ROUTE_CONFIG[t][1],
                PARTS=parts,PRELOAD=preload,num_warps=nw,launch_pdl=False)
        route()
        scores = torch.sigmoid(logits*inc.ROUTER_MULT)
        ref_ids = torch.argsort(scores+bias,dim=-1,descending=True,stable=True)[:,:8]
        assert torch.equal(ids,ref_ids.int())
        ref_wts = scores.gather(1,ref_ids)
        ref_wts /= ref_wts.sum(-1,keepdim=True)
        ref_lids = (ref_ids-96+256)%256
        ref_lids = torch.where(ref_lids<32,ref_lids,-1)
        assert torch.equal(lids,ref_lids.int())
        norm = pooled.float()*torch.rsqrt(pooled.float().square().mean(-1,keepdim=True)+1e-5)
        check = dict(ids_equal=True,local_ids_equal=True,weights=errors(wts,ref_wts),
                     normalized_input=errors(xm,(norm*(1+gpre.float())).bfloat16()))
        rows.append(dict(name='route_preexpert',tokens=t,scope='exact public selected portable Triton stage',validation=check,fn=route))
        branch = rand(t,1024,dtype=torch.bfloat16)
        carry = rand(t,8192)
        tail = carry[:,1024:].square().sum(-1).contiguous()
        beta = rand(8192)
        post = (1+.1*rand(8192)).bfloat16()
        nxt = (1+.1*rand(8192)).bfloat16()
        sout = torch.empty_like(carry)
        hout = torch.empty((t,8192),device='cuda',dtype=torch.bfloat16)
        nw,bc = inc.SL_CLUSTER_CONFIG[t]
        def finish(branch=branch,carry=carry,tail=tail,beta=beta,post=post,nxt=nxt,sout=sout,hout=hout,t=t,nw=nw,bc=bc):
            return inc._sl_cluster_finish[(t,8192//bc)](branch,t,carry,tail,beta,post,nxt,sout,hout,
                        BC=bc,num_warps=nw,launch_pdl=False)
        finish()
        f=(branch.float()*post[:1024].float()).bfloat16().float()
        bn=(f*torch.rsqrt(f.square().sum(-1,keepdim=True)/8192+1e-8)).bfloat16().float()
        sr=carry.clone()
        sr[:,:1024] += beta[:1024]*bn
        hr=(sr*torch.rsqrt(sr.square().mean(-1,keepdim=True)+1e-5)*nxt.float()).bfloat16()
        check=dict(stream=errors(sout,sr),hidden=errors(hout,hr))
        rows.append(dict(name='final_norm_residual',tokens=t,scope='exact public selected portable Triton stage',validation=check,fn=finish))
        keep.extend([pooled,logits,bias,gpre,ids,lids,wts,xm,ls,branch,carry,tail,beta,post,nxt,sout,hout])
    flush=torch.empty(256*1024*1024,device='cuda',dtype=torch.uint8)
    flush.zero_()
    for row in rows:
        fn=row.pop('fn')
        for _ in range(5): fn()
        torch.cuda.synchronize()
        one=torch.cuda.CUDAGraph()
        with torch.cuda.graph(one): fn()
        batch=torch.cuda.CUDAGraph()
        with torch.cuda.graph(batch):
            for _ in range(64): fn()
        row['_one']=one
        row['_batch']=batch
        row['cold_us']=[]
        row['warm_us']=[]
    start=torch.cuda.Event(enable_timing=True)
    stop=torch.cuda.Event(enable_timing=True)
    order=list(range(len(rows)))
    rng=random.Random(20261103)
    for rep in range(31):
        rng.shuffle(order)
        for i in order:
            row=rows[i]
            flush.add_(1)
            start.record(); row['_one'].replay(); stop.record(); stop.synchronize()
            row['cold_us'].append(start.elapsed_time(stop)*1000)
            if rep<11:
                row['_batch'].replay()
                start.record(); row['_batch'].replay(); stop.record(); stop.synchronize()
                row['warm_us'].append(start.elapsed_time(stop)*1000/64)
    for row in rows:
        for label in ('cold','warm'):
            vals=sorted(row[label+'_us'])
            row[label+'_summary_us']=dict(median=statistics.median(vals),p10=vals[int(.1*(len(vals)-1))],p90=vals[int(.9*(len(vals)-1))])
        row['minimum_weight_bandwidth_GBs_cold']=(row['minimum_weight_bytes']/row['cold_summary_us']['median']/1000
                                                    if 'minimum_weight_bytes' in row else None)
        print(json.dumps({k:v for k,v in row.items() if not k.startswith('_') and k not in ('cold_us','warm_us')}),flush=True)
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            for row in rows:
                with torch.profiler.record_function(row['name']+'_T'+str(row['tokens'])):
                    row['_one'].replay()
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(a.out/'trace.json'))
        (a.out/'profiler.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40))
        result['profiler']='trace.json and profiler.txt'
    except Exception as exc:
        result['profiler_error']=repr(exc)
    for row in rows:
        row.pop('_one');row.pop('_batch')
    result['rows']=rows
    result['finished_at']=datetime.now(timezone.utc).isoformat()
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RESULT',a.out/'result.json',flush=True)


if __name__=='__main__':
    main()
