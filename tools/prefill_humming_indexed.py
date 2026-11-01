"""Replace only baseline expert preparation/execution with Humming0.1.12 indexed BF16 NVFP4.

Own Triton metadata kernels; no copied entrant code. Humming v0.1.12 at636ba856.
Usage: python tools/prefill_humming_indexed.py BASELINE.py OUT.py
No judge dispatch from this generator. Root must review/version-check first.
"""
import ast
from pathlib import Path
import sys

BODY = r'''
import triton
import triton.language as tl
from humming import dtypes as _hd
from humming.layer import HummingLayer as _HLayer
from humming.tune import get_heuristics_config as _hcfg

@triton.jit
def _hi_rank(IDs, C, Rank, Local, R: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    e = tl.load(IDs+i, i<R, other=-1)
    valid = (i<R) & (e>=0) & (e<32)
    safe = tl.minimum(tl.maximum(e,0),31)
    rank = tl.atomic_add(C+safe, 1, mask=valid, sem="relaxed")
    tl.store(Rank+i, tl.where(valid,rank,-1), i<R)
    tl.store(Local+i, tl.where(valid,e,-1), i<R)

@triton.jit
def _hi_prefix(C, Off, NP, BM: tl.constexpr):
    e = tl.arange(0,32)
    c = tl.load(C+e)
    padded = tl.cdiv(c,BM)*BM
    end = tl.cumsum(padded)
    tl.store(Off+e,end-padded)
    tl.store(Off+32,tl.sum(padded,0))
    tl.store(NP,tl.sum(padded,0))

@triton.jit
def _hi_sorted(Rank, Local, Off, Sorted, R: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0)*B+tl.arange(0,B)
    r = tl.load(Rank+i,i<R,other=-1)
    e = tl.load(Local+i,i<R,other=-1)
    off = tl.load(Off+tl.maximum(e,0), (i<R)&(e>=0),other=0)
    tl.store(Sorted+off+tl.maximum(r,0),i,(i<R)&(r>=0))

@triton.jit
def _hi_blocks(C, Off, Experts, BM: tl.constexpr, BC: tl.constexpr):
    e = tl.program_id(0)
    i = tl.arange(0,BC)
    n = tl.cdiv(tl.load(C+e),BM)
    start = tl.load(Off+e)//BM
    tl.store(Experts+start+i,e,i<n)

def _hi_prepare(weights):
    layers=[]
    for name,n in (("w13",8192),("w2",4096)):
        layer=_HLayer(shape_n=n,shape_k=4096,num_experts=32,
            torch_dtype=torch.bfloat16,
            weight_config={"b_dtype":_hd.float4e2m1,"bs_dtype":_hd.float8e4m3,
                "weight_scale_group_size":16,"weight_scale_2_type":"tensor"},
            input_config={"dtype":_hd.bfloat16}).to(weights[name].device)
        layer.load_from_tensors({"weight":weights[name].contiguous().view(torch.int32),
            "weight_scale":weights[name+"_sf"].contiguous(),
            "weight_scale_2":weights[name+"_gs"].float().reshape(32,1)})
        layer.transform()
        layers.append(layer)
    return {"layers":layers,"workspace":{}}

def _hi_workspace(x,state):
    # Metadata-only cache: retain each shape's scratch for captured graph lifetimes.
    hum=state["hum"]
    m=x.shape[0]
    if m not in hum["workspace"]:
        r=m*8
        dev=x.device
        cfg=[_hcfg(layer.humming_config,shape_m=r,gemm_type="indexed",
                   use_f16_accum=False) for layer in hum["layers"]]
        ws={"rank":torch.empty((r,),dtype=torch.int32,device=dev),
            "local":torch.empty((r,),dtype=torch.int32,device=dev),
            "counts":torch.zeros((32,),dtype=torch.int32,device=dev),
            "c1":torch.zeros((r,8192),dtype=torch.bfloat16,device=dev),
            "c2":torch.zeros((r,4096),dtype=torch.bfloat16,device=dev),
            "c3":torch.zeros((r,4096),dtype=torch.bfloat16,device=dev),
            "cfg":cfg,"meta":[]}
        for config in cfg:
            bm=int(config["block_shape"][0])
            cap=(triton.cdiv(r,bm)+32)*bm
            ws["meta"].append({"bm":bm,"off":torch.empty((33,),dtype=torch.int32,device=dev),
                "np":torch.empty((1,),dtype=torch.int32,device=dev),
                "sorted":torch.empty((cap,),dtype=torch.int32,device=dev),
                "experts":torch.empty((cap//bm,),dtype=torch.int32,device=dev)})
        hum["workspace"][m]=ws
        print(f"HUMINDEX workspace M={m} cfg={cfg}",flush=True)
    return hum["workspace"][m]

def _hi_metadata(ws,m):
    r=m*8
    for meta in ws["meta"]:
        bm=meta["bm"]
        meta["sorted"].fill_(r)
        meta["experts"].zero_()
        _hi_prefix[(1,)](ws["counts"],meta["off"],meta["np"],bm)
        _hi_sorted[(triton.cdiv(r,256),)](ws["rank"],ws["local"],meta["off"],meta["sorted"],r,256)
        _hi_blocks[(32,)](ws["counts"],meta["off"],meta["experts"],bm,
            triton.next_power_of_2(triton.cdiv(r,bm)))

def _experts(x,topk_weights,topk_ids,state):
    # Caller has already shifted IDs by device local_start modulo256.
    m=x.shape[0]
    ws=_hi_workspace(x,state)
    ws["counts"].zero_()
    _hi_rank[(triton.cdiv(m*8,256),)](topk_ids,ws["counts"],ws["rank"],ws["local"],m*8,256)
    _hi_metadata(ws,m)
    # Humming writes only selected rows. Zero every other route so SiLU and
    # post-expert normalization never touch uninitialized nonlocal values.
    ws["c1"].zero_()
    ws["c2"].zero_()
    ws["c3"].zero_()
    l1,l2=state["hum"]["layers"]
    compute={"gemm_type":"indexed","use_f16_accum":False}
    a,b=ws["meta"]
    l1(x,outputs=ws["c1"],sorted_ids=a["sorted"],expert_ids=a["experts"],
        num_tokens_padded=a["np"],top_k=8,compute_config=compute,tuning_config=ws["cfg"][0])
    silu_and_mul(ws["c1"],ws["c2"])
    l2(ws["c2"],outputs=ws["c3"],sorted_ids=b["sorted"],expert_ids=b["experts"],
        num_tokens_padded=b["np"],top_k=1,compute_config=compute,tuning_config=ws["cfg"][1])
    return fused_post_expert_norm_weighted_sum(ws["c3"].view(m,8,4096),topk_weights,
        norm_weight=state["post_expert_w"],eps=POST_NORM_EPS,weight_first=True,
        slot_expert_ids=ws["local"].view(m,8))
'''


def build(src):
    tree=ast.parse(src)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_experts')
    lines=src.splitlines(keepends=True)
    lines[fn.lineno-1:fn.end_lineno]=[BODY+'\n']
    src=''.join(lines)
    anchor='"bank": _marlin_bank(weights),'
    assert src.count(anchor)==1
    src=src.replace(anchor,'"hum": _hi_prepare(weights),')
    if not src.startswith('# okc:'):
        src='# okc: problem=muse_prefill_layer\n'+src
    ast.parse(src)
    return src


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
