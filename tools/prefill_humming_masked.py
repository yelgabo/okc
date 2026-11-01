"""Bounded GROUPED_MASKED prototype for pinned Humming0.1.12; no dispatch.

Ordinary same-stream kernel boundaries preserve generic->async global visibility:
PTX8.7 (CUDA12.8) section8.9.4 API synchronization, final proxy-preserved-base-
causality paragraph. No PDL, concurrent dependent kernels or tensormap mutation.
Existing Humming shared-memory fences remain unchanged.
"""
import ast
from pathlib import Path
import sys
from prefill_humming_indexed import BODY, build as indexed_build

MASKED = r'''
import math as _hmath

@triton.jit
def _hm_gather(X, Rank, Local, GX, R:tl.constexpr, CAP:tl.constexpr,
               K:tl.constexpr=4096, B:tl.constexpr=256):
    route=tl.program_id(0)
    k=tl.program_id(1)*B+tl.arange(0,B)
    rank=tl.load(Rank+route)
    e=tl.load(Local+route)
    live=(e>=0)&(rank>=0)&(rank<CAP)
    x=tl.load(X+(route//8)*K+k,live&(k<K),other=0)
    row=(tl.maximum(e,0)*CAP+tl.maximum(rank,0)).to(tl.int64)
    tl.store(GX+row*K+k,x,live&(k<K))

@triton.jit
def _hm_tail(C, GX, M:tl.constexpr, CAP:tl.constexpr, BM:tl.constexpr,
             K:tl.constexpr=4096, B:tl.constexpr=256):
    e=tl.program_id(0)
    c=tl.load(C+e)
    tl.device_assert(c<=M,"unique top8 contract: expert count exceeds token count")
    i=tl.program_id(1)*B+tl.arange(0,B)
    row=c+i//K
    live=(row<tl.cdiv(c,BM)*BM)&(row<CAP)
    offset=(e*CAP+row).to(tl.int64)*K+i%K
    tl.store(GX+offset,0,live)

@triton.jit
def _hm_silu(Rank, Local, G1, G2, CAP:tl.constexpr, B:tl.constexpr=4096):
    route=tl.program_id(0)
    rank=tl.load(Rank+route)
    e=tl.load(Local+route)
    row=(tl.maximum(e,0)*CAP+tl.maximum(rank,0)).to(tl.int64)
    k=tl.arange(0,B)
    live=(e>=0)&(rank>=0)&(rank<CAP)
    a=tl.load(G1+row*8192+k,live,other=0).to(tl.float32)
    b=tl.load(G1+row*8192+4096+k,live,other=0).to(tl.float32)
    y=(a/(1+tl.exp(-a)))*b
    tl.store(G2+row*4096+k,y,live)

@triton.jit
def _hm_scatter(Rank, Local, G3, Y, CAP:tl.constexpr, B:tl.constexpr=256):
    route=tl.program_id(0)
    k=tl.program_id(1)*B+tl.arange(0,B)
    rank=tl.load(Rank+route)
    e=tl.load(Local+route)
    live=(e>=0)&(rank>=0)&(rank<CAP)
    row=(tl.maximum(e,0)*CAP+tl.maximum(rank,0)).to(tl.int64)
    y=tl.load(G3+row*4096+k,live,other=0)
    tl.store(Y+route*4096+k,y)

def _hi_workspace(x,state):
    hum=state["hum"]
    m=x.shape[0]
    if m not in hum["workspace"]:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Masked Humming scratch must be warmed eagerly")
        cfg=[_hcfg(l.humming_config,shape_m=m,gemm_type="grouped_masked",
                   use_f16_accum=False) for l in hum["layers"]]
        bm=[int(c["block_shape"][0]) for c in cfg]
        alignment=_hmath.lcm(*bm)
        cap=triton.cdiv(m,alignment)*alignment
        r=m*8
        # BF16 grouped X/G1/G2/G3, route Y; I32 rank/local/counts.
        planned=2*32*cap*(4096+8192+4096+4096)+2*r*4096+8*r+128
        free,total=torch.cuda.mem_get_info(x.device)
        print(f"HUMMASK plan M={m} cap={cap} bytes={planned} free={free} total={total} cfg={cfg}",flush=True)
        if planned+2**30>free:
            raise RuntimeError("Insufficient free memory for complete masked candidate")
        dev=x.device
        ws={"rank":torch.empty(r,device=dev,dtype=torch.int32),
            "local":torch.empty(r,device=dev,dtype=torch.int32),
            "counts":torch.empty(32,device=dev,dtype=torch.int32),
            "gx":torch.empty((32*cap,4096),device=dev,dtype=torch.bfloat16),
            "c1":torch.empty((32*cap,8192),device=dev,dtype=torch.bfloat16),
            "c2":torch.empty((32*cap,4096),device=dev,dtype=torch.bfloat16),
            "c3":torch.empty((32*cap,4096),device=dev,dtype=torch.bfloat16),
            "y":torch.empty((r,4096),device=dev,dtype=torch.bfloat16),
            "cap":cap,"bm":bm,"cfg":cfg}
        hum["workspace"][m]=ws
    return hum["workspace"][m]

def _experts(x,topk_weights,topk_ids,state):
    m=x.shape[0]; r=m*8
    x=x.contiguous()
    ids=topk_ids.to(torch.int32).contiguous()
    tw=topk_weights.to(torch.float32).contiguous()
    ws=_hi_workspace(x,state)
    cap=ws["cap"]
    ws["counts"].zero_()
    _hi_rank[(triton.cdiv(r,256),)](ids,ws["counts"],ws["rank"],ws["local"],r,256)
    _hm_gather[(r,1)](x,ws["rank"],ws["local"],ws["gx"],r,cap,B=4096)
    _hm_tail[(32,triton.cdiv(ws["bm"][0]*4096,256))](ws["counts"],ws["gx"],m,cap,ws["bm"][0],debug=True)
    l1,l2=state["hum"]["layers"]
    compute={"gemm_type":"grouped_masked","use_f16_accum":False}
    l1(ws["gx"],outputs=ws["c1"],expert_layout=ws["counts"],
       compute_config=compute,tuning_config=ws["cfg"][0])
    _hm_silu[(r,)](ws["rank"],ws["local"],ws["c1"],ws["c2"],cap)
    _hm_tail[(32,triton.cdiv(ws["bm"][1]*4096,256))](ws["counts"],ws["c2"],m,cap,ws["bm"][1],debug=True)
    l2(ws["c2"],outputs=ws["c3"],expert_layout=ws["counts"],
       compute_config=compute,tuning_config=ws["cfg"][1])
    _hm_scatter[(r,1)](ws["rank"],ws["local"],ws["c3"],ws["y"],cap,B=4096)
    return fused_post_expert_norm_weighted_sum(ws["y"].view(m,8,4096),tw,
        norm_weight=state["post_expert_w"],eps=POST_NORM_EPS,weight_first=True,
        slot_expert_ids=ws["local"].view(m,8))
'''


def build(src):
    # Reuse reviewed exact weight loading/ranking, replace only execution/workspace.
    out=indexed_build(src)
    for name in ('_hi_workspace','_hi_metadata','_experts'):
        node=next(n for n in ast.parse(out).body if isinstance(n,ast.FunctionDef) and n.name==name)
        lines=out.splitlines(keepends=True)
        lines[node.lineno-1:node.end_lineno]=[]
        out=''.join(lines)
    out+=MASKED
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
