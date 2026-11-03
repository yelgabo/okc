# okc: problem=muse_moe_w4a16
# Candidate: exact indexed Humming BF16/NVFP4 stages.
"""SGLang's NVFP4 expert path for Muse Spark Open on H200 (Marlin MoE runner), at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f: MuseSpark1xFp4Experts._forward_marlin, starting after the pre-expert norm.

prepare() repacks the experts into Marlin's layout with prepare_moe_nvfp4_layer_for_marlin,
as SGLang does at load. kernel_fn runs moe_align_block_size, two moe_wna16_marlin_gemm
(W4A16, FP32 reduce), silu_and_mul and the Triton post-expert-norm weighted sum.

Differs from SGLang: the expert all-reduce is not run."""

from typing import TypedDict

import types

import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def prepare(weights):
    device = weights["w13"].device
    # One repacked bank: this GPU's 32 experts under EP8.
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = data["topk_ids"].to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = data["topk_weights"].to(torch.float32).contiguous()

    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / bank["count"] / block_size_m < 0.9:
            break
    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, block_size_m)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, block_size_m, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


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
        assert layer.weight_scale_2.numel()==32 and layer.weight_scale_2.dtype==torch.float32
        assert weights[name+"_gs"].numel()==32 and weights[name+"_gs"].dtype==torch.float32
        layer.load_from_tensors({"weight":weights[name].contiguous().view(torch.int32),
            "weight_scale":weights[name+"_sf"].contiguous(),
            "weight_scale_2":weights[name+"_gs"].contiguous().reshape_as(layer.weight_scale_2)})
        layer.transform()
        layers.append(layer)
    return {"layers":layers,"workspace":{}}

def _hi_workspace(x,state):
    # Metadata-only cache: retain each shape's scratch for captured graph lifetimes.
    hum=state["hum"]
    m=x.shape[0]
    if m not in hum["workspace"]:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Humming scratch must be warmed eagerly before capture")
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
    x=x.contiguous()
    topk_ids=topk_ids.to(torch.int32).contiguous()
    topk_weights=topk_weights.to(torch.float32).contiguous()
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

def prepare(weights):
    assert weights["w13"].shape[0] % 32 == 0
    assert weights["w13"].shape[1:] == (8192,2048)
    banks = {}
    for start in range(0,weights["w13"].shape[0],32):
        local = {k:weights[k][start:start+32].contiguous()
                 for k in ("w13","w13_sf","w13_gs","w2","w2_sf","w2_gs")}
        banks[start] = _hi_prepare(local)
    return {"banks":banks,"post_expert_w":(weights["post_norm_w"].float()+1).to(torch.bfloat16)}

def kernel_fn(data,state):
    cfg = data["config"]
    assert cfg["local_count"] == 32
    start = cfg["local_start"]
    sub = {"hum":state["banks"][start],"post_expert_w":state["post_expert_w"]}
    local = data["topk_ids"].to(torch.int32).contiguous() - start
    return _experts(data["hidden_states"].contiguous(),
                    data["topk_weights"].float().contiguous(),local,sub)
