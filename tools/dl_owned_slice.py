"""Source-only VERIFY32 sliding slice of the BF16-matched owned cb71 control.
No judge dispatch. No runtime source reads in emitted artifact. Frozen diagnostic policy.
"""
import ast
import hashlib
import json
from pathlib import Path

PIN = "851c46951e1940b6874d35314d8ff7cc00a966a4718094881c764135ef13392a"

CUSTOM = r'''
def _ext():
    global _slice_main
    if _slice_main is None:
        _slice_main = load_inline(name="okc_owned_slice_main", cpp_sources=_CPP_SRC,
            cuda_sources=[_CUDA_SRC, _BND_SRC], functions=["route_fwd", "attn_boundary"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    return _slice_main

def _moe1_ext():
    global _slice_moe
    if _slice_moe is None:
        _slice_moe = _moe1_load()
    return _slice_moe

def _one_ext():
    global _slice_one
    if _slice_one is None:
        _slice_one = _one_load()
    return _slice_one

def _moe1_source():
    # Preserve the original ABI variant ordinal 10; reject every other route.
    return _MOE1_HOST_SRC.replace("//@MOE1_DISPATCH@",
        "        case 10: launch_main<51>(p, sms, stream); break;")

def _slice_proj(x, layer, state):
    entry = state["dlplt"].get((layer, x.shape[0]))
    w = state["dense"][layer]
    if entry is not None and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 256 == 0:
        return entry(x)
    return _gemm(x, w)

def _slice_lt(state):
    lt = _dlplt_build()
    state["dlplt"] = {}
    for layer, layout, cfg in (
        ("qkv_gate", 0, [66,124,1,0,0,0,35,0,2]),
        ("unpooler", 1, [66,20,1,0,0,4,35,0,3])):
        w = state["dense"][layer]; n,k = w.shape
        ws = torch.zeros(32 << 20,dtype=torch.uint8,device=w.device)
        wa = w.t().contiguous() if layout % 2 else w
        plan = lt.lt_plan_new(256,n,k,layout,ws.numel())
        idx = lt.lt_add_config(plan,list(cfg)) if plan >= 0 else -1
        assert idx >= 0, ("slice Lt pin rejected", layer)
        def run(x, plan=plan, idx=idx, wa=wa, ws=ws, n=n):
            out = torch.empty((256,n),dtype=torch.bfloat16,device=x.device)
            rc = lt.lt_run(plan,idx,wa,x,out,ws)
            if rc: raise RuntimeError(f"slice Lt status {rc}")
            return out
        state["dlplt"][layer,256] = run

def prepare(weights):
    device = weights["qkv_gate"].device
    pre = _norm(MuseSpark1xRMSNorm,weights["pre_ffn_norm"],HIDDEN,RMS_EPS)
    nxt = _norm(MuseSpark1xRMSNorm,weights["next_pre_attn_norm"],HIDDEN,RMS_EPS)
    post = _norm(MuseSpark1xRMSNorm,None,HIDDEN,POST_NORM_EPS,is_post_norm=True,
                 post_norm_gain_center_type="one",residual_id=1)
    postffn = _norm(MuseSpark1xRMSNormWithInputScale,weights["post_ffn_norm"],HIDDEN,
        POST_NORM_EPS,is_post_norm=True,post_norm_gain_center_type="one",residual_id=2)
    poste = _norm(MuseSpark1xRMSNormWithInputScale,weights["post_expert_norm"],D_EXPERT,
        POST_NORM_EPS,is_post_norm=True,post_norm_gain_center_type="one",residual_id=2)
    pren = _norm(MuseSpark1xRMSNorm,weights["pre_expert_norm"],D_EXPERT,RMS_EPS)
    ar = _residual(weights["attn_residual_gate"])
    fr = _residual(weights["ffn_residual_gate"])
    router = weights["router"]; hi=router.to(torch.bfloat16); lo=(router-hi.float()).to(torch.bfloat16)
    state={"dense":{k:weights[k].contiguous() for k in ("qkv_gate","o","pooler","unpooler")},
      "inv_freq":1.0/(ROPE_THETA**(torch.arange(0,64,2,dtype=torch.float,device=device)/64)),
      "pre_expert_w":pren._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
      "route_bias":weights["router_bias"].float().contiguous(),
      "pool_router":torch.cat((weights["pooler"],hi,lo),0).contiguous(),
      "abnd":{"alpha":ar._alpha.float().reshape(-1).contiguous(),
              "beta":ar._beta.float().reshape(-1).contiguous(),
              "w_next":pre._get_effective_weight().detach().to(torch.bfloat16).contiguous(),
              "eps_post":float(post.eps),"eps_next":float(pre.eps)},
      "ffn_boundary":FusedPostNormResidualPreNorm(postffn,fr,nxt),
      "_modules":(pre,nxt,post,postffn,poste,pren,ar,fr)}
    _ext(); _one_setup(state)
    assert _one_plan({"batch":32,"q_len":8,"page_size":1,"window":2048},86008,
                     state["one"]["resident"]) == (4,1,3), "slice attention selection mismatch"
    w=_moe1_prepare(weights)
    if w is None: raise RuntimeError("slice: unsupported expert scales; no fallback")
    state["moe1"]={"w":w,"nw":poste._get_effective_weight().detach().to(torch.bfloat16).float().contiguous(),
                   "scratch":_moe1_scratch(device)}
    _moe1_ext(); _slice_lt(state)
    torch.cuda.synchronize()
    print("OWNED_SLICE source-only policy verify32/sliding ONE(4,1,3), MOE51, Lt L0/L1; no autotuning",flush=True)
    return state

def kernel_fn(data,state):
    c=data["config"]
    assert all(c[k]==v for k,v in {"batch":32,"q_len":8,"tokens":256,"page_size":1,
        "window":2048,"hidden":8192,"num_q_heads":16,"num_kv_heads":2,"head_dim":64,
        "n_experts":256,"top_k":8,"local_experts":32}.items()), "unsupported slice config"
    assert data["page_table"].shape[1]==86008, "unsupported slice context capacity"
    qkv=_slice_proj(data["attn_input"],"qkv_gate",state)
    o=_attn_one(qkv,data,state)
    attn_output=_slice_proj(o,"o",state)
    ffn,stream,_=_abnd(data["stream"],attn_output,False,state,"c1x512")
    z=torch.mm(ffn,state["pool_router"].t(),out_dtype=torch.float32)
    x,ids,wts=_route_fwd(z[:,D_EXPERT:],N_EXPERTS,z[:,:D_EXPERT],state)
    expert=_moe1_run(x,wts,ids,data["local_start"],state,10,True)
    out=_slice_proj(expert,"unpooler",state)
    nxt,stream,_=state["ffn_boundary"](stream,out,branch_dtype=torch.bfloat16)
    return nxt,stream,ids
'''

def build(source, custom_source=None):
    assert hashlib.sha256(source.encode()).hexdigest()==PIN,"owned input pin mismatch"
    custom_source = CUSTOM if custom_source is None else custom_source
    from dl_owned_attention_boundary import patch_owned_attention
    source=patch_owned_attention(source)
    tree=ast.parse(source); custom=ast.parse(custom_source)
    overrides={n.name for n in custom.body if isinstance(n,ast.FunctionDef)}
    nodes={}
    for n in tree.body:
        if isinstance(n,(ast.FunctionDef,ast.ClassDef)): keys=[n.name]
        elif isinstance(n,(ast.Assign,ast.AnnAssign)):
            keys=[x.id for x in (n.targets if isinstance(n,ast.Assign) else [n.target]) if isinstance(x,ast.Name)]
        else: continue
        for k in keys:
            if k not in overrides: nodes[k]=n
    needed={x.id for x in ast.walk(custom) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load)}
    seen=set(); selected=set()
    while needed-seen:
        key=(needed-seen).pop();seen.add(key)
        if key not in nodes: continue
        n=nodes[key];selected.add(id(n))
        needed.update(x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Load))
    kept=[n for n in tree.body if id(n) in selected or isinstance(n,(ast.Import,ast.ImportFrom))]
    # Retain selected functions and device literals byte-for-byte, rather than unparse C++.
    body="\n\n".join(ast.get_source_segment(source,n) for n in kept)
    output="# okc: problem=muse_decode_layer\n# CHECK-ONLY single-load diagnostic; source/CPU qualification only.\n"+body+"\n_slice_main=_slice_moe=_slice_one=None\n"+custom_source
    ast.parse(output)
    literals={}
    for n in kept:
        if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) and len(n.value.value)>1000:
            literals[n.targets[0].id]={"bytes":len(n.value.value.encode()),"sha256":hashlib.sha256(n.value.value.encode()).hexdigest()}
    return output,{"input_sha256":PIN,"output_sha256":hashlib.sha256(output.encode()).hexdigest(),
      "bytes":len(output.encode()),"device_literals":literals,
      "functions":[n.name for n in kept if isinstance(n,ast.FunctionDef)],
      "qualification":"source-only; not H200 qualified; no performance claim"}

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("source");a.add_argument("output");args=a.parse_args()
    result,manifest=build(Path(args.source).read_text(encoding="utf8"))
    p=Path(args.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(result,encoding="utf8",newline="\n")
    p.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf8")
    print(json.dumps(manifest,indent=2))
