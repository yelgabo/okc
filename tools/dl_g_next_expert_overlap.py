"""G-local split front with ordinary fork/join. No dispatch; one fixed short policy."""
import ast
import hashlib
import inspect
from pathlib import Path
BASE="ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab"

def _g_overlap_join(torch, h, hr, side, pooler, router):
    """Unchanged stream logic can be tested with substitute arithmetic callbacks."""
    parent=torch.cuda.current_stream(device=h.device)
    capturing=torch.cuda.is_current_stream_capturing()
    side.wait_stream(parent)
    try:
        with torch.cuda.stream(side):
            if not capturing: hr.record_stream(side)
            logits=router(hr)
        pooled=pooler(h)
    finally:
        # Join even if a callback fails after issuing work; no side work escapes.
        parent.wait_stream(side)
    if not capturing: logits.record_stream(parent)
    return pooled,logits

BODY=r'''
class _GOverlapStream:
    def __init__(self, device):
        self.device=device
        self.stream=torch.cuda.Stream(device=device)
    def __deepcopy__(self,memo):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("G split-front owner copy during capture")
        clone=type(self)(self.device)
        memo[id(self)]=clone
        return clone

_g_overlap_parent_front=_pool_router_fp32

def _g_overlap_choice(T,state):
    b=state["bfp"]
    return b.get("choice",{}).get(("pooler",T)) or bfp__nearby_choice("pooler",T,b) or bfp__static_choice("pooler",T,b)

def _g_overlap_eligible(h,hr,T,state,ppick):
    return (state.get("_g_overlap_active",False) and "_g_overlap_stream" in state
        and T in (1,8) and not _pdl(T) and hr is not None
        and ppick in ("bfp","bfpf") and state.get("bfp") is not None
        and state.get("router_fp32") is not None and state.get("router_fp32_ext") is not None
        and hr.is_contiguous() and h.is_contiguous()
        and _g_overlap_choice(T,state)[0]=="sk2")

def _pool_router_fp32(h,hr,T,state,ppick):
    # The incumbent prepare/tuners run with precisely their original front.
    if not _g_overlap_eligible(h,hr,T,state,ppick):
        return _g_overlap_parent_front(h,hr,T,state,ppick)
    if bfp_SK2_PDL: raise RuntimeError("G split-front unexpected global PDL policy")
    return _g_overlap_join(torch,h,hr,state["_g_overlap_stream"].stream,
        lambda a:_bfp_gemm(state,"pool",a,T,D_EXPERT,HIDDEN),
        lambda a:_router_fp32_gemv(a,state)),-1

_g_overlap_parent_prepare=prepare

def prepare(weights):
    import time as _otime
    started=_otime.monotonic()
    state=_g_overlap_parent_prepare(weights)
    if _pdl(1) or _pdl(8) or GLE_T:
        raise RuntimeError("G split-front requires original non-deferred/noPDL dispatch")
    state["_g_overlap_stream"]=_GOverlapStream(weights["router"].device)
    state["_g_overlap_active"]=True
    _g_overlap_selftest(torch,weights,state,started)
    return state
'''

def build(source):
 if hashlib.sha256(source.encode()).hexdigest()!=BASE:raise ValueError("expected exact incumbent startup-validated #303 source")
 output=source+"\n"+inspect.getsource(_g_overlap_join)+"\n"+SELFTEST+"\n"+BODY
 tree=ast.parse(output);assert all(ast.dump(a)==ast.dump(b) for a,b in zip(ast.parse(source).body,tree.body))
 assert len(output.encode())<1000000
 return output

SELFTEST = r'''
def _g_overlap_selftest(torch,weights,state,started):
    import copy,time
    dev=weights["router"].device
    gen=torch.Generator(device=dev).manual_seed(940771)
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    old_cudnn=torch.backends.cudnn.allow_tf32
    def close(a,b,tol,rowtol):
        if a.shape!=b.shape or not bool(torch.isfinite(a).all()):raise RuntimeError("G split-front nonfinite/shape")
        d=a.float()-b.float();v=b.float()
        ge=float(d.norm()/v.norm().clamp_min(1e-8))
        re=float((d.norm(dim=-1)/v.norm(dim=-1).clamp_min(1e-8)).max())
        if ge>tol or re>rowtol:raise RuntimeError(f"G split-front math {ge} {re}")
    def routed(pool,logit,T,ls):
        ids=torch.empty((T,8),dtype=torch.int32,device=dev);lid=torch.empty_like(ids)
        wt=torch.empty((T,8),dtype=torch.float32,device=dev)
        x=torch.empty((T,4096),dtype=torch.bfloat16,device=dev)
        _sl_tile_route[(T,SL_TILE_CONFIG[T][1]+1)](pool,logit,logit,state["bias"],state["g_preexp"],ids,wt,x,
            ls,lid,SPLIT=True,EARLY=SL_ROUTE_CONFIG[T][2],MODE=SL_ROUTE_CONFIG[T][1],
            PARTS=SL_TILE_CONFIG[T][1],PRELOAD=SL_TILE_CONFIG[T][2],num_warps=SL_TILE_CONFIG[T][0],launch_pdl=False)
        return ids,lid,wt,x
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        # Different owners get distinct actual CUDA streams; concurrent capture on one
        # owner is outside the incumbent's shared MoE/attention-scratch contract too.
        clone=copy.deepcopy(state["_g_overlap_stream"])
        if clone is state["_g_overlap_stream"] or clone.stream.cuda_stream==state["_g_overlap_stream"].stream.cuda_stream:
            raise RuntimeError("G split-front deepcopy stream alias")
        refs={};banks={}
        ls=torch.tensor(96,dtype=torch.int32,device=dev)
        for T in (1,8):
            bank=torch.randn((4,T,8192),dtype=torch.float32,device=dev,generator=gen)
            bank=bank*torch.rsqrt(bank.square().mean(-1,keepdim=True)+1e-5)
            bank=bank*(1+weights["pre_ffn_norm"].float())
            banks[T]=bank
            rr=[]
            for i in range(4):
                pool=(bank[i].bfloat16().float()@weights["pooler"].float().T).bfloat16()
                logit=bank[i]@weights["router"].float().T
                route=routed(pool,logit,T,ls)
                rr.append((pool,logit,route))
            refs[T]=rr
        torch.cuda.synchronize()
        graphs=[]
        policy=[]
        for T in (1,8):
            for owner in range(2):
                if time.monotonic()-started>=280:raise RuntimeError("G split-front prepare budget")
                ownerstate=dict(state)
                ownerstate["bfp"]=dict(state["bfp"])
                ownerstate["bfp"]["dummy"]=state["bfp"]["dummy"].clone()
                ownerstate["_g_overlap_stream"]=copy.deepcopy(state["_g_overlap_stream"])
                parent=torch.cuda.Stream(device=dev)
                torch.cuda.synchronize()
                with torch.cuda.stream(parent):
                    hr=banks[T][0].clone();h=hr.bfloat16()
                    ppick=state.get("mm_pick",{}).get((T,"pr"),state.get("mm_pick",{}).get((_bucket(T),"pr"),False))
                    if owner==0: policy.append((T,repr(ppick),repr(_g_overlap_choice(T,ownerstate))))
                    if not _g_overlap_eligible(h,hr,T,ownerstate,ppick):
                        raise RuntimeError(f"G split-front expected qualification path unavailable T={T} ppick={ppick}")
                    def call():
                        p,l=_pool_router_fp32(h,hr,T,ownerstate,ppick)[0]
                        return p,l,routed(p,l,T,ls)
                    eager=call()
                torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=parent): captured=call()
                graphs.append((T,parent,hr,h,graph,captured,eager,ownerstate))
        coverage=[(T,parent.cuda_stream,ownerstate["_g_overlap_stream"].stream.cuda_stream) for T,parent,hr,h,graph,captured,eager,ownerstate in graphs]
        if len(graphs)!=4 or sorted(T for T,*_ in graphs)!=[1,1,8,8] or len({side for T,parent,side in coverage})!=4:
            raise RuntimeError("G split-front incomplete/aliased owner coverage")
        mutable=set()
        for T,parent,hr,h,graph,captured,eager,ownerstate in graphs:
            for tensor in (hr,h,ownerstate["bfp"]["dummy"],captured[0],captured[1],*captured[2]):
                ptr=tensor.untyped_storage().data_ptr()
                if ptr in mutable: raise RuntimeError("G split-front shared writable storage")
                mutable.add(ptr)
        # Alternate both parent streams/loads; each replay result is explicitly observed.
        checks=0
        for i in (0,3,1,2,0,2,3,1):
            for T,parent,hr,h,graph,captured,eager,ownerstate in graphs:
                with torch.cuda.stream(parent):
                    hr.copy_(banks[T][i]);h.copy_(banks[T][i].bfloat16());graph.replay()
                parent.synchronize()
                for actual,index in ((captured,i),(eager,0)):
                    p,l,route=actual;rp,rl,rr=refs[T][index]
                    close(p,rp,.005,.01);close(l,rl,1e-4,1e-4)
                    ids,lid,wt,x=route;ri,rli,rwt,rx=rr
                    if not torch.equal(ids,ri) or not torch.equal(lid,rli):raise RuntimeError("G split-front natural route mismatch")
                    close(wt,rwt,1e-4,1e-4);close(x,rx,.008,.075)
                checks+=1
        # Concurrent graph replays use independent inputs/outputs; joining both observes all work.
        for T,parent,hr,h,graph,captured,eager,ownerstate in graphs:
            with torch.cuda.stream(parent):graph.replay()
        torch.cuda.synchronize()
        for T,parent,hr,h,graph,captured,eager,ownerstate in graphs:
            p,l,route=captured;rp,rl,rr=refs[T][1]
            close(p,rp,.005,.01);close(l,rl,1e-4,1e-4)
            if not torch.equal(route[0],rr[0]) or not torch.equal(route[1],rr[1]):
                raise RuntimeError("G split-front concurrent routing mismatch")
            close(route[2],rr[2],1e-4,1e-4);close(route[3],rr[3],.008,.075)
        if checks!=32: raise RuntimeError("G split-front incomplete fresh checks")
        if time.monotonic()-started>=280: raise RuntimeError("G split-front prepare budget after checks")
        print("G_SPLIT_FRONT_STARTUP_OK",checks,"verified-replays",policy,coverage,
              "sizes1/8 owners2each private-sk2-debug+routeoutputs math/natural-routes/concurrent-replay",flush=True)
    finally:
        try: torch.cuda.synchronize()
        finally:
            torch.backends.cuda.matmul.allow_tf32=old_tf32
            torch.backends.cudnn.allow_tf32=old_cudnn
'''

if __name__=="__main__":
 import argparse
 a=argparse.ArgumentParser();a.add_argument("input");a.add_argument("output");p=a.parse_args()
 s=build(Path(p.input).read_text(encoding="utf8"));Path(p.output).write_text(s,encoding="utf8",newline="\n")
 print(len(s.encode()),hashlib.sha256(s.encode()).hexdigest())
