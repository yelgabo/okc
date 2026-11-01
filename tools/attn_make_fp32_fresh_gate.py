"""Prospective TASK-math fresh gate; failed strict studies stay failed.

Generate candidate2 plus independent control0, own chunked FP32 references for
ALL2000 unique inputs on BOTH batch1 shapes, official output/row/cache bounds,
and stale/zero/order1 corruption controls. No reference.py import or dispatch.
"""
from pathlib import Path
import sys
import attn_make_bench_one as base

ROOT = Path(__file__).resolve().parent.parent

BODY = r'''
import types as _fg_types
import triton as _fg_triton
import triton.language as _ft
_fg_control = _fg_types.ModuleType("late_math_control")
exec(compile(_FG_CONTROL_SRC,"late_math_control.py","exec"),_fg_control.__dict__)

def _fg_budget():
    assert _fg_clock.monotonic()-_FG_STARTED < 280, "FP32 fresh gate incomplete at280s; fail closed"

def _fg_references(data,bank,actual_k,actual_v,chunk=32):
    cfg=data["config"]
    count=bank.shape[0]
    seq=int(data["seq_lens"][0])
    start=max(0,seq+1-cfg["window"]) if cfg["window"] else 0
    past=torch.arange(start,seq,device="cuda")
    slots=data["page_table"][0].long()[past]
    ks=float(data["k_scale"]); vs=float(data["v_scale"])
    kp=(data["k_cache"].view(-1,2,64)[slots].float()*ks).permute(1,2,0).contiguous()
    vp=(data["v_cache"].view(-1,2,64)[slots].float()*vs).permute(1,0,2).contiguous()
    qi,ki,vi,gate=bank.split([1024,128,128,1024],-1)
    pos=torch.full((count,1),seq,device="cuda",dtype=torch.long)
    q=_ob_rms(qi.reshape(count,1,16,64))*cfg["qk_scale_factor"]
    k=_ob_rms(ki.reshape(count,1,2,64))
    if cfg["rope_theta"]:
        q=_ob_rope(q,pos); k=_ob_rope(k,pos)
    nk=(k/ks).to(FP8).float()
    nv=(vi.reshape(count,1,2,64).float()/vs).to(FP8).float()
    refs=torch.empty((2,count,1024),device="cuda",dtype=torch.bfloat16)
    # Same FP32 math as _ob_reference, batched over independent query inputs.
    for at in range(0,count,chunk):
        _fg_budget()
        end=min(count,at+chunk); n=end-at
        for fp8 in (0,1):
            qc=q[at:end]
            qc=((qc/cfg["qk_scale_factor"]).to(FP8).float()*cfg["qk_scale_factor"]
                if fp8 else qc.to(torch.bfloat16).float())
            q4=qc[:,0].reshape(n,2,8,64).permute(1,0,2,3)
            q2=q4.reshape(2,n*8,64)
            scores=torch.bmm(q2,kp)*cfg["sm_scale"]
            kn=actual_k[at:end].reshape(n,2,64).permute(1,0,2)*ks
            sn=(q4*kn[:,:,None,:]).sum(-1).reshape(2,n*8,1)*cfg["sm_scale"]
            probs=torch.cat((scores,sn),-1).softmax(-1)
            y=torch.bmm(probs[:,:,:-1],vp)
            vn=(actual_v[at:end].reshape(n,2,64).permute(1,0,2)*vs)[:,:,None,:].expand(-1,-1,8,-1)
            y=y+probs[:,:,-1:]*vn.reshape(2,n*8,64)
            y=y.reshape(2,n,8,64).permute(1,0,2,3).reshape(n,1,16,64)
            y=_ob_rms(y)*torch.sigmoid(gate[at:end].float()).reshape(n,1,16,64)
            refs[fp8,at:end].copy_(y.reshape(n,1024).to(torch.bfloat16))
            del scores,probs,sn,y
    e4=((refs[1].float()-refs[0].float()).square().sum(-1)/
        refs[0].float().square().sum(-1).clamp(min=1e-24)).sqrt()
    # Official TASK/checks formula, never fitted to candidate or null errors.
    bound=(2.5*e4+0.015).clamp(min=0.03,max=0.15)
    return refs,bound,nk.reshape(count,128),nv.reshape(count,128)

@_fg_triton.jit(do_not_specialize=["IDX"])
def _fg_collect(KC,VC,Loc,NK,NV,IDX):
    j=_ft.arange(0,128); loc=_ft.load(Loc)
    _ft.store(NK+IDX*128+j,_ft.load(KC+loc*128+j).to(_ft.float32))
    _ft.store(NV+IDX*128+j,_ft.load(VC+loc*128+j).to(_ft.float32))

@_fg_triton.jit(do_not_specialize=["IDX"])
def _fg_check(Out,Ref,Bound,NK,NV,AK,AV,KC,VC,Loc,S,IDX):
    i=_ft.arange(0,1024)
    a=_ft.load(Out+i).to(_ft.float32)
    b=_ft.load(Ref+IDX*1024+i).to(_ft.float32)
    tol=_ft.load(Bound+IDX)
    sq=_ft.sum((a-b)*(a-b),0)/_ft.maximum(_ft.sum(b*b,0),1e-24)
    ar=_ft.reshape(a,(16,64)); br=_ft.reshape(b,(16,64))
    den=_ft.sum(br*br,1)
    floor=_ft.maximum(0.1*_ft.sum(_ft.sqrt(den),0)/16,1e-12)
    row=_ft.max(_ft.sum((ar-br)*(ar-br),1)/_ft.maximum(den,floor*floor),0)
    j=_ft.arange(0,128)
    loc=_ft.load(Loc)
    kg=_ft.load(KC+loc*128+j).to(_ft.float32)
    vg=_ft.load(VC+loc*128+j).to(_ft.float32)
    kr=_ft.load(NK+IDX*128+j); vr=_ft.load(NV+IDX*128+j)
    ak=_ft.load(AK+IDX*128+j); av=_ft.load(AV+IDX*128+j)
    same=_ft.sum(((kg!=ak)|(vg!=av)).to(_ft.int32),0)==0
    ek=_ft.sum((kg-kr)*(kg-kr),0)/_ft.maximum(_ft.sum(kr*kr,0),1e-24)
    ev=_ft.sum((vg-vr)*(vg-vr),0)/_ft.maximum(_ft.sum(vr*vr,0),1e-24)
    finite=_ft.sum(((_ft.abs(a)==float("inf"))|(a!=a)).to(_ft.int32),0)==0
    finite=finite & (_ft.sum(((_ft.abs(kg)==float("inf"))|(_ft.abs(vg)==float("inf"))|
                              (kg!=kg)|(vg!=vg)).to(_ft.int32),0)==0)
    bad=(~finite)|(~same)|(sq>tol*tol)|(row>25*tol*tol)|(ek>0.08*0.08)|(ev>0.08*0.08)
    _ft.atomic_add(S,bad.to(_ft.float32))
    _ft.atomic_max(S+1,sq); _ft.atomic_max(S+2,row)
    _ft.atomic_max(S+3,ek); _ft.atomic_max(S+4,ev)

def _fg_run(replays=2000):
    # Force actual FP32 reference GEMMs: no TF32 rounding in the independent math.
    torch.backends.cuda.matmul.allow_tf32=False
    for tag,mod in (("candidate",_ONE),("control",_fg_control._ONE)):
        print(f"FRESHMATH RESOURCES {tag} fields=L,mode,regs,localbytes,staticsmem,dynamicsmem,threads,blocks_per_sm values={mod.attn_one_math_resources()}",flush=True)
    all_stats=[]
    for kind in ("sliding","full"):
        _fg_budget()
        data=_ob_inputs(1,80000,"tp8",kind,1,"fixed",
                        "soft" if kind=="sliding" else "coherent",1,1.,seed=311104)
        gen=torch.Generator(device="cuda").manual_seed(311105)
        bank=torch.randn((replays,*data["qkv"].shape),device="cuda",dtype=torch.bfloat16,generator=gen)
        fns={"control":_fg_control.kernel_fn,"candidate":kernel_fn}
        for fn in fns.values():
            for _ in range(3): fn(data)
        graphs={}; outs={}
        for tag,fn in fns.items():
            graphs[tag]=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[tag]): outs[tag]=fn(data)
            graphs[tag].replay()
        torch.cuda.synchronize()
        # Official reference consumes actual stored newKV. Collect the control's
        # rows for EVERY input; the gate separately requires candidate rows equal
        # these, in addition to official8% cache bounds against own quant math.
        actual_k=torch.empty((replays,128),device="cuda",dtype=torch.float32)
        actual_v=torch.empty_like(actual_k)
        _fg_collect[(1,)](data["k_cache"],data["v_cache"],data["cache_loc"],actual_k,actual_v,0)
        for idx in range(replays):
            data["qkv"].copy_(bank[idx]); graphs["control"].replay()
            _fg_collect[(1,)](data["k_cache"],data["v_cache"],data["cache_loc"],actual_k,actual_v,idx)
        torch.cuda.synchronize(); _fg_budget()
        before=_time.time()
        refs,bound,nk,nv=_fg_references(data,bank,actual_k,actual_v)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(refs.float()).all()) and bool(torch.isfinite(bound).all())
        print(f"FRESHMATH references {kind} inputs={replays} seconds={_time.time()-before:.3f} "
              f"chunk=32 score_cat_prob_max_bytes={3*32*16*(80001 if kind=='full' else 2048)*4} "
              f"reference_bytes={refs.numel()*refs.element_size()} bound_minmax={bound.min().item():.5g}/{bound.max().item():.5g}",flush=True)
        # Cross-check BOTH independent batched references against canonical eager math.
        for idx in (0,replays//2,replays-1):
            data["qkv"].copy_(bank[idx]); graphs["candidate"].replay()
            torch.cuda.synchronize()
            for fp8 in (0,1):
                canonical=_ob_reference(data,bool(fp8),(actual_k[idx].reshape(1,1,2,64),actual_v[idx].reshape(1,1,2,64))).to(torch.bfloat16)
                rel=_ob_rel(refs[fp8,idx],canonical)
                print(f"FRESHMATH canonical {kind} input={idx} fp8={fp8} rel={rel:.3e} "
                      f"{_ob_diffstat(refs[fp8,idx],canonical)}",flush=True)
                assert rel<1e-3, "batched FP32 reference disagrees with canonical math"
        stats={tag:torch.zeros(5,device="cuda") for tag in fns}
        def compare(tag,idx,acc,out=None):
            _fg_check[(1,)](outs[tag] if out is None else out,refs[0],bound,nk,nv,actual_k,actual_v,
                           data["k_cache"],data["v_cache"],data["cache_loc"],acc,idx)
        # IDX is a runtime nonspecialized argument: one checker for all2000inputs.
        compare("candidate",0,stats["candidate"])
        for s in stats.values(): s.zero_()
        _fg_budget()
        for idx in range(replays):
            data["qkv"].copy_(bank[idx])
            for tag in fns:
                graphs[tag].replay(); compare(tag,idx,stats[tag])
        torch.cuda.synchronize()
        _fg_budget()
        for tag,s in stats.items():
            bad,err,row,ek,ev=s.tolist()
            print(f"FRESHMATH {kind} {tag} inputs={replays} bad={bad:g} "
                  f"max_rel={err**.5:.7g} max_row={row**.5:.7g} max_k={ek**.5:.7g} max_v={ev**.5:.7g}",flush=True)
            all_stats.append((kind,tag,bad))
        # Explicit negative controls: stale complete output, zero, order1 one-head corruption.
        data["qkv"].copy_(bank[0]); graphs["candidate"].replay()
        stale=outs["candidate"].clone()
        data["qkv"].copy_(bank[1]); graphs["candidate"].replay()
        negatives={"stale":stale,"zero":torch.zeros_like(stale),"head":outs["candidate"].clone()}
        negatives["head"].view(-1)[:64].add_(10)
        for name,wrong in negatives.items():
            s=torch.zeros(5,device="cuda"); compare("candidate",1,s,wrong)
            torch.cuda.synchronize(); bad=s[0].item()
            print(f"FRESHMATH NEG {kind} {name} rejected={bad==1}",flush=True)
            assert bad==1, f"insensitive official-math comparator: {kind}/{name}"
        clean,msg=_ob_ws_check(data["qkv"].device)
        print(f"FRESHMATH workspace {kind} clean={clean} {msg}",flush=True)
        assert clean,msg
        del data,bank,graphs,outs,refs,bound,nk,nv,actual_k,actual_v,stats,negatives,stale
        torch.cuda.empty_cache()
    print(f"FRESHMATH FINAL {all_stats} prepare_elapsed={_fg_clock.monotonic()-_FG_STARTED:.2f}",flush=True)
    assert all(bad==0 for _,_,bad in all_stats), "prospective official-math fresh gate failed"

_ob_selftest()
_fg_run()
'''


def main():
    source=(ROOT/"muse_attention_decode/submission.py").read_text()
    assert source.count("#define ONE_LATE_TID 0")==1
    # Read-only host attribute export, avoiding buffered C++ stdout on import failure.
    decl="std::vector<int64_t> attn_one_prepare();"
    assert source.count(decl)==1
    source=source.replace(decl,decl+"\nstd::vector<int64_t> attn_one_math_resources();")
    marker="std::vector<int64_t> attn_one_prepare() {"
    assert source.count(marker)==1
    diag='''std::vector<int64_t> attn_one_math_resources() {
    std::vector<int64_t> out;
#define MRES(L,M) { Instance<L,M>::prepare(); cudaFuncAttributes a = {}; \\
    cudaFuncGetAttributes(&a, attn_one_kernel<L,M>); \\
    out.insert(out.end(), {L,M,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,Instance<L,M>::BYTES,Instance<L,M>::NT,Instance<L,M>::per_sm}); }
    ONE_INSTANCES(MRES)
#undef MRES
    return out;
}
'''
    source=source.replace(marker,diag+"\n"+marker)
    exports='functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop"],'
    assert source.count(exports)==1
    source=source.replace(exports,'functions=["attn_one", "attn_one_prepare", "attn_one_probe_coop", "attn_one_math_resources"],')
    candidate=source.replace("#define ONE_LATE_TID 0","#define ONE_LATE_TID 2")
    header,rest=candidate.split("\n",1)
    candidate=header+"\nimport time as _fg_clock\n_FG_STARTED = _fg_clock.monotonic()\n"+rest
    control=source.replace('name="okc_attn_one_v1"','name="okc_attn_fp32_gate_control"')
    runner=base.RUNNER.format(variants=["_one"],coop_probe=True)
    head,sep,_=runner.rpartition("\nimport os as _os\n"); assert sep
    Path(sys.argv[1]).write_text(candidate+head+"\n_FG_CONTROL_SRC = "+repr(control)+"\n"+BODY)


if __name__=="__main__": main()
