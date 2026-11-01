"""Single full-slot wait-policy check, unchanged public control, no hints/sweeps."""
import ast
from pathlib import Path
import sys
from moe_make_bench_ab import AB

HOOK=r'''
// Bit128 changes ONLY full-slot polling. Bit256 is an unused neutral duplicate.
__device__ __forceinline__ bool full_slot_ready(uint64_t* bar, unsigned parity) {
    if constexpr ((kMode & 128) != 0) {
        unsigned ok;
        asm volatile("{\n.reg .pred p;\n"
                     "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2;\n"
                     "selp.u32 %0, 1, 0, p;\n}\n"
                     : "=r"(ok) : "r"(smem_u32(bar)), "r"(parity) : "memory");
        return ok != 0;
    } else {
        return mbar_try(bar, parity);
    }
}
'''

RESOURCE=r'''
std::vector<int64_t> moe_wait_resources() {
    std::vector<int64_t> out;
@@QUERY@@
    return out;
}
'''

QUERY=r'''
    {
        cudaFuncAttributes a;
        int blocks=0;
        C10_CUDA_CHECK(cudaFuncSetAttribute(sk_8_3_@MODE@::moe_fused_kernel<1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(sk_8_3_@MODE@::Smem)));
        C10_CUDA_CHECK(cudaFuncGetAttributes(&a,sk_8_3_@MODE@::moe_fused_kernel<1>));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,
            sk_8_3_@MODE@::moe_fused_kernel<1>,sk_8_3_@MODE@::kNTL,sizeof(sk_8_3_@MODE@::Smem)));
        out.insert(out.end(),{@MODE@,a.numRegs,(int64_t)a.localSizeBytes,
            (int64_t)a.sharedSizeBytes,(int64_t)sizeof(sk_8_3_@MODE@::Smem),blocks,a.maxThreadsPerBlock});
    }
'''

BENCH=r'''
import math
import random as _tw_random
_tw_original_prepare=prepare
def prepare(weights):
    began=time.monotonic()
    state=_tw_original_prepare(weights)
    def guard():
        if time.monotonic()-began>280:
            raise RuntimeError("Testwait comparison incomplete: 280s guard")
    guard()
    dev=weights["w13"].device
    resources=_ext().moe_wait_resources()
    rows=[resources[i:i+7] for i in range(0,len(resources),7)]
    assert [r[0] for r in rows]==[0,256,128,384]
    assert all(r[1:]==rows[0][1:] for r in rows),f"testwait resource mismatch {rows}"
    variants={tag:SKINNY_VARIANTS.index((8,3,mode)) for tag,mode in
              (("base",0),("null",256),("test",128),("test2",384))}
    states={tag:dict(state,scratch=_scratch(dev)) for tag in variants}
    flush=torch.empty(256<<20,device=dev,dtype=torch.uint8)
    logs=[]; checks=0; postchecks=0; records=[]; stamps=[]
    def run(tag,data): return _run(data,states[tag],variants[tag])
    def check(value,reference):
        rel=_rel(value,reference)
        scale=reference.float().norm(dim=-1)
        scale=scale.clamp(min=.1*scale.mean()).clamp(min=1e-12)
        row=float(((value.float()-reference.float()).norm(dim=-1)/scale).max())
        assert bool(torch.isfinite(value).all()) and rel<TEST_TOL and row<.075,(rel,row)
        return rel,row
@@ROUTE@@
    for m in (1,8,16,32):
        for routing in ("layer","std"):
            for seed in (321,322):
                guard()
                data=inputs(m,routing,seed)
                ref=_ref_forward(data,weights)
                graphs={}; output={}; events={}; worst=[0.,0.]
                for tag in variants:
                    err=check(run(tag,data),ref); checks+=1
                    worst=[max(a,b) for a,b in zip(worst,err)]
                    graphs[tag]=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graphs[tag]): output[tag]=run(tag,data)
                    events[tag]=[]
                for round in range(20):
                    order=list(variants); _tw_random.Random(seed+round).shuffle(order)
                    for tag in order:
                        flush.zero_()
                        a=torch.cuda.Event(enable_timing=True); z=torch.cuda.Event(enable_timing=True)
                        a.record(); graphs[tag].replay(); z.record(); events[tag].append((a,z))
                torch.cuda.synchronize()
                for tag in variants:
                    err=check(output[tag],ref); postchecks+=1
                    worst=[max(a,b) for a,b in zip(worst,err)]
                    assert int(states[tag]["scratch"]["ctr"][8])==0,"testwait spin timeout"
                times={tag:[a.elapsed_time(z)*1000 for a,z in pairs] for tag,pairs in events.items()}
                median=lambda v:sorted(v)[len(v)//2]
                raw={tag:median(t) for tag,t in times.items()}
                paired={tag:median([a-b for a,b in zip(t,times["base"])]) for tag,t in times.items()}
                records.append((m,routing,seed,raw,paired))
                logs.append(f"TWCASE M={m} route={routing} seed={seed} raw_us="+
                    repr({k:round(v,3) for k,v in raw.items()})+" paired_us="+
                    repr({k:round(v,3) for k,v in paired.items()})+
                    f" worst_rel,row={worst[0]:.6g},{worst[1]:.6g}")
                del graphs,output,ref
    guard()
    # Standard routing is the standalone score. Unchanged128/256 contribute ratio1.
    # Aggregate per-shape seed means first, then four logratios / six loads.
    scores={}
    for tag in variants:
        logsums=[]
        for m in (1,8,16,32):
            cell=[r[3] for r in records if r[0]==m and r[1]=="std"]
            b=sum(r["base"] for r in cell)/len(cell)
            c=sum(r[tag] for r in cell)/len(cell)
            logsums.append(math.log(b/c))
        scores[tag]=math.exp(sum(logsums)/6)
    nullnoise=abs(math.log(scores["null"]))
    gain=all(math.log(scores[t])>=math.log(1.01) and math.log(scores[t])>2*nullnoise for t in ("test","test2"))
    regress=any(r[4][t]>max(.25,2*abs(r[4]["null"])) for r in records for t in ("test","test2"))
    # Fully counted corrected wait windows, untimed twins AFTER all four-arm timing.
    stamp_complete=True
    for m in (1,8,16,32):
        for routing in ("layer","std"):
            if time.monotonic()-began>265:
                stamp_complete=False; break
            data=inputs(m,routing,321)
            row=[]
            for mode in (2,130):
                v=SKINNY_VARIANTS.index((8,3,mode))
                flush.zero_(); _run(data,states["base"],v); torch.cuda.synchronize()
                g=torch.cuda.get_device_properties(dev).multi_processor_count
                d=states["base"]["scratch"]["inter"].view(torch.int64)[:g*16].reshape(g,16).double().cpu()
                ghz=(d[:,13]-d[:,12])/(d[:,7]-d[:,0])
                assert bool(torch.isfinite(ghz).all()) and bool((ghz>0).all())
                a=float((d[:,8]/ghz/1000).mean()); b=float((d[:,9]/ghz/1000).mean())
                row.append((mode,round(a,3),round(b,3)))
                assert int(states["base"]["scratch"]["ctr"][8])==0,"stamp spin timeout"
            stamps.append(f"TWSTAMP M={m} route={routing} mode,A_wait_us,B_wait_us={row}")
        if not stamp_complete: break
    summary=["TWRESULT core_complete16cases resources_equivalentTrue",
        f"TWRESOURCE mode,regs,local,static,dynamic,blocksSM,maxThreads={rows}",
        f"TWCHECK eager={checks}/64 graph={postchecks}/64 spin0True thresholdsrel{TEST_TOL}/row.075",
        *logs,*stamps,f"TWSTAMP_COMPLETE {stamp_complete} (noattributionifFalse)",
        f"TWSCORE sixload_contribution={scores} null_abslog={nullnoise:.8g} gain1pct_2xnull={gain} no_regress={not regress} KEEP={gain and not regress}"]
    text="\n".join(summary)
    assert len(text.encode())<10000,"testwait compact summary exceeds10KB"
    print(text,flush=True)
    return state
'''


def build(src):
    blocknode=next(n for n in ast.parse(src).body if isinstance(n,ast.Assign)
                   and any(isinstance(t,ast.Name) and t.id=='_SKINNY_BLOCK' for t in n.targets))
    block=ast.literal_eval(blocknode.value)
    anchor='__device__ __forceinline__ void hand_wait('
    assert block.count(anchor)==1
    block=block.replace(anchor,HOOK+'\n'+anchor)
    target='mbar_try(&S.full[slot], (q / kNS) & 1)'
    assert block.count(target)==2
    block=block.replace(target,'full_slot_ready(&S.full[slot], (q / kNS) & 1)')
    lines=src.splitlines(keepends=True)
    lines[blocknode.lineno-1:blocknode.end_lineno]=['_SKINNY_BLOCK = r"""'+block+'"""\n']
    out=''.join(lines)
    old='SKINNY_VARIANTS = ((8, 4, 0), (16, 4, 0), (8, 4, 1), (8, 3, 0))'
    assert out.count(old)==1
    out=out.replace(old,old[:-1]+', (8, 3, 256), (8, 3, 128), (8, 3, 384), (8, 3, 2), (8, 3, 130))')
    out=out.replace('int64_t moe_smem_bytes();','int64_t moe_smem_bytes();\nstd::vector<int64_t> moe_wait_resources();',1)
    resource=RESOURCE.replace('@@QUERY@@','\n'.join(QUERY.replace('@MODE@',str(m)) for m in (0,256,128,384)))
    assert out.count('int64_t moe_smem_bytes() {')==1
    out=out.replace('int64_t moe_smem_bytes() {',resource+'\nint64_t moe_smem_bytes() {',1)
    out=out.replace('functions=["moe_run", "moe_smem_bytes"],','functions=["moe_run", "moe_smem_bytes", "moe_wait_resources"],',1)
    route=AB[AB.index("    # The layer's judge routing"):AB.index('    pubfx = []')]
    a=route.index('    best = None'); z=route.index('    def inputs(',a)
    route=route[:a]+'    rt=router(8)  # fixed previously calibrated router\n'+route[z:]
    out+='\n'+BENCH.replace('@@ROUTE@@',route)
    ast.parse(out)
    return out


if __name__=='__main__':
    Path(sys.argv[2]).write_text(build(Path(sys.argv[1]).read_text()))
