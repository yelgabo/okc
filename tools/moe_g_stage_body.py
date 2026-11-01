"""CHECK ONLY: no file reads, fixed natural inputs; normal entry remains uninstrumented."""
_GSP_PREPARE=prepare
_GSP_SEEDS=(76111,76117,76123)

def _gsp_errors(out,ref):
 a,b=out.float(),ref.float()
 if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise RuntimeError("GSP_NONFINITE")
 delta=a-b;ge=float(delta.norm()/b.norm().clamp_min(1e-12));rn=b.norm(dim=-1);floor=(.1*rn.mean()).clamp_min(1e-12)
 re=float((delta.norm(dim=-1)/torch.maximum(rn,floor)).max())
 if not math.isfinite(ge) or not math.isfinite(re) or ge>.008 or re>.075:raise RuntimeError(f"GSP_MATH {ge} {re}")
 return ge,re

def prepare(weights):
 import time,statistics,math,json,zlib,base64,struct
 from concurrent.futures import ThreadPoolExecutor
 began=time.monotonic();old=torch.backends.cuda.matmul.allow_tf32
 def budget():
  if time.monotonic()-began>=275:raise RuntimeError("GSP_INNER_OR_FINAL_BUDGET")
 try:
  state=_GSP_PREPARE(weights)
  assert weights["w13"].shape[0]==weights["w2"].shape[0]==256
  assert state["lo"]==0 and state["cnt"]==32 and not PDL and not state.get("ps")
  dev=weights["w13"].device;grid=state["grid"];assert 3<=grid<=136
  assert EMAX==3072 and WGS_FIXW==4096 and WGS_XROWS>=3776
  flags=tuple(_main_defines(state["sfx"]))+tuple(state["sfxm"])
  assert {"-DOKC_WGS=1","-DOKC_SC6W=1","-DOKC_PRMTSC=1"}<=set(flags)
  # Base GPU preparation is over. Only extension compilation/host PDL setters
  # overlap here; no GPU tuning, numeric checks or graphs execute concurrently.
  control=_ext(flags)
  with ThreadPoolExecutor(max_workers=2) as pool:
   futures=[pool.submit(_ext,flags+(f"-DG_STAGE_PROBE={mode}",)) for mode in (3,1,2)]
   endpoint,ready,retire=[f.result() for f in futures]
  if time.monotonic()-began>=270:raise RuntimeError("GSP_PREPARE_COMPILE_BUDGET")
  modules=[control,control,endpoint,endpoint,ready,retire]
  resources=[m.sc6_resources() for m in (control,endpoint,ready,retire)]
  for r in resources:
   assert len(r)==12 and min(r[4],r[10])>=1
   for k in (0,6):
    assert r[k]<=resources[0][k] and r[k+2:k+6]==resources[0][k+2:k+6],("GSP_RESIDENCY_OR_GEOMETRY",resources)
  torch.backends.cuda.matmul.allow_tf32=False
  refns={"__name__":"_gsp_private_sglang","torch":torch};exec(_SC_BASELINE,refns)
  rw={k:(v.clone() if k=="post_norm_w" else v[:32].clone()) for k,v in weights.items()}
  reference_state=refns["prepare"](rw);del rw
  flush=torch.empty(64*1024*1024,dtype=torch.int32,device=dev)
  records=[];representative=None;worst=[0.,0.];tallies={"normal":0,"fresh":0,"skew":0,"remote":0,"restored":0}
  for m in (128,256):
   for seed in _GSP_SEEDS:
    if time.monotonic()-began>=270:raise RuntimeError("GSP_CASE_BUDGET_INCOMPLETE")
    data=_gen_input(m,seed+m,dev)
    original={k:data[k].clone() for k in ("hidden_states","topk_ids","topk_weights")}
    x,ids,tw=[data[k] for k in ("hidden_states","topk_ids","topk_weights")]
    scratches=[_make_scratch(dev) for _ in range(6)]
    traces=[None,None]+[torch.zeros(475,dtype=torch.int64,device=dev) for _ in range(4)]
    outputs=[torch.empty_like(x) for _ in range(6)]
    privateptrs=[t.untyped_storage().data_ptr() for s in scratches for t in s]+[t.untyped_storage().data_ptr() for t in outputs]+[t.untyped_storage().data_ptr() for t in traces if t is not None]
    assert len(set(privateptrs))==len(privateptrs),"GSP_MUTABLE_ALIAS"
    graphs=[]
    for i,module in enumerate(modules):
     ws=scratches[i] if i<2 else scratches[i]+[torch.empty(0,dtype=torch.int64,device=dev),traces[i]]
     args=(state["w"],ws,x,ids,tw,outputs[i],0,32,grid)
     module.okc_run(*args)
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):module.okc_run(*args)
     g.replay();graphs.append(g)
    torch.cuda.synchronize()
    def verify(label):
     ref=refns["kernel_fn"](data,reference_state)
     for out in outputs:
      a,b=_gsp_errors(out,ref);worst[0]=max(worst[0],a);worst[1]=max(worst[1],b);tallies[label]+=1
    verify("normal")
    times=[[] for _ in range(6)]
    for lap in range(25):
     budget()
     for offset in range(6):
      i=(lap+offset)%6
      if traces[i] is not None:traces[i].zero_()
      flush.zero_();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
      a.record();graphs[i].replay();b.record();b.synchronize()
      times[i].append(a.elapsed_time(b)*1000)
    med=[statistics.median(v) for v in times];verify("normal")
    fresh=_gen_input(m,5*seed+m,dev)
    for k in original:data[k].copy_(fresh[k])
    for g,out in zip(graphs,outputs):out.fill_(float("nan"));g.replay()
    torch.cuda.synchronize();verify("fresh")
    # Valid8distinct global IDs on every token; high local skew is correctness-only.
    ids.copy_(torch.arange(8,device=dev,dtype=torch.int32).expand(m,8))
    for g,out in zip(graphs,outputs):out.fill_(float("nan"));g.replay()
    torch.cuda.synchronize();verify("skew")
    ids.copy_(torch.arange(32,40,device=dev,dtype=torch.int32).expand(m,8))
    for g,out in zip(graphs,outputs):out.fill_(17);g.replay()
    torch.cuda.synchronize()
    for out in outputs:
     assert bool(torch.isfinite(out).all()) and int(torch.count_nonzero(out))==0,"GSP_EMPTY_NONZERO"
     tallies["remote"]+=1
    for k in original:data[k].copy_(original[k])
    # Three independent cold snapshots per instrumented arm. CPU transfer/parse
    # are outside timing and must finish before any results are interpreted.
    snapshots=[None,None,[],[],[],[]]
    routes=ids.cpu().tolist()
    for i in range(2,6):
     for repeat in range(3):
      budget()
      traces[i].zero_();flush.zero_();graphs[i].replay();torch.cuda.synchronize()
      _gsp_errors(outputs[i],refns["kernel_fn"](data,reference_state))
      raw=traces[i].cpu().tolist();info=analyze_trace(raw,routes,grid,3 if i<4 else 1 if i==4 else 2)
      snapshots[i].append((info,raw))
    for g in graphs:g.replay()
    torch.cuda.synchronize();verify("restored")
    phase_meds=[[0.]*6,[0.]*6]+[[statistics.median(info["phases"][k]["span_us"] for info,raw in snapshots[i]) for k in range(6)] for i in range(2,6)]
    gate=distortion(med,phase_meds)
    # Representative is chosen by phase span median, never by measured speedup.
    chosen=[None,None]+[sorted(snapshots[i],key=lambda p:sum(r["span_us"] for r in p[0]["phases"]))[1] for i in range(2,6)]
    details=[compact(chosen[i][0]) for i in (4,5)]
    records.append({"m":m,"seed":seed,"med_us":[round(v,4) for v in med],"gate":gate,"details":details,"counts":chosen[4][0]["counts"]})
    if m==256 and seed==_GSP_SEEDS[1]:
     masks=[sum(1<<e for e in row if 0<=e<32) for row in routes]
     payload=struct.pack("<4sHH",b"GSP1",grid,m)
     payload+=struct.pack("<1900Q",*(chosen[2][1]+chosen[3][1]+chosen[4][1]+chosen[5][1]))
     payload+=struct.pack(f"<{m}I",*masks)
     representative=base64.b64encode(zlib.compress(payload,9)).decode()
    del graphs,outputs,scratches,traces,chosen,snapshots
  assert len(records)==6 and representative is not None
  assert tallies=={"normal":72,"fresh":36,"skew":36,"remote":36,"restored":36},tallies
  packed=[]
  for r in records:
   gate=r["gate"]
   ready=[[q[k] for k in ("span_median_us","phase_span_max_us","selected_samples","startup_w_us","startup_x_us","steady_w_us","steady_x_us","ready_entry")] for q in r["details"][0]]
   retire=[[q[k] for k in ("span_median_us","phase_span_max_us","selected_samples","retire_us","release_us")] for q in r["details"][1]]
   gates=[[q[k] for k in ("arm","whole_displacement","phase_displacement","phase_duplicate_spread","pass")] for q in gate["detail_gates"]]
   packed.append([r["m"],r["seed"],r["med_us"],[gate["control_us"],gate["duplicate_spread"],gate["endpoint_displacement"]],gates,ready,retire,r["counts"]])
  report={"schema":"GSP1","backend":"exactG#303MAIN,32B4_UNMEASURED","resources":resources,"tallies":tallies,"worst":worst,"tolerance":[.008,.075],"cases":packed}
  summary=json.dumps(report,separators=(",",":"));assert len(summary)+len(representative)<11000,"GSP_LOG_BUDGET"
  budget()
  print("GSP_SUMMARY",summary,flush=True);print("GSP_CAPSULE",representative,flush=True)
  print("GSP_DONE",len(records),"prepare_s",time.monotonic()-began,"observed-ready/notphysicalTMA/noSpeedupClaim",flush=True)
  return state
 finally:
  torch.backends.cuda.matmul.allow_tf32=old
