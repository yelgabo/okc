"""Pure CPU analysis: observed readiness under original wait order, never HBM timing."""
import math
import statistics
ORDINALS=(0,7,15,31)
WORDS=475

def expected_jobs(ids):
 counts=[0]*32
 for row in ids:
  if len(row)!=8 or len(set(row))!=8:raise ValueError("valid unique top8 required")
  for e in row:
   if 0<=e<32:counts[e]+=1
 offsets=[];n=0
 for c in counts:offsets.append(n);n+=c
 fat=[];skinny=[]
 for e,c in enumerate(counts):
  if c>8:fat.extend((e,offsets[e]+q,min(64,c-q)) for q in range(0,c,64))
  elif c:skinny.append((e,offsets[e],c))
 return fat+skinny,counts

def stage_order(pb,g0,g1):
 ue=((g1-1)>>4)<<4
 sp=ue if pb==0 and (g1&15) and ue>g0 else g0
 return list(range(sp,g1))+list(range(g0,sp)),sp

def analyze_trace(words,ids,grid,mode):
 if len(words)!=WORDS or mode not in (1,2,3) or grid<3:raise ValueError("bad stage trace ABI")
 jobs,counts=expected_jobs(ids);nj=int(words[207])
 if nj!=len(jobs) or [tuple(words[208+3*j:211+3*j]) for j in range(nj)]!=jobs:
  raise ValueError("trace job metadata disagrees with actual replay input")
 result=[];samples=[];ctas=(0,grid//2,grid-1)
 for c,cta in enumerate(ctas):
  entry=words[472+c];finish=words[204+c]
  if not 0<entry<=finish:raise ValueError("missing CTA entry/consumer finish")
  phaseA_count=None
  for pb in (0,1):
   g0,g1,q0,q1,t0,t1=words[168+(c*2+pb)*6:174+(c*2+pb)*6]
   if not 0<=g0<=g1<=(nj<<(9-pb)) or q1-q0!=g1-g0 or not entry<=t0<t1<=finish:
    raise ValueError("invalid phase range/sequence/span")
   if pb==0:phaseA_count=g1-g0
   if q0!=(phaseA_count if pb else 0):raise ValueError("phase sequence origin drift")
   order,sp=stage_order(pb,g0,g1)
   observed=[]
   for k,ordinal in enumerate(ORDINALS):
    row=words[((c*2+pb)*4+k)*7:((c*2+pb)*4+k+1)*7]
    if mode==3:
     if any(row):raise ValueError("endpoint control unexpectedly has inner stage marks")
     continue
    if ordinal>=len(order):
     if any(row):raise ValueError("unconsumed stage was sampled")
     continue
    a,b,d=row[:3] if mode==1 else row[3:6]
    if not t0<=a<=b<=d<=t1 or not a:raise ValueError("incomplete/nonmonotone sampled stage")
    gs=order[ordinal];u=gs>>4;j=u>>(5-pb);stage=gs&15
    e,off,count=jobs[j]
    start,end=(sp,g1) if ordinal<g1-sp else (g0,sp)
    segment=(max(start,u*16)-u*16,min(end,(u+1)*16)-u*16)
    v={"cta":cta,"phase":pb,"ordinal":ordinal,"global_stage":gs,"unit":u,"stage":stage,
       "job":j,"expert":e,"count":count,"segment":segment,"seq":q0+ordinal,
       "slot":(q0+ordinal)%3,"parity":((q0+ordinal)//3)&1}
    if mode==1:
     if row[6] not in range(4):raise ValueError("bad read-only entry readiness flags")
     v.update(weight_wait_observed_us=(b-a)/1000,x_wait_after_weight_us=(d-b)/1000,
              weight_ready_at_entry=bool(row[6]&1),x_ready_at_entry=bool(row[6]&2),dispatch_to_entry_us=(a-t0)/1000)
    else:v.update(retire_observed_us=(b-a)/1000,release_observed_us=(d-b)/1000)
    samples.append(v);observed.append(v)
   expected=sum(i<len(order) for i in ORDINALS) if mode!=3 else 0
   if len(observed)!=expected:raise ValueError("selected ordinal coverage mismatch")
   result.append({"cta":cta,"phase":pb,"stages":g1-g0,"span_us":(t1-t0)/1000,
      "kernel_to_phase_us":(t0-entry)/1000,"consumer_finish_us":(finish-entry)/1000,"samples":expected})
 return {"mode":mode,"counts":counts,"jobs":jobs,"phases":result,"samples":samples}

def distortion(medians,phase_medians):
 if len(medians)!=6 or len(phase_medians)!=6 or any(len(x)!=6 for x in phase_medians):raise ValueError("six arms and six phase/CTA spans required")
 base=math.sqrt(medians[0]*medians[1]);ep=math.sqrt(medians[2]*medians[3])
 spread=max(abs(medians[0]/medians[1]-1),abs(medians[2]/medians[3]-1))
 endpoint=abs(ep/base-1)
 rows=[]
 for arm in (4,5):
  phase=max(abs(phase_medians[arm][k]/math.sqrt(phase_medians[2][k]*phase_medians[3][k])-1) for k in range(6))
  whole=abs(medians[arm]/base-1)
  phase_spread=max(abs(phase_medians[2][k]/phase_medians[3][k]-1) for k in range(6))
  rows.append({"arm":arm,"whole_displacement":whole,"phase_displacement":phase,
               "phase_duplicate_spread":phase_spread,"pass":spread<=.015 and endpoint<=.015 and whole<=.03 and phase<=.03 and phase_spread<=.015})
 return {"control_us":base,"duplicate_spread":spread,"endpoint_displacement":endpoint,"detail_gates":rows}

def compact(info):
 phases=[]
 for pb in (0,1):
  s=[r for r in info["samples"] if r["phase"]==pb];p=[r for r in info["phases"] if r["phase"]==pb]
  def med(k,rows):return round(statistics.median(r[k] for r in rows),4) if rows else None
  row={"phase":pb,"span_median_us":med("span_us",p),"phase_span_max_us":max(r["span_us"] for r in p),"selected_samples":len(s)}
  if info["mode"]==1:
   row.update(startup_w_us=med("weight_wait_observed_us",[r for r in s if r["ordinal"]==0]),
              startup_x_us=med("x_wait_after_weight_us",[r for r in s if r["ordinal"]==0]),
              steady_w_us=med("weight_wait_observed_us",[r for r in s if r["ordinal"]>0]),
              steady_x_us=med("x_wait_after_weight_us",[r for r in s if r["ordinal"]>0]),
              ready_entry=[sum(r[k] for r in s) for k in ("weight_ready_at_entry","x_ready_at_entry")])
  elif info["mode"]==2:row.update(retire_us=med("retire_observed_us",s),release_us=med("release_observed_us",s))
  phases.append(row)
 return phases
