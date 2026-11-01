"""Source/CPU-only owned low-token representation feasibility. No kernel changes.

python tools/moe_lowtoken_representation_audit.py --root PATH --output report.json
Traffic is logical requested payload, never a measured HBM transaction count.
"""
import argparse,ast,hashlib,json,math,statistics,struct
from pathlib import Path
PIN='c7854a71ad4639ed1c0ef7ff3013a831765344bf82f27caba15f034a030e03d7'
U=(512,256);UNIT=36864;H=4096;SLOTS=3;SHARED=155520

def bf16(bits):return struct.unpack('<f',struct.pack('<I',bits<<16))[0]
def value(code):return math.copysign((0.,.5,1.,1.5,2.,3.,4.,6.)[code&7],-1. if code&8 else 1.)
def prove_expansion():
 n=0
 for sc in range(8,127):
  scale=2.**((sc>>3)-7)*(1+(sc&7)/8)
  for code in range(16):
   raw=((code&1)<<6)|((code&2)<<6)|((code&4)<<6)|((code&8)<<12)
   a=bf16(raw)*bf16(sc*16+0x7780)
   b=value(code)*scale/128
   assert a==b and math.copysign(1,a)==math.copysign(1,b),(code,sc,a,b)
   # Positive-normal E4M3 times E2M1 needs <=5 significand bits: exact BF16.
   bits=struct.unpack('<I',struct.pack('<f',a))[0]
   assert bits&65535==0
   n+=1
 return n

def reloads(counts,grid=132):
 assert len(counts)==32 and all(0<=c<=256 for c in counts)
 live=[min(8,c-o) for c in counts for o in range(0,c,8)]
 out=[]
 for units in U:
  payload=0;runs=0
  for cta in range(grid):
   lo=len(live)*units*cta//grid;hi=len(live)*units*(cta+1)//grid
   for j,rows in enumerate(live):
    if max(lo,j*units)<min(hi,(j+1)*units):payload+=rows*H*2;runs+=1
  out.append({'job_cta_incidences':runs,'requested_activation_bytes':payload})
 return out

def run(root):
 src=(root/'muse_moe_w4a16/submission.py').read_text(encoding='utf8').replace('\r\n','\n')
 assert hashlib.sha256(src.encode()).hexdigest()==PIN
 tree=ast.parse(src)
 pack=ast.get_source_segment(src,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_repack'))
 assert all(s in pack for s in ('repack_expert(', 'repack_expert_fat(', 'scales_ok(sf)'))
 block=next(n.value.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SKINNY_BLOCK')
 assert 'if (j != jcur || phb != bcur)' in block and '__ldg(reinterpret_cast<const uint4*>(src' in block and '__ldcg(reinterpret_cast<const uint4*>(src' in block
 assert 'uint4 xr[kCPW][2]' in block and 'credit(p, S, jcred, ncred, lane)' in block
 census_path=root/'docs/steering/evidence/2026-11-01-owned-moe-locality/census.json'
 census=json.loads(census_path.read_text(encoding='utf8'))
 banks=[]
 for m in (16,32):
  rows=[r for r in census['records'] if r['m']==m];assert len(rows)==32
  data=[(r,reloads(r['counts'])) for r in rows]
  banks.append({'M':m,'banks':len(data),'mean_weight_requested_bytes':statistics.mean(r['jobs']*sum(U)*UNIT for r,_ in data),'mean_pair_activation_minimum_bytes':statistics.mean(sum(r['counts'])*H*2 for r,_ in data),'mean_activation_requested_A_bytes':statistics.mean(a[0]['requested_activation_bytes'] for _,a in data),'mean_activation_requested_B_bytes':statistics.mean(a[1]['requested_activation_bytes'] for _,a in data),'mean_unique_X_capacity_bytes_upper':m*H*2})
 for r in banks:
  w=r['mean_weight_requested_bytes'];x=r['mean_activation_requested_A_bytes']+r['mean_activation_requested_B_bytes']
  r['activation_to_total_requested_fraction']=x/(w+x)
  r['conditional_eliminate_all_activation_factor_one_load']=(1+x/w)**(1/6)
 receipt=root/'muse_moe_w4a16/profiles/multiply_decode.835253441a214a1c.summary.json'
 prior=json.loads(receipt.read_text(encoding='utf8'))
 return {'source_pin':PIN,'scope':'CPU/source model only; no HBM counters, latency prediction, or production change','expansion_math_cases':prove_expansion(),'immutable_prepare_packs':['skinny packed code+byte scales','fat packed code+byte scales'],'representation':{'current_unit_bytes':UNIT,'BF16_unit_bytes':16*H*2,'scale16_unit_bytes':32768+8192,'current_local32_skinny_bank_bytes':32*sum(U)*UNIT,'current_both_layout_bank_bytes':2*32*sum(U)*UNIT,'BF16_local32_bank_bytes':32*3*H*H*2,'scale16_local32_bank_bytes':32*sum(U)*(32768+8192),'active_ring_slots':SLOTS,'actual_H200_control_shared_bytes':SHARED,'BF16_same_ring_shared_bytes':SHARED-SLOTS*UNIT+SLOTS*16*H*2,'scale16_same_ring_shared_bytes':SHARED+SLOTS*4096,'traffic_ratio_BF16':(16*H*2)/UNIT,'traffic_ratio_scale16':(32768+8192)/UNIT},'retained_census_sha256':hashlib.sha256(census_path.read_bytes()).hexdigest(),'census_scope':census['scope'],'activation_model':banks,'measured_common_control_job':prior['job'],'measured_us':prior['benchmark_us'],'prior_decoder_four_cross':prior['six_load_four_cross'],'prior_duplicate_null':prior['duplicate_null'],'minimum_equal_latency_reduction_for_one_percent_six_load_score':{str(k):1-1.01**(-6/k) for k in (1,2,4,6)},'conditional_proportional_eliminate_all_activation_M16_M32_factor':math.prod(r['conditional_eliminate_all_activation_factor_one_load'] for r in banks),'scenario_warning':'All activation requests removed at zero cost and latency proportional to total requested bytes; not measured, not a timing upper bound.','decision':'NO_GPU_CANDIDATE: expansion admission/traffic cost; no measured exposed activation-read budget'}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(run(a.root),indent=2)+'\n',encoding='utf8')
