"""CPU ownership/byte model for exact non-SPLIT T1/T8 tiny path; no CUDA."""
from collections import Counter
UNIT_BYTES=36864
ROWS_A,ROWS_B=512,256

def model(counts,grid=132):
 assert len(counts)==32 and all(0<=c<=8 for c in counts)
 experts=[e for e,c in enumerate(counts) if c];requests=Counter()
 for phase,rows in ((0,ROWS_A),(1,ROWS_B)):
  total=len(experts)*rows
  for cta in range(grid):
   # Actual device muldiv_g uses exact quotient corrected around float estimate.
   a,b=total*cta//grid,total*(cta+1)//grid
   for v in range(a,b):requests[phase,experts[v//rows],v%rows]+=1
 assert all(n==1 for n in requests.values())
 return dict(active_experts=len(experts),local_routes=sum(counts),
  logical_weight_bytes=len(requests)*UNIT_BYTES,unique_tiles=len(requests),
  duplicate_tiles=sum(v-1 for v in requests.values()))

if __name__=="__main__":
 import json
 # Only fixtures, not measured route vectors. Match retained natural aggregate counts.
 fixtures={"m1_seed901_aggregate":[1,1]+[0]*30,"m1_seed902_aggregate":[1,1,1]+[0]*29,
  "m8_seed901_aggregate":[8,2,2]+[1]*8+[0]*21,
  "m8_seed902_aggregate":[8,2]+[1]*5+[0]*25,
  "allremote":[0]*32,"hot8":[8]+[0]*31}
 print(json.dumps({k:model(v) for k,v in fixtures.items()},indent=2))
