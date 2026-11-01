"""Independent bounded canonical skinny-credit and main queue capacity proof. CPU only."""
import json,math

def prove():
 cases=0
 for jobs in (0,1,2,31,32,60,95,543):
  total=jobs*512
  for grid in (1,2,7,16,132,136):
   credited=[0]*jobs
   for c in range(grid):
    a0=total*c//grid;a1=total*(c+1)//grid
    # Simulate actual helper's canonical job-change/phase-boundary flush.
    current=-1;credit=0
    for v in range(a0,a1):
     j=v>>9
     if j!=current:
      if credit:credited[current]+=credit
      current=j;credit=0
     credit+=1
    if credit:credited[current]+=credit # BEFORE first B partial wait, or end.
   assert credited==[512]*jobs,(jobs,grid)
   # Both exact A and B partitions cover their domains with no gaps/overlap.
   for unit in (512,256):
    end=0
    for c in range(grid):
     lo=jobs*unit*c//grid;hi=jobs*unit*(c+1)//grid
     assert lo==end and lo<=hi;end=hi
    assert end==jobs*unit
   cases+=1
 # Model the independent conservative host bound; it never underestimates
 # sum ceil(count/64) given sum count<=M*8 and active experts<=nloc.
 for m in range(1,513):
  for nloc in range(1,33):
   maxjobs=(m*8+63*nloc)//64
   bound=m*8//64+nloc
   assert bound>=maxjobs
   assert 132*63>=16*bound
 return dict(canonical_partition_cases=cases,skinny_credit512_every_job=True,main_host_bound_cases=512*32,H200132_queue_capacity=True,max_conservative_Bunits=16*(512*8//64+32),GPU=False)
if __name__=='__main__':print(json.dumps(prove(),indent=2))
