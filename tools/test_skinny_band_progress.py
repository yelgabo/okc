"""Independent CPU specification; no claim of CUDA memory-order validation."""
import unittest

def mapped(v,units,band,experts):
 j=v//units;lo=j;hi=j+1
 while lo and experts[lo-1]==experts[j]:lo-=1
 while hi<len(experts) and experts[hi]==experts[j]:hi+=1
 r=hi-lo;t=v-lo*units;b,rem=divmod(t,r*band);dj,off=divmod(rem,band)
 return lo+dj,b*band+off

def inverse(j,rb,units,band,experts):
 lo=j;hi=j+1
 while lo and experts[lo-1]==experts[j]:lo-=1
 while hi<len(experts) and experts[hi]==experts[j]:hi+=1
 return lo*units+(rb//band)*(hi-lo)*band+(j-lo)*band+rb%band

class SkinnyBand(unittest.TestCase):
 def test_band32_analytic_last_credit(self):
  def prefix(x,job,jobs,units):
   x=max(0,min(x,jobs*units));cycle,tail=divmod(x,jobs*32)
   return cycle*32+max(0,min(32,tail-job*32))
  for jobs in (1,2,3,4,7):
   for units in (512,256):
    es=[0]*jobs
    for grid in (1,2,7,132):
     totals=[0]*jobs
     for cta in range(grid):
      lo=jobs*units*cta//grid;hi=jobs*units*(cta+1)//grid
      actual=[mapped(v,units,32,es)[0] for v in range(lo,hi)]
      for j in range(jobs):
       count=prefix(hi,j,jobs,units)-prefix(lo,j,jobs,units)
       self.assertEqual(count,actual.count(j))
       if count:
        k=prefix(hi,j,jobs,units)-1
        last=(k//32)*jobs*32+j*32+k%32
        self.assertEqual(last,max(v for v in range(lo,hi) if mapped(v,units,32,es)[0]==j))
        totals[j]+=count
     self.assertEqual(totals,[units]*jobs)
 def test_phase_bijection_inverse(self):
  for es in ([],[0],[0,0],[0,0,0,2,3,3]):
   for units in (512,256):
    for band in (1,4,16,64,256):
     seq=[mapped(v,units,band,es) for v in range(len(es)*units)]
     self.assertEqual(set(seq),{(j,u) for j in range(len(es)) for u in range(units)})
     self.assertEqual(len(seq),len(set(seq)))
     for v,(j,u) in enumerate(seq):self.assertEqual(inverse(j,u,units,band,es),v)
 def test_partition_credit_and_phase_flush(self):
  es=[0,0,0,2,3,3]
  for grid in (1,2,3,7,132):
   for band in (1,16,64,256):
    total=[0]*len(es)
    for cta in range(grid):
     a0=len(es)*512*cta//grid;a1=len(es)*512*(cta+1)//grid
     pending_j=-1;pending_n=0;credit=[]
     for v in range(a0,a1):
      j,rb=mapped(v,512,band,es)
      if j!=pending_j:
       if pending_n:credit.append((pending_j,pending_n))
       pending_j=j;pending_n=0
      pending_n+=1
     # Mandatory before first phase-B partial wait, including an empty B range.
     if pending_n:credit.append((pending_j,pending_n))
     for j,n in credit:total[j]+=n
    self.assertEqual(total,[512]*len(es))
if __name__=='__main__':unittest.main()
