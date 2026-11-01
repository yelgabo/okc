"""CPU specification only: expert-local job/unit transpose, not device code."""
import unittest

def order(experts,units):
 out=[];start=0
 while start<len(experts):
  end=start+1
  while end<len(experts) and experts[end]==experts[start]:end+=1
  for unit in range(units):
   for job in range(start,end):out.append((job,unit))
  start=end
 return out

class ExpertRunOrder(unittest.TestCase):
 def test_bijection_inverse_and_tail_cost(self):
  for experts,counts in (([],[]),([1],[7]),([1,1],[64,57]),([0,2,2,2,3],[8,64,64,1,9])):
   for units in (16,32):
    pairs=order(experts,units);inverse={pair:i for i,pair in enumerate(pairs)}
    self.assertEqual(set(pairs),{(j,u) for j in range(len(experts)) for u in range(units)})
    self.assertEqual(len(pairs),len(inverse))
    stages=[(j,u,k) for j,u in pairs for k in range(16)]
    self.assertEqual(len(stages),len(set(stages)))
    for i,(j,u,k) in enumerate(stages):self.assertEqual(16*inverse[j,u]+k,i)
    # A cost prefix must use the mapped physical job, including its smaller tail.
    costs=[(counts[j]+7)//8 for j,u in pairs]
    self.assertEqual(sum(costs),sum((c+7)//8 for c in counts)*units)
 def test_reuse_distance(self):
  pairs=order([4,4],32)
  self.assertEqual(pairs[:4],[(0,0),(1,0),(0,1),(1,1)])
  self.assertEqual([order([4],32).index((0,u)) for u in range(32)],list(range(32)))
if __name__=='__main__':unittest.main()
