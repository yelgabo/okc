"""CPU algebra/shape proof for independent-query batched reference factorization."""
import math
import random
import unittest


def softmax(x):
    m=max(x); y=[math.exp(v-m) for v in x]; z=sum(y)
    return [v/z for v in y]


class ReferenceLayout(unittest.TestCase):
    def test_past_plus_per_input_new_matches_canonical(self):
        rng=random.Random(311108)
        for count,heads,groups,dims,past in ((3,2,8,64,0),(3,2,8,64,1),(5,2,8,64,17)):
            vec=lambda:[rng.uniform(-1,1) for _ in range(dims)]
            q=[[[vec() for _ in range(groups)] for _ in range(heads)] for _ in range(count)]
            kp=[[vec() for _ in range(past)] for _ in range(heads)]
            vp=[[vec() for _ in range(past)] for _ in range(heads)]
            kn=[[vec() for _ in range(heads)] for _ in range(count)]
            vn=[[vec() for _ in range(heads)] for _ in range(count)]
            direct={}
            for i in range(count):
                for h in range(heads):
                    keys=kp[h]+[kn[i][h]]; vals=vp[h]+[vn[i][h]]
                    for g in range(groups):
                        p=softmax([sum(a*b for a,b in zip(q[i][h][g],k))*.125 for k in keys])
                        direct[i,h,g]=[sum(p[t]*vals[t][d] for t in range(past+1)) for d in range(dims)]
            for h in range(heads):
                flatq=[q[i][h][g] for i in range(count) for g in range(groups)]
                for flat,query in enumerate(flatq):
                    i,g=divmod(flat,groups)
                    old=[sum(a*b for a,b in zip(query,k))*.125 for k in kp[h]]
                    new=sum(a*b for a,b in zip(query,kn[i][h]))*.125
                    p=softmax(old+[new])
                    factored=[sum(p[t]*vp[h][t][d] for t in range(past))+p[-1]*vn[i][h][d] for d in range(dims)]
                    for a,b in zip(factored,direct[i,h,g]): self.assertAlmostEqual(a,b,places=13)

    def test_capacity_and_work_model(self):
        inputs,chunk,hq,t,d=2000,32,16,80001,64
        flops=8*inputs*hq*t*d  # QK+PV, multiply+add, BF16-query/E4M3-query refs.
        peak_three_score_tensors=3*chunk*hq*t*4
        retained_reference_bytes=2*inputs*hq*d*2
        self.assertEqual(flops,1310736384000)
        self.assertEqual(peak_three_score_tensors,491526144)
        self.assertEqual(retained_reference_bytes,8192000)
        self.assertLess(peak_three_score_tensors,512*1024**2)


if __name__=="__main__": unittest.main()
