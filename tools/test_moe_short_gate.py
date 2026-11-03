"""Check absolute short-cell benefit, duplicate noise and all-load preservation."""
import unittest
from moe_short_gate import short_decision
from expert_top100_moe_harness_contract import BANKS, LOADS, DISTRIBUTIONS, CROSSES


def rows(token=8):
    result=[]
    for seed in BANKS:
        for dist in DISTRIBUTIONS:
            for t in LOADS:
                base=26. if t==1 else 67. if t==8 else 180.
                candidate=base-2.5 if t==token else base
                times=[base,base,candidate,candidate]
                result.append([seed,t,dist,times,[t,1,t],[times[a]/times[b] for a,b in CROSSES],[.0001,.0001]])
    return result


class Gate(unittest.TestCase):
    def test_t8_win_is_priced_without_a_t32_threshold(self):
        result=short_decision(rows())
        self.assertTrue(result['passed']);self.assertEqual(result['qualifying_tokens'],[8])
        self.assertLess(result['groups'][0]['factors'][0],1.01)
        self.assertTrue(short_decision(rows(1))['passed'])

    def test_noise_regressions_and_incomplete_rows_fail(self):
        for fault in ('noise','other_load','affected_loss','one_owner','one_bank'):
            data=rows()
            if fault=='noise':
                for r in data:r[6]=[.01,.01]
            elif fault=='other_load':data[3][5]=[.95]*4
            elif fault=='affected_loss':data[0][5]=[.95]*4
            elif fault=='one_owner':data[1][3][3]=67.;data[1][5][1]=1.;data[1][5][3]=1.
            else:data[-5][3][2:]=[67.,67.];data[-5][5]=[1.]*4
            with self.subTest(fault=fault):self.assertFalse(short_decision(data)['passed'])
        with self.assertRaises(RuntimeError):short_decision(rows()[:-1])


if __name__=='__main__':unittest.main()
