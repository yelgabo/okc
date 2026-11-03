"""Ensure confirmation changes inputs and identity, with the same kernel and gates."""
import ast
import unittest
from moe_short_ready_confirm import build,BANKS,_gate,SCREEN
from moe_short_ready_harness import build as screen
from moe_short_gate import short_decision
from expert_top100_moe_harness_contract import BANKS as ORIGINAL,LOADS,DISTRIBUTIONS,CROSSES
from moe_factor_y_harness import assignment
from cuda_issue_groups import sha


class Confirm(unittest.TestCase):
    def test_only_banks_and_runtime_identity_change(self):
        source,m=build();old,prior=screen()
        restored=assignment(source,'BANKS',ORIGINAL)
        restored=assignment(restored,'H_RUNTIME_ID',prior['runtime_id'])
        self.assertEqual(ast.dump(ast.parse(restored)),ast.dump(ast.parse(old)))
        self.assertEqual(sha(old),SCREEN)
        self.assertEqual(m['compiler_calls'],prior['compiler_calls'])
        self.assertEqual(m['candidate_native_sha256'],prior['candidate_native_sha256'])
        self.assertTrue(set(BANKS).isdisjoint(set(ORIGINAL)|{x+97 for x in ORIGINAL}))

    def test_heldout_gate_keeps_original_module_immutable(self):
        rows=[]
        for seed in BANKS:
            for dist in DISTRIBUTIONS:
                for t in LOADS:
                    times=[26.,26.,23.,23.] if t==1 else [100.]*4
                    rows.append([seed,t,dist,times,[t,1,t],[times[a]/times[b] for a,b in CROSSES],[0.,0.]])
        self.assertTrue(_gate(rows)['passed'])
        with self.assertRaises(RuntimeError):short_decision(rows)
        self.assertEqual(short_decision.__globals__['BANKS'],ORIGINAL)


if __name__=='__main__':unittest.main()
