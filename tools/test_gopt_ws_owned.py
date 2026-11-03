"""Check the emitted schedule, byte mapping, release isolation and frozen gates."""
import ast
import json
import math
from pathlib import Path
import unittest

from gopt_ws_owned import build, contract_source, ROOT, RELEASE_PIN, sha
from expert_top100_pregamma import capture


class SourceAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source, cls.manifest, cls.call = build()

    def test_disabled_builds_and_exact_affected_dispatch(self):
        baseline, _ = capture((ROOT/'muse_moe_w4a16/cand/current_control_release.py').read_text())
        current, _ = capture(self.source)
        self.assertEqual(current,baseline)
        self.assertEqual(self.manifest['release_sha256'],RELEASE_PIN)
        self.assertEqual([r['tokens'] for r in self.manifest['candidate_routes'] if r['key']=='owned'],[8,16])
        flags = self.call['extra_cuda_cflags']
        self.assertIn('-DOKC_NW=4',flags)
        self.assertIn('-DOKC_SPLIT=1',flags)
        self.assertNotIn('--use_fast_math',flags)
        self.assertEqual(self.call['functions'],['okc_run','okc_set_pdl'])

    def test_byte_permutation_and_every_consumer_fragment(self):
        unit,codes = 36864,32768
        old = list(range(4*unit))
        new = [-1]*(4*unit)
        for stage in range(4):
            for warp in range(4):
                for half in range(2):
                    k=stage*2+half
                    for size,src,dst in ((4096,k*4096,warp*8192+half*4096),
                                         (512,codes+k*512,codes+warp*1024+half*512)):
                        new[stage*unit+dst:stage*unit+dst+size]=old[warp*unit+src:warp*unit+src+size]
        self.assertEqual(sorted(new),old)
        checks=0
        for stage in range(4):
            for warp in range(4):
                for half in range(2):
                    for lane in range(32):
                        dst=stage*unit+codes+warp*1024+half*512+lane*16
                        src=warp*unit+codes+(stage*2+half)*512+lane*16
                        self.assertEqual(new[dst:dst+16],old[src:src+16])
                        for c in range(8):
                            dst=stage*unit+warp*8192+half*4096+c*512+lane*16
                            src=warp*unit+(stage*2+half)*4096+c*512+lane*16
                            self.assertEqual(new[dst:dst+16],old[src:src+16])
                            for k in range(4):
                                self.assertEqual(stage*1024+half*512+c*64+(lane&3)*16+k*4,
                                                 (stage*2+half)*512+c*64+(lane&3)*16+k*4)
                            checks+=1
        self.assertEqual(checks,8192)

    def test_partition_credits_and_replay_ring_parities(self):
        for jobs in range(89):
            credits=[0]*jobs
            for blocks in (128,64):
                seen=[0]*(jobs*blocks)
                for cta in range(132):
                    begin,end=jobs*blocks*cta//132,jobs*blocks*(cta+1)//132
                    for v in range(begin,end):
                        seen[v]+=1
                        if blocks==128:credits[v//blocks]+=4
                    if blocks==64:
                        total=(jobs*128*(cta+1)//132-jobs*128*cta//132+end-begin)*4
                        phases=[0]*3
                        for q in range(total):
                            slot=q%3
                            self.assertEqual(phases[slot],q//3)
                            phases[slot]+=1
                self.assertTrue(all(x==1 for x in seen))
            self.assertEqual(credits,[512]*jobs)

    def test_complete_score_and_unchanged_guards(self):
        ns={};exec(contract_source(),ns)
        def rows(gain):
            return [[seed,t,d,[100.,100.,100./(gain if t in (8,16) else 1)]*1+[100./(gain if t in (8,16) else 1)],
                     [t,1,t],[gain if t in (8,16) else 1]*4,[.0001,.0001]]
                    for seed in ns['BANKS'] for t in ns['LOADS'] for d in ns['DISTRIBUTIONS']]
        self.assertFalse(ns['decision'](rows(1.02))['component_pass'])
        self.assertTrue(ns['decision'](rows(1.05))['component_pass'])
        r=rows(1.05);r[0][5]=[.98]*4
        self.assertFalse(ns['decision'](r)['component_pass'])
        self.assertTrue(ns['decision'](r)['unresolved_drift'])
        r=rows(1.05);r[2][5]=[.98]*4
        self.assertFalse(ns['decision'](r)['component_pass'])

    def test_native_ordering_and_lifetimes_are_present(self):
        native=self.call['cuda_sources']
        self.assertIn('fence.proxy.async.shared::cta;',native)
        self.assertNotIn('float red[',native)
        self.assertIn('ld_acquire(p.doneA + j) < 512',native)
        self.assertIn('atomicAdd(p.doneA + j, 4 * count)',native)
        self.assertIn('cudaOccupancyMaxActiveBlocksPerMultiprocessor',native)
        self.assertIn('sm <= 160 * 1024',native)
        self.assertNotIn('__ldg(src',native[native.index('void owned_phase'):native.index('__global__ void __launch_bounds__')])
        ast.parse(self.source)


if __name__=='__main__':unittest.main()
