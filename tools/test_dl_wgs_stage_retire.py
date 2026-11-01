"""Offline edge/count audit of the exact coordinator stage-end retire delta.

This models stage order/arrival ownership, not CUDA memory execution or progress.
"""
import hashlib
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT.parent/'okc/.build/today/dqscreen'
PIN='a231faabd8b1613096b6288d3c7a77811c5378b8b7476c375d711eb9e1b38eef'


def split(pb,g0,g1):
    ue=((g1-1)>>4)<<4
    return ue if pb==0 and (g1&15)!=0 and ue>g0 else g0


def visits(pb,g0,g1):
    gs,ge=split(pb,g0,g1),g1
    result=[]
    while True:
        if gs>=ge:
            sp=split(pb,g0,g1)
            if ge==g1 and sp>g0: gs,ge=g0,sp
            else:break
        s0=gs&15;s1=min(16,s0+ge-gs)
        assert 0<=s0<s1<=16
        # Actual wgs_seg calls stage once, then s0+1 .. s1-1.
        result.extend((gs>>4)*16+st for st in range(s0,s1))
        gs+=s1-s0
    return result


class StageRetireTests(unittest.TestCase):
    def test_exact_delta(self):
        stage=(FOLDER/'stage-x.cu').read_text(encoding='utf-8')
        ctrl=(FOLDER/'ctrl.cu').read_text(encoding='utf-8')
        self.assertEqual(hashlib.sha256(stage.encode()).hexdigest(),PIN)
        old='''        if (ci == 0) {
          if (prev >= 0) {  // the previous stage's last group just retired: its x slot is free
            asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
            mbar_arrive1(S.bars + BX_E + prev);
            prev = -1;
          }
        }
'''
        expected=ctrl.replace('  int prev = -1;  // x slot the last committed group may still read\n','\n')
        self.assertEqual(expected.count(old),1);expected=expected.replace(old,'')
        expected=expected.replace('    prev = xs;\n','''    wg_wait<0>();
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
    mbar_arrive1(S.bars + BX_E + xs);
''')
        expected=expected.replace('''  wg_wait<0>();
  asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
            mbar_arrive1(S.bars + BX_E + prev);
''','  wg_wait<0>();\n')
        self.assertEqual(expected,stage)
        self.assertIn('k < BW_E ? 1 : k < BX_F ? NT : k < BX_E ? 1 : NT',stage)

    def test_exhaustive_stage_edges_and_phase_rollover(self):
        cases=0
        for g0 in range(129):
            for g1 in range(g0,257):
                seq=0
                for pb in (0,1):
                    actual=visits(pb,g0,g1)
                    expected=list(range(split(pb,g0,g1),g1))+list(range(g0,split(pb,g0,g1)))
                    self.assertEqual(actual,expected)
                    self.assertEqual(len(actual),g1-g0)
                    self.assertEqual(len(set(actual)),len(actual))
                    # Four128-reader warpgroups each release once per stage;
                    # full retirement precedes every one of512 arrivals. Every
                    # slot phase counts exactly512 before producer reuse.
                    generations={}
                    for _ in actual:
                        slot,phase=seq%3,(seq//3)&1
                        generation=seq//3
                        key=slot,generation,phase
                        self.assertNotIn(key,generations)
                        generations[key]=4*128
                        seq+=1
                    self.assertTrue(all(n==512 for n in generations.values()))
                    cases+=1
        print('STAGE_EDGE_CASES',cases,'both phases incl empty/partial/wrap',flush=True)

    def test_actual_sass_wait_mix_and_resources(self):
        for name,w1,w0 in (('ctrl',0,512),('stage-x',256,48),('g0-grid',256,16)):
            sass=(FOLDER/(name+'.sass')).read_text(encoding='utf-8')
            parts=re.split(r'\n\s*\.section\s+\.text\.([^,]+),[^\n]*\n',sass)
            count=0
            for i in range(1,len(parts),2):
                symbol,body=parts[i:i+2]
                if 'moe_fused' not in symbol:continue
                self.assertEqual(len(re.findall(r'\bHGMMA\.',body)),512)
                self.assertEqual(len(re.findall(r'WARPGROUP\.DEPBAR\.LE[^;]*, 0x1\s*;',body)),w1)
                self.assertEqual(len(re.findall(r'WARPGROUP\.DEPBAR\.LE[^;]*, 0x0\s*;',body)),w0)
                count+=1
            self.assertEqual(count,2)
        manifest=json.loads((FOLDER/'stage-x/manifest.json').read_text())
        self.assertEqual(manifest['source_sha256'],PIN)
        for entry in manifest['resources'].values():
            self.assertEqual(entry,dict(stack_bytes=8,spill_store_bytes=4,spill_load_bytes=32,registers=96,barriers=16))


if __name__=='__main__':unittest.main()
