import hashlib
from pathlib import Path
import unittest

import d32_split_copies as candidate
from test_attention_layer_continuity import config, environment, invoke


class SplitCopies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=(Path(__file__).resolve().parents[1]/candidate.BASE).read_text()

    def call(self, cfg, treatment):
        captures, owners=[],[]
        env,_,_=environment(self.base,captures,owners)
        if treatment:exec(candidate.HOST,env)
        invoke(env,cfg)
        return captures[-1][1],owners[-1]

    def test_actual_launch_scope_and_private_workspace_size(self):
        for batch in (1,16,17,24,32,33):
            for length in (1,8):
                for window in (0,2048):
                    cfg=config(batch,length,window)
                    before,_=self.call(cfg,False)
                    after,owner=self.call(cfg,True)
                    target=17<=batch<=32 and length==1 and window>0
                    if target:
                        self.assertEqual(after[22],after[20])
                        self.assertEqual(owner['observed'][-1],batch*after[22]*16)
                        self.assertEqual(after[:22],before[:22])
                        self.assertEqual(after[23:],before[23:])
                    else:self.assertEqual(before,after)

    def test_split_and_lane_destinations_have_one_writer(self):
        for splits in (1,8,11,15,264):
            addresses=set()
            sums=set()
            for split in range(splits):
                copy=split%splits
                for hk in range(2):
                    for group in range(8):
                        head=hk*8+group
                        self.assertNotIn((copy,head),sums)
                        sums.add((copy,head))
                        for tig in range(4):
                            for col in range(2):
                                for t4 in (0,4):
                                    for e in range(4):
                                        key=copy,head,16*tig+8*col+t4+e
                                        self.assertNotIn(key,addresses)
                                        addresses.add(key)
            self.assertEqual(len(addresses),splits*16*64)

    def test_frozen_device_and_explicit_override(self):
        source,manifest=candidate.build()
        self.assertTrue(source.startswith(self.base))
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),manifest['sha256'])
        captures,owners=[],[]
        env,_,_=environment(self.base,captures,owners)
        exec(candidate.HOST,env)
        state,data,_=invoke(env,config(32,1,2048))
        env['_launch7'](data,attention_state=state,nc=2)
        self.assertEqual(captures[-1][1][22],2)


if __name__=='__main__':unittest.main()
