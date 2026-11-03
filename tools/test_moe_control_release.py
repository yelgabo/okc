"""Check standalone control ownership, dispatch and graph-safe forwarding."""
import unittest

import torch

from expert_top100_moe_test import environment,inputs,prepare_small
from moe_control_release import build


class Release(unittest.TestCase):
    def test_exact_control_and_owned_forwarding(self):
        source,manifest=build();ns,calls,launches=environment(source)
        one=prepare_small(ns)[0];two=prepare_small(ns)[0]
        self.assertFalse(one['_expert_pair_arm'])
        self.assertEqual(len(calls),5)
        self.assertFalse(manifest['public_submission_approved'])
        self.assertFalse({x.data_ptr() for x in one['ma_scratch']}&{x.data_ptr() for x in two['ma_scratch']})
        for tokens in (1,4,8,16,32,33,128,256,257):
            data=inputs(tokens,tokens);result=ns['kernel_fn'](data,one)
            self.assertTrue(torch.equal(result,data['hidden_states']))
            for actual,key in zip(launches[-1][1][2:5],('hidden_states','topk_ids','topk_weights')):
                self.assertIs(actual,data[key])
        one['_expert_pair_arm']=True
        with self.assertRaisesRegex(RuntimeError,'STATE_IDENTITY_OR_ARM'):ns['kernel_fn'](inputs(32),one)
        with self.assertRaisesRegex(RuntimeError,'INVALID_REBIND'):ns['_moe_bind_control'](two,True)


if __name__=='__main__':unittest.main()
