import base64
import contextlib
import io
import json
import unittest
import zlib

import torch
import d32_identity_pair as builder
from d32_attn_worker_harness_torch_test import AttentionHost
from d32_attn_worker_read import report as read_report


class GeneratedIdentityHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source,cls.manifest=builder.build()

    def test_complete_host_replay_and_report(self):
        host=AttentionHost(self.source,self.manifest)
        report=host.run()
        self.assertEqual((len(report['rows']),report['math_checks'],report['fresh_checks'],
                          report['restores'],report['transitions']),(36,144,1152,144,96))
        self.assertEqual([r['tokens'] for r in report['rows'][:6]],[32]*6)
        self.assertEqual(read_report(host.log.getvalue(),self.manifest['pins']),report)
        with self.assertRaises(ValueError):read_report(host.log.getvalue())
        self.assertTrue(report['cleanup_restored'])
        self.assertEqual(len({s['ma_scratch'][0].untyped_storage().data_ptr()
                              for s in host.owners.values()}),4)

    def test_failure_has_partial_and_preserves_route_gate(self):
        host=AttentionHost(self.source,self.manifest)
        checkpoint=host.namespace['_d32_checkpoint']
        def corrupt_after_row(stage,deadline):
            checkpoint(stage,deadline)
            if stage=='row':host.failure='replay_output'
        host.namespace['_d32_checkpoint']=corrupt_after_row
        with self.assertRaises((RuntimeError,AssertionError)):host.run()
        records=[json.loads(zlib.decompress(base64.b64decode(s.split(' ',1)[1])))
                 for s in host.log.getvalue().splitlines() if s.startswith('D32_IDENTITY_PROGRESS ')]
        self.assertEqual(records[-1]['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
        self.assertGreater(records[-1]['covered'],0)
        self.assertEqual(records[-1]['covered'],len(records[-1]['rows']))
        self.assertNotIn('D32_ATTENTION_PAIR_REPORT ',host.log.getvalue())
        self.assertIsNone(host.namespace['_EH_CACHE'])
        self.assertIs(host.namespace['_ma_ext'],host.original['_EH_BUILD'])
        self.assertTrue(host.torch.backends.cuda.matmul.allow_tf32)
        host=AttentionHost(self.source,self.manifest)
        ids=torch.tensor([[1,2,3,4]],dtype=torch.int32)
        outputs=[(None,None,ids.clone()) for _ in range(4)]
        outputs[2][2][0,1:3]=outputs[2][2][0,1:3].flip(0)
        arms=[dict(_d32_pair_arm=i>=2,_d32_probe_seen=[i>=2]) for i in range(4)]
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(AssertionError,'ROUTING_MISMATCH'):
            host.namespace['_d32_require_same_routes'](outputs,arms,32,'full',1102031,'injected-swap')


if __name__=='__main__':unittest.main()
