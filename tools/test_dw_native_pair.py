"""Complete-entry real-Torch orchestration facades; no GPU/performance evidence."""
import base64
import contextlib
import copy
import hashlib
import io
import json
import types
import unittest
import zlib
import torch
import dw_native_pair_package as pair
import dw_native_pair_harness as builder
import dw_native_pair_read as reader
import test_expert_top100_harness_torch as prior_torch_tests
from test_expert_top100_harness import fixture
from test_g_expert_g224_prefix import capture,BASE


class DirectHost(prior_torch_tests.Host):
    def run(self,weights=None):
        ns=self.namespace;ns['_EH_OFFLINE_HOLD']=False;self.log=io.StringIO()
        with contextlib.redirect_stdout(self.log):self.state=ns['prepare'](self.weights() if weights is None else weights)
        blob=self.log.getvalue().split('DW_NATIVE_PAIR_REPORT ')[1].strip()
        self.report=json.loads(zlib.decompress(base64.b64decode(blob)))
        return self.report


class DWPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.source,cls.manifest=builder.build()

    def cleanup(self,host,success=False):
        prior_torch_tests.GeneratedTorch.assert_cleanup(self,host,success)

    def test_exact_control_and_candidate_factory_calls(self):
        selected,calls=pair.selected();old,_=capture(BASE,False)
        # Compare all five original calls by immutable module name.
        originals={c['name']:c for c in calls if c['name'] in {v['name'] for v in old}}
        self.assertEqual(originals,{c['name']:c for c in old})
        self.assertEqual(len(calls),7)
        from test_wgs_direct_weight_registers import selected as frozen
        candidate,_,_=frozen()
        self.assertEqual({c['name']:c for c in calls if c['name'] in {v['name'] for v in candidate}}, {c['name']:c for c in candidate})
        self.assertEqual(selected,self.manifest['selected'])

    def test_complete_entry_all_rows_inference_and_official_binding(self):
        for inference,infweights in ((False,False),(False,True),(True,False),(True,True)):
          with self.subTest(inference=inference,weights=infweights):
            host=DirectHost(self.source,self.manifest)
            with torch.inference_mode(infweights):weights=host.weights()
            with torch.inference_mode(inference):r=host.run(weights)
            self.assertEqual(r['official_arm'],'direct_weight')
            self.assertIs(host.state['_expert_pair_arm'],True)
            self.assertIs(host.state['_expert_pair_arm_bound'],True)
            self.assertTrue(host.state['ma_ext']['main']._module.__name__.startswith('direct_weight_'))
            self.assertEqual((len(r['rows']),r['math_checks'],r['fresh_checks'],r['restores'],r['transitions']),(36,144,1152,144,96))
            self.assertFalse(r['decision']['gate'])  # Constant synthetic timing is a null.
            self.assertEqual(len(host.owners),4)
            self.assertEqual(sorted(s['_expert_pair_arm'] for s in host.owners.values()),[False,False,True,True])
            self.assertEqual(len({id(s['ma_ext']) for s in host.owners.values()}),4)
            storages=[s['ma_scratch'][0].untyped_storage().data_ptr() for s in host.owners.values()]
            self.assertEqual(len(set(storages)),4)
            self.assertNotIn(host.state['ma_scratch'][0].untyped_storage().data_ptr(),storages)
            for s in host.owners.values():
                self.assertFalse(torch.is_inference(s['ma_scratch'][0]))
                self.assertIs(s['ma_w'][0],weights['qkv'])  # Shared immutable coefficient, not mutable state.
            self.assertFalse(any(host.reference_modes))
            self.cleanup(host,True)
            # CPU report devices must not be accepted as GPU evidence.
            with self.assertRaisesRegex(AssertionError,'readonly storage'):reader.capsule(host.log.getvalue(),self.manifest)

    def test_failures_restore_original_bindings_and_guards(self):
        for failure in ('prepare','native','replay_output','local_ids_after_native','guard','unknown_owner','output_alias'):
          with self.subTest(failure=failure):
            host=DirectHost(self.source,self.manifest,failure)
            with self.assertRaises((RuntimeError,AssertionError)):host.run()
            self.assertNotIn('DW_NATIVE_PAIR_REPORT ',host.log.getvalue())
            self.cleanup(host)

    def test_hold_rebinding_mutation_and_wrong_module(self):
        host=DirectHost(self.source,self.manifest);ns=host.namespace
        with self.assertRaisesRegex(RuntimeError,'STOP_CPU_HOST_GATE'):ns['prepare'](host.weights())
        self.assertFalse(host.loads)
        ns['_EH_START']=__import__('time').monotonic();ns['_ma_ext']=ns['_eh_build']
        state=host.prepare_device_state(host.weights());ns['_expert_pair_bind'](state,True)
        with self.assertRaisesRegex(RuntimeError,'INVALID_BIND'):ns['_expert_pair_bind'](state,False)
        state['_expert_pair_arm']=False
        with self.assertRaisesRegex(RuntimeError,'ARM_MUTATED'):ns['_EH_NATIVE'](32,state)
        state['_expert_pair_arm']=True
        other=host.prepare_device_state(host.weights());ns['_expert_pair_bind'](other,False)
        state['ma_ext']['b4']=other['ma_ext']['b4'];state['_wl_probe']=True
        with self.assertRaisesRegex(AssertionError,'SELECTED_SOURCE'):ns['_eh_native'](32,state)

    def test_strict_reader_source_job_official_and_duplicate_negatives(self):
        args=fixture(self.manifest,self.source)
        report=json.loads(zlib.decompress(base64.b64decode(args[2].split()[1])))
        report['official_arm']='direct_weight'
        def encode(r):return 'DW_NATIVE_PAIR_REPORT '+base64.b64encode(zlib.compress(json.dumps(r,separators=(',',':')).encode())).decode()+'\n'
        args[2]=encode(report);args[6]['sha256']=hashlib.sha256(args[2].encode()).hexdigest()
        observed=reader.read(*args);self.assertEqual(observed['official_arm'],'direct_weight')
        self.assertFalse(observed['release_eligible'])
        wrong=copy.deepcopy(args);wrong[1]='wrong'
        with self.assertRaisesRegex(AssertionError,'own private job'):reader.read(*wrong)
        wrong=copy.deepcopy(args);wrong[4]=wrong[4]+b'\n'
        with self.assertRaisesRegex(AssertionError,'source artifact'):reader.read(*wrong)
        report['official_arm']='exact607'
        with self.assertRaisesRegex(AssertionError,'official/restoration'):reader.capsule(encode(report),self.manifest)
        with self.assertRaisesRegex(AssertionError,'missing/duplicate'):reader.capsule(args[2]+args[2],self.manifest)


if __name__=='__main__':unittest.main(verbosity=2)
