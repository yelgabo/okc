"""Real prepare wrapper, receipt negatives and unchanged native source gates."""

import contextlib
import io
import json
import unittest

import g_pr_graph_owner_receipt as treatment
import test_g_pr_graph_owner_relaxed as scoped

scoped.events.legacy.treatment = treatment
scoped.events.legacy.SOURCE = treatment.build(scoped.events.legacy.ORIGINAL)
scoped.treatment = treatment
scoped.events.treatment = treatment


class ReceiptContracts(scoped.ScopeContracts):
    def test_actual_prepare_end_and_schema_no_values(self):
        legacy = scoped.events.legacy
        env, events, captured = legacy.integrated()
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            first = env['prepare']({})
            second = env['prepare']({})
        receipts = [json.loads(line.split(' ',1)[1]) for line in output.getvalue().splitlines()]
        self.assertEqual(len(receipts),2)
        self.assertEqual(events,['normal_prepare','complete_prior_work','original_startup']*2)
        for receipt in receipts:
            self.assertEqual(set(receipt),{'schema_version','status','outcome','accepted_pr_plans'})
            self.assertEqual(receipt['outcome'],'ACTIVE_TREATMENT')
            self.assertEqual([row['tokens'] for row in receipt['accepted_pr_plans']],[128,256])
            for row in receipt['accepted_pr_plans']:
                self.assertEqual(row,dict(tokens=row['tokens'],shape=[4608,8192],dtype='bf16',owner_tagged=True))
        self.assertIsNot(first,second)
        self.assertEqual(captured,[])
        self.assertIn('return state',treatment.APPEND)

    def test_no_treatment_and_prepare_exception_negatives(self):
        legacy = scoped.events.legacy
        for case in ('unsupported','no-plans','wrong-alias','missing-status','partial'):
            env, unused_events, unused_captured = legacy.integrated()
            with contextlib.redirect_stdout(io.StringIO()):
                state=env['prepare']({})
            if case=='unsupported':
                state['_pr_graph_status']='no-treatment:capability'
                state['graft_lt']={}
            elif case=='no-plans':
                state['graft_lt']={}
            elif case=='wrong-alias':
                for tokens in (128,256):
                    state['graft_lt'][('pr',tokens)]['weight']=state['pr']
            elif case=='missing-status':
                state.pop('_pr_graph_status')
            else:
                state['graft_lt'].pop(('pr',256))
            env['_pr_graph_receipt_prepare']=lambda weights: state
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIs(env['prepare']({}),state)
            receipt=json.loads(output.getvalue().split(' ',1)[1])
            self.assertEqual(receipt['outcome'],'NO_TREATMENT')
        env['_pr_graph_receipt_prepare']=lambda weights: (_ for _ in ()).throw(RuntimeError('prepare-failure'))
        output=io.StringIO()
        with contextlib.redirect_stdout(output),self.assertRaisesRegex(RuntimeError,'prepare-failure'):
            env['prepare']({})
        self.assertEqual(output.getvalue(),'')

    def test_frozen_prefix_and_timed_receipt_absence(self):
        source=treatment.build(treatment.BASE.read_bytes().decode())
        self.assertTrue(treatment.validate(source,treatment.BASE.read_bytes().decode())['receipt_only_append'])
        self.assertEqual(source[:-len(treatment.APPEND)],treatment.prior.build(treatment.BASE.read_bytes().decode()))
        self.assertEqual(treatment.HOST,treatment.prior.HOST)
        self.assertNotIn('RECEIPT',treatment.function(source,'_graft_lt_run'))
        with self.assertRaises(ValueError):
            treatment.validate(source.replace('flush=True','flush=False'),treatment.BASE.read_bytes().decode())


if __name__ == '__main__':
    unittest.main()

