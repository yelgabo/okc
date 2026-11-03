import ast
import base64
import contextlib
import io
import json
import types
import unittest
import zlib

import d32_identity_pair as builder
from attention_top100_screen import function
from test_attention_layer_continuity import config, environment, invoke


class IdentityPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.manifest=builder.build()
        cls.tree=ast.parse(cls.source)

    def test_actual_arm_selection(self):
        captures=[]
        env,_,_=environment(self.source,captures,[])
        wanted={'_d32_attention_scope','_d32_attention_module','_d32_pair_bind'}
        aliases={'_d32_pair_parent_module','_d32_pair_selected_module'}
        nodes=[n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted
               or isinstance(n,ast.Assign) and any(getattr(t,'id','') in aliases for t in n.targets)]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'identity-pair', 'exec'),env)
        env['_WL_KERNEL']=env['kernel_fn']
        function(self.tree,'_wl_run',env)
        module=types.SimpleNamespace(run=lambda *a:captures.append(('identity',a)))
        env['_prepare_attention_modules7']=lambda s:s.update(d32_attention_module=module,d32_attention_eligible=True)
        for batch,length in ((1,1),(1,8),(16,1),(32,1),(16,8),(32,8)):
            for window in (0,2048):
                for arm in (False,True):
                    state,data,_=invoke(env,config(batch,length,window))
                    env['_d32_pair_bind'](state,arm)
                    env['_wl_run'](data,state)
                    expected=arm and batch==32 and length==1 and window==0
                    self.assertEqual(state['_d32_probe_seen'],[expected])
        state['_d32_pair_arm']=not arm
        with self.assertRaises(RuntimeError):env['_wl_run'](data,state)

    def progress_env(self):
        env={'time':types.SimpleNamespace(monotonic=lambda:281.),'_EH_START':0.,
             '_EH_PINS':self.manifest['pins'],'_D32_PROGRESS':{'rows':[],'stage':'entry'}}
        function(self.tree,'_d32_emit',env)
        function(self.tree,'_d32_checkpoint',env)
        return env

    def test_partial_rows_and_budget_failure(self):
        env=self.progress_env()
        rows=[dict(tokens=t,context=c,seed=s,times_us=[10.,10.,9.,9.],selected=[[False],[False],[t==32 and c=='full'],[t==32 and c=='full']])
              for t in (32,1,8,16,128,256) for c in ('sliding','full') for s in (1102031,1102037,1102049)]
        env['_D32_PROGRESS']['rows']=rows
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaisesRegex(AssertionError,'D32_PHASE_BUDGET'):
                env['_d32_checkpoint']('row',280)
            env['_d32_emit']('failure',ValueError('injected route mismatch'))
        records=[json.loads(zlib.decompress(base64.b64decode(line.split(' ',1)[1])))
                 for line in out.getvalue().splitlines()]
        self.assertEqual(records[-1]['rows'],rows)
        self.assertEqual(records[-1]['pins'],self.manifest['pins'])
        self.assertEqual(records[-1]['status'],'FAILED_PARTIAL_NOT_QUALIFIED')
        self.assertIn('injected route mismatch',records[-1]['error'])
        self.assertLess(len(out.getvalue()),11000)

    def test_prepare_failure_retains_original_error_and_restores_hooks(self):
        class Failure(Exception):pass
        env=self.progress_env()
        env['time'].monotonic=lambda:0.
        def fail(_):raise Failure('compile failed')
        old=[object(),object(),object()]
        torch=types.SimpleNamespace(backends=types.SimpleNamespace(cuda=types.SimpleNamespace(
            matmul=types.SimpleNamespace(allow_tf32=True))))
        env.update(_EH_OFFLINE_HOLD=False,_EH_CACHE=None,_EH_BUILDS={},_GUARDS=[],_GUARDED_IDS=set(),
                   _ma_ext=old[0],_fork_native_moe=old[1],_ab_ref_back=old[2],
                   _eh_build=object(),_eh_native=object(),_eh_back=object(),_EH_PREPARE=fail,
                   _AB_DEQ={},torch=torch)
        function(self.tree,'_eh_budget',env)
        prepares=[n for n in self.tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare']
        exec(compile(ast.Module(body=[prepares[-1]],type_ignores=[]),'last-prepare','exec'),env)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaisesRegex(Failure,'compile failed'):env['prepare']({})
        self.assertEqual([env[k] for k in ('_ma_ext','_fork_native_moe','_ab_ref_back')],old)
        self.assertIn('D32_IDENTITY_PROGRESS',out.getvalue())

    def test_complete_coverage_and_private_hold(self):
        self.assertIn('for tokens in (32,1,8,16,128,256):',self.source)
        self.assertIn('checks==restores==144 and fresh_checks==1152',self.source)
        self.assertIn('for rep in range(24):',self.source)
        self.assertIn("_d32_checkpoint('prepared',180)",self.source)
        self.assertIn("_d32_checkpoint('owners',210)",self.source)
        self.assertIn("_d32_checkpoint('serialize',290)",self.source)
        enabled,_=builder.build(True)
        self.assertEqual(enabled,self.source.replace('_EH_OFFLINE_HOLD=True','_EH_OFFLINE_HOLD=False',1))


if __name__=='__main__':unittest.main()
