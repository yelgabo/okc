"""CPU proof for layer private storage/Lt/captures and complete score merger."""
import ast,contextlib,copy,io,unittest
from pathlib import Path
from dl_g1_layer_merge import decide

class Gate(unittest.TestCase):
    def owners(self):
        import torch
        env={'torch':torch,'_GUARDS':[]};exec(Path(__file__).with_name('dl_g1_layer_owners.py').read_text(encoding='utf8'),env);return env
    def test_storage_alias_and_nested_lt(self):
        import torch
        env=self.owners()
        class _GraftLt:pass
        class _RouterExt:pass
        weight=torch.randn(16);raw=torch.arange(64,dtype=torch.int32)
        lt=_GraftLt()
        for key in ('lib','handle','desc','alpha','beta'):setattr(lt,key,object())
        lt.workspace=torch.empty(100,dtype=torch.uint8)
        state={'qkv':weight,'ma_w':[weight],'ma_scratch':[raw[3:12],raw[17:21]],'attention_scratch':{},
               'ma_ext':{'main':_RouterExt(),'tiny':_RouterExt()},'graft_lt':{'x':{'lt':lt,'weight':weight,'algo':object(),'layouts':[object()]}}}
        env['_GUARDS']=[(raw,40)]
        readonly=env['_wl_readonly'](state,{'qkv_gate':weight})
        arms=[env['_wl_clone'](state,readonly,i>=2) for i in range(4)]
        self.assertEqual(len({env['_wl_storage'](a['ma_scratch'][0]) for a in arms}),4)
        self.assertEqual(env['_wl_storage'](arms[0]['ma_scratch'][0]),env['_wl_storage'](arms[0]['ma_scratch'][1]))
        self.assertEqual(arms[0]['ma_scratch'][0].storage_offset(),3)
        self.assertTrue(all(a['ma_w'][0] is weight for a in arms))
        self.assertEqual(len({env['_wl_storage'](a['graft_lt']['x']['lt'].workspace) for a in arms}),4)
        self.assertIs(arms[0]['graft_lt']['x']['lt'].handle,lt.handle)
        env['_wl_check_owners'](arms,readonly)
        # Disjoint-looking views of one storage still fail the conservative owner gate.
        arms[1]['ma_scratch']=[arms[0]['ma_scratch'][0][5:]]
        with self.assertRaises(AssertionError):env['_wl_check_owners'](arms,readonly)
    def test_uninventoried_object_fails(self):
        env=self.owners()
        class CallableOwner:
            def __call__(self):pass
        with self.assertRaises(RuntimeError):env['_wl_clone']({'unknown':CallableOwner()},set(),False)
    def reports(self,on=96):
        parts=[]
        for name,loads in [('decode',(1,16,32)),('verify',(8,128,256))]:
            rows=[{'tokens':t,'context':c,'seed':seed,'times_us':[100,100,on,on] if t in (32,128,256) else [100,100,100,100]} for t in loads for c in ('sliding','full') for seed in (1,2,3)]
            parts.append({'part':name,'candidate_pin':'pin','device_sha':'d','harness_core_sha':'h','weight_signature':'w','seeds':[1,2,3],'resources':{},'rows':rows,'judged_stage_on':True,'empty_native_checks':12,'math_checks':72,'fresh_final_checks':72})
        return parts
    def test_split_complete_and_weighted_gate(self):
        r=self.reports();self.assertTrue(decide(r)['decision'])
        with self.assertRaises(AssertionError):decide(r[:1])
        bad=copy.deepcopy(r);bad[1]['weight_signature']='different'
        with self.assertRaises(AssertionError):decide(bad)
        bad=copy.deepcopy(r);bad[1]['rows'][0]=bad[0]['rows'][0]
        with self.assertRaises(AssertionError):decide(bad)
        self.assertFalse(decide(self.reports(100))['decision'])
        bad=self.reports(80);bad[0]['rows'][0]['times_us']=[100,100,110,110]
        self.assertFalse(decide(bad)['decision'])
    def test_generated_parts_share_exact_device_and_core(self):
        from dl_g1_layer_gate import build,PIN
        root=Path(__file__).resolve().parents[1].parent/'okc'
        source=(root/'.build/today/g1b4_native_experts.py').read_text(encoding='utf8')
        outputs=[build(source,root,p) for p in ('decode','verify')]
        def assignments(source):return {getattr(n.targets[0],'id',''):ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','') in ('_WL_CORE_SHA','_WL_DEVICE_SHA','_WL_CONFORMANCE')}
        self.assertEqual(assignments(outputs[0]),assignments(outputs[1]))
        self.assertIsNone(assignments(outputs[0])['_WL_CONFORMANCE'])
        for out in outputs:
            self.assertLess(len(out.encode()),1_000_000)
            self.assertIn('G1b4 conformance receipt required before measurement',out)
            self.assertIn('OKC_STAGE_RETIRE=1',out)
            self.assertIn('wl_resources',out)

    def test_actual_b4_compiled_flag_gate(self):
        body=Path(__file__).with_name('dl_g1_layer_body.py').read_text(encoding='utf8')
        node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='_fork_native_moe')
        class Ext:
            _module=object()
        ext=Ext();state={'_wl_probe':True,'_fork_native_eligible':True,'ma_ok':True,'_wl_stage_on':True,'_wl_hits':[],'ma_ext':{'b4':ext}}
        flags={id(ext._module):('-DOKC_WGS=1','-DOKC_B4FIN=1','-DOKC_STAGE_RETIRE=1')}
        env={'_WL_NATIVE':lambda m,s:ext,'_pdl':lambda m:False,'_WL_EXT_FLAGS':flags}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'backend','exec'),env)
        self.assertIs(env['_fork_native_moe'](32,state),ext)
        self.assertEqual(state['_wl_hits'],[(32,'b4',True)])
        flags[id(ext._module)]=('-DOKC_WGS=1','-DOKC_B4FIN=1')
        with self.assertRaises(AssertionError):env['_fork_native_moe'](32,state)

    def test_tf32_restored_failclosed(self):
        import torch
        body=Path(__file__).with_name('dl_g1_layer_body.py').read_text(encoding='utf8')
        node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
        def failure(w):raise RuntimeError('budget')
        env={'torch':torch,'_wl_prepare':failure};exec(compile(ast.Module(body=[node],type_ignores=[]),'prepare','exec'),env)
        old=torch.backends.cuda.matmul.allow_tf32
        with self.assertRaises(RuntimeError):env['prepare']({})
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32,old)

if __name__=='__main__':unittest.main()
