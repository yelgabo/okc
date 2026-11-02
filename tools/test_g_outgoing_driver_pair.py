"""Reuse the fixed real-entry suite with explicit driver/native CPU stand-ins."""
from pathlib import Path

from g_short_rmw_check import once


text=Path(__file__).with_name('test_g_outgoing_pair.py').read_text(encoding='utf8')
text=once(text,'from g_outgoing_pair import build,MARKER','from g_outgoing_driver_pair import build,MARKER')
text=once(text,'import g_outgoing_pair_read as reader','import g_outgoing_driver_pair_read as reader')
text=once(text,'[0,1,10],[0,2,200 if arm>=2 else 100]','[0,1,1010],[0,2,1200 if arm>=2 else 1100]')
text=text.replace('[10,11]','[1010,1011]')
assert text.count(',200,True)')==2
text=text.replace(',200,True)',',1200,True)')
text=once(text,',100,False)',',1100,False)')
text=once(text,'cudaGraphGetEdges(graph,from.data(),to.data(),edges.data(),&count)','cuGraphGetEdges(graph,from.data(),to.data(),edges.data(),&count)')
text=once(text,'def seam_graph_edges(self,address):return graph_rows(address-1000)', '''def seam_driver_unpool_identity(self,row,arm):return self.pointer+1000
            def seam_driver_graph_edges(self,address,row,arm):return graph_rows(address-1000)''')
text=once(text,'def install(state,enabled):','''def install(state,enabled):
            ns['_OS_BUILDS'].append(dict(name='mock_secondary',functions=['seam_driver_graph_edges'],flags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],link_flags=['-lcuda']))''')
text=once(text,'patch.object(old.Native,"seam_native_identity",return_value=[1010,1011],create=True)', 'patch.object(old.Native,"seam_native_identity",return_value=[10,11],create=True),patch.object(old.Native,"seam_driver_primary_identity",return_value=[1010,1011],create=True)')
text=text.replace('OUTGOING_REPORT ','OUTGOING_DRIVER_REPORT ')
text=text.replace('if __name__=="__main__":unittest.main()','')
exec(compile(text,__file__+':fixed-suite','exec'),globals())


class IntegrationSpecific(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RealEntry.setUpClass()

    def test_exact_fixed_code_and_host_only_literal_changes(self):
        from g_outgoing_pair import build as original
        from g_common_rmw_pair import function
        import g_common_release as exports
        from g_outgoing_driver_query import PRIMARY
        old_source,_=original(ROOT)
        self.assertTrue(MANIFEST['reverse_exact'])
        for name in ('_wl_run','_wl_capture','_wl_check','_wl_check_owners','_wl_empty','_srp_occupied_empty','_os_secondary_gate','_os_edges','_os_score','_fork_native_moe','_os_install','_os_native_sources'):
            before=function(old_source,name);after=function(SOURCE,name)
            if name=='_os_install':
                before=before.replace("functions=['seam_sk2','seam_unpool_resources','seam_graph_edges']", "functions=['seam_sk2','seam_unpool_resources','seam_driver_unpool_identity','seam_driver_graph_edges']")
            self.assertEqual(before,after,name)
        for name in ('MOE_CUDA_SRC',"_MA_ALT_SRCS['moe_helper0']"):
            before=ast.literal_eval(exports.assignment(old_source,name).value)
            after=ast.literal_eval(exports.assignment(SOURCE,name).value)
            self.assertEqual(after,before+PRIMARY)
        for name in ('bfp_SK2_SRC','bfp_CUDA_SRC','bfp_MAP_SRC'):
            self.assertEqual(ast.literal_eval(exports.assignment(old_source,name).value),ast.literal_eval(exports.assignment(SOURCE,name).value))

    def test_real_loader_wires_linker_and_preserves_cuda_flags(self):
        from g_common_rmw_pair import function
        calls=[]
        ns=dict(time=__import__('time'),json=json,_OS_BUILDS=[],_OS_ORIGINAL_LOAD=lambda *args,**kwargs:calls.append(kwargs) or object())
        exec(function(SOURCE,'_os_load_inline'),ns)
        flags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-DSK2_DELAY_ESC=9']
        with contextlib.redirect_stdout(io.StringIO()):
            ns['_os_load_inline'](name='query',functions=['seam_driver_graph_edges'],extra_cuda_cflags=flags,cuda_sources='host',cpp_sources='decl')
            ns['_os_load_inline'](name='other',functions=['attention'],extra_cuda_cflags=flags)
        self.assertEqual(calls[0]['extra_ldflags'],['-lcuda'])
        self.assertNotIn('extra_ldflags',calls[1])
        self.assertEqual(calls[0]['extra_cuda_cflags'],flags)
        self.assertEqual(ns['_OS_BUILDS'][0]['link_flags'],['-lcuda'])

    def test_real_graph_entry_prefix_and_namespace_negatives(self):
        test=RealEntry();ns,_,_=test.fixture()
        messages=[];ns['_os_driver_emit']=lambda text,**kwargs:messages.append(text)
        saved=ns['_os_driver_validate']
        def wrong(rows,primary,consumer,enabled):
            primary=dict(primary,namespace='host_stub')
            return saved(rows,primary,consumer,enabled)
        ns['_os_driver_validate']=wrong
        with self.assertRaisesRegex(AssertionError,'DRIVER_IDENTITY_NAMESPACE'):
            test.call(ns)
        self.assertTrue(messages[0].startswith('SEAM_DRIVER_PREFIX '))
        self.assertIn('DRIVER_IDENTITY_NAMESPACE',messages[-1])
        self.assertIsNone(ns['_SRP_CACHE'])

    def test_integrated_policy_owner_and_foreign_failure_negatives(self):
        test=RealEntry();ns,_,_=test.fixture()
        with self.assertRaisesRegex(AssertionError,'DRIVER_ARM_POLICY'):
            ns['_os_driver_inspect'](None,None,None,row=0,arm=3,enabled=False,resources={'mock':True},build_census=['mock'])
        shared=torch.ones(1)
        with self.assertRaisesRegex(AssertionError,'mutable storage overlap'):
            ns['_wl_check_owners']([dict(attention_scratch={},mutable=shared),dict(attention_scratch={},mutable=shared)],set())
        messages=[];ns['_os_driver_emit']=lambda text,**kwargs:messages.append(text)
        def foreign(address,row,arm):raise RuntimeError(f'api=cuGraphKernelNodeGetParams_v2 code=98 node=4 row={row} arm={arm}')
        ns['bfp__get_ext']().seam_driver_graph_edges=foreign
        with self.assertRaisesRegex(RuntimeError,'api=cuGraphKernelNodeGetParams_v2 code=98 node=4'):
            test.call(ns)
        prefix=json.loads(messages[0].split(' ',1)[1])
        self.assertEqual(set(prefix['resources']),{'primary','secondary','common'})
        self.assertTrue(prefix['build_census'])
        self.assertIn('node=4',messages[-1]);self.assertIn('arm=1',messages[-1])

    def test_reader_rejects_identity_and_linkage_drift(self):
        test=RealEntry();ns,_,_=test.fixture();_,log=test.call(ns)
        reader.MANIFEST=MANIFEST;report,_=reader.capsule(log)
        for mutate in (lambda value:value.__setitem__('driver_identity_namespace','host_stub'),
                       lambda value:value['driver_secondary_identities']['1'].__setitem__(3,999),
                       lambda value:value['build_census'][0].__setitem__('link_flags',[])):
            changed=copy.deepcopy(report);mutate(changed)
            encoded='OUTGOING_DRIVER_REPORT '+base64.b64encode(zlib.compress(json.dumps(changed).encode(),9)).decode()+'\n'
            with self.assertRaises(AssertionError):reader.capsule(encoded)


if __name__=='__main__':unittest.main()
