"""CPU source/lifetime and timeline gates; pass pinned pure G1b4 as argv1."""
import ast,re,sys,unittest
from pathlib import Path
from dl_make_g1b4_edge import patch,edge_summary
SOURCE=Path(sys.argv.pop(1)).read_text(encoding='utf8')

class Edge(unittest.TestCase):
    def test_sync_and_math_inverse(self):
        _,new=patch(SOURCE)
        old=next(ast.literal_eval(n.value) for n in ast.parse(SOURCE).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_SL_GLUE_CUDA')
        start='template<int V,int NW,bool POST,bool SLICED,bool ROUTER,bool PRELOADED_BASE>'
        stop='template<typename F> void glue_launch'
        original=old[old.index(start):old.index(stop)]
        traced=new[new.index(start):new.index(stop)]
        traced=re.sub(r'  edge_mark\(a,[0-4]\);\n','',traced)
        self.assertEqual(original,traced)
        self.assertEqual(old.count('griddepcontrol.wait'),new.count('griddepcontrol.wait'))
        self.assertEqual(old.count('cluster.sync()'),new.count('cluster.sync()'))
        self.assertEqual(old.count('fence'),new.count('fence'))
        self.assertIn('programmaticStreamSerializationAllowed=0',new)
        self.assertEqual(new.count('a.edge_stamp[blockIdx.x*5+k]=t'),1)
        self.assertEqual({b*5+k for b in range(8) for k in range(5)},set(range(40)))

    def test_exposed_edge_not_summed_kernel_time(self):
        e=[(0,10000,1,'void sk2_kernel<o>'),(10000,13200,1,'void glue_cluster_boundary<4, 32, false, false, true, false>'),(12000,20000,2,'parallel'),(13200,25000,1,'route')]
        r=edge_summary(e);self.assertEqual(r['consumer_duration_us'],3.2)
        self.assertEqual(r['edge_gap_us'],0);self.assertEqual(r['graph_tail_after_consumer_us'],11.8)
        # Summed activity durations overstate the graph span when activity overlaps.
        self.assertGreater(sum(b-a for a,b,_,_ in e),max(b for _,b,_,_ in e))
        with self.assertRaises(ValueError):edge_summary(e+[(14000,17000,1,e[1][3])])

    def test_probe_guards_do_not_affect_prepare_or_judge(self):
        from dl_make_g1b4_edge import RUNTIME
        tree=ast.parse(RUNTIME)
        b1=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_edge_b1')
        calls=[]
        class Module:
            def b1(self,*args):calls.append(args);return 'original'
        env={'_sl_cluster_module':lambda:Module()}
        exec(compile(ast.Module(body=[b1],type_ignores=[]),'inactive_b1','exec'),env)
        self.assertEqual(env['_edge_b1']({},'unsupported shape allowed outside probe'),'original')
        self.assertEqual(env['_edge_b1']({'_edge_trace':False},'official nonD1'),'original')
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_edge_run')
        def fail(data,state):raise ValueError('simulated kernel failure')
        env={'_EDGE_ORIGINAL_KERNEL':fail,'PDL':False};exec(compile(ast.Module(body=[run],type_ignores=[]),'guard','exec'),env)
        st={'_fork_native_eligible':True}
        with self.assertRaises(ValueError):env['_edge_run']({},st)
        self.assertFalse(st['_edge_active'])

    def test_private_tensor_scope(self):
        import torch
        from dl_make_g1b4_edge import RUNTIME
        node=next(n for n in ast.parse(RUNTIME).body if isinstance(n,ast.FunctionDef) and n.name=='_edge_ptrs')
        env={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'ptrs','exec'),env)
        a={'cache':torch.ones(4),'scratch':[torch.empty(8)]};b={k:([x.clone() for x in v] if isinstance(v,list) else v.clone()) for k,v in a.items()}
        self.assertFalse(env['_edge_ptrs'](a)&env['_edge_ptrs'](b))
        b['scratch']=a['scratch'];self.assertTrue(env['_edge_ptrs'](a)&env['_edge_ptrs'](b))

if __name__=='__main__':unittest.main()
