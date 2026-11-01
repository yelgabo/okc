"""Source equivalence gate: only two skinny full-slot sites may change policy."""
import ast
from pathlib import Path
import re
import unittest
import contextlib
import copy
import io
import math
import random
from types import SimpleNamespace
from moe_skinny_testwait_check import build,HOOK,RESOURCE,QUERY

ROOT=Path(__file__).resolve().parent.parent


def constants(src):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(src).body
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('_CUDA_SRC','_CPP_SRC','_SKINNY_BLOCK','SKINNY_VARIANTS')}


class SkinnyWaitSource(unittest.TestCase):
    def test_exact_scope(self):
        src=(ROOT/'muse_moe_w4a16/submission.py').read_text()
        old=constants(src); new=constants(build(src))
        resource=RESOURCE.replace('@@QUERY@@','\n'.join(QUERY.replace('@MODE@',str(m)) for m in (0,256,128,384)))
        self.assertEqual(new['_CUDA_SRC'].replace(resource+'\n',''),old['_CUDA_SRC'])
        self.assertEqual(new['_CPP_SRC'].replace('\nstd::vector<int64_t> moe_wait_resources();',''),old['_CPP_SRC'])
        restored=new['_SKINNY_BLOCK'].replace(HOOK+'\n','').replace(
            'full_slot_ready(&S.full[slot], (q / kNS) & 1)',
            'mbar_try(&S.full[slot], (q / kNS) & 1)')
        self.assertEqual(restored,old['_SKINNY_BLOCK'])
        self.assertEqual(new['_SKINNY_BLOCK'].count('full_slot_ready(&S.full[slot]'),2)
        self.assertEqual(new['SKINNY_VARIANTS'][:4],old['SKINNY_VARIANTS'])
        self.assertEqual(set(map(int,re.findall(r'kMode\s*&\s*(\d+)',new['_SKINNY_BLOCK']))),{15,16,32,64,128})
        for mode in (0,256,128,384):
            self.assertEqual(mode&15,0)
            self.assertEqual(mode&(16|32|64),0)
            self.assertNotIn(mode,(3,5,6,7))
        # New modes never become production tuner candidates implicitly.
        self.assertFalse(any(m in (0,3,5,6,7) for _,_,m in new['SKINNY_VARIANTS'][4:]))

    def test_host_formatters_are_not_shadowed_by_loops(self):
        tree=ast.parse(build((ROOT/'muse_moe_w4a16/submission.py').read_text()))
        wrapper=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'][-1]
        loop_names={n.target.id for n in ast.walk(wrapper) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)}
        call_names={n.func.id for n in ast.walk(wrapper) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        self.assertFalse(loop_names & call_names, f"loop name shadows callable: {loop_names & call_names}")

    def test_actual_host_loop_format_and_score_gate_on_cpu(self):
        generated=build((ROOT/'muse_moe_w4a16/submission.py').read_text())
        wrapper=[n for n in ast.parse(generated).body if isinstance(n,ast.FunctionDef) and n.name=='prepare'][-1]
        loop=next(n for n in ast.walk(wrapper) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
                  and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range'
                  and ast.literal_eval(n.iter.args[0])==20)
        formatter=next(n for n in ast.walk(wrapper) if isinstance(n,ast.Expr)
                       and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute)
                       and n.value.func.attr=='append' and 'TWCASE' in ast.get_source_segment(generated,n))
        def assigned(node,name):
            return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)
        begin=next(i for i,n in enumerate(wrapper.body) if assigned(n,'scores'))
        end=next(i for i,n in enumerate(wrapper.body) if assigned(n,'stamp_complete'))
        summary=next(i for i,n in enumerate(wrapper.body) if assigned(n,'summary'))
        def execute(nodes,scope):
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                         'actual_testwait_host_scaffold','exec'),scope)
        class Event:
            def __init__(self,**kw): pass
            def record(self): pass
        tags=('base','null','test','test2')
        scope={'variants':dict.fromkeys(tags),'_tw_random':random,'seed':321,'m':8,'routing':'std',
               'torch':SimpleNamespace(cuda=SimpleNamespace(Event=Event)),
               'flush':SimpleNamespace(zero_=lambda:None),
               'graphs':{t:SimpleNamespace(replay=lambda:None) for t in tags},
               'events':{t:[] for t in tags},'logs':[],'worst':[.004,.005],
               'raw':dict(base=100.,null=100.01,test=95.,test2=95.01),
               'paired':dict(base=0.,null=.01,test=-5.,test2=-4.99)}
        execute([loop,formatter],scope)
        self.assertEqual(len(scope['logs']),1)
        self.assertTrue(all(len(v)==20 for v in scope['events'].values()))
        # Negative control recreates the exact former built-in shadowing failure.
        class OldShadow(ast.NodeTransformer):
            def visit_Name(self,node):
                return ast.copy_location(ast.Name(id='round',ctx=node.ctx),node) if node.id=='round_idx' else node
        bad=OldShadow().visit(copy.deepcopy(loop))
        with self.assertRaises(TypeError): execute([bad,formatter],dict(scope))
        for win in (True,False):
            times=dict(base=100.,null=100.01,test=95. if win else 100.1,test2=95.01 if win else 100.2)
            deltas={t:v-100. for t,v in times.items()}
            scope.update(math=math,records=[(m,r,s,times,deltas) for m in (1,8,16,32)
                for r in ('layer','std') for s in (321,322)],checks=64,postchecks=64,
                TEST_TOL=.008,stamps=[],stamp_complete=True,
                rows=[[mode,140,0,0,155520,1,288] for mode in (0,256,128,384)])
            execute(wrapper.body[begin:end],scope)
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                execute([n for n in wrapper.body[summary:] if not isinstance(n,ast.Return)],scope)
            self.assertEqual(scope['gain'] and not scope['regress'],win)
            self.assertIn(f'KEEP={win}',printed.getvalue())


if __name__=='__main__': unittest.main()
