import ast
import contextlib
import io
import math
from pathlib import Path
import random
import unittest
from types import SimpleNamespace
import moe_skinny_position_check as candidate
from test_moe_skinny_testwait_source import constants

ROOT=Path(__file__).resolve().parent.parent


class PositionGate(unittest.TestCase):
    def test_duplicate_partial_nonlocal_bijection(self):
        fixtures=[[],[-1]*8,[0]*8,list(range(8)),[0,1,0,-1,1,31,31],
                  [-1]*128,[31]*128,[i%5 if i%7 else -1 for i in range(128)]]
        rng=random.Random(433)
        fixtures += [[rng.randrange(-1,32) for _ in range(n)] for n in range(1,129)]
        for eids in fixtures:
            ordered=sorted((e,i) for i,e in enumerate(eids) if e>=0)
            ent=[i for e,i in ordered]
            pos=[-1]*len(eids)
            for q,i in enumerate(ent): pos[i]=q
            self.assertEqual(candidate.model(eids),(pos,ent))

    def test_source_scope(self):
        src=(ROOT/'.build/today-moe/production51.py').read_text()
        old=constants(src); new=constants(candidate.build(src))
        restored=new['_SKINNY_BLOCK'].replace(candidate.HOOK+'\n','').replace(candidate.NEW,candidate.OLD)
        self.assertEqual(restored,old['_SKINNY_BLOCK'])
        self.assertNotIn('full_slot_ready',new['_SKINNY_BLOCK'])
        self.assertIn('mbar_try(&S.full[slot]',new['_SKINNY_BLOCK'])
        self.assertEqual(new['SKINNY_VARIANTS'][4:],((8,3,256),(8,3,512),(8,3,768)))
        import re
        self.assertEqual(set(map(int,re.findall(r'kMode\s*&\s*(\d+)',new['_SKINNY_BLOCK']))),{15,16,32,64,512})
        # All fat changes must be resource queries only, like the reviewed prior harness.
        resource=candidate.prior.RESOURCE.replace('@@QUERY@@','\n'.join(
            candidate.prior.QUERY.replace('@MODE@',str(m)) for m in (0,256,512,768)))
        stripped=new['_CUDA_SRC'].replace(resource+'\n','')
        self.assertEqual(stripped,old['_CUDA_SRC'])
        self.assertEqual(new['_CPP_SRC'].replace('\nstd::vector<int64_t> moe_wait_resources();',''),old['_CPP_SRC'])

    def test_actual_formatter_score_and_cold_round_loop(self):
        src=candidate.build((ROOT/'.build/today-moe/production51.py').read_text())
        wrapper=[n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='prepare'][-1]
        loop=next(n for n in ast.walk(wrapper) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call)
                  and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and ast.literal_eval(n.iter.args[0])==20)
        formatter=next(n for n in ast.walk(wrapper) if isinstance(n,ast.Expr) and 'PWCASE' in ast.get_source_segment(src,n))
        first=next(i for i,n in enumerate(wrapper.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='scores' for t in n.targets))
        def execute(nodes,scope):
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual_position_host','exec'),scope)
        tags=('base','null','test','test2')
        class Event:
            def __init__(self,**kw): pass
            def record(self): pass
        for win in (True,False):
            scope=dict(variants=dict.fromkeys(tags),_tw_random=random,seed=321,m=1,routing='std',
                torch=SimpleNamespace(cuda=SimpleNamespace(Event=Event)),flush=SimpleNamespace(zero_=lambda:None),
                graphs={t:SimpleNamespace(replay=lambda:None) for t in tags},events={t:[] for t in tags},
                logs=[],worst=[.004,.005],math=math,checks=64,postchecks=64,TEST_TOL=.008,
                rows=[[mode,130,0,0,155520,1,288] for mode in (0,256,512,768)])
            execute([loop],scope)
            self.assertTrue(all(len(events)==20 for events in scope['events'].values()))
            records=[]
            for m in (1,8,16,32):
                for route in ('layer','std'):
                    for seed in (321,322):
                        times=dict(base=100.,null=100.01,test=95. if win and m<=16 else 100.1,
                                   test2=95.01 if win and m<=16 else 100.2)
                        if win and m==32: times=dict(base=100.,null=100.01,test=100.,test2=100.)
                        paired={k:v-100 for k,v in times.items()}
                        records.append((m,route,seed,times,paired))
                        scope.update(m=m,routing=route,seed=seed,raw=times,paired=paired)
                        execute([formatter],scope)
            scope['records']=records
            with contextlib.redirect_stdout(io.StringIO()) as output:
                execute([n for n in wrapper.body[first:] if not isinstance(n,ast.Return)],scope)
            self.assertEqual(scope['gain'] and not scope['regress'],win)
            self.assertEqual(len(scope['logs']),16)
            self.assertIn(f'KEEP={win}',output.getvalue())


if __name__=='__main__': unittest.main()
