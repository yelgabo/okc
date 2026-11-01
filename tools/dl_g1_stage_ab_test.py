"""CPU contracts for private stage-retirement A/B; no CUDA execution."""
import ast,contextlib,io,unittest
from pathlib import Path
BODY=Path(__file__).with_name('dl_g1_stage_ab_body.py').read_text(encoding='utf8')

def function(name,env=None):
    scope={} if env is None else env
    node=next(n for n in ast.parse(BODY).body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'stage_body','exec'),scope);return scope[name]

class StageAB(unittest.TestCase):
    def test_score_only_changed_loads(self):
        f=function('_st_summary',{'_ST_SEEDS':(1,2,3),'_ST_AFFECTED':(32,128,256)})
        def rows(times):return [(m,s,times(m)) for s in (1,2,3) for m in (1,8,16,32,128,256)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(f(rows(lambda m:[100,100,96,96])))
            self.assertFalse(f(rows(lambda m:[100,100,100,100] if m>=32 else [100,100,1,1])))
            self.assertFalse(f(rows(lambda m:[100,100,102,102])))
            self.assertFalse(f(rows(lambda m:[100,110,96,80])))
            self.assertFalse(f(rows(lambda m:[100,100,104,104] if m==32 else [100,100,80,80])))
    def test_tf32_restore_and_failclosed(self):
        class Back:allow_tf32=True
        class Cuda:matmul=Back()
        class Torch:cuda=Cuda()
        class Backend:cuda=Cuda()
        Torch.backends=Backend()
        def fail(w):raise ValueError('budget')
        f=function('prepare',{'torch':Torch,'_st_run':fail})
        with self.assertRaises(ValueError):f({})
        self.assertTrue(Torch.backends.cuda.matmul.allow_tf32)
    def test_observes_actual_route(self):
        # The observation wrapper must fail if the real ordinary route differs.
        env={'_ST_OBSERVE':False,'_ST_SEEN':[],'_st_defs':lambda a,m:('main',('expected',))}
        class X:shape=(128,4096)
        def kernel(d,a):env['_ST_SEEN'].append(('wrong',));return 1
        env['_ST_KERNEL']=kernel;f=function('_st_call',env)
        with self.assertRaises(AssertionError):f({'hidden_states':X()}, {})
        self.assertFalse(env['_ST_OBSERVE'])
        self.assertNotIn('OKC_SC6W=0',BODY)
        self.assertIn("arm['w'] is state['w']",BODY)

if __name__=='__main__':unittest.main()
