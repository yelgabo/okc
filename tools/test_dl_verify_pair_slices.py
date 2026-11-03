import ast
import builtins
from pathlib import Path
from types import SimpleNamespace
import symtable
import unittest

from dl_verify_pair_slices import build, literals
from dl_g_verify_slice import build as old_g
from dl_owned_slice import build as old_o
from dl_owned_main_slice import specialize

ROOT = Path(__file__).resolve().parents[1]


def function(source, name):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)


class VerifySlices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arms, cls.manifest = build(ROOT)

    def test_native_literals_and_global_bindings(self):
        original_g = old_g((ROOT / "muse_decode_layer/cand/g1b4_stage_release.py").read_text(encoding="utf8"))[0]
        original_o = specialize(old_o((ROOT / "muse_decode_layer/cand/layer_lt_boundary.py").read_text(encoding="utf8"))[0])[0]
        for name, original in (("G", original_g), ("owned", original_o)):
            before, after = literals(original), literals(self.arms[name])
            self.assertEqual(before, {k:after[k] for k in before})
            table = symtable.symtable(self.arms[name], name, "exec")
            available = set(dir(builtins)) | {s.get_name() for s in table.get_symbols()
                                             if s.is_assigned() or s.is_imported()}
            def globals_used(scope):
                return {s.get_name() for s in scope.get_symbols() if s.is_global()}.union(
                    *(globals_used(child) for child in scope.get_children()))
            used = set().union(*(globals_used(child) for child in table.get_children()))
            self.assertEqual(used - available, set(), name)
        self.assertLess(sum(m["bytes"] for m in self.manifest["arms"].values()) + 60000, 1000000)

    def test_owned_recorded_128_policy(self):
        class X:
            dtype="bf16"
            def __init__(self,shape):self.shape=shape
            def contiguous(self):return self
            def is_contiguous(self):return True
            def data_ptr(self):return 256
        events=[]
        def log(name):
            def call(*args):events.append(name);return args[0]
            return call
        env={"torch":SimpleNamespace(bfloat16="bf16"),"_gemm":log("torch")}
        exec(compile(ast.Module([function(self.arms["owned"],"_slice_proj")],[]),"owned-verify-proj","exec"),env)
        state={"verify_qkv128":log("WD128"),"dense":{"qkv_gate":object(),"o":object()},
            "dlplt":{("qkv_gate",256):log("Lt256")}}
        for m in (128,256):
            env["_slice_proj"](X((m,8192)),"qkv_gate",state)
            env["_slice_proj"](X((m,1024)),"o",state)
        self.assertEqual(events,["WD128","torch","Lt256","torch"])
        node=function(self.arms["owned"],"kernel_fn")
        geo=next(n.args[-1] for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="_abnd")
        self.assertEqual(eval(compile(ast.Expression(geo),"owned-boundary-policy","eval"),{"batch":16}),"sglang")
        self.assertEqual(eval(compile(ast.Expression(geo),"owned-boundary-policy","eval"),{"batch":32}),"c1x512")
        prep=function(self.arms["owned"],"prepare")
        self.assertIn('"attn_boundary":FusedPostNormResidualPreNorm(post,ar,pre)',ast.get_source_segment(self.arms["owned"],prep))

    def test_graft_actual_four_cell_entry(self):
        import torch
        for batch in (16, 32):
            for window in (0, 2048):
                with self.subTest(batch=batch, window=window):
                    m = batch * 8
                    events = []
                    class Launch:
                        def __init__(self, name): self.name = name
                        def __getitem__(self, grid):
                            def call(*args, **kwargs): events.append((self.name, grid, args, kwargs))
                            return call
                    def project(x, plan):
                        events.append((plan,))
                        return torch.empty(m, {"qkv":2304,"o":8192,"pr":4608,"un":8192}[plan],
                            dtype=torch.float32 if plan=="pr" else torch.bfloat16)
                    def attention(*args):
                        events.append(("attention",))
                        return torch.empty(m,1024,dtype=torch.bfloat16)
                    cfg = dict(batch=batch,q_len=8,tokens=m,window=window,page_size=1,
                        hidden=8192,num_q_heads=16,num_kv_heads=2,head_dim=64,top_k=8,
                        n_experts=256,local_experts=32,kv_scale=1.)
                    data = dict(config=cfg,attn_input=torch.empty(m,8192,dtype=torch.bfloat16),
                        stream=torch.empty(m,8192),page_table=SimpleNamespace(shape=(batch,86008)),
                        local_start=torch.tensor([0],dtype=torch.int32))
                    state = dict(graft_lt={(k,m):k for k in ("qkv","o","pr","un")},
                        graft={k:torch.empty(8192) for k in ("pre1","pre2","post2","alpha1","alpha2","beta1","beta2")},
                        bias=torch.empty(256),g_preexp=torch.empty(4096,dtype=torch.bfloat16),
                        native=SimpleNamespace(okc_run=lambda *a:events.append(("native",a))),
                        ma_w=[],ma_scratch=[],nsm=132)
                    env = dict(torch=torch,_decode_attention7=attention,_graft_lt_run=project,
                        _graft_row_boundary=Launch("boundary"),_graft_route=Launch("route"),_GRAFT_ROUTE_EARLY=False)
                    exec(compile(ast.Module([function(self.arms["G"],"kernel_fn")],[]),"G-verify-entry","exec"),env)
                    out = env["kernel_fn"](data,state)
                    self.assertEqual([e[0] for e in events],
                        ["qkv","attention","o","boundary","pr","route","native","un","boundary"])
                    boundaries = [e for e in events if e[0]=="boundary"]
                    self.assertEqual([e[3]["num_warps"] for e in boundaries],[32 if m==128 else 16]*2)
                    self.assertEqual([e[2][-1] for e in boundaries],[32 if m==128 else 16]*2)
                    self.assertEqual([v.dtype for v in out],[torch.bfloat16,torch.float32,torch.int32])
                    for key, value in (("tokens",m+1),("window",1024),("kv_scale",.5),("batch",8)):
                        old=cfg[key];cfg[key]=value
                        with self.assertRaises(RuntimeError):env["kernel_fn"](data,state)
                        cfg[key]=old

    def test_owned_three_lt_plans_and_closure_ownership(self):
        class Tensor:
            def __init__(self, shape): self.shape=shape;self.device="cpu"
            def t(self): return Tensor(self.shape[::-1])
            def contiguous(self): return self
            def numel(self): return self.shape[0]
        class Lt:
            def __init__(self): self.plans=[];self.runs=[]
            def lt_plan_new(self,*args):self.plans.append(args);return len(self.plans)-1
            def lt_add_config(self,plan,cfg):return plan+20
            def lt_run(self,*args):self.runs.append(args);return 0
        lt=Lt()
        torch=SimpleNamespace(uint8="u8",bfloat16="bf16",zeros=lambda n,**k:Tensor((n,)),
            empty=lambda shape,**k:Tensor(shape))
        env={"torch":torch,"_dlplt_build":lambda:lt}
        exec(compile(ast.Module([function(self.arms["owned"],"_slice_lt")],[]),"owned-verify-Lt","exec"),env)
        state={"dense":{"qkv_gate":Tensor((2304,8192)),"unpooler":Tensor((8192,4096))}}
        env["_slice_lt"](state)
        self.assertEqual([p[:4] for p in lt.plans],[(128,8192,4096,3),(256,8192,4096,1),(256,2304,8192,0)])
        ws=[]
        for (layer,m),run in state["dlplt"].items():
            out=run(Tensor((m,4096 if layer=="unpooler" else 8192)))
            self.assertEqual(out.shape,(m,8192 if layer=="unpooler" else 2304))
            names=run.__code__.co_varnames[1:run.__code__.co_argcount]
            defaults=dict(zip(names,run.__defaults__))
            ws.append(defaults["ws"])
            self.assertEqual(defaults["m"],m)
        self.assertEqual(len({id(w) for w in ws}),3)
        self.assertEqual([r[2].shape for r in lt.runs],[(4096,8192),(4096,8192),(2304,8192)])


if __name__ == "__main__": unittest.main()
