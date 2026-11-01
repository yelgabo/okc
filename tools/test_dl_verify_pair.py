import ast
import copy
import json
import base64
from pathlib import Path
import random
import unittest
import zlib

from dl_verify_pair import build
from dl_selected_pair import build as selected_build

ROOT=Path(__file__).resolve().parents[1]


class VerifyPair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.source,cls.manifest=build(ROOT)

    def env(self,names):
        env={}
        for n in ast.parse(self.source).body:
            if isinstance(n,ast.Assign) and any(getattr(x,"id","") in names for x in n.targets):
                exec(compile(ast.Module([n],[]),"verify-constants","exec"),env)
            elif isinstance(n,ast.FunctionDef) and n.name in names:
                exec(compile(ast.Module([n],[]),"verify-functions","exec"),env)
        return env

    def test_package_and_preserved_selected_payload(self):
        self.assertLess(self.manifest["bytes"],1000000)
        self.assertFalse(self.manifest["whole_six_load_score_claim"])
        self.assertFalse(self.manifest["H200_qualified"])
        _,old=selected_build(ROOT)
        self.assertEqual(old["bytes"],850769)
        self.assertEqual(old["sha256"],"305d8e2804d8d10c3fdda902d659c25177ce7ad31d7575dda803c318830e8dc7")
        self.assertIn("VERIFY_NATURAL_ROUTING_DRIFT",self.source)
        self.assertIn("VERIFY_FRESH_ROUTING_DRIFT",self.source)
        self.assertIn("VERIFY_WD_DIAGNOSTIC",self.source)
        self.assertIn("graph._verify_inputs=inputs",self.source)

    def test_weighted_restricted_report_and_lossless_transport(self):
        names={"_verify_report","_VERIFY_SEEDS","_VERIFY_CASES","_PAIR_PINS","_VERIFY_G_CELLS","_VERIFY_G_COSTS"}
        env=self.env(names)
        rng=random.Random(491071)
        rows=[dict(batch=b,layer=l,seed=s,times_us=[100.]*4,
                   samples_us=[[100.+rng.random() for _ in range(24)] for _ in range(4)])
              for b,l in env["_VERIFY_CASES"] for s in env["_VERIFY_SEEDS"]]
        report=env["_verify_report"](copy.deepcopy(rows),96,1.)
        self.assertTrue(all(r["two_load_cross_ratios"]==[1.]*4 for r in report["weighted_two_load_comparisons"]))
        self.assertTrue(all(r["conditional_cell_hybrid_factors"]==[1.]*4 for r in report["rows"]))
        self.assertFalse(report["whole_six_load_score_claim"])
        encoded=base64.b64encode(zlib.compress(json.dumps(report,separators=(",",":")).encode(),9))
        self.assertLess(len(encoded),11000)
        self.assertEqual(json.loads(zlib.decompress(base64.b64decode(encoded))),report)
        changed=copy.deepcopy(rows)
        changed[0]["times_us"]=[90.,90.,100.,100.]
        report=env["_verify_report"](changed,96,1.)
        w=46 if changed[0]["layer"]=="sliding" else 16
        b=changed[0]["batch"];l=changed[0]["layer"]
        base=env["_VERIFY_G_COSTS"][b];cell=env["_VERIFY_G_CELLS"][b,l]
        self.assertAlmostEqual(report["rows"][0]["conditional_cell_hybrid_factors"][0],
                               (base/(base-w*cell*.1))**(1/6))
        for invalid,count in ((rows[:-1],96),(rows,95)):
            with self.assertRaises(AssertionError):env["_verify_report"](invalid,count,1.)

    def test_lazy_module_and_lt_aliases_are_rejected(self):
        import torch
        env={"torch":torch}
        exec((ROOT/"tools/dl_g1_layer_owners.py").read_text(encoding="utf8"),env)
        node=next(n for n in ast.parse(self.source).body if isinstance(n,ast.FunctionDef) and n.name=="_pair_owned_owners")
        exec(compile(ast.Module([node],[]),"verify-owners","exec"),env)
        class Lazy(torch.nn.Module):
            def __init__(self):super().__init__();self.register_buffer("temporary",torch.empty(32))
        def owner():
            def runner():
                workspace=torch.empty(32,dtype=torch.uint8)
                def run(x,ws=workspace):return x
                return run
            return {"dlplt":{key:runner() for key in (("qkv_gate",256),("unpooler",128),("unpooler",256))},"module":Lazy()}
        first,second=owner(),owner()
        env["_pair_owned_owners"](first,second,{})
        second["module"].temporary=first["module"].temporary
        with self.assertRaisesRegex(AssertionError,"VERIFY_OWNED_MUTABLE_ALIAS"):
            env["_pair_owned_owners"](first,second,{})
        with self.assertRaisesRegex(AssertionError,"VERIFY_OWNED_LT_ALIAS"):
            env["_pair_owned_owners"](first,copy.deepcopy(first),{})


if __name__=="__main__":unittest.main()
