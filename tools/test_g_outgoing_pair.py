"""Real private entry under CPU CUDA/native stand-ins; not runtime qualification."""
import ast
import base64
import contextlib
import copy
import inspect
import io
import json
import math
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import zlib

from g_outgoing_pair import build,MARKER
import g_outgoing_pair_read as reader

ROOT=Path(__file__).resolve().parents[1]
SOURCE,MANIFEST=build(ROOT)

def rows():
    return [dict(seed=seed,tokens=tokens,context=context,times_us=[100.,100.,100.,95. if tokens in (1,8) else 100.]) for seed in (993811,993817,993821) for tokens in (1,8,16,32,128,256) for context in ("sliding","full")]

def graph_rows(arm):
    return [[2,0,1,1],[0,1,10],[0,2,200 if arm>=2 else 100],[0,3,30],[0,4,40],[1,3,1,0,0,0],[1,1,2,1 if arm==3 else 0,0,1 if arm==3 else 0],[1,2,4,0,0,0]]

class ScoreAndEdges(unittest.TestCase):
    def test_net_mapping_and_non_cancelling_null(self):
        result=reader.decision(rows());self.assertTrue(result["gate"])
        self.assertAlmostEqual(result["banks"][0]["net_log_gains"][0],math.log(100/95)/3)
        altered=rows()
        for row in altered:
            if row["tokens"] in (1,8):row["times_us"][2:]=[120.,110.]
        self.assertFalse(reader.decision(altered)["gate"])
        altered=rows()
        for row in altered:row["times_us"][1]=100*(1.1 if row["tokens"] in (1,16,128) else 1/1.1)
        self.assertAlmostEqual(reader.decision(altered)["banks"][0]["duplicate_log_null"],math.log(1.1))

    def test_missing_duplicate_nonfinite_rows(self):
        for altered in (rows()[:-1],rows()+rows()[:1]):
            with self.assertRaises(AssertionError):reader.decision(altered)
        altered=rows();altered[0]["times_us"][3]=float("nan")
        with self.assertRaises(AssertionError):reader.decision(altered)

    def test_graph_identity_and_edge_data_negatives(self):
        proof=reader.edge_gate(graph_rows(3),[10,11],200,True)
        self.assertTrue(proof["edge_data_nonnull"])
        for mutate in (lambda value:value.__setitem__(0,[2,0,1,0]),lambda value:value[2].__setitem__(2,999),lambda value:value[6].__setitem__(5,0),lambda value:value[5].__setitem__(5,1),lambda value:value[7].__setitem__(3,1)):
            altered=graph_rows(3);mutate(altered)
            with self.assertRaises(AssertionError):reader.edge_gate(altered,[10,11],200,True)
        self.assertEqual(reader.edge_gate(graph_rows(0),[10,11],100,False)["programmatic_edges"],0)

    def test_frozen_source_and_unchanged_instrument_paths(self):
        self.assertEqual((SOURCE,MANIFEST),build(ROOT));self.assertLess(MANIFEST["bytes"],1_000_000)
        from g_common_rmw_pair import build as prior,function
        old,_=prior(ROOT)
        for name in ("_wl_check","_wl_copy","_wl_check_owners","_srp_occupied_empty","_wl_empty","_srp_back"):
            self.assertEqual(ast.dump(ast.parse(function(old,name))),ast.dump(ast.parse(function(SOURCE,name))))
        self.assertIn("_os_install(arm,index==3)",SOURCE);self.assertIn("_wl_clone(state,readonly,False)",SOURCE)
        self.assertNotIn("i>=2",SOURCE.split(MARKER)[1]);self.assertIn("cudaGraphGetEdges(graph,from.data(),to.data(),edges.data(),&count)",SOURCE)

class RealEntry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch,prior,old
        import torch
        import test_g_common_rmw_pair as prior
        old=prior.old

    def fixture(self):
        text=inspect.getsource(old.fixture)
        text=text.replace("suffix=SOURCE[SOURCE.index('def _wl_storage('):]","suffix=SOURCE[SOURCE.index('def _gqb_signature('):]")
        text=text.replace("ns.update(prepare=prep,","ns.update(load_inline=lambda *a,**kw:None,bfp__sk2_src=lambda *a:None,_GUARDED_IDS=set(),_ab_ref_back=lambda f,w,ids:(f['ref_h'],f['ref_s']),_att7=lambda:None,prepare=prep,")
        text=text.replace("d['config'].update(page_size=1)","d['config'].update(page_size=1,num_kv_heads=2,window=2048 if c=='sliding' else 0)")
        text=text.replace("return s","s['bfp']['sk2']['slice8_r192']=(0,torch.ones(1),torch.ones(1),3,0,1024)\n        return s",1)
        text=text.replace("out=ns['_wl_run'](d,s)","out=ns['_wl_run'](d,s)\n        out=ns['_wl_run'](d,s)\n        out=ns['_wl_run'](d,s)")
        environment=dict(vars(old));environment["SOURCE"]=SOURCE;exec(text,environment)
        ns,stats,production=environment["fixture"]()
        audit=prior.Attention()
        ns["_att7"]=lambda:audit;ns["_WL_ATTN"]=lambda data,*args:audit.run(data)
        ns["_wl_weight_signature"]=lambda weights:"a"*64;ns["math"]=math
        class Query:
            def __init__(self,pointer):self.pointer=pointer
            def seam_unpool_resources(self,pmax):return [self.pointer,147,0,0,118240+pmax*8,1024,1,416]
            def seam_graph_edges(self,address):return graph_rows(address-1000)
        original=Query(100);adapter=Query(200)
        ns["bfp__get_ext"]=lambda:original
        def install(state,enabled):
            state["_os_unpool_bfp"]=dict(state["bfp"]);state["_os_unpool_bfp"]["_os_launch"]=(adapter,1073741826 if enabled else 0);return state
        ns["_os_install"]=install
        capture=ns["_wl_capture"]
        def captured(data,state):
            graph,out=capture(data,state);graph.raw_cuda_graph=lambda:1000+state["_os_placement"];ns["_os_graph"](graph,data,state);return graph,out
        ns["_wl_capture"]=captured
        return ns,production,audit

    def call(self,ns):
        log=io.StringIO();fill=torch.Tensor.fill_
        def recorded(tensor,value):
            if old.G.active is not None:old.G.active.ops.append(lambda:fill(tensor,value))
            return fill(tensor,value)
        resource=[96,0,0,231168,288,1024,1,132,1,132]*2
        with patch.object(torch.Tensor,"fill_",recorded),patch.object(old.Native,"okc_run",prior.native_run),patch.object(old.Native,"wl_resources",return_value=[96,0,0,231168,1,288]*2),patch.object(old.Native,"qual_native_resources",return_value=resource,create=True),patch.object(old.Native,"seam_native_identity",return_value=[10,11],create=True),contextlib.redirect_stdout(log),torch.inference_mode():
            result=ns["prepare"](old.WEIGHTS())
        text=log.getvalue();return result,text

    def test_full_entry_and_owning_capsule(self):
        ns,production,audit=self.fixture();saved=tuple(ns[key] for key in ("_ma_ext","_fork_native_moe","_decode_attention7","_ab_ref_back"))
        result,log=self.call(ns);self.assertIs(result,production[0]);self.assertNotIn("_os_unpool_bfp",result)
        self.assertEqual(saved,tuple(ns[key] for key in ("_ma_ext","_fork_native_moe","_decode_attention7","_ab_ref_back")));self.assertFalse(audit.active)
        reader.MANIFEST=MANIFEST;report,decision=reader.capsule(log);self.assertEqual(len(report["graph_edges"]),48);self.assertFalse(decision["gate"])
        from test_g_short_rmw_read import receipt
        self.assertTrue(reader.read(receipt(),"abc123",log,MANIFEST)["complete"])
        for key,value in (("id","wrong"),("board","other"),("mode","submit"),("status","running")):
            changed=receipt();changed[key]=value
            with self.assertRaises(AssertionError):reader.read(changed,"abc123",log,MANIFEST)
        with self.assertRaises(AssertionError):reader.capsule(log+log)
        for mutation in (lambda value:value["graph_edges"][0][4].__setitem__("edge_data_nonnull",False),lambda value:value["primary_identities"].__setitem__("tiny",[999,998]),lambda value:value["graph_edges"].pop(),lambda value:value["secondary_resources"]["1"][2].__setitem__(2,100),lambda value:value.__setitem__("device_sha","bad")):
            changed=copy.deepcopy(report);mutation(changed)
            text="OUTGOING_REPORT "+base64.b64encode(zlib.compress(json.dumps(changed).encode(),9)).decode()+"\n"
            with self.assertRaises(AssertionError):reader.capsule(text)
        report["decision"]["gate"]=True
        tampered="OUTGOING_REPORT "+base64.b64encode(zlib.compress(json.dumps(report).encode(),9)).decode()+"\n"
        with self.assertRaisesRegex(AssertionError,"printed gate"):reader.capsule(tampered)

    def test_graph_failure_restores_without_success(self):
        ns,_,audit=self.fixture();saved=ns["_fork_native_moe"];ns["_os_graph"]=lambda *args:(_ for _ in ()).throw(RuntimeError("edge unsupported"))
        with self.assertRaisesRegex(RuntimeError,"edge unsupported"):self.call(ns)
        self.assertIs(ns["_fork_native_moe"],saved);self.assertFalse(audit.active);self.assertIsNone(ns["_SRP_CACHE"])

if __name__=="__main__":unittest.main()
