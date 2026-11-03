"""Reject incomplete observer capsules and mismatched source identities."""
import base64
import copy
import json
import random
import unittest
import zlib

from expert_top100_moe_harness_contract import decision
from moe_phase_analysis import observer_decision,phase_trace
from moe_phase_harness import build
from moe_phase_read import capsule
from test_moe_phase import synthetic_trace


def encode(report):
    return 'MOE_PHASE_REPORT '+base64.b64encode(zlib.compress(json.dumps(report,separators=(',',':')).encode(),9)).decode()


class Reader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _,cls.manifest=build()
        rows=[[seed,t,dist,[100.]*4,[8,1,1] if t==1 else [t,1,t],[1.]*4,[0.,0.]]
              for seed in (1102031,1102037,1102049) for t in (1,8,16,32,128,256) for dist in ('normal','skewed')]
        summary,raw=phase_trace(synthetic_trace(),132)
        phases=[[*r[:3],[summary,summary]] for r in rows if r[1]>=32]
        sampling=[[*r[:3],[0.]*5] for r in rows if r[1]>=32]
        repeated=[[*r[:3],[[[v,v,v] for v in (70.,1.,19.)] for _ in range(2)]] for r in rows if r[1]>=32]
        cls.report=dict(runtime_id=cls.manifest['runtime_id'],official_arm='control0',placement_arms=[False,False,True,True],
                        cleanup_restored=True,shared_tensor_storage=[],owners_bytes=[100]*4,
                        counters=dict(observations=378,restorations=54,captures=216,timed_replays=4608,trace_replays=180,worst=[0.,0.]),
                        prepare_s=1.,rows=rows,component_pass=decision(rows)['component_pass'],phases=phases,
                        raw_trace=raw,sampling=sampling,repeated=repeated,observer_valid=observer_decision(rows,sampling)['valid'])

    def test_complete_and_injected_failures(self):
        result,_=capsule(encode(self.report),self.manifest)
        self.assertTrue(result['observer_valid'])
        for fault in ('source','official','rows','traces','sampling','repeated','raw'):
            report=copy.deepcopy(self.report)
            if fault=='source':report['runtime_id']='wrong'
            elif fault=='official':report['official_arm']='candidate2'
            elif fault=='rows':report['rows'].pop()
            elif fault=='traces':report['counters']['trace_replays']-=1
            elif fault=='sampling':report['sampling'].pop()
            elif fault=='repeated':report['repeated'][0][3][0][0]=[2.,1.,3.]
            else:report['raw_trace']['critical_ctas'][0][14]+=1000
            with self.subTest(fault=fault):
                with self.assertRaises((ValueError,RuntimeError)):capsule(encode(report),self.manifest)

    def test_variable_evidence_fits_log_tail(self):
        report=copy.deepcopy(self.report);rng=random.Random(91)
        def varied(value):
            if isinstance(value,list):return [varied(x) for x in value]
            if isinstance(value,dict):return {k:varied(x) for k,x in value.items()}
            return round(rng.uniform(1,300),4) if isinstance(value,float) else value
        report['rows']=varied(report['rows'])
        for key in ('phases','sampling','repeated'):
            def compact(value):
                if isinstance(value,list):return [compact(x) for x in value]
                if isinstance(value,dict):return {k:compact(x) for k,x in value.items()}
                return round(value,2) if isinstance(value,float) else value
            report[key]=compact(varied(report[key]))
        self.assertLess(len(encode(report)),12000)


if __name__=='__main__':unittest.main()
