"""Keep failed partial receipts separate from qualifying full comparisons."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
import zlib

import moe_pairlut_read as reader
from moe_pairlut_harness import build

ROOT=Path(__file__).resolve().parents[1]


class Partial(unittest.TestCase):
    def test_failed_partial_binds_source_and_never_qualifies(self):
        method=getattr(reader,'read_partial',None)
        self.assertTrue(callable(method),'failed owning capsule needs a partial reader')
        payload,manifest=build(); job='c985bbdb2980407a'
        receipt=json.loads((ROOT/'docs/steering/evidence/2026-11-02-moe-phase/status-wait.raw.json').read_text())
        receipt['id']=job;receipt['output']['log']='/synthetic/paired.'+job+'.log'
        partial=dict(runtime_id=manifest['runtime_id'],status='FAILED_PARTIAL_NOT_QUALIFIED',
                     error='RuntimeError:injected budget',stage='cleanup',elapsed_s=296.,
                     rows=[],counters=dict(observations=1,restorations=0,captures=4,timed_replays=0,worst=[0.,0.]),
                     loaded_builds=list(range(7)))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'profiles').mkdir()
            (folder/'paired.py').write_text(payload)
            (folder/'paired.manifest.json').write_text(json.dumps(manifest))
            (folder/'accepted.json').write_text(json.dumps(dict(job_id=job,source_sha256=manifest['source_sha256'])))
            def write(value):
                line='MOE_PAIRLUT_PARTIAL '+base64.b64encode(zlib.compress(json.dumps(value).encode())).decode()+'\n'
                receipt['result']['log_tail']=line
                (folder/'status-wait.raw.json').write_text(json.dumps(receipt))
                (folder/'profiles'/('paired.'+job+'.log')).write_text(line)
            write(partial);result=method(folder,job)
            self.assertIs(result['qualified'],False)
            self.assertEqual(result['retained_rows'],0)
            self.assertFalse(result['complete_timing_coverage'])
            for fault in ('runtime','success','wrong-job','excess-coverage','duplicate-row'):
                modified=copy.deepcopy(partial)
                if fault=='runtime':modified['runtime_id']='wrong'
                elif fault=='success':receipt['result']['passed']=True
                elif fault=='wrong-job':receipt['id']='wrong'
                elif fault=='excess-coverage':modified['counters']['captures']=217
                else:modified['rows']=[[1102031,1,'normal']]*2
                write(modified)
                with self.subTest(fault=fault),self.assertRaises((RuntimeError,ValueError)):
                    method(folder,job)
                receipt['result']['passed']=False;receipt['id']=job
            partial['rows']=[[seed,tokens,dist,[100.,100.,97.,97.] if tokens>=32 else [100.]*4,
                              [tokens,1,tokens],[100/97]*4 if tokens>=32 else [1.]*4,[0.,0.]]
                             for seed in (1102031,1102037,1102049) for tokens in (1,8,16,32,128,256)
                             for dist in ('normal','skewed')]
            partial['counters'].update(observations=378,restorations=54,captures=216,timed_replays=1152)
            write(partial);result=method(folder,job)
            self.assertTrue(result['complete_timing_coverage'])
            self.assertIs(result['qualified'],False)


if __name__=='__main__':unittest.main()
