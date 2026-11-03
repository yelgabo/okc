"""Validate synthetic owning associations and real target-row entropy with the strict reader."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from expert_top100_moe_harness_contract import decision
from moe_pairlut_contract import pairlut_decision
from moe_param_y_harness import build
from moe_param_y_read import read

ROOT=Path(__file__).resolve().parents[1]


class Reader(unittest.TestCase):
    def test_strict_source_record_coverage_and_target_entropy(self):
        source,manifest=build();payload=source.encode();job='e044121f822d47b1'
        receipt=json.loads((ROOT/'docs/steering/evidence/2026-11-02-moe-bqueue/status-wait.raw.json').read_text())
        rows=json.loads((ROOT/'docs/steering/evidence/2026-11-02-moe-phase/partial.json').read_text())['rows']
        gate=decision(rows)
        report=dict(runtime_id=manifest['runtime_id'],official_arm='candidate2',placement_arms=[False,False,True,True],
                    cleanup_restored=True,owners_bytes=[8000000000]*4,shared_tensor_storage=[],prepare_s=279.8,
                    rows=rows,counters=dict(observations=378,restorations=54,captures=216,timed_replays=1152,worst=[.001,.002]),
                    component_pass=gate['component_pass'],param_y_decision=pairlut_decision(rows,gate),loaded_builds=list(range(7)),table_words=[256,256,256,256],gamma_binding_checks=36,gamma_words=[4096]*4)
        log='EXPERT_MOE_CAPSULE '+json.dumps(report,separators=(',',':'))+'\n'
        self.assertLess(len(log),14000)
        receipt['output']['log']='/synthetic/paired.'+job+'.log';receipt['result']['log_tail']=log
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);(folder/'profiles').mkdir()
            (folder/'paired.py').write_bytes(payload)
            (folder/'paired.manifest.json').write_text(json.dumps(manifest))
            (folder/'accepted.json').write_text(json.dumps(dict(job_id=job,source_sha256=manifest['source_sha256'])))
            (folder/'status-wait.raw.json').write_text(json.dumps(receipt))
            logfile=folder/'profiles'/('paired.'+job+'.log');logfile.write_text(log)
            data=read(folder,job);self.assertFalse(data['target_qualified'])
            for fault in ('job','runtime','builds','table','rows','coverage','gamma'):
                modified=copy.deepcopy(report)
                if fault=='job':
                    with self.assertRaises(ValueError):read(folder,'wrong-job')
                    continue
                if fault=='runtime':modified['runtime_id']='wrong'
                elif fault=='builds':modified['loaded_builds'].pop()
                elif fault=='table':modified['table_words'][2]=16640
                elif fault=='rows':modified['rows'].pop()
                elif fault=='gamma':modified['gamma_words'][2]=32
                else:modified['counters']['captures']-=1
                logfile.write_text('EXPERT_MOE_CAPSULE '+json.dumps(modified,separators=(',',':'))+'\n')
                with self.subTest(fault=fault):
                    with self.assertRaises((ValueError,RuntimeError)):read(folder,job)


if __name__=='__main__':unittest.main()
