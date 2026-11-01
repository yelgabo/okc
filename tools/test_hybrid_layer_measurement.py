"""Owning archived receipt and rejection gates; no simulated performance claim."""
import copy
import json
from pathlib import Path
import unittest
from hybrid_layer_measurement import compare, read, qualification

ROOT=Path(__file__).resolve().parents[1]
JOB='607eb52bdaf24160'


class Reader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt=json.loads((ROOT/f'muse_decode_layer/profiles/scored.{JOB}.json').read_text())

    def test_archived_owner_full_score_and_raw_cells(self):
        summary=compare(self.receipt,self.receipt,JOB,JOB)
        self.assertEqual(len(summary['rows']),12)
        self.assertEqual(len(summary['weighted_loads']),6)
        self.assertEqual(summary['score_ratio_to_archived_control'],1)
        self.assertFalse(summary['paired']);self.assertFalse(summary['release_decision'])
        self.assertTrue(all(r['adjusted_change_percent']==r['raw_change_percent']==0 for r in summary['rows']))

    def test_wrong_owner_coverage_values_and_score_reject(self):
        for field,value in (('id','other'),('board','muse_moe_w4a16'),('mode','check')):
            d=copy.deepcopy(self.receipt);d[field]=value
            with self.assertRaises(AssertionError):read(d,JOB)
        for change in ('missing','nan','weight','flag','score','test'):
            d=copy.deepcopy(self.receipt);result=d['result']
            row=next(r for r in result['records'] if r.get('event')=='benchmark')
            if change=='missing':result['records'].remove(row)
            if change=='nan':row['median_raw_ns']=float('nan')
            if change=='weight':row['case']['weight']=16
            if change=='flag':row['flags']=['flagged']
            if change=='score':result['speedup']*=1.01
            if change=='test':next(r for r in result['records'] if r.get('event')=='test')['status']='fail'
            with self.assertRaises(AssertionError,msg=change):read(d,JOB)

    def test_startup_missing_fallback_and_local_memory_reject(self):
        log='HYBRID_QUAL_DONE selected=4 fallback=0 checks=48 captures=8 fresh=24 empty_checks=40 empty_captures=8 restored=32\n'
        for name in ('moe51','one8_mode3','one8_mode51'):
            log+=f'HYBRID_QUAL_RESOURCE {name} [96, 0, 0, 228736, 576, 576, 1, 132, 1, 132]\n'
        self.assertEqual(qualification(log)['startup_fields']['selected'],'4')
        for bad in ('',log+log,log.replace('selected=4','selected=0'),log.replace('[96, 0,','[96, 8,'),log.replace('empty_checks=40','empty_checks=0')):
            with self.assertRaises(AssertionError):qualification(bad)


if __name__=='__main__':unittest.main()
