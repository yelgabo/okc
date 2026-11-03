"""Check observer source selection, complete trace invariants and distortion gates."""
import copy
import unittest

from cuda_issue_groups import BASE,NATIVE_PIN,sha
from prod_size_compile_g import bindings
from expert_top100_pregamma import capture
from moe_phase import transform
from moe_phase_analysis import phase_trace,observer_decision,sampling_summary


def synthetic_trace(grid=132,jobs=8):
    result=[]
    for c in range(grid):
        entry=1000000+c*10
        result.extend([entry,entry+1000,entry+3000,*[entry+20000+i*10 for i in range(4)],
                       *[entry+50000+i*10 for i in range(4)],entry+51000,
                       1060000+c*2,1070000+c*2,entry+21000,entry+21100,
                       entry+19000,entry+22000,c,jobs,
                       jobs*512*c//grid,jobs*512*(c+1)//grid,
                       jobs*256*c//grid,jobs*256*(c+1)//grid])
    return result


class Phase(unittest.TestCase):
    def test_exact_selected_delta_and_producer_leaders(self):
        base=BASE.read_text();native=bindings(base)['MOE_CUDA_SRC']
        self.assertEqual(sha(native),NATIVE_PIN)
        revised,edits=transform(native);restored=revised
        for old,new,count in reversed(edits):
            self.assertEqual(restored.count(new),count);restored=restored.replace(new,old)
        self.assertEqual(restored,native)
        boundary='#include <torch/extension.h>'
        self.assertEqual(native[native.index(boundary):],revised[revised.index(boundary):])
        calls,routes=capture(base);after,other=capture(base.replace(native,revised))
        self.assertEqual(routes,other)
        for a,b in zip(calls,after):
            self.assertEqual(a['extra_cuda_cflags'],b['extra_cuda_cflags'])
            self.assertEqual(a['cuda_sources']!=b['cuda_sources'],'-DOKC_WGS=1'in a['extra_cuda_cflags'])
        for field in (14,15,16,17):self.assertIn('ph_single<'+str(field)+'>',revised)
        for name in ('wgs_seg','wgs_flush','wgs_epi_tile'):
            start=native.index('void '+name+'(');end=native.index('\n}',start)+2
            self.assertIn(native[start:end],revised)

    def test_boundary_protocol_and_stale_or_corrupt_traces(self):
        data=synthetic_trace();report,raw=phase_trace(data,132)
        self.assertEqual(report['grid'],132);self.assertEqual(report['unique_smids'],132)
        self.assertEqual(report['jobs'],8);self.assertLessEqual(len(raw['critical_ctas']),15)
        for change in ('stale','barrier','missing_producer','range'):
            bad=data[:]
            if change=='stale':bad[3]=bad[0]-1
            elif change=='barrier':bad[12]=bad[11]-1
            elif change=='missing_producer':bad[14]=0
            else:bad[21]+=1
            with self.assertRaises(ValueError):phase_trace(bad,132)

    def test_intrusion_is_not_treated_as_a_gain(self):
        rows=[[seed,t,dist,[100.,100.,100.4,100.4],None,None,None]
              for seed in (1102031,1102037,1102049) for t in (32,128,256)
              for dist in ('normal','skewed')]
        sampling=[[*r[:3],sampling_summary([[v]*32 for v in r[3]])] for r in rows]
        self.assertTrue(observer_decision(rows,sampling)['valid'])
        for value in (98.,102.):
            bad=copy.deepcopy(rows);bad[0][3][2:]=[value,value]
            self.assertFalse(observer_decision(bad,sampling)['valid'])
        noisy=copy.deepcopy(sampling);noisy[0][3][1:3]=[-2.,2.]
        self.assertFalse(observer_decision(rows,noisy)['valid'])
        with self.assertRaises(ValueError):observer_decision(rows[:-1],sampling)


if __name__=='__main__':unittest.main()
