"""Independent synthetic CPU checks of the published Tiny observer format."""
import copy
import json
import random
import unittest

from gopt_tiny_trace_contract import audit_native_buffer, audit_exported_case, cta_plan, encode, observer_costs, source_map
from moe_tiny_timeline import trace


def fixture(tokens=8,empty=False,entropy=False):
    ids=[[(32+i if empty else (token*8+i)%32) for i in range(8)] for token in range(tokens)]
    values=[0]*(136*1184)
    rng=random.Random(8827)
    for work in cta_plan(ids):
        r=[0]*1184
        entry=1793690000000000000+work['cta']*32
        r[0]=entry;r[1]=entry+32;r[2]=entry+64
        r[17:23]=*work['a'],*work['b'],2 if tokens==1 else 1,1
        t=r[2]+32;helper_end=t;a_count=work['a'][1]-work['a'][0]
        r[3]=r[2];r[4]=r[3]+32
        for q in range(work['units']):
            if q==a_count:
                r[3]=t+32;r[4]=t+64;t=r[4]+32
            t+=rng.randrange(1,10)*32 if entropy else 32
            begin=t+32;ready=t+64;end=t+96
            epi=end+(rng.randrange(1,10)*32 if entropy else 32) if q<work['helper_units'] else 0
            first=q>=a_count and (q==a_count or (work['b'][0]+q-a_count)%256==0)
            dep0,dep1=(t-16,t) if first else (0,0)
            r[32+q*6:38+q*6]=dep0,dep1,begin,ready,end,epi
            helper_end=max(helper_end,epi)
            t=max(end,epi)+32
        r[5]=t+32;r[16]=max(helper_end,r[1])+32
        r[6]=max(r[5],r[16])+32;r[7]=r[6]+32;r[8]=r[7]+32
        values[work['cta']*1184:(work['cta']+1)*1184]=r
    return ids,values


class Contract(unittest.TestCase):
    def test_selected_flags_and_native_anchors(self):
        m=source_map()
        self.assertEqual([m['builds'][str(t)]['tail_warps'] for t in (1,8)],[2,1])
        self.assertEqual(len(m['anchors']),13)

    def test_occupied_and_empty_buffers_and_actual_export(self):
        for t in (1,8):
            for empty in (False,True):
                ids,values=fixture(t,empty)
                result=audit_native_buffer(t,ids,values)
                self.assertTrue(result['schema_valid'])
                self.assertFalse(result['phase_gain_qualified'])
                self.assertEqual(result['units'],0 if empty else (8 if t==1 else 32)*768)

    def test_native_range_tail_and_dependency_corruption(self):
        changes=[lambda x:x.__setitem__(17,99),lambda x:x.__setitem__(21,2),
                 lambda x:x.__setitem__(32+3*6+2,0),lambda x:x.__setitem__(16,0)]
        ids,values=fixture(8)
        plan=cta_plan(ids)[0]
        q=plan['a'][1]-plan['a'][0]
        changes+=[lambda x:x.__setitem__(slice(32+q*6,34+q*6),[0,0])]
        for change in changes:
            raw=values.copy();change(raw)
            with self.assertRaises(ValueError):audit_native_buffer(8,ids,raw)

    def test_exported_missing_witness_and_unit_corruption(self):
        ids,values=fixture()
        exported=trace(values,132)
        for change in (lambda x:x['critical_units'].pop(),
                       lambda x:x['critical_units'][0][1].pop(),
                       lambda x:x['ctas'][0].__setitem__(9,0),
                       lambda x:x['critical_units'][0][1][-1].__setitem__(3,0)):
            data=copy.deepcopy(exported);change(data)
            with self.assertRaises(ValueError):audit_exported_case(8,data)

    def test_router_duplicate_and_wrong_token_size(self):
        with self.assertRaises(ValueError):cta_plan([[0]*8])
        with self.assertRaises(ValueError):cta_plan([[i for i in range(8)]]*16)

    def test_frozen_owner_cost_and_duplicate_caps(self):
        rows=[[1102031,t,'normal',[30.,30.1,30.2,30.3]] for t in (1,8,16,32,128,256)]
        self.assertTrue(all(r[-1] for r in observer_costs(rows)))
        rows[0][3]=[30.,30.,34.,34.]
        self.assertFalse(observer_costs(rows)[0][-1])
        rows[0][3]=[30.,31.,30.,30.]
        self.assertFalse(observer_costs(rows)[0][-1])
        rows[0][3]=[30.,30.,float('nan'),30.]
        with self.assertRaises(ValueError):observer_costs(rows)

    def test_lossless_budget_fails_without_deleting_raw_records(self):
        ids,values=fixture(1,True)
        self.assertLess(len(encode(trace(values,132))),12000)
        ids,values=fixture(8,entropy=True)
        before=values.copy()
        with self.assertRaisesRegex(ValueError,'no records may be silently dropped'):encode(dict(raw=values))
        self.assertEqual(before,values)
        audit_native_buffer(8,ids,values)


if __name__=='__main__':unittest.main()
