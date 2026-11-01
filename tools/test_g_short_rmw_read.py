import base64,copy,hashlib,json, tempfile,unittest,zlib
from pathlib import Path
from g_short_rmw_read import *

def report():
    pins=MANIFEST['capsule_pins'];r=dict(schema=1,part='all',seeds=list(SEEDS),candidate_pin=pins['_WL_PIN'],g1b4_conformance=pins['_WL_CONFORMANCE'],device_sha=pins['_WL_DEVICE_SHA'],harness_core_sha=pins['_WL_CORE_SHA'],math_checks=144,fresh_final_checks=144,empty_native_checks=32,fresh_replays_per_bank=8,judged_stage_on=True,judged_short_relaxed=False,prepare_s=200.,weight_signature='a'*64,owners_bytes=[100]*4)
    r['resources']={k:[[147,0,0,1000,1,288]*2 for _ in range(4)] for k in ('tiny','tiny2','mid','b4','main')}
    r['module_flags']={str(i)+':'+k: (['-DOKC_SHORT_FINAL_RELAXED=1'] if i==2 and k in ('tiny','tiny2') else [])+(['-DOKC_B4FIN=1'] if k=='b4' else []) for i in (0,2) for k in r['resources']}
    r['rows']=[]
    for t,c,s in sorted(CELLS):
        key='tiny' if t==1 else 'tiny2' if t==8 else 'mid' if t==16 else 'b4' if t==32 else 'main'
        r['rows'].append(dict(tokens=t,context=c,seed=s,times_us=[100.,100.,95. if t in (1,8) else 100.,95. if t in (1,8) else 100.],local_start=0,reference_group_counts=[t*8]+[0]*7,actual_routes=[[[t,key,i>=2]] for i in range(4)]))
    r['decision']=decision(r['rows']);return r

def log(r):return 'SRM_REPORT '+base64.b64encode(zlib.compress(json.dumps(r).encode())).decode()+'\n'

def receipt():
    rec=[dict(event='test',status='pass') for _ in range(13)]+[dict(event='setup',status='pass',error=None,prepare_s=201.)]
    for l,c in sorted(OFFICIAL):
        case=dict(load=l,layer=c,weight=46 if c=='sliding' else 16)
        rec.extend([dict(event='benchmark',case=case,status='pass',flags=[],median_ns=1,median_raw_ns=2,baseline_ns=3,baseline_raw_ns=4),dict(event='profile',case=case,report='native')])
    return dict(id='abc123',board='muse_decode_layer',mode='check',status='done',content_hash='b'*64,payload_ref='muse_decode_layer/'+'b'*64+'.py',output=dict(log='/tmp/abc123.log'),result=dict(status='ok',passed=True,returncode=0,checks=dict(ok=True),flags=[],error=None,records=rec))

class Reader(unittest.TestCase):
    def test_valid(self):
        out=read(receipt(),'abc123',log(report()));self.assertTrue(out['complete'] and out['private_raw_gate'])
    def test_receipt_rejects(self):
        for field,val in [('id','other'),('board','other'),('mode','run'),('status','running')]:
            r=receipt();r[field]=val
            with self.assertRaises(AssertionError):read(r,'abc123',log(report()))
        for field,val in [('passed',False),('flags',['guard']),('error','final owner guard'),('returncode',1)]:
            r=receipt();r['result'][field]=val
            with self.assertRaises(AssertionError):read(r,'abc123',log(report()))
        r=receipt();r['result']['records'].pop()
        with self.assertRaises(AssertionError):read(r,'abc123',log(report()))
    def test_capsule_negatives(self):
        for field,val in [('math_checks',143),('fresh_final_checks',143),('empty_native_checks',31),('device_sha','c'*64),('prepare_s',295),('judged_short_relaxed',True)]:
            r=report();r[field]=val
            with self.assertRaises(AssertionError):capsule(log(r))
        mutations=[lambda r:r['rows'].pop(),lambda r:r['rows'].__setitem__(0,r['rows'][1]),lambda r:r['rows'][0]['times_us'].__setitem__(0,float('nan')),lambda r:r['resources']['tiny'][2].__setitem__(0,148),lambda r:r['resources']['tiny'][0].__setitem__(4,0),lambda r:r['module_flags']['0:tiny'].append('-DOKC_SHORT_FINAL_RELAXED=1'),lambda r:r['decision'].__setitem__('gate',False),lambda r:r['rows'][0]['actual_routes'].__setitem__(0,[])]
        for change in mutations:
            r=report();change(r)
            with self.assertRaises((AssertionError,ValueError)):capsule(log(r))
        for bad in ('',log(report())+log(report()),log(report())[:-20]):
            with self.assertRaises((AssertionError,ValueError,zlib.error)):capsule(bad)
    def test_loss_is_complete_not_promoted(self):
        r=report()
        for row in r['rows']:row['times_us']=[100.]*4
        r['decision']=decision(r['rows']);o=read(receipt(),'abc123',log(r));self.assertFalse(o['private_raw_gate']);self.assertTrue(o['complete'])
    def test_log_binding(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'abc123.log';p.write_bytes(log(report()).encode());r=receipt();r['output']['log']=str(p)
            self.assertEqual(owning_log(r,p)[0],log(report()))
            q=Path(d)/'wrong.log';q.write_bytes(p.read_bytes())
            with self.assertRaises(AssertionError):verify_source(q)
            with self.assertRaises(AssertionError):owning_log(r,q)
            att=dict(job='abc123',owning_log=str(p),sha256=hashlib.sha256(q.read_bytes()).hexdigest(),source_sha256=MANIFEST['source_sha256'])
            owning_log(r,q,att);att['job']='wrong'
            with self.assertRaises(AssertionError):owning_log(r,q,att)

if __name__=='__main__':unittest.main()
