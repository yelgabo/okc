"""Synthetic receipt validation tests; no kernel execution or measured claims."""
import copy,json,os,struct,unittest
from pathlib import Path
from proj_sk_xor_codec import encode_report
from proj_sk_xor_receipt import (CHECKS,TARGET_NAMES,ANCHOR_IDS,FREEVARS,SHAPES,MODULES,
    SOURCE_PACK,analytical_layout,validate_receipt,ReceiptError,policy)
ROOT=Path(os.environ.get('SK_XOR_EVIDENCE_ROOT',Path(__file__).resolve().parents[1]))
EVIDENCE=ROOT/'docs/steering/evidence'
MANIFEST=json.loads((EVIDENCE/'2026-11-01-projection-sk-xor-private/harness-manifest.json').read_text(encoding='utf8'))
PROTOTYPE=json.loads((EVIDENCE/'2026-11-01-projection-sparse-xor/prototype-manifest.json').read_text(encoding='utf8'))
JOB='0123456789abcdef'


def event_us(ms):return 1000*struct.unpack('<f',struct.pack('<f',ms))[0]

def seal(receipt,report):
    prefix=json.dumps(dict(pins=report['pins'],checks=report['checks'],rows=90,rounds=24),separators=(',',':'))
    footer='SXDONE coverage=3x30 target=10 common=20 raw_pass='+str(report['raw_pass'])+' offsets=future_same_receipt_reader_only'
    receipt['result']['log_tail']='SXPINS '+prefix+'\nSXREPORT1 '+encode_report(report)+'\n'+footer+'\n'
    return receipt


def fixture():
    pins={k:v for k,v in MANIFEST.items() if k not in ('physical_sha256','bytes')}
    rows=[]
    for seed in policy.SEEDS:
        for m in policy.M_VALUES:
            for layer in policy.LAYERS:
                yes=(layer,m) in policy.TARGETS
                samples={a:[event_us(.010 if a.startswith('t') else .012)]*24 for a in (('c0','c1','t0','t1','anchor') if yes else ('common',))}
                rows.append(dict(layer=layer,m=m,seed=seed,target=yes,samples_us=samples))
    report=dict(format='PROJ_SK_XOR_V1',pins=pins,seeds=list(policy.SEEDS),rounds=24,rows=rows,
        checks=dict(CHECKS),complete=True,elapsed=200.,resources={},anchors={},build_names=list(MODULES),
        weight_inventory={name:dict(shape=[n,k],dtype='torch.float32' if name=='router' else 'torch.bfloat16',bytes=n*k*(4 if name=='router' else 2)) for name,(n,k) in SHAPES.items()},
        common_scope='Other20 actual common vectors, identically charged; no independent null.')
    report['raw_gate']=policy.score_model(rows);report['raw_pass']=report['raw_gate']['keep']
    for (layer,m),name in TARGET_NAMES.items():
        group={};meta=PROTOTYPE['metadata'][name];n=1024 if name=='slice8F' else SHAPES[layer][0];k=SHAPES[layer][1]
        for arm in ('c0','c1','t0','t1'):
            v=int(arm.startswith('t'));pmax=50 if v else 64;layout,owner,g=analytical_layout(meta['original_targs'],v,pmax)
            ns=k//(64*g['ups']);nc=n//g['r']
            pack=dict(name=name,variant=v,n=n,k=k,pmax=pmax,base4=54*0x01010101,escapes=100,source_pack_name=SOURCE_PACK[name])
            if v:pack.update(records=100,header_bytes=g['headers']*ns*nc,patch_bytes=pmax*4,rowmap_bytes=n*4)
            stream=(g['payload']+g['headers'])*ns*nc;table=(pmax+n)*4 if v else nc*(2*pmax+g['r'])*4
            group[arm]=dict(cfg=meta['ids'][v][0],name=name,variant=v,nearby=False,module_name=MODULES[int(arm[1])],
                resources=[64,0,0,layout['dynamic'],layout['threads'],layout['threads'],1],
                expected_layout=layout,expected=owner,register_thread_product=64*layout['threads'],
                pack_metadata=pack,immutable_bytes=[stream,table])
        report['resources'][name]=group;cid,full=ANCHOR_IDS[name]
        report['anchors'][str((layer,m))]=dict(code='run',cid=cid,full=full,vars=sorted(FREEVARS))
    records=[dict(event='test',index=i,status='pass') for i in range(9)]
    for m in policy.M_VALUES:
        for layer in policy.LAYERS:
            records.append(dict(event='benchmark',index=len(records)-9,status='pass',flags=[],case=dict(layer=layer,m=m,weight=62),median_raw_ns=12000.,median_ns=8000.))
    receipt=dict(id=JOB,board='muse_bf16_proj',mode='check',status='done',result=dict(gpu='H200',passed=True,status='ok',returncode=0,error=None,flags=[],checks=dict(ok=True,flagged=[],rejected=[]),records=records))
    return seal(receipt,report),report


class ReceiptTests(unittest.TestCase):
    def validate(self,r):return validate_receipt(r,JOB,MANIFEST,PROTOTYPE)
    def test_valid_and_contemporary_offsets(self):
        receipt,report=fixture();result=self.validate(receipt)
        self.assertTrue(result['raw_complete']);self.assertTrue(result['registered_decision']);self.assertTrue(result['calibration_domain_valid'])
        self.assertEqual(set(result['contemporary_offsets_us'].values()),{4.})
        self.assertEqual(result['raw'],policy.score_model(report['rows']))
        self.assertIn('not official candidate paired score',result['scope'])
    def test_invalid_adjusted_domain_preserves_raw(self):
        receipt,_=fixture()
        for row in receipt['result']['records']:
            if row['event']=='benchmark':row.update(median_raw_ns=100000.,median_ns=1000.)
        result=self.validate(receipt)
        self.assertTrue(result['raw_complete']);self.assertTrue(result['raw']['keep']);self.assertFalse(result['calibration_domain_valid'])
        self.assertFalse(result['registered_decision']);self.assertIsNone(result['adjusted_sensitivity']);self.assertEqual(len(result['report']['rows']),90)
    def test_positive_domain_can_fail_adjusted_gate(self):
        receipt,_=fixture()
        for row in receipt['result']['records']:
            if row['event']=='benchmark':row.update(median_raw_ns=1000.,median_ns=100000.)
        result=self.validate(receipt)
        self.assertTrue(result['raw']['keep']);self.assertTrue(result['calibration_domain_valid'])
        self.assertFalse(result['adjusted_sensitivity']['keep']);self.assertFalse(result['registered_decision'])
    def test_cli_requires_job(self):
        import contextlib,io,sys
        from unittest.mock import patch
        from proj_sk_xor_receipt import main
        with patch.object(sys,'argv',['proj_sk_xor_receipt.py','receipt.json']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:main()
        self.assertEqual(error.exception.code,2)
    def test_owning_job_board_private_terminal(self):
        receipt,_=fixture()
        for key,value in (('id','1111111111111111'),('board','muse_attention_decode'),('mode','run'),('status','running')):
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.assertRaises(ReceiptError):self.validate(bad)
        with self.assertRaises(ReceiptError):validate_receipt(receipt,'',MANIFEST,PROTOTYPE)
    def test_official_failure_flags_tests_and_benchmarks(self):
        receipt,_=fixture()
        mutations=[lambda r:r['result'].update(passed=False),lambda r:r['result'].update(flags=['x']),
            lambda r:r['result']['checks'].update(flagged=['x']),lambda r:r['result']['records'].pop(0),
            lambda r:r['result']['records'][0].update(status='fail'),lambda r:r['result']['records'][-1]['case'].update(weight=61),
            lambda r:r['result']['records'][-1].update(flags=['x']),lambda r:r['result']['records'][-1].update(case=copy.deepcopy(r['result']['records'][-2]['case']))]
        for fn in mutations:
            bad=copy.deepcopy(receipt);fn(bad)
            with self.assertRaises(ReceiptError):self.validate(bad)
    def test_missing_truncated_duplicate_messages(self):
        receipt,_=fixture();log=receipt['result']['log_tail']
        for badlog in (log.replace('SXREPORT1 ','LOST ',1),log+'SXDONE duplicate\n',log[:log.index('SXREPORT1 ')+25],log.replace('SX2:','SX2:!',1)):
            bad=copy.deepcopy(receipt);bad['result']['log_tail']=badlog
            with self.assertRaises(ReceiptError):self.validate(bad)
    def test_rows_counts_pins_and_header(self):
        receipt,report=fixture()
        changes=[lambda r:r['rows'].pop(),lambda r:r['rows'].__setitem__(1,copy.deepcopy(r['rows'][0])),
            lambda r:r['checks'].update(oracle_checks=839),lambda r:r['pins'].update(body_sha256='0'*64),
            lambda r:r.update(elapsed=295.),lambda r:r.update(complete=False),lambda r:r['seeds'].__setitem__(0,1),
            lambda r:r['rows'][0]['samples_us']['common'].pop()]
        for fn in changes:
            bad=copy.deepcopy(report);fn(bad)
            with self.assertRaises(ReceiptError):self.validate(seal(copy.deepcopy(receipt),bad))
        bad=copy.deepcopy(receipt);bad['result']['log_tail']=bad['result']['log_tail'].replace('"rows":90','"rows":89',1)
        with self.assertRaises(ReceiptError):self.validate(bad)
    def test_resource_native_ownership_and_expected_layout(self):
        receipt,report=fixture()
        changes=[lambda r:r['resources']['qkv8F']['c0']['resources'].__setitem__(1,4),
            lambda r:r['resources']['qkv8F']['t0'].update(cfg=0),
            lambda r:r['resources']['qkv8F']['c0'].update(cfg=False),
            lambda r:r['resources']['qkv8F']['t1'].update(module_name=MODULES[0]),
            lambda r:r['resources']['qkv8F']['t0']['expected_layout'].update(dynamic=1),
            lambda r:r['resources']['qkv8F']['t0']['resources'].__setitem__(3,1),
            lambda r:r['resources']['qkv8F']['c0']['pack_metadata'].update(pmax=65),
            lambda r:r['resources']['qkv8F']['c0']['immutable_bytes'].__setitem__(0,1),
            lambda r:r['anchors'][str(('qkv_gate',8))].update(full=0),
            lambda r:r['weight_inventory']['router'].update(dtype='torch.bfloat16')]
        for fn in changes:
            bad=copy.deepcopy(report);fn(bad)
            with self.assertRaises(ReceiptError):self.validate(seal(copy.deepcopy(receipt),bad))
    def test_false_gate_and_pass_rejected(self):
        receipt,report=fixture()
        for fn in (lambda r:r['raw_gate'].update(keep=False),lambda r:r.update(raw_pass=False),lambda r:r['raw_gate'].update(minimum_factor=100.)):
            bad=copy.deepcopy(report);fn(bad)
            with self.assertRaises(ReceiptError):self.validate(seal(copy.deepcopy(receipt),bad))
    def test_real_loser_is_complete_not_exception(self):
        receipt,report=fixture()
        for row in report['rows']:
            if row['target']:
                row['samples_us']['t0']=[event_us(.013)]*24;row['samples_us']['t1']=[event_us(.013)]*24
        report['raw_gate']=policy.score_model(report['rows']);report['raw_pass']=False
        result=self.validate(seal(receipt,report));self.assertTrue(result['raw_complete']);self.assertFalse(result['registered_decision'])

if __name__=='__main__':unittest.main()
