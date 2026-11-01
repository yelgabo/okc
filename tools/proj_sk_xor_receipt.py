"""Validate an owning PRIVATE SK-XOR receipt; never a public paired-score claim."""
import argparse,ast,hashlib,json,math,re,lzma,struct
from pathlib import Path
import proj_sk_xor_read as policy
from proj_sk_xor_codec import decode_report

POLICY_SHA256='00e99c9e51423bf349f69ac3aed7f58838c0bce788c59292289d886055b2e694'
CHECKS=dict(immutable_pack_formats=12,captured_placements=210,oracle_checks=840,
    fresh_replay_checks=420,post_timing_checks=210,onehot_nearby_checks=80,outlier_nearby_checks=80)
TARGET_NAMES={('qkv_gate',8):'qkv8F',('qkv_gate',16):'qkv16F',('qkv_gate',32):'qkv32',
    ('o',16):'o16F',('o',32):'o32F',('pooler',16):'pool16F',('pooler',32):'pool32F',
    ('unpooler',8):'slice8F',('unpooler',16):'unp16F',('unpooler',32):'unp32F'}
# Frozen C2 3044ee7 SK_CONFIGS ordering; FULL is chosen only at TOKR.
ANCHOR_IDS={'qkv8F':(0,1),'qkv16F':(3,4),'qkv32':(6,None),'o16F':(11,12),
    'o32F':(14,15),'pool16F':(20,21),'pool32F':(23,24),'slice8F':(26,27),
    'unp16F':(29,30),'unp32F':(32,33)}
SOURCE_PACK={n:n for n in TARGET_NAMES.values()}
SOURCE_PACK.update(qkv32='qkv16F',o32F='o16F',pool32F='pool16F',unp32F='unp16F')
SHAPES={'qkv_gate':(2304,8192),'o':(8192,1024),'pooler':(4096,8192),
    'unpooler':(8192,4096),'router':(256,8192)}
MODULES=['okc_owned_sx_sx_a','okc_owned_sx_sx_b']
FREEVARS={'base4','cid','diag','ext','full','k','n','out_dtype','pmax','stream','tab','tokr'}

class ReceiptError(ValueError):pass

def need(ok,message):
    if not ok:raise ReceiptError(message)

def integer(value):return type(value) is int

def finite(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)

def deep_close(a,b,path='value'):
    if isinstance(a,dict) and isinstance(b,dict):
        need(set(a)==set(b),path+' keys differ')
        for k in a:deep_close(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)) and isinstance(b,(tuple,list)):
        need(len(a)==len(b),path+' length differs')
        for i,(x,y) in enumerate(zip(a,b)):deep_close(x,y,path+f'[{i}]')
    elif isinstance(a,bool) or isinstance(b,bool):need(type(a) is type(b) and a==b,path+' boolean differs')
    elif type(b) is int:need(type(a) is int and a==b,path+' integer differs')
    elif isinstance(a,(int,float)) and isinstance(b,(int,float)):
        need(finite(a) and finite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),path+' number differs')
    else:need(type(a) is type(b) and a==b,path+' differs')

def analytical_layout(targs,variant,pmax):
    """Independent SkCfg translation from committed native template arguments."""
    ts=[t.strip() for t in targs.split(',')];need(len(ts)==13,'bad original_targs')
    need(all(re.fullmatch(r'\d+',ts[i]) for i in (0,1,2,3,4,5,6,9,10)),'bad integer targs')
    need(ts[7]=='true' and ts[8]=='false' and ts[10]=='1' and ts[11]=='__nv_bfloat16' and ts[12]=='false','unsupported native roles')
    mt,tokr,nt,nwr,wsk,upw,nstage=map(int,ts[:7]);nacc=int(ts[9]);need(not(nacc&16),'ESCG unsupported')
    need(min(mt,tokr,nt,nwr,wsk,upw,nstage)>0 and integer(pmax) and 0<=pmax<(1<<22),'invalid geometry/pmax')
    nw=nwr*wsk;r=nwr*nt*8;ups=wsk*upw;payload=nw*upw*nt*768
    headers=((nw*upw*nt*4+15)&~15) if variant else 0
    xby=ups*tokr*128;sst=(xby+payload+headers+1023)&~1023
    nesc=0 if variant else min(4,nstage);correction=nesc*((16*mt*(r+1)+3)&~3)*4
    fixed=1024+nstage*sst+16*nstage+32+correction+128+16
    common=dict(static=0,dynamic=fixed+(0 if variant else (2*pmax+r)*4),threads=(nw+1+nesc)*32,
        stage_stride=sst,headers_per_stage=headers,fixed=fixed,escape_warps=nesc)
    owner=dict(static=0,dynamic=common['dynamic'],threads=common['threads'],stage_stride=sst,
        weight_payload=payload,header_bytes=headers,weight_bytes=payload+headers,x_reservation=xby,
        stages=nstage,correction_bytes=correction,escape_readers=nesc,empty_arrivals=nw+nesc)
    return common,owner,dict(r=r,ups=ups,payload=payload,headers=headers)

def validate_resources(report,prototype):
    need(set(prototype['metadata'])==set(TARGET_NAMES.values()),'prototype target mismatch')
    need(report['build_names']==MODULES,'wrong independent build names')
    need(set(report['resources'])==set(TARGET_NAMES.values()),'requires ten resource groups')
    for (layer,m),name in TARGET_NAMES.items():
        group=report['resources'][name];meta=prototype['metadata'][name]
        need(set(group)=={'c0','c1','t0','t1'},'incomplete resource arms')
        vectors=[]
        for arm in ('c0','c1','t0','t1'):
            info=group[arm];variant=int(arm.startswith('t'));slot=int(arm[1]);pack=info['pack_metadata']
            need(integer(info['cfg']) and integer(info['variant']) and info['cfg']==meta['ids'][variant][0] and info['name']==name and info['variant']==variant and info['nearby'] is False,'resource native cfg mismatch')
            need(info['module_name']==MODULES[slot],'module/placement mismatch')
            need(integer(pack['variant']) and pack['name']==name and pack['variant']==variant and pack['source_pack_name']==SOURCE_PACK[name],'pack ownership/variant mismatch')
            n=1024 if name=='slice8F' else SHAPES[layer][0];k=SHAPES[layer][1]
            need(integer(pack['n']) and integer(pack['k']) and pack['n']==n and pack['k']==k,'pack extent mismatch')
            pmax=pack['pmax'];layout,owner,g=analytical_layout(meta['original_targs'],variant,pmax)
            deep_close(info['expected_layout'],layout,'reported analytical layout')
            deep_close(info['expected'],owner,'reported implementation layout')
            v=info['resources'];need(len(v)==7 and all(integer(x) for x in v),'bad resource vector')
            need(0<v[0]<=255 and v[1]==0 and v[2]==0 and v[3]==layout['dynamic'] and v[4]==v[5]==layout['threads'] and 0<v[6]<=32,'resource admission/layout mismatch')
            need(v[3]<=227*1024 and v[4]<=1024,'illegal launch resource size')
            need(integer(info['register_thread_product']) and info['register_thread_product']==v[0]*v[4],'wrong register product')
            need(n%g['r']==0 and k%(64*g['ups'])==0,'pack divisibility')
            ns=k//(64*g['ups']);nc=n//g['r'];stream=(g['payload']+g['headers'])*ns*nc
            if variant:
                need(integer(pack['records']) and 0<=pack['records']<1<<22 and pmax==(pack['records']+1)//2,'patch count mismatch')
                need(pack['header_bytes']==g['headers']*ns*nc and pack['patch_bytes']==pmax*4 and pack['rowmap_bytes']==n*4,'patch byte extents mismatch')
                table=(pmax+n)*4
            else:
                need(pmax>=2 and pmax%2==0,'control table extent invalid');table=nc*(2*pmax+g['r'])*4
            deep_close(info['immutable_bytes'],[stream,table],'immutable backing byte extent')
            vectors.append(v)
        deep_close(group['c0']['pack_metadata'],group['c1']['pack_metadata'],'shared control immutable pack')
        deep_close(group['t0']['pack_metadata'],group['t1']['pack_metadata'],'shared treatment immutable pack')
        need(vectors[0]==vectors[1] and vectors[2]==vectors[3],'independent build attribute mismatch')
        need(vectors[2][6]>=vectors[0][6],'resident blocks regressed')
    need(set(report['anchors'])=={str(k) for k in TARGET_NAMES},'requires ten actual anchors')
    for key,name in TARGET_NAMES.items():
        anchor=report['anchors'][str(key)];cid,full=ANCHOR_IDS[name]
        need(integer(anchor['cid']) and (anchor['full'] is None or integer(anchor['full'])) and anchor['code']=='run' and anchor['cid']==cid and anchor['full']==full,'wrong actual C2 anchor selection')
        need(len(anchor['vars'])==len(FREEVARS) and set(anchor['vars'])==FREEVARS,'wrong actual C2 closure ABI')
    need(set(report['weight_inventory'])==set(SHAPES),'missing actual weight inventory')
    for name,(n,k) in SHAPES.items():
        deep_close(report['weight_inventory'][name],dict(shape=[n,k],dtype='torch.float32' if name=='router' else 'torch.bfloat16',bytes=n*k*(4 if name=='router' else 2)),'weight inventory')

def _validate(receipt,expected_job,manifest,prototype):
    need(isinstance(expected_job,str) and bool(re.fullmatch('[0-9a-f]{16}',expected_job)),'explicit owning job ID required')
    need(receipt['id']==expected_job,'wrong owning job ID')
    need(receipt['board']=='muse_bf16_proj' and receipt['mode']=='check','requires PRIVATE projection check, not public run')
    need(receipt['status']=='done','receipt not terminal done')
    result=receipt['result'];need(result['gpu']=='H200','receipt is not official H200')
    need(result['passed'] is True and result['status']=='ok' and type(result['returncode']) is int and result['returncode']==0 and result['error'] is None,'official check failed')
    need(result['flags']==[] and result['checks']['ok'] is True and result['checks']['flagged']==[] and result['checks']['rejected']==[],'official flags/rejections present')
    records=result['records'];tests=[r for r in records if r.get('event')=='test'];bench=[r for r in records if r.get('event')=='benchmark']
    need(len(tests)==9 and {r['index'] for r in tests}==set(range(9)) and all(r['status']=='pass' and integer(r['index']) for r in tests),'requires nine passing official tests')
    need(len(bench)==30 and {r['index'] for r in bench}==set(range(30)) and all(integer(r['index']) for r in bench),'requires thirty official benchmarks')
    offsets={}
    for row in bench:
        case=row['case'];key=(case['layer'],case['m'])
        need(integer(case['m']) and key in {(l,m) for l in policy.LAYERS for m in policy.M_VALUES} and key not in offsets,'duplicate/unexpected official cell')
        need(row['status']=='pass' and row['flags']==[] and integer(case['weight']) and case['weight']==62,'invalid benchmark status/weight')
        need(finite(row['median_raw_ns']) and finite(row['median_ns']) and min(row['median_raw_ns'],row['median_ns'])>0,'invalid official latency')
        offsets[key]=(row['median_raw_ns']-row['median_ns'])/1000
    lines=result['log_tail'].splitlines();messages={}
    for prefix in ('SXPINS ','SXREPORT1 ','SXDONE '):
        found=[s[len(prefix):] for s in lines if s.startswith(prefix)]
        need(len(found)==1,'missing/truncated/duplicate '+prefix.strip());messages[prefix]=found[0]
    printed=json.loads(messages['SXPINS ']);encoded=messages['SXREPORT1 '];need(len(encoded)<9000,'measurement codec budget exceeded')
    report=decode_report(encoded);pins={k:v for k,v in manifest.items() if k not in ('physical_sha256','bytes')}
    deep_close(report['pins'],pins,'report pins');deep_close(printed['pins'],pins,'printed pins')
    need(prototype['runtime_sha256']==pins['augmented_sha256'] and prototype['source_pin']==pins['c2_sha256'] and prototype['source_lf_sha256']==pins['c2_lf_sha256'],'prototype/source pin mismatch')
    actual_policy=hashlib.sha256(Path(policy.__file__).read_text(encoding='utf8').encode()).hexdigest()
    need(actual_policy==pins['score_model_sha256']==POLICY_SHA256,'immutable scoring policy changed')
    deep_close(report['checks'],CHECKS,'coverage checks');deep_close(printed['checks'],CHECKS,'printed coverage')
    need(integer(printed['rows']) and integer(printed['rounds']) and integer(report['rounds']) and printed['rows']==90 and printed['rounds']==24 and report['seeds']==list(policy.SEEDS) and report['rounds']==24,'wrong registered seeds/rows/rounds')
    deep_close(report['seeds'],list(policy.SEEDS),'registered seeds')
    need(report['complete'] is True and finite(report['elapsed']) and 0<report['elapsed']<295,'incomplete/expired preparation')
    validate_resources(report,prototype)
    raw=policy.score_model(report['rows']);deep_close(report['raw_gate'],raw,'printed raw gate')
    need(type(report['raw_pass']) is bool and report['raw_pass']==raw['keep'],'false raw_pass')
    need(messages['SXDONE ']=='coverage=3x30 target=10 common=20 raw_pass='+str(raw['keep'])+' offsets=future_same_receipt_reader_only','false/incomplete final coverage footer')
    valid=True;error=None;adjusted=None
    try:adjusted=policy.score_model(report['rows'],offsets)
    except ValueError as exc:
        need(str(exc)=='nonpositive offset-adjusted latency','unexpected adjusted score error')
        valid=False;error=str(exc)
    return dict(job=expected_job,board=receipt['board'],mode='check',raw_complete=True,raw=raw,
        calibration_domain_valid=valid,calibration_domain_error=error,adjusted_sensitivity=adjusted,
        registered_decision=bool(raw['keep'] and valid and adjusted['keep']),
        contemporary_offsets_us={str(k):v for k,v in sorted(offsets.items())},report=report,
        scope='Private paired event factors and same-receipt common-offset sensitivity; not official candidate paired score. Other20 are actual common vectors charged identically, not independent nulls.')

def validate_receipt(receipt,expected_job,manifest,prototype):
    try:return _validate(receipt,expected_job,manifest,prototype)
    except ReceiptError:raise
    except (KeyError,TypeError,IndexError,AssertionError,ValueError,OverflowError,lzma.LZMAError,struct.error) as exc:raise ReceiptError('malformed/incomplete receipt: '+str(exc)) from exc

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--job',required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path);a=p.parse_args()
    evidence=a.root/'docs/steering/evidence'
    manifest=json.loads((evidence/'2026-11-01-projection-sk-xor-private/harness-manifest.json').read_text(encoding='utf8'))
    prototype=json.loads((evidence/'2026-11-01-projection-sparse-xor/prototype-manifest.json').read_text(encoding='utf8'))
    result=validate_receipt(json.loads(a.receipt.read_text(encoding='utf8')),a.job,manifest,prototype)
    text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.write_text(text,encoding='utf8')
    else:print(text,end='')

if __name__=='__main__':main()
