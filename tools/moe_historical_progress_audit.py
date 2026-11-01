"""Read-only audit of exact historical source+owning receipts. No GPU or mutations."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
SOURCE='b35d321730f67b8157eef0683011b118bf3644dc5e9602ee29d823d8056c4e14'
def function(source,anchor):
 assert source.count(anchor)==1,anchor
 a=source.index(anchor);z=source.index('{',a)+1;depth=1
 while depth:depth+=(source[z]=='{')-(source[z]=='}');z+=1
 return source[a:z]
def audit(root):
 raw=subprocess.check_output(['git','-C',str(root),'show','ec565a7:muse_moe_w4a16/submission.py']);assert len(raw)==235700 and hashlib.sha256(raw).hexdigest()==SOURCE
 source=raw.decode();receipt=root/'muse_moe_w4a16/profiles/scored.abe7651dbaa44458.json';rraw=receipt.read_bytes();job=json.loads(rraw);r=job['result']
 assert job['id']=='abe7651dbaa44458' and job['board']=='muse_moe_w4a16' and job['status']=='done' and r['passed'] is True and r['checks']['ok'] and not r['flags']
 check=json.loads((root/'muse_moe_w4a16/profiles/submission.b50351288b504e81.json').read_text());assert check['content_hash']==job['content_hash']
 main=function(source,'template <int MODE>\nvoid launch_main(');skinny=function(source,'template <int SPLIT>\nvoid launch(');spin=function(source,'__device__ __forceinline__ bool spin_failed(')
 assert 'moe_main_kernel<MODE><<<sms, kNTM, sizeof(SmemMain), stream>>>(p);' in main
 assert 'moe_fused_kernel<SPLIT><<<sms, kNTL, sizeof(Smem), stream>>>(p);' in skinny
 for helper in (main,skinny):assert 'cudaLaunchCooperative' not in helper and 'cudaOccupancy' not in helper
 assert 'atomicOr(ctr + 8, code);\n    return true;' in spin and 'trap' not in spin
 assert source.count('while (ld_acquire(bc) < G)')==2
 policy=next(line for line in r['log_tail'].splitlines() if line.startswith('POLICY '))
 assert policy=='POLICY {1: sk8w3s, 8: sk8w3s, 16: sk8w3s, 32: sk8w3s, 128: main-lean-grp-nohoist, 256: main-lean-grp-nohoist}'
 profiles={z['case']['bs']:z for z in r['records'] if z.get('event')=='profile'};bench={z['case']['bs']:z for z in r['records'] if z.get('event')=='benchmark'};assert set(profiles)==set(bench)=={1,8,16,32,128,256}
 rows=[]
 for n in (1,8,16,32,128,256):
  expected='sk_8_3_0::moe_fused_kernel<1>' if n<=32 else 'moe_main_kernel<47>'
  assert expected in profiles[n]['report'],(n,profiles[n]['report'])
  rows.append(dict(tokens=n,load=profiles[n]['case']['load'],selected_family=expected,ordinary_launch=True,software_global_join=True,adjusted_median_ns=bench[n]['median_ns'],profile_report=profiles[n]['report']))
 return dict(scope='Historical source/receipt progress-contract audit only; no new failure/performance observation, production change or public action.',source=dict(commit='ec565a7',path='muse_moe_w4a16/submission.py',bytes=len(raw),sha256=SOURCE),job=dict(id=job['id'],mode=job['mode'],board=job['board'],upload_content_hash=job['content_hash'],receipt_raw_sha256=hashlib.sha256(rraw).hexdigest(),check_same_upload_id=check['id'],status=r['status'],passed=r['passed'],score=r['speedup'],ci=r['speedup_ci'],correctness_tests=sum(z.get('event')=='test' and z.get('status')=='pass' for z in r['records'])),policy=policy,paths=rows,source_helpers=dict(main=main,skinny=skinny,spin_failed=spin),global_join_sites=2,eligibility_gap='Ordinary sms-sized launch does not request guaranteed cooperative co-residency; shared watchdog returns to dependent code on expiry. Affects all six actually profiled paths, not only main47.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=audit(a.root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['job']['score'],r['global_join_sites'],[(x['tokens'],x['selected_family']) for x in r['paths']])
