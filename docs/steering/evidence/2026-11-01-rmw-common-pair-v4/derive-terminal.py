import collections,csv,hashlib,json,math,pathlib
root=pathlib.Path('.');ev=root/'docs/steering/evidence/2026-11-01-rmw-common-pair-v4'
r=json.loads((ev/'own-terminal-receipt.json').read_bytes());o=json.loads((ev/'terminal-reader-observation.json').read_bytes());p=o['report']
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pins={str(f):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in [root/'.build/rmw-pair-v4/check.py',root/'tools/g_common_rmw_pair_read.py',root/'tools/g_common_rmw_pair_manifest.json']}
assert pins['tools/g_common_rmw_pair_manifest.json']['sha256']=='c5f0e3975e30f4c012be8320615c7a19bbf9a7b7a63381906c10f3a3b3d4c574'
assert pins['tools/g_common_rmw_pair_read.py']['sha256']=='745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d'
assert pins['.build/rmw-pair-v4/check.py']['sha256']==o['source_sha256']
cross=((0,2),(0,3),(1,2),(1,3));loads=(1,8,16,32,128,256)
weighted=[];regressions=[];official=[]
for seed in p['seeds']:
 for t in loads:
  rows={x['context']:x for x in p['rows'] if x['seed']==seed and x['tokens']==t}
  costs=[(46*rows['sliding']['times_us'][i]+16*rows['full']['times_us'][i])/62 for i in range(4)]
  logs=[math.log(costs[a]/costs[b]) for a,b in cross]
  weighted.append(dict(seed=seed,tokens=t,costs_us=costs,cross_log_gains=logs,cross_speed_gain_pct=[100*math.expm1(x) for x in logs],cross_latency_change_pct=[100*math.expm1(-x) for x in logs]))
  for context,row in rows.items():
   for a,b in cross:
    off,on=row['times_us'][a],row['times_us'][b];delta=on-off;cap=max(.5,.01*off)
    if delta>cap:regressions.append(dict(seed=seed,tokens=t,context=context,off=a,on=b,off_us=off,on_us=on,delta_us=delta,cap_us=cap,excess_us=delta-cap))
bench=[x for x in r['result']['records'] if x['event']=='benchmark']
for load in ('decode 1','decode 16','decode 32','verify 1','verify 16','verify 32'):
 pair={x['case']['layer']:x for x in bench if x['case']['load']==load}
 candidate=sum(w*pair[c]['median_ns']/1000 for w,c in ((46,'sliding'),(16,'full')))/62
 baseline=sum(w*pair[c]['baseline_ns']/1000 for w,c in ((46,'sliding'),(16,'full')))/62
 official.append(dict(load=load,sliding_us=pair['sliding']['median_ns']/1000,full_us=pair['full']['median_ns']/1000,weighted_us=candidate,weighted_baseline_us=baseline,ratio=baseline/candidate))
score=math.exp(sum(math.log(x['ratio']) for x in official)/6)
assert math.isclose(score,r['result']['speedup'],rel_tol=1e-12)
q=json.loads((ev/'terminal-quota.json').read_bytes());q=next(x for x in q['problems'] if x['name']=='muse_decode_layer')
jobs=json.loads((ev/'terminal-jobs.json').read_bytes());live=[x for x in jobs if x['problem']=='muse_decode_layer' and x['status'] not in ('done','failed','cancelled','error')]
out=dict(pins=pins,weighted_rows=weighted,regression_failures=regressions,official_rows=official,reconstructed_official_score=score,quota=q,layer_live_in_latest_jobs=live,jobs_observed=len(jobs),decision=o['decision'],prepare=p['prepare_s'],phases=p['prepare_phase_seconds'],resource_vectors={k:v[0] for k,v in p['common_resources'].items()},actual_attention_count=len(p['actual_attention']))
(ev/'terminal-derived.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
with (ev/'terminal-weighted-rows.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['seed','tokens','OFF0_us','OFF1_us','ON0_us','ON1_us','gain_0_2_pct','gain_0_3_pct','gain_1_2_pct','gain_1_3_pct'])
 for x in weighted:w.writerow([x['seed'],x['tokens'],*x['costs_us'],*x['cross_speed_gain_pct']])
print(json.dumps(dict(regressions=regressions,official=official,score=score,quota=q,live=live,jobs=len(jobs),ranges={str(t):[min(v for x in weighted if x['tokens']==t for v in x['cross_speed_gain_pct']),max(v for x in weighted if x['tokens']==t for v in x['cross_speed_gain_pct'])] for t in loads}),indent=2))
