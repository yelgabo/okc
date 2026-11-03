import collections,datetime,hashlib,json,pathlib,subprocess,sys
root=pathlib.Path('.'); ev=root/'docs/steering/evidence/2026-11-01-rmw-common-pair-v4'; job='c5e4f89672f743bb'
receipt=ev/'own-terminal-receipt.json'; raw=receipt.read_bytes(); r=json.loads(raw)
assert r['id']==job and r['board']=='muse_decode_layer' and r['mode']=='check' and r['status']=='done'
assert r['content_hash']=='28d360741906efb0ff5a90be98b92a053aff04480442d92992d47907ee557cab'
prefix=root/'muse_decode_layer/profiles'/('g-common-rmw-pair-v4.'+job)
source=root/'.build/rmw-pair-v4/check.py'; srcsha=hashlib.sha256(source.read_bytes()).hexdigest()
assert srcsha=='785a90ca08ddde0c3b194fc219a0645cb8cd14bac02620d8714f0b66163a0b66'
reader=root/'tools/g_common_rmw_pair_read.py'
assert hashlib.sha256(reader.read_bytes()).hexdigest()=='745d7bb508d170dbc6bcd4b86e0f8a0b39c9a9ba08885aec54a45a33e68f004d'
log=pathlib.Path(r['output']['log']); lograw=log.read_bytes()
archive_json=pathlib.Path(str(prefix)+'.json'); archive_log=pathlib.Path(str(prefix)+'.log'); archive_txt=pathlib.Path(str(prefix)+'.txt')
archive_json.write_bytes(raw); archive_log.write_bytes(lograw)
archive_txt.write_bytes(pathlib.Path('.build/rmw-pair-v4/profiles/check.'+job+'.txt').read_bytes())
att=dict(job=job,owning_log=r['output']['log'],sha256=hashlib.sha256(lograw).hexdigest(),source_sha256=srcsha)
attpath=ev/'terminal-log-attestation.json';attpath.write_text(json.dumps(att,indent=2)+'\n')
args=[sys.executable,str(reader),str(receipt),'--job',job,'--source',str(source),'--log',str(archive_log),'--archive-attestation',str(attpath),'--output',str(ev/'terminal-reader-observation.json')]
p=subprocess.run(args,capture_output=True,text=True)
(ev/'terminal-reader.stdout.txt').write_text(p.stdout)
(ev/'terminal-reader.stderr.txt').write_text(p.stderr)
(ev/'terminal-reader-transport.json').write_text(json.dumps(dict(command=args,exitcode=p.returncode,expected_rejection=False,reason='Frozen owning reader evaluated without edits.'),indent=2)+'\n')
records=r['result']['records']; counts=dict(collections.Counter(x['event'] for x in records))
utc=lambda t:datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat()
summary=dict(job=job,status=r['status'],result_status=r['result']['status'],passed=r['result']['passed'],returncode=r['result']['returncode'],checks=r['result']['checks'],flags=r['result']['flags'],error=r['result']['error'],events=counts,setup=records[0],created_at_utc=utc(r['created_at']),started_at_utc=utc(r['started_at']),finished_at_utc=utc(r['finished_at']),queue_s=r['started_at']-r['created_at'],worker_wall_s=r['finished_at']-r['started_at'],gpu=r['result']['gpu'],machine=r['result']['machine'],score_us=r['result']['score_us'],speedup=r['result']['speedup'],speedup_ci=r['result']['speedup_ci'],capsule_present=b'SRM_COMMON_REPORT ' in lograw,wl_prep_present=b'WL_PREP ' in lograw,reader_exitcode=p.returncode,reader_observation_exists=(ev/'terminal-reader-observation.json').exists(),source_sha256=srcsha)
(ev/'terminal-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
files=[receipt,ev/'own-terminal-transport.json',archive_json,archive_log,archive_txt,attpath,ev/'terminal-reader.stdout.txt',ev/'terminal-reader.stderr.txt',ev/'terminal-reader-transport.json',ev/'terminal-summary.json']
files += [ev/'terminal-reader-observation.json'] if (ev/'terminal-reader-observation.json').exists() else []
manifest={str(f):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files}
(ev/'terminal-archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print('READER STDERR:',p.stderr)
assert p.returncode==0 and summary['reader_observation_exists']
