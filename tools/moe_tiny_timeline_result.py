"""Bind the bounded timeline capsule to its exact owning private receipt."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

from expert_top100_moe_harness_read import require, strict_json
from moe_tiny_timeline_harness import build


def read(folder, job):
    manifest=strict_json((folder/'paired.manifest.json').read_text())
    accepted=strict_json((folder/'accepted.json').read_text())
    receipt=strict_json((folder/'status-wait.raw.json').read_text())
    source=(folder/'paired.py').read_bytes()
    require(manifest==json.loads(json.dumps(build()[1])) and hashlib.sha256(source).hexdigest()==manifest['source_sha256'],
            'exact reproducible source')
    require(accepted['job_id']==receipt['id']==job and accepted['source_sha256']==manifest['source_sha256'],
            'accepted owning source/job')
    require(receipt['team']=='yelgabo' and receipt['board']=='muse_moe_w4a16'
            and receipt['mode']=='check' and receipt['status']=='done','own terminal private MoE')
    path=folder/'profiles'/Path(receipt['output']['log']).name
    log=path.read_text()
    lines=[s for s in log.splitlines() if s.startswith('MOE_TINY_TIMELINE ')]
    require(len(lines)==1 and lines[0] in receipt['result']['log_tail'],'unique owning capsule')
    data=lines[0].split(' ',1)[1]
    require(len(data)<=12000,'encoded capsule bound')
    decoder=zlib.decompressobj()
    raw=decoder.decompress(base64.b85decode(data),100000)
    require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'decoded capsule bound')
    report=strict_json(raw.decode())
    require(report['runtime_id']==manifest['runtime_id'] and report['official_arm']=='control0','runtime/official arm')
    complete=report['status']=='complete'
    if complete:
        require(receipt['result']['passed'] is True and receipt['result']['returncode']==0,'passing complete receipt')
        require(report['loaded_builds']==list(range(7)) and len(report['owners_bytes'])==4,'actual native owners/builds')
        require(report['cleanup_restored'] is True and report['prepare_s']<295
                and not report.get('detail_compacted'),'complete cleanup/budget/detail')
        require([r[:3] for r in report['rows']]==[[1102031,t,'normal'] for t in (1,8,16,32,128,256)],
                'six fixed rows')
        expected=dict(observations=42,restorations=6,captures=24,timed_replays=768)
        require(all(report['counters'].get(k)==v for k,v in expected.items()),'complete counters')
        require(report['counters']['worst'][0]<=.015 and report['counters']['worst'][1]<=.075,'numerical gates')
        require([t for t,r in report['traces']]==[1,8],'two observed loads')
        for tokens,item in report['traces']:
            require(item['tick_ns']==10 and len(item['ctas'])==132
                    and [r[0] for r in item['ctas']]==list(range(132)),'CTA summary coverage')
            for row in item['ctas']:
                require(len(row)==14 and row[1:9]==sorted(row[1:9])
                        and all(x>=0 for x in row[1:]) and row[9]+row[10]<=192,'CTA summary bounds/order')
        ns={}
        from moe_tiny_timeline_harness import ENTRY
        import ast
        node=next(n for n in ast.parse(ENTRY).body if isinstance(n,ast.FunctionDef) and n.name=='_tt_costs')
        import statistics
        ns['statistics']=statistics
        exec(ast.get_source_segment(ENTRY,node),ns)
        costs=ns['_tt_costs'](report['rows'])
        require(costs==report['observer_costs'] and report['observer_valid'] is all(r[-1] for r in costs),
                'recomputed observation-cost gate')
    else:
        require(report['status']=='failed' and receipt['result']['passed'] is False,'failed remains failed')
    result=dict(job=job,source_sha256=manifest['source_sha256'],report=report,
                observer_usable=complete and report['observer_valid'],algorithm_gain=False,
                score_us=receipt['result'].get('score_us'),speedup=receipt['result'].get('speedup'),
                speedup_ci=receipt['result'].get('speedup_ci'),log_sha256=hashlib.sha256(log.encode()).hexdigest())
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--job',required=True)
    args=parser.parse_args()
    result=read(args.folder,args.job)
    print(json.dumps({k:v for k,v in result.items() if k!='report'},indent=2))
