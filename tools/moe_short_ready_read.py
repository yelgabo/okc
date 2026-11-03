"""Keep strict owning receipt checks while applying the frozen short-cell gate."""
import argparse
import base64
import json
from pathlib import Path
import zlib

import expert_top100_moe_harness_read as reader
from moe_short_ready_harness import build
from moe_short_gate import short_decision


def read(folder,job):
    receipt=reader.strict_json((folder/'status-wait.raw.json').read_text())
    manifest=reader.strict_json((folder/'paired.manifest.json').read_text())
    accepted=reader.strict_json((folder/'accepted.json').read_text())
    payload=(folder/'paired.py').read_bytes()
    reader.require(receipt['id']==accepted['job_id']==job and accepted['source_sha256']==reader.sha(payload),
                   'accepted source/job')
    log=(folder/'profiles'/Path(receipt['output']['log']).name).read_text()
    if receipt['result']['passed'] is False:
        marker='MOE_SHORT_READY_PARTIAL '
        lines=[x[len(marker):] for x in log.splitlines() if x.startswith(marker)]
        reader.require(receipt['mode']=='check' and receipt['board']=='muse_moe_w4a16'
                       and receipt['team']=='yelgabo' and receipt['status']=='done'
                       and receipt['result']['returncode']!=0,'failed owning check')
        reader.require(manifest==build()[1] and reader.sha(payload)==manifest['source_sha256'],'partial exact source')
        reader.require(len(lines)==1 and len(lines[0])<12000 and marker+lines[0] in receipt['result']['log_tail'],
                       'own bounded partial')
        decoder=zlib.decompressobj();raw=decoder.decompress(base64.b64decode(lines[0],validate=True),100000)
        reader.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,'bounded partial decode')
        report=reader.strict_json(raw.decode())
        reader.require(report['runtime_id']==manifest['runtime_id']
                       and report['status']=='FAILED_PARTIAL_NOT_QUALIFIED','partial remains failed')
        result=dict(job=job,target_qualified=False,report=report,scope='Failed partial evidence, no algorithm verdict')
    else:
        upload=dict(job=job,team='yelgabo',mode='check',board='muse_moe_w4a16',content_hash=receipt['content_hash'],
                    payload_ref=receipt['payload_ref'],source_sha256=manifest['source_sha256'],source_bytes=len(payload))
        attest=dict(job=job,owning_log=receipt['output']['log'],sha256=reader.sha(log),source_sha256=reader.sha(payload))
        previous_build,previous_gate=reader.build,reader.decision
        try:
            reader.build=build;reader.decision=short_decision
            result=reader.read(receipt,job,log,manifest,payload,upload,attest)
        finally:reader.build=previous_build;reader.decision=previous_gate
        reader.require(result['report']['loaded_builds']==list(range(7))
                       and result['report']['table_words']==[256]*4,'actual selected source/weights')
        reader.require(result['report']['counters']['flag_zero_checks']==1512,'every observed flag reset')
        (folder/'upload-association.json').write_text(json.dumps(upload,indent=2)+'\n')
        (folder/'log-attestation.json').write_text(json.dumps(attest,indent=2)+'\n')
    result['score_us']=receipt['result'].get('score_us')
    result['speedup']=receipt['result'].get('speedup')
    result['speedup_ci']=receipt['result'].get('speedup_ci')
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True);parser.add_argument('--job',required=True)
    args=parser.parse_args();result=read(args.folder,args.job)
    print(json.dumps({k:v for k,v in result.items() if k!='report'},indent=2))
