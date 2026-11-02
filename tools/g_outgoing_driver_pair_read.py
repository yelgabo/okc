"""Exact owning schema4 driver-identity reader; fixed original score gates."""
import argparse
import json
from pathlib import Path

import g_outgoing_pair_read as old
from g_common_rmw_pair import function


MANIFEST=None
decision=old.decision
edge_gate=old.edge_gate
verify_source=old.verify_source
scope=dict(vars(old))
text=Path(old.__file__).read_text(encoding='utf8')
graph_source=function(text,'graph_evidence').replace('resources[str(tokens)][arm][0]', 'report["driver_secondary_identities"][str(tokens)][arm]')
exec(graph_source,scope)
old_graph=scope['graph_evidence']


def graph_evidence(report):
    assert report['driver_identity_namespace']=='CUDA13:CUfunction','DRIVER_IDENTITY_NAMESPACE'
    assert set(report['primary_identities'])=={'tiny','tiny2'},'DRIVER_PRIMARY_COVERAGE'
    secondary=report['driver_secondary_identities']
    assert set(secondary)=={'1','8'},'DRIVER_SECONDARY_COVERAGE'
    for values in secondary.values():
        assert len(values)==4 and values[0]==values[1] and values[2]==values[3],'DRIVER_SECONDARY_PARITY'
        assert all(type(value)is int and value>0 for value in values),'DRIVER_SECONDARY_IDENTITY'
    old_graph(report)
    queries=[row for row in report['build_census'] if 'seam_driver_graph_edges' in row['functions']]
    assert queries and all('-lcuda' in row['link_flags'] for row in queries),'DRIVER_LINK_CENSUS'


scope['graph_evidence']=graph_evidence
capsule_source=function(text,'capsule').replace('OUTGOING_REPORT ', 'OUTGOING_DRIVER_REPORT ').replace('report["schema"]==3','report["schema"]==4')
exec(capsule_source,scope)


def capsule(log):
    scope['MANIFEST']=MANIFEST
    return scope['capsule'](log)


read_scope=dict(vars(old.receipt_rules));read_scope.update(capsule=capsule)
exec(function(Path(old.receipt_rules.__file__).read_text(),'read'),read_scope)


def read(receipt,job,log,manifest):
    global MANIFEST
    MANIFEST=manifest;read_scope['MANIFEST']=manifest
    return read_scope['read'](receipt,job,log)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('receipt',type=Path);parser.add_argument('--job',required=True);parser.add_argument('--source',required=True,type=Path);parser.add_argument('--manifest',required=True,type=Path);parser.add_argument('--log',required=True,type=Path);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    manifest=json.loads(args.manifest.read_text());verify_source(args.source,manifest);receipt=json.loads(args.receipt.read_text())
    assert args.log.resolve()==Path(receipt['output']['log']).resolve() and args.job in args.log.name,'owning log'
    import hashlib
    raw=args.log.read_bytes();result=read(receipt,args.job,raw.decode('utf8'),manifest);result['owning_log_sha256']=hashlib.sha256(raw).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(job=args.job,complete=True,private_raw_gate=result['private_raw_gate'])))
