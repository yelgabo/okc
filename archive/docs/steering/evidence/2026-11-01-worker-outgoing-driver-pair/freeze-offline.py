import ast
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
from g_outgoing_driver_pair import build
from g_outgoing_pair import build as original
from g_common_rmw_pair import function

folder=Path(__file__).resolve().parent
source,manifest=build(ROOT)
assert (source,manifest)==build(ROOT),'deterministic rebuild'
prior,_=original(ROOT)
def digest(data):return hashlib.sha256(data).hexdigest()
def functions(text):
    result={};counts={}
    for node in ast.parse(text).body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            ordinal=counts.get(node.name,0);counts[node.name]=ordinal+1
            result[node.name+':'+str(ordinal)]=ast.get_source_segment(text,node)
    return result
old_functions=functions(prior);new_functions=functions(source)
changed=[key for key,value in old_functions.items() if new_functions.get(key)!=value]
allowed={'_ma_ext','bfp__get_ext','_os_install','_srp_prepare','_wl_prepare','_os_graph','_os_load_inline'}
assert all(key.split(':')[0] in allowed for key in changed),changed
old_loop=function(prior,'_wl_prepare').split('    for tokens in _WL_LOADS:',1)[1].split('    report={',1)[0]
new_loop=function(source,'_wl_prepare').split('    for tokens in _WL_LOADS:',1)[1].split('    report={',1)[0]
assert old_loop==new_loop,'timed workload/round body drift'
for name in ('_wl_capture','_wl_run','_wl_copy','_wl_check','_wl_check_owners','_wl_empty','_srp_occupied_empty','_os_score','_os_edges','_fork_native_moe','_os_native_sources'):
    assert function(prior,name)==function(source,name),name
frozen={'.build/outgoing-pair/check.py':'f25d714e928835fba07741ca56fa8783137032f268652c22ec4f5511d771bc9a','tools/g_outgoing_pair_read.py':'5435108e9eb729c7fe1f705244845cc7c916b9af780081e682c83ae81f736c1b'}
for path,want in frozen.items():assert digest((ROOT/path).read_bytes())==want,path
target=ROOT/'.build/outgoing-driver-pair/check.py'
assert target.read_bytes()==source.encode(),'physical upload drift'
(folder/'check.py.gz').write_bytes(gzip.compress(source.encode(),mtime=0))
diff=''.join(difflib.unified_diff(prior.splitlines(True),source.splitlines(True),fromfile='frozen-f25/check.py',tofile='driver/check.py'))
(folder/'host-only.diff.gz').write_bytes(gzip.compress(diff.encode(),mtime=0))
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths=['tools/g_outgoing_driver_pair.py','tools/g_outgoing_driver_pair_read.py','tools/test_g_outgoing_driver_pair.py','tools/g_outgoing_driver_query.py','tools/g_outgoing_pair.py','tools/g_outgoing_pair_read.py','tools/g_outgoing_pair_body.py']
pins={path:dict(sha256=digest((ROOT/path).read_bytes()),bytes=(ROOT/path).stat().st_size) for path in paths}
audit=dict(changed_existing_top_level_functions=changed,added_functions=sorted(set(new_functions)-set(old_functions)),timed_row_loop_sha256=digest(old_loop.encode()),timed_row_loop_exact=True,reversible_to_f25=manifest['reverse_exact'],frozen_inputs=frozen,files=pins,host_compilation='NOT_RUN',target_support='unproved; same allocated check gate',source=manifest)
(folder/'offline-pins.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(source=manifest['source_sha256'],bytes=manifest['bytes'],reader=pins['tools/g_outgoing_driver_pair_read.py']['sha256'],changed_functions=changed,timed_loop_exact=True)))
