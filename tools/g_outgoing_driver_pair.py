"""Separately named fixed outgoing pair with canonical driver host queries."""
import argparse
import ast
import json
from pathlib import Path

import g_common_release as exports
import g_outgoing_driver_query as bridge
from g_common_rmw_pair import function, sha
from g_outgoing_pair import build as original, MARKER as OLD_MARKER
from g_outgoing_seam_query import SECONDARY as OLD_SECONDARY, DECL as OLD_DECL
from g_short_rmw_check import once


MARKER = "\n# PRIVATE CHECK ONLY: repaired outgoing driver query four placements. NEVER SUBMIT.\n"
SUPPORT = r'''
_OS_DRIVER_PRIMARY={}
_OS_DRIVER_SECONDARY={1:[None]*4,8:[None]*4}
_OS_DRIVER_RESOURCES={}
_OS_DRIVER_PREFIX_EMITTED=False

def _os_driver_emit(text,**kwargs):
    global _OS_DRIVER_PREFIX_EMITTED
    if text.startswith('SEAM_DRIVER_PREFIX '):
        if _OS_DRIVER_PREFIX_EMITTED:return
        _OS_DRIVER_PREFIX_EMITTED=True
    print(text,**kwargs)

def _os_graph(graph,data,state):
    tokens=data['config']['tokens']
    if tokens not in (1,8):return
    arm=state['_os_placement'];seed=state['_os_bank'];window=data['config']['window']
    row=(_WL_LOADS.index(tokens)*2+int(window==0))*3+_WL_SEEDS.index(seed)
    key='tiny' if tokens==1 else 'tiny2'
    query=state.get('_os_unpool_bfp',{}).get('_os_launch',(bfp__get_ext(),0))[0]
    print('SEAM_DRIVER_QUERY_CONTEXT',row,arm,tokens,window,seed,flush=True)
    record=_os_driver_inspect(graph,state['ma_ext'][key],query,row=row,arm=arm,enabled=arm==3,
                              resources=_OS_DRIVER_RESOURCES,build_census=list(_OS_BUILDS),emit=_os_driver_emit)
    primary=record['primary']['values'];consumer=record['consumer']['values'][0]
    if key in _OS_DRIVER_PRIMARY:assert _OS_DRIVER_PRIMARY[key]==primary,'DRIVER_PRIMARY_DRIFT'
    prior=_OS_DRIVER_SECONDARY[tokens][arm]
    assert prior is None or prior==consumer,'DRIVER_SECONDARY_DRIFT'
    _OS_DRIVER_PRIMARY[key]=primary;_OS_DRIVER_SECONDARY[tokens][arm]=consumer
    _OS_GRAPHS.append([tokens,window,seed,arm,record['proof']])
'''


def build(root):
    previous, manifest = original(root)
    assert sha(previous) == 'f25d714e928835fba07741ca56fa8783137032f268652c22ec4f5511d771bc9a'
    device, suffix = previous.split(OLD_MARKER)
    edits = []
    def change(text, old, new):
        edits.append((old, new))
        return once(text, old, new)
    def literal(text, name, value):
        node = exports.assignment(text, name)
        old = ast.get_source_segment(text, node)
        new = old[:old.index('=')+1] + ' ' + exports.literal(value)
        return change(text, old, new)
    for name in ('MOE_CUDA_SRC', "_MA_ALT_SRCS['moe_helper0']"):
        node = exports.assignment(device, name)
        device = literal(device, name, ast.literal_eval(node.value) + bridge.PRIMARY)
    node = exports.assignment(device, 'MOE_CPP_SRC')
    device = literal(device, 'MOE_CPP_SRC', ast.literal_eval(node.value) + '\n' + bridge.PRIMARY_DECL + '\n')
    resource_only = OLD_SECONDARY.split('std::vector<std::vector<int64_t>> seam_graph_edges(')[0]
    query_decl = '\n#include <vector>\nstd::vector<int64_t> seam_unpool_resources(int64_t pmax);\n' + bridge.SECONDARY_DECL
    node = exports.assignment(device, 'bfp_CPP_SRC')
    device = literal(device, 'bfp_CPP_SRC', once(ast.literal_eval(node.value), OLD_DECL, query_decl))
    device = literal(device, '_OS_QUERY_SOURCE', resource_only + bridge.SECONDARY)
    device = literal(device, '_OS_QUERY_DECL', query_decl)
    device = change(device, '+ ["wl_resources","qual_native_resources","seam_native_identity"],', '+ ["wl_resources","qual_native_resources","seam_native_identity","seam_driver_primary_identity"],')
    device = change(device, '"seam_unpool_resources","seam_graph_edges"],', '"seam_unpool_resources","seam_driver_unpool_identity","seam_driver_graph_edges"],')
    device = change(device, "functions=['seam_sk2','seam_unpool_resources','seam_graph_edges']", "functions=['seam_sk2','seam_unpool_resources','seam_driver_unpool_identity','seam_driver_graph_edges']")
    loader = function(suffix, '_os_load_inline')
    new_loader = once(loader, '    started=time.monotonic()', '''    if 'seam_driver_graph_edges' in kwargs.get('functions',[]):
        kwargs['extra_ldflags']=list(kwargs.get('extra_ldflags',[]))+['-lcuda']
    print('SEAM_DRIVER_BUILD_BEGIN',kwargs.get('name'),kwargs.get('functions'),kwargs.get('extra_cuda_cflags'),kwargs.get('extra_ldflags',[]),flush=True)
    started=time.monotonic()''')
    new_loader = once(new_loader, 'host_load_seconds=time.monotonic()-started', 'link_flags=kwargs.get("extra_ldflags",[]),host_load_seconds=time.monotonic()-started')
    new_loader = once(new_loader, '    return extension', "    print('SEAM_DRIVER_BUILD_DONE '+json.dumps(_OS_BUILDS[-1],sort_keys=True),flush=True)\n    return extension")
    suffix = change(suffix, loader, new_loader)
    suffix = change(suffix, function(suffix, '_os_graph'), function(SUPPORT, '_os_graph'))
    suffix = change(suffix, "global _ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back,load_inline", "global _ma_ext,_fork_native_moe,_decode_attention7,_ab_ref_back,load_inline,_OS_DRIVER_PREFIX_EMITTED")
    suffix = change(suffix, 'load_inline=_os_load_inline;_OS_BUILDS.clear()', 'load_inline=_os_load_inline;_OS_BUILDS.clear();_OS_DRIVER_PREFIX_EMITTED=False')
    suffix = change(suffix, "_OS_GRAPHS.clear();_OS_SECONDARY.clear()", "_OS_GRAPHS.clear();_OS_SECONDARY.clear();_OS_DRIVER_PRIMARY.clear()\n    _OS_DRIVER_SECONDARY.update({1:[None]*4,8:[None]*4});_OS_DRIVER_RESOURCES.clear()")
    suffix = change(suffix, 'secondary_resources=_os_secondary_gate(arms,budget)', "secondary_resources=_os_secondary_gate(arms,budget)\n    _OS_DRIVER_RESOURCES.update(primary=resources,common=common_resources,secondary=secondary_resources)")
    suffix = change(suffix, "'schema':3,", "'schema':4,'driver_identity_namespace':_OS_DRIVER_NAMESPACE,'driver_secondary_identities':dict(_OS_DRIVER_SECONDARY),")
    suffix = change(suffix, "{key:list(ext.seam_native_identity()) for key,ext in arms[0]['ma_ext'].items()}", 'dict(_OS_DRIVER_PRIMARY)')
    suffix = change(suffix, "print('OUTGOING_REPORT '+blob", "print('OUTGOING_DRIVER_REPORT '+blob")
    source_bridge = Path(bridge.__file__).read_text(encoding='utf8')
    validate = function(source_bridge, 'validate_edges').replace('def validate_edges(', 'def _os_driver_validate(').replace('IDENTITY_NAMESPACE', '_OS_DRIVER_NAMESPACE').replace('DRIVER__OS_DRIVER_NAMESPACE','DRIVER_IDENTITY_NAMESPACE').replace('edge_gate(', '_os_edges(')
    inspect = function(source_bridge, 'inspect_capture').replace('def inspect_capture(', 'def _os_driver_inspect(').replace('IDENTITY_NAMESPACE', '_OS_DRIVER_NAMESPACE').replace('validate_edges(', '_os_driver_validate(')
    suffix += '\nimport json\n_OS_DRIVER_NAMESPACE=' + repr(bridge.IDENTITY_NAMESPACE) + '\n' + SUPPORT[:SUPPORT.index('def _os_graph(')] + '\n' + validate + '\n' + inspect + '\n'
    device_hash = sha(device)
    node = exports.assignment(suffix, '_WL_DEVICE_SHA')
    suffix = change(suffix, ast.get_source_segment(suffix, node), '_WL_DEVICE_SHA=' + repr(device_hash))
    node = exports.assignment(suffix, '_WL_CORE_SHA')
    old_hash = ast.get_source_segment(suffix, node)
    suffix = change(suffix, old_hash, "_WL_CORE_SHA=''")
    core = sha(suffix)
    suffix = change(suffix, "_WL_CORE_SHA=''", '_WL_CORE_SHA=' + repr(core))
    source = device + MARKER + suffix
    reverse = source.replace(MARKER, OLD_MARKER)
    extra = '\nimport json\n_OS_DRIVER_NAMESPACE=' + repr(bridge.IDENTITY_NAMESPACE) + '\n' + SUPPORT[:SUPPORT.index('def _os_graph(')] + '\n' + validate + '\n' + inspect + '\n'
    reverse = once(reverse, extra, '')
    for old, new in reversed(edits):reverse = once(reverse, new, old)
    assert reverse == previous, 'HOST_DELTA_REVERSE'
    ast.parse(source)
    assert len(source.encode()) < 1_000_000
    manifest = dict(manifest, source_sha256=sha(source), bytes=len(source.encode()), device_sha256=device_hash, core_sha256=core,
                    parent_upload_sha256=sha(previous), driver_identity_namespace=bridge.IDENTITY_NAMESPACE,
                    driver_link_flags=list(bridge.QUERY_EXTRA_LDFLAGS), reverse_exact=True, schema=4,
                    allocation='captain explicit one repaired check, November2 05:38UTC; next failure closes lane',
                    repaired_authorization_commit='1a20b31',
                    host_delta_operations=len(edits), host_compilation='NOT_RUN; same allocated target check only')
    return source, manifest


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);parser.add_argument('out',type=Path);args=parser.parse_args()
    source,manifest=build(args.root);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_bytes(source.encode())
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
