"""Check pinned preparation selection with CPU fixtures, without device work."""
import ast
import hashlib
import json
from pathlib import Path

from prod_size_compile_g import bindings
from g_expert_g224_prefix import BASE, BASE_SHA, function_source

source = Path(BASE).read_text()
assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA
env = bindings(source)
for name in ('bfp__sk2_name', 'bfp__sk2_tokr', 'bfp__route', 'bfp__route_key',
             'bfp__mt_key', 'bfp__static_choice'):
    exec(function_source(source, name), env)
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
         and n.name == 'prepare'
         and '_stack_unpinned_prepare(weights)' in ast.get_source_segment(source, n)]
assert len(nodes) == 1
prepare = ast.get_source_segment(source, nodes[0])
exec(prepare, env)
records = []
for mode in ('all_banks', 'pooler_missing', 'router_missing', 'router_disabled', 'bfp_absent'):
    bfp = dict(sm90=True, old_paths=False, sk2={v:() for v in env['bfp_SK2_PICK'].values()},
               disabled=set(), choice={})
    if mode == 'pooler_missing':
        bfp['sk2'].pop(env['bfp__sk2_name']('pooler', 32))
    if mode == 'router_missing':
        bfp['sk2'].pop(env['bfp__sk2_name']('router', 32))
    if mode == 'router_disabled':
        bfp['disabled'].add(env['bfp__route_key']('router', 32, 'sk2'))
    state = {'bfp':None if mode == 'bfp_absent' else bfp,
             'mm_pick':{(32, 'pr'):'parent-choice'}}
    env['_stack_unpinned_prepare'] = lambda weights: state
    pick = env['prepare']({})['mm_pick'][32, 'pr']
    assert pick == ('bfpf' if mode == 'all_banks' else
                    'parent-choice' if mode == 'bfp_absent' else 'bfp'), (mode, pick)
    records.append(dict(mode=mode, t32_pick=pick,
                        pooler_choice=bfp['choice'].get(('pooler', 32)),
                        router_choice=bfp['choice'].get(('router', 32))))
result = dict(base_sha256=BASE_SHA, prepare_definition_line=nodes[0].lineno,
              prepare_sha256=hashlib.sha256(prepare.encode()).hexdigest(), records=records,
              scope='Actual pinned preparation selector with controlled CPU state; no device preparation, target selected choice or latency evidence',
              decision='Existing conditional fusion; no new fusion candidate without actual fallback and exposed-time evidence')
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2)+'\n')
print('Five preparation selector fixtures passed; target choice remains unobserved')
