import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = ROOT / 'docs/steering/evidence/2026-11-02-context-cleanup'
manifest = json.loads((EVIDENCE / 'manifest.json').read_text())
BASE = manifest['base']


def original(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT)


def native_tokens(source):
    result = subprocess.run(['clang++', '-x', 'c++', '-fsyntax-only', '-Xclang',
                             '-dump-raw-tokens', '-'], input=source, text=True,
                            capture_output=True, check=True)
    tokens = []
    for match in re.finditer(r"(?ms)^(\w+) '(.*?)'\t", result.stderr):
        kind, value = match.groups()
        if kind == 'comment' or kind == 'unknown' and value.isspace():
            continue
        tokens.append((kind, value))
    assert tokens
    return tokens


def native_bindings(tree):
    return {n.targets[0].id: n.value for n in tree.body
            if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
            and re.search(r'#include|__device__|__global__', n.value.value)}


def module(name, source):
    path = ROOT / 'tools' / (name + '.py')
    result = types.ModuleType('_audit_' + name)
    result.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), result.__dict__)
    result.ROOT = ROOT
    return result


def generated(mod, name, source, target):
    mod.SUB = target
    mod.SRC = ROOT / source
    mod.CU = ROOT / source
    saved = sys.argv
    sys.argv = [name, str(ROOT / source), str(target)]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
        first = target.read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            mod.main()
        assert target.read_bytes() == first, (name, 'repeat generation changed')
        return ast.parse(first)
    finally:
        sys.argv = saved


records = []
for name, source in [('dl_moe_inline', 'muse_moe_w4a16/submission.py'),
                     ('dl_proj_inline', 'muse_bf16_proj/submission.py'),
                     ('dl_one_inline', 'muse_attention_decode/attn_one.cu'),
                     ('dl_ws_inline', 'muse_attention_decode/attn_ws.cu'),
                     ('dl_attn_inline', 'muse_attention_decode/attn_kernel.cu'),
                     ('attn_inline_ws', 'muse_attention_decode/attn_ws.cu')]:
    old = module(name, original('tools/' + name + '.py'))
    new = module(name, (ROOT / 'tools' / (name + '.py')).read_bytes())
    base_source = 'muse_attention_decode/submission.py' if name == 'attn_inline_ws' else 'muse_decode_layer/submission.py'
    with tempfile.TemporaryDirectory(prefix='okc-cleanup-') as folder:
        a, b = Path(folder) / 'before.py', Path(folder) / 'after.py'
        a.write_bytes((ROOT / base_source).read_bytes())
        b.write_bytes(a.read_bytes())
        before = generated(old, name, source, a)
        after = generated(new, name, source, b)
        originals, revisions = native_bindings(before), native_bindings(after)
        assert originals.keys() == revisions.keys()
        natives = []
        for key in originals:
            x, y = originals[key].value, revisions[key].value
            if x != y:
                assert native_tokens(x) == native_tokens(y), (name, key, 'native token drift')
            natives.append(dict(binding=key,bytes_equal=x == y,
                                before_sha256=hashlib.sha256(x.encode()).hexdigest(),
                                after_sha256=hashlib.sha256(y.encode()).hexdigest()))
            originals[key].value = revisions[key].value = 'AUDITED_NATIVE_' + key
        if name == 'dl_proj_inline':
            removed = [n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == 'pack_router']
            assert len(removed) == 1
            before.body.remove(removed[0])
            assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
                           and n.id == 'pack_router' for n in ast.walk(after))
        assert ast.dump(before) == ast.dump(after), (name, 'generated Python AST drift')
        records.append(dict(generator=name,repeat_generation_identical=True,
                            generated_python_ast_equal_after_named_dead_definition=True,
                            native_bindings=natives))

for path, expected in manifest['unchanged_runtime_sources'].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    assert (ROOT / path).read_bytes() == original(path)
assert (ROOT / 'docs/steering/index-history-through-2026-11-02-context-cleanup.md').read_bytes() == subprocess.check_output(['git', 'show', manifest.get('index_archive_commit', BASE) + ':docs/steering/README.md'], cwd=ROOT)
for row in manifest['removed']:
    assert hashlib.sha256(original(row['path'])).hexdigest() == row['sha256']
    assert not (ROOT / row['path']).exists()

instructions = ['AGENTS.md', 'CLAUDE.md', 'docs/steering/README.md']
instructions += [p + '/AGENTS.md' for p in ['muse_bf16_proj', 'muse_attention_decode',
                                           'muse_moe_w4a16', 'muse_decode_layer', 'muse_prefill_layer']]
sizes = [dict(path=p,before_bytes=len(original(p)),after_bytes=(ROOT / p).stat().st_size,
              before_lines=len(original(p).splitlines()),
              after_lines=len((ROOT / p).read_bytes().splitlines())) for p in instructions]
result = dict(base=BASE,generators=records,instructions=sizes,
              instruction_bytes_before=sum(x['before_bytes'] for x in sizes),
              instruction_bytes_after=sum(x['after_bytes'] for x in sizes),
              current_runtime_source_hashes_unchanged=True,removed_sources_recoverable=True,
              clang=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0],
              scope='CPU generation and primary Clang lexical tokens only; no CUDA execution or performance claim')
(EVIDENCE / 'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: six real generators are repeatable; generated Python matches after one dead definition; native tokens match')
print('PASS: six runtime sources unchanged;',len(manifest['removed']),'retired sources recoverable at exact hashes')
print('Instruction bytes',result['instruction_bytes_before'],'->',result['instruction_bytes_after'])
