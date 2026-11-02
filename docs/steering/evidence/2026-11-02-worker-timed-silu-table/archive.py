import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
import g_timed_silu_table as treatment
from prod_size_compile_g import bindings
from test_g_timed_silu_table import capture, coverage

OUT = Path(__file__).resolve().parent
LOCAL = ROOT / '.build/worker-timed-silu-table'
source, generation = treatment.generate(OUT / 'generated')
final_native = bindings(source)['MOE_CUDA_SRC']
result = json.loads((LOCAL / 'compile-result.json').read_text())
shutil.copyfile(LOCAL / 'compile-result.json', OUT / 'compile-result.json')
shutil.copyfile(ROOT / '.build/worker-timed-silu-table-compile.log', OUT / 'compile.log')
receipt = dict(final=generation, records=[], numeric_equivalence='NOT RUN', GPU=False,
               short_paths='unchanged', startup_correction='allocator and guards only; compiled device prefix unchanged')
for record in result['records']:
    family = record['family']
    local = LOCAL / family
    folder = OUT / family
    folder.mkdir(exist_ok=True)
    native = (local / 'selected.cu').read_text()
    prefix = native.split('#include <torch/extension.h>')[0]
    assert prefix == final_native.split('#include <torch/extension.h>')[0]
    link = dict(family=family, compiled_full_native_sha256=treatment.sha(native),
                final_full_native_sha256=treatment.sha(final_native),
                exact_device_prefix_equal=True, device_prefix_sha256=treatment.sha(prefix), files={}, ptx={}, sass={})
    for name in ('manifest.json', 'nvrtc.log', 'ptxas.log'):
        shutil.copyfile(local / name, folder / name)
    for name in ('device.ptx', 'device.cubin', 'device.sass', 'selected.cu'):
        raw = (local / name).read_bytes()
        link['files'][name] = dict(bytes=len(raw), sha256=treatment.sha(raw), local_path=str(local / name))
    ptx = (local / 'device.ptx').read_text()
    lines = ptx.splitlines()
    excerpts = []
    for match in re.finditer(r'(?:\.visible|\.weak) \.entry ([^\s(]+)', ptx):
        symbol = match[1]
        opening = ptx.index('{', match.end())
        depth, end = 1, opening + 1
        while depth:
            depth += (ptx[end] == '{') - (ptx[end] == '}')
            end += 1
        body = ptx[match.start():end]
        if 'timed_silu_table_build' in symbol:
            for op in ('mul.f32', 'ex2.approx.f32', 'add.f32', 'div.approx.f32', 'st.global.f32'):
                assert body.count(op) == 1, (family, symbol, op)
            assert '.ftz.' not in body
            excerpts.append(body)
            link['ptx'][symbol] = dict(scalar_producer_sequence=True)
        else:
            windows = []
            body_lines = body.splitlines()
            for index, line in enumerate(body_lines):
                if re.search(r'\.loc\s+1\s+1976\s+9', line):
                    window = '\n'.join(body_lines[index - 26:index + 35])
                    if 'ld.global.nc.f32' in window:
                        assert window.count('cvt.rn.bf16.f32') >= 3
                        assert window.count('cvt.f32.bf16') >= 2
                        assert 'st.global.L2::cache_hint.b16' in window
                        assert 'ex2.approx' not in window and 'div.approx' not in window
                        assert 'mul.f32' in window and 'shr.u32' in window
                        windows.append(window)
            assert windows, (family, symbol, 'missing actual emitted gate')
            excerpts.extend(windows)
            link['ptx'][symbol] = dict(selected_site_windows=len(windows), rounding_lookup_multiply_store=True)
    (folder / 'codegen.ptx.txt').write_text('\n\n'.join(excerpts) + '\n')
    disassembler = '/root/.cache/okc-cuda130-88/cuda_nvdisasm-linux-x86_64-13.0.85-archive/bin/nvdisasm'
    sass = subprocess.run([disassembler, '-gi', str(local / 'device.cubin')], capture_output=True, text=True, check=True).stdout
    selected = []
    symbol, origin, after_instruction = None, None, True
    for line in sass.splitlines():
        match = re.match(r'\s*\.section\s+\.text\.([^,]+)', line)
        if match:
            symbol = match[1]
            link['sass'][symbol] = dict(source_line_ops={})
        location = re.search(r'//## File .*?, line (\d+)', line)
        if location and after_instruction:
            origin = int(location[1])
            after_instruction = False
        if re.match(r'\s*/\*[0-9a-f]+\*/', line):
            if symbol and (origin in (1975, 1976, 1977, 1978, 444) or 'timed_silu_table_build' in symbol):
                selected.append(str((symbol, origin)) + ' ' + line.strip())
                row = link['sass'][symbol]['source_line_ops']
                row[str(origin)] = row.get(str(origin), 0) + 1
            after_instruction = True
    (folder / 'codegen.sass.txt').write_text('\n'.join(selected) + '\n')
    receipt['records'].append(link)
calls, routes = capture(source)
receipt['routes'] = routes
receipt['coverage'] = dict(main=coverage(final_native), helper=coverage(bindings(source)['_MA_ALT_SRCS']['moe_helper0']))
budget = json.loads((ROOT / 'docs/steering/evidence/2026-11-01-silu-roi/conditional-budget.json').read_text())
affected = {'decode 32', 'verify 16', 'verify 32'}
def factor(fraction):
    loads = {}
    for cell in budget['cells']:
        row = loads.setdefault(cell['load'], [0.0, 0.0])
        row[0] += cell['weight'] * cell['adjusted_us']
        row[1] += cell['weight'] * (cell['adjusted_us'] - (fraction * cell['expert_profile_us'] if cell['load'] in affected else 0))
    product = 1.0
    for old, new in loads.values():
        product *= old / new
    return product ** (1 / 6)
lower, upper = 0.0, 0.5
for unused in range(80):
    middle = (lower + upper) / 2
    if factor(middle) < 1.01:
        lower = middle
    else:
        upper = middle
receipt['conditional_ROI'] = dict(affected_loads=sorted(affected), required_uniform_affected_expert_fraction_for_1percent_score=upper,
    scenarios=[dict(expert_fraction=value, whole_layer_factor=factor(value)) for value in (0.01, 0.03, 0.05, 0.1)],
    limitation='Arithmetic only, assumes zero replacement cost and full critical-path exposure; NOT a forecast')
(OUT / 'final-link-codegen-roi.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(dict(prefixes_equal=True, ptx=[row['ptx'] for row in receipt['records']], ROI=receipt['conditional_ROI']), indent=2))
