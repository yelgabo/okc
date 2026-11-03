"""Compile the helper0 device source of a candidate file for sm_90a with the tiny / tiny2 defines (docker nvcc 12.8.1).

Local observation only: register counts, spills, SASS opcode counts. Not H200 parity.
    python3 tl_compile.py <candidate.py> <out_dir> [tiny|tiny2|mid]
"""
import ast, json, subprocess, sys
from pathlib import Path

src_path, out = Path(sys.argv[1]), Path(sys.argv[2])
key = sys.argv[3] if len(sys.argv) > 3 else 'tiny'
out.mkdir(parents=True, exist_ok=True)
text = src_path.read_text()
tree = ast.parse(text)
ns = {}
want = {'MOE_TINY_DEFINES', 'MOE_TINY2_DEFINES', 'MOE_MID_DEFINES', '_MA_ALT_SRCS', 'DQ7', '_MA_SFX_IF_PRMT'}
keep = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') in want for t in n.targets)]
keep += [n for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Subscript)
         and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == '_MA_ALT_SRCS']
exec(compile(ast.Module(body=keep, type_ignores=[]), 'cand', 'exec'), ns)
defs = {'tiny': ns['MOE_TINY_DEFINES'], 'tiny2': ns['MOE_TINY2_DEFINES'], 'mid': ns['MOE_MID_DEFINES']}[key]
src = None
flags = []
for d in defs:
    if d.startswith('@'):
        src = ns['_MA_ALT_SRCS'][d[1:]]
    else:
        flags.append(d)
flags += ['-DOKC_NS=3', '-DOKC_FNST=4', '-DOKC_NSF=3', '-DOKC_DQ7=1', '-DOKC_PRMTSC=1']
device = src.split('#include <torch/extension.h>')[0]
device += '''
void compile_entries(okc::Params p) {
  okc::moe_fused<false><<<1,okc::NTL>>>(p);
  okc::moe_fused<true><<<1,okc::NTL>>>(p);
}
#include <cstdio>
int main() {
  const size_t base = ((sizeof(okc::Smem)+127)/128)*128;
  std::printf("{\\"smem_base\\":%zu,\\"threads\\":%d}\\n", base, (int)okc::NTL);
}
'''
(out / 'device.cu').write_text(device)
nv = ['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo', '-std=c++17', '-Xptxas=-v,--warn-on-spills'] + flags
docker = ['docker', 'run', '--rm', '--platform', 'linux/arm64', '--entrypoint', '', '-v', str(out.resolve()) + ':/w',
          'nvidia/cuda:12.8.1-devel-ubuntu24.04']
proc = subprocess.run(docker + ['nvcc', *nv, '/w/device.cu', '-o', '/w/device'], capture_output=True, text=True)
(out / 'device.log').write_text(proc.stdout + proc.stderr)
print('nvcc exit', proc.returncode, 'key', key, 'flags', ' '.join(flags))
for line in (proc.stdout + proc.stderr).splitlines():
    if any(k in line for k in ('error', 'warning', 'Compiling entry', 'registers', 'spill', 'C75')):
        print(line.replace('ptxas info    : ', ''))
if proc.returncode:
    sys.exit(proc.returncode)
sass = subprocess.run(docker + ['cuobjdump', '-sass', '/w/device'], capture_output=True, text=True, check=True).stdout
(out / 'device.sass').write_text(sass)
import collections, re
counts = collections.Counter()
cur = None
for line in sass.splitlines():
    m = re.match(r'\s+Function : (\S+)', line)
    if m:
        cur = m.group(1); continue
    m = re.match(r'\s+/\*[0-9a-f]+\*/\s+(@!?P\d\s+)?([A-Z0-9_.]+)', line)
    if m and cur:
        op = m.group(2).split('.')[0]
        counts[(cur, 'instr')] += 1
        if op in ('MEMBAR', 'BAR', 'FENCE', 'UBLKCP', 'SYNCS', 'ATOMG', 'REDG', 'LDL', 'STL', 'S2R', 'S2UR', 'CS2R', 'ERRBAR'):
            counts[(cur, op)] += 1
summary = {}
for (fn, k), v in sorted(counts.items()):
    summary.setdefault(fn[:60], {})[k] = v
print(json.dumps(summary, indent=1))
(out / 'sass_counts.json').write_text(json.dumps(summary, indent=1))
