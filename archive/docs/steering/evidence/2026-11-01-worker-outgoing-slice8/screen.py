"""Freeze the one outgoing seam source pair and run its CPU tests; no CUDA calls."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tools'))
from g_integration_common import build as common
from g_short_outgoing_seam import build, MARKER, sha
from test_g_short_outgoing_seam import literals, namespace, state, original_factory_flags

OUT = Path(__file__).parent
SCRATCH = ROOT / '.build/worker-outgoing-slice8'
SCRATCH.mkdir(parents=True, exist_ok=True)
public = (ROOT / 'muse_decode_layer/cand/g1b4_stage_release.py').read_text(encoding='utf8')
base, common_manifest = common(public, True)
summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'evidence_base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'scope': 'One fixed source-only outgoing expert-to-unpool seam. No GPU, compiler, host link, CUDA import, measurement or qualification.',
           'common': common_manifest, 'arms': {}, 'GPU_qualified': False, 'host_compiled': False,
           'future_gates': ['actual target host compilation and both-arm resources', 'cooperative-primary captured programmatic edge and runtime support',
                            'fresh/empty ownership numerical correctness and graph replay', 'independently prepared paired whole-six measurement']}
native = {}
for label, enabled in (('off', False), ('on', True)):
    source, manifest = build(base, enabled)
    assert source.split(MARKER)[0] == base
    (SCRATCH / (label + '.py')).write_bytes(source.encode())
    (OUT / (label + '.py.gz')).write_bytes(gzip.compress(source.encode(), mtime=0))
    (OUT / (label + '.manifest.json')).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf8')
    scope, extension, captures, fallback = namespace(enabled)
    scope['prepare'](state())
    assert captures[0]['extra_cuda_cflags'] == original_factory_flags(0)
    cuda, cpp = scope['_os_native_sources']()
    native[label] = (cuda, cpp)
    summary['arms'][label] = dict(manifest, cuda_sha256=sha(cuda), cpp_sha256=sha(cpp), selected_flags=captures[0]['extra_cuda_cflags'], archive_sha256=hashlib.sha256((OUT / (label + '.py.gz')).read_bytes()).hexdigest())
assert native['off'] == native['on']
cuda, cpp = native['on']
(OUT / 'secondary-adapter.cu.gz').write_bytes(gzip.compress(cuda.encode(), mtime=0))
(OUT / 'secondary-adapter.cpp').write_bytes(cpp.encode())
raw_sk2 = literals()['bfp_SK2_SRC']
boundary = raw_sk2.index('static void launch_sk2(')
summary['secondary_device_prefix_sha256'] = sha(raw_sk2[:boundary])
summary['native_pair_source_identical'] = True
summary['off_on_single_delta'] = '_OS_ENABLED = False -> True; installation freezes policy0 or1073741826 in per-state tuple'
summary['native_host_delta'] = 'explicit seam_pdl parameter, attribute1 only in pdl-positive launch branch, single fixed slice8 dispatch; no setter exports/calls'
summary['stages'] = 2
summary['pacing'] = [0, 0, 0]
summary['launch_policy'] = {'incoming_route_to_expert': 'ordinary completion unchanged', 'expert': 'original checked cooperative source unchanged',
                            'secondary': 'role un, T1/T8 only', 'final_boundary': 'ordinary completion unchanged'}
run = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tools', '-p', 'test_g_short_outgoing_seam.py', '-v'], cwd=ROOT, capture_output=True, text=True)
(OUT / 'cpu-tests.txt').write_text(run.stdout + run.stderr, encoding='utf8')
summary['cpu_test_returncode'] = run.returncode
summary['cpu_test_scope'] = 'Actual final dense -> prior dense -> patched _bfp_gemm -> bfp_kernel_fn -> bfp__run -> patched bfp__run_sk2 under CPU tensor/native stubs; not full layer execution or native API test'
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf8')
assert run.returncode == 0
print(json.dumps(summary, indent=2))
