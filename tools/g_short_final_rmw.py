"""OFF-by-default, exact607 helper final-counter representation experiment."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

PIN = 'ba9e327cd7b96595ace4d0c1d0d78fe27c128f23ed0debcda97c7169fc3164ab'
HELPER = 'b8ed85438c06673f411ff4203080bfbacd0c0c2dc5cb624e6f10d4877fe7a716'
FLAG = 'OKC_SHORT_FINAL_RELAXED'
OLD = r'''    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;\n" : "=r"(old) : "l"(bc), "r"(1) : "memory");'''
NEW = '''#if defined(OKC_SHORT_FINAL_RELAXED) && OKC_SHORT_FINAL_RELAXED
#if !OKC_TINY || !OKC_WS || OKC_SPLIT || OKC_WGS
#error SHORT_FINAL_RELAXED_requires_unsplit_WS_Tiny
#endif
''' + OLD.replace('atom.acq_rel.gpu.global.add.s32', 'atom.relaxed.gpu.global.add.u32') + '\n#else\n' + OLD + '\n#endif'


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def helper(source):
    return next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "_MA_ALT_SRCS['moe_helper0']")


def build(source, enabled=False):
    """Return source and manifest; enabling is explicit, never a runtime selector."""
    source = source.replace('\r\n', '\n')
    assert sha(source) == PIN, 'unrecognized incumbent'
    node = helper(source)
    device = ast.literal_eval(node.value)
    assert sha(device) == HELPER and device.count(OLD) == 1
    assert device.count('"l"(fc), "r"(pd.nst)') == 2  # release RED and acq_rel ATOM
    patched = device.replace(OLD, NEW)
    assert patched.replace(NEW, OLD) == device
    lines = source.splitlines(keepends=True)
    assignment = "_MA_ALT_SRCS['moe_helper0'] = r\"\"\"" + patched + '\"\"\"\n'
    lines[node.lineno - 1:node.end_lineno] = [assignment]
    result = ''.join(lines)
    # Appended fixed source policy changes ONLY the two selected short modules.
    # MID still aliases this source but does not receive the opt-in flag.
    if enabled:
        result += ('\n# Private OFF-prototype opt-in: T1/T8 only, no runtime timing selector.\n'
                   f'MOE_TINY_DEFINES += ("-D{FLAG}=1",)\n'
                   f'MOE_TINY2_DEFINES += ("-D{FLAG}=1",)\n')
    manifest = dict(control_sha256=PIN, helper_control_sha256=HELPER,
                    helper_patched_sha256=sha(patched), source_sha256=sha(result),
                    bytes=len(result.encode()), enabled=enabled, gpu_execution=False,
                    scope='Tiny/Tiny2 final bc RMW only; all other finalizers unchanged')
    return result, manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--enable', action='store_true')
    args = parser.parse_args()
    source, manifest = build(args.source.read_text(encoding='utf-8'), args.enable)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding='utf-8', newline='\n')
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest))
