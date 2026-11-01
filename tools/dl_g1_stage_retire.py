"""Retire each WGS activation slot after its own stage drains. Family G experiment."""
import argparse
import ast
import hashlib
from pathlib import Path


SIGNATURE = 'template <bool PB, int NTT>\n__device__ __forceinline__ void wgs_seg('
SEGMENT_SHA = '223b0525779c3023e3db3ba524deca01f750c4a34e9f61affcc20bef9e45cfbd'
G1B4_SHA = '1607293eee3403109df5d37b6e2d1ee34b528c3578afa6c2c8668440afde1658'
RELEASE = r'''    // All reads of this activation slot are complete before the ring can reuse it.
    wg_wait<0>();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    mbar_arrive1(S.bars + BX_E + xs);'''


def patch_device(cuda, switch=False):
    """Accept the common, exact segment shared by the repaired layer and component probe."""
    if cuda.count(SIGNATURE) != 1:
        raise ValueError('expected one WGS segment')
    begin = cuda.index(SIGNATURE)
    end = cuda.index('\n}\n', begin) + 2
    previous = cuda[begin:end]
    if hashlib.sha256(previous.encode()).hexdigest() != SEGMENT_SHA:
        raise ValueError('unreviewed WGS segment')
    # The original implementation releases X for the previous stage at ci==0
    # and handles the final stage separately. The new loop retires every stage.
    first_release = previous.index('        if (ci == 0) {')
    next_work = previous.index('        // the previous segment', first_release)
    revised = previous[:first_release] + previous[next_work:]
    final_drain = revised.index('  wg_wait<0>();\n', revised.index('  stage(std::true_type{})'))
    epilogue = revised.index('  if (OKC_WGSX', final_drain)
    revised = revised[:final_drain] + '  wg_wait<0>();\n' + revised[epilogue:]
    revised = revised.replace('  int prev = -1;  // x slot the last committed group may still read\n', '')
    revised = revised.replace('    prev = xs;', RELEASE)
    if 'BX_E + prev' in revised or revised.count('mbar_arrive1(S.bars + BX_E + xs)') != 1:
        raise ValueError('unexpected activation-slot release')
    if switch:
        revised = '#if defined(OKC_STAGE_RETIRE) && OKC_STAGE_RETIRE\n' + revised + '\n#else\n' + previous + '\n#endif'
    return cuda[:begin] + revised + cuda[end:]


def build_layer(source):
    if hashlib.sha256(source.encode()).hexdigest() != G1B4_SHA:
        raise ValueError('expected the pure G1b4 layer')
    literal = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and ast.unparse(node.targets[0]) == 'MOE_CUDA_SRC')
    device = ast.literal_eval(literal.value)
    lines = source.splitlines(keepends=True)
    assignment = ''.join(lines[literal.lineno - 1:literal.end_lineno])
    if assignment.count(device) != 1:
        raise ValueError('expected verbatim raw CUDA literal')
    lines[literal.lineno - 1:literal.end_lineno] = [assignment.replace(device, patch_device(device))]
    revised = ''.join(lines)
    header = '# Modified by yelgabo: G1b4,'
    if revised.count(header) != 1:
        raise ValueError('expected attribution header')
    revised = revised.replace(header, '# Modified by yelgabo: G1b4 stage-retire experiment,')
    compile(revised, '<stage retirement layer>', 'exec')
    return revised.encode()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pure_g1b4', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    payload = build_layer(args.pure_g1b4.read_text('utf8'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(len(payload), hashlib.sha256(payload).hexdigest())
