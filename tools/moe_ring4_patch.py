"""Construct bench-only mode87: mode51 with four fixed-stride x slots.

The existing ring byte budget stays fixed; larger x ring takes weight slots.
Fourth barriers use diagnostic-only S.dg[0][0:2], unused by mode51/87.
No shared layout or default production capacity changes.
"""
from pathlib import Path
import re
import sys


def patch(s):
    # Mode87 inherits every51 trait, including all fences and release behavior.
    s, n = re.subn(r'(__host__ __device__ constexpr \w+ \w+\(int mode\) \{)',
                   r'\1\n    if (mode == 87) mode = 51;', s)
    assert n >= 30, n
    anchor = '// Ring positions shared by the consumers and the producers:'
    assert s.count(anchor) == 1
    helpers = '''// Mode87: same total byte budget, four x slots. Extra barriers alias only
// diagnostic fields; mode87 has no light/consumer/stage instrumentation.
template <int MODE> constexpr int nsx() { return MODE == 87 ? 4 : kNSX; }
template <int MODE, bool FULL>
__device__ __forceinline__ uint64_t* xbar(SmemMain& S, int slot) {
    if constexpr (MODE == 87) {
        if (slot == 3) return reinterpret_cast<uint64_t*>(&S.dg[0][FULL ? 0 : 1]);
    }
    return FULL ? &S.xfull[slot] : &S.xempty[slot];
}

'''
    s = s.replace(anchor, helpers + anchor)
    # Helper's default expression must remain raw; substitute only later call sites.
    prefix, tail = s.split('// Ring positions shared by the consumers and the producers:', 1)
    tail, a = re.subn(r'&S\.xfull\[([^\]]+)\]', r'xbar<MODE, true>(S, \1)', tail)
    tail, b = re.subn(r'&S\.xempty\[([^\]]+)\]', r'xbar<MODE, false>(S, \1)', tail)
    assert a >= 3 and b >= 3, (a, b)
    s = prefix + '// Ring positions shared by the consumers and the producers:' + tail
    s = s.replace('__device__ __forceinline__ void ring_next_x(Ring& r) {',
                  'template <int MODE>\n__device__ __forceinline__ void ring_next_x(Ring& r) {')
    assert s.count('ring_next_x(rg);') == 7
    s = s.replace('ring_next_x(rg);', 'ring_next_x<MODE>(rg);')
    s = s.replace('if (++r.xs == kNSX)', 'if (++r.xs == nsx<MODE>())')
    s = s.replace('(kRingBytes - kNSX * xsz)', '(kRingBytes - nsx<MODE>() * xsz)')
    s = s.replace('for (int s = 0; s < kNSX; ++s)', 'for (int s = 0; s < nsx<MODE>(); ++s)')
    s = s.replace('|| abl(MODE) || MODE == 86', '|| abl(MODE) || MODE == 86 || MODE == 87')
    s = s.replace('51: "main-lean-grp-finss64",',
                  '51: "main-lean-grp-finss64", 87: "main-finss64-xring4",')
    return s


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(patch(Path(sys.argv[1]).read_text()))
