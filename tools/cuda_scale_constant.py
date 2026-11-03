"""Remove one opaque zero dependency from WGS scale reconstruction."""
import argparse
import json
from pathlib import Path

from cuda_issue_groups import compile_pair, generate


def transform(native):
    old = "const uint32_t k16c = (OKC_SC6W ? 4u : 16u) + p.zero;  // opaque 16 (4 with SC6W): the scale decode's multiply stays an IMAD (FMA pipe)"
    new = 'const uint32_t k16c = (OKC_SC6W ? 4u : 16u);'
    if native.count(old) != 1 or native.count('  p.zero = 0;') != 1:
        raise ValueError('WGS scale/host zero source drift')
    return native.replace(old, new), [(old, new, 1)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()
    print(json.dumps(generate(args.out, transform), indent=2), flush=True)
    if args.compile:
        compile_pair(args.out)
