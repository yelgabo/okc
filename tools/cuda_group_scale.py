"""Test issue batching and constant scales as one coherent pipeline treatment."""
import argparse
import json
from pathlib import Path

from cuda_issue_groups import compile_pair, generate, transform as group_transform
from cuda_scale_constant import transform as scale_transform


def transform(native):
    native, scale_edits = scale_transform(native)
    native, group_edits = group_transform(native)
    return native, scale_edits + group_edits


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.out, transform), indent=2), flush=True)
    compile_pair(args.out, arms=('candidate',))
