"""Generate device-only WGS decoder twins and summarize their nvdisasm -g output.

This compiler experiment does not produce a runnable submission. A different
word permutation must be used when packing weights for the second decoder.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


WGS_SHA = 'c5ececf50cb8893ca6189cda42a051228b4a8576faf71142cccfa0b5917c3fa4'
SCATTER = '''  constexpr uint32_t M = 0x81C081C0u;
  r0 = q & M;
  r1 = __funnelshift_l(q, q, 10) & M;
  r2 = __funnelshift_l(q, q, 13) & M;
  r3 = ((q & 0x001b001bu) * 16648u) & M;'''


def verify_mapping():
    mask = 0x81c081c0
    rotated = [((mask >> shift) | (mask << (32 - shift))) & 0xffffffff for shift in (0, 10, 13)]
    scatter_mask = 0x001b001b
    assert sum(word.bit_count() for word in [*rotated, scatter_mask]) == 32
    assert rotated[0] | rotated[1] | rotated[2] | scatter_mask == 0xffffffff
    positions = [1 << i for i in range(32) if (1 << i) & scatter_mask]
    outputs = [(bit * 16648) & mask for bit in positions]
    assert all(value.bit_count() == 1 for value in outputs) and len(set(outputs)) == 8
    # The multiplication must behave like wiring even when several inputs are
    # set. Check all inputs including carries from the low half to the high half.
    for subset in range(256):
        indices = [i for i in range(8) if subset & (1 << i)]
        word = sum(positions[i] for i in indices)
        assert ((word * 16648) & mask) == sum(outputs[i] for i in indices)


def generate(layer, out):
    verify_mapping()
    node = next(n for n in ast.parse(layer.read_text('utf8')).body
                if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'MOE_CUDA_SRC')
    cuda = ast.literal_eval(node.value)
    assert hashlib.sha256(cuda.encode()).hexdigest() == WGS_SHA
    start = cuda.index('  constexpr uint32_t M = ', cuda.index('void dq4('))
    end = cuda.index('\n}', start)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ctrl.cu').write_bytes(cuda.encode())
    (out / 'scatter.cu').write_bytes((cuda[:start] + SCATTER + cuda[end:]).encode())


def census(sass, cuda):
    """Count static instructions per entry. Source attribution is only used for integer decode."""
    lines = cuda.read_text('utf8').splitlines()
    begin = next(i for i, line in enumerate(lines, 1) if 'void dq4(' in line)
    end = next(i for i in range(begin + 1, len(lines) + 1) if lines[i - 1] == '}')
    entries = {}
    name = None
    location = 0
    for line in sass.read_text('utf8').splitlines():
        entry = re.match(r'\s*\.type\s+(_ZN3okc9moe_fused\w+),@function', line)
        if entry:
            name = entry[1]
            entries[name] = {'instructions': 0, 'selected_ops': Counter(), 'decoder_integer': Counter()}
        source = re.match(r'\s*//## File "([^"]+)", line (\d+)', line)
        if source:
            location = int(source[2]) if source[1].endswith('exact_device.cu') else 0
        inst = re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\d+\s+)?([\w.]+)(?:\s+([^;]*?))?\s*;', line)
        if not (name and inst):
            continue
        entry_stats = entries[name]
        entry_stats['instructions'] += 1
        opcode = inst[1].split('.')[0]
        if opcode in ('HGMMA', 'FENCE', 'BAR', 'SYNCS', 'WARPGROUP'):
            label = inst[1] if 'DEPBAR' not in inst[1] else f'{inst[1]} {inst[2]}'
            entry_stats['selected_ops'][label] += 1
        if begin <= location <= end and opcode in ('IMAD', 'LOP3', 'SHF'):
            entry_stats['decoder_integer'][opcode] += 1
    assert len(entries) == 2
    return entries


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('layer', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--census', action='store_true', help='also read NAME.sass for each twin')
    args = parser.parse_args()
    generate(args.layer, args.out)
    if args.census:
        print(json.dumps({tag: census(args.out / (tag + '.sass'), args.out / (tag + '.cu'))
                          for tag in ('ctrl', 'scatter')}, indent=2))
