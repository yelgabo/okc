#!/usr/bin/env python3
"""Can four rotations of the DQ7 E2M1 bf16 field mask (0x81C081C0) partition a 32-bit code word?

If they could, a lossless repack would let every one of the four A-fragment registers come from one rotate and one
mask (8 ops per word instead of 9). Prints every rotation triple that, with rotation 0, covers all 32 bits exactly
once; there are none. docs/runs-decode-layer.md "Unit loop at issue rate (box)".
"""
import itertools

M = [6, 7, 8, 15, 22, 23, 24, 31]


def main():
    sols = []
    for rots in itertools.combinations(range(1, 32), 3):
        cover = set(M)
        for r in rots:
            src = {(m - r) % 32 for m in M}
            if cover & src:
                break
            cover |= src
        else:
            if len(cover) == 32:
                sols.append(rots)
    print(f"{len(sols)} partitions", sols[:20])


if __name__ == "__main__":
    main()
