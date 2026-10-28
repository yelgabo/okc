"""Write muse_bf16_proj/bench.py: the submission plus alternative skinny-kernel geometries.

prepare() already times every candidate of a shape by cold-L2 graph replay and prints the
table, so a bench run is the submission with more candidates: the entries below are added to
SK_GEOM and listed in SK_EXTRA, and the autotuner's `tune` lines in the run log compare them
with the picks. `okc check muse_bf16_proj/bench.py` scores kernel_fn as usual.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# name: (MT, TOKR, NT, NWR, WSK, UPW, NSTAGE); the comment gives what the variant tests.
SWEEP_GEOM = {
    "qkv8b": (1, 8, 3, 1, 4, 1, 8),       # 256 k per stage, deeper ring
    "qkv16b": (1, 16, 3, 1, 4, 1, 8),
    "qkv32b": (2, 32, 3, 1, 4, 1, 6),
    "o8b": (1, 8, 2, 2, 2, 2, 4),         # two row warps x two k warps
    "pool8b": (1, 8, 2, 2, 2, 4, 5),      # R=32: 128 CTAs at one per SM
    "pool16b": (1, 16, 1, 2, 2, 4, 3),    # R=16: 256 CTAs at two per SM
    "pool32b": (2, 32, 1, 2, 2, 2, 4),    # R=16: 256 CTAs at two per SM
    "unp32b": (2, 32, 2, 2, 2, 2, 3),     # R=32: 256 CTAs at two per SM
}
SWEEP_ROUTER = {
    "router8b": (1, 8, 1, 1, 8, 1, 4),
    "router16b": (1, 16, 1, 1, 8, 1, 3),
    "router32b": (2, 32, 1, 1, 8, 1, 3),
}
SWEEP_USE = {
    ("qkv_gate", 8): ["qkv8b"], ("qkv_gate", 16): ["qkv16b"], ("qkv_gate", 32): ["qkv32b"],
    ("o", 8): ["o8b"],
    ("pooler", 8): ["pool8b"], ("pooler", 16): ["pool16b"], ("pooler", 32): ["pool32b"],
    ("unpooler", 32): ["unp32b"],
    ("router", 8): ["router8b"], ("router", 16): ["router16b"], ("router", 32): ["router32b"],
}


def main():
    src = (ROOT / "muse_bf16_proj/submission.py").read_text()
    geom = "".join(f'    "{k}": {v},\n' for k, v in SWEEP_GEOM.items())
    rgeom = "".join(f'    "{k}": {v},\n' for k, v in SWEEP_ROUTER.items())
    assert src.count("}\nROUTER_GEOM = {\n") == 1 and src.count("}\n# Which geometry") == 1
    src = src.replace("}\nROUTER_GEOM = {\n", geom + "}\nROUTER_GEOM = {\n", 1)
    src = src.replace("}\n# Which geometry", rgeom + "}\n# Which geometry", 1)
    assert src.count("SK_EXTRA = {}\n") == 1
    src = src.replace("SK_EXTRA = {}\n", f"SK_EXTRA = {SWEEP_USE!r}\n", 1)
    (ROOT / "muse_bf16_proj/bench.py").write_text(src)
    print(f"{len(SWEEP_GEOM) + len(SWEEP_ROUTER)} extra geometries")


if __name__ == "__main__":
    main()
