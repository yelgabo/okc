"""Generate an owned T<=8 boundary candidate. Keep the PDL wait on every CTA."""
from pathlib import Path
import sys


def candidate(source):
    begin = source.index("    const float4 pv =", source.index(" boundary_kernel("))
    end = source.index("    const float sa[PER]", begin)
    body = source[begin:end]
    wait = '    asm volatile("griddepcontrol.wait;" ::: "memory");\n'
    assert body.count(wait) == 1 and body.count("block_sum(ss, red)") == 1
    body = body.replace(wait, "").replace("float br[PER] = {", "float live_branch[PER] = {")
    body = body.replace("ss += br[k] * br[k]", "ss += live_branch[k] * live_branch[k]")
    body += "    for (int k = 0; k < PER; ++k) br[k] = bf16r(live_branch[k] * rb);\n"
    guarded = wait + "    float br[PER] = {};\n    if (rank == 0) {\n"
    guarded += "\n".join("    " + line for line in body.splitlines()) + "\n    }\n"
    old = "rank == 0 ? bf16r(br[k] * rb) : 0.f"
    assert source.count(old) == 1
    return (source[:begin]+guarded+source[end:]).replace(old, "br[k]")


if __name__ == "__main__":
    path = Path("muse_decode_layer/cand/layer_attn31s.py")
    Path(sys.argv[1]).write_text(candidate(path.read_text(encoding="utf-8")),
                                 encoding="utf-8", newline="\n")
