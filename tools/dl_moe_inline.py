"""Keep the layer synchronized with admitted standalone MoE variants."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUB = ROOT / "muse_decode_layer/submission.py"
SRC = ROOT / "muse_moe_w4a16/submission.py"

NAMES = {"_CUDA_SRC", "_SKINNY_BLOCK", "SKINNY_VARIANTS", "MAIN_MODES", "MAIN_CANDIDATES",
         "MAIN_MIN_M", "W4_BITPOS"}
FUNCS = {"_wgmma_code", "_skinny_tag"}
REPACK = ("unit_rows", "repack_expert", "repack_expert_fat", "_to_u8_words", "_cat_last")
RENAME = (("repack_expert_fat", "moe1_repack_expert_fat"), ("repack_expert", "moe1_repack_expert"),
          ("unit_rows", "moe1_unit_rows"))
MAIN_NAMES = {0: "main-d2", 5: "main-d3", 16: "main-m2-ar2-ub", 20: "main-own",
              23: "main-own-dynb", 26: "main-dynb-fs", 28: "main-lean", 37: "main-lean-silu-bar",
              47: "main-lean-grp-nohoist", 51: "main-lean-grp-finss64",
              78: "main-lean-grp-fin4ps"}


LAYER_MODES = (0, 16, 26, 28, 37, 47, 51)


LAYER_MAIN_EXTRA = ((78, 64),)


LAYER_SKINNY_EXTRA = ((8, 4, 3),)
SKINNY_SUFFIX = {0: "", 3: "-er"}


LO_HELPER = (
    "__device__ __forceinline__ int ld_lo(const int* p) {\n"
    "    int v;\n"
    "    asm volatile(\"ld.global.nc.s32 %0, [%1];\" : \"=r\"(v) : \"l\"(p));\n"
    "    return v;\n"
    "}\n\n"
)


PDL_HELPER = (
    "template <typename Kern>\n"
    "void moe1_launch(Kern kern, int grid, int block, int smem, cudaStream_t stream,\n"
    "                 const Params& p) {\n"
    "    if (!p.pdl) {\n"
    "        kern<<<grid, block, smem, stream>>>(p);\n"
    "        return;\n"
    "    }\n"
    "    cudaLaunchConfig_t cfg = {};\n"
    "    cfg.gridDim = dim3(grid);\n"
    "    cfg.blockDim = dim3(block);\n"
    "    cfg.dynamicSmemBytes = smem;\n"
    "    cfg.stream = stream;\n"
    "    cudaLaunchAttribute attr[1];\n"
    "    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;\n"
    "    attr[0].val.programmaticStreamSerializationAllowed = 1;\n"
    "    cfg.attrs = attr;\n"
    "    cfg.numAttrs = 1;\n"
    "    C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, p));\n"
    "}\n\n"
    "__device__ __forceinline__ void prefetch_l2_hint(const void* src, unsigned bytes) {\n"
    "    asm volatile(\"cp.async.bulk.prefetch.L2.global [%0], %1;\" :: \"l\"(src), \"r\"(bytes)\n"
    "                 : \"memory\");\n"
    "}\n\n"
)


MAIN_PREAMBLE = (
    "    const int E = p.M * kTopK;\n"
    "    if (p.pdl) {\n"
    "        if (warp == kNWarpsM - 2 && lane == 0) {\n"
    "            constexpr unsigned kRelP = lean_stage(MODE) ? kNC * 32 : kNC;\n"
    "#pragma unroll\n"
    "            for (int s = 0; s < kNSWMax; ++s) {\n"
    "                mbar_init(&S.wfull[s], 1);\n"
    "                mbar_init(&S.wempty[s], kRelP);\n"
    "            }\n"
    "#pragma unroll\n"
    "            for (int s = 0; s < kNSX; ++s) {\n"
    "                mbar_init(&S.xfull[s], 1);\n"
    "                mbar_init(&S.xempty[s], kRelP);\n"
    "            }\n"
    "            asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");\n"
    "        }\n"
    "        if (tid == 0) {\n"
    "            for (int i = cta; i < p.nloc * 16; i += G) {\n"
    "                prefetch_l2_hint(p.w13f + (static_cast<size_t>(i % p.nloc) * 32 * 16\n"
    "                                           + i / p.nloc) * kUnitBytes, kUnitBytes);\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
    "    constexpr bool kDiag = is_diag(MODE);\n"
)


SKINNY_PREFETCH = (
    "        if (tid == 0) {\n"
    "            for (int i = cta; i < p.nloc * 4; i += G) {\n"
    "                prefetch_l2_hint(p.w13 + (static_cast<size_t>(i % p.nloc) * kUA\n"
    "                                          + i / p.nloc) * kUnitBytes, kUnitBytes);\n"
    "            }\n"
    "        }\n"
)
SKINNY_INIT_RE = (r"        if \(lane == 0\) \{\n#pragma unroll\n            for \(int s = 0; s < kNS; \+\+s\) \{\n"
                  r"                mbar_init\(&S\.full\[s\], 1\);\n.*?"
                  r"            asm volatile\(\"fence\.mbarrier_init\.release\.cluster;\" ::: \"memory\"\);\n"
                  r"        \}\n")

SKINNY_INIT_LINE = re.compile(r"^\s*(#pragma unroll|for \(int \w+ = 0; \w+ < \w+; \+\+\w+\) \{|"
                              r"mbar_init\(&S\.\w+\[[\w +]+\], \w+\);|if constexpr \(\w+\) \{|\}|"
                              r"if \(lane == 0\) \{|asm volatile\(\"fence\.mbarrier_init\.release\.cluster;\" "
                              r"::: \"memory\"\);)$")


def skinny_init_block(block):
    """Private barriers do not depend on producer writes, so initialize them before the PDL wait."""
    found = re.findall(SKINNY_INIT_RE, block, flags=re.S)
    if len(found) != 1:
        raise SystemExit(f"skinny mbarrier init block found {len(found)} times, need 1")
    init = found[0]
    for line in init.splitlines():
        if not SKINNY_INIT_LINE.match(line):
            raise SystemExit(f"unexpected line in the skinny mbarrier init block: {line!r}")
    block = block.replace(init, init.replace("        if (lane == 0) {\n",
                                             "        if (lane == 0 && !p.pdl) {\n", 1))
    wait = "    // the instruction does nothing.\n    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
    if block.count(wait) != 1:
        raise SystemExit("skinny griddepcontrol.wait anchor not found once")
    pre = ("    // the instruction does nothing.\n    if (p.pdl) {\n        if (warp == kNW) {\n"
           + "".join("    " + ln + "\n" for ln in init.splitlines()) + "        }\n"
           + SKINNY_PREFETCH + "    }\n    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n")
    return block.replace(wait, pre)


CUDA_PATCHES = [
    ("    int cost[2][9];\n};\n",
     "    int cost[2][9];\n"
     "    // The layer's local_start is an input tensor, read here instead of the host value.\n"
     "    const int* lo_p;\n"
     "    int pdl;\n};\n\n" + PDL_HELPER),
    ("    const int E = p.M * kTopK;\n"
     "    asm volatile(\"griddepcontrol.wait;\" ::: \"memory\");\n"
     "    constexpr bool kDiag = is_diag(MODE);\n", MAIN_PREAMBLE),
    ("    } else if (warp == kNWarpsM - 2) {\n        if (lane == 0) {\n"
     "            constexpr unsigned kRel = lean_stage(MODE) ? kNC * 32 : kNC;\n#pragma unroll\n"
     "            for (int s = 0; s < kNSWMax; ++s) {\n                mbar_init(&S.wfull[s], 1);\n"
     "                mbar_init(&S.wempty[s], kRel);\n            }\n#pragma unroll\n"
     "            for (int s = 0; s < kNSX; ++s) {\n                mbar_init(&S.xfull[s], 1);\n"
     "                mbar_init(&S.xempty[s], kRel);\n            }\n"
     "            asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");\n        }\n",
     "    } else if (warp == kNWarpsM - 2) {\n        if (lane == 0 && !p.pdl) {\n"
     "            constexpr unsigned kRel = lean_stage(MODE) ? kNC * 32 : kNC;\n#pragma unroll\n"
     "            for (int s = 0; s < kNSWMax; ++s) {\n                mbar_init(&S.wfull[s], 1);\n"
     "                mbar_init(&S.wempty[s], kRel);\n            }\n#pragma unroll\n"
     "            for (int s = 0; s < kNSX; ++s) {\n                mbar_init(&S.xfull[s], 1);\n"
     "                mbar_init(&S.xempty[s], kRel);\n            }\n"
     "            asm volatile(\"fence.mbarrier_init.release.cluster;\" ::: \"memory\");\n        }\n"),
    ("    moe_main_kernel<MODE><<<sms, kNTM, sizeof(SmemMain), stream>>>(p);\n",
     "    moe1_launch(moe_main_kernel<MODE>, sms, kNTM, static_cast<int>(sizeof(SmemMain)), stream,\n"
     "                p);\n"),
    ("__device__ __forceinline__ int ld_acquire(const int* p) {\n",
     LO_HELPER + "__device__ __forceinline__ int ld_acquire(const int* p) {\n"),
    ("                const int ev[4] = {v[k].x - p.lo, v[k].y - p.lo, v[k].z - p.lo, v[k].w - p.lo};\n",
     "                const int lo = ld_lo(p.lo_p);\n"
     "                const int ev[4] = {v[k].x - lo, v[k].y - lo, v[k].z - lo, v[k].w - lo};\n"),
]
SKINNY_PATCHES = [
    ("                const int e = ev[k] - p.lo;\n",
     "                const int e = ev[k] - ld_lo(p.lo_p);\n"),
    ("    moe_fused_kernel<SPLIT><<<sms, kNTL, sizeof(Smem), stream>>>(p);\n",
     "    moe1_launch(moe_fused_kernel<SPLIT>, sms, kNTL, static_cast<int>(sizeof(Smem)), stream,\n"
     "                p);\n"),
]


def patch(text, patches):
    for old, new in patches:
        if text.count(old) != 1:
            raise SystemExit(f"anchor found {text.count(old)} times, need 1: {old!r}")
        text = text.replace(old, new)
    return text


def load(text):
    tree = ast.parse(text)
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in FUNCS)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in NAMES for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "moe_submission", "exec"), ns)
    funcs = {n.name: ast.get_source_segment(text, n) for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in REPACK}
    missing = set(REPACK) - set(funcs)
    if missing:
        raise SystemExit(f"repack functions missing from the source: {sorted(missing)}")
    return ns, funcs


def device_code(ns):
    cuda = patch(ns["_CUDA_SRC"], CUDA_PATCHES)
    block = skinny_init_block(patch(ns["_SKINNY_BLOCK"], SKINNY_PATCHES))
    tag = ns["_skinny_tag"]
    skinny_variants = [v for v in ns["SKINNY_VARIANTS"] if v[2] == 0] + list(LAYER_SKINNY_EXTRA)
    main_modes = [m for m in ns["MAIN_MODES"] if m in ns["MAIN_CANDIDATES"] and m in LAYER_MODES]
    for m, _ in LAYER_MAIN_EXTRA:
        if f"mode == {m}" not in ns["_CUDA_SRC"]:
            raise SystemExit(f"LAYER_MAIN_EXTRA mode {m} is not defined in the kernel source")
        main_modes.append(m)
    if not main_modes:
        raise SystemExit(f"none of LAYER_MODES {LAYER_MODES} is a fat candidate in the source")
    skinny = "\n".join(block.replace("@TAG@", tag(nw, ns_, mode)).replace("@NW@", str(nw))
                       .replace("@NS@", str(ns_)).replace("@MODE@", str(mode))
                       for nw, ns_, mode in skinny_variants)
    cuda = cuda.replace("//@WGMMA@", ns["_wgmma_code"]()).replace("//@SKINNY@", skinny)
    end = "\n}  // namespace\n"
    if cuda.count(end) != 1:
        raise SystemExit(f"expected one anonymous-namespace close, found {cuda.count(end)}")
    head = cuda[:cuda.index(end)]
    if "p.lo" in head.replace("p.lo_p", ""):
        raise SystemExit("an unpatched use of Params::lo remains in the device code")
    head, n = re.subn(r"^namespace \{$", "namespace {\nnamespace moe1 {", head, count=1, flags=re.M)
    if n != 1:
        raise SystemExit("no 'namespace {' line in the kernel source")
    code = head.rstrip() + "\n\n}  // namespace moe1\n}  // namespace\n"
    variants = [(f"sk{nw}w{ns_}s{SKINNY_SUFFIX[mode]}", "skinny", f"sk_{tag(nw, ns_, mode)}::launch<1>")
                for nw, ns_, mode in skinny_variants]
    variants += [(MAIN_NAMES.get(m, f"main-m{m}"), "main", f"launch_main<{m}>") for m in main_modes]
    max_m = {MAIN_NAMES.get(m, f"main-m{m}"): below for m, below in LAYER_MAIN_EXTRA}
    return code, variants, int(ns["MAIN_MIN_M"]), max_m


def repack_code(funcs):
    out = []
    for name in REPACK:
        text = funcs[name]
        for old, new in RENAME:
            text = re.sub(rf"\b{old}\b", new, text)
        out.append(text)
    return "\n\n\n".join(out)


def replace_once(src, pattern, repl, what):
    new, n = re.subn(pattern, repl, src, count=1, flags=re.S | re.M)
    if n != 1:
        raise SystemExit(f"no {what} in submission.py")
    return new


def main():
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC
    text = src_path.read_text()
    ns, funcs = load(text)
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else SUB
    sub = target.read_text()
    layer_bitpos = ast.literal_eval(re.search(r"^W4_BITPOS = (\(.*?\n\))", sub, re.S | re.M).group(1))
    if tuple(layer_bitpos) != tuple(ns["W4_BITPOS"]):
        raise SystemExit("the layer's W4_BITPOS differs from the fused kernel's; copy it too")
    for helper in ("_perm", "_stack_last", "_to_i64"):
        if f"\ndef {helper}(" not in sub:
            raise SystemExit(f"the layer lacks the repack helper {helper}")
    code, variants, main_min, max_m = device_code(ns)
    sub = replace_once(sub, r'_MOE1_SRC = r""".*?"""', lambda _: f'_MOE1_SRC = r"""\n{code}"""',
                       "_MOE1_SRC block")
    sub = replace_once(sub, r"^_MOE1_VARIANTS = \(.*?\n\)$",
                       lambda _: "_MOE1_VARIANTS = (\n" + "".join(f"    {v!r},\n" for v in variants) + ")",
                       "_MOE1_VARIANTS table")
    sub = replace_once(sub, r"^_MOE1_MAIN_MIN_M = \d+$", f"_MOE1_MAIN_MIN_M = {main_min}",
                       "_MOE1_MAIN_MIN_M line")

    max_line = f"_MOE1_MAIN_MAX_M = {max_m!r}"
    if re.search(r"^_MOE1_MAIN_MAX_M = .*$", sub, re.M):
        sub = replace_once(sub, r"^_MOE1_MAIN_MAX_M = .*?$", lambda _: max_line,
                           "_MOE1_MAIN_MAX_M line")
    else:
        sub = replace_once(sub, r"^(_MOE1_MAIN_MIN_M = \d+)$", lambda mo: mo.group(1) + "\n" + max_line,
                           "_MOE1_MAIN_MIN_M line")
    tune_old = 'if _MOE1_VARIANTS[v][1] == "skinny" or bs >= _MOE1_MAIN_MIN_M]'
    tune_new = ('if _MOE1_VARIANTS[v][1] == "skinny"\n'
                '                          or _MOE1_MAIN_MIN_M <= bs < _MOE1_MAIN_MAX_M.get(\n'
                '                              _MOE1_VARIANTS[v][0], 1 << 30)]')
    if sub.count(tune_old) == 1:
        sub = sub.replace(tune_old, tune_new)
    elif max_m and sub.count(tune_new) != 1:
        raise SystemExit("the MoE tuner's candidate filter is not where expected")
    sub = replace_once(sub, r"# @MOE1_REPACK_BEGIN@\n.*?# @MOE1_REPACK_END@\n",
                       lambda _: "# @MOE1_REPACK_BEGIN@\n" + repack_code(funcs) + "\n# @MOE1_REPACK_END@\n",
                       "@MOE1_REPACK@ region")
    target.write_text(sub)
    print(f"inlined {len(code.splitlines())} device lines and {len(REPACK)} repack functions from "
          f"{src_path}; variants {[v[0] for v in variants]}, fat path from {main_min} tokens")


if __name__ == "__main__":
    main()
