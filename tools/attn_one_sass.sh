#!/bin/sh
# Compile attn_one.cu instances for sm_90a (CUDA 12.8.1, Docker), print ptxas registers and
# spills, and per instance the highest register used inside the consumer region
# (USETMAXREG.TRY_ALLOC up to DEALLOC) and outside it, plus local-memory instructions. Leaves the
# SASS in .build/one/one.sass (or $ONE_WORK/one.sass). $ONE_DEFS passes extra nvcc flags, e.g.
# ONE_DEFS="-DONE_DIAG=0 -DONE_DIET=1".
#   tools/attn_one_sass.sh [cu file] [L,MODE ...]     (default: 8,51 8,19)
set -eu
cu=${1:-muse_attention_decode/attn_one.cu}
[ $# -gt 0 ] && shift
insts=${*:-8,51 8,19}
work=${ONE_WORK:-.build/one}
defs=${ONE_DEFS:-}
mkdir -p "$work"
python3 - "$cu" "$work/one.cu" $insts <<'PY'
import re, sys
text = open(sys.argv[1]).read()
text = "#define TORCH_CHECK(cond, ...) (void)(cond)\n" + re.sub(
    r'#include <(torch|ATen)/[^>]*>\n', '', text)
head, sep, tail = text.rpartition("}  // namespace")
inst = "".join(f"template struct Instance<{x}>;\n" for x in sys.argv[3:])
open(sys.argv[2], "w").write(head + inst + sep + "\n")
PY
docker run --rm --platform linux/arm64 -v "$PWD/$work:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c "nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a $defs -Xptxas -v -cubin \
        /w/one.cu -o /w/one.cubin > /w/ptxas.txt 2>&1; cuobjdump -sass /w/one.cubin > /w/one.sass" \
    > /dev/null 2>&1 || true
grep -E 'error|Compiling entry|registers|spill' "$work/ptxas.txt" | grep -v coop_probe \
    | sed -E 's/.*attn_one_kernelILi([0-9]+)ELi([0-9]+)E.*/attn_one_kernel<\1, \2>/; s/ptxas info *: //'
python3 - "$work/one.sass" <<'PY'
import re, sys
for f in re.split(r'\n\s+Function : ', open(sys.argv[1]).read())[1:]:
    name = f.split("\n", 1)[0]
    m = re.search(r'attn_one_kernelILi(\d+)ELi(\d+)E', name)
    if not m:
        continue
    inside, top_in, top_out, loc, n = False, 0, 0, 0, 0
    for line in f.split("\n"):
        if not re.search(r'/\*[0-9a-f]{4,}\*/ ', line):
            continue
        n += 1
        if "TRY_ALLOC" in line:
            inside = True
            continue
        if "DEALLOC" in line:
            inside = False
        if re.search(r'\b(LDL|STL)', line):
            loc += 1
        regs = [int(r) for r in re.findall(r"\bR(\d+)\b", line)]
        if regs:
            if inside:
                top_in = max(top_in, max(regs))
            else:
                top_out = max(top_out, max(regs))
    print(f"attn_one_kernel<{m.group(1)}, {m.group(2)}>: {n} instrs, consumer region top "
          f"R{top_in}, elsewhere R{top_out}, LDL/STL {loc}")
PY
