#!/bin/sh
# Compile one of the wgmma micro tests (default wg4.cu) for sm_90a in Docker and count, per
# kernel, the HGMMAs and the wgmma wait depths: a kernel whose every HGMMA is followed by
# "DEPBAR.LE gsb0, 0x0" is serialized. The tests pin down what makes ptxas serialize the fat
# path's wgmmas (results in docs/runs-moe.md, "MAIN path, wgmma pipelining").
#   tools/moe_wgmma_micro/run.sh [wg|wg2|wg3|wg4]
set -eu
name=${1:-wg4}
dir=$(cd "$(dirname "$0")" && pwd)
docker run --rm --platform linux/arm64 -v "$dir:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 \
    sh -c "cd /w && nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v -c $name.cu -o $name.o 2>&1 | grep -E 'C75|registers|error' ; cuobjdump -sass $name.o > $name.sass"
python3 - "$dir/$name.sass" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
parts = re.split(r"\n\s*Function : ", text)
for part in parts[1:]:
    name = part.split("\n", 1)[0]
    hg = len(re.findall(r"HGMMA", part))
    d0 = len(re.findall(r"DEPBAR\.LE gsb0, 0x0", part))
    d1 = len(re.findall(r"DEPBAR\.LE gsb0, 0x1", part))
    arr = len(re.findall(r"WARPGROUP\.ARRIVE", part))
    m = re.search(r"kILi(\d+)E", name)
    print(f"MODE {m.group(1) if m else name}: HGMMA {hg}, wait0 {d0}, wait1 {d1}, arrive {arr}")
PY
