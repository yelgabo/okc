#!/bin/sh
# Device compile of the f3 kernel of a layer file (bfp_CUDA_SRC + mbarrier helpers + bfp_MAP_SRC + _F3_SRC, the gle
# extension's translation unit without the sk2 kernels) for sm_90a in the okc-torch-headers image: ptxas registers,
# spills and smem, then the ordering-relevant SASS opcode counts (as tools/dl_mc_c1_sass.sh).
#   [SASS_OUT=file] sh tools/dl_f3_sass.sh muse_decode_layer/cand/g1b4_f3.py
set -eu
src=$1
mkdir -p .build && work=$(mktemp -d "$PWD/.build/f3.XXXXXX")
trap 'rm -rf "$work"' EXIT

python3 - "$src" "$work" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
blocks = dict(re.findall(r'^(\w+) = r"""(.*?)"""', text, re.S | re.M))
common = blocks["bfp_CUDA_SRC"].replace("@@MBAR@@", blocks["bfp_MBAR_HELPERS"]).replace("@@BODY@@", "")
open(f"{sys.argv[2]}/ext.cu", "w").write(common + blocks["bfp_MAP_SRC"] + blocks["_F3_SRC"])
PY
mkdir -p "$work/stub/c10/cuda/impl"
echo "#pragma once" > "$work/stub/c10/cuda/impl/cuda_cmake_macros.h"

docker run --rm --platform linux/arm64 -v "$work:/w" okc-torch-headers sh -c '
inc=$(python3 -c "import torch.utils.cpp_extension as c; print(\" \".join(\"-I\" + p for p in c.include_paths()))" 2>/dev/null)
inc="$inc -I/w/stub -I$(python3 -c "import sysconfig; print(sysconfig.get_paths()[\"include\"])")"
nvcc -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a -Xptxas -v \
    -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
    -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ --expt-relaxed-constexpr \
    -DTORCH_EXTENSION_NAME=dl_f3_check -DTORCH_API_INCLUDE_EXTENSION_H -D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler -fPIC $inc -c /w/ext.cu -o /w/ext.o > /w/nvcc.txt 2>&1 || { cat /w/nvcc.txt; exit 1; }
cuobjdump -sass /w/ext.o > /w/ext.sass'
grep -E "error|warning|Compiling entry|registers|spill|bytes smem" "$work/nvcc.txt" \
    | grep -v "#3189-D" | sed -E "s/ptxas info *: //"
if [ -n "${SASS_OUT:-}" ]; then cp "$work/ext.sass" "$SASS_OUT"; fi
awk '/Function :/ { f = $3; fs[f] = 1 }
     /^ *\/\*[0-9a-f]+\*\/ / && !/ NOP/ { n[f]++ }
     /FENCE.VIEW.ASYNC.S/ { fs_[f]++ }
     /FENCE.VIEW.ASYNC.G/ { fg[f]++ }
     /MEMBAR/ { mb[f]++ }
     /BAR.SYNC|BAR.ARV|BAR.RED/ { ba[f]++ }
     /ERRBAR/ { eb[f]++ }
     /CCTL/ { cc[f]++ }
     /ATOMG/ { ag[f]++ }
     /REDG/ { rg[f]++ }
     /NANOSLEEP/ { ns[f]++ }
     /UTMALDG/ { tm[f]++ }
     /UBLKCP/ { bk[f]++ }
     /LDL/ { ll[f]++ }
     /STL/ { sl[f]++ }
     END { for (k in fs) printf "%s instr %d FENCE.S %d FENCE.G %d MEMBAR %d BAR %d ERRBAR %d CCTL %d ATOMG %d REDG %d NANOSLEEP %d UTMALDG %d UBLKCP %d LDL %d STL %d\n",
                                 k, n[k], fs_[k], fg[k], mb[k], ba[k], eb[k], cc[k], ag[k], rg[k], ns[k], tm[k], bk[k], ll[k], sl[k] }' \
    "$work/ext.sass" | sort
