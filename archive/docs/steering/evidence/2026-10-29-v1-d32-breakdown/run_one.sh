#!/bin/sh
# Compile gate for ONE_TRIG: attn_one.cu at HEAD, new with ONE_TRIG=0 and ONE_TRIG=1, all 13 instances.
set -eu
G=$(cd "$(dirname "$0")" && pwd); W=/Users/gretchengabo/GitHub/okc/.claude/worktrees/agent-ae5a38858ad8cfbaf/.build/gate_one
mkdir -p "$W"
REPO=/Users/gretchengabo/GitHub/okc/.claude/worktrees/agent-ae5a38858ad8cfbaf
cd "$REPO"
git show HEAD:muse_attention_decode/attn_one.cu > "$W/base_src.cu"
for pair in "base_src.cu base.cu" ; do :; done
python3 "$G/prep_one.py" "$W/base_src.cu" "$W/base.cu"
python3 "$G/prep_one.py" muse_attention_decode/attn_one.cu "$W/new.cu"
docker run --rm --platform linux/arm64 -v "$W:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 sh -c '
cd /w
for v in base:base.cu:0 new0:new.cu:0 new1:new.cu:1; do
  n=${v%%:*}; r=${v#*:}; f=${r%%:*}; d=${r#*:}
  nvcc -O3 -std=c++17 -gencode=arch=compute_90a,code=sm_90a -DONE_TRIG=$d -Xptxas -v -cubin $f -o $n.cubin > $n.ptxas 2>&1 || echo "FAIL $n"
  cuobjdump -sass $n.cubin > $n.sass
done' 2>&1 | tail -20
