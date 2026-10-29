#!/bin/sh
# Compile gate for SK_PDL: base source (HEAD), new source with SK_PDL=0 and SK_PDL=1.
set -eu
G=$(cd "$(dirname "$0")" && pwd); W=/Users/gretchengabo/GitHub/okc/.claude/worktrees/agent-ae5a38858ad8cfbaf/.build/gate_sk; mkdir -p "$W"
REPO=/Users/gretchengabo/GitHub/okc/.claude/worktrees/agent-ae5a38858ad8cfbaf
cd "$REPO"
git show HEAD:muse_bf16_proj/submission.py > "$W/base_submission.py"
uv run --with torch python "$G/gen_sk.py" "$W/base_submission.py" "$W/base.cu"
uv run --with torch python "$G/gen_sk.py" muse_bf16_proj/submission.py "$W/new.cu"
docker run --rm --platform linux/arm64 -v "$W:/w" nvidia/cuda:12.8.1-devel-ubuntu24.04 sh -c '
cd /w
for v in base:base.cu:0 new0:new.cu:0 new1:new.cu:1; do
  n=${v%%:*}; r=${v#*:}; f=${r%%:*}; d=${r#*:}
  nvcc -O3 -gencode=arch=compute_90a,code=sm_90a -DSK_PDL=$d -Xptxas -v -cubin $f -o $n.cubin > $n.ptxas 2>&1 || echo "FAIL $n"
  cuobjdump -sass $n.cubin > $n.sass
done' 2>&1 | tail -20
