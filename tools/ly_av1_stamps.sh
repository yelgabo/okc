#!/bin/bash
# Attention-v1 round 2: verify 1 sliding attention stamps (tools/ly_box_atst.py) for each stamped file in FILES, one
# after another under gpu.lock. Logs and JSONs in .build/av1r2/atst_<file stem>.{log,json}.
#   ssh okc-h200 'cd /root/okc && FILES="ly_atst_s1.py ly_atst_s1n17.py" setsid nohup tools/ly_av1_stamps.sh \
#     > .build/av1r2/stamps.log 2>&1 < /dev/null &'
cd /root/okc && . tools/box_env.sh
mkdir -p .build/av1r2
for f in ${FILES:?FILES="a.py b.py"}; do
  t=${f%.py}
  python tools/ly_box_atst.py "muse_decode_layer/cand/$f" --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --router-seed 106218864389449120 --inputs "${INPUTS:-20}" --raw 2 --layers "${LAYERS:-sliding}" \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --out ".build/av1r2/atst_$t.json" > ".build/av1r2/atst_$t.log" 2>&1
  echo "$(date -u +%T) atst_$t rc=$?"
done
echo "$(date -u +%T) done"
