#!/bin/bash
# layer-d16 round 2 on the box: in-process arms of ly_v16pdl at verify 16 / 32 (tools/ly_arms.py): PDL off, all four
# T = 128 / 256 edges, each edge alone, all but the attention edge. FILE / TAG / RUNS / LOADS override.
#   ssh okc-h200 'cd /root/okc && (nohup bash tools/ly_v16_arms.sh > .build/d16r2/arms_v16.run 2>&1 < /dev/null &)'
cd /root/okc && . tools/box_env.sh
FILE=${FILE:-ly_v16pdl.py}; TAG=${TAG:-v16}; RUNS=${RUNS:-48}; LOADS=${LOADS:-verify 16,verify 32}
ON='"v16pdl_on": {128: True, 256: True}'
E() { echo "{\"state\": {$ON, \"v16pdl_edges\": {128: {$1}, 256: {$1}}}}"; }
mkdir -p .build/d16r2
python tools/ly_arms.py muse_decode_layer "muse_decode_layer/cand/$FILE" --loads "$LOADS" --runs "$RUNS" \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate \
  --arm 'off={"state": {"v16pdl_on": {}}}' \
  --arm "all={\"state\": {$ON}}" \
  --arm "attn=$(E '"attn"')" \
  --arm "b1=$(E '"b1"')" \
  --arm "route=$(E '"route"')" \
  --arm "b2=$(E '"b2"')" \
  --arm "noattn=$(E '"b1", "route", "b2"')" \
  --out ".build/d16r2/arms_$TAG.json" > ".build/d16r2/arms_$TAG.log" 2>&1
echo "$(date -u +%T) arms $TAG rc=$?"
