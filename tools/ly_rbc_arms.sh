#!/bin/bash
# layer-d16 round 2 on the box: in-process arms of ly_rbc at decode 16 / 32 (tools/ly_arms.py): the Gluon row
# boundary's (V, NW) and the route's (num_warps, PARTS, MODE) at T = 16 / 32, one table changed per arm.
#   ssh okc-h200 'cd /root/okc && (nohup bash tools/ly_rbc_arms.sh > .build/d16r2/arms_rbc.run 2>&1 < /dev/null &)'
cd /root/okc && . tools/box_env.sh
FILE=${FILE:-ly_rbc.py}; TAG=${TAG:-rbc}; RUNS=${RUNS:-48}; LOADS=${LOADS:-decode 16,decode 32}
RB() { echo "{\"glob\": {\"RB_CFG\": {(16, True): ($1, $2), (32, False): ($1, $2), (32, True): ($1, $2)}}}"; }
RT() { echo "{\"glob\": {\"RT_CFG\": {16: ($1, $2, $3), 32: ($1, $2, $3)}}}"; }
mkdir -p .build/d16r2
python tools/ly_arms.py muse_decode_layer "muse_decode_layer/cand/$FILE" --loads "$LOADS" --runs "$RUNS" \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate \
  --arm 'base={}' \
  --arm "rb4x16=$(RB 4 16)" --arm "rb8x8=$(RB 8 8)" --arm "rb8x16=$(RB 8 16)" --arm "rb8x4=$(RB 8 4)" \
  --arm "rt4=$(RT 4 1 0)" --arm "rt16=$(RT 16 1 0)" --arm "rtp2=$(RT 8 2 0)" --arm "rtm1=$(RT 8 1 1)" \
  --out ".build/d16r2/arms_$TAG.json" > ".build/d16r2/arms_$TAG.log" 2>&1
echo "$(date -u +%T) arms $TAG rc=$?"
