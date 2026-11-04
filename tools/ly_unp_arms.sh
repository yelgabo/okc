#!/bin/bash
# layer-d16 round 2 on the box: in-process arms of the unpooler sk2's pre-wait weight stages P (PCX_CFG[("un", T)])
# at decode 16 (NSTAGE 5, base P 2) and decode 32 (NSTAGE 3, base P 1), with the MoE -> unpooler edge programmatic
# (ly_stk1). Arms for one T are nulls at the other.
#   ssh okc-h200 'cd /root/okc && (nohup bash tools/ly_unp_arms.sh > .build/d16r2/arms_unp.run 2>&1 < /dev/null &)'
cd /root/okc && . tools/box_env.sh
FILE=${FILE:-ly_stk1.py}; TAG=${TAG:-unp}; RUNS=${RUNS:-48}; LOADS=${LOADS:-decode 16,decode 32}
P() { echo "{\"pcx\": {(\"un\", $1): ($2, 0, 0, 0)}}"; }
mkdir -p .build/d16r2
python tools/ly_arms.py muse_decode_layer "muse_decode_layer/cand/$FILE" --loads "$LOADS" --runs "$RUNS" \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate \
  --arm 'base={}' \
  --arm "d16p1=$(P 16 1)" --arm "d16p3=$(P 16 3)" --arm "d16p4=$(P 16 4)" --arm "d16p5=$(P 16 5)" \
  --arm "d32p2=$(P 32 2)" --arm "d32p3=$(P 32 3)" \
  --out ".build/d16r2/arms_$TAG.json" > ".build/d16r2/arms_$TAG.log" 2>&1
echo "$(date -u +%T) arms $TAG rc=$?"
