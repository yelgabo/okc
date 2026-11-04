#!/bin/bash
# layer-d16 round 2 on the box: in-process arms of ly_unpf at decode 16 / 32 (tools/ly_arms.py), the number of
# unpooler stages the MoE prefetches into L2 (UNPF_STAGES[T], 0 = off) swept per T.
#   ssh okc-h200 'cd /root/okc && (nohup bash tools/ly_unpf_arms.sh > .build/d16r2/arms_unpf.run 2>&1 < /dev/null &)'
cd /root/okc && . tools/box_env.sh
FILE=${FILE:-ly_unpf.py}; TAG=${TAG:-unpf}; RUNS=${RUNS:-48}; LOADS=${LOADS:-decode 16,decode 32}
K() { echo "{\"glob\": {\"UNPF_STAGES\": {16: $1, 32: $1}}}"; }
mkdir -p .build/d16r2
python tools/ly_arms.py muse_decode_layer "muse_decode_layer/cand/$FILE" --loads "$LOADS" --runs "$RUNS" \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate \
  --arm "k0=$(K 0)" --arm "k2=$(K 2)" --arm "k4=$(K 4)" --arm "k6=$(K 6)" --arm "k8=$(K 8)" --arm "k12=$(K 12)" \
  --out ".build/d16r2/arms_$TAG.json" > ".build/d16r2/arms_$TAG.log" 2>&1
echo "$(date -u +%T) arms $TAG rc=$?"
