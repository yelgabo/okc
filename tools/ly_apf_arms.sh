#!/bin/bash
# layer-d16 round 2 on the box: in-process arms of ly_apf at decode 16 / 32 (tools/ly_arms.py): the pooler stream
# bytes the attention prefetches into L2 at T = 16 / 32 (0 = the base: none).
#   ssh okc-h200 'cd /root/okc && (nohup bash tools/ly_apf_arms.sh > .build/d16r2/arms_apf.run 2>&1 < /dev/null &)'
cd /root/okc && . tools/box_env.sh
FILE=${FILE:-ly_apf.py}; TAG=${TAG:-apf}; RUNS=${RUNS:-48}; LOADS=${LOADS:-decode 16,decode 32}
A() { echo "{\"glob\": {\"APF_BYTES\": {1: 4194304, 8: 4194304, 16: $(( $1 << 20 )), 32: $(( $1 << 20 ))}}}"; }
mkdir -p .build/d16r2
python tools/ly_arms.py muse_decode_layer "muse_decode_layer/cand/$FILE" --loads "$LOADS" --runs "$RUNS" \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate \
  --arm "m0=$(A 0)" --arm "m4=$(A 4)" --arm "m8=$(A 8)" --arm "m12=$(A 12)" --arm "m16=$(A 16)" --arm "m24=$(A 24)" \
  --arm "m0b=$(A 0)" \
  --out ".build/d16r2/arms_$TAG.json" > ".build/d16r2/arms_$TAG.log" 2>&1
echo "$(date -u +%T) arms $TAG rc=$?"
