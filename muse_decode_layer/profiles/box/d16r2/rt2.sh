cd /root/okc && . tools/box_env.sh
RT() { echo "{\"glob\": {\"RT_CFG\": {16: ($1, $2, 0), 32: ($3, $4, 0)}}}"; }
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_rbc.py --loads "decode 16,decode 32" --runs 96 \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate --arm "base={}" --arm "rt4=$(RT 4 1 4 1)" --arm "rt2=$(RT 2 1 2 1)" --arm "rtp2=$(RT 8 2 8 2)" \
  --arm "rt4p2=$(RT 4 2 4 2)" --arm "base2={}" --out .build/d16r2/arms_rt2.json > .build/d16r2/arms_rt2.log 2>&1
echo "$(date -u +%T) arms rt2 rc=$?"
