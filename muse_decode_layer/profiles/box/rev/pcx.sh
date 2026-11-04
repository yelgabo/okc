# Revival lane sweep 1: pre-wait weight stages P per sk2 role at T = 1 / 8 / 16 on ly_stk3 (T = 32 entries untouched).
cd /root/okc && . tools/box_env.sh
mkdir -p .build/rev
P() { echo "{\"pcx\": {(\"$1\", 1): ($2, 0, 0, 0), (\"$1\", 8): ($2, 0, 0, 0), (\"$1\", 16): ($2, 0, 0, 0)}}"; }
A=(--arm "base={}")
for r in qkv o pool; do for p in 1 3 4; do A+=(--arm "$r$p=$(P $r $p)"); done; done
A+=(--arm "router1=$(P router 1)" --arm "router4=$(P router 4)" --arm "base2={}")
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_stk3.py --loads "decode 1,verify 1,decode 16" \
  --layers sliding --runs 48 --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate "${A[@]}" --out .build/rev/arms_pcx.json > .build/rev/arms_pcx.log 2>&1
echo "$(date -u +%T) arms pcx rc=$?"
