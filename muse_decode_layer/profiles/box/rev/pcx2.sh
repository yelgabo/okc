# Revival lane sweep 2: confirm the sweep 1 leads at their own T only (o P=3 at T=8, qkv P=1 at T=16), plus the
# T = 32 entries (P=1 in ly_stk3) at P=2/3, both layers, 96 runs. Arms keyed to another T act as extra controls.
cd /root/okc && . tools/box_env.sh
mkdir -p .build/rev
Q() { echo "{\"pcx\": {(\"$1\", $2): ($3, 0, 0, 0)}}"; }
A=(--arm "base={}" --arm "o3_8=$(Q o 8 3)" --arm "qkv1_16=$(Q qkv 16 1)" --arm "pool3_16=$(Q pool 16 3)" --arm "router1_8=$(Q router 8 1)")
for r in qkv o pool; do A+=(--arm "${r}2_32=$(Q $r 32 2)"); done
A+=(--arm "base2={}")
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_stk3.py --loads "verify 1,decode 16,decode 32" \
  --runs 96 --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate "${A[@]}" --out .build/rev/arms_pcx2.json > .build/rev/arms_pcx2.log 2>&1
echo "$(date -u +%T) arms pcx2 rc=$?"
