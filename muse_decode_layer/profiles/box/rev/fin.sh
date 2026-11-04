# Revival lane sweep 3: T <= 8 finish shape (SL_CLUSTER_CONFIG: warps, columns per program) and boundary/route tile
# shape (SL_TILE_CONFIG: warps, parts) at T = 1 / 8 on ly_stk3, decode 1 / verify 1, both layers, 48 runs.
cd /root/okc && . tools/box_env.sh
mkdir -p .build/rev
C() { echo "{\"glob\": {\"SL_CLUSTER_CONFIG\": {1: ($1, $2), 8: ($3, $4)}}}"; }
TL() { echo "{\"glob\": {\"SL_TILE_CONFIG\": {1: ($1, $2, True), 8: ($3, $4, True)}}}"; }
A=(--arm "base={}"
   --arm "f1nw4=$(C 4 256 16 1024)" --arm "f1nw16=$(C 16 256 16 1024)" --arm "f1bc128=$(C 8 128 16 1024)"
   --arm "f1bc512=$(C 8 512 16 1024)" --arm "f8bc512=$(C 8 256 16 512)" --arm "f8bc2048=$(C 8 256 16 2048)"
   --arm "t1nw8=$(TL 8 1 4 4)" --arm "t1p2=$(TL 4 2 4 4)" --arm "t8nw8=$(TL 4 1 8 4)" --arm "t8p2=$(TL 4 1 4 2)"
   --arm "t8p8=$(TL 4 1 4 8)" --arm "base2={}")
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_stk3.py --loads "decode 1,verify 1" \
  --runs 48 --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate "${A[@]}" --out .build/rev/arms_fin.json > .build/rev/arms_fin.log 2>&1
echo "$(date -u +%T) arms fin rc=$?"
