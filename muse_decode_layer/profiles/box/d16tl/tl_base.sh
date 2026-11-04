cd /root/okc && . tools/box_env.sh
COMMON=(--lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
        --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json)
echo "$(date -u +%T) start"
LY_TRACE_DIR=.build/d16tl/tl_v1un python tools/ly_tl_run.py muse_decode_layer muse_decode_layer/cand/ly_v1un.py \
  --loads "decode 16,decode 32" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate "${COMMON[@]}" \
  --out .build/d16tl/tl_v1un.json > .build/d16tl/tl_v1un.log 2>&1
echo "$(date -u +%T) rc=$?"
