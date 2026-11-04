cd /root/okc && . tools/box_env.sh
F=${F:?}; TAG=${TAG:?}; LOADS=${LOADS:-verify 16,verify 32}
LY_TRACE_DIR=".build/d16r2/tl_$TAG" python tools/ly_tl_run.py muse_decode_layer "muse_decode_layer/cand/$F" \
  --loads "$LOADS" --runs 20 --kernels --pin-pdl8 1,8,16,32 --skip-pdl8-gate \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --out ".build/d16r2/tl_$TAG.json" > ".build/d16r2/tl_$TAG.log" 2>&1
echo "$(date -u +%T) tl $TAG rc=$?"
