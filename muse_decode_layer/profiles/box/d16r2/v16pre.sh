cd /root/okc && . tools/box_env.sh
ON="\"v16pdl_on\": {128: True, 256: True}"
E() { echo "{\"state\": {$ON, \"v16pdl_edges\": {128: {$1}, 256: {$1}}}}"; }
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_v16pre.py --loads "verify 16,verify 32" --runs 48 \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate --arm "off={\"state\": {\"v16pdl_on\": {}}}" --arm "all={\"state\": {$ON}}" \
  --arm "rb=$(E "\"b1\", \"b2\"")" --arm "off2={\"state\": {\"v16pdl_on\": {}}}" \
  --out .build/d16r2/arms_v16pre.json > .build/d16r2/arms_v16pre.log 2>&1
echo "$(date -u +%T) arms v16pre rc=$?"
