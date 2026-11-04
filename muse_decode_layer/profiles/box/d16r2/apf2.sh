cd /root/okc && . tools/box_env.sh
A() { echo "{\"glob\": {\"APF_BYTES\": {1: 4194304, 8: 4194304, 16: $1, 32: $1}}}"; }
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_apf.py --loads "decode 16,decode 32" --runs 96 \
  --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
  --skip-pdl8-gate --arm "m0=$(A 0)" --arm "m2=$(A 2097152)" --arm "m4=$(A 4194304)" --arm "m6=$(A 6291456)" --arm "m0b=$(A 0)" \
  --out .build/d16r2/arms_apf2.json > .build/d16r2/arms_apf2.log 2>&1
echo "$(date -u +%T) arms apf2 rc=$?"
