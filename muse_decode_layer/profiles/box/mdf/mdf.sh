# moe-dfybf lane: ly_mdf MID data-flag hand-off against the ly_stk4 MID kernel (plain helper0), decode 16, two processes.
# Arm base swaps MOE_MID_DEFINES back to @moe_helper0 with its own ext cache, which is ly_stk4's MID build byte for byte.
cd /root/okc && . tools/box_env.sh
mkdir -p .build/mdf
B="('-DOKC_NW=8', '-DOKC_TINY=1', '-DOKC_SPLIT=1', '-DOKC_CPF=0', '-DOKC_EPI4=0', '-DOKC_FAST1=2', '-DOKC_NACC=4', '-DOKC_UWARP=0', '-DOKC_WS=1', '-DOKC_WSB=1', '-DOKC_WSBL=2', '-DOKC_WSBLK=2', '-DOKC_FINPF=1', '@moe_helper0')"
for r in 1 2; do
  python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_mdf.py --loads "decode 16" \
    --runs 48 --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json \
    --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext \
    --skip-pdl8-gate --arm "base={\"glob\": {\"MOE_MID_DEFINES\": $B}, \"state\": {\"ma_ext\": {}}}" --arm "df={}" \
    --arm "base2={\"glob\": {\"MOE_MID_DEFINES\": $B}, \"state\": {\"ma_ext\": {}}}" \
    --out .build/mdf/arms_r$r.json > .build/mdf/arms_r$r.log 2>&1
  echo "$(date -u +%T) round $r rc=$?" >> .build/mdf/done.txt
done
