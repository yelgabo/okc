cd /root/okc && . tools/box_env.sh
C=(--judge .build/judge/g1b4_pdl8d.f931abf568544018.json --router-seed 106218864389449120 --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext)
echo "$(date -u +%T) tl16 start"
python tools/ly_box_moe_tl16g.py muse_decode_layer/cand/ly_stk1_tl16.py --loads "decode 16" --inputs 40 --raw 6 "${C[@]}" --out .build/moe3/tl16_stk1.json > .build/moe3/tl16_stk1.log 2>&1
echo "$(date -u +%T) tl16 rc=$?"
python tools/ly_box_moe_tlb4g.py muse_decode_layer/cand/ly_stk1_tlb4.py --loads "decode 32" --inputs 40 --raw 6 "${C[@]}" --out .build/moe3/tlb4_stk1.json > .build/moe3/tlb4_stk1.log 2>&1
echo "$(date -u +%T) tlb4 rc=$?"
