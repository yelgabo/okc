# Revival round 2 sweep 1 on ly_rv2s (= ly_stk4 with launch-time knobs): attention dynamic shared memory per
# (B, L, sliding), the T = 1 cgemv threads of (1024, 4096) and a T = 1 finish reservation, decode 1 / verify 1.
# Sweep 1b: attention reservation at decode 16 / 32 full.
cd /root/okc && . tools/box_env.sh
mkdir -p .build/rev2
AS() { echo "{\"glob\": {\"ASMEM_CFG\": {($1, $2, $3): $4}}}"; }
F1() { echo "{\"glob\": {\"V1FS_T\": {8: 118784, 1: $1}}}"; }
CG() { echo "{\"glob\": {\"CG_NTH_OVR\": {(1024, 4096): $1}}}"; }
COMMON=(--runs 48 --router-seed 106218864389449120 --judge .build/judge/g1b4_pdl8d.f931abf568544018.json
  --lock /root/okc/.build/box/gpu.lock --ls-cache /root/okc/.build/box/ls_cache.json --ext-cache /root/.cache/bx_ext
  --skip-pdl8-gate)
A=(--arm "base={}"
   --arm "a1s100=$(AS 1 1 True 102400)" --arm "a1s160=$(AS 1 1 True 163840)" --arm "a1s227=$(AS 1 1 True 232448)"
   --arm "a1f100=$(AS 1 1 False 102400)" --arm "a1f160=$(AS 1 1 False 163840)"
   --arm "a8s100=$(AS 1 8 True 102400)" --arm "a8s160=$(AS 1 8 True 163840)"
   --arm "a8f100=$(AS 1 8 False 102400)" --arm "a8f160=$(AS 1 8 False 163840)"
   --arm "cg64=$(CG 64)" --arm "cg256=$(CG 256)"
   --arm "f1_60k=$(F1 61440)" --arm "f1_118k=$(F1 118784)" --arm "f1_160k=$(F1 163840)" --arm "base2={}")
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_rv2s.py --loads "decode 1,verify 1" "${COMMON[@]}" \
  "${A[@]}" --out .build/rev2/arms_s1.json > .build/rev2/arms_s1.log 2>&1
echo "$(date -u +%T) arms s1 rc=$?"
B=(--arm "base={}"
   --arm "d16f100=$(AS 16 1 False 102400)" --arm "d16f160=$(AS 16 1 False 163840)"
   --arm "d32f100=$(AS 32 1 False 102400)" --arm "d32f160=$(AS 32 1 False 163840)"
   --arm "d16s120=$(AS 16 1 True 122880)" --arm "d32s120=$(AS 32 1 True 122880)" --arm "base2={}")
python tools/ly_arms.py muse_decode_layer muse_decode_layer/cand/ly_rv2s.py --loads "decode 16,decode 32" "${COMMON[@]}" \
  "${B[@]}" --out .build/rev2/arms_s1b.json > .build/rev2/arms_s1b.log 2>&1
echo "$(date -u +%T) arms s1b rc=$?"
