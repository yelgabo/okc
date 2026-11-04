"""Revival round 2 sweep 2 on ly_rv2s: MoE TINY (decode 1) / TINY2 (verify 1) define variants as in-process arms
(MOE_X replaces the build's defines). Skips docs/ruled-out.md rows: TINY2 L2H=0 (row 230), NS, nw16, HALF, BCLM,
TINY2 WSBLK 2 (runs-moe tiny2wsblk2)."""
import subprocess
import sys

TINY = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_L2H=0", "-DOKC_NACC=4",
        "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=2")
TINY2 = ("-DOKC_NW=8", "-DOKC_TINY=1", "-DOKC_CPF=0", "-DOKC_EPI4=0", "-DOKC_FAST1=2", "-DOKC_NACC=4",
         "-DOKC_UWARP=0", "-DOKC_WS=1", "-DOKC_WSB=1", "-DOKC_WSBL=2", "-DOKC_WSBLK=1", "-DOKC_FINPF=1")


def setd(base, name, val):
    out = [d for d in base if not d.startswith(f"-DOKC_{name}=")]
    return tuple(out + [f"-DOKC_{name}={val}"])


V = {
    "t_wsblk1": ("tiny", setd(TINY, "WSBLK", 1)), "t_wsblk4": ("tiny", setd(TINY, "WSBLK", 4)),
    "t_finpf1": ("tiny", setd(TINY, "FINPF", 1)), "t_cpf0": ("tiny", setd(TINY, "CPF", 0)),
    "t_l2h1": ("tiny", setd(TINY, "L2H", 1)), "t_wsbl1": ("tiny", setd(TINY, "WSBL", 1)),
    "t_fast3": ("tiny", setd(TINY, "FAST1", 3)),
    "t2_cpf1": ("tiny2", setd(TINY2, "CPF", 1)), "t2_finpf0": ("tiny2", setd(TINY2, "FINPF", 0)),
    "t2_wsbl1": ("tiny2", setd(TINY2, "WSBL", 1)), "t2_fast3": ("tiny2", setd(TINY2, "FAST1", 3)),
    "t2_wsblk4": ("tiny2", setd(TINY2, "WSBLK", 4)),
}
arms = ["--arm", "base={}"]
for name, (key, d) in V.items():
    arms += ["--arm", f"{name}={{'glob': {{'MOE_X': {{'{key}': {d!r}}}}}}}"]
arms += ["--arm", "base2={}"]
cmd = [sys.executable, "tools/ly_arms.py", "muse_decode_layer", "muse_decode_layer/cand/ly_rv2s.py",
       "--loads", "decode 1,verify 1", "--runs", "48", "--router-seed", "106218864389449120",
       "--judge", ".build/judge/g1b4_pdl8d.f931abf568544018.json", "--lock", "/root/okc/.build/box/gpu.lock",
       "--ls-cache", "/root/okc/.build/box/ls_cache.json", "--ext-cache", "/root/.cache/bx_ext", "--skip-pdl8-gate",
       *arms, "--out", ".build/rev2/arms_s2.json"]
with open(".build/rev2/arms_s2.log", "w") as f:
    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
print("arms s2 rc", rc)
