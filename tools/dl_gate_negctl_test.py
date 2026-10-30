"""Offline test of the gate's negative controls: GATE_SRC from tools/dl_gate_patch.py on
synthetic rows of the layer's hidden width at BF16-level control noise, many seeds, with the
unrestricted single-element draw and with the M1 gate's elem_floor. Counts misses per kind.

Two row models: dense (every row live, like the layer's hidden and stream outputs) and sparse,
like a MoE output: each bank entry's row is exactly zero with probability ZERO_P (a token with
no local route; top-8 of 256 experts with 32 local gives (224 choose 8) / (256 choose 8), about
0.34), and zero rows carry no noise. The sparse model reproduces the route gates' 'row' misses
at 32 tokens in d6c30f88c617438b when run against the tool before the row fix.

    uv run --with torch python tools/dl_gate_negctl_test.py tools [TRIALS]
"""
import sys
import torch

sys.path.insert(0, sys.argv[1])
import dl_gate_patch as gp  # noqa: E402

ns = {"torch": torch, "_FAILED": []}
exec(gp.GATE_SRC, ns)
negctl, gate_acc = ns["_gate_negctl"], ns["_gate_acc"]
refs_of, gate_add = ns["_gate_refs"], ns["_gate_add"]

HIDDEN = 8192
ZERO_P = 0.34
torch.manual_seed(0)
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 200
fails = 0
for model in ("dense", "sparse"):
    for rows in (1, 8, 16, 32):
        for floor in (None, 0.125):
            misses = {}
            for t in range(trials):
                bank = [torch.randn(rows, HIDDEN) * 3 for _ in range(5)]
                if model == "sparse":
                    for b in bank:
                        b[torch.rand(rows) < ZERO_P] = 0
                refs = refs_of(bank)
                ctrl = gate_acc(1, "cpu")
                for k in range(20):
                    ref = bank[k % 5]
                    noisy = ref + 3e-3 * torch.randn(rows, HIDDEN) * (ref != 0)
                    gate_add(ctrl, [noisy], [refs], torch.tensor([k % 5]))
                obs = bank[0] + 3e-3 * torch.randn(rows, HIDDEN) * (bank[0] != 0)
                _, miss = negctl(obs.to(torch.bfloat16), refs, 0, 1, ctrl, 55 + t, floor)
                for m in miss:
                    misses[m] = misses.get(m, 0) + 1
            if "row" in misses or "none" in misses:
                fails += 1
            print(f"{model} rows {rows} elem_floor {floor}: misses over {trials} trials "
                  f"{misses or 'none'}")
print("row and none never missed" if not fails else f"row or none missed in {fails} settings")
sys.exit(1 if fails else 0)
