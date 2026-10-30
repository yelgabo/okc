"""Offline test of the gate's negative controls: GATE_SRC from tools/dl_gate_patch.py on
synthetic rows of the layer's hidden width at BF16-level control noise, many seeds, with the
unrestricted single-element draw and with the M1 gate's elem_floor. Counts misses per kind.

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
torch.manual_seed(0)
trials = int(sys.argv[2]) if len(sys.argv) > 2 else 200
for rows in (1, 8, 16, 32):
    for floor in (None, 0.125):
        misses = {}
        for t in range(trials):
            bank = [torch.randn(rows, HIDDEN) * 3 for _ in range(5)]
            refs = refs_of(bank)
            ctrl = gate_acc(1, "cpu")
            for k in range(20):
                noisy = bank[k % 5] + 3e-3 * torch.randn(rows, HIDDEN)
                gate_add(ctrl, [noisy], [refs], torch.tensor([k % 5]))
            obs = (bank[0] + 3e-3 * torch.randn(rows, HIDDEN)).to(torch.bfloat16)
            _, miss = negctl(obs, refs, 0, 1, ctrl, 55 + t, floor)
            for m in miss:
                misses[m] = misses.get(m, 0) + 1
        print(f"rows {rows} elem_floor {floor}: misses over {trials} trials {misses or 'none'}")
