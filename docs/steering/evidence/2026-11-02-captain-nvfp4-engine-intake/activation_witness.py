import json
import sys

import torch
import torch.nn.functional as functional

gate = torch.linspace(-8, 8, 1025).to(torch.bfloat16)
up = torch.tensor([0.125, 0.15625, 0.5, 0.75, 1.0, 1.0078125, 1.5, 3.0],
                  dtype=torch.bfloat16)
gate_grid = gate[:, None].expand(-1, up.numel())
up_grid = up[None, :].expand(gate.numel(), -1)
sigmoid_product = (gate_grid.float() / (1 + torch.exp(-gate_grid.float()))
                   * up_grid.float()).to(torch.bfloat16)
single_round = (functional.silu(gate_grid.float()) * up_grid.float()).to(torch.bfloat16)
extra_round = up_grid * functional.silu(gate_grid.float()).to(torch.bfloat16)
different = extra_round != single_round
attributable = different & (sigmoid_product == single_round)
positions = attributable.nonzero()
assert positions.numel() > 0
row, column = positions[0].tolist()
result = {'torch_version': torch.__version__, 'device': 'cpu',
          'sample_pairs': gate_grid.numel(),
          'different_after_extra_round': int(different.sum()),
          'different_with_original_formula_agreement': int(attributable.sum()),
          'witness': {'gate_bf16': float(gate_grid[row, column]),
                      'up_bf16': float(up_grid[row, column]),
                      'silu_fp32': float(functional.silu(gate_grid[row, column].float())),
                      'single_round_bf16': float(single_round[row, column]),
                      'original_formula_bf16': float(sigmoid_product[row, column]),
                      'extra_round_bf16': float(extra_round[row, column])},
          'limitations': ['Independent transcription of scalar boundaries; no foreign module executed.',
                          'Not an exhaustive BF16 domain proof, target result or official rejection.',
                          'No whole-operator norm, GEMM or performance test.']}
json.dump(result, sys.stdout, indent=2)
sys.stdout.write('\n')
