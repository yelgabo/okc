controls: c_a c_b c_c c_z

ly_micro: runs ly_micro_a ly_micro_b ly_micro_c
| cell | c_a | c_b | c_c | c_z | ly_micro_a | ly_micro_b | ly_micro_c | spread | delta (means) | per run | threshold | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 sliding | 65.89 | 65.89 | 66.38 | 66.59 | 66.80 | 66.54 | 66.26 | 0.70 | +0.35 | +0.61 +0.36 +0.07 | 1.41 | inside |
| decode 1 full | 72.32 | 72.54 | 72.46 | 72.54 | 72.13 | 73.30 | 72.35 | 0.22 | +0.12 | -0.34 +0.83 -0.12 | 0.50 | inside |
| verify 1 sliding | 102.67 | 102.03 | 102.82 | 102.90 | 102.80 | 101.90 | 102.94 | 0.86 | -0.05 | +0.20 -0.70 +0.34 | 1.73 | inside |
| verify 1 full | 111.82 | 112.13 | 112.75 | 112.80 | 112.30 | 111.82 | 112.27 | 0.98 | -0.24 | -0.07 -0.55 -0.10 | 1.95 | inside |

ly_micro2: runs ly_micro2_a ly_micro2_b ly_micro2_c
| cell | c_a | c_b | c_c | c_z | ly_micro2_a | ly_micro2_b | ly_micro2_c | spread | delta (means) | per run | threshold | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 sliding | 65.89 | 65.89 | 66.38 | 66.59 | 66.78 | 66.14 | 66.13 | 0.70 | +0.16 | +0.60 -0.04 -0.06 | 1.41 | inside |
| decode 1 full | 72.32 | 72.54 | 72.46 | 72.54 | 72.02 | 72.35 | 72.16 | 0.22 | -0.29 | -0.45 -0.12 -0.31 | 0.50 | inside |
| verify 1 sliding | 102.67 | 102.03 | 102.82 | 102.90 | 102.94 | 102.83 | 102.93 | 0.86 | +0.30 | +0.34 +0.23 +0.32 | 1.73 | inside |
| verify 1 full | 111.82 | 112.13 | 112.75 | 112.80 | 112.46 | 111.87 | 112.21 | 0.98 | -0.19 | +0.09 -0.50 -0.17 | 1.95 | inside |

checks not passing: none
