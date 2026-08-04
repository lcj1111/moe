# Q-TopoMoE Phase 1 BF16/FP8 statistical analysis

Completed means `failed == 0` and `completed == requests`; the legacy `acceptance` flag is reported but not used to discard measurements.

- Matrix rows: 78 (completed: 78, failed/incomplete: 0)
- acceptance=true rows: 0

## Grouped medians

| Format | Topology | Workload | C | Runs | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 | tp4_numa0 | medium | 1 | 1 | 487.96 | 4.23 | 1236.28 |
| bf16 | tp4_numa0 | medium | 32 | 1 | 1625.68 | 12.14 | 4255.19 |
| bf16 | tp4_numa0 | medium | 8 | 1 | 537.17 | 6.25 | 2040.00 |
| bf16 | tp4_numa0 | short | 1 | 1 | 125.95 | 3.85 | 615.03 |
| bf16 | tp4_numa0 | short | 32 | 1 | 1071.77 | 10.50 | 2306.31 |
| bf16 | tp4_numa0 | short | 8 | 1 | 359.80 | 5.62 | 1037.13 |
| bf16 | tp4_numa1 | medium | 1 | 1 | 487.28 | 3.99 | 1193.23 |
| bf16 | tp4_numa1 | medium | 32 | 1 | 1678.00 | 11.66 | 4202.28 |
| bf16 | tp4_numa1 | medium | 8 | 1 | 500.56 | 6.21 | 1991.00 |
| bf16 | tp4_numa1 | short | 1 | 1 | 112.54 | 3.88 | 605.28 |
| bf16 | tp4_numa1 | short | 32 | 1 | 1187.65 | 10.13 | 2406.24 |
| bf16 | tp4_numa1 | short | 8 | 1 | 399.85 | 5.34 | 1071.26 |
| bf16 | tp8_sys | medium | 1 | 1 | 478.16 | 4.84 | 1335.59 |
| bf16 | tp8_sys | medium | 32 | 1 | 1561.54 | 12.44 | 4252.47 |
| bf16 | tp8_sys | medium | 8 | 1 | 570.20 | 6.77 | 2235.23 |
| bf16 | tp8_sys | short | 1 | 1 | 104.64 | 4.81 | 715.33 |
| bf16 | tp8_sys | short | 32 | 1 | 1178.43 | 9.83 | 2418.59 |
| bf16 | tp8_sys | short | 8 | 1 | 699.17 | 6.29 | 1492.45 |
| fp8 | tp2_node_0_2 | medium | 1 | 1 | 353.29 | 4.58 | 1484.68 |
| fp8 | tp2_node_0_2 | medium | 32 | 1 | 2960.63 | 7.45 | 4220.07 |
| fp8 | tp2_node_0_2 | medium | 8 | 1 | 533.26 | 5.70 | 1768.11 |
| fp8 | tp2_node_0_2 | short | 1 | 1 | 110.79 | 4.57 | 691.16 |
| fp8 | tp2_node_0_2 | short | 32 | 1 | 1879.17 | 6.30 | 2540.89 |
| fp8 | tp2_node_0_2 | short | 8 | 1 | 443.07 | 5.30 | 1109.61 |
| fp8 | tp2_node_1_3 | medium | 1 | 1 | 361.72 | 4.65 | 1509.10 |
| fp8 | tp2_node_1_3 | medium | 32 | 1 | 3198.04 | 8.16 | 4445.79 |
| fp8 | tp2_node_1_3 | medium | 8 | 1 | 536.68 | 5.73 | 1784.82 |
| fp8 | tp2_node_1_3 | short | 1 | 1 | 89.82 | 4.52 | 663.48 |
| fp8 | tp2_node_1_3 | short | 32 | 1 | 1836.45 | 6.56 | 2505.88 |
| fp8 | tp2_node_1_3 | short | 8 | 1 | 456.27 | 5.36 | 1130.70 |
| fp8 | tp2_node_4_6 | medium | 1 | 1 | 434.04 | 4.60 | 1570.33 |
| fp8 | tp2_node_4_6 | medium | 32 | 1 | 2928.07 | 8.13 | 4143.28 |
| fp8 | tp2_node_4_6 | medium | 8 | 1 | 526.41 | 5.77 | 1774.34 |
| fp8 | tp2_node_4_6 | short | 1 | 1 | 95.79 | 4.65 | 685.93 |
| fp8 | tp2_node_4_6 | short | 32 | 1 | 1998.38 | 6.50 | 2666.57 |
| fp8 | tp2_node_4_6 | short | 8 | 1 | 381.63 | 5.48 | 1041.53 |
| fp8 | tp2_node_5_7 | medium | 1 | 1 | 402.14 | 4.58 | 1534.44 |
| fp8 | tp2_node_5_7 | medium | 32 | 1 | 2836.08 | 7.86 | 4203.52 |
| fp8 | tp2_node_5_7 | medium | 8 | 1 | 602.53 | 5.98 | 1888.71 |
| fp8 | tp2_node_5_7 | short | 1 | 1 | 113.31 | 4.38 | 669.45 |
| fp8 | tp2_node_5_7 | short | 32 | 1 | 2047.94 | 6.35 | 2716.05 |
| fp8 | tp2_node_5_7 | short | 8 | 1 | 322.97 | 5.34 | 993.56 |
| fp8 | tp2_pix_0_1 | medium | 1 | 1 | 456.88 | 4.58 | 1588.87 |
| fp8 | tp2_pix_0_1 | medium | 32 | 1 | 3245.37 | 8.25 | 4551.00 |
| fp8 | tp2_pix_0_1 | medium | 8 | 1 | 555.33 | 5.86 | 1844.22 |
| fp8 | tp2_pix_0_1 | short | 1 | 1 | 105.69 | 4.45 | 670.84 |
| fp8 | tp2_pix_0_1 | short | 32 | 1 | 2118.59 | 8.15 | 2811.88 |
| fp8 | tp2_pix_0_1 | short | 8 | 1 | 414.24 | 5.54 | 1112.53 |
| fp8 | tp2_pix_2_3 | medium | 1 | 1 | 444.38 | 4.63 | 1588.73 |
| fp8 | tp2_pix_2_3 | medium | 32 | 1 | 3118.73 | 8.91 | 4479.59 |
| fp8 | tp2_pix_2_3 | medium | 8 | 1 | 584.92 | 6.05 | 1881.25 |
| fp8 | tp2_pix_2_3 | short | 1 | 1 | 115.79 | 4.49 | 685.76 |
| fp8 | tp2_pix_2_3 | short | 32 | 1 | 2115.09 | 7.80 | 2811.90 |
| fp8 | tp2_pix_2_3 | short | 8 | 1 | 398.45 | 5.57 | 1101.22 |
| fp8 | tp2_pix_4_5 | medium | 1 | 1 | 454.11 | 4.58 | 1585.34 |
| fp8 | tp2_pix_4_5 | medium | 32 | 1 | 3142.14 | 9.05 | 4484.07 |
| fp8 | tp2_pix_4_5 | medium | 8 | 1 | 675.46 | 6.19 | 2138.08 |
| fp8 | tp2_pix_4_5 | short | 1 | 1 | 105.77 | 4.45 | 670.63 |
| fp8 | tp2_pix_4_5 | short | 32 | 1 | 2136.46 | 7.38 | 2829.03 |
| fp8 | tp2_pix_4_5 | short | 8 | 1 | 383.08 | 5.55 | 1080.83 |
| fp8 | tp2_pix_6_7 | medium | 1 | 1 | 454.86 | 4.60 | 1591.03 |
| fp8 | tp2_pix_6_7 | medium | 32 | 1 | 3141.16 | 8.12 | 4369.36 |
| fp8 | tp2_pix_6_7 | medium | 8 | 1 | 577.46 | 5.90 | 1855.16 |
| fp8 | tp2_pix_6_7 | short | 1 | 1 | 106.34 | 4.47 | 674.32 |
| fp8 | tp2_pix_6_7 | short | 32 | 1 | 2085.95 | 7.42 | 2779.86 |
| fp8 | tp2_pix_6_7 | short | 8 | 1 | 389.37 | 5.54 | 1087.03 |
| fp8 | tp4_numa0 | medium | 1 | 1 | 449.07 | 4.80 | 1672.92 |
| fp8 | tp4_numa0 | medium | 32 | 1 | 1527.30 | 11.69 | 4024.90 |
| fp8 | tp4_numa0 | medium | 8 | 1 | 550.52 | 6.14 | 2039.64 |
| fp8 | tp4_numa0 | short | 1 | 1 | 103.25 | 4.54 | 679.84 |
| fp8 | tp4_numa0 | short | 32 | 1 | 1081.11 | 10.44 | 2317.60 |
| fp8 | tp4_numa0 | short | 8 | 1 | 521.16 | 5.69 | 1237.60 |
| fp8 | tp4_numa1 | medium | 1 | 1 | 449.46 | 4.62 | 1628.82 |
| fp8 | tp4_numa1 | medium | 32 | 1 | 1665.61 | 11.63 | 4095.90 |
| fp8 | tp4_numa1 | medium | 8 | 1 | 519.39 | 6.00 | 1983.40 |
| fp8 | tp4_numa1 | short | 1 | 1 | 104.65 | 4.49 | 675.45 |
| fp8 | tp4_numa1 | short | 32 | 1 | 1077.22 | 9.90 | 2290.71 |
| fp8 | tp4_numa1 | short | 8 | 1 | 334.60 | 5.64 | 1044.74 |

## Overlapping BF16/FP8 configurations

| Topology | Workload | C | FP8 E2E p95 delta | FP8 TTFT p95 delta |
|---|---|---:|---:|---:|
| tp4_numa0 | medium | 1 | +35.32% | -7.97% |
| tp4_numa0 | medium | 8 | -0.02% | +2.49% |
| tp4_numa0 | medium | 32 | -5.41% | -6.05% |
| tp4_numa0 | short | 1 | +10.54% | -18.03% |
| tp4_numa0 | short | 8 | +19.33% | +44.85% |
| tp4_numa0 | short | 32 | +0.49% | +0.87% |
| tp4_numa1 | medium | 1 | +36.51% | -7.76% |
| tp4_numa1 | medium | 8 | -0.38% | +3.76% |
| tp4_numa1 | medium | 32 | -2.53% | -0.74% |
| tp4_numa1 | short | 1 | +11.59% | -7.01% |
| tp4_numa1 | short | 8 | -2.48% | -16.32% |
| tp4_numa1 | short | 32 | -4.80% | -9.30% |
