# Accuracy protocol — llama31_h100

| arm | mmlu | gsm8k | all | trunc% | fail% |
|---|---|---|---|---|---|
| acc1 | 66.9 | 85.7 | 76.3 | 1.8 | 1.1 |
| acc2_keep11520 | 58.4 | 54.1 | 56.2 | 2.1 | 1.4 |
| acc2_keep12288 | 59.2 | 67.6 | 63.4 | 1.7 | 1.9 |
| acc2_keep12800 | 64.1 | 70.4 | 67.2 | 1.6 | 1.1 |
| acc2_keep13568 | 63.3 | 79.3 | 71.3 | 1.9 | 0.8 |
| acc2_keep14080 | 67.1 | 83.6 | 75.3 | 1.4 | 1.1 |
| acc2_modelopt_keep11520 | 55.8 | 64.9 | 60.4 | 3.3 | 1.7 |
| acc2_modelopt_keep12288 | 60.3 | 72.1 | 66.2 | 2.2 | 1.4 |
| acc2_modelopt_keep12800 | 62.1 | 78.6 | 70.3 | 2.2 | 1.4 |
| acc2_modelopt_keep13568 | 63.2 | 81.7 | 72.5 | 1.8 | 1.1 |
| acc2_modelopt_keep14080 | 67.4 | 83.2 | 75.3 | 1.3 | 0.9 |
| acc3_b16_g00 | 66.7 | 86.1 | 76.4 | 1.8 | 1.1 |
| acc3_b16_g01 | 67.8 | 86.5 | 77.1 | 1.9 | 0.9 |
| acc3_b16_g02 | 67.3 | 85.8 | 76.5 | 2.0 | 0.9 |
| acc3_b16_g03 | 67.8 | 85.7 | 76.8 | 2.1 | 1.1 |
| acc3_b16_g04 | 67.9 | 86.4 | 77.1 | 1.8 | 1.0 |
| acc3_b16_g05 | 66.5 | 85.4 | 75.9 | 1.7 | 1.1 |
| acc3_b16_g06 | 66.9 | 85.2 | 76.0 | 2.1 | 0.9 |
| acc3_b16_g07 | 67.0 | 84.8 | 75.9 | 1.7 | 1.1 |
| acc3_b16_g08 | 66.1 | 86.2 | 76.1 | 1.9 | 1.1 |
| acc3_b16_g09 | 68.1 | 85.8 | 77.0 | 1.9 | 0.8 |

## Claims (mmlu)

- Acc1 (B=1 full) = 66.90
- Acc3 over 10 groupings: mean 67.21, std 0.64, min 66.10, max 68.10
- thresholds: min = 66.10, mean-std = 66.57

- acc2_keep11520: 58.40 | >min: fail | >mean-std: fail
- acc2_keep12288: 59.20 | >min: fail | >mean-std: fail
- acc2_keep12800: 64.10 | >min: fail | >mean-std: fail
- acc2_keep13568: 63.30 | >min: fail | >mean-std: fail
- acc2_keep14080: 67.10 | >min: PASS | >mean-std: PASS
- acc2_modelopt_keep11520: 55.80 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: 60.30 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: 62.10 | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: 63.20 | >min: fail | >mean-std: fail
- acc2_modelopt_keep14080: 67.40 | >min: PASS | >mean-std: PASS

## Claims (gsm8k)

- Acc1 (B=1 full) = 85.70
- Acc3 over 10 groupings: mean 85.79, std 0.51, min 84.80, max 86.50
- thresholds: min = 84.80, mean-std = 85.28

- acc2_keep11520: 54.10 | >min: fail | >mean-std: fail
- acc2_keep12288: 67.60 | >min: fail | >mean-std: fail
- acc2_keep12800: 70.40 | >min: fail | >mean-std: fail
- acc2_keep13568: 79.30 | >min: fail | >mean-std: fail
- acc2_keep14080: 83.60 | >min: fail | >mean-std: fail
- acc2_modelopt_keep11520: 64.90 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: 72.10 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: 78.60 | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: 81.70 | >min: fail | >mean-std: fail
- acc2_modelopt_keep14080: 83.20 | >min: fail | >mean-std: fail

## Claims (all)

- Acc1 (B=1 full) = 76.30
- Acc3 over 10 groupings: mean 76.50, std 0.46, min 75.90, max 77.15
- thresholds: min = 75.90, mean-std = 76.04

- acc2_keep11520: 56.25 | >min: fail | >mean-std: fail
- acc2_keep12288: 63.40 | >min: fail | >mean-std: fail
- acc2_keep12800: 67.25 | >min: fail | >mean-std: fail
- acc2_keep13568: 71.30 | >min: fail | >mean-std: fail
- acc2_keep14080: 75.35 | >min: fail | >mean-std: fail
- acc2_modelopt_keep11520: 60.35 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: 66.20 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: 70.35 | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: 72.45 | >min: fail | >mean-std: fail
- acc2_modelopt_keep14080: 75.30 | >min: fail | >mean-std: fail

## Paired flips vs Acc1 (McNemar exact)

- acc2_keep11520: n=2000, lost 541, gained 140, p=0.0000
- acc2_keep12288: n=2000, lost 386, gained 128, p=0.0000
- acc2_keep12800: n=2000, lost 322, gained 141, p=0.0000
- acc2_keep13568: n=2000, lost 233, gained 133, p=0.0000
- acc2_keep14080: n=2000, lost 141, gained 122, p=0.2670
- acc2_modelopt_keep11520: n=2000, lost 466, gained 147, p=0.0000
- acc2_modelopt_keep12288: n=2000, lost 362, gained 160, p=0.0000
- acc2_modelopt_keep12800: n=2000, lost 258, gained 139, p=0.0000
- acc2_modelopt_keep13568: n=2000, lost 205, gained 128, p=0.0000
- acc2_modelopt_keep14080: n=2000, lost 144, gained 124, p=0.2458
- acc3_b16_g00: n=2000, lost 67, gained 69, p=0.9317
- acc3_b16_g01: n=2000, lost 53, gained 70, p=0.1488
- acc3_b16_g02: n=2000, lost 58, gained 63, p=0.7163
- acc3_b16_g03: n=2000, lost 61, gained 70, p=0.4847
- acc3_b16_g04: n=2000, lost 60, gained 77, p=0.1714
- acc3_b16_g05: n=2000, lost 70, gained 63, p=0.6030
- acc3_b16_g06: n=2000, lost 65, gained 60, p=0.7207
- acc3_b16_g07: n=2000, lost 72, gained 64, p=0.5485
- acc3_b16_g08: n=2000, lost 80, gained 77, p=0.8732
- acc3_b16_g09: n=2000, lost 53, gained 66, p=0.2712
