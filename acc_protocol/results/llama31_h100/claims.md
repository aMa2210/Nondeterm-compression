# Accuracy protocol — llama31_h100

| arm | mmlu | gsm8k | all | trunc% | fail% |
|---|---|---|---|---|---|
| acc1 | 68.0 | 85.7 | 78.7 | 1.6 | 0.9 |
| acc2_keep11520 | nan | 54.1 | 54.1 | 1.4 | 0.0 |
| acc2_keep12288 | nan | 67.6 | 67.6 | 0.6 | 0.0 |
| acc2_keep12800 | nan | 70.4 | 70.4 | 0.7 | 0.0 |
| acc2_keep13568 | nan | 79.3 | 79.3 | 1.1 | 0.0 |
| acc2_keep14080 | nan | 83.6 | 83.6 | 0.8 | 0.0 |
| acc2_modelopt_keep11520 | nan | 64.9 | 64.9 | 1.7 | 0.0 |
| acc2_modelopt_keep12288 | nan | 72.1 | 72.1 | 0.8 | 0.0 |
| acc2_modelopt_keep12800 | nan | 78.6 | 78.6 | 1.5 | 0.0 |
| acc2_modelopt_keep13568 | nan | 81.7 | 81.7 | 1.5 | 0.0 |
| acc2_modelopt_keep14080 | nan | 83.2 | 83.2 | 0.8 | 0.0 |
| acc3_b16_g00 | nan | 86.1 | 86.1 | 1.1 | 0.0 |
| acc3_b16_g01 | nan | 86.5 | 86.5 | 1.1 | 0.0 |
| acc3_b16_g02 | nan | 85.8 | 85.8 | 1.2 | 0.0 |
| acc3_b16_g03 | nan | 85.7 | 85.7 | 1.4 | 0.0 |
| acc3_b16_g04 | nan | 86.4 | 86.4 | 1.1 | 0.0 |
| acc3_b16_g05 | nan | 85.4 | 85.4 | 1.2 | 0.0 |
| acc3_b16_g06 | nan | 85.2 | 85.2 | 1.5 | 0.0 |
| acc3_b16_g07 | nan | 84.8 | 84.8 | 1.2 | 0.0 |
| acc3_b16_g08 | nan | 86.2 | 86.2 | 1.2 | 0.0 |
| acc3_b16_g09 | nan | 85.8 | 85.8 | 1.6 | 0.0 |

## Claims (mmlu)

- Acc1 (B=1 full) = 67.95
- Acc3 over 10 groupings: mean nan, std nan, min nan, max nan
- thresholds: min = nan, mean-std = nan

- acc2_keep11520: nan | >min: fail | >mean-std: fail
- acc2_keep12288: nan | >min: fail | >mean-std: fail
- acc2_keep12800: nan | >min: fail | >mean-std: fail
- acc2_keep13568: nan | >min: fail | >mean-std: fail
- acc2_keep14080: nan | >min: fail | >mean-std: fail
- acc2_modelopt_keep11520: nan | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: nan | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: nan | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: nan | >min: fail | >mean-std: fail
- acc2_modelopt_keep14080: nan | >min: fail | >mean-std: fail

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

- Acc1 (B=1 full) = 78.71
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

## Paired flips vs Acc1 (McNemar exact)

- acc2_keep11520: n=1000, lost 351, gained 35, p=0.0000
- acc2_keep12288: n=1000, lost 223, gained 42, p=0.0000
- acc2_keep12800: n=1000, lost 193, gained 40, p=0.0000
- acc2_keep13568: n=1000, lost 113, gained 49, p=0.0000
- acc2_keep14080: n=1000, lost 66, gained 45, p=0.0572
- acc2_modelopt_keep11520: n=1000, lost 256, gained 48, p=0.0000
- acc2_modelopt_keep12288: n=1000, lost 186, gained 50, p=0.0000
- acc2_modelopt_keep12800: n=1000, lost 116, gained 45, p=0.0000
- acc2_modelopt_keep13568: n=1000, lost 87, gained 47, p=0.0007
- acc2_modelopt_keep14080: n=1000, lost 63, gained 38, p=0.0165
- acc3_b16_g00: n=1000, lost 14, gained 18, p=0.5966
- acc3_b16_g01: n=1000, lost 12, gained 20, p=0.2153
- acc3_b16_g02: n=1000, lost 17, gained 18, p=1.0000
- acc3_b16_g03: n=1000, lost 22, gained 22, p=1.0000
- acc3_b16_g04: n=1000, lost 20, gained 27, p=0.3817
- acc3_b16_g05: n=1000, lost 21, gained 18, p=0.7493
- acc3_b16_g06: n=1000, lost 23, gained 18, p=0.5327
- acc3_b16_g07: n=1000, lost 26, gained 17, p=0.2221
- acc3_b16_g08: n=1000, lost 20, gained 25, p=0.5515
- acc3_b16_g09: n=1000, lost 15, gained 16, p=1.0000
