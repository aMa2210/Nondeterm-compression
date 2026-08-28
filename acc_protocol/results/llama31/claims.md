# Accuracy protocol — llama31

| arm | mmlu | gsm8k | all | trunc% | fail% |
|---|---|---|---|---|---|
| acc1 | 66.4 | 82.8 | 74.6 | 2.2 | 1.0 |
| acc2_keep11520 | 58.8 | 49.6 | 54.2 | 3.8 | 1.8 |
| acc2_keep12288 | 60.4 | 62.4 | 61.4 | 1.6 | 1.2 |
| acc2_keep12800 | 64.8 | 68.8 | 66.8 | 1.4 | 0.6 |
| acc2_keep13568 | 64.8 | 75.6 | 70.2 | 1.8 | 0.6 |
| acc2_keep14080 | 67.2 | 80.0 | 73.6 | 2.0 | 0.8 |
| acc2_modelopt_keep11520 | 54.4 | 60.8 | 57.6 | 4.4 | 2.4 |
| acc2_modelopt_keep12288 | 56.8 | 67.2 | 62.0 | 2.4 | 1.2 |
| acc2_modelopt_keep12800 | 59.6 | 74.0 | 66.8 | 2.6 | 1.4 |
| acc2_modelopt_keep13568 | 61.6 | 79.6 | 70.6 | 2.2 | 0.8 |
| acc2_modelopt_keep14080 | 66.4 | 80.0 | 73.2 | 1.6 | 0.6 |
| acc3_b16_g00 | 67.2 | 81.6 | 74.4 | 1.6 | 0.6 |
| acc3_b16_g01 | 67.2 | 80.8 | 74.0 | 2.2 | 0.8 |
| acc3_b16_g02 | 70.0 | 80.4 | 75.2 | 2.0 | 0.8 |
| acc3_b16_g03 | 66.8 | 81.2 | 74.0 | 1.6 | 0.8 |
| acc3_b16_g04 | 69.2 | 82.8 | 76.0 | 2.8 | 1.2 |
| acc3_b16_g05 | 66.8 | 82.8 | 74.8 | 2.4 | 1.4 |
| acc3_b16_g06 | 68.8 | 79.2 | 74.0 | 3.0 | 1.0 |
| acc3_b16_g07 | 66.4 | 79.6 | 73.0 | 2.0 | 0.8 |
| acc3_b16_g08 | 68.8 | 80.8 | 74.8 | 2.2 | 0.8 |
| acc3_b16_g09 | 67.6 | 82.0 | 74.8 | 2.0 | 0.6 |

## Claims (mmlu)

- Acc1 (B=1 full) = 66.40
- Acc3 over 10 groupings: mean 67.88, std 1.16, min 66.40, max 70.00
- thresholds: min = 66.40, mean-std = 66.72

- acc2_keep11520: 58.80 | >min: fail | >mean-std: fail
- acc2_keep12288: 60.40 | >min: fail | >mean-std: fail
- acc2_keep12800: 64.80 | >min: fail | >mean-std: fail
- acc2_keep13568: 64.80 | >min: fail | >mean-std: fail
- acc2_keep14080: 67.20 | >min: PASS | >mean-std: PASS
- acc2_modelopt_keep11520: 54.40 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: 56.80 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: 59.60 | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: 61.60 | >min: fail | >mean-std: fail
- acc2_modelopt_keep14080: 66.40 | >min: fail | >mean-std: fail

## Claims (gsm8k)

- Acc1 (B=1 full) = 82.80
- Acc3 over 10 groupings: mean 81.12, std 1.16, min 79.20, max 82.80
- thresholds: min = 79.20, mean-std = 79.96

- acc2_keep11520: 49.60 | >min: fail | >mean-std: fail
- acc2_keep12288: 62.40 | >min: fail | >mean-std: fail
- acc2_keep12800: 68.80 | >min: fail | >mean-std: fail
- acc2_keep13568: 75.60 | >min: fail | >mean-std: fail
- acc2_keep14080: 80.00 | >min: PASS | >mean-std: PASS
- acc2_modelopt_keep11520: 60.80 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: 67.20 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: 74.00 | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: 79.60 | >min: PASS | >mean-std: fail
- acc2_modelopt_keep14080: 80.00 | >min: PASS | >mean-std: PASS

## Claims (all)

- Acc1 (B=1 full) = 74.60
- Acc3 over 10 groupings: mean 74.50, std 0.78, min 73.00, max 76.00
- thresholds: min = 73.00, mean-std = 73.72

- acc2_keep11520: 54.20 | >min: fail | >mean-std: fail
- acc2_keep12288: 61.40 | >min: fail | >mean-std: fail
- acc2_keep12800: 66.80 | >min: fail | >mean-std: fail
- acc2_keep13568: 70.20 | >min: fail | >mean-std: fail
- acc2_keep14080: 73.60 | >min: PASS | >mean-std: fail
- acc2_modelopt_keep11520: 57.60 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12288: 62.00 | >min: fail | >mean-std: fail
- acc2_modelopt_keep12800: 66.80 | >min: fail | >mean-std: fail
- acc2_modelopt_keep13568: 70.60 | >min: fail | >mean-std: fail
- acc2_modelopt_keep14080: 73.20 | >min: PASS | >mean-std: fail

## Paired flips vs Acc1 (McNemar exact)

- acc2_keep11520: n=500, lost 140, gained 38, p=0.0000
- acc2_keep12288: n=500, lost 97, gained 31, p=0.0000
- acc2_keep12800: n=500, lost 76, gained 37, p=0.0003
- acc2_keep13568: n=500, lost 57, gained 35, p=0.0280
- acc2_keep14080: n=500, lost 39, gained 34, p=0.6400
- acc2_modelopt_keep11520: n=500, lost 116, gained 31, p=0.0000
- acc2_modelopt_keep12288: n=500, lost 94, gained 31, p=0.0000
- acc2_modelopt_keep12800: n=500, lost 68, gained 29, p=0.0001
- acc2_modelopt_keep13568: n=500, lost 50, gained 30, p=0.0330
- acc2_modelopt_keep14080: n=500, lost 35, gained 28, p=0.4500
- acc3_b16_g00: n=500, lost 19, gained 18, p=1.0000
- acc3_b16_g01: n=500, lost 18, gained 15, p=0.7283
- acc3_b16_g02: n=500, lost 17, gained 20, p=0.7428
- acc3_b16_g03: n=500, lost 21, gained 18, p=0.7493
- acc3_b16_g04: n=500, lost 12, gained 19, p=0.2810
- acc3_b16_g05: n=500, lost 13, gained 14, p=1.0000
- acc3_b16_g06: n=500, lost 20, gained 17, p=0.7428
- acc3_b16_g07: n=500, lost 22, gained 14, p=0.2430
- acc3_b16_g08: n=500, lost 19, gained 20, p=1.0000
- acc3_b16_g09: n=500, lost 18, gained 19, p=1.0000
