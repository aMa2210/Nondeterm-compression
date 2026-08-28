# Accuracy protocol — gemma3

| arm | mmlu | gsm8k | all | trunc% | fail% |
|---|---|---|---|---|---|
| acc1 | 70.8 | 91.2 | 81.0 | 0.6 | 0.0 |
| acc2_keep13056 | 68.4 | 90.4 | 79.4 | 1.8 | 0.4 |
| acc2_keep13824 | 67.2 | 90.8 | 79.0 | 0.8 | 0.0 |
| acc2_keep14592 | 66.0 | 90.4 | 78.2 | 1.4 | 0.4 |
| acc2_keep15040 | 69.6 | 90.8 | 80.2 | 2.4 | 0.6 |
| acc3_b16_g00 | 70.0 | 90.8 | 80.4 | 1.2 | 0.0 |
| acc3_b16_g01 | 70.8 | 91.2 | 81.0 | 1.2 | 0.0 |
| acc3_b16_g02 | 68.8 | 91.2 | 80.0 | 1.2 | 0.0 |
| acc3_b16_g03 | 70.8 | 90.8 | 80.8 | 1.4 | 0.0 |
| acc3_b16_g04 | 70.0 | 90.4 | 80.2 | 1.2 | 0.0 |
| acc3_b16_g05 | 69.6 | 91.2 | 80.4 | 1.4 | 0.0 |
| acc3_b16_g06 | 69.6 | 90.8 | 80.2 | 1.2 | 0.0 |
| acc3_b16_g07 | 70.4 | 91.2 | 80.8 | 1.2 | 0.2 |
| acc3_b16_g08 | 70.8 | 90.4 | 80.6 | 0.8 | 0.0 |
| acc3_b16_g09 | 68.4 | 90.8 | 79.6 | 1.0 | 0.0 |

## Claims (mmlu)

- Acc1 (B=1 full) = 70.80
- Acc3 over 10 groupings: mean 69.92, std 0.80, min 68.40, max 70.80
- thresholds: min = 68.40, mean-std = 69.12

- acc2_keep13056: 68.40 | >min: fail | >mean-std: fail
- acc2_keep13824: 67.20 | >min: fail | >mean-std: fail
- acc2_keep14592: 66.00 | >min: fail | >mean-std: fail
- acc2_keep15040: 69.60 | >min: PASS | >mean-std: PASS

## Claims (gsm8k)

- Acc1 (B=1 full) = 91.20
- Acc3 over 10 groupings: mean 90.88, std 0.30, min 90.40, max 91.20
- thresholds: min = 90.40, mean-std = 90.58

- acc2_keep13056: 90.40 | >min: fail | >mean-std: fail
- acc2_keep13824: 90.80 | >min: PASS | >mean-std: PASS
- acc2_keep14592: 90.40 | >min: fail | >mean-std: fail
- acc2_keep15040: 90.80 | >min: PASS | >mean-std: PASS

## Claims (all)

- Acc1 (B=1 full) = 81.00
- Acc3 over 10 groupings: mean 80.40, std 0.40, min 79.60, max 81.00
- thresholds: min = 79.60, mean-std = 80.00

- acc2_keep13056: 79.40 | >min: fail | >mean-std: fail
- acc2_keep13824: 79.00 | >min: fail | >mean-std: fail
- acc2_keep14592: 78.20 | >min: fail | >mean-std: fail
- acc2_keep15040: 80.20 | >min: PASS | >mean-std: PASS

## Paired flips vs Acc1 (McNemar exact)

- acc2_keep13056: n=500, lost 29, gained 21, p=0.3222
- acc2_keep13824: n=500, lost 22, gained 12, p=0.1214
- acc2_keep14592: n=500, lost 23, gained 9, p=0.0201
- acc2_keep15040: n=500, lost 14, gained 10, p=0.5413
- acc3_b16_g00: n=500, lost 13, gained 10, p=0.6776
- acc3_b16_g01: n=500, lost 7, gained 7, p=1.0000
- acc3_b16_g02: n=500, lost 8, gained 3, p=0.2266
- acc3_b16_g03: n=500, lost 9, gained 8, p=1.0000
- acc3_b16_g04: n=500, lost 13, gained 9, p=0.5235
- acc3_b16_g05: n=500, lost 9, gained 6, p=0.6072
- acc3_b16_g06: n=500, lost 11, gained 7, p=0.4807
- acc3_b16_g07: n=500, lost 9, gained 8, p=1.0000
- acc3_b16_g08: n=500, lost 11, gained 9, p=0.8238
- acc3_b16_g09: n=500, lost 12, gained 5, p=0.1435
