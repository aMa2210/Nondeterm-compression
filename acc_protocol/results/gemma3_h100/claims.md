# Accuracy protocol — gemma3_h100

| arm | mmlu | gsm8k | all | trunc% | fail% |
|---|---|---|---|---|---|
| acc1 | 72.5 | 94.0 | 83.2 | 0.9 | 0.1 |
| acc2_keep12288 | 64.5 | 91.0 | 77.8 | 1.8 | 0.4 |
| acc2_keep13056 | 66.8 | 92.6 | 79.7 | 1.2 | 0.2 |
| acc2_keep13824 | 70.2 | 92.9 | 81.5 | 0.9 | 0.3 |
| acc2_keep14592 | 70.8 | 93.7 | 82.2 | 1.3 | 0.4 |
| acc2_keep15040 | 72.8 | 93.3 | 83.0 | 1.2 | 0.4 |
| acc3_b16_g00 | 73.4 | 94.3 | 83.9 | 1.1 | 0.1 |
| acc3_b16_g01 | 72.5 | 94.4 | 83.5 | 0.8 | 0.1 |
| acc3_b16_g02 | 73.7 | 93.9 | 83.8 | 1.4 | 0.3 |
| acc3_b16_g03 | 73.2 | 94.2 | 83.7 | 0.9 | 0.1 |
| acc3_b16_g04 | 72.3 | 94.1 | 83.2 | 1.0 | 0.1 |
| acc3_b16_g05 | 73.1 | 94.4 | 83.8 | 1.1 | 0.2 |
| acc3_b16_g06 | 72.5 | 93.9 | 83.2 | 0.8 | 0.1 |
| acc3_b16_g07 | 72.1 | 94.0 | 83.0 | 1.1 | 0.1 |
| acc3_b16_g08 | 71.6 | 94.1 | 82.8 | 1.1 | 0.4 |
| acc3_b16_g09 | 72.6 | 93.4 | 83.0 | 1.2 | 0.1 |

## Claims (mmlu)

- Acc1 (B=1 full) = 72.50
- Acc3 over 10 groupings: mean 72.70, std 0.61, min 71.60, max 73.70
- thresholds: min = 71.60, mean-std = 72.09

- acc2_keep12288: 64.50 | >min: fail | >mean-std: fail
- acc2_keep13056: 66.80 | >min: fail | >mean-std: fail
- acc2_keep13824: 70.20 | >min: fail | >mean-std: fail
- acc2_keep14592: 70.80 | >min: fail | >mean-std: fail
- acc2_keep15040: 72.80 | >min: PASS | >mean-std: PASS

## Claims (gsm8k)

- Acc1 (B=1 full) = 94.00
- Acc3 over 10 groupings: mean 94.07, std 0.28, min 93.40, max 94.40
- thresholds: min = 93.40, mean-std = 93.79

- acc2_keep12288: 91.00 | >min: fail | >mean-std: fail
- acc2_keep13056: 92.60 | >min: fail | >mean-std: fail
- acc2_keep13824: 92.90 | >min: fail | >mean-std: fail
- acc2_keep14592: 93.70 | >min: PASS | >mean-std: fail
- acc2_keep15040: 93.30 | >min: fail | >mean-std: fail

## Claims (all)

- Acc1 (B=1 full) = 83.25
- Acc3 over 10 groupings: mean 83.38, std 0.35, min 82.85, max 83.85
- thresholds: min = 82.85, mean-std = 83.03

- acc2_keep12288: 77.75 | >min: fail | >mean-std: fail
- acc2_keep13056: 79.70 | >min: fail | >mean-std: fail
- acc2_keep13824: 81.55 | >min: fail | >mean-std: fail
- acc2_keep14592: 82.25 | >min: fail | >mean-std: fail
- acc2_keep15040: 83.05 | >min: PASS | >mean-std: PASS

## Paired flips vs Acc1 (McNemar exact)

- acc2_keep12288: n=2000, lost 171, gained 61, p=0.0000
- acc2_keep13056: n=2000, lost 135, gained 64, p=0.0000
- acc2_keep13824: n=2000, lost 95, gained 61, p=0.0080
- acc2_keep14592: n=2000, lost 77, gained 57, p=0.1004
- acc2_keep15040: n=2000, lost 58, gained 54, p=0.7770
- acc3_b16_g00: n=2000, lost 33, gained 45, p=0.2127
- acc3_b16_g01: n=2000, lost 32, gained 36, p=0.7163
- acc3_b16_g02: n=2000, lost 31, gained 42, p=0.2416
- acc3_b16_g03: n=2000, lost 33, gained 42, p=0.3557
- acc3_b16_g04: n=2000, lost 36, gained 35, p=1.0000
- acc3_b16_g05: n=2000, lost 33, gained 43, p=0.3019
- acc3_b16_g06: n=2000, lost 36, gained 35, p=1.0000
- acc3_b16_g07: n=2000, lost 35, gained 31, p=0.7122
- acc3_b16_g08: n=2000, lost 34, gained 26, p=0.3663
- acc3_b16_g09: n=2000, lost 41, gained 36, p=0.6488
