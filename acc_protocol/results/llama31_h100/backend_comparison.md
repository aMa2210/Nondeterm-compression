# Pruning backend comparison — llama31_h100

topiary = salience width truncation (in-place); modelopt = NVIDIA Model-Optimizer (puzzletron) checkpoint

## mmlu

Acc1 = 66.90 · Acc3 mean 67.21 ± 0.64, min 66.10 (n=10 groupings)

| keep | topiary | modelopt | delta | topiary claims | modelopt claims |
|---|---|---|---|---|---|
| 14080 | 67.1 | 67.4 | +0.3 | >min:PASS >m-s:PASS | >min:PASS >m-s:PASS |
| 13568 | 63.3 | 63.2 | -0.1 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12800 | 64.1 | 62.1 | -2.0 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12288 | 59.2 | 60.3 | +1.1 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 11520 | 58.4 | 55.8 | -2.6 | >min:fail >m-s:fail | >min:fail >m-s:fail |

## gsm8k

Acc1 = 85.70 · Acc3 mean 85.79 ± 0.51, min 84.80 (n=10 groupings)

| keep | topiary | modelopt | delta | topiary claims | modelopt claims |
|---|---|---|---|---|---|
| 14080 | 83.6 | 83.2 | -0.4 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 13568 | 79.3 | 81.7 | +2.4 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12800 | 70.4 | 78.6 | +8.2 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12288 | 67.6 | 72.1 | +4.5 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 11520 | 54.1 | 64.9 | +10.8 | >min:fail >m-s:fail | >min:fail >m-s:fail |

## all

Acc1 = 76.30 · Acc3 mean 76.50 ± 0.46, min 75.90 (n=10 groupings)

| keep | topiary | modelopt | delta | topiary claims | modelopt claims |
|---|---|---|---|---|---|
| 14080 | 75.3 | 75.3 | -0.0 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 13568 | 71.3 | 72.5 | +1.2 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12800 | 67.2 | 70.3 | +3.1 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12288 | 63.4 | 66.2 | +2.8 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 11520 | 56.2 | 60.4 | +4.1 | >min:fail >m-s:fail | >min:fail >m-s:fail |

## Paired analyses (pooled, exact McNemar)

- keep14080: topiary-vs-modelopt n=2000, topiary-only-right 147, modelopt-only-right 146, agree 1707 (85%), p=1.0000
    - topiary vs Acc1: lost 141, gained 122, p=0.2670
    - modelopt vs Acc1: lost 144, gained 124, p=0.2458
- keep13568: topiary-vs-modelopt n=2000, topiary-only-right 159, modelopt-only-right 182, agree 1659 (83%), p=0.2335
    - topiary vs Acc1: lost 233, gained 133, p=0.0000
    - modelopt vs Acc1: lost 205, gained 128, p=0.0000
- keep12800: topiary-vs-modelopt n=2000, topiary-only-right 194, modelopt-only-right 256, agree 1550 (78%), p=0.0040
    - topiary vs Acc1: lost 322, gained 141, p=0.0000
    - modelopt vs Acc1: lost 258, gained 139, p=0.0000
- keep12288: topiary-vs-modelopt n=2000, topiary-only-right 220, modelopt-only-right 276, agree 1504 (75%), p=0.0134
    - topiary vs Acc1: lost 386, gained 128, p=0.0000
    - modelopt vs Acc1: lost 362, gained 160, p=0.0000
- keep11520: topiary-vs-modelopt n=2000, topiary-only-right 239, modelopt-only-right 321, agree 1440 (72%), p=0.0006
    - topiary vs Acc1: lost 541, gained 140, p=0.0000
    - modelopt vs Acc1: lost 466, gained 147, p=0.0000
