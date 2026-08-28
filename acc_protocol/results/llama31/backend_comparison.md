# Pruning backend comparison — llama31

topiary = salience width truncation (in-place); modelopt = NVIDIA Model-Optimizer (puzzletron) checkpoint

## mmlu

Acc1 = 66.40 · Acc3 mean 67.88 ± 1.16, min 66.40 (n=10 groupings)

| keep | topiary | modelopt | delta | topiary claims | modelopt claims |
|---|---|---|---|---|---|
| 14080 | 67.2 | 66.4 | -0.8 | >min:PASS >m-s:PASS | >min:fail >m-s:fail |
| 13568 | 64.8 | 61.6 | -3.2 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12800 | 64.8 | 59.6 | -5.2 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12288 | 60.4 | 56.8 | -3.6 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 11520 | 58.8 | 54.4 | -4.4 | >min:fail >m-s:fail | >min:fail >m-s:fail |

## gsm8k

Acc1 = 82.80 · Acc3 mean 81.12 ± 1.16, min 79.20 (n=10 groupings)

| keep | topiary | modelopt | delta | topiary claims | modelopt claims |
|---|---|---|---|---|---|
| 14080 | 80.0 | 80.0 | +0.0 | >min:PASS >m-s:PASS | >min:PASS >m-s:PASS |
| 13568 | 75.6 | 79.6 | +4.0 | >min:fail >m-s:fail | >min:PASS >m-s:fail |
| 12800 | 68.8 | 74.0 | +5.2 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12288 | 62.4 | 67.2 | +4.8 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 11520 | 49.6 | 60.8 | +11.2 | >min:fail >m-s:fail | >min:fail >m-s:fail |

## all

Acc1 = 74.60 · Acc3 mean 74.50 ± 0.78, min 73.00 (n=10 groupings)

| keep | topiary | modelopt | delta | topiary claims | modelopt claims |
|---|---|---|---|---|---|
| 14080 | 73.6 | 73.2 | -0.4 | >min:PASS >m-s:fail | >min:PASS >m-s:fail |
| 13568 | 70.2 | 70.6 | +0.4 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12800 | 66.8 | 66.8 | +0.0 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 12288 | 61.4 | 62.0 | +0.6 | >min:fail >m-s:fail | >min:fail >m-s:fail |
| 11520 | 54.2 | 57.6 | +3.4 | >min:fail >m-s:fail | >min:fail >m-s:fail |

## Paired analyses (pooled, exact McNemar)

- keep14080: topiary-vs-modelopt n=500, topiary-only-right 39, modelopt-only-right 37, agree 424 (85%), p=0.9088
    - topiary vs Acc1: lost 39, gained 34, p=0.6400
    - modelopt vs Acc1: lost 35, gained 28, p=0.4500
- keep13568: topiary-vs-modelopt n=500, topiary-only-right 47, modelopt-only-right 49, agree 404 (81%), p=0.9188
    - topiary vs Acc1: lost 57, gained 35, p=0.0280
    - modelopt vs Acc1: lost 50, gained 30, p=0.0330
- keep12800: topiary-vs-modelopt n=500, topiary-only-right 63, modelopt-only-right 63, agree 374 (75%), p=1.0000
    - topiary vs Acc1: lost 76, gained 37, p=0.0003
    - modelopt vs Acc1: lost 68, gained 29, p=0.0001
- keep12288: topiary-vs-modelopt n=500, topiary-only-right 57, modelopt-only-right 60, agree 383 (77%), p=0.8534
    - topiary vs Acc1: lost 97, gained 31, p=0.0000
    - modelopt vs Acc1: lost 94, gained 31, p=0.0000
- keep11520: topiary-vs-modelopt n=500, topiary-only-right 72, modelopt-only-right 89, agree 339 (68%), p=0.2072
    - topiary vs Acc1: lost 140, gained 38, p=0.0000
    - modelopt vs Acc1: lost 116, gained 31, p=0.0000
