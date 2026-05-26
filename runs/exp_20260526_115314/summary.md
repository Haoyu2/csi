# Experiment Sweep Results
Generated: 2026-05-26T13:12:11
Source manifest: users 1-3, gestures 1-6, keep-all dedup (26822 samples)

## Config 1 — Aggregate user classification

Predict user (3 classes), motion 1-6 mixed. Split: by-key 3-way, val_frac=0.1, test_frac=0.1.

| Mode | Test acc | Train acc | Val acc | Time |
|---|---|---|---|---|
| BVP | 0.5078 | 0.4817 | 0.4648 | 463.6s |
| BAP | 0.5078 | 0.4809 | 0.4648 | 469.4s |

### Confusion matrices (row-normalized)

**BVP**

```
true\pred  user1   user2   user3
user1     0.31    0.69    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

**BAP**

```
true\pred  user1   user2   user3
user1     0.31    0.69    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```


## Config 2 — Per-motion user classification

Predict user (3 classes). Split: by-key 3-way, val_frac=0.1, test_frac=0.1.

| Motion | Mode | Test acc | Val acc | n_train | n_val | n_test |
|---|---|---|---|---|---|---|
| 1 | BVP | 0.4574 | 0.4820 | 3758 | 471 | 470 |
| 1 | BAP | 0.4404 | 0.4607 | 3758 | 471 | 470 |
| 2 | BVP | 0.5087 | 0.4803 | 3659 | 458 | 458 |
| 2 | BAP | 0.5568 | 0.5022 | 3659 | 458 | 458 |
| 3 | BVP | 0.4672 | 0.4956 | 3658 | 458 | 458 |
| 3 | BAP | 0.5087 | 0.4782 | 3658 | 458 | 458 |
| 4 | BVP | 0.5236 | 0.5359 | 3559 | 446 | 445 |
| 4 | BAP | 0.5169 | 0.5247 | 3559 | 446 | 445 |
| 5 | BVP | 0.5127 | 0.5196 | 3459 | 433 | 433 |
| 5 | BAP | 0.5497 | 0.5681 | 3459 | 433 | 433 |
| 6 | BVP | 0.5667 | 0.5330 | 3355 | 424 | 420 |
| 6 | BAP | 0.5833 | 0.5660 | 3355 | 424 | 420 |

### Confusion matrices per motion (row-normalized)

**Motion 1 — BVP**

```
true\pred  user1   user2   user3
user1     0.26    0.54    0.20
user2     0.00    0.69    0.31
user3     0.03    0.67    0.30
```

**Motion 1 — BAP**

```
true\pred  user1   user2   user3
user1     0.28    0.34    0.38
user2     0.01    0.47    0.52
user3     0.01    0.38    0.61
```

**Motion 2 — BVP**

```
true\pred  user1   user2   user3
user1     0.28    0.54    0.17
user2     0.04    0.88    0.08
user3     0.05    0.76    0.19
```

**Motion 2 — BAP**

```
true\pred  user1   user2   user3
user1     0.59    0.19    0.22
user2     0.17    0.67    0.16
user3     0.29    0.40    0.31
```

**Motion 3 — BVP**

```
true\pred  user1   user2   user3
user1     0.27    0.58    0.15
user2     0.05    0.80    0.15
user3     0.04    0.79    0.18
```

**Motion 3 — BAP**

```
true\pred  user1   user2   user3
user1     0.35    0.49    0.15
user2     0.09    0.79    0.12
user3     0.12    0.63    0.25
```

**Motion 4 — BVP**

```
true\pred  user1   user2   user3
user1     0.43    0.56    0.01
user2     0.16    0.83    0.01
user3     0.10    0.80    0.10
```

**Motion 4 — BAP**

```
true\pred  user1   user2   user3
user1     0.26    0.74    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

**Motion 5 — BVP**

```
true\pred  user1   user2   user3
user1     0.54    0.18    0.28
user2     0.14    0.43    0.43
user3     0.20    0.19    0.61
```

**Motion 5 — BAP**

```
true\pred  user1   user2   user3
user1     0.52    0.23    0.26
user2     0.09    0.54    0.37
user3     0.20    0.19    0.61
```

**Motion 6 — BVP**

```
true\pred  user1   user2   user3
user1     0.64    0.35    0.01
user2     0.15    0.85    0.01
user3     0.31    0.69    0.00
```

**Motion 6 — BAP**

```
true\pred  user1   user2   user3
user1     0.52    0.28    0.19
user2     0.09    0.77    0.14
user3     0.31    0.33    0.37
```


## Config 3 — Per-cell user classification (ideal settings)

Predict user (3 classes). One experiment per (motion, orientation, location) cell. Split: random 3-way 0.6/0.2/0.2.

**Overall mean across 150 cells:**

- BVP: **0.4148** (std 0.1106)
- BAP: **0.3897** (std 0.0991)

### Per-motion mean

| Motion | BVP | BAP | Cells |
|---|---|---|---|
| 1 | 0.4847 | 0.4482 | 25 |
| 2 | 0.2970 | 0.2873 | 25 |
| 3 | 0.3091 | 0.2970 | 25 |
| 4 | 0.4700 | 0.4537 | 25 |
| 5 | 0.4787 | 0.4465 | 25 |
| 6 | 0.4493 | 0.4053 | 25 |

### Aggregated confusion matrix per motion (summed over all cells)

**Motion 1 — BVP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.04    0.93    0.03
user2     0.04    0.94    0.03
user3     0.04    0.94    0.03
```

**Motion 1 — BAP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.11    0.83    0.06
user2     0.12    0.84    0.05
user3     0.09    0.88    0.03
```

**Motion 2 — BVP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.29    0.06    0.65
user2     0.23    0.07    0.71
user3     0.23    0.09    0.68
```

**Motion 2 — BAP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.17    0.06    0.77
user2     0.21    0.07    0.73
user3     0.22    0.09    0.69
```

**Motion 3 — BVP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.28    0.11    0.61
user2     0.27    0.08    0.64
user3     0.23    0.08    0.69
```

**Motion 3 — BAP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.25    0.11    0.64
user2     0.25    0.06    0.68
user3     0.23    0.08    0.69
```

**Motion 4 — BVP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.06    0.94    0.00
user2     0.08    0.91    0.01
user3     0.09    0.90    0.01
```

**Motion 4 — BAP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.13    0.83    0.04
user2     0.13    0.83    0.03
user3     0.14    0.82    0.04
```

**Motion 5 — BVP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.16    0.68    0.16
user2     0.09    0.77    0.14
user3     0.10    0.73    0.17
```

**Motion 5 — BAP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.11    0.66    0.23
user2     0.12    0.69    0.19
user3     0.16    0.61    0.23
```

**Motion 6 — BVP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.00    0.85    0.14
user2     0.02    0.82    0.16
user3     0.03    0.83    0.14
```

**Motion 6 — BAP** (25 cells pooled)

```
true\pred  user1   user2   user3
user1     0.02    0.74    0.24
user2     0.05    0.73    0.22
user3     0.03    0.86    0.10
```


### Sample cells per motion (5 spread by BVP accuracy)

For each motion, picking the worst, 25th-pct, median, 75th-pct, and best cells by BVP accuracy.


#### Motion 1

**ori=1, loc=3** — BVP=0.3824, BAP=0.2353, n_test=34

BVP:

```
true\pred  user1   user2   user3
user1     0.50    0.50    0.00
user2     0.47    0.53    0.00
user3     0.36    0.55    0.09
```

BAP:

```
true\pred  user1   user2   user3
user1     0.50    0.50    0.00
user2     0.71    0.29    0.00
user3     0.55    0.45    0.00
```

**ori=5, loc=1** — BVP=0.4706, BAP=0.4706, n_test=34

BVP:

```
true\pred  user1   user2   user3
user1     0.17    0.83    0.00
user2     0.12    0.88    0.00
user3     0.09    0.91    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.17    0.83    0.00
user2     0.12    0.88    0.00
user3     0.09    0.91    0.00
```

**ori=3, loc=3** — BVP=0.5000, BAP=0.5000, n_test=34

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    0.94    0.06
user3     0.00    0.91    0.09
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    0.94    0.06
user3     0.00    0.91    0.09
```

**ori=4, loc=4** — BVP=0.5000, BAP=0.5000, n_test=34

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

**ori=2, loc=5** — BVP=0.5294, BAP=0.4412, n_test=34

BVP:

```
true\pred  user1   user2   user3
user1     0.17    0.83    0.00
user2     0.00    1.00    0.00
user3     0.18    0.82    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.06    0.88    0.06
user3     0.18    0.82    0.00
```


#### Motion 2

**ori=3, loc=4** — BVP=0.1515, BAP=0.2424, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.50    0.00    0.50
user2     0.50    0.00    0.50
user3     0.73    0.00    0.27
```

BAP:

```
true\pred  user1   user2   user3
user1     0.25    0.00    0.75
user2     0.28    0.00    0.72
user3     0.36    0.00    0.64
```

**ori=5, loc=3** — BVP=0.2424, BAP=0.2424, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.75    0.00    0.25
user2     0.33    0.00    0.67
user3     0.55    0.00    0.45
```

BAP:

```
true\pred  user1   user2   user3
user1     0.25    0.00    0.75
user2     0.28    0.11    0.61
user3     0.27    0.27    0.45
```

**ori=1, loc=5** — BVP=0.3030, BAP=0.2727, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.00    0.25    0.75
user2     0.11    0.28    0.61
user3     0.55    0.00    0.45
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.00    1.00
user2     0.17    0.17    0.67
user3     0.27    0.18    0.55
```

**ori=5, loc=1** — BVP=0.3333, BAP=0.3030, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.75    0.00    0.25
user2     0.39    0.11    0.50
user3     0.09    0.36    0.55
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.50    0.50
user2     0.00    0.17    0.83
user3     0.00    0.36    0.64
```

**ori=1, loc=1** — BVP=0.4848, BAP=0.3333, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.25    0.50    0.25
user2     0.06    0.44    0.50
user3     0.00    0.36    0.64
```

BAP:

```
true\pred  user1   user2   user3
user1     0.25    0.00    0.75
user2     0.39    0.11    0.50
user3     0.00    0.27    0.73
```


#### Motion 3

**ori=3, loc=5** — BVP=0.1515, BAP=0.3636, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.40    0.40    0.20
user2     0.71    0.00    0.29
user3     0.64    0.09    0.27
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.20    0.80
user2     0.18    0.12    0.71
user3     0.09    0.00    0.91
```

**ori=3, loc=4** — BVP=0.2424, BAP=0.2727, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.50    0.00    0.50
user2     0.44    0.00    0.56
user3     0.36    0.09    0.55
```

BAP:

```
true\pred  user1   user2   user3
user1     0.50    0.00    0.50
user2     0.17    0.00    0.83
user3     0.36    0.00    0.64
```

**ori=1, loc=5** — BVP=0.3333, BAP=0.3333, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.50    0.25    0.25
user2     0.28    0.22    0.50
user3     0.36    0.18    0.45
```

BAP:

```
true\pred  user1   user2   user3
user1     0.75    0.00    0.25
user2     0.33    0.00    0.67
user3     0.27    0.00    0.73
```

**ori=1, loc=2** — BVP=0.3636, BAP=0.2121, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.00    0.25    0.75
user2     0.11    0.28    0.61
user3     0.00    0.36    0.64
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.00    1.00
user2     0.44    0.00    0.56
user3     0.27    0.09    0.64
```

**ori=4, loc=1** — BVP=0.5758, BAP=0.4242, n_test=33

BVP:

```
true\pred  user1   user2   user3
user1     0.50    0.50    0.00
user2     0.22    0.50    0.28
user3     0.00    0.27    0.73
```

BAP:

```
true\pred  user1   user2   user3
user1     0.25    0.25    0.50
user2     0.28    0.33    0.39
user3     0.00    0.36    0.64
```


#### Motion 4

**ori=4, loc=2** — BVP=0.3438, BAP=0.3438, n_test=32

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.31    0.69    0.00
user3     0.40    0.60    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.31    0.69    0.00
user3     0.50    0.50    0.00
```

**ori=3, loc=1** — BVP=0.4688, BAP=0.5312, n_test=32

BVP:

```
true\pred  user1   user2   user3
user1     0.67    0.33    0.00
user2     0.31    0.69    0.00
user3     0.50    0.50    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.17    0.83    0.00
user2     0.00    1.00    0.00
user3     0.20    0.80    0.00
```

**ori=1, loc=4** — BVP=0.5000, BAP=0.5000, n_test=32

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

**ori=3, loc=4** — BVP=0.5000, BAP=0.4062, n_test=32

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.06    0.81    0.12
user3     0.20    0.80    0.00
```

**ori=5, loc=5** — BVP=0.5000, BAP=0.5000, n_test=32

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```


#### Motion 5

**ori=4, loc=3** — BVP=0.3226, BAP=0.4516, n_test=31

BVP:

```
true\pred  user1   user2   user3
user1     0.20    0.60    0.20
user2     0.38    0.31    0.31
user3     0.20    0.40    0.40
```

BAP:

```
true\pred  user1   user2   user3
user1     0.20    0.80    0.00
user2     0.25    0.69    0.06
user3     0.40    0.40    0.20
```

**ori=5, loc=1** — BVP=0.4194, BAP=0.4516, n_test=31

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.12    0.75    0.12
user3     0.10    0.80    0.10
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.60    0.40
user2     0.06    0.56    0.38
user3     0.00    0.50    0.50
```

**ori=5, loc=4** — BVP=0.4839, BAP=0.4839, n_test=31

BVP:

```
true\pred  user1   user2   user3
user1     0.40    0.40    0.20
user2     0.06    0.75    0.19
user3     0.20    0.70    0.10
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.60    0.40
user2     0.00    0.56    0.44
user3     0.10    0.30    0.60
```

**ori=2, loc=4** — BVP=0.5161, BAP=0.3548, n_test=31

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.40    0.60
user2     0.12    0.62    0.25
user3     0.10    0.80    0.10
```

**ori=1, loc=2** — BVP=0.6129, BAP=0.4839, n_test=31

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    0.94    0.06
user3     0.00    0.60    0.40
```

BAP:

```
true\pred  user1   user2   user3
user1     0.40    0.60    0.00
user2     0.25    0.75    0.00
user3     0.30    0.60    0.10
```


#### Motion 6

**ori=4, loc=2** — BVP=0.1667, BAP=0.1667, n_test=30

BVP:

```
true\pred  user1   user2   user3
user1     0.08    0.85    0.08
user2     0.19    0.25    0.56
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.92    0.08
user2     0.25    0.31    0.44
user3     0.00    1.00    0.00
```

**ori=5, loc=4** — BVP=0.4000, BAP=0.4000, n_test=30

BVP:

```
true\pred  user1   user2   user3
user1     0.00    0.85    0.15
user2     0.00    0.69    0.31
user3     0.00    0.00    1.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.69    0.31
user2     0.00    0.75    0.25
user3     0.00    1.00    0.00
```

**ori=1, loc=1** — BVP=0.5333, BAP=0.4000, n_test=30

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.77    0.23
user2     0.00    0.75    0.25
user3     0.00    1.00    0.00
```

**ori=3, loc=2** — BVP=0.5333, BAP=0.3333, n_test=30

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.54    0.46
user2     0.00    0.62    0.38
user3     0.00    1.00    0.00
```

**ori=5, loc=5** — BVP=0.5333, BAP=0.2667, n_test=30

BVP:

```
true\pred  user1   user2   user3
user1     0.00    1.00    0.00
user2     0.00    1.00    0.00
user3     0.00    1.00    0.00
```

BAP:

```
true\pred  user1   user2   user3
user1     0.00    0.38    0.62
user2     0.19    0.50    0.31
user3     0.00    1.00    0.00
```

