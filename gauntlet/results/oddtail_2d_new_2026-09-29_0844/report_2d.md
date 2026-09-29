# gauntlet report (2D)

run: `oddtail_2d_new_2026-09-29_0844`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 25 listed, 25 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.112..1.328


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       23x23  23           chain+rb replayed         967       1824    1.84     24.8  1.1e-15
       23x45  23           chain  replayed        1555       2680    1.60     33.3  9.3e-16
       23x47  23           chain+rb replayed        2228       5037    2.24     24.4  1.8e-15
       23x63  23           csk    replayed        2401       4089    1.70     31.7  7.9e-16
       29x23  29           csk+rb replayed        1434       2086    1.42     21.8  8.9e-16
       29x45  29           chain  replayed        2510       3065    1.21     26.9  5.5e-16
       29x47  29           chain+rb replayed        3311       5919    1.70     21.4  1.3e-15
       29x63  29           chain  replayed        3199       4665    1.27     30.9  6.2e-16
       37x23  37           chain+rb replayed        1785       2693    1.48     23.2  1.1e-15
       37x45  37           csk    replayed        2995       4381    1.40     29.7  9.4e-16
       37x47  37           chain+rb replayed        3976       7902    1.91     23.5  1.5e-15
       37x63  37           csk    replayed        4256       6326    1.49     30.6  9.3e-16
       43x23  43           csk+rb replayed        2440       3318    1.23     20.2  9.6e-16
       43x45  43           chain  replayed        3749       5282    1.40     28.2  5.9e-16
       43x47  43           csk+rb replayed        4977       9281    1.85     22.3  1.5e-15
       43x63  43           chain  replayed        5333       7492    1.39     29.0  6.8e-16
       47x23  47           chain+rb replayed        2457       5050    1.98     22.2  1.8e-15
       47x45  47           chain  replayed        4376       9088    2.06     26.7  1.3e-15
       47x47  47           chain+rb replayed        5605      13364    2.34     21.9  1.5e-15
       47x63  47           csk    replayed        6113      12478    2.03     27.9  1.3e-15
       256x3  2^8          turn   replayed         703       1539    2.17     52.3  3.6e-16
      1024x3  2^10         turn   replayed        4024       7219    1.79     44.2  3.6e-16
      1024x5  2^10         turn   replayed        6964      10287    1.46     45.3  3.9e-16
      1024x7  2^10         turn   replayed       10603      15773    1.48     43.3  3.1e-16
      4096x3  2^12         turn   replayed       18499      32643    1.62     45.1  4.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb                         7      0      0   1.48   1.91   2.34    1.91
 chain                            6      0      0   1.21   1.39   2.06    1.46
 turn                             5      0      0   1.46   1.62   2.17    1.68
 csk                              4      0      0   1.40   1.59   2.03    1.64
 csk+rb                           3      0      0   1.23   1.42   1.85    1.48
 ALL                             25      0      0   1.27   1.62   2.17    1.65
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 prime column                    20      0      0   1.27   1.65   2.24    1.64
 pow2 column                      5      0      0   1.46   1.62   2.17    1.68
 ALL                             25      0      0   1.27   1.62   2.17    1.65
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                      17      0      0   1.27   1.70   2.24    1.69
 257..1024                        5      0      0   1.23   1.48   2.17    1.59
 4097..65536                      3      0      0   1.46   1.48   1.62    1.52
 ALL                             25      0      0   1.27   1.62   2.17    1.65
```


worst 10: 29x45 (chain 1.21), 43x23 (csk+rb 1.23), 29x63 (chain 1.27), 43x63 (chain 1.39), 43x45 (chain 1.40), 37x45 (csk 1.40), 29x23 (csk+rb 1.42), 1024x5 (turn 1.46), 37x23 (chain+rb 1.48), 1024x7 (turn 1.48)
best 5: 47x47 (chain+rb 2.34), 23x47 (chain+rb 2.24), 256x3 (turn 2.17), 47x45 (chain 2.06), 47x63 (csk 2.03)
