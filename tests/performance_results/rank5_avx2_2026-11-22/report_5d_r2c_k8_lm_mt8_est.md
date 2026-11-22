# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=8 lane-major_r2c_k8_lm_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 14.639..23.571

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x4x8x16x32


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             209900     788500    3.33     93.7  3.5e-16
      4x4x4x4x4  2^2          lm     -               7939      79377    9.46     51.6  0.0e+00
    4x4x4x4x256  2^2          lm     -             157475    1221912    7.15    266.3  4.4e-16
      6x6x6x6x6  2.3          lm     -              27953     309478   10.88    143.8  5.8e-16
      8x8x8x8x8  2^3          lm     -              57000    1041525   14.68    344.9  3.4e-16
     8x8x53x8x8  2^3          lm     -             537363    5574243    9.23    286.5  9.1e-16
 10x10x10x10x10  2.5          lm     -             222450    2159700    7.97    298.7  5.4e-16
 12x12x12x12x12  2^2.3        lm     -             526500    4412937    7.40    338.9  7.8e-16
   16x16x16x8x8  2^4          lm     -             697338    5481312    7.45    270.7  3.0e-16
 16x16x16x16x16  2^4          lm     -            9078712   27840513    2.95     92.4  4.0e-16
   20x16x12x8x4  2^2.5        lm     -             300362    3685063   10.72    276.7  4.4e-16
  24x12x6x12x24  2^3.3        lm     -             970937   11114681   10.30    388.0  7.6e-16
 32x16x16x16x16  2^5          lm     -           19503713   61300450    3.01     90.3  4.3e-16
     53x8x8x8x8  53           lm     -             552388    6389793   11.06    278.7  8.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   3.01   8.60  11.06    7.44
 ALL                             14      0      0   3.01   8.60  11.06    7.44
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.95   7.30  14.68    6.14
 even column                      5      0      0   7.40  10.30  10.88    9.34
 prime column                     1      0      0  11.06  11.06  11.06   11.06
 ALL                             14      0      0   3.01   8.60  11.06    7.44
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   2.95   7.97  11.06    7.07
 4097..65536                      4      0      0   3.33   9.02  14.68    7.86
 257..1024                        1      0      0   9.46   9.46   9.46    9.46
 ALL                             14      0      0   3.01   8.60  11.06    7.44
```


worst 10: 16x16x16x16x16 (lm 2.95), 32x16x16x16x16 (lm 3.01), 2x4x8x16x32 (lm 3.33), 4x4x4x4x256 (lm 7.15), 12x12x12x12x12 (lm 7.40), 16x16x16x8x8 (lm 7.45), 10x10x10x10x10 (lm 7.97), 8x8x53x8x8 (lm 9.23), 4x4x4x4x4 (lm 9.46), 24x12x6x12x24 (lm 10.30)
best 5: 8x8x8x8x8 (lm 14.68), 53x8x8x8x8 (lm 11.06), 6x6x6x6x6 (lm 10.88), 20x16x12x8x4 (lm 10.72), 24x12x6x12x24 (lm 10.30)
