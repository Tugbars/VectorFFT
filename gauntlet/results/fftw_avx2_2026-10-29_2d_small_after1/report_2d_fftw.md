# gauntlet report (2D)

run: `fftw_avx2_2026-10-29_2d_small_after1`  contract: 2D c2c interleaved, natural, out of place, K=1_fftw  cells: 31 listed, 31 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 0.933..1.067


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            csk+rb replayed           9          6    0.70      4.6  1.4e-16
         2x4  2            csk+rb replayed          10          8    0.72     11.6  1.7e-16
         2x8  2            csk+rb replayed          14         12    0.82     22.4  1.9e-16
        2x16  2            csk+rb2 replayed          25         21    0.86     32.1  2.0e-16
        2x32  2            csk+rb2 replayed          39         37    0.93     48.8  2.7e-16
        2x64  2            csk    replayed          76         74    0.96     58.6  1.8e-16
       2x128  2            csk    replayed         156        176    1.10     65.8  2.5e-16
         4x2  2^2          csk+rb replayed          10          8    0.78     12.0  8.9e-17
         4x4  2^2          csk+rb replayed          13         10    0.79     24.5  2.0e-16
         4x8  2^2          csk+rb replayed          19         19    0.97     42.1  1.7e-16
        4x16  2^2          csk+rb2 replayed          38         34    0.90     50.4  2.8e-16
        4x32  2^2          csk+rb2 replayed          72         71    0.98     62.3  1.7e-16
        4x64  2^2          csk+rb2 replayed         151        151    0.97     68.0  2.6e-16
         8x2  2^3          csk+rb replayed          14         12    0.82     22.8  1.2e-16
         8x4  2^3          csk+rb replayed          19         18    0.93     41.6  1.6e-16
         8x8  2^3          csk+rb replayed          32         34    1.06     60.7  1.3e-16
        8x16  2^3          csk+rb2 replayed          70         68    0.97     64.5  2.3e-16
        8x32  2^3          csk+rb2 replayed         142        148    1.00     72.2  2.6e-16
       8x512  2^3          csk    replayed        4150       4087    0.98     59.2  3.0e-16
        16x2  2^4          csk+rb replayed          22         19    0.86     35.6  2.2e-16
        16x4  2^4          csk+rb replayed          35         34    0.95     54.4  2.5e-16
        16x8  2^4          csk+rb replayed          66         67    1.02     68.0  2.0e-16
       16x16  2^4          csk+rb2 replayed         158        137    0.86     64.8  2.5e-16
        32x2  2^5          chain+rb replayed          48         35    0.73     40.0  1.9e-16
        32x4  2^5          chain+rb replayed          78         71    0.88     57.6  1.7e-16
        32x8  2^5          chain+rb replayed         147        148    0.98     69.7  2.9e-16
        64x2  2^6          chain+rb replayed          98         75    0.75     45.6  3.0e-16
        64x4  2^6          chain+rb replayed         172        146    0.85     59.4  3.4e-16
       64x64  2^6          csk    replayed        5284       5802    1.07     46.5  3.8e-16
       128x2  2^7          turn   replayed         209        166    0.78     49.1  2.8e-16
     256x256  2^8          chain  replayed      112117     148460    1.32     46.8  3.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk+rb                          12      4     10   0.72   0.84   1.02    0.86
 csk+rb2                          8      0      8   0.86   0.95   1.00    0.93
 chain+rb                         5      2      5   0.73   0.85   0.98    0.84
 csk                              4      0      2   0.96   1.03   1.10    1.03
 turn                             1      1      1   0.78   0.78   0.78    0.78
 chain                            1      0      0   1.32   1.32   1.32    1.32
 ALL                             31      7     26   0.75   0.93   1.06    0.90
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     31      7     26   0.75   0.93   1.06    0.90
 ALL                             31      7     26   0.75   0.93   1.06    0.90
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 <= 256 points                   28      7     25   0.73   0.89   1.02    0.88
 1025..4096                       2      0      1   0.98   1.03   1.07    1.02
 4097..65536                      1      0      0   1.32   1.32   1.32    1.32
 ALL                             31      7     26   0.75   0.93   1.06    0.90
```


worst 10: 2x2 (csk+rb 0.70), 2x4 (csk+rb 0.72), 32x2 (chain+rb 0.73), 64x2 (chain+rb 0.75), 128x2 (turn 0.78), 4x2 (csk+rb 0.78), 4x4 (csk+rb 0.79), 2x8 (csk+rb 0.82), 8x2 (csk+rb 0.82), 64x4 (chain+rb 0.85)
best 5: 256x256 (chain 1.32), 2x128 (csk 1.10), 64x64 (csk 1.07), 8x8 (csk+rb 1.06), 16x8 (csk+rb 1.02)
