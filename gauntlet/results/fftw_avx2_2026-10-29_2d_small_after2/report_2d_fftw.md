# gauntlet report (2D)

run: `fftw_avx2_2026-10-29_2d_small_after2`  contract: 2D c2c interleaved, natural, out of place, K=1_fftw  cells: 31 listed, 31 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 0.929..1.075


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            csk+rb replayed           9          6    0.69      4.5  1.4e-16
         2x4  2            csk+rb replayed          10          8    0.72     11.5  1.7e-16
         2x8  2            csk+rb replayed          14         12    0.82     22.5  1.9e-16
        2x16  2            csk+rb2 replayed          25         21    0.83     32.1  2.0e-16
        2x32  2            csk+rb2 replayed          39         37    0.93     48.7  2.7e-16
        2x64  2            csk    replayed          76         72    0.94     58.6  1.8e-16
       2x128  2            csk    replayed         156        166    1.06     65.8  2.5e-16
         4x2  2^2          csk+rb replayed          10          8    0.76     12.0  8.9e-17
         4x4  2^2          csk+rb replayed          13         10    0.78     24.5  2.0e-16
         4x8  2^2          csk+rb replayed          19         19    0.97     42.1  1.7e-16
        4x16  2^2          csk+rb2 replayed          38         34    0.90     50.9  2.8e-16
        4x32  2^2          csk+rb2 replayed          72         70    0.98     62.5  1.7e-16
        4x64  2^2          csk+rb2 replayed         150        146    0.97     68.1  2.6e-16
         8x2  2^3          csk+rb replayed          14         12    0.82     22.8  1.2e-16
         8x4  2^3          csk+rb replayed          19         18    0.92     41.7  1.6e-16
         8x8  2^3          csk+rb replayed          32         34    1.06     60.5  1.3e-16
        8x16  2^3          csk+rb2 replayed          70         68    0.97     64.3  2.3e-16
        8x32  2^3          csk+rb2 replayed         142        152    1.04     72.2  2.6e-16
       8x512  2^3          csk    replayed        4092       4134    1.01     60.1  3.0e-16
        16x2  2^4          csk+rb replayed          22         19    0.85     35.6  2.2e-16
        16x4  2^4          csk+rb replayed          35         36    0.95     54.3  2.5e-16
        16x8  2^4          csk+rb replayed          66         69    1.02     68.1  2.0e-16
       16x16  2^4          csk+rb2 replayed         158        137    0.86     64.8  2.5e-16
        32x2  2^5          chain+rb replayed          48         37    0.74     40.1  1.9e-16
        32x4  2^5          chain+rb replayed          78         68    0.88     57.8  1.7e-16
        32x8  2^5          chain+rb replayed         147        140    0.94     69.7  2.9e-16
        64x2  2^6          chain+rb replayed          97         73    0.75     46.0  3.0e-16
        64x4  2^6          chain+rb replayed         172        146    0.84     59.5  3.4e-16
       64x64  2^6          csk    replayed        5298       5799    0.97     46.4  3.8e-16
       128x2  2^7          turn   replayed         209        166    0.79     49.0  2.8e-16
     256x256  2^8          chain  replayed      114273     144048    1.23     45.9  3.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk+rb                          12      4     10   0.72   0.84   1.02    0.86
 csk+rb2                          8      0      7   0.83   0.95   1.04    0.93
 chain+rb                         5      2      5   0.74   0.84   0.94    0.83
 csk                              4      0      2   0.94   0.99   1.06    1.00
 turn                             1      1      1   0.79   0.79   0.79    0.79
 chain                            1      0      0   1.23   1.23   1.23    1.23
 ALL                             31      7     25   0.75   0.92   1.04    0.90
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     31      7     25   0.75   0.92   1.04    0.90
 ALL                             31      7     25   0.75   0.92   1.04    0.90
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 <= 256 points                   28      7     24   0.74   0.89   1.04    0.88
 1025..4096                       2      0      1   0.97   0.99   1.01    0.99
 4097..65536                      1      0      0   1.23   1.23   1.23    1.23
 ALL                             31      7     25   0.75   0.92   1.04    0.90
```


worst 10: 2x2 (csk+rb 0.69), 2x4 (csk+rb 0.72), 32x2 (chain+rb 0.74), 64x2 (chain+rb 0.75), 4x2 (csk+rb 0.76), 4x4 (csk+rb 0.78), 128x2 (turn 0.79), 2x8 (csk+rb 0.82), 8x2 (csk+rb 0.82), 2x16 (csk+rb2 0.83)
best 5: 256x256 (chain 1.23), 2x128 (csk 1.06), 8x8 (csk+rb 1.06), 8x32 (csk+rb2 1.04), 16x8 (csk+rb 1.02)
