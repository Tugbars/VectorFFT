# gauntlet report (3D)

run: `rank3_cubes_fftw_avx2_2026-11-22`  contract: 3D c2c interleaved, natural, out of place, K=1_fftw_est  cells: 7 listed, 7 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.380..1.470


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       8x8x8  2^3          -      -                362        382    1.05     63.6  2.4e-16
    12x12x12  2^2.3        -      -               1502       1367    0.90     61.9  5.9e-16
    15x15x15  3.5          -      -               5186       4272    0.80     38.1  5.3e-16
    16x16x16  2^4          -      -               4512       5694    1.24     54.5  4.0e-16
    20x20x20  2^2.5        -      -              15977       9299    0.56     32.5  4.2e-16
    25x25x25  5^2          -      -              30848      31125    0.96     35.3  5.1e-16
    32x32x32  2^5          -      -              42656      54099    1.14     57.6  4.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                                7      1      4   0.56   0.96   1.24    0.92
 ALL                              7      1      4   0.56   0.96   1.24    0.92
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      3      0      0   1.05   1.14   1.24    1.14
 even column                      2      1      2   0.56   0.73   0.90    0.71
 odd column                       2      0      2   0.80   0.88   0.96    0.88
 ALL                              7      1      4   0.56   0.96   1.24    0.92
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                       3      0      2   0.80   0.90   1.24    0.96
 4097..65536                      3      1      2   0.56   0.96   1.14    0.85
 257..1024                        1      0      0   1.05   1.05   1.05    1.05
 ALL                              7      1      4   0.56   0.96   1.24    0.92
```


worst 10: 20x20x20 (- 0.56), 15x15x15 (- 0.80), 12x12x12 (- 0.90), 25x25x25 (- 0.96), 8x8x8 (- 1.05), 32x32x32 (- 1.14), 16x16x16 (- 1.24)
best 5: 16x16x16 (- 1.24), 32x32x32 (- 1.14), 8x8x8 (- 1.05), 25x25x25 (- 0.96), 12x12x12 (- 0.90)
