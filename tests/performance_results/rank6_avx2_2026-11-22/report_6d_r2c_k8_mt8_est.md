# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D r2c interleaved, natural, out of place, K=8 contiguous_r2c_k8_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 7.124..9.164

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               2152      35742   14.76     42.8  0.0e+00
     2x4x6x8x10x12  2            -      -              46950     920868   11.53    304.1  5.6e-16
       3x3x3x3x3x3  3            -      -               4196      53833    7.88     33.0  4.8e-16
       3x5x7x7x5x3  3            -      -              31723     436436   11.74     93.3  4.6e-16
       4x4x4x4x4x4  2^2          -      -               8218     182331   19.71    119.6  0.0e+00
      4x4x4x4x4x53  2^2          -      -             190462     952762    3.17     89.6  6.3e-16
       4x4x4x8x8x8  2^2          -      -              32813     583475   15.45    299.6  3.4e-16
       5x5x5x5x5x5  5            -      -              36327     596557    9.44    119.8  4.2e-16
       6x6x6x6x6x6  2.3          -      -              58875    1211793   14.40    245.8  6.8e-16
       8x8x8x4x4x4  2^3          -      -              41800    1311193   16.11    235.2  2.3e-16
       8x8x8x8x8x8  2^3          -      -             436813    3489181    7.29    216.0  3.2e-16
       9x9x9x9x9x9  3^2          -      -            2083575    6145543    2.72     97.0  7.7e-16
 10x10x10x10x10x10  2.5          -      -            4150900   10450994    2.40     96.0  5.8e-16
     12x10x8x6x4x2  2^2.3        -      -             207850    2004500    7.32     68.7  5.7e-16
     16x4x4x4x4x16  2^4          -      -              64175    1018118    8.27    326.8  3.3e-16
      53x4x4x4x4x4  53           -      -             140612    2180337   12.19    121.4  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               16      0      0   2.72  10.49  16.11    8.74
 ALL                             16      0      0   2.72  10.49  16.11    8.74
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.17  13.15  19.71   10.60
 prime column                     4      0      0   7.88  10.59  12.19   10.16
 even column                      3      0      0   2.40   7.32  14.40    6.33
 odd column                       1      0      0   2.72   2.72   2.72    2.72
 ALL                             16      0      0   2.72  10.49  16.11    8.74
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   7.32  11.64  16.11   10.07
 > 65536 points                   3      0      0   2.40   2.72   7.29    3.62
 257..1024                        2      0      0   7.88  11.32  14.76   10.78
 1025..4096                       1      0      0  19.71  19.71  19.71   19.71
 ALL                             16      0      0   2.72  10.49  16.11    8.74
```


worst 10: 10x10x10x10x10x10 (- 2.40), 9x9x9x9x9x9 (- 2.72), 4x4x4x4x4x53 (- 3.17), 8x8x8x8x8x8 (- 7.29), 12x10x8x6x4x2 (- 7.32), 3x3x3x3x3x3 (- 7.88), 16x4x4x4x4x16 (- 8.27), 5x5x5x5x5x5 (- 9.44), 2x4x6x8x10x12 (- 11.53), 3x5x7x7x5x3 (- 11.74)
best 5: 4x4x4x4x4x4 (- 19.71), 8x8x8x4x4x4 (- 16.11), 4x4x4x8x8x8 (- 15.45), 2x4x2x4x2x4 (- 14.76), 6x6x6x6x6x6 (- 14.40)
