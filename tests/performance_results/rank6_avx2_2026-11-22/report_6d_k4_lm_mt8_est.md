# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=4 lane-major_k4_lm_mt8_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.293..9.062

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 2x4x6x8x10x12


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               3497      79134   21.50     26.4  3.2e-16
     2x4x6x8x10x12  2            lm     -             304040    1212050    3.62     47.0  5.4e-16
       3x3x3x3x3x3  3            lm     -               4496      95598   18.48     30.8  5.4e-16
       3x5x7x7x5x3  3            lm     -              37058     386393    8.15     79.9  5.4e-16
       4x4x4x4x4x4  2^2          lm     -              11566     376869   32.46     85.0  2.5e-16
      4x4x4x4x4x53  2^2          lm     -             249633     984977    3.68     68.4  1.1e-15
       4x4x4x8x8x8  2^2          lm     -              62020     621270    8.07    158.5  5.4e-16
       5x5x5x5x5x5  5            lm     -              34432     286713    6.21    126.4  4.4e-16
       6x6x6x6x6x6  2.3          lm     -              75060     582925    5.84    192.8  8.4e-16
       8x8x8x4x4x4  2^3          lm     -              47853     448393    7.28    205.4  3.3e-16
       8x8x8x8x8x8  2^3          lm     -             295037    2270943    6.34    319.9  1.2e-15
       9x9x9x9x9x9  3^2          lm     -            1393012    9663937    6.62    145.1  8.8e-16
 10x10x10x10x10x10  2.5          lm     -            3750063   27382019    7.08    106.3  6.0e-16
     12x10x8x6x4x2  2^2.3        lm     -             101000    1211175    9.61    141.4  5.1e-16
     16x4x4x4x4x16  2^4          lm     -              75012     892543    8.56    279.6  5.2e-16
      53x4x4x4x4x4  53           lm     -             118489     933111    6.86    144.1  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              16      0      0   3.68   7.18  21.50    8.29
 ALL                             16      0      0   3.68   7.18  21.50    8.29
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   3.62   7.68  32.46    8.59
 prime column                     4      0      0   6.21   7.51  18.48    8.95
 even column                      3      0      0   5.84   7.08   9.61    7.35
 odd column                       1      0      0   6.62   6.62   6.62    6.62
 ALL                             16      0      0   3.68   7.18  21.50    8.29
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   3.68   7.07   9.61    6.48
 > 65536 points                   3      0      0   6.34   6.62   7.08    6.67
 257..1024                        2      0      0  18.48  19.99  21.50   19.93
 1025..4096                       1      0      0  32.46  32.46  32.46   32.46
 ALL                             16      0      0   3.68   7.18  21.50    8.29
```


worst 10: 2x4x6x8x10x12 (lm 3.62), 4x4x4x4x4x53 (lm 3.68), 6x6x6x6x6x6 (lm 5.84), 5x5x5x5x5x5 (lm 6.21), 8x8x8x8x8x8 (lm 6.34), 9x9x9x9x9x9 (lm 6.62), 53x4x4x4x4x4 (lm 6.86), 10x10x10x10x10x10 (lm 7.08), 8x8x8x4x4x4 (lm 7.28), 4x4x4x8x8x8 (lm 8.07)
best 5: 4x4x4x4x4x4 (lm 32.46), 2x4x2x4x2x4 (lm 21.50), 3x3x3x3x3x3 (lm 18.48), 12x10x8x6x4x2 (lm 9.61), 16x4x4x4x4x16 (lm 8.56)
