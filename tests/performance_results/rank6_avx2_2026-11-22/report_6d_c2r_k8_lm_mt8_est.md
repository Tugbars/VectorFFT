# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=8 lane-major_c2r_k8_lm_mt8_est  cells: 10 listed, 10 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.319..3.471

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 2x4x6x8x10x12


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               6174      41467    6.47     29.9  3.3e-16
     2x4x6x8x10x12  2            lm     -             541600    1063406    1.36     52.7  1.3e-15
       4x4x4x4x4x4  2^2          lm     -              16533     188370   10.62    118.9  4.4e-16
       4x4x4x8x8x8  2^2          lm     -              93100     857387    6.15    211.2  7.8e-16
       6x6x6x6x6x6  2.3          lm     -             112937    1048100    7.86    256.3  1.9e-15
       8x8x8x4x4x4  2^3          lm     -              69687     970150   13.07    282.1  6.7e-16
       8x8x8x8x8x8  2^3          lm     -            1343863    4760225    3.16    140.4  8.9e-16
 10x10x10x10x10x10  2.5          lm     -            8128538   16740100    1.70     98.1  1.6e-15
     16x4x4x4x4x16  2^4          lm     -             121737    1170574    7.78    344.5  6.7e-16
      53x4x4x4x4x4  53           lm     -             163038    2116606   10.42    209.4  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              10      0      0   1.70   7.12  13.07    5.51
 ALL                             10      0      0   1.70   7.12  13.07    5.51
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      7      0      0   1.36   6.47  13.07    5.65
 even column                      2      0      0   1.70   4.78   7.86    3.65
 prime column                     1      0      0  10.42  10.42  10.42   10.42
 ALL                             10      0      0   1.70   7.12  13.07    5.51
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      6      0      0   1.36   7.82  13.07    6.41
 > 65536 points                   2      0      0   1.70   2.43   3.16    2.31
 257..1024                        1      0      0   6.47   6.47   6.47    6.47
 1025..4096                       1      0      0  10.62  10.62  10.62   10.62
 ALL                             10      0      0   1.70   7.12  13.07    5.51
```


worst 10: 2x4x6x8x10x12 (lm 1.36), 10x10x10x10x10x10 (lm 1.70), 8x8x8x8x8x8 (lm 3.16), 4x4x4x8x8x8 (lm 6.15), 2x4x2x4x2x4 (lm 6.47), 16x4x4x4x4x16 (lm 7.78), 6x6x6x6x6x6 (lm 7.86), 53x4x4x4x4x4 (lm 10.42), 4x4x4x4x4x4 (lm 10.62), 8x8x8x4x4x4 (lm 13.07)
best 5: 8x8x8x4x4x4 (lm 13.07), 4x4x4x4x4x4 (lm 10.62), 53x4x4x4x4x4 (lm 10.42), 6x6x6x6x6x6 (lm 7.86), 16x4x4x4x4x16 (lm 7.78)
