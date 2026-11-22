# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D c2c interleaved, natural, out of place, K=4 lane-major_k4_lm_mt8_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 9.230..11.313

threaded plans that ran serial (engaged = 0 at both flips): 4: 2x2x2x2x2x2x2, 2x2x2x2x2x2x53, 2x3x4x5x6x7x8, 2x4x8x2x4x8x2


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            lm     -               1394      40422   27.84     12.9  1.8e-16
  2x2x2x2x2x2x53  2            lm     -              49096     255031    4.37     16.2  1.3e-15
   2x3x4x5x6x7x8  2            lm     -             284008    1395058    4.22     43.4  4.4e-16
   2x4x8x2x4x8x2  2            lm     -              70089     979861   13.27     30.4  2.6e-16
   3x3x3x3x3x3x3  3            lm     -              10531     296298   23.13     46.1  5.0e-16
   3x5x7x3x5x7x3  3            lm     -             101587    1006696    9.86     97.8  4.6e-16
   4x4x4x4x4x4x4  2^2          lm     -              40817     451212    9.40    112.4  3.8e-16
   4x4x4x4x8x8x8  2^2          lm     -             253100    2356050    8.71    176.1  1.7e-15
   5x5x5x5x5x5x5  5            lm     -             148775    1311149    8.37    170.7  5.0e-16
   6x6x6x6x6x6x6  2.3          lm     -             458913    3767738    7.99    220.8  8.8e-16
   8x8x8x8x8x8x4  2^3          lm     -            3839175   29980131    7.55    109.2  1.8e-15
   8x8x8x8x8x8x8  2^3          lm     -           10100050   83355024    8.08     87.2  2.3e-15
 10x8x6x4x6x8x10  2.5          lm     -            3415625   29169506    8.50    106.9  5.5e-16
  53x2x2x2x2x2x2  53           lm     -              11228     245428   18.19     70.9  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   4.37   8.61  23.13    9.84
 ALL                             14      0      0   4.37   8.61  23.13    9.84
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.22   8.40  27.84    8.74
 prime column                     4      0      0   8.37  14.02  23.13   13.65
 even column                      2      0      0   7.99   8.25   8.50    8.24
 ALL                             14      0      0   4.37   8.61  23.13    9.84
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   7.55   8.22   8.71    8.19
 4097..65536                      4      0      0   4.22   9.63  13.27    8.49
 1025..4096                       3      0      0   4.37  18.19  23.13   12.24
 <= 256 points                    1      0      0  27.84  27.84  27.84   27.84
 ALL                             14      0      0   4.37   8.61  23.13    9.84
```


worst 10: 2x3x4x5x6x7x8 (lm 4.22), 2x2x2x2x2x2x53 (lm 4.37), 8x8x8x8x8x8x4 (lm 7.55), 6x6x6x6x6x6x6 (lm 7.99), 8x8x8x8x8x8x8 (lm 8.08), 5x5x5x5x5x5x5 (lm 8.37), 10x8x6x4x6x8x10 (lm 8.50), 4x4x4x4x8x8x8 (lm 8.71), 4x4x4x4x4x4x4 (lm 9.40), 3x5x7x3x5x7x3 (lm 9.86)
best 5: 2x2x2x2x2x2x2 (lm 27.84), 3x3x3x3x3x3x3 (lm 23.13), 53x2x2x2x2x2x2 (lm 18.19), 2x4x8x2x4x8x2 (lm 13.27), 3x5x7x3x5x7x3 (lm 9.86)
