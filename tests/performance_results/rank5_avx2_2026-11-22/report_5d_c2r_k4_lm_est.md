# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2r interleaved, natural, out of place, K=4 lane-major_c2r_k4_lm_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 7.968..8.198


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             149713     935747    6.02     65.7  7.8e-16
      4x4x4x4x4  2^2          lm     -               5299     117041   21.87     38.6  4.4e-16
    4x4x4x4x256  2^2          lm     -             321487    1202450    3.71     65.2  8.9e-16
      6x6x6x6x6  2.3          lm     -              33848     413837   12.18     59.4  1.3e-15
      8x8x8x8x8  2^3          lm     -             159513    1263660    7.52     61.6  7.8e-16
     8x8x53x8x8  2^3          lm     -            1826562   12016913    6.56     42.1  1.9e-15
 10x10x10x10x10  2.5          lm     -             478912    2977763    6.20     69.4  1.5e-15
 12x12x12x12x12  2^2.3        lm     -            1321363    6517718    4.87     67.5  1.9e-15
   16x16x16x8x8  2^4          lm     -            1938637    8743437    4.47     48.7  7.8e-16
 16x16x16x16x16  2^4          lm     -           11019737   32043237    2.86     38.1  8.9e-16
   20x16x12x8x4  2^2.5        lm     -             915612    6922800    7.29     45.4  1.0e-15
  24x12x6x12x24  2^3.3        lm     -            3221887   13317443    3.93     58.5  2.0e-15
 32x16x16x16x16  2^5          lm     -           23244112   67450244    2.90     37.9  1.0e-15
     53x8x8x8x8  53           lm     -            1988288   12618125    6.19     38.7  2.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   2.90   6.10  12.18    5.87
 ALL                             14      0      0   2.90   6.10  12.18    5.87
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   2.86   5.24  21.87    5.54
 even column                      5      0      0   3.93   6.20  12.18    6.38
 prime column                     1      0      0   6.19   6.19   6.19    6.19
 ALL                             14      0      0   2.90   6.10  12.18    5.87
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   2.86   4.87   7.29    4.78
 4097..65536                      4      0      0   3.71   6.77  12.18    6.72
 257..1024                        1      0      0  21.87  21.87  21.87   21.87
 ALL                             14      0      0   2.90   6.10  12.18    5.87
```


worst 10: 16x16x16x16x16 (lm 2.86), 32x16x16x16x16 (lm 2.90), 4x4x4x4x256 (lm 3.71), 24x12x6x12x24 (lm 3.93), 16x16x16x8x8 (lm 4.47), 12x12x12x12x12 (lm 4.87), 2x4x8x16x32 (lm 6.02), 53x8x8x8x8 (lm 6.19), 10x10x10x10x10 (lm 6.20), 8x8x53x8x8 (lm 6.56)
best 5: 4x4x4x4x4 (lm 21.87), 6x6x6x6x6 (lm 12.18), 8x8x8x8x8 (lm 7.52), 20x16x12x8x4 (lm 7.29), 8x8x53x8x8 (lm 6.56)
