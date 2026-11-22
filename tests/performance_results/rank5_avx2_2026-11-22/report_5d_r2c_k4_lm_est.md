# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=4 lane-major_r2c_k4_lm_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 14.259..14.388


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -              98107    1070353   10.89    100.2  3.3e-16
      4x4x4x4x4  2^2          lm     -               4145     127800   30.62     49.4  0.0e+00
    4x4x4x4x256  2^2          lm     -             218988    1804587    8.23     95.8  3.9e-16
      6x6x6x6x6  2.3          lm     -              27666     472909   16.98     72.7  6.4e-16
      8x8x8x8x8  2^3          lm     -             107727    1528880   14.13     91.3  3.6e-16
     8x8x53x8x8  2^3          lm     -            1642775   14189412    8.59     46.9  6.7e-16
 10x10x10x10x10  2.5          lm     -             400025    3895306    9.65     83.0  5.3e-16
 12x12x12x12x12  2^2.3        lm     -            1095212    8482437    7.66     81.5  6.6e-16
   16x16x16x8x8  2^4          lm     -            1683413   12140294    7.01     56.1  3.4e-16
 16x16x16x16x16  2^4          lm     -           10333800   43564962    4.18     40.6  4.2e-16
   20x16x12x8x4  2^2.5        lm     -             892125    8584812    9.61     46.6  4.5e-16
  24x12x6x12x24  2^3.3        lm     -            2830163   17049387    5.97     66.6  8.5e-16
 32x16x16x16x16  2^5          lm     -           24339525  105078506    4.27     36.2  4.5e-16
     53x8x8x8x8  53           lm     -            1614187   14797106    9.09     47.7  6.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              14      0      0   4.27   8.84  16.98    9.11
 ALL                             14      0      0   4.27   8.84  16.98    9.11
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   4.18   8.41  30.62    8.96
 even column                      5      0      0   5.97   9.61  16.98    9.36
 prime column                     1      0      0   9.09   9.09   9.09    9.09
 ALL                             14      0      0   4.27   8.84  16.98    9.11
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   9      0      0   4.18   7.66   9.65    7.02
 4097..65536                      4      0      0   8.23  12.51  16.98   12.11
 257..1024                        1      0      0  30.62  30.62  30.62   30.62
 ALL                             14      0      0   4.27   8.84  16.98    9.11
```


worst 10: 16x16x16x16x16 (lm 4.18), 32x16x16x16x16 (lm 4.27), 24x12x6x12x24 (lm 5.97), 16x16x16x8x8 (lm 7.01), 12x12x12x12x12 (lm 7.66), 4x4x4x4x256 (lm 8.23), 8x8x53x8x8 (lm 8.59), 53x8x8x8x8 (lm 9.09), 20x16x12x8x4 (lm 9.61), 10x10x10x10x10 (lm 9.65)
best 5: 4x4x4x4x4 (lm 30.62), 6x6x6x6x6 (lm 16.98), 8x8x8x8x8 (lm 14.13), 2x4x8x16x32 (lm 10.89), 10x10x10x10x10 (lm 9.65)
