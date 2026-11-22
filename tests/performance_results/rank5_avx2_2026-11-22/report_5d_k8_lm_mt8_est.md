# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D c2c interleaved, natural, out of place, K=8 lane-major_k8_lm_mt8_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 7.970..15.611

threaded plans that ran serial (engaged = 0 at both flips): 1: 2x4x8x16x32


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            lm     -             516963    2065269    3.64     38.0  8.0e-16
     3x5x7x9x11  3            lm     -              49600     448925    7.95    111.9  5.2e-16
      4x4x4x4x4  2^2          lm     -               6107      89093   13.78     67.1  2.3e-16
    4x4x4x4x256  2^2          lm     -             260788    2199813    7.82    160.8  1.2e-15
      5x5x5x5x5  5            lm     -              13265     313802   19.98    109.4  4.2e-16
      6x6x6x6x6  2.3          lm     -              25931     665100   21.06    155.0  7.1e-16
      8x8x8x8x8  2^3          lm     -              73300     830250    8.11    268.2  6.8e-16
     8x8x8x8x53  2^3          lm     -             998825    5077631    4.74    154.1  1.2e-15
     8x8x53x8x8  2^3          lm     -             936987    5763143    5.08    164.3  1.3e-15
      9x9x9x9x9  3^2          lm     -             189662    1899681    8.64    197.4  8.0e-16
 10x10x10x10x10  2.5          lm     -             271525    2768618    8.79    244.7  4.8e-16
 12x12x12x12x12  2^2.3        lm     -             979650    8752169    7.70    182.1  8.9e-16
     15x9x7x5x3  3.5          lm     -              53229     468682    8.01    146.9  5.0e-16
 15x15x15x15x15  3.5          lm     -            6804800   54267225    7.77     87.2  6.9e-16
   16x16x16x8x8  2^4          lm     -            2109325   10472388    4.46     89.5  7.5e-16
 16x16x16x16x16  2^4          lm     -           18038163   86277887    4.16     46.5  1.9e-15
   20x16x12x8x4  2^2.5        lm     -             365650    3547437    8.69    227.3  5.0e-16
  24x12x6x12x24  2^3.3        lm     -            3140025   27858194    8.78    120.0  7.9e-16
    32x16x8x4x2  2^5          lm     -              96575    1383481   12.94    203.6  5.7e-16
 32x16x16x16x16  2^5          lm     -           37912050  196997919    4.96     46.5  1.8e-15
     53x8x8x8x8  53           lm     -             946287    6981149    5.52    162.7  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              21      0      0   4.46   7.95  13.78    7.74
 ALL                             21      0      0   4.46   7.95  13.78    7.74
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   4.16   5.02  13.78    6.25
 even column                      5      0      0   7.70   8.78  21.06   10.17
 prime column                     3      0      0   5.52   7.95  19.98    9.57
 odd column                       3      0      0   7.77   8.01   8.64    8.13
 ALL                             21      0      0   4.46   7.95  13.78    7.74
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   4.46   5.52   8.78    6.17
 4097..65536                      8      0      0   3.64   8.06  21.06    8.76
 257..1024                        1      0      0  13.78  13.78  13.78   13.78
 1025..4096                       1      0      0  19.98  19.98  19.98   19.98
 ALL                             21      0      0   4.46   7.95  13.78    7.74
```


worst 10: 2x4x8x16x32 (lm 3.64), 16x16x16x16x16 (lm 4.16), 16x16x16x8x8 (lm 4.46), 8x8x8x8x53 (lm 4.74), 32x16x16x16x16 (lm 4.96), 8x8x53x8x8 (lm 5.08), 53x8x8x8x8 (lm 5.52), 12x12x12x12x12 (lm 7.70), 15x15x15x15x15 (lm 7.77), 4x4x4x4x256 (lm 7.82)
best 5: 6x6x6x6x6 (lm 21.06), 5x5x5x5x5 (lm 19.98), 4x4x4x4x4 (lm 13.78), 32x16x8x4x2 (lm 12.94), 10x10x10x10x10 (lm 8.79)
