# gauntlet report (5D)

run: `rank5_avx2_2026-11-22`  contract: 5D r2c interleaved, natural, out of place, K=1_r2c_mt4_est  cells: 21 listed, 21 benched, comparator: MKL DFTI 5D (out of place)

control cell 8x8x8x8x8: 4 readings, 5.970..6.875

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x8x16x32, 4x4x4x4x4


## every shape

```
          shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
    2x4x8x16x32  2            child  -              27248      84577    3.08     45.1  3.1e-16
     3x5x7x9x11  3            child  -              15043      45683    2.92     23.1  6.1e-16
      4x4x4x4x4  2^2          child  -               1084      10587    9.57     23.6  0.0e+00
    4x4x4x4x256  2^2          child  -              27913      74463    2.43     93.9  3.7e-16
      5x5x5x5x5  5            child  -               5057      30471    6.00     17.9  2.9e-16
      6x6x6x6x6  2.3          child  -               6740      38950    5.73     37.3  5.8e-16
      8x8x8x8x8  2^3          child  -              14674      87329    5.84     83.7  2.7e-16
     8x8x8x8x53  2^3          child  -             185800     581644    3.07     51.8  7.1e-16
     8x8x53x8x8  2^3          child  -             131961     728567    5.36     72.9  6.2e-16
      9x9x9x9x9  3^2          child  -              41685     169578    3.36     56.1  6.9e-16
 10x10x10x10x10  2.5          child  -              37292     183569    4.83    111.3  6.4e-16
 12x12x12x12x12  2^2.3        child  -              94731     342890    3.57    117.7  7.2e-16
     15x9x7x5x3  3.5          child  -              16821      96440    5.58     29.1  5.5e-16
 15x15x15x15x15  3.5          child  -             590662     771637    1.30     62.8  8.0e-16
   16x16x16x8x8  2^4          child  -             115167     495560    4.25    102.4  3.6e-16
 16x16x16x16x16  2^4          child  -             428300    1126131    2.53    122.4  3.5e-16
   20x16x12x8x4  2^2.5        child  -              58422     448123    7.01     88.9  4.2e-16
  24x12x6x12x24  2^3.3        child  -             192475     575962    2.62    122.3  7.4e-16
    32x16x8x4x2  2^5          child  -              43333     278800    6.26     28.4  3.6e-16
 32x16x16x16x16  2^5          child  -             949175    2326912    2.20    116.0  3.9e-16
     53x8x8x8x8  53           child  -             104578     768414    6.48     92.0  6.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           21      0      0   2.43   4.25   6.48    4.03
 ALL                             21      0      0   2.43   4.25   6.48    4.03
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   2.43   3.67   9.57    3.99
 even column                      5      0      0   2.62   4.83   7.01    4.49
 prime column                     3      0      0   2.92   6.00   6.48    4.84
 odd column                       3      0      0   1.30   3.36   5.58    2.90
 ALL                             21      0      0   2.43   4.25   6.48    4.03
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  11      0      0   2.20   3.57   6.48    3.52
 4097..65536                      8      0      0   2.43   4.47   6.26    4.14
 257..1024                        1      0      0   9.57   9.57   9.57    9.57
 1025..4096                       1      0      0   6.00   6.00   6.00    6.00
 ALL                             21      0      0   2.43   4.25   6.48    4.03
```


worst 10: 15x15x15x15x15 (child 1.30), 32x16x16x16x16 (child 2.20), 4x4x4x4x256 (child 2.43), 16x16x16x16x16 (child 2.53), 24x12x6x12x24 (child 2.62), 3x5x7x9x11 (child 2.92), 8x8x8x8x53 (child 3.07), 2x4x8x16x32 (child 3.08), 9x9x9x9x9 (child 3.36), 12x12x12x12x12 (child 3.57)
best 5: 4x4x4x4x4 (child 9.57), 20x16x12x8x4 (child 7.01), 53x8x8x8x8 (child 6.48), 32x16x8x4x2 (child 6.26), 5x5x5x5x5 (child 6.00)
