# gauntlet report (4D)

run: `rank4_avx2_2026-11-22`  contract: 4D r2c interleaved, natural, out of place, K=1_r2c_mt8_est  cells: 36 listed, 36 benched, comparator: MKL DFTI 4D (out of place)

control cell 16x16x16x16: 4 readings, 1.792..3.208

threaded plans that ran serial (engaged = 0 at both flips): 1: 4x4x4x4


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x64x64x64  2            child  -             218875     610350    2.35    113.8  4.6e-16
      4x4x4x4  2^2          child  -                238       3281   13.33     21.5  0.0e+00
   4x32x32x32  2^2          child  -              47010     111226    2.34    118.5  3.3e-16
     5x7x9x11  5            child  -               3379       9103    2.65     30.1  4.6e-16
   6x12x24x96  2.3          child  -              43767     109310    1.82    164.3  5.1e-16
      8x8x8x8  2^3          child  -               2302       7609    2.99     53.4  2.1e-16
    8x8x8x256  2^3          child  -              26337      54791    1.70    211.5  4.2e-16
  8x8x128x128  2^3          child  -             200538     416225    1.43    261.4  4.2e-16
      9x9x9x9  3^2          child  -               5490      16698    2.10     37.9  5.5e-16
  10x12x14x16  2.5          child  -              12423      36949    2.54     79.6  4.9e-16
  12x12x12x12  2^2.3        child  -               7675      28317    2.66     96.9  8.5e-16
  15x15x15x15  3.5          child  -              20676      38067    1.02     95.7  5.5e-16
  16x16x16x16  2^4          child  -              16982      53614    2.85    154.4  3.2e-16
  16x16x16x32  2^4          child  -              27297      68288    2.20    204.1  3.1e-16
  16x16x16x53  2^4          child  -             105617     277258    2.45     91.1  6.8e-16
  16x16x16x64  2^4          child  -              48933      83353    1.41    241.1  3.6e-16
  16x53x16x16  2^4          child  -              63300     236066    2.87    152.0  7.1e-16
  20x20x20x20  2^2.5        child  -              53525     114716    1.36    129.2  4.5e-16
  24x24x24x24  2^3.3        child  -              87133     245587    1.66    174.6  6.3e-16
  25x25x25x25  5^2          child  -             144500     284955    1.24    125.5  5.5e-16
     27x9x9x9  3^3          child  -              11540      31137    2.00     60.8  6.9e-16
  32x32x32x32  2^5          child  -             199813     418750    1.55    262.4  4.4e-16
  32x32x32x64  2^5          child  -             359200     728318    1.34    306.5  4.9e-16
    35x15x9x5  5.7          child  -              14670      42856    1.96     58.5  5.6e-16
  40x30x20x10  2^3.5        child  -              52825     147524    2.01    203.0  4.4e-16
  45x45x15x15  3^2.5        child  -             141525     310081    1.80    151.3  6.8e-16
  48x48x48x48  2^4.3        child  -            2104125    2865706    1.19    140.9  6.7e-16
  53x16x16x16  53           child  -              57678     243938    3.06    166.8  6.5e-16
  64x16x16x16  2^6          child  -              47133     124343    2.23    250.3  3.8e-16
   64x64x64x4  2^6          child  -             355125    1152243    2.44    147.6  4.3e-16
  64x64x64x64  2^6          child  -           13505837   14150268    1.03     74.5  4.4e-16
   96x24x12x6  2^5.3        child  -              48854     187367    3.43    147.2  6.0e-16
    101x8x8x8  101          child  -              20605      80333    3.10     98.2  1.4e-15
 127x12x12x12  127          child  -              61572     319655    3.06    158.1  7.5e-16
  128x128x8x8  2^7          child  -             218725     888100    3.14    239.7  4.7e-16
    256x8x8x8  2^8          child  -              38883     159825    2.80    143.3  3.6e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           36      0      0   1.24   2.21   3.10    2.16
 ALL                             36      0      0   1.24   2.21   3.10    2.16
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      0      0   1.34   2.34   3.14    2.30
 even column                      8      0      0   1.19   1.92   3.43    1.97
 odd column                       6      0      0   1.02   1.88   2.10    1.63
 prime column                     4      0      0   2.65   3.06   3.10    2.96
 ALL                             36      0      0   1.24   2.21   3.10    2.16
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  25      0      0   1.24   2.01   3.06    1.96
 4097..65536                      8      0      0   1.02   2.32   3.10    2.17
 1025..4096                       2      0      0   2.65   2.82   2.99    2.81
 <= 256 points                    1      0      0  13.33  13.33  13.33   13.33
 ALL                             36      0      0   1.24   2.21   3.10    2.16
```


worst 10: 15x15x15x15 (child 1.02), 64x64x64x64 (child 1.03), 48x48x48x48 (child 1.19), 25x25x25x25 (child 1.24), 32x32x32x64 (child 1.34), 20x20x20x20 (child 1.36), 16x16x16x64 (child 1.41), 8x8x128x128 (child 1.43), 32x32x32x32 (child 1.55), 24x24x24x24 (child 1.66)
best 5: 4x4x4x4 (child 13.33), 96x24x12x6 (child 3.43), 128x128x8x8 (child 3.14), 101x8x8x8 (child 3.10), 53x16x16x16 (child 3.06)
