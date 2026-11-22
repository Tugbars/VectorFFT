# gauntlet report (4D)

run: `rank4_avx2_2026-11-22`  contract: 4D c2r interleaved, natural, out of place, K=1_c2r_mt8_est  cells: 36 listed, 36 benched, comparator: MKL DFTI 4D (out of place)

control cell 16x16x16x16: 4 readings, 2.277..4.975

threaded plans that ran serial (engaged = 0 at both flips): 1: 4x4x4x4


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x64x64x64  2            child  -             281225     724500    2.41     88.6  1.1e-15
      4x4x4x4  2^2          child  -                253       3521   11.89     20.3  3.3e-16
   4x32x32x32  2^2          child  -              36970     138808    3.22    150.7  1.1e-15
     5x7x9x11  5            child  -               3145      11412    2.78     32.4  1.0e-15
   6x12x24x96  2.3          child  -              39946     123631    2.63    180.0  1.3e-15
      8x8x8x8  2^3          child  -               1763       8285    4.04     69.7  6.7e-16
    8x8x8x256  2^3          child  -              24263      54291    2.10    229.6  1.0e-15
  8x8x128x128  2^3          child  -             232788     384268    1.40    225.2  1.3e-15
      9x9x9x9  3^2          child  -               6279      18922    2.64     33.1  1.6e-15
  10x12x14x16  2.5          child  -               8182      40908    4.07    120.8  1.2e-15
  12x12x12x12  2^2.3        child  -               6194      33925    4.71    120.0  1.2e-15
  15x15x15x15  3.5          child  -              29934      35365    0.93     66.1  1.6e-15
  16x16x16x16  2^4          child  -              16428      44745    2.34    159.6  1.0e-15
  16x16x16x32  2^4          child  -              21470      59906    2.34    259.5  9.4e-16
  16x16x16x53  2^4          child  -             158394     330755    1.74     60.7  1.6e-15
  16x16x16x64  2^4          child  -              46513     119220    2.38    253.6  1.1e-15
  16x53x16x16  2^4          child  -              63822     275719    3.14    150.8  1.9e-15
  20x20x20x20  2^2.5        child  -              53846     153218    2.34    128.4  1.1e-15
  24x24x24x24  2^3.3        child  -              89717     193916    1.81    169.6  1.4e-15
  25x25x25x25  5^2          child  -             149820     283985    1.28    121.1  1.6e-15
     27x9x9x9  3^3          child  -              10900      34740    2.33     64.4  1.7e-15
  32x32x32x32  2^5          child  -             174550     577962    1.92    300.4  1.2e-15
  32x32x32x64  2^5          child  -             574200     786406    1.30    191.7  1.3e-15
    35x15x9x5  5.7          child  -              17617      43175    1.89     48.7  1.4e-15
  40x30x20x10  2^3.5        child  -              55769     188481    1.91    192.3  1.3e-15
  45x45x15x15  3^2.5        child  -             183875     323649    1.48    116.4  1.9e-15
  48x48x48x48  2^4.3        child  -            2917000    3925781    1.34    101.6  1.8e-15
  53x16x16x16  53           child  -              56606     275330    3.52    170.0  1.9e-15
  64x16x16x16  2^6          child  -              42053     144860    2.25    280.5  1.1e-15
   64x64x64x4  2^6          child  -             256800    1142274    3.06    204.2  1.1e-15
  64x64x64x64  2^6          child  -           11255937   26070644    1.97     89.4  1.7e-15
   96x24x12x6  2^5.3        child  -              36775     204810    2.77    195.5  1.3e-15
    101x8x8x8  101          child  -              16905     107649    3.47    119.7  1.2e-15
 127x12x12x12  127          child  -              58500     333564    3.21    166.4  2.0e-15
  128x128x8x8  2^7          child  -             210375     908456    2.85    249.2  1.0e-15
    256x8x8x8  2^8          child  -              29753     140993    3.23    187.2  8.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           36      0      1   1.34   2.36   4.04    2.44
 ALL                             36      0      1   1.34   2.36   4.04    2.44
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      0      0   1.40   2.36   4.04    2.57
 even column                      8      0      0   1.34   2.49   4.71    2.50
 odd column                       6      0      1   0.93   1.69   2.64    1.66
 prime column                     4      0      0   2.78   3.34   3.52    3.23
 ALL                             36      0      1   1.34   2.36   4.04    2.44
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  25      0      0   1.34   2.34   3.22    2.20
 4097..65536                      8      0      1   0.93   2.49   4.71    2.53
 1025..4096                       2      0      0   2.78   3.41   4.04    3.35
 <= 256 points                    1      0      0  11.89  11.89  11.89   11.89
 ALL                             36      0      1   1.34   2.36   4.04    2.44
```


worst 10: 15x15x15x15 (child 0.93), 25x25x25x25 (child 1.28), 32x32x32x64 (child 1.30), 48x48x48x48 (child 1.34), 8x8x128x128 (child 1.40), 45x45x15x15 (child 1.48), 16x16x16x53 (child 1.74), 24x24x24x24 (child 1.81), 35x15x9x5 (child 1.89), 40x30x20x10 (child 1.91)
best 5: 4x4x4x4 (child 11.89), 12x12x12x12 (child 4.71), 10x12x14x16 (child 4.07), 8x8x8x8 (child 4.04), 53x16x16x16 (child 3.52)
