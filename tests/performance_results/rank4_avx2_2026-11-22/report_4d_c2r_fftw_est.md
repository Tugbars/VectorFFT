# gauntlet report (4D)

run: `rank4_avx2_2026-11-22`  contract: 4D c2r interleaved, natural, out of place, K=1_c2r_fftw_est  cells: 36 listed, 36 benched, comparator: FFTW 4D (out of place, MEASURE)

control cell 16x16x16x16: 4 readings, 1.175..1.197


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x64x64x64  2            child  -             761713     734200    0.96     32.7  1.1e-15
      4x4x4x4  2^2          child  -                257        234    0.87     19.9  3.3e-16
   4x32x32x32  2^2          child  -             134637     130513    0.95     41.4  8.9e-16
     5x7x9x11  5            child  -               8471       3324    0.38     12.0  9.3e-16
   6x12x24x96  2.3          child  -             190317     206779    1.07     37.8  1.5e-15
      8x8x8x8  2^3          child  -               2773       3007    1.05     44.3  7.5e-16
    8x8x8x256  2^3          child  -             123477     119883    0.94     45.1  1.1e-15
  8x8x128x128  2^3          child  -            1620075    1313837    0.77     32.4  1.3e-15
      9x9x9x9  3^2          child  -              14050       6118    0.43     14.8  1.5e-15
  10x12x14x16  2.5          child  -              22597      23432    1.02     43.8  9.2e-16
  12x12x12x12  2^2.3        child  -              15296      17276    1.12     48.6  1.2e-15
  15x15x15x15  3.5          child  -             126441      46768    0.37     15.6  1.5e-15
  16x16x16x16  2^4          child  -              60349      72368    1.13     43.4  7.8e-16
  16x16x16x32  2^4          child  -             138777     157903    1.10     40.1  8.9e-16
  16x16x16x53  2^4          child  -             906933    1889969    1.86     10.6  1.6e-15
  16x16x16x64  2^4          child  -             286373     322383    1.11     41.2  8.9e-16
  16x53x16x16  2^4          child  -             381200     827639    2.16     25.2  2.1e-15
  20x20x20x20  2^2.5        child  -             311921     167966    0.54     22.2  1.1e-15
  24x24x24x24  2^3.3        child  -             526400     824541    1.54     28.9  1.4e-15
  25x25x25x25  5^2          child  -            1053470     625615    0.59     17.2  1.6e-15
     27x9x9x9  3^3          child  -              45535      29507    0.64     15.4  1.7e-15
  32x32x32x32  2^5          child  -            1462425    1634681    1.05     35.9  1.1e-15
  32x32x32x64  2^5          child  -            3969800    3579987    0.89     27.7  1.4e-15
    35x15x9x5  5.7          child  -              88121      38322    0.42      9.7  1.2e-15
  40x30x20x10  2^3.5        child  -             373988     450890    1.17     28.7  1.2e-15
  45x45x15x15  3^2.5        child  -            1315988     943512    0.71     16.3  1.7e-15
  48x48x48x48  2^4.3        child  -           11413900   17905850    1.54     26.0  1.7e-15
  53x16x16x16  53           child  -             366061     870991    2.35     26.3  1.9e-15
  64x16x16x16  2^6          child  -             294540     368880    1.24     40.1  1.0e-15
   64x64x64x4  2^6          child  -            2727625    2836725    1.02     19.2  1.1e-15
  64x64x64x64  2^6          child  -           40845562   44304850    1.07     24.6  1.4e-15
   96x24x12x6  2^5.3        child  -             251088     346212    1.37     28.6  1.3e-15
    101x8x8x8  101          child  -              82138     242271    2.91     24.6  1.4e-15
 127x12x12x12  127          child  -             403117     792364    1.94     24.1  1.9e-15
  128x128x8x8  2^7          child  -            1867150    1946462    1.02     28.1  1.1e-15
    256x8x8x8  2^8          child  -             174017     176228    0.99     32.0  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           36      9     15   0.43   1.04   1.94    1.00
 ALL                             36      9     15   0.43   1.04   1.94    1.00
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      1      7   0.87   1.04   1.86    1.09
 even column                      8      1      1   0.54   1.15   1.54    1.12
 odd column                       6      6      6   0.37   0.51   0.71    0.51
 prime column                     4      1      1   0.38   2.15   2.91    1.50
 ALL                             36      9     15   0.43   1.04   1.94    1.00
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  25      4      9   0.71   1.07   1.94    1.12
 4097..65536                      8      4      4   0.37   0.83   2.91    0.80
 1025..4096                       2      1      1   0.38   0.72   1.05    0.63
 <= 256 points                    1      0      1   0.87   0.87   0.87    0.87
 ALL                             36      9     15   0.43   1.04   1.94    1.00
```


worst 10: 15x15x15x15 (child 0.37), 5x7x9x11 (child 0.38), 35x15x9x5 (child 0.42), 9x9x9x9 (child 0.43), 20x20x20x20 (child 0.54), 25x25x25x25 (child 0.59), 27x9x9x9 (child 0.64), 45x45x15x15 (child 0.71), 8x8x128x128 (child 0.77), 4x4x4x4 (child 0.87)
best 5: 101x8x8x8 (child 2.91), 53x16x16x16 (child 2.35), 16x53x16x16 (child 2.16), 127x12x12x12 (child 1.94), 16x16x16x53 (child 1.86)
