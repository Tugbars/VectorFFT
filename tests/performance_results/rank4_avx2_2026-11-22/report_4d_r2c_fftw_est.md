# gauntlet report (4D)

run: `rank4_avx2_2026-11-22`  contract: 4D r2c interleaved, natural, out of place, K=1_r2c_fftw_est  cells: 36 listed, 36 benched, comparator: FFTW 4D (out of place, MEASURE)

control cell 16x16x16x16: 4 readings, 1.391..1.489


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x64x64x64  2            child  -             747413     728744    0.95     33.3  4.6e-16
      4x4x4x4  2^2          child  -                249        269    0.95     20.5  0.0e+00
   4x32x32x32  2^2          child  -             126213     130176    1.00     44.1  3.2e-16
     5x7x9x11  5            child  -               4581       3300    0.71     22.2  4.2e-16
   6x12x24x96  2.3          child  -             167358     222129    1.29     43.0  4.8e-16
      8x8x8x8  2^3          child  -               2488       3564    1.43     49.4  2.9e-16
    8x8x8x256  2^3          child  -             100693     116986    1.14     55.3  3.4e-16
  8x8x128x128  2^3          child  -            1268650    1251881    0.79     41.3  4.2e-16
      9x9x9x9  3^2          child  -               9267       7084    0.75     22.4  5.1e-16
  10x12x14x16  2.5          child  -              20164      26495    1.30     49.0  5.2e-16
  12x12x12x12  2^2.3        child  -              14879      19902    1.33     50.0  8.5e-16
  15x15x15x15  3.5          child  -              92173      56153    0.55     21.5  5.5e-16
  16x16x16x16  2^4          child  -              55585      77551    1.39     47.2  3.5e-16
  16x16x16x32  2^4          child  -             127917     152318    1.15     43.5  3.1e-16
  16x16x16x53  2^4          child  -             598072    1852955    3.04     16.1  6.8e-16
  16x16x16x64  2^4          child  -             273587     336920    1.22     43.1  3.6e-16
  16x53x16x16  2^4          child  -             342350     873605    2.54     28.1  6.8e-16
  20x20x20x20  2^2.5        child  -             296304     193189    0.64     23.3  4.5e-16
  24x24x24x24  2^3.3        child  -             546267     791796    1.44     27.8  6.4e-16
  25x25x25x25  5^2          child  -             973490     730480    0.72     18.6  5.1e-16
     27x9x9x9  3^3          child  -              33700      32764    0.96     20.8  6.4e-16
  32x32x32x32  2^5          child  -            1330712    1631950    1.23     39.4  4.4e-16
  32x32x32x64  2^5          child  -            2833100    3634399    1.26     38.9  4.1e-16
    35x15x9x5  5.7          child  -              51413      38796    0.75     16.7  5.6e-16
  40x30x20x10  2^3.5        child  -             301969     486762    1.61     35.5  4.4e-16
  45x45x15x15  3^2.5        child  -             936250    1000456    1.06     22.9  7.2e-16
  48x48x48x48  2^4.3        child  -           10535588   17763894    1.61     28.1  6.5e-16
  53x16x16x16  53           child  -             326250     918030    2.69     29.5  6.8e-16
  64x16x16x16  2^6          child  -             262780     407460    1.53     44.9  3.2e-16
   64x64x64x4  2^6          child  -            3077538    3300938    1.03     17.0  4.3e-16
  64x64x64x64  2^6          child  -           38698112   42917893    1.01     26.0  4.7e-16
   96x24x12x6  2^5.3        child  -             240258     352175    1.46     29.9  5.9e-16
    101x8x8x8  101          child  -              82994     252055    3.03     24.4  1.4e-15
 127x12x12x12  127          child  -             374467     890502    2.36     26.0  7.9e-16
  128x128x8x8  2^7          child  -            1742463    2314637    1.29     30.1  4.7e-16
    256x8x8x8  2^8          child  -             148507     207560    1.39     37.5  3.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           36      7     11   0.72   1.24   2.54    1.23
 ALL                             36      7     11   0.72   1.24   2.54    1.23
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     18      1      4   0.95   1.22   2.54    1.27
 even column                      8      1      1   0.64   1.38   1.61    1.29
 odd column                       6      4      5   0.55   0.75   1.06    0.78
 prime column                     4      1      1   0.71   2.52   3.03    1.92
 ALL                             36      7     11   0.72   1.24   2.54    1.23
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  25      3      5   0.79   1.26   2.54    1.31
 4097..65536                      8      3      4   0.55   1.13   3.03    1.10
 1025..4096                       2      1      1   0.71   1.07   1.43    1.00
 <= 256 points                    1      0      1   0.95   0.95   0.95    0.95
 ALL                             36      7     11   0.72   1.24   2.54    1.23
```


worst 10: 15x15x15x15 (child 0.55), 20x20x20x20 (child 0.64), 5x7x9x11 (child 0.71), 25x25x25x25 (child 0.72), 9x9x9x9 (child 0.75), 35x15x9x5 (child 0.75), 8x8x128x128 (child 0.79), 2x64x64x64 (child 0.95), 4x4x4x4 (child 0.95), 27x9x9x9 (child 0.96)
best 5: 16x16x16x53 (child 3.04), 101x8x8x8 (child 3.03), 53x16x16x16 (child 2.69), 16x53x16x16 (child 2.54), 127x12x12x12 (child 2.36)
