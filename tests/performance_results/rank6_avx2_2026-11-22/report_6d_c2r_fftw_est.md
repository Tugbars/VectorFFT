# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=1_c2r_fftw_est  cells: 17 listed, 17 benched, comparator: FFTW 6D (out of place, MEASURE)

control cell 8x8x8x8x8x8: 4 readings, 0.966..0.973


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                974        662    0.68     11.8  3.3e-16
     2x4x6x8x10x12  2            child  -              48756      48333    0.98     36.6  1.0e-15
       3x3x3x3x3x3  3            child  -               3201        975    0.29      5.4  1.0e-15
       3x5x7x7x5x3  3            child  -              48309      14183    0.29      7.7  9.5e-16
       4x4x4x4x4x4  2^2          child  -               5809       4751    0.80     21.2  4.4e-16
      4x4x4x4x4x53  2^2          child  -             214604     461360    1.89      9.9  1.4e-15
       4x4x4x8x8x8  2^2          child  -              34922      32859    0.93     35.2  6.7e-16
       5x5x5x5x5x5  5            child  -              64758      18735    0.28      8.4  8.6e-16
       6x6x6x6x6x6  2.3          child  -              59525      52571    0.85     30.4  1.6e-15
       8x8x8x4x4x4  2^3          child  -              50098      40475    0.80     24.5  7.8e-16
       8x8x8x8x8x8  2^3          child  -             346760     330947    0.93     34.0  1.1e-15
       9x9x9x9x9x9  3^2          child  -            1505600     940431    0.50     16.8  2.3e-15
 10x10x10x10x10x10  2.5          child  -            1721425    1375950    0.80     28.9  1.4e-15
     12x10x8x6x4x2  2^2.3        child  -             231431      85632    0.37      7.7  9.6e-16
     16x4x4x4x4x16  2^4          child  -              73138      80645    1.09     35.8  7.2e-16
 16x16x16x16x16x16  2^4          child  -           53839038   59919450    1.04     18.7  1.4e-15
      53x4x4x4x4x4  53           child  -             130695     288382    2.18     16.3  1.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      8     13   0.29   0.80   1.89    0.73
 ALL                             17      8     13   0.29   0.80   1.89    0.73
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      2      6   0.68   0.93   1.89    0.97
 prime column                     4      3      3   0.28   0.29   2.18    0.47
 even column                      3      2      3   0.37   0.80   0.85    0.63
 odd column                       1      1      1   0.50   0.50   0.50    0.50
 ALL                             17      8     13   0.29   0.80   1.89    0.73
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      3      7   0.29   0.89   2.18    0.78
 > 65536 points                   4      2      3   0.50   0.86   1.04    0.79
 257..1024                        2      2      2   0.29   0.48   0.68    0.44
 1025..4096                       1      1      1   0.80   0.80   0.80    0.80
 ALL                             17      8     13   0.29   0.80   1.89    0.73
```


worst 10: 5x5x5x5x5x5 (child 0.28), 3x3x3x3x3x3 (child 0.29), 3x5x7x7x5x3 (child 0.29), 12x10x8x6x4x2 (child 0.37), 9x9x9x9x9x9 (child 0.50), 2x4x2x4x2x4 (child 0.68), 10x10x10x10x10x10 (child 0.80), 4x4x4x4x4x4 (child 0.80), 8x8x8x4x4x4 (child 0.80), 6x6x6x6x6x6 (child 0.85)
best 5: 53x4x4x4x4x4 (child 2.18), 4x4x4x4x4x53 (child 1.89), 16x4x4x4x4x16 (child 1.09), 16x16x16x16x16x16 (child 1.04), 2x4x6x8x10x12 (child 0.98)
