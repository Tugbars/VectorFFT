# gauntlet report (7D)

run: `rank7_avx2_2026-11-22`  contract: 7D r2c interleaved, natural, out of place, K=1_r2c_est  cells: 14 listed, 14 benched, comparator: MKL DFTI 7D (out of place)

control cell 4x4x4x4x4x4x4: 4 readings, 31.863..32.012


## every shape

```
           shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
   2x2x2x2x2x2x2  2            child  -               1124      16883   14.99      2.0  0.0e+00
  2x2x2x2x2x2x53  2            child  -              13554      72666    5.30      7.3  4.7e-16
   2x3x4x5x6x7x8  2            child  -              44516     959433   21.43     34.6  5.1e-16
   2x4x8x2x4x8x2  2            child  -              25527     596757   23.31     10.4  2.2e-16
   3x3x3x3x3x3x3  3            child  -               6328     135289   21.14      9.6  5.8e-16
   3x5x7x3x5x7x3  3            child  -              71871    1375788   19.13     17.3  6.0e-16
   4x4x4x4x4x4x4  2^2          child  -              21023     675104   31.95     27.3  0.0e+00
   4x4x4x4x8x8x8  2^2          child  -             117327    2388291   20.20     47.5  2.4e-16
   5x5x5x5x5x5x5  5            child  -             166888    2189390   12.91     19.0  4.5e-16
   6x6x6x6x6x6x6  2.3          child  -             343157    4951407   14.37     36.9  7.9e-16
   8x8x8x8x8x8x4  2^3          child  -            1320100   23119550   17.36     39.7  3.4e-16
   8x8x8x8x8x8x8  2^3          child  -            2482250   25988031   10.32     44.4  3.7e-16
 10x8x6x4x6x8x10  2.5          child  -             882725   10716675   12.08     51.7  5.8e-16
  53x2x2x2x2x2x2  53           child  -              34820     424146   11.69      2.9  1.1e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           14      0      0  10.32  16.17  23.31   15.56
 ALL                             14      0      0  10.32  16.17  23.31   15.56
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   5.30  18.78  31.95   16.13
 prime column                     4      0      0  11.69  16.02  21.14   15.72
 even column                      2      0      0  12.08  13.22  14.37   13.17
 ALL                             14      0      0  10.32  16.17  23.31   15.56
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0  10.32  13.64  20.20   14.17
 4097..65536                      4      0      0  19.13  22.37  31.95   23.51
 1025..4096                       3      0      0   5.30  11.69  21.14   10.94
 <= 256 points                    1      0      0  14.99  14.99  14.99   14.99
 ALL                             14      0      0  10.32  16.17  23.31   15.56
```


worst 10: 2x2x2x2x2x2x53 (child 5.30), 8x8x8x8x8x8x8 (child 10.32), 53x2x2x2x2x2x2 (child 11.69), 10x8x6x4x6x8x10 (child 12.08), 5x5x5x5x5x5x5 (child 12.91), 6x6x6x6x6x6x6 (child 14.37), 2x2x2x2x2x2x2 (child 14.99), 8x8x8x8x8x8x4 (child 17.36), 3x5x7x3x5x7x3 (child 19.13), 4x4x4x4x8x8x8 (child 20.20)
best 5: 4x4x4x4x4x4x4 (child 31.95), 2x4x8x2x4x8x2 (child 23.31), 2x3x4x5x6x7x8 (child 21.43), 3x3x3x3x3x3x3 (child 21.14), 4x4x4x4x8x8x8 (child 20.20)
