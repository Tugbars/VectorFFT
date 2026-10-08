# gauntlet report (3D)

run: `mt8_3d_c2r_ip`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_ip_mt8  cells: 11 listed, 11 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.529..0.572

threaded plans that ran serial (engaged = 0 at both flips): 1: 9x16x30


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          plane+payonce raced         800888     471931    0.58    137.5  1.3e-15
     9x16x30  3^2          payonce raced           5040       3198    0.62     25.9  8.6e-16
    16x16x16  2^4          plane+payonce raced           1806       2534    1.35     68.0  6.7e-16
  16x64x1024  2^4          plane+payonce raced         196137     171219    0.81    267.3  1.3e-15
  16x256x256  2^4          plane+payonce raced         615063     215787    0.33     85.2  1.1e-15
   32x8x1000  2^5          plane+payonce raced          54087      60876    1.07    212.6  1.7e-15
    32x32x32  2^5          plane+child raced           8420      10006    1.15    145.9  7.8e-16
  32x256x256  2^5          plane+payonce raced         673200     446438    0.64    163.5  1.2e-15
    64x64x64  2^6          plane+payonce+strips raced          97720      54160    0.54    120.7  1.1e-15
  64x128x128  2^6          plane+payonce raced         301238     199050    0.64    174.0  1.3e-15
 128x128x128  2^7          plane+payonce raced         477150     408393    0.83    230.7  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+payonce                    8      4      6   0.33   0.73   1.35    0.72
 payonce                          1      1      1   0.62   0.62   0.62    0.62
 plane+child                      1      0      0   1.15   1.15   1.15    1.15
 plane+payonce+strips             1      1      1   0.54   0.54   0.54    0.54
 ALL                             11      6      8   0.54   0.64   1.15    0.73
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      5      7   0.54   0.73   1.35    0.74
 odd column                       1      1      1   0.62   0.62   0.62    0.62
 ALL                             11      6      8   0.54   0.64   1.15    0.73
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      5      7   0.33   0.64   1.07    0.65
 4097..65536                      2      1      1   0.62   0.89   1.15    0.85
 1025..4096                       1      0      0   1.35   1.35   1.35    1.35
 ALL                             11      6      8   0.54   0.64   1.15    0.73
```


worst 10: 16x256x256 (plane+payonce 0.33), 64x64x64 (plane+payonce+strips 0.54), 8x128x2048 (plane+payonce 0.58), 9x16x30 (payonce 0.62), 32x256x256 (plane+payonce 0.64), 64x128x128 (plane+payonce 0.64), 16x64x1024 (plane+payonce 0.81), 128x128x128 (plane+payonce 0.83), 32x8x1000 (plane+payonce 1.07), 32x32x32 (plane+child 1.15)
best 5: 16x16x16 (plane+payonce 1.35), 32x32x32 (plane+child 1.15), 32x8x1000 (plane+payonce 1.07), 128x128x128 (plane+payonce 0.83), 16x64x1024 (plane+payonce 0.81)
