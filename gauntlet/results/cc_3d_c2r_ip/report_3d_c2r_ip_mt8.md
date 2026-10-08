# gauntlet report (3D)

run: `cc_3d_c2r_ip`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_ip_mt8  cells: 11 listed, 11 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 0.770..0.799

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          plane+payonce raced         731200     497225    0.67    150.6  1.3e-15
     9x16x30  3^2          plane+payonce+strips raced           4242       3225    0.70     30.7  9.4e-16
    16x16x16  2^4          plane+child raced           3021       2591    0.61     40.7  6.7e-16
  16x64x1024  2^4          plane+payonce raced         213663     171762    0.79    245.4  1.1e-15
  16x256x256  2^4          plane+payonce raced         246300     219344    0.85    212.9  1.1e-15
   32x8x1000  2^5          plane+payonce raced          55127      63343    1.12    208.6  1.7e-15
    32x32x32  2^5          plane+payonce raced           9534      10189    0.98    128.9  8.9e-16
  32x256x256  2^5          plane+payonce raced         529275     498887    0.91    208.0  1.2e-15
    64x64x64  2^6          plane+payonce raced          71233      56217    0.78    165.6  1.0e-15
  64x128x128  2^6          plane+payonce raced         233088     202744    0.78    224.9  1.3e-15
 128x128x128  2^7          plane+payonce raced         473613     410849    0.85    232.5  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+payonce                    9      4      8   0.67   0.85   1.12    0.85
 plane+payonce+strips             1      1      1   0.70   0.70   0.70    0.70
 plane+child                      1      1      1   0.61   0.61   0.61    0.61
 ALL                             11      6     10   0.67   0.79   0.98    0.81
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      5      9   0.67   0.82   1.12    0.82
 odd column                       1      1      1   0.70   0.70   0.70    0.70
 ALL                             11      6     10   0.67   0.79   0.98    0.81
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      4      7   0.67   0.82   1.12    0.84
 4097..65536                      2      1      2   0.70   0.84   0.98    0.83
 1025..4096                       1      1      1   0.61   0.61   0.61    0.61
 ALL                             11      6     10   0.67   0.79   0.98    0.81
```


worst 10: 16x16x16 (plane+child 0.61), 8x128x2048 (plane+payonce 0.67), 9x16x30 (plane+payonce+strips 0.70), 64x128x128 (plane+payonce 0.78), 64x64x64 (plane+payonce 0.78), 16x64x1024 (plane+payonce 0.79), 128x128x128 (plane+payonce 0.85), 16x256x256 (plane+payonce 0.85), 32x256x256 (plane+payonce 0.91), 32x32x32 (plane+payonce 0.98)
best 5: 32x8x1000 (plane+payonce 1.12), 32x32x32 (plane+payonce 0.98), 32x256x256 (plane+payonce 0.91), 16x256x256 (plane+payonce 0.85), 128x128x128 (plane+payonce 0.85)
