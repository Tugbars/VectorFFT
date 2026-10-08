# gauntlet report (3D)

run: `ip3d_c2r_ip_fftw`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_ip_fftw  cells: 11 listed, 11 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.079..1.091


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          child  raced        4179750    2360906    0.56     26.3  1.3e-15
     9x16x30  3^2          child  raced           5665       4223    0.74     23.0  9.7e-16
    16x16x16  2^4          child  raced           2858       3108    1.08     43.0  6.7e-16
  16x64x1024  2^4          child  raced        1372825    1050300    0.75     38.2  1.3e-15
  16x256x256  2^4          child  raced        1732412    1705043    0.95     30.3  1.2e-15
   32x8x1000  2^5          child  raced         349653     264546    0.75     32.9  1.5e-15
    32x32x32  2^5          child  raced          32915      31957    0.96     37.3  8.9e-16
  32x256x256  2^5          child  raced        3503825    3508631    0.96     31.4  1.4e-15
    64x64x64  2^6          child  raced         295440     329870    1.09     39.9  1.1e-15
  64x128x128  2^6          child  raced        1350613    1415444    1.05     38.8  1.4e-15
 128x128x128  2^7          child  raced        2761888    3046262    1.09     39.9  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           11      4      7   0.74   0.96   1.09    0.89
 ALL                             11      4      7   0.74   0.96   1.09    0.89
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      3      6   0.75   0.96   1.09    0.91
 odd column                       1      1      1   0.74   0.74   0.74    0.74
 ALL                             11      4      7   0.74   0.96   1.09    0.89
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      3      5   0.56   0.96   1.09    0.88
 4097..65536                      2      1      2   0.74   0.85   0.96    0.84
 1025..4096                       1      0      0   1.08   1.08   1.08    1.08
 ALL                             11      4      7   0.74   0.96   1.09    0.89
```


worst 10: 8x128x2048 (child 0.56), 9x16x30 (child 0.74), 32x8x1000 (child 0.75), 16x64x1024 (child 0.75), 16x256x256 (child 0.95), 32x32x32 (child 0.96), 32x256x256 (child 0.96), 64x128x128 (child 1.05), 16x16x16 (child 1.08), 64x64x64 (child 1.09)
best 5: 128x128x128 (child 1.09), 64x64x64 (child 1.09), 16x16x16 (child 1.08), 64x128x128 (child 1.05), 32x256x256 (child 0.96)
