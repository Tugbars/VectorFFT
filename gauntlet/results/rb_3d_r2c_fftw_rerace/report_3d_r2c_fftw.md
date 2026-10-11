# gauntlet report (3D)

run: `rb_3d_r2c_fftw_rerace`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 59 listed, 59 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.083..1.183


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       4x4x4  2^2          child  raced             50         47    0.94     19.3  0.0e+00
       4x4x8  2^2          child  raced             69         79    1.15     32.4  2.7e-16
     4x4x256  2^2          child  raced           2564       3189    1.00     47.9  3.0e-16
       4x8x4  2^2          child  raced             82         89    1.07     27.2  1.5e-16
     4x8x256  2^2          child  raced           5233       5453    1.03     50.9  3.4e-16
     4x32x16  2^2          child  raced           1066       1434    1.33     52.8  2.4e-16
    4x32x256  2^2          child  raced          23849      24539    1.02     51.5  3.8e-16
    4x64x256  2^2          child  raced          50346      52192    1.02     52.1  4.3e-16
       8x4x4  2^3          child  raced             94        101    1.08     23.9  1.4e-16
       8x4x8  2^3          child  raced            133        167    1.19     38.5  1.8e-16
       8x8x4  2^3          child  raced            160        181    1.11     32.0  4.0e-16
    8x128x32  2^3          child  raced          29161      31548    1.06     42.1  3.6e-16
     8x256x4  2^3          child  raced           9535      11394    1.15     27.9  3.1e-16
     8x256x8  2^3          child  raced          15128      19314    1.27     37.9  2.8e-16
      16x4x4  2^4          child  raced            192        200    1.04     26.6  1.5e-16
      16x4x8  2^4          child  raced            310        344    1.10     37.2  2.6e-16
    16x4x128  2^4          child  raced           5060       5986    1.17     52.6  3.0e-16
      16x8x4  2^4          child  raced            346        383    1.11     33.3  1.9e-16
     16x32x4  2^4          child  raced           1531       1751    1.13     36.8  2.3e-16
     27x9x15  3^3          child  raced           5202       5155    0.99     20.7  5.4e-16
      32x4x4  2^5          child  raced            413        404    0.96     27.9  3.6e-16
      32x4x8  2^5          child  raced            617        700    1.12     41.5  3.2e-16
     32x4x16  2^5          child  raced           1099       1461    1.32     51.2  2.7e-16
      32x8x4  2^5          child  raced            742        816    1.09     34.5  2.1e-16
      32x8x8  2^5          child  raced           1270       1467    1.10     44.4  2.7e-16
  32x128x256  2^5          child  raced        1408137    1400150    0.92     37.2  4.6e-16
  32x256x128  2^5          child+strips raced        2166038    1547362    0.68     24.2  4.3e-16
      64x4x4  2^6          child  raced            886        884    0.98     28.9  2.7e-16
      64x4x8  2^6          child  raced           1331       1524    1.11     42.3  3.5e-16
      64x8x4  2^6          child  raced           1619       1749    1.07     34.8  2.6e-16
      64x8x8  2^6          child  raced           2608       3668    1.37     47.1  2.8e-16
    64x8x256  2^6          child  raced         108840     113363    1.03     51.2  4.5e-16
   64x16x128  2^6          child  raced         111103     117800    1.05     50.1  3.6e-16
   64x16x256  2^6          child  raced         244240     261333    1.07     48.3  4.2e-16
    64x32x32  2^6          child  raced          61730      68939    1.11     42.5  3.9e-16
   64x32x128  2^6          child  raced         249520     264236    1.05     47.3  4.1e-16
   64x32x256  2^6          child  raced         533813     569825    1.06     46.7  4.7e-16
     64x64x8  2^6          child  raced          37207      45473    1.20     33.0  3.5e-16
    64x64x15  2^6          child+strips raced         101698      76493    0.75     24.0  4.7e-16
    64x64x16  2^6          child  raced          86195      83053    0.96     30.4  3.9e-16
   64x64x256  2^6          child  raced        1355488    1331844    0.95     38.7  4.4e-16
    64x128x4  2^6          child  raced          51151      61221    1.17     24.0  3.8e-16
    64x128x8  2^6          child  raced          80205      96781    1.19     32.7  4.6e-16
  64x128x128  2^6          child  raced        1341562    1394506    1.00     39.1  5.7e-16
     128x4x4  2^7          child  raced           2368       1813    0.76     23.8  2.1e-16
     128x4x8  2^7          child  raced           3685       3705    0.99     33.4  3.1e-16
   128x4x256  2^7          child  raced         127710     117410    0.92     43.6  3.4e-16
     128x8x4  2^7          child  raced           4122       4837    1.15     29.8  2.8e-16
   128x8x128  2^7          child  raced         126330     119053    0.87     44.1  3.3e-16
   128x8x256  2^7          child  raced         305320     269766    0.87     38.6  3.7e-16
   128x16x32  2^7          payonce raced          80280      68370    0.81     32.7  3.4e-16
  128x16x128  2^7          child  raced         305387     266293    0.86     38.6  3.5e-16
  128x16x256  2^7          child  raced         694262     566850    0.79     35.9  4.1e-16
  128x32x128  2^7          child+strips raced         645875     649437    0.98     38.6  4.8e-16
  128x32x256  2^7          child+strips raced        1368087    1338706    0.95     38.3  4.3e-16
  128x64x128  2^7          child+strips raced        1411275    1414356    1.00     37.1  4.9e-16
  128x128x32  2^7          child+strips raced         760125     815137    1.07     32.8  4.0e-16
     256x4x4  2^8          child  raced           4543       5867    1.25     27.0  3.8e-16
     256x8x4  2^8          child  raced           9416      13150    1.39     28.3  3.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           52      2     14   0.92   1.07   1.25    1.06
 child+strips                     6      2      5   0.68   0.97   1.07    0.89
 payonce                          1      0      1   0.81   0.81   0.81    0.81
 ALL                             59      4     20   0.86   1.06   1.25    1.04
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     58      4     19   0.86   1.06   1.25    1.04
 odd column                       1      0      1   0.99   0.99   0.99    0.99
 ALL                             59      4     20   0.86   1.06   1.25    1.04
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  18      2     11   0.79   0.97   1.07    0.95
 4097..65536                     15      1      3   0.81   1.11   1.27    1.07
 1025..4096                      13      1      3   0.99   1.11   1.33    1.11
 <= 256 points                    7      0      1   0.94   1.08   1.19    1.08
 257..1024                        6      0      2   0.96   1.10   1.12    1.06
 ALL                             59      4     20   0.86   1.06   1.25    1.04
```


worst 10: 32x256x128 (child+strips 0.68), 64x64x15 (child+strips 0.75), 128x4x4 (child 0.76), 128x16x256 (child 0.79), 128x16x32 (payonce 0.81), 128x16x128 (child 0.86), 128x8x256 (child 0.87), 128x8x128 (child 0.87), 128x4x256 (child 0.92), 32x128x256 (child 0.92)
best 5: 256x8x4 (child 1.39), 64x8x8 (child 1.37), 4x32x16 (child 1.33), 32x4x16 (child 1.32), 8x256x8 (child 1.27)
