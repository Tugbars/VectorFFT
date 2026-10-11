# gauntlet report (3D)

run: `rb_3d_r2c_fftw_replay`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 59 listed, 59 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 4 readings, 1.080..1.095


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       4x4x4  2^2          child  raced             55         47    0.84     17.3  0.0e+00
       4x4x8  2^2          child  raced             70         79    1.11     32.1  2.7e-16
     4x4x256  2^2          child  raced           2359       2555    1.04     52.1  3.0e-16
       4x8x4  2^2          payonce raced            122         89    0.73     18.4  1.5e-16
     4x8x256  2^2          child  raced           5224       5484    0.92     51.0  3.1e-16
     4x32x16  2^2          child  raced           1095       1467    1.32     51.5  2.4e-16
    4x32x256  2^2          child  raced          24197      24492    1.01     50.8  3.8e-16
    4x64x256  2^2          child  raced          50948      52278    1.02     51.5  4.1e-16
       8x4x4  2^3          child  raced            123        100    0.81     18.2  1.4e-16
       8x4x8  2^3          child  raced            138        167    1.20     37.1  1.8e-16
       8x8x4  2^3          payonce raced            233        181    0.78     22.0  4.0e-16
    8x128x32  2^3          child  raced          29211      31562    1.05     42.1  4.0e-16
     8x256x4  2^3          child  raced          12882      11476    0.89     20.7  3.1e-16
     8x256x8  2^3          payonce raced          19291      19404    1.00     29.7  3.8e-16
      16x4x4  2^4          child  raced            232        201    0.85     22.1  1.5e-16
      16x4x8  2^4          child  raced            297        343    1.15     38.8  2.6e-16
    16x4x128  2^4          payonce raced           5848       5941    1.00     45.5  3.0e-16
      16x8x4  2^4          payonce raced            485        384    0.78     23.7  1.9e-16
     16x32x4  2^4          child  raced           1930       1774    0.90     29.2  2.3e-16
     27x9x15  3^3          child  raced           5300       5009    0.94     20.3  5.2e-16
      32x4x4  2^5          child  raced            478        399    0.83     24.1  2.4e-16
      32x4x8  2^5          payonce raced            789        736    0.90     32.5  2.4e-16
     32x4x16  2^5          child  raced           1178       1497    1.23     47.8  2.7e-16
      32x8x4  2^5          payonce raced           1007        837    0.83     25.4  2.1e-16
      32x8x8  2^5          child  raced           1343       1533    1.11     41.9  2.7e-16
  32x128x256  2^5          payonce raced        1611787    1419831    0.81     32.5  4.6e-16
  32x256x128  2^5          payonce+strips raced        1692563    1538112    0.83     31.0  4.3e-16
      64x4x4  2^6          child  raced            962        843    0.82     26.6  3.1e-16
      64x4x8  2^6          child  raced           1389       1538    1.02     40.5  3.5e-16
      64x8x4  2^6          payonce raced           2401       1756    0.73     23.5  2.6e-16
      64x8x8  2^6          child  raced           3264       3584    1.07     37.6  2.5e-16
    64x8x256  2^6          child  raced         130970     113203    0.86     42.5  3.6e-16
   64x16x128  2^6          child  raced         129763     117486    0.90     42.9  4.0e-16
   64x16x256  2^6          child  raced         303947     264126    0.72     38.8  4.2e-16
    64x32x32  2^6          payonce raced          71121      68765    0.88     36.9  3.9e-16
   64x32x128  2^6          child  raced         308107     265340    0.83     38.3  3.5e-16
   64x32x256  2^6          child  raced         666150     571187    0.83     37.4  3.9e-16
     64x64x8  2^6          child  raced          45005      46056    0.93     27.3  3.5e-16
    64x64x15  2^6          child  raced          94823      77814    0.81     25.8  4.1e-16
    64x64x16  2^6          child  raced          83430      83227    0.97     31.4  3.9e-16
   64x64x256  2^6          payonce raced        1488750    1356112    0.87     35.2  4.4e-16
    64x128x4  2^6          child  raced          59469      59135    0.91     20.7  3.1e-16
    64x128x8  2^6          child  raced          97051      96158    0.93     27.0  4.0e-16
  64x128x128  2^6          child+strips raced        1453637    1403244    0.93     36.1  5.7e-16
     128x4x4  2^7          child  raced           2279       1856    0.81     24.7  2.1e-16
     128x4x8  2^7          child  raced           3538       3739    1.04     34.7  3.1e-16
   128x4x256  2^7          child  raced         131857     114901    0.84     42.2  3.4e-16
     128x8x4  2^7          payonce raced           5495       5008    0.90     22.4  2.8e-16
   128x8x128  2^7          child  raced         128493     119511    0.92     43.4  3.3e-16
   128x8x256  2^7          payonce raced         297260     262943    0.87     39.7  3.7e-16
   128x16x32  2^7          child  raced          66479      68569    0.95     39.4  3.5e-16
  128x16x128  2^7          child  raced         294547     264813    0.86     40.0  3.5e-16
  128x16x256  2^7          child  raced         588188     569481    0.89     42.3  4.1e-16
  128x32x128  2^7          child+strips raced         646162     648062    1.00     38.5  4.8e-16
  128x32x256  2^7          child+strips raced        1381188    1363887    0.91     38.0  4.3e-16
  128x64x128  2^7          child+strips raced        1582075    1442056    0.90     33.1  4.9e-16
  128x128x32  2^7          child+strips raced         770463     822744    1.06     32.3  4.4e-16
     256x4x4  2^8          child  raced           5039       5838    1.14     24.4  3.0e-16
     256x8x4  2^8          payonce raced          12113      13328    1.07     22.0  3.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           39      1     25   0.81   0.92   1.15    0.95
 payonce                         14      4     11   0.73   0.87   1.00    0.86
 child+strips                     5      0      4   0.90   0.93   1.06    0.96
 payonce+strips                   1      0      1   0.83   0.83   0.83    0.83
 ALL                             59      5     41   0.81   0.90   1.11    0.93
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     58      5     40   0.81   0.90   1.11    0.92
 odd column                       1      0      1   0.94   0.94   0.94    0.94
 ALL                             59      5     41   0.81   0.90   1.11    0.93
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  18      1     17   0.81   0.87   1.00    0.88
 4097..65536                     15      0      9   0.88   0.95   1.05    0.95
 1025..4096                      13      1      5   0.81   1.04   1.23    1.01
 <= 256 points                    7      2      5   0.73   0.84   1.20    0.89
 257..1024                        6      1      5   0.78   0.83   1.15    0.88
 ALL                             59      5     41   0.81   0.90   1.11    0.93
```


worst 10: 64x16x256 (child 0.72), 64x8x4 (payonce 0.73), 4x8x4 (payonce 0.73), 8x8x4 (payonce 0.78), 16x8x4 (payonce 0.78), 32x128x256 (payonce 0.81), 64x64x15 (child 0.81), 8x4x4 (child 0.81), 128x4x4 (child 0.81), 64x4x4 (child 0.82)
best 5: 4x32x16 (child 1.32), 32x4x16 (child 1.23), 8x4x8 (child 1.20), 16x4x8 (child 1.15), 256x4x4 (child 1.14)
