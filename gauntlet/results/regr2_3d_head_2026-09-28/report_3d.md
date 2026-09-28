# gauntlet report (3D)

run: `regr2_3d_head_2026-09-28`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 50 listed, 50 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.251..1.286


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  2x256x4096  2            chain  raced        8934200   11656181    1.28     24.6  5.6e-15
   2x512x243  2            chain  raced         711875    1279650    1.79     31.3  7.8e-16
       4x4x4  2^2          chain  replayed          83         71    0.86     23.2  3.3e-16
   4x64x8192  2^2          chain  raced        7841875    8707662    1.08     28.1  2.9e-15
   4x128x135  2^2          chain  raced         145729     230080    1.54     38.1  6.7e-16
       7x8x8  7            chain  raced            348       1839    4.54     56.7  2.9e-16
      7x9x11  7            chain  raced            737       2776    3.75     44.4  3.7e-16
       8x8x8  2^3          chain  raced            385        366    0.94     59.8  2.4e-16
     8x12x16  2^3          chain  raced           1350       3546    2.62     60.2  3.2e-16
   8x16x1009  2^3          chain  raced         723527     932200    1.25     15.2  1.0e-15
     8x23x16  2^3          chain  raced           4001       9293    2.15     42.4  9.4e-16
    8x64x243  2^3          chain  raced         231844     369362    1.56     45.4  7.1e-16
  8x128x2048  2^3          chain  raced        7891200    8215531    1.01     27.9  2.0e-15
   8x8192x32  2^3          chain  raced        9094038    9051850    0.99     24.2  2.0e-15
       9x9x9  3^2          chain  raced            849        805    0.92     40.8  6.6e-16
     12x8x16  2^2.3        chain  raced           1348       4002    2.93     60.3  3.0e-16
    12x12x12  2^2.3        chain  raced           1607       1434    0.87     57.8  7.3e-16
    12x20x45  2^2.3        chain  raced          15815      29103    1.82     45.7  5.4e-16
    15x15x15  3.5          chain  raced           5515       4202    0.67     35.9  5.9e-16
    16x16x16  2^4          chain  raced           4976       7048    1.41     49.4  3.5e-16
    16x16x23  2^4          chain  raced           8907      22791    2.30     41.4  5.3e-16
   16x16x512  2^4          chain  raced         236360     368096    1.47     47.1  3.5e-16
    16x32x45  2^4          chain  raced          35121      50957    1.29     47.5  5.3e-16
    16x64x25  2^4          chain  raced          43327      57522    1.32     43.3  4.3e-16
    17x17x17  17           chain  raced           8264      31594    3.81     36.4  9.7e-16
      23x8x8  23           chain  raced           2024       6957    3.28     38.3  8.6e-16
    24x24x24  2^3.3        chain  raced          19994      35964    1.78     47.5  5.1e-16
     27x9x15  3^3          chain  raced           5898       8816    1.49     36.6  5.6e-16
    31x32x32  31           chain  raced          66484     160976    2.21     35.7  6.5e-16
    32x16x81  2^5          chain  raced          65125      85435    1.16     48.8  6.5e-16
    32x32x32  2^5          chain  raced          44951      62120    1.37     54.7  3.5e-16
    32x32x75  2^5          chain  raced         135696     173494    1.27     45.9  6.1e-16
   32x32x256  2^5          chain  raced         580312     818537    1.39     40.7  4.2e-16
    36x20x28  2^2.3^2      chain  raced          29829      56457    1.85     48.3  4.4e-16
    45x45x45  3^2.5        chain  raced         200676     258985    1.29     37.4  6.8e-16
    48x48x48  2^4.3        chain  raced         199494     272980    1.21     46.4  7.6e-16
    60x60x60  2^2.3.5      chain  raced         421833     556566    1.29     45.4  7.1e-16
   64x4x4096  2^6          chain  raced        2578575    3526425    1.36     40.7  1.2e-15
    64x32x15  2^6          chain  raced          52954      63770    1.18     43.2  5.4e-16
    64x64x64  2^6          chain  raced         618725     803437    1.28     38.1  5.1e-16
    81x27x27  3^4          chain  raced         116248     165282    1.39     40.3  7.0e-16
    96x96x96  2^5.3        chain  raced        2101388    2287200    1.08     41.6  6.1e-16
    97x16x16  97           tpc    raced         107045     196429    1.83     16.9  7.5e-16
   120x16x64  2^3.3.5      chain  raced         211906     260143    1.21     49.0  4.9e-16
    125x25x5  5^3          chain  raced          31917      66160    2.06     34.1  4.0e-16
    128x8x27  2^7          chain  raced          44136      82704    1.84     46.2  5.9e-16
   128x16x16  2^7          chain  replayed       44254      59656    1.34     55.5  3.7e-16
 128x128x128  2^7          chain  raced        7005312   10457225    1.49     31.4  1.6e-15
     256x4x4  2^8          chain  raced           8184      42755    5.20     30.0  4.4e-16
    2048x8x8  2^11         chain  raced         278353     620996    2.13     40.0  4.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           49      1      6   0.94   1.39   3.28    1.56
 tpc                              1      0      0   1.83   1.83   1.83    1.83
 ALL                             50      1      6   0.99   1.39   3.28    1.57
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     29      0      3   0.99   1.36   2.30    1.47
 even column                      9      0      1   0.87   1.29   2.93    1.47
 prime column                     6      0      0   1.83   3.52   4.54    3.09
 odd column                       6      1      2   0.67   1.34   2.06    1.23
 ALL                             50      1      6   0.99   1.39   3.28    1.57
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  20      0      1   1.08   1.29   1.79    1.33
 4097..65536                     16      0      0   1.18   1.80   2.30    1.70
 1025..4096                       9      1      2   0.67   2.15   5.20    1.91
 257..1024                        4      0      2   0.92   2.35   4.54    1.96
 <= 256 points                    1      0      1   0.86   0.86   0.86    0.86
 ALL                             50      1      6   0.99   1.39   3.28    1.57
```


worst 10: 15x15x15 (chain 0.67), 4x4x4 (chain 0.86), 12x12x12 (chain 0.87), 9x9x9 (chain 0.92), 8x8x8 (chain 0.94), 8x8192x32 (chain 0.99), 8x128x2048 (chain 1.01), 96x96x96 (chain 1.08), 4x64x8192 (chain 1.08), 32x16x81 (chain 1.16)
best 5: 256x4x4 (chain 5.20), 7x8x8 (chain 4.54), 17x17x17 (chain 3.81), 7x9x11 (chain 3.75), 23x8x8 (chain 3.28)
