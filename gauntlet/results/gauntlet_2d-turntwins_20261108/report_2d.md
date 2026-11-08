# gauntlet report (2D)

run: `gauntlet_2d-turntwins_20261108`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 52 listed, 52 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.487..1.541


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            csk+rb raced              9         13    1.44      4.5  1.4e-16
         2x8  2            csk+rb raced             14        168   11.57     22.2  1.9e-16
        2x32  2            csk+rb raced             34        305    8.46     55.9  2.7e-16
        2x64  2            csk+rb raced             71        467    6.47     63.5  2.7e-16
         4x2  2^2          csk+rb raced             10        155   15.48     12.0  8.9e-17
         4x8  2^2          csk+rb raced             19        174    9.21     42.4  1.7e-16
        4x32  2^2          csk+rb raced             63        315    4.73     71.1  2.1e-16
        4x64  2^2          csk+rb raced            135        504    3.73     76.1  2.9e-16
         8x2  2^3          csk+rb raced             14        167   11.91     22.8  1.2e-16
         8x8  2^3          csk+rb raced             32         37    1.15     60.5  1.5e-16
        8x32  2^3          csk+rb2 raced            130        403    3.08     78.8  2.6e-16
        8x64  2^3          csk+rb raced            290        697    2.36     79.3  4.0e-16
        16x2  2^4          csk+rb raced             22        194    8.66     35.8  2.2e-16
        16x8  2^4          csk+rb raced             63        244    3.87     71.3  2.0e-16
       16x32  2^4          csk+rb2 raced            289        359    1.05     79.6  3.0e-16
       16x64  2^4          chain+rb raced            647        720    1.06     79.1  4.8e-16
        32x2  2^5          chain+rb raced             45        271    5.95     42.2  2.6e-16
        32x8  2^5          chain+rb raced            141        400    2.83     72.6  1.9e-16
       32x32  2^5          chain+rb2 raced            646        720    1.10     79.3  2.4e-16
       32x64  2^5          csk+rb raced           1832       1943    1.03     61.5  3.4e-16
        64x2  2^6          turn   raced             93        413    4.42     48.1  3.4e-16
        64x8  2^6          chain+rb raced            310        686    2.20     74.2  3.2e-16
       64x32  2^6          chain+rb raced           1774       2009    1.11     63.5  4.2e-16
       64x64  2^6          csk+rb raced           4698       7106    1.50     52.3  4.8e-16
       128x2  2^7          turn   raced            190        740    3.81     53.9  4.3e-16
       128x8  2^7          turn   raced            948       1528    1.60     54.0  3.2e-16
      128x32  2^7          chain+rb raced           5188       7153    1.35     47.4  3.6e-16
      128x64  2^7          chain+rb raced          10493      15165    1.40     50.7  4.0e-16
       256x2  2^8          turn   raced            387       1350    3.48     59.5  3.3e-16
       256x8  2^8          turn   raced           2371       3523    1.46     47.5  3.6e-16
      256x32  2^8          chain+rb raced          10750      14826    1.34     49.5  3.0e-16
      256x64  2^8          chain+rb raced          21766      31021    1.39     52.7  4.1e-16
       512x2  2^9          turn   raced            990       2902    2.85     51.7  3.7e-16
       512x8  2^9          turn   raced           5163       7497    1.44     47.6  4.1e-16
      512x32  2^9          chain+rb raced          22952      34778    1.51     50.0  4.3e-16
      512x64  2^9          chain+rb raced          46818      72009    1.28     52.5  4.0e-16
      1024x2  2^10         turn   raced           2603       6265    2.40     43.3  4.7e-16
      1024x8  2^10         turn   raced          11193      15474    1.38     47.6  3.0e-16
     1024x32  2^10         chain+rb2 raced          47954      76438    1.59     51.2  4.0e-16
     1024x64  2^10         chain+rb raced         108333     162953    1.50     48.4  4.8e-16
      2048x2  2^11         turn   raced           5734      12812    2.23     42.9  2.6e-16
      2048x8  2^11         turn   raced          24552      33063    1.33     46.7  3.7e-16
     2048x32  2^11         chain+rb2 raced         106257     159768    1.49     49.3  3.1e-16
     2048x64  2^11         chain+rb raced         259773     397746    1.50     42.9  4.7e-16
      4096x2  2^12         turn   raced          12022      26271    2.18     44.3  4.9e-16
      4096x8  2^12         turn   raced          51925      72342    1.30     47.3  3.9e-16
     4096x32  2^12         chain+rb2 raced         260027     374786    1.40     42.8  4.1e-16
     4096x64  2^12         chain+rb raced         540150     877693    1.55     43.7  5.3e-16
      8192x2  2^13         turn   raced          25111      53057    2.08     45.7  3.6e-16
      8192x8  2^13         turn   raced         126233     164483    1.28     41.5  4.1e-16
     8192x32  2^13         chain+rb raced         543188     945974    1.63     43.4  4.5e-16
     8192x64  2^13         chain+rb2 raced        1200538    2003262    1.64     41.5  1.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk+rb                          15      0      0   1.15   4.73  11.91    4.36
 chain+rb                        15      0      0   1.11   1.50   2.83    1.64
 turn                            15      0      0   1.30   2.08   3.81    2.03
 chain+rb2                        5      0      0   1.10   1.49   1.64    1.43
 csk+rb2                          2      0      0   1.05   2.07   3.08    1.80
 ALL                             52      0      0   1.15   1.60   8.46    2.29
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     52      0      0   1.15   1.60   8.46    2.29
 ALL                             52      0      0   1.15   1.60   8.46    2.29
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 <= 256 points                   17      0      0   1.44   4.73  11.91    5.06
 4097..65536                     14      0      0   1.28   1.40   2.08    1.48
 257..1024                        8      0      0   1.05   1.90   3.48    1.78
 1025..4096                       8      0      0   1.03   1.45   2.40    1.50
 > 65536 points                   5      0      0   1.40   1.55   1.64    1.54
 ALL                             52      0      0   1.15   1.60   8.46    2.29
```


worst 10: 32x64 (csk+rb 1.03), 16x32 (csk+rb2 1.05), 16x64 (chain+rb 1.06), 32x32 (chain+rb2 1.10), 64x32 (chain+rb 1.11), 8x8 (csk+rb 1.15), 512x64 (chain+rb 1.28), 8192x8 (turn 1.28), 4096x8 (turn 1.30), 2048x8 (turn 1.33)
best 5: 4x2 (csk+rb 15.48), 8x2 (csk+rb 11.91), 2x8 (csk+rb 11.57), 4x8 (csk+rb 9.21), 16x2 (csk+rb 8.66)
