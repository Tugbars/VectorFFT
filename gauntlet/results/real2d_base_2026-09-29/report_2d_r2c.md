# gauntlet report (2D)

run: `real2d_base_2026-09-29`  contract: 2D c2c interleaved, natural, out of place, K=1_r2c  cells: 67 listed, 67 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.026..1.057


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          chain  replayed         327        130    0.40     15.7  1.9e-16
       16x32  2^4          chain  replayed         424        520    1.22     27.2  2.8e-16
       16x64  2^4          chain  replayed         659        777    1.16     38.9  3.3e-16
      16x128  2^4          chain  replayed        1207       1450    1.18     46.7  3.7e-16
      16x256  2^4          chain  replayed        2239       3121    1.37     54.9  3.9e-16
      16x512  2^4          chain  replayed        7198       6679    0.91     37.0  4.1e-16
     16x1024  2^4          chain  replayed       15518      13638    0.87     37.0  5.2e-16
     16x4096  2^4          chain  replayed       66126      65977    0.99     39.6  3.8e-16
       23x64  23           chain  replayed        1253       2656    1.80     30.9  1.2e-15
       29x64  29           chain  replayed        1818       2894    1.58     27.7  5.8e-16
       32x16  2^5          chain  replayed         642        542    0.84     18.0  3.8e-16
       32x32  2^5          chain  replayed         910        598    0.66     28.1  3.2e-16
       32x64  2^5          chain  replayed        1409       1554    1.09     40.0  3.3e-16
      32x128  2^5          chain  replayed        2735       3053    1.09     44.9  3.4e-16
      32x256  2^5          chain  replayed        5146       6558    1.25     51.7  4.0e-16
      32x512  2^5          chain  replayed       11720      14236    0.91     48.9  4.1e-16
     32x1024  2^5          chain  replayed       23532      29096    1.13     52.2  4.7e-16
       47x64  47           chain  replayed        3533       7011    1.97     24.6  1.3e-15
       48x48  2^4.3        chain  replayed        2234       2841    1.27     28.8  4.8e-16
       63x64  3^2.7        chain  replayed        3773       3599    0.94     32.0  4.2e-16
       64x16  2^6          chain  replayed        1322       1041    0.78     19.4  3.1e-16
       64x30  2^6          chain  replayed        2266       1832    0.77     23.1  4.4e-16
       64x32  2^6          chain  replayed        1912       2000    1.03     29.5  3.4e-16
       64x50  2^6          chain  replayed        2884       3565    1.13     32.3  3.2e-16
       64x62  2^6          chain  replayed        5520       9502    1.70     21.5  4.5e-16
       64x64  2^6          chain  replayed        3072       3289    1.05     40.0  2.7e-16
      64x128  2^6          chain  replayed        6038       6653    1.10     44.1  3.3e-16
      64x256  2^6          chain  replayed       11427      14029    1.18     50.2  3.9e-16
      64x512  2^6          chain  replayed       24458      30575    1.21     50.2  4.6e-16
     64x1024  2^6          chain  replayed       53584      61861    1.15     48.9  5.4e-16
     64x4096  2^6          chain  replayed      261160     318873    1.22     45.2  4.0e-16
       96x96  2^5.3        chain  replayed        8537       9370    1.09     35.5  5.2e-16
      100x64  2^2.5^2      chain  replayed        6956       5310    0.76     29.1  4.2e-16
      128x16  2^7          chain  replayed        2831       2084    0.73     19.9  2.7e-16
      128x32  2^7          chain  replayed        5504       4130    0.74     22.3  2.7e-16
      128x64  2^7          chain  replayed        7968       6803    0.83     33.4  3.2e-16
     128x128  2^7          chain  replayed       15734      13517    0.86     36.4  3.0e-16
     128x256  2^7          chain  replayed       30092      28229    0.92     40.8  3.4e-16
     128x512  2^7          chain  replayed       65059      62829    0.96     40.3  3.8e-16
    128x1024  2^7          chain  replayed      168313     131713    0.77     33.1  5.3e-16
     192x192  2^6.3        chain  replayed       35778      39227    0.89     39.1  5.2e-16
      256x16  2^8          chain  replayed        6543       4846    0.73     18.8  4.5e-16
      256x32  2^8          chain  replayed        9655       9717    0.97     27.6  3.9e-16
      256x64  2^8          chain  replayed       17495      16551    0.94     32.8  3.8e-16
     256x128  2^8          chain  replayed       35611      33575    0.93     34.5  3.7e-16
     256x256  2^8          chain  replayed       64348      69890    1.08     40.7  5.4e-16
     256x512  2^8          chain  replayed      154757     153445    0.81     36.0  4.1e-16
    256x1024  2^8          chain  replayed      301847     327753    0.93     39.1  6.1e-16
     480x480  2^5.3.5      chain  replayed      295235     368917    1.23     34.8  5.5e-16
      512x16  2^9          chain  replayed       13949      11453    0.82     19.1  4.0e-16
      512x32  2^9          chain  replayed       20461      22852    1.10     28.0  3.9e-16
      512x64  2^9          chain  replayed       35070      40572    1.12     35.0  3.8e-16
     512x128  2^9          chain  replayed       69044      82365    1.14     38.0  3.7e-16
     512x256  2^9          chain  replayed      154917     173992    1.12     36.0  4.9e-16
     512x512  2^9          chain  replayed      346640     393426    1.12     34.0  5.1e-16
    512x1024  2^9          chain  replayed      624838     813319    1.28     39.9  6.6e-16
   1000x1000  2^3.5^3      chain  replayed     1672687    1784781    1.06     29.8  5.4e-16
     1024x16  2^10         chain  replayed       28943      23218    0.77     19.8  3.4e-16
     1024x32  2^10         chain  replayed       42505      46120    1.08     28.9  3.9e-16
     1024x64  2^10         chain  replayed       71675      81679    1.04     36.6  3.4e-16
    1024x128  2^10         chain  replayed      156803     168120    1.06     35.5  4.6e-16
    1024x256  2^10         chain  replayed      325093     370880    1.10     36.3  4.5e-16
    1024x512  2^10         chain  replayed      728137     804263    1.02     34.2  4.1e-16
   1024x1024  2^10         chain  replayed     1468662    1675044    1.01     35.7  5.6e-16
   2048x2048  2^11         chain  replayed     9013975   10385325    1.08     25.6  5.6e-16
     4096x16  2^12         chain  replayed      155159     107992    0.69     16.9  4.3e-16
     4096x64  2^12         chain  replayed      496147     404010    0.81     23.8  4.3e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           67     11     29   0.76   1.05   1.27    1.01
 ALL                             67     11     29   0.76   1.05   1.27    1.01
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     57     10     26   0.74   1.03   1.22    0.98
 even column                      6      1      2   0.76   1.08   1.27    1.03
 prime column                     3      0      0   1.58   1.80   1.97    1.78
 odd column                       1      0      1   0.94   0.94   0.94    0.94
 ALL                             67     11     29   0.76   1.05   1.27    1.01
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     29      3     16   0.77   0.97   1.18    0.98
 1025..4096                      17      4      5   0.73   1.09   1.80    1.13
 > 65536 points                  15      1      4   0.81   1.06   1.23    1.03
 257..1024                        5      2      3   0.66   0.84   1.22    0.91
 <= 256 points                    1      1      1   0.40   0.40   0.40    0.40
 ALL                             67     11     29   0.76   1.05   1.27    1.01
```


worst 10: 16x16 (chain 0.40), 32x32 (chain 0.66), 4096x16 (chain 0.69), 128x16 (chain 0.73), 256x16 (chain 0.73), 128x32 (chain 0.74), 100x64 (chain 0.76), 1024x16 (chain 0.77), 64x30 (chain 0.77), 128x1024 (chain 0.77)
best 5: 47x64 (chain 1.97), 23x64 (chain 1.80), 64x62 (chain 1.70), 29x64 (chain 1.58), 16x256 (chain 1.37)
