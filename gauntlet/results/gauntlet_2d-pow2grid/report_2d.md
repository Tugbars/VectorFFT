# gauntlet report (2D)

run: `gauntlet_2d-pow2grid`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.052..1.222


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain+rb raced             13         13    0.95      3.0  1.4e-16
         2x4  2            chain+rb raced             15        157   10.32      7.9  1.7e-16
         2x8  2            chain+rb raced             19        168    8.72     16.7  1.9e-16
        2x16  2            chain+rb raced             35        205    5.82     22.9  2.0e-16
        2x32  2            chain  raced             53        305    5.79     36.6  2.0e-16
        2x64  2            chain  raced             86        468    5.41     51.9  2.2e-16
       2x128  2            chain  raced            164        830    5.04     62.3  3.7e-16
       2x256  2            chain  raced            326       1535    4.71     70.7  3.9e-16
       2x512  2            chain  raced            999       2966    2.77     51.3  4.5e-16
      2x1024  2            chain  raced           2623       6071    2.12     42.9  4.0e-16
      2x2048  2            chain  raced           5186      12286    2.37     47.4  2.7e-16
      2x4096  2            chain  raced          10843      25269    2.32     49.1  3.2e-16
      2x8192  2            chain  raced          22857      51568    2.24     50.2  3.4e-16
         4x2  2^2          chain+rb raced             15        155   10.35      8.0  8.9e-17
         4x4  2^2          chain+rb raced             18         19    1.02     17.7  2.0e-16
         4x8  2^2          chain+rb raced             25        175    7.04     32.4  1.7e-16
        4x16  2^2          chain+rb raced             48        212    4.44     40.2  2.8e-16
        4x32  2^2          chain  raced             95        317    3.32     47.2  1.7e-16
        4x64  2^2          chain  raced            167        504    3.02     61.4  2.9e-16
       4x128  2^2          chain  raced            330        949    2.81     69.9  3.7e-16
       4x256  2^2          chain  raced            694       1799    2.55     73.7  3.7e-16
       4x512  2^2          chain  raced           1974       3615    1.82     57.1  4.6e-16
      4x1024  2^2          chain  raced           4438       7493    1.66     55.4  3.7e-16
      4x2048  2^2          chain  raced           9888      15437    1.53     53.9  3.6e-16
      4x4096  2^2          chain  raced          21171      32132    1.50     54.2  3.1e-16
      4x8192  2^2          chain  raced          44730      66994    1.50     54.9  3.6e-16
         8x2  2^3          chain+rb raced             19        166    8.76     16.8  1.2e-16
         8x4  2^3          chain+rb raced             25        179    6.99     31.5  1.6e-16
         8x8  2^3          chain+rb raced             39         37    0.95     49.8  1.5e-16
        8x16  2^3          chain+rb raced             82        246    2.97     54.5  2.8e-16
        8x32  2^3          chain  raced            193        404    2.09     52.9  2.6e-16
        8x64  2^3          chain  raced            353        688    1.94     65.3  3.0e-16
       8x128  2^3          chain  raced            718       1402    1.91     71.4  3.3e-16
       8x256  2^3          chain  raced           1923       2856    1.28     58.6  3.5e-16
       8x512  2^3          chain  raced           4928       5793    1.17     49.9  4.6e-16
      8x1024  2^3          chain  raced          10358      11881    1.14     51.4  3.8e-16
      8x2048  2^3          chain  raced          23128      24946    1.08     49.6  3.8e-16
      8x4096  2^3          chain  raced          48789      52915    1.08     50.4  5.4e-16
      8x8192  2^3          chain  raced          96357     117100    1.19     54.4  5.2e-16
        16x2  2^4          chain+rb raced             27        195    7.35     30.1  2.2e-16
        16x4  2^4          chain+rb raced             41        214    5.25     47.2  2.5e-16
        16x8  2^4          chain+rb raced             70        245    3.48     63.8  2.0e-16
       16x16  2^4          chain+rb raced            163        148    0.90     62.9  2.5e-16
       16x32  2^4          chain+rb raced            428        362    0.76     53.9  3.0e-16
       16x64  2^4          chain  raced            775        716    0.91     66.0  3.9e-16
      16x128  2^4          chain  raced           1928       1625    0.84     58.4  4.6e-16
      16x256  2^4          chain  raced           6537       7010    1.06     37.6  3.8e-16
      16x512  2^4          chain  raced          11201      14473    1.28     47.5  3.8e-16
     16x1024  2^4          chain  raced          26897      31422    1.16     42.6  3.7e-16
     16x2048  2^4          chain  raced          57877      67702    1.15     42.5  3.7e-16
     16x4096  2^4          chain  raced         130147     133290    1.01     40.3  3.1e-16
     16x8192  2^4          chain  raced         277693     317867    1.07     40.1  4.2e-16
        32x2  2^5          chain+rb raced             49        272    5.52     38.9  2.6e-16
        32x4  2^5          chain+rb raced             79        319    3.95     56.6  2.0e-16
        32x8  2^5          chain+rb raced            152        399    2.62     67.3  2.4e-16
       32x16  2^5          chain+rb raced            347        340    0.98     66.4  2.3e-16
       32x32  2^5          chain+rb raced            862        780    0.79     59.4  2.9e-16
       32x64  2^5          chain  raced           2722       1919    0.64     41.4  3.0e-16
      32x128  2^5          chain  raced           5603       6900    1.23     43.9  3.5e-16
      32x256  2^5          chain  raced          12849      15196    1.15     41.4  4.0e-16
      32x512  2^5          chain  raced          27300      31971    1.17     42.0  4.2e-16
     32x1024  2^5          chain  raced          54849      69976    1.20     44.8  3.5e-16
     32x2048  2^5          chain  raced         119427     150112    1.21     43.9  3.6e-16
     32x4096  2^5          chain  raced         265540     324963    1.22     42.0  3.8e-16
     32x8192  2^5          chain  raced         583425     939750    1.53     40.4  5.2e-16
        64x2  2^6          chain+rb raced             98        414    4.18     45.7  3.4e-16
        64x4  2^6          chain+rb raced            174        510    2.92     58.7  3.4e-16
        64x8  2^6          chain+rb raced            342        683    1.96     67.3  3.2e-16
       64x16  2^6          chain+rb raced            757        861    1.09     67.6  3.2e-16
       64x32  2^6          chain+rb raced           2389       2010    0.81     47.2  4.2e-16
       64x64  2^6          chain  raced           5849       7138    1.13     42.0  3.8e-16
      64x128  2^6          chain  raced          13454      14004    1.04     39.6  4.2e-16
      64x256  2^6          chain  raced          25672      32553    1.25     44.7  3.8e-16
      64x512  2^6          chain  raced          59841      67156    1.09     41.1  5.0e-16
     64x1024  2^6          chain  raced         128483     143021    1.09     40.8  3.6e-16
     64x2048  2^6          chain  raced         303993     331833    1.07     36.6  3.2e-16
     64x4096  2^6          chain  raced         604262     828393    1.32     39.0  5.6e-16
     64x8192  2^6          chain  raced        1390263    1573900    1.12     35.8  9.7e-16
       128x2  2^7          chain+rb raced            336        726    2.14     30.5  3.9e-16
       128x4  2^7          chain+rb raced            665        967    1.45     34.7  3.4e-16
       128x8  2^7          chain+rb raced           1196       1507    1.16     42.8  3.5e-16
      128x16  2^7          chain  raced           3658       1988    0.54     30.8  3.3e-16
      128x32  2^7          chain  raced           7517       7220    0.84     32.7  4.1e-16
      128x64  2^7          chain  raced          16107      15178    0.90     33.1  4.0e-16
     128x128  2^7          chain  raced          28744      30034    1.03     39.9  4.3e-16
     128x256  2^7          chain  raced          81152      61150    0.75     30.3  5.9e-16
     128x512  2^7          chain  raced         123747     131325    1.02     42.4  4.8e-16
    128x1024  2^7          chain  raced         284027     313856    1.09     39.2  3.7e-16
    128x2048  2^7          chain  raced         664325     754950    1.13     35.5  5.8e-16
    128x4096  2^7          chain  raced        1367750    1520500    0.95     36.4  1.1e-15
    128x8192  2^7          chain  raced        3226162    3332956    1.03     32.5  1.1e-15
       256x2  2^8          chain+rb raced            716       1356    1.81     32.2  4.1e-16
       256x4  2^8          chain+rb raced           1802       1963    1.06     28.4  3.5e-16
       256x8  2^8          chain+rb raced           2888       3487    1.20     39.0  4.4e-16
      256x16  2^8          chain+rb raced           7211       7415    0.89     34.1  3.6e-16
      256x32  2^8          chain  raced          18284      14802    0.71     29.1  3.7e-16
      256x64  2^8          chain  raced          28721      31500    1.09     39.9  4.8e-16
     256x128  2^8          chain  raced          76895      61908    0.75     32.0  5.1e-16
     256x256  2^8          chain  raced         121437     128041    1.01     43.2  5.8e-16
     256x512  2^8          chain  raced         286147     332963    1.11     38.9  6.6e-16
    256x1024  2^8          chain  raced         589725     743825    1.16     40.0  5.1e-16
    256x2048  2^8          chain  raced        1360588    1589344    1.07     36.6  1.3e-15
    256x4096  2^8          chain  raced        2838512    3594424    1.20     36.9  1.1e-15
    256x8192  2^8          chain  raced        7328163    8027263    1.09     30.0  1.5e-15
       512x2  2^9          chain+rb raced           2111       2934    1.14     24.3  3.2e-16
       512x4  2^9          chain+rb raced           3606       4160    1.13     31.2  5.7e-16
       512x8  2^9          chain+rb raced           6742       7513    1.11     36.5  5.1e-16
      512x16  2^9          chain+rb raced          12839      17850    1.35     41.5  4.5e-16
      512x32  2^9          chain+rb raced          27602      34797    1.14     41.5  5.3e-16
      512x64  2^9          chain  raced          59884      71852    1.19     41.0  5.5e-16
     512x128  2^9          chain  raced         121600     145425    1.15     43.1  5.3e-16
     512x256  2^9          chain  raced         275820     347463    1.26     40.4  5.1e-16
     512x512  2^9          chain  raced         615138     876162    1.37     38.4  5.4e-16
    512x1024  2^9          chain  raced        1305650    1721537    1.17     38.1  1.1e-15
    512x2048  2^9          chain  raced        2979838    3630418    1.20     35.2  1.2e-15
    512x4096  2^9          chain  raced        6976725    9216537    1.29     31.6  1.5e-15
    512x8192  2^9          chain  raced       16467638   20794418    1.23     28.0  2.8e-15
      1024x2  2^10         chain+rb raced           4502       6253    1.37     25.0  5.5e-16
      1024x4  2^10         chain+rb raced           8263       8782    0.95     29.7  5.3e-16
      1024x8  2^10         chain+rb raced          13595      15406    1.10     39.2  4.3e-16
     1024x16  2^10         chain+rb raced          34691      37978    1.01     33.1  4.9e-16
     1024x32  2^10         chain  raced          56557      76263    1.21     43.5  5.4e-16
     1024x64  2^10         chain  raced         132343     162897    1.22     39.6  4.8e-16
    1024x128  2^10         chain  raced         264033     379673    1.43     42.2  5.9e-16
    1024x256  2^10         chain  raced         541075    1084781    1.98     43.6  5.7e-16
    1024x512  2^10         chain  raced        1256013    2359294    1.86     39.7  1.2e-15
   1024x1024  2^10         chain  raced        3101713    4742343    1.51     33.8  1.3e-15
   1024x2048  2^10         chain  raced        7668888   13649256    1.40     28.7  1.7e-15
   1024x4096  2^10         chain  raced       17538163   27081975    1.50     26.3  2.6e-15
      2048x2  2^11         chain+rb raced          25614      12839    0.50      9.6  3.4e-16
      2048x4  2^11         chain+rb raced          21095      18317    0.75     25.2  5.5e-16
      2048x8  2^11         chain+rb raced          36562      32753    0.77     31.4  4.6e-16
     2048x16  2^11         chain+rb raced          74931      78607    1.04     32.8  4.8e-16
     2048x32  2^11         chain  raced         164423     159782    0.93     31.9  4.3e-16
     2048x64  2^11         chain  raced         337193     392013    1.13     33.0  5.8e-16
    2048x128  2^11         chain  raced         775625    1062281    1.36     30.4  5.8e-16
    2048x256  2^11         chain  raced        1542762    2402631    1.54     32.3  1.2e-15
    2048x512  2^11         chain  raced        3472175    5891394    1.65     30.2  1.3e-15
   2048x1024  2^11         chain  raced        8729225   13626287    1.52     25.2  1.5e-15
   2048x2048  2^11         chain  raced       18408487   25927231    1.39     25.1  3.2e-15
      4096x2  2^12         chain+rb raced          54417      26077    0.46      9.8  4.4e-16
      4096x4  2^12         chain+rb raced          51086      38047    0.73     22.4  5.1e-16
      4096x8  2^12         chain+rb raced          80256      68398    0.84     30.6  5.2e-16
     4096x16  2^12         chain+rb raced         156077     144385    0.89     33.6  4.8e-16
     4096x32  2^12         chain+rb raced         371673     396583    1.04     30.0  5.7e-16
     4096x64  2^12         chain  raced         755313     884112    1.15     31.2  7.0e-16
    4096x128  2^12         chain  raced        1921175    1969137    0.93     25.9  1.2e-15
    4096x256  2^12         chain  raced        3870288    4446775    1.14     27.1  1.3e-15
    4096x512  2^12         chain  raced        8366425   13423443    1.59     26.3  1.7e-15
   4096x1024  2^12         chain  raced       19548287   29843587    1.16     23.6  3.0e-15
      8192x2  2^13         chain+rb raced          64255      52992    0.82     17.8  5.4e-16
      8192x4  2^13         chain+rb raced          93697      79243    0.81     26.2  6.1e-16
      8192x8  2^13         chain+rb raced         188567     158326    0.83     27.8  5.8e-16
     8192x16  2^13         chain+rb raced         377793     414293    1.06     29.5  6.0e-16
     8192x32  2^13         chain  raced         851200     886106    1.04     27.7  5.3e-16
     8192x64  2^13         chain  raced        1707488    1942356    1.10     29.2  1.3e-15
    8192x128  2^13         chain  raced        3703863    4731137    1.26     28.3  1.3e-15
    8192x256  2^13         chain  raced        8273338   13831006    1.59     26.6  1.5e-15
    8192x512  2^13         chain  raced       18964588   39532287    2.07     24.3  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                          103      5     12   0.93   1.20   2.32    1.35
 chain+rb                        56      7     19   0.77   1.12   7.04    1.67
 ALL                            159     12     31   0.83   1.17   4.18    1.46
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159     12     31   0.83   1.17   4.18    1.46
 ALL                            159     12     31   0.83   1.17   4.18    1.46
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      7     14   0.75   1.09   1.50    1.06
 > 65536 points                  45      0      2   1.04   1.20   1.59    1.27
 <= 256 points                   28      0      3   0.95   4.31   8.76    3.80
 1025..4096                      21      3      8   0.64   1.13   1.82    1.09
 257..1024                       17      2      4   0.79   1.45   2.81    1.53
 ALL                            159     12     31   0.83   1.17   4.18    1.46
```


worst 10: 4096x2 (chain+rb 0.46), 2048x2 (chain+rb 0.50), 128x16 (chain 0.54), 32x64 (chain 0.64), 256x32 (chain 0.71), 4096x4 (chain+rb 0.73), 2048x4 (chain+rb 0.75), 128x256 (chain 0.75), 256x128 (chain 0.75), 16x32 (chain+rb 0.76)
best 5: 4x2 (chain+rb 10.35), 2x4 (chain+rb 10.32), 8x2 (chain+rb 8.76), 2x8 (chain+rb 8.72), 16x2 (chain+rb 7.35)
