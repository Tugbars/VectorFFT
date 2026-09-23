# gauntlet report (2D)

run: `gauntlet_2d-pow2grid2`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.137..1.292


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain+rb raced             13         13    0.94      3.0  1.4e-16
         2x4  2            chain+rb raced             15        157   10.26      7.8  1.7e-16
         2x8  2            chain+rb raced             19        167    8.71     16.7  1.9e-16
        2x16  2            chain+rb2 raced             31        205    6.67     26.1  2.0e-16
        2x32  2            chain+rb2 raced             45        305    6.74     42.5  2.0e-16
        2x64  2            chain+rb2 raced             84        466    5.52     53.0  2.2e-16
       2x128  2            chain  raced            164        831    5.05     62.3  3.7e-16
       2x256  2            chain  raced            325       1536    4.73     71.0  3.9e-16
       2x512  2            chain  raced            965       2969    2.92     53.1  4.5e-16
      2x1024  2            chain  raced           2309       6035    2.61     48.8  4.0e-16
      2x2048  2            chain  raced           5192      12303    2.36     47.3  2.7e-16
      2x4096  2            chain  raced          10786      25135    2.33     49.4  3.2e-16
      2x8192  2            chain  raced          22543      51535    2.19     50.9  3.4e-16
         4x2  2^2          chain+rb raced             15        155   10.17      7.9  8.9e-17
         4x4  2^2          chain+rb raced             18         18    1.01     17.7  2.0e-16
         4x8  2^2          chain+rb raced             25        176    6.66     31.8  1.7e-16
        4x16  2^2          chain+rb2 raced             42        212    5.00     45.2  2.8e-16
        4x32  2^2          chain+rb2 raced             75        319    4.22     59.4  1.7e-16
        4x64  2^2          chain+rb2 raced            155        507    3.18     65.9  2.9e-16
       4x128  2^2          chain  raced            330        953    2.89     69.9  3.7e-16
       4x256  2^2          chain  raced            697       1803    2.52     73.5  3.7e-16
       4x512  2^2          chain  raced           2077       3625    1.72     54.2  4.6e-16
      4x1024  2^2          chain  raced           4374       7501    1.69     56.2  3.7e-16
      4x2048  2^2          chain  raced           9919      15436    1.54     53.7  3.6e-16
      4x4096  2^2          chain  raced          21058      32054    1.52     54.5  3.1e-16
      4x8192  2^2          chain  raced          44315      67045    1.50     55.5  3.6e-16
         8x2  2^3          chain+rb raced             19        166    8.62     16.6  1.2e-16
         8x4  2^3          chain+rb raced             25        179    6.91     32.2  1.6e-16
         8x8  2^3          chain+rb raced             38         37    0.95     50.0  1.5e-16
        8x16  2^3          chain+rb2 raced             75        246    3.27     59.6  2.8e-16
        8x32  2^3          chain+rb2 raced            147        407    2.76     69.5  2.6e-16
        8x64  2^3          chain+rb2 raced            322        691    2.14     71.6  2.9e-16
       8x128  2^3          chain  raced            712       1410    1.97     71.9  3.3e-16
       8x256  2^3          chain  raced           1896       2848    1.27     59.4  3.5e-16
       8x512  2^3          chain  raced           4886       5783    1.18     50.3  4.6e-16
      8x1024  2^3          chain  raced          10288      11882    1.13     51.8  3.8e-16
      8x2048  2^3          chain  raced          23037      24956    1.08     49.8  3.8e-16
      8x4096  2^3          chain  raced          48811      52957    1.08     50.3  5.4e-16
      8x8192  2^3          chain  raced          95640     116812    1.21     54.8  5.2e-16
        16x2  2^4          chain+rb raced             26        195    7.38     30.2  2.2e-16
        16x4  2^4          chain+rb raced             40        214    5.17     47.6  2.5e-16
        16x8  2^4          chain+rb raced             70        246    3.50     63.9  2.0e-16
       16x16  2^4          chain+rb raced            163        146    0.90     63.0  2.5e-16
       16x32  2^4          chain+rb2 raced            320        353    1.10     72.0  3.0e-16
       16x64  2^4          chain+rb2 raced            735        719    0.96     69.6  3.9e-16
      16x128  2^4          chain  raced           1962       1612    0.82     57.4  4.6e-16
      16x256  2^4          chain  raced           5963       6986    1.06     41.2  3.8e-16
      16x512  2^4          chain  raced          11155      14450    1.28     47.7  3.8e-16
     16x1024  2^4          chain  raced          26578      31948    1.20     43.2  3.7e-16
     16x2048  2^4          chain  raced          59657      67792    1.12     41.2  3.7e-16
     16x4096  2^4          chain  raced         132770     133316    1.00     39.5  3.1e-16
     16x8192  2^4          chain  raced         282320     317693    1.11     39.5  4.2e-16
        32x2  2^5          chain+rb raced             49        273    5.46     39.0  2.6e-16
        32x4  2^5          chain+rb raced             79        320    4.04     57.0  2.0e-16
        32x8  2^5          chain+rb raced            150        400    2.61     68.3  2.4e-16
       32x16  2^5          chain+rb2 raced            348        340    0.98     66.2  2.3e-16
       32x32  2^5          chain+rb2 raced            715        720    0.97     71.6  3.9e-16
       32x64  2^5          chain+rb2 raced           2035       1919    0.81     55.4  3.5e-16
      32x128  2^5          chain  raced           5633       6902    1.18     43.6  3.5e-16
      32x256  2^5          chain  raced          12971      15172    1.15     41.1  4.0e-16
      32x512  2^5          chain  raced          26869      31799    1.15     42.7  4.6e-16
     32x1024  2^5          chain  raced          56743      69445    1.21     43.3  3.5e-16
     32x2048  2^5          chain  raced         120653     152392    1.25     43.5  3.6e-16
     32x4096  2^5          chain  raced         268520     322776    1.18     41.5  3.8e-16
     32x8192  2^5          chain  raced         590150     860556    1.42     40.0  5.2e-16
        64x2  2^6          chain+rb raced             98        414    4.16     45.5  3.4e-16
        64x4  2^6          chain+rb raced            174        508    2.88     58.9  3.4e-16
        64x8  2^6          chain+rb raced            333        685    2.03     69.1  3.2e-16
       64x16  2^6          chain+rb raced            755        827    0.89     67.8  3.2e-16
       64x32  2^6          chain+rb2 raced           1891       2014    0.79     59.6  3.0e-16
       64x64  2^6          chain+rb2 raced           5621       7130    1.18     43.7  3.8e-16
      64x128  2^6          chain  raced          12737      14051    1.06     41.8  3.9e-16
      64x256  2^6          chain  raced          29625      32653    1.10     38.7  4.2e-16
      64x512  2^6          chain  raced          60533      66961    1.09     40.6  5.0e-16
     64x1024  2^6          chain  raced         129487     143010    1.09     40.5  3.6e-16
     64x2048  2^6          chain  raced         304473     331153    1.08     36.6  3.2e-16
     64x4096  2^6          chain  raced         577700     777168    1.27     40.8  5.6e-16
     64x8192  2^6          chain  raced        1347975    1544206    1.10     37.0  9.7e-16
       128x2  2^7          chain+rb raced            431        726    1.64     23.7  4.3e-16
       128x4  2^7          chain+rb raced            638        967    1.49     36.1  3.4e-16
       128x8  2^7          chain+rb raced           1223       1509    1.08     41.9  3.5e-16
      128x16  2^7          chain+rb2 raced           2941       1993    0.66     38.3  3.0e-16
      128x32  2^7          chain+rb2 raced           6932       7233    0.90     35.5  4.1e-16
      128x64  2^7          chain  raced          19814      15274    0.76     26.9  4.0e-16
     128x128  2^7          chain  raced          31434      30340    0.78     36.5  4.8e-16
     128x256  2^7          chain  raced          64220      62005    0.91     38.3  5.3e-16
     128x512  2^7          chain  raced         118230     131306    1.10     44.3  4.8e-16
    128x1024  2^7          chain  raced         280333     311076    1.10     39.7  3.7e-16
    128x2048  2^7          chain  raced         625812     721606    1.10     37.7  5.8e-16
    128x4096  2^7          chain  raced        1381050    1484263    1.04     36.1  1.3e-15
    128x8192  2^7          chain  raced        3088525    3479188    1.06     34.0  1.4e-15
       256x2  2^8          chain+rb raced            667       1352    1.99     34.5  3.3e-16
       256x4  2^8          chain+rb raced           1582       1977    1.03     32.4  3.5e-16
       256x8  2^8          chain+rb raced           2820       3477    1.07     39.9  4.4e-16
      256x16  2^8          chain+rb2 raced           9031       7492    0.81     27.2  4.1e-16
      256x32  2^8          chain+rb2 raced          16277      14864    0.88     32.7  3.7e-16
      256x64  2^8          chain+rb2 raced          32348      31024    0.81     35.5  5.3e-16
     256x128  2^8          chain  raced          67451      61520    0.78     36.4  5.1e-16
     256x256  2^8          chain  raced         117237     133855    1.03     44.7  5.8e-16
     256x512  2^8          chain  raced         286293     326376    1.10     38.9  6.6e-16
    256x1024  2^8          chain  raced         602825     782162    1.25     39.1  5.5e-16
    256x2048  2^8          chain  raced        1392875    1517156    1.04     35.8  1.3e-15
    256x4096  2^8          chain  raced        2938425    3558600    1.19     35.7  1.1e-15
    256x8192  2^8          chain  raced        7032700    8036419    1.13     31.3  1.5e-15
       512x2  2^9          chain+rb raced           2179       2914    1.11     23.5  3.2e-16
       512x4  2^9          chain+rb raced           3707       4173    1.11     30.4  5.7e-16
       512x8  2^9          chain+rb raced           7571       7475    0.83     32.5  4.6e-16
      512x16  2^9          chain+rb2 raced          16052      17800    1.01     33.2  4.0e-16
      512x32  2^9          chain+rb2 raced          31291      34861    1.06     36.7  4.3e-16
      512x64  2^9          chain+rb2 raced          66084      72172    1.08     37.2  4.6e-16
     512x128  2^9          chain  raced         120853     146976    1.21     43.4  5.3e-16
     512x256  2^9          chain  raced         248807     356896    1.43     44.8  5.1e-16
     512x512  2^9          chain  raced         582800     884943    1.51     40.5  5.4e-16
    512x1024  2^9          chain  raced        1331838    1805868    1.15     37.4  1.1e-15
    512x2048  2^9          chain  raced        3053250    3805725    1.18     34.3  1.2e-15
    512x4096  2^9          chain  raced        6916300    8914531    1.24     31.8  1.5e-15
    512x8192  2^9          chain  raced       16082250   19298131    1.20     28.7  2.8e-15
      1024x2  2^10         chain+rb raced           4549       6259    1.36     24.8  4.7e-16
      1024x4  2^10         chain+rb raced           8739       8770    0.92     28.1  4.2e-16
      1024x8  2^10         chain+rb raced          17819      15428    0.79     29.9  4.9e-16
     1024x16  2^10         chain+rb2 raced          26892      37937    1.27     42.6  4.9e-16
     1024x32  2^10         chain+rb2 raced          53410      76344    1.29     46.0  5.1e-16
     1024x64  2^10         chain+rb2 raced         122647     162968    1.32     42.7  5.3e-16
    1024x128  2^10         chain  raced         261667     390537    1.41     42.6  5.9e-16
    1024x256  2^10         chain  raced         536300    1021787    1.90     44.0  5.7e-16
    1024x512  2^10         chain  raced        1208813    2213781    1.78     41.2  1.2e-15
   1024x1024  2^10         chain  raced        2985213    4570281    1.50     35.1  1.3e-15
   1024x2048  2^10         chain  raced        7861525   10238456    1.26     28.0  1.7e-15
   1024x4096  2^10         chain  raced       17464525   33346469    1.44     26.4  2.6e-15
      2048x2  2^11         chain+rb raced          24593      12801    0.49     10.0  4.3e-16
      2048x4  2^11         chain+rb raced          21355      18444    0.86     24.9  5.1e-16
      2048x8  2^11         chain+rb raced          37359      33023    0.75     30.7  4.6e-16
     2048x16  2^11         chain+rb2 raced          71466      79104    0.94     34.4  4.5e-16
     2048x32  2^11         chain+rb2 raced         155550     161203    1.03     33.7  4.7e-16
     2048x64  2^11         chain+rb2 raced         362553     415380    1.14     30.7  5.5e-16
    2048x128  2^11         chain  raced         734200    1035150    1.36     32.1  5.8e-16
    2048x256  2^11         chain  raced        1508775    2285806    1.49     33.0  1.2e-15
    2048x512  2^11         chain  raced        3589413    5297631    1.45     29.2  1.3e-15
   2048x1024  2^11         chain  raced        7359125   11092894    1.48     29.9  1.5e-15
   2048x2048  2^11         chain  raced       18188750   26696549    1.38     25.4  3.1e-15
      4096x2  2^12         chain+rb raced          51948      25989    0.49     10.2  3.7e-16
      4096x4  2^12         chain+rb raced          42184      37938    0.90     27.2  5.6e-16
      4096x8  2^12         chain+rb raced          80931      67779    0.83     30.4  5.8e-16
     4096x16  2^12         chain+rb2 raced         176477     146007    0.81     29.7  4.4e-16
     4096x32  2^12         chain+rb2 raced         354280     352083    0.98     31.4  5.7e-16
     4096x64  2^12         chain+rb raced         801413     823619    1.03     29.4  7.1e-16
    4096x128  2^12         chain  raced        1675762    1781744    1.01     29.7  1.2e-15
    4096x256  2^12         chain+rb2 raced        3738000    4412631    1.16     28.1  1.3e-15
    4096x512  2^12         chain  raced        9114563   13039062    1.42     24.2  1.8e-15
   4096x1024  2^12         chain  raced       18629225   29533000    1.16     24.8  3.1e-15
      8192x2  2^13         chain+rb raced          69348      53276    0.70     16.5  5.4e-16
      8192x4  2^13         chain+rb raced          97393      78845    0.72     25.2  5.5e-16
      8192x8  2^13         chain+rb raced         202817     163247    0.78     25.9  5.4e-16
     8192x16  2^13         chain+rb2 raced         371813     412280    1.10     30.0  4.6e-16
     8192x32  2^13         chain+rb2 raced         774725     874937    1.08     30.5  4.9e-16
     8192x64  2^13         chain+rb2 raced        1609687    1908419    1.11     30.9  1.3e-15
    8192x128  2^13         chain  raced        3708450    4533299    1.20     28.3  1.3e-15
    8192x256  2^13         chain+rb2 raced        8317037   14408006    1.68     26.5  1.5e-15
    8192x512  2^13         chain  raced       17871263   37869987    2.09     25.8  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           80      3      5   1.04   1.21   2.33    1.35
 chain+rb                        42      7     16   0.75   1.09   7.38    1.74
 chain+rb2                       37      2     13   0.81   1.10   5.00    1.42
 ALL                            159     12     34   0.81   1.18   4.22    1.46
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159     12     34   0.81   1.18   4.22    1.46
 ALL                            159     12     34   0.81   1.18   4.22    1.46
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      9     17   0.76   1.08   1.50    1.04
 > 65536 points                  45      0      1   1.04   1.19   1.51    1.26
 <= 256 points                   28      0      3   0.95   4.61   8.71    3.89
 1025..4096                      21      3      9   0.79   1.07   1.72    1.09
 257..1024                       17      0      4   0.96   1.49   2.92    1.59
 ALL                            159     12     34   0.81   1.18   4.22    1.46
```


worst 10: 2048x2 (chain+rb 0.49), 4096x2 (chain+rb 0.49), 128x16 (chain+rb2 0.66), 8192x2 (chain+rb 0.70), 8192x4 (chain+rb 0.72), 2048x8 (chain+rb 0.75), 128x64 (chain 0.76), 256x128 (chain 0.78), 128x128 (chain 0.78), 8192x8 (chain+rb 0.78)
best 5: 2x4 (chain+rb 10.26), 4x2 (chain+rb 10.17), 2x8 (chain+rb 8.71), 8x2 (chain+rb 8.62), 16x2 (chain+rb 7.38)
