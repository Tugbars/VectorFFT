# gauntlet report (2D)

run: `gauntlet_2d-pow2grid5`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.233..1.409


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain+rb raced             14         13    0.92      2.9  1.4e-16
         2x4  2            chain+rb raced             15        157   10.27      7.9  1.7e-16
         2x8  2            chain+rb raced             19        167    8.66     16.6  1.9e-16
        2x16  2            chain+rb2 raced             30        205    6.76     26.4  2.0e-16
        2x32  2            chain+rb2 raced             45        305    6.78     42.7  2.0e-16
        2x64  2            chain+rb2 raced             84        467    5.55     53.3  2.2e-16
       2x128  2            chain  raced            164        830    4.96     62.3  3.7e-16
       2x256  2            chain  raced            325       1534    4.72     70.9  3.9e-16
       2x512  2            chain  raced            987       2965    2.69     51.9  4.5e-16
      2x1024  2            chain  raced           2286       6042    2.63     49.3  4.0e-16
      2x2048  2            chain  raced           5193      12291    2.36     47.3  2.7e-16
      2x4096  2            chain  raced          10784      25144    2.33     49.4  3.2e-16
      2x8192  2            chain  raced          22522      51527    2.27     50.9  3.4e-16
         4x2  2^2          chain+rb raced             15        155   10.28      8.0  8.9e-17
         4x4  2^2          chain+rb raced             18         19    1.02     17.6  2.0e-16
         4x8  2^2          chain+rb raced             25        175    6.92     31.9  1.7e-16
        4x16  2^2          chain+rb2 raced             43        212    4.93     45.0  2.8e-16
        4x32  2^2          chain+rb2 raced             75        317    4.22     59.6  1.7e-16
        4x64  2^2          chain+rb2 raced            155        505    3.25     66.1  2.9e-16
       4x128  2^2          chain  raced            331        949    2.84     69.7  3.7e-16
       4x256  2^2          chain  raced            694       1799    2.56     73.8  3.7e-16
       4x512  2^2          chain  raced           2074       3625    1.74     54.3  4.6e-16
      4x1024  2^2          chain  raced           4441       7491    1.69     55.3  3.7e-16
      4x2048  2^2          chain  raced           9817      15450    1.54     54.2  3.6e-16
      4x4096  2^2          chain  raced          21353      32050    1.50     53.7  3.1e-16
      4x8192  2^2          chain  raced          44252      67030    1.50     55.5  3.6e-16
         8x2  2^3          chain+rb raced             19        166    8.63     16.8  1.2e-16
         8x4  2^3          chain+rb raced             24        179    6.46     32.7  1.6e-16
         8x8  2^3          chain+rb raced             39         36    0.94     49.7  1.5e-16
        8x16  2^3          chain+rb2 raced             75        246    3.26     59.7  2.8e-16
        8x32  2^3          chain+rb2 raced            147        406    2.75     69.6  2.6e-16
        8x64  2^3          chain+rb2 raced            321        689    2.14     71.7  2.9e-16
       8x128  2^3          chain  raced            713       1403    1.94     71.8  3.3e-16
       8x256  2^3          chain  raced           1948       2861    1.46     57.8  3.5e-16
       8x512  2^3          chain  raced           4765       5795    1.18     51.6  4.6e-16
      8x1024  2^3          chain  raced          10224      11894    1.15     52.1  3.8e-16
      8x2048  2^3          chain  raced          22972      24950    1.07     49.9  3.8e-16
      8x4096  2^3          chain  raced          48413      52860    1.09     50.8  5.4e-16
      8x8192  2^3          chain  raced          95363     118160    1.22     55.0  5.2e-16
        16x2  2^4          chain+rb raced             26        195    7.31     30.4  2.2e-16
        16x4  2^4          chain+rb raced             40        214    5.20     47.8  2.5e-16
        16x8  2^4          chain+rb raced             70        245    3.49     63.8  2.0e-16
       16x16  2^4          chain+rb2 raced            154        146    0.94     66.4  2.5e-16
       16x32  2^4          chain+rb2 raced            320        353    1.10     72.1  3.0e-16
       16x64  2^4          chain+rb2 raced            717        742    1.00     71.4  3.9e-16
      16x128  2^4          chain  raced           1889       1634    0.84     59.6  4.6e-16
      16x256  2^4          chain  raced           5222       7000    1.17     47.1  3.8e-16
      16x512  2^4          chain  raced          13040      14674    1.05     40.8  3.8e-16
     16x1024  2^4          chain  raced          25968      31975    1.23     44.2  3.7e-16
     16x2048  2^4          chain  raced          57721      68028    1.16     42.6  3.7e-16
     16x4096  2^4          chain  raced         130653     135201    1.03     40.1  3.1e-16
     16x8192  2^4          chain  raced         287947     319576    1.10     38.7  4.2e-16
        32x2  2^5          chain+rb raced             49        273    5.47     38.9  2.6e-16
        32x4  2^5          chain+rb raced             78        320    4.07     57.3  2.0e-16
        32x8  2^5          chain+rb raced            148        400    2.60     69.0  2.4e-16
       32x16  2^5          chain+rb raced            359        340    0.79     64.2  2.5e-16
       32x32  2^5          chain+rb2 raced            714        720    1.01     71.7  2.4e-16
       32x64  2^5          chain+rb2 raced           1796       1879    0.72     62.7  3.5e-16
      32x128  2^5          chain  raced           5640       6918    1.23     43.6  3.5e-16
      32x256  2^5          chain  raced          11698      15109    1.14     45.5  4.0e-16
      32x512  2^5          chain  raced          27116      32054    1.18     42.3  4.2e-16
     32x1024  2^5          chain  raced          56995      68648    1.19     43.1  3.5e-16
     32x2048  2^5          chain  raced         121793     151313    1.23     43.0  3.6e-16
     32x4096  2^5          chain  raced         265407     327573    1.20     42.0  3.8e-16
     32x8192  2^5          chain  raced         610487     906338    1.47     38.6  5.2e-16
        64x2  2^6          chain+rb raced             99        414    4.17     45.3  3.4e-16
        64x4  2^6          chain+rb raced            177        509    2.83     57.7  3.4e-16
        64x8  2^6          chain+rb raced            336        690    2.04     68.6  3.2e-16
       64x16  2^6          chain+rb2 raced            836        827    0.89     61.3  3.4e-16
       64x32  2^6          chain+rb2 raced           2088       2015    0.85     53.9  3.0e-16
       64x64  2^6          chain+rb2 raced           5254       7063    1.23     46.8  3.8e-16
      64x128  2^6          chain  raced          12325      13990    1.13     43.2  4.2e-16
      64x256  2^6          chain  raced          29016      32369    1.11     39.5  3.8e-16
      64x512  2^6          chain  raced          47805      67033    1.23     51.4  4.4e-16
     64x1024  2^6          chain  raced         122253     143653    1.17     42.9  3.2e-16
     64x2048  2^6          chain  raced         294393     333353    1.12     37.8  4.2e-16
     64x4096  2^6          chain  raced         609063     812850    1.28     38.7  5.6e-16
     64x8192  2^6          chain  raced        1434663    1649062    1.14     34.7  1.1e-15
       128x2  2^7          turn   raced            206        726    3.48     49.7  3.5e-16
       128x4  2^7          turn   raced            403        968    2.38     57.2  3.4e-16
       128x8  2^7          turn   raced           1040       1527    1.45     49.2  4.3e-16
      128x16  2^7          turn   raced           2514       1982    0.77     44.8  3.0e-16
      128x32  2^7          chain+rb2 raced           5839       7768    1.13     42.1  3.1e-16
      128x64  2^7          chain+rb2 raced          12394      15715    1.23     43.0  4.3e-16
     128x128  2^7          chain  raced          25839      31985    1.05     44.4  4.3e-16
     128x256  2^7          chain  raced          52116      67461    1.23     47.2  5.3e-16
     128x512  2^7          chain  raced         124330     134100    1.06     42.2  4.8e-16
    128x1024  2^7          chain  raced         297580     317100    1.01     37.4  3.7e-16
    128x2048  2^7          chain  raced         662650     808831    1.18     35.6  5.8e-16
    128x4096  2^7          chain  raced        1412100    1502425    1.05     35.3  1.3e-15
    128x8192  2^7          chain  raced        3067975    3649025    1.09     34.2  1.1e-15
       256x2  2^8          turn   raced            417       1350    3.23     55.3  4.1e-16
       256x4  2^8          turn   raced           1013       1991    1.92     50.5  3.3e-16
       256x8  2^8          turn   raced           2453       3490    1.41     45.9  3.6e-16
      256x16  2^8          chain+rb2 raced           6526       7457    1.14     37.7  3.4e-16
      256x32  2^8          chain+rb2 raced          12531      14851    1.16     42.5  3.7e-16
      256x64  2^8          chain+rb2 raced          22514      31133    1.38     50.9  4.3e-16
     256x128  2^8          chain  raced          49813      61942    1.12     49.3  5.1e-16
     256x256  2^8          chain  raced         120370     128010    1.02     43.6  5.8e-16
     256x512  2^8          chain  raced         301220     335786    1.09     37.0  6.6e-16
    256x1024  2^8          chain  raced         629988     822331    1.29     37.5  5.5e-16
    256x2048  2^8          chain  raced        1417675    1686400    1.16     35.1  1.4e-15
    256x4096  2^8          chain  raced        3881213    3998712    1.02     27.0  1.1e-15
    256x8192  2^8          chain  raced        7111712   10366406    1.19     31.0  1.5e-15
       512x2  2^9          turn   raced           1046       2905    2.76     49.0  3.7e-16
       512x4  2^9          turn   raced           2451       4167    1.70     46.0  4.3e-16
       512x8  2^9          turn   raced           5359       7400    1.35     45.9  4.1e-16
      512x16  2^9          chain+rb raced          13575      17702    1.30     39.2  4.3e-16
      512x32  2^9          chain+rb2 raced          26351      34791    1.31     43.5  4.3e-16
      512x64  2^9          chain+rb2 raced          52505      71731    1.35     46.8  6.1e-16
     512x128  2^9          chain  raced         125403     145843    1.16     41.8  5.3e-16
     512x256  2^9          chain  raced         288353     355236    1.18     38.6  5.1e-16
     512x512  2^9          chain  raced         637213     912294    1.43     37.0  5.4e-16
    512x1024  2^9          chain  raced        1348737    1935687    1.43     36.9  1.1e-15
    512x2048  2^9          chain  raced        3090462    4037262    1.30     33.9  1.2e-15
    512x4096  2^9          chain  raced        6992763   10387093    1.47     31.5  1.5e-15
    512x8192  2^9          chain  raced       16125238   22315887    1.36     28.6  2.8e-15
      1024x2  2^10         turn   raced           2643       6247    2.35     42.6  4.7e-16
      1024x4  2^10         turn   raced           5365       8800    1.63     45.8  4.2e-16
      1024x8  2^10         turn   raced          11560      15425    1.33     46.1  3.0e-16
     1024x16  2^10         chain+rb2 raced          29960      38032    1.25     38.3  4.6e-16
     1024x32  2^10         chain+rb2 raced          48808      76386    1.55     50.4  5.4e-16
     1024x64  2^10         chain+rb2 raced         124133     163122    1.29     42.2  4.4e-16
    1024x128  2^10         chain  raced         299347     380383    1.24     37.2  5.9e-16
    1024x256  2^10         chain  raced         544600    1074888    1.96     43.3  5.7e-16
    1024x512  2^10         chain  raced        1413613    2299812    1.62     35.2  1.2e-15
   1024x1024  2^10         chain  raced        3017575    5241144    1.54     34.7  1.3e-15
   1024x2048  2^10         chain  raced        7922962   11001043    1.35     27.8  1.6e-15
   1024x4096  2^10         chain  raced       17323475   37088506    1.70     26.6  2.6e-15
      2048x2  2^11         turn   raced           5843      12833    2.19     42.1  2.6e-16
      2048x4  2^11         turn   raced          11820      18421    1.55     45.0  3.4e-16
      2048x8  2^11         turn   raced          25505      32733    1.28     45.0  3.7e-16
     2048x16  2^11         turn   raced          58138      79004    1.34     42.3  5.2e-16
     2048x32  2^11         turn   raced         148393     160133    1.04     35.3  3.9e-16
     2048x64  2^11         chain+rb2 raced         345993     409817    1.17     32.2  5.2e-16
    2048x128  2^11         chain  raced         756562    1118974    1.44     31.2  5.8e-16
    2048x256  2^11         chain  raced        1591175    2323375    1.45     31.3  1.2e-15
    2048x512  2^11         chain  raced        3651937    5877250    1.59     28.7  1.3e-15
   2048x1024  2^11         chain  raced        8633500   15236762    1.75     25.5  1.5e-15
   2048x2048  2^11         chain  raced       19041388   36601143    1.57     24.2  3.2e-15
      4096x2  2^12         turn   raced          12215      25983    2.13     43.6  3.7e-16
      4096x4  2^12         turn   raced          24706      37823    1.52     46.4  3.7e-16
      4096x8  2^12         turn   raced          53170      67586    1.27     46.2  3.9e-16
     4096x16  2^12         turn   raced         138143     146935    1.06     38.0  2.9e-16
     4096x32  2^12         turn   raced         327493     389423    1.11     34.0  4.1e-16
     4096x64  2^12         turn   raced         757312     887275    1.12     31.2  5.3e-16
    4096x128  2^12         chain  raced        1812388    1854093    0.98     27.5  1.2e-15
    4096x256  2^12         chain  raced        4000912    4777406    1.19     26.2  1.4e-15
    4096x512  2^12         chain  raced        8760563   13599131    1.52     25.1  1.8e-15
   4096x1024  2^12         chain  raced       19827888   37396193    1.88     23.3  3.0e-15
      8192x2  2^13         turn   raced          25862      53092    2.04     44.3  3.6e-16
      8192x4  2^13         turn   raced          52543      78093    1.48     46.8  3.6e-16
      8192x8  2^13         turn   raced         130960     164281    1.25     40.0  4.1e-16
     8192x16  2^13         turn   raced         317047     452617    1.41     35.1  3.1e-16
     8192x32  2^13         turn   raced         737913     973131    1.31     32.0  4.7e-16
     8192x64  2^13         turn   raced        1660400    2029031    1.16     30.0  5.5e-16
    8192x128  2^13         chain  raced        4079838    5188356    1.17     25.7  1.3e-15
    8192x256  2^13         chain  raced       10465000   14129019    1.33     21.0  1.5e-15
    8192x512  2^13         chain  raced       22762800   40421868    1.77     20.3  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           81      0      2   1.05   1.23   2.27    1.40
 turn                            30      1      1   1.11   1.43   2.76    1.57
 chain+rb2                       28      1      4   0.89   1.27   5.55    1.70
 chain+rb                        20      1      3   0.94   4.12  10.27    3.46
 ALL                            159      3     10   1.02   1.34   4.22    1.66
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159      3     10   1.02   1.34   4.22    1.66
 ALL                            159      3     10   1.02   1.34   4.22    1.66
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      0      0   1.05   1.23   1.55    1.29
 > 65536 points                  45      0      1   1.09   1.29   1.70    1.31
 <= 256 points                   28      0      3   0.94   4.57   8.66    4.00
 1025..4096                      21      2      4   0.84   1.35   2.35    1.37
 257..1024                       17      1      2   0.89   2.04   3.23    1.85
 ALL                            159      3     10   1.02   1.34   4.22    1.66
```


worst 10: 32x64 (chain+rb2 0.72), 128x16 (turn 0.77), 32x16 (chain+rb 0.79), 16x128 (chain 0.84), 64x32 (chain+rb2 0.85), 64x16 (chain+rb2 0.89), 2x2 (chain+rb 0.92), 8x8 (chain+rb 0.94), 16x16 (chain+rb2 0.94), 4096x128 (chain 0.98)
best 5: 4x2 (chain+rb 10.28), 2x4 (chain+rb 10.27), 2x8 (chain+rb 8.66), 8x2 (chain+rb 8.63), 16x2 (chain+rb 7.31)
