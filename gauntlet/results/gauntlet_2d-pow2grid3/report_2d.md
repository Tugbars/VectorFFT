# gauntlet report (2D)

run: `gauntlet_2d-pow2grid3`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.145..1.267


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain+rb raced             13         13    0.94      3.0  1.4e-16
         2x4  2            chain+rb raced             15        158   10.30      7.8  1.7e-16
         2x8  2            chain+rb raced             19        168    8.77     16.7  1.9e-16
        2x16  2            chain+rb2 raced             31        206    6.71     26.1  2.0e-16
        2x32  2            chain+rb2 raced             45        307    6.78     42.4  2.0e-16
        2x64  2            chain+rb2 raced             84        468    5.54     53.4  2.2e-16
       2x128  2            chain  raced            165        832    4.99     62.2  3.7e-16
       2x256  2            chain  raced            325       1538    4.72     70.9  3.9e-16
       2x512  2            chain  raced           1014       2971    2.70     50.5  4.5e-16
      2x1024  2            chain  raced           2292       6051    2.61     49.1  4.0e-16
      2x2048  2            chain  raced           5144      12274    2.36     47.8  2.7e-16
      2x4096  2            chain  raced          10871      25108    2.31     49.0  3.2e-16
      2x8192  2            chain  raced          23203      51542    2.20     49.4  3.4e-16
         4x2  2^2          chain+rb raced             15        155   10.18      7.9  8.9e-17
         4x4  2^2          chain+rb raced             18         18    1.00     17.8  2.0e-16
         4x8  2^2          chain+rb raced             25        175    6.92     31.7  1.7e-16
        4x16  2^2          chain+rb2 raced             43        211    4.94     45.1  2.8e-16
        4x32  2^2          chain+rb2 raced             75        318    4.22     59.4  1.7e-16
        4x64  2^2          chain+rb2 raced            155        507    3.26     65.9  2.9e-16
       4x128  2^2          chain  raced            329        952    2.89     70.0  3.7e-16
       4x256  2^2          chain  raced            695       1806    2.56     73.7  3.7e-16
       4x512  2^2          chain  raced           2111       3609    1.66     53.4  4.6e-16
      4x1024  2^2          chain  raced           4435       7493    1.68     55.4  3.7e-16
      4x2048  2^2          chain  raced          10038      15430    1.53     53.0  3.6e-16
      4x4096  2^2          chain  raced          21302      32113    1.50     53.8  3.1e-16
      4x8192  2^2          chain  raced          44361      66979    1.51     55.4  3.6e-16
         8x2  2^3          chain+rb raced             19        166    8.63     16.6  1.2e-16
         8x4  2^3          chain+rb raced             25        180    7.12     32.1  1.6e-16
         8x8  2^3          chain+rb raced             38         36    0.95     50.0  1.5e-16
        8x16  2^3          chain+rb2 raced             75        246    3.28     59.7  2.8e-16
        8x32  2^3          chain+rb2 raced            147        405    2.74     69.5  2.6e-16
        8x64  2^3          chain+rb2 raced            322        691    2.15     71.6  2.9e-16
       8x128  2^3          chain  raced            712       1407    1.98     71.9  3.3e-16
       8x256  2^3          chain  raced           1894       2857    1.27     59.5  3.5e-16
       8x512  2^3          chain  raced           4773       5782    1.20     51.5  4.6e-16
      8x1024  2^3          chain  raced          10255      11872    1.15     51.9  3.8e-16
      8x2048  2^3          chain  raced          23033      24945    1.07     49.8  3.8e-16
      8x4096  2^3          chain  raced          48507      52975    1.09     50.7  5.4e-16
      8x8192  2^3          chain  raced          94207     117221    1.21     55.7  5.2e-16
        16x2  2^4          chain+rb raced             26        195    7.36     30.3  2.2e-16
        16x4  2^4          chain+rb raced             40        214    5.30     47.6  2.5e-16
        16x8  2^4          chain+rb raced             70        245    3.49     63.9  2.0e-16
       16x16  2^4          chain+rb raced            163        146    0.84     63.0  2.5e-16
       16x32  2^4          chain+rb2 raced            320        353    1.10     72.0  3.0e-16
       16x64  2^4          chain+rb2 raced            716        712    0.98     71.5  3.9e-16
      16x128  2^4          chain  raced           2059       1615    0.78     54.7  4.6e-16
      16x256  2^4          chain  raced           5275       7003    1.17     46.6  3.8e-16
      16x512  2^4          chain  raced          12291      14463    1.17     43.3  4.9e-16
     16x1024  2^4          chain  raced          25977      30948    0.98     44.1  3.7e-16
     16x2048  2^4          chain  raced          57608      68308    1.18     42.7  3.7e-16
     16x4096  2^4          chain  raced         132403     133338    1.00     39.6  3.1e-16
     16x8192  2^4          chain  raced         289060     313726    1.08     38.5  4.2e-16
        32x2  2^5          chain+rb raced             49        273    5.47     39.0  2.6e-16
        32x4  2^5          chain+rb raced             79        319    3.94     57.0  2.0e-16
        32x8  2^5          chain+rb raced            150        399    2.49     68.2  2.4e-16
       32x16  2^5          chain+rb raced            347        341    0.98     66.3  2.3e-16
       32x32  2^5          chain+rb2 raced            713        720    1.01     71.8  2.4e-16
       32x64  2^5          chain+rb2 raced           1936       1884    0.68     58.2  3.5e-16
      32x128  2^5          chain  raced           5255       6914    1.16     46.8  3.5e-16
      32x256  2^5          chain  raced          10730      15096    1.21     49.6  4.0e-16
      32x512  2^5          chain  raced          26767      31944    1.18     42.8  4.6e-16
     32x1024  2^5          chain  raced          56920      68516    1.19     43.2  3.5e-16
     32x2048  2^5          chain  raced         123000     150236    1.22     42.6  3.6e-16
     32x4096  2^5          chain  raced         277880     321857    1.14     40.1  4.1e-16
     32x8192  2^5          chain  raced         594175     855037    1.42     39.7  5.2e-16
        64x2  2^6          chain+rb raced             98        414    4.20     45.5  3.4e-16
        64x4  2^6          chain+rb raced            178        508    2.42     57.7  3.4e-16
        64x8  2^6          chain+rb raced            334        684    1.99     69.0  3.2e-16
       64x16  2^6          chain+rb2 raced            819        828    1.00     62.5  3.4e-16
       64x32  2^6          chain+rb2 raced           2161       2014    0.91     52.1  3.0e-16
       64x64  2^6          chain+rb2 raced           5746       7083    1.21     42.8  3.8e-16
      64x128  2^6          chain  raced          12723      13991    1.10     41.9  4.2e-16
      64x256  2^6          chain  raced          25608      32273    1.13     44.8  3.8e-16
      64x512  2^6          chain  raced          49811      67096    1.24     49.3  4.4e-16
     64x1024  2^6          chain  raced         128990     144070    1.11     40.6  3.6e-16
     64x2048  2^6          chain  raced         298747     332646    1.11     37.3  4.2e-16
     64x4096  2^6          chain  raced         582063     751056    1.23     40.5  5.6e-16
     64x8192  2^6          chain  raced        1370738    1572925    1.14     36.3  1.1e-15
       128x2  2^7          chain+rb raced            336        726    2.16     30.5  3.9e-16
       128x4  2^7          chain+rb raced            640        967    1.46     36.0  3.4e-16
       128x8  2^7          chain+rb raced           1225       1517    1.12     41.8  3.5e-16
      128x16  2^7          chain+rb2 raced           3819       2000    0.52     29.5  3.7e-16
      128x32  2^7          chain+rb2 raced           5544       7218    1.13     44.3  3.1e-16
      128x64  2^7          chain+rb2 raced          12818      15277    1.18     41.5  4.3e-16
     128x128  2^7          chain  raced          28474      30031    1.05     40.3  4.3e-16
     128x256  2^7          chain  raced          50189      61036    1.06     49.0  5.3e-16
     128x512  2^7          chain  raced         119190     132643    1.09     44.0  4.8e-16
    128x1024  2^7          chain  raced         277267     313077    1.10     40.2  3.7e-16
    128x2048  2^7          chain  raced         618050     714425    1.11     38.2  5.8e-16
    128x4096  2^7          chain  raced        1322113    1492131    1.11     37.7  1.3e-15
    128x8192  2^7          chain  raced        2964700    3327375    1.10     35.4  1.4e-15
       256x2  2^8          chain+rb raced            668       1349    1.99     34.5  3.3e-16
       256x4  2^8          chain+rb raced           1517       1964    1.18     33.7  4.4e-16
       256x8  2^8          chain+rb raced           3165       3457    1.09     35.6  5.8e-16
      256x16  2^8          chain+rb raced           5666       7483    1.15     43.4  3.6e-16
      256x32  2^8          chain+rb2 raced          10925      14779    1.14     48.7  3.7e-16
      256x64  2^8          chain+rb2 raced          22416      31171    1.22     51.2  4.3e-16
     256x128  2^8          chain  raced          54059      61272    1.03     45.5  5.1e-16
     256x256  2^8          chain  raced         116567     128251    1.10     45.0  5.8e-16
     256x512  2^8          chain  raced         281973     331126    1.13     39.5  6.6e-16
    256x1024  2^8          chain  raced         594012     796512    1.31     39.7  5.5e-16
    256x2048  2^8          chain  raced        1334075    1557650    1.15     37.3  1.4e-15
    256x4096  2^8          chain  raced        3769800    3400399    0.88     27.8  1.1e-15
    256x8192  2^8          chain  raced        6927050    7805262    1.11     31.8  1.5e-15
       512x2  2^9          chain+rb raced           2227       2917    1.29     23.0  4.2e-16
       512x4  2^9          chain+rb raced           3408       4143    1.17     33.1  5.7e-16
       512x8  2^9          chain+rb raced           6765       7408    1.08     36.3  4.1e-16
      512x16  2^9          chain+rb2 raced          12765      17694    1.27     41.7  4.5e-16
      512x32  2^9          chain+rb2 raced          30940      34823    1.11     37.1  4.3e-16
      512x64  2^9          chain+rb2 raced          47010      72655    1.53     52.3  6.1e-16
     512x128  2^9          chain  raced         121933     145493    1.19     43.0  5.3e-16
     512x256  2^9          chain  raced         278567     343523    1.22     40.0  5.1e-16
     512x512  2^9          chain  raced         602487     854669    1.40     39.2  5.4e-16
    512x1024  2^9          chain  raced        1296775    1815324    1.39     38.4  1.1e-15
    512x2048  2^9          chain  raced        2749575    3472444    1.22     38.1  1.2e-15
    512x4096  2^9          chain  raced        6809600    8292031    1.20     32.3  1.5e-15
    512x8192  2^9          chain  raced       15799525   18393331    1.16     29.2  2.8e-15
      1024x2  2^10         chain+rb raced           4484       6258    1.34     25.1  5.1e-16
      1024x4  2^10         chain+rb raced           8273       8768    0.98     29.7  5.3e-16
      1024x8  2^10         chain+rb raced          13537      15436    1.12     39.3  4.3e-16
     1024x16  2^10         chain+rb2 raced          27044      38019    1.40     42.4  4.9e-16
     1024x32  2^10         chain+rb2 raced          49826      76160    1.52     49.3  5.4e-16
     1024x64  2^10         chain+rb2 raced         120923     162232    1.34     43.4  4.4e-16
    1024x128  2^10         chain  raced         290067     366586    1.26     38.4  5.9e-16
    1024x256  2^10         chain  raced         613600    1005200    1.62     38.5  5.7e-16
    1024x512  2^10         chain  raced        1337438    2176581    1.59     37.2  1.2e-15
   1024x1024  2^10         chain  raced        2836313    4664562    1.55     37.0  1.3e-15
   1024x2048  2^10         chain  raced        7707588    9740012    1.24     28.6  1.6e-15
   1024x4096  2^10         chain  raced       15711600   29946706    1.43     29.4  2.8e-15
      2048x2  2^11         chain+rb raced          12204      12819    0.98     20.1  4.3e-16
      2048x4  2^11         chain+rb raced          20606      18350    0.77     25.8  4.8e-16
      2048x8  2^11         chain+rb raced          49920      32732    0.65     23.0  4.6e-16
     2048x16  2^11         chain+rb2 raced          70934      78528    1.10     34.6  4.5e-16
     2048x32  2^11         chain+rb2 raced         155547     159368    1.02     33.7  4.9e-16
     2048x64  2^11         chain+rb2 raced         335100     437510    1.24     33.2  5.2e-16
    2048x128  2^11         chain  raced         751575    1040262    1.35     31.4  5.5e-16
    2048x256  2^11         chain  raced        1440825    2261750    1.44     34.6  1.2e-15
    2048x512  2^11         chain  raced        3245837    5247950    1.59     32.3  1.3e-15
   2048x1024  2^11         chain  raced        8066812   12751893    1.31     27.3  1.5e-15
   2048x2048  2^11         chain  raced       16770750   24400581    1.40     27.5  3.1e-15
      4096x2  2^12         chain+rb raced          27916      25985    0.90     19.1  4.4e-16
      4096x4  2^12         chain+rb raced          50421      37847    0.75     22.7  5.1e-16
      4096x8  2^12         chain+rb raced          92334      67600    0.72     26.6  5.2e-16
     4096x16  2^12         chain+rb2 raced         167690     162933    0.93     31.3  4.8e-16
     4096x32  2^12         chain+rb2 raced         338667     358863    1.03     32.9  5.2e-16
     4096x64  2^12         chain+rb2 raced         740500     821718    1.08     31.9  7.0e-16
    4096x128  2^12         chain  raced        1710025    1745506    0.96     29.1  1.2e-15
    4096x256  2^12         chain  raced        3456112    4473381    1.27     30.3  1.3e-15
    4096x512  2^12         chain  raced        8369150   12941769    1.47     26.3  1.8e-15
   4096x1024  2^12         chain  raced       18410125   22212937    1.19     25.1  3.1e-15
      8192x2  2^13         chain+rb raced          56855      53096    0.84     20.2  5.4e-16
      8192x4  2^13         chain+rb raced          93874      78254    0.83     26.2  6.1e-16
      8192x8  2^13         chain+rb raced         193483     158596    0.81     27.1  5.8e-16
     8192x16  2^13         chain+rb2 raced         334387     395883    1.16     33.3  4.7e-16
     8192x32  2^13         chain+rb2 raced         748625     872225    1.14     31.5  4.9e-16
     8192x64  2^13         chain+rb2 raced        1655312    1876438    1.12     30.1  1.3e-15
    8192x128  2^13         chain  raced        3528350    4551200    1.22     29.7  1.3e-15
    8192x256  2^13         chain  raced        7642550   13525524    1.71     28.8  1.5e-15
    8192x512  2^13         chain  raced       18570100   36173487    1.87     24.8  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           81      1      4   1.07   1.22   2.20    1.36
 chain+rb                        42      4     14   0.81   1.17   7.36    1.84
 chain+rb2                       36      2      5   0.93   1.17   4.94    1.52
 ALL                            159      7     23   0.94   1.21   4.22    1.51
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159      7     23   0.94   1.21   4.22    1.51
 ALL                            159      7     23   0.94   1.21   4.22    1.51
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      4     10   0.81   1.13   1.52    1.13
 > 65536 points                  45      0      2   1.08   1.22   1.59    1.25
 <= 256 points                   28      0      3   0.95   4.58   8.77    3.90
 1025..4096                      21      3      6   0.78   1.16   1.68    1.17
 257..1024                       17      0      2   0.98   1.46   2.89    1.63
 ALL                            159      7     23   0.94   1.21   4.22    1.51
```


worst 10: 128x16 (chain+rb2 0.52), 2048x8 (chain+rb 0.65), 32x64 (chain+rb2 0.68), 4096x8 (chain+rb 0.72), 4096x4 (chain+rb 0.75), 2048x4 (chain+rb 0.77), 16x128 (chain 0.78), 8192x8 (chain+rb 0.81), 8192x4 (chain+rb 0.83), 16x16 (chain+rb 0.84)
best 5: 2x4 (chain+rb 10.30), 4x2 (chain+rb 10.18), 2x8 (chain+rb 8.77), 8x2 (chain+rb 8.63), 16x2 (chain+rb 7.36)
