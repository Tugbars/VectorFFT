# gauntlet report (2D)

run: `kfr_2d_pow2`  contract: 2D c2c interleaved, natural, out of place, K=1_kfr  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 0.930..1.301


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            csk+rb replayed          11        103    9.61      3.7  1.4e-16
         2x4  2            csk+rb replayed          12        128   10.23      9.7  1.7e-16
         2x8  2            csk+rb replayed          17        267   16.02     19.3  1.9e-16
        2x16  2            csk+rb2 replayed          27        463   16.84     29.3  2.0e-16
        2x32  2            csk+rb2 replayed          42        792   18.58     46.0  2.0e-16
        2x64  2            csk    replayed          78       1512   18.92     57.4  2.2e-16
       2x128  2            csk    replayed         158       2821   17.79     64.8  2.5e-16
       2x256  2            csk    replayed         321       5519   17.15     71.8  2.6e-16
       2x512  2            chain  replayed         944      11887   10.29     54.2  3.4e-16
      2x1024  2            csk    replayed        2181      23016   10.11     51.6  3.3e-16
      2x2048  2            csk    replayed        4653      46039    9.84     52.8  3.6e-16
      2x4096  2            csk    replayed        9782      93536    9.53     54.4  4.5e-16
      2x8192  2            csk    replayed       21125     193595    9.01     54.3  3.8e-16
         4x2  2^2          csk+rb replayed          12        129   10.43      9.8  8.9e-17
         4x4  2^2          csk+rb replayed          15        117    7.91     21.7  2.0e-16
         4x8  2^2          csk+rb replayed          21        209    9.74     37.5  1.7e-16
        4x16  2^2          csk+rb2 replayed          40        223    5.52     47.5  2.8e-16
        4x32  2^2          csk+rb2 replayed          75        563    7.53     60.0  1.7e-16
        4x64  2^2          csk+rb2 replayed         152       1044    5.93     67.3  2.6e-16
       4x128  2^2          csk    replayed         319       1978    6.18     72.2  3.0e-16
       4x256  2^2          chain  replayed         685       3902    5.59     74.8  2.3e-16
       4x512  2^2          chain  replayed        2010       8281    4.10     56.0  3.1e-16
      4x1024  2^2          csk    replayed        4196      16978    3.98     58.6  2.8e-16
      4x2048  2^2          csk    replayed        8720      34907    3.97     61.1  3.6e-16
      4x4096  2^2          csk    replayed       18743      73202    3.87     61.2  3.1e-16
      4x8192  2^2          csk    replayed       40390     155214    3.80     60.8  3.6e-16
         8x2  2^3          csk+rb replayed          16        270   16.92     20.0  1.2e-16
         8x4  2^3          csk+rb replayed          21        210    9.75     37.7  1.6e-16
         8x8  2^3          csk+rb replayed          34        169    4.91     56.2  1.5e-16
        8x16  2^3          csk+rb2 replayed          72        473    6.55     62.3  1.9e-16
        8x32  2^3          csk+rb2 replayed         144        853    5.91     71.1  2.6e-16
        8x64  2^3          csk+rb2 replayed         318        873    2.74     72.4  3.4e-16
       8x128  2^3          chain  replayed         704       3417    4.50     72.7  3.3e-16
       8x256  2^3          csk    replayed        2070       6489    2.98     54.4  2.8e-16
       8x512  2^3          csk    replayed        4359      14161    3.08     56.4  3.6e-16
      8x1024  2^3          csk    replayed        8417      29552    3.48     63.3  2.9e-16
      8x2048  2^3          csk    replayed       17634      62038    3.38     65.0  3.8e-16
      8x4096  2^3          csk    replayed       37984     128528    3.38     64.7  4.1e-16
      8x8192  2^3          chain  replayed       97587     272998    2.79     53.7  4.8e-16
        16x2  2^4          csk+rb replayed          23        463   19.73     34.3  2.2e-16
        16x4  2^4          csk+rb replayed          36        226    6.21     53.0  2.5e-16
        16x8  2^4          csk+rb replayed          67        474    7.05     67.0  2.0e-16
       16x16  2^4          csk+rb2 replayed         157        338    1.71     65.1  2.5e-16
       16x32  2^4          csk+rb2 replayed         317       1509    3.72     72.6  3.0e-16
       16x64  2^4          csk    replayed         750       3008    3.93     68.2  2.5e-16
      16x128  2^4          csk    replayed        1748       5844    3.00     64.4  3.1e-16
      16x256  2^4          csk+rb2 replayed        4144       7685    1.73     59.3  3.8e-16
      16x512  2^4          csk    replayed        8651      27854    3.02     61.6  2.8e-16
     16x1024  2^4          csk    replayed       19589      60939    2.27     58.5  3.3e-16
     16x2048  2^4          csk    replayed       38772     120842    2.80     63.4  3.7e-16
     16x4096  2^4          csk    replayed       96833     265346    2.67     54.1  3.4e-16
     16x8192  2^4          csk    replayed      214713     590750    2.67     51.9  4.2e-16
        32x2  2^5          chain+rb replayed          50        796   15.89     38.4  2.6e-16
        32x4  2^5          chain+rb replayed          79        566    7.08     56.4  2.1e-16
        32x8  2^5          chain+rb replayed         150        854    4.86     68.5  2.9e-16
       32x16  2^5          chain+rb replayed         347       1579    4.31     66.4  2.3e-16
       32x32  2^5          chain+rb2 replayed         732       1298    1.77     70.0  2.4e-16
       32x64  2^5          chain+rb2 replayed        2604       6531    2.39     43.3  3.4e-16
      32x128  2^5          csk    replayed        4888      14128    2.88     50.3  3.5e-16
      32x256  2^5          chain  replayed       11275      28925    2.41     47.2  2.8e-16
      32x512  2^5          csk    replayed       23092      57305    1.87     49.7  3.7e-16
     32x1024  2^5          csk    replayed       43975      73597    1.46     55.9  3.5e-16
     32x2048  2^5          csk    replayed      108643     270208    2.47     48.3  4.4e-16
     32x4096  2^5          chain  replayed      261227     551470    2.05     42.6  4.1e-16
     32x8192  2^5          csk    replayed      673075    1291124    1.90     35.1  5.5e-16
        64x2  2^6          chain+rb replayed         102       1480   14.52     44.1  2.6e-16
        64x4  2^6          chain+rb replayed         173       1046    6.01     59.1  3.4e-16
        64x8  2^6          chain+rb replayed         337        874    2.58     68.4  3.0e-16
       64x16  2^6          chain+rb replayed         762       3451    3.36     67.2  3.2e-16
       64x32  2^6          chain+rb2 replayed        2277       8650    3.45     49.5  3.0e-16
       64x64  2^6          csk    replayed        5235       6698    1.26     46.9  3.8e-16
      64x128  2^6          csk    replayed       11801      29471    2.21     45.1  3.9e-16
      64x256  2^6          chain  replayed       23931      59199    2.47     47.9  3.4e-16
      64x512  2^6          csk    replayed       51975     118137    1.96     47.3  4.4e-16
     64x1024  2^6          chain  replayed      118723     248103    2.03     44.2  3.2e-16
     64x2048  2^6          chain  replayed      292107     526446    1.73     38.1  4.2e-16
     64x4096  2^6          chain  replayed      579412     875143    1.50     40.7  4.9e-16
     64x8192  2^6          chain  replayed     1286862    2584731    2.00     38.7  1.1e-15
       128x2  2^7          turn   replayed         207       2833   13.65     49.4  2.8e-16
       128x4  2^7          turn   replayed         407       1983    4.83     56.6  3.4e-16
       128x8  2^7          turn   replayed        1001       3420    3.22     51.1  2.1e-16
      128x16  2^7          turn   replayed        2467       6527    2.49     45.7  2.7e-16
      128x32  2^7          chain+rb2 replayed        5261      13096    2.46     46.7  3.3e-16
      128x64  2^7          chain+rb2 replayed       10907      28858    2.58     48.8  3.3e-16
     128x128  2^7          chain  replayed       24101      28461    1.16     47.6  3.8e-16
     128x256  2^7          chain  replayed       47813     118291    2.42     51.4  4.0e-16
     128x512  2^7          chain  replayed      120653     241762    1.93     43.5  3.6e-16
    128x1024  2^7          chain  replayed      272067     529006    1.92     41.0  4.6e-16
    128x2048  2^7          chain  replayed      615325    1147950    1.83     38.3  5.8e-16
    128x4096  2^7          chain  replayed     1290237    2462894    1.86     38.6  1.2e-15
    128x8192  2^7          chain  replayed     3065725    5515575    1.77     34.2  1.1e-15
       256x2  2^8          turn   replayed         410       5570   13.55     56.2  2.5e-16
       256x4  2^8          turn   replayed         986       4118    3.92     51.9  2.2e-16
       256x8  2^8          turn   replayed        2439       6652    2.66     46.2  2.9e-16
      256x16  2^8          chain+rb2 replayed        5902       7864    1.30     41.6  3.6e-16
      256x32  2^8          chain+rb2 replayed       11030      27347    2.45     48.3  3.0e-16
      256x64  2^8          chain+rb2 replayed       22689      57971    2.51     50.5  3.8e-16
     256x128  2^8          chain  replayed       51064     118141    2.30     48.1  5.1e-16
     256x256  2^8          chain  replayed      111210     123145    1.10     47.1  3.5e-16
     256x512  2^8          chain  replayed      265940     512353    1.83     41.9  4.4e-16
    256x1024  2^8          chain  replayed      589375    1121087    1.86     40.0  5.5e-16
    256x2048  2^8          chain  replayed     1299887    2464987    1.86     38.3  1.3e-15
    256x4096  2^8          chain  replayed     3141187    5568906    1.72     33.4  1.1e-15
    256x8192  2^8          chain  replayed     8686012   13737056    1.54     25.4  1.7e-15
       512x2  2^9          turn   replayed        1042      11176   10.63     49.1  2.3e-16
       512x4  2^9          turn   replayed        2467       8121    3.20     45.7  2.8e-16
       512x8  2^9          turn   replayed        5437      14535    2.65     45.2  3.1e-16
      512x16  2^9          chain+rb2 replayed       12755      28433    2.17     41.7  3.7e-16
      512x32  2^9          chain+rb2 replayed       24461      56720    2.26     46.9  3.8e-16
      512x64  2^9          chain+rb2 replayed       50921     118821    2.29     48.3  3.7e-16
     512x128  2^9          chain  replayed      115930     242605    2.07     45.2  3.7e-16
     512x256  2^9          chain  replayed      250753     510687    2.02     44.4  4.0e-16
     512x512  2^9          chain  replayed      579388     611500    1.05     40.7  4.6e-16
    512x1024  2^9          chain  replayed     1253812    2349562    1.73     39.7  1.1e-15
    512x2048  2^9          chain  replayed     2995150    5370975    1.73     35.0  1.2e-15
    512x4096  2^9          chain  replayed     7059925   13391425    1.80     31.2  1.4e-15
    512x8192  2^9          chain  replayed    16197250   32093937    1.92     28.5  2.7e-15
      1024x2  2^10         turn   replayed        2643      22254    8.18     42.6  3.2e-16
      1024x4  2^10         turn   replayed        5402      17284    3.18     45.5  3.2e-16
      1024x8  2^10         turn   replayed       11520      29026    2.49     46.2  3.0e-16
     1024x16  2^10         turn   replayed       26598      59995    2.24     43.1  3.9e-16
     1024x32  2^10         chain+rb2 replayed       50090      69452    1.38     49.1  4.0e-16
     1024x64  2^10         chain+rb2 replayed      124523     243998    1.75     42.1  4.4e-16
    1024x128  2^10         chain  replayed      266500     509167    1.90     41.8  4.7e-16
    1024x256  2^10         chain  replayed      548500    1102500    1.99     43.0  5.2e-16
    1024x512  2^10         chain  replayed     1267237    2316556    1.77     39.3  1.1e-15
   1024x1024  2^10         chain  replayed     3178012    3278893    1.03     33.0  1.3e-15
   1024x2048  2^10         chain  replayed     7504112   14943037    1.61     29.3  1.2e-15
   1024x4096  2^10         chain  replayed    17558087   31552762    1.74     26.3  2.9e-15
      2048x2  2^11         turn   replayed        5826      44928    7.70     42.2  2.8e-16
      2048x4  2^11         turn   replayed       11801      34332    2.90     45.1  4.1e-16
      2048x8  2^11         turn   replayed       25333      60488    2.35     45.3  3.7e-16
     2048x16  2^11         turn   replayed       58384     120137    2.01     42.1  3.9e-16
     2048x32  2^11         chain+rb replayed      154000     238261    1.50     34.0  3.7e-16
     2048x64  2^11         chain+rb2 replayed      314813     519836    1.63     35.4  4.7e-16
    2048x128  2^11         chain  replayed      731625    1113600    1.51     32.2  4.6e-16
    2048x256  2^11         chain  replayed     1452963    2374806    1.63     34.3  9.5e-16
    2048x512  2^11         chain  replayed     3004462    5258781    1.73     34.9  1.3e-15
   2048x1024  2^11         chain  replayed     8672563   13592012    1.38     25.4  1.5e-15
   2048x2048  2^11         chain  replayed    16934138   24817869    1.39     27.2  3.0e-15
      4096x2  2^12         turn   replayed       12520      95325    7.50     42.5  3.7e-16
      4096x4  2^12         turn   replayed       25205      71512    2.82     45.5  3.7e-16
      4096x8  2^12         turn   replayed       53189     126163    2.35     46.2  5.2e-16
     4096x16  2^12         turn   replayed      135897     265890    1.91     38.6  3.3e-16
     4096x32  2^12         chain+rb2 replayed      343147     529137    1.51     32.5  4.1e-16
     4096x64  2^12         chain+rb2 replayed      769912     860918    1.11     30.6  4.3e-16
    4096x128  2^12         chain  replayed     1694750    2541031    1.46     29.4  1.2e-15
    4096x256  2^12         chain  replayed     4001250    5986099    1.48     26.2  1.0e-15
    4096x512  2^12         chain  replayed     7891250   13542781    1.58     27.9  1.2e-15
   4096x1024  2^12         chain  replayed    16415213   32415700    1.90     28.1  2.8e-15
      8192x2  2^13         turn   replayed       25795     191337    7.37     44.5  3.8e-16
      8192x4  2^13         turn   replayed       52205     150204    2.87     47.1  3.6e-16
      8192x8  2^13         turn   replayed      127257     268565    2.02     41.2  4.1e-16
     8192x16  2^13         turn   replayed      313820     585493    1.84     35.5  3.6e-16
     8192x32  2^13         chain+rb2 replayed      779788    1240493    1.54     30.3  4.1e-16
     8192x64  2^13         chain  replayed     1692412    2639756    1.52     29.4  1.3e-15
    8192x128  2^13         chain  replayed     3901063    6126243    1.37     26.9  1.2e-15
    8192x256  2^13         chain  replayed    10377700   15445175    1.45     21.2  1.6e-15
    8192x512  2^13         chain  replayed    16711387   34001625    1.99     27.6  2.7e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           52      0      0   1.38   1.81   2.47    1.90
 csk                             32      0      0   1.90   3.23  10.11    3.90
 turn                            26      0      0   2.01   2.88  10.63    3.74
 chain+rb2                       17      0      0   1.30   2.17   2.58    1.95
 csk+rb                          12      0      0   6.21   9.74  16.92    9.90
 csk+rb2                         11      0      0   1.73   5.91  16.84    5.29
 chain+rb                         9      0      0   1.50   4.86  15.89    5.18
 ALL                            159      0      0   1.50   2.49  10.23    3.16
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159      0      0   1.50   2.49  10.23    3.16
 ALL                            159      0      0   1.50   2.49  10.23    3.16
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      0      0   1.50   2.42   3.97    2.58
 > 65536 points                  45      0      0   1.38   1.73   1.99    1.67
 <= 256 points                   28      0      0   4.91   9.68  18.58    9.21
 1025..4096                      21      0      0   1.73   3.00   8.18    3.31
 257..1024                       17      0      0   2.58   4.31  13.55    4.95
 ALL                            159      0      0   1.50   2.49  10.23    3.16
```


worst 10: 1024x1024 (chain 1.03), 512x512 (chain 1.05), 256x256 (chain 1.10), 4096x64 (chain+rb2 1.11), 128x128 (chain 1.16), 64x64 (csk 1.26), 256x16 (chain+rb2 1.30), 8192x128 (chain 1.37), 2048x1024 (chain 1.38), 1024x32 (chain+rb2 1.38)
best 5: 16x2 (csk+rb 19.73), 2x64 (csk 18.92), 2x32 (csk+rb2 18.58), 2x128 (csk 17.79), 2x256 (csk 17.15)
