# gauntlet report (2D)

run: `mt8_2d_pow2_estimate_20261109`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.195..1.547

threaded plans that ran serial (engaged = 0 at both flips): 49: 128x2, 128x4, 128x8, 16x16, 16x2, 16x32, 16x4, 16x64, 16x8, 256x2, 256x4, 2x1024, 2x128, 2x16, 2x2, 2x2048, 2x256, 2x32, 2x4, 2x4096, 2x512, 2x64, 2x8, 2x8192, 32x16, 32x2, 32x32, 32x4, 32x8, 4x128, 4x16, 4x2, 4x256, 4x32, 4x4, 4x64, 4x8, 512x2, 64x16, 64x2 ...


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            csk+rb replayed          20         13    0.67      2.0  1.4e-16
         2x4  2            csk+rb replayed          20        169    8.20      6.0  1.7e-16
         2x8  2            csk+rb replayed          26        182    6.88     12.3  1.9e-16
        2x16  2            csk+rb replayed          27        217    7.85     29.3  2.0e-16
        2x32  2            csk+rb replayed          47        319    6.25     40.6  2.7e-16
        2x64  2            csk+rb replayed          81        488    5.84     55.3  2.7e-16
       2x128  2            csk    replayed         168        852    4.97     61.0  2.7e-16
       2x256  2            csk    replayed         332       1574    4.62     69.3  3.4e-16
       2x512  2            csk    replayed        1070       3016    2.75     47.9  4.5e-16
      2x1024  2            csk    replayed        2072       8235    3.45     54.4  2.7e-16
      2x2048  2            csk    replayed        4602      11252    2.34     53.4  2.7e-16
      2x4096  2            csk    replayed        9327      18396    1.60     57.1  3.9e-16
      2x8192  2            csk    replayed       20527      29115    1.32     55.9  4.2e-16
         4x2  2^2          csk+rb replayed          21        170    7.60      5.7  8.9e-17
         4x4  2^2          csk+rb replayed          22         19    0.87     14.8  2.0e-16
         4x8  2^2          csk+rb replayed          30        190    6.05     26.3  1.7e-16
        4x16  2^2          csk+rb replayed          41        224    5.39     47.2  2.8e-16
        4x32  2^2          csk+rb replayed          74        327    4.37     60.6  2.1e-16
        4x64  2^2          csk+rb replayed         146        514    3.47     70.2  2.9e-16
       4x128  2^2          csk    replayed         319        969    3.00     72.3  2.2e-16
       4x256  2^2          csk    replayed         894       1796    2.00     57.3  3.7e-16
       4x512  2^2          csk    replayed        2287       5993    2.56     49.3  4.6e-16
      4x1024  2^2          csk    replayed        3268       8081    2.17     75.2  3.7e-16
      4x2048  2^2          csk    replayed        5469      11115    1.88     97.4  3.6e-16
      4x4096  2^2          csk    replayed        9009      16475    1.62    127.3  3.1e-16
      4x8192  2^2          csk    replayed       18177      28552    1.46    135.2  3.6e-16
         8x2  2^3          csk+rb replayed          24        184    7.47     13.1  1.2e-16
         8x4  2^3          csk+rb replayed          27        194    7.12     29.7  1.3e-16
         8x8  2^3          csk+rb replayed          42         36    0.86     45.4  1.5e-16
        8x16  2^3          csk+rb replayed          70        256    3.64     63.8  2.8e-16
        8x32  2^3          csk+rb replayed         143        416    2.91     71.7  2.6e-16
        8x64  2^3          csk+rb replayed         302        699    2.27     76.3  4.0e-16
       8x128  2^3          csk    replayed         729       1437    1.81     70.2  2.8e-16
       8x256  2^3          csk    replayed        1827       2796    1.50     61.7  2.8e-16
       8x512  2^3          csk    replayed        2547       4080    1.37     96.5  4.1e-16
      8x1024  2^3          csk    replayed        3932       6143    1.43    135.4  3.8e-16
      8x2048  2^3          csk    replayed        6213      10270    1.59    184.6  3.8e-16
      8x4096  2^3          csk    replayed       11334      16490    1.44    216.8  4.1e-16
      8x8192  2^3          csk    replayed       16810      31886    1.80    311.9  5.2e-16
        16x2  2^4          csk+rb replayed          32        205    5.59     24.6  2.2e-16
        16x4  2^4          csk+rb replayed          42        222    5.26     46.2  2.5e-16
        16x8  2^4          csk+rb replayed          73        254    3.49     61.5  2.0e-16
       16x16  2^4          csk+rb replayed         142        145    1.02     71.9  2.5e-16
       16x32  2^4          csk+rb replayed         306        356    1.14     75.3  3.0e-16
       16x64  2^4          csk+rb replayed         674       1856    2.66     75.9  4.8e-16
      16x128  2^4          csk    replayed        1801       2013    1.11     62.5  3.1e-16
      16x256  2^4          csk    replayed        2600       3271    1.13     94.5  3.8e-16
      16x512  2^4          csk    replayed        4008       5352    1.18    132.9  4.2e-16
     16x1024  2^4          csk    replayed        6523       9886    1.25    175.8  2.7e-16
     16x2048  2^4          csk    replayed       11752      16465    1.30    209.1  3.7e-16
     16x4096  2^4          csk    replayed       15317      26310    1.46    342.3  2.9e-16
     16x8192  2^4          csk    replayed       30327      52856    1.45    367.4  4.2e-16
        32x2  2^5          chain+rb replayed          55        281    5.07     34.8  2.6e-16
        32x4  2^5          chain+rb replayed          84        328    3.90     53.3  2.0e-16
        32x8  2^5          chain+rb replayed         151        404    2.67     67.8  2.4e-16
       32x16  2^5          chain+rb replayed         314        340    1.08     73.3  2.3e-16
       32x32  2^5          chain+rb replayed         673        719    1.07     76.1  3.9e-16
       32x64  2^5          chain+rb replayed        1864       2313    1.21     60.4  3.0e-16
      32x128  2^5          chain  replayed        2789       3075    0.95     88.1  3.0e-16
      32x256  2^5          chain  replayed        4465       6195    1.29    119.3  3.3e-16
      32x512  2^5          chain  replayed        8000      10160    1.24    143.4  2.8e-16
     32x1024  2^5          chain  replayed       15482      18565    0.98    158.7  3.5e-16
     32x2048  2^5          chain  replayed       17903      33855    1.85    292.8  4.0e-16
     32x4096  2^5          chain  replayed       38707      64966    1.66    287.8  4.1e-16
     32x8192  2^5          chain  replayed       77013     151856    1.76    306.4  5.2e-16
        64x2  2^6          chain+rb replayed         109        421    3.84     41.0  3.4e-16
        64x4  2^6          chain+rb replayed         171        514    2.99     59.8  3.4e-16
        64x8  2^6          chain+rb replayed         325        689    2.12     71.0  3.2e-16
       64x16  2^6          chain+rb replayed         655       1837    2.54     78.2  3.2e-16
       64x32  2^6          chain+rb replayed        1851       2035    1.03     60.9  4.2e-16
       64x64  2^6          chain+rb replayed        2509       3835    1.38     98.0  3.1e-16
      64x128  2^6          chain  replayed        3965       4887    1.09    134.3  3.3e-16
      64x256  2^6          chain  replayed        7488      11879    1.41    153.2  3.4e-16
      64x512  2^6          chain  replayed       12075      19641    1.52    203.5  3.8e-16
     64x1024  2^6          chain  replayed       19100      34358    1.66    274.5  3.2e-16
     64x2048  2^6          chain  replayed       36067      46876    1.22    308.9  3.8e-16
     64x4096  2^6          chain  replayed       71662     118637    1.30    329.2  4.9e-16
     64x8192  2^6          chain  replayed      137025     223162    1.41    363.5  1.1e-15
       128x2  2^7          turn   replayed         210        743    3.49     48.7  4.3e-16
       128x4  2^7          turn   replayed         409        985    2.33     56.3  3.4e-16
       128x8  2^7          turn   replayed        1018       1515    1.46     50.3  3.2e-16
      128x16  2^7          turn   replayed        2449       2303    0.87     46.0  2.3e-16
      128x32  2^7          chain+rb replayed        3508       3191    0.80     70.1  4.0e-16
      128x64  2^7          chain+rb replayed        4234       4931    1.06    125.8  3.3e-16
     128x128  2^7          chain  replayed        7993      13320    1.66    143.5  3.4e-16
     128x256  2^7          chain  replayed       11403      15079    1.14    215.5  4.0e-16
     128x512  2^7          chain  replayed       20787      38328    1.66    252.2  3.2e-16
    128x1024  2^7          chain  replayed       38993      65726    1.38    285.7  3.4e-16
    128x2048  2^7          chain  replayed       76175     125437    1.31    309.7  6.2e-16
    128x4096  2^7          chain  replayed      148138     327894    2.08    336.2  1.3e-15
    128x8192  2^7          chain  replayed      315650     455319    1.34    332.2  1.3e-15
       256x2  2^8          turn   replayed         434       1357    3.12     53.1  3.3e-16
       256x4  2^8          turn   replayed        1022       2017    1.95     50.1  3.3e-16
       256x8  2^8          turn   replayed        2423       8712    3.26     46.5  3.6e-16
      256x16  2^8          turn   replayed        3403       5115    1.32     72.2  3.1e-16
      256x32  2^8          chain+rb replayed        5593       5747    0.99     95.2  3.7e-16
      256x64  2^8          chain+rb replayed        9617      14974    1.37    119.3  4.8e-16
     256x128  2^8          chain  replayed       16590      20528    1.05    148.1  3.8e-16
     256x256  2^8          chain  replayed       25990      37475    1.28    201.7  3.3e-16
     256x512  2^8          chain  replayed       45553      61303    1.22    244.6  4.4e-16
    256x1024  2^8          chain  replayed       80138     121512    1.32    294.4  5.9e-16
    256x2048  2^8          chain  replayed      147412     275843    1.29    337.9  1.3e-15
    256x4096  2^8          chain  replayed      355863     502200    1.30    294.7  1.0e-15
    256x8192  2^8          chain  replayed     1325525    1924844    1.20    166.1  1.5e-15
       512x2  2^9          turn   replayed        1002       2972    2.86     51.1  3.7e-16
       512x4  2^9          turn   replayed        2673       6117    2.06     42.1  4.3e-16
       512x8  2^9          turn   replayed        3162      11669    3.48     77.7  4.1e-16
      512x16  2^9          turn   replayed        4853       8805    1.67    109.7  3.1e-16
      512x32  2^9          chain+rb replayed        9107      14399    1.51    125.9  4.8e-16
      512x64  2^9          chain+rb replayed       17066      19886    1.07    144.0  4.0e-16
     512x128  2^9          chain  replayed       27350      33241    1.19    191.7  5.3e-16
     512x256  2^9          chain  replayed       44873      68553    1.42    248.3  3.4e-16
     512x512  2^9          chain  replayed       91125     175406    1.58    258.9  4.6e-16
    512x1024  2^9          chain  replayed      160013     323200    1.69    311.3  1.1e-15
    512x2048  2^9          chain  replayed      361775     505418    1.28    289.8  1.2e-15
    512x4096  2^9          chain  replayed     1439287    1612574    1.06    153.0  1.5e-15
    512x8192  2^9          chain  replayed     4389463    4961619    1.07    105.1  2.8e-15
      1024x2  2^10         turn   replayed        3599      12920    3.39     31.3  4.7e-16
      1024x4  2^10         turn   replayed        4195      11978    2.56     58.6  5.3e-16
      1024x8  2^10         turn   replayed        4914      18657    3.69    108.4  3.0e-16
     1024x16  2^10         turn   replayed        7588      24072    2.92    151.2  3.9e-16
     1024x32  2^10         chain+rb replayed       16495      27775    1.67    149.0  4.7e-16
     1024x64  2^10         chain+rb replayed       26083      33518    1.17    201.0  4.4e-16
    1024x128  2^10         chain  replayed       52987      81447    1.25    210.3  4.7e-16
    1024x256  2^10         chain  replayed      108938     250118    2.16    216.6  5.2e-16
    1024x512  2^10         chain  replayed      184850     377081    1.73    269.4  1.2e-15
   1024x1024  2^10         chain  replayed      405987     700750    1.58    258.3  1.3e-15
   1024x2048  2^10         chain  replayed     1630575    1751931    1.05    135.0  1.6e-15
   1024x4096  2^10         chain  replayed     4781937    5757319    1.09     96.5  2.6e-15
      2048x2  2^11         turn   replayed        5113      18349    3.14     48.1  2.6e-16
      2048x4  2^11         turn   replayed        5570      22483    3.59     95.6  3.4e-16
      2048x8  2^11         turn   replayed        7372      37832    4.85    155.6  3.7e-16
     2048x16  2^11         turn   replayed       14892      45994    2.95    165.0  5.2e-16
     2048x32  2^11         chain+rb replayed       29517      49628    1.59    177.6  3.9e-16
     2048x64  2^11         chain+rb replayed       59467      94566    1.58    187.4  4.7e-16
    2048x128  2^11         chain  replayed      118525     158019    1.24    199.1  5.2e-16
    2048x256  2^11         chain  replayed      212337     360750    1.67    234.6  1.2e-15
    2048x512  2^11         chain  replayed      487750     807075    1.45    215.0  1.4e-15
   2048x1024  2^11         chain  replayed     1768537    2045356    1.05    124.5  1.5e-15
   2048x2048  2^11         chain  replayed     4994013    5734400    1.12     92.4  2.9e-15
      4096x2  2^12         turn   replayed        8366      31358    2.76     63.6  4.9e-16
      4096x4  2^12         turn   replayed        9701      41876    4.03    118.2  3.7e-16
      4096x8  2^12         turn   replayed       12872      60252    4.39    190.9  3.9e-16
     4096x16  2^12         turn   replayed       26237      81933    3.03    199.8  3.7e-16
     4096x32  2^12         chain+rb replayed       50680      92353    1.68    219.8  4.1e-16
     4096x64  2^12         chain+rb replayed      106563     188375    1.37    221.4  5.3e-16
    4096x128  2^12         chain  replayed      210287     355293    1.32    236.9  1.3e-15
    4096x256  2^12         chain  replayed      464237     625787    1.28    225.9  1.3e-15
    4096x512  2^12         chain  replayed     1536937    1993750    1.24    143.3  1.7e-15
   4096x1024  2^12         chain  replayed     6438013    6818356    1.03     71.7  3.0e-15
      8192x2  2^13         turn   replayed       16258      50529    2.57     70.5  3.6e-16
      8192x4  2^13         turn   replayed       18362      70572    3.45    133.8  3.6e-16
      8192x8  2^13         turn   replayed       21170     130343    5.96    247.7  4.1e-16
     8192x16  2^13         turn   replayed       47133     138303    2.77    236.4  3.1e-16
     8192x32  2^13         chain+rb replayed      115438     221506    1.73    204.4  4.9e-16
     8192x64  2^13         chain+rb replayed      235075     339056    1.22    211.9  1.0e-15
    8192x128  2^13         chain  replayed      503612     677238    1.25    208.2  9.3e-16
    8192x256  2^13         chain  replayed     1664188    2334100    1.34    132.3  1.2e-15
    8192x512  2^13         chain  replayed     6267000    7230619    1.13     73.6  2.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           53      0      2   1.05   1.29   1.69    1.33
 csk                             28      0      0   1.18   1.59   3.45    1.81
 turn                            28      0      1   1.46   2.99   4.39    2.79
 chain+rb                        26      1      2   1.03   1.45   3.84    1.62
 csk+rb                          24      1      3   0.87   4.81   7.60    3.51
 ALL                            159      2      8   1.06   1.59   4.85    1.92
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159      2      8   1.06   1.59   4.85    1.92
 ALL                            159      2      8   1.06   1.59   4.85    1.92
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      0      2   1.07   1.55   3.69    1.75
 > 65536 points                  45      0      0   1.07   1.32   1.73    1.38
 <= 256 points                   28      1      3   0.87   4.67   7.60    3.81
 1025..4096                      21      1      3   0.95   1.50   3.39    1.74
 257..1024                       17      0      0   1.08   2.27   3.12    2.12
 ALL                            159      2      8   1.06   1.59   4.85    1.92
```


worst 10: 2x2 (csk+rb 0.67), 128x32 (chain+rb 0.80), 8x8 (csk+rb 0.86), 128x16 (turn 0.87), 4x4 (csk+rb 0.87), 32x128 (chain 0.95), 32x1024 (chain 0.98), 256x32 (chain+rb 0.99), 16x16 (csk+rb 1.02), 4096x1024 (chain 1.03)
best 5: 2x4 (csk+rb 8.20), 2x16 (csk+rb 7.85), 4x2 (csk+rb 7.60), 8x2 (csk+rb 7.47), 8x4 (csk+rb 7.12)
