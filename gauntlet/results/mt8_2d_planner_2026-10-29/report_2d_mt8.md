# gauntlet report (2D)

run: `mt8_2d_planner_2026-10-29`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 0.880..1.315

threaded plans that ran serial (engaged = 0 at both flips): 55: 1024x2, 128x16, 128x2, 128x4, 128x8, 16x16, 16x2, 16x32, 16x4, 16x64, 16x8, 256x2, 256x4, 2x1024, 2x128, 2x16, 2x2, 2x2048, 2x256, 2x32, 2x4, 2x4096, 2x512, 2x64, 2x8, 2x8192, 32x16, 32x2, 32x32, 32x4, 32x64, 32x8, 4x128, 4x16, 4x2, 4x256, 4x32, 4x4, 4x512, 4x64 ...


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            csk+rb replayed          20         13    0.66      2.0  1.4e-16
         2x4  2            csk+rb replayed          21        164    7.62      5.6  1.7e-16
         2x8  2            csk+rb replayed          25        174    6.83     12.7  1.9e-16
        2x16  2            csk+rb2 replayed          35        212    5.92     22.7  2.0e-16
        2x32  2            csk+rb2 replayed          50        311    6.10     38.3  2.0e-16
        2x64  2            csk    replayed          87        471    5.38     51.4  2.0e-16
       2x128  2            csk    replayed         167        833    4.97     61.2  2.7e-16
       2x256  2            csk    replayed         330       1530    4.63     69.7  3.4e-16
       2x512  2            chain  replayed         981       2950    2.92     52.2  4.5e-16
      2x1024  2            csk    replayed        2202       7913    3.28     51.2  4.0e-16
      2x2048  2            csk    replayed        4558      11103    2.07     53.9  2.9e-16
      2x4096  2            csk    replayed        9743      16145    1.44     54.7  3.9e-16
      2x8192  2            csk    replayed       19993      27262    1.29     57.4  3.4e-16
         4x2  2^2          csk+rb replayed          21        163    7.72      5.8  8.9e-17
         4x4  2^2          csk+rb replayed          24         19    0.79     13.6  2.0e-16
         4x8  2^2          csk+rb replayed          30        183    5.90     26.9  1.7e-16
        4x16  2^2          csk+rb2 replayed          48        218    4.51     40.2  2.8e-16
        4x32  2^2          csk+rb2 replayed          80        319    3.98     55.9  1.7e-16
        4x64  2^2          csk+rb2 replayed         160        510    3.18     63.9  2.6e-16
       4x128  2^2          csk    replayed         327        952    2.91     70.5  2.2e-16
       4x256  2^2          chain  replayed         753       1794    2.29     68.0  3.7e-16
       4x512  2^2          chain  replayed        2024       6127    2.67     55.7  4.6e-16
      4x1024  2^2          chain  replayed        4682       7882    1.60     52.5  3.7e-16
      4x2048  2^2          csk    replayed        5020      10736    1.71    106.1  3.4e-16
      4x4096  2^2          csk    replayed        9030      15806    1.39    127.0  3.1e-16
      4x8192  2^2          csk    replayed       17939      26453    1.21    137.0  3.6e-16
         8x2  2^3          csk+rb replayed          24        176    7.30     13.3  1.2e-16
         8x4  2^3          csk+rb replayed          30        189    6.40     27.1  1.6e-16
         8x8  2^3          csk+rb replayed          42         36    0.85     45.7  1.5e-16
        8x16  2^3          csk+rb2 replayed          80        253    3.14     56.3  1.9e-16
        8x32  2^3          csk+rb2 replayed         152        411    2.69     67.4  2.6e-16
        8x64  2^3          csk+rb2 replayed         327        691    2.11     70.6  2.7e-16
       8x128  2^3          chain  replayed         712       1404    1.95     71.9  2.8e-16
       8x256  2^3          csk    replayed        1852       2662    1.32     60.8  2.8e-16
       8x512  2^3          csk    replayed        2801       3225    1.02     87.8  4.1e-16
      8x1024  2^3          csk    replayed        4514       5499    1.11    118.0  3.8e-16
      8x2048  2^3          csk    replayed        5991       8555    1.35    191.4  3.8e-16
      8x4096  2^3          csk    replayed       11487      13974    1.20    213.9  4.1e-16
      8x8192  2^3          chain  replayed       95720      19945    0.20     54.8  5.2e-16
        16x2  2^4          csk+rb replayed          32        203    6.32     24.9  2.2e-16
        16x4  2^4          csk+rb replayed          45        220    4.89     42.7  2.5e-16
        16x8  2^4          csk+rb replayed          75        252    3.33     59.5  2.0e-16
       16x16  2^4          csk+rb2 replayed         170        145    0.85     60.3  2.5e-16
       16x32  2^4          csk+rb2 replayed         325        349    1.07     70.9  3.0e-16
       16x64  2^4          chain+rb2 replayed         741       1794    2.17     69.1  2.7e-16
      16x128  2^4          chain  replayed        1919       1946    0.97     58.7  3.1e-16
      16x256  2^4          csk    replayed        2775       2977    1.03     88.6  3.8e-16
      16x512  2^4          csk    replayed        4114       4734    1.13    129.4  4.2e-16
     16x1024  2^4          csk    replayed        6752       8131    1.16    169.8  3.7e-16
     16x2048  2^4          csk    replayed       12790      15982    1.23    192.1  3.7e-16
     16x4096  2^4          csk    replayed       15707      21960    1.37    333.8  2.9e-16
     16x8192  2^4          csk    replayed       31100      43850    1.39    358.2  4.2e-16
        32x2  2^5          chain+rb replayed          58        279    4.73     32.9  1.3e-16
        32x4  2^5          chain+rb replayed          90        325    3.56     49.8  2.0e-16
        32x8  2^5          chain+rb replayed         160        405    2.53     64.1  2.4e-16
       32x16  2^5          chain+rb2 replayed         356        339    0.95     64.7  2.3e-16
       32x32  2^5          chain+rb2 replayed         721        753    0.99     71.1  2.4e-16
       32x64  2^5          chain+rb2 replayed        2553       2164    0.83     44.1  2.8e-16
      32x128  2^5          csk    replayed        2392       2986    1.24    102.7  3.5e-16
      32x256  2^5          csk    replayed        4030       5152    1.26    132.1  4.6e-16
      32x512  2^5          csk    replayed        7091       9078    1.20    161.7  3.7e-16
     32x1024  2^5          csk    replayed       12384      17360    1.37    198.5  3.5e-16
     32x2048  2^5          csk    replayed       16333      23135    1.39    321.0  4.4e-16
     32x4096  2^5          chain  replayed       38540      44753    1.15    289.1  4.1e-16
     32x8192  2^5          chain  replayed       83637      93319    1.07    282.1  5.2e-16
        64x2  2^6          chain+rb replayed         115        418    3.61     38.9  3.4e-16
        64x4  2^6          chain+rb replayed         187        512    2.48     54.8  3.4e-16
        64x8  2^6          chain+rb replayed         347        689    1.97     66.3  3.2e-16
       64x16  2^6          chain+rb replayed         743       1697    1.90     68.9  3.2e-16
       64x32  2^6          chain+rb2 replayed        1915       1892    0.94     58.8  3.0e-16
       64x64  2^6          chain+rb2 replayed        2772       3667    1.05     88.7  3.8e-16
      64x128  2^6          csk    replayed        3943       4808    1.15    135.0  5.2e-16
      64x256  2^6          csk+rb2 replayed        8205      10904    1.25    139.8  5.1e-16
      64x512  2^6          csk    replayed       12182      18726    1.50    201.7  4.4e-16
     64x1024  2^6          chain  replayed       18153      27735    1.42    288.8  3.2e-16
     64x2048  2^6          chain  replayed       35593      46616    1.28    313.0  3.8e-16
     64x4096  2^6          chain  replayed       71212      91669    1.27    331.3  4.9e-16
     64x8192  2^6          chain  replayed      137387     229637    1.42    362.5  1.1e-15
       128x2  2^7          turn   replayed         211        728    3.45     48.5  4.3e-16
       128x4  2^7          chain+rb replayed         591        966    1.63     39.0  3.4e-16
       128x8  2^7          chain+rb replayed        1135       1525    1.14     45.1  2.7e-16
      128x16  2^7          turn   replayed        2366       2356    0.91     47.6  2.3e-16
      128x32  2^7          chain+rb2 replayed        3156       2995    0.92     77.9  3.1e-16
      128x64  2^7          chain+rb2 replayed        4390       4893    1.09    121.3  2.6e-16
     128x128  2^7          chain  replayed        8401      12109    1.32    136.5  3.8e-16
     128x256  2^7          chain  replayed       11815      14751    1.03    208.0  4.0e-16
     128x512  2^7          chain  replayed       20943      26782    1.25    250.3  3.2e-16
    128x1024  2^7          chain  replayed       41100      49670    1.21    271.1  3.4e-16
    128x2048  2^7          chain  replayed       72550      94831    1.24    325.2  6.2e-16
    128x4096  2^7          chain  replayed      154737     218287    1.34    321.9  1.3e-15
    128x8192  2^7          chain  replayed      311975     400350    1.22    336.1  1.1e-15
       256x2  2^8          turn   replayed         431       1354    3.13     53.4  3.3e-16
       256x4  2^8          turn   replayed        1067       1992    1.84     48.0  3.3e-16
       256x8  2^8          chain+rb replayed        1870       7701    3.90     60.2  4.4e-16
      256x16  2^8          turn   replayed        3392       4947    1.35     72.5  3.1e-16
      256x32  2^8          chain+rb2 replayed        4782       5291    1.08    111.3  2.6e-16
      256x64  2^8          chain+rb replayed        8246      13676    1.19    139.1  4.8e-16
     256x128  2^8          chain  replayed       14380      17709    1.18    170.9  3.8e-16
     256x256  2^8          chain  replayed       22740      32545    1.33    230.6  3.3e-16
     256x512  2^8          chain  replayed       56713      58083    1.00    196.4  4.4e-16
    256x1024  2^8          chain  replayed       81362     133131    1.59    290.0  5.9e-16
    256x2048  2^8          chain  replayed      158375     235931    1.41    314.5  1.3e-15
    256x4096  2^8          chain  replayed      333812     479425    1.43    314.1  9.1e-16
    256x8192  2^8          chain  replayed     1175950    1375637    1.15    187.3  1.5e-15
       512x2  2^9          turn   replayed         986       2903    2.90     51.9  3.7e-16
       512x4  2^9          turn   replayed        2419       5961    2.22     46.6  4.3e-16
       512x8  2^9          turn   replayed        3327      11658    3.10     73.9  4.1e-16
      512x16  2^9          turn   replayed        4976       8433    1.43    107.0  3.1e-16
      512x32  2^9          chain+rb2 replayed        8975      14156    1.41    127.8  3.3e-16
      512x64  2^9          chain+rb2 replayed       13874      18366    1.16    177.1  3.4e-16
     512x128  2^9          chain  replayed       27333      32753    1.19    191.8  3.5e-16
     512x256  2^9          chain  replayed       54193      66986    1.21    205.6  3.4e-16
     512x512  2^9          chain  replayed       87450     140418    1.59    269.8  4.6e-16
    512x1024  2^9          chain  replayed      159788     261206    1.60    311.7  1.1e-15
    512x2048  2^9          chain  replayed      352700     492919    1.38    297.3  1.2e-15
    512x4096  2^9          chain  replayed     1253850    1550969    1.17    175.6  1.5e-15
    512x8192  2^9          chain  replayed     4240113    4762094    1.11    108.8  2.8e-15
      1024x2  2^10         turn   replayed        2639       6432    2.23     42.7  4.7e-16
      1024x4  2^10         turn   replayed        3982      12503    2.94     61.7  4.2e-16
      1024x8  2^10         turn   replayed        5206      19149    3.06    102.3  3.0e-16
     1024x16  2^10         turn   replayed        7511      23546    2.71    152.7  3.9e-16
     1024x32  2^10         chain+rb2 replayed       15462      27980    1.75    158.9  4.0e-16
     1024x64  2^10         chain+rb2 replayed       27743      32472    1.16    189.0  3.5e-16
    1024x128  2^10         chain  replayed       52313      69406    1.31    213.0  4.7e-16
    1024x256  2^10         chain  replayed      105187     162206    1.47    224.3  5.2e-16
    1024x512  2^10         chain  replayed      215338     325606    1.43    231.3  1.2e-15
   1024x1024  2^10         chain  replayed      387175     575168    1.43    270.8  1.3e-15
   1024x2048  2^10         chain  replayed     1482975    1610406    1.06    148.5  1.2e-15
   1024x4096  2^10         chain  replayed     7008112    5069212    0.71     65.8  2.8e-15
      2048x2  2^11         turn   replayed        5371      14038    2.58     45.8  2.6e-16
      2048x4  2^11         turn   replayed        5556      26118    4.13     95.8  3.4e-16
      2048x8  2^11         turn   replayed        7407      38109    4.90    154.8  3.7e-16
     2048x16  2^11         turn   replayed       14370      47463    3.16    171.0  3.2e-16
     2048x32  2^11         turn   replayed       30197      45355    1.43    173.6  3.9e-16
     2048x64  2^11         chain+rb2 replayed       52747      70917    1.33    211.2  4.7e-16
    2048x128  2^11         chain  replayed       98763     160925    1.61    238.9  5.4e-16
    2048x256  2^11         chain+rb2 replayed      199388     332156    1.61    249.8  1.2e-15
    2048x512  2^11         chain  replayed      598113     688137    1.12    175.3  1.4e-15
   2048x1024  2^11         chain  replayed     1650087    1881106    1.06    133.4  1.5e-15
   2048x2048  2^11         chain  replayed     4674900    6097950    1.12     98.7  2.9e-15
      4096x2  2^12         turn   replayed        8815      24636    2.60     60.4  4.9e-16
      4096x4  2^12         turn   replayed        9861      47700    4.57    116.3  3.7e-16
      4096x8  2^12         turn   replayed       12293      62401    4.43    199.9  3.9e-16
     4096x16  2^12         turn   replayed       25090      73875    2.64    209.0  2.9e-16
     4096x32  2^12         turn   replayed       55960      83033    1.43    199.1  4.7e-16
     4096x64  2^12         chain+rb2 replayed      124712     146118    1.17    189.2  4.6e-16
    4096x128  2^12         chain  replayed      203975     278719    1.35    244.2  1.3e-15
    4096x256  2^12         chain+rb2 replayed      517600     590325    1.06    202.6  1.3e-15
    4096x512  2^12         chain  replayed     2051463    1844219    0.89    107.3  1.2e-15
   4096x1024  2^12         chain  replayed     5339438    5722975    1.06     86.4  2.7e-15
      8192x2  2^13         turn   replayed       16327      44297    2.18     70.2  3.6e-16
      8192x4  2^13         turn   replayed       18616      80553    3.61    132.0  3.6e-16
      8192x8  2^13         turn   replayed       20807     111210    5.02    252.0  4.1e-16
     8192x16  2^13         turn   replayed       47267     141486    2.87    235.7  3.1e-16
     8192x32  2^13         chain+rb2 replayed      111638     159544    1.40    211.3  4.5e-16
     8192x64  2^13         chain+rb2 replayed      219900     300624    1.34    226.5  1.2e-15
    8192x128  2^13         chain  replayed      541125     643787    1.17    193.8  1.3e-15
    8192x256  2^13         chain  replayed     1563563    2086844    1.31    140.8  1.2e-15
    8192x512  2^13         chain  replayed     5791900    6828731    1.16     79.7  2.6e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           50      2      4   1.03   1.26   1.61    1.26
 csk                             29      0      0   1.11   1.35   4.63    1.58
 turn                            27      0      1   1.43   2.87   4.57    2.62
 chain+rb2                       19      0      5   0.92   1.16   1.75    1.20
 csk+rb                          12      2      3   0.79   6.11   7.62    3.62
 csk+rb2                         11      0      1   1.07   3.14   5.92    2.63
 chain+rb                        11      0      0   1.19   2.48   3.90    2.35
 ALL                            159      4     14   1.02   1.41   4.63    1.76
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159      4     14   1.02   1.41   4.63    1.76
 ALL                            159      4     14   1.02   1.41   4.63    1.76
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      1      1   1.11   1.36   4.13    1.58
 > 65536 points                  45      1      2   1.06   1.28   1.59    1.28
 <= 256 points                   28      2      4   0.85   4.24   7.30    3.55
 1025..4096                      21      0      5   0.92   1.35   3.10    1.60
 257..1024                       17      0      2   0.99   1.97   3.13    1.96
 ALL                            159      4     14   1.02   1.41   4.63    1.76
```


worst 10: 8x8192 (chain 0.20), 2x2 (csk+rb 0.66), 1024x4096 (chain 0.71), 4x4 (csk+rb 0.79), 32x64 (chain+rb2 0.83), 8x8 (csk+rb 0.85), 16x16 (csk+rb2 0.85), 4096x512 (chain 0.89), 128x16 (turn 0.91), 128x32 (chain+rb2 0.92)
best 5: 4x2 (csk+rb 7.72), 2x4 (csk+rb 7.62), 8x2 (csk+rb 7.30), 2x8 (csk+rb 6.83), 8x4 (csk+rb 6.40)
