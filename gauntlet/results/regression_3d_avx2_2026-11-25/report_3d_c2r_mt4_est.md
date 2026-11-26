# gauntlet report (3D)

run: `regression_3d_avx2_2026-11-25`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_mt4_est  cells: 85 listed, 85 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.197..1.230

threaded plans that ran serial (engaged = 0 at both flips): 10: 10x8x8, 17x5x7, 18x6x6, 2x16x16, 32x4x4, 5x9x15, 65x3x3, 6x10x14, 8x8x8, 9x8x8


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      2x16x16  2            child  -                230       3370   13.68     50.0  4.5e-16
      4x32x32  2^2          plane+child+strips -               1946       5995    3.04     63.1  5.6e-16
       5x9x15  5            child  -               1277       2474    1.85     12.4  7.6e-16
      6x10x14  2.3          child  -                986       2498    2.45     20.7  6.7e-16
        8x8x8  2^3          child  -                242        547    2.25     47.7  4.5e-16
      8x64x64  2^3          plane+child+strips -              12120      31026    2.38    101.4  8.9e-16
        9x8x8  3^2          child  -                289       2352    7.65     45.7  6.2e-16
      9x17x31  3^2          plane+child+strips -               6218      11836    1.85     23.3  1.5e-15
       10x8x8  2.5          child  -                312       2418    7.65     47.8  5.3e-16
     10x18x30  2.5          plane+child+strips -               4689       6025    1.26     35.7  1.1e-15
     16x16x16  2^4          plane+child+strips -               2102       3064    1.41     58.5  6.7e-16
   16x128x128  2^4          plane+child -              69993     105860    1.50    168.5  1.2e-15
       17x5x7  17           child  -               1521       2628    1.68      9.0  8.9e-16
     17x16x16  17           plane+child+strips -               2492       4963    1.93     52.8  1.0e-15
     17x35x63  17           plane+child+strips -              24536      54788    2.20     58.0  1.5e-15
       18x6x6  2.3^2        child  -                468       2635    5.42     32.3  7.0e-16
     18x16x16  2.3^2        plane+child+strips -               2508       3575    1.42     55.9  8.9e-16
     18x34x62  2.3^2        plane+child+strips -              27753      50931    1.80     52.0  1.6e-15
       32x4x4  2^5          child  -                419       3008    6.93     27.5  4.4e-16
     32x32x32  2^5          plane+child -              12028      17828    1.45    102.2  7.8e-16
   32x256x256  2^5          plane+child -             839563    1022975    1.19    131.1  1.2e-15
     34x10x14  2.17         plane+child+strips -               3951       6571    1.51     36.8  1.1e-15
     34x32x32  2.17         plane+child+strips -              16075      27188    1.68     81.7  1.3e-15
    34x70x126  2.17         plane+child+strips -             134054     205330    1.50    101.8  1.7e-15
      35x9x15  5.7          plane+child+strips -               5394       6464    1.09     26.7  1.2e-15
     35x32x32  5.7          plane+child+strips -              15004      20697    1.32     90.4  9.8e-16
    35x65x125  5.7          plane+child+strips -             163779     185939    1.12     78.6  1.7e-15
     53x32x32  53           plane+child -              24382      81620    3.31     87.5  1.6e-15
    53x67x127  53           plane+child -             587350    1885850    3.17     36.1  3.0e-15
       64x8x8  2^6          plane+child -               3144       7146    2.25     39.1  6.7e-16
     64x64x64  2^6          plane+child -              87667      99453    1.11    134.6  1.1e-15
       65x3x3  5.13         child  -               2135       4698    2.18      6.3  1.0e-15
     65x17x31  5.13         plane+child+strips -              35295      62203    1.74     36.6  1.9e-15
     65x64x64  5.13         plane+child+strips -              99687     109556    1.07    120.3  1.5e-15
   65x135x255  5.13         plane+child+strips -            1600975    2797156    1.68     73.7  2.4e-15
   65x259x507  5.13         plane+child+strips -           10122813   16685856    1.62     48.5  2.6e-15
     67x64x64  67           plane+child -             123100     480850    3.88    100.7  1.6e-15
   67x131x251  67           plane+child -            3625562    9045875    2.44     32.0  3.4e-15
   67x257x509  67           plane+child -           20001687   40082281    2.00     25.3  2.8e-15
     70x18x30  2.5.7        plane+child+strips -              30438      26699    0.83     47.2  1.3e-15
     70x64x64  2.5.7        plane+child+strips -             109462     115573    1.05    118.7  1.2e-15
   70x130x250  2.5.7        plane+child+strips -            1343862    1222250    0.80     89.4  2.0e-15
   70x270x510  2.5.7        plane+child+strips -            7917500   11920481    1.50     70.6  2.5e-15
    106x64x64  2.53         plane+child+strips -             244767     494478    1.96     83.1  2.0e-15
  106x134x254  2.53         plane+child+strips -            4589562   10900168    2.34     42.8  3.2e-15
  106x262x511  2.53         plane+child+strips -           27718525  108591106    3.88     30.4  3.1e-15
    128x16x16  2^7          plane+child+strips -              15765      19905    1.20     77.9  1.0e-15
  128x128x128  2^7          plane+child+strips -             964938    1099937    1.11    114.1  1.6e-15
  128x256x256  2^7          plane+child+strips -            6219988    7658762    1.23     77.5  1.4e-15
      130x6x6  2.5.13       plane+child+strips -               7313       9219    1.24     19.5  1.4e-15
    130x34x62  2.5.13       plane+child+strips -             185200     291193    1.57     66.8  2.2e-15
  130x128x128  2.5.13       plane+child+strips -            1098462    1162075    1.03    101.9  2.1e-15
    131x53x61  131          plane+child -             536022    1727905    3.08     36.9  4.2e-15
  131x128x128  131          plane+child -            1396575    3057331    2.16     80.8  2.3e-15
  134x128x128  2.67         plane+child+strips -            1456550    3386531    2.31     79.4  1.8e-15
      135x5x7  3^3.5        plane+child+strips -              10477       8863    0.80     13.8  1.2e-15
    135x35x63  3^3.5        plane+child+strips -             163269     206535    1.24     82.9  1.7e-15
  135x128x128  3^3.5        plane+child+strips -            1060100    1211369    1.06    109.9  1.7e-15
    256x32x32  2^8          plane+child+strips -              98447     133186    1.34    119.8  1.0e-15
   257x67x127  257          plane+child -            3162475    9165750    2.80     36.4  2.7e-15
  257x131x251  257          plane+child -           15427600   33039356    2.13     31.5  3.2e-15
     259x9x15  7.37         plane+child+strips -              41074      69138    1.63     32.1  1.4e-15
   259x65x125  7.37         plane+child+strips -            1623250    2942556    1.79     68.1  2.3e-15
  259x135x255  7.37         plane+child+strips -            8932287   18509725    2.06     57.6  2.7e-15
  262x106x122  2.131        plane+child+strips -            4237200   16585699    3.89     43.4  2.9e-15
  262x134x254  2.131        plane+child+strips -           12459137   40388993    3.21     41.3  3.3e-15
    270x10x14  2.3^3.5      plane+child+strips -              34298      42566    1.20     41.9  1.3e-15
   270x70x126  2.3^3.5      plane+child+strips -            1413050    1402612    0.94     89.2  2.0e-15
  270x130x250  2.3^3.5      plane+child+strips -            7030788    6906725    0.97     72.0  2.5e-15
    512x64x64  2^9          plane+child+strips -            1084812    1272819    1.12    101.5  1.3e-15
  512x128x128  2^9          plane+child+strips -            6748925    7889175    1.16     71.5  1.4e-15
  514x128x128  2.257        plane+child+strips -            8633088   15761212    1.81     56.1  2.0e-15
    516x18x30  2^2.3.43     plane+child+strips -             229107     371361    1.54     55.0  2.1e-15
  516x128x128  2^2.3.43     plane+child+strips -            7931350   11809856    1.47     61.3  1.9e-15
  521x128x128  521          plane+child -            8568050   15941350    1.85     57.3  2.7e-15
    525x17x31  3.5^2.7      plane+child+strips -             275164     501621    1.78     45.4  2.1e-15
  525x128x128  3.5^2.7      plane+child+strips -            7494900    7502287    1.00     66.1  2.0e-15
 1027x106x122  13.79        plane+child+strips -           18789238   59781062    3.06     41.8  3.4e-15
   1029x35x63  3.7^3        plane+child+strips -            1771675    2114606    1.08     67.6  1.9e-15
  1029x65x125  3.7^3        plane+child+strips -            8041288    9253787    1.11     59.8  2.5e-15
   1031x53x61  1031         plane+child -            6937575   14953825    2.14     26.0  5.2e-15
  1031x67x127  1031         plane+child -           18967587   39760844    2.06     26.7  4.2e-15
   1200x34x62  2^4.3.5^2    plane+child+strips -            2289625    3094506    1.14     58.8  2.3e-15
  1200x70x126  2^4.3.5^2    plane+child+strips -           10195537    9972406    0.97     60.6  2.1e-15
   2048x64x64  2^11         plane+child+strips -            8187875    8423619    1.02     58.9  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child+strips              58      1      7   0.97   1.41   2.38    1.47
 plane+child                     17      0      0   1.19   2.14   3.31    2.14
 child                           10      0      0   1.85   3.94  13.68    4.03
 ALL                             85      1      7   1.03   1.68   3.31    1.78
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     30      0      5   0.97   1.50   3.89    1.69
 odd column                      21      1      2   1.06   1.62   2.18    1.55
 pow2 column                     18      0      0   1.11   1.37   6.93    1.80
 prime column                    16      0      0   1.85   2.15   3.31    2.35
 ALL                             85      1      7   1.03   1.68   3.31    1.78
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  52      0      5   1.02   1.52   3.08    1.61
 4097..65536                     20      1      2   1.09   1.48   2.38    1.50
 257..1024                       10      0      0   1.85   3.94  13.68    4.03
 1025..4096                       3      0      0   1.41   2.25   3.04    2.13
 ALL                             85      1      7   1.03   1.68   3.31    1.78
```


worst 10: 135x5x7 (plane+child+strips 0.80), 70x130x250 (plane+child+strips 0.80), 70x18x30 (plane+child+strips 0.83), 270x70x126 (plane+child+strips 0.94), 270x130x250 (plane+child+strips 0.97), 1200x70x126 (plane+child+strips 0.97), 525x128x128 (plane+child+strips 1.00), 2048x64x64 (plane+child+strips 1.02), 130x128x128 (plane+child+strips 1.03), 70x64x64 (plane+child+strips 1.05)
best 5: 2x16x16 (child 13.68), 9x8x8 (child 7.65), 10x8x8 (child 7.65), 32x4x4 (child 6.93), 18x6x6 (child 5.42)
