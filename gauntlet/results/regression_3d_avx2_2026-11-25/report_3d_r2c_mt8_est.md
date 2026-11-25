# gauntlet report (3D)

run: `regression_3d_avx2_2026-11-25`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_mt8_est  cells: 85 listed, 85 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.084..1.818

threaded plans that ran serial (engaged = 0 at both flips): 11: 10x8x8, 17x5x7, 18x6x6, 2x16x16, 32x4x4, 4x32x32, 5x9x15, 65x3x3, 6x10x14, 8x8x8, 9x8x8


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      2x16x16  2            child  -                224       6162   26.66     51.3  2.7e-16
      4x32x32  2^2          child  -               2408       7186    2.97     51.0  2.6e-16
       5x9x15  5            child  -                914       4497    4.89     17.3  5.0e-16
      6x10x14  2.3          child  -                940       4151    4.35     21.7  3.4e-16
        8x8x8  2^3          child  -                259        325    1.25     44.5  2.7e-16
      8x64x64  2^3          plane+child+strips -              12257      14519    1.07    100.2  3.4e-16
        9x8x8  3^2          child  -                314       2528    7.79     42.0  2.6e-16
      9x17x31  3^2          plane+child+strips -               5097       6346    1.20     28.4  6.9e-16
       10x8x8  2.5          child  -                331       2588    7.47     45.0  3.7e-16
     10x18x30  2.5          plane+child+strips -               3894       3522    0.85     43.0  5.0e-16
     16x16x16  2^4          plane+child+strips -               2420       2969    1.17     50.8  1.9e-16
   16x128x128  2^4          plane+child -              43387      77386    1.78    271.9  4.0e-16
       17x5x7  17           child  -                832       4787    5.58     16.5  4.5e-16
     17x16x16  17           plane+child+strips -               2877       4278    1.46     45.7  5.4e-16
     17x35x63  17           plane+child+strips -              19152      20750    1.07     74.3  5.9e-16
       18x6x6  2.3^2        child  -                486       4335    8.41     31.1  4.5e-16
     18x16x16  2.3^2        plane+payonce+stages -               3224       3528    1.06     43.5  4.3e-16
     18x34x62  2.3^2        plane+payonce+stages -              16050      30641    1.89     89.9  7.9e-16
       32x4x4  2^5          child  -                414       5958   13.21     27.8  2.4e-16
     32x32x32  2^5          plane+child -              14449      10545    0.69     85.0  3.4e-16
   32x256x256  2^5          plane+child -             432413     461856    1.05    254.6  4.6e-16
     34x10x14  2.17         plane+payonce+stages -               3210       4523    1.29     45.3  5.0e-16
     34x32x32  2.17         plane+payonce+stages -              12632      17154    1.25    104.0  5.7e-16
    34x70x126  2.17         plane+payonce+stages -              77377     107996    1.34    176.3  7.4e-16
      35x9x15  5.7          plane+payonce+stages -               3766       3040    0.78     38.3  5.6e-16
     35x32x32  5.7          plane+payonce+stages -              14225      12294    0.85     95.3  4.4e-16
    35x65x125  5.7          plane+payonce+stages -              86786      86564    0.99    148.4  6.7e-16
     53x32x32  53           plane+child -              21142      50345    2.31    100.9  7.5e-16
    53x67x127  53           plane+child -             269250     892593    3.19     78.7  1.2e-15
       64x8x8  2^6          plane+child -               4321       7015    1.48     28.4  2.5e-16
     64x64x64  2^6          plane+child -              47740      67850    1.09    247.1  4.2e-16
       65x3x3  5.13         child  -               1258       7445    5.88     10.7  5.4e-16
     65x17x31  5.13         plane+payonce+stages -              19620      32146    1.47     65.8  8.6e-16
     65x64x64  5.13         plane+payonce+stages -              54520      57436    1.01    220.0  6.2e-16
   65x135x255  5.13         plane+child+strips -             700850    1222569    1.65    168.4  9.8e-16
   65x259x507  5.13         plane+child+strips -            5074962    8189194    1.58     96.8  9.7e-16
     67x64x64  67           plane+child -              70279     243618    3.45    176.4  6.4e-16
   67x131x251  67           plane+child -            1457725    4464043    2.82     79.6  1.2e-15
   67x257x509  67           plane+child -            7971950   21019194    2.62     63.4  1.1e-15
     70x18x30  2.5.7        plane+payonce+stages -              15341      16615    1.06     93.7  5.8e-16
     70x64x64  2.5.7        plane+payonce+stages -              58808      73015    0.98    221.0  4.9e-16
   70x130x250  2.5.7        plane+child+strips -             624687     558206    0.85    192.3  7.8e-16
   70x270x510  2.5.7        plane+child+strips -            5739150    6161625    1.06     97.4  7.8e-16
    106x64x64  2.53         plane+child+strips -             162078     246533    1.43    125.4  7.1e-16
  106x134x254  2.53         plane+child+strips -            2149187    5265025    2.31     91.4  9.6e-16
  106x262x511  2.53         plane+child+strips -           15373338   56286781    3.61     54.8  9.7e-16
    128x16x16  2^7          plane+payonce+stages -              11011      15012    1.27    111.6  3.4e-16
  128x128x128  2^7          plane+child+strips -             411037     432969    1.03    267.9  4.4e-16
  128x256x256  2^7          plane+child+strips -            4248463    4261456    1.00    113.5  4.5e-16
      130x6x6  2.5.13       plane+payonce+stages -               3513       9719    2.65     40.6  6.1e-16
    130x34x62  2.5.13       plane+payonce+stages -              85707     133160    1.54    144.4  8.4e-16
  130x128x128  2.5.13       plane+child+strips -             458700     464662    1.00    244.0  6.8e-16
    131x53x61  131          plane+child -             283011     807033    2.84     69.9  1.5e-15
  131x128x128  131          plane+child -             549800    1383337    2.47    205.3  7.4e-16
  134x128x128  2.67         plane+child+strips -             653325    1588937    2.42    177.0  7.5e-16
      135x5x7  3^3.5        plane+payonce+stages -               4306      10318    2.32     33.5  5.4e-16
    135x35x63  3^3.5        plane+payonce+stages -              79338      84961    0.93    170.6  6.7e-16
  135x128x128  3^3.5        plane+child+strips -             463337     483231    1.03    251.5  5.7e-16
    256x32x32  2^8          plane+payonce+stages -              54727      84013    1.27    215.6  4.4e-16
   257x67x127  257          plane+child -            1303725    4468412    3.34     88.3  1.1e-15
  257x131x251  257          plane+child -            6783062   16867444    2.28     71.7  1.1e-15
     259x9x15  7.37         plane+payonce+stages -              18433      35421    1.89     71.6  5.9e-16
   259x65x125  7.37         plane+child+strips -             835912    1315000    1.54    132.2  8.3e-16
  259x135x255  7.37         plane+child+strips -            5143287    8665481    1.61    100.1  8.5e-16
  262x106x122  2.131        plane+child+strips -            1998775    8270300    3.98     91.9  9.7e-16
  262x134x254  2.131        plane+child+strips -            6590537   21149538    2.96     78.1  1.1e-15
    270x10x14  2.3^3.5      plane+payonce+stages -              16795      22469    1.32     85.6  7.5e-16
   270x70x126  2.3^3.5      plane+child+strips -             573200     668018    1.16    220.0  7.1e-16
  270x130x250  2.3^3.5      plane+child+strips -            4721187    4353950    0.91    107.2  7.6e-16
    512x64x64  2^9          plane+child+strips -             443725     592937    1.31    248.1  4.4e-16
  512x128x128  2^9          plane+child+strips -            3947175    4892606    1.20    122.2  5.3e-16
  514x128x128  2.257        plane+child+strips -            4796375    8942481    1.85    101.0  7.0e-16
    516x18x30  2^2.3.43     plane+payonce+stages -             101007     178128    1.76    124.7  6.9e-16
  516x128x128  2^2.3.43     plane+child+strips -            4477800    6806794    1.49    108.6  7.5e-16
  521x128x128  521          plane+child -            4736288    8909294    1.78    103.7  1.3e-15
    525x17x31  3.5^2.7      plane+payonce+stages -             122607     210150    1.68    102.0  7.5e-16
  525x128x128  3.5^2.7      plane+child+strips -            4246913    4851293    1.14    116.6  8.1e-16
 1027x106x122  13.79        plane+child+strips -           10620688   30629250    2.83     74.0  1.1e-15
   1029x35x63  3.7^3        plane+child+strips -             682513     837169    1.09    175.5  6.2e-16
  1029x65x125  3.7^3        plane+child+strips -            4877425    4999093    1.00     98.5  8.3e-16
   1031x53x61  1031         plane+child -            3116800    7335069    2.27     57.9  1.8e-15
  1031x67x127  1031         plane+child -            9600200   20475175    2.12     52.7  1.7e-15
   1200x34x62  2^4.3.5^2    plane+child+strips -             938487    1297200    1.38    143.3  7.2e-16
  1200x70x126  2^4.3.5^2    plane+child+strips -            6832650    6373712    0.92     90.4  7.5e-16
   2048x64x64  2^11         plane+child+strips -            4579937    5261499    1.14    105.3  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child+strips              35      0      6   0.92   1.20   2.83    1.39
 plane+payonce+stages            22      1      5   0.93   1.28   1.89    1.32
 plane+child                     17      1      1   1.05   2.28   3.34    2.04
 child                           11      0      0   2.97   5.88  13.21    6.11
 ALL                             85      2     12   0.98   1.48   4.35    1.79
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     30      0      6   0.92   1.40   4.35    1.72
 odd column                      21      1      5   0.93   1.47   2.83    1.54
 pow2 column                     18      1      1   1.00   1.22  13.21    1.67
 prime column                    16      0      0   1.46   2.55   4.89    2.57
 ALL                             85      2     12   0.98   1.48   4.35    1.79
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  52      0      8   0.99   1.52   2.96    1.59
 4097..65536                     20      2      4   0.85   1.26   2.32    1.30
 257..1024                       10      0      0   4.35   6.68  26.66    6.57
 1025..4096                       3      0      0   1.17   1.48   2.97    1.72
 ALL                             85      2     12   0.98   1.48   4.35    1.79
```


worst 10: 32x32x32 (plane+child 0.69), 35x9x15 (plane+payonce+stages 0.78), 70x130x250 (plane+child+strips 0.85), 35x32x32 (plane+payonce+stages 0.85), 10x18x30 (plane+child+strips 0.85), 270x130x250 (plane+child+strips 0.91), 1200x70x126 (plane+child+strips 0.92), 135x35x63 (plane+payonce+stages 0.93), 70x64x64 (plane+payonce+stages 0.98), 35x65x125 (plane+payonce+stages 0.99)
best 5: 2x16x16 (child 26.66), 32x4x4 (child 13.21), 18x6x6 (child 8.41), 9x8x8 (child 7.79), 10x8x8 (child 7.47)
