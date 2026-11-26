# gauntlet report (3D)

run: `regression_3d_avx2_2026-11-25`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_mt4_est  cells: 85 listed, 85 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.063..1.077

threaded plans that ran serial (engaged = 0 at both flips): 10: 10x8x8, 17x5x7, 18x6x6, 2x16x16, 32x4x4, 5x9x15, 65x3x3, 6x10x14, 8x8x8, 9x8x8


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      2x16x16  2            child  -                224       3509   15.11     51.3  2.7e-16
      4x32x32  2^2          plane+child+strips -               4014       4777    1.15     30.6  2.6e-16
       5x9x15  5            child  -                914       2300    2.33     17.3  5.0e-16
      6x10x14  2.3          child  -                937       2494    2.60     21.8  3.4e-16
        8x8x8  2^3          child  -                255        320    1.25     45.1  2.7e-16
      8x64x64  2^3          plane+child+strips -              15566      18603    1.17     78.9  3.4e-16
        9x8x8  3^2          child  -                308       2474    7.75     42.8  2.6e-16
      9x17x31  3^2          plane+child+strips -               6802       9123    1.32     21.3  6.9e-16
       10x8x8  2.5          child  -                352       2443    6.82     42.4  3.7e-16
     10x18x30  2.5          plane+child+strips -               4703       4393    0.91     35.6  5.0e-16
     16x16x16  2^4          plane+child+strips -               2860       3714    1.21     43.0  1.9e-16
   16x128x128  2^4          plane+child -              65280      86570    1.30    180.7  4.0e-16
       17x5x7  17           child  -                880       2874    3.21     15.6  4.5e-16
     17x16x16  17           plane+child+strips -               3476       5831    1.65     37.8  5.4e-16
     17x35x63  17           plane+child+strips -              30820      33722    1.06     46.2  5.9e-16
       18x6x6  2.3^2        child  -                493       2657    5.22     30.7  4.5e-16
     18x16x16  2.3^2        plane+payonce+stages -               3490       4324    1.11     40.2  4.3e-16
     18x34x62  2.3^2        plane+payonce+stages -              25542      40467    1.47     56.5  7.9e-16
       32x4x4  2^5          child  -                428       3771    8.64     26.9  2.4e-16
     32x32x32  2^5          plane+child -              25183      16042    0.63     48.8  3.4e-16
   32x256x256  2^5          plane+child -             894475     972787    0.94    123.1  4.6e-16
     34x10x14  2.17         plane+payonce+stages -               4325       6472    1.46     33.6  5.0e-16
     34x32x32  2.17         plane+payonce+stages -              19105      25760    1.33     68.7  5.7e-16
    34x70x126  2.17         plane+payonce+stages -             134308     191088    1.40    101.6  7.4e-16
      35x9x15  5.7          plane+payonce+stages -               4533       3997    0.79     31.8  5.6e-16
     35x32x32  5.7          plane+payonce+stages -              22114      18784    0.80     61.3  4.4e-16
    35x65x125  5.7          plane+payonce+stages -             135600     157543    1.13     95.0  6.7e-16
     53x32x32  53           plane+child -              28022      82622    2.70     76.2  7.5e-16
    53x67x127  53           plane+child -             547150    1801674    3.07     38.7  1.2e-15
       64x8x8  2^6          plane+child -               3081       7806    2.19     39.9  2.5e-16
     64x64x64  2^6          plane+child -              86713      90537    0.99    136.0  4.2e-16
       65x3x3  5.13         child  -               1315       6039    4.56     10.2  5.4e-16
     65x17x31  5.13         plane+payonce+stages -              33884      48887    1.34     38.1  8.6e-16
     65x64x64  5.13         plane+payonce+stages -              84920     108016    1.21    141.3  6.2e-16
   65x135x255  5.13         plane+child+strips -            1394212    2505138    1.74     84.6  9.8e-16
   65x259x507  5.13         plane+child+strips -            7517987   13725325    1.73     65.4  9.7e-16
     67x64x64  67           plane+child -             123771     493971    3.93    100.1  6.4e-16
   67x131x251  67           plane+child -            2777925    8551931    2.88     41.8  1.2e-15
   67x257x509  67           plane+child -           14195138   37264319    2.49     35.6  1.1e-15
     70x18x30  2.5.7        plane+payonce+stages -              25790      18839    0.71     55.7  5.8e-16
     70x64x64  2.5.7        plane+payonce+stages -              92962     102157    1.09    139.8  4.9e-16
   70x130x250  2.5.7        plane+child+strips -            1176200    1187506    0.85    102.1  7.8e-16
   70x270x510  2.5.7        plane+child+strips -            7427988    9447631    1.25     75.3  7.8e-16
    106x64x64  2.53         plane+child+strips -             273222     484989    1.75     74.4  7.1e-16
  106x134x254  2.53         plane+child+strips -            4147825   10123718    2.30     47.4  9.6e-16
  106x262x511  2.53         plane+child+strips -           25269587  103267718    3.94     33.4  9.7e-16
    128x16x16  2^7          plane+payonce+stages -              14105      17824    1.26     87.1  3.4e-16
  128x128x128  2^7          plane+child+strips -            1009512     881131    0.67    109.1  4.4e-16
  128x256x256  2^7          plane+child+strips -            5778863    6316056    1.09     83.5  4.5e-16
      130x6x6  2.5.13       plane+payonce+stages -               4497      10613    2.35     31.7  6.1e-16
    130x34x62  2.5.13       plane+payonce+stages -             149021     250603    1.67     83.0  8.4e-16
  130x128x128  2.5.13       plane+child+strips -             917487     881143    0.90    122.0  6.8e-16
    131x53x61  131          plane+child -             521611    1566833    2.98     37.9  1.5e-15
  131x128x128  131          plane+child -            1155775    2785594    2.34     97.6  7.4e-16
  134x128x128  2.67         plane+child+strips -            1328300    3140325    2.34     87.0  7.5e-16
      135x5x7  3^3.5        plane+payonce+stages -               5565      11065    1.93     25.9  5.4e-16
    135x35x63  3^3.5        plane+payonce+stages -             126315     156569    1.03    107.1  6.7e-16
  135x128x128  3^3.5        plane+child+strips -             932950     933481    0.94    124.9  5.7e-16
    256x32x32  2^8          plane+payonce+stages -              84847     118200    1.39    139.0  4.4e-16
   257x67x127  257          plane+child -            2570275    8354269    3.21     44.8  1.1e-15
  257x131x251  257          plane+child -           12939800   31850681    2.43     37.6  1.1e-15
     259x9x15  7.37         plane+payonce+stages -              23164      52535    2.21     57.0  5.9e-16
   259x65x125  7.37         plane+child+strips -            1588050    2666362    1.41     69.6  8.3e-16
  259x135x255  7.37         plane+child+strips -            7985937   15890137    1.98     64.4  8.5e-16
  262x106x122  2.131        plane+child+strips -            3955412   16054050    4.03     46.5  9.7e-16
  262x134x254  2.131        plane+child+strips -           11366075   39648687    3.45     45.3  1.1e-15
    270x10x14  2.3^3.5      plane+payonce+stages -              22056      26899    1.20     65.2  7.5e-16
   270x70x126  2.3^3.5      plane+child+strips -            1167825    1403412    1.14    108.0  7.1e-16
  270x130x250  2.3^3.5      plane+child+strips -            6471388    6602275    1.00     78.2  7.6e-16
    512x64x64  2^9          plane+child+strips -             840525    1151756    1.30    131.0  4.4e-16
  512x128x128  2^9          plane+child+strips -            5558800    7129812    1.22     86.8  5.3e-16
  514x128x128  2.257        plane+child+strips -            7747675   15526875    1.98     62.5  7.0e-16
    516x18x30  2^2.3.43     plane+payonce+stages -             160714     312493    1.91     78.4  6.9e-16
  516x128x128  2^2.3.43     plane+child+strips -            6500300   11101669    1.67     74.8  7.5e-16
  521x128x128  521          plane+child -            7297687   14651962    2.00     67.3  1.3e-15
    525x17x31  3.5^2.7      plane+payonce+stages -             216264     385821    1.77     57.8  7.5e-16
  525x128x128  3.5^2.7      plane+child+strips -            5850463    6609481    1.08     84.7  8.1e-16
 1027x106x122  13.79        plane+child+strips -           17440688   55684081    3.17     45.0  1.1e-15
   1029x35x63  3.7^3        plane+child+strips -            1313137    1664744    1.16     91.2  6.2e-16
  1029x65x125  3.7^3        plane+child+strips -            7351800    7447794    0.99     65.4  8.3e-16
   1031x53x61  1031         plane+child -            5962462   13454950    2.22     30.3  1.8e-15
  1031x67x127  1031         plane+child -           16591938   35988862    2.16     30.5  1.7e-15
   1200x34x62  2^4.3.5^2    plane+child+strips -            1731200    2570250    1.41     77.7  7.2e-16
  1200x70x126  2^4.3.5^2    plane+child+strips -            9024400    9159700    0.98     68.4  7.5e-16
   2048x64x64  2^11         plane+child+strips -            6476137    7337593    1.05     74.5  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child+strips              36      1      7   0.91   1.24   3.17    1.42
 plane+payonce+stages            22      3      3   0.80   1.33   1.93    1.32
 plane+child                     17      1      3   0.94   2.34   3.21    2.05
 child                           10      0      0   2.33   4.89  15.11    4.57
 ALL                             85      5     13   0.94   1.46   3.93    1.72
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     30      1      5   0.91   1.46   4.03    1.69
 odd column                      21      2      4   0.94   1.34   3.17    1.57
 pow2 column                     18      2      4   0.67   1.22   8.64    1.46
 prime column                    16      0      0   1.65   2.46   3.21    2.44
 ALL                             85      5     13   0.94   1.46   3.93    1.72
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  52      1      8   0.98   1.54   3.17    1.62
 4097..65536                     20      4      5   0.79   1.29   2.35    1.27
 257..1024                       10      0      0   2.33   4.89  15.11    4.57
 1025..4096                       3      0      0   1.15   1.21   2.19    1.45
 ALL                             85      5     13   0.94   1.46   3.93    1.72
```


worst 10: 32x32x32 (plane+child 0.63), 128x128x128 (plane+child+strips 0.67), 70x18x30 (plane+payonce+stages 0.71), 35x9x15 (plane+payonce+stages 0.79), 35x32x32 (plane+payonce+stages 0.80), 70x130x250 (plane+child+strips 0.85), 130x128x128 (plane+child+strips 0.90), 10x18x30 (plane+child+strips 0.91), 32x256x256 (plane+child 0.94), 135x128x128 (plane+child+strips 0.94)
best 5: 2x16x16 (child 15.11), 32x4x4 (child 8.64), 9x8x8 (child 7.75), 10x8x8 (child 6.82), 18x6x6 (child 5.22)
