# gauntlet report (3D)

run: `regression_3d_avx2_2026-11-25`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_mt8_est  cells: 85 listed, 85 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.206..1.279

threaded plans that ran serial (engaged = 0 at both flips): 11: 10x8x8, 17x5x7, 18x6x6, 2x16x16, 32x4x4, 4x32x32, 5x9x15, 65x3x3, 6x10x14, 8x8x8, 9x8x8


## every shape

```
        shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      2x16x16  2            child  -                239       5825   23.99     48.2  4.5e-16
      4x32x32  2^2          child  -               2692       7824    2.84     45.7  5.6e-16
       5x9x15  5            child  -               1290       4474    3.36     12.3  7.6e-16
      6x10x14  2.3          child  -                974       4174    4.23     20.9  6.7e-16
        8x8x8  2^3          child  -                242        548    2.23     47.5  4.5e-16
      8x64x64  2^3          plane+child+strips -               7639      18958    2.38    160.9  8.9e-16
        9x8x8  3^2          child  -                295       2302    7.55     44.7  6.2e-16
      9x17x31  3^2          plane+child+strips -               4612       8404    1.76     31.4  1.5e-15
       10x8x8  2.5          child  -                311       2367    7.32     48.0  5.3e-16
     10x18x30  2.5          plane+child+strips -               3327       4490    1.32     50.3  1.1e-15
     16x16x16  2^4          plane+child+strips -               1995       2414    1.18     61.6  6.7e-16
   16x128x128  2^4          plane+child -              38527      57320    1.46    306.2  1.2e-15
       17x5x7  17           child  -               1516       4738    3.09      9.0  8.9e-16
     17x16x16  17           plane+child+strips -               2261       3409    1.46     58.2  1.0e-15
     17x35x63  17           plane+child+strips -              14858      37136    2.48     95.8  1.5e-15
       18x6x6  2.3^2        child  -                475       4275    8.86     31.9  7.0e-16
     18x16x16  2.3^2        plane+child+strips -               2282       2936    1.27     61.4  8.9e-16
     18x34x62  2.3^2        plane+child+strips -              17246      32578    1.88     83.7  1.6e-15
       32x4x4  2^5          child  -                416       5599   12.91     27.7  4.4e-16
     32x32x32  2^5          plane+child -               6759      10382    1.51    181.8  7.8e-16
   32x256x256  2^5          plane+child -             485688     511050    1.02    226.7  1.2e-15
     34x10x14  2.17         plane+child+strips -               3525       5496    1.51     41.2  1.1e-15
     34x32x32  2.17         plane+child+strips -               9353      16016    1.67    140.4  1.3e-15
    34x70x126  2.17         plane+child+strips -              77438     113688    1.42    176.1  1.7e-15
      35x9x15  5.7          plane+child+strips -               4784       4925    0.90     30.1  1.2e-15
     35x32x32  5.7          plane+child+strips -               8911      13141    1.28    152.1  9.8e-16
    35x65x125  5.7          plane+child+strips -              95886     111539    1.10    134.3  1.7e-15
     53x32x32  53           plane+child -              15790      43784    2.76    135.1  1.6e-15
    53x67x127  53           plane+child -             298862     952175    3.09     70.9  3.0e-15
       64x8x8  2^6          plane+child -               3043       4521    1.37     40.4  6.7e-16
     64x64x64  2^6          plane+child -              45067      71793    1.14    261.8  1.1e-15
       65x3x3  5.13         child  -               2299       7065    3.04      5.8  1.0e-15
     65x17x31  5.13         plane+child+strips -              21911      37629    1.68     58.9  1.9e-15
     65x64x64  5.13         plane+child+strips -              59480      81516    1.02    201.7  1.5e-15
   65x135x255  5.13         plane+child+strips -             957287    1508775    1.56    123.3  2.4e-15
   65x259x507  5.13         plane+child+strips -            6260912   10557362    1.65     78.5  2.6e-15
     67x64x64  67           plane+child -              66864     283492    3.76    185.4  1.6e-15
   67x131x251  67           plane+child -            1897213    5005650    2.53     61.2  3.4e-15
   67x257x509  67           plane+child -           11545637   22594856    1.77     43.8  2.8e-15
     70x18x30  2.5.7        plane+child+strips -              17778      20633    1.12     80.8  1.3e-15
     70x64x64  2.5.7        plane+child+strips -              63446      65284    0.97    204.8  1.2e-15
   70x130x250  2.5.7        plane+child+strips -             775100     643331    0.79    155.0  2.0e-15
   70x270x510  2.5.7        plane+child+strips -            5701212    7418743    1.29     98.1  2.5e-15
    106x64x64  2.53         plane+child+strips -             138200     257239    1.81    147.1  2.0e-15
  106x134x254  2.53         plane+child+strips -            2719300    5669774    2.05     72.2  3.2e-15
  106x262x511  2.53         plane+child+strips -           17242025   58612956    3.32     48.9  3.1e-15
    128x16x16  2^7          plane+child+strips -              13301      14037    1.04     92.4  1.0e-15
  128x128x128  2^7          plane+child+strips -             535263     526193    0.95    205.7  1.6e-15
  128x256x256  2^7          plane+child+strips -            4716312    4681062    0.97    102.3  1.4e-15
      130x6x6  2.5.13       plane+child+strips -               7048       7795    1.09     20.2  1.4e-15
    130x34x62  2.5.13       plane+child+strips -             104557     181318    1.55    118.4  2.2e-15
  130x128x128  2.5.13       plane+child+strips -             587925     569975    0.94    190.4  2.1e-15
    131x53x61  131          plane+child -             278656     877128    2.94     71.0  4.2e-15
  131x128x128  131          plane+child -             711338    1512812    2.11    158.7  2.3e-15
  134x128x128  2.67         plane+child+strips -             819262    1679175    2.05    141.1  1.8e-15
      135x5x7  3^3.5        plane+child+strips -               8083       8699    0.96     17.8  1.2e-15
    135x35x63  3^3.5        plane+child+strips -              90231     118434    1.30    150.0  1.7e-15
  135x128x128  3^3.5        plane+child+strips -             626000     599062    0.91    186.2  1.7e-15
    256x32x32  2^8          plane+child+strips -              61293      90993    1.20    192.5  1.0e-15
   257x67x127  257          plane+child -            1638350    5036675    2.93     70.3  2.7e-15
  257x131x251  257          plane+child -            8924750   18918162    2.01     54.5  3.2e-15
     259x9x15  7.37         plane+child+strips -              28057      52693    1.83     47.0  1.4e-15
   259x65x125  7.37         plane+child+strips -             881175    1727362    1.90    125.4  2.3e-15
  259x135x255  7.37         plane+child+strips -            5613225   10224318    1.80     91.7  2.7e-15
  262x106x122  2.131        plane+child+strips -            2480100    8692850    3.38     74.1  2.9e-15
  262x134x254  2.131        plane+child+strips -            7469050   21074325    2.78     68.9  3.3e-15
    270x10x14  2.3^3.5      plane+child+strips -              23778      32906    1.37     60.4  1.3e-15
   270x70x126  2.3^3.5      plane+child+strips -             843150     708231    0.78    149.6  2.0e-15
  270x130x250  2.3^3.5      plane+child+strips -            5198363    4528894    0.83     97.3  2.5e-15
    512x64x64  2^9          plane+child+strips -             587613     693825    1.08    187.4  1.3e-15
  512x128x128  2^9          plane+child+strips -            5108263    5138281    0.98     94.4  1.4e-15
  514x128x128  2.257        plane+child+strips -            6191288    9150462    1.45     78.2  2.0e-15
    516x18x30  2^2.3.43     plane+child+strips -             142329     224350    1.46     88.5  2.1e-15
  516x128x128  2^2.3.43     plane+child+strips -            5893437    7062518    1.16     82.5  1.9e-15
  521x128x128  521          plane+child -            5805187    9289681    1.57     84.6  2.7e-15
    525x17x31  3.5^2.7      plane+child+strips -             151164     275296    1.79     82.7  2.1e-15
  525x128x128  3.5^2.7      plane+child+strips -            5594550    4756531    0.84     88.5  2.0e-15
 1027x106x122  13.79        plane+child+strips -           11649337   33178850    2.81     67.4  3.4e-15
   1029x35x63  3.7^3        plane+child+strips -             889387    1188737    1.30    134.7  1.9e-15
  1029x65x125  3.7^3        plane+child+strips -            5282137    6066131    1.11     91.0  2.5e-15
   1031x53x61  1031         plane+child -            3763150    8226194    1.94     48.0  5.2e-15
  1031x67x127  1031         plane+child -           10667775   22874937    2.14     47.4  4.2e-15
   1200x34x62  2^4.3.5^2    plane+child+strips -            1285013    1622781    1.16    104.7  2.3e-15
  1200x70x126  2^4.3.5^2    plane+child+strips -            8295500    7108613    0.84     74.4  2.1e-15
   2048x64x64  2^11         plane+child+strips -            5633662    5903818    1.02     85.6  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child+strips              57      2     13   0.90   1.30   2.38    1.37
 plane+child                     17      0      0   1.14   2.01   3.09    1.99
 child                           11      0      0   2.84   4.23  12.91    5.46
 ALL                             85      2     13   0.95   1.55   3.36    1.76
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     30      2      6   0.84   1.44   4.23    1.64
 odd column                      21      0      4   0.91   1.56   2.81    1.55
 pow2 column                     18      0      3   0.97   1.19  12.91    1.75
 prime column                    16      0      0   1.57   2.50   3.36    2.41
 ALL                             85      2     13   0.95   1.55   3.36    1.76
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  52      2     11   0.91   1.45   2.93    1.50
 4097..65536                     20      0      2   1.04   1.48   2.48    1.49
 257..1024                       10      0      0   3.04   5.77  23.99    5.83
 1025..4096                       3      0      0   1.18   1.37   2.84    1.66
 ALL                             85      2     13   0.95   1.55   3.36    1.76
```


worst 10: 270x70x126 (plane+child+strips 0.78), 70x130x250 (plane+child+strips 0.79), 270x130x250 (plane+child+strips 0.83), 1200x70x126 (plane+child+strips 0.84), 525x128x128 (plane+child+strips 0.84), 35x9x15 (plane+child+strips 0.90), 135x128x128 (plane+child+strips 0.91), 130x128x128 (plane+child+strips 0.94), 128x128x128 (plane+child+strips 0.95), 135x5x7 (plane+child+strips 0.96)
best 5: 2x16x16 (child 23.99), 32x4x4 (child 12.91), 18x6x6 (child 8.86), 9x8x8 (child 7.55), 10x8x8 (child 7.32)
