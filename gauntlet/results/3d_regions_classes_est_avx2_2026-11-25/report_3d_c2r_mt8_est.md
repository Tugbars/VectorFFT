# gauntlet report (3D)

run: `3d_regions_classes_est_avx2_2026-11-25`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_mt8_est  cells: 160 listed, 160 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 6 readings, 1.218..2.011

threaded plans that ran serial (engaged = 0 at both flips): 17: 10x10x10, 10x10x9, 10x9x10, 10x9x9, 6x10x70, 6x6x35, 6x6x42, 6x9x21, 6x9x24, 9x10x10, 9x10x9, 9x6x21, 9x6x24, 9x9x10, 9x9x15, 9x9x18, 9x9x9


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      6x6x35  2.3          child  -               1404       2773    1.85     23.1  1.0e-15
      6x6x42  2.3          child  -               1490       2666    1.78     26.8  1.1e-15
      6x9x21  2.3          child  -               1614       4763    2.90     17.8  9.9e-16
      6x9x24  2.3          child  -               1008       4654    4.58     33.2  9.9e-16
     6x10x70  2.3          child  -               3658       6312    1.71     34.6  1.0e-15
     6x15x70  2.3          plane+child+strips -               3076       7102    2.02     64.6  1.3e-15
     6x15x75  2.3          plane+child+strips -               3280       8458    2.45     65.4  1.3e-15
    6x42x135  2.3          plane+child+strips -              12312      23723    1.67    104.0  1.5e-15
    6x45x135  2.3          plane+child+strips -              12598      26807    2.05    109.6  1.7e-15
    6x54x150  2.3          plane+child+strips -              14071      27552    1.66    134.4  1.7e-15
    6x135x42  2.3          plane+child+strips -              13035      30997    2.12     98.2  1.5e-15
    6x135x45  2.3          plane+child+strips -              13050      33265    2.40    105.8  1.6e-15
   6x135x225  2.3          plane+child+strips -              56267     111557    1.69    141.5  1.9e-15
    6x150x54  2.3          plane+child+strips -              15976      31177    1.91    118.4  1.5e-15
   6x225x135  2.3          plane+child+strips -              60267     111026    1.78    132.1  1.9e-15
      9x6x21  3^2          child  -               1567       4828    3.00     18.4  1.1e-15
      9x6x24  3^2          child  -               1014       4738    4.50     33.0  8.6e-16
       9x9x9  3^2          child  -                927        538    0.58     18.7  9.9e-16
      9x9x10  3^2          child  -                447       1884    4.13     43.8  9.6e-16
      9x9x15  3^2          child  -               1651       2194    1.12     18.9  8.9e-16
      9x9x18  3^2          child  -               1880       2064    1.03     20.4  1.3e-15
      9x9x70  3^2          plane+child+strips -               3201       5038    1.52     55.2  1.1e-15
      9x9x75  3^2          plane+child+strips -               3418       6869    1.81     55.8  1.4e-15
      9x10x9  3^2          child  -                966       1833    1.86     20.3  9.9e-16
     9x10x10  3^2          child  -                465       1692    3.63     47.5  6.9e-16
     9x10x70  3^2          plane+child+strips -               3441       5332    1.51     57.8  1.1e-15
     9x10x75  3^2          plane+child+strips -               3790       6907    1.73     56.6  1.2e-15
    9x35x135  3^2          plane+child+strips -              16823      32018    1.89     97.2  1.3e-15
    9x35x150  3^2          plane+child+strips -              16346      28686    1.69    112.2  1.8e-15
    9x75x210  3^2          plane+child+strips -              43071      65541    1.44    140.8  1.7e-15
    9x135x35  3^2          plane+child+strips -              19803      32350    1.62     82.5  1.5e-15
    9x150x35  3^2          plane+child+strips -              21720      35291    1.59     84.4  1.6e-15
    9x210x75  3^2          plane+child+strips -              59464      98585    1.48    102.0  1.7e-15
   9x225x315  3^2          plane+child+strips -             227487     341037    1.49    135.2  2.1e-15
   9x300x420  3^2          plane+child+strips -             377500     484113    1.22    151.0  2.0e-15
   9x315x225  3^2          plane+child+strips -             247638     357488    1.43    124.2  2.1e-15
   9x420x300  3^2          plane+child+strips -             369300     529006    1.26    154.4  2.0e-15
      10x9x9  2.5          child  -                967       1987    1.99     20.2  9.4e-16
     10x9x10  2.5          child  -                493       1944    3.84     44.8  8.5e-16
     10x10x9  2.5          child  -               1061       1890    1.71     20.8  8.8e-16
    10x10x10  2.5          child  -                533        637    1.18     46.7  7.5e-16
    10x10x75  2.5          plane+child+strips -               3882       7172    1.71     62.2  1.3e-15
   10x35x150  2.5          plane+child+strips -              16707      30487    1.81    123.2  1.4e-15
   10x70x210  2.5          plane+child+strips -              42326      70529    1.55    149.0  1.5e-15
   10x75x210  2.5          plane+child+strips -              43944      69448    1.40    154.7  1.6e-15
   10x150x35  2.5          plane+child+strips -              22086      36430    1.56     93.2  1.5e-15
   10x210x70  2.5          plane+child+strips -              48178      83196    1.54    130.9  1.6e-15
   10x210x75  2.5          plane+child+strips -              61152     107582    1.70    111.2  1.7e-15
  10x420x525  2.5          plane+child+strips -            1237675    1473819    1.09     93.9  2.2e-15
  10x525x420  2.5          plane+child+strips -            1179162     947362    0.78     98.5  2.0e-15
   15x18x135  3.5          plane+child+strips -              10845      26735    2.26    127.3  1.5e-15
   15x54x225  3.5          plane+child+strips -              49038      73219    1.21    162.4  1.8e-15
   15x135x18  3.5          plane+child+strips -              13296      24314    1.76    103.9  1.5e-15
   15x225x54  3.5          plane+child+strips -              54762      61459    0.95    145.4  1.8e-15
  15x315x525  3.5          plane+child+strips -            1154587    1074569    0.85    114.1  2.2e-15
  15x420x630  3.5          plane+child+strips -            2133888    1617293    0.70    101.9  2.3e-15
  15x525x315  3.5          plane+child+strips -            1092675    1089269    0.92    120.6  2.0e-15
  15x630x420  3.5          plane+child+strips -            2194775    1624750    0.73     99.1  2.2e-15
    18x18x18  2.3^2        plane+child+strips -               3704       4649    1.23     49.2  1.4e-15
    18x18x21  2.3^2        plane+child+strips -               4228       5807    1.35     51.2  1.3e-15
    18x21x18  2.3^2        plane+child+strips -               4018       6062    1.46     53.9  1.3e-15
   18x42x225  2.3^2        plane+child+strips -              53061      65254    1.22    139.3  1.7e-15
  18x150x420  2.3^2        plane+child+strips -             308162     300644    0.97    185.0  2.1e-15
   18x225x42  2.3^2        plane+child+strips -              54148      59082    1.05    136.5  2.1e-15
  18x420x150  2.3^2        plane+child+strips -             306762     384893    1.19    185.9  2.3e-15
    21x18x18  3.7          plane+child+strips -               3916       6652    1.66     55.3  1.4e-15
    21x21x21  3.7          plane+child+strips -               4762      10348    2.11     64.1  1.2e-15
    21x21x24  3.7          plane+child+strips -               4152       7590    1.70     85.2  9.7e-16
    21x24x21  3.7          plane+child+strips -               5553      10188    1.73     63.7  1.3e-15
   21x35x225  3.7          plane+child+strips -              45175      80175    1.49    158.7  1.6e-15
  21x210x525  3.7          plane+child+strips -            1039800    1060681    1.01    117.7  2.3e-15
   21x225x35  3.7          plane+child+strips -              51946      74383    1.37    138.0  1.8e-15
  21x525x210  3.7          plane+child+strips -             970762     847562    0.87    126.1  2.2e-15
    24x21x21  2^3.3        plane+child+strips -               5139      10552    1.61     68.8  1.2e-15
    24x42x42  2^3.3        plane+child+strips -              13857      22682    1.63    117.4  1.3e-15
   24x70x315  2^3.3        plane+child+strips -             125025     232406    1.29    201.2  1.8e-15
   24x75x315  2^3.3        plane+child+strips -             134713     180006    1.32    201.1  1.8e-15
   24x315x70  2^3.3        plane+child+strips -             135138     204419    1.15    186.1  1.8e-15
   24x315x75  2^3.3        plane+child+strips -             173675     231075    1.25    156.0  2.1e-15
   35x10x150  5.7          plane+child+strips -              14483      23446    1.33    142.1  1.4e-15
    35x35x35  5.7          plane+child+strips -              15273      28370    1.69    108.0  1.2e-15
    35x35x42  5.7          plane+child+strips -              19635      22043    1.05    102.5  1.4e-15
    35x42x35  5.7          plane+child+strips -              18309      24215    1.27    110.0  1.3e-15
    35x42x42  5.7          plane+child+strips -              20812      29567    1.14    118.0  1.3e-15
   35x75x420  5.7          plane+child+strips -             276025     277213    0.99    200.4  1.7e-15
   35x150x10  5.7          plane+child+strips -              16329      26949    1.48    126.0  1.3e-15
   35x420x75  5.7          plane+child+strips -             375112     482206    1.26    147.5  1.9e-15
    42x35x35  2.3.7        plane+child+strips -              18326      28845    1.18    109.8  1.5e-15
    42x35x42  2.3.7        plane+child+strips -              22209      23744    1.05    110.6  1.4e-15
    42x42x35  2.3.7        plane+child+strips -              21341      25511    1.19    115.1  1.5e-15
  42x105x525  2.3.7        plane+child+strips -             965850     895156    0.92    126.7  1.9e-15
  42x150x630  2.3.7        plane+child+strips -            2000550    1527499    0.74    108.7  2.4e-15
  42x525x105  2.3.7        plane+child+strips -             935263    1028806    1.06    130.8  2.1e-15
  42x630x150  2.3.7        plane+child+strips -            2070937    1733493    0.82    105.0  2.2e-15
    45x45x54  3^2.5        plane+child+strips -              31675      46862    1.14    144.5  1.8e-15
    45x54x45  3^2.5        plane+child+strips -              32019      44439    1.30    142.9  1.7e-15
    45x54x54  3^2.5        plane+child+strips -              36000      52870    1.10    154.9  2.0e-15
    45x75x75  3^2.5        plane+child+strips -              73660     110406    1.20    154.2  1.7e-15
  45x135x630  3^2.5        plane+child+strips -            1905912    1321081    0.68    109.8  2.7e-15
  45x630x135  3^2.5        plane+child+strips -            1973062    1876612    0.95    106.0  2.2e-15
    54x45x45  2.3^3        plane+child+strips -              31967      45987    1.22    143.1  1.9e-15
    54x45x54  2.3^3        plane+child+strips -              38783      54813    0.97    143.8  2.1e-15
    54x54x45  2.3^3        plane+child+strips -              37963      56668    1.32    146.9  2.0e-15
    54x54x54  2.3^3        plane+child+strips -              43064      47618    0.92    157.8  2.1e-15
   75x10x210  3.5^2        plane+child+strips -              37416      59020    1.38    181.7  1.5e-15
   75x24x315  3.5^2        plane+child+strips -             144175     187112    1.29    187.9  1.7e-15
  75x105x105  3.5^2        plane+child+strips -             243275     304693    1.18    167.0  2.2e-15
   75x210x10  3.5^2        plane+child+strips -              43760      70704    1.58    155.3  1.5e-15
   75x315x24  3.5^2        plane+child+strips -             174575     197637    1.03    155.2  1.6e-15
 100x100x100  2^2.5^2      plane+child+strips -             242600     289806    1.10    205.4  1.9e-15
 100x100x105  2^2.5^2      plane+child+strips -             304612     400493    1.25    172.4  1.6e-15
 100x105x100  2^2.5^2      plane+child+strips -             241875     299844    1.17    217.1  1.8e-15
 100x105x105  2^2.5^2      plane+child+strips -             309012     417587    1.25    179.0  1.9e-15
 105x100x100  3.5.7        plane+child+strips -             258738     291537    1.13    202.9  1.8e-15
 105x100x105  3.5.7        plane+child+strips -             320362     517931    1.26    172.7  1.9e-15
 105x105x100  3.5.7        plane+child+strips -             255513     324499    1.22    216.5  2.0e-15
    135x6x42  3^3.5        plane+child+strips -              18445      25808    1.29     69.4  1.4e-15
    135x6x45  3^3.5        plane+child+strips -              17692      28137    1.47     78.1  1.5e-15
    135x9x35  3^3.5        plane+child+strips -              19980      29465    1.31     81.8  1.5e-15
   135x15x18  3^3.5        plane+child+strips -              19146      20135    1.04     72.1  1.7e-15
 135x135x135  3^3.5        plane+child+strips -             991187     892225    0.90    131.7  2.3e-15
 135x135x150  3^3.5        plane+child+strips -            1078300     816162    0.71    135.5  2.3e-15
 135x150x135  3^3.5        plane+child+strips -            1177638    1071925    0.83    124.1  2.3e-15
 135x150x150  3^3.5        plane+child+strips -            1336025     896231    0.65    122.4  2.3e-15
    150x6x54  2.3.5^2      plane+child+strips -              22248      24416    1.07     85.0  1.6e-15
    150x9x35  2.3.5^2      plane+child+strips -              21957      29891    1.31     83.5  1.3e-15
   150x10x35  2.3.5^2      plane+child+strips -              22892      32971    1.27     89.9  1.3e-15
   150x35x10  2.3.5^2      plane+child+strips -              24345      33228    1.23     84.5  1.1e-15
 150x135x135  2.3.5^2      plane+child+strips -            1166812    1083837    0.90    125.2  2.2e-15
 150x135x150  2.3.5^2      plane+child+strips -            1339100    1088681    0.66    122.1  2.3e-15
 150x150x135  2.3.5^2      plane+child+strips -            1378562    1164518    0.84    118.6  2.3e-15
 150x150x150  2.3.5^2      plane+child+strips -            1577288    1025056    0.64    116.0  2.2e-15
    210x9x75  2.3.5.7      plane+child+strips -              46311      77823    1.54    131.0  1.9e-15
   210x10x70  2.3.5.7      plane+child+strips -              45185      63104    1.20    139.6  1.5e-15
   210x10x75  2.3.5.7      plane+child+strips -              50624      79280    1.55    134.3  1.8e-15
   210x75x10  2.3.5.7      plane+child+strips -              51836      97724    1.70    131.1  1.5e-15
   225x6x135  3^2.5^2      plane+child+strips -              55462      83843    1.42    143.6  1.8e-15
   225x15x54  3^2.5^2      plane+child+strips -              60148      79507    1.03    132.4  1.9e-15
   225x18x42  3^2.5^2      plane+child+strips -              63552      72565    1.07    116.3  1.7e-15
   225x21x35  3^2.5^2      plane+child+strips -              54396      84775    1.35    131.8  1.6e-15
   300x9x420  2^2.3.5^2    plane+child+strips -             243100     411231    1.67    234.6  1.9e-15
  300x21x630  2^2.3.5^2    plane+child+strips -            1966150    1847756    0.92    110.6  2.1e-15
   300x420x9  2^2.3.5^2    plane+child+strips -             407250     736131    1.80    140.0  2.2e-15
  300x630x21  2^2.3.5^2    plane+child+strips -            2279337    2412369    1.05     95.4  2.0e-15
   315x9x225  3^2.5.7      plane+child+strips -             177125     302387    1.59    173.6  2.0e-15
   315x24x70  3^2.5.7      plane+child+strips -             145762     171394    1.17    172.6  1.7e-15
   315x24x75  3^2.5.7      plane+child+strips -             183875     227881    1.18    147.3  1.8e-15
   315x75x24  3^2.5.7      plane+child+strips -             151675     208062    1.32    178.6  1.6e-15
   420x9x300  2^2.3.5.7    plane+child+strips -             243487     442788    1.74    234.2  2.3e-15
  420x18x150  2^2.3.5.7    plane+child+strips -             275362     373800    1.34    207.1  2.0e-15
   420x35x75  2^2.3.5.7    plane+child+strips -             372600     450769    1.20    148.5  1.9e-15
   420x300x9  2^2.3.5.7    plane+child+strips -             411262     683612    1.65    138.6  2.1e-15
  525x10x420  3.5^2.7      plane+child+strips -             676825     856512    1.23    171.6  1.8e-15
  525x15x315  3.5^2.7      plane+child+strips -             958813     927050    0.91    137.4  2.1e-15
  525x21x210  3.5^2.7      plane+child+strips -             838150     923581    1.02    146.0  2.2e-15
  525x42x105  3.5^2.7      plane+child+strips -             855300     921387    1.03    143.1  2.1e-15
  630x15x420  2.3^2.5.7    plane+child+strips -            2216625    1541169    0.67     98.1  2.1e-15
  630x42x150  2.3^2.5.7    plane+child+strips -            2264825    1607993    0.70     96.0  2.1e-15
  630x45x135  2.3^2.5.7    plane+child+strips -            2061975    1849831    0.85    101.5  2.1e-15
  630x300x21  2.3^2.5.7    plane+child+strips -            2348975    2536456    1.06     92.6  2.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child+strips             143     11     29   0.85   1.26   1.74    1.25
 child                           17      1      1   1.03   1.86   4.50    2.10
 ALL                            160     12     30   0.85   1.28   1.91    1.32
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     80      6     15   0.85   1.30   2.02    1.35
 odd column                      80      6     15   0.87   1.27   1.86    1.30
 ALL                            160     12     30   0.85   1.28   1.91    1.32
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  96     11     29   0.74   1.18   1.58    1.13
 4097..65536                     48      0      0   1.14   1.62   2.11    1.55
 1025..4096                       8      0      0   1.03   2.38   4.58    2.26
 257..1024                        8      1      1   0.58   1.93   4.13    1.99
 ALL                            160     12     30   0.85   1.28   1.91    1.32
```


worst 10: 9x9x9 (child 0.58), 150x150x150 (plane+child+strips 0.64), 135x150x150 (plane+child+strips 0.65), 150x135x150 (plane+child+strips 0.66), 630x15x420 (plane+child+strips 0.67), 45x135x630 (plane+child+strips 0.68), 630x42x150 (plane+child+strips 0.70), 15x420x630 (plane+child+strips 0.70), 135x135x150 (plane+child+strips 0.71), 15x630x420 (plane+child+strips 0.73)
best 5: 6x9x24 (child 4.58), 9x6x24 (child 4.50), 9x9x10 (child 4.13), 10x9x10 (child 3.84), 9x10x10 (child 3.63)
