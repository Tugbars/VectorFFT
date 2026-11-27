# gauntlet report (3D)

run: `3d_regions_classes_est_avx2_2026-11-25`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_mt4_est  cells: 160 listed, 160 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 8 readings, 1.182..1.264

threaded plans that ran serial (engaged = 0 at both flips): 16: 10x10x10, 10x10x9, 10x9x10, 10x9x9, 6x6x35, 6x6x42, 6x9x21, 6x9x24, 9x10x10, 9x10x9, 9x6x21, 9x6x24, 9x9x10, 9x9x15, 9x9x18, 9x9x9


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      6x6x35  2.3          child  -               1465       3127    2.07     22.2  1.0e-15
      6x6x42  2.3          child  -               1509       3377    2.17     26.5  1.1e-15
      6x9x21  2.3          child  -               1659       3259    1.89     17.3  9.9e-16
      6x9x24  2.3          child  -               1059       3480    3.21     31.6  9.9e-16
     6x10x70  2.3          plane+child+strips -               2892       5335    1.65     43.7  1.0e-15
     6x15x70  2.3          plane+child+strips -               4296       7120    1.44     46.3  1.3e-15
     6x15x75  2.3          plane+child+strips -               5149       9024    1.71     41.7  1.3e-15
    6x42x135  2.3          plane+child+strips -              19948      33540    1.66     64.2  1.5e-15
    6x45x135  2.3          plane+child+strips -              22175      35761    1.48     62.3  1.7e-15
    6x54x150  2.3          plane+child+strips -              24501      42988    1.74     77.2  1.7e-15
    6x135x42  2.3          plane+child+strips -              21114      42673    1.89     60.6  1.5e-15
    6x135x45  2.3          plane+child+strips -              24082      49326    1.98     57.3  1.6e-15
   6x135x225  2.3          plane+child+strips -              98467     159569    1.53     80.9  1.9e-15
    6x150x54  2.3          plane+child+strips -              28756      44671    1.51     65.8  1.5e-15
   6x225x135  2.3          plane+child+strips -             105143     162381    1.50     75.7  1.9e-15
      9x6x21  3^2          child  -               1648       3406    1.95     17.5  1.1e-15
      9x6x24  3^2          child  -               1062       3425    3.19     31.6  8.6e-16
       9x9x9  3^2          child  -                928        537    0.49     18.7  9.9e-16
      9x9x10  3^2          child  -                472       1770    3.74     41.5  9.6e-16
      9x9x15  3^2          child  -               1735       2407    1.17     17.9  8.9e-16
      9x9x18  3^2          child  -               1942       2305    1.16     19.7  1.3e-15
      9x9x70  3^2          plane+child+strips -               3966       6369    1.57     44.6  1.1e-15
      9x9x75  3^2          plane+child+strips -               4778       8351    1.72     39.9  1.4e-15
      9x10x9  3^2          child  -               1008       2005    1.98     19.4  9.9e-16
     9x10x10  3^2          child  -                485       1931    3.66     45.5  6.9e-16
     9x10x70  3^2          plane+child+strips -               4277       5768    1.31     46.5  1.1e-15
     9x10x75  3^2          plane+child+strips -               5132       8288    1.53     41.8  1.2e-15
    9x35x135  3^2          plane+child+strips -              24551      46530    1.86     66.6  1.3e-15
    9x35x150  3^2          plane+child+strips -              23376      41397    1.68     78.5  1.8e-15
    9x75x210  3^2          plane+child+strips -              61621      73832    1.11     98.4  1.7e-15
    9x135x35  3^2          plane+child+strips -              29287      46585    1.51     55.8  1.5e-15
    9x150x35  3^2          plane+child+strips -              33344      48114    1.40     55.0  1.6e-15
    9x210x75  3^2          plane+child+strips -              87918     115482    1.29     69.0  1.7e-15
   9x225x315  3^2          plane+child+strips -             360150     531250    1.46     85.4  2.1e-15
   9x300x420  3^2          plane+child+strips -             632700     711637    1.11     90.1  2.0e-15
   9x315x225  3^2          plane+child+strips -             391963     552656    1.38     78.5  2.1e-15
   9x420x300  3^2          plane+child+strips -             617612     718181    1.15     92.3  2.0e-15
      10x9x9  2.5          child  -               1015       1978    1.91     19.3  9.4e-16
     10x9x10  2.5          child  -                519       1901    3.59     42.5  8.5e-16
     10x10x9  2.5          child  -               1115       2006    1.75     19.8  8.8e-16
    10x10x10  2.5          child  -                531        636    1.18     46.9  7.5e-16
    10x10x75  2.5          plane+child+strips -               5179       8683    1.56     46.6  1.3e-15
   10x35x150  2.5          plane+child+strips -              24830      43448    1.73     82.9  1.4e-15
   10x70x210  2.5          plane+child+strips -              60033      69766    1.16    105.1  1.5e-15
   10x75x210  2.5          plane+child+strips -              64232      73956    1.14    105.8  1.6e-15
   10x150x35  2.5          plane+child+strips -              33787      49746    1.37     60.9  1.5e-15
   10x210x70  2.5          plane+child+strips -              67089      77992    1.12     94.0  1.6e-15
   10x210x75  2.5          plane+child+strips -              88960     115106    1.27     76.4  1.7e-15
  10x420x525  2.5          plane+child+strips -            1799313    2067012    1.14     64.6  2.2e-15
  10x525x420  2.5          plane+child+strips -            1678312    1542256    0.86     69.2  2.0e-15
   15x18x135  3.5          plane+child+strips -              19744      32100    1.62     69.9  1.5e-15
   15x54x225  3.5          plane+child+strips -              85952     110923    1.23     92.6  1.8e-15
   15x135x18  3.5          plane+child+strips -              24436      33762    1.29     56.5  1.5e-15
   15x225x54  3.5          plane+child+strips -              81124      96357    1.17     98.2  1.8e-15
  15x315x525  3.5          plane+child+strips -            1894625    2417800    1.06     69.5  2.2e-15
  15x420x630  3.5          plane+child+strips -            3058000    3296887    1.03     71.1  2.3e-15
  15x525x315  3.5          plane+child+strips -            1791313    2040869    1.14     73.5  2.0e-15
  15x630x420  3.5          plane+child+strips -            3142513    3123987    0.97     69.2  2.2e-15
    18x18x18  2.3^2        plane+child+strips -               4903       6527    1.28     37.2  1.4e-15
    18x18x21  2.3^2        plane+child+strips -               5929       8219    1.27     36.5  1.3e-15
    18x21x18  2.3^2        plane+child+strips -               5735       9432    1.45     37.8  1.3e-15
   18x42x225  2.3^2        plane+child+strips -              83522     106100    1.15     88.5  1.7e-15
  18x150x420  2.3^2        plane+child+strips -             556537     510181    0.87    102.5  2.1e-15
   18x225x42  2.3^2        plane+child+strips -              87726      95078    1.08     84.2  2.1e-15
  18x420x150  2.3^2        plane+child+strips -             532300     655925    1.11    107.1  2.3e-15
    21x18x18  3.7          plane+child+strips -               5590       9313    1.53     38.7  1.4e-15
    21x21x21  3.7          plane+child+strips -               7794      17022    2.16     39.1  1.2e-15
    21x21x24  3.7          plane+child+strips -               6642      12392    1.83     53.3  9.7e-16
    21x24x21  3.7          plane+child+strips -               8565      14412    1.51     41.3  1.3e-15
   21x35x225  3.7          plane+child+strips -              80758     126302    1.44     88.7  1.6e-15
  21x210x525  3.7          plane+child+strips -            1760875    2273912    1.23     69.5  2.3e-15
   21x225x35  3.7          plane+child+strips -              96229     138483    1.37     74.5  1.8e-15
  21x525x210  3.7          plane+child+strips -            1611950    1764069    1.05     75.9  2.2e-15
    24x21x21  2^3.3        plane+child+strips -               8333      15148    1.77     42.5  1.2e-15
    24x42x42  2^3.3        plane+child+strips -              24226      30821    1.20     67.1  1.3e-15
   24x70x315  2^3.3        plane+child+strips -             295100     318194    1.05     85.2  1.8e-15
   24x75x315  2^3.3        plane+child+strips -             262662     344031    1.15    103.1  1.8e-15
   24x315x70  2^3.3        plane+child+strips -             242725     299587    1.20    103.6  1.8e-15
   24x315x75  2^3.3        plane+child+strips -             329538     412187    1.21     82.2  2.1e-15
   35x10x150  5.7          plane+child+strips -              23036      26883    1.15     89.3  1.4e-15
    35x35x35  5.7          plane+child+strips -              24661      29973    1.14     66.9  1.2e-15
    35x35x42  5.7          plane+child+strips -              28186      29472    1.04     71.4  1.4e-15
    35x42x35  5.7          plane+child+strips -              28588      34788    1.14     70.4  1.3e-15
    35x42x42  5.7          plane+child+strips -              32822      35279    1.03     74.8  1.3e-15
   35x75x420  5.7          plane+child+strips -             506513     509138    0.96    109.2  1.7e-15
   35x150x10  5.7          plane+child+strips -              24401      40969    1.66     84.3  1.3e-15
   35x420x75  5.7          plane+child+strips -             637812     852094    1.31     86.7  1.9e-15
    42x35x35  2.3.7        plane+child+strips -              29013      35426    1.14     69.4  1.5e-15
    42x35x42  2.3.7        plane+child+strips -              32812      35595    1.03     74.9  1.4e-15
    42x42x35  2.3.7        plane+child+strips -              34311      42218    1.23     71.6  1.5e-15
  42x105x525  2.3.7        plane+child+strips -            1677137    1700856    0.95     73.0  1.9e-15
  42x150x630  2.3.7        plane+child+strips -            2976738    2682281    0.89     73.1  2.4e-15
  42x525x105  2.3.7        plane+child+strips -            1556300    1963550    1.24     78.6  2.1e-15
  42x630x150  2.3.7        plane+child+strips -            3025750    2983881    0.97     71.9  2.2e-15
    45x45x54  3^2.5        plane+child+strips -              54256      60616    1.11     84.3  1.8e-15
    45x54x45  3^2.5        plane+child+strips -              56125      78782    1.33     81.5  1.7e-15
    45x54x54  3^2.5        plane+child+strips -              62783      73183    1.04     88.8  2.0e-15
    45x75x75  3^2.5        plane+child+strips -             139860     165566    1.15     81.2  1.7e-15
  45x135x630  3^2.5        plane+child+strips -            2919150    2619768    0.88     71.7  2.7e-15
  45x630x135  3^2.5        plane+child+strips -            3027975    3948831    1.21     69.1  2.2e-15
    54x45x45  2.3^3        plane+child+strips -              54389      78428    1.42     84.1  1.9e-15
    54x45x54  2.3^3        plane+child+strips -              63450      72698    1.13     87.9  2.1e-15
    54x54x45  2.3^3        plane+child+strips -              66797      92805    1.34     83.5  2.0e-15
    54x54x54  2.3^3        plane+child+strips -              72324      84806    1.16     94.0  2.1e-15
   75x10x210  3.5^2        plane+child+strips -              61172      72578    1.16    111.1  1.5e-15
   75x24x315  3.5^2        plane+child+strips -             270937     390000    1.43    100.0  1.7e-15
  75x105x105  3.5^2        plane+child+strips -             469938     563475    1.16     86.5  2.2e-15
   75x210x10  3.5^2        plane+child+strips -              72280     122092    1.66     94.1  1.5e-15
   75x315x24  3.5^2        plane+child+strips -             296063     365175    1.18     91.5  1.6e-15
 100x100x100  2^2.5^2      plane+child+strips -             461113     479012    1.00    108.1  1.9e-15
 100x100x105  2^2.5^2      plane+child+strips -             567837     692381    1.16     92.5  1.6e-15
 100x105x100  2^2.5^2      plane+child+strips -             455913     545225    1.07    115.2  1.8e-15
 100x105x105  2^2.5^2      plane+child+strips -             584500     759925    1.28     94.7  1.9e-15
 105x100x100  3.5.7        plane+child+strips -             475450     539350    1.02    110.4  1.8e-15
 105x100x105  3.5.7        plane+child+strips -             611275     743362    1.10     90.5  1.9e-15
 105x105x100  3.5.7        plane+child+strips -             489437     585493    1.18    113.0  2.0e-15
    135x6x42  3^3.5        plane+child+strips -              26126      32484    1.23     49.0  1.4e-15
    135x6x45  3^3.5        plane+child+strips -              23407      38953    1.57     59.0  1.5e-15
    135x9x35  3^3.5        plane+child+strips -              27250      38169    1.40     60.0  1.5e-15
   135x15x18  3^3.5        plane+child+strips -              27067      28477    1.00     51.0  1.7e-15
 135x135x135  3^3.5        plane+child+strips -            1676587    1858231    1.09     77.9  2.3e-15
 135x135x150  3^3.5        plane+child+strips -            1875250    1560025    0.80     77.9  2.3e-15
 135x150x135  3^3.5        plane+child+strips -            1907738    2095406    1.01     76.6  2.3e-15
 135x150x150  3^3.5        plane+child+strips -            2127713    1846337    0.84     76.9  2.3e-15
    150x6x54  2.3.5^2      plane+child+strips -              29766      38613    1.18     63.5  1.6e-15
    150x9x35  2.3.5^2      plane+child+strips -              29894      41436    1.33     61.4  1.3e-15
   150x10x35  2.3.5^2      plane+child+strips -              34375      41060    1.18     59.9  1.3e-15
   150x35x10  2.3.5^2      plane+child+strips -              33067      52094    1.47     62.2  1.1e-15
 150x135x135  2.3.5^2      plane+child+strips -            1925200    2117437    1.06     75.9  2.2e-15
 150x135x150  2.3.5^2      plane+child+strips -            2003013    1813506    0.85     81.6  2.3e-15
 150x150x135  2.3.5^2      plane+child+strips -            2220763    2386694    1.04     73.6  2.3e-15
 150x150x150  2.3.5^2      plane+child+strips -            2425363    2006368    0.81     75.4  2.2e-15
    210x9x75  2.3.5.7      plane+child+strips -              80525     115220    1.25     75.3  1.9e-15
   210x10x70  2.3.5.7      plane+child+strips -              73926      83116    0.99     85.3  1.5e-15
   210x10x75  2.3.5.7      plane+child+strips -              87740     116970    1.33     77.5  1.8e-15
   210x75x10  2.3.5.7      plane+child+strips -              84108     158592    1.86     80.8  1.5e-15
   225x6x135  3^2.5^2      plane+child+strips -              98586     148169    1.45     80.8  1.8e-15
   225x15x54  3^2.5^2      plane+child+strips -              94348     105721    1.09     84.4  1.9e-15
   225x18x42  3^2.5^2      plane+child+strips -             108735     107989    0.89     68.0  1.7e-15
   225x21x35  3^2.5^2      plane+child+strips -              98008     154987    1.41     73.1  1.6e-15
   300x9x420  2^2.3.5^2    plane+child+strips -             480550     660174    1.36    118.7  1.9e-15
  300x21x630  2^2.3.5^2    plane+child+strips -            3059250    3489262    1.11     71.1  2.1e-15
   300x420x9  2^2.3.5^2    plane+child+strips -             815037    1381019    1.67     70.0  2.2e-15
  300x630x21  2^2.3.5^2    plane+child+strips -            3842763    4696056    1.21     56.6  2.0e-15
   315x9x225  3^2.5.7      plane+child+strips -             314938     505093    1.54     97.6  2.0e-15
   315x24x70  3^2.5.7      plane+child+strips -             261700     313794    1.18     96.1  1.7e-15
   315x24x75  3^2.5.7      plane+child+strips -             329350     425093    1.24     82.3  1.8e-15
   315x75x24  3^2.5.7      plane+child+strips -             275763     374475    1.26     98.2  1.6e-15
   420x9x300  2^2.3.5.7    plane+child+strips -             474000     716006    1.49    120.3  2.3e-15
  420x18x150  2^2.3.5.7    plane+child+strips -             581875     644337    1.08     98.0  2.0e-15
   420x35x75  2^2.3.5.7    plane+child+strips -             647600     790969    1.16     85.4  1.9e-15
   420x300x9  2^2.3.5.7    plane+child+strips -             820712    1280737    1.47     69.5  2.1e-15
  525x10x420  3.5^2.7      plane+child+strips -            1410513    1431174    0.91     82.4  1.8e-15
  525x15x315  3.5^2.7      plane+child+strips -            1941650    1914543    0.95     67.8  2.1e-15
  525x21x210  3.5^2.7      plane+child+strips -            1573013    1887556    1.02     77.8  2.2e-15
  525x42x105  3.5^2.7      plane+child+strips -            1755775    1839625    0.98     69.7  2.1e-15
  630x15x420  2.3^2.5.7    plane+child+strips -            3332625    2788100    0.82     65.3  2.1e-15
  630x42x150  2.3^2.5.7    plane+child+strips -            3342550    2887806    0.84     65.1  2.1e-15
  630x45x135  2.3^2.5.7    plane+child+strips -            3211925    3377637    1.02     65.1  2.1e-15
  630x300x21  2.3^2.5.7    plane+child+strips -            4026125    4351456    1.08     54.0  2.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+child+strips             144      0     20   0.96   1.19   1.66    1.23
 child                           16      1      1   1.16   1.96   3.66    1.95
 ALL                            160      1     21   0.97   1.21   1.86    1.29
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     80      0     11   0.97   1.22   1.89    1.30
 odd column                      80      1     10   0.97   1.20   1.83    1.28
 ALL                            160      1     21   0.97   1.21   1.86    1.29
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  96      0     20   0.89   1.14   1.45    1.14
 4097..65536                     48      0      0   1.14   1.47   1.83    1.44
 1025..4096                       8      0      0   1.16   2.01   3.21    1.97
 257..1024                        8      1      1   0.49   1.94   3.74    1.92
 ALL                            160      1     21   0.97   1.21   1.86    1.29
```


worst 10: 9x9x9 (child 0.49), 135x135x150 (plane+child+strips 0.80), 150x150x150 (plane+child+strips 0.81), 630x15x420 (plane+child+strips 0.82), 135x150x150 (plane+child+strips 0.84), 630x42x150 (plane+child+strips 0.84), 150x135x150 (plane+child+strips 0.85), 10x525x420 (plane+child+strips 0.86), 18x150x420 (plane+child+strips 0.87), 45x135x630 (plane+child+strips 0.88)
best 5: 9x9x10 (child 3.74), 9x10x10 (child 3.66), 10x9x10 (child 3.59), 6x9x24 (child 3.21), 9x6x24 (child 3.19)
