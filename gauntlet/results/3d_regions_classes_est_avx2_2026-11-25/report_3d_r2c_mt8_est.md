# gauntlet report (3D)

run: `3d_regions_classes_est_avx2_2026-11-25`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_mt8_est  cells: 160 listed, 160 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 6 readings, 1.054..1.810

threaded plans that ran serial (engaged = 0 at both flips): 17: 10x10x10, 10x10x9, 10x9x10, 10x9x9, 6x10x70, 6x6x35, 6x6x42, 6x9x21, 6x9x24, 9x10x10, 9x10x9, 9x6x21, 9x6x24, 9x9x10, 9x9x15, 9x9x18, 9x9x9


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      6x6x35  2.3          child  -               1527       2366    1.52     21.2  5.2e-16
      6x6x42  2.3          child  -               1493       2497    1.65     26.7  5.7e-16
      6x9x21  2.3          child  -               1729       4885    2.77     16.6  4.2e-16
      6x9x24  2.3          child  -               1197       4514    3.71     28.0  4.4e-16
     6x10x70  2.3          child  -               3643       6257    1.66     34.7  4.1e-16
     6x15x70  2.3          plane+child+strips -               3825       7089    1.78     52.0  5.7e-16
     6x15x75  2.3          plane+child+strips -               4323       7507    1.66     49.7  7.0e-16
    6x42x135  2.3          plane+child+strips -              16917      24202    1.23     75.7  5.9e-16
    6x45x135  2.3          plane+child+strips -              17594      25699    1.40     78.5  6.4e-16
    6x54x150  2.3          plane+child+strips -              21282      27088    1.27     88.9  6.2e-16
    6x135x42  2.3          plane+child+strips -              17044      26965    1.43     75.1  5.9e-16
    6x135x45  2.3          plane+child+strips -              21672      29523    1.31     63.7  7.0e-16
   6x135x225  2.3          plane+child+strips -              69129      95481    1.18    115.2  7.1e-16
    6x150x54  2.3          plane+child+strips -              21377      31250    1.41     88.5  6.2e-16
   6x225x135  2.3          plane+child+strips -              71276      94459    1.12    111.7  8.2e-16
      9x6x21  3^2          child  -               1679       5035    2.92     17.1  5.4e-16
      9x6x24  3^2          child  -               1192       4589    3.84     28.1  5.0e-16
       9x9x9  3^2          child  -                890        561    0.62     19.5  4.7e-16
      9x9x10  3^2          child  -                489       1728    3.43     40.0  4.3e-16
      9x9x15  3^2          child  -               1649       1534    0.89     18.9  5.5e-16
      9x9x18  3^2          child  -               1961       2193    1.11     19.5  6.8e-16
      9x9x70  3^2          plane+child+strips -               3420       3758    1.04     51.7  5.8e-16
      9x9x75  3^2          plane+child+strips -               4394       4234    0.91     43.4  6.1e-16
      9x10x9  3^2          child  -                927       1692    1.74     21.1  5.1e-16
     9x10x10  3^2          child  -                513       1550    2.95     43.1  4.6e-16
     9x10x70  3^2          plane+child+strips -               3896       4038    1.01     51.0  4.6e-16
     9x10x75  3^2          plane+child+strips -               4644       4183    0.87     46.2  5.0e-16
    9x35x135  3^2          plane+child+strips -              21926      26578    1.17     74.6  5.4e-16
    9x35x150  3^2          plane+child+strips -              22518      22643    0.98     81.5  6.4e-16
    9x75x210  3^2          plane+child+strips -              55400      52852    0.95    109.5  6.2e-16
    9x135x35  3^2          plane+child+strips -              26855      29759    1.10     60.9  6.8e-16
    9x150x35  3^2          plane+child+strips -              30299      31055    0.94     60.5  5.0e-16
    9x210x75  3^2          plane+child+strips -              67179      62671    0.80     90.3  7.0e-16
   9x225x315  3^2          plane+child+strips -             272225     289993    1.05    113.0  8.2e-16
   9x300x420  3^2          plane+child+strips -             421550     485356    1.13    135.3  6.4e-16
   9x315x225  3^2          plane+child+strips -             294113     328431    1.09    104.6  6.6e-16
   9x420x300  3^2          plane+child+strips -             414650     498768    1.18    137.5  6.8e-16
      10x9x9  2.5          child  -                940       1736    1.78     20.8  4.9e-16
     10x9x10  2.5          child  -                539       1833    3.38     41.0  4.6e-16
     10x10x9  2.5          child  -               1014       1803    1.66     21.8  4.8e-16
    10x10x10  2.5          child  -                586        649    1.10     42.5  4.2e-16
    10x10x75  2.5          plane+child+strips -               4806       4311    0.88     50.2  4.3e-16
   10x35x150  2.5          plane+child+strips -              23405      23343    0.94     87.9  5.5e-16
   10x70x210  2.5          plane+child+strips -              51848      48431    0.92    121.7  5.3e-16
   10x75x210  2.5          plane+child+strips -              57396      66620    1.03    118.4  6.6e-16
   10x150x35  2.5          plane+child+strips -              30943      33053    1.04     66.5  6.4e-16
   10x210x70  2.5          plane+child+strips -              53919      52105    0.92    117.0  7.1e-16
   10x210x75  2.5          plane+child+strips -              69020      62754    0.90     98.5  6.7e-16
  10x420x525  2.5          plane+child+strips -             983200    1293531    1.19    118.1  6.7e-16
  10x525x420  2.5          plane+child+strips -             923675     992143    1.05    125.8  7.8e-16
   15x18x135  3.5          plane+child+strips -              15048      16219    0.97     91.8  6.8e-16
   15x54x225  3.5          plane+child+strips -              60567      68738    0.83    131.5  6.6e-16
   15x135x18  3.5          plane+child+strips -              16925      15927    0.92     81.6  7.2e-16
   15x225x54  3.5          plane+child+strips -              58043      55507    0.78    137.2  6.5e-16
  15x315x525  3.5          plane+child+strips -             816413    1006293    1.15    161.4  7.7e-16
  15x420x630  3.5          plane+child+strips -            1712425    1444462    0.83    127.0  8.0e-16
  15x525x315  3.5          plane+child+strips -             871825     876519    0.97    151.1  8.8e-16
  15x630x420  3.5          plane+child+strips -            1725613    1602150    0.93    126.0  7.3e-16
    18x18x18  2.3^2        plane+payonce+stages -               5355       3799    0.69     34.1  6.2e-16
    18x18x21  2.3^2        plane+payonce+stages -               6780       5903    0.83     31.9  5.4e-16
    18x21x18  2.3^2        plane+payonce+stages -               6459       5373    0.80     33.5  5.1e-16
   18x42x225  2.3^2        plane+payonce+stages -              49517      60360    1.01    149.2  6.7e-16
  18x150x420  2.3^2        plane+payonce+stages -             275013     318181    1.15    207.3  6.6e-16
   18x225x42  2.3^2        plane+payonce+stages -              68857      58830    0.76    107.3  7.6e-16
  18x420x150  2.3^2        plane+payonce+stages -             296012     344312    1.14    192.6  7.3e-16
    21x18x18  3.7          plane+payonce+stages -               5893       4632    0.69     36.8  5.0e-16
    21x21x21  3.7          plane+payonce+stages -               7539       8734    1.09     40.5  4.6e-16
    21x21x24  3.7          plane+payonce+stages -               6334       8674    1.27     55.9  4.3e-16
    21x24x21  3.7          plane+payonce+stages -               7881       8839    1.08     44.9  5.4e-16
   21x35x225  3.7          plane+payonce+stages -              44854      61752    1.27    159.8  6.1e-16
  21x210x525  3.7          plane+child+strips -             743013    1022856    1.36    164.7  7.1e-16
   21x225x35  3.7          plane+payonce+stages -              53542      55631    0.92    133.9  5.9e-16
  21x525x210  3.7          plane+child+strips -             654950     811944    1.22    186.8  7.1e-16
    24x21x21  2^3.3        plane+payonce+stages -               7273       8143    1.11     48.6  5.1e-16
    24x42x42  2^3.3        plane+payonce+stages -              18046      17375    0.90     90.1  5.7e-16
   24x70x315  2^3.3        plane+payonce+stages -             123088     146975    1.09    204.4  7.1e-16
   24x75x315  2^3.3        plane+payonce+stages -             147125     157981    1.03    184.1  6.6e-16
   24x315x70  2^3.3        plane+payonce+stages -             139250     216068    1.53    180.6  6.0e-16
   24x315x75  2^3.3        plane+payonce+stages -             173875     234612    1.05    155.8  6.5e-16
   35x10x150  5.7          plane+payonce+stages -              15399      19510    1.10    133.6  5.9e-16
    35x35x35  5.7          plane+payonce+stages -              21739      21013    0.94     75.9  4.8e-16
    35x35x42  5.7          plane+payonce+stages -              26488      22014    0.77     76.0  5.9e-16
    35x42x35  5.7          plane+payonce+stages -              22330      24661    1.10     90.2  5.3e-16
    35x42x42  5.7          plane+payonce+stages -              22327      28604    1.26    110.0  5.8e-16
   35x75x420  5.7          plane+payonce+stages -             238225     327368    1.18    232.2  6.6e-16
   35x150x10  5.7          plane+payonce+stages -              17776      26260    1.46    115.8  6.8e-16
   35x420x75  5.7          plane+payonce+stages -             326313     480262    1.10    169.5  8.1e-16
    42x35x35  2.3.7        plane+payonce+stages -              22077      22938    0.81     91.2  5.1e-16
    42x35x42  2.3.7        plane+payonce+stages -              26619      28452    1.03     92.3  5.4e-16
    42x42x35  2.3.7        plane+payonce+stages -              23502      32779    1.38    104.5  6.1e-16
  42x105x525  2.3.7        plane+child+strips -             735738     915575    1.21    166.3  7.4e-16
  42x150x630  2.3.7        plane+child+strips -            1326750    1472787    0.92    163.9  7.8e-16
  42x525x105  2.3.7        plane+child+strips -             686125     826431    1.10    178.4  7.1e-16
  42x630x150  2.3.7        plane+child+strips -            1323050    1534487    1.13    164.4  7.2e-16
    45x45x54  3^2.5        plane+payonce+stages -              37842      41836    0.92    120.9  6.6e-16
    45x54x45  3^2.5        plane+payonce+stages -              33111      56883    1.70    138.2  6.0e-16
    45x54x54  3^2.5        plane+payonce+stages -              41630      52518    1.25    134.0  5.9e-16
    45x75x75  3^2.5        plane+payonce+stages -              83073      96513    0.86    136.7  7.0e-16
  45x135x630  3^2.5        plane+child+strips -            1195513    1132368    0.94    175.0  8.9e-16
  45x630x135  3^2.5        plane+child+strips -            1213350    1481993    1.17    172.4  8.4e-16
    54x45x45  2.3^3        plane+payonce+stages -              35706      48617    1.14    128.2  7.4e-16
    54x45x54  2.3^3        plane+payonce+stages -              38040      47882    1.01    146.6  8.3e-16
    54x54x45  2.3^3        plane+payonce+stages -              37890      38475    0.94    147.2  7.1e-16
    54x54x54  2.3^3        plane+payonce+stages -              42156      64088    1.47    161.2  6.0e-16
   75x10x210  3.5^2        plane+payonce+stages -              35756      39570    1.05    190.1  6.2e-16
   75x24x315  3.5^2        plane+payonce+stages -             136763     166275    1.02    198.1  6.5e-16
  75x105x105  3.5^2        plane+payonce+stages -             259712     229100    0.78    156.5  7.3e-16
   75x210x10  3.5^2        plane+payonce+stages -              40348      45282    1.07    168.5  7.2e-16
   75x315x24  3.5^2        plane+payonce+stages -             162888     227006    1.19    166.3  6.5e-16
 100x100x100  2^2.5^2      plane+payonce+stages -             223763     237406    1.06    222.7  5.3e-16
 100x100x105  2^2.5^2      plane+payonce+stages -             284488     277368    0.97    184.6  5.4e-16
 100x105x100  2^2.5^2      plane+payonce+stages -             239413     253862    1.03    219.3  6.6e-16
 100x105x105  2^2.5^2      plane+payonce+stages -             290125     323112    0.92    190.7  6.9e-16
 105x100x100  3.5.7        plane+payonce+stages -             234963     268131    1.12    223.5  6.0e-16
 105x100x105  3.5.7        plane+payonce+stages -             332462     305649    0.89    166.4  7.2e-16
 105x105x100  3.5.7        plane+payonce+stages -             240588     288981    1.18    230.0  7.5e-16
    135x6x42  3^3.5        plane+payonce+stages -              14215      30703    2.15     90.1  6.0e-16
    135x6x45  3^3.5        plane+payonce+stages -              14083      30629    2.12     98.0  7.6e-16
    135x9x35  3^3.5        plane+payonce+stages -              15941      20778    1.21    102.5  6.0e-16
   135x15x18  3^3.5        plane+payonce+stages -              17258      19521    1.09     80.0  7.2e-16
 135x135x135  3^3.5        plane+child+strips -             725350     769988    1.05    180.0  8.2e-16
 135x135x150  3^3.5        plane+child+strips -             708050     774731    1.03    206.4  8.6e-16
 135x150x135  3^3.5        plane+child+strips -             804000     871899    1.06    181.8  7.7e-16
 135x150x150  3^3.5        plane+child+strips -             790025     897163    1.10    207.0  8.7e-16
    150x6x54  2.3.5^2      plane+payonce+stages -              17432      36398    1.93    108.5  6.7e-16
    150x9x35  2.3.5^2      plane+payonce+stages -              16727      22273    1.19    109.7  6.4e-16
   150x10x35  2.3.5^2      plane+payonce+stages -              20458      25764    1.07    100.6  5.1e-16
   150x35x10  2.3.5^2      plane+payonce+stages -              16891      24960    1.38    121.8  5.6e-16
 150x135x135  2.3.5^2      plane+child+strips -             787213     856331    0.96    185.6  7.6e-16
 150x135x150  2.3.5^2      plane+child+strips -             804925     906712    1.06    203.2  8.5e-16
 150x150x135  2.3.5^2      plane+child+strips -             899425    1006537    1.08    181.8  8.8e-16
 150x150x150  2.3.5^2      plane+child+strips -             910037    1128287    1.23    201.1  8.9e-16
    210x9x75  2.3.5.7      plane+payonce+stages -              41354      48437    1.00    146.6  7.3e-16
   210x10x70  2.3.5.7      plane+payonce+stages -              34544      51222    1.22    182.6  6.7e-16
   210x10x75  2.3.5.7      plane+payonce+stages -              41304      62798    1.15    164.6  8.0e-16
   210x75x10  2.3.5.7      plane+payonce+stages -              37708      51732    1.35    180.3  8.9e-16
   225x6x135  3^2.5^2      plane+payonce+stages -              57043     103171    1.73    139.6  6.6e-16
   225x15x54  3^2.5^2      plane+payonce+stages -              45324      53840    1.15    175.7  6.3e-16
   225x18x42  3^2.5^2      plane+payonce+stages -              48722      53024    1.06    151.7  6.0e-16
   225x21x35  3^2.5^2      plane+payonce+stages -              48658      72127    1.37    147.3  5.8e-16
   300x9x420  2^2.3.5^2    plane+payonce+stages -             229675     322687    1.38    248.3  8.7e-16
  300x21x630  2^2.3.5^2    plane+child+strips -            1211350    1647406    1.27    179.6  8.0e-16
   300x420x9  2^2.3.5^2    plane+payonce+stages -             366075     463069    1.16    155.8  9.0e-16
  300x630x21  2^2.3.5^2    plane+child+strips -            1460263    1775681    1.03    148.9  8.0e-16
   315x9x225  3^2.5.7      plane+payonce+stages -             151213     216987    1.29    203.4  7.6e-16
   315x24x70  3^2.5.7      plane+payonce+stages -             113800     201606    1.38    221.0  5.3e-16
   315x24x75  3^2.5.7      plane+payonce+stages -             140600     186019    1.23    192.7  6.1e-16
   315x75x24  3^2.5.7      plane+payonce+stages -             143975     187781    1.25    188.2  6.6e-16
   420x9x300  2^2.3.5.7    plane+payonce+stages -             225600     346812    1.44    252.8  6.6e-16
  420x18x150  2^2.3.5.7    plane+payonce+stages -             230950     307806    1.31    246.9  6.8e-16
   420x35x75  2^2.3.5.7    plane+payonce+stages -             308525     427981    1.14    179.3  8.2e-16
   420x300x9  2^2.3.5.7    plane+payonce+stages -             370375     491262    1.10    154.0  7.7e-16
  525x10x420  3.5^2.7      plane+child+strips -             642075     632987    0.98    180.9  7.0e-16
  525x15x315  3.5^2.7      plane+child+strips -             737262     852150    1.04    178.7  7.9e-16
  525x21x210  3.5^2.7      plane+child+strips -             681662     853293    1.19    179.5  6.6e-16
  525x42x105  3.5^2.7      plane+child+strips -             716325     745225    1.03    170.8  7.1e-16
  630x15x420  2.3^2.5.7    plane+child+strips -            1253875    1345618    1.05    173.5  7.5e-16
  630x42x150  2.3^2.5.7    plane+child+strips -            1320087    1476094    1.07    164.8  8.9e-16
  630x45x135  2.3^2.5.7    plane+child+strips -            1307088    1553812    1.08    160.1  9.2e-16
  630x300x21  2.3^2.5.7    plane+child+strips -            1587938    1843475    1.08    137.0  8.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+payonce+stages            76      6     17   0.83   1.11   1.46    1.13
 plane+child+strips              67      1     22   0.90   1.05   1.31    1.07
 child                           17      1      2   0.89   1.74   3.71    1.90
 ALL                            160      8     41   0.89   1.10   1.66    1.17
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     80      3     16   0.92   1.12   1.66    1.19
 odd column                      80      5     25   0.86   1.09   1.73    1.14
 ALL                            160      8     41   0.89   1.10   1.66    1.17
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  96      3     23   0.92   1.08   1.35    1.09
 4097..65536                     48      4     16   0.81   1.09   1.66    1.13
 1025..4096                       8      0      1   0.89   2.21   3.84    2.03
 257..1024                        8      1      1   0.62   1.76   3.43    1.82
 ALL                            160      8     41   0.89   1.10   1.66    1.17
```


worst 10: 9x9x9 (child 0.62), 21x18x18 (plane+payonce+stages 0.69), 18x18x18 (plane+payonce+stages 0.69), 18x225x42 (plane+payonce+stages 0.76), 35x35x42 (plane+payonce+stages 0.77), 15x225x54 (plane+child+strips 0.78), 75x105x105 (plane+payonce+stages 0.78), 18x21x18 (plane+payonce+stages 0.80), 9x210x75 (plane+child+strips 0.80), 42x35x35 (plane+payonce+stages 0.81)
best 5: 9x6x24 (child 3.84), 6x9x24 (child 3.71), 9x9x10 (child 3.43), 10x9x10 (child 3.38), 9x10x10 (child 2.95)
