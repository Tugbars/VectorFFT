# gauntlet report (3D)

run: `3d_regions_classes_est_avx2_2026-11-25`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_mt4_est  cells: 160 listed, 160 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 6 readings, 1.024..1.140

threaded plans that ran serial (engaged = 0 at both flips): 16: 10x10x10, 10x10x9, 10x9x10, 10x9x9, 6x6x35, 6x6x42, 6x9x21, 6x9x24, 9x10x10, 9x10x9, 9x6x21, 9x6x24, 9x9x10, 9x9x15, 9x9x18, 9x9x9


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      6x6x35  2.3          child  -               1534       2989    1.85     21.1  5.2e-16
      6x6x42  2.3          child  -               1542       3245    2.06     25.9  5.7e-16
      6x9x21  2.3          child  -               1777       3293    1.82     16.2  4.2e-16
      6x9x24  2.3          child  -               1188       2981    2.34     28.2  4.4e-16
     6x10x70  2.3          plane+child+strips -               3495       5250    1.43     36.2  4.1e-16
     6x15x70  2.3          plane+child+strips -               5012       6795    1.32     39.7  5.7e-16
     6x15x75  2.3          plane+child+strips -               5759       7446    1.19     37.3  7.0e-16
    6x42x135  2.3          plane+child+strips -              23574      34722    1.37     54.3  5.9e-16
    6x45x135  2.3          plane+child+strips -              33383      36960    1.05     41.4  6.4e-16
    6x54x150  2.3          plane+child+strips -              33129      39923    1.15     57.1  6.2e-16
    6x135x42  2.3          plane+child+strips -              26989      36915    1.31     47.4  5.9e-16
    6x135x45  2.3          plane+child+strips -              35098      43938    1.24     39.3  7.0e-16
   6x135x225  2.3          plane+child+strips -             112119     155412    1.37     71.0  7.1e-16
    6x150x54  2.3          plane+child+strips -              33649      47208    1.35     56.2  6.2e-16
   6x225x135  2.3          plane+child+strips -             108119     163009    1.35     73.6  8.2e-16
      9x6x21  3^2          child  -               1718       3338    1.91     16.7  5.4e-16
      9x6x24  3^2          child  -               1199       2988    2.35     27.9  5.0e-16
       9x9x9  3^2          child  -                890        561    0.62     19.5  4.7e-16
      9x9x10  3^2          child  -                492       1562    3.07     39.7  4.3e-16
      9x9x15  3^2          child  -               1653       1684    0.97     18.8  5.5e-16
      9x9x18  3^2          child  -               1975       2019    0.99     19.4  6.8e-16
      9x9x70  3^2          plane+child+strips -               4339       4711    1.05     40.7  5.8e-16
      9x9x75  3^2          plane+child+strips -               5242       5169    0.96     36.4  6.1e-16
      9x10x9  3^2          child  -                926       1572    1.67     21.1  5.1e-16
     9x10x10  3^2          child  -                508       1559    2.95     43.4  4.6e-16
     9x10x70  3^2          plane+child+strips -               4760       5194    1.07     41.8  4.6e-16
     9x10x75  3^2          plane+child+strips -               5885       5605    0.91     36.5  5.0e-16
    9x35x135  3^2          plane+child+strips -              34723      34230    0.90     47.1  5.4e-16
    9x35x150  3^2          plane+child+strips -              31343      30864    0.97     58.5  6.4e-16
    9x75x210  3^2          plane+child+strips -              74329      70543    0.95     81.6  6.2e-16
    9x135x35  3^2          plane+child+strips -              37976      37787    0.94     43.0  6.8e-16
    9x150x35  3^2          plane+child+strips -              42111      44236    1.01     43.6  5.0e-16
    9x210x75  3^2          plane+child+strips -              89521      87205    0.97     67.7  7.0e-16
   9x225x315  3^2          plane+child+strips -             381713     437056    1.13     80.6  8.2e-16
   9x300x420  3^2          plane+child+strips -             621925     735956    1.17     91.7  6.4e-16
   9x315x225  3^2          plane+child+strips -             418300     500668    1.04     73.5  6.6e-16
   9x420x300  3^2          plane+child+strips -             603275     742837    1.21     94.5  6.8e-16
      10x9x9  2.5          child  -                937       1744    1.85     20.9  4.9e-16
     10x9x10  2.5          child  -                541       1638    3.02     40.8  4.6e-16
     10x10x9  2.5          child  -               1006       1660    1.62     21.9  4.8e-16
    10x10x10  2.5          child  -                585        649    1.11     42.6  4.2e-16
    10x10x75  2.5          plane+child+strips -               5894       5590    0.92     41.0  4.3e-16
   10x35x150  2.5          plane+child+strips -              32462      31630    0.95     63.4  5.5e-16
   10x70x210  2.5          plane+child+strips -              67022      65524    0.97     94.1  5.3e-16
   10x75x210  2.5          plane+child+strips -              77348      70748    0.90     87.9  6.6e-16
   10x150x35  2.5          plane+child+strips -              42630      43770    0.93     48.3  6.4e-16
   10x210x70  2.5          plane+child+strips -              70944      67611    0.93     88.9  7.1e-16
   10x210x75  2.5          plane+child+strips -              94996      87698    0.76     71.6  6.7e-16
  10x420x525  2.5          plane+child+strips -            1586825    2039350    1.19     73.2  6.7e-16
  10x525x420  2.5          plane+child+strips -            1409063    1519469    1.06     82.4  7.8e-16
   15x18x135  3.5          plane+child+strips -              26430      22817    0.70     52.2  6.8e-16
   15x54x225  3.5          plane+child+strips -             103024      95300    0.84     77.3  6.6e-16
   15x135x18  3.5          plane+child+strips -              29968      25770    0.79     46.1  7.2e-16
   15x225x54  3.5          plane+child+strips -              88329      82552    0.88     90.1  6.5e-16
  15x315x525  3.5          plane+child+strips -            1603587    2013168    1.13     82.2  7.7e-16
  15x420x630  3.5          plane+child+strips -            2345137    2841056    1.15     92.7  8.0e-16
  15x525x315  3.5          plane+child+strips -            1634375    1824594    1.07     80.6  8.8e-16
  15x630x420  3.5          plane+child+strips -            2605887    2974213    1.12     83.5  7.3e-16
    18x18x18  2.3^2        plane+payonce+stages -               8029       4611    0.53     22.7  6.2e-16
    18x18x21  2.3^2        plane+payonce+stages -               9311       7647    0.80     23.3  5.4e-16
    18x21x18  2.3^2        plane+payonce+stages -               9324       7308    0.73     23.2  5.1e-16
   18x42x225  2.3^2        plane+payonce+stages -              86378      90391    1.04     85.5  6.7e-16
  18x150x420  2.3^2        plane+payonce+stages -             514775     520099    1.00    110.8  6.6e-16
   18x225x42  2.3^2        plane+payonce+stages -             118122      77954    0.64     62.6  7.6e-16
  18x420x150  2.3^2        plane+payonce+stages -             488162     554025    1.11    116.8  7.3e-16
    21x18x18  3.7          plane+payonce+stages -               7935       5796    0.67     27.3  5.0e-16
    21x21x21  3.7          plane+payonce+stages -              10491      13044    1.22     29.1  4.6e-16
    21x21x24  3.7          plane+payonce+stages -               9216      10563    1.09     38.4  4.3e-16
    21x24x21  3.7          plane+payonce+stages -              10891      12475    1.12     32.5  5.4e-16
   21x35x225  3.7          plane+payonce+stages -              73958     105819    1.30     96.9  6.1e-16
  21x210x525  3.7          plane+child+strips -            1575875    1890487    1.12     77.7  7.1e-16
   21x225x35  3.7          plane+payonce+stages -             100713     106790    0.91     71.2  5.9e-16
  21x525x210  3.7          plane+child+strips -            1452488    1639637    1.10     84.3  7.1e-16
    24x21x21  2^3.3        plane+payonce+stages -               7880      12719    1.46     44.9  5.1e-16
    24x42x42  2^3.3        plane+payonce+stages -              20730      27632    1.29     78.5  5.7e-16
   24x70x315  2^3.3        plane+payonce+stages -             231313     288800    1.16    108.7  7.1e-16
   24x75x315  2^3.3        plane+payonce+stages -             250400     310700    1.11    108.2  6.6e-16
   24x315x70  2^3.3        plane+payonce+stages -             215125     249387    1.10    116.9  6.0e-16
   24x315x75  2^3.3        plane+payonce+stages -             272600     332449    1.21     99.4  6.5e-16
   35x10x150  5.7          plane+payonce+stages -              26041      22813    0.84     79.0  5.9e-16
    35x35x35  5.7          plane+payonce+stages -              27569      24291    0.87     59.8  4.8e-16
    35x35x42  5.7          plane+payonce+stages -              30016      34382    1.07     67.1  5.9e-16
    35x42x35  5.7          plane+payonce+stages -              27332      28567    1.03     73.7  5.3e-16
    35x42x42  5.7          plane+payonce+stages -              33200      29651    0.88     74.0  5.8e-16
   35x75x420  5.7          plane+payonce+stages -             452550     513419    1.10    122.2  6.6e-16
   35x150x10  5.7          plane+payonce+stages -              21595      26552    1.21     95.3  6.8e-16
   35x420x75  5.7          plane+payonce+stages -             536150     643681    1.19    103.2  8.1e-16
    42x35x35  2.3.7        plane+payonce+stages -              29851      28698    0.86     67.4  5.1e-16
    42x35x42  2.3.7        plane+payonce+stages -              33878      29938    0.86     72.5  5.4e-16
    42x42x35  2.3.7        plane+payonce+stages -              31889      33905    1.04     77.0  6.1e-16
  42x105x525  2.3.7        plane+child+strips -            1414388    1693075    1.13     86.5  7.4e-16
  42x150x630  2.3.7        plane+child+strips -            2327113    2372768    1.01     93.5  7.8e-16
  42x525x105  2.3.7        plane+child+strips -            1428962    1569050    1.02     85.6  7.1e-16
  42x630x150  2.3.7        plane+child+strips -            2367325    2592900    1.05     91.9  7.2e-16
    45x45x54  3^2.5        plane+payonce+stages -              49808      57548    1.13     91.9  6.6e-16
    45x54x45  3^2.5        plane+payonce+stages -              51758      60521    1.09     88.4  6.0e-16
    45x54x54  3^2.5        plane+payonce+stages -              57790      67090    1.14     96.5  5.9e-16
    45x75x75  3^2.5        plane+payonce+stages -             117127     132560    1.13     97.0  7.0e-16
  45x135x630  3^2.5        plane+child+strips -            2233425    2294281    1.00     93.7  8.9e-16
  45x630x135  3^2.5        plane+child+strips -            2193375    2860637    1.22     95.4  8.4e-16
    54x45x45  2.3^3        plane+payonce+stages -              51303      59326    0.95     89.2  7.4e-16
    54x45x54  2.3^3        plane+payonce+stages -              55827      66738    1.16     99.9  8.3e-16
    54x54x45  2.3^3        plane+payonce+stages -              61267      71496    1.10     91.0  7.1e-16
    54x54x54  2.3^3        plane+payonce+stages -              66364      79328    1.18    102.4  6.0e-16
   75x10x210  3.5^2        plane+payonce+stages -              55168      65012    1.15    123.2  6.2e-16
   75x24x315  3.5^2        plane+payonce+stages -             288075     315181    1.05     94.0  6.5e-16
  75x105x105  3.5^2        plane+payonce+stages -             462050     424987    0.80     87.9  7.3e-16
   75x210x10  3.5^2        plane+payonce+stages -              58772      76612    1.22    115.7  7.2e-16
   75x315x24  3.5^2        plane+payonce+stages -             277738     356044    1.10     97.5  6.5e-16
 100x100x100  2^2.5^2      plane+payonce+stages -             433138     433962    1.00    115.0  5.3e-16
 100x100x105  2^2.5^2      plane+payonce+stages -             563250     516537    0.84     93.2  5.4e-16
 100x105x100  2^2.5^2      plane+payonce+stages -             463525     464700    0.97    113.3  6.6e-16
 100x105x105  2^2.5^2      plane+payonce+stages -             603212     566425    0.91     91.7  6.9e-16
 105x100x100  3.5.7        plane+payonce+stages -             408750     477269    1.02    128.5  6.0e-16
 105x100x105  3.5.7        plane+payonce+stages -             661663     567287    0.83     83.6  7.2e-16
 105x105x100  3.5.7        plane+payonce+stages -             498487     529362    1.04    111.0  7.5e-16
    135x6x42  3^3.5        plane+payonce+stages -              23615      37866    1.54     54.2  6.0e-16
    135x6x45  3^3.5        plane+payonce+stages -              22980      41289    1.51     60.1  7.6e-16
    135x9x35  3^3.5        plane+payonce+stages -              24835      29344    1.07     65.8  6.0e-16
   135x15x18  3^3.5        plane+payonce+stages -              29217      25636    0.87     47.3  7.2e-16
 135x135x135  3^3.5        plane+child+strips -            1447100    1551437    1.02     90.2  8.2e-16
 135x135x150  3^3.5        plane+child+strips -            1534588    1520106    0.98     95.2  8.6e-16
 135x150x135  3^3.5        plane+child+strips -            1703938    1778043    0.96     85.8  7.7e-16
 135x150x150  3^3.5        plane+child+strips -            1567300    1757569    0.96    104.3  8.7e-16
    150x6x54  2.3.5^2      plane+payonce+stages -              21930      48461    2.10     86.3  6.7e-16
    150x9x35  2.3.5^2      plane+payonce+stages -              26565      31182    1.08     69.0  6.4e-16
   150x10x35  2.3.5^2      plane+payonce+stages -              29092      31872    1.05     70.7  5.1e-16
   150x35x10  2.3.5^2      plane+payonce+stages -              23041      33182    1.18     89.3  5.6e-16
 150x135x135  2.3.5^2      plane+child+strips -            1605463    1791894    0.98     91.0  7.6e-16
 150x135x150  2.3.5^2      plane+child+strips -            1739538    1793194    0.98     94.0  8.5e-16
 150x150x135  2.3.5^2      plane+child+strips -            1760800    2033156    1.02     92.9  8.8e-16
 150x150x150  2.3.5^2      plane+child+strips -            1709537    2039162    1.13    107.0  8.9e-16
    210x9x75  2.3.5.7      plane+payonce+stages -              72311      80291    1.02     83.9  7.3e-16
   210x10x70  2.3.5.7      plane+payonce+stages -              53193      69892    1.23    118.6  6.7e-16
   210x10x75  2.3.5.7      plane+payonce+stages -              73580      83406    0.96     92.4  8.0e-16
   210x75x10  2.3.5.7      plane+payonce+stages -              63664      89934    1.15    106.8  8.9e-16
   225x6x135  3^2.5^2      plane+payonce+stages -              83400     151983    1.54     95.5  6.6e-16
   225x15x54  3^2.5^2      plane+payonce+stages -              69852      89774    1.24    114.0  6.3e-16
   225x18x42  3^2.5^2      plane+payonce+stages -              73217      86365    1.09    100.9  6.0e-16
   225x21x35  3^2.5^2      plane+payonce+stages -              86567     126179    1.40     82.8  5.8e-16
   300x9x420  2^2.3.5^2    plane+payonce+stages -             510425     581819    1.09    111.7  8.7e-16
  300x21x630  2^2.3.5^2    plane+child+strips -            2275850    3328450    1.44     95.6  8.0e-16
   300x420x9  2^2.3.5^2    plane+payonce+stages -             704275     813993    1.08     81.0  9.0e-16
  300x630x21  2^2.3.5^2    plane+child+strips -            2947775    3374637    0.96     73.8  8.0e-16
   315x9x225  3^2.5.7      plane+payonce+stages -             329362     404644    1.17     93.4  7.6e-16
   315x24x70  3^2.5.7      plane+payonce+stages -             191275     284012    1.45    131.5  5.3e-16
   315x24x75  3^2.5.7      plane+payonce+stages -             263625     338593    1.25    102.8  6.1e-16
   315x75x24  3^2.5.7      plane+payonce+stages -             253275     327856    1.07    107.0  6.6e-16
   420x9x300  2^2.3.5.7    plane+payonce+stages -             478812     595175    1.24    119.1  6.6e-16
  420x18x150  2^2.3.5.7    plane+payonce+stages -             487750     551106    1.11    116.9  6.8e-16
   420x35x75  2^2.3.5.7    plane+payonce+stages -             507063     651331    1.09    109.1  8.2e-16
   420x300x9  2^2.3.5.7    plane+payonce+stages -             692638     760525    1.05     82.3  7.7e-16
  525x10x420  3.5^2.7      plane+child+strips -            1247675    1294187    1.01     93.1  7.0e-16
  525x15x315  3.5^2.7      plane+child+strips -            1503925    1747537    1.05     87.6  7.9e-16
  525x21x210  3.5^2.7      plane+child+strips -            1367200    1796674    1.15     89.5  6.6e-16
  525x42x105  3.5^2.7      plane+child+strips -            1562000    1520981    0.89     78.3  7.1e-16
  630x15x420  2.3^2.5.7    plane+child+strips -            2562050    2662862    1.01     84.9  7.5e-16
  630x42x150  2.3^2.5.7    plane+child+strips -            2317187    2669400    0.99     93.9  8.9e-16
  630x45x135  2.3^2.5.7    plane+child+strips -            2653712    2860775    1.01     78.8  9.2e-16
  630x300x21  2.3^2.5.7    plane+child+strips -            3292675    3478231    1.00     66.1  8.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+payonce+stages            76      4     21   0.84   1.09   1.30    1.07
 plane+child+strips              68      3     27   0.90   1.02   1.32    1.05
 child                           16      1      3   0.97   1.85   3.02    1.73
 ALL                            160      8     51   0.87   1.08   1.51    1.11
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     80      4     25   0.90   1.09   1.62    1.13
 odd column                      80      4     26   0.87   1.07   1.51    1.10
 ALL                            160      8     51   0.87   1.08   1.51    1.11
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  96      2     28   0.91   1.08   1.24    1.07
 4097..65536                     48      5     20   0.79   1.05   1.43    1.04
 1025..4096                       8      0      2   0.97   1.88   2.35    1.70
 257..1024                        8      1      1   0.62   1.76   3.07    1.77
 ALL                            160      8     51   0.87   1.08   1.51    1.11
```


worst 10: 18x18x18 (plane+payonce+stages 0.53), 9x9x9 (child 0.62), 18x225x42 (plane+payonce+stages 0.64), 21x18x18 (plane+payonce+stages 0.67), 15x18x135 (plane+child+strips 0.70), 18x21x18 (plane+payonce+stages 0.73), 10x210x75 (plane+child+strips 0.76), 15x135x18 (plane+child+strips 0.79), 75x105x105 (plane+payonce+stages 0.80), 18x18x21 (plane+payonce+stages 0.80)
best 5: 9x9x10 (child 3.07), 10x9x10 (child 3.02), 9x10x10 (child 2.95), 9x6x24 (child 2.35), 6x9x24 (child 2.34)
