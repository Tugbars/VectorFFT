# gauntlet report (2D)

run: `gauntlet_2d-odd`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 140 listed, 140 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.184..1.265


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
        2x64  2            chain  raced             86        467    5.45     52.3  2.2e-16
       2x128  2            chain  raced            163        829    5.00     63.0  3.7e-16
       2x256  2            chain  raced            323       1533    4.48     71.3  3.9e-16
       2x512  2            chain  raced           1079       2977    2.69     47.4  4.5e-16
        3x64  3            chain  raced            125        410    3.26     58.2  3.1e-16
       3x128  3            chain  raced            246        734    2.96     67.0  3.8e-16
       3x256  3            chain  raced            490       1345    2.74     75.1  3.9e-16
       3x512  3            chain  raced           1523       2646    1.70     53.4  3.9e-16
        5x64  5            chain  raced            216        539    2.49     61.8  3.4e-16
       5x128  5            chain  raced            429       1033    2.40     69.6  2.6e-16
       5x256  5            chain  raced           1082       2006    1.84     61.0  4.9e-16
       5x512  5            chain  raced           2628       4067    1.54     55.1  5.1e-16
        6x64  2.3          chain  raced            265        589    2.23     62.3  3.9e-16
       6x128  2.3          chain  raced            526       1148    2.17     70.0  3.7e-16
       6x256  2.3          chain  raced           1368       2257    1.61     59.4  4.7e-16
       6x512  2.3          chain  raced           3238       4666    1.43     54.9  4.8e-16
        7x64  7            chain  raced            318        715    2.20     62.1  2.3e-16
       7x128  7            chain  raced            643       1396    2.14     68.4  5.0e-16
       7x256  7            chain  raced           1671       2792    1.46     58.0  4.6e-16
       7x512  7            chain  raced           4005       5800    1.39     52.8  5.5e-16
        9x64  3^2          chain  raced            433        868    2.00     61.0  4.6e-16
       9x128  3^2          chain  raced            873       1780    2.03     67.1  4.1e-16
       9x256  3^2          chain  raced           2527       3708    1.45     50.9  5.4e-16
       9x512  3^2          chain  raced           5522       7672    1.36     50.8  4.6e-16
       10x64  2.5          chain  raced            469        892    1.90     63.6  2.5e-16
      10x128  2.5          chain  raced            980       1860    1.88     67.4  5.0e-16
      10x256  2.5          chain  raced           2897       4011    1.37     50.0  3.5e-16
      10x512  2.5          chain  raced           6224       8199    1.29     50.7  5.1e-16
       11x64  11           chain  raced            567       1096    1.93     58.7  2.7e-16
      11x128  11           chain  raced           1169       2302    1.87     63.0  3.9e-16
      11x256  11           chain  raced           3481       5156    1.47     46.4  4.5e-16
      11x512  11           chain  raced           7519      10459    1.37     46.7  5.3e-16
       12x64  2^2.3        chain  raced            584        937    1.55     63.0  4.0e-16
      12x128  2^2.3        chain  raced           1236       2028    1.63     65.8  4.1e-16
      12x256  2^2.3        chain  raced           3873       4527    1.17     45.9  5.4e-16
      12x512  2^2.3        chain  raced           8254       9125    1.10     46.8  5.2e-16
       13x64  13           chain  raced            710       1390    1.94     56.9  5.5e-16
      13x128  13           chain  raced           1498       2970    1.98     59.4  6.8e-16
      13x256  13           chain  raced           5115       6911    1.34     38.1  6.9e-16
      13x512  13           chain  raced          10624      13838    1.27     39.8  7.6e-16
       14x64  2.7          chain  raced            706       1253    1.77     62.3  3.9e-16
      14x128  2.7          chain  raced           1581       2727    1.43     61.2  4.2e-16
      14x256  2.7          chain  raced           5174       6508    1.24     40.9  4.6e-16
      14x512  2.7          chain  raced           9964      13168    1.13     46.1  5.7e-16
       15x64  3.5          chain  raced           1027       1347    1.28     46.3  3.5e-16
      15x128  3.5          chain  raced           1847       2959    1.52     56.7  4.6e-16
      15x256  3.5          chain  raced           5797       7137    1.21     39.4  4.2e-16
      15x512  3.5          chain  raced          13575      14241    1.00     36.5  5.0e-16
       17x64  17           chain  raced           1038       1844    1.56     52.9  5.5e-16
      17x128  17           chain  raced           2413       4053    1.67     50.0  5.8e-16
      17x256  17           chain  raced           6832       9029    1.31     38.5  6.5e-16
      17x512  17           chain  raced          14363      18365    1.28     39.7  7.7e-16
       19x64  19           chain  raced           1250       2292    1.82     49.9  5.2e-16
      19x128  19           chain  raced           3042       4998    1.61     45.0  6.6e-16
      19x256  19           chain  raced           7641      10983    1.39     39.0  6.3e-16
      19x512  19           chain  raced          15802      22190    1.37     40.8  6.6e-16
       21x64  3.7          chain  raced           1397       2024    1.38     50.0  4.0e-16
      21x128  3.7          chain  raced           3679       4545    1.21     41.6  5.1e-16
      21x256  3.7          chain  raced           8265       9679    1.16     40.3  4.9e-16
      21x512  3.7          chain  raced          17223      19569    1.08     41.8  4.8e-16
       22x64  2.11         chain  raced           1370       1249    0.90     53.8  3.7e-16
      22x128  2.11         chain  raced           3840       3078    0.78     42.0  4.6e-16
      22x256  2.11         chain  raced           8408      11474    1.36     41.7  4.7e-16
      22x512  2.11         chain  raced          17780      23769    1.33     42.6  4.5e-16
       23x64  23           chain  raced           1814       3419    1.79     42.7  1.1e-15
      23x128  23           chain  raced           4513       7457    1.61     37.6  7.5e-16
      23x256  23           chain  raced           9846      15419    1.45     37.4  8.2e-16
      23x512  23           chain  raced          20718      31149    1.44     38.4  8.4e-16
       25x64  5^2          chain  raced           2228       1556    0.69     38.2  3.8e-16
      25x128  5^2          chain  raced           5413       4935    0.91     34.4  4.8e-16
      25x256  5^2          chain  raced           9089      11745    1.28     44.5  5.7e-16
      25x512  5^2          chain  raced          25310      24343    0.95     34.5  6.0e-16
       26x64  2.13         chain  raced           1707       2773    1.51     52.1  5.7e-16
      26x128  2.13         chain  raced           4815       6263    1.27     40.4  5.6e-16
      26x256  2.13         chain  raced          11245      12829    1.14     37.6  6.9e-16
      26x512  2.13         chain  raced          22620      26006    1.11     40.3  7.2e-16
       27x64  3^3          chain  raced           2396       1830    0.65     38.8  4.2e-16
      27x128  3^3          chain  raced           7387       5853    0.77     27.5  4.3e-16
      27x256  3^3          chain  raced          11812      13642    1.15     37.3  4.3e-16
      27x512  3^3          chain  raced          27058      28164    1.02     35.1  5.4e-16
       29x64  29           chain  raced           2677       3743    1.16     37.6  6.3e-16
      29x128  29           chain  raced           7495       8112    1.04     29.4  5.9e-16
      29x256  29           chain  raced          14300      16679    1.06     33.4  5.8e-16
      29x512  29           chain  raced          29202      33857    1.02     35.2  6.3e-16
       31x64  31           chain  raced           3077       3881    0.97     35.3  5.6e-16
      31x128  31           chain  raced           7123       8484    1.02     33.3  6.0e-16
      31x256  31           chain  raced          15163      17559    1.01     33.9  6.0e-16
      31x512  31           chain  raced          31006      35499    0.95     35.7  7.3e-16
       33x64  3.11         chain  raced           2641       3648    1.28     44.2  4.5e-16
      33x128  3.11         chain  raced           5598       7737    1.25     45.4  5.8e-16
      33x256  3.11         chain  raced          12478      16116    1.24     44.2  5.3e-16
      33x512  3.11         chain  raced          30611      32737    1.06     38.8  5.9e-16
       35x64  5.7          chain  raced           2751       2697    0.86     45.3  4.2e-16
      35x128  5.7          chain  raced           5808       6017    0.96     46.8  4.7e-16
      35x256  5.7          chain  raced          13649      13670    0.83     43.1  4.3e-16
      35x512  5.7          chain  raced          29003      28935    0.91     43.7  4.6e-16
       37x64  37           chain  raced           3912       5399    1.38     33.9  9.2e-16
      37x128  37           chain  raced           8574      10957    1.27     33.7  9.1e-16
      37x256  37           chain  raced          16758      22628    1.30     37.3  9.7e-16
      37x512  37           chain  raced          34182      45886    1.12     39.4  1.1e-15
       39x64  3.13         chain  raced           3328       4670    1.40     42.3  6.0e-16
      39x128  3.13         chain  raced           6984       9682    1.35     43.9  5.8e-16
      39x256  3.13         chain  raced          25975      20047    0.77     25.5  7.3e-16
      39x512  3.13         chain  raced          42156      40757    0.96     33.8  7.4e-16
       41x64  41           chain  raced           4831       5999    1.13     30.8  7.7e-16
      41x128  41           chain  raced           9987      11850    1.17     32.5  7.3e-16
      41x256  41           chain  raced          19221      24442    1.25     36.5  5.9e-16
      41x512  41           chain  raced          41238      49672    1.09     36.5  6.6e-16
       43x64  43           chain  raced           5093       6477    1.16     30.9  5.5e-16
      43x128  43           chain  raced          10642      12974    1.12     32.1  6.8e-16
      43x256  43           chain  raced          21439      26977    1.15     34.5  5.9e-16
      43x512  43           chain  raced          43242      54250    1.25     36.7  6.0e-16
       44x64  2^2.11       chain  raced           4630       3606    0.77     34.8  5.7e-16
      44x128  2^2.11       chain  raced           8345       8023    0.91     42.0  6.3e-16
      44x256  2^2.11       chain  raced          17244      18078    0.99     44.0  6.3e-16
      44x512  2^2.11       chain  raced          39574      37766    0.93     41.2  5.9e-16
       45x64  3^2.5        chain  raced           3534       3632    0.93     46.8  5.6e-16
      45x128  3^2.5        chain  raced           8181       7943    0.97     44.0  5.0e-16
      45x256  3^2.5        chain  raced          16236      17612    1.04     47.9  5.6e-16
      45x512  3^2.5        chain  raced          41413      36825    0.88     40.3  5.9e-16
       46x64  2.23         chain  raced           6674       7274    1.01     25.4  1.0e-15
      46x128  2.23         chain  raced          13446      14517    0.95     27.4  7.4e-16
      46x256  2.23         chain  raced          26352      29876    1.11     30.2  7.7e-16
      46x512  2.23         chain  raced          55480      60669    1.07     30.8  8.0e-16
       47x64  47           chain  raced           5973      11429    1.76     29.1  1.3e-15
      47x128  47           chain  raced          12284      22877    1.86     30.7  1.5e-15
      47x256  47           chain  raced          24214      46545    1.76     33.7  1.5e-15
      47x512  47           chain  raced          50046      93876    1.66     35.0  1.4e-15
       58x64  2.29         chain  raced           8973       8007    0.77     24.5  5.7e-16
      58x128  2.29         chain  raced          18490      16394    0.85     25.8  5.4e-16
      58x256  2.29         chain  raced          37547      33572    0.89     27.4  5.1e-16
      58x512  2.29         chain  raced          80433      68747    0.63     27.4  5.4e-16
       62x64  2.31         chain  raced           8054       8233    0.80     29.4  6.2e-16
      62x128  2.31         chain  raced          19764      16959    0.85     26.0  5.4e-16
      62x256  2.31         chain  raced          34990      34920    0.82     31.6  6.4e-16
      62x512  2.31         chain  raced          73983      71261    0.85     32.1  6.5e-16
       94x64  2.47         chain  raced          14004      38743    2.75     27.0  4.9e-16
      94x128  2.47         chain  raced          30730      78168    2.40     26.5  5.1e-16
      94x256  2.47         chain  raced          56565     157080    2.64     31.0  6.0e-16
      94x512  2.47         chain  raced         125849     317697    2.50     29.7  7.5e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                          140      9     31   0.86   1.28   2.23    1.34
 ALL                            140      9     31   0.86   1.28   2.23    1.34
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 prime column                    56      0      2   1.04   1.42   2.20    1.49
 even column                     44      5     15   0.80   1.15   2.23    1.24
 odd column                      36      4     14   0.77   1.07   1.45    1.10
 pow2 column                      4      0      0   2.69   4.74   5.45    4.25
 ALL                            140      9     31   0.86   1.28   2.23    1.34
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     67      2     19   0.85   1.12   1.66    1.17
 1025..4096                      53      7     12   0.78   1.38   1.82    1.27
 257..1024                       17      0      0   1.55   2.17   2.96    2.20
 <= 256 points                    3      0      0   3.26   5.00   5.45    4.46
 ALL                            140      9     31   0.86   1.28   2.23    1.34
```


worst 10: 58x512 (chain 0.63), 27x64 (chain 0.65), 25x64 (chain 0.69), 58x64 (chain 0.77), 39x256 (chain 0.77), 44x64 (chain 0.77), 27x128 (chain 0.77), 22x128 (chain 0.78), 62x64 (chain 0.80), 62x256 (chain 0.82)
best 5: 2x64 (chain 5.45), 2x128 (chain 5.00), 2x256 (chain 4.48), 3x64 (chain 3.26), 3x128 (chain 2.96)
