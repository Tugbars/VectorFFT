# gauntlet report (2D)

run: `tail2d_oddgrid_2026-09-29`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 169 listed, 169 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 6 readings, 1.073..1.339


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         9x9  3^2          csk+rb raced             65         69    1.07     39.7  4.6e-16
        9x15  3^2          csk+rb raced            137        340    2.43     34.9  3.8e-16
        9x21  3^2          csk    raced            208        426    1.94     34.4  2.8e-16
        9x25  3^2          csk    raced            236        485    2.02     37.3  3.8e-16
        9x27  3^2          chain  raced            266        521    1.95     36.3  6.0e-16
        9x33  3^2          csk    raced            324        614    1.81     37.6  4.4e-16
        9x35  3^2          csk    raced            307        640    2.08     42.5  3.8e-16
        9x45  3^2          csk    raced            396        803    1.99     44.2  4.4e-16
        9x49  3^2          csk    raced            433        893    2.02     44.7  5.9e-16
        9x63  3^2          csk    raced            571       1119    1.91     45.4  5.0e-16
        9x75  3^2          csk    raced            688       1457    2.02     46.1  6.3e-16
        9x99  3^2          chain  raced            938       1915    2.04     46.6  4.2e-16
       9x121  3^2          chain  raced           1181       2186    1.79     46.5  4.9e-16
        15x9  3.5          csk+rb raced            120        326    2.65     39.7  4.5e-16
       15x15  3.5          csk+rb raced            253        183    0.72     34.8  4.5e-16
       15x21  3.5          csk    raced            416        612    1.47     31.5  4.2e-16
       15x25  3.5          csk    raced            458        719    1.52     35.0  3.8e-16
       15x27  3.5          chain  raced            525        774    1.40     33.4  6.9e-16
       15x33  3.5          csk    raced            615        933    1.51     36.0  3.6e-16
       15x35  3.5          chain  raced            610        968    1.54     38.9  4.0e-16
       15x45  3.5          csk    raced            791       1206    1.52     40.1  4.2e-16
       15x49  3.5          csk    raced            859       1391    1.60     40.7  4.7e-16
       15x63  3.5          csk    raced           1142       1708    1.48     40.9  5.2e-16
       15x75  3.5          csk    raced           1387       2332    1.61     41.1  6.0e-16
       15x99  3.5          csk    raced           2055       3137    1.11     38.1  5.2e-16
      15x121  3.5          csk    raced           2614       3589    1.36     37.6  4.9e-16
        21x9  3.7          csk+rb raced            197        415    2.07     36.3  4.7e-16
       21x15  3.7          csk+rb raced            402        620    1.45     32.5  5.7e-16
       21x21  3.7          chain+rb raced            628        825    1.28     30.8  4.7e-16
       21x25  3.7          chain  raced            679        975    1.43     34.9  2.6e-16
       21x27  3.7          chain  raced            738       1063    1.44     35.1  4.5e-16
       21x33  3.7          chain  raced            970       1289    1.32     33.7  4.4e-16
       21x35  3.7          chain  raced            892       1364    1.49     39.2  4.5e-16
       21x45  3.7          csk    raced           1280       1680    1.28     36.5  4.8e-16
       21x49  3.7          csk    raced           1427       1951    1.31     36.1  5.4e-16
       21x63  3.7          csk    raced           1897       2553    1.32     36.2  4.5e-16
       21x75  3.7          csk    raced           2279       3273    1.38     36.7  6.8e-16
       21x99  3.7          csk    raced           3211       4489    1.35     35.7  6.1e-16
      21x121  3.7          chain  raced           3820       5155    1.33     37.6  5.0e-16
        25x9  5^2          chain+rb raced            244        483    1.96     36.1  4.0e-16
       25x15  5^2          chain+rb raced            460        736    1.56     34.9  5.2e-16
       25x21  5^2          chain+rb raced            767        771    1.00     30.9  3.8e-16
       25x25  5^2          chain  raced            835        867    0.96     34.8  3.9e-16
       25x27  5^2          chain  raced            873       1043    1.15     36.3  4.6e-16
       25x33  5^2          chain  raced           1159       1220    1.04     34.5  4.0e-16
       25x35  5^2          chain  raced           1142       1220    1.03     37.4  4.6e-16
       25x45  5^2          chain  raced           1671       1696    0.96     34.1  5.2e-16
       25x49  5^2          chain  raced           1850       1696    0.92     34.0  3.6e-16
       25x63  5^2          chain  raced           2490       2458    0.98     33.6  4.3e-16
       25x75  5^2          csk    raced           3220       2927    0.75     31.7  5.1e-16
       25x99  5^2          chain  raced           4176       4175    0.86     33.4  4.6e-16
      25x121  5^2          chain  raced           4957       4784    0.96     35.3  5.0e-16
        27x9  3^3          chain+rb raced            265        514    1.94     36.3  4.2e-16
       27x15  3^3          chain+rb raced            504        792    1.56     34.8  4.2e-16
       27x21  3^3          chain+rb raced            778        878    0.98     33.3  5.1e-16
       27x25  3^3          chain  raced            924        991    1.06     34.3  5.5e-16
       27x27  3^3          chain  raced            988       1187    1.19     35.1  4.0e-16
       27x33  3^3          chain  raced           1355       1390    1.00     32.2  5.4e-16
       27x35  3^3          csk    raced           1728       1386    0.80     27.0  4.6e-16
       27x45  3^3          csk    raced           2082       1929    0.84     29.9  6.1e-16
       27x49  3^3          chain  raced           2101       1946    0.90     32.6  5.0e-16
       27x63  3^3          chain  raced           2643       2770    1.04     34.5  4.8e-16
       27x75  3^3          chain  raced           3274       3302    0.99     34.0  5.8e-16
       27x99  3^3          chain  raced           4212       4761    1.12     36.1  4.1e-16
      27x121  3^3          chain  raced           5267       5366    1.01     36.2  7.3e-16
        33x9  3.11         turn   raced            377        624    1.65     32.3  4.2e-16
       33x15  3.11         chain+rb raced            676        960    1.41     32.8  3.8e-16
       33x21  3.11         chain+rb raced           1059       1308    1.23     30.9  5.0e-16
       33x25  3.11         chain  raced           1511       1569    0.99     26.5  4.5e-16
       33x27  3.11         chain  raced           1443       1796    1.19     30.3  4.3e-16
       33x33  3.11         chain  raced           1949       2112    1.06     28.2  5.0e-16
       33x35  3.11         chain  raced           1980       2215    1.11     29.7  4.0e-16
       33x45  3.11         chain  raced           2594       3084    1.17     30.2  4.8e-16
       33x49  3.11         chain  raced           2614       3272    1.25     33.0  4.8e-16
       33x63  3.11         chain  raced           3333       4600    1.35     34.4  4.6e-16
       33x75  3.11         chain  raced           4132       5409    1.28     33.8  6.2e-16
       33x99  3.11         chain  raced           5377       7377    1.36     35.5  5.9e-16
      33x121  3.11         chain  raced           6713       8541    1.26     35.6  4.5e-16
        35x9  5.7          chain+rb raced            346        645    1.86     37.8  4.8e-16
       35x15  5.7          chain+rb raced            656        990    1.47     36.1  4.7e-16
       35x21  5.7          chain+rb raced           1089       1167    0.95     32.1  4.4e-16
       35x25  5.7          chain  raced           1219       1350    1.08     35.1  3.9e-16
       35x27  5.7          chain  raced           1441       1598    1.08     32.4  4.9e-16
       35x33  5.7          chain  raced           1882       1871    0.95     31.2  4.6e-16
       35x35  5.7          chain  raced           1853       1881    1.01     33.9  3.9e-16
       35x45  5.7          chain  raced           2591       2624    1.01     32.3  6.2e-16
       35x49  5.7          chain  raced           2752       2657    0.96     33.5  5.0e-16
       35x63  5.7          chain  raced           3458       3692    1.01     35.4  5.2e-16
       35x75  5.7          chain  raced           4328       4410    1.00     34.4  5.5e-16
       35x99  5.7          chain  raced           5674       6138    1.06     35.9  5.5e-16
      35x121  5.7          chain  raced           6835       7044    1.03     37.3  5.0e-16
        45x9  3^2.5        turn   raced            483        784    1.50     36.3  4.5e-16
       45x15  3^2.5        chain+rb raced            866       1239    1.34     36.6  5.0e-16
       45x21  3^2.5        chain+rb raced           1520       1537    0.99     30.7  5.5e-16
       45x25  3^2.5        chain  raced           1857       1759    0.94     30.7  4.3e-16
       45x27  3^2.5        chain  raced           1979       2074    1.03     31.5  6.2e-16
       45x33  3^2.5        chain  raced           2476       2467    0.99     31.6  5.8e-16
       45x35  3^2.5        chain  raced           2476       2531    1.00     33.8  4.5e-16
       45x45  3^2.5        chain  raced           3157       3508    1.10     35.2  5.6e-16
       45x49  3^2.5        chain  raced           3524       3539    1.00     34.7  4.9e-16
       45x63  3^2.5        chain  raced           4396       4971    1.11     37.0  7.0e-16
       45x75  3^2.5        chain  raced           5493       5799    1.05     36.0  7.5e-16
       45x99  3^2.5        chain  raced           7148       8096    1.13     37.8  6.1e-16
      45x121  3^2.5        chain  raced           8795       9441    0.89     38.4  4.0e-16
        49x9  7^2          chain+rb raced            526        901    1.69     36.8  4.0e-16
       49x15  7^2          chain+rb raced            966       1426    1.43     36.2  4.9e-16
       49x21  7^2          chain  raced           1682       1640    0.97     30.6  4.0e-16
       49x25  7^2          chain  raced           1977       1880    0.95     31.8  3.5e-16
       49x27  7^2          chain  raced           2147       2244    1.03     32.0  5.1e-16
       49x33  7^2          chain  raced           2705       2616    0.68     31.9  4.7e-16
       49x35  7^2          chain  raced           2752       2719    0.98     33.5  5.0e-16
       49x45  7^2          chain  raced           3571       3714    1.02     34.3  6.2e-16
       49x49  7^2          chain  raced           3864       3731    0.96     34.9  5.4e-16
       49x63  7^2          chain  raced           4831       5193    1.07     37.0  5.7e-16
       49x75  7^2          chain  raced           6008       6131    1.00     36.2  6.1e-16
       49x99  7^2          chain  raced           7767       8594    1.10     38.2  5.6e-16
      49x121  7^2          chain  raced           9892       9900    1.00     37.6  7.3e-16
        63x9  3^2.7        chain+rb raced            663       1074    1.59     39.1  5.9e-16
       63x15  3^2.7        chain+rb raced           1515       1756    0.98     30.8  4.0e-16
       63x21  3^2.7        chain+rb raced           2319       2172    0.90     29.6  5.3e-16
       63x25  3^2.7        chain  raced           2652       2492    0.93     31.5  5.3e-16
       63x27  3^2.7        chain  raced           2852       2981    1.03     32.0  4.5e-16
       63x33  3^2.7        chain  raced           3461       3500    1.00     33.1  5.7e-16
       63x35  3^2.7        chain  raced           3656       3637    0.99     33.5  5.3e-16
       63x45  3^2.7        chain  raced           4625       4971    1.06     35.1  4.8e-16
       63x49  3^2.7        chain  raced           4852       5053    1.01     36.9  5.7e-16
       63x63  3^2.7        chain  raced           6245       6872    1.08     38.0  5.0e-16
       63x75  3^2.7        chain  raced           7930       8185    1.02     36.4  6.3e-16
       63x99  3^2.7        chain  raced          10012      11612    1.13     39.3  5.2e-16
      63x121  3^2.7        chain  raced          12561      13739    1.05     39.1  5.4e-16
        75x9  3.5^2        chain+rb raced            815       1457    1.70     38.9  7.2e-16
       75x15  3.5^2        chain+rb raced           1886       2420    1.27     30.2  6.3e-16
       75x21  3.5^2        chain  raced           3094       2573    0.82     27.0  4.4e-16
       75x25  3.5^2        chain  raced           3324       2972    0.88     30.7  4.2e-16
       75x27  3.5^2        chain  raced           3687       3533    0.94     30.2  6.1e-16
       75x33  3.5^2        chain  raced           4344       4474    0.96     32.1  5.1e-16
       75x35  3.5^2        chain  raced           4660       4269    0.89     32.0  4.2e-16
       75x45  3.5^2        chain  raced           5625       5830    1.01     35.2  5.2e-16
       75x49  3.5^2        chain  raced           6203       5893    0.95     35.1  4.5e-16
       75x63  3.5^2        chain  raced           7836       8152    1.01     36.8  5.1e-16
       75x75  3.5^2        chain  raced           9851       9702    0.97     35.6  6.1e-16
       75x99  3.5^2        chain  raced          12758      13523    1.02     37.4  5.7e-16
      75x121  3.5^2        chain  raced          15743      15610    0.99     37.9  6.0e-16
        99x9  3^2.11       turn   raced           1228       2030    1.59     35.5  5.7e-16
       99x15  3^2.11       chain+rb raced           2603       3443    1.16     30.1  5.7e-16
       99x21  3^2.11       chain  raced           3809       3580    0.91     30.1  5.1e-16
       99x25  3^2.11       chain  raced           4537       4148    0.90     30.8  5.2e-16
       99x27  3^2.11       chain  raced           4728       4918    1.03     32.2  6.8e-16
       99x33  3^2.11       chain  raced           5735       5716    0.99     33.2  5.5e-16
       99x35  3^2.11       chain  raced           6139       5917    0.96     33.2  5.0e-16
       99x45  3^2.11       chain  raced           7638       8049    1.04     35.3  6.7e-16
       99x49  3^2.11       chain  raced           8290       8145    0.95     35.8  5.0e-16
       99x63  3^2.11       chain  raced          10296      11397    1.08     38.2  6.6e-16
       99x75  3^2.11       chain  raced          12930      13361    1.03     36.9  7.3e-16
       99x99  3^2.11       chain  raced          17678      18599    1.03     36.8  6.2e-16
      99x121  3^2.11       chain  raced          21352      21489    0.90     38.0  5.7e-16
       121x9  11^2         turn   raced           1650       2466    1.48     33.3  5.4e-16
      121x15  11^2         chain+rb raced           3254       4213    1.29     30.2  4.2e-16
      121x21  11^2         chain  raced           4980       4490    0.90     28.9  5.2e-16
      121x25  11^2         chain  raced           5726       5207    0.91     30.5  5.1e-16
      121x27  11^2         chain  raced           6088       6156    1.01     31.3  5.7e-16
      121x33  11^2         chain  raced           7302       7163    0.96     32.7  5.3e-16
      121x35  11^2         chain  raced           7662       7436    0.96     33.3  5.2e-16
      121x45  11^2         chain  raced           9649      10137    1.04     35.0  5.1e-16
      121x49  11^2         chain  raced          10571      10430    0.96     35.1  4.9e-16
      121x63  11^2         chain  raced          13037      14208    1.07     37.7  5.3e-16
      121x75  11^2         chain  raced          16688      16697    0.85     35.7  5.9e-16
      121x99  11^2         chain  raced          21460      23183    0.90     37.8  5.6e-16
     121x121  11^2         chain  raced          27147      26958    0.99     37.3  4.8e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                          111      1     48   0.90   1.01   1.32    1.05
 csk                             25      2      3   0.84   1.51   2.02    1.46
 chain+rb                        23      0      6   0.98   1.34   1.86    1.33
 csk+rb                           6      1      1   0.72   1.76   2.65    1.57
 turn                             4      0      0   1.48   1.54   1.65    1.55
 ALL                            169      4     58   0.90   1.05   1.79    1.17
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                     169      4     58   0.90   1.05   1.79    1.17
 ALL                            169      4     58   0.90   1.05   1.79    1.17
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                      81      2     38   0.90   1.01   1.33    1.05
 257..1024                       53      1      8   0.99   1.44   1.91    1.37
 4097..65536                     25      0     11   0.90   1.02   1.10    1.00
 <= 256 points                   10      1      1   1.07   1.96   2.65    1.77
 ALL                            169      4     58   0.90   1.05   1.79    1.17
```


worst 10: 49x33 (chain 0.68), 15x15 (csk+rb 0.72), 25x75 (csk 0.75), 27x35 (csk 0.80), 75x21 (chain 0.82), 27x45 (csk 0.84), 121x75 (chain 0.85), 25x99 (chain 0.86), 75x25 (chain 0.88), 45x121 (chain 0.89)
best 5: 15x9 (csk+rb 2.65), 9x15 (csk+rb 2.43), 9x35 (csk 2.08), 21x9 (csk+rb 2.07), 9x99 (chain 2.04)
