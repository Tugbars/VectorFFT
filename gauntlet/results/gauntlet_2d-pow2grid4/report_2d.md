# gauntlet report (2D)

run: `gauntlet_2d-pow2grid4`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 159 listed, 159 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 10 readings, 1.079..1.341


## every shape

```
       N1xN2  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain+rb raced             14         13    0.92      2.9  1.4e-16
         2x4  2            chain+rb raced             15        158   10.32      7.9  1.7e-16
         2x8  2            chain+rb raced             19        168    8.64     16.6  1.9e-16
        2x16  2            chain+rb2 raced             30        206    6.78     26.4  2.0e-16
        2x32  2            chain+rb2 raced             45        307    6.80     42.6  2.0e-16
        2x64  2            chain+rb2 raced             84        470    5.58     53.3  2.2e-16
       2x128  2            chain  raced            164        832    5.05     62.4  3.7e-16
       2x256  2            chain  raced            325       1536    4.73     70.9  3.9e-16
       2x512  2            chain  raced           1016       2971    2.76     50.4  4.5e-16
      2x1024  2            chain  raced           2284       6041    2.64     49.3  4.0e-16
      2x2048  2            chain  raced           5183      12269    2.37     47.4  2.7e-16
      2x4096  2            chain  raced          10849      25132    2.31     49.1  3.2e-16
      2x8192  2            chain  raced          22567      51571    2.27     50.8  3.4e-16
         4x2  2^2          chain+rb raced             15        155   10.30      8.0  8.9e-17
         4x4  2^2          chain+rb raced             18         18    0.99     17.6  2.0e-16
         4x8  2^2          chain+rb raced             25        175    6.93     31.9  1.7e-16
        4x16  2^2          chain+rb2 raced             42        212    5.01     45.4  2.8e-16
        4x32  2^2          chain+rb2 raced             75        317    4.22     59.7  1.7e-16
        4x64  2^2          chain+rb2 raced            155        506    3.25     65.9  2.9e-16
       4x128  2^2          chain  raced            332        949    2.85     69.4  3.7e-16
       4x256  2^2          chain  raced            680       1801    2.56     75.3  3.7e-16
       4x512  2^2          chain  raced           1975       3616    1.76     57.0  4.6e-16
      4x1024  2^2          chain  raced           4423       7501    1.69     55.6  3.7e-16
      4x2048  2^2          chain  raced           9824      15438    1.55     54.2  3.6e-16
      4x4096  2^2          chain  raced          21211      32183    1.51     54.1  3.1e-16
      4x8192  2^2          chain  raced          44610      67223    1.50     55.1  3.6e-16
         8x2  2^3          chain+rb raced             19        164    8.57     16.8  1.2e-16
         8x4  2^3          chain+rb raced             25        179    7.31     32.6  1.6e-16
         8x8  2^3          chain+rb raced             39         36    0.94     49.7  1.5e-16
        8x16  2^3          chain+rb2 raced             75        246    3.26     59.7  2.8e-16
        8x32  2^3          chain+rb2 raced            148        404    2.72     69.0  2.6e-16
        8x64  2^3          chain+rb2 raced            321        689    2.14     71.7  2.9e-16
       8x128  2^3          chain  raced            715       1404    1.96     71.7  3.3e-16
       8x256  2^3          chain  raced           1918       2834    1.44     58.7  3.5e-16
       8x512  2^3          chain  raced           4765       5791    1.20     51.6  4.6e-16
      8x1024  2^3          chain  raced          10321      11877    1.15     51.6  3.8e-16
      8x2048  2^3          chain  raced          22978      24994    1.08     49.9  3.8e-16
      8x4096  2^3          chain  raced          49016      53104    1.08     50.1  5.4e-16
      8x8192  2^3          chain  raced          94727     116216    1.19     55.3  5.2e-16
        16x2  2^4          chain+rb raced             26        195    7.42     30.4  2.2e-16
        16x4  2^4          chain+rb raced             40        214    5.23     47.7  2.5e-16
        16x8  2^4          chain+rb raced             70        246    3.50     63.9  2.0e-16
       16x16  2^4          chain+rb2 raced            155        146    0.94     66.2  2.5e-16
       16x32  2^4          chain+rb2 raced            319        352    1.10     72.1  3.0e-16
       16x64  2^4          chain+rb2 raced            718        709    0.98     71.3  3.9e-16
      16x128  2^4          chain  raced           1952       1630    0.83     57.7  4.6e-16
      16x256  2^4          chain+rb2 raced           7205       7000    0.97     34.1  4.2e-16
      16x512  2^4          chain  raced          11355      14696    1.17     46.9  3.8e-16
     16x1024  2^4          chain  raced          26091      31552    1.00     44.0  3.7e-16
     16x2048  2^4          chain  raced          59444      68008    0.98     41.3  3.7e-16
     16x4096  2^4          chain  raced         134293     136486    0.98     39.0  3.6e-16
     16x8192  2^4          chain  raced         286060     316930    1.10     38.9  4.2e-16
        32x2  2^5          chain+rb raced             49        272    5.51     38.9  2.6e-16
        32x4  2^5          chain+rb raced             78        321    4.10     57.3  2.0e-16
        32x8  2^5          chain+rb raced            148        399    2.59     69.0  2.4e-16
       32x16  2^5          chain+rb raced            347        344    0.98     66.5  2.3e-16
       32x32  2^5          chain+rb2 raced            735        723    0.93     69.7  3.9e-16
       32x64  2^5          chain+rb2 raced           2087       1895    0.76     54.0  3.5e-16
      32x128  2^5          chain  raced           5586       6845    1.17     44.0  3.5e-16
      32x256  2^5          chain  raced          13001      15170    1.16     41.0  4.0e-16
      32x512  2^5          chain  raced          26935      32121    1.16     42.6  4.6e-16
     32x1024  2^5          chain  raced          56690      68710    1.17     43.4  3.5e-16
     32x2048  2^5          chain  raced         120707     153998    1.26     43.4  3.6e-16
     32x4096  2^5          chain  raced         270860     381870    1.21     41.1  4.0e-16
     32x8192  2^5          chain  raced         614663     974306    1.52     38.4  5.2e-16
        64x2  2^6          chain+rb raced             98        414    4.12     45.5  3.4e-16
        64x4  2^6          chain+rb raced            174        509    2.89     58.7  3.4e-16
        64x8  2^6          chain+rb raced            335        687    2.04     68.8  3.2e-16
       64x16  2^6          chain+rb raced            761        831    0.96     67.3  3.2e-16
       64x32  2^6          chain+rb2 raced           1962       2016    0.96     57.4  3.0e-16
       64x64  2^6          chain  raced           6067       7079    1.16     40.5  3.8e-16
      64x128  2^6          chain  raced          12547      13978    1.11     42.4  4.2e-16
      64x256  2^6          chain  raced          25895      32356    1.12     44.3  3.8e-16
      64x512  2^6          chain  raced          48577      67219    1.21     50.6  4.4e-16
     64x1024  2^6          chain  raced         128927     147661    1.13     40.7  3.6e-16
     64x2048  2^6          chain  raced         307293     333866    1.08     36.3  3.2e-16
     64x4096  2^6          chain  raced         620062     856906    1.34     38.0  5.6e-16
     64x8192  2^6          chain  raced        1593937    1710812    1.01     31.2  9.7e-16
       128x2  2^7          turn   raced            209        726    3.46     49.0  3.5e-16
       128x4  2^7          turn   raced            405        967    2.38     56.9  3.4e-16
       128x8  2^7          turn   raced           1020       1504    1.46     50.2  4.3e-16
      128x16  2^7          turn   raced           2570       2018    0.78     43.8  3.0e-16
      128x32  2^7          chain+rb2 raced           5555       7249    1.29     44.2  3.1e-16
      128x64  2^7          chain+rb2 raced          11301      15277    1.21     47.1  4.0e-16
     128x128  2^7          chain  raced          25807      30280    1.17     44.4  4.3e-16
     128x256  2^7          chain  raced          58341      61224    1.05     42.1  5.3e-16
     128x512  2^7          chain  raced         123680     143712    1.10     42.4  4.8e-16
    128x1024  2^7          chain  raced         294960     314670    1.03     37.8  3.7e-16
    128x2048  2^7          chain  raced         638925     814412    1.17     36.9  6.2e-16
    128x4096  2^7          chain  raced        1604138    1631644    0.92     31.0  1.3e-15
    128x8192  2^7          chain  raced        3228275    3885419    1.19     32.5  1.1e-15
       256x2  2^8          turn   raced            415       1350    3.25     55.5  4.1e-16
       256x4  2^8          turn   raced           1062       1953    1.84     48.2  3.3e-16
       256x8  2^8          turn   raced           2675       3500    1.29     42.1  3.6e-16
      256x16  2^8          chain+rb2 raced           6465       7449    1.14     38.0  3.4e-16
      256x32  2^8          chain+rb2 raced          11400      14939    1.16     46.7  3.7e-16
      256x64  2^8          chain+rb2 raced          22608      31152    1.37     50.7  4.3e-16
     256x128  2^8          chain  raced          49148      61388    1.10     50.0  5.1e-16
     256x256  2^8          chain  raced         115197     146646    1.21     45.5  5.8e-16
     256x512  2^8          chain  raced         297340     355303    1.08     37.5  5.5e-16
    256x1024  2^8          chain  raced         598437     767837    1.26     39.4  5.5e-16
    256x2048  2^8          chain  raced        1387512    1826825    1.10     35.9  1.4e-15
    256x4096  2^8          chain  raced        3086738    4341787    1.31     34.0  1.1e-15
    256x8192  2^8          chain  raced        7142438    8199862    1.11     30.8  1.5e-15
       512x2  2^9          turn   raced            997       2916    2.71     51.4  3.7e-16
       512x4  2^9          turn   raced           2638       4178    1.57     42.7  4.3e-16
       512x8  2^9          turn   raced           5736       7409    1.27     42.8  4.1e-16
      512x16  2^9          chain+rb raced          12329      17816    1.30     43.2  4.3e-16
      512x32  2^9          chain+rb2 raced          26690      34828    1.01     43.0  4.3e-16
      512x64  2^9          chain+rb2 raced          53785      72129    1.25     45.7  6.1e-16
     512x128  2^9          chain  raced         118433     148093    1.23     44.3  5.3e-16
     512x256  2^9          chain  raced         283420     366060    1.23     39.3  5.1e-16
     512x512  2^9          chain  raced         618937     889962    1.43     38.1  5.4e-16
    512x1024  2^9          chain  raced        1347287    1935619    1.41     37.0  1.1e-15
    512x2048  2^9          chain  raced        3523837    4492006    1.06     29.8  1.2e-15
    512x4096  2^9          chain  raced        7089750    9823212    1.37     31.1  1.5e-15
    512x8192  2^9          chain  raced       16321750   21657462    1.19     28.3  2.8e-15
      1024x2  2^10         turn   raced           2753       6255    2.26     40.9  4.7e-16
      1024x4  2^10         turn   raced           5518       8823    1.59     44.5  4.2e-16
      1024x8  2^10         turn   raced          12129      15388    1.26     43.9  3.0e-16
     1024x16  2^10         chain+rb2 raced          30049      48908    1.27     38.2  4.6e-16
     1024x32  2^10         chain+rb2 raced          49125      76352    1.37     50.0  5.4e-16
     1024x64  2^10         chain+rb2 raced         127583     164891    1.25     41.1  5.3e-16
    1024x128  2^10         chain  raced         300133     380526    1.23     37.1  5.9e-16
    1024x256  2^10         chain  raced         558000    1139744    2.01     42.3  5.7e-16
    1024x512  2^10         chain  raced        1278575    2464406    1.89     39.0  1.2e-15
   1024x1024  2^10         chain  raced        3200750    6160743    1.90     32.8  1.3e-15
   1024x2048  2^10         chain  raced        8008400   15593162    1.69     27.5  1.6e-15
   1024x4096  2^10         chain  raced       17255138   30078850    1.49     26.7  2.8e-15
      2048x2  2^11         turn   raced           6043      12843    2.12     40.7  2.6e-16
      2048x4  2^11         turn   raced          12100      18407    1.51     44.0  3.4e-16
      2048x8  2^11         turn   raced          26658      35079    1.23     43.0  3.7e-16
     2048x16  2^11         turn   raced          78131      78833    1.00     31.5  5.2e-16
     2048x32  2^11         chain+rb2 raced         147817     160015    1.03     35.5  4.7e-16
     2048x64  2^11         chain+rb2 raced         325727     406723    1.23     34.2  6.4e-16
    2048x128  2^11         chain  raced         759162    1075681    1.35     31.1  5.8e-16
    2048x256  2^11         chain  raced        2004913    2374487    1.16     24.8  1.2e-15
    2048x512  2^11         chain  raced        3503100    5990950    1.70     29.9  1.3e-15
   2048x1024  2^11         chain  raced        9411762   17591656    1.70     23.4  1.5e-15
   2048x2048  2^11         chain  raced       19736300   28019650    1.37     23.4  3.1e-15
      4096x2  2^12         turn   raced          12626      26126    2.07     42.2  3.7e-16
      4096x4  2^12         turn   raced          25264      38084    1.50     45.4  3.7e-16
      4096x8  2^12         turn   raced          55923      67813    1.21     43.9  3.9e-16
     4096x16  2^12         chain+rb raced         173137     146485    0.83     30.3  4.8e-16
     4096x32  2^12         chain+rb2 raced         321453     393743    1.18     34.7  5.0e-16
     4096x64  2^12         chain+rb2 raced         786112     864112    1.08     30.0  7.0e-16
    4096x128  2^12         chain  raced        1811525    1922806    1.01     27.5  1.2e-15
    4096x256  2^12         chain  raced        4063125    6444738    1.51     25.8  1.4e-15
    4096x512  2^12         chain  raced        8767950   15767025    1.59     25.1  1.8e-15
   4096x1024  2^12         chain  raced       19774650   25185368    1.20     23.3  3.0e-15
      8192x2  2^13         turn   raced          26798      53177    1.97     42.8  3.6e-16
      8192x4  2^13         turn   raced          53770      78742    1.46     45.7  3.6e-16
      8192x8  2^13         turn   raced         129740     159513    1.14     40.4  4.1e-16
     8192x16  2^13         chain+rb raced         335267     434453    1.24     33.2  4.6e-16
     8192x32  2^13         chain+rb2 raced         834287     922324    0.98     28.3  4.9e-16
     8192x64  2^13         chain+rb2 raced        1739938    1994887    1.11     28.6  1.3e-15
    8192x128  2^13         chain  raced        4540937    5263725    1.14     23.1  1.3e-15
    8192x256  2^13         chain  raced        9948900   14933793    1.41     22.1  1.5e-15
    8192x512  2^13         chain  raced       23365200   41675581    1.71     19.7  3.0e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           81      0      5   1.05   1.21   2.27    1.39
 chain+rb2                       32      1      7   0.96   1.22   5.01    1.60
 chain+rb                        23      0      6   0.94   3.50   8.64    2.98
 turn                            23      1      2   1.14   1.51   2.71    1.64
 ALL                            159      2     20   0.98   1.27   4.22    1.63
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    159      2     20   0.98   1.27   4.22    1.63
 ALL                            159      2     20   0.98   1.27   4.22    1.63
```


## by plane size (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     48      0      5   1.00   1.20   1.55    1.25
 > 65536 points                  45      0      2   1.03   1.23   1.70    1.29
 <= 256 points                   28      0      4   0.94   4.61   8.64    4.02
 1025..4096                      21      2      5   0.83   1.29   2.26    1.35
 257..1024                       17      0      4   0.96   2.04   3.25    1.88
 ALL                            159      2     20   0.98   1.27   4.22    1.63
```


worst 10: 32x64 (chain+rb2 0.76), 128x16 (turn 0.78), 4096x16 (chain+rb 0.83), 16x128 (chain 0.83), 2x2 (chain+rb 0.92), 128x4096 (chain 0.92), 32x32 (chain+rb2 0.93), 16x16 (chain+rb2 0.94), 8x8 (chain+rb 0.94), 64x32 (chain+rb2 0.96)
best 5: 2x4 (chain+rb 10.32), 4x2 (chain+rb 10.30), 2x8 (chain+rb 8.64), 8x2 (chain+rb 8.57), 16x2 (chain+rb 7.42)
