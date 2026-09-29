# gauntlet report (2D)

run: `real2d_base_2026-09-29`  contract: 2D c2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 67 listed, 67 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.126..1.224


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          chain  raced            327        156    0.47     15.7  1.7e-16
       16x32  2^4          chain  raced            437        375    0.80     26.4  2.5e-16
       16x64  2^4          chain  raced            657        773    1.07     38.9  3.3e-16
      16x128  2^4          chain  raced           1228       1204    0.98     45.9  3.1e-16
      16x256  2^4          chain  raced           2260       2460    1.07     54.4  2.6e-16
      16x512  2^4          chain  raced           7129       5279    0.73     37.3  3.2e-16
     16x1024  2^4          chain  raced          15653      11025    0.70     36.6  2.8e-16
     16x4096  2^4          chain  raced          57367      52731    0.81     45.7  3.8e-16
       23x64  23           chain  raced           1257       3236    2.35     30.8  4.7e-16
       29x64  29           chain  raced           1823       4808    2.34     27.6  5.8e-16
       32x16  2^5          chain  raced            641        320    0.50     18.0  3.0e-16
       32x32  2^5          chain  raced            910        737    0.78     28.1  3.2e-16
       32x64  2^5          chain  raced           1417       1600    1.06     39.7  3.3e-16
      32x128  2^5          chain  raced           2725       2618    0.95     45.1  4.0e-16
      32x256  2^5          chain  raced           5218       5336    0.97     51.0  3.0e-16
      32x512  2^5          chain  raced          11411      11147    0.95     50.3  3.4e-16
     32x1024  2^5          chain  raced          23713      22954    0.88     51.8  4.5e-16
       47x64  47           chain  raced           3538      11052    3.10     24.6  5.8e-16
       48x48  2^4.3        chain  raced           2281       2541    1.08     28.2  3.9e-16
       63x64  3^2.7        chain  raced           3758       5132    1.33     32.1  4.5e-16
       64x16  2^6          chain  raced           1320        670    0.50     19.4  3.6e-16
       64x30  2^6          chain  raced           2267       1762    0.74     23.1  4.0e-16
       64x32  2^6          chain  raced           1877       1473    0.77     30.0  3.4e-16
       64x50  2^6          chain  raced           2909       2763    0.94     32.0  4.0e-16
       64x62  2^6          chain  raced           5550      10491    1.88     21.4  5.0e-16
       64x64  2^6          chain  raced           3089       3576    1.15     39.8  2.7e-16
      64x128  2^6          chain  raced           6018       5664    0.91     44.2  3.9e-16
      64x256  2^6          chain  raced          11225      11498    1.00     51.1  3.9e-16
      64x512  2^6          chain  raced          24620      23841    0.96     49.9  4.6e-16
     64x1024  2^6          chain  raced          49684      50689    1.02     52.8  4.1e-16
     64x4096  2^6          chain  raced         258880     259420    1.00     45.6  4.0e-16
       96x96  2^5.3        chain  raced           8510       9525    1.11     35.7  5.2e-16
      100x64  2^2.5^2      chain  raced           6979       7267    1.04     29.0  3.9e-16
      128x16  2^7          chain  raced           2867       1456    0.47     19.6  3.2e-16
      128x32  2^7          chain  raced           5474       3294    0.60     22.4  2.7e-16
      128x64  2^7          chain  raced           7976       7309    0.86     33.4  4.3e-16
     128x128  2^7          chain  raced          15725      12215    0.64     36.5  3.0e-16
     128x256  2^7          chain  raced          30349      24448    0.74     40.5  3.4e-16
     128x512  2^7          chain  raced          66082      51654    0.77     39.7  3.8e-16
    128x1024  2^7          chain  raced         155967     109446    0.66     35.7  4.4e-16
     192x192  2^6.3        chain  raced          35649      38116    1.06     39.2  5.4e-16
      256x16  2^8          chain  raced           6582       3804    0.56     18.7  4.5e-16
      256x32  2^8          chain  raced           9645       8285    0.85     27.6  3.9e-16
      256x64  2^8          chain  raced          17575      18595    1.05     32.6  3.8e-16
     256x128  2^8          chain  raced          33736      30281    0.84     36.4  4.1e-16
     256x256  2^8          chain  raced          61654      68725    1.08     42.5  4.1e-16
     256x512  2^8          chain  raced         155223     158137    1.01     35.9  4.1e-16
    256x1024  2^8          chain  raced         323547     318550    0.92     36.5  4.3e-16
     480x480  2^5.3.5      chain  raced         286871     318312    1.11     35.8  5.2e-16
      512x16  2^9          chain  raced          13957       8406    0.60     19.1  4.0e-16
      512x32  2^9          chain  raced          20648      18392    0.85     27.8  3.2e-16
      512x64  2^9          chain  raced          35233      38315    1.05     34.9  4.3e-16
     512x128  2^9          chain  raced          68439      65926    0.96     38.3  3.7e-16
     512x256  2^9          chain  raced         148713     146873    0.98     37.5  4.1e-16
     512x512  2^9          chain  raced         344207     355383    0.94     34.3  4.1e-16
    512x1024  2^9          chain  raced         627562     748418    1.17     39.7  4.9e-16
   1000x1000  2^3.5^3      chain  raced        1673775    1643318    0.98     29.8  5.4e-16
     1024x16  2^10         chain  raced          28273      22128    0.76     20.3  3.4e-16
     1024x32  2^10         chain  raced          42314      44269    1.02     29.0  3.9e-16
     1024x64  2^10         chain  raced          71557      93310    1.19     36.6  3.9e-16
    1024x128  2^10         chain  raced         157623     171210    1.08     35.3  3.5e-16
    1024x256  2^10         chain  raced         320953     356310    1.08     36.8  4.5e-16
    1024x512  2^10         chain  raced         724500     739406    1.01     34.4  4.1e-16
   1024x1024  2^10         chain  raced        1318375    1521643    1.13     39.8  4.5e-16
   2048x2048  2^11         chain  raced        8998588   11285712    1.20     25.6  5.6e-16
     4096x16  2^12         chain  raced         154290     105744    0.68     17.0  5.0e-16
     4096x64  2^12         chain  raced         500513     469493    0.91     23.6  3.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           67     19     38   0.60   0.96   1.19    0.93
 ALL                             67     19     38   0.60   0.96   1.19    0.93
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     57     19     37   0.60   0.94   1.13    0.87
 even column                      6      0      1   0.98   1.07   1.11    1.06
 prime column                     3      0      0   2.34   2.35   3.10    2.57
 odd column                       1      0      0   1.33   1.33   1.33    1.33
 ALL                             67     19     38   0.60   0.96   1.19    0.93
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     29      8     19   0.68   0.91   1.08    0.89
 1025..4096                      17      5      8   0.56   1.06   2.35    1.10
 > 65536 points                  15      1      6   0.91   1.01   1.17    1.00
 257..1024                        5      4      4   0.50   0.78   1.07    0.70
 <= 256 points                    1      1      1   0.47   0.47   0.47    0.47
 ALL                             67     19     38   0.60   0.96   1.19    0.93
```


worst 10: 128x16 (chain 0.47), 16x16 (chain 0.47), 32x16 (chain 0.50), 64x16 (chain 0.50), 256x16 (chain 0.56), 128x32 (chain 0.60), 512x16 (chain 0.60), 128x128 (chain 0.64), 128x1024 (chain 0.66), 4096x16 (chain 0.68)
best 5: 47x64 (chain 3.10), 23x64 (chain 2.35), 29x64 (chain 2.34), 64x62 (chain 1.88), 63x64 (chain 1.33)
