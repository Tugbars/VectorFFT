# gauntlet report (2D)

run: `mt8_losers_2026-09-24`  contract: 2D c2c interleaved, natural, out of place, K=1_mt8  cells: 77 listed, 77 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.286..1.653

threaded plans that ran serial (engaged = 0 at both flips): 3: 16x16, 32x16, 32x32


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      4x4096  2^2          chain  raced          10904      16344    1.45    105.2  3.1e-16
      4x8192  2^2          chain  raced          18731      27870    1.30    131.2  3.6e-16
       8x512  2^3          chain  raced           2632       3271    1.19     93.4  4.1e-16
      8x1024  2^3          chain  raced           4467       5263    1.15    119.2  3.8e-16
      8x2048  2^3          csk    raced           5803       8695    1.46    197.6  3.8e-16
      8x4096  2^3          csk    raced          10616      14337    1.30    231.5  4.1e-16
      8x8192  2^3          csk    raced          16303      20298    1.20    321.6  5.2e-16
       16x16  2^4          chain+rb2 raced            164        149    0.64     62.5  2.5e-16
      16x128  2^4          csk    raced           1432       1899    1.15     78.6  3.1e-16
      16x256  2^4          csk+rb2 raced           2258       3195    1.39    108.8  2.8e-16
      16x512  2^4          csk    raced           3927       4796    1.10    135.6  4.2e-16
     16x1024  2^4          chain  raced           7269       8224    1.12    157.8  3.7e-16
     16x2048  2^4          csk    raced          12823      15954    1.11    191.7  3.7e-16
     16x4096  2^4          chain  raced          15590      21978    0.71    336.3  2.9e-16
     16x8192  2^4          csk    raced          31320      44006    1.40    355.7  4.2e-16
       32x16  2^5          chain+rb2 raced            350        341    0.65     65.9  2.3e-16
       32x32  2^5          chain+rb2 raced            728        718    0.96     70.3  2.4e-16
       32x64  2^5          chain+rb2 raced           1462       2263    1.51     77.0  2.7e-16
      32x128  2^5          chain  raced           2479       3055    1.16     99.1  3.0e-16
      32x256  2^5          csk+rb2 raced           4231       5321    1.24    125.9  4.6e-16
      32x512  2^5          chain  raced           7592      10132    1.11    151.1  3.7e-16
     32x1024  2^5          chain  raced          12985      17419    1.32    189.3  3.5e-16
     32x2048  2^5          chain  raced          16770      24317    1.24    312.6  4.4e-16
       64x32  2^6          chain+rb2 raced           1578       1852    1.03     71.4  3.0e-16
       64x64  2^6          chain+rb2 raced           2383       3809    1.44    103.1  3.8e-16
      64x128  2^6          csk    raced           4080       4800    1.14    130.5  5.2e-16
      64x256  2^6          csk+rb2 raced           8188      11303    1.31    140.1  5.1e-16
      64x512  2^6          csk    raced          12543      18382    1.40    195.9  4.4e-16
     64x1024  2^6          csk    raced          20040      30150    1.47    261.6  4.9e-16
      128x32  2^7          chain+rb2 raced           2710       2986    0.93     90.7  3.1e-16
      128x64  2^7          turn   raced           5612       4949    0.85     94.9  4.3e-16
     128x128  2^7          chain  raced          15116      12273    0.79     75.9  3.4e-16
     128x256  2^7          chain+rb2 raced          25672      14959    0.56     95.7  4.0e-16
     128x512  2^7          chain  raced          35190      28130    0.68    149.0  3.2e-16
    128x2048  2^7          chain  raced          84263      89738    0.98    280.0  5.8e-16
    128x8192  2^7          chain  raced         340388     448275    1.26    308.1  1.3e-15
      256x16  2^8          chain+rb2 raced           2668       4715    1.62     92.1  4.1e-16
      256x32  2^8          chain  raced           6598       5612    0.78     80.7  2.6e-16
      256x64  2^8          chain+rb2 raced          15002      14575    0.95     76.4  3.8e-16
     256x128  2^8          chain  raced          27608      18149    0.65     89.0  3.8e-16
     256x256  2^8          chain+rb2 raced          37003      32868    0.85    141.7  4.1e-16
     256x512  2^8          chain  raced          48507      59113    1.00    229.7  4.4e-16
    256x4096  2^8          chain  raced         343025     481149    1.22    305.7  1.0e-15
    256x8192  2^8          chain  raced        1280462    1326744    0.87    172.0  1.5e-15
      512x16  2^9          turn   raced           5139       9067    1.75    103.6  3.1e-16
      512x32  2^9          turn   raced           9575      14246    1.31    119.8  3.8e-16
      512x64  2^9          chain+rb2 raced          20910      18707    0.65    117.5  3.7e-16
     512x128  2^9          chain  raced          39750      32491    0.76    131.9  3.5e-16
     512x256  2^9          chain+rb2 raced          57200      66770    1.14    194.8  3.7e-16
    512x2048  2^9          chain  raced         360325     493324    1.34    291.0  1.2e-15
    512x4096  2^9          chain  raced        1284800    1907419    1.30    171.4  1.5e-15
    512x8192  2^9          chain  raced        4295600    4598056    1.07    107.4  2.8e-15
     1024x16  2^10         turn   raced           7566      24562    3.09    151.6  3.9e-16
     1024x64  2^10         turn   raced          34220      32880    0.93    153.2  4.6e-16
    1024x128  2^10         chain  raced          60993      68053    0.96    182.7  4.7e-16
    1024x256  2^10         chain+rb2 raced         106075     165912    1.42    222.4  5.3e-16
    1024x512  2^10         chain  raced         209238     323481    1.33    238.0  1.2e-15
   1024x2048  2^10         chain  raced        1532350    1842956    1.13    143.7  1.6e-15
   1024x4096  2^10         chain  raced        4780187    6379688    1.19     96.5  2.7e-15
     2048x16  2^11         turn   raced          14351      50437    3.13    171.3  5.2e-16
     2048x32  2^11         turn   raced          31070      47095    1.40    168.7  3.9e-16
     2048x64  2^11         turn   raced          66453      67283    1.01    167.7  4.4e-16
    2048x128  2^11         chain  raced         143412     156981    1.08    164.5  5.8e-16
    2048x256  2^11         chain+rb2 raced         235813     345575    1.32    211.2  1.2e-15
    2048x512  2^11         chain  raced         604487     784275    0.99    173.5  1.3e-15
   2048x2048  2^11         chain  raced        5521662    5683018    0.95     83.6  3.1e-15
     4096x32  2^12         turn   raced          59393      86010    1.39    187.6  4.1e-16
     4096x64  2^12         turn   raced         125587     151318    1.14    187.9  5.3e-16
    4096x128  2^12         chain  raced         299175     284469    0.93    166.5  1.2e-15
    4096x256  2^12         chain+rb2 raced         801450     625769    0.74    130.8  1.3e-15
   4096x1024  2^12         chain  raced        7058763    6361918    0.89     65.4  3.1e-15
      8192x8  2^13         turn   raced          21210     117330    5.28    247.2  4.1e-16
     8192x16  2^13         turn   raced          48733     156903    3.07    228.6  3.1e-16
     8192x32  2^13         chain+rb2 raced         113137     162343    1.00    208.5  4.9e-16
    8192x128  2^13         chain  raced         954212     693256    0.71    109.9  1.2e-15
    8192x256  2^13         chain+rb2 raced        3290438    2441094    0.71     66.9  1.6e-15
    8192x512  2^13         chain  raced        7987538    7691512    0.94     57.8  2.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           34      7     15   0.71   1.07   1.32    1.02
 chain+rb2                       18      6     11   0.64   0.96   1.51    0.96
 turn                            12      0      2   0.93   1.40   3.13    1.71
 csk                             10      0      0   1.11   1.25   1.47    1.27
 csk+rb2                          3      0      0   1.24   1.31   1.39    1.31
 ALL                             77     13     28   0.71   1.14   1.47    1.13
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     77     13     28   0.71   1.14   1.47    1.13
 ALL                             77     13     28   0.71   1.14   1.47    1.13
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     35      8     12   0.68   1.15   1.75    1.17
 > 65536 points                  30      3     12   0.87   1.07   1.40    1.10
 1025..4096                       9      0      1   0.93   1.19   1.62    1.25
 257..1024                        2      1      2   0.65   0.80   0.96    0.79
 <= 256 points                    1      1      1   0.64   0.64   0.64    0.64
 ALL                             77     13     28   0.71   1.14   1.47    1.13
```


worst 10: 128x256 (chain+rb2 0.56), 16x16 (chain+rb2 0.64), 32x16 (chain+rb2 0.65), 512x64 (chain+rb2 0.65), 256x128 (chain 0.65), 128x512 (chain 0.68), 16x4096 (chain 0.71), 8192x256 (chain+rb2 0.71), 8192x128 (chain 0.71), 4096x256 (chain+rb2 0.74)
best 5: 8192x8 (turn 5.28), 2048x16 (turn 3.13), 1024x16 (turn 3.09), 8192x16 (turn 3.07), 512x16 (turn 1.75)
