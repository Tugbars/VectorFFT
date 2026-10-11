# gauntlet report (2D)

run: `vsfftw_2d_r2c`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 159 listed, 146 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 6 readings, 1.211..1.294


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain  refused            -          -       -        - not benched
         2x4  2            chain  raced              9          9    0.95      6.5  0.0e+00
         2x8  2            chain  raced             11         12    1.11     15.2  9.4e-17
        2x16  2            chain  raced             16         21    1.30     25.0  1.4e-16
        2x32  2            chain  raced             33         41    1.22     28.7  2.4e-16
        2x64  2            chain  raced             61         82    1.35     36.9  1.8e-16
       2x128  2            chain  raced            122        141    1.15     41.8  2.6e-16
       2x256  2            chain  raced            236        279    1.04     48.9  2.7e-16
       2x512  2            chain  raced            496        504    1.01     51.6  3.6e-16
      2x1024  2            chain  raced           1051       1123    1.05     53.6  3.7e-16
      2x2048  2            chain  raced           2576       2549    0.96     47.7  2.8e-16
      2x4096  2            chain  raced           5744       5852    1.01     46.3  3.0e-16
      2x8192  2            chain  raced          12275      12909    1.04     46.7  3.8e-16
         4x2  2^2          chain  refused            -          -       -        - not benched
         4x4  2^2          chain  raced             11         13    1.14     14.1  0.0e+00
         4x8  2^2          chain  raced             14         19    1.39     29.1  2.5e-16
        4x16  2^2          chain  raced             28         38    1.09     33.8  1.9e-16
        4x32  2^2          chain  raced             51         80    1.58     44.0  1.7e-16
        4x64  2^2          chain  raced            119        194    1.61     43.1  1.5e-16
       4x128  2^2          chain  raced            242        282    1.15     47.5  2.5e-16
       4x256  2^2          chain  raced            487        513    1.04     52.6  3.0e-16
       4x512  2^2          chain  raced            983       1043    1.05     57.3  3.0e-16
      4x1024  2^2          chain  raced           2488       2461    0.99     49.4  3.1e-16
      4x2048  2^2          chain  raced           5479       5452    0.98     48.6  2.9e-16
      4x4096  2^2          chain  raced          11845      11733    0.99     48.4  3.3e-16
      4x8192  2^2          chain  raced          25159      26160    1.03     48.8  4.0e-16
         8x2  2^3          chain  refused            -          -       -        - not benched
         8x4  2^3          chain  raced             15         21    1.32     26.0  7.2e-17
         8x8  2^3          chain  raced             24         36    1.43     40.4  1.8e-16
        8x16  2^3          chain  raced             51         77    1.51     44.0  1.3e-16
        8x32  2^3          chain  raced            107        169    1.57     47.9  2.4e-16
        8x64  2^3          chain  raced            257        357    1.38     44.7  3.1e-16
       8x128  2^3          chain  raced            510        593    1.15     50.2  3.5e-16
       8x256  2^3          chain  raced           1007       1052    1.00     55.9  2.5e-16
       8x512  2^3          chain  raced           2273       2346    1.02     54.1  3.9e-16
      8x1024  2^3          chain  raced           5187       5051    0.97     51.3  3.3e-16
      8x2048  2^3          chain  raced          12780      11095    0.86     44.9  3.6e-16
      8x4096  2^3          chain  raced          24240      27181    0.99     50.7  3.5e-16
      8x8192  2^3          chain  raced          50841      54096    1.06     51.6  3.8e-16
        16x2  2^4          chain  refused            -          -       -        - not benched
        16x4  2^4          chain  raced             34         44    0.66     28.3  1.9e-16
        16x8  2^4          chain  raced             56         80    1.43     40.2  1.6e-16
       16x16  2^4          chain  raced            108        155    1.08     47.6  2.1e-16
       16x32  2^4          chain  raced            230        347    1.45     50.2  2.5e-16
       16x64  2^4          chain  raced            545        828    1.50     47.0  2.5e-16
      16x128  2^4          chain  raced           1098       1265    1.14     51.3  3.1e-16
      16x256  2^4          chain  raced           2253       2392    0.87     54.5  2.6e-16
      16x512  2^4          chain  raced           6127       5300    0.86     43.5  3.2e-16
     16x1024  2^4          chain  raced          12606      11072    0.86     45.5  2.8e-16
     16x2048  2^4          chain  raced          27508      23990    0.77     44.7  4.6e-16
     16x4096  2^4          chain  raced          58098      52511    0.90     45.1  3.5e-16
     16x8192  2^4          chain  raced         121790     119946    0.98     45.7  3.6e-16
        32x2  2^5          chain  refused            -          -       -        - not benched
        32x4  2^5          chain  raced             71         81    1.01     31.5  1.3e-16
        32x8  2^5          chain  raced            117        146    1.24     43.6  2.8e-16
       32x16  2^5          chain  raced            233        331    1.37     49.4  3.2e-16
       32x32  2^5          chain  raced            504        721    1.42     50.8  3.2e-16
       32x64  2^5          chain  raced           1163       1739    1.29     48.4  2.8e-16
      32x128  2^5          chain  raced           2417       2800    1.12     50.8  3.0e-16
      32x256  2^5          chain  raced           5219       5317    1.00     51.0  3.0e-16
      32x512  2^5          chain  raced          11480      11175    0.97     50.0  3.4e-16
     32x1024  2^5          chain  raced          23361      23188    0.99     52.6  3.8e-16
     32x2048  2^5          chain  raced          51336      50892    0.98     51.1  4.1e-16
     32x4096  2^5          chain  raced         112997     114445    0.97     49.3  4.1e-16
     32x8192  2^5          chain  raced         262233     281480    1.06     45.0  4.0e-16
        64x2  2^6          chain  refused            -          -       -        - not benched
        64x4  2^6          chain  raced            169        208    1.22     30.2  2.1e-16
        64x8  2^6          chain  raced            268        325    1.14     43.0  2.3e-16
       64x16  2^6          chain  raced            524        676    1.28     48.8  3.6e-16
       64x32  2^6          chain  raced           1114       1490    1.33     50.5  3.4e-16
       64x64  2^6          chain  raced           2607       3307    1.25     47.1  2.8e-16
      64x128  2^6          chain  raced           5367       6268    1.16     49.6  3.9e-16
      64x256  2^6          chain  raced          11596      11353    0.83     49.5  3.9e-16
      64x512  2^6          chain  raced          24560      23514    0.95     50.0  4.6e-16
     64x1024  2^6          chain  raced          50449      50307    0.99     52.0  4.1e-16
     64x2048  2^6          chain  raced         108477     109583    1.00     51.4  4.6e-16
     64x4096  2^6          chain  raced         258820     259813    0.97     45.6  4.0e-16
     64x8192  2^6          chain  raced         559625     589369    1.04     44.5  3.9e-16
       128x2  2^7          chain  refused            -          -       -        - not benched
       128x4  2^7          chain  raced            391        413    1.06     29.5  2.0e-16
       128x8  2^7          chain  raced            596        810    1.32     42.9  3.0e-16
      128x16  2^7          chain  raced           1151       1410    1.22     48.9  3.2e-16
      128x32  2^7          chain  raced           2970       3285    1.10     41.4  2.7e-16
      128x64  2^7          chain  raced           5714       7792    1.36     46.6  4.4e-16
     128x128  2^7          chain  raced          11539      13154    1.13     49.7  3.6e-16
     128x256  2^7          chain  raced          23262      24391    1.03     52.8  3.4e-16
     128x512  2^7          chain  raced          50292      52465    1.04     52.1  5.0e-16
    128x1024  2^7          chain  raced         108310     108372    0.99     51.4  5.0e-16
    128x2048  2^7          chain  raced         246900     250936    1.01     47.8  4.5e-16
    128x4096  2^7          chain  raced         535262     556750    1.01     46.5  4.5e-16
    128x8192  2^7          chain  raced        1170525    1230525    1.04     44.8  4.7e-16
       256x2  2^8          chain  refused            -          -       -        - not benched
       256x4  2^8          chain  raced            980        890    0.86     26.1  3.3e-16
       256x8  2^8          chain  raced           1715       1643    0.95     32.8  2.6e-16
      256x16  2^8          chain  raced           3148       3778    1.16     39.0  3.0e-16
      256x32  2^8          chain  raced           7282       8199    1.11     36.6  3.0e-16
      256x64  2^8          chain  raced          14286      17341    1.21     40.1  3.4e-16
     256x128  2^8          chain  raced          29360      32046    1.08     41.9  3.7e-16
     256x256  2^8          chain  raced          60446      69510    1.15     43.4  3.7e-16
     256x512  2^8          chain  raced         148623     156830    1.05     37.5  3.3e-16
    256x1024  2^8          chain  raced         301027     323783    1.05     39.2  4.3e-16
    256x2048  2^8          chain  raced         664975     702356    0.98     37.5  4.1e-16
    256x4096  2^8          chain  raced        1308225    1571643    1.16     40.1  5.2e-16
    256x8192  2^8          chain  raced        3464338    3642769    0.99     31.8  4.2e-16
       512x2  2^9          chain  refused            -          -       -        - not benched
       512x4  2^9          chain  raced           2453       1841    0.74     23.0  3.2e-16
       512x8  2^9          chain  raced           3933       4160    1.05     31.2  2.9e-16
      512x16  2^9          chain  raced           6889       8408    1.18     38.6  3.0e-16
      512x32  2^9          chain  raced          15253      18260    1.19     37.6  3.9e-16
      512x64  2^9          chain  raced          31324      39643    1.26     39.2  3.8e-16
     512x128  2^9          chain  raced          60175      69808    1.15     43.6  3.7e-16
     512x256  2^9          chain  raced         139283     146408    1.04     40.0  3.9e-16
     512x512  2^9          chain  raced         318313     365420    1.10     37.1  5.1e-16
    512x1024  2^9          chain  raced         603737     737068    1.21     41.2  4.3e-16
    512x2048  2^9          chain  raced        1290900    1482887    1.14     40.6  5.4e-16
    512x4096  2^9          chain  raced        2995800    3723306    1.23     36.8  4.2e-16
    512x8192  2^9          chain  raced        8816425    9415387    1.06     26.2  4.8e-16
      1024x2  2^10         chain  refused            -          -       -        - not benched
      1024x4  2^10         chain  raced           5663       5509    0.94     21.7  4.0e-16
      1024x8  2^10         chain  raced           8676      12444    1.43     30.7  2.8e-16
     1024x16  2^10         chain  raced          15513      22113    1.42     37.0  4.1e-16
     1024x32  2^10         chain  raced          32880      44271    1.34     37.4  3.9e-16
     1024x64  2^10         chain  raced          64605      92158    1.42     40.6  3.4e-16
    1024x128  2^10         chain  raced         138513     182538    1.30     40.2  3.8e-16
    1024x256  2^10         chain  raced         307513     352753    1.14     38.4  3.4e-16
    1024x512  2^10         chain  raced         706700     747756    1.05     35.2  3.5e-16
   1024x1024  2^10         chain  raced        1342887    1549787    1.10     39.0  4.5e-16
   1024x2048  2^10         chain  raced        3030225    4001643    1.31     36.3  4.5e-16
   1024x4096  2^10         chain  raced        7058788    9298600    1.20     32.7  5.2e-16
      2048x2  2^11         chain  refused            -          -       -        - not benched
      2048x4  2^11         chain  raced          13424      14350    1.07     19.8  4.0e-16
      2048x8  2^11         chain  raced          19452      26473    1.32     29.5  4.0e-16
     2048x16  2^11         chain  raced          35734      47860    1.28     34.4  3.7e-16
     2048x32  2^11         chain  raced          73367      91286    1.21     35.7  3.8e-16
     2048x64  2^11         chain  raced         153987     195580    1.25     36.2  4.2e-16
    2048x128  2^11         chain  raced         325120     401333    1.22     36.3  3.5e-16
    2048x256  2^11         chain  raced         682350     792181    1.15     36.5  3.7e-16
    2048x512  2^11         chain  raced        1512163    1806869    1.17     34.7  3.8e-16
   2048x1024  2^11         chain  raced        3905400    4433656    1.08     28.2  5.0e-16
   2048x2048  2^11         chain  raced        7602150   11179037    1.45     30.3  5.6e-16
      4096x2  2^12         chain  refused            -          -       -        - not benched
      4096x4  2^12         chain  raced          32989      31620    0.94     17.4  3.5e-16
      4096x8  2^12         chain  raced          50139      59316    1.05     24.5  3.5e-16
     4096x16  2^12         chain  raced          83474     104870    1.24     31.4  3.8e-16
     4096x32  2^12         chain  raced         191083     206892    1.07     29.2  4.0e-16
     4096x64  2^12         chain  raced         356367     456303    1.27     33.1  3.2e-16
    4096x128  2^12         chain  raced         682150     940650    1.36     36.5  4.2e-16
    4096x256  2^12         chain  raced        1482312    1933687    1.28     35.4  4.5e-16
    4096x512  2^12         chain  raced        4180175    4872087    1.15     26.3  4.5e-16
   4096x1024  2^12         chain  raced        8397950   11866056    1.36     27.5  4.9e-16
      8192x2  2^13         chain  refused            -          -       -        - not benched
      8192x4  2^13         chain  raced          78407      65512    0.81     15.7  3.5e-16
      8192x8  2^13         chain  raced         101643     115497    1.12     25.8  3.8e-16
     8192x16  2^13         chain  raced         198280     224052    1.13     28.1  4.4e-16
     8192x32  2^13         chain  raced         425407     470346    1.07     27.7  4.1e-16
     8192x64  2^13         chain  raced         779187     999112    1.27     32.0  4.4e-16
    8192x128  2^13         chain  raced        1587650    2012006    1.26     33.0  4.0e-16
    8192x256  2^13         chain  raced        4346800    4993919    1.07     25.3  4.7e-16
    8192x512  2^13         chain  raced        9104437   18519800    2.01     25.3  4.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                          146      3     34   0.95   1.11   1.39    1.12
 ALL                            146      3     34   0.95   1.11   1.39    1.12
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    146      3     34   0.95   1.11   1.39    1.12
 ALL                            146      3     34   0.95   1.11   1.39    1.12
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     46      1     18   0.86   1.05   1.34    1.07
 > 65536 points                  45      0      7   0.99   1.10   1.31    1.14
 <= 256 points                   21      1      2   1.01   1.24   1.57    1.23
 1025..4096                      19      1      6   0.87   1.05   1.29    1.05
 257..1024                       15      0      1   1.01   1.15   1.45    1.20
 ALL                            146      3     34   0.95   1.11   1.39    1.12
```


worst 10: 16x4 (chain 0.66), 512x4 (chain 0.74), 16x2048 (chain 0.77), 8192x4 (chain 0.81), 64x256 (chain 0.83), 16x512 (chain 0.86), 8x2048 (chain 0.86), 256x4 (chain 0.86), 16x1024 (chain 0.86), 16x256 (chain 0.87)
best 5: 8192x512 (chain 2.01), 4x64 (chain 1.61), 4x32 (chain 1.58), 8x32 (chain 1.57), 8x16 (chain 1.51)
