# gauntlet report (2D)

run: `real2d_base_2026-09-29`  contract: 2D c2c interleaved, natural, out of place, K=1_c2r_fftw  cells: 67 listed, 67 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 0.964..1.078


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       16x16  2^4          chain  replayed         334        153    0.46     15.3  4.2e-16
       16x32  2^4          chain  raced            430        361    0.82     26.8  4.5e-16
       16x64  2^4          chain  raced            652        704    1.07     39.2  7.2e-16
      16x128  2^4          chain  raced           1251       1253    0.97     45.0  6.7e-16
      16x256  2^4          chain  raced           3185       2566    0.78     38.6  7.8e-16
      16x512  2^4          chain  raced           6747       5195    0.77     39.5  8.9e-16
     16x1024  2^4          chain  raced          15276      11205    0.68     37.5  1.0e-15
     16x4096  2^4          chain  raced          63161      50340    0.78     41.5  1.0e-15
       23x64  23           chain  raced           1241       3139    2.37     31.2  8.4e-16
       29x64  29           chain  raced           2051       4670    2.10     24.6  9.1e-16
       32x16  2^5          chain  raced            394        309    0.75     29.2  5.6e-16
       32x32  2^5          chain  raced            895        756    0.81     28.6  5.6e-16
       32x64  2^5          chain  raced           1429       1476    1.02     39.4  5.6e-16
      32x128  2^5          chain  raced           3005       2749    0.85     40.9  6.7e-16
      32x256  2^5          chain  raced           6647       5466    0.81     40.1  8.9e-16
      32x512  2^5          chain  raced          12074      10932    0.89     47.5  1.0e-15
     32x1024  2^5          chain  raced          28148      21864    0.75     43.7  1.2e-15
       47x64  47           chain  raced           3494      10787    3.05     24.9  1.1e-15
       48x48  2^4.3        chain  replayed        2504       2567    1.00     25.7  9.9e-16
       63x64  3^2.7        chain  replayed        3775       5162    1.32     32.0  1.0e-15
       64x16  2^6          chain  raced            842        650    0.72     30.4  6.7e-16
       64x30  2^6          chain  raced           2190       1670    0.71     23.9  7.9e-16
       64x32  2^6          chain  raced           1900       1562    0.80     29.6  7.8e-16
       64x50  2^6          chain  raced           3053       2774    0.89     30.5  8.9e-16
       64x62  2^6          chain  raced           5864      10342    1.72     20.2  1.0e-15
       64x64  2^6          chain  raced           3233       3271    0.96     38.0  6.7e-16
      64x128  2^6          chain  raced           6784       6043    0.85     39.2  1.0e-15
      64x256  2^6          chain  raced          13919      11592    0.81     41.2  8.9e-16
      64x512  2^6          chain  raced          26081      23077    0.85     47.1  1.2e-15
     64x1024  2^6          chain  raced          60703      47706    0.76     43.2  1.1e-15
     64x4096  2^6          chain  raced         281573     246320    0.87     41.9  1.3e-15
       96x96  2^5.3        chain  replayed        8477       9069    1.06     35.8  1.1e-15
      100x64  2^2.5^2      chain  replayed        6683       7280    1.08     30.3  8.8e-16
      128x16  2^7          chain  replayed        3100       1410    0.45     18.2  8.9e-16
      128x32  2^7          chain  replayed        5459       3352    0.60     22.5  6.7e-16
      128x64  2^7          chain  replayed        7747       7065    0.89     34.4  7.8e-16
     128x128  2^7          chain  replayed       15479      12854    0.82     37.0  1.0e-15
     128x256  2^7          chain  replayed       35160      25226    0.70     34.9  1.1e-15
     128x512  2^7          chain  replayed       65234      50761    0.77     40.2  1.2e-15
    128x1024  2^7          chain  replayed      161237     103310    0.63     34.5  1.3e-15
     192x192  2^6.3        chain  replayed       36430      36203    0.89     38.4  1.4e-15
      256x16  2^8          chain  replayed        6664       3559    0.53     18.4  6.7e-16
      256x32  2^8          chain  replayed        9866       8268    0.82     27.0  7.8e-16
      256x64  2^8          chain  replayed       16964      17789    0.99     33.8  7.8e-16
     256x128  2^8          chain  replayed       32616      30783    0.94     37.7  1.0e-15
     256x256  2^8          chain  replayed       72111      69493    0.95     36.4  1.1e-15
     256x512  2^8          chain  replayed      145743     150801    1.03     38.2  1.1e-15
    256x1024  2^8          chain  replayed      331167     321383    0.96     35.6  1.2e-15
     480x480  2^5.3.5      chain  replayed      291053     312067    1.06     35.3  1.5e-15
      512x16  2^9          chain  replayed       14034       7869    0.52     19.0  7.8e-16
      512x32  2^9          chain  replayed       21096      17705    0.80     27.2  8.9e-16
      512x64  2^9          chain  replayed       36747      36599    0.98     33.4  1.2e-15
     512x128  2^9          chain  replayed       69664      66688    0.92     37.6  1.0e-15
     512x256  2^9          chain  replayed      170340     147838    0.80     32.7  1.3e-15
     512x512  2^9          chain  replayed      320320     352733    1.09     36.8  1.3e-15
    512x1024  2^9          chain  replayed      626288     708100    1.11     39.8  1.4e-15
   1000x1000  2^3.5^3      chain  replayed     1809338    1689525    0.91     27.5  1.6e-15
     1024x16  2^10         chain  replayed       28135      20732    0.73     20.4  1.0e-15
     1024x32  2^10         chain  replayed       41340      43827    0.98     29.7  1.0e-15
     1024x64  2^10         chain  replayed       71484      93199    1.30     36.7  1.1e-15
    1024x128  2^10         chain  replayed      159907     174213    1.08     34.8  1.2e-15
    1024x256  2^10         chain  replayed      339947     364303    1.06     34.7  1.2e-15
    1024x512  2^10         chain  replayed      689875     752456    1.06     36.1  1.3e-15
   1024x1024  2^10         chain  replayed     1512538    1538700    0.92     34.7  1.4e-15
   2048x2048  2^11         chain  replayed     9186750   11574181    1.24     25.1  1.6e-15
     4096x16  2^12         chain  replayed      144756     100506    0.67     18.1  1.0e-15
     4096x64  2^12         chain  replayed      413113     460760    1.10     28.6  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           67     19     48   0.67   0.89   1.24    0.91
 ALL                             67     19     48   0.67   0.89   1.24    0.91
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     57     19     45   0.63   0.85   1.09    0.85
 even column                      6      0      3   0.89   1.03   1.08    1.00
 prime column                     3      0      0   2.10   2.37   3.05    2.48
 odd column                       1      0      0   1.32   1.32   1.32    1.32
 ALL                             67     19     48   0.67   0.89   1.24    0.91
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     29     10     26   0.68   0.82   1.06    0.84
 1025..4096                      17      5     11   0.53   0.96   2.37    1.03
 > 65536 points                  15      1      6   0.80   1.06   1.11    0.98
 257..1024                        5      2      4   0.72   0.81   1.07    0.83
 <= 256 points                    1      1      1   0.46   0.46   0.46    0.46
 ALL                             67     19     48   0.67   0.89   1.24    0.91
```


worst 10: 128x16 (chain 0.45), 16x16 (chain 0.46), 512x16 (chain 0.52), 256x16 (chain 0.53), 128x32 (chain 0.60), 128x1024 (chain 0.63), 4096x16 (chain 0.67), 16x1024 (chain 0.68), 128x256 (chain 0.70), 64x30 (chain 0.71)
best 5: 47x64 (chain 3.05), 23x64 (chain 2.37), 29x64 (chain 2.10), 64x62 (chain 1.72), 63x64 (chain 1.32)
