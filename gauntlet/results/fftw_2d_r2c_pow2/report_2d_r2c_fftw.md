# gauntlet report (2D)

run: `fftw_2d_r2c_pow2`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 159 listed, 146 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 6 readings, 1.329..1.439


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         2x2  2            chain  refused            -          -       -        - not benched
         2x4  2            chain  replayed           9          9    1.04      6.7  0.0e+00
         2x8  2            chain  replayed          10         12    1.15     15.5  9.4e-17
        2x16  2            chain  replayed          16         21    1.28     24.5  1.4e-16
        2x32  2            chain  replayed          35         41    1.16     27.3  2.4e-16
        2x64  2            chain  replayed          64         85    1.32     34.8  1.8e-16
       2x128  2            chain  replayed         126        135    1.04     40.6  2.6e-16
       2x256  2            chain  replayed         239        250    1.04     48.3  2.7e-16
       2x512  2            chain  replayed         488        502    1.03     52.5  2.9e-16
      2x1024  2            chain  replayed        1125       1114    0.97     50.1  3.7e-16
      2x2048  2            chain  replayed        2607       2614    0.99     47.1  2.8e-16
      2x4096  2            chain  replayed        5813       5790    0.99     45.8  3.0e-16
      2x8192  2            chain  replayed       12301      12949    1.05     46.6  3.8e-16
         4x2  2^2          chain  refused            -          -       -        - not benched
         4x4  2^2          chain  replayed          17         13    0.77      9.5  0.0e+00
         4x8  2^2          chain  replayed          19         19    0.91     20.8  2.5e-16
        4x16  2^2          chain  replayed          25         37    1.48     38.2  1.9e-16
        4x32  2^2          chain  replayed          57         90    1.41     39.5  1.7e-16
        4x64  2^2          chain  replayed         123        194    1.57     41.6  1.5e-16
       4x128  2^2          chain  replayed         241        276    1.14     47.8  2.5e-16
       4x256  2^2          chain  replayed         499        513    1.01     51.3  3.0e-16
       4x512  2^2          chain  replayed         991       1042    1.05     56.8  3.0e-16
      4x1024  2^2          chain  replayed        2428       2409    0.98     50.6  3.1e-16
      4x2048  2^2          chain  replayed        5450       5474    0.99     48.8  3.9e-16
      4x4096  2^2          chain  replayed       11968      11742    0.97     47.9  3.3e-16
      4x8192  2^2          chain  replayed       25303      26067    1.03     48.6  4.0e-16
         8x2  2^3          chain  refused            -          -       -        - not benched
         8x4  2^3          chain  replayed          37         21    0.55     10.8  7.2e-17
         8x8  2^3          chain  replayed          30         36    1.20     32.2  1.8e-16
        8x16  2^3          chain  replayed          55         77    1.39     40.7  1.3e-16
        8x32  2^3          chain  replayed         111        169    1.52     46.1  2.4e-16
        8x64  2^3          chain  replayed         256        344    1.33     45.0  2.5e-16
       8x128  2^3          chain  replayed         504        568    1.12     50.8  2.6e-16
       8x256  2^3          chain  replayed        1002       1054    1.01     56.2  2.5e-16
       8x512  2^3          chain  replayed        2268       2386    0.85     54.2  3.9e-16
      8x1024  2^3          chain  replayed        5136       5072    0.98     51.8  3.3e-16
      8x2048  2^3          chain  replayed       11301      11366    1.00     50.7  3.7e-16
      8x4096  2^3          chain  replayed       24400      24346    0.96     50.4  3.5e-16
      8x8192  2^3          chain  replayed       61234      54852    0.88     42.8  3.8e-16
        16x2  2^4          chain  refused            -          -       -        - not benched
        16x4  2^4          chain  replayed          33         44    1.35     29.3  1.9e-16
        16x8  2^4          chain  replayed          52         75    1.08     42.8  1.6e-16
       16x16  2^4          chain  replayed         107        155    1.45     48.0  2.1e-16
       16x32  2^4          chain  replayed         236        347    1.46     48.8  2.5e-16
       16x64  2^4          chain  replayed         547        823    1.49     46.8  2.5e-16
      16x128  2^4          chain  replayed        1101       1197    1.08     51.2  2.8e-16
      16x256  2^4          chain  replayed        2712       2408    0.86     45.3  2.6e-16
      16x512  2^4          chain  replayed        6337       5295    0.83     42.0  3.2e-16
     16x1024  2^4          chain  replayed       12911      11067    0.85     44.4  2.8e-16
     16x2048  2^4          chain  replayed       27977      24665    0.86     43.9  4.6e-16
     16x4096  2^4          chain  replayed       59457      53166    0.88     44.1  3.5e-16
     16x8192  2^4          chain  replayed      122957     122730    0.99     45.3  3.6e-16
        32x2  2^5          chain  refused            -          -       -        - not benched
        32x4  2^5          chain  replayed          71         81    1.14     31.6  1.3e-16
        32x8  2^5          chain  replayed         115        150    1.31     44.7  2.8e-16
       32x16  2^5          chain  replayed         240        315    1.31     48.0  2.5e-16
       32x32  2^5          chain  replayed         512        717    1.39     50.0  2.4e-16
       32x64  2^5          chain  replayed        1176       1718    1.40     47.9  2.8e-16
      32x128  2^5          chain  replayed        2426       2667    1.07     50.6  2.7e-16
      32x256  2^5          chain  replayed        5108       5351    1.02     52.1  3.0e-16
      32x512  2^5          chain  replayed       11365      11403    0.99     50.5  3.4e-16
     32x1024  2^5          chain  replayed       24393      22951    0.92     50.4  3.8e-16
     32x2048  2^5          chain  replayed       51707      51362    0.99     50.7  4.1e-16
     32x4096  2^5          chain  replayed      114523     116915    1.00     48.6  4.1e-16
     32x8192  2^5          chain  replayed      266180     288220    1.05     44.3  4.0e-16
        64x2  2^6          chain  refused            -          -       -        - not benched
        64x4  2^6          chain  replayed         169        174    1.02     30.2  2.1e-16
        64x8  2^6          chain  replayed         267        309    1.11     43.1  2.3e-16
       64x16  2^6          chain  replayed         530        658    0.82     48.3  3.6e-16
       64x32  2^6          chain  replayed        1126       1507    1.32     50.0  3.4e-16
       64x64  2^6          chain  replayed        2662       3571    1.33     46.2  3.0e-16
      64x128  2^6          chain  replayed        5332       5701    1.04     49.9  3.4e-16
      64x256  2^6          chain  replayed       11368      11344    0.99     50.4  3.3e-16
      64x512  2^6          chain  replayed       24122      24295    1.00     50.9  4.6e-16
     64x1024  2^6          chain  replayed       50651      50226    0.98     51.8  4.1e-16
     64x2048  2^6          chain  replayed      111267     112253    0.98     50.1  5.5e-16
     64x4096  2^6          chain  replayed      263747     261483    0.95     44.7  4.0e-16
     64x8192  2^6          chain  replayed      555050     594212    1.04     44.9  3.9e-16
       128x2  2^7          chain  refused            -          -       -        - not benched
       128x4  2^7          chain  replayed         404        376    0.93     28.5  2.0e-16
       128x8  2^7          chain  replayed         628        806    1.26     40.8  3.0e-16
      128x16  2^7          chain  replayed        1190       1410    1.18     47.3  3.2e-16
      128x32  2^7          chain  replayed        2949       3296    1.07     41.7  3.2e-16
      128x64  2^7          chain  replayed        5770       7752    1.32     46.1  4.4e-16
     128x128  2^7          chain  replayed       11594      12104    1.03     49.5  3.6e-16
     128x256  2^7          chain  replayed       23752      25085    1.04     51.7  3.4e-16
     128x512  2^7          chain  replayed       51157      52642    1.02     51.2  4.1e-16
    128x1024  2^7          chain  replayed      108337     107830    0.98     51.4  5.0e-16
    128x2048  2^7          chain  replayed      245800     259646    1.03     48.0  4.5e-16
    128x4096  2^7          chain  replayed      534738     563612    1.05     46.6  4.5e-16
    128x8192  2^7          chain  replayed     1167875    1253906    1.07     44.9  4.7e-16
       256x2  2^8          chain  refused            -          -       -        - not benched
       256x4  2^8          chain  replayed         955        879    0.91     26.8  3.3e-16
       256x8  2^8          chain  replayed        1719       1697    0.94     32.8  2.6e-16
      256x16  2^8          chain  replayed        3244       3729    1.12     37.9  3.0e-16
      256x32  2^8          chain  replayed        7305       8144    1.10     36.4  3.0e-16
      256x64  2^8          chain  replayed       14376      17605    1.21     39.9  3.2e-16
     256x128  2^8          chain  replayed       28502      30164    1.05     43.1  3.7e-16
     256x256  2^8          chain  replayed       60767      68533    1.12     43.1  4.1e-16
     256x512  2^8          chain  replayed      149133     157171    1.05     37.4  3.3e-16
    256x1024  2^8          chain  replayed      302827     324866    1.06     39.0  4.3e-16
    256x2048  2^8          chain  replayed      717325     732000    1.01     34.7  4.9e-16
    256x4096  2^8          chain  replayed     1447400    1527525    1.03     36.2  4.6e-16
    256x8192  2^8          chain  replayed     3523675    4012600    1.10     31.2  4.2e-16
       512x2  2^9          chain  refused            -          -       -        - not benched
       512x4  2^9          chain  replayed        2521       1841    0.72     22.3  3.2e-16
       512x8  2^9          chain  replayed        3763       4134    1.09     32.7  3.7e-16
      512x16  2^9          chain  replayed        6926       8429    1.21     38.4  3.0e-16
      512x32  2^9          chain  replayed       14488      18496    1.26     39.6  4.5e-16
      512x64  2^9          chain  replayed       31573      37760    1.10     38.9  3.8e-16
     512x128  2^9          chain  replayed       60695      66882    1.08     43.2  3.7e-16
     512x256  2^9          chain  replayed      141357     147296    1.01     39.4  3.9e-16
     512x512  2^9          chain  replayed      333633     348316    1.04     35.4  5.1e-16
    512x1024  2^9          chain  replayed      605463     736644    1.21     41.1  4.5e-16
    512x2048  2^9          chain  replayed     1290800    1560462    1.18     40.6  4.6e-16
    512x4096  2^9          chain  replayed     3186237    3803268    1.17     34.6  4.2e-16
    512x8192  2^9          chain  replayed     7609437    9584437    1.23     30.3  4.5e-16
      1024x2  2^10         chain  refused            -          -       -        - not benched
      1024x4  2^10         chain  replayed        5757       5505    0.94     21.3  3.2e-16
      1024x8  2^10         chain  replayed        8528      12436    1.44     31.2  2.8e-16
     1024x16  2^10         chain  replayed       15534      22002    1.41     36.9  3.4e-16
     1024x32  2^10         chain  replayed       32771      44636    1.27     37.5  3.9e-16
     1024x64  2^10         chain  replayed       66303      93132    1.39     39.5  3.5e-16
    1024x128  2^10         chain  replayed      147570     170120    1.13     37.7  3.8e-16
    1024x256  2^10         chain  replayed      308220     353613    1.13     38.3  3.9e-16
    1024x512  2^10         chain  replayed      686725     789450    1.14     36.3  4.1e-16
   1024x1024  2^10         chain  replayed     1366388    1565800    1.13     38.4  4.5e-16
   1024x2048  2^10         chain  replayed     3219850    3999512    1.23     34.2  5.1e-16
   1024x4096  2^10         chain  replayed     7364475    9511375    1.21     31.3  5.2e-16
      2048x2  2^11         chain  refused            -          -       -        - not benched
      2048x4  2^11         chain  replayed       13292      14195    1.04     20.0  4.0e-16
      2048x8  2^11         chain  replayed       20182      26187    1.28     28.4  3.7e-16
     2048x16  2^11         chain  replayed       34199      45993    1.33     35.9  3.7e-16
     2048x32  2^11         chain  replayed       73820      92571    1.20     35.5  3.8e-16
     2048x64  2^11         chain  replayed      153677     199255    1.29     36.2  4.2e-16
    2048x128  2^11         chain  replayed      345293     372190    1.05     34.2  4.7e-16
    2048x256  2^11         chain  replayed      676525     812024    1.18     36.8  4.4e-16
    2048x512  2^11         chain  replayed     1607163    1872218    1.16     32.6  3.8e-16
   2048x1024  2^11         chain  replayed     4110675    4964218    1.15     26.8  4.3e-16
   2048x2048  2^11         chain  replayed     7724937   11281275    1.44     29.9  5.6e-16
      4096x2  2^12         chain  refused            -          -       -        - not benched
      4096x4  2^12         chain  replayed       33057      31812    0.96     17.3  3.8e-16
      4096x8  2^12         chain  replayed       48998      56215    1.03     25.1  3.5e-16
     4096x16  2^12         chain  replayed       83559     105118    1.25     31.4  3.8e-16
     4096x32  2^12         chain  replayed      195267     206676    1.05     28.5  4.0e-16
     4096x64  2^12         chain  replayed      360967     466533    1.28     32.7  3.7e-16
    4096x128  2^12         chain  replayed      679225     891493    1.30     36.7  5.0e-16
    4096x256  2^12         chain  replayed     1464812    1963219    1.31     35.8  4.5e-16
    4096x512  2^12         chain  replayed     4241363    4972587    1.17     26.0  4.5e-16
   4096x1024  2^12         chain  replayed     9654737   11984243    1.20     23.9  5.3e-16
      8192x2  2^13         chain  refused            -          -       -        - not benched
      8192x4  2^13         chain  replayed       79727      64307    0.80     15.4  3.5e-16
      8192x8  2^13         chain  replayed      116780     116429    0.99     22.4  3.8e-16
     8192x16  2^13         chain  replayed      201987     223075    1.10     27.6  3.8e-16
     8192x32  2^13         chain  replayed      424553     473273    1.11     27.8  4.1e-16
     8192x64  2^13         chain  replayed      756150     999337    1.30     32.9  3.7e-16
    8192x128  2^13         chain  replayed     1612863    1923244    1.17     32.5  4.0e-16
    8192x256  2^13         chain  replayed     4344050    5199344    1.19     25.3  4.7e-16
    8192x512  2^13         chain  replayed     9565700   19748450    2.01     24.1  4.9e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                          146      4     36   0.92   1.08   1.39    1.10
 ALL                            146      4     36   0.92   1.08   1.39    1.10
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    146      4     36   0.92   1.08   1.39    1.10
 ALL                            146      4     36   0.92   1.08   1.39    1.10
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     46      1     18   0.88   1.03   1.32    1.06
 > 65536 points                  45      0      4   1.00   1.13   1.30    1.14
 <= 256 points                   21      2      3   0.91   1.20   1.48    1.16
 1025..4096                      19      1      8   0.85   1.05   1.33    1.04
 257..1024                       15      0      3   0.91   1.12   1.46    1.14
 ALL                            146      4     36   0.92   1.08   1.39    1.10
```


worst 10: 8x4 (chain 0.55), 512x4 (chain 0.72), 4x4 (chain 0.77), 8192x4 (chain 0.80), 64x16 (chain 0.82), 16x512 (chain 0.83), 8x512 (chain 0.85), 16x1024 (chain 0.85), 16x256 (chain 0.86), 16x2048 (chain 0.86)
best 5: 8192x512 (chain 2.01), 4x64 (chain 1.57), 8x32 (chain 1.52), 16x64 (chain 1.49), 4x16 (chain 1.48)
