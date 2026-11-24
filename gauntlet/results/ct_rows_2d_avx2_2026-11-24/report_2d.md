# gauntlet report (2D)

run: `ct_rows_2d_avx2_2026-11-24`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 11 listed, 11 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.456..1.501


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
        9x21  3^2          chain+rb raced            180        427    2.33     39.7  2.8e-16
       21x21  3.7          csk+rb raced            459        824    1.80     42.2  3.4e-16
       35x21  5.7          chain+rb raced            745       1175    1.55     47.0  4.1e-16
       75x21  3.5^2        chain+rb raced           2176       2552    1.06     38.4  4.4e-16
      135x21  3^3.5        chain+rb raced           4346       4776    1.08     37.4  6.8e-16
      315x21  3^2.5.7      chain+rb raced          11258      15107    1.34     37.3  5.7e-16
      525x21  3.5^2.7      chain+rb raced          20522      26496    1.29     36.1  7.9e-16
     1029x21  3.7^3        chain+rb raced          42839      68757    1.59     36.3  5.9e-16
     2187x21  3^7          chain+rb raced          97016     125597    1.28     36.7  7.3e-16
     4375x21  5^4.7        chain+rb raced         240838     273478    1.13     31.4  6.1e-16
     8505x21  3^5.5.7      chain+rb raced         452500     888372    1.96     34.4  8.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb                        10      0      0   1.08   1.31   2.33    1.42
 csk+rb                           1      0      0   1.80   1.80   1.80    1.80
 ALL                             11      0      0   1.08   1.34   1.96    1.45
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      11      0      0   1.08   1.34   1.96    1.45
 ALL                             11      0      0   1.08   1.34   1.96    1.45
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                      4      0      0   1.28   1.31   1.59    1.37
 257..1024                        2      0      0   1.55   1.67   1.80    1.67
 1025..4096                       2      0      0   1.06   1.07   1.08    1.07
 > 65536 points                   2      0      0   1.13   1.55   1.96    1.49
 <= 256 points                    1      0      0   2.33   2.33   2.33    2.33
 ALL                             11      0      0   1.08   1.34   1.96    1.45
```


worst 10: 75x21 (chain+rb 1.06), 135x21 (chain+rb 1.08), 4375x21 (chain+rb 1.13), 2187x21 (chain+rb 1.28), 525x21 (chain+rb 1.29), 315x21 (chain+rb 1.34), 35x21 (chain+rb 1.55), 1029x21 (chain+rb 1.59), 21x21 (csk+rb 1.80), 8505x21 (chain+rb 1.96)
best 5: 9x21 (chain+rb 2.33), 8505x21 (chain+rb 1.96), 21x21 (csk+rb 1.80), 1029x21 (chain+rb 1.59), 35x21 (chain+rb 1.55)
