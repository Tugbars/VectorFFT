# gauntlet report (2D)

run: `2d_rule_cells_avx2_2026-11-24`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 53 listed, 53 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.449..1.481


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       15x25  3.5          csk+rb raced            364        718    1.72     44.1  3.7e-16
       15x42  3.5          csk    raced            619       1066    1.72     47.3  4.1e-16
      15x256  3.5          csk+rb2 raced           4055       7075    1.72     56.4  4.7e-16
     15x1050  3.5          chain  raced          29322      36265    1.23     37.4  5.8e-16
       21x25  3.7          csk+rb raced            587        976    1.60     40.4  3.3e-16
       21x27  3.7          chain+rb raced            597       1060    1.78     43.5  5.3e-16
       21x42  3.7          csk    raced            985       1479    1.50     43.8  4.6e-16
      21x256  3.7          csk+rb2 raced           6221       9656    1.54     53.5  4.9e-16
     21x1050  3.7          chain  raced          41850      52249    1.25     38.0  4.7e-16
       25x21  5^2          csk+rb raced            532        763    1.41     44.5  3.8e-16
       25x25  5^2          chain+rb raced            662        874    0.92     43.8  4.7e-16
       25x42  5^2          chain  raced           1180       1426    1.19     44.7  5.0e-16
      25x256  5^2          csk+rb2 raced           8088      11604    1.36     50.0  4.1e-16
     25x1050  5^2          chain  raced          55868      49363    0.88     34.5  6.6e-16
       27x21  3^3          csk+rb raced            588        870    1.45     44.1  6.2e-16
       27x27  3^3          chain+rb raced            768       1179    1.52     45.1  5.0e-16
       27x42  3^3          chain  raced           1320       1640    1.23     43.6  5.6e-16
      27x256  3^3          csk+rb2 raced           8597      13687    1.58     51.3  4.3e-16
     27x1050  3^3          chain  raced          59416      55840    0.93     35.3  6.1e-16
       45x42  3^2.5        chain  raced           2434       2934    1.20     42.3  4.6e-16
      45x256  3^2.5        chain  raced          16201      17555    1.08     48.0  4.5e-16
     45x1050  3^2.5        chain  raced          95190      96566    1.01     38.5  5.8e-16
       63x42  3^2.7        chain  raced           3555       4140    1.15     42.3  5.9e-16
      63x256  3^2.7        chain  raced          23764      28851    1.21     47.4  4.2e-16
     63x1050  3^2.7        chain  raced         139660     143895    1.01     37.9  5.6e-16
       64x25  2^6          chain+rb raced           1676       2327    1.20     50.8  4.2e-16
       64x27  2^6          chain+rb raced           2025       2780    1.37     45.9  5.7e-16
       75x25  3.5^2        chain+rb raced           2775       2953    1.06     36.7  5.3e-16
       75x42  3.5^2        chain  raced           4796       4898    1.01     38.2  5.5e-16
      75x256  3.5^2        chain  raced          27147      29462    1.07     50.3  4.5e-16
     75x1050  3.5^2        chain  raced         179848     170804    0.94     35.6  6.8e-16
      100x27  2^2.5^2      chain+rb raced           4383       4658    1.06     35.1  5.6e-16
      105x25  3.5.7        chain+rb raced           4181       4211    0.97     35.7  5.8e-16
      105x42  3.5.7        chain  raced           7236       6906    0.93     36.9  7.1e-16
     105x256  3.5.7        chain  raced          38984      53017    1.26     50.7  4.5e-16
    105x1050  3.5.7        chain  raced         262533     242461    0.87     35.2  6.3e-16
      135x27  3^3.5        chain+rb raced           5583       6631    1.16     38.6  6.5e-16
      135x42  3^3.5        chain  raced           8911       9054    0.99     39.7  6.4e-16
     135x256  3^3.5        chain  raced          52805      63473    1.20     49.3  4.7e-16
    135x1050  3^3.5        chain  raced         336107     325378    0.94     36.1  6.3e-16
      225x42  3^2.5^2      chain  raced          15612      15698    0.99     40.0  6.3e-16
     225x256  3^2.5^2      chain  raced         103612     117126    1.11     44.0  5.3e-16
    225x1050  3^2.5^2      chain  raced         581775     585869    0.99     36.2  6.6e-16
      315x42  3^2.5.7      chain  raced          22113      28031    1.23     41.0  5.0e-16
     315x256  3^2.5.7      chain  raced         149683     191691    1.22     43.9  5.6e-16
    315x1050  3^2.5.7      chain  raced         835738     995750    1.18     36.3  5.8e-16
      525x42  3.5^2.7      chain  raced          39976      49347    1.23     39.8  6.8e-16
     525x256  3.5^2.7      chain  raced         261229     356450    1.34     43.8  5.6e-16
    525x1050  3.5^2.7      chain  raced        1695450    1763012    1.02     31.0  8.2e-16
     1024x25  2^10         chain+rb raced          43599      61999    1.40     43.0  4.0e-16
     1024x27  2^10         chain+rb raced          47306      70396    1.40     43.1  5.6e-16
     2187x27  3^7          chain+rb raced         137279     174578    1.20     34.1  7.8e-16
     4375x25  5^4.7        chain+rb raced         281556     325606    1.13     32.5  6.2e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           30      0      9   0.93   1.10   1.25    1.09
 chain+rb                        13      0      2   0.97   1.20   1.52    1.22
 csk+rb                           4      0      0   1.41   1.53   1.72    1.54
 csk+rb2                          4      0      0   1.36   1.56   1.72    1.54
 csk                              2      0      0   1.50   1.61   1.72    1.61
 ALL                             53      0     11   0.94   1.20   1.58    1.20
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      48      0     11   0.93   1.20   1.60    1.19
 pow2 column                      4      0      0   1.20   1.39   1.40    1.34
 even column                      1      0      0   1.06   1.06   1.06    1.06
 ALL                             53      0     11   0.94   1.20   1.58    1.20
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     22      0      5   0.93   1.21   1.40    1.17
 1025..4096                      12      0      1   1.01   1.18   1.37    1.18
 > 65536 points                  10      0      4   0.94   1.01   1.34    1.05
 257..1024                        9      0      1   0.92   1.52   1.78    1.49
 ALL                             53      0     11   0.94   1.20   1.58    1.20
```


worst 10: 105x1050 (chain 0.87), 25x1050 (chain 0.88), 25x25 (chain+rb 0.92), 105x42 (chain 0.93), 27x1050 (chain 0.93), 75x1050 (chain 0.94), 135x1050 (chain 0.94), 105x25 (chain+rb 0.97), 225x1050 (chain 0.99), 225x42 (chain 0.99)
best 5: 21x27 (chain+rb 1.78), 15x25 (csk+rb 1.72), 15x42 (csk 1.72), 15x256 (csk+rb2 1.72), 21x25 (csk+rb 1.60)
