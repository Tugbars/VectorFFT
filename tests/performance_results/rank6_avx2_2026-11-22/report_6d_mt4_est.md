# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=1_mt4_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 3.478..3.766

threaded plans that ran serial (engaged = 0 at both flips): 2: 2x4x2x4x2x4, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            -      -               1213      12209    9.47     19.0  1.4e-16
     2x4x6x8x10x12  2            -      -              51002     163330    3.10     70.0  5.5e-16
       3x3x3x3x3x3  3            -      -               2024      10139    4.74     17.1  5.3e-16
       3x5x7x7x5x3  3            -      -              17344      82345    3.29     42.7  5.6e-16
       4x4x4x4x4x4  2^2          -      -               5015      24086    4.54     49.0  2.2e-16
      4x4x4x4x4x53  2^2          -      -              65667     125708    1.87     65.0  1.1e-15
       4x4x4x8x8x8  2^2          -      -              23284      86170    2.76    105.6  2.4e-16
       5x5x5x5x5x5  5            -      -              28765      92327    2.71     37.8  5.1e-16
       6x6x6x6x6x6  2.3          -      -              29412     136877    4.47    123.0  7.6e-16
       8x8x8x4x4x4  2^3          -      -              26123     150632    5.05     94.1  3.5e-16
       8x8x8x8x8x8  2^3          -      -             113612     423506    3.67    207.7  4.9e-16
       9x9x9x9x9x9  3^2          -      -             364400    1298874    3.14    138.7  8.2e-16
 10x10x10x10x10x10  2.5          -      -             584887    1929718    3.12    170.4  6.2e-16
     12x10x8x6x4x2  2^2.3        -      -              40344     323060    7.72     88.5  5.8e-16
     16x4x4x4x4x16  2^4          -      -              30297     138878    4.58    173.1  3.3e-16
 16x16x16x16x16x16  2^4          -      -           37013112   43817568    1.18     54.4  4.5e-15
      53x4x4x4x4x4  53           -      -              53919     256009    4.40     79.2  1.3e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 -                               17      0      0   1.87   3.67   7.72    3.69
 ALL                             17      0      0   1.87   3.67   7.72    3.69
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.18   3.67   9.47    3.45
 prime column                     4      0      0   2.71   3.84   4.74    3.69
 even column                      3      0      0   3.12   4.47   7.72    4.75
 odd column                       1      0      0   3.14   3.14   3.14    3.14
 ALL                             17      0      0   1.87   3.67   7.72    3.69
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   2.71   3.84   7.72    3.72
 > 65536 points                   4      0      0   1.18   3.13   3.67    2.55
 257..1024                        2      0      0   4.74   7.10   9.47    6.70
 1025..4096                       1      0      0   4.54   4.54   4.54    4.54
 ALL                             17      0      0   1.87   3.67   7.72    3.69
```


worst 10: 16x16x16x16x16x16 (- 1.18), 4x4x4x4x4x53 (- 1.87), 5x5x5x5x5x5 (- 2.71), 4x4x4x8x8x8 (- 2.76), 2x4x6x8x10x12 (- 3.10), 10x10x10x10x10x10 (- 3.12), 9x9x9x9x9x9 (- 3.14), 3x5x7x7x5x3 (- 3.29), 8x8x8x8x8x8 (- 3.67), 53x4x4x4x4x4 (- 4.40)
best 5: 2x4x2x4x2x4 (- 9.47), 12x10x8x6x4x2 (- 7.72), 8x8x8x4x4x4 (- 5.05), 3x3x3x3x3x3 (- 4.74), 16x4x4x4x4x16 (- 4.58)
