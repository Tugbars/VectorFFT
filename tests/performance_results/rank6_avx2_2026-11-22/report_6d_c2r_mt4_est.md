# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2r interleaved, natural, out of place, K=1_c2r_mt4_est  cells: 17 listed, 17 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 6.376..6.895

threaded plans that ran serial (engaged = 0 at both flips): 3: 2x4x2x4x2x4, 2x4x6x8x10x12, 3x3x3x3x3x3


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            child  -                967      17028   17.14     11.9  3.3e-16
     2x4x6x8x10x12  2            child  -              47540     259901    5.39     37.5  1.1e-15
       3x3x3x3x3x3  3            child  -               3105      16029    5.09      5.6  1.0e-15
       3x5x7x7x5x3  3            child  -              18606     140483    6.94     19.9  1.1e-15
       4x4x4x4x4x4  2^2          child  -               3224      40575   12.37     38.1  4.4e-16
      4x4x4x4x4x53  2^2          child  -              69114     208237    2.62     30.9  1.4e-15
       4x4x4x8x8x8  2^2          child  -              13912     142298    9.35     88.3  6.7e-16
       5x5x5x5x5x5  5            child  -              27891     152922    5.29     19.5  1.1e-15
       6x6x6x6x6x6  2.3          child  -              28694     239093    7.89     63.0  1.6e-15
       8x8x8x4x4x4  2^3          child  -              19343     225045   11.29     63.5  7.8e-16
       8x8x8x8x8x8  2^3          child  -             105393     690760    5.94    111.9  1.0e-15
       9x9x9x9x9x9  3^2          child  -             523312    1609506    3.05     48.3  2.2e-15
 10x10x10x10x10x10  2.5          child  -             553400    2157006    3.55     90.0  1.6e-15
     12x10x8x6x4x2  2^2.3        child  -              66362     439216    6.25     26.9  1.0e-15
     16x4x4x4x4x16  2^4          child  -              20528     164293    7.90    127.7  7.8e-16
 16x16x16x16x16x16  2^4          child  -           24784262   39166825    1.52     40.6  1.3e-15
      53x4x4x4x4x4  53           child  -              40436     511504   10.98     52.8  1.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                           17      0      0   2.62   6.25  12.37    6.14
 ALL                             17      0      0   2.62   6.25  12.37    6.14
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      9      0      0   1.52   7.90  17.14    6.56
 prime column                     4      0      0   5.09   6.12  10.98    6.73
 even column                      3      0      0   3.55   6.25   7.89    5.59
 odd column                       1      0      0   3.05   3.05   3.05    3.05
 ALL                             17      0      0   2.62   6.25  12.37    6.14
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   5.29   7.42  11.29    6.87
 > 65536 points                   4      0      0   1.52   3.30   5.94    3.14
 257..1024                        2      0      0   5.09  11.12  17.14    9.35
 1025..4096                       1      0      0  12.37  12.37  12.37   12.37
 ALL                             17      0      0   2.62   6.25  12.37    6.14
```


worst 10: 16x16x16x16x16x16 (child 1.52), 4x4x4x4x4x53 (child 2.62), 9x9x9x9x9x9 (child 3.05), 10x10x10x10x10x10 (child 3.55), 3x3x3x3x3x3 (child 5.09), 5x5x5x5x5x5 (child 5.29), 2x4x6x8x10x12 (child 5.39), 8x8x8x8x8x8 (child 5.94), 12x10x8x6x4x2 (child 6.25), 3x5x7x7x5x3 (child 6.94)
best 5: 2x4x2x4x2x4 (child 17.14), 4x4x4x4x4x4 (child 12.37), 8x8x8x4x4x4 (child 11.29), 53x4x4x4x4x4 (child 10.98), 4x4x4x8x8x8 (child 9.35)
