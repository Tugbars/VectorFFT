# gauntlet report (2D)

run: `2d_scaling_avx2_2026-11-23`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 31 listed, 31 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.425..1.491


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     21x4375  3.7          chain  replayed      241695     286633    1.18     31.3  5.9e-16
    135x4375  3^3.5        chain  replayed     1991038    1809150    0.90     28.4  6.8e-16
     150x150  2.3.5^2      chain  replayed       36700      38014    1.03     44.3  6.9e-16
    257x8209  257          chain  raced       21863162   58084631    2.56     10.1  1.1e-15
    263x8219  263          chain  raced       30055963   50216925    1.67      7.6  1.4e-15
    270x8750  2.3^3.5      chain  replayed     9259412   10965650    1.15     27.0  6.8e-16
     521x521  521          chain  raced        1863988    5049469    2.69     13.1  9.9e-16
     525x525  3.5^2.7      chain  replayed      763950     859712    1.05     32.6  7.1e-16
    557x4099  557          chain  raced       35390713   54642487    1.53      6.8  1.2e-15
     588x588  2^2.3.7^2    chain  replayed      811188    1025137    1.26     39.2  6.2e-16
    588x4374  2^2.3.7^2    chain  replayed    11032037   13771144    1.23     24.8  7.4e-16
   1029x1029  3.7^3        chain  replayed     3921987    4929775    1.25     27.0  6.8e-16
   1031x1031  1031         chain  raced       14803250   21948431    1.47      7.2  1.2e-15
   1033x1033  1033         chain  raced       12929625   24370413    1.72      8.3  9.8e-16
   1033x4129  1033         chain  raced       59579175  118768762    1.80      7.9  1.1e-15
   1050x1050  2.3.5^2.7    chain  replayed     3624600    4042306    1.10     30.5  7.5e-16
   2053x2053  2053         chain  raced       53055600  101024406    1.90      8.7  1.2e-15
   2058x2058  2.3.7^3      chain  replayed    17678550   23973831    1.35     26.4  6.7e-16
   2063x1031  2063         tpc    raced       35833138   47653693    1.33      6.2  9.4e-16
   2063x2063  2063         tpc    raced       76874913  100858537    1.31      6.1  1.1e-15
    2187x525  3^7          chain  replayed     4137013    5348293    1.26     27.9  7.9e-16
   2187x2187  3^7          chain  replayed    21970675   33687906    1.51     24.2  9.1e-16
    4099x557  4099         tpc    raced       39150787   55485418    1.41      6.2  1.1e-15
   4129x1033  4129         chain  raced       56017025  110276206    1.96      8.4  1.5e-15
    4374x150  2.3^7        chain  replayed     1569413    2059262    1.31     40.4  7.5e-16
   4374x1050  2.3^7        chain  replayed    20829038   26232137    1.22     24.4  7.7e-16
    4375x315  5^4.7        chain  replayed     5040300    6928875    1.35     27.9  9.4e-16
    8209x257  8209         chain  raced       19255238   53310219    2.67     11.5  2.0e-15
    8219x263  8219         tpc    raced       35984575   51729425    1.42      6.3  1.3e-15
    8750x270  2.5^4.7      chain  replayed     7820388   11927769    1.51     32.0  8.0e-16
    8750x588  2.5^4.7      chain  replayed    19594837   32411299    1.63     29.3  9.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           27      0      1   1.05   1.35   2.56    1.47
 tpc                              4      0      0   1.31   1.37   1.42    1.37
 ALL                             31      0      1   1.10   1.35   1.96    1.45
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 prime column                    14      0      0   1.33   1.69   2.67    1.76
 even column                     10      0      0   1.10   1.24   1.63    1.27
 odd column                       7      0      1   0.90   1.25   1.51    1.20
 ALL                             31      0      1   1.10   1.35   1.96    1.45
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  30      0      1   1.15   1.38   2.56    1.47
 4097..65536                      1      0      0   1.03   1.03   1.03    1.03
 ALL                             31      0      1   1.10   1.35   1.96    1.45
```


worst 10: 135x4375 (chain 0.90), 150x150 (chain 1.03), 525x525 (chain 1.05), 1050x1050 (chain 1.10), 270x8750 (chain 1.15), 21x4375 (chain 1.18), 4374x1050 (chain 1.22), 588x4374 (chain 1.23), 1029x1029 (chain 1.25), 588x588 (chain 1.26)
best 5: 521x521 (chain 2.69), 8209x257 (chain 2.67), 257x8209 (chain 2.56), 4129x1033 (chain 1.96), 2053x2053 (chain 1.90)
