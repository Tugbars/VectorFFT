# gauntlet report (2D)

run: `oddtail2d_before_2026-09-29`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 25 listed, 25 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.017..1.352


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       23x23  23           chain+rb raced            951       1826    1.92     25.2  1.1e-15
       23x45  23           chain  raced           1464       2681    1.82     35.4  9.3e-16
       23x47  23           chain+rb raced           2901       5023    1.71     18.8  1.8e-15
       23x63  23           csk    raced           2366       4075    1.56     32.2  7.9e-16
       29x23  29           csk+rb raced           1424       2038    1.30     22.0  8.9e-16
       29x45  29           chain  raced           2272       3071    1.19     29.7  5.5e-16
       29x47  29           chain+rb raced           3849       5872    1.48     18.4  1.3e-15
       29x63  29           chain  raced           3570       4623    0.99     27.7  6.2e-16
       37x23  37           chain+rb raced           1843       2693    1.43     22.5  1.1e-15
       37x45  37           csk    raced           3168       4448    1.37     28.1  9.4e-16
       37x47  37           chain+rb raced           4797       7891    1.61     19.5  1.5e-15
       37x63  37           csk    raced           4419       6329    1.41     29.5  9.3e-16
       43x23  43           csk+rb raced           2422       3146    1.19     20.3  9.6e-16
       43x45  43           chain  raced           4370       5281    1.19     24.2  5.9e-16
       43x47  43           csk+rb raced           5886       9261    1.48     18.9  1.5e-15
       43x63  43           chain  raced           5622       7402    1.31     27.5  6.8e-16
       47x23  47           chain+rb raced           2771       5058    1.70     19.7  1.8e-15
       47x45  47           chain  raced           4653       8995    1.91     25.1  1.3e-15
       47x47  47           chain+rb raced           6565      13392    2.03     18.7  1.5e-15
       47x63  47           csk    raced           6826      12533    1.71     25.0  1.3e-15
       256x3  2^8          turn   raced            701       1534    2.18     52.5  3.6e-16
      1024x3  2^10         turn   raced           4052       7203    1.77     43.9  3.6e-16
      1024x5  2^10         turn   raced           6981      10300    1.46     45.2  3.9e-16
      1024x7  2^10         turn   raced          10556      15666    1.47     43.5  3.1e-16
      4096x3  2^12         turn   raced          18498      30342    1.63     45.1  4.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain+rb                         7      0      0   1.43   1.70   2.03    1.69
 chain                            6      0      1   0.99   1.25   1.91    1.36
 turn                             5      0      0   1.46   1.63   2.18    1.68
 csk                              4      0      0   1.37   1.48   1.71    1.51
 csk+rb                           3      0      0   1.19   1.30   1.48    1.32
 ALL                             25      0      1   1.19   1.48   1.92    1.53
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 prime column                    20      0      1   1.19   1.48   1.92    1.49
 pow2 column                      5      0      0   1.46   1.63   2.18    1.68
 ALL                             25      0      1   1.19   1.48   1.92    1.53
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 1025..4096                      17      0      1   1.19   1.56   1.91    1.52
 257..1024                        5      0      0   1.19   1.43   2.18    1.56
 4097..65536                      3      0      0   1.46   1.47   1.63    1.52
 ALL                             25      0      1   1.19   1.48   1.92    1.53
```


worst 10: 29x63 (chain 0.99), 29x45 (chain 1.19), 43x23 (csk+rb 1.19), 43x45 (chain 1.19), 29x23 (csk+rb 1.30), 43x63 (chain 1.31), 37x45 (csk 1.37), 37x63 (csk 1.41), 37x23 (chain+rb 1.43), 1024x5 (turn 1.46)
best 5: 256x3 (turn 2.18), 47x47 (chain+rb 2.03), 23x23 (chain+rb 1.92), 47x45 (chain 1.91), 23x45 (chain 1.82)
