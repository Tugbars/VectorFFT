# gauntlet report (2D)

run: `2d_csk_cells_avx2_2026-11-24`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 22 listed, 22 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.334..1.529


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      21x128  3.7          csk    raced           2702       4559    1.41     56.7  4.4e-16
      21x512  3.7          csk    raced          12843      19582    1.50     56.1  4.8e-16
      21x600  3.7          csk    raced          18823      30208    1.56     45.6  5.2e-16
     21x2048  3.7          csk    raced          61985      84609    1.35     53.4  4.2e-16
       25x64  5^2          csk+rb raced           1409       1570    1.11     60.4  4.5e-16
      25x100  5^2          csk    raced           2810       3926    1.30     50.2  5.1e-16
      25x128  5^2          csk    raced           3327       4863    1.40     56.0  4.4e-16
      25x150  5^2          chain  raced           5547       5862    1.05     40.1  4.7e-16
      25x384  5^2          csk    raced          11604      20318    1.72     54.7  4.6e-16
      25x512  5^2          csk    raced          15504      24580    1.53     56.3  4.2e-16
      25x600  5^2          chain  raced          22659      28853    1.26     45.9  7.0e-16
     25x2048  5^2          csk    raced          84503     122082    1.38     47.4  4.3e-16
     25x4096  5^2          csk    raced         190758     238581    1.06     44.7  4.3e-16
       27x64  3^3          csk+rb raced           1712       1838    1.03     54.3  5.9e-16
      27x100  3^3          csk    raced           3253       4396    1.30     47.3  5.4e-16
      27x128  3^3          csk    raced           3742       5896    1.57     54.3  4.7e-16
      27x150  3^3          csk    raced           5843       6671    1.14     41.5  5.5e-16
      27x384  3^3          chain  raced          14099      23995    1.69     49.0  4.6e-16
      27x512  3^3          csk    raced          17060      28571    1.64     55.7  4.8e-16
      27x600  3^3          chain  raced          24107      32478    1.34     47.0  5.5e-16
     27x2048  3^3          csk    raced          91850     140076    1.44     47.4  5.5e-16
     27x4096  3^3          chain  raced         225111     281114    1.14     41.2  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk                             15      0      0   1.14   1.41   1.64    1.41
 chain                            5      0      0   1.05   1.26   1.69    1.28
 csk+rb                           2      0      0   1.03   1.07   1.11    1.07
 ALL                             22      0      0   1.06   1.37   1.64    1.34
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      22      0      0   1.06   1.37   1.64    1.34
 ALL                             22      0      0   1.06   1.37   1.64    1.34
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     11      0      0   1.34   1.50   1.69    1.48
 1025..4096                       9      0      0   1.03   1.30   1.57    1.24
 > 65536 points                   2      0      0   1.06   1.10   1.14    1.10
 ALL                             22      0      0   1.06   1.37   1.64    1.34
```


worst 10: 27x64 (csk+rb 1.03), 25x150 (chain 1.05), 25x4096 (csk 1.06), 25x64 (csk+rb 1.11), 27x150 (csk 1.14), 27x4096 (chain 1.14), 25x600 (chain 1.26), 25x100 (csk 1.30), 27x100 (csk 1.30), 27x600 (chain 1.34)
best 5: 25x384 (csk 1.72), 27x384 (chain 1.69), 27x512 (csk 1.64), 27x128 (csk 1.57), 21x600 (csk 1.56)
