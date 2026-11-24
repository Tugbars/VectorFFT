# gauntlet report (2D)

run: `2d_csk_cells_est_avx2_2026-11-24`  contract: 2D c2c interleaved, natural, out of place, K=1_est  cells: 22 listed, 22 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.403..1.429


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      21x128  3.7          csk    -               2706       4644    1.63     56.6  4.3e-16
      21x512  3.7          csk    -              13299      19590    1.44     54.1  4.8e-16
      21x600  3.7          csk    -              18680      30134    1.59     45.9  5.7e-16
     21x2048  3.7          csk    -              61080      85348    1.30     54.2  4.2e-16
       25x64  5^2          csk+rb -               1476       1569    0.98     57.7  4.5e-16
      25x100  5^2          csk    -               2845       3950    1.38     49.6  5.1e-16
      25x128  5^2          csk    -               3369       4801    1.39     55.3  4.8e-16
      25x150  5^2          csk    -               5644       5854    1.01     39.4  4.8e-16
      25x384  5^2          csk    -              12365      20357    1.57     51.4  4.9e-16
      25x512  5^2          csk    -              16724      24597    1.46     52.2  4.2e-16
      25x600  5^2          csk    -              23058      28870    1.24     45.1  6.3e-16
     25x2048  5^2          csk    -              78372     113786    1.39     51.1  3.8e-16
     25x4096  5^2          csk    -             181584     237960    1.28     46.9  4.9e-16
       27x64  3^3          csk+rb -               1566       1833    1.16     59.3  5.9e-16
      27x100  3^3          csk    -               3185       4374    1.28     48.3  5.4e-16
      27x128  3^3          csk    -               3755       5849    1.47     54.1  4.2e-16
      27x150  3^3          csk    -               5939       6659    1.11     40.9  5.7e-16
      27x384  3^3          csk    -              13686      23777    1.73     50.5  5.2e-16
      27x512  3^3          csk    -              17918      28135    1.54     53.1  5.3e-16
      27x600  3^3          csk    -              25246      32511    1.28     44.9  5.2e-16
     27x2048  3^3          csk    -              88864     139447    1.50     49.0  5.1e-16
     27x4096  3^3          csk    -             202439     290719    1.37     45.8  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 csk                             20      0      0   1.24   1.39   1.63    1.39
 csk+rb                           2      0      1   0.98   1.07   1.16    1.06
 ALL                             22      0      1   1.11   1.38   1.59    1.35
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      22      0      1   1.11   1.38   1.59    1.35
 ALL                             22      0      1   1.11   1.38   1.59    1.35
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     11      0      0   1.28   1.46   1.59    1.45
 1025..4096                       9      0      1   0.98   1.28   1.63    1.25
 > 65536 points                   2      0      0   1.28   1.33   1.37    1.33
 ALL                             22      0      1   1.11   1.38   1.59    1.35
```


worst 10: 25x64 (csk+rb 0.98), 25x150 (csk 1.01), 27x150 (csk 1.11), 27x64 (csk+rb 1.16), 25x600 (csk 1.24), 27x100 (csk 1.28), 27x600 (csk 1.28), 25x4096 (csk 1.28), 21x2048 (csk 1.30), 27x4096 (csk 1.37)
best 5: 27x384 (csk 1.73), 21x128 (csk 1.63), 21x600 (csk 1.59), 25x384 (csk 1.57), 27x512 (csk 1.54)
