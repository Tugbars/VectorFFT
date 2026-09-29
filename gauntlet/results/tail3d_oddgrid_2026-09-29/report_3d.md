# gauntlet report (3D)

run: `tail3d_oddgrid_2026-09-29`  contract: 3D c2c interleaved, natural, out of place, K=1  cells: 20 listed, 20 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.346..1.440


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       9x9x9  3^2          chain  raced            787        832    1.02     44.1  4.6e-16
     9x63x25  3^2          chain  raced          26944      36865    1.35     36.3  7.0e-16
    15x15x15  3.5          chain  raced           5938       4197    0.70     33.3  5.9e-16
    15x21x25  3.5          chain  raced          14371      22795    1.57     35.5  5.5e-16
    15x64x64  3.5          chain  raced         111531     152214    1.33     43.8  4.0e-16
    21x21x21  3.7          chain  raced          17765      33365    1.87     34.3  4.7e-16
    21x25x27  3.7          chain  raced          26146      42984    1.62     37.4  6.3e-16
    25x25x25  5^2          chain  raced          29003      37681    1.29     37.5  4.9e-16
    27x27x27  3^3          chain  raced          41906      53192    1.26     33.5  7.4e-16
    27x35x45  3^3          chain  raced          90955     116166    1.24     35.9  7.2e-16
    32x45x45  2^5          chain  raced         142770     163401    1.14     36.3  5.9e-16
    33x35x39  3.11         chain  raced         102093     144827    1.40     34.1  7.7e-16
    35x35x35  5.7          chain  raced          85241     113712    1.28     38.7  4.2e-16
    45x32x32  3^2.5        chain  raced          73542      86065    1.17     48.5  4.9e-16
     45x45x9  3^2.5        chain  raced          35328      46606    1.32     36.5  5.7e-16
    45x45x45  3^2.5        chain  raced         200210     259590    1.29     37.5  6.8e-16
    63x63x15  3^2.7        chain  raced         131618     162093    1.21     35.9  6.3e-16
    63x63x63  3^2.7        chain  raced         586988     752850    1.27     38.2  6.3e-16
    64x15x15  2^6          chain  raced          25822      28198    1.09     38.5  4.0e-16
     75x15x9  3.5^2        chain  raced          17384      22002    1.25     38.7  5.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           20      1      1   1.09   1.28   1.62    1.26
 ALL                             20      1      1   1.09   1.28   1.62    1.26
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 odd column                      18      1      1   1.02   1.29   1.62    1.28
 pow2 column                      2      0      0   1.09   1.11   1.14    1.11
 ALL                             20      1      1   1.09   1.28   1.62    1.26
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     16      0      0   1.14   1.29   1.62    1.33
 > 65536 points                   2      0      0   1.27   1.28   1.29    1.28
 257..1024                        1      0      0   1.02   1.02   1.02    1.02
 1025..4096                       1      1      1   0.70   0.70   0.70    0.70
 ALL                             20      1      1   1.09   1.28   1.62    1.26
```


worst 10: 15x15x15 (chain 0.70), 9x9x9 (chain 1.02), 64x15x15 (chain 1.09), 32x45x45 (chain 1.14), 45x32x32 (chain 1.17), 63x63x15 (chain 1.21), 27x35x45 (chain 1.24), 75x15x9 (chain 1.25), 27x27x27 (chain 1.26), 63x63x63 (chain 1.27)
best 5: 21x21x21 (chain 1.87), 21x25x27 (chain 1.62), 15x21x25 (chain 1.57), 33x35x39 (chain 1.40), 9x63x25 (chain 1.35)
