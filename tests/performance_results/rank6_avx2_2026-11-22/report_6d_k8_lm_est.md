# gauntlet report (6D)

run: `rank6_avx2_2026-11-22`  contract: 6D c2c interleaved, natural, out of place, K=8 lane-major_k8_lm_est  cells: 16 listed, 16 benched, comparator: MKL DFTI 6D (out of place)

control cell 8x8x8x8x8x8: 4 readings, 13.561..13.799


## every shape

```
             shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       2x4x2x4x2x4  2            lm     -               5619     145181   25.82     32.8  2.1e-16
     2x4x6x8x10x12  2            lm     -             630212    6482450   10.26     45.3  6.1e-16
       3x3x3x3x3x3  3            lm     -               8312     182192   21.58     33.4  4.6e-16
       3x5x7x7x5x3  3            lm     -             135977    2018125   14.28     43.6  5.4e-16
       4x4x4x4x4x4  2^2          lm     -              42075     733930   17.19     46.7  2.6e-16
      4x4x4x4x4x53  2^2          lm     -            1431050    8553312    5.94     23.9  1.2e-15
       4x4x4x8x8x8  2^2          lm     -             387300    4804137   12.37     50.8  9.5e-16
       5x5x5x5x5x5  5            lm     -             184720    2303543   12.43     47.1  4.4e-16
       6x6x6x6x6x6  2.3          lm     -             601375    7560106   12.41     48.1  8.3e-16
       8x8x8x4x4x4  2^3          lm     -             418413    6858531   15.73     47.0  7.5e-16
       8x8x8x8x8x8  2^3          lm     -            5032775   69672512   13.78     37.5  8.1e-16
       9x9x9x9x9x9  3^2          lm     -           13243600  162330100   12.22     30.5  8.2e-16
 10x10x10x10x10x10  2.5          lm     -           26695450  315833550   11.79     29.9  5.9e-16
     12x10x8x6x4x2  2^2.3        lm     -             786962   15487131   19.08     36.3  5.5e-16
     16x4x4x4x4x16  2^4          lm     -            1043225   10487781    9.91     40.2  8.1e-16
      53x4x4x4x4x4  53           lm     -            1274025   13002062   10.12     26.8  1.2e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 lm                              16      0      0   9.91  12.42  21.58   13.29
 ALL                             16      0      0   9.91  12.42  21.58   13.29
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      8      0      0   5.94  13.07  25.82   12.80
 prime column                     4      0      0  10.12  13.35  21.58   14.03
 even column                      3      0      0  11.79  12.41  19.08   14.08
 odd column                       1      0      0  12.22  12.22  12.22   12.22
 ALL                             16      0      0   9.91  12.42  21.58   13.29
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     10      0      0   9.91  12.39  19.08   11.75
 > 65536 points                   3      0      0  11.79  12.22  13.78   12.57
 257..1024                        2      0      0  21.58  23.70  25.82   23.60
 1025..4096                       1      0      0  17.19  17.19  17.19   17.19
 ALL                             16      0      0   9.91  12.42  21.58   13.29
```


worst 10: 4x4x4x4x4x53 (lm 5.94), 16x4x4x4x4x16 (lm 9.91), 53x4x4x4x4x4 (lm 10.12), 2x4x6x8x10x12 (lm 10.26), 10x10x10x10x10x10 (lm 11.79), 9x9x9x9x9x9 (lm 12.22), 4x4x4x8x8x8 (lm 12.37), 6x6x6x6x6x6 (lm 12.41), 5x5x5x5x5x5 (lm 12.43), 8x8x8x8x8x8 (lm 13.78)
best 5: 2x4x2x4x2x4 (lm 25.82), 3x3x3x3x3x3 (lm 21.58), 12x10x8x6x4x2 (lm 19.08), 4x4x4x4x4x4 (lm 17.19), 8x8x8x4x4x4 (lm 15.73)
