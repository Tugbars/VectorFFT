# gauntlet report (3D)

run: `mt8_3d_c2r_oop`  contract: 3D c2r interleaved, natural, out of place, K=1_c2r_mt8  cells: 11 listed, 11 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.107..1.206

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  8x128x2048  2^3          plane+payonce raced         439563     677531    1.53    250.5  1.3e-15
     9x16x30  3^2          plane+child raced           2911       3175    1.01     44.8  1.1e-15
    16x16x16  2^4          plane+child raced           1749       2306    1.10     70.3  6.7e-16
  16x64x1024  2^4          plane+payonce raced         162087     217556    1.32    323.5  1.1e-15
  16x256x256  2^4          plane+payonce raced         174363     237968    1.28    300.7  1.1e-15
   32x8x1000  2^5          plane+payonce raced          45267      73360    1.43    254.0  1.7e-15
    32x32x32  2^5          plane+child raced           6860      10278    1.29    179.1  7.8e-16
  32x256x256  2^5          plane+payonce raced         477125     512038    1.07    230.8  1.3e-15
    64x64x64  2^6          plane+payonce raced          47013      53757    1.10    250.9  1.0e-15
  64x128x128  2^6          plane+payonce raced         175887     250118    1.40    298.1  1.3e-15
 128x128x128  2^7          plane+payonce raced         429325     528475    1.13    256.4  1.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 plane+payonce                    8      0      0   1.07   1.30   1.53    1.27
 plane+child                      3      0      0   1.01   1.10   1.29    1.13
 ALL                             11      0      0   1.07   1.28   1.43    1.23
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     10      0      0   1.10   1.28   1.53    1.26
 odd column                       1      0      0   1.01   1.01   1.01    1.01
 ALL                             11      0      0   1.07   1.28   1.43    1.23
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   8      0      0   1.07   1.30   1.53    1.27
 4097..65536                      2      0      0   1.01   1.15   1.29    1.14
 1025..4096                       1      0      0   1.10   1.10   1.10    1.10
 ALL                             11      0      0   1.07   1.28   1.43    1.23
```


worst 10: 9x16x30 (plane+child 1.01), 32x256x256 (plane+payonce 1.07), 64x64x64 (plane+payonce 1.10), 16x16x16 (plane+child 1.10), 128x128x128 (plane+payonce 1.13), 16x256x256 (plane+payonce 1.28), 32x32x32 (plane+child 1.29), 16x64x1024 (plane+payonce 1.32), 64x128x128 (plane+payonce 1.40), 32x8x1000 (plane+payonce 1.43)
best 5: 8x128x2048 (plane+payonce 1.53), 32x8x1000 (plane+payonce 1.43), 64x128x128 (plane+payonce 1.40), 16x64x1024 (plane+payonce 1.32), 32x32x32 (plane+child 1.29)
