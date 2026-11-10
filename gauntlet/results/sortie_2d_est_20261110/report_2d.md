# gauntlet report (2D)

run: `sortie_2d_est_20261110`  contract: 2D c2c interleaved, natural, out of place, K=1  cells: 6 listed, 6 benched, comparator: MKL DFTI 2D (out of place)

control cell 64x64: 4 readings, 1.368..1.497


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
     512x512  2^9          chain  replayed      523175     947693    1.78     45.1  4.6e-16
    720x1280  2^4.3^2.5    chain  replayed     3572187    4857318    1.33     25.6  6.5e-16
   1024x1024  2^10         chain  replayed     3390162    6110937    1.75     30.9  1.3e-15
   1080x1920  2^3.3^3.5    chain  replayed     8888875   13153237    1.44     24.5  9.4e-16
   2048x1024  2^11         chain  replayed     8023862   21654475    2.66     27.4  1.5e-15
   2048x2048  2^11         chain  replayed    17120425   29835550    1.67     26.9  2.9e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                            6      0      0   1.33   1.71   2.66    1.73
 ALL                              6      0      0   1.33   1.71   2.66    1.73
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                      4      0      0   1.67   1.77   2.66    1.93
 even column                      2      0      0   1.33   1.39   1.44    1.39
 ALL                              6      0      0   1.33   1.71   2.66    1.73
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                   6      0      0   1.33   1.71   2.66    1.73
 ALL                              6      0      0   1.33   1.71   2.66    1.73
```


worst 10: 720x1280 (chain 1.33), 1080x1920 (chain 1.44), 2048x2048 (chain 1.67), 1024x1024 (chain 1.75), 512x512 (chain 1.78), 2048x1024 (chain 2.66)
best 5: 2048x1024 (chain 2.66), 512x512 (chain 1.78), 1024x1024 (chain 1.75), 2048x2048 (chain 1.67), 1080x1920 (chain 1.44)
