# gauntlet report (2D)

run: `rb_2d_r2c_fftw_replay`  contract: 2D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 36 listed, 36 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 4 readings, 1.323..1.422


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
      2x1024  2            chain  raced           1129       1101    0.96     49.9  3.7e-16
      2x2048  2            chain  raced           2548       2528    0.99     48.2  2.8e-16
      2x4096  2            chain  raced           5798       5825    1.00     45.9  3.0e-16
         4x4  2^2          chain  raced             11         13    1.24     15.2  0.0e+00
         4x8  2^2          chain  raced             14         19    1.39     29.1  2.5e-16
      4x1024  2^2          chain  raced           2444       2471    1.00     50.3  3.1e-16
      4x2048  2^2          chain  raced           5524       5339    0.97     48.2  3.9e-16
      4x4096  2^2          chain  raced          11951      11759    0.97     48.0  3.3e-16
         8x4  2^3          chain  raced             35         21    0.59     11.6  7.2e-17
       8x512  2^3          chain  raced           2263       2351    1.02     54.3  3.9e-16
      8x1024  2^3          chain  raced           5131       5059    0.97     51.9  3.3e-16
      8x4096  2^3          chain  raced          24170      26619    0.99     50.8  3.5e-16
      8x8192  2^3          chain  raced          50913      54022    0.85     51.5  4.4e-16
      16x256  2^4          chain  raced           2354       2393    0.97     52.2  2.6e-16
      16x512  2^4          chain  raced           6404       5263    0.63     41.6  3.2e-16
     16x1024  2^4          chain  raced          12908      11087    0.78     44.4  2.8e-16
     16x2048  2^4          chain  raced          28373      23915    0.84     43.3  3.7e-16
     16x4096  2^4          chain  raced          58023      53278    0.89     45.2  3.5e-16
     16x8192  2^4          chain  raced         118720     120771    0.84     46.9  3.5e-16
      32x512  2^5          chain  raced          11462      11156    0.97     50.0  3.4e-16
     32x1024  2^5          chain  raced          24420      22904    0.93     50.3  3.8e-16
     32x2048  2^5          chain  raced          52807      50987    0.96     49.6  5.1e-16
       64x16  2^6          chain  raced            522        699    1.29     49.0  3.6e-16
      64x256  2^6          chain  raced          11065      11503    0.90     51.8  3.3e-16
     64x1024  2^6          chain  raced          50390      49959    0.99     52.0  4.1e-16
     64x2048  2^6          chain  raced         111010     110305    0.99     50.2  4.6e-16
     64x4096  2^6          chain  raced         262607     263216    0.99     44.9  4.0e-16
       128x4  2^7          chain  raced            399        375    0.93     28.9  2.0e-16
    128x1024  2^7          chain  raced         108717     110160    0.99     51.2  5.0e-16
       256x4  2^8          chain  raced            919        861    0.93     27.9  3.3e-16
       256x8  2^8          chain  raced           1708       1628    0.94     33.0  2.6e-16
       512x4  2^9          chain  raced           2444       1980    0.75     23.0  3.2e-16
      1024x4  2^10         chain  raced           5731       5537    0.96     21.4  3.2e-16
      4096x4  2^12         chain  raced          32636      31305    0.95     17.6  3.8e-16
      8192x4  2^13         chain  raced          79404      65483    0.80     15.5  3.0e-16
      8192x8  2^13         chain  raced         116561     117087    0.99     22.5  3.1e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           36      4     32   0.78   0.97   1.02    0.94
 ALL                             36      4     32   0.78   0.97   1.02    0.94
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     36      4     32   0.78   0.97   1.02    0.94
 ALL                             36      4     32   0.78   0.97   1.02    0.94
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                     18      2     18   0.78   0.96   0.99    0.90
 1025..4096                       8      1      7   0.75   0.97   1.02    0.94
 > 65536 points                   4      0      4   0.84   0.99   0.99    0.95
 <= 256 points                    3      1      1   0.59   1.24   1.39    1.01
 257..1024                        3      0      2   0.93   0.93   1.29    1.04
 ALL                             36      4     32   0.78   0.97   1.02    0.94
```


worst 10: 8x4 (chain 0.59), 16x512 (chain 0.63), 512x4 (chain 0.75), 16x1024 (chain 0.78), 8192x4 (chain 0.80), 16x2048 (chain 0.84), 16x8192 (chain 0.84), 8x8192 (chain 0.85), 16x4096 (chain 0.89), 64x256 (chain 0.90)
best 5: 4x8 (chain 1.39), 64x16 (chain 1.29), 4x4 (chain 1.24), 8x512 (chain 1.02), 4x1024 (chain 1.00)
