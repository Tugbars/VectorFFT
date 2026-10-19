# gauntlet report

run: `pow2rec_t1_fftw_2026-10-20`  contract file suffix: `_fftw`  cells: 23 listed, 23 benched, comparator: FFTW

control cell: 4 readings, 1.375..1.418 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     replayed           2          3    1.26  0.0e+00  
         4  2^2              mono     replayed           3          3    1.11  0.0e+00  
         8  2^3              mono     replayed           5          4    0.79  5.6e-17  
        16  2^4              ztt      replayed           8          8    0.84  7.5e-17  
        32  2^5              2p       replayed          15         17    1.08  1.3e-16  
        64  2^6              2p       replayed          30         30    0.70  1.5e-16  flips differ 1.43x
       128  2^7              2p       replayed          64         59    0.92  2.1e-16  
       256  2^8              2p       replayed         134        124    0.92  3.1e-16  
       512  2^9              2p       replayed         297        274    0.92  3.1e-16  
      1024  2^10             ztt      replayed         715        740    1.03  3.0e-16  
      2048  2^11             ztt      replayed        1607       2068    1.14  3.1e-16  
      4096  2^12             ztt      replayed        3562       5069    1.41  3.4e-16  
      8192  2^13             ztt      replayed        7939      10629    1.33  3.4e-16  
     16384  2^14             ztt      replayed       17661      24557    1.14  3.6e-16  
     32768  2^15             ztt      replayed       39284      53627    1.14  3.4e-16  
     65536  2^16             ztt      replayed       88123     127420    1.42  3.3e-16  
    131072  2^17             ztt      replayed      244333     357180    1.45  4.6e-16  
    262144  2^18             ztt      replayed      574125     846775    1.39  3.3e-16  
    524288  2^19             fs       replayed     1396225    2235737    1.57  4.5e-16  
   1048576  2^20             fs       replayed     3387662    5483118    1.61  1.1e-15  
   2097152  2^21             fs       replayed     8396325   12438868    1.46  2.7e-15  
   4194304  2^22             fs       replayed    18675237   29118806    1.44  8.0e-16  
   8388608  2^23             fs       replayed    41052675   62806893    1.48  9.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt         10      0      1   1.03   1.23   1.45    1.21
 fs           5      0      0   1.44   1.48   1.61    1.51
 2p           5      1      4   0.70   0.92   1.08    0.90
 mono         3      1      1   0.79   1.11   1.26    1.04
 ALL         23      2      6   0.84   1.14   1.48    1.17
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   2   1.19      0      0
 8..31                  2   0.82      2      1
 32..127                2   0.89      1      1
 128..511               2   0.92      2      0
 512..2047              2   0.97      1      0
 2048..8191             2   1.27      0      0
 8192..32767            2   1.23      0      0
 32768..131071          2   1.28      0      0
 131072..524287         2   1.42      0      0
 524288..2097151        2   1.59      0      0
 2097152..8388607       2   1.45      0      0
 8388608..8388608       1   1.48      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            23   1.14      6   1.17
```


flip agreement: our two readings more than 25% apart at 1 of 23 cells.

worst 10: 64 (2p 0.70), 8 (mono 0.79), 16 (ztt 0.84), 512 (2p 0.92), 128 (2p 0.92), 256 (2p 0.92), 1024 (ztt 1.03), 32 (2p 1.08), 4 (mono 1.11), 16384 (ztt 1.14)
best 5: 1048576 (fs 1.61), 524288 (fs 1.57), 8388608 (fs 1.48), 2097152 (fs 1.46), 131072 (ztt 1.45)
