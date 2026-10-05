# gauntlet report

run: `fftw_5cells_2026-10-05`  contract file suffix: `_fftw`  cells: 5 listed, 5 benched, comparator: FFTW

control cell: 4 readings, 1.379..1.409 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        97  97               prime    raced            222        540    2.42  5.4e-16  rader
      1000  2^3.5^3          chain3   raced           1061        919    0.86  3.3e-16  
      1024  2^10             ztt      raced            725        802    1.09  4.0e-16  
      4096  2^12             ztt      raced           3563       5008    1.39  2.9e-16  
     16384  2^14             ztt      raced          17220      28328    1.37  3.6e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          3      0      0   1.09   1.37   1.39    1.28
 chain3       1      0      1   0.86   0.86   0.86    0.86
 prime        1      0      0   2.42   2.42   2.42    2.42
 ALL          5      0      1   0.86   1.37   2.42    1.34
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                1   2.42      0      0
 512..2047              2   0.98      1      0
 2048..8191             1   1.39      0      0
 8192..16384            1   1.37      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                             3   1.37      0   1.28
 prime N, rader                                   1   2.42      0   2.42
 chain3                                           1   0.86      1   0.86
```


flip agreement: our two readings more than 25% apart at 0 of 5 cells.

worst 10: 1000 (chain3 0.86), 1024 (ztt 1.09), 16384 (ztt 1.37), 4096 (ztt 1.39), 97 (prime 2.42)
best 5: 97 (prime 2.42), 4096 (ztt 1.39), 16384 (ztt 1.37), 1024 (ztt 1.09), 1000 (chain3 0.86)
