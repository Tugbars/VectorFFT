# gauntlet report

run: `odd15_c2c_v3_2026-10-04`  contract file suffix: `(oop, T=1)`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 4 readings, 1.068..1.083 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          chain3   raced           5144       6917    1.31  5.9e-16  
      6561  3^8              chain3   raced          15512      15728    1.00  6.6e-16  
     10125  3^4.5^3          chain3   raced          23647      23835    1.00  6.9e-16  
     15625  5^6              flat     raced          32648      36172    1.11  5.5e-16  
     19683  3^9              flat     raced          45163      52547    1.16  1.0e-15  
     30375  3^5.5^3          flat     raced          67129      80050    1.16  8.0e-16  
     45927  3^8.7            flat     raced         111588     128387    0.91  8.0e-16  flips differ 1.27x
     50625  3^4.5^4          flat     raced         120423     142432    1.11  1.0e-15  
     59049  3^10             flat     raced         157776     185356    1.17  1.1e-15  
     99225  3^4.5^2.7^2      flat     raced         269765     347917    1.28  9.2e-16  
    117649  7^6              flat     raced         333806     364162    1.07  7.8e-16  
    151875  3^5.5^4          flat     raced         456315     598688    1.31  8.6e-16  
    177147  3^11             flat     raced         554745     742559    1.33  1.3e-15  
    225225  3^2.5^2.7.11.13  flat     raced         740737     962693    1.30  9.4e-16  
    253125  3^4.5^5          flat     raced         800113    1101856    1.35  1.0e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 flat        12      0      1   1.07   1.17   1.33    1.18
 chain3       3      0      1   1.00   1.00   1.31    1.09
 ALL         15      0      2   1.00   1.16   1.33    1.16
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   1.15      1      0
 8192..32767            4   1.13      0      0
 32768..131071          5   1.11      1      0
 131072..253125         4   1.32      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 flat                                            12   1.17      1   1.18
 chain3                                           3   1.00      1   1.09
```


flip agreement: our two readings more than 25% apart at 1 of 15 cells.

worst 10: 45927 (flat 0.91), 6561 (chain3 1.00), 10125 (chain3 1.00), 117649 (flat 1.07), 15625 (flat 1.11), 50625 (flat 1.11), 19683 (flat 1.16), 30375 (flat 1.16), 59049 (flat 1.17), 99225 (flat 1.28)
best 5: 253125 (flat 1.35), 177147 (flat 1.33), 3375 (chain3 1.31), 151875 (flat 1.31), 225225 (flat 1.30)
