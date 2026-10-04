# gauntlet report

run: `odd15_c2c_v2_2026-10-04`  contract file suffix: `(oop, T=1)`  cells: 15 listed, 15 benched, comparator: MKL

control cell: 4 readings, 1.074..1.089 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      3375  3^3.5^3          chain3   raced           5170       6908    1.33  5.9e-16  
      6561  3^8              flat     raced          13919      15717    0.99  9.0e-16  
     10125  3^4.5^3          flat     raced          19605      23775    1.14  7.6e-16  
     15625  5^6              flat     raced          32549      36135    1.10  5.5e-16  
     19683  3^9              flat     raced          44444      52496    1.07  1.0e-15  
     30375  3^5.5^3          flat     raced          65058      80000    1.21  8.3e-16  
     45927  3^8.7            flat     raced         112328     128201    1.12  8.3e-16  
     50625  3^4.5^4          flat     raced         120969     144024    1.15  9.1e-16  
     59049  3^10             flat     raced         154230     184245    1.19  1.2e-15  
     99225  3^4.5^2.7^2      flat     raced         311425     344567    1.01  1.4e-15  
    117649  7^6              flat     raced         332169     361340    1.09  7.8e-16  
    151875  3^5.5^4          flat     raced         417446     600265    1.38  8.2e-16  
    177147  3^11             flat     raced         561209     744545    1.29  1.2e-15  
    225225  3^2.5^2.7.11.13  flat     raced         765375     975650    1.24  1.0e-15  
    253125  3^4.5^5          flat     raced         863237    1117519    1.23  1.5e-15  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 flat        14      0      1   1.01   1.15   1.29    1.15
 chain3       1      0      0   1.33   1.33   1.33    1.33
 ALL         15      0      1   1.01   1.15   1.33    1.16
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             2   1.16      1      0
 8192..32767            4   1.12      0      0
 32768..131071          5   1.12      0      0
 131072..253125         4   1.26      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 flat                                            14   1.15      1   1.15
 chain3                                           1   1.33      0   1.33
```


flip agreement: our two readings more than 25% apart at 0 of 15 cells.

worst 10: 6561 (flat 0.99), 99225 (flat 1.01), 19683 (flat 1.07), 117649 (flat 1.09), 15625 (flat 1.10), 45927 (flat 1.12), 10125 (flat 1.14), 50625 (flat 1.15), 59049 (flat 1.19), 30375 (flat 1.21)
best 5: 151875 (flat 1.38), 3375 (chain3 1.33), 177147 (flat 1.29), 225225 (flat 1.24), 253125 (flat 1.23)
