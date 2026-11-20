# gauntlet report

run: `prime_sides_r2c_probe_avx2_2026-11-20`  contract file suffix: `_r2c_mt8_est`  cells: 9 listed, 9 benched, comparator: MKL

control cell: 4 readings, 1.133..1.239 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
     14357  7^2.293          -        -              34172     117664    3.09  7.0e-16  
     28714  2.7^2.293        -        -              41913      45348    0.92  7.1e-16  
     40983  3.19.719         -        -              51380     760109   14.76  9.0e-16  
     81966  2.3.19.719       -        -              95875     122291    0.93  8.9e-16  
    229377  3.157.487        -        -            1367347    4025594    2.93  1.0e-15  
    458754  2.3.157.487      -        -            1208538     503093    0.36  1.2e-15  
   1048577  17.61681         -        -            7124788   56386275    7.01  8.7e-16  
   2097151  7^2.127.337      -        -           15454275   60719606    3.56  8.1e-16  
   2097154  2.17.61681       -        -           15086288    8476193    0.55  9.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -            9      2      4   0.36   2.93  14.76    2.02
 ALL          9      2      4   0.36   2.93  14.76    2.02
```


## by size
```
 band               cells median   <1.0   <0.8
 8192..32767            2   2.01      1      0
 32768..131071          2   7.84      1      0
 131072..524287         2   1.64      1      1
 524288..2097151        2   5.29      0      0
 2097152..2097154       1   0.55      1      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                                9   2.93      4   2.02
```


flip agreement: our two readings more than 25% apart at 0 of 9 cells.

worst 10: 458754 (- 0.36), 2097154 (- 0.55), 28714 (- 0.92), 81966 (- 0.93), 229377 (- 2.93), 14357 (- 3.09), 2097151 (- 3.56), 1048577 (- 7.01), 40983 (- 14.76)
best 5: 40983 (- 14.76), 1048577 (- 7.01), 2097151 (- 3.56), 14357 (- 3.09), 229377 (- 2.93)
