# gauntlet report

run: `t8_pow2_now_20261021`  contract file suffix: `_mt8`  cells: 11 listed, 11 benched, comparator: MKL

control cell: 4 readings, 1.017..1.053 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      4096  2^12             ztt      replayed        3585       3782    1.02  3.9e-16  
      8192  2^13             ztt      replayed        6072       8255    1.27  3.4e-16  
     16384  2^14             ztt      replayed       11341      14539    1.21  3.6e-16  
     32768  2^15             ztt      replayed       21270      21325    0.93  3.5e-16  
     65536  2^16             ztt      replayed       26643      40746    1.48  4.0e-16  
    131072  2^17             ztt      replayed       53513      81040    1.45  4.6e-16  
    262144  2^18             ztt      replayed      109863     166894    1.45  4.0e-16  
    524288  2^19             fs       raced         283213     330618    1.14  4.3e-16  
   1048576  2^20             fs       raced         586425     896293    1.51  1.1e-15  
   2097152  2^21             fs       raced        1777100    2636862    1.46  2.7e-15  
   4194304  2^22             fs       raced        6236288    7430343    1.16  9.2e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt          7      0      1   0.93   1.27   1.48    1.24
 fs           4      0      0   1.14   1.31   1.51    1.31
 ALL         11      0      1   1.02   1.27   1.48    1.27
```


## by size
```
 band               cells median   <1.0   <0.8
 2048..8191             1   1.02      0      0
 8192..32767            2   1.24      0      0
 32768..131071          2   1.20      1      0
 131072..524287         2   1.45      0      0
 524288..2097151        2   1.32      0      0
 2097152..4194304       2   1.31      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 pow2                                            11   1.27      1   1.27
```


flip agreement: our two readings more than 25% apart at 0 of 11 cells.

worst 10: 32768 (ztt 0.93), 4096 (ztt 1.02), 524288 (fs 1.14), 4194304 (fs 1.16), 16384 (ztt 1.21), 8192 (ztt 1.27), 131072 (ztt 1.45), 262144 (ztt 1.45), 2097152 (fs 1.46), 65536 (ztt 1.48)
best 5: 1048576 (fs 1.51), 65536 (ztt 1.48), 2097152 (fs 1.46), 262144 (ztt 1.45), 131072 (ztt 1.45)
