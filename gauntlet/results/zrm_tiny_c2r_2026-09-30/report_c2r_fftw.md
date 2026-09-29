# gauntlet report

run: `zrm_tiny_c2r_2026-09-30`  contract file suffix: `_c2r_fftw`  cells: 63 listed, 62 benched, comparator: FFTW

control cell: 4 readings, 1.084..1.092 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     refused            -          -       -        -  not benched
         3  3                -        raced              4          4    0.98  2.2e-16  
         4  2^2              -        raced              4          4    0.98  0.0e+00  
         5  5                -        raced              5          5    0.97  1.3e-16  
         6  2.3              -        raced              5          4    0.80  1.9e-16  
         7  7                -        raced              6          6    0.95  8.0e-17  
         8  2^3              -        raced              6          6    0.96  1.7e-16  
         9  3^2              -        raced              7          7    0.87  2.2e-16  
        10  2.5              -        raced              7          7    0.95  2.2e-16  
        11  11               -        raced              9         10    1.13  4.3e-16  
        12  2^2.3            -        raced              9          8    0.84  2.3e-16  
        13  13               -        raced             13         11    0.87  4.7e-16  
        14  2.7              -        raced             11         10    0.95  2.9e-16  
        15  3.5              -        raced             15         10    0.69  4.1e-16  
        16  2^4              -        raced             12         10    0.83  1.1e-16  
        17  17               -        raced             20         59    2.87  4.6e-16  
        18  2.3^2            -        raced             15         15    0.96  4.6e-16  
        19  19               -        raced             27         71    2.65  2.3e-16  
        20  2^2.5            -        raced             16         12    0.78  2.7e-16  
        21  3.7              -        raced             30         49    1.60  6.2e-16  
        22  2.11             -        raced             19         17    0.82  4.0e-16  
        23  23               -        raced             38         89    2.34  4.0e-16  
        24  2^3.3            -        raced             14         17    1.20  3.0e-16  
        25  5^2              -        raced             34         19    0.57  3.5e-16  
        26  2.13             -        raced             24         18    0.77  5.7e-16  
        27  3^3              -        raced             39         53    1.35  3.9e-16  
        28  2^2.7            -        raced             22         19    0.82  3.2e-16  
        29  29               -        raced             56        114    2.02  3.0e-16  
        30  2.3.5            -        raced             26         20    0.73  4.7e-16  
        31  31               -        raced             63        132    2.09  6.8e-16  
        32  2^5              -        raced             16         20    1.22  2.3e-16  
        33  3.11             -        replayed          43         68    1.50  3.3e-16  
        34  2.17             -        raced             27         52    1.91  6.8e-16  
        35  5.7              -        replayed          40         65    1.59  2.6e-16  
        36  2^2.3^2          -        raced             19         22    1.18  4.5e-16  
        37  37               -        raced             87        170    1.94  3.5e-16  
        38  2.19             -        raced             32         63    1.97  4.3e-16  
        39  3.13             -        replayed          54         75    1.35  7.1e-16  
        40  2^3.5            -        raced             17         24    1.44  3.2e-16  
        41  41               -        raced            106        199    1.87  4.4e-16  
        42  2.3.7            -        raced             30         28    0.94  4.7e-16  
        43  43               -        raced            120        215    1.79  6.4e-16  
        44  2^2.11           -        raced             34         27    0.81  5.3e-16  
        45  3^2.5            -        replayed          48         72    1.47  6.3e-16  
        46  2.23             -        raced             39         85    2.12  5.3e-16  
        47  47               -        raced            132        249    1.88  7.2e-16  
        48  2^4.3            -        raced             19         28    1.44  3.8e-16  
        49  7^2              -        replayed          52         87    1.64  2.3e-16  
        50  2.5^2            -        raced             31         31    0.92  4.2e-16  
        51  3.17             -        replayed          72        150    2.08  5.6e-16  
        52  2^2.13           -        raced             40         30    0.75  7.0e-16  
        53  53               -        replayed         216        303    1.40  6.8e-16  
        54  2.3^3            -        raced             34         33    0.92  5.7e-16  
        55  5.11             -        replayed          60        101    1.68  3.3e-16  
        56  2^3.7            -        raced             30         31    1.01  3.8e-16  
        57  3.19             -        replayed          80        182    2.24  3.5e-16  
        58  2.29             -        raced             62        124    1.99  5.8e-16  
        59  59               -        replayed         214        366    1.70  4.9e-16  
        60  2^2.3.5          -        raced             29         32    1.12  4.8e-16  
        61  61               -        replayed         159        410    2.51  6.7e-16  
        62  2.31             -        raced             70        148    2.10  8.4e-16  
        63  3^2.7            -        replayed          65        103    1.57  6.6e-16  
        64  2^6              -        raced             24         33    1.33  2.2e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           62      6     25   0.80   1.27   2.10    1.28
 ALL         62      6     25   0.80   1.27   2.10    1.28
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   5   0.97      5      0
 8..31                 24   0.95     15      5
 32..64                33   1.57      5      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               57   1.35     22   1.30
 pow2                                             5   0.98      3   1.05
```


flip agreement: our two readings more than 25% apart at 0 of 62 cells.

worst 10: 25 (- 0.57), 15 (- 0.69), 30 (- 0.73), 52 (- 0.75), 26 (- 0.77), 20 (- 0.78), 6 (- 0.80), 44 (- 0.81), 28 (- 0.82), 22 (- 0.82)
best 5: 17 (- 2.87), 19 (- 2.65), 61 (- 2.51), 23 (- 2.34), 57 (- 2.24)
