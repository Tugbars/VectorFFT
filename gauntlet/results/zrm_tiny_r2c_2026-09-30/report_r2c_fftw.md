# gauntlet report

run: `zrm_tiny_r2c_2026-09-30`  contract file suffix: `_r2c_fftw`  cells: 63 listed, 63 benched, comparator: FFTW

control cell: 4 readings, 1.135..1.159 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                -        replayed           9          4    0.44  0.0e+00  
         3  3                -        raced              4          4    1.06  0.0e+00  
         4  2^2              -        raced              4          4    1.12  0.0e+00  
         5  5                -        raced              5          5    1.11  1.6e-17  
         6  2.3              -        raced              5          5    0.96  1.4e-16  
         7  7                -        raced              6          6    1.06  8.1e-17  
         8  2^3              -        raced              6          6    1.11  7.4e-17  
         9  3^2              -        raced              7          8    1.09  2.7e-16  
        10  2.5              -        raced              7          7    1.03  1.6e-16  
        11  11               -        raced              9         10    1.11  2.4e-16  
        12  2^2.3            -        raced              8          8    0.98  2.7e-16  
        13  13               -        raced             12         10    0.87  2.0e-16  
        14  2.7              -        raced             10         11    1.05  2.4e-16  
        15  3.5              -        raced             15         11    0.70  1.9e-16  
        16  2^4              -        raced             10         10    0.99  4.6e-17  
        17  17               -        raced             20         56    2.73  3.2e-16  
        18  2.3^2            -        raced             18         15    0.81  4.1e-16  
        19  19               -        raced             25         69    2.73  2.0e-16  
        20  2^2.5            -        raced             18         12    0.67  1.9e-16  
        21  3.7              -        raced             28         47    1.69  3.8e-16  
        22  2.11             -        raced             20         17    0.84  2.9e-16  
        23  23               -        raced             36         86    2.35  3.7e-16  
        24  2^3.3            -        raced             17         17    1.04  3.3e-16  
        25  5^2              -        raced             29         27    0.91  2.7e-16  
        26  2.13             -        raced             23         20    0.85  4.4e-16  
        27  3^3              -        raced             48         54    1.11  4.3e-16  
        28  2^2.7            -        raced             22         20    0.90  1.7e-16  
        29  29               -        raced             55        111    1.98  2.1e-16  
        30  2.3.5            -        raced             29         21    0.72  2.0e-16  
        31  31               -        raced             61        129    2.09  5.5e-16  
        32  2^5              -        raced             18         18    1.03  1.2e-16  
        33  3.11             -        raced             40         65    1.63  1.8e-16  
        34  2.17             -        raced             27         53    1.95  4.0e-16  
        35  5.7              -        raced             40         65    1.59  2.8e-16  
        36  2^2.3^2          -        raced             22         22    1.02  2.6e-16  
        37  37               -        raced             84        167    1.95  3.3e-16  
        38  2.19             -        raced             32         64    2.00  3.1e-16  
        39  3.13             -        raced             45         70    1.42  2.1e-16  
        40  2^3.5            -        raced             19         23    1.24  1.7e-16  
        41  41               -        raced            102        196    1.88  2.6e-16  
        42  2.3.7            -        raced             31         30    0.96  2.6e-16  
        43  43               -        raced            116        210    1.77  6.1e-16  
        44  2^2.11           -        raced             34         28    0.82  2.1e-16  
        45  3^2.5            -        raced             45         70    1.43  3.4e-16  
        46  2.23             -        raced             42         87    2.06  4.1e-16  
        47  47               -        raced            128        244    1.90  4.3e-16  
        48  2^4.3            -        raced             22         29    1.34  2.0e-16  
        49  7^2              -        raced             47         86    1.81  2.8e-16  
        50  2.5^2            -        raced             33         31    0.94  2.5e-16  
        51  3.17             -        raced             63        150    1.43  4.1e-16  flips differ 1.70x
        52  2^2.13           -        raced             41         32    0.78  5.8e-16  
        53  53               -        raced            211        299    1.42  4.1e-16  
        54  2.3^3            -        raced             39         33    0.83  3.2e-16  
        55  5.11             -        raced             53         97    1.80  3.1e-16  
        56  2^3.7            -        raced             31         30    0.95  2.1e-16  
        57  3.19             -        raced             87        189    1.61  2.8e-16  flips differ 1.34x
        58  2.29             -        raced             61        125    2.05  4.9e-16  
        59  59               -        raced            209        362    1.72  2.7e-16  
        60  2^2.3.5          -        raced             29         33    1.16  3.2e-16  
        61  61               -        raced            155        386    2.46  5.0e-16  
        62  2.31             -        raced             71        148    2.09  4.8e-16  
        63  3^2.7            -        raced             58        103    1.70  4.1e-16  
        64  2^6              -        raced             26         34    1.32  1.6e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 -           63      5     19   0.82   1.12   2.06    1.27
 ALL         63      5     19   0.82   1.12   2.06    1.27
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   1.06      2      1
 8..31                 24   1.04     11      3
 32..64                33   1.59      6      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 -                                               57   1.24     17   1.31
 pow2                                             6   1.07      2   0.95
```


flip agreement: our two readings more than 25% apart at 2 of 63 cells.

worst 10: 2 (- 0.44), 20 (- 0.67), 15 (- 0.70), 30 (- 0.72), 52 (- 0.78), 18 (- 0.81), 44 (- 0.82), 54 (- 0.83), 22 (- 0.84), 26 (- 0.85)
best 5: 19 (- 2.73), 17 (- 2.73), 61 (- 2.46), 23 (- 2.35), 31 (- 2.09)
