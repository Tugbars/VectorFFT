# gauntlet report

run: `regr_1d_head_2026-09-28`  contract file suffix: `(oop, T=1)`  cells: 135 listed, 129 benched, comparator: MKL

control cell: 6 readings, 0.924..1.081 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced              3         11    3.23  0.0e+00  
         3  3                mono     raced              4         11    2.94  1.5e-16  
         4  2^2              mono     raced              4         11    2.94  0.0e+00  
         5  5                mono     raced              5         12    2.20  7.5e-17  
         6  2.3              mono     raced              6         12    2.06  1.9e-16  
         7  7                mono     raced              7         12    1.83  2.6e-16  
         8  2^3              mono     raced              6         11    1.85  7.4e-17  
         9  3^2              mono     raced              8         13    1.55  4.2e-16  
        11  11               mono     replayed          10         14    1.30  1.2e-16  
        12  2^2.3            2p       raced             10         13    1.33  2.4e-16  
        13  13               mono     raced             14         15    1.07  5.0e-16  
        15  3.5              2p       raced             15         14    0.94  3.3e-16  
        16  2^4              ztt      raced              9         12    1.35  7.5e-17  
        17  17               mono     raced             22         40    1.77  2.0e-16  
        19  19               mono     raced             27         45    1.69  4.6e-16  
        20  2^2.5            2p       raced             14         22    0.93  2.2e-16  flips differ 1.71x
        21  3.7              2p       raced             19         25    1.28  3.3e-16  
        23  23               mono     raced             38         66    1.72  2.5e-16  
        24  2^3.3            2p       raced             13         29    2.12  2.8e-16  
        25  5^2              2p       raced             19         26    1.36  1.8e-16  
        27  3^3              flat     raced             28         32    1.15  3.8e-16  
        29  29               mono     raced             66        107    1.61  2.8e-16  
        31  31               mono     raced             71        121    1.71  3.3e-16  
        32  2^5              2p       raced             16         16    1.04  1.7e-16  
        37  37               prime    raced            101        185    1.80  6.9e-16  rader
        45  3^2.5            2p       raced             34         49    1.30  3.7e-16  
        48  2^4.3            2p       raced             27         46    1.64  2.7e-16  
        59  59               prime    raced            220        553    2.50  4.9e-16  bluestein
        60  2^2.3.5          2p       raced             35         62    1.74  3.4e-16  
        63  3^2.7            2p       raced             47         67    1.44  4.2e-16  
        64  2^6              2p       raced             30         31    1.02  1.5e-16  
        75  3.5^2            2p       raced             58         79    1.33  4.4e-16  
        89  89               prime    raced            213       1508    7.08  3.9e-16  rader
        96  2^5.3            2p       raced             51         97    1.88  3.0e-16  
       105  3.5.7            2p       raced             83        113    1.02  5.0e-16  flips differ 1.34x
       111  3.37             2p       raced            223        565    2.53  4.8e-16  
       120  2^3.3.5          2p       raced             77        126    1.63  3.7e-16  
       125  5^3              2p       raced            104        133    1.28  4.8e-16  
       128  2^7              2p       raced             66         68    0.82  3.2e-16  flips differ 1.26x
       135  3^3.5            chain3   raced            112        156    1.39  4.3e-16  
       139  139              prime    raced            465        960    1.84  7.5e-16  rader
       143  11.13            2p       raced            129        149    1.15  4.8e-16  
       187  11.17            2p       raced            188        452    1.85  7.1e-16  flips differ 1.30x
       192  2^6.3            2p       raced            110        183    1.64  3.4e-16  
       197  197              prime    raced            576       1296    2.23  5.3e-16  rader
       225  3^2.5^2          chain3   raced            192        268    1.40  4.8e-16  
       240  2^4.3.5          chain3   raced            164        250    1.52  3.4e-16  
       243  3^5              chain3   raced            221        298    1.31  4.6e-16  
       256  2^8              2p       raced            134        137    0.98  0.0e+00  
       281  281              prime    raced            814       2114    2.51  6.3e-16  rader
       343  7^3              chain3   raced            346        404    1.16  4.9e-16  
       360  2^3.3^2.5        chain3   raced            289        434    1.50  5.5e-16  
       375  3.5^3            chain3   raced            343        478    1.37  6.8e-16  
       391  17.23            2p       raced            549       1779    2.75  6.7e-16  
       397  397              prime    raced           1506       3160    2.07  8.9e-16  rader
       441  3^2.7^2          chain3   raced            423        567    1.25  5.3e-16  
       443  443              prime    raced           2287       2639    1.01  5.2e-16  bluestein
       480  2^5.3.5          chain3   raced            354        585    1.18  3.2e-16  flips differ 1.41x
       512  2^9              2p       raced            297        295    0.93  2.3e-16  
       521  521              prime    raced           1971       4597    2.21  1.1e-15  rader
       613  613              prime    raced           2776       4839    1.56  1.0e-15  rader
       625  5^4              2p       raced            677        873    1.26  4.1e-16  
       640  2^7.5            2p       raced            506        823    1.59  4.2e-16  
       667  23.29            2p       raced           1350       4116    2.35  5.4e-16  flips differ 1.30x
       720  2^4.3^2.5        chain3   raced            719        938    1.30  3.9e-16  
       727  727              prime    raced           2996       6318    1.92  6.5e-16  rader
       729  3^6              2p       raced            826       1087    1.30  5.1e-16  
       739  739              prime    raced           4925       6326    1.28  5.1e-16  bluestein
       953  953              prime    raced           4105       5994    1.45  8.2e-16  rader
       971  971              prime    raced           5270       5934    1.12  6.9e-16  bluestein
      1000  2^3.5^3          chain3   raced           1054       1418    1.33  3.7e-16  
      1024  2^10             ztt      raced            718        822    1.13  4.0e-16  
      1125  3^2.5^3          flat     raced           1584       1818    1.11  7.8e-16  
      1147  31.37            flat     raced           3325      10133    3.02  8.4e-16  
      1193  1193             prime    raced          10679      11606    0.92  6.0e-16  bluestein
      1200  2^4.3.5^2        chain3   raced           1294       1710    1.31  4.8e-16  
      1201  1201             prime    raced           6702      11581    1.49  8.0e-16  rader
      1327  1327             prime    raced           6249      13526    2.13  1.0e-15  rader
      1487  1487             prime    raced          10570      13685    1.27  5.8e-16  bluestein
      1531  1531             prime    raced           6945      13715    1.92  7.5e-16  rader
      1536  2^9.3            chain3   raced           1639       2399    1.45  4.5e-16  
      1667  1667             prime    raced          10643      12496    1.16  5.9e-16  bluestein
      1871  1871             prime    raced           8748      12785    1.45  8.6e-16  rader
      2000  2^4.5^3          flat     raced           2979       3041    1.01  4.5e-16  
      2048  2^11             ztt      raced           1629       2040    1.24  3.1e-16  
      2069  2069             prime    raced          16732      22111    1.04  1.0e-15  flips differ 1.27x rader
      2143  2143             prime    raced          11903      22078    1.69  1.0e-15  rader
      2187  3^7              chain3   raced           3737       4510    1.19  5.2e-16  
      2351  2351             prime    raced          22949      24471    1.05  7.8e-16  rader
      2377  2377             prime    raced          15376      24563    1.54  9.2e-16  rader
      2400  2^5.3.5^2        ztt      raced           2482       4049    1.60  3.8e-16  
      2491  47.53            prime    raced          23202      37826    1.63  6.9e-16  bluestein
      2609  2609             prime    raced          22720      30326    1.30  6.6e-16  bluestein
      2689  2689             prime    raced           9833      30389    3.03  5.8e-16  rader
      2791  2791             prime    raced          20094      30362    1.49  9.2e-16  rader
      3023  3023             prime    raced          22512      30574    1.35  5.8e-16  bluestein
      3072  2^10.3           ztt      raced           2730       5231    1.91  3.0e-16  
      3121  3121             prime    raced          15176      31445    1.78  9.9e-16  rader
      3299  3299             prime    raced          22992      26441    1.14  7.0e-16  bluestein
      3375  3^3.5^3          chain3   raced           5178       6892    1.26  5.9e-16  
      3527  3527             prime    raced          23117      26633    1.13  7.9e-16  bluestein
      3599  59.61            prime    raced          23201      69450    2.96  6.6e-16  bluestein
      3697  3697             prime    raced          20274      27086    1.33  9.2e-16  rader
      3840  2^8.3.5          ztt      raced           4190       6729    1.47  4.1e-16  
      3907  3907             prime    raced          23350      26922    1.15  7.3e-16  bluestein
      4001  4001             prime    raced          15195      27219    1.78  7.8e-16  rader
      4096  2^12             ztt      raced           3575       3846    1.07  3.9e-16  
      4099  4099             prime    raced          50916      54065    1.06  4.9e-16  bluestein
      6144  2^11.3           ztt      raced           6020      11878    1.97  3.1e-16  
      6561  3^8              flat     raced          14131      15707    1.05  8.3e-16  
      8191  8191             prime    raced          49159      57814    1.15  6.8e-16  bluestein
      8192  2^13             ztt      raced           7756       8453    1.08  4.1e-16  
     10000  2^4.5^4          ztt      raced          11529      19170    1.66  4.7e-16  
     12288  2^12.3           ztt      raced          13022      24864    1.90  3.8e-16  
     15625  5^6              flat     raced          35275      36032    1.00  6.9e-16  
     16381  16381            prime    raced         115197     119420    1.01  6.3e-16  bluestein
     16384  2^14             ztt      raced          17183      18988    1.10  3.6e-16  
     19683  3^9              flat     raced          48199      52803    0.97  1.1e-15  
     24576  2^13.3           ztt      raced          27794      56839    1.58  4.0e-16  flips differ 1.31x
     32768  2^15             ztt      raced          38630      36907    0.95  3.4e-16  
     40000  2^6.5^4          ztt      raced          49810      87190    1.75  4.7e-16  
     49152  2^14.3           ztt      raced          65067     115612    1.77  4.1e-16  
     59049  3^10             flat     raced         163415     181850    1.08  1.0e-15  
     65536  2^16             ztt      raced          87993     111926    1.24  4.0e-16  
     65537  65537            prime    raced         356097    1771556    4.94  1.0e-15  rader
     98304  2^15.3           ztt      raced         159615     345002    2.09  4.0e-16  
    131072  2^17             ztt      raced         231080     267220    1.15  4.6e-16  
    196608  2^16.3           ztt      raced         394800     768565    1.92  4.7e-16  
    262144  2^18             ztt      raced         573988     737044    1.23  4.0e-16  
    524288  2^19             fs       refused            -          -       -        -  not benched
   1048576  2^20             fs       refused            -          -       -        -  not benched
   2097152  2^21             fs       refused            -          -       -        -  not benched
   4194304  2^22             fs       refused            -          -       -        -  not benched
   8388608  2^23             fs       refused            -          -       -        -  not benched
  16777216  2^24             fs       refused            -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime       41      0      1   1.05   1.49   2.51    1.63
 2p          29      0      5   0.93   1.33   2.35    1.39
 ztt         21      0      1   1.08   1.47   1.92    1.44
 mono        15      0      0   1.30   1.77   2.94    1.88
 chain3      15      0      0   1.18   1.31   1.50    1.32
 flat         8      0      1   0.97   1.07   3.02    1.20
 ALL        129      0      8   1.02   1.40   2.35    1.50
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   2.57      0      0
 8..31                 17   1.36      2      0
 32..127               15   1.63      0      0
 128..511              20   1.39      2      0
 512..2047             26   1.32      2      0
 2048..8191            27   1.33      0      0
 8192..32767            8   1.09      1      0
 32768..131071          7   1.75      1      0
 131072..262144         3   1.23      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 2p                                              24   1.40      2   1.50
 prime N, rader                                  24   1.79      0   1.92
 pow2                                            18   1.12      4   1.25
 prime N, bluestein                              15   1.15      1   1.20
 chain3                                          15   1.31      0   1.32
 mono                                            12   1.71      0   1.74
 ztt                                             11   1.77      0   1.77
 flat                                             8   1.07      1   1.20
 composite with a prime >= 53 (prime cell)        2   2.30      0   2.20
```


flip agreement: our two readings more than 25% apart at 8 of 129 cells.

worst 10: 128 (2p 0.82), 1193 (prime 0.92), 512 (2p 0.93), 20 (2p 0.93), 15 (2p 0.94), 32768 (ztt 0.95), 19683 (flat 0.97), 256 (2p 0.98), 15625 (flat 1.00), 443 (prime 1.01)
best 5: 89 (prime 7.08), 65537 (prime 4.94), 2 (mono 3.23), 2689 (prime 3.03), 1147 (flat 3.02)
