# gauntlet report

run: `zen4_2_2048_A_intel`  contract file suffix: `_fftw`  cells: 2047 listed, 2047 benched, comparator: FFTW

control cell: 44 readings, 1.016..4.305 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     replayed           7          8    1.01  0.0e+00  
         3  3                mono     replayed          11          9    0.71  1.5e-16  flips differ 1.27x
         4  2^2              mono     replayed          10         10    0.89  0.0e+00  
         5  5                mono     replayed          15         13    0.82  1.5e-16  
         6  2.3              mono     replayed          15         13    0.77  2.8e-16  
         7  7                mono     replayed          19         15    0.80  2.6e-16  
         8  2^3              mono     replayed          17         13    0.76  5.6e-17  
         9  3^2              mono     replayed          22         18    0.83  4.2e-16  
        10  2.5              mono     replayed          22         16    0.71  1.8e-16  
        11  11               mono     replayed          30         21    0.72  1.6e-16  
        12  2^2.3            2p       replayed          27         17    0.64  2.8e-16  
        13  13               mono     replayed          40         24    0.61  6.0e-16  
        14  2.7              mono     replayed          32         22    0.70  1.6e-16  
        15  3.5              2p       replayed          34         24    0.69  3.3e-16  
        16  2^4              ztt      replayed          21         21    0.96  1.5e-16  
        17  17               mono     replayed          53        104    1.88  2.4e-16  
        18  2.3^2            2p       replayed          39         34    0.85  3.6e-16  
        19  19               mono     replayed          61        125    2.01  3.9e-16  
        20  2^2.5            2p       replayed          36         31    0.85  2.2e-16  
        21  3.7              2p       replayed          42         53    1.25  3.3e-16  
        22  2.11             mono     replayed          53         45    0.85  2.2e-16  
        23  23               mono     replayed          84        178    2.12  3.1e-16  
        24  2^3.3            2p       replayed          43         33    0.77  3.3e-16  
        25  5^2              2p       replayed          49         52    1.04  1.8e-16  
        26  2.13             mono     replayed          68         49    0.71  4.4e-16  
        27  3^3              flat     replayed          71         68    0.92  3.8e-16  
        28  2^2.7            2p       replayed          43         40    0.94  1.3e-16  
        29  29               mono     replayed         123        266    2.12  3.7e-16  
        30  2.3.5            2p       replayed          50         48    0.96  2.6e-16  
        31  31               mono     replayed         125        301    2.40  2.7e-16  
        32  2^5              2p       replayed          38         38    1.01  1.7e-16  
        33  3.11             2p       replayed          63         81    1.28  2.2e-16  
        34  2.17             flat     replayed         119        237    1.97  5.2e-16  
        35  5.7              2p       replayed          59         77    1.29  3.9e-16  
        36  2^2.3^2          2p       replayed          52         47    0.89  2.4e-16  
        37  37               mono     replayed         159        411    2.52  3.6e-16  
        38  2.19             flat     replayed         137        275    1.91  3.5e-16  
        39  3.13             2p       replayed          78         92    1.18  5.3e-16  
        40  2^3.5            2p       replayed          48         48    1.00  1.4e-16  
        41  41               mono     replayed         194        484    2.48  4.1e-16  
        42  2.3.7            2p       replayed          62         64    0.98  4.3e-16  
        43  43               prime    replayed         227        531    2.33  5.3e-16  rader
        44  2^2.11           2p       replayed          63         81    1.28  4.0e-16  
        45  3^2.5            2p       replayed          75         97    1.29  3.7e-16  
        46  2.23             flat     replayed         143        409    2.86  4.9e-16  
        47  47               mono     replayed         239        573    2.39  4.0e-16  
        48  2^4.3            2p       replayed          58         54    0.94  3.0e-16  
        49  7^2              2p       replayed          75        137    1.82  3.9e-16  
        50  2.5^2            2p       replayed          70         75    1.07  3.6e-16  
        51  3.17             2p       replayed         116        372    3.18  3.5e-16  
        52  2^2.13           2p       replayed          80         91    1.13  5.3e-16  
        53  53               prime    replayed         290        734    2.48  5.8e-16  rader
        54  2.3^3            2p       replayed          78         81    1.03  3.8e-16  
        55  5.11             2p       replayed          94        120    1.27  4.2e-16  
        56  2^3.7            2p       replayed          71         69    0.96  1.8e-16  
        57  3.19             2p       replayed         137        418    3.05  5.4e-16  
        58  2.29             flat     replayed         195        557    2.85  4.0e-16  
        59  59               prime    replayed         374        700    1.86  3.7e-16  bluestein
        60  2^2.3.5          2p       replayed          73         69    0.95  3.4e-16  
        61  61               prime    replayed         313        594    1.73  5.1e-16  rader
        62  2.31             flat     replayed         213        641    2.94  3.5e-16  
        63  3^2.7            2p       replayed          97        133    1.36  3.5e-16  
        64  2^6              2p       replayed          62         66    1.07  1.5e-16  
        65  5.13             2p       replayed         116        142    1.22  4.7e-16  
        66  2.3.11           2p       replayed          96        117    1.22  3.0e-16  
        67  67               prime    replayed         361       1084    2.98  4.6e-16  rader
        68  2^2.17           2p       replayed         115        461    4.00  3.2e-16  
        69  3.23             2p       replayed         202        604    2.99  4.1e-16  
        70  2.5.7            2p       replayed          90        106    1.18  2.3e-16  
        71  71               prime    replayed         358        808    2.25  4.5e-16  rader
        72  2^3.3^2          2p       replayed          92         81    0.88  3.2e-16  
        73  73               prime    replayed         352        700    1.99  8.3e-16  rader
        74  2.37             flat     replayed         457        868    1.84  4.3e-16  
        75  3.5^2            2p       replayed         111        147    1.30  2.5e-16  
        76  2^2.19           2p       replayed         146        613    4.01  3.6e-16  
        77  7.11             2p       replayed         123        158    1.28  2.5e-16  
        78  2.3.13           2p       replayed         120        133    0.91  5.4e-16  
        79  79               prime    replayed         443       1017    1.65  1.1e-15  flips differ 1.39x rader
        80  2^4.5            2p       replayed          85         85    0.99  4.1e-16  
        81  3^4              2p       replayed         123        214    1.73  4.3e-16  
        82  2.41             prime    replayed         689       1066    1.50  3.7e-16  
        83  83               prime    replayed         686       1054    1.53  4.7e-16  bluestein
        84  2^2.3.7          2p       replayed         108        121    0.87  3.5e-16  
        85  5.17             2p       replayed         174        654    3.59  4.4e-16  
        86  2.43             prime    replayed         687       1112    1.59  3.8e-16  
        87  3.29             flat     replayed         334        922    2.57  5.5e-16  
        88  2^3.11           2p       replayed         114        204    1.69  2.8e-16  
        89  89               prime    replayed         460       1075    2.33  4.4e-16  rader
        90  2.3^2.5          2p       replayed         116        123    1.06  4.0e-16  
        91  7.13             2p       replayed         159        178    1.12  5.0e-16  
        92  2^2.23           2p       replayed         175        838    4.80  4.1e-16  
        93  3.31             flat     replayed         370       1001    2.69  4.4e-16  
        94  2.47             prime    replayed         711       1387    1.95  4.9e-16  
        95  5.19             2p       replayed         213        715    3.26  3.0e-16  
        96  2^5.3            2p       replayed          98         97    0.98  3.0e-16  
        97  97               prime    replayed         446        912    2.03  6.5e-16  rader
        98  2.7^2            flat     replayed         223        155    0.69  3.8e-16  
        99  3^2.11           2p       replayed         167        237    1.19  5.3e-16  
       100  2^2.5^2          2p       replayed         114        110    0.96  2.5e-16  
       101  101              prime    replayed         489        958    1.96  6.3e-16  rader
       102  2.3.17           2p       replayed         175        693    3.76  4.6e-16  
       103  103              prime    replayed         621       1394    2.17  1.2e-15  rader
       104  2^3.13           2p       replayed         146        169    1.16  4.8e-16  
       105  3.5.7            2p       replayed         150        190    1.20  4.0e-16  
       106  2.53             prime    replayed         790       1692    2.10  3.9e-16  bluestein
       107  107              prime    replayed         796       1478    1.76  6.2e-16  bluestein
       108  2^2.3^3          2p       replayed         141        158    1.02  4.4e-16  
       109  109              prime    replayed         557       1157    2.00  6.9e-16  rader
       110  2.5.11           2p       replayed         156        190    1.22  4.1e-16  
       111  3.37             prime    replayed         710       1342    1.84  4.4e-16  
       112  2^4.7            2p       replayed         139        124    0.89  2.9e-16  
       113  113              prime    replayed         555       1636    2.93  6.3e-16  rader
       114  2.3.19           2p       replayed         209        824    3.88  3.9e-16  
       115  5.23             2p       replayed         300        988    3.29  4.1e-16  
       116  2^2.29           flat     replayed         342       1114    1.85  5.5e-16  flips differ 1.76x
       117  3^2.13           2p       replayed         197        228    1.15  5.7e-16  
       118  2.59             prime    replayed         801       1644    1.98  3.9e-16  bluestein
       119  7.17             2p       replayed         228        818    3.52  3.7e-16  
       120  2^3.3.5          2p       replayed         133        131    0.89  4.7e-16  
       121  11^2             2p       replayed         209        366    1.75  3.3e-16  
       122  2.61             prime    replayed         802       1273    1.55  4.2e-16  bluestein
       123  3.41             prime    replayed         731       1539    2.04  3.4e-16  
       124  2^2.31           flat     replayed         372       1250    3.00  5.9e-16  
       125  5^3              2p       replayed         209        273    1.30  4.5e-16  
       126  2.3^2.7          2p       replayed         170        197    1.08  2.8e-16  
       127  127              prime    replayed         831       1371    1.64  9.2e-16  rader
       128  2^7              2p       replayed         124        125    1.00  2.1e-16  
       129  3.43             flat     replayed         504       1793    3.50  4.4e-16  
       130  2.5.13           2p       replayed         191        217    1.14  4.7e-16  
       131  131              prime    replayed         726       1780    2.36  6.7e-16  rader
       132  2^2.3.11         2p       replayed         180        277    1.54  4.6e-16  
       133  7.19             2p       replayed         271       1050    3.60  3.5e-16  
       134  2.67             prime    replayed        1555       2476    1.49  4.1e-16  bluestein
       135  3^3.5            2p       replayed         194        246    1.26  3.0e-16  
       136  2^3.17           2p       replayed         215        921    4.28  3.8e-16  
       137  137              prime    replayed         805       1769    2.11  8.3e-16  rader
       138  2.3.23           2p       replayed         264       1148    4.32  4.8e-16  
       139  139              prime    replayed         963       1676    1.73  7.4e-16  rader
       140  2^2.5.7          chain3   replayed         261        179    0.64  2.7e-16  
       141  3.47             flat     replayed         577       2035    3.53  3.9e-16  
       142  2.71             prime    replayed        1532       1783    1.16  5.9e-16  bluestein
       143  11.13            2p       replayed         261        437    1.61  4.0e-16  
       144  2^4.3^2          chain3   replayed         231        156    0.68  3.6e-16  
       145  5.29             2p       replayed         384       1414    3.67  4.0e-16  
       146  2.73             prime    replayed        1540       1485    0.96  6.2e-16  bluestein
       147  3.7^2            chain3   replayed         304        413    1.33  3.1e-16  
       148  2^2.37           2p       replayed         353       1744    4.81  2.8e-16  
       149  149              prime    replayed        1321       1718    1.24  5.8e-16  rader
       150  2.3.5^2          2p       replayed         186        215    1.12  3.5e-16  
       151  151              prime    replayed         904       1521    1.67  6.8e-16  rader
       152  2^3.19           2p       replayed         258       1097    4.10  4.1e-16  
       153  3^2.17           2p       replayed         294       1063    3.51  4.2e-16  
       154  2.7.11           flat     replayed         362        396    1.08  4.6e-16  
       155  5.31             2p       replayed         603       1634    2.56  3.6e-16  
       156  2^2.3.13         2p       replayed         228        306    1.34  5.9e-16  
       157  157              prime    replayed         860       1635    1.90  1.2e-15  rader
       158  2.79             prime    replayed        1474       2098    1.25  4.9e-16  bluestein
       159  3.53             prime    replayed        1469       2468    1.67  4.5e-16  bluestein
       160  2^5.5            2p       replayed         161        164    1.01  3.0e-16  
       161  7.23             2p       replayed         351       1484    3.93  4.5e-16  
       162  2.3^4            2p       replayed         214        316    1.39  4.2e-16  
       163  163              prime    replayed        1047       1810    1.73  6.8e-16  rader
       164  2^2.41           2p       replayed         622       2095    2.74  4.5e-16  
       165  3.5.11           2p       replayed         253        367    1.45  3.8e-16  
       166  2.83             prime    replayed        1524       2861    1.68  6.5e-16  bluestein
       167  167              prime    replayed        1534       2290    1.45  4.9e-16  bluestein
       168  2^3.3.7          chain3   replayed         284        198    0.70  2.6e-16  
       169  13^2             2p       replayed         318        647    1.84  6.0e-16  
       170  2.5.17           2p       replayed         278       1162    4.17  7.6e-16  
       171  3^2.19           2p       replayed         350       1264    3.59  6.3e-16  
       172  2^2.43           2p       replayed         464       2254    4.84  4.3e-16  
       173  173              prime    replayed        1473       2218    1.31  4.6e-16  bluestein
       174  2.3.29           2p       replayed         380       1808    4.75  5.5e-16  
       175  5^2.7            chain3   replayed         362        383    0.93  3.7e-16  
       176  2^4.11           2p       replayed         238        351    1.00  2.4e-16  flips differ 1.47x
       177  3.59             prime    replayed        1562       2416    1.53  5.5e-16  bluestein
       178  2.89             prime    replayed        1478       2748    1.86  5.5e-16  bluestein
       179  179              prime    replayed        1468       2220    1.49  6.3e-16  bluestein
       180  2^2.3^2.5        2p       replayed         220        242    1.09  4.0e-16  
       181  181              prime    replayed        1032       1802    1.73  6.9e-16  rader
       182  2.7.13           flat     replayed         442        448    1.00  5.3e-16  
       183  3.61             prime    replayed        1539       2002    1.30  4.9e-16  bluestein
       184  2^3.23           2p       replayed         439       1642    3.72  4.2e-16  
       185  5.37             2p       replayed         816       2270    2.76  4.3e-16  
       186  2.3.31           2p       replayed         417       1886    4.52  6.3e-16  
       187  11.17            flat     replayed         463       1495    3.23  6.5e-16  
       188  2^2.47           2p       replayed         957       2613    2.61  4.5e-16  
       189  3^3.7            chain3   replayed         357        513    1.43  4.7e-16  
       190  2.5.19           2p       replayed         336       1386    4.06  3.5e-16  
       191  191              prime    replayed        1201       2026    1.61  8.4e-16  rader
       192  2^6.3            2p       replayed         199        199    0.98  2.7e-16  
       193  193              prime    replayed         942       1788    1.88  5.4e-16  rader
       194  2.97             prime    replayed        1482       1906    1.11  4.6e-16  bluestein
       195  3.5.13           chain3   replayed         382        430    1.12  7.1e-16  
       196  2^2.7^2          chain3   replayed         408        370    0.90  2.8e-16  
       197  197              prime    replayed        1363       2222    1.57  5.3e-16  rader
       198  2.3^2.11         chain3   replayed         363        455    1.13  4.9e-16  
       199  199              prime    replayed        1358       2218    1.63  6.8e-16  rader
       200  2^3.5^2          chain3   replayed         301        219    0.72  3.5e-16  
       201  3.67             prime    replayed        1584       3449    2.15  8.0e-16  bluestein
       202  2.101            prime    replayed        1578       2132    1.33  4.0e-16  bluestein
       203  7.29             2p       replayed         506       1978    3.86  4.5e-16  
       204  2^2.3.17         chain3   replayed         399       1490    3.70  5.3e-16  
       205  5.41             2p       replayed         665       2203    3.31  4.3e-16  
       206  2.103            prime    replayed        1653       4137    2.48  7.7e-16  bluestein
       207  3^2.23           2p       replayed         553       1761    3.15  4.4e-16  
       208  2^4.13           2p       replayed         277        404    1.45  5.0e-16  
       209  11.19            2p       replayed         453       1764    3.88  3.6e-16  
       210  2.3.5.7          chain3   replayed         340        304    0.88  3.5e-16  
       211  211              prime    replayed        1380       2835    1.82  7.1e-16  rader
       212  2^2.53           prime    replayed        1584       3536    2.21  4.5e-16  bluestein
       213  3.71             prime    replayed        1597       2706    1.61  6.3e-16  bluestein
       214  2.107            prime    replayed        1582       3103    1.87  6.1e-16  bluestein
       215  5.43             flat     replayed         763       2889    3.70  4.5e-16  
       216  2^3.3^3          2p       replayed         298        364    1.21  3.9e-16  
       217  7.31             2p       replayed         739       2217    2.51  4.2e-16  
       218  2.109            prime    replayed        1507       2360    1.47  4.8e-16  bluestein
       219  3.73             prime    replayed        1599       2428    1.47  5.5e-16  bluestein
       220  2^2.5.11         chain3   replayed         388        445    0.99  2.8e-16  
       221  13.17            2p       replayed         456       1766    3.80  6.2e-16  
       222  2.3.37           2p       replayed         537       2550    4.75  3.9e-16  
       223  223              prime    replayed        1586       2915    1.83  6.1e-16  bluestein
       224  2^5.7            2p       replayed         264        249    0.94  3.3e-16  
       225  3^2.5^2          2p       replayed         346        503    1.33  4.8e-16  
       226  2.113            prime    replayed        1521       3515    2.17  4.8e-16  bluestein
       227  227              prime    replayed        1577       2672    1.68  4.3e-16  bluestein
       228  2^2.3.19         flat     replayed         656       1891    2.63  1.1e-15  
       229  229              prime    replayed        1427       2699    1.85  6.3e-16  rader
       230  2.5.23           2p       replayed         437       1910    4.36  4.7e-16  
       231  3.7.11           chain3   replayed         440        643    1.43  3.5e-16  
       232  2^3.29           flat     replayed         665       2264    3.35  5.3e-16  
       233  233              prime    replayed        1600       2676    1.64  5.0e-16  bluestein
       234  2.3^2.13         chain3   replayed         495        502    0.93  5.1e-16  
       235  5.47             2p       replayed        1265       3438    2.56  3.9e-16  
       236  2^2.59           prime    replayed        1598       3205    1.96  4.8e-16  bluestein
       237  3.79             prime    replayed        1603       3191    1.96  6.1e-16  bluestein
       238  2.7.17           flat     replayed         677       1858    2.54  1.2e-15  
       239  239              prime    replayed        1609       2685    1.66  4.9e-16  bluestein
       240  2^4.3.5          2p       replayed         302        257    0.84  3.0e-16  
       241  241              prime    replayed        1401       2286    1.55  4.0e-16  rader
       242  2.11^2           flat     replayed         604       1013    1.36  7.6e-16  
       243  3^5              2p       replayed         380        643    1.68  4.9e-16  
       244  2^2.61           prime    replayed        1636       2545    1.53  4.4e-16  bluestein
       245  5.7^2            chain3   replayed         502        721    1.43  2.8e-16  
       246  2.3.41           2p       replayed         661       3175    4.79  3.0e-16  
       247  13.19            2p       replayed         557       2081    3.73  4.7e-16  
       248  2^3.31           flat     replayed         748       2803    3.73  4.5e-16  
       249  3.83             prime    replayed        1631       3346    1.96  6.4e-16  bluestein
       250  2.5^3            chain3   replayed         419        431    1.00  3.7e-16  
       251  251              prime    replayed        1534       2688    1.74  6.6e-16  rader
       252  2^2.3^2.7        chain3   replayed         458        451    0.97  4.5e-16  
       253  11.23            2p       replayed         571       2391    4.17  5.1e-16  
       254  2.127            prime    replayed        1526       2993    1.95  3.8e-16  bluestein
       255  3.5.17           2p       replayed         466       1884    3.92  5.5e-16  
       256  2^8              2p       replayed         236        256    1.05  2.5e-16  
       257  257              prime    replayed        1277       2522    1.87  4.4e-16  rader
       258  2.3.43           2p       replayed        1037       3364    2.80  4.3e-16  
       259  7.37             2p       replayed         720       2999    4.14  4.4e-16  
       260  2^2.5.13         chain3   replayed         485        524    1.06  4.4e-16  
       261  3^2.29           2p       replayed         653       2817    4.29  4.5e-16  
       262  2.131            prime    replayed        3271       3681    1.09  6.1e-16  bluestein
       263  263              prime    replayed        3256       3469    1.02  5.4e-16  bluestein
       264  2^3.3.11         chain3   replayed         456        560    1.22  4.8e-16  
       265  5.53             prime    replayed        3272       4125    1.23  5.7e-16  bluestein
       266  2.7.19           flat     replayed         712       2081    2.85  4.4e-16  
       267  3.89             prime    replayed        3375       3401    1.00  6.6e-16  bluestein
       268  2^2.67           prime    replayed        3351       4544    1.31  6.9e-16  bluestein
       269  269              prime    replayed        3269       3541    1.06  5.7e-16  bluestein
       270  2.3^3.5          chain3   replayed         446        486    1.09  4.9e-16  
       271  271              prime    replayed        1640       2927    1.75  7.2e-16  rader
       272  2^4.17           2p       replayed         412       2075    4.58  5.8e-16  
       273  3.7.13           2p       replayed         471        707    1.50  6.8e-16  
       274  2.137            prime    replayed        3354       4299    1.23  5.8e-16  bluestein
       275  5^2.11           chain3   replayed         538        816    1.51  3.4e-16  
       276  2^2.3.23         2p       replayed         521       2934    5.46  4.5e-16  
       277  277              prime    replayed        1922       3669    1.83  7.8e-16  rader
       278  2.139            prime    replayed        3505       3721    1.05  6.1e-16  bluestein
       279  3^2.31           2p       replayed         926       2922    2.66  4.9e-16  
       280  2^3.5.7          chain3   replayed         417        328    0.78  3.6e-16  
       281  281              prime    replayed        1728       3441    1.96  5.4e-16  rader
       282  2.3.47           2p       replayed        1419       3933    2.68  4.3e-16  
       283  283              prime    replayed        3487       3519    0.95  4.8e-16  bluestein
       284  2^2.71           prime    replayed        3403       3421    1.00  4.4e-16  bluestein
       285  3.5.19           2p       replayed         578       2275    3.79  4.3e-16  
       286  2.11.13          flat     replayed         741        986    0.98  4.9e-16  flips differ 1.36x
       287  7.41             2p       replayed         887       3604    3.83  3.0e-16  
       288  2^5.3^2          2p       replayed         355        424    1.19  3.5e-16  
       289  17^2             2p       replayed         660       3892    4.84  7.9e-16  flips differ 1.25x
       290  2.5.29           2p       replayed         675       2831    3.86  5.0e-16  
       291  3.97             prime    replayed        3319       3049    0.90  6.8e-16  bluestein
       292  2^2.73           prime    replayed        3658       2981    0.81  6.2e-16  bluestein
       293  293              prime    replayed        3309       3939    1.11  5.8e-16  bluestein
       294  2.3.7^2          chain3   replayed         546        600    1.02  4.3e-16  
       295  5.59             prime    replayed        3342       4411    1.27  5.7e-16  bluestein
       296  2^3.37           2p       replayed         695       3499    4.88  4.0e-16  
       297  3^3.11           chain3   replayed         582        785    1.35  5.2e-16  
       298  2.149            prime    replayed        3317       4504    1.30  4.3e-16  bluestein
       299  13.23            2p       replayed         705       3055    4.18  4.8e-16  
       300  2^2.3.5^2        chain3   replayed         486        419    0.86  3.8e-16  
       301  7.43             flat     replayed        1068       4495    4.19  4.1e-16  
       302  2.151            prime    replayed        3288       3334    0.97  5.1e-16  bluestein
       303  3.101            prime    replayed        3356       3072    0.91  4.6e-16  bluestein
       304  2^4.19           2p       replayed         524       2288    4.31  2.7e-16  
       305  5.61             prime    replayed        3291       3621    1.07  5.5e-16  bluestein
       306  2.3^2.17         chain3   replayed         628       2282    3.03  6.0e-16  
       307  307              prime    replayed        2406       5638    2.23  1.0e-15  rader
       308  2^2.7.11         chain3   replayed         549        759    1.24  3.4e-16  
       309  3.103            prime    replayed        3292       4703    1.42  7.1e-16  bluestein
       310  2.5.31           2p       replayed         866       3156    3.64  4.0e-16  
       311  311              prime    replayed        2514       5517    2.02  7.1e-16  rader
       312  2^3.3.13         chain3   replayed         569        672    1.15  5.6e-16  
       313  313              prime    replayed        2101       5564    2.61  8.4e-16  rader
       314  2.157            prime    replayed        3320       4120    1.24  5.1e-16  bluestein
       315  3^2.5.7          chain3   replayed         537        767    1.43  5.5e-16  
       316  2^2.79           prime    replayed        3318       4163    1.25  4.4e-16  bluestein
       317  317              prime    replayed        3288       3425    1.02  5.2e-16  bluestein
       318  2.3.53           prime    replayed        3406       5421    1.54  4.4e-16  bluestein
       319  11.29            2p       replayed         819       3512    4.09  4.5e-16  
       320  2^6.5            2p       replayed         346        347    1.00  3.0e-16  
       321  3.107            prime    replayed        3391       4719    1.30  6.1e-16  bluestein
       322  2.7.23           flat     replayed         966       2907    2.93  5.9e-16  
       323  17.19            2p       replayed         882       4582    5.04  5.8e-16  
       324  2^2.3^4          chain3   replayed         518        622    1.17  4.3e-16  
       325  5^2.13           2p       replayed         562        924    1.62  6.4e-16  
       326  2.163            prime    replayed        3606       4015    1.03  6.3e-16  bluestein
       327  3.109            prime    replayed        3400       3701    1.06  6.6e-16  bluestein
       328  2^3.41           2p       replayed        1226       4085    3.30  4.9e-16  
       329  7.47             flat     replayed        1236       5330    4.09  4.3e-16  
       330  2.3.5.11         chain3   replayed         636        739    1.12  5.3e-16  
       331  331              prime    replayed        2128       5501    2.57  6.7e-16  rader
       332  2^2.83           prime    replayed        3303       4608    1.35  5.4e-16  bluestein
       333  3^2.37           2p       replayed        1321       3925    2.93  3.8e-16  
       334  2.167            prime    replayed        3308       6601    1.92  5.3e-16  bluestein
       335  5.67             prime    replayed        3429       6176    1.68  5.9e-16  bluestein
       336  2^4.3.7          2p       replayed         435        576    1.23  3.6e-16  
       337  337              prime    replayed        1905       5555    2.84  8.1e-16  rader
       338  2.13^2           flat     replayed         915       1370    1.29  1.2e-15  
       339  3.113            prime    replayed        3394       5276    1.52  5.5e-16  bluestein
       340  2^2.5.17         chain3   replayed         682       2547    3.69  6.0e-16  
       341  11.31            flat     replayed        1156       3862    3.31  9.0e-16  
       342  2.3^2.19         chain3   replayed         751       3182    3.50  3.7e-16  flips differ 1.40x
       343  7^3              chain3   replayed        1016       1887    1.82  3.7e-16  
       344  2^3.43           2p       replayed         921       6771    5.14  4.4e-16  flips differ 1.43x
       345  3.5.23           2p       replayed         845       3267    3.64  4.9e-16  
       346  2.173            prime    replayed        3344       4607    1.37  5.1e-16  bluestein
       347  347              prime    replayed        3322       4415    1.25  6.5e-16  bluestein
       348  2^2.3.29         2p       replayed         782       3471    4.30  5.5e-16  
       349  349              prime    replayed        2889       4690    1.56  7.8e-16  rader
       350  2.5^2.7          chain3   replayed         587        674    1.06  2.6e-16  
       351  3^3.13           chain3   replayed         715        900    1.25  5.3e-16  
       352  2^5.11           2p       replayed         494        701    1.39  2.6e-16  
       353  353              prime    replayed        1943       4516    2.24  6.5e-16  rader
       354  2.3.59           prime    replayed        3415       4416    1.21  5.4e-16  bluestein
       355  5.71             prime    replayed        3370       4514    1.31  5.7e-16  bluestein
       356  2^2.89           prime    replayed        3349       4663    1.36  7.4e-16  bluestein
       357  3.7.17           chain3   replayed         784       2692    3.15  4.8e-16  
       358  2.179            prime    replayed        3325       4656    1.39  6.2e-16  bluestein
       359  359              prime    replayed        3339       4783    1.25  5.3e-16  bluestein
       360  2^3.3^2.5        chain3   replayed         616        598    0.93  3.8e-16  
       361  19^2             2p       replayed         911       5058    5.54  4.5e-16  
       362  2.181            prime    replayed        3315       3773    1.13  6.8e-16  bluestein
       363  3.11^2           chain3   replayed         785       1661    1.96  4.6e-16  
       364  2^2.7.13         chain3   replayed         685        849    1.18  4.9e-16  
       365  5.73             prime    replayed        3328       3808    1.13  5.3e-16  bluestein
       366  2.3.61           prime    replayed        3410       3807    1.11  5.5e-16  bluestein
       367  367              prime    replayed        3350       5168    1.50  8.6e-16  bluestein
       368  2^4.23           2p       replayed         787       3498    4.12  3.5e-16  
       369  3^2.41           2p       replayed        1598       4666    2.90  3.8e-16  
       370  2.5.37           2p       replayed         912       4409    4.83  4.0e-16  
       371  7.53             prime    replayed        3483       7112    1.70  3.9e-16  flips differ 1.38x bluestein
       372  2^2.3.31         flat     replayed        1310       5590    2.75  4.4e-16  flips differ 1.55x
       373  373              prime    replayed        3327       4952    1.45  5.6e-16  bluestein
       374  2.11.17          flat     replayed        1033       3730    3.57  1.1e-15  
       375  3.5^3            chain3   replayed         654        924    1.12  5.1e-16  flips differ 1.27x
       376  2^3.47           2p       replayed        1597       5687    2.94  3.7e-16  
       377  13.29            2p       replayed        1226       4148    2.89  5.3e-16  
       378  2.3^3.7          chain3   replayed         710        776    1.08  4.8e-16  
       379  379              prime    replayed        3030       4049    1.30  9.3e-16  rader
       380  2^2.5.19         chain3   replayed         795       2906    3.52  4.1e-16  
       381  3.127            prime    replayed        3324       4318    1.28  4.7e-16  bluestein
       382  2.191            prime    replayed        3320       4073    1.21  5.5e-16  bluestein
       383  383              prime    replayed        3486       4003    1.11  5.3e-16  bluestein
       384  2^7.3            2p       replayed         411        410    0.90  3.5e-16  
       385  5.7.11           chain3   replayed         777       1365    1.38  4.3e-16  flips differ 1.50x
       386  2.193            prime    replayed        4854       5635    1.13  5.4e-16  bluestein
       387  3^2.43           2p       replayed        1222       5106    2.93  4.5e-16  flips differ 1.43x
       388  2^2.97           prime    replayed        3342       3965    1.17  4.7e-16  bluestein
       389  389              prime    replayed        3322       4791    1.40  4.7e-16  bluestein
       390  2.3.5.13         chain3   replayed         769        830    1.07  5.8e-16  
       391  17.23            2p       replayed        1025       5771    5.53  6.2e-16  
       392  2^3.7^2          chain3   replayed         660        762    1.13  2.7e-16  
       393  3.131            prime    replayed        3376       5609    1.66  5.3e-16  bluestein
       394  2.197            prime    replayed        3355       4749    1.36  5.5e-16  bluestein
       395  5.79             prime    replayed        3400       5238    1.53  5.1e-16  bluestein
       396  2^2.3^2.11       chain3   replayed         747        891    1.16  4.5e-16  
       397  397              prime    replayed        2863       4721    1.64  8.9e-16  rader
       398  2.199            prime    replayed        3395       4655    1.35  4.5e-16  bluestein
       399  3.7.19           chain3   replayed         915       3193    3.46  5.7e-16  
       400  2^4.5^2          2p       replayed         527        550    0.95  3.1e-16  
       401  401              prime    replayed        2317       3897    1.66  5.1e-16  rader
       402  2.3.67           prime    replayed        3537       6762    1.80  7.6e-16  bluestein
       403  13.31            2p       replayed        1375       4564    2.89  6.4e-16  
       404  2^2.101          prime    replayed        3426       4097    1.15  5.9e-16  bluestein
       405  3^4.5            chain3   replayed         714        997    1.29  6.4e-16  
       406  2.7.29           flat     replayed        1382       4221    3.00  5.6e-16  
       407  11.37            2p       replayed        1165       5178    4.43  4.6e-16  
       408  2^3.3.17         chain3   replayed         811       3111    3.74  5.1e-16  
       409  409              prime    replayed        3029       6549    1.78  8.5e-16  flips differ 1.40x rader
       410  2.5.41           2p       replayed        2208       7683    3.44  3.5e-16  
       411  3.137            prime    replayed        3434       5468    1.57  8.2e-16  bluestein
       412  2^2.103          prime    replayed        3388       5813    1.69  6.1e-16  bluestein
       413  7.59             prime    replayed        3513       5989    1.70  5.8e-16  bluestein
       414  2.3^2.23         chain3   replayed        1057       3905    3.38  5.1e-16  
       415  5.83             prime    replayed        3488       5620    1.58  5.7e-16  bluestein
       416  2^5.13           2p       replayed         590        842    1.37  4.9e-16  
       417  3.139            prime    replayed        3428       6818    1.92  4.8e-16  bluestein
       418  2.11.19          flat     replayed        1212       3838    3.11  1.0e-15  
       419  419              prime    replayed        3365       5296    1.51  6.9e-16  bluestein
       420  2^2.3.5.7        chain3   replayed         683        725    1.04  4.3e-16  
       421  421              prime    replayed        5263       7503    1.39  6.8e-16  rader
       422  2.211            prime    replayed        4548       7862    1.65  5.9e-16  bluestein
       423  3^2.47           flat     replayed        1561       5967    3.81  6.3e-16  
       424  2^3.53           prime    replayed        3404       6565    1.93  4.9e-16  bluestein
       425  5^2.17           2p       replayed         825       3240    3.89  6.4e-16  
       426  2.3.71           prime    replayed        3386       5425    1.52  4.8e-16  bluestein
       427  7.61             prime    replayed        3439       4543    1.29  5.1e-16  bluestein
       428  2^2.107          prime    replayed        3473       5925    1.62  7.2e-16  bluestein
       429  3.11.13          chain3   replayed         945       1543    1.62  5.6e-16  
       430  2.5.43           2p       replayed        1195       5821    4.87  3.4e-16  
       431  431              prime    replayed        3409       5304    1.53  6.9e-16  bluestein
       432  2^4.3^3          2p       replayed         569        725    1.15  4.2e-16  
       433  433              prime    replayed        2499       4665    1.85  7.6e-16  rader
       434  2.7.31           flat     replayed        1502       4633    3.05  4.6e-16  
       435  3.5.29           2p       replayed        1071       4518    4.20  4.8e-16  
       436  2^2.109          prime    replayed        3472       4710    1.33  6.3e-16  bluestein
       437  19.23            2p       replayed        1213       6622    5.42  4.7e-16  
       438  2.3.73           prime    replayed        3413       4552    1.31  6.2e-16  bluestein
       439  439              prime    replayed        3412       6244    1.70  7.2e-16  bluestein
       440  2^3.5.11         chain3   replayed         754        925    1.23  4.0e-16  
       441  3^2.7^2          chain3   replayed         892       1230    1.22  4.5e-16  
       442  2.13.17          flat     replayed        1290       4664    2.53  1.1e-15  flips differ 1.37x
       443  443              prime    replayed        3479       6124    1.57  6.1e-16  bluestein
       444  2^2.3.37         2p       replayed        1101       5377    4.84  5.2e-16  
       445  5.89             prime    replayed        3900       5835    1.43  4.5e-16  bluestein
       446  2.223            prime    replayed        3423       6209    1.78  5.4e-16  bluestein
       447  3.149            prime    replayed        3443       6861    1.82  6.4e-16  bluestein
       448  2^6.7            chain3   replayed         623        550    0.83  2.8e-16  
       449  449              prime    replayed        2535       5801    2.27  5.8e-16  rader
       450  2.3^2.5^2        chain3   replayed         774        829    1.07  5.1e-16  
       451  11.41            2p       replayed        1977       6178    2.58  4.6e-16  
       452  2^2.113          prime    replayed        3455       7125    1.94  9.3e-16  bluestein
       453  3.151            prime    replayed        3380       5216    1.51  7.8e-16  bluestein
       454  2.227            prime    replayed        3381       5726    1.58  5.9e-16  bluestein
       455  5.7.13           chain3   replayed         947       1206    1.26  4.5e-16  
       456  2^3.3.19         chain3   replayed        1019       3777    3.56  3.2e-16  
       457  457              prime    replayed        3418       5611    1.61  1.0e-15  rader
       458  2.229            prime    replayed        3403       5598    1.64  5.9e-16  bluestein
       459  3^3.17           chain3   replayed        1029       3495    3.23  4.5e-16  
       460  2^2.5.23         flat     replayed        1278       4135    3.19  5.3e-16  
       461  461              prime    replayed        3460       5621    1.48  4.8e-16  bluestein
       462  2.3.7.11         chain3   replayed         957       1098    1.13  4.0e-16  
       463  463              prime    replayed        3234       5833    1.67  7.2e-16  rader
       464  2^4.29           2p       replayed        1231       4740    3.31  3.5e-16  
       465  3.5.31           2p       replayed        1505       5055    3.28  5.6e-16  
       466  2.233            prime    replayed        3447       7203    2.04  5.6e-16  bluestein
       467  467              prime    replayed        3402       5594    1.60  4.7e-16  bluestein
       468  2^2.3^2.13       chain3   replayed         928       1021    1.09  5.1e-16  
       469  7.67             prime    replayed        3434       8102    2.30  4.7e-16  bluestein
       470  2.5.47           2p       replayed        2012       6975    2.80  5.1e-16  
       471  3.157            prime    replayed        3496       5712    1.49  5.5e-16  bluestein
       472  2^3.59           prime    replayed        3396       6652    1.86  5.5e-16  bluestein
       473  11.43            2p       replayed        1518       6980    4.44  5.6e-16  
       474  2.3.79           prime    replayed        3413       6426    1.84  4.7e-16  bluestein
       475  5^2.19           2p       replayed         971       4047    4.01  5.0e-16  
       476  2^2.7.17         chain3   replayed        1225       3836    3.09  4.7e-16  
       477  3^2.53           prime    replayed        3431       8100    2.32  5.1e-16  bluestein
       478  2.239            prime    replayed        3417       5659    1.65  5.7e-16  bluestein
       479  479              prime    replayed        3405       5615    1.65  5.3e-16  bluestein
       480  2^5.3.5          2p       replayed         647        692    0.99  3.6e-16  
       481  13.37            2p       replayed        1416       6388    4.18  5.5e-16  
       482  2.241            prime    replayed        3420       4781    1.34  4.4e-16  bluestein
       483  3.7.23           2p       replayed        1306       4418    3.33  5.1e-16  
       484  2^2.11^2         chain3   replayed        1181       1963    1.64  3.3e-16  
       485  5.97             prime    replayed        3416       5175    1.40  4.6e-16  bluestein
       486  2.3^5            chain3   replayed        1307       1473    0.77  4.6e-16  flips differ 1.47x
       487  487              prime    replayed        4884       7961    1.37  4.2e-16  bluestein
       488  2^3.61           prime    replayed        3590       5108    1.06  4.5e-16  flips differ 1.36x bluestein
       489  3.163            prime    replayed        3428       5840    1.69  6.9e-16  bluestein
       490  2.5.7^2          chain3   replayed         832       1006    1.19  4.8e-16  
       491  491              prime    replayed        3014       6782    2.13  7.1e-16  rader
       492  2^2.3.41         2p       replayed        1979       6644    2.97  4.0e-16  
       493  17.29            2p       replayed        1434       7960    5.36  5.6e-16  
       494  2.13.19          flat     replayed        1454       4861    3.10  1.2e-15  
       495  3^2.5.11         flat     replayed        1203       1449    1.18  5.0e-16  
       496  2^4.31           flat     replayed        1565       5613    3.57  6.6e-16  
       497  7.71             prime    replayed        3459       6673    1.83  5.5e-16  bluestein
       498  2.3.83           prime    replayed        3465       7351    1.98  8.9e-16  bluestein
       499  499              prime    replayed        3719       6507    1.64  7.2e-16  bluestein
       500  2^2.5^3          chain3   replayed         831        941    1.13  5.4e-16  
       501  3.167            prime    replayed        3452      10347    2.74  5.7e-16  bluestein
       502  2.251            prime    replayed        3535       6054    1.53  5.6e-16  bluestein
       503  503              prime    replayed        3626       5842    1.34  4.6e-16  bluestein
       504  2^3.3^2.7        chain3   replayed         932        889    0.93  4.3e-16  
       505  5.101            prime    replayed        3864       5471    1.33  5.5e-16  bluestein
       506  2.11.23          flat     replayed        1561       5086    3.17  1.5e-15  
       507  3.13^2           chain3   replayed        1150       1885    1.51  5.2e-16  
       508  2^2.127          prime    replayed        3497       6061    1.60  5.6e-16  bluestein
       509  509              prime    replayed        3541       5981    1.52  5.3e-16  bluestein
       510  2.3.5.17         chain3   replayed        1080       3988    3.63  5.9e-16  
       511  7.73             prime    replayed        3584       5787    1.35  7.1e-16  bluestein
       512  2^9              2p       replayed         520        591    1.08  3.1e-16  
       513  3^3.19           2p       replayed        1077       4218    3.90  6.2e-16  
       514  2.257            prime    replayed        6846       5208    0.74  5.9e-16  bluestein
       515  5.103            prime    replayed        7084       7477    0.92  5.3e-16  bluestein
       516  2^2.3.43         flat     replayed        1916       6987    3.63  4.3e-16  
       517  11.47            flat     replayed        1966       8184    4.13  5.5e-16  
       518  2.7.37           flat     replayed        1939       6396    3.23  5.8e-16  
       519  3.173            prime    replayed        7256       7104    0.79  3.5e-16  bluestein
       520  2^3.5.13         chain3   replayed         932       1128    1.16  6.2e-16  
       521  521              prime    replayed        3557       7469    2.00  9.9e-16  rader
       522  2.3^2.29         chain3   replayed        1468       5490    3.58  3.9e-16  
       523  523              prime    replayed        4924       7180    1.39  6.9e-16  rader
       524  2^2.131          prime    replayed        8024       8038    0.90  4.3e-16  bluestein
       525  3.5^2.7          2p       replayed         929       1442    1.54  4.3e-16  
       526  2.263            prime    replayed        8112       7377    0.84  5.5e-16  bluestein
       527  17.31            2p       replayed        1579       8701    5.47  7.5e-16  
       528  2^4.3.11         chain3   replayed         974       1169    1.17  5.5e-16  
       529  23^2             2p       replayed        1765       8626    4.84  4.5e-16  
       530  2.5.53           prime    replayed        8073       8741    1.05  4.0e-16  bluestein
       531  3^2.59           prime    replayed        8076       7930    0.95  5.3e-16  bluestein
       532  2^2.7.19         flat     replayed        1337       4592    3.05  5.0e-16  
       533  13.41            2p       replayed        2053       7364    2.87  5.6e-16  flips differ 1.26x
       534  2.3.89           prime    replayed        8206       7312    0.79  6.1e-16  bluestein
       535  5.107            prime    replayed        8131       7833    0.93  5.3e-16  bluestein
       536  2^3.67           prime    replayed        8145       9458    1.05  5.9e-16  bluestein
       537  3.179            prime    replayed        7464       7389    0.98  6.2e-16  bluestein
       538  2.269            prime    replayed        7980       7408    0.88  5.4e-16  bluestein
       539  7^2.11           flat     replayed        1308       1593    1.18  4.6e-16  
       540  2^2.3^3.5        chain3   replayed         923        987    1.05  4.8e-16  
       541  541              prime    replayed        5859       6041    1.03  6.8e-16  rader
       542  2.271            prime    replayed        8383       6565    0.68  5.8e-16  bluestein
       543  3.181            prime    replayed        8111       6157    0.72  4.4e-16  bluestein
       544  2^5.17           2p       replayed        1061       4028    3.10  4.4e-16  
       545  5.109            prime    replayed        8624       6114    0.71  5.5e-16  bluestein
       546  2.3.7.13         chain3   replayed        1196       1494    1.07  7.0e-16  flips differ 1.32x
       547  547              prime    replayed        3957      10675    2.47  9.4e-16  rader
       548  2^2.137          prime    replayed        8346       7614    0.86  4.5e-16  bluestein
       549  3^2.61           prime    replayed        8182       5924    0.64  6.2e-16  bluestein
       550  2.5^2.11         chain3   replayed        1053       1258    0.97  4.1e-16  
       551  19.29            2p       replayed        1717       9413    5.12  4.0e-16  
       552  2^3.3.23         flat     replayed        1492       4948    3.16  5.3e-16  
       553  7.79             prime    replayed        7927       8067    0.98  3.9e-16  bluestein
       554  2.277            prime    replayed        8252       7712    0.90  5.0e-16  bluestein
       555  3.5.37           2p       replayed        1521       6737    4.41  5.1e-16  
       556  2^2.139          prime    replayed        8060       7130    0.88  6.3e-16  bluestein
       557  557              prime    replayed        8092       9872    1.19  7.1e-16  bluestein
       558  2.3^2.31         chain3   replayed        1618       5992    3.59  5.9e-16  
       559  13.43            2p       replayed        1855       8004    4.24  6.4e-16  
       560  2^4.5.7          chain3   replayed         842        898    1.04  3.5e-16  
       561  3.11.17          flat     replayed        1574       5156    3.21  1.3e-15  
       562  2.281            prime    replayed        8155       7286    0.86  5.5e-16  bluestein
       563  563              prime    replayed        8145      10049    1.17  6.2e-16  bluestein
       564  2^2.3.47         flat     replayed        2280       8431    3.67  4.1e-16  
       565  5.113            prime    replayed        6880       9099    1.20  5.0e-16  bluestein
       566  2.283            prime    replayed        8334       7223    0.82  6.1e-16  bluestein
       567  3^4.7            2p       replayed         989       1585    1.56  6.0e-16  
       568  2^3.71           prime    replayed        6919       7120    0.88  3.7e-16  bluestein
       569  569              prime    replayed        7147       7201    0.99  5.0e-16  bluestein
       570  2.3.5.19         flat     replayed        1636       4713    2.12  5.4e-16  flips differ 1.36x
       571  571              prime    replayed        4327       7226    1.67  6.9e-16  rader
       572  2^2.11.13        chain3   replayed        1235       2110    1.38  4.5e-16  flips differ 1.25x
       573  3.191            prime    replayed        7790       6512    0.79  6.8e-16  bluestein
       574  2.7.41           flat     replayed        2396       7753    3.19  2.1e-15  
       575  5^2.23           2p       replayed        1584       5496    3.30  6.6e-16  
       576  2^6.3^2          2p       replayed         842        905    1.05  3.5e-16  
       577  577              prime    replayed        3462       7408    1.78  6.1e-16  flips differ 1.37x rader
       578  2.17^2           flat     replayed        2461      11701    4.68  8.0e-16  
       579  3.193            prime    replayed        8019       5953    0.51  4.9e-16  flips differ 1.43x bluestein
       580  2^2.5.29         chain3   replayed        1789       6134    3.42  4.1e-16  
       581  7.83             prime    replayed        8326       8740    1.03  5.4e-16  bluestein
       582  2.3.97           prime    replayed        8327       5962    0.67  4.8e-16  bluestein
       583  11.53            prime    replayed        8214      11993    1.30  4.5e-16  bluestein
       584  2^3.73           prime    replayed        9199       6572    0.67  4.4e-16  bluestein
       585  3^2.5.13         chain3   replayed        1272       1913    1.34  5.1e-16  
       586  2.293            prime    replayed        8621       8158    0.89  5.8e-16  bluestein
       587  587              prime    replayed        8280       9401    0.88  6.2e-16  bluestein
       588  2^2.3.7^2        flat     replayed        1320       1190    0.82  5.0e-16  
       589  19.31            2p       replayed        1835      10280    5.50  5.1e-16  
       590  2.5.59           prime    replayed        8633       7668    0.81  4.6e-16  bluestein
       591  3.197            prime    replayed        8838       8847    0.87  5.8e-16  flips differ 1.27x bluestein
       592  2^4.37           2p       replayed        1452       7396    4.27  4.3e-16  
       593  593              prime    replayed        5361       8474    1.32  6.9e-16  rader
       594  2.3^3.11         chain3   replayed        1290       1551    1.12  5.3e-16  
       595  5.7.17           chain3   replayed        1415       4924    2.78  5.1e-16  
       596  2^2.149          prime    replayed        8777       7778    0.71  4.7e-16  bluestein
       597  3.199            prime    replayed        8048       7081    0.81  4.6e-16  bluestein
       598  2.13.23          flat     replayed        1935       6715    3.33  8.4e-16  
       599  599              prime    replayed        8175       8210    0.96  7.2e-16  bluestein
       600  2^3.3.5^2        chain3   replayed        1102       1272    0.86  3.8e-16  flips differ 1.47x
       601  601              prime    replayed        5384       6555    1.18  7.6e-16  rader
       602  2.7.43           flat     replayed        2511       9036    3.37  6.5e-16  
       603  3^2.67           prime    replayed        8469      11781    1.32  4.5e-16  bluestein
       604  2^2.151          prime    replayed        8493       6893    0.80  5.6e-16  bluestein
       605  5.11^2           flat     replayed        1546       2311    1.46  4.2e-16  
       606  2.3.101          prime    replayed        8421       6768    0.75  4.1e-16  bluestein
       607  607              prime    replayed        8392      10481    1.21  7.6e-16  bluestein
       608  2^5.19           2p       replayed        1075       5093    4.64  3.8e-16  
       609  3.7.29           2p       replayed        2311       6577    2.37  4.5e-16  
       610  2.5.61           prime    replayed        8421       6714    0.77  6.9e-16  bluestein
       611  13.47            flat     replayed        2458       9587    3.74  6.5e-16  
       612  2^2.3^2.17       chain3   replayed        1339       4805    3.53  7.3e-16  
       613  613              prime    replayed        5029       9747    1.90  8.7e-16  rader
       614  2.307            prime    replayed        8257      12220    1.44  6.2e-16  bluestein
       615  3.5.41           2p       replayed        1929       8603    4.36  4.0e-16  
       616  2^3.7.11         chain3   replayed        1193       1444    1.21  3.6e-16  
       617  617              prime    replayed        4206      10032    2.29  7.6e-16  rader
       618  2.3.103          prime    replayed        7707      10305    1.24  5.5e-16  bluestein
       619  619              prime    replayed        7439      11425    1.36  5.6e-16  bluestein
       620  2^2.5.31         flat     replayed        2110       7353    3.35  5.6e-16  
       621  3^3.23           2p       replayed        1637       7322    3.14  5.2e-16  
       622  2.311            prime    replayed        8848      13090    1.29  6.4e-16  bluestein
       623  7.89             prime    replayed        8669       9893    1.09  4.4e-16  bluestein
       624  2^4.3.13         chain3   replayed        1112       1341    1.18  4.7e-16  
       625  5^4              2p       replayed        1122       1835    1.58  4.4e-16  
       626  2.313            prime    replayed        8351      12492    1.40  5.7e-16  bluestein
       627  3.11.19          flat     replayed        1810       5854    3.18  7.6e-16  
       628  2^2.157          prime    replayed        7843       7459    0.90  4.5e-16  bluestein
       629  17.37            2p       replayed        2682      12027    4.22  7.8e-16  
       630  2.3^2.5.7        chain3   replayed        1147       1395    1.08  3.6e-16  
       631  631              prime    replayed        5113       8463    1.31  7.5e-16  rader
       632  2^3.79           prime    replayed        7944       9259    1.10  5.2e-16  bluestein
       633  3.211            prime    replayed        8948       8837    0.96  4.5e-16  bluestein
       634  2.317            prime    replayed        8615       7320    0.82  4.4e-16  bluestein
       635  5.127            prime    replayed        8798       7994    0.86  4.6e-16  bluestein
       636  2^2.3.53         prime    replayed        8590      11108    1.22  3.9e-16  bluestein
       637  7^2.13           flat     replayed        1899       1921    0.83  8.4e-16  flips differ 1.28x
       638  2.11.29          flat     replayed        2435       8626    3.04  6.5e-16  
       639  3^2.71           prime    replayed        8519       8975    0.97  6.1e-16  bluestein
       640  2^7.5            2p       replayed         975        792    0.59  4.9e-16  flips differ 1.35x
       641  641              prime    replayed        3547       7531    1.81  6.2e-16  rader
       642  2.3.107          prime    replayed        8784      11376    0.87  5.9e-16  flips differ 1.49x bluestein
       643  643              prime    replayed        8659      11627    1.14  7.2e-16  bluestein
       644  2^2.7.23         flat     replayed        1894       6160    3.22  5.3e-16  
       645  3.5.43           2p       replayed        2119      13181    5.63  4.3e-16  
       646  2.17.19          flat     replayed        2077      10986    4.85  1.0e-15  
       647  647              prime    replayed        8278       9219    0.99  7.4e-16  bluestein
       648  2^3.3^4          chain3   replayed        1205       1494    1.11  4.8e-16  
       649  11.59            prime    replayed        8257      10760    1.23  3.8e-16  bluestein
       650  2.5^2.13         chain3   replayed        1331       1702    1.10  5.4e-16  
       651  3.7.31           2p       replayed        2274       7854    2.53  4.4e-16  flips differ 1.42x
       652  2^2.163          prime    replayed        9415       8637    0.84  5.2e-16  bluestein
       653  653              prime    replayed       11096      12891    0.96  6.0e-16  bluestein
       654  2.3.109          prime    replayed        9570       8828    0.89  5.5e-16  bluestein
       655  5.131            prime    replayed        8616      11408    1.25  6.7e-16  bluestein
       656  2^4.41           2p       replayed        2617      10213    3.79  4.6e-16  
       657  3^2.73           prime    replayed        9447       9438    0.86  6.4e-16  bluestein
       658  2.7.47           flat     replayed        2879      10721    3.54  5.8e-16  
       659  659              prime    replayed        8838      10856    1.06  5.3e-16  bluestein
       660  2^2.3.5.11       chain3   replayed        1384       2013    1.37  4.4e-16  
       661  661              prime    replayed        5406       9562    1.76  7.7e-16  rader
       662  2.331            prime    replayed        8818      12636    1.35  7.1e-16  bluestein
       663  3.13.17          flat     replayed        1885       6436    3.16  1.0e-15  
       664  2^3.83           prime    replayed        8609      10134    1.17  4.8e-16  bluestein
       665  5.7.19           flat     replayed        1851       5630    2.62  4.6e-16  
       666  2.3^2.37         chain3   replayed        2079       8771    3.73  4.8e-16  
       667  23.29            2p       replayed        2581      13023    4.21  4.7e-16  
       668  2^2.167          prime    replayed       10038      11054    0.97  4.8e-16  bluestein
       669  3.223            prime    replayed        8271      11590    1.28  6.9e-16  bluestein
       670  2.5.67           prime    replayed       11856      12400    1.02  6.4e-16  bluestein
       671  11.61            prime    replayed        8667       8835    0.99  7.1e-16  bluestein
       672  2^5.3.7          2p       replayed        1010       1413    1.26  3.3e-16  
       673  673              prime    replayed        4045      10323    2.33  6.7e-16  rader
       674  2.337            prime    replayed        8686      13310    1.38  7.2e-16  bluestein
       675  3^3.5^2          2p       replayed        1598       2060    1.10  6.2e-16  
       676  2^2.13^2         chain3   replayed        1789       2864    1.38  6.7e-16  
       677  677              prime    replayed        6572      10130    1.53  1.0e-15  rader
       678  2.3.113          prime    replayed        8567      11182    1.23  3.9e-16  bluestein
       679  7.97             prime    replayed        9624       8537    0.66  4.7e-16  bluestein
       680  2^3.5.17         flat     replayed        1797       5556    2.42  6.6e-16  
       681  3.227            prime    replayed        7566      11153    1.23  5.8e-16  bluestein
       682  2.11.31          flat     replayed        2466      11167    3.07  5.8e-16  flips differ 1.44x
       683  683              prime    replayed        8982      10878    0.42  6.0e-16  flips differ 3.30x bluestein
       684  2^2.3^2.19       chain3   replayed        1513       5700    3.34  4.7e-16  
       685  5.137            prime    replayed        8890       9853    1.02  5.5e-16  bluestein
       686  2.7^3            flat     replayed        1740       1574    0.89  4.1e-16  
       687  3.229            prime    replayed        8514      11106    1.25  4.9e-16  bluestein
       688  2^4.43           flat     replayed        2529       9754    3.74  5.8e-16  
       689  13.53            prime    replayed        8164      12052    1.21  5.7e-16  bluestein
       690  2.3.5.23         chain3   replayed        1656       6581    3.90  5.0e-16  
       691  691              prime    replayed        5537       9896    1.63  8.6e-16  rader
       692  2^2.173          prime    replayed        8257      10605    0.60  5.5e-16  flips differ 2.19x bluestein
       693  3^2.7.11         flat     replayed        1684       2084    1.12  6.0e-16  
       694  2.347            prime    replayed        8237       9557    1.07  5.2e-16  bluestein
       695  5.139            prime    replayed        8285       9044    1.03  6.5e-16  bluestein
       696  2^3.3.29         flat     replayed        2571       7846    2.91  4.8e-16  
       697  17.41            2p       replayed        2507      14197    5.09  6.4e-16  
       698  2.349            prime    replayed        7768      15293    1.20  6.8e-16  bluestein
       699  3.233            prime    replayed        7887       9073    1.00  4.7e-16  bluestein
       700  2^2.5^2.7        chain3   replayed        1176       1372    0.88  3.2e-16  
       701  701              prime    replayed        6688       9820    1.41  9.5e-16  rader
       702  2.3^3.13         chain3   replayed        1503       1804    1.13  5.4e-16  
       703  19.37            2p       replayed        2325      13159    5.55  4.6e-16  
       704  2^6.11           2p       replayed        1127       1569    1.31  5.0e-16  
       705  3.5.47           2p       replayed        2823      14837    4.42  5.3e-16  
       706  2.353            prime    replayed        8205       9497    1.10  6.7e-16  bluestein
       707  7.101            prime    replayed        8300       7390    0.87  6.1e-16  bluestein
       708  2^2.3.59         prime    replayed        8026      10736    1.31  7.1e-16  bluestein
       709  709              prime    replayed        7943      10731    1.17  5.5e-16  bluestein
       710  2.5.71           prime    replayed        8174       8984    1.09  5.5e-16  bluestein
       711  3^2.79           prime    replayed        6946       9871    1.35  4.5e-16  bluestein
       712  2^3.89           prime    replayed        8138       9088    1.09  6.0e-16  bluestein
       713  23.31            2p       replayed        2784      13441    4.73  6.0e-16  
       714  2.3.7.17         chain3   replayed        1701       7457    3.46  6.2e-16  
       715  5.11.13          flat     replayed        1831       2821    1.53  5.5e-16  
       716  2^2.179          prime    replayed        8024      10969    1.18  5.6e-16  bluestein
       717  3.239            prime    replayed        8058       8605    1.04  6.2e-16  bluestein
       718  2.359            prime    replayed        8741      13057    1.27  5.1e-16  bluestein
       719  719              prime    replayed       11642      13025    1.11  6.0e-16  bluestein
       720  2^4.3^2.5        chain3   replayed        1286       1201    0.87  5.0e-16  
       721  7.103            prime    replayed        8161      10344    1.16  5.1e-16  bluestein
       722  2.19^2           flat     replayed        2327      11195    4.63  1.0e-15  
       723  3.241            prime    replayed        8217       7733    0.81  3.9e-16  bluestein
       724  2^2.181          prime    replayed        8147       7775    0.93  6.5e-16  bluestein
       725  5^2.29           flat     replayed        2810       7555    2.69  5.7e-16  
       726  2.3.11^2         chain3   replayed        1668       2577    1.48  4.5e-16  
       727  727              prime    replayed        5286      11882    2.23  7.1e-16  rader
       728  2^3.7.13         chain3   replayed        1437       1608    1.03  5.1e-16  
       729  3^6              2p       replayed        1333       2381    1.42  4.5e-16  flips differ 1.42x
       730  2.5.73           prime    replayed       11584      10796    0.93  4.9e-16  bluestein
       731  17.43            2p       replayed        2592      17248    5.49  5.2e-16  flips differ 1.42x
       732  2^2.3.61         prime    replayed        8199       7841    0.85  5.6e-16  bluestein
       733  733              prime    replayed        8006      10152    1.23  4.1e-16  bluestein
       734  2.367            prime    replayed        7119      10249    1.20  5.0e-16  bluestein
       735  3.5.7^2          flat     replayed        1942       1969    0.78  5.4e-16  
       736  2^5.23           2p       replayed        1434       6420    4.44  4.1e-16  
       737  11.67            prime    replayed        8187      13387    1.63  6.9e-16  bluestein
       738  2.3^2.41         chain3   replayed        3000      10059    3.31  4.5e-16  
       739  739              prime    replayed        8229      10500    1.26  5.8e-16  bluestein
       740  2^2.5.37         flat     replayed        2621       9633    3.67  4.8e-16  
       741  3.13.19          chain3   replayed        1929       6962    3.60  5.6e-16  
       742  2.7.53           prime    replayed        7324      11437    1.51  4.9e-16  bluestein
       743  743              prime    replayed       11251      14478    1.27  5.3e-16  bluestein
       744  2^3.3.31         flat     replayed        2622       9509    2.46  4.5e-16  flips differ 1.73x
       745  5.149            prime    replayed        8147       9517    1.13  5.5e-16  bluestein
       746  2.373            prime    replayed        8124      10209    1.25  5.3e-16  bluestein
       747  3^2.83           prime    replayed        7980      10232    1.27  6.8e-16  bluestein
       748  2^2.11.17        flat     replayed        2341       6813    2.90  6.8e-16  
       749  7.107            prime    replayed        7260      10544    1.44  4.6e-16  bluestein
       750  2.3.5^3          chain3   replayed        1374       1416    0.94  4.7e-16  
       751  751              prime    replayed        7979       8201    0.93  5.7e-16  bluestein
       752  2^4.47           flat     replayed        2917      10810    3.68  5.8e-16  
       753  3.251            prime    replayed        7892       8853    1.06  5.1e-16  bluestein
       754  2.13.29          flat     replayed        2640       9501    3.53  9.0e-16  
       755  5.151            prime    replayed        7745       8249    1.02  6.7e-16  bluestein
       756  2^2.3^3.7        chain3   replayed        1304       1585    1.19  4.8e-16  
       757  757              prime    replayed        7340       9554    1.26  5.8e-16  rader
       758  2.379            prime    replayed        8157       8657    1.05  5.0e-16  bluestein
       759  3.11.23          flat     replayed        2505       7768    3.03  6.2e-16  
       760  2^3.5.19         flat     replayed        1928       6176    3.03  4.9e-16  
       761  761              prime    replayed        6105       9432    1.45  8.0e-16  rader
       762  2.3.127          prime    replayed        8082       9936    1.20  4.1e-16  bluestein
       763  7.109            prime    replayed        8303       8936    0.92  6.2e-16  bluestein
       764  2^2.191          prime    replayed        8391       8576    1.02  5.0e-16  bluestein
       765  3^2.5.17         flat     replayed        2078       6487    3.09  6.3e-16  
       766  2.383            prime    replayed       10578      10111    0.82  5.3e-16  bluestein
       767  13.59            prime    replayed        9688      14211    1.19  4.7e-16  flips differ 1.29x bluestein
       768  2^8.3            2p       replayed        1034       1282    0.93  3.5e-16  flips differ 1.36x
       769  769              prime    replayed        4194       8011    1.84  5.3e-16  rader
       770  2.5.7.11         chain3   replayed        1507       1816    1.20  3.4e-16  
       771  3.257            prime    replayed        7708       8325    1.06  5.8e-16  bluestein
       772  2^2.193          prime    replayed        8537       8332    0.94  6.4e-16  bluestein
       773  773              prime    replayed        8195      11049    1.29  6.2e-16  bluestein
       774  2.3^2.43         chain3   replayed        2757      13532    4.04  4.9e-16  
       775  5^2.31           2p       replayed        2070       8670    4.06  5.1e-16  
       776  2^3.97           prime    replayed        8236       8264    0.87  3.3e-16  bluestein
       777  3.7.37           2p       replayed        2276      10657    4.62  4.9e-16  
       778  2.389            prime    replayed        8287      11109    0.76  6.1e-16  flips differ 1.63x bluestein
       779  19.41            2p       replayed        2896      15757    5.16  5.0e-16  
       780  2^2.3.5.13       chain3   replayed        1532       1762    1.14  7.1e-16  
       781  11.71            prime    replayed        8736      12315    1.22  5.8e-16  bluestein
       782  2.17.23          flat     replayed        2771      12407    4.41  1.9e-15  
       783  3^3.29           flat     replayed        3388       9433    2.69  4.5e-16  
       784  2^4.7^2          chain3   replayed        1264       1458    1.05  5.2e-16  
       785  5.157            prime    replayed        8487       9663    1.11  6.4e-16  bluestein
       786  2.3.131          prime    replayed        7909      11770    1.44  5.7e-16  bluestein
       787  787              prime    replayed        8090      10449    1.22  5.4e-16  bluestein
       788  2^2.197          prime    replayed        8259      10349    0.89  6.7e-16  flips differ 1.45x bluestein
       789  3.263            prime    replayed        8213      12311    1.32  6.3e-16  bluestein
       790  2.5.79           prime    replayed        8134      13057    1.46  4.7e-16  bluestein
       791  7.113            prime    replayed        8447      13502    1.49  6.7e-16  bluestein
       792  2^3.3^2.11       chain3   replayed        1461       1950    1.18  5.5e-16  
       793  13.61            prime    replayed        8240       9675    1.10  5.4e-16  bluestein
       794  2.397            prime    replayed        8626      11956    1.34  6.6e-16  bluestein
       795  3.5.53           prime    replayed        8418      14018    1.44  3.9e-16  bluestein
       796  2^2.199          prime    replayed        8311      10510    1.21  6.5e-16  bluestein
       797  797              prime    replayed        8253      10575    1.27  5.1e-16  bluestein
       798  2.3.7.19         flat     replayed        2423       8218    2.84  5.0e-16  
       799  17.47            2p       replayed        2992      17274    5.48  4.6e-16  
       800  2^5.5^2          2p       replayed        1231       1439    1.03  3.1e-16  
       801  3^2.89           prime    replayed        8585      11341    1.16  7.2e-16  bluestein
       802  2.401            prime    replayed        8198       8942    0.98  4.9e-16  bluestein
       803  11.73            prime    replayed        8140      10126    1.19  6.2e-16  bluestein
       804  2^2.3.67         prime    replayed        7909      14247    1.53  5.3e-16  bluestein
       805  5.7.23           flat     replayed        2553       8341    2.95  5.6e-16  
       806  2.13.31          flat     replayed        2996       9970    3.27  8.8e-16  
       807  3.269            prime    replayed        7815      11803    1.29  5.0e-16  bluestein
       808  2^3.101          prime    replayed        8144       8755    1.03  5.3e-16  bluestein
       809  809              prime    replayed        7367      11831    1.48  4.9e-16  bluestein
       810  2.3^4.5          chain3   replayed        1573       1832    1.09  5.5e-16  
       811  811              prime    replayed        7599       9654    1.24  5.9e-16  bluestein
       812  2^2.7.29         flat     replayed        2798       9095    2.80  4.9e-16  
       813  3.271            prime    replayed        8219       9692    1.17  6.2e-16  bluestein
       814  2.11.37          flat     replayed        3145      11999    3.53  8.1e-16  
       815  5.163            prime    replayed        8339      10234    1.08  6.2e-16  bluestein
       816  2^4.3.17         chain3   replayed        1723       6605    3.57  4.7e-16  
       817  19.43            flat     replayed        3248      16886    5.01  8.3e-16  
       818  2.409            prime    replayed        7857      11205    1.36  6.4e-16  bluestein
       819  3^2.7.13         flat     replayed        2071       2259    1.08  7.7e-16  
       820  2^2.5.41         flat     replayed        2904      11657    3.84  4.2e-16  
       821  821              prime    replayed        7538      11900    1.44  6.9e-16  bluestein
       822  2.3.137          prime    replayed        8335      11351    1.35  5.7e-16  bluestein
       823  823              prime    replayed        8731      11147    1.25  6.4e-16  bluestein
       824  2^3.103          prime    replayed        8117      12220    1.43  7.9e-16  bluestein
       825  3.5^2.11         chain3   replayed        1754       2410    1.33  4.4e-16  
       826  2.7.59           prime    replayed        7883      11396    0.81  5.2e-16  flips differ 1.74x bluestein
       827  827              prime    replayed        8332      13728    1.49  5.9e-16  bluestein
       828  2^2.3^2.23       chain3   replayed        2250       7282    3.18  5.1e-16  
       829  829              prime    replayed        7562      11603    1.47  1.1e-15  rader
       830  2.5.83           prime    replayed        7915      11987    1.47  5.4e-16  bluestein
       831  3.277            prime    replayed        9369      12322    1.25  7.0e-16  bluestein
       832  2^6.13           2p       replayed        1449       1803    1.21  5.1e-16  
       833  7^2.17           flat     replayed        2345       6658    2.29  6.8e-16  
       834  2.3.139          prime    replayed        8700      11925    1.21  6.2e-16  bluestein
       835  5.167            prime    replayed        8687      11989    1.38  5.7e-16  bluestein
       836  2^2.11.19        flat     replayed        2764       7735    2.71  8.2e-16  
       837  3^3.31           flat     replayed        3469      10338    2.76  7.1e-16  
       838  2.419            prime    replayed        9418      13882    1.39  5.3e-16  bluestein
       839  839              prime    replayed        8091      11108    1.15  6.0e-16  bluestein
       840  2^3.3.5.7        chain3   replayed        1367       1654    1.12  3.0e-16  
       841  29^2             flat     replayed        3835      16583    3.62  1.2e-15  
       842  2.421            prime    replayed        8122      11263    1.30  6.8e-16  bluestein
       843  3.281            prime    replayed        7763      11896    1.29  5.1e-16  bluestein
       844  2^2.211          prime    replayed        8228      12289    1.31  4.7e-16  bluestein
       845  5.13^2           flat     replayed        2340       3144    1.30  9.2e-16  
       846  2.3^2.47         flat     replayed        4710      12724    2.63  5.7e-16  
       847  7.11^2           flat     replayed        2240       3309    1.45  5.4e-16  
       848  2^4.53           prime    replayed        8370      13738    1.59  4.4e-16  bluestein
       849  3.283            prime    replayed        8408      11473    1.33  5.0e-16  bluestein
       850  2.5^2.17         chain3   replayed        1910       6724    3.50  4.9e-16  
       851  23.37            2p       replayed        3172      17082    4.99  5.5e-16  
       852  2^2.3.71         prime    replayed        8397      10639    1.26  5.5e-16  bluestein
       853  853              prime    replayed        8101      11188    1.31  7.3e-16  bluestein
       854  2.7.61           prime    replayed        8238       9276    1.08  4.6e-16  bluestein
       855  3^2.5.19         flat     replayed        2396       7495    3.08  5.9e-16  
       856  2^3.107          prime    replayed        8184      11856    1.44  7.9e-16  bluestein
       857  857              prime    replayed        7872      11363    1.41  6.4e-16  bluestein
       858  2.3.11.13        flat     replayed        2516       3007    1.17  9.6e-16  
       859  859              prime    replayed        6579      12741    1.56  8.7e-16  flips differ 1.39x rader
       860  2^2.5.43         flat     replayed        3369      15044    3.75  6.3e-16  flips differ 1.38x
       861  3.7.41           2p       replayed        2760      12000    4.33  4.1e-16  
       862  2.431            prime    replayed        8056      12194    1.43  6.0e-16  bluestein
       863  863              prime    replayed        8235      12256    1.31  6.5e-16  bluestein
       864  2^5.3^3          2p       replayed        1423       1627    1.00  5.8e-16  
       865  5.173            prime    replayed        7060      11991    1.31  5.0e-16  flips differ 1.33x bluestein
       866  2.433            prime    replayed        7883      10138    1.25  5.7e-16  bluestein
       867  3.17^2           flat     replayed        2602      12560    4.79  1.1e-15  
       868  2^2.7.31         flat     replayed        2997       9464    2.90  5.3e-16  
       869  11.79            prime    replayed        7941      12712    1.50  4.6e-16  bluestein
       870  2.3.5.29         flat     replayed        3138       9029    2.81  5.6e-16  
       871  13.67            prime    replayed        7419      15717    1.91  6.4e-16  bluestein
       872  2^3.109          prime    replayed        8160       9840    1.02  6.4e-16  bluestein
       873  3^2.97           prime    replayed        8612      10086    0.96  6.1e-16  bluestein
       874  2.19.23          flat     replayed        3455      15381    4.43  1.1e-15  
       875  5^3.7            flat     replayed        2128       2603    1.18  4.5e-16  
       876  2^2.3.73         prime    replayed        8511       9725    1.01  5.0e-16  bluestein
       877  877              prime    replayed        8190      12720    1.26  5.2e-16  bluestein
       878  2.439            prime    replayed        8141      12686    1.53  5.9e-16  bluestein
       879  3.293            prime    replayed        8636      12874    1.46  4.9e-16  bluestein
       880  2^4.5.11         chain3   replayed        1552       1980    1.26  4.3e-16  
       881  881              prime    replayed        5809      12501    2.13  5.0e-16  rader
       882  2.3^2.7^2        flat     replayed        2459       1976    0.79  5.4e-16  
       883  883              prime    replayed        9785      14097    1.35  5.5e-16  bluestein
       884  2^2.13.17        chain3   replayed        2185       8665    3.56  7.0e-16  
       885  3.5.59           prime    replayed        8370      12576    1.46  5.8e-16  bluestein
       886  2.443            prime    replayed        8654      14110    1.54  6.4e-16  bluestein
       887  887              prime    replayed        8244      12470    1.44  6.8e-16  bluestein
       888  2^3.3.37         chain3   replayed        2718      12020    4.11  4.5e-16  
       889  7.127            prime    replayed        7587      10734    1.08  5.5e-16  flips differ 1.32x bluestein
       890  2.5.89           prime    replayed        8441      11712    1.30  6.3e-16  bluestein
       891  3^4.11           chain3   replayed        2114       2627    1.20  5.5e-16  
       892  2^2.223          prime    replayed        9446      12789    0.71  5.7e-16  flips differ 1.90x bluestein
       893  19.47            flat     replayed        3832      22646    5.04  8.8e-16  flips differ 1.31x
       894  2.3.149          prime    replayed        8973      15181    1.57  6.8e-16  bluestein
       895  5.179            prime    replayed        9331      13467    1.34  5.4e-16  bluestein
       896  2^7.7            chain3   replayed        1324       1396    0.95  3.0e-16  
       897  3.13.23          flat     replayed        3199       9978    3.01  1.1e-15  
       898  2.449            prime    replayed        8751      13903    1.41  7.0e-16  bluestein
       899  29.31            2p       replayed        3763      18403    4.84  5.6e-16  
       900  2^2.3^2.5^2      chain3   replayed        1583       1920    1.10  3.7e-16  
       901  17.53            prime    replayed        8450      21471    2.33  6.0e-16  bluestein
       902  2.11.41          flat     replayed        4082      15421    3.10  8.0e-16  
       903  3.7.43           flat     replayed        4204      14674    3.24  1.9e-15  
       904  2^3.113          prime    replayed        8579      15302    1.68  6.9e-16  bluestein
       905  5.181            prime    replayed        8404      10620    0.78  5.3e-16  flips differ 1.59x bluestein
       906  2.3.151          prime    replayed        8638      11980    1.26  4.0e-16  bluestein
       907  907              prime    replayed        9061      22678    2.20  7.8e-16  bluestein
       908  2^2.227          prime    replayed        9054      13016    1.36  5.8e-16  bluestein
       909  3^2.101          prime    replayed       10025      10156    0.85  5.3e-16  bluestein
       910  2.5.7.13         chain3   replayed        1795       2393    1.31  5.4e-16  
       911  911              prime    replayed        8179      18001    2.11  5.2e-16  bluestein
       912  2^4.3.19         chain3   replayed        1896       7591    3.12  4.7e-16  
       913  11.83            prime    replayed        8002      13699    1.70  4.6e-16  bluestein
       914  2.457            prime    replayed        7939      11968    1.32  5.0e-16  bluestein
       915  3.5.61           prime    replayed        8569      10933    1.24  6.1e-16  bluestein
       916  2^2.229          prime    replayed        8798      13113    1.29  5.1e-16  bluestein
       917  7.131            prime    replayed        8571      13687    1.40  7.9e-16  bluestein
       918  2.3^3.17         flat     replayed        2650       8103    2.94  6.6e-16  
       919  919              prime    replayed        8221      20225    2.29  6.3e-16  bluestein
       920  2^3.5.23         chain3   replayed        2326       8544    3.65  3.6e-16  
       921  3.307            prime    replayed        8216      18717    2.06  7.7e-16  bluestein
       922  2.461            prime    replayed        8502      13326    1.43  6.0e-16  bluestein
       923  13.71            prime    replayed        7506      13210    1.65  5.2e-16  bluestein
       924  2^2.3.7.11       chain3   replayed        1778       2586    1.37  4.8e-16  
       925  5^2.37           2p       replayed        4550      12464    2.52  5.0e-16  
       926  2.463            prime    replayed        8350      12274    1.34  5.5e-16  bluestein
       927  3^2.103          prime    replayed        8492      15357    1.73  6.3e-16  bluestein
       928  2^5.29           2p       replayed        2116      12465    5.81  4.5e-16  
       929  929              prime    replayed        7392      17741    2.30  7.1e-16  rader
       930  2.3.5.31         chain3   replayed        3660      11177    2.89  4.5e-16  
       931  7^2.19           flat     replayed        2733       8131    2.31  6.2e-16  
       932  2^2.233          prime    replayed        9309      13140    0.70  6.0e-16  flips differ 1.93x bluestein
       933  3.311            prime    replayed        8530      18170    1.74  6.2e-16  bluestein
       934  2.467            prime    replayed        8921      12612    1.29  5.4e-16  bluestein
       935  5.11.17          flat     replayed        2740       9158    3.20  6.2e-16  
       936  2^3.3^2.13       chain3   replayed        1819       2396    1.15  6.9e-16  
       937  937              prime    replayed        6638      19449    2.75  7.6e-16  rader
       938  2.7.67           prime    replayed        8663      17370    1.72  6.9e-16  bluestein
       939  3.313            prime    replayed        8324      18919    1.91  5.1e-16  bluestein
       940  2^2.5.47         flat     replayed        3622      15129    3.92  5.2e-16  
       941  941              prime    replayed        7403      12569    1.63  5.9e-16  bluestein
       942  2.3.157          prime    replayed        7692      10747    1.34  5.6e-16  bluestein
       943  23.41            2p       replayed        3772      18089    4.01  5.2e-16  
       944  2^4.59           prime    replayed        8116      12948    1.54  5.2e-16  bluestein
       945  3^3.5.7          chain3   replayed        2016       2424    1.15  5.0e-16  
       946  2.11.43          flat     replayed        3944      15083    3.78  7.5e-16  
       947  947              prime    replayed        8845      16294    1.68  6.1e-16  bluestein
       948  2^2.3.79         prime    replayed        9043      13781    1.50  4.6e-16  bluestein
       949  13.73            prime    replayed        8640      11341    1.30  6.9e-16  bluestein
       950  2.5^2.19         flat     replayed        3276       8055    2.20  4.3e-16  
       951  3.317            prime    replayed        8462      11542    1.26  5.6e-16  bluestein
       952  2^3.7.17         flat     replayed        2375       8587    3.24  8.3e-16  
       953  953              prime    replayed        7469      13278    1.23  8.1e-16  flips differ 1.39x rader
       954  2.3^2.53         prime    replayed        8803      19854    1.88  7.1e-16  bluestein
       955  5.191            prime    replayed        8786      10926    1.02  5.2e-16  bluestein
       956  2^2.239          prime    replayed        8316      12617    1.47  5.1e-16  bluestein
       957  3.11.29          chain3   replayed        3078      13646    3.88  4.6e-16  flips differ 1.28x
       958  2.479            prime    replayed        8744      12656    1.38  6.0e-16  bluestein
       959  7.137            prime    replayed        9186      18682    1.42  6.9e-16  bluestein
       960  2^6.3.5          2p       replayed        1529       1646    0.84  3.9e-16  
       961  31^2             flat     replayed        4467      21130    4.30  8.2e-16  
       962  2.13.37          flat     replayed        3727      15703    3.75  8.0e-16  
       963  3^2.107          prime    replayed        8153      14764    1.62  6.9e-16  bluestein
       964  2^2.241          prime    replayed        9642      10203    0.98  5.2e-16  bluestein
       965  5.193            prime    replayed        9549      11275    1.09  5.7e-16  bluestein
       966  2.3.7.23         chain3   replayed        2883      11314    3.29  5.8e-16  
       967  967              prime    replayed        7822      18072    1.50  5.2e-16  flips differ 1.52x bluestein
       968  2^3.11^2         chain3   replayed        2143       4069    1.69  4.3e-16  
       969  3.17.19          flat     replayed        3300      14355    4.22  9.4e-16  
       970  2.5.97           prime    replayed        8756      10478    1.05  5.5e-16  bluestein
       971  971              prime    replayed        8621      16481    1.57  8.0e-16  bluestein
       972  2^2.3^5          chain3   replayed        1767       2120    1.01  4.9e-16  
       973  7.139            prime    replayed        8561      14554    1.57  6.1e-16  bluestein
       974  2.487            prime    replayed        9281      13859    1.35  6.5e-16  bluestein
       975  3.5^2.13         flat     replayed        3084       3559    1.07  6.2e-16  
       976  2^4.61           prime    replayed        9283      12812    1.27  4.7e-16  bluestein
       977  977              prime    replayed       10011      16914    1.23  5.7e-16  flips differ 1.56x bluestein
       978  2.3.163          prime    replayed        8515      12541    1.42  6.3e-16  bluestein
       979  11.89            prime    replayed        9150      17672    1.66  6.8e-16  bluestein
       980  2^2.5.7^2        flat     replayed        2299       2125    0.83  3.9e-16  
       981  3^2.109          prime    replayed        9261      12437    1.29  8.2e-16  bluestein
       982  2.491            prime    replayed        9029      14749    1.43  6.1e-16  bluestein
       983  983              prime    replayed        8397      13597    1.42  6.1e-16  bluestein
       984  2^3.3.41         chain3   replayed        3162      13162    3.91  5.2e-16  
       985  5.197            prime    replayed        8807      13511    1.35  5.9e-16  bluestein
       986  2.17.29          flat     replayed        3841      20340    4.68  6.8e-16  
       987  3.7.47           2p       replayed        3759      16558    4.13  4.1e-16  
       988  2^2.13.19        chain3   replayed        2589      13445    4.41  5.0e-16  
       989  23.43            flat     replayed        4306      21965    4.83  9.1e-16  
       990  2.3^2.5.11       chain3   replayed        1954       2671    1.34  5.1e-16  
       991  991              prime    replayed        8048      13828    1.66  4.7e-16  bluestein
       992  2^5.31           flat     replayed        3140      12245    3.72  1.5e-15  
       993  3.331            prime    replayed        8066      18163    2.23  7.4e-16  bluestein
       994  2.7.71           prime    replayed        8451      14431    1.46  5.2e-16  flips differ 1.26x bluestein
       995  5.199            prime    replayed        9723      18731    0.95  5.9e-16  flips differ 1.95x bluestein
       996  2^2.3.83         prime    replayed        8876      19584    1.61  6.5e-16  flips differ 1.56x bluestein
       997  997              prime    replayed        8683      14330    1.57  6.2e-16  bluestein
       998  2.499            prime    replayed        8551      14966    1.41  5.4e-16  bluestein
       999  3^3.37           2p       replayed        6026      13813    1.14  5.0e-16  flips differ 1.88x
      1000  2^3.5^3          chain3   replayed        1724       1974    1.05  4.1e-16  
      1001  7.11.13          flat     replayed        2880       4446    1.23  6.1e-16  
      1002  2.3.167          prime    replayed       12310      15235    1.22  8.3e-16  bluestein
      1003  17.59            prime    replayed        8729      26522    2.42  5.6e-16  bluestein
      1004  2^2.251          prime    replayed        8210      12986    1.35  4.6e-16  bluestein
      1005  3.5.67           prime    replayed        7891      18890    2.37  6.1e-16  bluestein
      1006  2.503            prime    replayed        7733      12566    1.50  5.4e-16  bluestein
      1007  19.53            prime    replayed        8943      23747    2.36  6.4e-16  bluestein
      1008  2^4.3^2.7        chain3   replayed        1724       1834    0.93  5.1e-16  
      1009  1009             prime    replayed        8600      18139    2.07  7.5e-16  bluestein
      1010  2.5.101          prime    replayed        7602      11636    1.51  5.5e-16  bluestein
      1011  3.337            prime    replayed        8220      18588    2.10  5.5e-16  bluestein
      1012  2^2.11.23        chain3   replayed        3083      10678    3.40  5.5e-16  
      1013  1013             prime    replayed        8884      19774    1.85  6.3e-16  bluestein
      1014  2.3.13^2         chain3   replayed        2449       4162    1.46  7.4e-16  
      1015  5.7.29           flat     replayed        4111      13557    2.90  5.5e-16  
      1016  2^3.127          prime    replayed        9667      13374    1.33  4.8e-16  bluestein
      1017  3^2.113          prime    replayed        9004      18440    1.95  5.8e-16  bluestein
      1018  2.509            prime    replayed        8707      13024    1.43  4.7e-16  bluestein
      1019  1019             prime    replayed        9215      13557    1.43  5.7e-16  bluestein
      1020  2^2.3.5.17       chain3   replayed        2461       8272    3.35  5.6e-16  
      1021  1021             prime    replayed        8420      13084    1.40  6.1e-16  bluestein
      1022  2.7.73           prime    replayed        8845      12042    1.24  5.5e-16  bluestein
      1023  3.11.31          chain3   replayed        3288      13239    3.86  5.3e-16  
      1024  2^10             ztt      replayed        1310       1387    0.94  4.0e-16  
      1025  5^2.41           2p       replayed        3775      16810    3.33  5.0e-16  flips differ 1.41x
      1026  2.3^3.19         chain3   replayed        2961      10805    3.57  5.4e-16  
      1027  13.79            prime    replayed       18275      18048    0.85  4.9e-16  bluestein
      1028  2^2.257          prime    raced          15863      12552    0.74  4.8e-16  bluestein
      1029  3.7^3            flat     replayed        2579       3855    1.38  4.9e-16  
      1030  2.5.103          prime    raced          17016      18576    0.79  6.2e-16  flips differ 1.34x bluestein
      1031  1031             prime    replayed       17725      16069    0.79  6.2e-16  bluestein
      1032  2^3.3.43         chain3   replayed        3861      17184    4.17  5.2e-16  
      1033  1033             prime    replayed       12138      16276    1.19  1.1e-15  rader
      1034  2.11.47          flat     replayed        4487      17026    3.73  7.6e-16  
      1035  3^2.5.23         flat     replayed        3157      10517    3.22  5.7e-16  
      1036  2^2.7.37         flat     replayed        3673      13618    3.70  4.7e-16  
      1037  17.61            prime    replayed       15644      18324    1.13  4.5e-16  bluestein
      1038  2.3.173          prime    raced          16497      16325    0.66  7.6e-16  flips differ 1.45x bluestein
      1039  1039             prime    replayed       16424      15052    0.88  5.7e-16  bluestein
      1040  2^4.5.13         chain3   replayed        1913       2424    1.17  6.1e-16  
      1041  3.347            prime    replayed       15788      14387    0.89  6.2e-16  bluestein
      1042  2.521            prime    raced          14975      15303    0.99  6.5e-16  bluestein
      1043  7.149            prime    replayed       15385      13231    0.80  4.5e-16  bluestein
      1044  2^2.3^2.29       chain3   replayed        3219      11743    3.48  5.3e-16  
      1045  5.11.19          flat     replayed        3113      10270    3.18  5.5e-16  
      1046  2.523            prime    raced          15195      15640    0.88  5.5e-16  bluestein
      1047  3.349            prime    replayed       16421      14939    0.87  6.2e-16  bluestein
      1048  2^3.131          prime    raced          15423      15918    1.01  5.7e-16  bluestein
      1049  1049             prime    replayed       17487      16270    0.91  5.6e-16  bluestein
      1050  2.3.5^2.7        chain3   replayed        2248       2119    0.91  4.3e-16  
      1051  1051             prime    replayed        8480      16655    1.75  7.3e-16  rader
      1052  2^2.263          prime    raced          21961      15239    0.33  5.2e-16  flips differ 2.13x bluestein
      1053  3^4.13           chain3   replayed        2369       3437    1.34  6.9e-16  
      1054  2.17.31          flat     replayed        4052      19122    4.61  1.6e-15  
      1055  5.211            prime    replayed       15368      17847    1.08  6.5e-16  bluestein
      1056  2^5.3.11         chain3   replayed        2021       2447    0.90  4.8e-16  flips differ 1.32x
      1057  7.151            prime    replayed       18252      14774    0.75  5.5e-16  bluestein
      1058  2.23^2           flat     replayed        4216      20486    4.40  1.4e-15  
      1059  3.353            prime    replayed       14580      14275    0.75  5.8e-16  flips differ 1.31x bluestein
      1060  2^2.5.53         prime    raced          15696      19956    1.18  4.6e-16  bluestein
      1061  1061             prime    replayed       15333      15999    0.97  5.8e-16  bluestein
      1062  2.3^2.59         prime    raced          16167      15741    0.78  5.3e-16  bluestein
      1063  1063             prime    replayed       17658      15861    0.85  6.5e-16  bluestein
      1064  2^3.7.19         chain3   replayed        2560       8785    2.50  4.2e-16  flips differ 1.34x
      1065  3.5.71           prime    replayed       15840      14514    0.85  4.5e-16  bluestein
      1066  2.13.41          flat     replayed        4527      19363    3.87  1.1e-15  
      1067  11.97            prime    replayed       15520      13001    0.73  5.1e-16  bluestein
      1068  2^2.3.89         prime    raced          15378      16057    0.96  6.4e-16  bluestein
      1069  1069             prime    replayed       15572      14816    0.92  4.9e-16  bluestein
      1070  2.5.107          prime    raced          16437      15965    0.83  5.3e-16  bluestein
      1071  3^2.7.17         flat     replayed        3075      10197    3.30  5.3e-16  
      1072  2^4.67           prime    raced          15673      20406    1.28  4.9e-16  bluestein
      1073  29.37            2p       replayed        4718      23276    4.59  4.9e-16  
      1074  2.3.179          prime    raced          15805      14653    0.92  5.3e-16  bluestein
      1075  5^2.43           2p       replayed        3544      17169    4.75  5.1e-16  
      1076  2^2.269          prime    raced          15695      16220    0.90  6.0e-16  bluestein
      1077  3.359            prime    replayed       15913      14710    0.73  5.4e-16  flips differ 1.30x bluestein
      1078  2.7^2.11         flat     replayed        3136       3587    0.80  6.1e-16  flips differ 1.44x
      1079  13.83            prime    replayed       16658      16494    0.97  5.0e-16  bluestein
      1080  2^3.3^3.5        chain3   replayed        1862       2308    1.06  4.7e-16  
      1081  23.47            2p       replayed        6002      25558    3.51  5.0e-16  
      1082  2.541            prime    raced          16612      13695    0.74  5.3e-16  bluestein
      1083  3.19^2           chain3   replayed        3318      18890    5.52  3.8e-16  
      1084  2^2.271          prime    raced          15786      13894    0.69  5.1e-16  bluestein
      1085  5.7.31           chain3   replayed        4051      12461    2.85  4.2e-16  
      1086  2.3.181          prime    raced          15891      12944    0.69  5.3e-16  bluestein
      1087  1087             prime    replayed       15507      24006    1.35  6.6e-16  bluestein
      1088  2^6.17           2p       replayed        2242       8570    3.80  5.7e-16  
      1089  3^2.11^2         chain3   replayed        2564       4355    1.64  5.7e-16  
      1090  2.5.109          prime    raced          15658      13131    0.81  5.6e-16  bluestein
      1091  1091             prime    replayed       15292      22878    1.43  8.5e-16  bluestein
      1092  2^2.3.7.13       chain3   replayed        2139       2950    1.35  4.8e-16  
      1093  1093             prime    replayed       10665      22515    1.96  1.1e-15  rader
      1094  2.547            prime    raced          17577      24068    1.20  8.0e-16  bluestein
      1095  3.5.73           prime    replayed       15066      15201    0.83  5.8e-16  bluestein
      1096  2^3.137          prime    raced          15658      15360    0.95  5.6e-16  bluestein
      1097  1097             prime    replayed       15830      19965    1.13  5.3e-16  bluestein
      1098  2.3^2.61         prime    raced          15663      12520    0.79  5.4e-16  bluestein
      1099  7.157            prime    replayed       16207      15819    0.85  5.3e-16  bluestein
      1100  2^2.5^2.11       chain3   replayed        2175       2956    1.21  4.0e-16  
      1101  3.367            prime    replayed       16050      17456    0.97  5.6e-16  bluestein
      1102  2.19.29          flat     replayed        4277      20779    4.73  1.1e-15  
      1103  1103             prime    replayed       15599      17239    1.00  4.9e-16  bluestein
      1104  2^4.3.23         chain3   replayed        2505      11867    3.98  7.1e-16  flips differ 1.30x
      1105  5.13.17          flat     replayed        3317      11927    3.35  6.9e-16  
      1106  2.7.79           prime    raced          14636      16255    0.85  4.7e-16  flips differ 1.30x bluestein
      1107  3^3.41           2p       replayed        3518      15879    4.50  5.2e-16  
      1108  2^2.277          prime    raced          15971      16644    0.86  5.3e-16  bluestein
      1109  1109             prime    replayed       15822      16068    1.00  6.9e-16  bluestein
      1110  2.3.5.37         chain3   replayed        3289      15232    4.05  5.7e-16  
      1111  11.101           prime    replayed       14684      13713    0.89  6.2e-16  bluestein
      1112  2^3.139          prime    raced          16736      15164    0.79  5.1e-16  bluestein
      1113  3.7.53           prime    replayed       16615      20504    1.15  3.8e-16  bluestein
      1114  2.557            prime    raced          15309      26290    1.40  5.4e-16  flips differ 1.26x bluestein
      1115  5.223            prime    replayed       16236      17297    0.96  5.4e-16  bluestein
      1116  2^2.3^2.31       flat     replayed        4139      14303    3.29  5.3e-16  
      1117  1117             prime    replayed       12231      18734    1.42  9.0e-16  rader
      1118  2.13.43          flat     replayed        4924      17023    3.33  1.0e-15  
      1119  3.373            prime    replayed       15586      16095    0.97  5.8e-16  bluestein
      1120  2^5.5.7          chain3   replayed        1770       2008    0.90  4.3e-16  
      1121  19.59            prime    replayed       15244      23670    1.50  4.6e-16  bluestein
      1122  2.3.11.17        chain3   replayed        3078       9993    2.97  5.3e-16  
      1123  1123             prime    replayed        9495      17580    1.73  8.4e-16  rader
      1124  2^2.281          prime    raced          16695      19265    0.98  5.5e-16  bluestein
      1125  3^2.5^3          flat     replayed        3137       3869    1.05  5.9e-16  
      1126  2.563            prime    raced          17456      22027    1.01  6.2e-16  bluestein
      1127  7^2.23           chain3   replayed        3498      14683    4.12  5.2e-16  
      1128  2^3.3.47         flat     replayed        4545      20302    3.93  6.9e-16  
      1129  1129             prime    replayed       15864      16322    0.95  5.3e-16  bluestein
      1130  2.5.113          prime    raced          15270      19624    1.18  5.2e-16  bluestein
      1131  3.13.29          chain3   replayed        3725      13631    3.60  6.0e-16  
      1132  2^2.283          prime    raced          15885      18790    0.86  4.7e-16  flips differ 1.33x bluestein
      1133  11.103           prime    replayed       16033      18813    0.96  5.7e-16  bluestein
      1134  2.3^4.7          chain3   replayed        2264       2923    1.16  6.5e-16  
      1135  5.227            prime    replayed       15820      14930    0.87  5.4e-16  bluestein
      1136  2^4.71           prime    raced          15797      16038    0.96  5.8e-16  bluestein
      1137  3.379            prime    replayed       16377      14052    0.73  5.2e-16  bluestein
      1138  2.569            prime    raced          16034      16142    0.99  3.7e-16  bluestein
      1139  17.67            prime    replayed       16131      30088    1.66  5.4e-16  bluestein
      1140  2^2.3.5.19       chain3   replayed        2625       9484    3.53  4.7e-16  
      1141  7.163            prime    replayed       15999      14739    0.86  4.9e-16  bluestein
      1142  2.571            prime    raced          16193      18289    0.97  5.9e-16  bluestein
      1143  3^2.127          prime    replayed       19336      17206    0.63  5.1e-16  bluestein
      1144  2^3.11.13        chain3   replayed        2579       4363    1.60  5.6e-16  
      1145  5.229            prime    replayed       15821      14765    0.78  3.8e-16  bluestein
      1146  2.3.191          prime    raced          16215      13036    0.80  4.3e-16  bluestein
      1147  31.37            flat     replayed        5105      26225    4.97  1.0e-15  
      1148  2^2.7.41         chain3   replayed        4218      16200    3.80  4.1e-16  
      1149  3.383            prime    replayed       16009      13441    0.83  5.4e-16  bluestein
      1150  2.5^2.23         chain3   replayed        2927      11402    3.88  4.6e-16  
      1151  1151             prime    replayed        9819      15441    1.51  7.5e-16  rader
      1152  2^7.3^2          chain3   replayed        1739       2201    1.15  4.9e-16  
      1153  1153             prime    replayed        6867      12792    1.61  8.8e-16  rader
      1154  2.577            prime    raced          16023      13839    0.73  4.4e-16  bluestein
      1155  3.5.7.11         chain3   replayed        2641       3646    1.28  3.5e-16  
      1156  2^2.17^2         chain3   replayed        3718      17035    4.20  5.8e-16  
      1157  13.89            prime    replayed       16115      17598    0.89  4.8e-16  bluestein
      1158  2.3.193          prime    raced          15860      12029    0.70  5.9e-16  bluestein
      1159  19.61            prime    replayed       15837      21577    1.23  4.3e-16  bluestein
      1160  2^3.5.29         flat     replayed        3704      13122    2.77  8.9e-16  
      1161  3^3.43           2p       replayed        3896      18660    4.47  5.8e-16  
      1162  2.7.83           prime    raced          16230      16792    0.98  5.3e-16  bluestein
      1163  1163             prime    replayed       16579      17232    0.96  4.8e-16  bluestein
      1164  2^2.3.97         prime    raced          15860      15732    0.94  5.4e-16  bluestein
      1165  5.233            prime    replayed       16230      17403    1.06  5.1e-16  bluestein
      1166  2.11.53          prime    raced          18679      21635    1.05  4.5e-16  bluestein
      1167  3.389            prime    replayed       16594      19832    1.09  5.1e-16  bluestein
      1168  2^4.73           prime    raced          15455      13257    0.83  6.1e-16  bluestein
      1169  7.167            prime    replayed       15534      17748    0.69  6.6e-16  flips differ 1.58x bluestein
      1170  2.3^2.5.13       chain3   replayed        2436       2874    1.16  5.6e-16  
      1171  1171             prime    replayed        8454      17668    1.90  8.3e-16  rader
      1172  2^2.293          prime    raced          15737      16532    0.78  5.9e-16  flips differ 1.32x bluestein
      1173  3.17.23          flat     replayed        4738      20275    3.33  8.3e-16  flips differ 1.28x
      1174  2.587            prime    raced          17001      18902    1.01  5.9e-16  bluestein
      1175  5^2.47           flat     replayed        4981      20155    3.76  5.6e-16  
      1176  2^3.3.7^2        chain3   replayed        2696       2351    0.85  3.9e-16  
      1177  11.107           prime    replayed       16111      19189    1.01  5.6e-16  bluestein
      1178  2.19.31          flat     replayed        4650      24593    4.83  1.2e-15  
      1179  3^2.131          prime    replayed       16232      19166    1.16  4.9e-16  bluestein
      1180  2^2.5.59         prime    raced          15414      18525    1.10  4.7e-16  bluestein
      1181  1181             prime    replayed       16025      17026    0.43  5.7e-16  flips differ 2.37x bluestein
      1182  2.3.197          prime    raced          15770      17447    0.96  5.3e-16  bluestein
      1183  7.13^2           chain3   replayed        2973       5112    1.53  7.3e-16  
      1184  2^5.37           2p       replayed        4379      15148    2.79  3.7e-16  flips differ 1.26x
      1185  3.5.79           prime    replayed       15910      17637    1.08  4.7e-16  bluestein
      1186  2.593            prime    raced          16174      18144    0.78  4.5e-16  flips differ 1.54x bluestein
      1187  1187             prime    replayed       15741      16676    1.00  6.4e-16  bluestein
      1188  2^2.3^3.11       chain3   replayed        2261       3316    1.32  4.5e-16  
      1189  29.41            2p       replayed        5582      28136    4.55  5.0e-16  
      1190  2.5.7.17         chain3   replayed        2711      12128    4.20  6.0e-16  
      1191  3.397            prime    replayed       15787      18235    1.02  6.7e-16  bluestein
      1192  2^3.149          prime    raced          15977      15324    0.95  5.0e-16  bluestein
      1193  1193             prime    replayed       15176      16181    0.75  5.0e-16  flips differ 1.39x bluestein
      1194  2.3.199          prime    raced          15362      15345    0.96  4.8e-16  bluestein
      1195  5.239            prime    replayed       15769      14918    0.92  5.6e-16  bluestein
      1196  2^2.13.23        chain3   replayed        3701      13527    3.38  5.6e-16  
      1197  3^2.7.19         flat     replayed        3970      10910    2.59  1.0e-15  
      1198  2.599            prime    raced          18439      17675    0.92  5.9e-16  bluestein
      1199  11.109           prime    replayed       16305      15981    0.79  6.3e-16  bluestein
      1200  2^4.3.5^2        chain3   replayed        2823       2181    0.75  6.0e-16  
      1201  1201             prime    replayed       10581      13507    1.22  7.2e-16  rader
      1202  2.601            prime    raced          15610      14401    0.87  5.3e-16  bluestein
      1203  3.401            prime    replayed       18013      15010    0.79  4.0e-16  bluestein
      1204  2^2.7.43         flat     replayed        5194      18465    3.36  6.5e-16  
      1205  5.241            prime    replayed       15876      12655    0.51  5.6e-16  flips differ 1.58x bluestein
      1206  2.3^2.67         prime    raced          14859      24403    1.49  6.1e-16  bluestein
      1207  17.71            prime    replayed       14924      24258    1.56  6.3e-16  bluestein
      1208  2^3.151          prime    raced          15823      14091    0.85  4.6e-16  bluestein
      1209  3.13.31          chain3   replayed        4071      15421    3.35  6.5e-16  
      1210  2.5.11^2         chain3   replayed        2630       4620    1.69  3.3e-16  
      1211  7.173            prime    replayed       17747      19398    0.83  5.8e-16  flips differ 1.29x bluestein
      1212  2^2.3.101        prime    raced          17999      13131    0.24  4.8e-16  flips differ 3.07x bluestein
      1213  1213             prime    replayed       16294      20912    1.07  6.5e-16  bluestein
      1214  2.607            prime    raced          15803      23921    1.32  6.0e-16  bluestein
      1215  3^5.5            chain3   replayed        2705       3323    1.16  5.4e-16  
      1216  2^6.19           2p       replayed        2739      10662    3.79  3.9e-16  
      1217  1217             prime    replayed        8805      20488    1.79  7.1e-16  flips differ 1.29x rader
      1218  2.3.7.29         flat     replayed        4742      13576    2.60  7.8e-16  
      1219  23.53            prime    replayed       16922      32485    0.78  4.6e-16  flips differ 2.50x bluestein
      1220  2^2.5.61         prime    raced          16591      15812    0.84  4.9e-16  bluestein
      1221  3.11.37          chain3   replayed        4301      18629    4.29  4.6e-16  
      1222  2.13.47          flat     replayed        5538      18427    2.84  7.6e-16  
      1223  1223             prime    replayed       15771      19360    1.09  4.9e-16  bluestein
      1224  2^3.3^2.17       chain3   replayed        2606      12032    4.25  5.4e-16  
      1225  5^2.7^2          flat     replayed        2945       3712    1.21  5.0e-16  
      1226  2.613            prime    raced          17363      20482    1.16  5.4e-16  bluestein
      1227  3.409            prime    replayed       17493      17632    0.84  7.9e-16  bluestein
      1228  2^2.307          prime    raced          15719      25215    1.53  4.7e-16  bluestein
      1229  1229             prime    replayed       16610      21594    1.16  5.6e-16  bluestein
      1230  2.3.5.41         chain3   replayed        4171      16584    3.71  4.4e-16  
      1231  1231             prime    replayed       15961      19372    1.06  5.3e-16  bluestein
      1232  2^4.7.11         chain3   replayed        2172       3316    1.47  4.1e-16  
      1233  3^2.137          prime    replayed       15277      17800    1.08  5.4e-16  bluestein
      1234  2.617            prime    raced          15976      20819    1.29  6.7e-16  bluestein
      1235  5.13.19          flat     replayed        3998      11566    1.56  6.7e-16  flips differ 1.91x
      1236  2^2.3.103        prime    raced          16798      20735    1.02  5.7e-16  bluestein
      1237  1237             prime    replayed       15167      18972    1.22  6.7e-16  bluestein
      1238  2.619            prime    raced          15186      19798    1.24  5.7e-16  bluestein
      1239  3.7.59           prime    replayed       15551      19693    1.05  4.8e-16  flips differ 1.31x bluestein
      1240  2^3.5.31         chain3   replayed        3868      18886    3.27  4.5e-16  flips differ 1.32x
      1241  17.73            prime    replayed       17176      23696    1.03  5.0e-16  flips differ 1.30x bluestein
      1242  2.3^3.23         flat     replayed        4680      13610    2.75  9.2e-16  
      1243  11.113           prime    replayed       17753      22836    0.91  5.9e-16  flips differ 1.32x bluestein
      1244  2^2.311          prime    raced          15697      25771    1.55  6.6e-16  bluestein
      1245  3.5.83           prime    replayed       17340      22678    1.25  4.8e-16  bluestein
      1246  2.7.89           prime    raced          16559      22997    1.15  4.7e-16  bluestein
      1247  29.43            flat     replayed        6857      30677    3.52  2.8e-15  flips differ 1.34x
      1248  2^5.3.13         chain3   replayed        2352       3404    1.30  5.6e-16  
      1249  1249             prime    replayed        8140      20167    2.28  1.0e-15  rader
      1250  2.5^4            chain3   replayed        3121       2893    0.62  5.1e-16  flips differ 1.49x
      1251  3^2.139          prime    replayed       15642      17526    1.09  5.1e-16  bluestein
      1252  2^2.313          prime    raced          15251      23421    1.39  5.6e-16  bluestein
      1253  7.179            prime    replayed       15741      17978    0.73  6.2e-16  flips differ 1.51x bluestein
      1254  2.3.11.19        chain3   replayed        3215      11737    2.62  5.0e-16  flips differ 1.40x
      1255  5.251            prime    replayed       16354      15912    0.43  5.4e-16  flips differ 2.40x bluestein
      1256  2^3.157          prime    raced          15538      14797    0.84  6.2e-16  bluestein
      1257  3.419            prime    replayed       17665      21537    0.99  7.6e-16  bluestein
      1258  2.17.37          flat     replayed        5042      27269    5.15  1.1e-15  
      1259  1259             prime    replayed       18498      17708    0.73  5.6e-16  flips differ 1.38x bluestein
      1260  2^2.3^2.5.7      chain3   replayed        2321       2685    1.08  5.8e-16  
      1261  13.97            prime    replayed       15846      17006    0.94  3.9e-16  bluestein
      1262  2.631            prime    raced          19035      19017    0.89  4.6e-16  bluestein
      1263  3.421            prime    replayed       18180      18412    0.89  6.0e-16  bluestein
      1264  2^4.79           prime    raced          22532      21200    0.87  5.5e-16  bluestein
      1265  5.11.23          flat     replayed        4390      13527    2.32  8.0e-16  flips differ 1.34x
      1266  2.3.211          prime    raced          17318      17467    0.92  7.3e-16  bluestein
      1267  7.181            prime    replayed       15953      13668    0.82  6.0e-16  bluestein
      1268  2^2.317          prime    raced          16089      14321    0.77  4.9e-16  bluestein
      1269  3^3.47           flat     replayed        6275      21512    3.04  6.4e-16  
      1270  2.5.127          prime    raced          16096      15949    0.93  4.6e-16  bluestein
      1271  31.41            flat     replayed        5819      29855    5.02  1.1e-15  
      1272  2^3.3.53         prime    raced          17881      21339    1.14  3.9e-16  bluestein
      1273  19.67            prime    replayed       15822      30279    1.60  6.5e-16  bluestein
      1274  2.7^2.13         flat     replayed        3580       4059    1.09  5.3e-16  
      1275  3.5^2.17         flat     replayed        3735      10985    2.85  6.9e-16  
      1276  2^2.11.29        chain3   replayed        4188      15151    3.17  4.1e-16  
      1277  1277             prime    replayed       14168      21454    1.20  7.0e-16  rader
      1278  2.3^2.71         prime    raced          15359      19401    0.94  4.6e-16  bluestein
      1279  1279             prime    replayed       15677      16500    1.01  5.5e-16  bluestein
      1280  2^8.5            chain3   replayed        2299       2255    0.71  4.3e-16  
      1281  3.7.61           prime    replayed       17433      15810    0.70  3.9e-16  flips differ 1.31x bluestein
      1282  2.641            prime    raced          16494      13768    0.68  4.7e-16  bluestein
      1283  1283             prime    replayed       16682      19470    1.06  6.1e-16  bluestein
      1284  2^2.3.107        prime    raced          17417      18969    0.87  6.3e-16  flips differ 1.25x bluestein
      1285  5.257            prime    replayed       15995      14209    0.77  5.4e-16  bluestein
      1286  2.643            prime    raced          22106      21058    0.64  5.7e-16  flips differ 1.35x bluestein
      1287  3^2.11.13        chain3   replayed        3356       5022    1.09  5.8e-16  flips differ 1.38x
      1288  2^3.7.23         chain3   replayed        3187      12265    3.66  5.0e-16  
      1289  1289             prime    replayed       10974      20832    1.81  6.7e-16  rader
      1290  2.3.5.43         flat     replayed        6791      20385    2.72  7.4e-16  
      1291  1291             prime    replayed       16107      19388    1.04  7.2e-16  bluestein
      1292  2^2.17.19        chain3   replayed        4214      19589    4.54  5.9e-16  
      1293  3.431            prime    replayed       16424      17750    0.84  5.8e-16  flips differ 1.34x bluestein
      1294  2.647            prime    raced          15121      18921    1.15  6.6e-16  bluestein
      1295  5.7.37           flat     replayed        4783      16390    2.83  7.7e-16  
      1296  2^4.3^4          chain3   replayed        2229       2703    1.10  5.4e-16  
      1297  1297             prime    replayed       11259      14446    1.14  9.9e-16  rader
      1298  2.11.59          prime    raced          15496      21963    1.25  6.1e-16  bluestein
      1299  3.433            prime    replayed       15288      14453    0.89  5.9e-16  bluestein
      1300  2^2.5^2.13       flat     replayed        3325       2905    0.86  6.7e-16  
      1301  1301             prime    replayed       13030      18495    1.36  8.1e-16  rader
      1302  2.3.7.31         chain3   replayed        4248      14431    3.23  6.4e-16  
      1303  1303             prime    replayed       15288      20607    1.13  4.5e-16  bluestein
      1304  2^3.163          prime    raced          15537      16303    0.94  5.3e-16  bluestein
      1305  3^2.5.29         chain3   replayed        3853      14242    3.51  7.8e-16  
      1306  2.653            prime    raced          15797      21848    1.26  5.8e-16  bluestein
      1307  1307             prime    replayed       17226      19737    1.00  4.7e-16  bluestein
      1308  2^2.3.109        prime    raced          16902      15762    0.87  4.9e-16  bluestein
      1309  7.11.17          flat     replayed        3704      14075    3.44  9.4e-16  
      1310  2.5.131          prime    raced          15357      19234    1.18  6.7e-16  bluestein
      1311  3.19.23          chain3   replayed        4131      22808    5.17  5.6e-16  
      1312  2^5.41           2p       replayed        3800      25459    5.90  4.6e-16  
      1313  13.101           prime    replayed       16079      16433    0.95  6.1e-16  bluestein
      1314  2.3^2.73         prime    raced          16616      15001    0.86  5.3e-16  bluestein
      1315  5.263            prime    replayed       15718      20546    1.00  6.1e-16  bluestein
      1316  2^2.7.47         chain3   replayed        6089      21970    3.22  4.5e-16  
      1317  3.439            prime    replayed       15312      19183    1.15  5.2e-16  bluestein
      1318  2.659            prime    raced          16833      20570    1.07  5.7e-16  bluestein
      1319  1319             prime    replayed       15280      19358    1.06  7.0e-16  bluestein
      1320  2^3.3.5.11       chain3   replayed        2490       3094    1.22  5.1e-16  
      1321  1321             prime    replayed       10091      22033    1.98  6.9e-16  rader
      1322  2.661            prime    raced          15328      21190    1.26  5.5e-16  bluestein
      1323  3^3.7^2          flat     replayed        3431       4710    1.34  8.3e-16  
      1324  2^2.331          prime    raced          15383      24346    1.57  6.7e-16  bluestein
      1325  5^2.53           prime    replayed       15322      22276    1.40  4.9e-16  bluestein
      1326  2.3.13.17        chain3   replayed        3527      11880    3.14  6.2e-16  
      1327  1327             prime    replayed       11397      19555    1.69  9.3e-16  rader
      1328  2^4.83           prime    raced          16581      21578    1.16  4.6e-16  bluestein
      1329  3.443            prime    replayed       15786      19795    1.10  5.5e-16  bluestein
      1330  2.5.7.19         chain3   replayed        3248      10878    2.56  4.8e-16  flips differ 1.32x
      1331  11^3             chain3   replayed        3440       6512    1.88  4.9e-16  
      1332  2^2.3^2.37       chain3   replayed        3998      18387    4.54  5.5e-16  
      1333  31.43            2p       replayed        6626      33609    4.74  6.3e-16  
      1334  2.23.29          flat     replayed        5626      25309    4.37  1.5e-15  
      1335  3.5.89           prime    replayed       16731      20421    1.10  7.1e-16  bluestein
      1336  2^3.167          prime    raced          15178      19620    1.02  5.0e-16  bluestein
      1337  7.191            prime    replayed       15631      14739    0.90  4.7e-16  bluestein
      1338  2.3.223          prime    raced          15668      19906    1.11  5.7e-16  bluestein
      1339  13.103           prime    replayed       15898      22638    1.40  5.6e-16  bluestein
      1340  2^2.5.67         prime    raced          15721      24244    1.41  6.3e-16  bluestein
      1341  3^2.149          prime    replayed       16164      20042    1.16  5.4e-16  bluestein
      1342  2.11.61          prime    raced          15340      17734    0.94  4.9e-16  bluestein
      1343  17.79            prime    replayed       18961      28097    1.41  4.6e-16  bluestein
      1344  2^6.3.7          2p       replayed        2226       2578    1.11  4.1e-16  
      1345  5.269            prime    replayed       15207      22415    1.21  5.3e-16  flips differ 1.28x bluestein
      1346  2.673            prime    raced          15672      24145    1.32  7.9e-16  bluestein
      1347  3.449            prime    replayed       15935      19437    1.16  5.4e-16  bluestein
      1348  2^2.337          prime    raced          16353      25255    1.49  5.7e-16  bluestein
      1349  19.71            prime    replayed       15742      28498    1.21  7.3e-16  flips differ 1.41x bluestein
      1350  2.3^3.5^2        chain3   replayed        2521       2802    0.96  4.6e-16  
      1351  7.193            prime    replayed       15323      15214    0.72  5.1e-16  flips differ 1.28x bluestein
      1352  2^3.13^2         chain3   replayed        3028       5330    1.65  6.9e-16  
      1353  3.11.41          chain3   replayed        5418      21440    3.33  5.1e-16  
      1354  2.677            prime    raced          16195      24389    1.41  5.9e-16  bluestein
      1355  5.271            prime    replayed       16431      18046    1.06  5.8e-16  bluestein
      1356  2^2.3.113        prime    raced          18316      28731    1.49  6.0e-16  bluestein
      1357  23.59            prime    replayed       16386      30323    1.68  4.5e-16  bluestein
      1358  2.7.97           prime    raced          15337      15800    0.79  4.4e-16  flips differ 1.26x bluestein
      1359  3^2.151          prime    replayed       14970      17629    1.04  5.6e-16  bluestein
      1360  2^4.5.17         chain3   replayed        2728      12396    4.18  5.3e-16  
      1361  1361             prime    replayed       10467      22290    1.92  8.0e-16  rader
      1362  2.3.227          prime    raced          15906      19166    1.12  5.6e-16  bluestein
      1363  29.47            2p       replayed       10456      36024    3.27  4.5e-16  
      1364  2^2.11.31        flat     replayed        5211      18482    2.98  7.3e-16  
      1365  3.5.7.13         chain3   replayed        3478       4232    1.12  6.9e-16  
      1366  2.683            prime    raced          15690      20081    1.03  6.2e-16  bluestein
      1367  1367             prime    replayed       16204      19177    1.15  5.5e-16  bluestein
      1368  2^3.3^2.19       chain3   replayed        3165      11740    3.61  5.9e-16  
      1369  37^2             2p       replayed        7153      34373    3.02  6.8e-16  flips differ 1.52x
      1370  2.5.137          prime    raced          18362      20128    0.84  4.5e-16  bluestein
      1371  3.457            prime    replayed       15910      22005    1.18  5.9e-16  bluestein
      1372  2^2.7^3          flat     replayed        3155       3464    1.08  4.0e-16  
      1373  1373             prime    replayed       15670      20519    1.15  5.5e-16  bluestein
      1374  2.3.229          prime    raced          15758      18603    0.84  4.1e-16  flips differ 1.41x bluestein
      1375  5^3.11           chain3   replayed        2950       5584    1.74  4.2e-16  
      1376  2^5.43           2p       replayed        4092      26237    5.56  4.7e-16  
      1377  3^4.17           chain3   replayed        3347      12248    3.42  6.7e-16  
      1378  2.13.53          prime    raced          25889      26176    0.97  4.6e-16  bluestein
      1379  7.197            prime    replayed       15938      19577    1.16  5.3e-16  bluestein
      1380  2^2.3.5.23       chain3   replayed        3532      13790    3.60  4.4e-16  
      1381  1381             prime    replayed       13801      22156    1.35  8.6e-16  rader
      1382  2.691            prime    raced          16510      23916    1.31  4.9e-16  bluestein
      1383  3.461            prime    replayed       18216      23082    1.21  5.4e-16  bluestein
      1384  2^3.173          prime    raced          17758      22908    1.17  7.5e-16  bluestein
      1385  5.277            prime    replayed       15060      23931    1.32  5.8e-16  bluestein
      1386  2.3^2.7.11       chain3   replayed        3273       4263    1.00  5.1e-16  
      1387  19.73            prime    replayed       15555      26331    1.60  5.0e-16  bluestein
      1388  2^2.347          prime    raced          15492      22100    1.30  5.3e-16  bluestein
      1389  3.463            prime    replayed       15468      19061    1.14  6.2e-16  bluestein
      1390  2.5.139          prime    raced          18100      18617    0.90  5.4e-16  bluestein
      1391  13.107           prime    replayed       16501      23686    1.14  5.2e-16  bluestein
      1392  2^4.3.29         flat     replayed        6492      22709    3.18  5.7e-16  
      1393  7.199            prime    replayed       20100      18259    0.91  6.3e-16  bluestein
      1394  2.17.41          flat     replayed        6509      33744    4.04  1.4e-15  
      1395  3^2.5.31         flat     replayed        5583      16127    2.38  5.2e-16  
      1396  2^2.349          prime    raced          16119      19247    1.14  6.1e-16  bluestein
      1397  11.127           prime    replayed       15781      20203    1.24  5.2e-16  bluestein
      1398  2.3.233          prime    raced          16081      18757    1.07  5.4e-16  bluestein
      1399  1399             prime    replayed       16240      19634    0.74  5.8e-16  flips differ 1.63x bluestein
      1400  2^3.5^2.7        chain3   replayed        3048       2696    0.86  4.1e-16  
      1401  3.467            prime    replayed       19196      21052    0.97  5.9e-16  bluestein
      1402  2.701            prime    raced          15336      20195    1.22  7.5e-16  bluestein
      1403  23.61            prime    replayed       15297      26948    1.71  5.2e-16  bluestein
      1404  2^2.3^3.13       chain3   replayed        3407       4973    1.35  5.6e-16  
      1405  5.281            prime    replayed       16557      20442    1.04  4.9e-16  bluestein
      1406  2.19.37          flat     replayed        6102      31890    3.42  1.5e-15  flips differ 1.52x
      1407  3.7.67           prime    replayed       16485      27744    1.40  5.6e-16  bluestein
      1408  2^7.11           chain3   replayed        2610       3399    1.26  4.0e-16  
      1409  1409             prime    replayed        8674      18838    2.11  6.0e-16  rader
      1410  2.3.5.47         flat     replayed        6762      23522    3.03  5.6e-16  
      1411  17.83            prime    replayed       15815      32901    1.77  6.5e-16  bluestein
      1412  2^2.353          prime    raced          16443      21704    1.16  5.5e-16  bluestein
      1413  3^2.157          prime    replayed       16432      17140    0.95  7.2e-16  bluestein
      1414  2.7.101          prime    raced          16484      17065    0.97  5.2e-16  bluestein
      1415  5.283            prime    replayed       16071      20566    1.20  6.8e-16  bluestein
      1416  2^3.3.59         prime    raced          17293      22532    0.99  6.1e-16  bluestein
      1417  13.109           prime    replayed       15441      18774    1.12  5.9e-16  bluestein
      1418  2.709            prime    raced          16511      31357    1.51  5.7e-16  flips differ 1.37x bluestein
      1419  3.11.43          flat     replayed        6376      24503    3.69  6.0e-16  
      1420  2^2.5.71         prime    raced          16819      22577    1.18  5.5e-16  bluestein
      1421  7^2.29           chain3   replayed        4919      16126    2.58  5.2e-16  flips differ 1.27x
      1422  2.3^2.79         prime    raced          16335      20838    1.25  5.1e-16  bluestein
      1423  1423             prime    replayed       15838      19989    1.06  5.3e-16  bluestein
      1424  2^4.89           prime    raced          15272      20255    1.30  5.1e-16  bluestein
      1425  3.5^2.19         flat     replayed        4364      12447    2.32  5.9e-16  
      1426  2.23.31          flat     replayed        6430      27547    4.17  1.5e-15  
      1427  1427             prime    replayed       15549      18564    0.96  5.4e-16  bluestein
      1428  2^2.3.7.17       chain3   replayed        3044      11110    3.62  7.2e-16  
      1429  1429             prime    replayed       14037      19331    1.34  1.1e-15  rader
      1430  2.5.11.13        chain3   replayed        3159       5068    1.60  5.6e-16  
      1431  3^3.53           prime    replayed       15419      23583    1.52  3.8e-16  bluestein
      1432  2^3.179          prime    raced          16371      18741    1.02  4.5e-16  bluestein
      1433  1433             prime    replayed       15278      18550    1.21  5.4e-16  bluestein
      1434  2.3.239          prime    raced          15246      17363    1.11  5.4e-16  bluestein
      1435  5.7.41           chain3   replayed        5644      20021    3.53  5.2e-16  
      1436  2^2.359          prime    raced          15025      18454    1.17  6.9e-16  bluestein
      1437  3.479            prime    replayed       15385      18278    1.18  5.4e-16  bluestein
      1438  2.719            prime    raced          15444      19126    1.23  6.8e-16  bluestein
      1439  1439             prime    replayed       15307      18703    1.21  5.2e-16  bluestein
      1440  2^5.3^2.5        chain3   replayed        2302       2498    1.07  4.4e-16  
      1441  11.131           prime    replayed       14972      22837    1.49  5.4e-16  bluestein
      1442  2.7.103          prime    raced          15011      21745    1.44  5.6e-16  bluestein
      1443  3.13.37          flat     replayed        5646      19256    3.31  6.7e-16  
      1444  2^2.19^2         chain3   replayed        4777      22511    4.68  3.6e-16  
      1445  5.17^2           chain3   replayed        4024      20578    4.98  6.1e-16  
      1446  2.3.241          prime    raced          15031      16132    1.05  5.2e-16  bluestein
      1447  1447             prime    replayed       15079      23262    1.51  7.5e-16  bluestein
      1448  2^3.181          prime    raced          15084      15658    0.97  6.0e-16  bluestein
      1449  3^2.7.23         flat     replayed        5027      14620    2.83  5.4e-16  
      1450  2.5^2.29         chain3   replayed        4026      14927    3.68  4.3e-16  
      1451  1451             prime    replayed       15107      23482    1.48  5.0e-16  bluestein
      1452  2^2.3.11^2       chain3   replayed        2994       5555    1.84  4.8e-16  
      1453  1453             prime    replayed       10148      23920    2.28  8.2e-16  rader
      1454  2.727            prime    raced          15243      25032    1.63  6.2e-16  bluestein
      1455  3.5.97           prime    replayed       15056      16523    1.00  6.0e-16  bluestein
      1456  2^4.7.13         chain3   replayed        2636       3581    1.34  7.7e-16  
      1457  31.47            2p       replayed        9420      34791    2.98  5.7e-16  
      1458  2.3^6            chain3   replayed        3294       3950    1.18  5.5e-16  
      1459  1459             prime    replayed       13669      18456    1.30  1.1e-15  rader
      1460  2^2.5.73         prime    raced          16125      15852    0.95  5.1e-16  bluestein
      1461  3.487            prime    replayed       15413      17702    1.14  5.9e-16  bluestein
      1462  2.17.43          flat     replayed        6506      31921    4.86  1.0e-15  
      1463  7.11.19          chain3   replayed        3820      15539    3.97  3.7e-16  
      1464  2^3.3.61         prime    raced          15058      16072    0.99  5.6e-16  bluestein
      1465  5.293            prime    replayed       14754      19669    1.30  5.7e-16  bluestein
      1466  2.733            prime    raced          15278      22435    1.31  7.1e-16  bluestein
      1467  3^2.163          prime    replayed       15127      20203    1.31  7.2e-16  bluestein
      1468  2^2.367          prime    raced          15235      20957    1.31  6.1e-16  bluestein
      1469  13.113           prime    replayed       15443      24627    1.35  5.9e-16  bluestein
      1470  2.3.5.7^2        chain3   replayed        2796       3041    1.07  4.6e-16  
      1471  1471             prime    replayed       12266      20444    1.62  9.4e-16  rader
      1472  2^6.23           2p       replayed        4049      12914    3.09  4.5e-16  
      1473  3.491            prime    replayed       15229      21050    1.34  5.1e-16  bluestein
      1474  2.11.67          prime    raced          15337      28578    1.81  6.4e-16  bluestein
      1475  5^2.59           prime    replayed       15414      20790    1.14  5.6e-16  bluestein
      1476  2^2.3^2.41       chain3   replayed        5351      19759    3.25  4.7e-16  
      1477  7.211            prime    replayed       15648      19999    1.18  5.8e-16  bluestein
      1478  2.739            prime    raced          15263      21396    1.38  5.5e-16  bluestein
      1479  3.17.29          flat     replayed        5614      25250    4.40  1.1e-15  
      1480  2^3.5.37         flat     replayed        5013      18484    3.60  6.6e-16  
      1481  1481             prime    replayed       15539      20526    1.27  5.7e-16  bluestein
      1482  2.3.13.19        chain3   replayed        4273      13609    3.16  6.4e-16  
      1483  1483             prime    replayed       13020      20477    1.50  9.1e-16  rader
      1484  2^2.7.53         prime    raced          15758      24322    1.50  5.0e-16  bluestein
      1485  3^3.5.11         chain3   replayed        3600       4658    1.07  4.8e-16  
      1486  2.743            prime    raced          16400      21490    1.29  5.1e-16  bluestein
      1487  1487             prime    replayed       15815      20700    1.28  5.2e-16  bluestein
      1488  2^4.3.31         chain3   replayed        4025      16318    4.01  4.9e-16  
      1489  1489             prime    replayed       16028      20882    1.28  5.7e-16  bluestein
      1490  2.5.149          prime    raced          15359      19941    1.16  6.0e-16  bluestein
      1491  3.7.71           prime    replayed       17292      21612    1.00  4.8e-16  flips differ 1.27x bluestein
      1492  2^2.373          prime    raced          15851      23769    1.38  5.4e-16  bluestein
      1493  1493             prime    replayed       18323      22971    1.19  5.9e-16  bluestein
      1494  2.3^2.83         prime    raced          16765      23393    1.12  5.6e-16  bluestein
      1495  5.13.23          chain3   replayed        4545      16113    3.49  6.1e-16  
      1496  2^3.11.17        chain3   replayed        3620      13283    3.61  4.5e-16  
      1497  3.499            prime    replayed       15437      20822    1.33  6.5e-16  bluestein
      1498  2.7.107          prime    raced          15071      21902    1.31  6.6e-16  bluestein
      1499  1499             prime    replayed       15542      21491    1.30  5.9e-16  bluestein
      1500  2^2.3.5^3        chain3   replayed        2818       3106    1.06  5.3e-16  
      1501  19.79            prime    replayed       17051      31820    1.83  5.4e-16  bluestein
      1502  2.751            prime    raced          15837      18680    1.13  4.9e-16  bluestein
      1503  3^2.167          prime    replayed       16335      23839    1.22  6.2e-16  bluestein
      1504  2^5.47           2p       replayed        4805      24747    5.11  3.7e-16  
      1505  5.7.43           chain3   replayed        8334      23093    2.65  4.5e-16  
      1506  2.3.251          prime    raced          16030      17636    1.08  5.0e-16  bluestein
      1507  11.137           prime    replayed       18633      22509    0.71  6.1e-16  flips differ 1.70x bluestein
      1508  2^2.13.29        flat     replayed        5409      18023    3.23  1.1e-15  
      1509  3.503            prime    replayed       15698      19292    1.16  6.1e-16  bluestein
      1510  2.5.151          prime    raced          16641      16586    0.94  4.4e-16  bluestein
      1511  1511             prime    replayed       17043      20491    1.16  7.3e-16  bluestein
      1512  2^3.3^3.7        chain3   replayed        2992       3135    1.03  5.4e-16  
      1513  17.89            prime    replayed       16925      30089    1.65  5.1e-16  bluestein
      1514  2.757            prime    raced          16800      19467    1.11  5.3e-16  bluestein
      1515  3.5.101          prime    replayed       16035      18341    0.95  5.9e-16  flips differ 1.32x bluestein
      1516  2^2.379          prime    raced          16942      18077    1.03  6.9e-16  bluestein
      1517  37.41            2p       replayed        8232      37529    4.38  5.9e-16  
      1518  2.3.11.23        chain3   replayed        4387      16636    3.54  4.8e-16  
      1519  7^2.31           flat     replayed        5389      16755    2.98  7.1e-16  
      1520  2^4.5.19         chain3   replayed        3224      14319    4.18  4.6e-16  
      1521  3^2.13^2         chain3   replayed        3773       5904    1.51  6.7e-16  
      1522  2.761            prime    raced          15856      20107    1.22  4.7e-16  bluestein
      1523  1523             prime    replayed       15720      20778    1.28  6.1e-16  bluestein
      1524  2^2.3.127        prime    raced          15911      20491    1.08  6.3e-16  bluestein
      1525  5^2.61           prime    replayed       15568      18364    1.17  4.9e-16  bluestein
      1526  2.7.109          prime    raced          15566      19423    1.08  5.6e-16  bluestein
      1527  3.509            prime    replayed       15444      18299    1.01  5.9e-16  bluestein
      1528  2^3.191          prime    raced          15792      16651    0.89  3.9e-16  bluestein
      1529  11.139           prime    replayed       15422      26431    1.68  4.6e-16  bluestein
      1530  2.3^2.5.17       chain3   replayed        3456      12741    3.62  5.0e-16  
      1531  1531             prime    replayed       11529      20138    1.68  7.7e-16  rader
      1532  2^2.383          prime    raced          15453      18241    1.11  4.9e-16  bluestein
      1533  3.7.73           prime    replayed       18115      18397    0.98  5.0e-16  bluestein
      1534  2.13.59          prime    raced          15384      22998    1.43  5.7e-16  bluestein
      1535  5.307            prime    replayed       15760      29043    1.58  6.9e-16  bluestein
      1536  2^9.3            chain3   replayed        2312       2280    0.98  3.4e-16  
      1537  29.53            prime    replayed       16931      42236    2.33  5.4e-16  bluestein
      1538  2.769            prime    raced          17121      17056    0.95  5.9e-16  bluestein
      1539  3^4.19           chain3   replayed        3827      15154    3.88  5.8e-16  
      1540  2^2.5.7.11       flat     replayed        3823       4481    0.98  4.6e-16  
      1541  23.67            prime    replayed       16140      44501    2.08  5.7e-16  bluestein
      1542  2.3.257          prime    raced          18719      17837    0.75  5.6e-16  flips differ 1.27x bluestein
      1543  1543             prime    replayed       16688      37387    1.81  4.3e-16  flips differ 1.27x bluestein
      1544  2^3.193          prime    raced          16647      17426    0.83  5.4e-16  bluestein
      1545  3.5.103          prime    replayed       15927      24191    1.28  5.4e-16  bluestein
      1546  2.773            prime    raced          17769      21645    0.94  6.0e-16  flips differ 1.28x bluestein
      1547  7.13.17          flat     replayed        4602      16713    3.01  7.6e-16  flips differ 1.27x
      1548  2^2.3^2.43       chain3   replayed        5293      24252    3.97  5.4e-16  
      1549  1549             prime    replayed       16092      33814    1.98  5.3e-16  bluestein
      1550  2.5^2.31         chain3   replayed        4433      19407    4.01  6.0e-16  
      1551  3.11.47          chain3   replayed        7701      28521    3.02  4.6e-16  
      1552  2^4.97           prime    raced          18339      19037    0.86  5.2e-16  bluestein
      1553  1553             prime    replayed       15335      36529    2.07  5.1e-16  bluestein
      1554  2.3.7.37         flat     replayed        7265      23290    2.38  7.8e-16  
      1555  5.311            prime    replayed       15926      29276    1.60  8.9e-16  bluestein
      1556  2^2.389          prime    raced          15730      19856    1.22  5.4e-16  bluestein
      1557  3^2.173          prime    replayed       15697      24300    1.52  5.0e-16  bluestein
      1558  2.19.41          flat     replayed        6819      31886    4.66  1.4e-15  
      1559  1559             prime    replayed       15682      33517    2.05  6.2e-16  bluestein
      1560  2^3.3.5.13       chain3   replayed        2985       3604    1.20  5.0e-16  
      1561  7.223            prime    replayed       15450      23374    1.49  5.0e-16  bluestein
      1562  2.11.71          prime    raced          15524      22929    1.41  7.0e-16  bluestein
      1563  3.521            prime    replayed       15396      23706    1.48  6.2e-16  bluestein
      1564  2^2.17.23        chain3   replayed        5099      25475    4.79  5.4e-16  
      1565  5.313            prime    replayed       15352      28632    1.83  5.0e-16  bluestein
      1566  2.3^3.29         chain3   replayed        5031      16725    2.94  4.5e-16  
      1567  1567             prime    replayed       14971      20948    1.39  6.3e-16  bluestein
      1568  2^5.7^2          chain3   replayed        2495       3009    1.18  5.7e-16  
      1569  3.523            prime    replayed       15093      23124    1.29  5.9e-16  bluestein
      1570  2.5.157          prime    raced          15490      18612    1.12  5.2e-16  bluestein
      1571  1571             prime    replayed       15951      20741    1.25  5.8e-16  bluestein
      1572  2^2.3.131        prime    raced          15565      25065    1.42  6.7e-16  bluestein
      1573  11^2.13          chain3   replayed        4086       8074    1.81  6.6e-16  
      1574  2.787            prime    raced          15726      22458    1.29  5.6e-16  bluestein
      1575  3^2.5^2.7        flat     replayed        3976       4575    1.02  4.6e-16  
      1576  2^3.197          prime    raced          16145      21812    1.22  4.9e-16  bluestein
      1577  19.83            prime    replayed       15985      34080    1.93  5.3e-16  bluestein
      1578  2.3.263          prime    raced          15848      24408    1.36  5.9e-16  bluestein
      1579  1579             prime    replayed       15542      21090    1.13  4.6e-16  bluestein
      1580  2^2.5.79         prime    raced          15458      22288    1.11  4.5e-16  flips differ 1.26x bluestein
      1581  3.17.31          chain3   replayed        6043      29732    3.76  4.9e-16  flips differ 1.28x
      1582  2.7.113          prime    raced          15349      26223    1.64  5.0e-16  bluestein
      1583  1583             prime    replayed       15556      20948    1.14  5.5e-16  bluestein
      1584  2^4.3^2.11       chain3   replayed        2871       4065    1.41  4.5e-16  
      1585  5.317            prime    replayed       15351      18805    0.95  5.0e-16  bluestein
      1586  2.13.61          prime    raced          16188      19917    1.10  5.4e-16  bluestein
      1587  3.23^2           flat     replayed        6035      27897    4.61  1.4e-15  
      1588  2^2.397          prime    raced          15559      20719    1.11  6.3e-16  bluestein
      1589  7.227            prime    replayed       18940      20081    1.00  4.9e-16  bluestein
      1590  2.3.5.53         prime    raced          15495      26800    1.71  4.5e-16  bluestein
      1591  37.43            2p       replayed       10176      38833    3.81  5.8e-16  
      1592  2^3.199          prime    raced          16798      21364    0.97  5.8e-16  bluestein
      1593  3^3.59           prime    replayed       16886      23692    1.06  6.8e-16  bluestein
      1594  2.797            prime    raced          16978      26122    1.42  7.0e-16  bluestein
      1595  5.11.29          flat     replayed        5527      19931    3.44  5.8e-16  
      1596  2^2.3.7.19       chain3   replayed        3660      13424    3.64  5.2e-16  
      1597  1597             prime    replayed       15584      20584    1.27  7.1e-16  bluestein
      1598  2.17.47          flat     replayed        7200      36315    4.81  1.6e-15  
      1599  3.13.41          chain3   replayed        6021      22886    3.72  5.5e-16  
      1600  2^6.5^2          chain3   replayed        2542       2959    1.09  3.4e-16  
      1601  1601             prime    replayed        9632      17926    1.78  5.8e-16  rader
      1602  2.3^2.89         prime    raced          15423      21335    1.34  4.6e-16  bluestein
      1603  7.229            prime    replayed       15448      20372    1.17  4.6e-16  bluestein
      1604  2^2.401          prime    raced          15666      16409    1.03  4.0e-16  bluestein
      1605  3.5.107          prime    replayed       15307      23481    1.51  5.3e-16  bluestein
      1606  2.11.73          prime    raced          14869      20962    1.35  5.6e-16  bluestein
      1607  1607             prime    replayed       15567      24107    1.32  6.1e-16  bluestein
      1608  2^3.3.67         prime    raced          15322      28236    1.78  5.6e-16  bluestein
      1609  1609             prime    replayed       15479      26758    1.33  6.2e-16  bluestein
      1610  2.5.7.23         chain3   replayed        4019      14353    3.57  4.7e-16  
      1611  3^2.179          prime    replayed       14919      22739    1.46  6.2e-16  bluestein
      1612  2^2.13.31        flat     replayed        6062      19207    3.01  7.9e-16  
      1613  1613             prime    replayed       15299      24043    1.52  4.7e-16  bluestein
      1614  2.3.269          prime    raced          15444      22439    1.26  5.6e-16  bluestein
      1615  5.17.19          chain3   replayed        5100      23055    4.09  5.4e-16  
      1616  2^4.101          prime    raced          15528      16961    1.05  4.9e-16  bluestein
      1617  3.7^2.11         chain3   replayed        3865       5558    1.40  4.6e-16  
      1618  2.809            prime    raced          15409      23494    1.51  5.6e-16  bluestein
      1619  1619             prime    replayed       15529      24732    1.47  7.9e-16  bluestein
      1620  2^2.3^4.5        chain3   replayed        3001       3287    1.05  6.0e-16  
      1621  1621             prime    replayed       15600      18489    1.15  5.5e-16  bluestein
      1622  2.811            prime    raced          15355      19892    1.27  5.5e-16  bluestein
      1623  3.541            prime    replayed       15632      19415    1.20  4.6e-16  bluestein
      1624  2^3.7.29         flat     replayed        5084      20228    3.88  6.6e-16  
      1625  5^3.13           flat     replayed        4440       7103    1.58  6.6e-16  
      1626  2.3.271          prime    raced          15042      18695    1.04  5.5e-16  bluestein
      1627  1627             prime    replayed       15000      31506    2.09  6.8e-16  bluestein
      1628  2^2.11.37        chain3   replayed        6518      22329    3.29  4.6e-16  
      1629  3^2.181          prime    replayed       14959      17682    1.17  5.0e-16  bluestein
      1630  2.5.163          prime    raced          15716      19425    1.22  6.8e-16  bluestein
      1631  7.233            prime    replayed       14873      21497    1.36  6.6e-16  bluestein
      1632  2^5.3.17         chain3   replayed        3391      12201    3.54  4.5e-16  
      1633  23.71            prime    replayed       15463      34016    2.15  5.1e-16  bluestein
      1634  2.19.43          flat     replayed        7473      36396    4.48  8.3e-16  
      1635  3.5.109          prime    replayed       15395      19265    1.25  5.5e-16  bluestein
      1636  2^2.409          prime    raced          15437      22398    1.43  7.0e-16  bluestein
      1637  1637             prime    replayed       15073      31702    2.09  5.4e-16  bluestein
      1638  2.3^2.7.13       chain3   replayed        4021       4620    1.06  5.5e-16  
      1639  11.149           prime    replayed       15380      24480    1.57  5.8e-16  bluestein
      1640  2^3.5.41         flat     replayed        5710      26242    4.47  8.1e-16  
      1641  3.547            prime    replayed       15353      31210    1.99  6.6e-16  bluestein
      1642  2.821            prime    raced          15384      24276    1.00  6.9e-16  flips differ 1.51x bluestein
      1643  31.53            prime    replayed       15709      43392    2.67  6.0e-16  bluestein
      1644  2^2.3.137        prime    raced          15487      22509    1.22  6.0e-16  bluestein
      1645  5.7.47           flat     replayed        7469      25664    3.29  5.6e-16  
      1646  2.823            prime    raced          15562      22971    1.29  6.3e-16  bluestein
      1647  3^3.61           prime    replayed       15362      20214    1.31  4.7e-16  bluestein
      1648  2^4.103          prime    raced          15509      24712    1.39  6.3e-16  bluestein
      1649  17.97            prime    replayed       15796      27264    1.19  4.6e-16  flips differ 1.44x bluestein
      1650  2.3.5^2.11       chain3   replayed        3335       4069    1.19  3.9e-16  
      1651  13.127           prime    replayed       15443      21843    1.36  4.2e-16  bluestein
      1652  2^2.7.59         prime    raced          15361      25145    1.62  4.9e-16  bluestein
      1653  3.19.29          chain3   replayed        5735      30472    5.29  4.6e-16  
      1654  2.827            prime    raced          15662      24324    1.47  6.6e-16  bluestein
      1655  5.331            prime    replayed       15534      28887    1.81  5.6e-16  bluestein
      1656  2^3.3^2.23       chain3   replayed        4159      14781    3.53  5.3e-16  
      1657  1657             prime    replayed       15646      31803    1.98  6.2e-16  bluestein
      1658  2.829            prime    raced          15526      23965    1.51  6.4e-16  bluestein
      1659  3.7.79           prime    replayed       15458      24176    1.52  6.3e-16  bluestein
      1660  2^2.5.83         prime    raced          15525      25255    1.54  6.1e-16  bluestein
      1661  11.151           prime    replayed       15832      20061    1.26  5.0e-16  bluestein
      1662  2.3.277          prime    raced          15515      24342    1.50  5.9e-16  bluestein
      1663  1663             prime    replayed       15476      32105    2.06  5.1e-16  bluestein
      1664  2^7.13           chain3   replayed        2919       3943    1.23  4.6e-16  
      1665  3^2.5.37         chain3   replayed        6701      20840    3.08  4.8e-16  
      1666  2.7^2.17         flat     replayed        4988      13639    2.65  7.7e-16  
      1667  1667             prime    replayed       15316      32256    2.00  5.7e-16  bluestein
      1668  2^2.3.139        prime    raced          15334      22486    1.37  5.2e-16  bluestein
      1669  1669             prime    replayed       16045      32000    1.99  6.4e-16  bluestein
      1670  2.5.167          prime    raced          15718      23380    1.47  6.0e-16  bluestein
      1671  3.557            prime    replayed       15097      30691    1.95  6.0e-16  bluestein
      1672  2^3.11.19        chain3   replayed        4229      15166    3.50  5.0e-16  
      1673  7.239            prime    replayed       15878      20443    1.10  6.3e-16  bluestein
      1674  2.3^3.31         flat     replayed        6095      18813    2.90  8.3e-16  
      1675  5^2.67           prime    replayed       15361      30027    1.89  5.5e-16  bluestein
      1676  2^2.419          prime    raced          15364      22184    1.44  7.2e-16  bluestein
      1677  3.13.43          chain3   replayed        6424      27275    4.20  7.0e-16  
      1678  2.839            prime    raced          15348      23802    1.49  6.2e-16  bluestein
      1679  23.73            prime    replayed       15063      31876    2.05  6.1e-16  bluestein
      1680  2^4.3.5.7        chain3   replayed        2845       3155    1.07  4.3e-16  
      1681  41^2             2p       replayed        9409      43311    3.99  4.3e-16  
      1682  2.29^2           flat     replayed        7385      33216    4.43  1.7e-15  
      1683  3^2.11.17        chain3   replayed        4358      19658    3.40  7.2e-16  flips differ 1.43x
      1684  2^2.421          prime    raced          21422      32066    1.25  7.5e-16  bluestein
      1685  5.337            prime    replayed       15505      35943    1.90  6.5e-16  flips differ 1.44x bluestein
      1686  2.3.281          prime    raced          15722      22377    1.21  6.5e-16  bluestein
      1687  7.241            prime    replayed       15460      17748    1.14  6.2e-16  bluestein
      1688  2^3.211          prime    raced          15960      23932    1.32  6.0e-16  bluestein
      1689  3.563            prime    replayed       15394      30985    1.69  5.3e-16  bluestein
      1690  2.5.13^2         chain3   replayed        3855       6278    1.53  6.3e-16  
      1691  19.89            prime    replayed       15727      34581    2.16  5.7e-16  bluestein
      1692  2^2.3^2.47       chain3   replayed        5750      25030    4.33  5.0e-16  
      1693  1693             prime    replayed       15387      23804    1.52  6.1e-16  bluestein
      1694  2.7.11^2         flat     replayed        4853       6955    1.36  9.5e-16  
      1695  3.5.113          prime    replayed       15723      27289    1.51  7.3e-16  bluestein
      1696  2^5.53           prime    raced          15708      33825    1.75  4.4e-16  bluestein
      1697  1697             prime    replayed       15337      23318    1.49  5.5e-16  bluestein
      1698  2.3.283          prime    raced          15372      22315    1.05  5.6e-16  flips differ 1.36x bluestein
      1699  1699             prime    replayed       14951      23837    1.22  6.3e-16  flips differ 1.33x bluestein
      1700  2^2.5^2.17       chain3   replayed        3811      13413    3.48  5.5e-16  
      1701  3^5.7            chain3   replayed        3473       5802    1.63  4.7e-16  
      1702  2.23.37          flat     replayed        7329      34208    4.64  1.3e-15  
      1703  13.131           prime    replayed       14956      27592    1.73  5.7e-16  bluestein
      1704  2^3.3.71         prime    raced          15056      21390    1.42  5.0e-16  bluestein
      1705  5.11.31          chain3   replayed        7116      20683    2.75  5.4e-16  
      1706  2.853            prime    raced          15138      23190    1.50  7.0e-16  bluestein
      1707  3.569            prime    replayed       15062      24033    1.45  6.2e-16  bluestein
      1708  2^2.7.61         prime    raced          15162      18793    1.24  4.9e-16  bluestein
      1709  1709             prime    replayed       15736      23710    1.50  6.0e-16  bluestein
      1710  2.3^2.5.19       chain3   replayed        4232      14615    3.26  5.1e-16  
      1711  29.59            prime    replayed       18230      38243    2.02  6.8e-16  bluestein
      1712  2^4.107          prime    raced          14880      24211    1.58  6.8e-16  bluestein
      1713  3.571            prime    replayed       15070      22717    1.50  5.4e-16  bluestein
      1714  2.857            prime    raced          14982      23586    1.50  6.2e-16  bluestein
      1715  5.7^3            flat     replayed        4427       6452    1.43  4.9e-16  
      1716  2^2.3.11.13      chain3   replayed        3672       6516    1.75  5.8e-16  
      1717  17.101           prime    replayed       15709      28685    1.79  5.4e-16  bluestein
      1718  2.859            prime    raced          15647      23761    1.50  5.2e-16  bluestein
      1719  3^2.191          prime    replayed       15424      19294    1.06  5.2e-16  bluestein
      1720  2^3.5.43         chain3   replayed        5757      24904    4.30  4.6e-16  
      1721  1721             prime    replayed       15106      25517    1.51  5.9e-16  bluestein
      1722  2.3.7.41         chain3   replayed        5984      24323    4.04  4.7e-16  
      1723  1723             prime    replayed       15545      24545    1.51  5.5e-16  bluestein
      1724  2^2.431          prime    raced          21881      31972    1.43  9.0e-16  bluestein
      1725  3.5^2.23         chain3   replayed        6654      23424    3.37  5.5e-16  
      1726  2.863            prime    raced          15722      23225    1.47  6.3e-16  bluestein
      1727  11.157           prime    replayed       15426      28335    1.75  6.5e-16  bluestein
      1728  2^6.3^3          chain3   replayed        2897       3119    1.05  6.4e-16  
      1729  7.13.19          chain3   replayed        4665      16452    3.48  6.0e-16  
      1730  2.5.173          prime    raced          15069      23893    1.55  5.6e-16  bluestein
      1731  3.577            prime    replayed       15421      19857    1.17  5.7e-16  bluestein
      1732  2^2.433          prime    raced          15731      19526    1.22  6.1e-16  bluestein
      1733  1733             prime    replayed       15459      26043    1.52  6.6e-16  bluestein
      1734  2.3.17^2         chain3   replayed        4774      32452    5.03  4.2e-16  flips differ 1.44x
      1735  5.347            prime    replayed       22266      33284    1.31  5.6e-16  bluestein
      1736  2^3.7.31         flat     replayed        8641      26426    2.52  6.9e-16  
      1737  3^2.193          prime    replayed       15004      18293    0.86  4.7e-16  flips differ 1.42x bluestein
      1738  2.11.79          prime    raced          15455      26238    1.69  5.6e-16  bluestein
      1739  37.47            flat     replayed        8947      44935    4.88  1.8e-15  
      1740  2^2.3.5.29       flat     replayed        6292      18398    2.55  5.9e-16  
      1741  1741             prime    replayed       16512      25632    1.43  5.0e-16  bluestein
      1742  2.13.67          prime    raced          16311      34015    1.75  5.7e-16  bluestein
      1743  3.7.83           prime    replayed       18494      25890    1.27  4.8e-16  bluestein
      1744  2^4.109          prime    raced          18563      21227    0.43  6.3e-16  flips differ 2.64x bluestein
      1745  5.349            prime    replayed       15941      24270    1.17  6.0e-16  flips differ 1.28x bluestein
      1746  2.3^2.97         prime    raced          17065      20004    0.66  4.4e-16  flips differ 1.74x bluestein
      1747  1747             prime    replayed       16061      27120    1.38  5.1e-16  bluestein
      1748  2^2.19.23        chain3   replayed        6933      36018    4.61  5.0e-16  
      1749  3.11.53          prime    replayed       17446      32206    1.76  4.6e-16  bluestein
      1750  2.5^3.7          chain3   replayed        3422       4660    1.25  4.9e-16  
      1751  17.103           prime    replayed       17720      40046    2.03  6.8e-16  bluestein
      1752  2^3.3.73         prime    raced          15918      20269    1.08  6.7e-16  bluestein
      1753  1753             prime    replayed       15682      23629    1.41  5.2e-16  bluestein
      1754  2.877            prime    raced          18551      24737    1.31  5.3e-16  bluestein
      1755  3^3.5.13         chain3   replayed        4380       5908    1.29  6.7e-16  
      1756  2^2.439          prime    raced          15800      24297    1.53  5.5e-16  bluestein
      1757  7.251            prime    replayed       15432      20817    1.28  5.3e-16  bluestein
      1758  2.3.293          prime    raced          15431      24570    1.49  6.1e-16  bluestein
      1759  1759             prime    replayed       15404      23923    1.43  5.6e-16  bluestein
      1760  2^5.5.11         chain3   replayed        3099       4052    1.30  3.9e-16  
      1761  3.587            prime    replayed       15326      24092    1.56  5.9e-16  bluestein
      1762  2.881            prime    raced          15870      24721    1.51  6.4e-16  bluestein
      1763  41.43            2p       replayed       10080      45304    4.46  6.4e-16  
      1764  2^2.3^2.7^2      chain3   replayed        3555       4342    1.20  6.0e-16  
      1765  5.353            prime    replayed       15558      23185    1.18  5.4e-16  flips differ 1.26x bluestein
      1766  2.883            prime    raced          15968      24797    1.54  6.0e-16  bluestein
      1767  3.19.31          chain3   replayed        6234      33275    5.31  4.2e-16  
      1768  2^3.13.17        chain3   replayed        4483      15240    3.08  6.8e-16  
      1769  29.61            prime    replayed       16585      35428    2.10  6.4e-16  bluestein
      1770  2.3.5.59         prime    raced          16627      23004    1.38  4.5e-16  bluestein
      1771  7.11.23          flat     replayed        5836      20449    3.47  7.4e-16  
      1772  2^2.443          prime    raced          16656      25517    1.22  5.9e-16  bluestein
      1773  3^2.197          prime    replayed       18442      21425    0.96  4.9e-16  bluestein
      1774  2.887            prime    raced          15917      25440    1.25  5.5e-16  bluestein
      1775  5^2.71           prime    replayed       18220      24410    1.24  6.2e-16  bluestein
      1776  2^4.3.37         chain3   replayed        5090      22467    4.38  5.9e-16  
      1777  1777             prime    replayed       15885      24188    1.15  6.9e-16  flips differ 1.35x bluestein
      1778  2.7.127          prime    raced          16542      21593    1.00  5.2e-16  flips differ 1.31x bluestein
      1779  3.593            prime    replayed       18056      24349    1.33  4.7e-16  bluestein
      1780  2^2.5.89         prime    raced          16104      23461    1.43  5.7e-16  bluestein
      1781  13.137           prime    replayed       18117      25576    1.39  6.1e-16  bluestein
      1782  2.3^4.11         chain3   replayed        4368       5142    1.15  5.7e-16  
      1783  1783             prime    replayed       18653      23566    1.25  5.8e-16  bluestein
      1784  2^3.223          prime    raced          18451      24919    1.33  6.0e-16  bluestein
      1785  3.5.7.17         chain3   replayed        4594      15198    2.85  6.5e-16  
      1786  2.19.47          flat     replayed        8146      39134    4.76  1.5e-15  
      1787  1787             prime    replayed       15170      24145    1.55  6.2e-16  bluestein
      1788  2^2.3.149        prime    raced          15243      22578    1.47  6.3e-16  bluestein
      1789  1789             prime    replayed       18218      24607    1.28  6.6e-16  bluestein
      1790  2.5.179          prime    raced          15634      23586    1.46  5.8e-16  bluestein
      1791  3^2.199          prime    replayed       15229      23805    1.55  6.0e-16  bluestein
      1792  2^8.7            chain3   replayed        2858       3202    1.11  3.3e-16  
      1793  11.163           prime    replayed       16017      23591    1.47  4.2e-16  bluestein
      1794  2.3.13.23        chain3   replayed        5255      18806    3.57  6.5e-16  
      1795  5.359            prime    replayed       18463      23564    1.26  6.1e-16  bluestein
      1796  2^2.449          prime    raced          17662      24342    1.34  6.5e-16  bluestein
      1797  3.599            prime    replayed       18018      27042    1.46  5.9e-16  bluestein
      1798  2.29.31          flat     replayed        9374      43705    4.43  2.0e-15  
      1799  7.257            prime    replayed       19642      18370    0.91  5.9e-16  bluestein
      1800  2^3.3^2.5^2      chain3   replayed        3177       3367    1.04  6.3e-16  
      1801  1801             prime    replayed       18206      19758    1.08  6.0e-16  bluestein
      1802  2.17.53          prime    raced          18292      43683    2.32  5.0e-16  bluestein
      1803  3.601            prime    replayed       15982      20485    1.24  4.9e-16  bluestein
      1804  2^2.11.41        chain3   replayed        6729      26230    3.82  5.6e-16  
      1805  5.19^2           flat     replayed        5967      26574    4.21  1.3e-15  
      1806  2.3.7.43         chain3   replayed        7778      25111    3.19  5.7e-16  
      1807  13.139           prime    replayed       15583      27079    1.44  5.4e-16  flips differ 1.25x bluestein
      1808  2^4.113          prime    raced          15440      28169    1.56  5.6e-16  bluestein
      1809  3^3.67           prime    replayed       18160      32373    1.74  6.1e-16  bluestein
      1810  2.5.181          prime    raced          18026      19366    1.06  4.7e-16  bluestein
      1811  1811             prime    replayed       18120      34781    1.88  6.4e-16  bluestein
      1812  2^2.3.151        prime    raced          15498      20425    1.11  5.4e-16  bluestein
      1813  7^2.37           chain3   replayed        6814      22423    3.28  5.0e-16  
      1814  2.907            prime    raced          18039      36937    1.99  7.0e-16  bluestein
      1815  3.5.11^2         chain3   replayed        4327       7746    1.78  4.9e-16  
      1816  2^3.227          prime    raced          18287      23785    1.20  5.7e-16  bluestein
      1817  23.79            prime    replayed       19598      38762    1.71  5.0e-16  bluestein
      1818  2.3^2.101        prime    raced          18037      19333    1.06  5.9e-16  bluestein
      1819  17.107           prime    replayed       18196      38465    1.98  6.1e-16  bluestein
      1820  2^2.5.7.13       flat     replayed        4680       4671    0.99  8.1e-16  
      1821  3.607            prime    replayed       18155      34917    1.79  7.2e-16  bluestein
      1822  2.911            prime    raced          18039      35931    1.98  5.5e-16  bluestein
      1823  1823             prime    replayed       18188      34768    1.85  6.7e-16  bluestein
      1824  2^5.3.19         chain3   replayed        3870      14398    3.57  4.8e-16  
      1825  5^2.73           prime    replayed       18179      21446    1.17  5.9e-16  bluestein
      1826  2.11.83          prime    raced          18218      28290    1.52  6.9e-16  bluestein
      1827  3^2.7.29         flat     replayed        6423      19345    2.97  6.0e-16  
      1828  2^2.457          prime    raced          18159      24287    1.25  5.5e-16  bluestein
      1829  31.59            prime    replayed       18148      43266    2.19  4.8e-16  bluestein
      1830  2.3.5.61         prime    raced          16039      19966    1.24  6.5e-16  bluestein
      1831  1831             prime    replayed       18608      28486    1.52  5.4e-16  bluestein
      1832  2^3.229          prime    raced          15644      22587    1.43  5.1e-16  bluestein
      1833  3.13.47          flat     replayed        8031      29277    3.58  7.3e-16  
      1834  2.7.131          prime    raced          18115      26986    1.48  5.8e-16  bluestein
      1835  5.367            prime    replayed       16132      27446    1.55  6.3e-16  bluestein
      1836  2^2.3^3.17       chain3   replayed        4017      14316    3.48  6.3e-16  
      1837  11.167           prime    replayed       15104      29072    1.83  7.7e-16  bluestein
      1838  2.919            prime    raced          17951      36491    2.00  5.6e-16  bluestein
      1839  3.613            prime    replayed       15577      29253    1.86  6.2e-16  bluestein
      1840  2^4.5.23         chain3   replayed        4324      16195    3.74  5.0e-16  
      1841  7.263            prime    replayed       15089      26257    1.35  5.9e-16  flips differ 1.28x bluestein
      1842  2.3.307          prime    raced          16038      35114    2.12  5.3e-16  bluestein
      1843  19.97            prime    replayed       18112      31308    1.73  5.7e-16  bluestein
      1844  2^2.461          prime    raced          15700      24326    1.32  5.0e-16  bluestein
      1845  3^2.5.41         chain3   replayed        6548      26039    3.87  4.4e-16  
      1846  2.13.71          prime    raced          15187      25461    1.62  6.3e-16  bluestein
      1847  1847             prime    replayed       18457      30011    1.49  6.8e-16  bluestein
      1848  2^3.3.7.11       flat     replayed        5252       5253    0.96  9.1e-16  
      1849  43^2             flat     replayed        9695      48327    4.29  1.5e-15  
      1850  2.5^2.37         chain3   replayed        5674      22728    3.95  4.9e-16  
      1851  3.617            prime    replayed       18094      30771    1.63  6.5e-16  bluestein
      1852  2^2.463          prime    raced          18048      23350    1.24  6.3e-16  bluestein
      1853  17.109           prime    replayed       15612      32225    2.02  5.8e-16  bluestein
      1854  2.3^2.103        prime    raced          16597      27532    1.64  5.3e-16  bluestein
      1855  5.7.53           prime    replayed       15663      32330    1.98  5.0e-16  bluestein
      1856  2^6.29           2p       replayed        4653      19170    4.06  4.1e-16  
      1857  3.619            prime    replayed       18075      31110    1.55  5.9e-16  bluestein
      1858  2.929            prime    raced          15339      36320    2.27  5.7e-16  bluestein
      1859  11.13^2          chain3   replayed        4962       9039    1.78  7.7e-16  
      1860  2^2.3.5.31       chain3   replayed        5230      20118    3.82  6.5e-16  
      1861  1861             prime    replayed       18048      29363    1.57  6.7e-16  bluestein
      1862  2.7^2.19         flat     replayed        5900      17253    2.89  7.0e-16  
      1863  3^4.23           chain3   replayed        5565      18990    3.32  6.2e-16  
      1864  2^3.233          prime    raced          15617      23135    1.44  5.5e-16  bluestein
      1865  5.373            prime    replayed       16024      27928    1.62  4.4e-16  bluestein
      1866  2.3.311          prime    raced          18333      40415    1.95  4.5e-16  bluestein
      1867  1867             prime    replayed       18126      30052    1.61  7.1e-16  bluestein
      1868  2^2.467          prime    raced          18288      23950    1.25  5.8e-16  bluestein
      1869  3.7.89           prime    replayed       18230      27359    1.48  5.6e-16  bluestein
      1870  2.5.11.17        chain3   replayed        4522      17325    3.62  5.5e-16  
      1871  1871             prime    replayed       20076      31273    1.47  6.8e-16  bluestein
      1872  2^4.3^2.13       chain3   replayed        3686       4940    1.28  7.0e-16  
      1873  1873             prime    replayed       12727      30710    2.35  1.2e-15  rader
      1874  2.937            prime    raced          17676      39217    2.13  7.1e-16  bluestein
      1875  3.5^4            flat     replayed        5182       6288    1.07  4.4e-16  
      1876  2^2.7.67         prime    raced          19230      34559    1.71  4.9e-16  bluestein
      1877  1877             prime    replayed       15932      26029    1.48  6.5e-16  bluestein
      1878  2.3.313          prime    raced          16193      35933    2.03  7.3e-16  bluestein
      1879  1879             prime    replayed       15640      25570    1.32  5.6e-16  flips differ 1.28x bluestein
      1880  2^3.5.47         flat     replayed        7762      30864    3.91  5.5e-16  
      1881  3^2.11.19        chain3   replayed        5022      19431    3.81  5.3e-16  
      1882  2.941            prime    raced          16032      25115    1.32  5.2e-16  bluestein
      1883  7.269            prime    replayed       18831      28003    1.44  5.6e-16  bluestein
      1884  2^2.3.157        prime    raced          18375      24542    1.07  6.5e-16  flips differ 1.33x bluestein
      1885  5.13.29          chain3   replayed        7264      25621    2.92  7.0e-16  
      1886  2.23.41          flat     replayed        8790      48870    5.37  1.4e-15  
      1887  3.17.37          chain3   replayed        7112      37285    5.03  6.2e-16  
      1888  2^5.59           prime    raced          18898      24971    1.21  4.8e-16  bluestein
      1889  1889             prime    replayed       18476      26294    1.33  7.0e-16  bluestein
      1890  2.3^3.5.7        chain3   replayed        3942       4222    0.94  4.9e-16  
      1891  31.61            prime    replayed       18603      41221    2.20  5.8e-16  bluestein
      1892  2^2.11.43        chain3   replayed        7448      29278    3.33  4.1e-16  
      1893  3.631            prime    replayed       18612      26708    1.23  5.7e-16  bluestein
      1894  2.947            prime    raced          18795      26547    1.38  6.7e-16  bluestein
      1895  5.379            prime    replayed       18415      22745    1.07  5.9e-16  bluestein
      1896  2^3.3.79         prime    raced          19104      28363    1.18  5.9e-16  bluestein
      1897  7.271            prime    replayed       18637      22373    1.18  5.2e-16  bluestein
      1898  2.13.73          prime    raced          18904      23353    1.18  6.0e-16  bluestein
      1899  3^2.211          prime    replayed       15977      26757    1.30  7.1e-16  flips differ 1.27x bluestein
      1900  2^2.5^2.19       flat     replayed        5727      15784    2.57  4.7e-16  
      1901  1901             prime    replayed       16639      26766    1.55  6.0e-16  bluestein
      1902  2.3.317          prime    raced          16956      24168    1.34  5.5e-16  bluestein
      1903  11.173           prime    replayed       16795      28613    1.20  5.4e-16  flips differ 1.39x bluestein
      1904  2^4.7.17         chain3   replayed        3966      16503    3.95  5.8e-16  
      1905  3.5.127          prime    replayed       16629      24237    1.45  6.0e-16  bluestein
      1906  2.953            prime    raced          19639      29305    1.27  4.9e-16  bluestein
      1907  1907             prime    replayed       20537      26378    1.25  5.3e-16  bluestein
      1908  2^2.3^2.53       prime    raced          17700      37712    1.81  4.6e-16  bluestein
      1909  23.83            prime    replayed       19236      43918    2.04  5.4e-16  bluestein
      1910  2.5.191          prime    raced          15332      21773    1.38  4.3e-16  bluestein
      1911  3.7^2.13         chain3   replayed        4254       7937    1.77  5.1e-16  
      1912  2^3.239          prime    raced          18449      23990    1.20  6.6e-16  bluestein
      1913  1913             prime    replayed       18797      26076    1.15  5.5e-16  bluestein
      1914  2.3.11.29        flat     replayed        7601      24948    3.12  2.0e-15  
      1915  5.383            prime    replayed       15682      21932    1.35  5.0e-16  bluestein
      1916  2^2.479          prime    raced          18941      28925    1.37  5.5e-16  bluestein
      1917  3^3.71           prime    replayed       22357      25851    0.79  6.1e-16  flips differ 1.43x bluestein
      1918  2.7.137          prime    raced          17237      27135    1.47  5.3e-16  bluestein
      1919  19.101           prime    replayed       18668      36171    1.86  5.4e-16  bluestein
      1920  2^7.3.5          chain3   replayed        3229       3291    0.96  4.7e-16  
      1921  17.113           prime    replayed       16078      44616    2.73  4.6e-16  bluestein
      1922  2.31^2           flat     replayed        9009      40390    4.38  2.2e-15  
      1923  3.641            prime    replayed       19095      21554    1.10  4.5e-16  bluestein
      1924  2^2.13.37        chain3   replayed        6794      26794    3.82  4.6e-16  
      1925  5^2.7.11         flat     replayed        5435       6873    1.26  6.0e-16  
      1926  2.3^2.107        prime    raced          18804      29548    1.50  6.8e-16  bluestein
      1927  41.47            prime    replayed       15438      55062    3.00  4.7e-16  bluestein
      1928  2^3.241          prime    raced          26203      23743    0.89  6.0e-16  bluestein
      1929  3.643            prime    replayed       18656      29351    1.53  5.7e-16  bluestein
      1930  2.5.193          prime    raced          18494      20004    1.07  5.6e-16  bluestein
      1931  1931             prime    replayed       20640      32026    1.31  8.6e-16  flips differ 1.26x bluestein
      1932  2^2.3.7.23       flat     replayed        9694      25910    2.58  6.6e-16  
      1933  1933             prime    replayed       25988      41368    1.58  8.0e-16  bluestein
      1934  2.967            prime    raced          25812      43285    1.58  7.5e-16  bluestein
      1935  3^2.5.43         flat     replayed       10777      38272    3.52  5.4e-16  
      1936  2^4.11^2         chain3   replayed        3857       7654    1.85  4.6e-16  
      1937  13.149           prime    replayed       18948      27403    1.44  5.8e-16  bluestein
      1938  2.3.17.19        chain3   replayed        5955      39591    4.80  6.9e-16  flips differ 1.60x
      1939  7.277            prime    replayed       18555      26677    1.41  6.4e-16  bluestein
      1940  2^2.5.97         prime    raced          16223      20400    1.22  5.5e-16  bluestein
      1941  3.647            prime    replayed       18438      28310    1.49  6.8e-16  bluestein
      1942  2.971            prime    raced          18016      34556    1.74  7.0e-16  bluestein
      1943  29.67            prime    replayed       16105      54063    3.27  5.7e-16  bluestein
      1944  2^3.3^5          chain3   replayed        4090       4633    1.13  5.1e-16  
      1945  5.389            prime    replayed       16062      31702    1.93  6.7e-16  bluestein
      1946  2.7.139          prime    raced          18132      26147    1.42  5.1e-16  bluestein
      1947  3.11.59          prime    replayed       18389      31512    1.42  6.3e-16  bluestein
      1948  2^2.487          prime    raced          15875      23900    1.33  6.6e-16  bluestein
      1949  1949             prime    replayed       18304      26571    1.41  5.8e-16  bluestein
      1950  2.3.5^2.13       chain3   replayed        4392       4670    1.05  5.8e-16  
      1951  1951             prime    replayed       15886      26475    1.45  6.2e-16  bluestein
      1952  2^5.61           prime    raced          15249      21816    1.32  5.2e-16  bluestein
      1953  3^2.7.31         flat     replayed        7022      22395    3.00  8.5e-16  
      1954  2.977            prime    raced          15725      27109    1.35  5.6e-16  bluestein
      1955  5.17.23          chain3   replayed        6370      32436    4.78  5.3e-16  
      1956  2^2.3.163        prime    raced          18150      25113    1.34  7.4e-16  bluestein
      1957  19.103           prime    replayed       18280      43446    2.26  6.9e-16  bluestein
      1958  2.11.89          prime    raced          18525      30669    1.63  6.0e-16  bluestein
      1959  3.653            prime    replayed       19210      33295    1.41  6.3e-16  bluestein
      1960  2^3.5.7^2        flat     replayed        4864       4392    0.79  5.4e-16  
      1961  37.53            prime    replayed       18699      57546    3.04  5.6e-16  bluestein
      1962  2.3^2.109        prime    raced          18149      22268    1.15  5.9e-16  bluestein
      1963  13.151           prime    replayed       18240      24397    1.29  5.8e-16  bluestein
      1964  2^2.491          prime    raced          19435      33801    1.43  5.6e-16  bluestein
      1965  3.5.131          prime    replayed       19086      33418    1.60  7.7e-16  bluestein
      1966  2.983            prime    raced          16695      30223    1.37  5.9e-16  flips differ 1.29x bluestein
      1967  7.281            prime    replayed       15819      27124    1.42  5.5e-16  bluestein
      1968  2^4.3.41         flat     replayed        7917      25726    3.20  5.6e-16  
      1969  11.179           prime    replayed       18361      28324    1.51  6.3e-16  bluestein
      1970  2.5.197          prime    raced          18810      24298    1.23  5.2e-16  bluestein
      1971  3^3.73           prime    replayed       18137      23392    1.26  5.3e-16  bluestein
      1972  2^2.17.29        flat     replayed        7760      33403    4.09  9.9e-16  
      1973  1973             prime    replayed       18636      26494    1.31  6.4e-16  bluestein
      1974  2.3.7.47         chain3   replayed        7255      29064    4.00  4.3e-16  
      1975  5^2.79           prime    replayed       15283      28805    1.88  5.5e-16  bluestein
      1976  2^3.13.19        chain3   replayed        5146      17918    2.75  6.5e-16  flips differ 1.27x
      1977  3.659            prime    replayed       18226      30264    1.55  6.2e-16  bluestein
      1978  2.23.43          flat     replayed        9381      45621    3.49  1.8e-15  flips differ 1.40x
      1979  1979             prime    replayed       22332      36803    1.65  4.5e-16  bluestein
      1980  2^2.3^2.5.11     chain3   replayed        4000       5236    0.91  5.5e-16  flips differ 1.43x
      1981  7.283            prime    replayed       18133      26154    1.41  5.5e-16  bluestein
      1982  2.991            prime    raced          17880      32932    1.65  4.9e-16  bluestein
      1983  3.661            prime    replayed       18529      30788    1.62  6.5e-16  bluestein
      1984  2^6.31           2p       replayed        5238      23484    4.26  4.5e-16  
      1985  5.397            prime    replayed       18143      25289    1.36  5.8e-16  bluestein
      1986  2.3.331          prime    raced          15586      35370    2.00  6.0e-16  bluestein
      1987  1987             prime    replayed       18211      28158    1.49  7.3e-16  bluestein
      1988  2^2.7.71         prime    raced          18120      26423    1.34  4.6e-16  bluestein
      1989  3^2.13.17        chain3   replayed        5838      18381    3.09  6.5e-16  
      1990  2.5.199          prime    raced          16120      25242    1.43  4.9e-16  bluestein
      1991  11.181           prime    replayed       17786      26590    1.19  7.1e-16  flips differ 1.36x bluestein
      1992  2^3.3.83         prime    raced          15709      29266    1.39  5.7e-16  flips differ 1.35x bluestein
      1993  1993             prime    replayed       16298      27939    1.38  6.2e-16  bluestein
      1994  2.997            prime    raced          15123      27079    1.74  4.9e-16  bluestein
      1995  3.5.7.19         chain3   replayed        5055      17078    3.18  4.7e-16  
      1996  2^2.499          prime    raced          15232      27159    1.48  5.7e-16  bluestein
      1997  1997             prime    replayed       18500      27376    1.39  5.9e-16  bluestein
      1998  2.3^3.37         chain3   replayed        9556      26534    2.62  5.3e-16  
      1999  1999             prime    replayed       15717      28320    1.70  7.0e-16  bluestein
      2000  2^4.5^3          chain3   replayed        3988       3575    0.82  3.7e-16  
      2001  3.23.29          chain3   replayed        7679      39584    5.09  5.1e-16  
      2002  2.7.11.13        flat     replayed        6265       8687    1.26  7.2e-16  
      2003  2003             prime    replayed       15758      32574    1.97  5.9e-16  bluestein
      2004  2^2.3.167        prime    raced          19074      28811    1.41  5.2e-16  bluestein
      2005  5.401            prime    replayed       15787      21986    1.16  6.0e-16  bluestein
      2006  2.17.59          prime    raced          19398      43601    1.23  5.8e-16  flips differ 1.77x bluestein
      2007  3^2.223          prime    replayed       18128      28811    1.55  7.5e-16  bluestein
      2008  2^3.251          prime    raced          18275      23739    1.29  5.5e-16  bluestein
      2009  7^2.41           chain3   replayed        8602      26852    2.70  4.9e-16  
      2010  2.3.5.67         prime    raced          18856      36298    1.54  6.1e-16  flips differ 1.28x bluestein
      2011  2011             prime    replayed       20424      36364    1.72  6.1e-16  bluestein
      2012  2^2.503          prime    raced          19251      25792    1.22  5.2e-16  bluestein
      2013  3.11.61          prime    replayed       19067      25699    1.32  6.2e-16  bluestein
      2014  2.19.53          prime    raced          17213      49576    2.63  6.2e-16  bluestein
      2015  5.13.31          flat     replayed        8446      29012    3.30  6.5e-16  
      2016  2^5.3^2.7        chain3   replayed        3481       4254    1.11  4.7e-16  
      2017  2017             prime    replayed       19343      32501    1.64  7.2e-16  bluestein
      2018  2.1009           prime    raced          18823      40143    2.06  7.9e-16  bluestein
      2019  3.673            prime    replayed       18429      30434    1.62  5.8e-16  bluestein
      2020  2^2.5.101        prime    raced          18428      22166    0.51  4.9e-16  flips differ 2.26x bluestein
      2021  43.47            prime    replayed       19239      56075    2.86  4.9e-16  bluestein
      2022  2.3.337          prime    raced          18958      36018    1.86  5.6e-16  bluestein
      2023  7.17^2           chain3   replayed        5923      29352    4.85  6.7e-16  
      2024  2^3.11.23        chain3   replayed        5734      21640    3.33  4.9e-16  
      2025  3^4.5^2          chain3   replayed        4358       6135    1.40  4.3e-16  
      2026  2.1013           prime    raced          19152      38696    1.66  6.9e-16  bluestein
      2027  2027             prime    replayed       18914      28928    1.47  5.5e-16  bluestein
      2028  2^2.3.13^2       chain3   replayed        4495       8006    1.75  6.7e-16  
      2029  2029             prime    replayed       14968      28026    1.56  1.0e-15  rader
      2030  2.5.7.29         flat     replayed        8432      22524    2.62  5.1e-16  
      2031  3.677            prime    replayed       18731      32212    1.63  5.8e-16  bluestein
      2032  2^4.127          prime    raced          18175      27947    1.42  7.1e-16  bluestein
      2033  19.107           prime    replayed       16184      42891    2.61  6.1e-16  bluestein
      2034  2.3^2.113        prime    raced          17000      32222    1.51  6.2e-16  bluestein
      2035  5.11.37          chain3   replayed        8327      31591    3.25  5.7e-16  
      2036  2^2.509          prime    raced          18889      24465    1.26  6.2e-16  bluestein
      2037  3.7.97           prime    replayed       18772      24051    1.27  6.0e-16  bluestein
      2038  2.1019           prime    raced          15774      25343    1.57  6.6e-16  bluestein
      2039  2039             prime    replayed       18618      27708    1.48  7.6e-16  bluestein
      2040  2^3.3.5.17       chain3   replayed        4772      15146    3.12  5.5e-16  
      2041  13.157           prime    replayed       18685      29942    1.57  5.5e-16  bluestein
      2042  2.1021           prime    raced          15865      25319    1.59  5.4e-16  bluestein
      2043  3^2.227          prime    replayed       18797      27990    1.45  5.8e-16  bluestein
      2044  2^2.7.73         prime    raced          17913      23293    1.24  6.5e-16  bluestein
      2045  5.409            prime    replayed       20056      33715    1.58  5.9e-16  bluestein
      2046  2.3.11.31        chain3   replayed        8604      25995    2.99  4.9e-16  
      2047  23.89            prime    replayed       18549      43432    2.30  5.7e-16  bluestein
      2048  2^11             ztt      replayed        2750       3047    1.04  3.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     1180     70    231   0.87   1.31   1.92    1.29
 chain3     346      9     39   0.97   1.67   4.01    1.96
 2p         256      4     32   0.98   3.01   4.84    2.38
 flat       240      4     16   1.12   3.08   4.47    2.70
 mono        22      8     13   0.71   0.84   2.40    1.14
 ztt          3      0      2   0.94   0.96   1.04    0.98
 ALL       2047     95    333   0.90   1.42   3.76    1.63
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   0.81      5      2
 8..31                 24   0.85     17      9
 32..127               96   1.67     15      1
 128..511             384   1.64     33      6
 512..2047           1536   1.38    263     77
 2048..2048             1   1.04      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 composite with a prime >= 53 (prime cell)      878   1.26    209   1.23
 chain3                                         346   1.67     39   1.96
 2p                                             251   3.09     32   2.42
 flat                                           240   3.08     16   2.70
 prime N, bluestein                             188   1.34     22   1.32
 prime N, rader                                 107   1.76      0   1.78
 mono                                            19   0.83     11   1.18
 pow2                                            11   1.01      4   0.98
 composite, prime cell by race                    7   1.95      0   2.04
```


flip agreement: our two readings more than 25% apart at 150 of 2047 cells.

worst 10: 1212 (prime 0.24), 1052 (prime 0.33), 683 (prime 0.42), 1255 (prime 0.43), 1181 (prime 0.43), 1744 (prime 0.43), 1205 (prime 0.51), 579 (prime 0.51), 2020 (prime 0.51), 640 (2p 0.59)
best 5: 1312 (2p 5.90), 928 (2p 5.81), 645 (2p 5.63), 1376 (2p 5.56), 703 (2p 5.55)
