# gauntlet report

run: `zen4_2_2048_B_recal`  contract file suffix: `_fftw`  cells: 2047 listed, 2047 benched, comparator: FFTW

control cell: 44 readings, 1.029..2.969 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced              7          8    1.05  0.0e+00  
         3  3                mono     raced             10          9    0.69  1.5e-16  flips differ 1.30x
         4  2^2              mono     raced             10         10    0.84  0.0e+00  
         5  5                mono     raced             15         13    0.82  1.5e-16  
         6  2.3              mono     raced             15         11    0.74  2.8e-16  
         7  7                mono     raced             19         15    0.80  2.6e-16  
         8  2^3              mono     raced             17         15    0.76  5.6e-17  
         9  3^2              mono     raced             21         18    0.85  4.2e-16  
        10  2.5              mono     raced             22         16    0.73  1.8e-16  
        11  11               mono     raced             29         22    0.73  1.6e-16  
        12  2^2.3            mono     raced             25         17    0.69  2.8e-16  
        13  13               mono     raced             40         24    0.58  6.0e-16  
        14  2.7              mono     raced             32         22    0.67  1.6e-16  
        15  3.5              2p       raced             34         24    0.69  3.3e-16  
        16  2^4              ztt      raced             21         21    0.96  1.5e-16  
        17  17               mono     raced             55        103    1.89  2.4e-16  
        18  2.3^2            2p       raced             34         34    0.99  3.2e-16  
        19  19               mono     raced             60        124    2.04  3.9e-16  
        20  2^2.5            2p       raced             36         30    0.85  2.2e-16  
        21  3.7              2p       raced             41         51    1.25  3.3e-16  
        22  2.11             mono     raced             52         50    0.89  2.2e-16  
        23  23               mono     raced             82        180    2.12  3.1e-16  
        24  2^3.3            2p       raced             37         34    0.92  2.8e-16  
        25  5^2              2p       raced             49         52    0.60  1.8e-16  flips differ 1.76x
        26  2.13             mono     raced             68         49    0.72  4.4e-16  
        27  3^3              2p       raced             50         66    1.30  3.8e-16  
        28  2^2.7            2p       raced             42         40    0.94  2.6e-16  
        29  29               mono     raced            120        260    2.17  3.7e-16  
        30  2.3.5            2p       raced             47         46    0.96  2.6e-16  
        31  31               mono     raced            122        293    2.41  2.7e-16  
        32  2^5              ztt      raced             33         38    1.16  1.7e-16  
        33  3.11             2p       raced             63         90    1.41  2.2e-16  
        34  2.17             flat     raced            119        232    1.83  5.2e-16  
        35  5.7              2p       raced             59         77    1.29  3.9e-16  
        36  2^2.3^2          2p       raced             51         47    0.91  3.2e-16  
        37  37               mono     raced            159        400    2.24  3.6e-16  
        38  2.19             flat     raced            137        280    2.01  3.5e-16  
        39  3.13             2p       raced             78         92    1.18  5.3e-16  
        40  2^3.5            2p       raced             48         47    0.97  1.4e-16  
        41  41               mono     raced            194        483    2.48  4.1e-16  
        42  2.3.7            2p       raced             61         70    1.13  3.9e-16  
        43  43               mono     raced            209        538    2.54  3.6e-16  
        44  2^2.11           2p       raced             63         81    1.28  4.0e-16  
        45  3^2.5            2p       raced             74         94    1.27  4.5e-16  
        46  2.23             flat     raced            143        380    2.66  4.9e-16  
        47  47               mono     raced            239        574    2.40  4.0e-16  
        48  2^4.3            2p       raced             53         54    1.03  3.7e-16  
        49  7^2              2p       raced             75        105    1.38  3.9e-16  
        50  2.5^2            2p       raced             70         75    1.07  3.6e-16  
        51  3.17             2p       raced            116        354    3.04  3.5e-16  
        52  2^2.13           2p       raced             79         91    1.15  5.3e-16  
        53  53               prime    raced            288        724    2.48  6.8e-16  rader
        54  2.3^3            2p       raced             75         81    1.06  4.0e-16  
        55  5.11             2p       raced             94        120    1.28  4.2e-16  
        56  2^3.7            2p       raced             69         69    0.97  1.7e-16  
        57  3.19             2p       raced            137        462    3.09  5.4e-16  
        58  2.29             flat     raced            195        562    2.85  4.0e-16  
        59  59               prime    raced            373        690    1.75  4.3e-16  bluestein
        60  2^2.3.5          2p       raced             73         69    0.94  3.4e-16  
        61  61               prime    raced            297        596    2.00  4.9e-16  rader
        62  2.31             flat     raced            212        625    2.82  3.5e-16  
        63  3^2.7            2p       raced             95        130    1.36  3.8e-16  
        64  2^6              ztt      raced             59         66    1.13  3.1e-16  
        65  5.13             2p       raced            116        140    1.16  4.7e-16  
        66  2.3.11           2p       raced             96        117    1.22  3.7e-16  
        67  67               prime    raced            361       1084    2.95  5.5e-16  rader
        68  2^2.17           2p       raced            115        471    3.95  3.2e-16  
        69  3.23             2p       raced            182        592    3.19  3.5e-16  
        70  2.5.7            2p       raced             90        106    1.16  2.3e-16  
        71  71               prime    raced            358        806    2.24  4.5e-16  rader
        72  2^3.3^2          2p       raced             84         81    0.96  3.8e-16  
        73  73               prime    raced            370        752    1.67  5.8e-16  flips differ 1.29x rader
        74  2.37             flat     raced            273       1059    3.22  4.3e-16  flips differ 1.33x
        75  3.5^2            2p       raced            111        164    1.47  2.5e-16  
        76  2^2.19           2p       raced            152        549    3.53  3.6e-16  
        77  7.11             2p       raced            123        206    1.63  2.5e-16  
        78  2.3.13           2p       raced            120        133    1.10  4.7e-16  
        79  79               prime    raced            440       1007    1.95  6.8e-16  rader
        80  2^4.5            2p       raced             87         85    0.97  2.8e-16  
        81  3^4              2p       raced            123        163    1.32  4.3e-16  
        82  2.41             flat     raced            311       1073    3.41  6.2e-16  
        83  83               prime    raced            688       1060    1.54  4.7e-16  bluestein
        84  2^2.3.7          2p       raced            108        101    0.93  4.1e-16  
        85  5.17             2p       raced            174        595    3.42  4.0e-16  
        86  2.43             flat     raced            332       1113    3.27  1.3e-15  
        87  3.29             2p       raced            258        896    3.47  4.4e-16  
        88  2^3.11           2p       raced            115        196    1.69  2.8e-16  
        89  89               prime    raced            461       1073    2.33  4.4e-16  rader
        90  2.3^2.5          2p       raced            115        356    1.15  3.7e-16  
        91  7.13             2p       raced            153        178    1.10  4.6e-16  
        92  2^2.23           2p       raced            176        759    4.10  4.1e-16  
        93  3.31             2p       raced            285       1001    3.51  3.8e-16  
        94  2.47             flat     raced            377       1373    3.60  8.6e-16  
        95  5.19             2p       raced            213        911    3.27  3.0e-16  
        96  2^5.3            2p       raced             98         97    0.99  3.0e-16  
        97  97               prime    raced            458        907    1.96  4.3e-16  rader
        98  2.7^2            flat     raced            230        155    0.67  3.8e-16  
        99  3^2.11           2p       raced            158        258    1.61  4.7e-16  
       100  2^2.5^2          2p       raced            114        110    0.96  2.5e-16  
       101  101              prime    raced            489        963    1.97  6.3e-16  rader
       102  2.3.17           2p       raced            175        777    4.43  4.6e-16  
       103  103              prime    raced            628       1421    2.24  7.6e-16  rader
       104  2^3.13           2p       raced            146        176    1.14  4.3e-16  
       105  3.5.7            2p       raced            148        191    1.28  3.7e-16  
       106  2.53             prime    raced            719       1698    2.03  3.3e-16  bluestein
       107  107              prime    raced            702       1426    1.68  7.5e-16  bluestein
       108  2^2.3^3          2p       raced            140        159    1.02  4.2e-16  
       109  109              prime    raced            558       1113    1.99  6.9e-16  rader
       110  2.5.11           2p       raced            153        190    1.24  2.8e-16  
       111  3.37             2p       raced            365       1358    3.68  4.3e-16  
       112  2^4.7            2p       raced            125        125    0.99  2.9e-16  
       113  113              prime    raced            563       1635    2.88  5.8e-16  rader
       114  2.3.19           2p       raced            213        826    3.60  3.6e-16  
       115  5.23             2p       raced            266        962    3.61  3.2e-16  
       116  2^2.29           2p       raced            251       1180    4.69  4.4e-16  
       117  3^2.13           2p       raced            203        296    1.46  6.2e-16  
       118  2.59             prime    raced            711       1591    2.24  3.2e-16  bluestein
       119  7.17             2p       raced            231        821    3.55  4.6e-16  
       120  2^3.3.5          2p       raced            133        127    0.96  3.7e-16  
       121  11^2             2p       raced            209        365    1.74  3.7e-16  
       122  2.61             prime    raced            752       1257    1.67  3.1e-16  bluestein
       123  3.41             2p       raced            448       1541    3.43  4.0e-16  
       124  2^2.31           2p       raced            275       1275    4.63  5.3e-16  
       125  5^3              2p       raced            190        280    1.44  4.5e-16  
       126  2.3^2.7          2p       raced            171        214    0.97  2.8e-16  
       127  127              prime    raced            738       1434    1.83  3.8e-16  bluestein
       128  2^7              2p       raced            120        125    1.02  2.1e-16  
       129  3.43             2p       raced            473       1769    3.73  4.6e-16  
       130  2.5.13           2p       raced            191        293    1.52  4.7e-16  
       131  131              prime    raced            722       1785    2.47  6.0e-16  rader
       132  2^2.3.11         2p       raced            180        277    1.54  4.6e-16  
       133  7.19             2p       raced            271        981    3.45  3.5e-16  
       134  2.67             prime    raced           1438       2626    1.81  4.5e-16  bluestein
       135  3^3.5            2p       raced            193        247    1.27  3.0e-16  
       136  2^3.17           2p       raced            215        921    4.28  3.8e-16  
       137  137              prime    raced            793       1722    2.16  8.3e-16  rader
       138  2.3.23           2p       raced            264       1144    4.34  4.8e-16  
       139  139              prime    raced            973       1705    1.71  1.0e-15  rader
       140  2^2.5.7          chain3   raced            261        161    0.61  2.7e-16  
       141  3.47             2p       raced            542       2038    3.75  3.6e-16  
       142  2.71             prime    raced           1445       1788    1.12  5.4e-16  bluestein
       143  11.13            2p       raced            257        423    1.65  5.1e-16  
       144  2^4.3^2          2p       raced            156        156    1.00  4.7e-16  
       145  5.29             2p       raced            381       1408    3.67  4.0e-16  
       146  2.73             prime    raced           1508       1474    0.95  5.7e-16  bluestein
       147  3.7^2            2p       raced            217        427    1.96  4.4e-16  
       148  2^2.37           2p       raced            353       1702    3.86  2.8e-16  
       149  149              prime    raced           1306       1751    1.31  5.8e-16  rader
       150  2.3.5^2          2p       raced            191        211    1.09  3.0e-16  
       151  151              prime    raced            857       1515    1.76  7.3e-16  rader
       152  2^3.19           2p       raced            258       1098    4.12  4.1e-16  
       153  3^2.17           2p       raced            294       1106    3.58  4.2e-16  
       154  2.7.11           flat     raced            356        399    1.11  3.5e-16  
       155  5.31             2p       raced            423       1583    3.72  3.6e-16  
       156  2^2.3.13         2p       raced            223        306    1.36  3.9e-16  
       157  157              prime    raced            859       1748    1.92  1.4e-15  rader
       158  2.79             prime    raced           1470       2320    1.32  3.3e-16  bluestein
       159  3.53             prime    raced           1459       2481    1.68  4.5e-16  bluestein
       160  2^5.5            2p       raced            163        169    1.02  3.0e-16  
       161  7.23             2p       raced            352       1348    3.76  4.5e-16  
       162  2.3^4            2p       raced            214        316    1.45  4.2e-16  
       163  163              prime    raced           1048       1812    1.72  6.8e-16  rader
       164  2^2.41           2p       raced            435       2121    4.68  4.0e-16  
       165  3.5.11           2p       raced            253        367    1.45  3.8e-16  
       166  2.83             prime    raced           1472       2777    1.84  4.9e-16  bluestein
       167  167              prime    raced           1589       2249    1.29  4.1e-16  bluestein
       168  2^3.3.7          2p       raced            210        198    0.91  3.2e-16  
       169  13^2             2p       raced            318        497    1.53  6.0e-16  
       170  2.5.17           2p       raced            288       1195    3.94  7.6e-16  
       171  3^2.19           2p       raced            349       1535    4.36  6.3e-16  
       172  2^2.43           2p       raced            465       2243    4.80  4.3e-16  
       173  173              prime    raced           1484       2238    1.51  4.6e-16  bluestein
       174  2.3.29           2p       raced            380       1702    4.42  5.5e-16  
       175  5^2.7            2p       raced            259        372    1.43  3.7e-16  
       176  2^4.11           2p       raced            222        351    1.58  2.6e-16  
       177  3.59             prime    raced           1478       2424    1.63  6.5e-16  bluestein
       178  2.89             prime    raced           1479       2243    1.51  5.5e-16  bluestein
       179  179              prime    raced           1473       2242    1.51  6.3e-16  bluestein
       180  2^2.3^2.5        2p       raced            220        245    1.10  4.0e-16  
       181  181              prime    raced           1033       1866    1.74  6.9e-16  rader
       182  2.7.13           flat     raced            443        448    0.98  4.7e-16  
       183  3.61             prime    raced           1474       2003    1.35  5.1e-16  bluestein
       184  2^3.23           2p       raced            334       1611    4.82  5.6e-16  
       185  5.37             2p       raced            540       2332    4.12  5.0e-16  
       186  2.3.31           2p       raced            417       1974    4.56  4.7e-16  
       187  11.17            2p       raced            372       1691    4.32  6.0e-16  
       188  2^2.47           2p       raced            535       2898    5.06  4.5e-16  
       189  3^3.7            2p       raced            287        546    1.78  3.6e-16  
       190  2.5.19           2p       raced            342       1409    4.05  4.6e-16  
       191  191              prime    raced           1181       2073    1.61  8.4e-16  rader
       192  2^6.3            chain3   raced            238        197    0.79  2.7e-16  
       193  193              prime    raced            912       1787    1.95  4.7e-16  rader
       194  2.97             prime    raced           1578       2016    1.22  5.4e-16  bluestein
       195  3.5.13           2p       raced            319        457    1.34  5.7e-16  
       196  2^2.7^2          chain3   raced            361        369    1.02  3.1e-16  
       197  197              prime    raced           1370       2125    1.52  5.3e-16  rader
       198  2.3^2.11         chain3   raced            362        451    1.22  5.7e-16  
       199  199              prime    raced           1363       2122    1.55  6.8e-16  rader
       200  2^3.5^2          2p       raced            249        220    0.61  2.7e-16  flips differ 1.47x
       201  3.67             prime    raced           1506       3459    2.26  6.4e-16  bluestein
       202  2.101            prime    raced           1599       2161    1.28  4.0e-16  bluestein
       203  7.29             2p       raced            507       1983    3.90  4.5e-16  
       204  2^2.3.17         2p       raced            377       1485    3.04  5.0e-16  flips differ 1.29x
       205  5.41             2p       raced            664       2699    3.05  4.3e-16  flips differ 1.27x
       206  2.103            prime    raced           1495       4427    2.93  6.9e-16  bluestein
       207  3^2.23           2p       raced            453       1791    3.85  5.0e-16  
       208  2^4.13           2p       raced            279        392    1.38  5.9e-16  
       209  11.19            2p       raced            438       1764    3.89  5.8e-16  
       210  2.3.5.7          2p       raced            284        319    1.07  5.3e-16  
       211  211              prime    raced           1261       2697    2.12  6.3e-16  rader
       212  2^2.53           prime    raced           1509       3411    2.12  4.5e-16  bluestein
       213  3.71             prime    raced           1507       2692    1.63  4.0e-16  bluestein
       214  2.107            prime    raced           1503       3434    1.94  4.9e-16  bluestein
       215  5.43             2p       raced            711       2989    4.14  5.2e-16  
       216  2^3.3^3          2p       raced            268        379    1.36  3.9e-16  
       217  7.31             2p       raced            559       2450    4.32  4.6e-16  
       218  2.109            prime    raced           1503       2510    1.60  4.8e-16  bluestein
       219  3.73             prime    raced           1513       2378    1.47  4.5e-16  bluestein
       220  2^2.5.11         chain3   raced            387        436    1.12  3.1e-16  
       221  13.17            2p       raced            452       1923    3.86  6.2e-16  
       222  2.3.37           2p       raced            540       2572    4.66  3.9e-16  
       223  223              prime    raced           1544       2948    1.70  6.1e-16  bluestein
       224  2^5.7            2p       raced            264        249    0.94  3.3e-16  
       225  3^2.5^2          2p       raced            325        478    1.46  4.8e-16  
       226  2.113            prime    raced           1507       4165    2.67  4.8e-16  bluestein
       227  227              prime    raced           1582       2670    1.68  5.1e-16  bluestein
       228  2^2.3.19         2p       raced            397       1716    4.28  4.5e-16  
       229  229              prime    raced           1418       2690    1.86  6.3e-16  rader
       230  2.5.23           2p       raced            437       1926    4.25  4.7e-16  
       231  3.7.11           2p       raced            382        637    1.61  4.1e-16  
       232  2^3.29           2p       raced            487       2375    4.76  3.5e-16  
       233  233              prime    raced           1517       2764    1.77  4.4e-16  bluestein
       234  2.3^2.13         chain3   raced            437        503    1.15  5.1e-16  
       235  5.47             2p       raced            820       3303    3.96  3.4e-16  
       236  2^2.59           prime    raced           1522       2954    1.93  5.1e-16  bluestein
       237  3.79             prime    raced           1519       3052    1.93  5.8e-16  bluestein
       238  2.7.17           flat     raced            617       1782    2.85  6.3e-16  
       239  239              prime    raced           1528       2868    1.88  4.9e-16  bluestein
       240  2^4.3.5          2p       raced            273        256    0.94  3.5e-16  
       241  241              prime    raced           1355       2267    1.67  4.4e-16  rader
       242  2.11^2           flat     raced            585       1004    1.64  7.6e-16  
       243  3^5              2p       raced            371        648    1.74  4.9e-16  
       244  2^2.61           prime    raced           1519       2522    1.65  4.4e-16  bluestein
       245  5.7^2            chain3   raced            501        732    1.45  2.8e-16  
       246  2.3.41           2p       raced            660       3249    4.78  3.0e-16  
       247  13.19            2p       raced            547       2076    3.76  4.1e-16  
       248  2^3.31           2p       raced            537       2506    4.66  5.1e-16  
       249  3.83             prime    raced           1537       3404    2.20  5.4e-16  bluestein
       250  2.5^3            2p       raced            348        433    1.22  3.7e-16  
       251  251              prime    raced           1528       2684    1.74  4.8e-16  bluestein
       252  2^2.3^2.7        2p       raced            356        447    1.12  4.5e-16  
       253  11.23            2p       raced            574       2468    4.19  5.1e-16  
       254  2.127            prime    raced           1600       3022    1.86  4.8e-16  bluestein
       255  3.5.17           2p       raced            486       1863    3.83  4.7e-16  
       256  2^8              2p       raced            234        256    1.09  2.5e-16  
       257  257              prime    raced           1185       2370    1.99  4.1e-16  rader
       258  2.3.43           2p       raced            715       3367    4.64  4.8e-16  
       259  7.37             2p       raced            720       2997    4.16  4.4e-16  
       260  2^2.5.13         chain3   raced            473        507    1.07  3.7e-16  
       261  3^2.29           2p       raced            653       2560    3.68  4.2e-16  
       262  2.131            prime    raced           3189       3718    1.16  4.9e-16  bluestein
       263  263              prime    raced           3200       3498    1.07  4.6e-16  bluestein
       264  2^3.3.11         chain3   raced            460        562    1.21  3.6e-16  
       265  5.53             prime    raced           3208       4248    1.28  3.4e-16  bluestein
       266  2.7.19           flat     raced            737       2087    2.74  4.4e-16  
       267  3.89             prime    raced           3285       4260    1.20  4.8e-16  bluestein
       268  2^2.67           prime    raced           3212       4503    1.39  5.5e-16  bluestein
       269  269              prime    raced           3199       3603    1.09  5.7e-16  bluestein
       270  2.3^3.5          2p       raced            381        490    1.18  4.3e-16  
       271  271              prime    raced           1644       2964    1.70  7.2e-16  rader
       272  2^4.17           2p       raced            440       1958    4.42  5.5e-16  
       273  3.7.13           2p       raced            469        708    1.51  5.7e-16  
       274  2.137            prime    raced           3217       4328    1.31  5.4e-16  bluestein
       275  5^2.11           2p       raced            453        796    1.74  4.0e-16  
       276  2^2.3.23         2p       raced            522       2505    4.62  5.6e-16  
       277  277              prime    raced           1913       3454    1.79  7.8e-16  rader
       278  2.139            prime    raced           3353       3824    1.10  4.6e-16  bluestein
       279  3^2.31           2p       raced            720       2878    3.99  4.9e-16  
       280  2^3.5.7          chain3   raced            420        361    0.79  3.4e-16  
       281  281              prime    raced           1730       3591    2.00  6.3e-16  rader
       282  2.3.47           2p       raced            814       4116    5.04  4.3e-16  
       283  283              prime    raced           2921       3453    1.18  7.2e-16  rader
       284  2^2.71           prime    raced           3231       3407    1.05  4.4e-16  bluestein
       285  3.5.19           2p       raced            553       2224    4.00  4.0e-16  
       286  2.11.13          flat     raced            744        976    1.27  4.9e-16  
       287  7.41             2p       raced            887       4123    4.64  3.0e-16  
       288  2^5.3^2          2p       raced            359        423    1.13  3.5e-16  
       289  17^2             2p       raced            661       3820    5.74  7.9e-16  
       290  2.5.29           2p       raced            636       2815    4.42  5.0e-16  
       291  3.97             prime    raced           3194       3062    0.95  4.9e-16  bluestein
       292  2^2.73           prime    raced           3226       2978    0.91  6.2e-16  bluestein
       293  293              prime    raced           3248       3703    1.13  5.8e-16  bluestein
       294  2.3.7^2          chain3   raced            550        599    1.07  4.3e-16  
       295  5.59             prime    raced           3204       4043    1.26  5.1e-16  bluestein
       296  2^3.37           2p       raced            701       3274    4.49  4.7e-16  
       297  3^3.11           chain3   raced            586        789    1.32  3.6e-16  
       298  2.149            prime    raced           3198       4316    1.29  4.9e-16  bluestein
       299  13.23            2p       raced            705       2839    4.01  4.7e-16  
       300  2^2.3.5^2        2p       raced            405        418    1.01  5.7e-16  
       301  7.43             2p       raced            944       4509    4.75  4.7e-16  
       302  2.151            prime    raced           3207       3324    1.02  4.5e-16  bluestein
       303  3.101            prime    raced           3245       3412    1.01  4.3e-16  bluestein
       304  2^4.19           2p       raced            496       2296    4.55  3.6e-16  
       305  5.61             prime    raced           3208       3186    0.98  5.5e-16  bluestein
       306  2.3^2.17         chain3   raced            629       2283    3.61  5.5e-16  
       307  307              prime    raced           2359       5488    2.31  1.0e-15  rader
       308  2^2.7.11         chain3   raced            550        742    1.34  4.0e-16  
       309  3.103            prime    raced           3276       4634    1.12  4.9e-16  flips differ 1.26x bluestein
       310  2.5.31           2p       raced            702       3160    4.49  4.2e-16  
       311  311              prime    raced           2840       5772    2.01  7.1e-16  rader
       312  2^3.3.13         chain3   raced            566        642    1.13  5.6e-16  
       313  313              prime    raced           2066       5521    2.67  8.4e-16  rader
       314  2.157            prime    raced           3201       4039    1.11  4.1e-16  bluestein
       315  3^2.5.7          2p       raced            477        774    1.57  4.2e-16  
       316  2^2.79           prime    raced           3192       4008    1.22  5.5e-16  bluestein
       317  317              prime    raced           3206       3410    1.05  3.9e-16  bluestein
       318  2.3.53           prime    raced           3210       5206    1.35  4.7e-16  bluestein
       319  11.29            2p       raced            819       3470    4.22  5.1e-16  
       320  2^6.5            2p       raced            365        356    0.91  3.0e-16  
       321  3.107            prime    raced           3197       4491    1.18  5.0e-16  bluestein
       322  2.7.23           flat     raced            955       3247    3.31  5.9e-16  
       323  17.19            2p       raced            781       4388    5.61  5.8e-16  
       324  2^2.3^4          2p       raced            452        613    1.35  4.9e-16  
       325  5^2.13           2p       raced            567        929    1.62  6.4e-16  
       326  2.163            prime    raced           3209       4002    1.24  5.4e-16  bluestein
       327  3.109            prime    raced           3225       3882    1.14  6.6e-16  bluestein
       328  2^3.41           2p       raced            865       4369    4.85  3.9e-16  
       329  7.47             2p       raced           1088       4616    4.22  4.5e-16  
       330  2.3.5.11         chain3   raced            604        721    1.19  4.3e-16  
       331  331              prime    raced           2238       5365    2.32  7.2e-16  rader
       332  2^2.83           prime    raced           3362       4534    1.33  3.8e-16  bluestein
       333  3^2.37           2p       raced            931       4072    4.18  4.4e-16  
       334  2.167            prime    raced           3265       6748    1.93  6.3e-16  bluestein
       335  5.67             prime    raced           3333       5763    1.71  5.9e-16  bluestein
       336  2^4.3.7          2p       raced            427        544    1.27  3.6e-16  
       337  337              prime    raced           1902       5402    2.73  6.9e-16  rader
       338  2.13^2           flat     raced            897       1364    1.51  1.2e-15  
       339  3.113            prime    raced           3220       5172    1.44  4.4e-16  bluestein
       340  2^2.5.17         chain3   raced            684       2553    3.72  5.7e-16  
       341  11.31            2p       raced            901       3941    4.27  6.2e-16  
       342  2.3^2.19         chain3   raced            749       2798    3.60  4.6e-16  
       343  7^3              chain3   raced            707       1013    1.39  3.7e-16  
       344  2^3.43           2p       raced            933       4700    5.03  4.4e-16  
       345  3.5.23           2p       raced            734       3266    4.12  4.4e-16  
       346  2.173            prime    raced           3266       4690    1.38  5.1e-16  bluestein
       347  347              prime    raced           3371       4422    1.16  5.4e-16  bluestein
       348  2^2.3.29         2p       raced            762       3658    4.80  5.5e-16  
       349  349              prime    raced           2707       4396    1.50  8.9e-16  rader
       350  2.5^2.7          chain3   raced            696        711    0.98  3.5e-16  
       351  3^3.13           2p       raced            605        902    1.49  6.1e-16  
       352  2^5.11           2p       raced            477        705    1.41  2.6e-16  
       353  353              prime    raced           1974       4488    2.23  5.2e-16  rader
       354  2.3.59           prime    raced           3423       4433    1.28  6.5e-16  bluestein
       355  5.71             prime    raced           3280       4296    1.28  4.6e-16  bluestein
       356  2^2.89           prime    raced           3361       4890    1.34  4.6e-16  bluestein
       357  3.7.17           2p       raced            678       2782    3.71  4.8e-16  
       358  2.179            prime    raced           3242       4811    1.48  5.2e-16  bluestein
       359  359              prime    raced           3238       4439    1.20  4.8e-16  bluestein
       360  2^3.3^2.5        chain3   raced            618        595    0.96  4.4e-16  
       361  19^2             2p       raced            908       5232    5.55  4.5e-16  
       362  2.181            prime    raced           3466       3799    1.06  4.6e-16  bluestein
       363  3.11^2           chain3   raced            771       1593    2.06  4.3e-16  
       364  2^2.7.13         chain3   raced            679        876    1.28  4.9e-16  
       365  5.73             prime    raced           3248       3775    1.12  4.8e-16  bluestein
       366  2.3.61           prime    raced           3232       3808    1.17  5.5e-16  bluestein
       367  367              prime    raced           3259       4936    1.30  6.5e-16  bluestein
       368  2^4.23           2p       raced            691       3440    4.03  3.9e-16  
       369  3^2.41           2p       raced           1139       5009    4.39  4.5e-16  
       370  2.5.37           2p       raced            909       4623    5.07  4.0e-16  
       371  7.53             prime    raced           3251       6119    1.87  4.4e-16  bluestein
       372  2^2.3.31         2p       raced            838       3926    4.44  4.4e-16  
       373  373              prime    raced           3251       4944    1.52  4.9e-16  bluestein
       374  2.11.17          flat     raced            985       3383    2.95  7.8e-16  
       375  3.5^3            2p       raced            577        936    1.60  5.7e-16  
       376  2^3.47           2p       raced           1064       5256    4.36  4.6e-16  
       377  13.29            2p       raced           1006       4086    4.05  5.6e-16  
       378  2.3^3.7          chain3   raced            643        761    1.14  4.8e-16  
       379  379              prime    raced           3056       4209    1.31  1.0e-15  rader
       380  2^2.5.19         chain3   raced            801       3229    3.82  4.1e-16  
       381  3.127            prime    raced           3444       4579    1.25  4.5e-16  bluestein
       382  2.191            prime    raced           3275       4161    1.17  5.0e-16  bluestein
       383  383              prime    raced           3408       4106    1.20  4.8e-16  bluestein
       384  2^7.3            2p       raced            411        412    0.87  3.5e-16  
       385  5.7.11           chain3   raced            764       1160    1.43  5.4e-16  
       386  2.193            prime    raced           3259       3955    1.16  5.4e-16  bluestein
       387  3^2.43           2p       raced           1215       5119    4.20  4.5e-16  
       388  2^2.97           prime    raced           3271       3849    1.18  4.3e-16  bluestein
       389  389              prime    raced           3267       4784    1.42  5.1e-16  bluestein
       390  2.3.5.13         chain3   raced            734        822    1.11  6.2e-16  
       391  17.23            flat     raced           1171       5716    4.75  9.8e-16  
       392  2^3.7^2          chain3   raced            668        750    1.07  3.1e-16  
       393  3.131            prime    raced           3270       5640    1.72  5.6e-16  bluestein
       394  2.197            prime    raced           3325       4375    1.17  5.5e-16  bluestein
       395  5.79             prime    raced           3268       5161    1.56  4.8e-16  bluestein
       396  2^2.3^2.11       chain3   raced            736        892    1.18  4.1e-16  
       397  397              prime    raced           2866       4895    1.67  7.9e-16  rader
       398  2.199            prime    raced           3296       4419    1.22  6.3e-16  bluestein
       399  3.7.19           2p       raced            809       3185    3.90  5.2e-16  
       400  2^4.5^2          2p       raced            509        481    0.92  3.1e-16  
       401  401              prime    raced           2318       3803    1.64  5.7e-16  rader
       402  2.3.67           prime    raced           3270       6740    2.04  5.7e-16  bluestein
       403  13.31            2p       raced           1142       4619    3.99  7.5e-16  
       404  2^2.101          prime    raced           3291       4106    1.24  5.9e-16  bluestein
       405  3^4.5            2p       raced            631       1228    1.55  5.5e-16  flips differ 1.34x
       406  2.7.29           flat     raced           1390       4149    2.71  9.4e-16  
       407  11.37            2p       raced           1162       5205    4.25  4.6e-16  
       408  2^3.3.17         chain3   raced            820       2961    3.57  5.1e-16  
       409  409              prime    raced           2957       5464    1.75  8.5e-16  rader
       410  2.5.41           2p       raced           1119       5465    4.80  4.9e-16  
       411  3.137            prime    raced           3282       6515    1.94  5.1e-16  bluestein
       412  2^2.103          prime    raced           3267       6105    1.72  7.8e-16  bluestein
       413  7.59             prime    raced           3290       5742    1.72  5.8e-16  bluestein
       414  2.3^2.23         chain3   raced           1050       3758    3.50  6.1e-16  
       415  5.83             prime    raced           3488       5676    1.52  4.7e-16  bluestein
       416  2^5.13           2p       raced            609        827    1.34  5.4e-16  
       417  3.139            prime    raced           3289       5905    1.62  4.0e-16  bluestein
       418  2.11.19          flat     raced           1159       3742    3.21  6.0e-16  
       419  419              prime    raced           3293       5274    1.49  6.0e-16  bluestein
       420  2^2.3.5.7        chain3   raced            679        715    1.05  3.5e-16  
       421  421              prime    raced           3270       5321    1.60  6.2e-16  bluestein
       422  2.211            prime    raced           3493       5497    1.34  5.7e-16  bluestein
       423  3^2.47           2p       raced           1396       6021    4.31  5.0e-16  
       424  2^3.53           prime    raced           3426       7048    2.00  4.9e-16  bluestein
       425  5^2.17           2p       raced            904       3272    3.59  5.9e-16  
       426  2.3.71           prime    raced           3275       5151    1.57  5.6e-16  bluestein
       427  7.61             prime    raced           3371       4639    1.31  4.3e-16  bluestein
       428  2^2.107          prime    raced           3270       6013    1.76  6.3e-16  bluestein
       429  3.11.13          chain3   raced            958       1544    1.61  5.6e-16  
       430  2.5.43           2p       raced           1192       6054    5.05  3.6e-16  
       431  431              prime    raced           3304       5336    1.45  5.2e-16  bluestein
       432  2^4.3^3          2p       raced            614        704    1.15  4.6e-16  
       433  433              prime    raced           2503       4556    1.75  8.3e-16  rader
       434  2.7.31           flat     raced           1493       4638    3.08  4.6e-16  
       435  3.5.29           2p       raced           1065       4646    4.36  4.8e-16  
       436  2^2.109          prime    raced           3321       4971    1.38  5.4e-16  bluestein
       437  19.23            2p       raced           1197       6836    5.50  4.7e-16  
       438  2.3.73           prime    raced           3599       4488    1.23  5.3e-16  bluestein
       439  439              prime    raced           3297       5754    1.74  5.8e-16  bluestein
       440  2^3.5.11         chain3   raced            811       1201    1.28  3.5e-16  
       441  3^2.7^2          2p       raced            763       1230    1.61  4.9e-16  
       442  2.13.17          flat     raced           1247       4264    3.42  7.5e-16  
       443  443              prime    raced           3291       5865    1.71  4.4e-16  bluestein
       444  2^2.3.37         2p       raced           1083       5650    4.85  5.2e-16  
       445  5.89             prime    raced           3467       5966    1.59  4.5e-16  bluestein
       446  2.223            prime    raced           3281       6315    1.76  5.2e-16  bluestein
       447  3.149            prime    raced           3389       6559    1.78  5.9e-16  bluestein
       448  2^6.7            chain3   raced            605        573    0.87  3.5e-16  
       449  449              prime    raced           2480       5842    2.33  6.9e-16  rader
       450  2.3^2.5^2        chain3   raced            770        820    1.07  5.1e-16  
       451  11.41            2p       raced           1418       6160    4.31  4.2e-16  
       452  2^2.113          prime    raced           3362       7108    2.01  5.8e-16  bluestein
       453  3.151            prime    raced           3308       5319    1.35  5.1e-16  bluestein
       454  2.227            prime    raced           3372       5787    1.40  4.8e-16  bluestein
       455  5.7.13           chain3   raced            970       1309    1.32  4.5e-16  
       456  2^3.3.19         chain3   raced           1007       3987    3.94  3.9e-16  
       457  457              prime    raced           3313       6333    1.22  6.4e-16  flips differ 1.45x bluestein
       458  2.229            prime    raced           3299       5640    1.60  4.6e-16  bluestein
       459  3^3.17           chain3   raced           1038       3546    3.33  5.1e-16  
       460  2^2.5.23         chain3   raced           1156       4113    3.29  4.8e-16  
       461  461              prime    raced           3345       5924    1.73  4.5e-16  bluestein
       462  2.3.7.11         chain3   raced            894       1085    1.21  4.0e-16  
       463  463              prime    raced           3219       5927    1.83  7.2e-16  rader
       464  2^4.29           2p       raced           1011       4622    4.55  3.0e-16  
       465  3.5.31           2p       raced           1173       5041    4.20  5.2e-16  
       466  2.233            prime    raced           3300       5670    1.69  5.3e-16  bluestein
       467  467              prime    raced           3307       5841    1.66  5.0e-16  bluestein
       468  2^2.3^2.13       chain3   raced            903       1013    1.08  6.3e-16  
       469  7.67             prime    raced           3298       8138    2.42  4.7e-16  bluestein
       470  2.5.47           2p       raced           1382       6963    4.21  5.3e-16  
       471  3.157            prime    raced           3481       5455    1.50  6.7e-16  bluestein
       472  2^3.59           prime    raced           3461       6097    1.71  4.7e-16  bluestein
       473  11.43            flat     raced           1726       7091    3.50  6.0e-16  
       474  2.3.79           prime    raced           3329       6491    1.79  4.9e-16  bluestein
       475  5^2.19           chain3   raced           1103       3811    3.45  5.0e-16  
       476  2^2.7.17         chain3   raced            979       3579    3.57  6.8e-16  
       477  3^2.53           prime    raced           3349       7515    2.23  4.3e-16  bluestein
       478  2.239            prime    raced           3321       5662    1.70  5.7e-16  bluestein
       479  479              prime    raced           3463       5975    1.72  4.5e-16  bluestein
       480  2^5.3.5          2p       raced            633        675    0.73  3.2e-16  flips differ 1.47x
       481  13.37            2p       raced           1415       6114    4.08  5.5e-16  
       482  2.241            prime    raced           3533       4779    1.35  4.4e-16  bluestein
       483  3.7.23           2p       raced           1081       4496    4.12  5.1e-16  
       484  2^2.11^2         chain3   raced            985       1758    1.76  3.0e-16  
       485  5.97             prime    raced           3382       5208    1.49  4.4e-16  bluestein
       486  2.3^5            chain3   raced            865       1032    1.10  5.4e-16  
       487  487              prime    raced           3326       5896    1.68  4.2e-16  bluestein
       488  2^3.61           prime    raced           3383       5377    1.55  4.1e-16  bluestein
       489  3.163            prime    raced           3373       5917    1.73  6.9e-16  bluestein
       490  2.5.7^2          chain3   raced            835       1021    1.21  4.8e-16  
       491  491              prime    raced           3034       6399    2.06  7.5e-16  rader
       492  2^2.3.41         2p       raced           1339       6606    4.91  4.7e-16  
       493  17.29            2p       raced           1433       7940    5.51  6.4e-16  
       494  2.13.19          flat     raced           1450       5245    3.46  6.9e-16  
       495  3^2.5.11         chain3   raced            959       1345    1.36  4.1e-16  
       496  2^4.31           2p       raced           1094       5282    4.81  4.9e-16  
       497  7.71             prime    raced           3338       6079    1.57  5.5e-16  bluestein
       498  2.3.83           prime    raced           3389       7422    2.04  8.9e-16  bluestein
       499  499              prime    raced           3539       6408    1.81  6.6e-16  bluestein
       500  2^2.5^3          chain3   raced            826        860    1.01  4.5e-16  
       501  3.167            prime    raced           3402       7025    2.00  5.2e-16  bluestein
       502  2.251            prime    raced           3407       5834    1.58  4.5e-16  bluestein
       503  503              prime    raced           3578       5597    1.55  3.7e-16  bluestein
       504  2^3.3^2.7        chain3   raced            871        901    1.01  3.4e-16  
       505  5.101            prime    raced           3476       5169    1.37  4.7e-16  bluestein
       506  2.11.23          flat     raced           1529       5478    3.28  6.7e-16  
       507  3.13^2           chain3   raced           1148       1804    1.55  5.2e-16  
       508  2^2.127          prime    raced           3682       6006    1.59  6.0e-16  bluestein
       509  509              prime    raced           3421       5712    1.38  5.7e-16  bluestein
       510  2.3.5.17         chain3   raced           1119       3905    3.04  7.2e-16  
       511  7.73             prime    raced           3477       5371    1.30  5.3e-16  bluestein
       512  2^9              2p       raced            521        561    1.06  3.1e-16  
       513  3^3.19           2p       raced           1058       4722    4.12  4.6e-16  
       514  2.257            prime    raced           7003       5421    0.73  5.9e-16  bluestein
       515  5.103            prime    raced           8236       8354    0.72  4.6e-16  flips differ 1.28x bluestein
       516  2^2.3.43         2p       raced           1437       7426    4.93  4.9e-16  
       517  11.47            2p       raced           1741       8008    4.50  4.6e-16  
       518  2.7.37           flat     raced           1902       6659    3.25  6.7e-16  
       519  3.173            prime    raced           8178       7515    0.79  5.4e-16  bluestein
       520  2^3.5.13         chain3   raced            926       1093    1.16  6.5e-16  
       521  521              prime    raced           3550       7464    1.88  9.9e-16  rader
       522  2.3^2.29         chain3   raced           1469       5828    3.73  3.9e-16  
       523  523              prime    raced           5194       7545    1.38  6.9e-16  rader
       524  2^2.131          prime    raced           7407       9342    1.08  5.0e-16  bluestein
       525  3.5^2.7          2p       raced            922       1435    1.34  5.1e-16  
       526  2.263            prime    raced           7500       7476    0.95  6.0e-16  bluestein
       527  17.31            2p       raced           1576       8873    5.53  7.5e-16  
       528  2^4.3.11         chain3   raced            951       1130    1.15  4.2e-16  
       529  23^2             flat     raced           1747       8856    4.91  9.0e-16  
       530  2.5.53           prime    raced           6857       8765    1.03  4.8e-16  bluestein
       531  3^2.59           prime    raced           7414       6679    0.33  4.9e-16  flips differ 2.70x bluestein
       532  2^2.7.19         chain3   raced           1155       4446    3.82  5.0e-16  
       533  13.41            2p       raced           1726       7420    4.27  5.6e-16  
       534  2.3.89           prime    raced           7607       7004    0.86  6.1e-16  bluestein
       535  5.107            prime    raced           8121       7867    0.85  5.3e-16  bluestein
       536  2^3.67           prime    raced           7231       9385    1.20  5.9e-16  bluestein
       537  3.179            prime    raced           7255       9476    1.07  5.0e-16  bluestein
       538  2.269            prime    raced           7109       9807    1.06  4.6e-16  bluestein
       539  7^2.11           chain3   raced           1165       1943    1.06  3.8e-16  flips differ 1.60x
       540  2^2.3^3.5        chain3   raced            994       1023    0.84  4.8e-16  
       541  541              prime    raced           5287       5852    1.09  6.2e-16  rader
       542  2.271            prime    raced           7345       6390    0.84  5.8e-16  bluestein
       543  3.181            prime    raced           8204       6525    0.73  4.4e-16  bluestein
       544  2^5.17           2p       raced            911       4450    4.75  6.4e-16  
       545  5.109            prime    raced           8549       6426    0.61  5.5e-16  bluestein
       546  2.3.7.13         chain3   raced           1104       1294    1.15  5.5e-16  
       547  547              prime    raced           3999      10931    2.52  1.1e-15  rader
       548  2^2.137          prime    raced           8424       8484    0.93  5.2e-16  bluestein
       549  3^2.61           prime    raced           8285       6544    0.76  5.8e-16  bluestein
       550  2.5^2.11         chain3   raced           1050       1244    0.95  4.4e-16  flips differ 1.25x
       551  19.29            2p       raced           1670       9179    5.49  4.0e-16  
       552  2^3.3.23         chain3   raced           1309       5965    4.48  5.3e-16  
       553  7.79             prime    raced           8159       8132    0.84  4.6e-16  bluestein
       554  2.277            prime    raced           6970       7247    1.01  4.6e-16  bluestein
       555  3.5.37           2p       raced           1566       7188    3.42  6.0e-16  flips differ 1.36x
       556  2^2.139          prime    raced           8211       7699    0.82  5.0e-16  bluestein
       557  557              prime    raced           7485      10056    1.20  6.3e-16  bluestein
       558  2.3^2.31         chain3   raced           1631       5972    3.63  5.9e-16  
       559  13.43            2p       raced           1970       8158    2.58  6.4e-16  flips differ 1.62x
       560  2^4.5.7          chain3   raced            912       1142    1.01  3.3e-16  
       561  3.11.17          chain3   raced           1407       5336    3.36  6.9e-16  
       562  2.281            prime    raced           9286       7905    0.81  7.3e-16  bluestein
       563  563              prime    raced           7328       9937    1.34  5.9e-16  bluestein
       564  2^2.3.47         2p       raced           1707      10045    5.48  4.7e-16  
       565  5.113            prime    raced           7406       8983    1.14  5.0e-16  bluestein
       566  2.283            prime    raced           7685       7809    0.92  5.4e-16  bluestein
       567  3^4.7            chain3   raced           1174       1585    1.33  5.6e-16  
       568  2^3.71           prime    raced           7008       7055    0.89  3.7e-16  bluestein
       569  569              prime    raced           7127       7262    0.92  5.0e-16  bluestein
       570  2.3.5.19         chain3   raced           1270       4697    3.46  4.4e-16  
       571  571              prime    raced           4288       7207    1.62  6.9e-16  rader
       572  2^2.11.13        chain3   raced           1207       1987    1.35  4.9e-16  
       573  3.191            prime    raced           8113       6333    0.76  6.0e-16  bluestein
       574  2.7.41           flat     raced           2298       7483    2.99  6.2e-16  
       575  5^2.23           2p       raced           1297       5217    4.00  4.8e-16  
       576  2^6.3^2          2p       raced            835        930    1.08  3.5e-16  
       577  577              prime    raced           3201       5991    1.84  6.8e-16  rader
       578  2.17^2           flat     raced           1732       8250    4.74  8.0e-16  
       579  3.193            prime    raced           8135       5875    0.61  4.9e-16  bluestein
       580  2^2.5.29         chain3   raced           1721       5907    3.11  4.9e-16  
       581  7.83             prime    raced           8240       8202    0.91  5.1e-16  bluestein
       582  2.3.97           prime    raced           7955       5902    0.72  4.8e-16  bluestein
       583  11.53            prime    raced           7200      10161    1.33  3.6e-16  bluestein
       584  2^3.73           prime    raced           8104       6012    0.71  4.4e-16  bluestein
       585  3^2.5.13         chain3   raced           1207       2088    1.29  5.1e-16  
       586  2.293            prime    raced           8102       7891    0.85  6.6e-16  bluestein
       587  587              prime    raced           7230       8023    0.99  4.8e-16  bluestein
       588  2^2.3.7^2        chain3   raced            970       1107    1.14  4.4e-16  
       589  19.31            2p       raced           1824      10100    5.44  5.1e-16  
       590  2.5.59           prime    raced           8108       8026    0.94  4.6e-16  bluestein
       591  3.197            prime    raced           6995       6814    0.93  4.3e-16  bluestein
       592  2^4.37           2p       raced           1419       7060    4.95  4.3e-16  
       593  593              prime    raced           5294       7785    1.32  7.2e-16  rader
       594  2.3^3.11         chain3   raced           1195       1465    1.19  5.3e-16  
       595  5.7.17           chain3   raced           1360       4540    3.32  5.1e-16  
       596  2^2.149          prime    raced           8376       9374    1.00  5.8e-16  bluestein
       597  3.199            prime    raced           8235       7101    0.80  5.3e-16  bluestein
       598  2.13.23          flat     raced           1986       6892    3.39  8.4e-16  
       599  599              prime    raced           7251       8218    1.09  5.3e-16  bluestein
       600  2^3.3.5^2        chain3   raced           1199        993    0.80  4.0e-16  
       601  601              prime    raced           5043       6321    1.23  7.0e-16  rader
       602  2.7.43           flat     raced           3446      11759    3.36  6.7e-16  
       603  3^2.67           prime    raced           7094      14797    1.49  5.2e-16  flips differ 1.38x bluestein
       604  2^2.151          prime    raced           8228       6475    0.69  5.6e-16  bluestein
       605  5.11^2           chain3   raced           1355       2796    1.86  4.2e-16  
       606  2.3.101          prime    raced           8100       6171    0.73  4.1e-16  bluestein
       607  607              prime    raced           7487      10489    1.36  6.1e-16  bluestein
       608  2^5.19           2p       raced           1061       4697    4.38  3.7e-16  
       609  3.7.29           2p       raced           1566       6471    4.07  5.1e-16  
       610  2.5.61           prime    raced           8170       6885    0.71  6.9e-16  bluestein
       611  13.47            2p       raced           2112       9314    4.40  5.3e-16  
       612  2^2.3^2.17       chain3   raced           1248       4456    3.56  6.2e-16  
       613  613              prime    raced           4915       9693    1.95  8.4e-16  rader
       614  2.307            prime    raced           7010      11608    1.53  5.1e-16  bluestein
       615  3.5.41           2p       raced           1869       9308    4.82  4.0e-16  
       616  2^3.7.11         chain3   raced           1147       1578    1.21  2.8e-16  
       617  617              prime    raced           4088       9797    2.38  6.5e-16  rader
       618  2.3.103          prime    raced           6982       9603    1.35  5.5e-16  bluestein
       619  619              prime    raced           8119       9648    1.11  5.6e-16  bluestein
       620  2^2.5.31         chain3   raced           1956       6961    3.55  5.2e-16  
       621  3^3.23           2p       raced           1419       5845    4.10  5.2e-16  
       622  2.311            prime    raced           6986      11934    1.69  5.6e-16  bluestein
       623  7.89             prime    raced           7365       8220    1.08  4.9e-16  bluestein
       624  2^4.3.13         chain3   raced           1075       1318    1.22  6.6e-16  
       625  5^4              2p       raced           1126       2092    1.61  4.4e-16  
       626  2.313            prime    raced           7227      11845    1.56  5.3e-16  bluestein
       627  3.11.19          chain3   raced           1566       5738    3.62  4.8e-16  
       628  2^2.157          prime    raced           8208       7082    0.86  4.5e-16  bluestein
       629  17.37            2p       raced           2008      11433    5.59  6.2e-16  
       630  2.3^2.5.7        chain3   raced           1224       1239    0.94  3.6e-16  
       631  631              prime    raced           4944       7214    1.44  8.2e-16  rader
       632  2^3.79           prime    raced           8276       8327    1.00  6.2e-16  bluestein
       633  3.211            prime    raced           7331       9126    1.13  4.5e-16  bluestein
       634  2.317            prime    raced           7227       7102    0.94  4.9e-16  bluestein
       635  5.127            prime    raced           6956       7761    1.05  5.7e-16  bluestein
       636  2^2.3.53         prime    raced           7969      10978    1.27  4.6e-16  bluestein
       637  7^2.13           chain3   raced           1336       2312    1.59  5.8e-16  
       638  2.11.29          flat     raced           2204       8141    3.67  1.6e-15  
       639  3^2.71           prime    raced           6982       7983    1.10  6.1e-16  bluestein
       640  2^7.5            2p       raced            798        767    0.86  4.9e-16  
       641  641              prime    raced           3369       7511    1.84  5.0e-16  flips differ 1.40x rader
       642  2.3.107          prime    raced          11412      12816    1.12  5.2e-16  bluestein
       643  643              prime    raced           8766       8777    0.88  7.2e-16  bluestein
       644  2^2.7.23         chain3   raced           1663       5718    3.21  4.0e-16  
       645  3.5.43           2p       raced           1998       9188    4.46  4.3e-16  
       646  2.17.19          flat     raced           1984       9124    4.57  1.0e-15  
       647  647              prime    raced           7281       8948    1.17  6.6e-16  bluestein
       648  2^3.3^4          chain3   raced           1131       1266    1.07  5.8e-16  
       649  11.59            prime    raced           8328      10102    0.98  4.4e-16  flips differ 1.28x bluestein
       650  2.5^2.13         chain3   raced           1278       1420    1.10  5.7e-16  
       651  3.7.31           2p       raced           1732       7110    4.09  4.4e-16  
       652  2^2.163          prime    raced           7198       8497    1.17  4.5e-16  bluestein
       653  653              prime    raced           7206      10191    1.28  5.4e-16  bluestein
       654  2.3.109          prime    raced           6976       7094    0.99  5.5e-16  bluestein
       655  5.131            prime    raced           7240       9543    1.26  5.4e-16  bluestein
       656  2^4.41           2p       raced           1770       8558    4.79  4.2e-16  
       657  3^2.73           prime    raced           6957       6865    0.96  6.4e-16  bluestein
       658  2.7.47           flat     raced           2755       9790    3.47  5.8e-16  
       659  659              prime    raced           8282       9849    1.10  5.3e-16  bluestein
       660  2^2.3.5.11       chain3   raced           1220       1453    1.13  5.1e-16  
       661  661              prime    raced           5318       9633    1.80  7.7e-16  rader
       662  2.331            prime    raced           8050      11388    1.36  7.1e-16  bluestein
       663  3.13.17          chain3   raced           1673       6577    3.90  6.6e-16  
       664  2^3.83           prime    raced           7199       9110    1.25  5.6e-16  bluestein
       665  5.7.19           chain3   raced           1574       5369    3.40  4.0e-16  
       666  2.3^2.37         chain3   raced           2062       7988    3.83  4.3e-16  
       667  23.29            flat     raced           2446      11897    4.82  1.1e-15  
       668  2^2.167          prime    raced           6992      10295    1.43  4.8e-16  bluestein
       669  3.223            prime    raced           7222      10061    1.39  6.9e-16  bluestein
       670  2.5.67           prime    raced           8141      11657    1.40  6.5e-16  bluestein
       671  11.61            prime    raced           6979       7817    1.09  5.7e-16  bluestein
       672  2^5.3.7          2p       raced           1140       1143    0.92  5.1e-16  
       673  673              prime    raced           3866      11630    2.44  6.7e-16  flips differ 1.43x rader
       674  2.337            prime    raced           9870      15589    1.58  7.2e-16  bluestein
       675  3^3.5^2          2p       raced           1275       2414    1.35  6.6e-16  flips differ 1.41x
       676  2^2.13^2         chain3   raced           1484       2352    1.58  6.0e-16  
       677  677              prime    raced           5410       9113    1.66  8.8e-16  rader
       678  2.3.113          prime    raced           8268      11232    1.34  3.9e-16  bluestein
       679  7.97             prime    raced           8226       6857    0.83  4.7e-16  bluestein
       680  2^3.5.17         chain3   raced           1431       4950    3.42  6.4e-16  
       681  3.227            prime    raced           8263       9048    1.05  6.6e-16  bluestein
       682  2.11.31          flat     raced           2422       8085    3.31  1.2e-15  
       683  683              prime    raced           7346       9348    1.25  6.0e-16  bluestein
       684  2^2.3^2.19       chain3   raced           1493       5394    3.55  4.9e-16  
       685  5.137            prime    raced           8181       8910    1.02  5.5e-16  bluestein
       686  2.7^3            flat     raced           1694       1509    0.70  5.1e-16  flips differ 1.28x
       687  3.229            prime    raced           8162       8562    1.03  5.4e-16  bluestein
       688  2^4.43           2p       raced           1925      14539    5.96  5.1e-16  flips differ 1.45x
       689  13.53            prime    raced          11550      16433    1.40  6.4e-16  bluestein
       690  2.3.5.23         chain3   raced           1644       6225    2.68  4.5e-16  flips differ 1.42x
       691  691              prime    raced           5691       9278    1.58  8.6e-16  rader
       692  2^2.173          prime    raced           7165       9998    1.38  6.2e-16  bluestein
       693  3^2.7.11         chain3   raced           1419       1924    1.34  4.7e-16  
       694  2.347            prime    raced           7386       9301    1.18  5.7e-16  bluestein
       695  5.139            prime    raced           8036       9131    1.13  4.7e-16  bluestein
       696  2^3.3.29         chain3   raced           1846       7039    3.81  4.1e-16  
       697  17.41            2p       raced           2446      13567    5.53  6.0e-16  
       698  2.349            prime    raced           8263       9519    0.95  6.1e-16  bluestein
       699  3.233            prime    raced           8255       9078    1.04  4.4e-16  bluestein
       700  2^2.5^2.7        chain3   raced           1171       1255    1.06  3.7e-16  
       701  701              prime    raced           6533       9425    1.41  9.5e-16  rader
       702  2.3^3.13         chain3   raced           1450       1666    1.14  4.8e-16  
       703  19.37            2p       raced           2341      13055    5.54  4.0e-16  
       704  2^6.11           chain3   raced           1096       1600    1.40  3.7e-16  
       705  3.5.47           2p       raced           2310      10336    4.08  4.7e-16  
       706  2.353            prime    raced           8654       9128    0.97  8.0e-16  bluestein
       707  7.101            prime    raced           8345       7296    0.87  5.3e-16  bluestein
       708  2^2.3.59         prime    raced           7852       9499    1.09  7.1e-16  bluestein
       709  709              prime    raced           8301       9208    1.07  5.2e-16  bluestein
       710  2.5.71           prime    raced           8308       9215    1.07  5.5e-16  bluestein
       711  3^2.79           prime    raced           8093       9413    1.07  5.5e-16  bluestein
       712  2^3.89           prime    raced           9034      13463    0.98  5.4e-16  flips differ 1.48x bluestein
       713  23.31            2p       raced           3617      18347    4.82  6.0e-16  
       714  2.3.7.17         chain3   raced           1590       5267    2.33  7.0e-16  flips differ 1.42x
       715  5.11.13          chain3   raced           1619       2656    1.56  5.2e-16  
       716  2^2.179          prime    raced           8285      10253    1.24  5.7e-16  bluestein
       717  3.239            prime    raced           8048       8913    1.09  6.2e-16  bluestein
       718  2.359            prime    raced           6974       9349    1.30  5.2e-16  bluestein
       719  719              prime    raced           8361       9403    1.10  5.0e-16  bluestein
       720  2^4.3^2.5        chain3   raced           1589       1188    0.70  3.9e-16  
       721  7.103            prime    raced           6992      10678    1.51  6.3e-16  bluestein
       722  2.19^2           flat     raced           2333      10965    4.60  1.0e-15  
       723  3.241            prime    raced           6995       7696    1.08  4.5e-16  bluestein
       724  2^2.181          prime    raced           8250       7656    0.92  6.5e-16  bluestein
       725  5^2.29           2p       raced           1897       7677    4.02  4.8e-16  
       726  2.3.11^2         chain3   raced           1586       2713    1.61  4.5e-16  
       727  727              prime    raced           5349      12331    2.23  7.1e-16  rader
       728  2^3.7.13         chain3   raced           1413       1673    1.15  6.7e-16  
       729  3^6              chain3   raced           1366       2101    1.44  5.2e-16  
       730  2.5.73           prime    raced           6985       8681    1.15  4.9e-16  bluestein
       731  17.43            2p       raced           2612      14503    5.41  5.2e-16  
       732  2^2.3.61         prime    raced           8530       7864    0.87  5.6e-16  bluestein
       733  733              prime    raced           8300      10840    1.16  4.9e-16  bluestein
       734  2.367            prime    raced           8054      10631    1.23  5.3e-16  bluestein
       735  3.5.7^2          chain3   raced           1376       1882    1.36  6.1e-16  
       736  2^5.23           2p       raced           1438       6545    4.29  4.0e-16  
       737  11.67            prime    raced          10309      18932    1.83  6.9e-16  bluestein
       738  2.3^2.41         chain3   raced           2487      13571    3.84  4.5e-16  flips differ 1.42x
       739  739              prime    raced           6997      10747    1.46  5.6e-16  bluestein
       740  2^2.5.37         flat     raced           2457       9078    3.66  4.8e-16  
       741  3.13.19          chain3   raced           1950       7685    3.88  4.9e-16  
       742  2.7.53           prime    raced           7014      12157    1.70  3.6e-16  bluestein
       743  743              prime    raced           8121      10790    1.23  5.3e-16  bluestein
       744  2^3.3.31         chain3   raced           2043       7882    3.85  4.5e-16  
       745  5.149            prime    raced           8338       9243    1.10  5.6e-16  bluestein
       746  2.373            prime    raced           8322      10887    1.28  4.7e-16  bluestein
       747  3^2.83           prime    raced           7430      10410    1.02  5.0e-16  flips differ 1.35x bluestein
       748  2^2.11.17        chain3   raced           2489       9543    3.60  5.7e-16  
       749  7.107            prime    raced          10215      21011    2.06  4.6e-16  bluestein
       750  2.3.5^3          chain3   raced           1356       1723    0.96  4.7e-16  flips differ 1.54x
       751  751              prime    raced           6392       8179    1.28  7.6e-16  rader
       752  2^4.47           2p       raced           2183      13117    5.97  4.4e-16  
       753  3.251            prime    raced           8074       9164    1.10  6.0e-16  bluestein
       754  2.13.29          flat     raced           2918       9578    2.82  2.1e-15  
       755  5.151            prime    raced           7573       8543    0.90  6.7e-16  bluestein
       756  2^2.3^3.7        chain3   raced           1745       1770    0.92  5.1e-16  
       757  757              prime    raced           7248       9262    1.24  4.6e-16  bluestein
       758  2.379            prime    raced           7523       8449    1.02  5.0e-16  bluestein
       759  3.11.23          chain3   raced           2195       8889    3.95  4.7e-16  
       760  2^3.5.19         chain3   raced           1625       6251    3.77  3.6e-16  
       761  761              prime    raced           6017       9470    1.54  8.0e-16  rader
       762  2.3.127          prime    raced           7020       9100    1.24  4.7e-16  bluestein
       763  7.109            prime    raced           8422       8404    0.98  8.6e-16  bluestein
       764  2^2.191          prime    raced           8312       8427    0.99  5.0e-16  bluestein
       765  3^2.5.17         chain3   raced           1755       6007    3.32  5.3e-16  
       766  2.383            prime    raced           8404       8441    0.99  4.2e-16  bluestein
       767  13.59            prime    raced           8241      11435    1.20  4.7e-16  bluestein
       768  2^8.3            2p       raced            955       1115    0.98  3.2e-16  
       769  769              prime    raced           3991       8036    1.62  5.5e-16  rader
       770  2.5.7.11         chain3   raced           1512       1885    1.17  3.4e-16  
       771  3.257            prime    raced           8473       7845    0.83  5.8e-16  bluestein
       772  2^2.193          prime    raced           7131       7691    1.06  5.1e-16  bluestein
       773  773              prime    raced           7122      10441    1.44  6.9e-16  bluestein
       774  2.3^2.43         chain3   raced           2676      10722    3.95  4.9e-16  
       775  5^2.31           2p       raced           2095       9095    4.04  6.1e-16  
       776  2^3.97           prime    raced           8513       8099    0.93  3.8e-16  bluestein
       777  3.7.37           2p       raced           2307       9621    3.96  4.9e-16  
       778  2.389            prime    raced           7645      10362    1.33  6.1e-16  bluestein
       779  19.41            2p       raced           2806      15933    5.36  4.7e-16  
       780  2^2.3.5.13       chain3   raced           1484       1743    1.13  7.1e-16  
       781  11.71            prime    raced           8416      10718    1.25  5.5e-16  bluestein
       782  2.17.23          flat     raced           2685      13814    4.00  1.9e-15  flips differ 1.38x
       783  3^3.29           2p       raced           2889      11712    3.98  4.0e-16  
       784  2^4.7^2          chain3   raced           1706       2068    1.02  5.2e-16  
       785  5.157            prime    raced           7467      14601    1.42  5.3e-16  flips differ 1.38x bluestein
       786  2.3.131          prime    raced           7940      11362    1.37  5.7e-16  bluestein
       787  787              prime    raced           7352      10220    1.38  5.7e-16  bluestein
       788  2^2.197          prime    raced           7009       9716    1.30  4.7e-16  bluestein
       789  3.263            prime    raced           8557      11238    1.27  6.3e-16  bluestein
       790  2.5.79           prime    raced           7736      13836    1.39  5.3e-16  bluestein
       791  7.113            prime    raced           7372      14720    1.95  4.9e-16  bluestein
       792  2^3.3^2.11       chain3   raced           1456       1948    1.28  4.3e-16  
       793  13.61            prime    raced           8612       9274    1.06  5.4e-16  bluestein
       794  2.397            prime    raced           7437      10744    1.38  5.4e-16  bluestein
       795  3.5.53           prime    raced           8451      13463    1.57  4.2e-16  bluestein
       796  2^2.199          prime    raced           7464       8965    1.14  6.5e-16  bluestein
       797  797              prime    raced           9027      14462    1.37  5.1e-16  bluestein
       798  2.3.7.19         chain3   raced           2650       8912    3.30  4.7e-16  
       799  17.47            2p       raced           4235      23694    5.56  5.6e-16  
       800  2^5.5^2          2p       raced           1173       1260    1.02  3.3e-16  
       801  3^2.89           prime    raced           8553      10536    1.20  5.6e-16  bluestein
       802  2.401            prime    raced           7493       8499    1.00  4.9e-16  bluestein
       803  11.73            prime    raced           7141       9294    1.28  5.6e-16  bluestein
       804  2^2.3.67         prime    raced           7229      14157    1.90  4.7e-16  bluestein
       805  5.7.23           chain3   raced           2197       7350    3.29  4.4e-16  
       806  2.13.31          flat     raced           2939       9751    3.19  9.9e-16  
       807  3.269            prime    raced           7378      11165    1.38  5.6e-16  bluestein
       808  2^3.101          prime    raced           8546       8766    0.79  4.8e-16  flips differ 1.31x bluestein
       809  809              prime    raced           8570      11648    1.28  6.1e-16  bluestein
       810  2.3^4.5          chain3   raced           1569       1769    1.05  4.6e-16  
       811  811              prime    raced           7506       9186    1.17  7.7e-16  rader
       812  2^2.7.29         chain3   raced           2922       8738    2.79  4.3e-16  
       813  3.271            prime    raced           7189       9408    1.28  5.6e-16  bluestein
       814  2.11.37          flat     raced           3035      11118    3.63  8.1e-16  
       815  5.163            prime    raced           7131      10215    1.37  6.2e-16  bluestein
       816  2^4.3.17         chain3   raced           1794       7447    3.38  5.3e-16  flips differ 1.28x
       817  19.43            2p       raced           3374      18637    5.23  5.3e-16  
       818  2.409            prime    raced           7284      11522    1.45  5.3e-16  bluestein
       819  3^2.7.13         chain3   raced           1701       2257    1.29  5.8e-16  
       820  2^2.5.41         flat     raced           3326      12639    3.75  4.1e-16  
       821  821              prime    raced           7054      11499    0.88  5.5e-16  flips differ 1.85x bluestein
       822  2.3.137          prime    raced           7391      11632    1.52  4.8e-16  bluestein
       823  823              prime    raced           7165      11892    1.27  6.1e-16  flips differ 1.34x bluestein
       824  2^3.103          prime    raced           7494      12389    1.62  6.6e-16  bluestein
       825  3.5^2.11         chain3   raced           1695       2431    1.42  5.0e-16  
       826  2.7.59           prime    raced           7122      10757    1.48  5.2e-16  bluestein
       827  827              prime    raced           7081      11301    1.57  5.0e-16  bluestein
       828  2^2.3^2.23       chain3   raced           1938       7715    3.97  5.1e-16  
       829  829              prime    raced           7140      11239    1.53  7.1e-16  bluestein
       830  2.5.83           prime    raced           7052      12551    1.69  6.7e-16  bluestein
       831  3.277            prime    raced           7084      10960    1.53  7.0e-16  bluestein
       832  2^6.13           chain3   raced           1363       1761    1.18  4.1e-16  
       833  7^2.17           chain3   raced           1916       7129    3.56  6.8e-16  
       834  2.3.139          prime    raced           9325      16323    1.25  5.6e-16  bluestein
       835  5.167            prime    raced           7384      13038    1.75  4.6e-16  bluestein
       836  2^2.11.19        chain3   raced           2003       7590    3.77  4.7e-16  
       837  3^3.31           2p       raced           2255       9265    4.08  7.7e-16  
       838  2.419            prime    raced           7079      11210    1.54  5.8e-16  bluestein
       839  839              prime    raced           8250      11321    1.34  6.0e-16  bluestein
       840  2^3.3.5.7        chain3   raced           1733       1489    0.80  3.0e-16  
       841  29^2             2p       raced           3339      16159    4.82  6.3e-16  
       842  2.421            prime    raced           7066      11404    1.56  6.8e-16  bluestein
       843  3.281            prime    raced           8679      11047    1.01  5.9e-16  flips differ 1.26x bluestein
       844  2^2.211          prime    raced           8262      11142    1.34  5.2e-16  bluestein
       845  5.13^2           chain3   raced           2003       3870    1.83  6.0e-16  
       846  2.3^2.47         chain3   raced           3064      12387    3.82  5.7e-16  
       847  7.11^2           chain3   raced           1964       4039    1.92  4.3e-16  
       848  2^4.53           prime    raced           7327      13770    1.87  4.7e-16  bluestein
       849  3.283            prime    raced           7388      10981    1.46  5.0e-16  bluestein
       850  2.5^2.17         chain3   raced           1910       6505    3.30  4.0e-16  
       851  23.37            flat     raced           3463      16694    4.80  1.5e-15  
       852  2^2.3.71         prime    raced           8219      11040    1.28  5.4e-16  bluestein
       853  853              prime    raced           7380      11754    1.42  6.1e-16  bluestein
       854  2.7.61           prime    raced           8299       9378    1.13  5.2e-16  bluestein
       855  3^2.5.19         chain3   raced           2087       7872    3.73  5.8e-16  
       856  2^3.107          prime    raced           8268      12133    1.44  6.9e-16  bluestein
       857  857              prime    raced           7112      11311    1.58  5.6e-16  bluestein
       858  2.3.11.13        chain3   raced           1933       3013    1.55  6.0e-16  
       859  859              prime    raced           6466      11406    1.74  8.7e-16  rader
       860  2^2.5.43         chain3   raced           3170      12290    3.75  5.0e-16  
       861  3.7.41           2p       raced           2683      11554    4.24  4.1e-16  
       862  2.431            prime    raced           7127      11263    1.56  6.0e-16  bluestein
       863  863              prime    raced           8262      11262    1.35  7.8e-16  bluestein
       864  2^5.3^3          2p       raced           1258       1473    1.11  5.8e-16  
       865  5.173            prime    raced           7790      12318    1.52  5.3e-16  bluestein
       866  2.433            prime    raced           7075       9843    1.38  5.7e-16  bluestein
       867  3.17^2           flat     raced           2685      12013    4.46  1.1e-15  
       868  2^2.7.31         chain3   raced           2694       9434    3.03  4.7e-16  
       869  11.79            prime    raced           7108      12617    1.71  4.9e-16  bluestein
       870  2.3.5.29         chain3   raced           2344       8963    3.82  4.9e-16  
       871  13.67            prime    raced           7117      16417    2.23  6.1e-16  bluestein
       872  2^3.109          prime    raced           7391       9565    1.29  5.5e-16  bluestein
       873  3^2.97           prime    raced           7107       9192    1.24  6.1e-16  bluestein
       874  2.19.23          flat     raced           2963      13903    4.62  8.8e-16  
       875  5^3.7            chain3   raced           1657       2537    1.33  4.1e-16  
       876  2^2.3.73         prime    raced           7426       9262    1.23  5.0e-16  bluestein
       877  877              prime    raced           7094      11985    1.68  4.3e-16  bluestein
       878  2.439            prime    raced           8404      12190    1.27  5.3e-16  bluestein
       879  3.293            prime    raced           8250      11873    1.42  4.4e-16  bluestein
       880  2^4.5.11         chain3   raced           1593       1918    1.19  3.2e-16  
       881  881              prime    raced           5861      12008    2.01  5.0e-16  rader
       882  2.3^2.7^2        chain3   raced           1925       1913    0.98  4.2e-16  
       883  883              prime    raced           7419      12328    1.59  5.0e-16  bluestein
       884  2^2.13.17        chain3   raced           2122       8488    3.97  7.6e-16  
       885  3.5.59           prime    raced           7404      11678    1.57  5.1e-16  bluestein
       886  2.443            prime    raced           7136      12213    1.61  5.3e-16  bluestein
       887  887              prime    raced           7083      11916    1.51  5.7e-16  bluestein
       888  2^3.3.37         chain3   raced           2634      12908    4.88  4.9e-16  
       889  7.127            prime    raced           7245      10502    1.24  5.8e-16  bluestein
       890  2.5.89           prime    raced           7434      12616    1.60  7.0e-16  bluestein
       891  3^4.11           chain3   raced           1833       2540    1.18  4.4e-16  
       892  2^2.223          prime    raced           7109      12403    1.52  6.3e-16  bluestein
       893  19.47            2p       raced           3407      18826    5.48  4.7e-16  
       894  2.3.149          prime    raced           7388      12247    1.63  6.8e-16  bluestein
       895  5.179            prime    raced           7401      11959    1.60  4.9e-16  bluestein
       896  2^7.7            chain3   raced           1278       1196    0.93  2.5e-16  
       897  3.13.23          chain3   raced           2535      10239    4.01  8.4e-16  
       898  2.449            prime    raced           7413      12144    1.62  5.6e-16  bluestein
       899  29.31            2p       raced           4005      17661    4.10  5.0e-16  
       900  2^2.3^2.5^2      chain3   raced           1702       1676    0.98  4.3e-16  
       901  17.53            prime    raced           7484      20207    2.63  5.4e-16  bluestein
       902  2.11.41          flat     raced           3592      12837    3.55  7.4e-16  
       903  3.7.43           2p       raced           2878      12598    4.36  5.4e-16  
       904  2^3.113          prime    raced           7134      13874    1.69  5.2e-16  bluestein
       905  5.181            prime    raced           7528       9711    1.28  4.3e-16  bluestein
       906  2.3.151          prime    raced           7392      10845    1.45  5.0e-16  bluestein
       907  907              prime    raced           8163      17821    2.16  6.1e-16  bluestein
       908  2^2.227          prime    raced           7410      11414    1.52  5.8e-16  bluestein
       909  3^2.101          prime    raced           7181       9429    1.21  4.2e-16  bluestein
       910  2.5.7.13         chain3   raced           1875       2080    1.11  5.1e-16  
       911  911              prime    raced           6298      22327    2.80  7.9e-16  rader
       912  2^4.3.19         chain3   raced           1877       7243    3.76  4.7e-16  
       913  11.83            prime    raced           7162      14038    1.85  4.6e-16  bluestein
       914  2.457            prime    raced           7400      12572    1.60  5.2e-16  bluestein
       915  3.5.61           prime    raced           8159      11097    1.34  6.6e-16  bluestein
       916  2^2.229          prime    raced           8496      11313    1.31  4.6e-16  bluestein
       917  7.131            prime    raced           7463      13428    1.59  6.7e-16  bluestein
       918  2.3^3.17         chain3   raced           2103       7042    3.26  6.6e-16  
       919  919              prime    raced           7127      17762    2.44  1.2e-15  rader
       920  2^3.5.23         chain3   raced           2128       8096    3.77  3.6e-16  
       921  3.307            prime    raced           7433      18419    2.41  7.4e-16  bluestein
       922  2.461            prime    raced           7478      12682    1.67  5.5e-16  bluestein
       923  13.71            prime    raced           7511      12540    1.63  4.9e-16  bluestein
       924  2^2.3.7.11       chain3   raced           1751       2461    1.23  4.6e-16  
       925  5^2.37           2p       raced           2648      11426    4.30  5.0e-16  
       926  2.463            prime    raced           8084      12475    1.47  7.3e-16  bluestein
       927  3^2.103          prime    raced           7463      13670    1.66  5.3e-16  bluestein
       928  2^5.29           2p       raced           2117      12043    5.57  4.5e-16  
       929  929              prime    raced           7167      17678    2.36  3.4e-16  bluestein
       930  2.3.5.31         chain3   raced           2562      10689    4.12  5.1e-16  
       931  7^2.19           chain3   raced           2246       8473    3.74  4.9e-16  
       932  2^2.233          prime    raced           7414      11372    1.52  5.8e-16  bluestein
       933  3.311            prime    raced           8368      17347    2.06  5.2e-16  bluestein
       934  2.467            prime    raced           7535      12308    1.52  5.4e-16  bluestein
       935  5.11.17          chain3   raced           2391       9370    3.92  5.6e-16  
       936  2^3.3^2.13       chain3   raced           1746       2145    1.22  6.4e-16  
       937  937              prime    raced           6392      17645    2.70  7.6e-16  rader
       938  2.7.67           prime    raced           7443      18241    2.42  6.3e-16  bluestein
       939  3.313            prime    raced           7426      17263    2.29  5.1e-16  bluestein
       940  2^2.5.47         flat     raced           3576      14381    3.48  4.1e-16  
       941  941              prime    raced           8311      12103    1.43  5.7e-16  bluestein
       942  2.3.157          prime    raced           7485      11648    0.79  5.0e-16  flips differ 1.94x bluestein
       943  23.41            flat     raced           3915      24833    5.62  1.1e-15  
       944  2^4.59           prime    raced           7498      12652    1.63  7.0e-16  bluestein
       945  3^3.5.7          chain3   raced           2236       2828    1.13  4.8e-16  
       946  2.11.43          flat     raced           4013      17907    4.31  7.5e-16  
       947  947              prime    raced           8521      18381    1.42  6.1e-16  bluestein
       948  2^2.3.79         prime    raced           7332      12851    1.75  4.4e-16  bluestein
       949  13.73            prime    raced           8340      12206    1.31  6.9e-16  bluestein
       950  2.5^2.19         chain3   raced           2182       8283    3.66  4.0e-16  
       951  3.317            prime    raced           8428      10997    1.28  5.6e-16  bluestein
       952  2^3.7.17         chain3   raced           2069       7075    3.31  5.1e-16  
       953  953              prime    raced           7269      12480    1.66  8.2e-16  rader
       954  2.3^2.53         prime    raced           7734      16992    2.19  4.7e-16  bluestein
       955  5.191            prime    raced           8661      12521    1.32  5.2e-16  bluestein
       956  2^2.239          prime    raced           7662      13232    1.58  5.7e-16  bluestein
       957  3.11.29          flat     raced           3288      11613    3.23  8.0e-16  
       958  2.479            prime    raced           7658      14850    1.64  6.3e-16  bluestein
       959  7.137            prime    raced           7505      13148    1.73  5.7e-16  bluestein
       960  2^6.3.5          2p       raced           1517       1837    1.06  4.0e-16  
       961  31^2             flat     raced           4649      20069    2.85  8.2e-16  flips differ 1.52x
       962  2.13.37          flat     raced           3831      13541    3.48  1.1e-15  
       963  3^2.107          prime    raced           7878      19976    1.73  6.3e-16  flips differ 1.59x bluestein
       964  2^2.241          prime    raced           8158      10800    1.20  5.7e-16  bluestein
       965  5.193            prime    raced           8399      10474    1.15  4.5e-16  bluestein
       966  2.3.7.23         chain3   raced           2959      11036    3.44  5.2e-16  
       967  967              prime    raced           7636      15901    1.79  6.3e-16  bluestein
       968  2^3.11^2         chain3   raced           2125       4269    1.66  4.3e-16  
       969  3.17.19          chain3   raced           2704      15039    4.22  6.5e-16  flips differ 1.38x
       970  2.5.97           prime    raced           9224      11745    1.13  5.5e-16  bluestein
       971  971              prime    raced           7570      15548    1.97  6.6e-16  bluestein
       972  2^2.3^5          chain3   raced           1692       2296    1.24  4.9e-16  
       973  7.139            prime    raced           8410      16060    1.74  5.2e-16  bluestein
       974  2.487            prime    raced           7423      12810    1.63  5.7e-16  bluestein
       975  3.5^2.13         chain3   raced           2141       2899    1.31  5.9e-16  
       976  2^4.61           prime    raced           8462      11094    1.19  6.2e-16  bluestein
       977  977              prime    raced           7127      13768    1.72  5.4e-16  bluestein
       978  2.3.163          prime    raced           7595      11634    1.51  7.4e-16  bluestein
       979  11.89            prime    raced           8368      14861    1.68  6.3e-16  bluestein
       980  2^2.5.7^2        flat     raced           2134       1989    0.88  4.2e-16  
       981  3^2.109          prime    raced           8183      10837    1.29  8.2e-16  bluestein
       982  2.491            prime    raced           8301      13580    1.62  5.3e-16  bluestein
       983  983              prime    raced           7625      13024    1.44  6.1e-16  bluestein
       984  2^3.3.41         chain3   raced           3194      14421    4.16  4.6e-16  
       985  5.197            prime    raced           7503      11168    1.41  5.9e-16  bluestein
       986  2.17.29          flat     raced           3645      16601    3.03  1.6e-15  flips differ 1.48x
       987  3.7.47           2p       raced           3402      17206    4.95  4.1e-16  
       988  2^2.13.19        chain3   raced           2468       9274    3.71  5.0e-16  
       989  23.43            flat     raced           4206      21365    4.90  1.0e-15  
       990  2.3^2.5.11       chain3   raced           1926       2375    1.23  4.8e-16  
       991  991              prime    raced           6607      12922    1.92  6.6e-16  rader
       992  2^5.31           2p       raced           2316      10629    4.58  4.9e-16  
       993  3.331            prime    raced           8420      17197    1.94  5.2e-16  bluestein
       994  2.7.71           prime    raced           7389      14403    1.86  4.7e-16  bluestein
       995  5.199            prime    raced           8268      11146    1.16  5.9e-16  bluestein
       996  2^2.3.83         prime    raced           8118      14012    1.72  6.5e-16  bluestein
       997  997              prime    raced           7447      13238    1.73  5.9e-16  bluestein
       998  2.499            prime    raced           8353      13539    1.56  6.2e-16  bluestein
       999  3^3.37           2p       raced           4078      17574    4.25  5.9e-16  
      1000  2^3.5^3          chain3   raced           2402       2484    1.03  4.6e-16  
      1001  7.11.13          chain3   raced           2303       3742    1.62  4.7e-16  
      1002  2.3.167          prime    raced           8233      14647    1.76  6.0e-16  bluestein
      1003  17.59            prime    raced           7448      20924    2.71  4.7e-16  bluestein
      1004  2^2.251          prime    raced           8116      12300    1.44  4.6e-16  bluestein
      1005  3.5.67           prime    raced           7737      18784    2.33  5.1e-16  bluestein
      1006  2.503            prime    raced           7378      12242    1.64  4.9e-16  bluestein
      1007  19.53            prime    raced           7729      23183    2.87  5.4e-16  bluestein
      1008  2^4.3^2.7        chain3   raced           2059       1825    0.87  4.4e-16  
      1009  1009             prime    raced           7700      18222    2.25  7.9e-16  rader
      1010  2.5.101          prime    raced           7152      10679    1.46  5.2e-16  bluestein
      1011  3.337            prime    raced           8133      17510    2.08  6.1e-16  bluestein
      1012  2^2.11.23        chain3   raced           2737      10414    3.76  4.8e-16  
      1013  1013             prime    raced           7243      18269    2.46  5.8e-16  bluestein
      1014  2.3.13^2         chain3   raced           2378       3585    1.46  5.8e-16  
      1015  5.7.29           chain3   raced           3255      10712    2.74  4.2e-16  
      1016  2^3.127          prime    raced           7463      12848    1.53  5.5e-16  bluestein
      1017  3^2.113          prime    raced           7217      16361    2.23  5.8e-16  bluestein
      1018  2.509            prime    raced           7152      11843    1.64  6.0e-16  bluestein
      1019  1019             prime    raced           7447      12578    1.67  5.9e-16  bluestein
      1020  2^2.3.5.17       chain3   raced           2182       8529    3.79  4.7e-16  
      1021  1021             prime    raced           7182      12296    1.71  5.2e-16  bluestein
      1022  2.7.73           prime    raced           7224      11846    1.64  6.6e-16  bluestein
      1023  3.11.31          chain3   raced           3258      13555    3.81  5.3e-16  
      1024  2^10             2p       raced           1303       1251    0.95  2.5e-16  
      1025  5^2.41           2p       raced           3225      14966    4.62  4.5e-16  
      1026  2.3^3.19         chain3   raced           2460       9337    3.50  5.4e-16  
      1027  13.79            prime    raced          15414      14495    0.92  4.8e-16  bluestein
      1028  2^2.257          prime    raced          15041      10104    0.67  4.6e-16  bluestein
      1029  3.7^3            chain3   raced           1980       3664    1.80  3.9e-16  
      1030  2.5.103          prime    raced          15224      15119    0.96  6.2e-16  bluestein
      1031  1031             prime    raced          14926      14591    0.84  5.6e-16  bluestein
      1032  2^3.3.43         chain3   raced           3419      14932    4.27  5.5e-16  
      1033  1033             prime    raced          11487      15175    1.23  8.2e-16  rader
      1034  2.11.47          flat     raced           4386      17140    3.84  7.6e-16  
      1035  3^2.5.23         chain3   raced           2739       9638    3.40  4.3e-16  
      1036  2^2.7.37         chain3   raced           3433      12732    3.70  4.3e-16  
      1037  17.61            prime    raced          14884      17515    0.96  4.5e-16  bluestein
      1038  2.3.173          prime    raced          14980      14421    0.94  6.3e-16  bluestein
      1039  1039             prime    raced          15094      15378    0.89  5.7e-16  bluestein
      1040  2^4.5.13         chain3   raced           1948       2225    1.13  5.1e-16  
      1041  3.347            prime    raced          14952      13941    0.82  5.1e-16  bluestein
      1042  2.521            prime    raced          15003      15284    0.98  6.5e-16  bluestein
      1043  7.149            prime    raced          14938      15492    0.96  4.9e-16  bluestein
      1044  2^2.3^2.29       chain3   raced           2746      10791    3.85  7.7e-16  
      1045  5.11.19          chain3   raced           2704       9890    3.52  5.0e-16  
      1046  2.523            prime    raced          15036      15845    0.98  5.5e-16  bluestein
      1047  3.349            prime    raced          16446      13943    0.83  6.2e-16  bluestein
      1048  2^3.131          prime    raced          14928      15351    0.96  6.0e-16  bluestein
      1049  1049             prime    raced          15081      14462    0.92  5.6e-16  bluestein
      1050  2.3.5^2.7        chain3   raced           1916       2066    1.04  4.2e-16  
      1051  1051             prime    raced           8560      14907    1.49  1.0e-15  rader
      1052  2^2.263          prime    raced          15346      14553    0.94  5.1e-16  bluestein
      1053  3^4.13           chain3   raced           2228       2947    1.23  8.0e-16  
      1054  2.17.31          flat     raced           4024      18241    4.52  1.6e-15  
      1055  5.211            prime    raced          14931      14488    0.85  6.0e-16  bluestein
      1056  2^5.3.11         chain3   raced           1960       2597    1.32  4.8e-16  
      1057  7.151            prime    raced          15191      11651    0.74  6.5e-16  bluestein
      1058  2.23^2           flat     raced           3963      19134    4.55  1.4e-15  
      1059  3.353            prime    raced          15395      14815    0.91  4.8e-16  bluestein
      1060  2^2.5.53         prime    raced          17464      18510    0.91  4.5e-16  bluestein
      1061  1061             prime    raced          15196      14582    0.88  5.8e-16  bluestein
      1062  2.3^2.59         prime    raced          15694      14328    0.85  5.3e-16  bluestein
      1063  1063             prime    raced          17439      14768    0.73  6.5e-16  bluestein
      1064  2^3.7.19         chain3   raced           2425       8363    3.43  3.6e-16  
      1065  3.5.71           prime    raced          15056      14584    0.80  5.3e-16  bluestein
      1066  2.13.41          flat     raced           4433      16478    3.55  1.1e-15  
      1067  11.97            prime    raced          15239      12218    0.80  5.6e-16  bluestein
      1068  2^2.3.89         prime    raced          15689      14495    0.87  6.4e-16  bluestein
      1069  1069             prime    raced          15788      15399    0.92  5.7e-16  bluestein
      1070  2.5.107          prime    raced          15485      16046    0.96  5.3e-16  bluestein
      1071  3^2.7.17         chain3   raced           2477       8505    3.29  4.4e-16  
      1072  2^4.67           prime    raced          15270      19330    1.21  5.1e-16  bluestein
      1073  29.37            flat     raced           4535      22726    3.95  1.1e-15  flips differ 1.26x
      1074  2.3.179          prime    raced          15781      15311    0.91  5.3e-16  bluestein
      1075  5^2.43           2p       raced           3452      15415    4.44  4.7e-16  
      1076  2^2.269          prime    raced          15076      15056    0.97  5.5e-16  bluestein
      1077  3.359            prime    raced          15255      14085    0.87  4.5e-16  bluestein
      1078  2.7^2.11         flat     raced           2896       3923    1.31  6.1e-16  
      1079  13.83            prime    raced          15154      16100    1.04  5.6e-16  bluestein
      1080  2^3.3^3.5        chain3   raced           2093       2013    0.93  5.7e-16  
      1081  23.47            2p       raced           4746      25143    5.15  6.0e-16  
      1082  2.541            prime    raced          15085      12890    0.84  6.4e-16  bluestein
      1083  3.19^2           chain3   raced           3153      17107    5.30  3.8e-16  
      1084  2^2.271          prime    raced          15084      12958    0.84  5.4e-16  bluestein
      1085  5.7.31           chain3   raced           3632      11973    2.63  5.2e-16  
      1086  2.3.181          prime    raced          15449      11587    0.74  5.3e-16  bluestein
      1087  1087             prime    raced          15072      22280    1.33  6.6e-16  bluestein
      1088  2^6.17           chain3   raced           2100       7968    3.76  5.4e-16  
      1089  3^2.11^2         chain3   raced           2527       4176    1.44  4.4e-16  
      1090  2.5.109          prime    raced          15090      11975    0.79  4.6e-16  bluestein
      1091  1091             prime    raced          14973      22744    1.43  7.5e-16  bluestein
      1092  2^2.3.7.13       chain3   raced           2121       2950    1.34  5.0e-16  
      1093  1093             prime    raced          10070      22965    1.93  1.1e-15  rader
      1094  2.547            prime    raced          15842      20906    1.22  6.8e-16  bluestein
      1095  3.5.73           prime    raced          15057      12200    0.76  4.6e-16  bluestein
      1096  2^3.137          prime    raced          15160      15022    0.86  6.1e-16  bluestein
      1097  1097             prime    raced          14957      16994    1.04  6.4e-16  bluestein
      1098  2.3^2.61         prime    raced          14989      12012    0.73  5.4e-16  bluestein
      1099  7.157            prime    raced          15040      13892    0.77  5.3e-16  bluestein
      1100  2^2.5^2.11       chain3   raced           2109       2702    1.20  4.0e-16  
      1101  3.367            prime    raced          14994      16991    1.01  5.4e-16  bluestein
      1102  2.19.29          flat     raced           4219      20706    4.77  1.4e-15  
      1103  1103             prime    raced          15159      15813    1.04  5.3e-16  bluestein
      1104  2^4.3.23         chain3   raced           2553      11966    4.64  5.4e-16  
      1105  5.13.17          chain3   raced           3015      12801    3.51  7.7e-16  flips differ 1.38x
      1106  2.7.79           prime    raced          21162      20801    0.98  5.2e-16  bluestein
      1107  3^3.41           2p       raced           4991      21205    4.23  5.2e-16  
      1108  2^2.277          prime    raced          14958      14563    0.68  4.6e-16  flips differ 1.44x bluestein
      1109  1109             prime    raced          14936      15682    1.04  6.3e-16  bluestein
      1110  2.3.5.37         chain3   raced           3544      13536    3.49  6.8e-16  
      1111  11.101           prime    raced          14883      12738    0.80  5.4e-16  bluestein
      1112  2^3.139          prime    raced          14894      14235    0.94  5.5e-16  bluestein
      1113  3.7.53           prime    raced          14960      18221    1.04  3.9e-16  bluestein
      1114  2.557            prime    raced          15382      19907    1.22  5.2e-16  bluestein
      1115  5.223            prime    raced          14964      16575    1.01  6.5e-16  bluestein
      1116  2^2.3^2.31       chain3   raced           3039      13923    4.41  5.0e-16  
      1117  1117             prime    raced          11058      15716    1.26  9.5e-16  rader
      1118  2.13.43          flat     raced           4657      17025    3.57  1.0e-15  
      1119  3.373            prime    raced          14940      16151    1.07  6.2e-16  bluestein
      1120  2^5.5.7          chain3   raced           2166       1968    0.88  3.8e-16  
      1121  19.59            prime    raced          15204      22551    1.45  4.6e-16  bluestein
      1122  2.3.11.17        chain3   raced           2729      10128    3.70  5.4e-16  
      1123  1123             prime    raced           9064      16011    1.74  8.4e-16  rader
      1124  2^2.281          prime    raced          15160      14893    0.96  5.5e-16  bluestein
      1125  3^2.5^3          chain3   raced           2786       3163    1.13  5.9e-16  
      1126  2.563            prime    raced          14979      20554    1.30  5.2e-16  bluestein
      1127  7^2.23           chain3   raced           3110      10337    3.08  5.8e-16  
      1128  2^3.3.47         chain3   raced           3926      16596    4.15  5.3e-16  
      1129  1129             prime    raced          13158      14751    0.99  1.1e-15  rader
      1130  2.5.113          prime    raced          14914      17643    1.16  5.2e-16  bluestein
      1131  3.13.29          chain3   raced           3894      13305    3.30  7.1e-16  
      1132  2^2.283          prime    raced          15410      15446    0.97  5.1e-16  bluestein
      1133  11.103           prime    raced          17476      34779    1.60  5.0e-16  bluestein
      1134  2.3^4.7          chain3   raced           3122       3629    1.14  5.7e-16  
      1135  5.227            prime    raced          15045      14269    0.66  5.0e-16  flips differ 1.43x bluestein
      1136  2^4.71           prime    raced          14974      14054    0.89  4.8e-16  bluestein
      1137  3.379            prime    raced          15151      12863    0.78  4.7e-16  bluestein
      1138  2.569            prime    raced          14896      15223    0.92  3.9e-16  bluestein
      1139  17.67            prime    raced          15048      30974    1.98  5.4e-16  bluestein
      1140  2^2.3.5.19       chain3   raced           2551       8839    3.37  4.9e-16  
      1141  7.163            prime    raced          14923      13712    0.91  6.2e-16  bluestein
      1142  2.571            prime    raced          15400      15156    0.98  4.9e-16  bluestein
      1143  3^2.127          prime    raced          15266      13912    0.74  4.3e-16  bluestein
      1144  2^3.11.13        chain3   raced           2517       3971    1.57  6.7e-16  
      1145  5.229            prime    raced          14988      14665    0.94  4.4e-16  bluestein
      1146  2.3.191          prime    raced          15443      12776    0.70  4.3e-16  bluestein
      1147  31.37            flat     raced           4958      25236    4.88  8.4e-16  
      1148  2^2.7.41         chain3   raced           4087      15178    3.65  4.1e-16  
      1149  3.383            prime    raced          15246      12970    0.82  6.5e-16  bluestein
      1150  2.5^2.23         chain3   raced           2861      10195    3.56  4.6e-16  
      1151  1151             prime    raced           9550      14548    1.51  7.5e-16  rader
      1152  2^7.3^2          chain3   raced           2242       1971    0.84  3.9e-16  
      1153  1153             prime    raced           6584      12239    1.84  9.3e-16  rader
      1154  2.577            prime    raced          15029      12934    0.85  4.4e-16  bluestein
      1155  3.5.7.11         chain3   raced           2422       3443    1.42  3.5e-16  
      1156  2^2.17^2         chain3   raced           3026      16155    5.24  6.2e-16  
      1157  13.89            prime    raced          15085      19809    1.29  4.8e-16  bluestein
      1158  2.3.193          prime    raced          15010      12360    0.82  3.9e-16  bluestein
      1159  19.61            prime    raced          14985      20134    1.28  4.8e-16  bluestein
      1160  2^3.5.29         chain3   raced           3036      12169    3.92  5.2e-16  
      1161  3^3.43           2p       raced           3766      16759    4.39  5.8e-16  
      1162  2.7.83           prime    raced          18038      17024    0.93  6.1e-16  bluestein
      1163  1163             prime    raced          15023      15373    1.01  4.8e-16  bluestein
      1164  2^2.3.97         prime    raced          15002      12068    0.78  4.9e-16  bluestein
      1165  5.233            prime    raced          16576      18085    0.93  5.1e-16  flips differ 1.28x bluestein
      1166  2.11.53          prime    raced          15135      28625    1.23  5.0e-16  flips differ 1.55x bluestein
      1167  3.389            prime    raced          15038      15288    1.01  5.2e-16  bluestein
      1168  2^4.73           prime    raced          15237      12352    0.80  5.0e-16  bluestein
      1169  7.167            prime    raced          15205      17043    1.06  5.5e-16  bluestein
      1170  2.3^2.5.13       chain3   raced           2487       2954    1.12  5.5e-16  
      1171  1171             prime    raced           8026      15249    1.88  7.8e-16  rader
      1172  2^2.293          prime    raced          15766      15548    0.96  5.6e-16  bluestein
      1173  3.17.23          chain3   raced           3581      18162    5.04  6.3e-16  
      1174  2.587            prime    raced          14998      17746    1.06  6.2e-16  bluestein
      1175  5^2.47           2p       raced           4322      23915    4.57  4.5e-16  flips differ 1.31x
      1176  2^3.3.7^2        chain3   raced           2996       3265    1.09  3.7e-16  
      1177  11.107           prime    raced          15312      21884    1.18  6.5e-16  flips differ 1.41x bluestein
      1178  2.19.31          flat     raced           4585      22091    4.61  1.2e-15  
      1179  3^2.131          prime    raced          15199      18364    1.18  4.9e-16  bluestein
      1180  2^2.5.59         prime    raced          15432      15042    0.97  5.2e-16  bluestein
      1181  1181             prime    raced          15044      15214    1.00  5.0e-16  bluestein
      1182  2.3.197          prime    raced          15310      13289    0.81  5.3e-16  bluestein
      1183  7.13^2           chain3   raced           2921       5679    1.91  7.5e-16  
      1184  2^5.37           2p       raced           3085      14465    4.68  4.0e-16  
      1185  3.5.79           prime    raced          15172      17035    1.09  5.4e-16  bluestein
      1186  2.593            prime    raced          14962      15962    0.90  5.5e-16  bluestein
      1187  1187             prime    raced          15564      15291    0.97  6.5e-16  bluestein
      1188  2^2.3^3.11       chain3   raced           2257       3217    1.41  5.5e-16  
      1189  29.41            flat     raced           5333      26145    4.88  1.3e-15  
      1190  2.5.7.17         chain3   raced           2696       9127    3.34  6.3e-16  
      1191  3.397            prime    raced          16983      15489    0.87  5.8e-16  bluestein
      1192  2^3.149          prime    raced          14990      15068    1.00  5.0e-16  bluestein
      1193  1193             prime    raced          15047      15965    0.96  6.0e-16  bluestein
      1194  2.3.199          prime    raced          15537      14422    0.88  4.5e-16  bluestein
      1195  5.239            prime    raced          15266      15812    0.88  5.1e-16  bluestein
      1196  2^2.13.23        chain3   raced           3323      12530    3.73  6.7e-16  
      1197  3^2.7.19         chain3   raced           2950      10146    3.12  4.8e-16  
      1198  2.599            prime    raced          15024      16417    0.99  6.9e-16  bluestein
      1199  11.109           prime    raced          15015      14892    0.95  5.7e-16  bluestein
      1200  2^4.3.5^2        chain3   raced           2834       2213    0.71  4.0e-16  
      1201  1201             prime    raced          10058      13151    1.27  8.6e-16  rader
      1202  2.601            prime    raced          17075      16936    0.84  4.8e-16  bluestein
      1203  3.401            prime    raced          21214      18114    0.85  4.1e-16  bluestein
      1204  2^2.7.43         chain3   raced           4347      16423    3.74  4.6e-16  
      1205  5.241            prime    raced          15020      12183    0.81  5.0e-16  bluestein
      1206  2.3^2.67         prime    raced          15147      24240    1.59  5.8e-16  bluestein
      1207  17.71            prime    raced          15033      22479    1.47  5.6e-16  bluestein
      1208  2^3.151          prime    raced          14972      12965    0.82  5.5e-16  bluestein
      1209  3.13.31          chain3   raced           3921      16004    4.04  6.5e-16  
      1210  2.5.11^2         chain3   raced           2701       4350    1.61  3.5e-16  
      1211  7.173            prime    raced          15012      19366    1.19  6.2e-16  bluestein
      1212  2^2.3.101        prime    raced          15285      12779    0.77  3.9e-16  bluestein
      1213  1213             prime    raced          15068      19147    1.22  7.5e-16  bluestein
      1214  2.607            prime    raced          17964      20943    1.11  6.0e-16  bluestein
      1215  3^5.5            chain3   raced           3008       3231    1.06  3.9e-16  
      1216  2^6.19           2p       raced           2369      10104    4.05  4.3e-16  
      1217  1217             prime    raced           8306      18669    2.21  7.1e-16  rader
      1218  2.3.7.29         chain3   raced           3526      12655    3.51  5.3e-16  
      1219  23.53            prime    raced          18813      38777    1.83  5.7e-16  bluestein
      1220  2^2.5.61         prime    raced          15138      19655    0.92  4.9e-16  flips differ 1.40x bluestein
      1221  3.11.37          flat     raced           4730      21011    4.40  1.8e-15  
      1222  2.13.47          flat     raced           5371      20659    3.83  2.4e-15  
      1223  1223             prime    raced          15126      18427    1.18  4.6e-16  bluestein
      1224  2^3.3^2.17       chain3   raced           2726       9036    3.25  4.9e-16  
      1225  5^2.7^2          chain3   raced           2341       3633    1.39  4.0e-16  
      1226  2.613            prime    raced          15028      20661    1.35  5.4e-16  bluestein
      1227  3.409            prime    raced          14934      17196    1.11  6.8e-16  bluestein
      1228  2^2.307          prime    raced          15133      23189    1.49  4.7e-16  bluestein
      1229  1229             prime    raced          16495      22106    1.11  4.8e-16  flips differ 1.32x bluestein
      1230  2.3.5.41         chain3   raced           5706      25604    4.48  4.9e-16  
      1231  1231             prime    raced          14549      18378    0.91  9.7e-16  flips differ 1.39x rader
      1232  2^4.7.11         chain3   raced           2185       3066    1.40  3.5e-16  
      1233  3^2.137          prime    raced          14992      16368    1.05  4.5e-16  bluestein
      1234  2.617            prime    raced          15091      19805    1.29  6.4e-16  bluestein
      1235  5.13.19          chain3   raced           3277      13013    3.92  6.3e-16  
      1236  2^2.3.103        prime    raced          15094      18301    1.20  6.7e-16  bluestein
      1237  1237             prime    raced          15042      18586    1.21  6.7e-16  bluestein
      1238  2.619            prime    raced          15231      19530    1.25  5.0e-16  bluestein
      1239  3.7.59           prime    raced          15061      16487    1.08  4.4e-16  bluestein
      1240  2^3.5.31         chain3   raced           3320      13431    3.99  4.5e-16  
      1241  17.73            prime    raced          15151      21392    1.35  5.5e-16  bluestein
      1242  2.3^3.23         chain3   raced           3241      11482    3.47  6.1e-16  
      1243  11.113           prime    raced          15032      20679    1.36  5.9e-16  bluestein
      1244  2^2.311          prime    raced          15084      24124    1.50  5.7e-16  bluestein
      1245  3.5.83           prime    raced          15428      18309    1.14  5.1e-16  bluestein
      1246  2.7.89           prime    raced          15924      17319    1.07  4.8e-16  bluestein
      1247  29.43            flat     raced           5664      28866    4.92  1.5e-15  
      1248  2^5.3.13         chain3   raced           2311       2801    1.21  5.1e-16  
      1249  1249             prime    raced           8059      18605    2.10  1.0e-15  rader
      1250  2.5^4            chain3   raced           2311       2799    1.09  5.1e-16  
      1251  3^2.139          prime    raced          15330      16425    1.05  5.0e-16  bluestein
      1252  2^2.313          prime    raced          15069      22972    1.48  7.4e-16  bluestein
      1253  7.179            prime    raced          15164      17063    1.12  6.2e-16  bluestein
      1254  2.3.11.19        chain3   raced           3148      11378    3.59  5.0e-16  
      1255  5.251            prime    raced          15013      14305    0.94  4.9e-16  bluestein
      1256  2^3.157          prime    raced          21296      20004    0.93  6.2e-16  bluestein
      1257  3.419            prime    raced          25278      23778    0.93  6.6e-16  bluestein
      1258  2.17.37          flat     raced           5014      24680    4.60  1.1e-15  
      1259  1259             prime    raced          15158      16158    1.06  5.0e-16  bluestein
      1260  2^2.3^2.5.7      chain3   raced           2696       2444    0.87  4.4e-16  
      1261  13.97            prime    raced          15046      14239    0.93  3.9e-16  bluestein
      1262  2.631            prime    raced          15354      14883    0.95  4.6e-16  bluestein
      1263  3.421            prime    raced          15087      16879    1.03  6.0e-16  bluestein
      1264  2^4.79           prime    raced          15278      18904    1.10  4.6e-16  bluestein
      1265  5.11.23          flat     raced           3927      14666    3.67  8.0e-16  
      1266  2.3.211          prime    raced          15178      17055    1.10  7.3e-16  bluestein
      1267  7.181            prime    raced          15136      14184    0.93  6.0e-16  bluestein
      1268  2^2.317          prime    raced          15919      14633    0.87  4.9e-16  bluestein
      1269  3^3.47           2p       raced           4307      18975    4.39  6.4e-16  
      1270  2.5.127          prime    raced          15119      15021    0.98  6.4e-16  bluestein
      1271  31.41            flat     raced           5927      29441    4.81  1.1e-15  
      1272  2^3.3.53         prime    raced          16188      25320    1.24  4.3e-16  flips differ 1.43x bluestein
      1273  19.67            prime    raced          21448      46986    2.19  5.6e-16  bluestein
      1274  2.7^2.13         flat     raced           3575       4561    1.27  5.6e-16  
      1275  3.5^2.17         chain3   raced           3111      11167    3.47  5.4e-16  
      1276  2^2.11.29        chain3   raced           3871      14853    3.83  4.3e-16  
      1277  1277             prime    raced          13079      16764    1.17  6.6e-16  rader
      1278  2.3^2.71         prime    raced          15020      16043    1.05  4.8e-16  bluestein
      1279  1279             prime    raced          15122      16236    0.91  6.4e-16  bluestein
      1280  2^8.5            chain3   raced           2036       1930    0.81  3.8e-16  
      1281  3.7.61           prime    raced          15130      14283    0.90  3.8e-16  bluestein
      1282  2.641            prime    raced          16183      14247    0.57  5.0e-16  flips differ 1.62x bluestein
      1283  1283             prime    raced          15017      18922    1.22  5.6e-16  bluestein
      1284  2^2.3.107        prime    raced          15154      18838    1.21  5.8e-16  bluestein
      1285  5.257            prime    raced          15238      13161    0.83  5.7e-16  bluestein
      1286  2.643            prime    raced          15106      18117    1.18  6.2e-16  bluestein
      1287  3^2.11.13        chain3   raced           3137       5133    1.57  5.8e-16  
      1288  2^3.7.23         chain3   raced           3230      11456    3.53  4.5e-16  
      1289  1289             prime    raced          10630      19617    1.81  7.1e-16  rader
      1290  2.3.5.43         chain3   raced           4253      17597    4.11  5.8e-16  
      1291  1291             prime    raced          14615      18954    1.24  9.2e-16  rader
      1292  2^2.17.19        chain3   raced           3494      18648    5.31  6.0e-16  
      1293  3.431            prime    raced          15131      17369    1.00  5.8e-16  bluestein
      1294  2.647            prime    raced          15218      18771    1.20  6.1e-16  bluestein
      1295  5.7.37           chain3   raced           5398      23384    3.49  4.8e-16  
      1296  2^4.3^4          chain3   raced           3944       3562    0.89  5.2e-16  
      1297  1297             prime    raced          11180      14344    1.21  9.9e-16  rader
      1298  2.11.59          prime    raced          15152      21215    1.35  5.6e-16  bluestein
      1299  3.433            prime    raced          15115      14773    0.95  7.8e-16  bluestein
      1300  2^2.5^2.13       chain3   raced           2556       2978    1.13  5.4e-16  
      1301  1301             prime    raced          13138      19153    1.39  8.1e-16  rader
      1302  2.3.7.31         chain3   raced           3918      14354    3.64  5.7e-16  
      1303  1303             prime    raced          12762      18396    1.40  8.1e-16  rader
      1304  2^3.163          prime    raced          15924      17338    0.90  5.3e-16  flips differ 1.35x bluestein
      1305  3^2.5.29         chain3   raced           5487      20566    3.72  5.6e-16  
      1306  2.653            prime    raced          15262      20971    1.03  5.5e-16  flips differ 1.43x bluestein
      1307  1307             prime    raced          15141      18505    1.21  5.7e-16  bluestein
      1308  2^2.3.109        prime    raced          15096      14685    0.97  4.9e-16  bluestein
      1309  7.11.17          chain3   raced           3288      12846    3.87  5.6e-16  
      1310  2.5.131          prime    raced          17622      18883    1.06  6.2e-16  bluestein
      1311  3.19.23          chain3   raced           4122      21124    5.01  5.6e-16  
      1312  2^5.41           2p       raced           3684      17086    4.59  5.3e-16  
      1313  13.101           prime    raced          15177      15075    0.95  4.3e-16  bluestein
      1314  2.3^2.73         prime    raced          15069      14480    0.93  5.3e-16  bluestein
      1315  5.263            prime    raced          15167      18255    1.14  5.6e-16  bluestein
      1316  2^2.7.47         chain3   raced           5004      22355    4.10  5.6e-16  
      1317  3.439            prime    raced          15201      18367    1.17  6.1e-16  bluestein
      1318  2.659            prime    raced          15215      20276    1.29  4.8e-16  bluestein
      1319  1319             prime    raced          15170      18900    1.07  6.2e-16  bluestein
      1320  2^3.3.5.11       chain3   raced           2653       3110    1.17  5.1e-16  
      1321  1321             prime    raced           9818      18657    1.88  6.9e-16  rader
      1322  2.661            prime    raced          15043      19978    1.28  5.5e-16  bluestein
      1323  3^3.7^2          chain3   raced           2659       4666    1.65  4.6e-16  
      1324  2^2.331          prime    raced          15331      22537    1.35  6.4e-16  bluestein
      1325  5^2.53           prime    raced          15072      23027    1.45  4.1e-16  bluestein
      1326  2.3.13.17        chain3   raced           3490      12060    3.23  7.1e-16  
      1327  1327             prime    raced          10922      18522    1.65  9.3e-16  rader
      1328  2^4.83           prime    raced          15213      22319    1.39  5.0e-16  bluestein
      1329  3.443            prime    raced          18192      18398    0.91  7.0e-16  bluestein
      1330  2.5.7.19         chain3   raced           3106      10487    3.31  4.1e-16  
      1331  11^3             chain3   raced           3336       6700    2.00  4.9e-16  
      1332  2^2.3^2.37       chain3   raced           5591      22882    4.04  5.1e-16  
      1333  31.43            flat     raced           6204      43610    4.91  1.5e-15  flips differ 1.43x
      1334  2.23.29          flat     raced           5365      24368    4.34  1.1e-15  
      1335  3.5.89           prime    raced          15474      19263    1.11  5.8e-16  bluestein
      1336  2^3.167          prime    raced          16724      18686    1.10  5.0e-16  bluestein
      1337  7.191            prime    raced          15073      14819    0.97  5.5e-16  bluestein
      1338  2.3.223          prime    raced          15071      19131    1.26  6.1e-16  bluestein
      1339  13.103           prime    raced          15413      21334    1.26  5.1e-16  bluestein
      1340  2^2.5.67         prime    raced          15397      23542    1.49  5.4e-16  bluestein
      1341  3^2.149          prime    raced          15907      17256    1.08  6.3e-16  bluestein
      1342  2.11.61          prime    raced          21845      23257    1.06  4.9e-16  bluestein
      1343  17.79            prime    raced          15188      32048    1.72  5.8e-16  flips differ 1.42x bluestein
      1344  2^6.3.7          2p       raced           2240       2386    1.04  4.1e-16  
      1345  5.269            prime    raced          15356      18341    1.19  4.6e-16  bluestein
      1346  2.673            prime    raced          15221      19678    1.27  6.0e-16  bluestein
      1347  3.449            prime    raced          15510      18916    1.17  6.4e-16  bluestein
      1348  2^2.337          prime    raced          15263      22284    1.45  4.8e-16  bluestein
      1349  19.71            prime    raced          15123      26508    1.48  6.2e-16  bluestein
      1350  2.3^3.5^2        chain3   raced           2717       2728    1.00  5.5e-16  
      1351  7.193            prime    raced          17406      13809    0.79  6.1e-16  bluestein
      1352  2^3.13^2         chain3   raced           3036       5399    1.64  6.9e-16  
      1353  3.11.41          chain3   raced           5010      19513    3.85  5.5e-16  
      1354  2.677            prime    raced          15323      19148    1.24  5.5e-16  bluestein
      1355  5.271            prime    raced          15258      16206    0.91  5.8e-16  bluestein
      1356  2^2.3.113        prime    raced          15491      21654    1.35  5.3e-16  bluestein
      1357  23.59            prime    raced          18707      27903    1.39  4.9e-16  bluestein
      1358  2.7.97           prime    raced          15559      15668    1.00  4.2e-16  bluestein
      1359  3^2.151          prime    raced          15916      15085    0.93  6.4e-16  bluestein
      1360  2^4.5.17         chain3   raced           2727       9988    3.64  6.2e-16  
      1361  1361             prime    raced           9873      18318    1.79  8.4e-16  rader
      1362  2.3.227          prime    raced          16005      17175    1.04  5.4e-16  bluestein
      1363  29.47            flat     raced           6449      32197    4.79  9.8e-16  
      1364  2^2.11.31        chain3   raced           4366      21106    3.89  6.0e-16  flips differ 1.39x
      1365  3.5.7.13         chain3   raced           4275       5775    1.30  4.6e-16  
      1366  2.683            prime    raced          21626      27291    0.95  6.2e-16  flips differ 1.33x bluestein
      1367  1367             prime    raced          21430      26239    1.22  5.5e-16  bluestein
      1368  2^3.3^2.19       chain3   raced           3089      10694    3.38  5.9e-16  
      1369  37^2             flat     raced           6355      31174    4.88  1.9e-15  
      1370  2.5.137          prime    raced          15531      18186    1.16  4.1e-16  bluestein
      1371  3.457            prime    raced          15072      18369    1.21  5.9e-16  bluestein
      1372  2^2.7^3          flat     raced           3244       3538    1.03  4.5e-16  
      1373  1373             prime    raced          15219      20576    1.07  5.5e-16  bluestein
      1374  2.3.229          prime    raced          15330      17159    1.06  4.2e-16  bluestein
      1375  5^3.11           chain3   raced           2865       4933    1.71  4.7e-16  
      1376  2^5.43           2p       raced           4095      18902    4.37  4.9e-16  
      1377  3^4.17           chain3   raced           3233      11040    3.33  6.7e-16  
      1378  2.13.53          prime    raced          17358      25807    1.46  4.6e-16  bluestein
      1379  7.197            prime    raced          15118      17738    1.02  5.3e-16  bluestein
      1380  2^2.3.5.23       chain3   raced           3401      12272    3.57  4.6e-16  
      1381  1381             prime    raced          13239      18461    1.32  9.6e-16  rader
      1382  2.691            prime    raced          15212      19239    1.25  5.5e-16  bluestein
      1383  3.461            prime    raced          15566      18510    1.17  4.9e-16  bluestein
      1384  2^3.173          prime    raced          15182      18866    1.23  6.6e-16  bluestein
      1385  5.277            prime    raced          15368      18327    1.13  5.3e-16  bluestein
      1386  2.3^2.7.11       chain3   raced           2914       3895    1.26  4.6e-16  
      1387  19.73            prime    raced          15186      23862    1.57  5.0e-16  bluestein
      1388  2^2.347          prime    raced          15223      18423    1.19  5.8e-16  bluestein
      1389  3.463            prime    raced          15610      18562    1.11  4.9e-16  bluestein
      1390  2.5.139          prime    raced          15122      17880    1.17  5.9e-16  bluestein
      1391  13.107           prime    raced          15317      30477    1.92  4.5e-16  bluestein
      1392  2^4.3.29         chain3   raced           3649      14262    3.85  4.4e-16  
      1393  7.199            prime    raced          15633      16186    0.94  4.7e-16  bluestein
      1394  2.17.41          flat     raced           5966      29500    4.81  2.3e-15  
      1395  3^2.5.31         chain3   raced           4285      15507    3.39  5.9e-16  
      1396  2^2.349          prime    raced          15089      18421    1.18  5.9e-16  bluestein
      1397  11.127           prime    raced          15165      18587    1.21  4.3e-16  bluestein
      1398  2.3.233          prime    raced          15260      17217    1.11  5.8e-16  bluestein
      1399  1399             prime    raced          15513      18504    1.18  5.8e-16  bluestein
      1400  2^3.5^2.7        chain3   raced           2515       2580    0.94  4.1e-16  
      1401  3.467            prime    raced          17543      18513    1.04  6.7e-16  bluestein
      1402  2.701            prime    raced          15529      19312    1.22  5.6e-16  bluestein
      1403  23.61            prime    raced          15137      25813    1.69  6.1e-16  bluestein
      1404  2^2.3^3.13       chain3   raced           2859       3786    1.32  6.1e-16  
      1405  5.281            prime    raced          15345      19278    1.19  5.6e-16  bluestein
      1406  2.19.37          flat     raced           5844      28895    4.81  2.5e-15  
      1407  3.7.67           prime    raced          17795      26599    1.46  6.5e-16  bluestein
      1408  2^7.11           chain3   raced           2559       3312    1.22  4.0e-16  
      1409  1409             prime    raced           8442      18462    2.16  6.0e-16  rader
      1410  2.3.5.47         chain3   raced           4922      20451    3.79  5.1e-16  
      1411  17.83            prime    raced          15226      28672    1.86  5.5e-16  bluestein
      1412  2^2.353          prime    raced          17824      18884    1.02  5.5e-16  bluestein
      1413  3^2.157          prime    raced          15524      16058    1.02  4.9e-16  bluestein
      1414  2.7.101          prime    raced          15282      16491    1.02  5.5e-16  bluestein
      1415  5.283            prime    raced          15101      18864    1.22  6.8e-16  bluestein
      1416  2^3.3.59         prime    raced          15274      19094    1.20  8.7e-16  bluestein
      1417  13.109           prime    raced          15541      17262    1.10  5.9e-16  bluestein
      1418  2.709            prime    raced          15481      18929    1.18  5.4e-16  bluestein
      1419  3.11.43          chain3   raced           5405      21262    3.92  4.3e-16  
      1420  2^2.5.71         prime    raced          15344      17715    1.01  6.0e-16  bluestein
      1421  7^2.29           chain3   raced           4564      14911    3.24  5.4e-16  
      1422  2.3^2.79         prime    raced          15267      19461    1.22  5.1e-16  bluestein
      1423  1423             prime    raced          15239      18458    1.18  5.3e-16  bluestein
      1424  2^4.89           prime    raced          15096      20775    1.35  4.9e-16  bluestein
      1425  3.5^2.19         chain3   raced           3589      12289    3.34  4.8e-16  
      1426  2.23.31          flat     raced           5849      27167    4.56  1.5e-15  
      1427  1427             prime    raced          17528      19593    1.03  5.0e-16  bluestein
      1428  2^2.3.7.17       chain3   raced           3193      10991    3.41  5.4e-16  
      1429  1429             prime    raced          13742      18498    1.31  9.2e-16  rader
      1430  2.5.11.13        chain3   raced           3162       5075    1.58  5.6e-16  
      1431  3^3.53           prime    raced          18330      24902    1.36  3.8e-16  bluestein
      1432  2^3.179          prime    raced          15183      19049    1.25  5.6e-16  bluestein
      1433  1433             prime    raced          15121      19584    1.21  5.5e-16  bluestein
      1434  2.3.239          prime    raced          15254      17520    1.04  5.4e-16  bluestein
      1435  5.7.41           chain3   raced           5645      19144    3.30  4.3e-16  
      1436  2^2.359          prime    raced          15303      18504    1.04  7.9e-16  bluestein
      1437  3.479            prime    raced          15644      18481    1.15  5.8e-16  bluestein
      1438  2.719            prime    raced          15233      19354    1.27  6.5e-16  bluestein
      1439  1439             prime    raced          15702      18642    1.17  6.3e-16  bluestein
      1440  2^5.3^2.5        chain3   raced           3555       2509    0.69  5.3e-16  
      1441  11.131           prime    raced          15721      22840    1.43  5.5e-16  bluestein
      1442  2.7.103          prime    raced          15305      22323    1.44  5.6e-16  bluestein
      1443  3.13.37          chain3   raced           5023      18495    3.68  5.7e-16  
      1444  2^2.19^2         chain3   raced           4144      21167    5.04  5.8e-16  
      1445  5.17^2           chain3   raced           4229      21080    4.62  5.8e-16  
      1446  2.3.241          prime    raced          15114      14714    0.95  5.2e-16  bluestein
      1447  1447             prime    raced          15404      30386    1.43  6.6e-16  flips differ 1.71x bluestein
      1448  2^3.181          prime    raced          15942      15667    0.92  5.0e-16  bluestein
      1449  3^2.7.23         chain3   raced           3884      13672    3.44  5.8e-16  
      1450  2.5^2.29         chain3   raced           4019      15244    3.72  4.5e-16  
      1451  1451             prime    raced          13425      24916    1.59  6.0e-16  rader
      1452  2^2.3.11^2       chain3   raced           3007       6186    2.05  4.0e-16  
      1453  1453             prime    raced          10106      23377    2.03  7.9e-16  rader
      1454  2.727            prime    raced          15285      24676    1.51  5.0e-16  bluestein
      1455  3.5.97           prime    raced          15268      16289    1.02  5.5e-16  bluestein
      1456  2^4.7.13         chain3   raced           2669       3645    1.36  6.3e-16  
      1457  31.47            flat     raced           7042      34912    4.52  1.4e-15  
      1458  2.3^6            chain3   raced           2849       4307    1.43  4.9e-16  
      1459  1459             prime    raced          14835      18868    1.23  1.1e-15  rader
      1460  2^2.5.73         prime    raced          15210      15696    1.02  5.1e-16  bluestein
      1461  3.487            prime    raced          15305      17754    1.16  5.9e-16  bluestein
      1462  2.17.43          flat     raced           6352      31056    4.83  9.4e-16  
      1463  7.11.19          chain3   raced           3808      14753    3.81  4.6e-16  
      1464  2^3.3.61         prime    raced          15650      15865    1.01  4.6e-16  bluestein
      1465  5.293            prime    raced          15256      20730    1.28  5.7e-16  bluestein
      1466  2.733            prime    raced          15580      20933    1.30  6.0e-16  bluestein
      1467  3^2.163          prime    raced          15267      19840    1.28  7.2e-16  bluestein
      1468  2^2.367          prime    raced          16070      25653    1.34  6.9e-16  flips differ 1.39x bluestein
      1469  13.113           prime    raced          15432      28209    1.66  6.4e-16  bluestein
      1470  2.3.5.7^2        chain3   raced           2784       3155    1.10  4.6e-16  
      1471  1471             prime    raced          12167      21947    1.66  1.0e-15  rader
      1472  2^6.23           2p       raced           3322      15008    4.16  4.5e-16  
      1473  3.491            prime    raced          16040      21127    1.19  5.8e-16  bluestein
      1474  2.11.67          prime    raced          17752      37907    1.90  6.4e-16  bluestein
      1475  5^2.59           prime    raced          16109      21156    1.08  6.6e-16  bluestein
      1476  2^2.3^2.41       chain3   raced           4739      21466    4.09  4.9e-16  
      1477  7.211            prime    raced          15591      19988    1.27  5.0e-16  bluestein
      1478  2.739            prime    raced          15380      21920    1.25  5.5e-16  bluestein
      1479  3.17.29          chain3   raced           5152      26299    4.76  6.1e-16  
      1480  2^3.5.37         chain3   raced           4301      17858    4.13  4.1e-16  
      1481  1481             prime    raced          16008      21918    1.26  4.9e-16  bluestein
      1482  2.3.13.19        chain3   raced           4019      14484    2.92  6.0e-16  
      1483  1483             prime    raced          13190      20450    1.51  9.1e-16  rader
      1484  2^2.7.53         prime    raced          15264      24800    1.59  5.9e-16  bluestein
      1485  3^3.5.11         chain3   raced           3235       4659    1.18  4.8e-16  
      1486  2.743            prime    raced          15254      21943    1.37  5.5e-16  bluestein
      1487  1487             prime    raced          15296      21269    1.35  6.1e-16  bluestein
      1488  2^4.3.31         chain3   raced           3933      16456    4.16  5.3e-16  
      1489  1489             prime    raced          13606      20014    1.37  9.2e-16  rader
      1490  2.5.149          prime    raced          16136      22626    1.33  6.2e-16  bluestein
      1491  3.7.71           prime    raced          15450      19992    1.26  4.9e-16  bluestein
      1492  2^2.373          prime    raced          15368      21072    1.32  6.0e-16  bluestein
      1493  1493             prime    raced          15315      20695    1.33  5.9e-16  bluestein
      1494  2.3^2.83         prime    raced          15660      21251    1.25  4.8e-16  bluestein
      1495  5.13.23          chain3   raced           4533      15202    3.35  6.3e-16  
      1496  2^3.11.17        chain3   raced           3583      13543    3.77  4.6e-16  
      1497  3.499            prime    raced          15317      20533    1.14  6.5e-16  bluestein
      1498  2.7.107          prime    raced          16064      21962    1.36  5.7e-16  bluestein
      1499  1499             prime    raced          15333      20254    1.30  7.2e-16  bluestein
      1500  2^2.3.5^3        chain3   raced           2716       2928    1.07  5.3e-16  
      1501  19.79            prime    raced          15427      30287    1.94  6.4e-16  bluestein
      1502  2.751            prime    raced          15598      17708    1.04  5.4e-16  bluestein
      1503  3^2.167          prime    raced          15630      23698    1.51  5.7e-16  bluestein
      1504  2^5.47           2p       raced           4715      21780    4.56  4.3e-16  
      1505  5.7.43           chain3   raced           6037      22018    3.63  5.0e-16  
      1506  2.3.251          prime    raced          15236      17284    1.13  5.0e-16  bluestein
      1507  11.137           prime    raced          15327      21671    1.40  5.3e-16  bluestein
      1508  2^2.13.29        chain3   raced           4710      17356    3.57  5.5e-16  
      1509  3.503            prime    raced          15272      18139    1.11  5.5e-16  bluestein
      1510  2.5.151          prime    raced          15802      16380    1.00  5.1e-16  bluestein
      1511  1511             prime    raced          15728      20767    1.28  6.9e-16  bluestein
      1512  2^3.3^3.7        chain3   raced           3096       3630    0.98  4.4e-16  
      1513  17.89            prime    raced          17072      35325    1.71  6.8e-16  flips differ 1.28x bluestein
      1514  2.757            prime    raced          24988      26930    1.07  5.8e-16  bluestein
      1515  3.5.101          prime    raced          15476      17214    1.11  5.9e-16  bluestein
      1516  2^2.379          prime    raced          18061      17999    0.94  6.0e-16  bluestein
      1517  37.41            flat     raced           7391      37879    5.03  2.0e-15  
      1518  2.3.11.23        chain3   raced           4287      15353    3.57  4.2e-16  
      1519  7^2.31           chain3   raced           5173      16669    3.18  5.8e-16  
      1520  2^4.5.19         chain3   raced           3222      11810    3.66  4.4e-16  
      1521  3^2.13^2         chain3   raced           3771       5798    1.44  7.5e-16  
      1522  2.761            prime    raced          15295      19653    0.98  5.1e-16  flips differ 1.29x bluestein
      1523  1523             prime    raced          15235      20493    1.29  7.0e-16  bluestein
      1524  2^2.3.127        prime    raced          15385      20350    1.25  5.4e-16  bluestein
      1525  5^2.61           prime    raced          15824      17019    1.05  4.5e-16  bluestein
      1526  2.7.109          prime    raced          15220      18014    1.18  4.6e-16  bluestein
      1527  3.509            prime    raced          15593      18699    1.15  5.5e-16  bluestein
      1528  2^3.191          prime    raced          21566      25325    1.08  3.8e-16  bluestein
      1529  11.139           prime    raced          16101      31601    1.61  4.6e-16  flips differ 1.42x bluestein
      1530  2.3^2.5.17       chain3   raced           3615      11474    3.17  5.0e-16  
      1531  1531             prime    raced          11599      19894    1.66  7.7e-16  rader
      1532  2^2.383          prime    raced          15268      17100    1.10  6.6e-16  bluestein
      1533  3.7.73           prime    raced          15652      17016    0.98  5.0e-16  bluestein
      1534  2.13.59          prime    raced          15140      21694    1.37  6.8e-16  bluestein
      1535  5.307            prime    raced          15549      31232    1.97  5.6e-16  bluestein
      1536  2^9.3            chain3   raced           2704       2477    0.84  4.2e-16  
      1537  29.53            prime    raced          17267      39146    2.18  4.5e-16  bluestein
      1538  2.769            prime    raced          18377      22938    1.06  5.0e-16  bluestein
      1539  3^4.19           chain3   raced           5325      17993    3.32  5.4e-16  
      1540  2^2.5.7.11       flat     raced           3611       3962    1.09  4.2e-16  
      1541  23.67            prime    raced          15179      38665    2.52  6.3e-16  bluestein
      1542  2.3.257          prime    raced          15195      16393    1.07  6.0e-16  bluestein
      1543  1543             prime    raced          15565      31750    2.03  4.7e-16  bluestein
      1544  2^3.193          prime    raced          15240      15827    0.91  5.6e-16  bluestein
      1545  3.5.103          prime    raced          15380      23930    1.54  5.0e-16  bluestein
      1546  2.773            prime    raced          16120      22027    1.34  6.8e-16  bluestein
      1547  7.13.17          chain3   raced           4024      15765    3.15  7.5e-16  
      1548  2^2.3^2.43       chain3   raced           5073      21132    4.16  5.4e-16  
      1549  1549             prime    raced          15289      31822    2.08  4.0e-16  bluestein
      1550  2.5^2.31         chain3   raced           4428      16584    3.71  5.5e-16  
      1551  3.11.47          chain3   raced           6164      26536    4.19  4.6e-16  
      1552  2^4.97           prime    raced          16927      20424    1.04  5.7e-16  flips differ 1.29x bluestein
      1553  1553             prime    raced          21779      45441    2.07  5.1e-16  bluestein
      1554  2.3.7.37         chain3   raced           5004      20192    4.03  5.2e-16  
      1555  5.311            prime    raced          15267      31255    2.00  6.2e-16  bluestein
      1556  2^2.389          prime    raced          15584      20397    1.19  4.7e-16  bluestein
      1557  3^2.173          prime    raced          17829      21707    0.99  5.9e-16  bluestein
      1558  2.19.41          flat     raced           7014      32783    4.60  2.8e-15  
      1559  1559             prime    raced          15230      33055    2.05  5.7e-16  bluestein
      1560  2^3.3.5.13       chain3   raced           3074       3544    1.14  6.3e-16  
      1561  7.223            prime    raced          15201      23228    1.44  5.7e-16  bluestein
      1562  2.11.71          prime    raced          15238      24006    1.54  6.0e-16  bluestein
      1563  3.521            prime    raced          15450      22682    1.45  5.0e-16  bluestein
      1564  2^2.17.23        chain3   raced           4670      24521    4.89  6.2e-16  
      1565  5.313            prime    raced          15288      29355    1.91  5.6e-16  bluestein
      1566  2.3^3.29         chain3   raced           4568      16335    3.54  4.7e-16  
      1567  1567             prime    raced          15242      20817    1.25  1.0e-15  rader
      1568  2^5.7^2          chain3   raced           2499       3153    1.19  4.8e-16  
      1569  3.523            prime    raced          15810      26236    1.25  5.9e-16  flips differ 1.46x bluestein
      1570  2.5.157          prime    raced          15626      17696    1.02  5.2e-16  bluestein
      1571  1571             prime    raced          16986      20458    0.63  5.9e-16  flips differ 1.89x bluestein
      1572  2^2.3.131        prime    raced          15420      23252    1.42  5.0e-16  bluestein
      1573  11^2.13          chain3   raced           4083       7685    1.87  6.6e-16  
      1574  2.787            prime    raced          18114      21231    1.15  5.6e-16  bluestein
      1575  3^2.5^2.7        chain3   raced           3632       4578    1.24  5.8e-16  
      1576  2^3.197          prime    raced          15324      18212    1.01  4.6e-16  bluestein
      1577  19.83            prime    raced          15522      33127    1.92  6.2e-16  bluestein
      1578  2.3.263          prime    raced          15222      22606    1.34  8.5e-16  bluestein
      1579  1579             prime    raced          15430      21056    1.35  4.6e-16  bluestein
      1580  2^2.5.79         prime    raced          15629      21038    1.34  4.5e-16  bluestein
      1581  3.17.31          chain3   raced           5547      29054    5.23  5.0e-16  
      1582  2.7.113          prime    raced          15234      25240    1.64  5.1e-16  bluestein
      1583  1583             prime    raced          15290      21068    1.36  5.1e-16  bluestein
      1584  2^4.3^2.11       chain3   raced           2863       4218    1.44  4.5e-16  
      1585  5.317            prime    raced          15343      18125    1.15  6.1e-16  bluestein
      1586  2.13.61          prime    raced          15254      19528    1.24  5.4e-16  bluestein
      1587  3.23^2           flat     raced           5894      28187    4.67  1.4e-15  
      1588  2^2.397          prime    raced          15356      20478    1.32  6.6e-16  bluestein
      1589  7.227            prime    raced          15229      20241    1.32  5.8e-16  bluestein
      1590  2.3.5.53         prime    raced          17473      23944    1.35  4.1e-16  bluestein
      1591  37.43            flat     raced           7893      39712    4.92  1.7e-15  
      1592  2^3.199          prime    raced          17462      18413    1.01  5.8e-16  bluestein
      1593  3^3.59           prime    raced          17496      21844    1.23  6.8e-16  bluestein
      1594  2.797            prime    raced          17721      22553    1.24  6.5e-16  bluestein
      1595  5.11.29          chain3   raced           7324      26281    3.58  3.9e-16  
      1596  2^2.3.7.19       chain3   raced           3842      16117    3.47  4.2e-16  flips differ 1.41x
      1597  1597             prime    raced          15301      21594    1.37  5.8e-16  bluestein
      1598  2.17.47          flat     raced           7242      34589    4.64  3.5e-15  
      1599  3.13.41          chain3   raced           5996      23076    3.65  5.5e-16  
      1600  2^6.5^2          2p       raced           2692       2775    1.00  3.4e-16  
      1601  1601             prime    raced          12547      17196    1.35  7.7e-16  rader
      1602  2.3^2.89         prime    raced          15312      22069    1.34  4.6e-16  bluestein
      1603  7.229            prime    raced          15443      20422    1.23  5.2e-16  bluestein
      1604  2^2.401          prime    raced          15526      17531    1.10  4.0e-16  bluestein
      1605  3.5.107          prime    raced          17538      24509    1.39  5.8e-16  bluestein
      1606  2.11.73          prime    raced          17640      21338    1.20  6.5e-16  bluestein
      1607  1607             prime    raced          17595      24180    1.09  5.8e-16  flips differ 1.26x bluestein
      1608  2^3.3.67         prime    raced          19567      28315    1.32  6.4e-16  bluestein
      1609  1609             prime    raced          15716      25211    1.55  6.6e-16  bluestein
      1610  2.5.7.23         chain3   raced           4019      14394    3.55  4.7e-16  
      1611  3^2.179          prime    raced          15823      29961    1.63  6.5e-16  bluestein
      1612  2^2.13.31        chain3   raced           5497      19614    3.12  6.5e-16  
      1613  1613             prime    raced          15878      24695    1.45  5.0e-16  bluestein
      1614  2.3.269          prime    raced          15222      23391    1.45  6.4e-16  bluestein
      1615  5.17.19          chain3   raced           4624      25483    5.39  6.1e-16  
      1616  2^4.101          prime    raced          21705      25449    1.17  4.9e-16  bluestein
      1617  3.7^2.11         chain3   raced           4945       7920    1.60  5.5e-16  
      1618  2.809            prime    raced          15391      23764    1.47  5.6e-16  bluestein
      1619  1619             prime    raced          16197      25258    1.41  7.9e-16  bluestein
      1620  2^2.3^4.5        chain3   raced           3373       3284    0.96  5.8e-16  
      1621  1621             prime    raced          15441      18712    1.13  4.7e-16  bluestein
      1622  2.811            prime    raced          16156      21174    1.22  6.2e-16  bluestein
      1623  3.541            prime    raced          15840      18675    1.13  4.2e-16  bluestein
      1624  2^3.7.29         chain3   raced           4515      16967    3.64  4.5e-16  
      1625  5^3.13           chain3   raced           3509       5998    1.70  6.0e-16  
      1626  2.3.271          prime    raced          15261      18939    1.23  5.5e-16  bluestein
      1627  1627             prime    raced          15889      32517    1.93  5.3e-16  bluestein
      1628  2^2.11.37        chain3   raced           5575      23074    3.57  3.9e-16  
      1629  3^2.181          prime    raced          24921      26079    1.02  6.4e-16  bluestein
      1630  2.5.163          prime    raced          15250      24297    1.31  5.9e-16  flips differ 1.43x bluestein
      1631  7.233            prime    raced          15443      20615    1.15  6.6e-16  bluestein
      1632  2^5.3.17         chain3   raced           3537      11943    3.36  5.2e-16  
      1633  23.71            prime    raced          17997      36208    1.80  6.4e-16  bluestein
      1634  2.19.43          flat     raced           7297      40806    5.35  2.3e-15  
      1635  3.5.109          prime    raced          17599      19370    0.90  5.5e-16  flips differ 1.25x bluestein
      1636  2^2.409          prime    raced          16110      24962    1.32  4.8e-16  flips differ 1.29x bluestein
      1637  1637             prime    raced          15720      32396    2.00  5.4e-16  bluestein
      1638  2.3^2.7.13       chain3   raced           3595       4999    0.98  5.5e-16  flips differ 1.29x
      1639  11.149           prime    raced          15945      28637    1.58  6.2e-16  bluestein
      1640  2^3.5.41         chain3   raced           5233      25266    3.86  4.9e-16  flips differ 1.28x
      1641  3.547            prime    raced          16779      31879    1.70  6.6e-16  bluestein
      1642  2.821            prime    raced          18332      24672    1.30  6.9e-16  bluestein
      1643  31.53            prime    raced          15680      45400    2.69  4.6e-16  bluestein
      1644  2^2.3.137        prime    raced          15661      23082    1.42  5.6e-16  bluestein
      1645  5.7.47           flat     raced           6965      25743    3.65  6.4e-16  
      1646  2.823            prime    raced          15611      24338    1.18  6.7e-16  flips differ 1.35x bluestein
      1647  3^3.61           prime    raced          16788      19474    1.08  5.3e-16  bluestein
      1648  2^4.103          prime    raced          15564      27617    1.48  5.6e-16  bluestein
      1649  17.97            prime    raced          18017      30608    1.52  4.2e-16  bluestein
      1650  2.3.5^2.11       chain3   raced           3373       4022    1.07  4.3e-16  
      1651  13.127           prime    raced          17918      21992    1.08  4.9e-16  bluestein
      1652  2^2.7.59         prime    raced          17804      24526    1.08  4.9e-16  bluestein
      1653  3.19.29          flat     raced           6295      29432    4.52  3.2e-15  
      1654  2.827            prime    raced          15477      23409    1.37  5.7e-16  bluestein
      1655  5.331            prime    raced          17527      29992    1.35  7.7e-16  bluestein
      1656  2^3.3^2.23       chain3   raced           4163      17293    4.05  5.3e-16  
      1657  1657             prime    raced          14176      32479    2.18  9.7e-16  rader
      1658  2.829            prime    raced          18285      29710    1.49  6.0e-16  bluestein
      1659  3.7.79           prime    raced          15503      23603    1.50  5.8e-16  bluestein
      1660  2^2.5.83         prime    raced          16065      25122    1.51  5.7e-16  bluestein
      1661  11.151           prime    raced          17926      20118    1.12  4.6e-16  bluestein
      1662  2.3.277          prime    raced          15584      23361    1.36  6.0e-16  bluestein
      1663  1663             prime    raced          18955      36103    1.78  5.8e-16  bluestein
      1664  2^7.13           chain3   raced           3107       4202    1.16  4.6e-16  
      1665  3^2.5.37         chain3   raced           5425      23201    3.97  5.2e-16  
      1666  2.7^2.17         flat     raced           4868      14061    2.38  6.5e-16  
      1667  1667             prime    raced          16199      33708    2.05  6.3e-16  bluestein
      1668  2^2.3.139        prime    raced          17685      27116    1.31  5.5e-16  flips differ 1.31x bluestein
      1669  1669             prime    raced          22002      44070    1.98  6.0e-16  bluestein
      1670  2.5.167          prime    raced          17623      27755    0.91  5.6e-16  flips differ 1.75x bluestein
      1671  3.557            prime    raced          16161      32125    1.89  4.5e-16  bluestein
      1672  2^3.11.19        chain3   raced           4344      17910    3.53  5.0e-16  
      1673  7.239            prime    raced          15518      22534    0.81  5.9e-16  flips differ 1.86x bluestein
      1674  2.3^3.31         chain3   raced           5076      18645    3.46  5.4e-16  
      1675  5^2.67           prime    raced          18030      31866    1.37  7.3e-16  bluestein
      1676  2^2.419          prime    raced          15371      22986    1.40  7.2e-16  bluestein
      1677  3.13.43          chain3   raced           6418      25352    3.89  6.3e-16  
      1678  2.839            prime    raced          18086      24276    1.18  6.2e-16  bluestein
      1679  23.73            prime    raced          15554      33259    1.98  6.1e-16  bluestein
      1680  2^4.3.5.7        chain3   raced           3441       3061    0.88  4.8e-16  
      1681  41^2             flat     raced           8676      41834    4.75  2.0e-15  
      1682  2.29^2           flat     raced           7458      34324    4.45  1.7e-15  
      1683  3^2.11.17        chain3   raced           4390      15721    3.56  5.9e-16  
      1684  2^2.421          prime    raced          15288      22528    1.32  7.5e-16  bluestein
      1685  5.337            prime    raced          15276      28606    1.77  6.5e-16  bluestein
      1686  2.3.281          prime    raced          15452      22282    1.43  5.7e-16  bluestein
      1687  7.241            prime    raced          15421      18196    1.09  6.2e-16  bluestein
      1688  2^3.211          prime    raced          15237      22590    1.48  5.9e-16  bluestein
      1689  3.563            prime    raced          15288      30546    1.96  4.7e-16  bluestein
      1690  2.5.13^2         chain3   raced           3874       6565    1.66  6.1e-16  
      1691  19.89            prime    raced          15590      34517    2.15  5.8e-16  bluestein
      1692  2^2.3^2.47       chain3   raced           5842      24914    4.25  4.8e-16  
      1693  1693             prime    raced          15293      23569    1.52  6.1e-16  bluestein
      1694  2.7.11^2         flat     raced           4834       7546    1.43  7.1e-16  
      1695  3.5.113          prime    raced          15755      28287    1.72  6.9e-16  bluestein
      1696  2^5.53           prime    raced          16630      28561    1.70  6.1e-16  bluestein
      1697  1697             prime    raced          15337      23817    1.53  5.6e-16  bluestein
      1698  2.3.283          prime    raced          15370      23361    1.44  6.0e-16  bluestein
      1699  1699             prime    raced          15210      24554    1.55  6.3e-16  bluestein
      1700  2^2.5^2.17       chain3   raced           3902      13532    3.09  5.3e-16  
      1701  3^5.7            chain3   raced           4106       5850    1.42  4.9e-16  
      1702  2.23.37          flat     raced           7284      34114    4.59  1.3e-15  
      1703  13.131           prime    raced          15334      27310    1.72  5.7e-16  bluestein
      1704  2^3.3.71         prime    raced          15342      22721    1.47  4.8e-16  bluestein
      1705  5.11.31          chain3   raced           5659      20791    3.67  4.6e-16  
      1706  2.853            prime    raced          15324      23679    1.51  5.9e-16  bluestein
      1707  3.569            prime    raced          17557      23183    1.30  5.4e-16  bluestein
      1708  2^2.7.61         prime    raced          17566      18747    1.06  4.9e-16  bluestein
      1709  1709             prime    raced          15367      23557    1.17  6.8e-16  flips differ 1.29x bluestein
      1710  2.3^2.5.19       chain3   raced           4115      13703    3.31  5.6e-16  
      1711  29.59            prime    raced          15551      38155    2.43  5.5e-16  bluestein
      1712  2^4.107          prime    raced          15401      25029    1.56  5.9e-16  bluestein
      1713  3.571            prime    raced          17867      23623    1.25  7.1e-16  bluestein
      1714  2.857            prime    raced          17724      24049    1.16  6.2e-16  bluestein
      1715  5.7^3            flat     raced           4347       6459    1.42  7.4e-16  
      1716  2^2.3.11.13      chain3   raced           3700       6533    1.76  5.8e-16  
      1717  17.101           prime    raced          15374      28720    1.85  5.4e-16  bluestein
      1718  2.859            prime    raced          15454      23657    1.51  5.5e-16  bluestein
      1719  3^2.191          prime    raced          17726      19439    1.08  4.6e-16  bluestein
      1720  2^3.5.43         chain3   raced           5517      23361    4.19  5.4e-16  
      1721  1721             prime    raced          15365      23829    1.50  5.9e-16  bluestein
      1722  2.3.7.41         chain3   raced           6049      19705    3.24  4.7e-16  
      1723  1723             prime    raced          15408      24089    1.56  5.3e-16  bluestein
      1724  2^2.431          prime    raced          15621      23485    1.33  6.3e-16  bluestein
      1725  3.5^2.23         chain3   raced           4684      16552    3.51  5.5e-16  
      1726  2.863            prime    raced          15325      23785    1.51  6.7e-16  bluestein
      1727  11.157           prime    raced          15345      21652    1.36  5.5e-16  bluestein
      1728  2^6.3^3          2p       raced           2945       3128    1.02  4.7e-16  
      1729  7.13.19          chain3   raced           4676      16237    3.38  6.0e-16  
      1730  2.5.173          prime    raced          15360      24764    1.53  5.6e-16  bluestein
      1731  3.577            prime    raced          15366      19701    1.23  4.9e-16  bluestein
      1732  2^2.433          prime    raced          15701      19770    1.23  6.1e-16  bluestein
      1733  1733             prime    raced          17632      24017    1.35  5.8e-16  bluestein
      1734  2.3.17^2         chain3   raced           4765      25674    5.24  4.2e-16  
      1735  5.347            prime    raced          15433      23370    1.50  6.2e-16  bluestein
      1736  2^3.7.31         chain3   raced           4945      19006    3.82  5.4e-16  
      1737  3^2.193          prime    raced          17611      20161    1.13  4.7e-16  bluestein
      1738  2.11.79          prime    raced          15472      26312    1.63  6.0e-16  bluestein
      1739  37.47            flat     raced           9008      45723    4.83  1.6e-15  
      1740  2^2.3.5.29       chain3   raced           4826      17799    3.65  5.0e-16  
      1741  1741             prime    raced          15227      25308    1.34  5.1e-16  bluestein
      1742  2.13.67          prime    raced          15405      32924    2.10  5.3e-16  bluestein
      1743  3.7.83           prime    raced          15405      25869    1.55  4.8e-16  bluestein
      1744  2^4.109          prime    raced          15517      19569    1.21  6.7e-16  bluestein
      1745  5.349            prime    raced          17565      25122    1.28  6.0e-16  bluestein
      1746  2.3^2.97         prime    raced          15415      19467    0.99  4.7e-16  bluestein
      1747  1747             prime    raced          15777      25132    1.48  5.3e-16  bluestein
      1748  2^2.19.23        chain3   raced           5865      34754    5.34  5.4e-16  
      1749  3.11.53          prime    raced          17602      30675    1.72  4.8e-16  bluestein
      1750  2.5^3.7          chain3   raced           3356       4129    1.21  4.5e-16  
      1751  17.103           prime    raced          15372      36974    2.39  5.9e-16  bluestein
      1752  2^3.3.73         prime    raced          15968      18945    1.17  6.2e-16  bluestein
      1753  1753             prime    raced          15342      23566    1.52  5.7e-16  bluestein
      1754  2.877            prime    raced          15373      25724    1.65  4.9e-16  bluestein
      1755  3^3.5.13         chain3   raced           3807       5667    1.31  6.7e-16  
      1756  2^2.439          prime    raced          15272      25299    1.61  5.4e-16  bluestein
      1757  7.251            prime    raced          15394      20415    1.32  5.7e-16  bluestein
      1758  2.3.293          prime    raced          15383      23471    1.36  5.7e-16  bluestein
      1759  1759             prime    raced          15359      23552    1.32  5.4e-16  bluestein
      1760  2^5.5.11         chain3   raced           3173       4216    1.29  3.9e-16  
      1761  3.587            prime    raced          15432      25433    1.59  6.7e-16  bluestein
      1762  2.881            prime    raced          15419      24802    1.60  5.6e-16  bluestein
      1763  41.43            flat     raced           9186      42856    4.61  2.0e-15  
      1764  2^2.3^2.7^2      chain3   raced           3397       4213    1.23  4.9e-16  
      1765  5.353            prime    raced          15391      24666    1.51  5.0e-16  bluestein
      1766  2.883            prime    raced          15348      25081    1.59  6.0e-16  bluestein
      1767  3.19.31          chain3   raced           6246      31270    4.88  4.2e-16  
      1768  2^3.13.17        chain3   raced           4359      16227    3.47  6.4e-16  
      1769  29.61            prime    raced          15511      36550    2.33  6.4e-16  bluestein
      1770  2.3.5.59         prime    raced          15409      24001    1.48  4.7e-16  bluestein
      1771  7.11.23          chain3   raced           5739      20582    3.58  4.4e-16  
      1772  2^2.443          prime    raced          15345      25030    1.58  5.3e-16  bluestein
      1773  3^2.197          prime    raced          15409      20404    1.27  4.9e-16  bluestein
      1774  2.887            prime    raced          15423      25435    1.58  5.5e-16  bluestein
      1775  5^2.71           prime    raced          15512      23991    1.54  6.4e-16  bluestein
      1776  2^4.3.37         chain3   raced           6370      32877    3.48  5.0e-16  flips differ 1.82x
      1777  1777             prime    raced          17111      30267    1.67  6.0e-16  bluestein
      1778  2.7.127          prime    raced          15471      22649    1.27  5.6e-16  bluestein
      1779  3.593            prime    raced          15398      24234    1.56  6.4e-16  bluestein
      1780  2^2.5.89         prime    raced          15420      23544    1.47  5.0e-16  bluestein
      1781  13.137           prime    raced          15339      28777    1.71  5.3e-16  bluestein
      1782  2.3^4.11         chain3   raced           3743       5639    1.31  5.7e-16  
      1783  1783             prime    raced          15627      23678    1.50  6.4e-16  bluestein
      1784  2^3.223          prime    raced          15416      25803    1.60  6.4e-16  bluestein
      1785  3.5.7.17         chain3   raced           4326      14463    3.10  6.7e-16  
      1786  2.19.47          flat     raced           8080      41358    4.97  1.5e-15  
      1787  1787             prime    raced          15575      23970    1.43  5.4e-16  bluestein
      1788  2^2.3.149        prime    raced          16965      24055    1.33  5.7e-16  bluestein
      1789  1789             prime    raced          15415      23831    1.53  6.5e-16  bluestein
      1790  2.5.179          prime    raced          15467      23800    1.38  5.2e-16  bluestein
      1791  3^2.199          prime    raced          15642      20881    1.29  5.1e-16  bluestein
      1792  2^8.7            chain3   raced           3024       3240    1.06  3.3e-16  
      1793  11.163           prime    raced          16206      23693    1.46  4.2e-16  bluestein
      1794  2.3.13.23        chain3   raced           5169      18392    3.53  5.7e-16  
      1795  5.359            prime    raced          16046      23431    1.45  6.1e-16  bluestein
      1796  2^2.449          prime    raced          15394      24295    1.58  6.1e-16  bluestein
      1797  3.599            prime    raced          15460      24309    1.55  5.9e-16  bluestein
      1798  2.29.31          flat     raced           8059      37142    4.48  2.2e-15  
      1799  7.257            prime    raced          16368      19323    1.09  5.5e-16  bluestein
      1800  2^3.3^2.5^2      chain3   raced           3385       3315    0.96  4.2e-16  
      1801  1801             prime    raced          15398      19680    1.27  5.2e-16  bluestein
      1802  2.17.53          prime    raced          15424      43423    2.76  4.3e-16  bluestein
      1803  3.601            prime    raced          15547      20010    1.26  6.1e-16  bluestein
      1804  2^2.11.41        chain3   raced           6785      31455    4.63  5.2e-16  
      1805  5.19^2           chain3   raced           5416      27410    5.03  5.0e-16  
      1806  2.3.7.43         chain3   raced           6439      25014    3.87  4.3e-16  
      1807  13.139           prime    raced          15544      25414    1.50  6.0e-16  bluestein
      1808  2^4.113          prime    raced          16219      31305    1.66  6.0e-16  flips differ 1.28x bluestein
      1809  3^3.67           prime    raced          17331      32340    1.78  5.5e-16  bluestein
      1810  2.5.181          prime    raced          15622      19837    1.25  4.7e-16  bluestein
      1811  1811             prime    raced          15527      35840    2.25  6.4e-16  bluestein
      1812  2^2.3.151        prime    raced          15466      20489    1.30  4.7e-16  bluestein
      1813  7^2.37           flat     raced           6912      24271    3.35  6.6e-16  
      1814  2.907            prime    raced          15378      37470    2.32  6.7e-16  bluestein
      1815  3.5.11^2         chain3   raced           4231       7720    1.79  4.9e-16  
      1816  2^3.227          prime    raced          15504      22552    1.44  5.1e-16  bluestein
      1817  23.79            prime    raced          15411      38191    2.45  5.0e-16  bluestein
      1818  2.3^2.101        prime    raced          15341      19661    1.27  5.3e-16  bluestein
      1819  17.107           prime    raced          15670      38333    2.33  6.1e-16  bluestein
      1820  2^2.5.7.13       flat     raced           4557       5150    1.13  7.3e-16  
      1821  3.607            prime    raced          15392      34464    1.94  6.3e-16  bluestein
      1822  2.911            prime    raced          15351      36832    2.30  5.5e-16  bluestein
      1823  1823             prime    raced          17699      35086    1.97  8.1e-16  bluestein
      1824  2^5.3.19         chain3   raced           3984      17359    4.34  5.6e-16  
      1825  5^2.73           prime    raced          16182      21625    1.29  5.9e-16  bluestein
      1826  2.11.83          prime    raced          16648      28533    1.62  6.7e-16  bluestein
      1827  3^2.7.29         chain3   raced           5458      19456    3.55  4.5e-16  
      1828  2^2.457          prime    raced          15329      25126    1.57  4.8e-16  bluestein
      1829  31.59            prime    raced          17667      43506    2.45  4.5e-16  bluestein
      1830  2.3.5.61         prime    raced          16234      19984    1.07  5.7e-16  bluestein
      1831  1831             prime    raced          16163      30546    1.77  5.2e-16  bluestein
      1832  2^3.229          prime    raced          16463      22616    1.23  5.4e-16  bluestein
      1833  3.13.47          chain3   raced           7335      30022    4.02  7.3e-16  
      1834  2.7.131          prime    raced          16200      27141    1.67  7.3e-16  bluestein
      1835  5.367            prime    raced          16541      26462    1.52  5.5e-16  bluestein
      1836  2^2.3^3.17       chain3   raced           4129      14493    3.48  6.3e-16  
      1837  11.167           prime    raced          17622      28720    1.30  7.7e-16  flips differ 1.27x bluestein
      1838  2.919            prime    raced          16116      36810    2.02  6.1e-16  bluestein
      1839  3.613            prime    raced          16154      29509    1.51  6.2e-16  bluestein
      1840  2^4.5.23         chain3   raced           4420      16607    3.73  6.2e-16  
      1841  7.263            prime    raced          15460      26406    1.66  6.0e-16  bluestein
      1842  2.3.307          prime    raced          15847      34415    2.03  6.1e-16  bluestein
      1843  19.97            prime    raced          15640      31434    1.94  5.7e-16  bluestein
      1844  2^2.461          prime    raced          16436      24779    1.50  5.3e-16  bluestein
      1845  3^2.5.41         chain3   raced           7042      24701    3.49  4.4e-16  
      1846  2.13.71          prime    raced          15351      28162    1.64  7.0e-16  bluestein
      1847  1847             prime    raced          15470      29140    1.87  6.8e-16  bluestein
      1848  2^3.3.7.11       chain3   raced           3963       5062    1.25  5.0e-16  
      1849  43^2             flat     raced           9858      48154    4.87  1.5e-15  
      1850  2.5^2.37         chain3   raced           5663      23957    4.23  5.4e-16  
      1851  3.617            prime    raced          15343      29833    1.82  5.8e-16  bluestein
      1852  2^2.463          prime    raced          15712      24677    1.45  5.5e-16  bluestein
      1853  17.109           prime    raced          15697      32451    2.05  5.8e-16  bluestein
      1854  2.3^2.103        prime    raced          15585      28657    1.82  6.6e-16  bluestein
      1855  5.7.53           prime    raced          15561      31845    2.01  4.5e-16  bluestein
      1856  2^6.29           2p       raced           4649      19268    4.12  4.3e-16  
      1857  3.619            prime    raced          15687      30707    1.91  6.3e-16  bluestein
      1858  2.929            prime    raced          15632      36654    2.28  5.7e-16  bluestein
      1859  11.13^2          chain3   raced           4958       9109    1.83  8.2e-16  
      1860  2^2.3.5.31       chain3   raced           5300      23437    4.41  5.7e-16  
      1861  1861             prime    raced          15502      29172    1.85  6.3e-16  bluestein
      1862  2.7^2.19         flat     raced           5802      16392    2.80  7.0e-16  
      1863  3^4.23           chain3   raced           4985      17432    3.35  5.5e-16  
      1864  2^3.233          prime    raced          15737      23777    1.48  5.5e-16  bluestein
      1865  5.373            prime    raced          18138      32797    1.48  5.6e-16  flips differ 1.28x bluestein
      1866  2.3.311          prime    raced          17113      38000    1.80  4.8e-16  flips differ 1.32x bluestein
      1867  1867             prime    raced          16454      30731    1.81  6.3e-16  bluestein
      1868  2^2.467          prime    raced          16898      26932    1.43  6.4e-16  bluestein
      1869  3.7.89           prime    raced          18366      28351    1.51  5.4e-16  bluestein
      1870  2.5.11.17        chain3   raced           4748      19169    3.37  6.7e-16  flips differ 1.28x
      1871  1871             prime    raced          15340      30185    1.57  1.1e-15  rader
      1872  2^4.3^2.13       chain3   raced           3565       4948    1.34  6.7e-16  
      1873  1873             prime    raced          12724      37900    2.75  1.2e-15  rader
      1874  2.937            prime    raced          17107      42020    2.19  6.3e-16  bluestein
      1875  3.5^4            chain3   raced           4530       6565    1.34  4.4e-16  
      1876  2^2.7.67         prime    raced          18788      34652    1.69  5.1e-16  bluestein
      1877  1877             prime    raced          16711      34950    1.94  5.7e-16  bluestein
      1878  2.3.313          prime    raced          16173      40568    2.36  7.3e-16  bluestein
      1879  1879             prime    raced          17038      28582    1.52  4.9e-16  bluestein
      1880  2^3.5.47         chain3   raced           6686      37048    3.95  3.9e-16  flips differ 1.25x
      1881  3^2.11.19        chain3   raced           5442      26202    3.97  6.7e-16  flips differ 1.29x
      1882  2.941            prime    raced          16651      28671    1.58  4.4e-16  bluestein
      1883  7.269            prime    raced          16701      28247    1.65  6.4e-16  bluestein
      1884  2^2.3.157        prime    raced          16196      23447    1.39  5.0e-16  bluestein
      1885  5.13.29          chain3   raced           6374      22688    3.53  7.7e-16  
      1886  2.23.41          flat     raced           8889      44781    4.94  1.4e-15  
      1887  3.17.37          chain3   raced           7290      39879    5.32  6.2e-16  
      1888  2^5.59           prime    raced          17545      28589    1.40  5.5e-16  bluestein
      1889  1889             prime    raced          16475      26742    1.52  7.0e-16  bluestein
      1890  2.3^3.5.7        chain3   raced           4035       4324    1.04  5.7e-16  
      1891  31.61            prime    raced          16363      41837    2.40  5.5e-16  bluestein
      1892  2^2.11.43        chain3   raced           7278      31917    3.69  4.1e-16  
      1893  3.631            prime    raced          16155      25692    1.47  5.9e-16  bluestein
      1894  2.947            prime    raced          16531      27809    1.50  6.7e-16  bluestein
      1895  5.379            prime    raced          17345      22792    1.26  5.5e-16  bluestein
      1896  2^3.3.79         prime    raced          17062      26584    1.31  6.3e-16  bluestein
      1897  7.271            prime    raced          17242      23359    1.35  5.4e-16  bluestein
      1898  2.13.73          prime    raced          16248      27273    1.38  6.7e-16  bluestein
      1899  3^2.211          prime    raced          18267      27568    1.42  6.3e-16  bluestein
      1900  2^2.5^2.19       chain3   raced           4854      17784    3.21  4.7e-16  
      1901  1901             prime    raced          16311      28946    1.36  6.1e-16  flips differ 1.44x bluestein
      1902  2.3.317          prime    raced          15792      21898    1.17  5.3e-16  bluestein
      1903  11.173           prime    raced          16400      40527    2.44  6.7e-16  bluestein
      1904  2^4.7.17         chain3   raced           3866      15752    4.07  5.3e-16  
      1905  3.5.127          prime    raced          16136      23698    1.44  5.9e-16  bluestein
      1906  2.953            prime    raced          15501      26336    1.64  5.2e-16  bluestein
      1907  1907             prime    raced          15960      25031    1.48  6.0e-16  bluestein
      1908  2^2.3^2.53       prime    raced          15566      38820    2.20  5.4e-16  bluestein
      1909  23.83            prime    raced          19533      47715    2.00  5.7e-16  bluestein
      1910  2.5.191          prime    raced          16920      22895    1.29  4.4e-16  bluestein
      1911  3.7^2.13         chain3   raced           4304       7890    1.53  5.3e-16  
      1912  2^3.239          prime    raced          15819      25673    1.54  6.6e-16  bluestein
      1913  1913             prime    raced          16484      26028    1.49  6.3e-16  bluestein
      1914  2.3.11.29        chain3   raced           5985      22747    3.59  5.9e-16  
      1915  5.383            prime    raced          16832      22650    1.31  6.5e-16  bluestein
      1916  2^2.479          prime    raced          15933      29716    1.48  6.4e-16  bluestein
      1917  3^3.71           prime    raced          15639      28689    1.78  6.1e-16  bluestein
      1918  2.7.137          prime    raced          16731      27954    1.54  6.6e-16  bluestein
      1919  19.101           prime    raced          16334      34254    1.96  5.4e-16  bluestein
      1920  2^7.3.5          chain3   raced           4154       3556    0.79  4.7e-16  
      1921  17.113           prime    raced          19079      49083    2.21  5.3e-16  bluestein
      1922  2.31^2           flat     raced           8837      40941    4.54  2.2e-15  
      1923  3.641            prime    raced          16623      22631    1.28  5.2e-16  bluestein
      1924  2^2.13.37        chain3   raced           6639      28990    4.14  5.4e-16  
      1925  5^2.7.11         chain3   raced           4767       7235    1.38  4.5e-16  
      1926  2.3^2.107        prime    raced          16206      28479    1.55  5.5e-16  bluestein
      1927  41.47            flat     raced          10130      53388    5.11  1.9e-15  
      1928  2^3.241          prime    raced          17193      20532    1.15  5.5e-16  bluestein
      1929  3.643            prime    raced          15999      30084    1.70  7.1e-16  bluestein
      1930  2.5.193          prime    raced          17048      19586    1.13  5.4e-16  bluestein
      1931  1931             prime    raced          15757      30031    1.75  6.4e-16  bluestein
      1932  2^2.3.7.23       chain3   raced           4816      18247    3.67  5.9e-16  
      1933  1933             prime    raced          16115      29675    1.55  7.2e-16  bluestein
      1934  2.967            prime    raced          18172      37786    1.98  8.3e-16  bluestein
      1935  3^2.5.43         chain3   raced           7282      30081    4.03  5.6e-16  
      1936  2^4.11^2         chain3   raced           4018       8177    1.60  4.6e-16  
      1937  13.149           prime    raced          16917      28184    1.47  6.4e-16  bluestein
      1938  2.3.17.19        chain3   raced           5969      29001    4.26  6.8e-16  
      1939  7.277            prime    raced          16763      30456    1.29  6.4e-16  bluestein
      1940  2^2.5.97         prime    raced          16180      20891    1.21  5.7e-16  bluestein
      1941  3.647            prime    raced          15867      30232    1.71  6.8e-16  bluestein
      1942  2.971            prime    raced          16509      32111    1.92  7.0e-16  bluestein
      1943  29.67            prime    raced          16046      55220    3.30  6.1e-16  bluestein
      1944  2^3.3^5          chain3   raced           4213       4803    1.13  5.1e-16  
      1945  5.389            prime    raced          16628      27260    1.31  6.4e-16  flips differ 1.29x bluestein
      1946  2.7.139          prime    raced          16184      32315    1.63  5.1e-16  bluestein
      1947  3.11.59          prime    raced          17693      30965    1.61  6.3e-16  bluestein
      1948  2^2.487          prime    raced          16696      24873    1.45  6.6e-16  bluestein
      1949  1949             prime    raced          15705      28897    1.64  7.0e-16  bluestein
      1950  2.3.5^2.13       chain3   raced           4250       7201    1.12  5.8e-16  flips differ 1.25x
      1951  1951             prime    raced          16777      29591    1.65  9.2e-16  rader
      1952  2^5.61           prime    raced          17645      24138    1.26  5.2e-16  bluestein
      1953  3^2.7.31         chain3   raced           6999      27121    3.60  6.6e-16  
      1954  2.977            prime    raced          16224      32316    1.55  6.0e-16  bluestein
      1955  5.17.23          chain3   raced           6428      31928    4.79  5.2e-16  
      1956  2^2.3.163        prime    raced          16352      24805    1.44  7.0e-16  bluestein
      1957  19.103           prime    raced          16436      44088    2.46  5.6e-16  bluestein
      1958  2.11.89          prime    raced          16144      30044    1.67  5.6e-16  bluestein
      1959  3.653            prime    raced          16382      31545    1.59  6.0e-16  bluestein
      1960  2^3.5.7^2        flat     raced           4588       4539    0.87  6.1e-16  
      1961  37.53            prime    raced          16429      57054    3.14  6.6e-16  bluestein
      1962  2.3^2.109        prime    raced          16621      23508    1.37  5.9e-16  bluestein
      1963  13.151           prime    raced          16386      26693    1.57  5.8e-16  bluestein
      1964  2^2.491          prime    raced          16516      30728    1.73  5.3e-16  bluestein
      1965  3.5.131          prime    raced          17804      31640    1.70  7.1e-16  bluestein
      1966  2.983            prime    raced          16376      28271    1.48  5.9e-16  bluestein
      1967  7.281            prime    raced          16319      27686    1.66  5.7e-16  bluestein
      1968  2^4.3.41         chain3   raced           6700      32479    4.23  6.0e-16  
      1969  11.179           prime    raced          16507      30994    1.67  6.0e-16  bluestein
      1970  2.5.197          prime    raced          17163      29099    1.23  5.4e-16  bluestein
      1971  3^3.73           prime    raced          16555      23100    1.37  6.0e-16  bluestein
      1972  2^2.17.29        chain3   raced           6792      35424    4.74  7.1e-16  
      1973  1973             prime    raced          16316      45908    2.25  6.5e-16  bluestein
      1974  2.3.7.47         chain3   raced           7748      30893    3.85  4.4e-16  
      1975  5^2.79           prime    raced          18852      31015    1.50  5.5e-16  bluestein
      1976  2^3.13.19        chain3   raced           5231      18902    3.55  7.3e-16  
      1977  3.659            prime    raced          15954      30933    1.61  6.9e-16  bluestein
      1978  2.23.43          flat     raced           9280      45777    4.87  1.8e-15  
      1979  1979             prime    raced          15747      26945    1.61  3.9e-16  bluestein
      1980  2^2.3^2.5.11     chain3   raced           3929       5276    1.24  4.5e-16  
      1981  7.283            prime    raced          15769      26253    1.64  6.3e-16  bluestein
      1982  2.991            prime    raced          15690      27575    1.71  5.5e-16  bluestein
      1983  3.661            prime    raced          15752      31195    1.89  6.1e-16  bluestein
      1984  2^6.31           2p       raced           5167      21318    3.94  4.9e-16  
      1985  5.397            prime    raced          17292      26207    1.47  5.8e-16  bluestein
      1986  2.3.331          prime    raced          15809      34986    2.15  6.0e-16  bluestein
      1987  1987             prime    raced          19881      32201    1.40  5.9e-16  bluestein
      1988  2^2.7.71         prime    raced          17196      30375    1.46  4.8e-16  flips differ 1.35x bluestein
      1989  3^2.13.17        chain3   raced           5379      22774    4.00  7.3e-16  
      1990  2.5.199          prime    raced          16910      27959    1.61  4.9e-16  bluestein
      1991  11.181           prime    raced          16918      24321    1.32  5.8e-16  bluestein
      1992  2^3.3.83         prime    raced          16572      33297    1.89  5.0e-16  bluestein
      1993  1993             prime    raced          16397      27951    1.53  6.9e-16  bluestein
      1994  2.997            prime    raced          15742      27287    1.71  5.1e-16  bluestein
      1995  3.5.7.19         chain3   raced           5075      19160    3.53  4.6e-16  
      1996  2^2.499          prime    raced          18475      29016    1.53  5.6e-16  bluestein
      1997  1997             prime    raced          15964      27865    1.70  7.0e-16  bluestein
      1998  2.3^3.37         chain3   raced           6373      24608    3.70  6.2e-16  
      1999  1999             prime    raced          16111      28018    1.58  6.1e-16  bluestein
      2000  2^4.5^3          chain3   raced           4143       3529    0.82  4.5e-16  
      2001  3.23.29          flat     raced           8104      39417    4.78  1.5e-15  
      2002  2.7.11.13        flat     raced           5864       8281    1.38  8.6e-16  
      2003  2003             prime    raced          15590      32063    1.99  6.7e-16  bluestein
      2004  2^2.3.167        prime    raced          15649      28780    1.82  5.8e-16  bluestein
      2005  5.401            prime    raced          15553      20889    1.33  5.8e-16  bluestein
      2006  2.17.59          prime    raced          15547      42789    2.55  5.1e-16  bluestein
      2007  3^2.223          prime    raced          15600      28484    1.78  5.0e-16  bluestein
      2008  2^3.251          prime    raced          15910      23314    1.45  5.5e-16  bluestein
      2009  7^2.41           flat     raced           7889      30180    2.93  5.2e-16  
      2010  2.3.5.67         prime    raced          15745      35509    2.03  6.9e-16  bluestein
      2011  2011             prime    raced          16975      31890    1.75  6.1e-16  bluestein
      2012  2^2.503          prime    raced          15713      24458    1.54  5.5e-16  bluestein
      2013  3.11.61          prime    raced          15592      24656    1.56  6.2e-16  bluestein
      2014  2.19.53          prime    raced          15651      50474    3.11  5.8e-16  bluestein
      2015  5.13.31          chain3   raced           6759      24492    3.60  6.5e-16  
      2016  2^5.3^2.7        chain3   raced           4103       3816    0.92  4.3e-16  
      2017  2017             prime    raced          15640      32202    2.01  7.1e-16  bluestein
      2018  2.1009           prime    raced          15601      38051    2.38  6.5e-16  bluestein
      2019  3.673            prime    raced          15653      30686    1.91  6.0e-16  bluestein
      2020  2^2.5.101        prime    raced          15717      21329    1.30  5.6e-16  bluestein
      2021  43.47            flat     raced          10915      58899    4.62  1.8e-15  
      2022  2.3.337          prime    raced          16026      34104    2.12  5.0e-16  bluestein
      2023  7.17^2           flat     raced           6288      29999    4.72  1.6e-15  
      2024  2^3.11.23        chain3   raced           5502      21101    3.73  6.0e-16  
      2025  3^4.5^2          chain3   raced           5039       7071    1.20  4.3e-16  
      2026  2.1013           prime    raced          15934      40638    2.42  9.2e-16  bluestein
      2027  2027             prime    raced          18776      28520    1.50  6.2e-16  bluestein
      2028  2^2.3.13^2       chain3   raced           4712       7889    1.67  6.3e-16  
      2029  2029             prime    raced          14861      29757    1.93  1.0e-15  rader
      2030  2.5.7.29         chain3   raced           5864      23410    3.63  4.4e-16  
      2031  3.677            prime    raced          16254      29577    1.78  5.8e-16  bluestein
      2032  2^4.127          prime    raced          17896      26916    1.48  6.3e-16  bluestein
      2033  19.107           prime    raced          16570      47204    1.83  6.9e-16  flips differ 1.61x bluestein
      2034  2.3^2.113        prime    raced          17336      41819    1.98  5.9e-16  flips differ 1.30x bluestein
      2035  5.11.37          chain3   raced           7558      28465    3.11  5.7e-16  
      2036  2^2.509          prime    raced          16386      30380    1.53  6.2e-16  bluestein
      2037  3.7.97           prime    raced          16130      23666    1.39  5.4e-16  bluestein
      2038  2.1019           prime    raced          16685      28860    1.71  6.8e-16  bluestein
      2039  2039             prime    raced          16441      29347    1.75  7.6e-16  bluestein
      2040  2^3.3.5.17       chain3   raced           4753      26268    3.64  6.9e-16  flips differ 1.53x
      2041  13.157           prime    raced          16744      31994    1.65  5.5e-16  bluestein
      2042  2.1021           prime    raced          16482      30854    1.77  6.1e-16  bluestein
      2043  3^2.227          prime    raced          16574      29024    1.54  6.6e-16  bluestein
      2044  2^2.7.73         prime    raced          16476      24126    1.40  5.4e-16  bluestein
      2045  5.409            prime    raced          17174      30365    1.59  7.3e-16  bluestein
      2046  2.3.11.31        chain3   raced           6760      27123    3.86  5.4e-16  
      2047  23.89            prime    raced          17543      48108    2.49  5.0e-16  bluestein
      2048  2^11             ztt      raced           2797       3256    1.15  3.1e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     1172     36    174   0.93   1.36   1.98    1.36
 chain3     435      7     36   1.04   3.24   4.09    2.24
 2p         292      4     31   0.99   3.80   4.91    2.56
 flat       120      2      5   1.38   3.66   4.88    3.28
 mono        24      9     14   0.69   0.84   2.41    1.14
 ztt          4      0      1   0.96   1.14   1.16    1.10
 ALL       2047     58    261   0.96   1.51   4.12    1.74
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   0.81      5      2
 8..31                 24   0.90     17      9
 32..127               96   1.68     13      1
 128..511             384   1.68     19      5
 512..2047           1536   1.46    207     41
 2048..2048             1   1.15      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 composite with a prime >= 53 (prime cell)      878   1.31    158   1.31
 chain3                                         435   3.24     36   2.24
 2p                                             288   3.86     30   2.59
 prime N, bluestein                             178   1.43     14   1.40
 flat                                           120   3.66      5   3.28
 prime N, rader                                 116   1.76      2   1.76
 mono                                            21   0.85     12   1.19
 pow2                                            11   1.05      4   1.01
```


flip agreement: our two readings more than 25% apart at 96 of 2047 cells.

worst 10: 531 (prime 0.33), 1282 (prime 0.57), 13 (mono 0.58), 25 (2p 0.60), 545 (prime 0.61), 579 (prime 0.61), 200 (2p 0.61), 140 (chain3 0.61), 1571 (prime 0.63), 1135 (prime 0.66)
best 5: 752 (2p 5.97), 688 (2p 5.96), 289 (2p 5.74), 943 (flat 5.62), 323 (2p 5.61)
