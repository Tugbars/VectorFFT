# gauntlet report

run: `ip_2_2048_2026-09-25`  contract file suffix: `_ip`  cells: 1031 listed, 1023 benched, comparator: MKL

control cell: 24 readings, 1.065..1.079 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     replayed           4         10    2.89  0.0e+00  
         3  3                mono     replayed           4         11    2.40  1.5e-16  
         4  2^2              mono     replayed           5         11    2.33  0.0e+00  
         5  5                mono     replayed           6         11    1.79  7.5e-17  
         6  2.3              mono     replayed           7         11    1.69  1.9e-16  
         7  7                mono     replayed           8         12    1.49  2.6e-16  
         8  2^3              mono     replayed           7         11    1.54  7.4e-17  
         9  3^2              mono     replayed           9         12    1.37  4.2e-16  
        10  2.5              mono     replayed           9         12    1.32  1.8e-16  
        11  11               mono     replayed          11         13    1.15  1.2e-16  
        12  2^2.3            mono     replayed          11         12    1.06  2.8e-16  
        13  13               mono     replayed          15         14    0.94  5.0e-16  
        14  2.7              flat     replayed          19         13    0.66  2.4e-16  
        15  3.5              mono     replayed          18         14    0.76  4.4e-16  
        16  2^4              2p       replayed          10         12    1.14  0.0e+00  
        17  17               mono     replayed          23         39    1.65  2.0e-16  
        18  2.3^2            2p       replayed          18         23    1.28  3.6e-16  
        19  19               mono     replayed          28         44    1.58  4.6e-16  
        20  2^2.5            2p       replayed          16         22    0.93  2.2e-16  flips differ 1.50x
        21  3.7              2p       replayed          21         24    1.18  3.3e-16  
        22  2.11             flat     replayed          31         30    0.99  2.8e-16  
        23  23               mono     replayed          39         65    1.68  2.5e-16  
        24  2^3.3            2p       replayed          15         28    1.82  2.8e-16  
        25  5^2              2p       replayed          20         26    1.30  1.8e-16  
        26  2.13             flat     replayed          36         36    0.98  8.6e-16  
        27  3^3              2p       replayed          23         31    1.35  3.8e-16  
        28  2^2.7            2p       replayed          19         40    1.54  1.8e-16  
        29  29               mono     raced             65        106    1.60  2.8e-16  
        30  2.3.5            2p       replayed          23         31    1.34  3.0e-16  
        31  31               mono     raced             71        121    1.71  3.3e-16  
        32  2^5              2p       replayed          17         16    0.93  1.7e-16  
        33  3.11             2p       replayed          29         35    1.06  2.7e-16  
        34  2.17             flat     replayed          49         74    1.49  6.3e-16  
        35  5.7              2p       replayed          27         35    1.28  2.6e-16  
        36  2^2.3^2          2p       replayed          23         41    1.74  2.4e-16  
        37  37               prime    raced             95        184    1.94  7.2e-16  rader
        38  2.19             flat     replayed          57         87    1.51  2.6e-16  
        39  3.13             2p       replayed          35         44    1.27  6.4e-16  
        40  2^3.5            2p       replayed          23         41    1.75  1.5e-16  
        41  41               prime    raced             93        241    2.58  4.3e-16  rader
        42  2.3.7            2p       replayed          30         41    1.31  3.4e-16  
        43  43               prime    raced            110        265    2.41  6.2e-16  rader
        44  2^2.11           2p       replayed          30         43    1.46  2.6e-16  
        45  3^2.5            2p       replayed          35         48    1.36  5.0e-16  
        46  2.23             flat     replayed          79        121    1.23  5.9e-16  flips differ 1.25x
        47  47               mono     raced            187        327    1.59  4.0e-16  
        48  2^4.3            2p       replayed          28         44    1.48  2.7e-16  
        49  7^2              2p       replayed          36         47    1.17  3.2e-16  
        50  2.5^2            2p       replayed          34         53    1.33  3.6e-16  
        51  3.17             2p       replayed          55        114    1.99  4.1e-16  
        52  2^2.13           2p       replayed          38         50    1.28  5.3e-16  
        53  53               prime    raced            128        437    3.23  5.8e-16  rader
        54  2.3^3            2p       replayed          36         58    1.58  4.8e-16  
        55  5.11             2p       replayed          44         54    1.23  2.8e-16  
        56  2^3.7            2p       replayed          37         55    1.48  1.8e-16  
        57  3.19             2p       replayed          65        136    2.09  4.1e-16  
        58  2.29             flat     raced            131        185    1.39  4.3e-16  
        59  59               prime    raced            201        553    2.73  7.4e-16  bluestein
        60  2^2.3.5          2p       replayed          37         61    1.67  3.4e-16  
        61  61               prime    raced            145        602    4.15  5.9e-16  rader
        62  2.31             flat     raced            115        206    1.78  5.2e-16  
        63  3^2.7            2p       replayed          48         67    1.38  4.2e-16  
        64  2^6              2p       replayed          31         30    0.96  1.5e-16  
        65  5.13             2p       replayed          54         67    1.23  6.3e-16  
        66  2.3.11           2p       replayed          48         66    1.36  3.0e-16  
        67  67               prime    raced            170        740    4.36  4.9e-16  rader
        68  2^2.17           2p       replayed          61        138    2.25  3.7e-16  
        69  3.23             2p       replayed          94        198    2.09  4.1e-16  
        70  2.5.7            2p       replayed          47         78    1.65  2.3e-16  
        71  71               prime    raced            170        844    4.88  6.4e-16  rader
        72  2^3.3^2          2p       replayed          47         75    1.60  3.0e-16  
        73  73               prime    raced            170        904    5.22  8.7e-16  rader
        74  2.37             flat     raced            138        296    2.09  4.8e-16  
        75  3.5^2            2p       replayed          62         78    1.25  4.2e-16  
        76  2^2.19           flat     replayed          94        164    1.74  6.0e-16  
        77  7.11             2p       replayed          62         75    1.20  3.6e-16  
        78  2.3.13           2p       replayed          59         80    1.21  5.7e-16  
        79  79               prime    raced            206       1094    5.28  1.3e-15  rader
        80  2^4.5            2p       replayed          45         70    1.54  3.5e-16  
        81  3^4              2p       replayed          63         89    1.39  4.3e-16  
        82  2.41             flat     raced            164        370    2.25  6.2e-16  
        83  83               prime    raced            458       1252    2.73  5.4e-16  bluestein
        84  2^2.3.7          2p       replayed          55         87    1.57  3.1e-16  
        85  5.17             2p       replayed          86        190    2.15  4.7e-16  
        86  2.43             flat     raced            177        407    2.21  1.3e-15  
        87  3.29             2p       raced            156        331    1.84  4.4e-16  
        88  2^3.11           2p       replayed          61         87    1.42  2.8e-16  
        89  89               prime    raced            215       1507    6.99  3.9e-16  rader
        90  2.3^2.5          2p       replayed          64         95    1.47  4.0e-16  
        91  7.13             2p       replayed          76         91    1.20  5.0e-16  
        92  2^2.23           2p       replayed         115        234    1.94  4.1e-16  
        93  3.31             2p       raced            171        376    1.85  6.0e-16  
        94  2.47             flat     raced            201        509    2.46  8.6e-16  
        95  5.19             2p       replayed          99        227    2.15  4.4e-16  
        96  2^5.3            2p       replayed          52         96    1.84  3.0e-16  
        97  97               prime    raced            220        648    2.75  6.5e-16  rader
        98  2.7^2            flat     replayed         130        124    0.95  4.7e-16  
        99  3^2.11           2p       replayed          83        112    1.35  5.3e-16  
       100  2^2.5^2          2p       replayed          64        103    1.60  3.2e-16  
       101  101              prime    raced            249        560    2.19  8.0e-16  rader
       102  2.3.17           2p       replayed          87        203    2.29  5.8e-16  
       103  103              prime    raced            289        561    1.91  9.6e-16  rader
       104  2^3.13           2p       replayed          76        102    1.34  4.8e-16  
       105  3.5.7            2p       replayed          84        112    1.33  6.2e-16  
       106  2.53             prime    raced            447        665    1.47  3.1e-16  bluestein
       107  107              prime    raced            447        565    1.26  1.0e-15  bluestein
       108  2^2.3^3          2p       replayed          78        122    1.53  4.4e-16  
       109  109              prime    raced            272        572    2.06  8.0e-16  rader
       110  2.5.11           2p       replayed          83        129    1.36  3.7e-16  
       111  3.37             2p       raced            230        565    2.44  4.8e-16  
       112  2^4.7            2p       replayed          69         94    1.37  2.9e-16  
       113  113              prime    raced            274        569    2.07  5.3e-16  rader
       114  2.3.19           2p       replayed         108        245    2.11  3.4e-16  
       115  5.23             2p       replayed         141        342    2.36  3.7e-16  
       116  2^2.29           2p       raced            213        363    1.50  4.4e-16  
       117  3^2.13           2p       replayed         103        137    1.34  5.2e-16  
       118  2.59             prime    raced            454        837    1.83  2.8e-16  bluestein
       119  7.17             2p       replayed         114        278    2.34  3.7e-16  
       120  2^3.3.5          2p       replayed          79        125    1.58  3.7e-16  
       121  11^2             2p       replayed         107        122    1.15  3.7e-16  
       122  2.61             prime    raced            457        913    1.93  4.2e-16  bluestein
       123  3.41             2p       raced            285        727    2.49  4.7e-16  
       124  2^2.31           2p       raced            183        409    1.52  4.3e-16  flips differ 1.47x
       125  5^3              chain3   replayed         116        133    1.14  4.8e-16  
       126  2.3^2.7          2p       replayed          92        135    1.47  3.1e-16  
       127  127              prime    raced            405        588    1.44  5.4e-16  rader
       128  2^7              2p       replayed          66         68    1.03  3.2e-16  
       129  3.43             flat     raced            327        802    2.42  4.1e-16  
       130  2.5.13           2p       replayed         104        156    1.50  4.4e-16  
       131  131              prime    raced            353        973    2.70  5.7e-16  rader
       132  2^2.3.11         2p       replayed         102        145    1.42  4.8e-16  
       133  7.19             2p       replayed         148        326    2.19  4.5e-16  
       134  2.67             prime    raced            938       1111    1.18  3.6e-16  bluestein
       135  3^3.5            2p       replayed         114        155    1.36  3.2e-16  
       136  2^3.17           2p       replayed         117        275    2.34  6.0e-16  
       137  137              prime    raced            386        957    2.48  7.4e-16  rader
       138  2.3.23           2p       replayed         145        351    2.15  4.8e-16  
       139  139              prime    raced            465        959    1.92  7.5e-16  rader
       140  2^2.5.7          chain3   replayed         114        144    1.23  3.6e-16  
       141  3.47             flat     raced            521        985    1.88  5.4e-16  
       142  2.71             prime    raced            991       1265    1.23  4.3e-16  bluestein
       143  11.13            2p       replayed         133        150    1.12  5.5e-16  
       144  2^4.3^2          2p       replayed          95        142    1.48  3.7e-16  
       145  5.29             2p       raced            223        555    2.38  4.4e-16  
       146  2.73             prime    raced            976       1359    1.38  4.8e-16  bluestein
       147  3.7^2            2p       replayed         133        158    1.19  4.0e-16  
       148  2^2.37           2p       raced            195        589    2.93  3.2e-16  
       149  149              prime    raced            608        973    1.32  5.8e-16  rader
       150  2.3.5^2          2p       replayed         112        159    1.41  3.5e-16  
       151  151              prime    raced            404        970    2.40  6.5e-16  rader
       152  2^3.19           2p       replayed         142        331    2.17  4.1e-16  
       153  3^2.17           2p       replayed         147        371    2.49  5.6e-16  
       154  2.7.11           flat     replayed         194        214    0.96  3.5e-16  
       155  5.31             2p       raced            251        631    2.10  4.3e-16  
       156  2^2.3.13         2p       replayed         131        172    1.31  4.3e-16  
       157  157              prime    raced            434        981    2.25  9.9e-16  rader
       158  2.79             prime    raced            960       1610    1.67  3.3e-16  bluestein
       159  3.53             prime    raced            965       1333    1.38  3.0e-16  bluestein
       160  2^5.5            2p       replayed         101        163    1.61  2.2e-16  
       161  7.23             2p       replayed         200        474    2.14  4.5e-16  
       162  2.3^4            2p       replayed         122        189    1.51  4.6e-16  
       163  163              prime    raced            472       1250    2.64  6.8e-16  rader
       164  2^2.41           2p       raced            234        734    3.03  3.2e-16  
       165  3.5.11           2p       replayed         146        186    1.27  4.0e-16  
       166  2.83             prime    raced            959       1792    1.87  4.6e-16  bluestein
       167  167              prime    raced           1004       1253    1.25  4.1e-16  bluestein
       168  2^3.3.7          chain3   replayed         126        105    0.83  2.7e-16  
       169  13^2             2p       replayed         162        192    1.17  7.3e-16  
       170  2.5.17           2p       replayed         146        371    2.54  7.8e-16  
       171  3^2.19           2p       replayed         190        443    2.20  6.3e-16  
       172  2^2.43           2p       raced            251        808    3.21  5.2e-16  
       173  173              prime    raced            967       1261    1.30  3.0e-16  bluestein
       174  2.3.29           2p       raced            262        548    2.05  4.9e-16  
       175  5^2.7            2p       replayed         150        198    1.28  5.0e-16  
       176  2^4.11           2p       replayed         122        167    1.33  2.6e-16  
       177  3.59             prime    raced            981       1676    1.65  5.1e-16  bluestein
       178  2.89             prime    raced            970       2156    2.21  3.8e-16  bluestein
       179  179              prime    raced            970       1263    1.30  6.3e-16  bluestein
       180  2^2.3^2.5        chain3   replayed         136        206    1.52  4.4e-16  
       181  181              prime    raced            521       1268    2.42  7.5e-16  rader
       182  2.7.13           flat     replayed         242        250    1.03  4.7e-16  
       183  3.61             prime    raced            973       1836    1.88  5.1e-16  bluestein
       184  2^3.23           2p       replayed         210        477    2.27  3.8e-16  
       185  5.37             2p       raced            329        955    2.89  5.3e-16  
       186  2.3.31           2p       raced            307        616    1.57  4.3e-16  flips differ 1.28x
       187  11.17            2p       replayed         191        449    2.26  7.1e-16  
       188  2^2.47           2p       raced            297       1012    3.32  4.5e-16  
       189  3^3.7            2p       replayed         164        229    1.39  4.4e-16  
       190  2.5.19           2p       replayed         192        444    2.31  4.2e-16  
       191  191              prime    raced            590       1276    2.13  8.0e-16  rader
       192  2^6.3            2p       replayed         111        182    1.63  3.4e-16  
       193  193              prime    raced            475       1291    2.72  6.2e-16  rader
       194  2.97             prime    raced            980       2579    2.62  4.7e-16  bluestein
       195  3.5.13           2p       replayed         179        236    1.31  5.7e-16  
       196  2^2.7^2          chain3   replayed         168        219    1.30  3.1e-16  
       197  197              prime    raced            577       1295    2.24  5.3e-16  rader
       198  2.3^2.11         chain3   replayed         173        234    1.29  4.9e-16  
       199  199              prime    raced            617       1306    2.11  6.8e-16  rader
       200  2^3.5^2          chain3   replayed         141        217    1.54  2.5e-16  
       201  3.67             prime    raced           1016       2248    2.16  4.8e-16  bluestein
       202  2.101            prime    raced            974       2845    2.91  3.6e-16  bluestein
       203  7.29             2p       raced            318        783    1.99  3.9e-16  
       204  2^2.3.17         2p       replayed         194        440    2.14  6.1e-16  
       205  5.41             2p       raced            412       1231    2.91  5.7e-16  
       206  2.103            prime    raced           1006       2897    2.79  6.1e-16  bluestein
       207  3^2.23           2p       replayed         298        635    2.05  5.0e-16  
       208  2^4.13           2p       replayed         153        200    1.24  5.3e-16  
       209  11.19            2p       replayed         239        527    2.20  5.1e-16  
       210  2.3.5.7          2p       replayed         166        225    1.35  3.8e-16  
       211  211              prime    raced            713       1226    1.68  7.1e-16  rader
       212  2^2.53           prime    raced           1027       1325    1.21  6.0e-16  bluestein
       213  3.71             prime    raced           1017       2558    2.45  4.0e-16  bluestein
       214  2.107            prime    raced           1015       3147    3.06  4.9e-16  bluestein
       215  5.43             2p       raced            468       1355    2.89  6.7e-16  
       216  2^3.3^3          2p       replayed         162        250    1.53  3.9e-16  
       217  7.31             2p       raced            363        896    2.29  5.4e-16  
       218  2.109            prime    raced            997       3355    3.35  3.6e-16  bluestein
       219  3.73             prime    raced           1027       2750    2.40  5.5e-16  bluestein
       220  2^2.5.11         chain3   replayed         185        251    1.35  3.4e-16  
       221  13.17            2p       replayed         246        549    2.12  6.2e-16  
       222  2.3.37           2p       raced            301        911    2.87  3.9e-16  
       223  223              prime    raced           1018       1238    1.18  8.4e-16  bluestein
       224  2^5.7            2p       replayed         147        225    1.51  2.6e-16  
       225  3^2.5^2          chain3   replayed         193        271    1.39  4.8e-16  
       226  2.113            prime    raced           1000       3610    3.61  5.8e-16  bluestein
       227  227              prime    raced            985       1241    1.21  8.4e-16  bluestein
       228  2^2.3.19         2p       replayed         213        526    2.40  4.1e-16  
       229  229              prime    raced            735       1248    1.69  6.3e-16  rader
       230  2.5.23           2p       replayed         240        635    2.64  5.4e-16  
       231  3.7.11           chain3   replayed         216        278    1.28  3.5e-16  
       232  2^3.29           2p       raced            331        739    1.75  3.5e-16  flips differ 1.27x
       233  233              prime    raced            988       1251    1.26  5.8e-16  bluestein
       234  2.3^2.13         chain3   replayed         210        282    1.34  5.1e-16  
       235  5.47             flat     raced            706       1662    2.35  3.4e-16  
       236  2^2.59           prime    raced           1019       1668    1.63  6.0e-16  bluestein
       237  3.79             prime    raced           1050       3275    3.11  7.2e-16  bluestein
       238  2.7.17           flat     replayed         343        570    1.62  5.9e-16  
       239  239              prime    raced           1028       1257    1.20  8.5e-16  bluestein
       240  2^4.3.5          2p       replayed         170        250    1.47  4.0e-16  
       241  241              prime    raced            654       1256    1.88  5.0e-16  rader
       242  2.11^2           flat     replayed         340        367    1.08  8.7e-16  
       243  3^5              2p       replayed         227        302    1.30  6.5e-16  
       244  2^2.61           prime    raced           1047       1822    1.72  4.4e-16  bluestein
       245  5.7^2            chain3   replayed         234        281    1.19  3.0e-16  
       246  2.3.41           2p       raced            356       1107    3.08  3.2e-16  
       247  13.19            2p       replayed         296        646    2.18  5.2e-16  
       248  2^3.31           2p       raced            373        832    1.51  4.8e-16  flips differ 1.48x
       249  3.83             prime    raced           1036       3727    3.52  5.6e-16  bluestein
       250  2.5^3            chain3   replayed         192        308    1.60  3.7e-16  
       251  251              prime    raced            692       1275    1.84  7.2e-16  rader
       252  2^2.3^2.7        chain3   replayed         209        286    1.37  3.9e-16  
       253  11.23            2p       replayed         311        768    2.09  7.1e-16  
       254  2.127            prime    raced           1015       4534    4.46  4.3e-16  bluestein
       255  3.5.17           chain3   replayed         271        639    2.24  5.5e-16  
       256  2^8              2p       replayed         135        137    1.01  0.0e+00  
       257  257              prime    raced            659       2077    3.14  5.8e-16  rader
       258  2.3.43           2p       raced            389       1222    3.12  6.0e-16  
       259  7.37             2p       raced            439       1342    2.88  4.1e-16  
       260  2^2.5.13         chain3   replayed         230        299    1.30  4.4e-16  
       261  3^2.29           chain3   raced            414       1062    2.36  4.8e-16  
       262  2.131            prime    raced           2048       4844    2.34  5.5e-16  bluestein
       263  263              prime    raced           2071       2081    1.00  5.4e-16  bluestein
       264  2^3.3.11         chain3   replayed         228        306    1.32  3.8e-16  
       265  5.53             prime    raced           2054       2239    1.09  5.7e-16  bluestein
       266  2.7.19           flat     replayed         409        685    1.67  9.8e-16  
       267  3.89             prime    raced           2048       4531    2.20  4.9e-16  bluestein
       268  2^2.67           prime    raced           2172       2224    0.98  6.9e-16  bluestein
       269  269              prime    raced           2110       2089    0.98  6.2e-16  bluestein
       270  2.3^3.5          chain3   replayed         231        329    1.42  4.9e-16  
       271  271              prime    raced            782       2090    2.67  7.2e-16  rader
       272  2^4.17           2p       replayed         236        544    2.30  6.4e-16  
       273  3.7.13           chain3   replayed         262        335    1.27  6.2e-16  
       274  2.137            prime    raced           2223       5409    2.43  5.8e-16  bluestein
       275  5^2.11           chain3   replayed         252        332    1.31  3.4e-16  
       276  2^2.3.23         2p       replayed         300        759    2.48  4.5e-16  
       277  277              prime    raced           1009       2101    1.95  7.5e-16  rader
       278  2.139            prime    raced           2179       5477    2.47  4.9e-16  bluestein
       279  3^2.31           chain3   raced            468       1193    1.65  4.3e-16  flips differ 1.54x
       280  2^3.5.7          chain3   replayed         211        300    1.42  3.9e-16  
       281  281              prime    raced            808       2105    2.57  6.3e-16  rader
       282  2.3.47           2p       raced            454       1527    3.28  5.1e-16  
       283  283              prime    raced           2137       2102    0.97  5.4e-16  bluestein
       284  2^2.71           prime    raced           2121       2533    1.19  3.3e-16  bluestein
       285  3.5.19           chain3   replayed         322        758    2.24  4.4e-16  
       286  2.11.13          flat     replayed         417        441    1.05  5.6e-16  
       287  7.41             2p       raced            541       1727    2.98  3.7e-16  
       288  2^5.3^2          2p       replayed         206        336    1.63  3.8e-16  
       289  17^2             2p       replayed         342       1107    3.06  7.9e-16  
       290  2.5.29           2p       raced            605        982    1.56  4.4e-16  
       291  3.97             prime    raced           2063       2390    1.15  6.7e-16  bluestein
       292  2^2.73           prime    raced           2062       2719    1.31  5.1e-16  bluestein
       293  293              prime    raced           2134       2392    1.12  4.6e-16  bluestein
       294  2.3.7^2          chain3   replayed         253        341    1.35  3.7e-16  
       295  5.59             prime    raced           2088       2812    1.34  4.4e-16  bluestein
       296  2^3.37           2p       raced            393       1194    3.04  4.0e-16  
       297  3^3.11           chain3   replayed         298        392    1.31  5.7e-16  
       298  2.149            prime    raced           2195       6477    2.94  6.5e-16  bluestein
       299  13.23            2p       replayed         426        939    2.04  4.5e-16  
       300  2^2.3.5^2        chain3   replayed         235        357    1.52  4.1e-16  
       301  7.43             2p       raced            613       1904    2.95  4.3e-16  
       302  2.151            prime    raced           2077       2406    1.14  5.6e-16  bluestein
       303  3.101            prime    raced           2097       2402    1.14  4.1e-16  bluestein
       304  2^4.19           2p       replayed         288        657    2.12  2.7e-16  
       305  5.61             prime    raced           2045       3086    1.49  5.5e-16  bluestein
       306  2.3^2.17         chain3   replayed         327        685    2.09  5.5e-16  
       307  307              prime    raced           1166       2404    2.06  9.1e-16  rader
       308  2^2.7.11         chain3   replayed         275        386    1.39  4.5e-16  
       309  3.103            prime    raced           2129       2410    1.13  6.0e-16  bluestein
       310  2.5.31           2p       raced            406       1103    2.30  5.6e-16  
       311  311              prime    raced           1581       2416    1.50  7.1e-16  rader
       312  2^3.3.13         chain3   replayed         278        366    1.31  5.1e-16  
       313  313              prime    raced           1033       2417    2.30  9.6e-16  rader
       314  2.157            prime    raced           2090       2415    1.15  4.3e-16  bluestein
       315  3^2.5.7          chain3   replayed         292        399    1.36  5.0e-16  
       316  2^2.79           prime    raced           2166       3211    1.44  4.4e-16  bluestein
       317  317              prime    raced           2162       2419    1.08  6.3e-16  bluestein
       318  2.3.53           prime    raced           2090       2001    0.94  4.7e-16  bluestein
       319  11.29            2p       raced            522       1254    1.72  6.8e-16  flips differ 1.40x
       320  2^6.5            2p       replayed         206        323    1.56  3.0e-16  
       321  3.107            prime    raced           2093       2649    1.26  6.1e-16  bluestein
       322  2.7.23           flat     replayed         548        962    1.64  4.9e-16  
       323  17.19            2p       replayed         419       1294    3.06  6.8e-16  
       324  2^2.3^4          chain3   replayed         276        411    1.47  4.3e-16  
       325  5^2.13           chain3   replayed         306        409    1.33  5.9e-16  
       326  2.163            prime    raced           2221       2657    1.16  5.4e-16  bluestein
       327  3.109            prime    raced           2090       2649    1.26  6.6e-16  bluestein
       328  2^3.41           2p       raced            477       1485    3.03  3.9e-16  
       329  7.47             2p       raced            830       2337    2.76  5.1e-16  
       330  2.3.5.11         chain3   replayed         304        410    1.30  4.4e-16  
       331  331              prime    raced           1058       2649    2.49  5.7e-16  rader
       332  2^2.83           prime    raced           2106       3575    1.70  6.5e-16  bluestein
       333  3^2.37           2p       raced            573       1779    3.06  5.2e-16  
       334  2.167            prime    raced           2098       2655    1.25  5.3e-16  bluestein
       335  5.67             prime    raced           2218       3772    1.66  5.9e-16  bluestein
       336  2^4.3.7          chain3   replayed         248        346    1.40  2.5e-16  
       337  337              prime    raced           1007       2660    2.61  8.1e-16  rader
       338  2.13^2           flat     replayed         515        551    1.07  9.8e-16  
       339  3.113            prime    raced           2266       2657    1.15  6.3e-16  bluestein
       340  2^2.5.17         chain3   replayed         371        751    1.94  5.7e-16  
       341  11.31            2p       raced            623       1423    1.41  5.6e-16  flips differ 1.62x
       342  2.3^2.19         chain3   replayed         381        819    2.15  4.2e-16  
       343  7^3              chain3   replayed         355        403    1.13  4.9e-16  
       344  2^3.43           2p       raced            520       1635    3.14  4.4e-16  
       345  3.5.23           2p       replayed         423       1116    2.28  5.4e-16  
       346  2.173            prime    raced           2230       2668    1.19  5.1e-16  bluestein
       347  347              prime    raced           2108       2666    1.26  5.9e-16  bluestein
       348  2^2.3.29         2p       raced            448       1165    1.93  4.5e-16  flips differ 1.35x
       349  349              prime    raced           1506       2680    1.76  7.8e-16  rader
       350  2.5^2.7          chain3   replayed         288        438    1.50  3.5e-16  
       351  3^3.13           chain3   replayed         363        482    1.32  5.7e-16  
       352  2^5.11           2p       replayed         265        403    1.52  2.1e-16  
       353  353              prime    raced           1022       2678    2.59  6.5e-16  rader
       354  2.3.59           prime    raced           2248       2521    1.08  5.4e-16  bluestein
       355  5.71             prime    raced           2117       4296    2.02  4.7e-16  bluestein
       356  2^2.89           prime    raced           2228       4312    1.90  5.5e-16  bluestein
       357  3.7.17           2p       replayed         373        928    2.45  4.4e-16  
       358  2.179            prime    raced           2103       2685    1.27  5.2e-16  bluestein
       359  359              prime    raced           2103       2678    1.26  6.7e-16  bluestein
       360  2^3.3^2.5        chain3   replayed         294        433    1.48  4.4e-16  
       361  19^2             2p       replayed         495       1538    3.00  5.0e-16  
       362  2.181            prime    raced           2112       2689    1.27  5.7e-16  bluestein
       363  3.11^2           chain3   replayed         380        457    1.20  4.8e-16  
       364  2^2.7.13         chain3   replayed         367        463    1.26  5.2e-16  
       365  5.73             prime    raced           2136       4615    2.15  7.9e-16  bluestein
       366  2.3.61           prime    raced           2251       2776    1.20  4.6e-16  bluestein
       367  367              prime    raced           2214       2687    1.21  6.5e-16  bluestein
       368  2^4.23           2p       replayed         429        951    2.21  3.8e-16  
       369  3^2.41           2p       raced            685       2275    3.30  5.8e-16  
       370  2.5.37           2p       raced            517       1566    2.95  3.7e-16  
       371  7.53             prime    raced           2201       3146    1.41  4.9e-16  bluestein
       372  2^2.3.31         chain3   raced            586       1308    1.59  4.8e-16  flips differ 1.40x
       373  373              prime    raced           2105       2699    1.27  6.9e-16  bluestein
       374  2.11.17          flat     replayed         584        963    1.65  1.1e-15  
       375  3.5^3            chain3   replayed         341        474    1.38  6.8e-16  
       376  2^3.47           2p       raced            618       2046    3.29  4.2e-16  
       377  13.29            2p       raced            564       1518    2.29  5.3e-16  
       378  2.3^3.7          chain3   replayed         348        458    1.31  4.8e-16  
       379  379              prime    raced           1689       2699    1.58  1.1e-15  rader
       380  2^2.5.19         chain3   replayed         426        901    2.11  3.7e-16  
       381  3.127            prime    raced           2125       2704    1.27  5.2e-16  bluestein
       382  2.191            prime    raced           2112       2704    1.27  7.2e-16  bluestein
       383  383              prime    raced           2243       2701    1.15  6.4e-16  bluestein
       384  2^7.3            chain3   replayed         248        437    1.75  3.4e-16  
       385  5.7.11           chain3   replayed         374        480    1.27  5.6e-16  
       386  2.193            prime    raced           2256       3139    1.35  7.6e-16  bluestein
       387  3^2.43           2p       raced            754       2499    3.23  5.4e-16  
       388  2^2.97           prime    raced           2175       5226    2.37  4.7e-16  bluestein
       389  389              prime    raced           2122       3139    1.47  6.1e-16  bluestein
       390  2.3.5.13         chain3   replayed         368        480    1.29  6.2e-16  
       391  17.23            2p       replayed         556       1753    2.85  6.7e-16  
       392  2^3.7^2          chain3   replayed         331        429    1.30  2.7e-16  
       393  3.131            prime    raced           2216       3148    1.42  7.4e-16  bluestein
       394  2.197            prime    raced           2132       3147    1.46  6.0e-16  bluestein
       395  5.79             prime    raced           2130       5496    2.54  5.1e-16  bluestein
       396  2^2.3^2.11       flat     replayed         474        508    1.06  5.6e-16  
       397  397              prime    raced           1475       3151    2.11  8.9e-16  rader
       398  2.199            prime    raced           2135       3151    1.48  5.6e-16  bluestein
       399  3.7.19           2p       replayed         433       1083    2.50  6.2e-16  
       400  2^4.5^2          chain3   replayed         307        438    1.43  3.6e-16  
       401  401              prime    raced           1305       2591    1.82  7.6e-16  rader
       402  2.3.67           prime    raced           2350       3351    1.42  5.9e-16  bluestein
       403  13.31            2p       raced            994       1722    1.59  6.4e-16  
       404  2^2.101          prime    raced           2323       5774    2.48  4.9e-16  bluestein
       405  3^4.5            chain3   replayed         384        547    1.43  4.6e-16  
       406  2.7.29           flat     raced            917       1457    1.56  5.6e-16  
       407  11.37            2p       raced            681       2138    3.11  4.1e-16  
       408  2^3.3.17         chain3   replayed         410        905    2.21  5.5e-16  
       409  409              prime    raced           1648       2616    1.56  8.1e-16  rader
       410  2.5.41           2p       raced            610       1934    3.15  3.3e-16  
       411  3.137            prime    raced           2274       2594    1.11  1.0e-15  bluestein
       412  2^2.103          prime    raced           2233       5872    2.62  5.5e-16  bluestein
       413  7.59             prime    raced           2205       3953    1.78  4.9e-16  bluestein
       414  2.3^2.23         chain3   replayed         543       1162    1.92  4.6e-16  
       415  5.83             prime    raced           2246       6249    2.67  4.7e-16  bluestein
       416  2^5.13           2p       replayed         337        485    1.43  5.4e-16  
       417  3.139            prime    raced           2298       2613    1.08  6.4e-16  bluestein
       418  2.11.19          flat     replayed         685       1141    1.66  7.5e-16  
       419  419              prime    raced           2157       2600    1.20  6.0e-16  bluestein
       420  2^2.3.5.7        chain3   replayed         352        494    1.40  4.3e-16  
       421  421              prime    raced           1904       2625    1.36  7.0e-16  rader
       422  2.211            prime    raced           2272       2615    1.15  7.9e-16  bluestein
       423  3^2.47           2p       raced           1006       3069    2.97  6.0e-16  
       424  2^3.53           prime    raced           2150       2682    1.23  4.4e-16  bluestein
       425  5^2.17           chain3   replayed         460       1084    2.23  5.1e-16  
       426  2.3.71           prime    raced           2163       3832    1.75  5.2e-16  bluestein
       427  7.61             prime    raced           2182       4339    1.88  4.3e-16  bluestein
       428  2^2.107          prime    raced           2354       6369    2.68  5.4e-16  bluestein
       429  3.11.13          chain3   replayed         479        579    1.16  5.2e-16  
       430  2.5.43           2p       raced            669       2129    3.10  3.4e-16  
       431  431              prime    raced           2138       2617    1.22  7.7e-16  bluestein
       432  2^4.3^3          chain3   replayed         348        506    1.45  4.6e-16  
       433  433              prime    raced           1541       2832    1.73  8.8e-16  rader
       434  2.7.31           flat     raced            937       1630    1.53  1.5e-15  
       435  3.5.29           2p       raced            638       1783    2.19  5.5e-16  flips differ 1.28x
       436  2^2.109          prime    raced           2293       6765    2.92  5.4e-16  bluestein
       437  19.23            2p       replayed         668       2098    2.88  4.7e-16  
       438  2.3.73           prime    raced           2154       4094    1.89  5.3e-16  bluestein
       439  439              prime    raced           2170       2618    1.20  6.7e-16  bluestein
       440  2^3.5.11         chain3   replayed         380        539    1.42  5.3e-16  
       441  3^2.7^2          chain3   replayed         440        567    1.28  4.9e-16  
       442  2.13.17          flat     replayed         725       1175    1.56  1.3e-15  
       443  443              prime    raced           2147       2640    1.16  6.1e-16  bluestein
       444  2^2.3.37         2p       raced            635       1867    2.73  6.7e-16  
       445  5.89             prime    raced           2254       7595    3.36  4.5e-16  bluestein
       446  2.223            prime    raced           2162       2637    1.20  8.6e-16  bluestein
       447  3.149            prime    raced           2150       2637    1.21  7.3e-16  bluestein
       448  2^6.7            chain3   replayed         319        444    1.39  2.8e-16  
       449  449              prime    raced           1494       2654    1.74  8.3e-16  rader
       450  2.3^2.5^2        chain3   replayed         398        564    1.41  5.1e-16  
       451  11.41            2p       raced            836       2748    3.26  5.0e-16  
       452  2^2.113          prime    raced           2164       7310    3.36  7.0e-16  bluestein
       453  3.151            prime    raced           2155       2668    1.22  6.9e-16  bluestein
       454  2.227            prime    raced           2152       2644    1.21  7.6e-16  bluestein
       455  5.7.13           chain3   replayed         454        584    1.26  6.0e-16  
       456  2^3.3.19         chain3   replayed         527       1085    2.04  3.5e-16  
       457  457              prime    raced           1877       2648    1.31  8.2e-16  rader
       458  2.229            prime    raced           2178       2659    1.20  5.9e-16  bluestein
       459  3^3.17           chain3   replayed         562       1476    2.15  5.1e-16  
       460  2^2.5.23         chain3   replayed         589       1311    1.97  6.3e-16  
       461  461              prime    raced           2162       3036    1.23  7.4e-16  bluestein
       462  2.3.7.11         chain3   replayed         453        606    1.29  3.2e-16  
       463  463              prime    raced           1734       2654    1.51  6.4e-16  rader
       464  2^4.29           2p       raced            677       1480    2.13  2.8e-16  
       465  3.5.31           chain3   raced           1034       2015    1.91  4.7e-16  
       466  2.233            prime    raced           2163       2657    1.22  8.3e-16  bluestein
       467  467              prime    raced           2224       2665    1.18  6.2e-16  bluestein
       468  2^2.3^2.13       chain3   replayed         455        609    1.15  4.9e-16  
       469  7.67             prime    raced           2274       5301    2.29  3.9e-16  bluestein
       470  2.5.47           2p       raced            780       2654    3.32  5.3e-16  
       471  3.157            prime    raced           2165       2662    1.22  7.7e-16  bluestein
       472  2^3.59           prime    raced           2172       3377    1.41  4.7e-16  bluestein
       473  11.43            2p       raced            960       3026    3.06  6.0e-16  
       474  2.3.79           prime    raced           2238       4832    2.15  4.7e-16  bluestein
       475  5^2.19           chain3   replayed         581       1310    2.19  5.0e-16  
       476  2^2.7.17         chain3   replayed         500       1105    2.20  5.5e-16  
       477  3^2.53           prime    raced           2181       4111    1.87  5.1e-16  bluestein
       478  2.239            prime    raced           2169       2674    1.23  6.5e-16  bluestein
       479  479              prime    raced           2166       2657    1.22  6.1e-16  bluestein
       480  2^5.3.5          chain3   replayed         350        582    1.66  3.8e-16  
       481  13.37            2p       raced            802       2575    3.08  5.5e-16  
       482  2.241            prime    raced           2173       2684    1.21  7.3e-16  bluestein
       483  3.7.23           chain3   replayed         669       1583    2.15  6.0e-16  
       484  2^2.11^2         chain3   replayed         492        655    1.32  4.4e-16  
       485  5.97             prime    raced           2183       2679    1.22  5.9e-16  bluestein
       486  2.3^5            chain3   replayed         443        655    1.46  5.7e-16  
       487  487              prime    raced           2234       2671    1.18  4.2e-16  bluestein
       488  2^3.61           prime    raced           2172       3679    1.66  4.5e-16  bluestein
       489  3.163            prime    raced           2226       2689    1.21  6.1e-16  bluestein
       490  2.5.7^2          chain3   replayed         431        704    1.63  6.6e-16  
       491  491              prime    raced           1773       2679    1.48  7.1e-16  rader
       492  2^2.3.41         2p       raced            763       2302    2.96  4.4e-16  
       493  17.29            2p       raced            858       2655    2.50  6.2e-16  
       494  2.13.19          flat     replayed         819       1393    1.65  1.0e-15  
       495  3^2.5.11         chain3   replayed         517        666    1.29  5.6e-16  
       496  2^4.31           2p       raced            753       1666    1.62  4.5e-16  flips differ 1.37x
       497  7.71             prime    raced           2177       6039    2.75  4.7e-16  bluestein
       498  2.3.83           prime    raced           2305       5376    2.31  5.3e-16  bluestein
       499  499              prime    raced           2178       2687    1.22  8.3e-16  bluestein
       500  2^2.5^3          chain3   replayed         422        630    1.49  3.8e-16  
       501  3.167            prime    raced           2183       2689    1.23  7.5e-16  bluestein
       502  2.251            prime    raced           2258       2696    1.19  5.2e-16  bluestein
       503  503              prime    raced           2191       2684    1.20  5.6e-16  bluestein
       504  2^3.3^2.7        chain3   replayed         441        663    1.49  4.3e-16  
       505  5.101            prime    raced           2187       2699    1.23  6.3e-16  bluestein
       506  2.11.23          flat     replayed         890       1751    1.71  1.3e-15  
       507  3.13^2           chain3   replayed         576        711    1.23  6.2e-16  
       508  2^2.127          prime    raced           2183       9133    4.18  5.1e-16  bluestein
       509  509              prime    raced           2275       2690    1.17  5.0e-16  bluestein
       510  2.3.5.17         chain3   replayed         541       1263    2.29  5.7e-16  
       511  7.73             prime    raced           2184       6490    2.79  6.6e-16  bluestein
       512  2^9              2p       replayed         298        293    0.97  2.3e-16  
       513  3^3.19           2p       raced            580       1483    2.53  5.4e-16  
       514  2.257            prime    raced           4787       4571    0.94  6.9e-16  bluestein
       515  5.103            prime    raced           4768       4562    0.95  6.1e-16  bluestein
       516  2^2.3.43         2p       raced            836       2774    3.29  4.9e-16  
       517  11.47            2p       raced           1189       3711    3.12  5.5e-16  
       518  2.7.37           flat     raced           1180       2536    2.13  6.7e-16  
       519  3.173            prime    raced           4785       4565    0.95  4.7e-16  bluestein
       520  2^3.5.13         chain3   raced            484        767    1.58  6.2e-16  
       521  521              prime    raced           1940       4580    2.34  1.1e-15  rader
       522  2.3^2.29         chain3   raced            781       1970    2.15  4.3e-16  
       523  523              prime    raced           2897       4566    1.56  7.4e-16  rader
       524  2^2.131          prime    raced           4809       9772    2.02  4.7e-16  bluestein
       525  3.5^2.7          chain3   raced            489        724    1.47  6.4e-16  
       526  2.263            prime    raced           4850       4578    0.94  6.8e-16  bluestein
       527  17.31            2p       raced            911       2971    2.61  7.5e-16  flips differ 1.25x
       528  2^4.3.11         chain3   raced            483        728    1.50  4.2e-16  
       529  23^2             2p       raced            894       2783    2.71  6.5e-16  
       530  2.5.53           prime    raced           4818       4032    0.82  6.4e-16  bluestein
       531  3^2.59           prime    raced           4819       5166    1.07  4.5e-16  bluestein
       532  2^2.7.19         chain3   raced            609       1447    2.19  4.2e-16  
       533  13.41            2p       raced           1016       3303    3.23  6.4e-16  
       534  2.3.89           prime    raced           4831       6474    1.33  5.4e-16  bluestein
       535  5.107            prime    raced           4782       4572    0.94  5.3e-16  bluestein
       536  2^3.67           prime    raced           4851       4488    0.92  5.1e-16  bluestein
       537  3.179            prime    raced           4771       4587    0.96  5.8e-16  bluestein
       538  2.269            prime    raced           4795       4596    0.92  5.8e-16  bluestein
       539  7^2.11           chain3   raced            618        726    1.14  4.2e-16  
       540  2^2.3^3.5        chain3   raced            481        759    1.47  4.0e-16  
       541  541              prime    raced           3201       4630    1.37  8.2e-16  rader
       542  2.271            prime    raced           4872       4590    0.94  6.7e-16  bluestein
       543  3.181            prime    raced           4864       4668    0.95  5.1e-16  bluestein
       544  2^5.17           2p       raced            474       1351    2.59  5.2e-16  
       545  5.109            prime    raced           4944       4592    0.92  5.5e-16  bluestein
       546  2.3.7.13         chain3   raced            555        808    1.45  5.7e-16  
       547  547              prime    raced           2187       4618    2.10  1.0e-15  rader
       548  2^2.137          prime    raced           4864      10972    2.23  5.2e-16  bluestein
       549  3^2.61           prime    raced           4963       5711    0.73  4.6e-16  flips differ 1.59x bluestein
       550  2.5^2.11         chain3   raced            517        819    1.54  4.9e-16  
       551  19.29            2p       raced            981       3136    2.33  4.2e-16  flips differ 1.37x
       552  2^3.3.23         chain3   raced            681       1744    2.54  4.0e-16  
       553  7.79             prime    raced           6606       7900    1.07  4.6e-16  bluestein
       554  2.277            prime    raced           4856       4617    0.94  5.1e-16  bluestein
       555  3.5.37           2p       raced            922       3021    3.21  6.0e-16  
       556  2^2.139          prime    raced           4801      11031    2.24  5.2e-16  bluestein
       557  557              prime    raced           4834       4618    0.94  6.3e-16  bluestein
       558  2.3^2.31         chain3   raced            868       2213    1.77  5.0e-16  flips differ 1.44x
       559  13.43            2p       raced           1149       3652    3.15  5.6e-16  
       560  2^4.5.7          chain3   raced            449        654    1.44  3.3e-16  
       561  3.11.17          chain3   raced            687       1482    2.14  9.5e-16  
       562  2.281            prime    raced           4825       4639    0.95  5.3e-16  bluestein
       563  563              prime    raced           4839       4633    0.93  9.4e-16  bluestein
       564  2^2.3.47         2p       raced            963       3440    3.55  5.0e-16  
       565  5.113            prime    raced           4830       4645    0.96  5.7e-16  bluestein
       566  2.283            prime    raced           4781       4630    0.96  7.6e-16  bluestein
       567  3^4.7            chain3   raced            567        837    1.48  6.0e-16  
       568  2^3.71           prime    raced           4807       5105    1.06  3.7e-16  bluestein
       569  569              prime    raced           4805       4634    0.96  5.8e-16  bluestein
       570  2.3.5.19         chain3   raced            633       1505    2.35  5.1e-16  
       571  571              prime    raced           2522       4617    1.82  7.5e-16  rader
       572  2^2.11.13        chain3   raced            616        864    1.38  5.3e-16  
       573  3.191            prime    raced           4869       4627    0.94  6.8e-16  bluestein
       574  2.7.41           flat     raced           1519       3060    1.93  7.3e-16  
       575  5^2.23           2p       raced            747       1887    2.21  5.3e-16  
       576  2^6.3^2          chain3   raced            460        715    1.46  3.5e-16  
       577  577              prime    raced           1941       4783    2.46  7.9e-16  rader
       578  2.17^2           flat     raced           1016       2071    1.75  8.0e-16  
       579  3.193            prime    raced           4808       4765    0.97  4.9e-16  bluestein
       580  2^2.5.29         flat     raced            863       2118    2.08  6.2e-16  
       581  7.83             prime    raced           4821       8771    1.80  5.0e-16  bluestein
       582  2.3.97           prime    raced           4867       7845    1.61  6.0e-16  bluestein
       583  11.53            prime    raced           4808       4986    1.01  5.1e-16  bluestein
       584  2^3.73           prime    raced           4844       5476    1.13  4.4e-16  bluestein
       585  3^2.5.13         chain3   raced            639        854    1.30  5.1e-16  
       586  2.293            prime    raced           4811       4791    0.99  6.0e-16  bluestein
       587  587              prime    raced           4832       4782    0.98  6.6e-16  bluestein
       588  2^2.3.7^2        chain3   raced            539        824    1.52  5.8e-16  
       589  19.31            2p       raced           1068       3498    2.66  5.1e-16  
       590  2.5.59           prime    raced           4846       4916    1.00  4.6e-16  bluestein
       591  3.197            prime    raced           4828       4785    0.90  5.2e-16  bluestein
       592  2^4.37           2p       raced            799       2395    2.99  3.6e-16  
       593  593              prime    raced           2987       4809    1.59  7.0e-16  rader
       594  2.3^3.11         chain3   raced            612        890    1.44  4.7e-16  
       595  5.7.17           chain3   raced            707       1576    2.22  6.6e-16  
       596  2^2.149          prime    raced           4831      12971    2.63  5.4e-16  bluestein
       597  3.199            prime    raced           4929       4808    0.97  4.9e-16  bluestein
       598  2.13.23          flat     raced           1079       2120    1.96  9.1e-16  
       599  599              prime    raced           4872       4788    0.98  5.6e-16  bluestein
       600  2^3.3.5^2        chain3   raced            525        822    1.50  4.5e-16  
       601  601              prime    raced           3401       4812    1.41  8.9e-16  rader
       602  2.7.43           flat     raced           1737       3357    1.92  6.5e-16  
       603  3^2.67           prime    raced           4865       6904    1.42  3.9e-16  bluestein
       604  2^2.151          prime    raced           4808       4798    0.98  6.0e-16  bluestein
       605  5.11^2           chain3   raced            679        826    1.21  5.7e-16  
       606  2.3.101          prime    raced           4862       8694    1.32  4.8e-16  flips differ 1.36x bluestein
       607  607              prime    raced           4842       4829    0.96  6.6e-16  bluestein
       608  2^5.19           2p       raced            562       1626    2.86  3.6e-16  
       609  3.7.29           2p       raced           1447       2517    1.64  5.8e-16  
       610  2.5.61           prime    raced           4837       5372    1.10  6.2e-16  bluestein
       611  13.47            2p       raced           1402       4457    3.17  5.5e-16  
       612  2^2.3^2.17       chain3   raced            682       1659    2.33  5.5e-16  
       613  613              prime    raced           2866       4826    1.66  1.0e-15  rader
       614  2.307            prime    raced           4819       4825    0.93  6.2e-16  bluestein
       615  3.5.41           2p       raced           1138       3853    3.33  6.3e-16  
       616  2^3.7.11         chain3   raced            601        869    1.41  3.5e-16  
       617  617              prime    raced           2343       4828    2.04  7.3e-16  rader
       618  2.3.103          prime    raced           4883       8827    1.80  6.2e-16  bluestein
       619  619              prime    raced           4878       4807    0.98  6.0e-16  bluestein
       620  2^2.5.31         flat     raced            983       2384    2.00  6.2e-16  
       621  3^3.23           2p       raced            803       2086    2.60  6.7e-16  
       622  2.311            prime    raced           4793       4828    0.98  6.4e-16  bluestein
       623  7.89             prime    raced           4848      10676    2.18  4.8e-16  bluestein
       624  2^4.3.13         chain3   raced            601        873    1.34  4.1e-16  
       625  5^4              chain3   raced            756        944    1.05  3.7e-16  
       626  2.313            prime    raced           4862       4861    0.98  6.3e-16  bluestein
       627  3.11.19          chain3   raced            876       1752    1.79  5.5e-16  
       628  2^2.157          prime    raced           5006       4822    0.96  5.7e-16  bluestein
       629  17.37            2p       raced           1179       4224    3.26  6.4e-16  
       630  2.3^2.5.7        chain3   raced            602        867    1.39  4.5e-16  
       631  631              prime    raced           2954       4834    1.59  8.9e-16  rader
       632  2^3.79           prime    raced           4888       6460    1.32  5.4e-16  bluestein
       633  3.211            prime    raced           5035       4844    0.95  4.2e-16  bluestein
       634  2.317            prime    raced           4855       4998    0.83  4.9e-16  flips differ 1.28x bluestein
       635  5.127            prime    raced           4836       4827    0.98  5.7e-16  bluestein
       636  2^2.3.53         prime    raced           4838       4421    0.90  3.7e-16  bluestein
       637  7^2.13           chain3   raced            751        873    1.15  5.6e-16  
       638  2.11.29          flat     raced           1411       2600    1.75  5.8e-16  
       639  3^2.71           prime    raced           4908       7847    1.59  4.6e-16  bluestein
       640  2^7.5            2p       raced            505        823    1.62  4.2e-16  
       641  641              prime    raced           1931       6251    3.16  7.5e-16  rader
       642  2.3.107          prime    raced           4919       9558    1.94  5.2e-16  bluestein
       643  643              prime    raced           4895       6226    1.25  6.3e-16  bluestein
       644  2^2.7.23         flat     raced            849       2020    2.19  5.3e-16  
       645  3.5.43           2p       raced           1294       4246    3.25  5.5e-16  
       646  2.17.19          flat     raced           1113       2414    1.97  1.0e-15  
       647  647              prime    raced           4822       6232    1.29  7.4e-16  bluestein
       648  2^3.3^4          chain3   raced            620        896    1.44  5.2e-16  
       649  11.59            prime    raced           4894       6256    1.27  3.8e-16  bluestein
       650  2.5^2.13         chain3   raced            682        986    1.44  6.7e-16  
       651  3.7.31           2p       raced           1013       2845    2.79  5.2e-16  
       652  2^2.163          prime    raced           4899       6232    1.27  4.5e-16  bluestein
       653  653              prime    raced           4866       6253    1.28  5.4e-16  bluestein
       654  2.3.109          prime    raced           4838      10166    2.07  5.5e-16  bluestein
       655  5.131            prime    raced           4910       6231    1.22  6.0e-16  bluestein
       656  2^4.41           2p       raced            970       2974    2.95  5.3e-16  
       657  3^2.73           prime    raced           4844       8421    1.71  6.0e-16  bluestein
       658  2.7.47           flat     raced           2100       4121    1.94  6.5e-16  
       659  659              prime    raced           4916       6233    1.26  6.0e-16  bluestein
       660  2^2.3.5.11       chain3   raced            656        982    1.46  4.4e-16  
       661  661              prime    raced           2792       6266    2.21  7.4e-16  rader
       662  2.331            prime    raced           4878       6263    1.25  7.8e-16  bluestein
       663  3.13.17          chain3   raced            926       1809    1.86  5.5e-16  
       664  2^3.83           prime    raced           4854       7192    1.46  5.9e-16  bluestein
       665  5.7.19           flat     raced            924       1866    1.95  5.1e-16  
       666  2.3^2.37         chain3   raced           1183       3087    2.49  4.2e-16  
       667  23.29            2p       raced           1710       4110    2.22  6.1e-16  
       668  2^2.167          prime    raced           4921       6245    1.27  5.4e-16  bluestein
       669  3.223            prime    raced           4893       6271    1.28  5.4e-16  bluestein
       670  2.5.67           prime    raced           4843       6416    1.32  4.8e-16  bluestein
       671  11.61            prime    raced           4913       6875    1.37  7.1e-16  bluestein
       672  2^5.3.7          chain3   raced            591        877    1.48  3.3e-16  
       673  673              prime    raced           2466       6268    2.41  7.7e-16  rader
       674  2.337            prime    raced           4861       6278    1.26  7.2e-16  bluestein
       675  3^3.5^2          2p       raced            736        994    1.35  5.5e-16  
       676  2^2.13^2         chain3   raced            891       1028    1.15  8.2e-16  
       677  677              prime    raced           2840       6266    2.09  9.2e-16  rader
       678  2.3.113          prime    raced           4847      10963    2.22  4.9e-16  bluestein
       679  7.97             prime    raced           4870       6259    1.28  5.8e-16  bluestein
       680  2^3.5.17         chain3   raced            804       1709    2.12  5.9e-16  
       681  3.227            prime    raced           4957       6284    1.24  6.6e-16  bluestein
       682  2.11.31          flat     raced           1476       2908    1.97  1.1e-15  
       683  683              prime    raced           4932       6262    1.27  4.8e-16  bluestein
       684  2^2.3^2.19       chain3   raced            853       1889    2.21  6.2e-16  
       685  5.137            prime    raced           4850       6282    1.29  6.2e-16  bluestein
       686  2.7^3            flat     raced            940       1071    1.13  4.1e-16  
       687  3.229            prime    raced           4937       6273    1.27  5.4e-16  bluestein
       688  2^4.43           2p       raced           1082       3282    2.88  5.1e-16  
       689  13.53            prime    raced           4899       5965    1.20  5.0e-16  bluestein
       690  2.3.5.23         chain3   raced           1117       2121    1.77  4.2e-16  
       691  691              prime    raced           3683       6315    1.69  8.0e-16  rader
       692  2^2.173          prime    raced           4895       6345    1.22  4.8e-16  bluestein
       693  3^2.7.11         chain3   raced            786       1007    1.24  6.7e-16  
       694  2.347            prime    raced           4855       6295    1.29  6.5e-16  bluestein
       695  5.139            prime    raced           4905       6282    1.27  5.9e-16  bluestein
       696  2^3.3.29         chain3   raced           1108       2629    2.29  4.8e-16  
       697  17.41            2p       raced           1477       5270    3.50  8.6e-16  
       698  2.349            prime    raced           4903       6301    1.27  6.1e-16  bluestein
       699  3.233            prime    raced           4950       6285    1.24  5.9e-16  bluestein
       700  2^2.5^2.7        chain3   raced            664        928    1.38  3.7e-16  
       701  701              prime    raced           4170       6332    1.48  9.5e-16  rader
       702  2.3^3.13         chain3   raced            828       1071    1.29  5.7e-16  
       703  19.37            2p       raced           1411       4909    3.23  4.6e-16  
       704  2^6.11           2p       raced            655        909    1.37  3.1e-16  
       705  3.5.47           2p       raced           1600       5179    3.16  6.6e-16  
       706  2.353            prime    raced           4868       6314    1.29  5.3e-16  bluestein
       707  7.101            prime    raced           4921       6299    1.26  5.5e-16  bluestein
       708  2^2.3.59         prime    raced           4990       5509    1.08  5.7e-16  bluestein
       709  709              prime    raced           4973       6306    1.27  5.5e-16  bluestein
       710  2.5.71           prime    raced           4904       7215    1.46  5.5e-16  bluestein
       711  3^2.79           prime    raced           4921      10022    2.03  6.4e-16  bluestein
       712  2^3.89           prime    raced           4939       8647    1.75  5.2e-16  bluestein
       713  23.31            2p       raced           1554       4557    2.50  6.7e-16  
       714  2.3.7.17         chain3   raced            914       1807    1.96  7.4e-16  
       715  5.11.13          chain3   raced            880        997    1.11  5.8e-16  
       716  2^2.179          prime    raced           4887       6304    1.23  6.3e-16  bluestein
       717  3.239            prime    raced           5042       6419    1.20  6.2e-16  bluestein
       718  2.359            prime    raced           4900       6330    1.29  5.1e-16  bluestein
       719  719              prime    raced           4917       6369    1.29  5.4e-16  bluestein
       720  2^4.3^2.5        chain3   raced            677        937    1.38  3.9e-16  
       721  7.103            prime    raced           4876       6364    1.29  5.1e-16  bluestein
       722  2.19^2           flat     raced           1344       2800    2.01  1.0e-15  
       723  3.241            prime    raced           4878       6312    1.27  4.5e-16  bluestein
       724  2^2.181          prime    raced           4868       6307    1.29  7.2e-16  bluestein
       725  5^2.29           2p       raced           1386       3026    1.87  5.3e-16  
       726  2.3.11^2         chain3   raced            865       1086    1.23  4.5e-16  
       727  727              prime    raced           2975       6315    2.10  6.5e-16  rader
       728  2^3.7.13         chain3   raced            777       1047    1.35  5.1e-16  
       729  3^6              2p       raced            822       1083    1.28  5.1e-16  
       730  2.5.73           prime    raced           4953       7716    1.55  4.3e-16  bluestein
       731  17.43            2p       raced           1625       5763    3.46  6.0e-16  
       732  2^2.3.61         prime    raced           4920       5967    1.20  6.0e-16  bluestein
       733  733              prime    raced           4878       6332    1.26  6.2e-16  bluestein
       734  2.367            prime    raced           4994       6343    1.27  5.3e-16  bluestein
       735  3.5.7^2          flat     raced            917       1041    1.11  6.9e-16  
       736  2^5.23           2p       raced            831       2269    2.38  3.3e-16  
       737  11.67            prime    raced           4884       8379    1.70  5.5e-16  bluestein
       738  2.3^2.41         chain3   raced           1511       3779    2.48  5.1e-16  
       739  739              prime    raced           4898       6339    1.28  5.1e-16  bluestein
       740  2^2.5.37         flat     raced           1323       3343    2.36  5.4e-16  
       741  3.13.19          chain3   raced           1128       2137    1.88  5.9e-16  
       742  2.7.53           prime    raced           4953       5301    1.03  4.4e-16  bluestein
       743  743              prime    raced           4932       6330    1.21  6.3e-16  bluestein
       744  2^3.3.31         flat     raced           1368       2939    2.00  5.3e-16  
       745  5.149            prime    raced           4890       6347    1.30  6.8e-16  bluestein
       746  2.373            prime    raced           4916       6354    1.28  7.4e-16  bluestein
       747  3^2.83           prime    raced           4941      11389    2.28  6.1e-16  bluestein
       748  2^2.11.17        flat     raced           1083       1920    1.77  7.9e-16  
       749  7.107            prime    raced           4965       6354    1.28  6.6e-16  bluestein
       750  2.3.5^3          chain3   raced            845       1036    1.21  4.7e-16  
       751  751              prime    raced           4897       6342    1.14  6.3e-16  bluestein
       752  2^4.47           2p       raced           1264       4112    3.06  4.4e-16  
       753  3.251            prime    raced           5134       6371    1.22  6.0e-16  bluestein
       754  2.13.29          flat     raced           1641       3136    1.61  1.0e-15  
       755  5.151            prime    raced           4878       6342    1.27  5.7e-16  bluestein
       756  2^2.3^3.7        chain3   raced            761       1054    1.37  3.9e-16  
       757  757              prime    raced           4447       6352    1.28  6.3e-16  rader
       758  2.379            prime    raced           5072       6376    1.24  6.3e-16  bluestein
       759  3.11.23          flat     raced           1336       2524    1.82  1.6e-15  
       760  2^3.5.19         chain3   raced            925       2034    2.17  4.9e-16  
       761  761              prime    raced           3359       6356    1.83  7.6e-16  rader
       762  2.3.127          prime    raced           4988      13832    1.71  5.9e-16  flips differ 1.63x bluestein
       763  7.109            prime    raced           4903       6363    1.28  5.5e-16  bluestein
       764  2^2.191          prime    raced           4888       6369    1.28  6.1e-16  bluestein
       765  3^2.5.17         flat     raced           1060       2118    1.95  6.0e-16  
       766  2.383            prime    raced           4898       6382    1.23  5.3e-16  bluestein
       767  13.59            prime    raced           4926       7483    1.52  5.9e-16  bluestein
       768  2^8.3            2p       raced            626        905    1.43  4.3e-16  
       769  769              prime    raced           2321       6335    2.66  6.4e-16  rader
       770  2.5.7.11         chain3   raced            839       1301    1.54  3.9e-16  
       771  3.257            prime    raced           4958       6317    1.27  7.5e-16  bluestein
       772  2^2.193          prime    raced           4884       6354    1.27  5.9e-16  bluestein
       773  773              prime    raced           4917       6329    1.27  6.4e-16  bluestein
       774  2.3^2.43         chain3   raced           1665       4163    2.48  4.7e-16  
       775  5^2.31           2p       raced           1551       3424    1.40  6.3e-16  flips differ 1.58x
       776  2^3.97           prime    raced           4908      10490    2.13  3.1e-16  bluestein
       777  3.7.37           2p       raced           1318       4237    3.20  5.5e-16  
       778  2.389            prime    raced           4936       6339    1.28  6.1e-16  bluestein
       779  19.41            2p       raced           1747       6127    3.33  4.7e-16  
       780  2^2.3.5.13       chain3   raced            865       1180    1.35  7.1e-16  
       781  11.71            prime    raced           4927       9550    1.92  5.4e-16  bluestein
       782  2.17.23          flat     raced           1607       3263    2.00  1.4e-15  
       783  3^3.29           2p       raced           1207       3390    2.76  4.1e-16  
       784  2^4.7^2          chain3   raced            787        904    1.13  3.9e-16  
       785  5.157            prime    raced           4943       6347    1.27  7.0e-16  bluestein
       786  2.3.131          prime    raced           4888      14748    3.00  6.0e-16  bluestein
       787  787              prime    raced           5096       6311    1.24  5.7e-16  bluestein
       788  2^2.197          prime    raced           4945       6321    1.26  6.3e-16  bluestein
       789  3.263            prime    raced           5086       6348    1.18  6.9e-16  bluestein
       790  2.5.79           prime    raced           4893       8998    1.82  6.3e-16  bluestein
       791  7.113            prime    raced           5083       6326    1.22  6.1e-16  bluestein
       792  2^3.3^2.11       chain3   raced            809       1180    1.45  5.8e-16  
       793  13.61            prime    raced           4994       8201    1.64  5.1e-16  bluestein
       794  2.397            prime    raced           5120       6355    1.20  7.6e-16  bluestein
       795  3.5.53           prime    raced           4931       6930    1.40  5.7e-16  bluestein
       796  2^2.199          prime    raced           5044       6326    1.20  5.8e-16  bluestein
       797  797              prime    raced           4911       6360    1.28  5.4e-16  bluestein
       798  2.3.7.19         chain3   raced           1081       2150    1.92  3.9e-16  
       799  17.47            2p       raced           1968       6934    3.35  5.6e-16  
       800  2^5.5^2          2p       raced            740       1094    1.47  3.1e-16  
       801  3^2.89           prime    raced           4993      13822    2.67  7.2e-16  bluestein
       802  2.401            prime    raced           4919       5848    1.18  4.9e-16  bluestein
       803  11.73            prime    raced           4961      10250    1.97  5.0e-16  bluestein
       804  2^2.3.67         prime    raced           4944       7235    1.45  4.1e-16  bluestein
       805  5.7.23           flat     raced           1323       2685    1.77  5.5e-16  
       806  2.13.31          flat     raced           1809       3499    1.91  1.8e-15  
       807  3.269            prime    raced           5180       5925    1.03  5.0e-16  bluestein
       808  2^3.101          prime    raced           5004      11596    2.24  6.0e-16  bluestein
       809  809              prime    raced           4988       5845    1.17  4.9e-16  bluestein
       810  2.3^4.5          chain3   raced            925       1199    1.30  4.5e-16  
       811  811              prime    raced           4974       5826    1.17  5.2e-16  bluestein
       812  2^2.7.29         flat     raced           1478       3058    1.76  6.1e-16  
       813  3.271            prime    raced           4940       5844    1.17  5.6e-16  bluestein
       814  2.11.37          flat     raced           1952       4036    2.03  2.7e-15  
       815  5.163            prime    raced           4904       5816    1.18  6.5e-16  bluestein
       816  2^4.3.17         chain3   raced            985       2007    2.02  4.7e-16  
       817  19.43            2p       raced           1908       6685    3.30  5.0e-16  
       818  2.409            prime    raced           4919       5856    1.18  5.4e-16  bluestein
       819  3^2.7.13         chain3   raced           1138       1213    1.06  7.5e-16  
       820  2^2.5.41         flat     raced           1435       4109    2.85  4.8e-16  
       821  821              prime    raced           5005       5850    1.17  5.5e-16  bluestein
       822  2.3.137          prime    raced           4948      16436    3.30  5.4e-16  bluestein
       823  823              prime    raced           4998       5842    1.15  7.7e-16  bluestein
       824  2^3.103          prime    raced           5144      11856    2.29  5.9e-16  bluestein
       825  3.5^2.11         flat     raced           1029       1221    1.18  4.4e-16  
       826  2.7.59           prime    raced           4987       6584    1.32  5.2e-16  bluestein
       827  827              prime    raced           4967       5834    1.17  6.2e-16  bluestein
       828  2^2.3^2.23       flat     raced           1226       2654    2.02  4.5e-16  
       829  829              prime    raced           4469       5891    1.29  1.1e-15  rader
       830  2.5.83           prime    raced           4966       9971    2.01  6.7e-16  bluestein
       831  3.277            prime    raced           4959       5826    1.16  7.0e-16  bluestein
       832  2^6.13           2p       raced            834       1081    1.29  5.0e-16  
       833  7^2.17           flat     raced           1229       2309    1.82  5.8e-16  
       834  2.3.139          prime    raced           4922      16624    3.32  6.2e-16  bluestein
       835  5.167            prime    raced           4991       5840    1.16  6.9e-16  bluestein
       836  2^2.11.19        chain3   raced           1222       2292    1.87  4.7e-16  
       837  3^3.31           2p       raced           1669       3812    2.28  5.8e-16  
       838  2.419            prime    raced           4996       5881    1.17  5.3e-16  bluestein
       839  839              prime    raced           4954       5838    1.16  7.2e-16  bluestein
       840  2^3.3.5.7        chain3   raced            853       1143    1.34  4.0e-16  
       841  29^2             2p       raced           2441       6004    2.29  5.0e-16  
       842  2.421            prime    raced           4932       5876    1.18  5.7e-16  bluestein
       843  3.281            prime    raced           4982       5838    1.16  4.8e-16  bluestein
       844  2^2.211          prime    raced           4941       5847    1.16  5.2e-16  bluestein
       845  5.13^2           flat     raced           1187       1257    1.05  6.3e-16  
       846  2.3^2.47         chain3   raced           1891       5148    2.58  5.0e-16  
       847  7.11^2           chain3   raced           1139       1196    1.02  3.8e-16  
       848  2^4.53           prime    raced           4927       5392    1.09  4.1e-16  bluestein
       849  3.283            prime    raced           5043       5880    1.17  5.0e-16  bluestein
       850  2.5^2.17         chain3   raced           1138       2180    1.88  5.9e-16  
       851  23.37            flat     raced           2058       6392    2.79  9.7e-16  
       852  2^2.3.71         prime    raced           5100       8204    1.57  5.0e-16  bluestein
       853  853              prime    raced           5043       5885    1.15  6.7e-16  bluestein
       854  2.7.61           prime    raced           5064       7158    1.41  4.3e-16  bluestein
       855  3^2.5.19         flat     raced           1250       2499    1.97  5.4e-16  
       856  2^3.107          prime    raced           5135      12842    2.46  6.6e-16  bluestein
       857  857              prime    raced           4998       5896    1.17  6.4e-16  bluestein
       858  2.3.11.13        chain3   raced           1108       1309    1.18  6.3e-16  
       859  859              prime    raced           3622       5869    1.61  9.7e-16  rader
       860  2^2.5.43         flat     raced           1569       4502    2.72  5.7e-16  
       861  3.7.41           2p       raced           1655       5424    3.26  4.6e-16  
       862  2.431            prime    raced           4978       5912    1.18  7.2e-16  bluestein
       863  863              prime    raced           5004       5852    1.16  5.8e-16  bluestein
       864  2^5.3^3          2p       raced            800       1196    1.46  5.8e-16  
       865  5.173            prime    raced           5054       5895    1.16  5.6e-16  bluestein
       866  2.433            prime    raced           4931       5957    1.18  5.7e-16  bluestein
       867  3.17^2           chain3   raced           1436       3631    2.44  6.4e-16  
       868  2^2.7.31         flat     raced           1502       3454    1.93  4.7e-16  
       869  11.79            prime    raced           4974      12225    2.43  4.6e-16  bluestein
       870  2.3.5.29         chain3   raced           1526       3212    1.63  5.6e-16  flips differ 1.29x
       871  13.67            prime    raced           5025      10083    2.00  6.4e-16  bluestein
       872  2^3.109          prime    raced           5026      13653    2.72  6.4e-16  bluestein
       873  3^2.97           prime    raced           4947       5911    1.17  6.1e-16  bluestein
       874  2.19.23          flat     raced           1648       3766    1.98  1.1e-15  
       875  5^3.7            flat     raced           1049       1296    1.21  4.5e-16  
       876  2^2.3.73         prime    raced           4988       8828    1.75  3.9e-16  bluestein
       877  877              prime    raced           5021       5900    1.18  4.3e-16  bluestein
       878  2.439            prime    raced           5009       5905    1.17  5.9e-16  bluestein
       879  3.293            prime    raced           5107       5876    1.15  5.3e-16  bluestein
       880  2^4.5.11         chain3   raced            931       1191    1.26  3.2e-16  
       881  881              prime    raced           3426       5902    1.69  5.0e-16  rader
       882  2.3^2.7^2        chain3   raced           1270       1254    0.97  5.2e-16  
       883  883              prime    raced           5026       5885    1.17  7.0e-16  bluestein
       884  2^2.13.17        chain3   raced           1284       2289    1.76  6.3e-16  
       885  3.5.59           prime    raced           5196       8676    1.65  5.1e-16  bluestein
       886  2.443            prime    raced           5089       5921    1.15  7.0e-16  bluestein
       887  887              prime    raced           5102       6032    1.06  5.7e-16  bluestein
       888  2^3.3.37         flat     raced           1670       4126    2.34  5.9e-16  
       889  7.127            prime    raced           5101       8791    1.16  5.2e-16  flips differ 1.85x bluestein
       890  2.5.89           prime    raced           4992      11979    2.38  5.6e-16  bluestein
       891  3^4.11           chain3   raced           1209       1412    1.16  5.6e-16  
       892  2^2.223          prime    raced           5060       5978    1.17  4.6e-16  bluestein
       893  19.47            2p       raced           2203       7987    3.62  5.3e-16  
       894  2.3.149          prime    raced           4957      19548    3.94  7.8e-16  bluestein
       895  5.179            prime    raced           5011       5891    1.17  5.4e-16  bluestein
       896  2^7.7            chain3   raced            944       1151    1.19  4.0e-16  
       897  3.13.23          flat     raced           1616       3100    1.81  1.0e-15  
       898  2.449            prime    raced           5049       5924    1.17  5.6e-16  bluestein
       899  29.31            flat     raced           2294       6631    2.44  1.0e-15  
       900  2^2.3^2.5^2      chain3   raced            963       1304    1.35  3.7e-16  
       901  17.53            prime    raced           5140       9040    1.71  4.8e-16  bluestein
       902  2.11.41          flat     raced           2421       4905    2.00  8.6e-16  
       903  3.7.43           2p       raced           1825       5963    3.27  5.1e-16  
       904  2^3.113          prime    raced           5064      14742    2.83  5.8e-16  bluestein
       905  5.181            prime    raced           4962       5934    1.18  4.8e-16  bluestein
       906  2.3.151          prime    raced           4988       5937    1.16  5.0e-16  bluestein
       907  907              prime    raced           5032       5907    1.09  7.8e-16  bluestein
       908  2^2.227          prime    raced           4974       5901    1.18  5.2e-16  bluestein
       909  3^2.101          prime    raced           5144       5948    1.12  6.3e-16  bluestein
       910  2.5.7.13         chain3   raced           1084       1561    1.41  5.9e-16  
       911  911              prime    raced           5100       6010    1.16  6.2e-16  bluestein
       912  2^4.3.19         chain3   raced           1172       2389    2.02  4.7e-16  
       913  11.83            prime    raced           5062      13910    2.74  4.6e-16  bluestein
       914  2.457            prime    raced           5001       5945    1.17  8.1e-16  bluestein
       915  3.5.61           prime    raced           5078       9531    1.63  6.6e-16  bluestein
       916  2^2.229          prime    raced           5084       5918    1.15  6.1e-16  bluestein
       917  7.131            prime    raced           5067       5963    1.06  8.5e-16  bluestein
       918  2.3^3.17         chain3   raced           1255       2368    1.84  7.8e-16  
       919  919              prime    raced           4989       5926    1.16  6.0e-16  bluestein
       920  2^3.5.23         flat     raced           1365       2875    1.94  5.4e-16  
       921  3.307            prime    raced           5060       5962    1.18  6.8e-16  bluestein
       922  2.461            prime    raced           5062       5975    1.12  7.1e-16  bluestein
       923  13.71            prime    raced           5106      11509    2.25  4.9e-16  bluestein
       924  2^2.3.7.11       chain3   raced            995       1468    1.46  4.6e-16  
       925  5^2.37           flat     raced           1773       5092    2.63  5.5e-16  
       926  2.463            prime    raced           5207       5958    1.13  5.3e-16  bluestein
       927  3^2.103          prime    raced           5055       5935    1.15  5.8e-16  bluestein
       928  2^5.29           2p       raced           2140       3431    1.13  5.8e-16  flips differ 1.42x
       929  929              prime    raced           4812       5967    1.20  5.3e-16  rader
       930  2.3.5.31         chain3   raced           1763       3622    1.69  5.7e-16  
       931  7^2.19           flat     raced           1423       2658    1.76  7.7e-16  
       932  2^2.233          prime    raced           5085       5905    1.16  7.0e-16  bluestein
       933  3.311            prime    raced           5080       6013    1.18  6.2e-16  bluestein
       934  2.467            prime    raced           5155       5995    1.14  6.0e-16  bluestein
       935  5.11.17          flat     raced           1406       2548    1.81  6.5e-16  
       936  2^3.3^2.13       chain3   raced           1023       1423    1.39  6.9e-16  
       937  937              prime    raced           3668       5986    1.62  7.3e-16  rader
       938  2.7.67           prime    raced           5166       8826    1.70  7.5e-16  bluestein
       939  3.313            prime    raced           5074       6006    1.18  4.4e-16  bluestein
       940  2^2.5.47         flat     raced           1819       5672    3.08  5.2e-16  
       941  941              prime    raced           5100       5967    1.17  7.1e-16  bluestein
       942  2.3.157          prime    raced           5076       5986    1.16  5.6e-16  bluestein
       943  23.41            flat     raced           2455       7886    3.10  1.3e-15  
       944  2^4.59           prime    raced           5054       6836    1.35  5.8e-16  bluestein
       945  3^3.5.7          flat     raced           1228       1456    1.18  6.6e-16  
       946  2.11.43          flat     raced           2673       5475    2.02  5.6e-16  
       947  947              prime    raced           5131       5942    1.14  6.6e-16  bluestein
       948  2^2.3.79         prime    raced           5244      10352    1.82  5.1e-16  bluestein
       949  13.73            prime    raced           5060      12387    2.44  6.9e-16  bluestein
       950  2.5^2.19         chain3   raced           1336       2606    1.95  4.9e-16  
       951  3.317            prime    raced           5045       6128    1.19  8.4e-16  bluestein
       952  2^3.7.17         chain3   raced           1173       2389    1.99  6.0e-16  
       953  953              prime    raced           4118       6105    1.46  8.2e-16  rader
       954  2.3^2.53         prime    raced           5108       6693    1.31  5.9e-16  bluestein
       955  5.191            prime    raced           5011       5961    1.18  6.3e-16  bluestein
       956  2^2.239          prime    raced           5070       5962    1.17  5.7e-16  bluestein
       957  3.11.29          chain3   raced           2209       4038    1.33  5.4e-16  flips differ 1.38x
       958  2.479            prime    raced           5291       5997    1.13  6.3e-16  bluestein
       959  7.137            prime    raced           5065       5928    1.08  7.8e-16  bluestein
       960  2^6.3.5          2p       raced           1009       1272    0.98  4.7e-16  flips differ 1.30x
       961  31^2             2p       raced           3327       7539    1.98  6.0e-16  
       962  2.13.37          flat     raced           2245       4914    2.12  9.1e-16  
       963  3^2.107          prime    raced           5002       5921    1.18  7.5e-16  bluestein
       964  2^2.241          prime    raced           5068       5946    1.17  6.3e-16  bluestein
       965  5.193            prime    raced           5397       6032    1.11  7.4e-16  bluestein
       966  2.3.7.23         chain3   raced           1709       3018    1.60  5.2e-16  
       967  967              prime    raced           5086       5972    1.16  5.2e-16  bluestein
       968  2^3.11^2         chain3   raced           1129       1447    1.28  4.3e-16  
       969  3.17.19          flat     raced           1682       4098    2.39  1.4e-15  
       970  2.5.97           prime    raced           5159      14343    2.69  5.5e-16  bluestein
       971  971              prime    raced           5108       5970    1.11  6.3e-16  bluestein
       972  2^2.3^5          chain3   raced           1081       1469    1.35  5.4e-16  
       973  7.139            prime    raced           5106       6037    1.14  5.9e-16  bluestein
       974  2.487            prime    raced           5105       6013    1.17  6.5e-16  bluestein
       975  3.5^2.13         flat     raced           1499       1497    0.85  1.2e-15  
       976  2^4.61           prime    raced           5082       7447    1.46  5.2e-16  bluestein
       977  977              prime    raced           5178       5999    1.11  6.5e-16  bluestein
       978  2.3.163          prime    raced           5024       5995    1.18  6.2e-16  bluestein
       979  11.89            prime    raced           5146      16895    3.16  6.0e-16  bluestein
       980  2^2.5.7^2        flat     raced           1107       1404    1.25  6.0e-16  
       981  3^2.109          prime    raced           5190       6004    1.11  6.4e-16  bluestein
       982  2.491            prime    raced           5121       6042    1.17  5.2e-16  bluestein
       983  983              prime    raced           5104       5948    1.16  7.3e-16  bluestein
       984  2^3.3.41         flat     raced           1940       5127    2.51  5.8e-16  
       985  5.197            prime    raced           5135       6003    1.12  7.0e-16  bluestein
       986  2.17.29          flat     raced           2523       4779    1.76  6.8e-16  
       987  3.7.47           2p       raced           2236       7355    3.24  5.1e-16  
       988  2^2.13.19        chain3   raced           1587       2797    1.70  6.6e-16  
       989  23.43            2p       raced           2653       8656    2.90  7.0e-16  
       990  2.3^2.5.11       chain3   raced           1164       1497    1.27  6.0e-16  
       991  991              prime    raced           5027       5977    1.18  6.5e-16  bluestein
       992  2^5.31           2p       raced           1661       4050    2.16  4.3e-16  
       993  3.331            prime    raced           5050       6012    1.19  6.3e-16  bluestein
       994  2.7.71           prime    raced           5056       9854    1.95  4.7e-16  bluestein
       995  5.199            prime    raced           5062       5999    1.18  7.1e-16  bluestein
       996  2^2.3.83         prime    raced           5106      11515    2.21  5.6e-16  bluestein
       997  997              prime    raced           4987       6027    1.08  6.5e-16  bluestein
       998  2.499            prime    raced           5024       6056    1.18  5.1e-16  bluestein
       999  3^3.37           2p       raced           1773       5679    3.08  7.6e-16  
      1000  2^3.5^3          chain3   raced           1058       1411    1.33  3.7e-16  
      1001  7.11.13          chain3   raced           1469       1510    0.95  6.1e-16  
      1002  2.3.167          prime    raced           5155       6040    1.12  7.1e-16  bluestein
      1003  17.59            prime    raced           5051      11248    2.21  4.5e-16  bluestein
      1004  2^2.251          prime    raced           5105       5961    1.16  5.8e-16  bluestein
      1005  3.5.67           prime    raced           5124      11770    2.27  5.5e-16  bluestein
      1006  2.503            prime    raced           5126       6016    1.16  5.9e-16  bluestein
      1007  19.53            prime    raced           5127      10511    2.00  6.4e-16  bluestein
      1008  2^4.3^2.7        chain3   raced           1060       1315    1.23  5.1e-16  
      1009  1009             prime    raced           5136       6040    1.15  7.9e-16  bluestein
      1010  2.5.101          prime    raced           5143      15776    2.64  5.0e-16  bluestein
      1011  3.337            prime    raced           5071       6120    1.16  6.8e-16  bluestein
      1012  2^2.11.23        chain3   raced           1717       3300    1.71  5.1e-16  
      1013  1013             prime    raced           5232       6064    1.15  6.8e-16  bluestein
      1014  2.3.13^2         chain3   raced           1392       1621    1.14  6.3e-16  
      1015  5.7.29           flat     raced           1854       4379    1.98  5.8e-16  
      1016  2^3.127          prime    raced           5038      18404    3.59  5.0e-16  bluestein
      1017  3^2.113          prime    raced           5132       6026    1.15  6.7e-16  bluestein
      1018  2.509            prime    raced           5136       6044    1.17  5.0e-16  bluestein
      1019  1019             prime    raced           5164       5999    1.14  7.2e-16  bluestein
      1020  2^2.3.5.17       chain3   raced           1309       2687    2.03  5.4e-16  
      1021  1021             prime    raced           5191       6062    1.14  6.6e-16  bluestein
      1022  2.7.73           prime    raced           5119      10507    2.05  6.6e-16  bluestein
      1023  3.11.31          flat     raced           2125       4630    1.76  6.9e-16  
      1024  2^10             ztt      replayed         719        797    1.08  4.0e-16  
      1025  5^2.41           2p       raced              -          -       -        -  not benched
      1026  2.3^3.19         chain3   raced              -          -       -        -  not benched
      1027  13.79            prime    raced              -          -       -        -  not benched
      1028  2^2.257          prime    raced              -          -       -        -  not benched
      1029  3.7^3            flat     raced              -          -       -        -  not benched
      1030  2.5.103          prime    raced              -          -       -        -  not benched
      1031  1031             prime    raced              -          -       -        -  not benched
      1032  2^3.3.43         chain3   raced              -          -       -        -  not benched
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime      490      1     42   1.06   1.27   2.62    1.49
 2p         243      0      5   1.27   2.12   3.21    1.99
 chain3     179      0      3   1.18   1.44   2.20    1.52
 flat        91      1      6   1.06   1.82   2.42    1.72
 mono        19      1      2   0.94   1.59   2.40    1.53
 ztt          1      0      0   1.08   1.08   1.08    1.08
 ALL       1023      3     58   1.12   1.48   2.76    1.63
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   2.06      0      0
 8..31                 24   1.31      6      2
 32..127               96   1.59      3      0
 128..511             384   1.61      6      0
 512..1024            513   1.37     43      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 composite with a prime >= 53 (prime cell)      330   1.27     33   1.46
 2p                                             237   2.14      2   2.03
 chain3                                         179   1.44      3   1.52
 flat                                            91   1.82      6   1.72
 prime N, rader                                  80   2.06      0   2.09
 prime N, bluestein                              80   1.17      9   1.19
 mono                                            16   1.58      2   1.44
 pow2                                            10   1.06      3   1.28
```


flip agreement: our two readings more than 25% apart at 26 of 1023 cells.

worst 10: 14 (flat 0.66), 549 (prime 0.73), 15 (mono 0.76), 530 (prime 0.82), 634 (prime 0.83), 168 (chain3 0.83), 975 (flat 0.85), 591 (prime 0.90), 636 (prime 0.90), 536 (prime 0.92)
best 5: 89 (prime 6.99), 79 (prime 5.28), 73 (prime 5.22), 71 (prime 4.88), 254 (prime 4.46)
