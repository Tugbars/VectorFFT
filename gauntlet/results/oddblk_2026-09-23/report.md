# gauntlet report

run: `oddblk_2026-09-23`  contract file suffix: `(oop, T=1)`  cells: 2050 listed, 2050 benched, comparator: MKL

control cell: 46 readings, 1.064..1.091 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         7  7                mono     replayed           7         12    1.57  2.6e-16  
         9  3^2              mono     replayed           9         13    1.39  4.2e-16  
        11  11               mono     replayed          12         13    1.16  1.2e-16  
        13  13               mono     replayed          15         15    0.97  5.0e-16  
        14  2.7              mono     replayed          13         14    0.97  2.0e-16  
        15  3.5              2p       replayed          16         14    0.52  3.3e-16  flips differ 1.69x
        17  17               mono     replayed          23         41    1.68  2.0e-16  
        18  2.3^2            2p       replayed          15         23    1.54  3.8e-16  
        19  19               mono     replayed          27         45    1.64  4.6e-16  
        21  3.7              2p       replayed          18         25    1.40  3.3e-16  
        22  2.11             mono     replayed          24         31    1.27  1.7e-16  
        23  23               mono     replayed          40         69    1.72  2.5e-16  
        25  5^2              2p       replayed          20         26    1.31  2.7e-16  
        26  2.13             mono     replayed          31         36    1.19  3.7e-16  
        27  3^3              flat     replayed          29         36    1.08  3.8e-16  
        28  2^2.7            2p       replayed          19         30    1.57  1.5e-16  
        29  29               mono     replayed          67        107    1.60  2.8e-16  
        30  2.3.5            2p       replayed          20         31    1.53  3.0e-16  
        31  31               mono     replayed          72        121    1.69  3.3e-16  
        33  3.11             2p       replayed          27         36    1.32  3.3e-16  
        34  2.17             flat     replayed          49         75    1.50  6.3e-16  
        35  5.7              2p       replayed          25         36    1.36  2.6e-16  
        36  2^2.3^2          2p       replayed          22         42    1.88  2.4e-16  
        37  37               mono     replayed         106        185    1.70  4.8e-16  
        38  2.19             flat     replayed          57         88    1.46  2.6e-16  
        39  3.13             2p       replayed          33         45    1.34  6.4e-16  
        41  41               mono     replayed         133        242    1.82  3.7e-16  
        42  2.3.7            2p       replayed          28         41    1.46  3.4e-16  
        43  43               prime    replayed         108        265    2.45  6.2e-16  rader
        44  2^2.11           2p       replayed          28         44    1.53  4.8e-16  
        45  3^2.5            2p       replayed          33         49    1.46  5.2e-16  
        46  2.23             flat     replayed          79        121    1.54  5.9e-16  
        47  47               mono     replayed         187        327    1.75  4.0e-16  
        49  7^2              2p       replayed          36         47    1.29  4.3e-16  
        50  2.5^2            2p       replayed          33         53    1.60  3.6e-16  
        51  3.17             2p       replayed          54        114    2.02  4.1e-16  
        52  2^2.13           2p       replayed          38         50    1.30  5.3e-16  
        53  53               prime    replayed         128        438    3.42  7.2e-16  rader
        54  2.3^3            2p       replayed          36         58    1.63  4.8e-16  
        55  5.11             2p       replayed          43         54    1.25  3.5e-16  
        56  2^3.7            2p       replayed          34         56    1.64  1.8e-16  
        57  3.19             2p       replayed          64        136    2.05  3.4e-16  
        58  2.29             flat     replayed         108        185    1.40  4.3e-16  
        59  59               prime    replayed         201        552    2.74  5.1e-16  bluestein
        60  2^2.3.5          2p       replayed          36         61    1.67  5.5e-16  
        61  61               prime    replayed         143        602    4.17  6.1e-16  rader
        62  2.31             flat     replayed         115        206    1.45  5.9e-16  
        63  3^2.7            2p       replayed          47         67    1.42  4.5e-16  
        65  5.13             2p       replayed          54         67    1.22  6.3e-16  
        66  2.3.11           2p       replayed          47         67    1.40  3.8e-16  
        67  67               prime    replayed         171        739    4.32  6.4e-16  rader
        68  2^2.17           2p       replayed          60        138    2.29  3.6e-16  
        69  3.23             2p       replayed          94        198    2.04  4.1e-16  
        70  2.5.7            2p       replayed          46         78    1.54  3.5e-16  
        71  71               prime    replayed         173        841    4.85  7.6e-16  rader
        72  2^3.3^2          2p       replayed          45         76    1.67  3.0e-16  
        73  73               prime    replayed         173        904    5.21  7.9e-16  rader
        74  2.37             flat     replayed         177        296    1.67  4.8e-16  
        75  3.5^2            2p       replayed          58         78    1.34  4.7e-16  
        76  2^2.19           2p       replayed          64        165    2.56  3.3e-16  
        77  7.11             2p       replayed          61         75    1.22  3.0e-16  
        78  2.3.13           2p       replayed          59         81    1.37  5.4e-16  
        79  79               prime    replayed         207       1085    5.22  1.2e-15  rader
        81  3^4              2p       replayed          63         87    1.37  4.3e-16  
        82  2.41             prime    replayed         439        371    0.83  7.4e-16  bluestein
        83  83               prime    replayed         458       1240    2.70  7.1e-16  bluestein
        84  2^2.3.7          2p       replayed          54         88    1.59  3.3e-16  
        85  5.17             2p       replayed          83        190    2.19  5.7e-16  
        86  2.43             prime    replayed         442        408    0.92  6.9e-16  bluestein
        87  3.29             flat     replayed         184        332    1.79  6.6e-16  
        88  2^3.11           2p       replayed          61         87    1.44  2.8e-16  
        89  89               prime    replayed         215       1514    7.04  5.7e-16  rader
        90  2.3^2.5          2p       replayed          63         96    1.53  4.3e-16  
        91  7.13             2p       replayed          76         91    1.19  6.0e-16  
        92  2^2.23           2p       replayed          94        234    1.95  4.7e-16  flips differ 1.28x
        93  3.31             flat     replayed         209        376    1.74  6.6e-16  
        94  2.47             prime    replayed         442        509    1.14  8.6e-16  bluestein
        95  5.19             2p       replayed          99        227    2.19  4.9e-16  
        97  97               prime    replayed         219        648    2.95  7.1e-16  rader
        98  2.7^2            flat     replayed         117        124    1.07  4.7e-16  
        99  3^2.11           2p       replayed          82        112    1.36  4.4e-16  
       100  2^2.5^2          2p       replayed          64        104    1.61  3.9e-16  
       101  101              prime    replayed         236        561    2.38  7.6e-16  rader
       102  2.3.17           2p       replayed          85        204    2.36  4.6e-16  
       103  103              prime    replayed         290        562    1.94  1.1e-15  rader
       104  2^3.13           2p       replayed          75        102    1.26  4.8e-16  
       105  3.5.7            2p       replayed          86        113    1.31  5.0e-16  
       106  2.53             prime    replayed         448        666    1.46  6.7e-16  bluestein
       107  107              prime    replayed         448        564    1.24  1.0e-15  bluestein
       108  2^2.3^3          2p       replayed          78        120    1.54  5.2e-16  
       109  109              prime    replayed         273        568    2.08  8.5e-16  rader
       110  2.5.11           2p       replayed          81        130    1.60  3.7e-16  
       111  3.37             prime    replayed         449        567    1.26  6.4e-16  bluestein
       112  2^4.7            2p       replayed          72         96    1.33  2.9e-16  
       113  113              prime    replayed         275        569    2.04  8.4e-16  rader
       114  2.3.19           2p       replayed         100        245    2.44  4.8e-16  
       115  5.23             2p       replayed         147        338    2.29  4.6e-16  
       116  2^2.29           flat     replayed         155        364    1.88  4.4e-16  flips differ 1.25x
       117  3^2.13           2p       replayed         102        138    1.35  5.2e-16  
       118  2.59             prime    replayed         463        837    1.15  5.6e-16  flips differ 1.58x bluestein
       119  7.17             2p       replayed         114        277    2.41  4.5e-16  
       120  2^3.3.5          2p       replayed          78        125    1.61  5.6e-16  
       121  11^2             2p       replayed         106        122    1.15  2.8e-16  
       122  2.61             prime    replayed         456        912    2.00  7.6e-16  bluestein
       123  3.41             prime    replayed         464        729    1.56  6.4e-16  bluestein
       124  2^2.31           flat     replayed         176        409    1.84  5.3e-16  flips differ 1.26x
       125  5^3              2p       replayed         104        133    1.28  5.5e-16  
       126  2.3^2.7          2p       replayed          91        136    1.48  3.1e-16  
       127  127              prime    replayed         385        587    1.45  6.2e-16  rader
       129  3.43             flat     replayed         348        804    2.29  6.6e-16  
       130  2.5.13           2p       replayed         102        155    1.52  5.1e-16  
       131  131              prime    replayed         354        950    2.68  6.4e-16  rader
       132  2^2.3.11         2p       replayed         102        145    1.42  5.5e-16  
       133  7.19             2p       replayed         140        326    2.32  5.0e-16  
       134  2.67             prime    replayed         935       1129    1.19  7.7e-16  bluestein
       135  3^3.5            2p       replayed         114        156    1.37  3.6e-16  
       136  2^3.17           2p       replayed         111        276    2.45  7.7e-16  
       137  137              prime    replayed         388        957    2.42  7.4e-16  rader
       138  2.3.23           2p       replayed         144        352    2.17  4.8e-16  
       139  139              prime    replayed         468        959    1.92  7.4e-16  rader
       140  2^2.5.7          chain3   replayed         112        144    1.27  3.2e-16  
       141  3.47             flat     replayed         531        985    1.83  7.4e-16  
       142  2.71             prime    replayed         985       1264    1.25  8.5e-16  bluestein
       143  11.13            2p       replayed         130        150    1.15  6.2e-16  
       144  2^4.3^2          chain3   replayed         107        143    1.33  5.4e-16  
       145  5.29             2p       replayed         223        555    2.48  4.7e-16  
       146  2.73             prime    replayed        1021       1356    1.31  9.5e-16  bluestein
       147  3.7^2            chain3   replayed         135        158    1.16  4.9e-16  
       148  2^2.37           2p       replayed         194        588    2.81  3.2e-16  
       149  149              prime    replayed         764        970    1.24  6.3e-16  rader
       150  2.3.5^2          2p       replayed         111        159    1.43  3.5e-16  
       151  151              prime    replayed         405        970    2.39  7.3e-16  rader
       152  2^3.19           2p       replayed         132        332    2.52  4.1e-16  
       153  3^2.17           2p       replayed         146        371    2.52  5.6e-16  
       154  2.7.11           flat     replayed         193        208    1.05  7.0e-16  
       155  5.31             2p       replayed         258        631    2.45  5.4e-16  
       156  2^2.3.13         2p       replayed         126        173    1.37  6.2e-16  
       157  157              prime    replayed         434        978    2.25  1.2e-15  rader
       158  2.79             prime    replayed         964       1635    1.51  5.7e-16  bluestein
       159  3.53             prime    replayed         963       1328    1.38  4.6e-16  bluestein
       161  7.23             2p       replayed         211        477    2.16  4.5e-16  
       162  2.3^4            2p       replayed         123        189    1.54  4.2e-16  
       163  163              prime    replayed         475       1248    2.62  9.0e-16  rader
       164  2^2.41           2p       replayed         298        734    1.55  3.4e-16  flips differ 1.59x
       165  3.5.11           2p       replayed         145        188    1.29  4.6e-16  
       166  2.83             prime    replayed         950       1782    1.86  9.7e-16  bluestein
       167  167              prime    replayed         982       1253    1.26  8.7e-16  bluestein
       168  2^3.3.7          chain3   replayed         127        106    0.84  2.8e-16  
       169  13^2             2p       replayed         163        189    1.16  6.8e-16  
       170  2.5.17           2p       replayed         145        371    2.55  8.7e-16  
       171  3^2.19           2p       replayed         190        462    2.34  5.5e-16  
       172  2^2.43           2p       replayed         253        809    2.98  5.2e-16  
       173  173              prime    replayed         970       1260    1.30  7.0e-16  bluestein
       174  2.3.29           2p       replayed         217        557    1.63  4.2e-16  flips differ 1.60x
       175  5^2.7            chain3   replayed         153        197    1.21  4.8e-16  
       176  2^4.11           2p       replayed         124        167    1.35  2.4e-16  
       177  3.59             prime    replayed        1011       1677    1.64  1.2e-15  bluestein
       178  2.89             prime    replayed         972       2157    2.15  8.4e-16  bluestein
       179  179              prime    replayed         985       1263    1.27  7.9e-16  bluestein
       180  2^2.3^2.5        2p       replayed         141        206    1.47  5.1e-16  
       181  181              prime    replayed         525       1268    2.37  8.8e-16  rader
       182  2.7.13           flat     replayed         243        250    1.03  4.4e-16  
       183  3.61             prime    replayed         972       1836    1.89  1.1e-15  bluestein
       184  2^3.23           2p       replayed         212        478    1.99  4.6e-16  
       185  5.37             2p       replayed         492        956    1.42  3.5e-16  flips differ 1.37x
       186  2.3.31           2p       replayed         242        617    2.16  4.3e-16  
       187  11.17            flat     replayed         245        450    1.82  6.7e-16  
       188  2^2.47           2p       replayed         417       1013    2.41  3.7e-16  
       189  3^3.7            chain3   replayed         169        229    1.36  4.4e-16  
       190  2.5.19           2p       replayed         170        444    2.61  4.6e-16  
       191  191              prime    replayed         588       1274    2.15  9.2e-16  rader
       193  193              prime    replayed         476       1290    2.41  7.4e-16  rader
       194  2.97             prime    replayed         983       2578    2.61  7.3e-16  bluestein
       195  3.5.13           chain3   replayed         178        236    1.33  8.5e-16  
       196  2^2.7^2          chain3   replayed         168        238    1.30  3.5e-16  
       197  197              prime    replayed         578       1319    2.24  7.2e-16  rader
       198  2.3^2.11         chain3   replayed         173        234    1.35  6.6e-16  
       199  199              prime    replayed         613       1295    2.11  8.1e-16  rader
       200  2^3.5^2          chain3   replayed         140        218    1.55  4.7e-16  
       201  3.67             prime    replayed        1031       2249    2.13  8.0e-16  bluestein
       202  2.101            prime    replayed         979       2840    2.90  5.4e-16  bluestein
       203  7.29             2p       replayed         303        783    2.58  3.9e-16  
       204  2^2.3.17         chain3   replayed         208        440    2.10  6.3e-16  
       205  5.41             2p       replayed         407       1232    2.88  5.7e-16  
       206  2.103            prime    replayed        1032       2900    2.75  1.1e-15  bluestein
       207  3^2.23           2p       replayed         280        634    2.08  4.4e-16  
       208  2^4.13           2p       replayed         155        199    1.28  6.7e-16  
       209  11.19            2p       replayed         219        525    2.39  5.1e-16  
       210  2.3.5.7          chain3   replayed         167        226    1.35  4.1e-16  
       211  211              prime    replayed         701       1230    1.75  7.9e-16  rader
       212  2^2.53           prime    replayed        1023       1327    1.27  1.1e-15  bluestein
       213  3.71             prime    replayed        1030       2559    2.48  7.2e-16  bluestein
       214  2.107            prime    replayed        1045       3149    2.99  7.8e-16  bluestein
       215  5.43             flat     replayed         485       1357    2.79  5.2e-16  
       216  2^3.3^3          2p       replayed         162        266    1.52  3.6e-16  
       217  7.31             2p       replayed         362        892    1.88  6.8e-16  flips differ 1.32x
       218  2.109            prime    replayed         998       3346    3.35  7.2e-16  bluestein
       219  3.73             prime    replayed        1038       2753    2.56  8.9e-16  bluestein
       220  2^2.5.11         chain3   replayed         190        251    1.32  3.9e-16  
       221  13.17            2p       replayed         227        545    2.37  6.1e-16  
       222  2.3.37           2p       replayed         300        888    2.82  4.5e-16  
       223  223              prime    replayed        1012       1238    1.22  1.2e-15  bluestein
       224  2^5.7            2p       replayed         149        225    1.51  3.9e-16  
       225  3^2.5^2          2p       replayed         208        271    1.28  6.0e-16  
       226  2.113            prime    replayed         998       3613    3.61  8.2e-16  bluestein
       227  227              prime    replayed         985       1242    1.26  1.1e-15  bluestein
       228  2^2.3.19         flat     replayed         331        526    1.56  1.2e-15  
       229  229              prime    replayed         740       1256    1.68  8.8e-16  rader
       230  2.5.23           2p       replayed         240        636    2.20  5.5e-16  
       231  3.7.11           chain3   replayed         215        275    1.25  4.7e-16  
       232  2^3.29           flat     replayed         326        740    1.91  5.1e-16  
       233  233              prime    replayed         985       1250    1.26  8.2e-16  bluestein
       234  2.3^2.13         chain3   replayed         228        282    1.24  6.4e-16  
       235  5.47             2p       replayed         882       1665    1.46  4.5e-16  flips differ 1.29x
       236  2^2.59           prime    replayed        1045       1671    1.60  8.3e-16  bluestein
       237  3.79             prime    replayed        1074       3274    3.04  7.9e-16  bluestein
       238  2.7.17           flat     replayed         359        571    1.59  1.2e-15  
       239  239              prime    replayed        1032       1255    1.16  9.6e-16  bluestein
       240  2^4.3.5          2p       replayed         170        251    1.47  7.4e-16  
       241  241              prime    replayed         655       1260    1.92  5.5e-16  rader
       242  2.11^2           flat     replayed         341        374    1.08  7.6e-16  
       243  3^5              2p       replayed         225        295    1.31  6.5e-16  
       244  2^2.61           prime    replayed        1045       1820    1.62  6.0e-16  bluestein
       245  5.7^2            chain3   replayed         229        282    1.22  3.5e-16  
       246  2.3.41           2p       replayed         359       1109    3.08  3.9e-16  
       247  13.19            2p       replayed         267        647    1.90  4.9e-16  flips differ 1.27x
       248  2^3.31           flat     replayed         422        832    1.67  7.1e-16  
       249  3.83             prime    replayed        1034       3729    3.52  5.5e-16  bluestein
       250  2.5^3            chain3   replayed         191        308    1.61  4.1e-16  
       251  251              prime    replayed         700       1268    1.81  9.6e-16  rader
       252  2^2.3^2.7        chain3   replayed         211        286    1.34  8.7e-16  
       253  11.23            2p       replayed         311        767    2.16  7.7e-16  
       254  2.127            prime    replayed        1017       4533    4.45  7.0e-16  bluestein
       255  3.5.17           2p       replayed         259        641    2.47  7.2e-16  
       257  257              prime    replayed         657       2080    3.14  7.0e-16  rader
       258  2.3.43           2p       replayed         519       1223    1.27  5.1e-16  flips differ 1.85x
       259  7.37             2p       replayed         439       1351    3.07  4.1e-16  
       260  2^2.5.13         chain3   replayed         229        312    1.31  4.4e-16  
       261  3^2.29           2p       replayed         378       1065    2.81  5.3e-16  
       262  2.131            prime    replayed        2063       4848    2.35  8.1e-16  bluestein
       263  263              prime    replayed        2049       2083    1.00  6.1e-16  bluestein
       264  2^3.3.11         chain3   replayed         227        322    1.31  6.6e-16  
       265  5.53             prime    replayed        2039       2242    1.08  5.8e-16  bluestein
       266  2.7.19           flat     replayed         401        686    1.68  4.4e-16  
       267  3.89             prime    replayed        2055       4536    2.19  8.7e-16  bluestein
       268  2^2.67           prime    replayed        2233       2223    0.99  9.7e-16  bluestein
       269  269              prime    replayed        2101       2091    0.99  9.4e-16  bluestein
       270  2.3^3.5          chain3   replayed         231        329    1.42  6.2e-16  
       271  271              prime    replayed         795       2092    2.11  7.7e-16  rader
       272  2^4.17           2p       replayed         221        544    2.46  5.5e-16  
       273  3.7.13           2p       replayed         264        335    1.27  6.8e-16  
       274  2.137            prime    replayed        2192       5412    2.27  8.5e-16  bluestein
       275  5^2.11           chain3   replayed         252        332    1.27  4.5e-16  
       276  2^2.3.23         2p       replayed         301        759    2.30  5.6e-16  
       277  277              prime    replayed         991       2119    2.04  8.8e-16  rader
       278  2.139            prime    replayed        2262       5478    2.40  8.1e-16  bluestein
       279  3^2.31           2p       replayed         475       1193    2.48  4.9e-16  
       280  2^3.5.7          chain3   replayed         212        310    1.46  5.0e-16  
       281  281              prime    replayed         820       2106    2.56  6.3e-16  rader
       282  2.3.47           2p       replayed        1103       1531    1.30  4.9e-16  
       283  283              prime    replayed        2130       2105    0.98  5.7e-16  bluestein
       284  2^2.71           prime    replayed        2119       2531    1.18  9.1e-16  bluestein
       285  3.5.19           2p       replayed         328        758    2.21  4.6e-16  
       286  2.11.13          flat     replayed         420        443    1.05  5.6e-16  
       287  7.41             2p       replayed         534       1729    3.06  3.7e-16  
       288  2^5.3^2          2p       replayed         207        338    1.63  5.0e-16  
       289  17^2             2p       replayed         341       1102    3.03  8.4e-16  
       290  2.5.29           2p       replayed         438        983    2.21  5.5e-16  
       291  3.97             prime    replayed        2125       2392    1.11  7.9e-16  bluestein
       292  2^2.73           prime    replayed        2073       2718    1.31  7.2e-16  bluestein
       293  293              prime    replayed        2112       2399    1.12  7.5e-16  bluestein
       294  2.3.7^2          chain3   replayed         255        342    1.34  4.3e-16  
       295  5.59             prime    replayed        2053       2813    1.36  8.5e-16  bluestein
       296  2^3.37           2p       replayed         393       1194    2.95  4.0e-16  
       297  3^3.11           chain3   replayed         298        391    1.31  5.7e-16  
       298  2.149            prime    replayed        2140       6506    2.98  8.6e-16  bluestein
       299  13.23            2p       replayed         374        936    2.18  5.3e-16  
       300  2^2.3.5^2        chain3   replayed         234        360    1.53  5.4e-16  
       301  7.43             flat     replayed         681       1909    2.69  4.9e-16  
       302  2.151            prime    replayed        2099       2418    1.14  9.0e-16  bluestein
       303  3.101            prime    replayed        2064       2406    1.15  7.7e-16  bluestein
       304  2^4.19           2p       replayed         306        658    2.14  2.9e-16  
       305  5.61             prime    replayed        2052       3088    1.50  7.2e-16  bluestein
       306  2.3^2.17         chain3   replayed         315        685    2.04  5.5e-16  
       307  307              prime    replayed        1211       2408    1.99  1.2e-15  rader
       308  2^2.7.11         chain3   replayed         275        387    1.41  5.7e-16  
       309  3.103            prime    replayed        2125       2411    1.12  5.9e-16  bluestein
       310  2.5.31           2p       replayed         489       1103    1.58  5.1e-16  flips differ 1.43x
       311  311              prime    replayed        1432       2419    1.42  1.0e-15  rader
       312  2^3.3.13         chain3   replayed         279        366    1.30  5.2e-16  
       313  313              prime    replayed        1054       2417    2.25  9.8e-16  rader
       314  2.157            prime    replayed        2097       2422    1.12  7.6e-16  bluestein
       315  3^2.5.7          chain3   replayed         292        396    1.35  6.9e-16  
       316  2^2.79           prime    replayed        2177       3209    1.46  6.6e-16  bluestein
       317  317              prime    replayed        2191       2423    1.09  7.9e-16  bluestein
       318  2.3.53           prime    replayed        2086       2002    0.96  7.5e-16  bluestein
       319  11.29            2p       replayed         548       1255    2.29  5.6e-16  
       321  3.107            prime    replayed        2063       2653    1.27  1.1e-15  bluestein
       322  2.7.23           flat     replayed         545        963    1.67  4.9e-16  
       323  17.19            2p       replayed         396       1283    3.07  6.3e-16  
       324  2^2.3^4          chain3   replayed         273        411    1.47  6.8e-16  
       325  5^2.13           2p       replayed         330        410    1.24  8.0e-16  
       326  2.163            prime    replayed        2228       2657    1.18  6.3e-16  bluestein
       327  3.109            prime    replayed        2088       2661    1.21  1.3e-15  bluestein
       328  2^3.41           2p       replayed         611       1484    1.56  4.9e-16  flips differ 1.56x
       329  7.47             flat     replayed         897       2338    2.52  5.1e-16  
       330  2.3.5.11         chain3   replayed         303        400    1.31  5.3e-16  
       331  331              prime    replayed        1119       2658    2.36  6.4e-16  rader
       332  2^2.83           prime    replayed        2100       3574    1.69  9.8e-16  bluestein
       333  3^2.37           2p       replayed         783       1776    1.62  6.4e-16  flips differ 1.40x
       334  2.167            prime    replayed        2081       2663    1.27  9.5e-16  bluestein
       335  5.67             prime    replayed        2229       3771    1.69  7.9e-16  bluestein
       336  2^4.3.7          2p       replayed         246        348    1.41  4.0e-16  
       337  337              prime    replayed        1041       2670    2.52  1.0e-15  rader
       338  2.13^2           flat     replayed         517        551    1.07  1.1e-15  
       339  3.113            prime    replayed        2239       2688    1.15  7.5e-16  bluestein
       340  2^2.5.17         chain3   replayed         338        752    2.10  6.3e-16  
       341  11.31            flat     replayed         645       1422    1.85  6.8e-16  
       342  2.3^2.19         chain3   replayed         381        819    2.06  5.1e-16  
       343  7^3              chain3   replayed         350        400    1.14  3.7e-16  
       344  2^3.43           2p       replayed         516       1638    3.05  5.5e-16  
       345  3.5.23           2p       replayed         476       1114    2.17  6.4e-16  
       346  2.173            prime    replayed        2173       2674    1.21  8.1e-16  bluestein
       347  347              prime    replayed        2122       2676    1.26  9.2e-16  bluestein
       348  2^2.3.29         2p       replayed         450       1167    2.12  4.5e-16  
       349  349              prime    replayed        1534       2683    1.74  1.1e-15  rader
       350  2.5^2.7          chain3   replayed         286        433    1.51  3.5e-16  
       351  3^3.13           chain3   replayed         363        483    1.32  6.1e-16  
       352  2^5.11           2p       replayed         272        404    1.48  3.4e-16  
       353  353              prime    replayed        1026       2693    2.60  6.7e-16  rader
       354  2.3.59           prime    replayed        2289       2520    1.09  8.6e-16  bluestein
       355  5.71             prime    replayed        2100       4294    2.03  7.1e-16  bluestein
       356  2^2.89           prime    replayed        2209       4309    1.85  6.0e-16  bluestein
       357  3.7.17           chain3   replayed         414        915    2.21  5.3e-16  
       358  2.179            prime    replayed        2095       2692    1.22  9.6e-16  bluestein
       359  359              prime    replayed        2087       2689    1.28  9.6e-16  bluestein
       360  2^3.3^2.5        chain3   replayed         308        435    1.41  4.6e-16  
       361  19^2             2p       replayed         470       1538    3.12  5.2e-16  
       362  2.181            prime    replayed        2107       2696    1.27  1.0e-15  bluestein
       363  3.11^2           chain3   replayed         383        458    1.20  5.7e-16  
       364  2^2.7.13         chain3   replayed         341        461    1.35  5.6e-16  
       365  5.73             prime    replayed        2097       4619    2.19  9.0e-16  bluestein
       366  2.3.61           prime    replayed        2239       2746    1.22  6.4e-16  bluestein
       367  367              prime    replayed        2200       2696    1.22  6.5e-16  bluestein
       368  2^4.23           2p       replayed         475        961    1.98  3.1e-16  
       369  3^2.41           2p       replayed        1014       2278    1.80  6.3e-16  
       370  2.5.37           2p       replayed         510       1567    2.87  4.9e-16  
       371  7.53             prime    replayed        2184       3149    1.42  9.8e-16  bluestein
       372  2^2.3.31         flat     replayed         607       1317    2.16  5.2e-16  
       373  373              prime    replayed        2095       2705    1.29  8.5e-16  bluestein
       374  2.11.17          flat     replayed         585        963    1.56  1.3e-15  
       375  3.5^3            chain3   replayed         349        476    1.36  5.7e-16  
       376  2^3.47           2p       replayed         831       2045    1.60  4.6e-16  flips differ 1.53x
       377  13.29            2p       replayed         632       1519    1.94  5.3e-16  
       378  2.3^3.7          chain3   replayed         349        458    1.31  6.5e-16  
       379  379              prime    replayed        1728       2730    1.57  1.1e-15  rader
       380  2^2.5.19         chain3   replayed         405        901    2.18  4.7e-16  
       381  3.127            prime    replayed        2095       2714    1.29  9.4e-16  bluestein
       382  2.191            prime    replayed        2103       2716    1.28  9.6e-16  bluestein
       383  383              prime    replayed        2298       2710    1.18  9.7e-16  bluestein
       385  5.7.11           chain3   replayed         373        490    1.29  5.9e-16  
       386  2.193            prime    replayed        2242       3152    1.40  1.3e-15  bluestein
       387  3^2.43           2p       replayed         769       2510    3.23  5.9e-16  
       388  2^2.97           prime    replayed        2169       5227    2.41  8.0e-16  bluestein
       389  389              prime    replayed        2126       3151    1.48  1.3e-15  bluestein
       390  2.3.5.13         chain3   replayed         369        480    1.28  7.7e-16  
       391  17.23            2p       replayed         543       1760    2.71  6.7e-16  
       392  2^3.7^2          chain3   replayed         332        430    1.29  4.5e-16  
       393  3.131            prime    replayed        2237       3159    1.40  8.8e-16  bluestein
       394  2.197            prime    replayed        2119       3154    1.47  1.1e-15  bluestein
       395  5.79             prime    replayed        2122       5495    2.58  1.4e-15  bluestein
       396  2^2.3^2.11       chain3   replayed         373        509    1.36  5.6e-16  
       397  397              prime    replayed        1503       3162    2.10  8.6e-16  rader
       398  2.199            prime    replayed        2123       3162    1.49  9.7e-16  bluestein
       399  3.7.19           chain3   replayed         489       1085    2.19  6.2e-16  
       400  2^4.5^2          2p       replayed         310        436    1.40  4.5e-16  
       401  401              prime    replayed        1306       2606    1.96  8.0e-16  rader
       402  2.3.67           prime    replayed        2263       3352    1.47  7.6e-16  bluestein
       403  13.31            2p       replayed         715       1718    2.36  7.0e-16  
       404  2^2.101          prime    replayed        2314       5778    2.47  7.9e-16  bluestein
       405  3^4.5            chain3   replayed         392        548    1.30  6.6e-16  
       406  2.7.29           flat     replayed         840       1455    1.50  6.3e-16  
       407  11.37            2p       replayed         683       2138    2.99  4.6e-16  
       408  2^3.3.17         chain3   replayed         437        907    2.02  5.3e-16  
       409  409              prime    replayed        1644       2624    1.56  9.8e-16  rader
       410  2.5.41           2p       replayed         795       1935    1.56  4.1e-16  flips differ 1.56x
       411  3.137            prime    replayed        2245       2613    1.15  1.5e-15  bluestein
       412  2^2.103          prime    replayed        2212       5871    2.57  8.0e-16  bluestein
       413  7.59             prime    replayed        2208       3951    1.79  7.8e-16  bluestein
       414  2.3^2.23         chain3   replayed         536       1163    2.17  6.1e-16  
       415  5.83             prime    replayed        2245       6248    2.67  9.4e-16  bluestein
       416  2^5.13           2p       replayed         337        485    1.44  6.0e-16  
       417  3.139            prime    replayed        2279       2624    1.11  7.2e-16  bluestein
       418  2.11.19          flat     replayed         661       1150    1.68  1.0e-15  
       419  419              prime    replayed        2142       2624    1.22  8.9e-16  bluestein
       420  2^2.3.5.7        chain3   replayed         361        498    1.36  6.5e-16  
       421  421              prime    replayed        1985       2626    1.30  9.7e-16  rader
       422  2.211            prime    replayed        2237       2636    1.16  8.9e-16  bluestein
       423  3^2.47           flat     replayed        1132       3062    2.55  7.0e-16  
       424  2^3.53           prime    replayed        2144       2681    1.25  1.1e-15  bluestein
       425  5^2.17           2p       replayed         462       1114    2.34  6.4e-16  
       426  2.3.71           prime    replayed        2148       3817    1.73  6.0e-16  bluestein
       427  7.61             prime    replayed        2117       4338    1.94  8.8e-16  bluestein
       428  2^2.107          prime    replayed        2333       6363    2.72  7.6e-16  bluestein
       429  3.11.13          chain3   replayed         477        555    1.16  5.6e-16  
       430  2.5.43           2p       replayed         670       2136    3.14  4.1e-16  
       431  431              prime    replayed        2124       2628    1.23  9.5e-16  bluestein
       432  2^4.3^3          2p       replayed         335        510    1.52  6.3e-16  
       433  433              prime    replayed        1554       2646    1.55  9.1e-16  rader
       434  2.7.31           flat     replayed        1043       1629    1.56  5.4e-16  
       435  3.5.29           2p       replayed         642       1781    2.39  6.9e-16  
       436  2^2.109          prime    replayed        2295       6769    2.83  1.1e-15  bluestein
       437  19.23            2p       replayed         645       2103    2.79  6.3e-16  
       438  2.3.73           prime    replayed        2133       4093    1.80  8.0e-16  bluestein
       439  439              prime    replayed        2154       2639    1.15  9.6e-16  bluestein
       440  2^3.5.11         chain3   replayed         383        541    1.41  4.9e-16  
       441  3^2.7^2          chain3   replayed         446        566    1.26  4.9e-16  
       442  2.13.17          flat     replayed         709       1176    1.64  9.4e-16  
       443  443              prime    replayed        2158       2650    1.23  9.6e-16  bluestein
       444  2^2.3.37         2p       replayed         639       1863    2.75  8.3e-16  
       445  5.89             prime    replayed        2259       7593    3.18  7.4e-16  bluestein
       446  2.223            prime    replayed        2127       2668    1.22  8.6e-16  bluestein
       447  3.149            prime    replayed        2135       2660    1.23  1.0e-15  bluestein
       448  2^6.7            chain3   replayed         351        444    1.26  3.0e-16  
       449  449              prime    replayed        1481       2664    1.74  8.8e-16  rader
       450  2.3^2.5^2        chain3   replayed         401        577    1.40  5.1e-16  
       451  11.41            2p       replayed        1193       2746    1.65  6.7e-16  flips differ 1.40x
       452  2^2.113          prime    replayed        2142       7301    3.39  1.3e-15  bluestein
       453  3.151            prime    replayed        2145       2671    1.24  1.1e-15  bluestein
       454  2.227            prime    replayed        2158       2662    1.23  1.2e-15  bluestein
       455  5.7.13           chain3   replayed         459        584    1.26  5.6e-16  
       456  2^3.3.19         chain3   replayed         490       1085    2.21  6.3e-16  
       457  457              prime    replayed        1921       2662    1.35  9.1e-16  rader
       458  2.229            prime    replayed        2163       2666    1.21  9.2e-16  bluestein
       459  3^3.17           chain3   replayed         542       1206    2.12  5.8e-16  
       460  2^2.5.23         flat     replayed         591       1285    2.17  6.9e-16  
       461  461              prime    replayed        2179       2679    1.21  1.1e-15  bluestein
       462  2.3.7.11         chain3   replayed         456        606    1.30  6.4e-16  
       463  463              prime    replayed        1804       2674    1.47  7.8e-16  rader
       464  2^4.29           2p       replayed         655       1480    2.26  3.9e-16  
       465  3.5.31           2p       replayed         806       2012    1.79  5.6e-16  flips differ 1.39x
       466  2.233            prime    replayed        2162       2676    1.20  9.1e-16  bluestein
       467  467              prime    replayed        2216       2664    1.18  7.8e-16  bluestein
       468  2^2.3^2.13       chain3   replayed         482        610    1.20  7.5e-16  
       469  7.67             prime    replayed        2237       5343    2.37  7.6e-16  bluestein
       470  2.5.47           2p       replayed        1074       2650    1.61  5.3e-16  flips differ 1.54x
       471  3.157            prime    replayed        2188       2672    1.20  1.1e-15  bluestein
       472  2^3.59           prime    replayed        2203       3376    1.45  9.9e-16  bluestein
       473  11.43            2p       replayed         927       3051    3.23  6.9e-16  
       474  2.3.79           prime    replayed        2221       4830    2.17  6.9e-16  bluestein
       475  5^2.19           2p       replayed         541       1284    2.33  5.0e-16  
       476  2^2.7.17         chain3   replayed         560       1106    1.94  5.1e-16  
       477  3^2.53           prime    replayed        2165       4111    1.86  9.4e-16  bluestein
       478  2.239            prime    replayed        2191       2687    1.19  9.8e-16  bluestein
       479  479              prime    replayed        2190       2681    1.21  9.1e-16  bluestein
       480  2^5.3.5          2p       replayed         378        582    1.48  5.3e-16  
       481  13.37            2p       replayed         812       2577    3.10  5.8e-16  
       482  2.241            prime    replayed        2176       2704    1.19  8.4e-16  bluestein
       483  3.7.23           2p       replayed         701       1588    2.18  5.1e-16  
       484  2^2.11^2         chain3   replayed         529        656    1.20  4.4e-16  
       485  5.97             prime    replayed        2160       2698    1.24  7.3e-16  bluestein
       486  2.3^5            chain3   replayed         484        655    1.35  6.6e-16  
       487  487              prime    replayed        2229       2692    1.20  7.9e-16  bluestein
       488  2^3.61           prime    replayed        2160       3679    1.69  7.2e-16  bluestein
       489  3.163            prime    replayed        2221       2701    1.21  1.0e-15  bluestein
       490  2.5.7^2          chain3   replayed         452        705    1.49  5.8e-16  
       491  491              prime    replayed        1883       2693    1.42  8.9e-16  rader
       492  2^2.3.41         2p       replayed         964       2303    1.35  4.8e-16  flips differ 1.77x
       493  17.29            2p       replayed         807       2677    2.76  6.4e-16  
       494  2.13.19          flat     replayed         848       1395    1.56  1.4e-15  
       495  3^2.5.11         flat     replayed         595        671    1.12  6.2e-16  
       496  2^4.31           flat     replayed         804       1666    2.00  5.7e-16  
       497  7.71             prime    replayed        2167       6036    2.78  7.9e-16  bluestein
       498  2.3.83           prime    replayed        2267       5381    2.37  8.6e-16  bluestein
       499  499              prime    replayed        2157       2708    1.24  9.5e-16  bluestein
       500  2^2.5^3          chain3   replayed         429        644    1.45  5.4e-16  
       501  3.167            prime    replayed        2207       2726    1.22  9.7e-16  bluestein
       502  2.251            prime    replayed        2255       2721    1.20  9.8e-16  bluestein
       503  503              prime    replayed        2207       2708    1.22  8.0e-16  bluestein
       504  2^3.3^2.7        chain3   replayed         458        663    1.26  7.2e-16  
       505  5.101            prime    replayed        2182       2724    1.25  1.0e-15  bluestein
       506  2.11.23          flat     replayed         891       1746    1.74  1.2e-15  
       507  3.13^2           chain3   replayed         628        727    1.10  6.3e-16  
       508  2^2.127          prime    replayed        2198       9125    4.15  1.1e-15  bluestein
       509  509              prime    replayed        2308       2727    1.18  8.6e-16  bluestein
       510  2.3.5.17         chain3   replayed         545       1274    2.28  7.6e-16  
       511  7.73             prime    replayed        2182       6480    2.94  9.5e-16  bluestein
       513  3^3.19           2p       replayed         580       1473    2.53  5.6e-16  
       515  5.103            prime    replayed        4780       4573    0.94  6.6e-16  bluestein
       516  2^2.3.43         flat     replayed         940       2776    2.95  5.7e-16  
       517  11.47            flat     replayed        1329       3708    2.73  7.3e-16  
       518  2.7.37           flat     replayed        1199       2530    2.10  6.2e-16  
       519  3.173            prime    replayed        4769       4578    0.94  5.2e-16  bluestein
       520  2^3.5.13         chain3   replayed         520        768    1.46  7.1e-16  
       521  521              prime    replayed        1959       4597    2.33  1.1e-15  rader
       522  2.3^2.29         chain3   replayed         841       1970    1.95  5.4e-16  
       523  523              prime    replayed        2955       4598    1.28  8.4e-16  rader
       525  3.5^2.7          2p       replayed         541        720    1.20  6.9e-16  
       527  17.31            2p       replayed         912       2966    2.71  9.4e-16  
       528  2^4.3.11         chain3   replayed         486        731    1.40  5.5e-16  
       529  23^2             2p       replayed         894       2775    3.10  6.5e-16  
       531  3^2.59           prime    replayed        4761       5163    1.08  7.6e-16  bluestein
       532  2^2.7.19         flat     replayed         662       1437    2.16  5.0e-16  
       533  13.41            2p       replayed        1003       3297    3.27  6.4e-16  
       535  5.107            prime    replayed        4796       4591    0.95  9.0e-16  bluestein
       537  3.179            prime    replayed        4796       4608    0.95  7.4e-16  bluestein
       539  7^2.11           flat     replayed         677        721    1.03  7.7e-16  
       541  541              prime    replayed        3037       4616    1.51  8.0e-16  rader
       543  3.181            prime    replayed        4784       4613    0.94  7.2e-16  bluestein
       544  2^5.17           2p       replayed         522       1354    2.55  5.6e-16  
       545  5.109            prime    replayed        5015       4619    0.91  7.7e-16  bluestein
       546  2.3.7.13         chain3   replayed         640        808    1.26  6.6e-16  
       547  547              prime    replayed        2206       4612    2.07  1.0e-15  rader
       549  3^2.61           prime    replayed        4816       5655    1.12  8.4e-16  bluestein
       550  2.5^2.11         chain3   replayed         592        819    1.35  5.3e-16  
       551  19.29            2p       replayed         947       3120    3.27  4.0e-16  
       552  2^3.3.23         flat     replayed         875       1743    1.90  6.0e-16  
       553  7.79             prime    replayed        4897       7714    1.42  7.8e-16  bluestein
       555  3.5.37           2p       replayed         932       3007    3.18  7.2e-16  
       557  557              prime    replayed        4863       4644    0.93  8.2e-16  bluestein
       558  2.3^2.31         chain3   replayed         959       2226    2.24  6.6e-16  
       559  13.43            2p       replayed        1104       3640    3.17  6.4e-16  
       560  2^4.5.7          chain3   replayed         478        655    1.34  4.6e-16  
       561  3.11.17          flat     replayed         851       1479    1.71  1.3e-15  
       563  563              prime    replayed        4856       4633    0.94  1.2e-15  bluestein
       564  2^2.3.47         flat     replayed        1141       3428    2.93  4.6e-16  
       565  5.113            prime    replayed        4922       4687    0.93  6.4e-16  bluestein
       567  3^4.7            2p       replayed         602        835    1.37  7.1e-16  
       569  569              prime    replayed        4791       4707    0.96  8.3e-16  bluestein
       570  2.3.5.19         flat     replayed         878       1508    1.62  6.9e-16  
       571  571              prime    replayed        2533       4646    1.69  8.5e-16  rader
       572  2^2.11.13        chain3   replayed         739        868    1.14  6.4e-16  
       573  3.191            prime    replayed        4846       4650    0.95  1.0e-15  bluestein
       574  2.7.41           flat     replayed        1749       3059    1.74  2.1e-15  
       575  5^2.23           2p       replayed         848       1882    1.99  6.2e-16  
       577  577              prime    replayed        1934       4803    2.46  1.0e-15  rader
       578  2.17^2           flat     replayed         975       2066    2.09  1.1e-15  
       579  3.193            prime    replayed        4793       4781    0.99  7.0e-16  bluestein
       580  2^2.5.29         chain3   replayed        1031       2123    1.58  4.7e-16  flips differ 1.30x
       581  7.83             prime    replayed        4818       8770    1.80  7.6e-16  bluestein
       583  11.53            prime    replayed        4916       5048    0.96  7.9e-16  bluestein
       585  3^2.5.13         chain3   replayed         789        855    1.05  6.8e-16  
       587  587              prime    replayed        4939       4803    0.95  1.0e-15  bluestein
       588  2^2.3.7^2        flat     replayed         614        830    1.32  5.4e-16  
       589  19.31            2p       replayed        1061       3497    2.58  5.1e-16  flips differ 1.28x
       591  3.197            prime    replayed        4792       4793    0.99  8.3e-16  bluestein
       592  2^4.37           2p       replayed         794       2393    2.83  5.7e-16  
       593  593              prime    replayed        3070       4818    1.31  8.6e-16  rader
       594  2.3^3.11         chain3   replayed         747        891    1.18  5.7e-16  
       595  5.7.17           chain3   replayed         899       1575    1.74  5.8e-16  
       597  3.199            prime    replayed        4936       4818    0.96  9.5e-16  bluestein
       598  2.13.23          flat     replayed        1080       2118    1.50  9.1e-16  flips differ 1.31x
       599  599              prime    replayed        4824       4801    0.94  8.6e-16  bluestein
       601  601              prime    replayed        3138       4824    1.41  8.9e-16  rader
       602  2.7.43           flat     replayed        1766       3348    1.87  6.8e-16  
       603  3^2.67           prime    replayed        4841       6912    1.40  7.5e-16  bluestein
       605  5.11^2           flat     replayed         768        836    1.08  6.6e-16  
       607  607              prime    replayed        4882       4809    0.98  1.1e-15  bluestein
       608  2^5.19           2p       replayed         567       1613    2.84  5.8e-16  
       609  3.7.29           2p       replayed        1310       2517    1.90  7.3e-16  
       611  13.47            flat     replayed        1675       4525    2.54  7.8e-16  
       612  2^2.3^2.17       chain3   replayed         856       1592    1.84  5.9e-16  
       613  613              prime    replayed        2722       4834    1.75  1.3e-15  rader
       615  3.5.41           2p       replayed        1146       3848    3.35  6.8e-16  
       616  2^3.7.11         chain3   replayed         661        871    1.31  3.8e-16  
       617  617              prime    replayed        2316       4842    1.97  1.0e-15  rader
       619  619              prime    replayed        4955       4820    0.96  8.3e-16  bluestein
       620  2^2.5.31         flat     replayed        1183       2369    1.96  7.2e-16  
       621  3^3.23           2p       replayed         809       2092    2.55  6.7e-16  
       623  7.89             prime    replayed        4911      10655    2.16  7.5e-16  bluestein
       624  2^4.3.13         chain3   replayed         722        878    1.21  5.8e-16  
       625  5^4              2p       replayed         692        872    1.25  4.9e-16  
       627  3.11.19          flat     replayed         993       1752    1.74  7.6e-16  
       629  17.37            2p       replayed        1563       4224    1.88  1.0e-15  flips differ 1.44x
       630  2.3^2.5.7        chain3   replayed         710        838    1.18  4.8e-16  
       631  631              prime    replayed        2949       4830    1.61  9.3e-16  rader
       633  3.211            prime    replayed        5016       4854    0.95  6.7e-16  bluestein
       635  5.127            prime    replayed        4877       4837    0.98  8.1e-16  bluestein
       637  7^2.13           flat     replayed         923        873    0.92  7.0e-16  
       638  2.11.29          flat     replayed        1347       2599    1.71  8.3e-16  
       639  3^2.71           prime    replayed        4892       7852    1.58  7.8e-16  bluestein
       641  641              prime    replayed        1901       6249    3.16  7.5e-16  rader
       643  643              prime    replayed        4843       6225    1.27  1.1e-15  bluestein
       644  2^2.7.23         flat     replayed         977       2018    2.05  6.0e-16  
       645  3.5.43           2p       replayed        1284       4233    3.23  6.1e-16  
       646  2.17.19          flat     replayed        1107       2415    2.05  1.0e-15  
       647  647              prime    replayed        4874       6234    1.25  1.1e-15  bluestein
       649  11.59            prime    replayed        4913       6253    1.25  6.8e-16  bluestein
       650  2.5^2.13         chain3   replayed         791        986    1.22  8.0e-16  
       651  3.7.31           2p       replayed        1240       2845    1.73  5.6e-16  flips differ 1.33x
       653  653              prime    replayed        4979       6298    1.22  7.1e-16  bluestein
       655  5.131            prime    replayed        4855       6247    1.27  8.0e-16  bluestein
       656  2^4.41           2p       replayed         976       2974    2.96  4.6e-16  
       657  3^2.73           prime    replayed        4850       8426    1.72  8.4e-16  bluestein
       658  2.7.47           flat     replayed        2144       4121    1.90  5.8e-16  
       659  659              prime    replayed        4893       6249    1.27  8.0e-16  bluestein
       660  2^2.3.5.11       chain3   replayed         694        975    1.39  6.2e-16  
       661  661              prime    replayed        2895       6273    2.15  9.4e-16  rader
       663  3.13.17          flat     replayed        1060       1837    1.71  8.2e-16  
       665  5.7.19           flat     replayed         932       1871    2.00  5.3e-16  
       666  2.3^2.37         chain3   replayed        1263       3086    2.27  4.8e-16  
       667  23.29            2p       replayed        1472       4100    2.36  6.8e-16  
       669  3.223            prime    replayed        4915       6281    1.27  8.1e-16  bluestein
       671  11.61            prime    replayed        4965       6861    1.33  9.9e-16  bluestein
       672  2^5.3.7          2p       replayed         577        881    1.50  4.0e-16  
       673  673              prime    replayed        2490       6285    2.52  9.1e-16  rader
       675  3^3.5^2          2p       replayed         751        998    1.33  7.3e-16  
       676  2^2.13^2         chain3   replayed         938       1027    1.09  8.1e-16  
       677  677              prime    replayed        2887       6294    2.12  1.1e-15  rader
       679  7.97             prime    replayed        4965       6269    1.26  9.0e-16  bluestein
       680  2^3.5.17         flat     replayed         870       1709    1.95  6.6e-16  
       681  3.227            prime    replayed        4928       6298    1.24  1.0e-15  bluestein
       682  2.11.31          flat     replayed        1557       2902    1.71  5.8e-16  
       683  683              prime    replayed        4865       6282    1.25  6.0e-16  bluestein
       684  2^2.3^2.19       chain3   replayed         912       1901    2.07  6.2e-16  
       685  5.137            prime    replayed        4866       6300    1.26  8.3e-16  bluestein
       686  2.7^3            flat     replayed         968       1073    1.10  4.6e-16  
       687  3.229            prime    replayed        4940       6296    1.26  8.2e-16  bluestein
       688  2^4.43           flat     replayed        1283       3279    2.55  7.2e-16  
       689  13.53            prime    replayed        4883       6005    1.15  9.6e-16  bluestein
       690  2.3.5.23         chain3   replayed        1164       2121    1.82  5.5e-16  
       691  691              prime    replayed        3646       6279    1.70  9.3e-16  rader
       693  3^2.7.11         flat     replayed         904        998    1.09  7.4e-16  
       695  5.139            prime    replayed        4961       6295    1.25  8.8e-16  bluestein
       696  2^3.3.29         flat     replayed        1357       2629    1.51  6.2e-16  flips differ 1.28x
       697  17.41            2p       replayed        1461       5274    3.45  8.6e-16  
       699  3.233            prime    replayed        5015       6289    1.25  8.2e-16  bluestein
       700  2^2.5^2.7        chain3   replayed         735        926    1.25  4.1e-16  
       701  701              prime    replayed        3724       6329    1.46  1.1e-15  rader
       702  2.3^3.13         chain3   replayed         917       1073    1.16  5.7e-16  
       703  19.37            2p       replayed        1380       4918    3.49  5.2e-16  
       704  2^6.11           2p       replayed         653        901    1.25  3.7e-16  
       705  3.5.47           2p       replayed        1577       5177    3.16  8.0e-16  
       707  7.101            prime    replayed        4898       6303    1.24  1.1e-15  bluestein
       709  709              prime    replayed        5003       6319    1.26  8.2e-16  bluestein
       711  3^2.79           prime    replayed        4999      10018    2.00  7.1e-16  bluestein
       713  23.31            2p       replayed        1720       4572    1.66  9.4e-16  flips differ 1.60x
       714  2.3.7.17         chain3   replayed         990       1805    1.79  7.2e-16  
       715  5.11.13          flat     replayed         935        999    1.06  7.7e-16  
       717  3.239            prime    replayed        4970       6325    1.22  9.4e-16  bluestein
       719  719              prime    replayed        4883       6310    1.27  8.3e-16  bluestein
       721  7.103            prime    replayed        4967       6343    1.21  7.3e-16  bluestein
       722  2.19^2           flat     replayed        1311       2804    2.08  1.0e-15  
       723  3.241            prime    replayed        4876       6314    1.29  6.8e-16  bluestein
       725  5^2.29           flat     replayed        1236       3031    2.45  6.2e-16  
       726  2.3.11^2         chain3   replayed         979       1084    1.10  4.8e-16  
       727  727              prime    replayed        2978       6308    2.09  8.7e-16  rader
       728  2^3.7.13         chain3   replayed         831       1048    1.26  6.1e-16  
       729  3^6              2p       replayed         826       1090    1.32  6.8e-16  
       731  17.43            2p       replayed        1598       5761    3.43  6.4e-16  
       733  733              prime    replayed        4942       6365    1.28  9.2e-16  bluestein
       735  3.5.7^2          flat     replayed        1007       1040    1.02  6.1e-16  
       736  2^5.23           2p       replayed         906       2270    1.92  4.1e-16  flips differ 1.31x
       737  11.67            prime    replayed        4938       8383    1.68  9.0e-16  bluestein
       738  2.3^2.41         chain3   replayed        2265       3777    1.64  5.8e-16  
       739  739              prime    replayed        4977       6335    1.20  8.3e-16  bluestein
       740  2^2.5.37         flat     replayed        1338       3346    2.11  7.1e-16  
       741  3.13.19          chain3   replayed        1357       2143    1.35  5.9e-16  
       743  743              prime    replayed        5142       6332    1.19  7.8e-16  bluestein
       744  2^3.3.31         flat     replayed        1584       2937    1.80  7.3e-16  
       745  5.149            prime    replayed        5050       6379    1.23  8.5e-16  bluestein
       747  3^2.83           prime    replayed        5092      11438    2.24  1.1e-15  bluestein
       748  2^2.11.17        flat     replayed        1120       1922    1.59  8.6e-16  
       749  7.107            prime    replayed        5006       6367    1.20  9.2e-16  bluestein
       751  751              prime    replayed        4993       6380    1.26  1.0e-15  bluestein
       752  2^4.47           flat     replayed        1577       4115    2.60  5.8e-16  
       753  3.251            prime    replayed        5000       6375    1.21  9.2e-16  bluestein
       754  2.13.29          flat     replayed        1645       3143    1.90  1.0e-15  
       755  5.151            prime    replayed        4919       6350    1.28  9.4e-16  bluestein
       756  2^2.3^3.7        chain3   replayed         779       1055    1.35  4.8e-16  
       757  757              prime    replayed        4416       6370    1.33  7.4e-16  rader
       759  3.11.23          flat     replayed        1421       2563    1.79  7.1e-16  
       760  2^3.5.19         flat     replayed        1080       2041    1.83  6.7e-16  
       761  761              prime    replayed        3371       6389    1.89  8.7e-16  rader
       763  7.109            prime    replayed        4979       6366    1.27  8.0e-16  bluestein
       765  3^2.5.17         flat     replayed        1125       2129    1.77  6.7e-16  
       767  13.59            prime    replayed        5011       7473    1.49  6.6e-16  bluestein
       769  769              prime    replayed        2305       6338    2.71  7.6e-16  rader
       770  2.5.7.11         chain3   replayed         953       1297    1.35  4.8e-16  
       771  3.257            prime    replayed        4963       6303    1.24  1.5e-15  bluestein
       773  773              prime    replayed        5021       6355    1.22  9.7e-16  bluestein
       774  2.3^2.43         chain3   replayed        1747       4157    2.37  5.3e-16  
       775  5^2.31           2p       replayed        1253       3462    2.73  5.8e-16  
       777  3.7.37           2p       replayed        1303       4247    3.22  6.7e-16  
       779  19.41            2p       replayed        1792       6116    3.25  5.0e-16  
       780  2^2.3.5.13       chain3   replayed         873       1173    1.34  9.1e-16  
       781  11.71            prime    replayed        4973       9574    1.87  7.8e-16  bluestein
       782  2.17.23          flat     replayed        1510       3264    1.90  2.0e-15  
       783  3^3.29           flat     replayed        1553       3439    1.98  5.8e-16  
       784  2^4.7^2          chain3   replayed         821        907    1.09  5.2e-16  
       785  5.157            prime    replayed        5029       6429    1.25  1.1e-15  bluestein
       787  787              prime    replayed        5106       6323    1.22  8.6e-16  bluestein
       789  3.263            prime    replayed        5069       6387    1.25  8.8e-16  bluestein
       791  7.113            prime    replayed        4981       6328    1.26  9.8e-16  bluestein
       792  2^3.3^2.11       chain3   replayed         820       1184    1.44  6.6e-16  
       793  13.61            prime    replayed        5045       8223    1.57  7.9e-16  bluestein
       795  3.5.53           prime    replayed        4940       6923    1.36  6.7e-16  bluestein
       797  797              prime    replayed        5076       6384    1.23  7.4e-16  bluestein
       798  2.3.7.19         flat     replayed        1336       2155    1.60  6.5e-16  
       799  17.47            2p       replayed        1919       6924    3.48  6.2e-16  
       801  3^2.89           prime    replayed        5088      13851    2.59  9.2e-16  bluestein
       803  11.73            prime    replayed        4959      10236    1.95  6.9e-16  bluestein
       805  5.7.23           flat     replayed        1267       2676    2.11  7.2e-16  
       806  2.13.31          flat     replayed        2117       3500    1.60  8.8e-16  
       807  3.269            prime    replayed        5144       5791    1.10  6.9e-16  bluestein
       809  809              prime    replayed        4986       5854    1.17  8.8e-16  bluestein
       811  811              prime    replayed        5029       5796    1.14  6.5e-16  bluestein
       812  2^2.7.29         flat     replayed        1645       3058    1.74  6.7e-16  
       813  3.271            prime    replayed        4984       5935    1.12  8.3e-16  bluestein
       814  2.11.37          flat     replayed        1965       4036    1.95  9.4e-16  
       815  5.163            prime    replayed        5013       5926    1.16  1.0e-15  bluestein
       816  2^4.3.17         chain3   replayed        1322       2003    1.51  4.7e-16  
       817  19.43            flat     replayed        2080       6685    3.13  8.9e-16  
       819  3^2.7.13         flat     replayed        1147       1217    1.04  9.0e-16  
       820  2^2.5.41         flat     replayed        1516       4108    2.66  5.7e-16  
       821  821              prime    replayed        4996       5842    1.15  8.2e-16  bluestein
       823  823              prime    replayed        5072       5888    1.10  1.4e-15  bluestein
       825  3.5^2.11         chain3   replayed        1262       1220    0.95  4.4e-16  
       827  827              prime    replayed        4939       5819    1.15  6.9e-16  bluestein
       828  2^2.3^2.23       chain3   replayed        1533       2659    1.72  6.4e-16  
       829  829              prime    replayed        4484       5856    1.30  1.4e-15  rader
       831  3.277            prime    replayed        5108       5847    1.12  9.6e-16  bluestein
       832  2^6.13           2p       replayed         807       1093    1.34  5.0e-16  
       833  7^2.17           flat     replayed        1207       2234    1.84  7.7e-16  
       835  5.167            prime    replayed        5063       5859    1.15  8.0e-16  bluestein
       836  2^2.11.19        flat     replayed        1455       2301    1.54  8.2e-16  
       837  3^3.31           flat     replayed        1860       3810    2.03  7.7e-16  
       839  839              prime    replayed        5072       5922    1.14  8.6e-16  bluestein
       840  2^3.3.5.7        chain3   replayed         879       1144    1.29  5.4e-16  
       841  29^2             flat     replayed        2364       6003    2.38  1.2e-15  
       843  3.281            prime    replayed        4987       5814    1.15  7.4e-16  bluestein
       845  5.13^2           flat     replayed        1352       1266    0.93  9.2e-16  
       846  2.3^2.47         flat     replayed        2945       5143    1.69  5.8e-16  
       847  7.11^2           flat     replayed        1173       1203    0.99  5.1e-16  
       849  3.283            prime    replayed        4951       5887    1.17  9.3e-16  bluestein
       850  2.5^2.17         chain3   replayed        1416       2180    1.45  5.7e-16  
       851  23.37            2p       replayed        1952       6375    2.91  6.4e-16  
       853  853              prime    replayed        5062       5873    1.11  1.3e-15  bluestein
       855  3^2.5.19         flat     replayed        1367       2500    1.81  6.7e-16  
       857  857              prime    replayed        4964       5874    1.16  1.0e-15  bluestein
       858  2.3.11.13        flat     replayed        1432       1308    0.91  9.6e-16  
       859  859              prime    replayed        3630       5896    1.58  1.2e-15  rader
       860  2^2.5.43         flat     replayed        1713       4501    2.52  7.6e-16  
       861  3.7.41           2p       replayed        1636       5428    3.26  4.6e-16  
       863  863              prime    replayed        4996       5827    1.14  1.0e-15  bluestein
       865  5.173            prime    replayed        5033       5870    1.16  7.5e-16  bluestein
       867  3.17^2           flat     replayed        1535       3524    2.23  1.1e-15  
       868  2^2.7.31         flat     replayed        1501       3423    1.78  7.7e-16  flips differ 1.28x
       869  11.79            prime    replayed        5092      12187    2.39  9.2e-16  bluestein
       870  2.3.5.29         flat     replayed        2262       3227    1.41  6.5e-16  
       871  13.67            prime    replayed        5030      10082    1.98  1.0e-15  bluestein
       873  3^2.97           prime    replayed        5040       6023    1.19  8.1e-16  bluestein
       874  2.19.23          flat     replayed        1729       3777    1.80  1.3e-15  
       875  5^3.7            flat     replayed        1131       1303    1.10  5.4e-16  
       877  877              prime    replayed        5069       5896    1.16  6.9e-16  bluestein
       879  3.293            prime    replayed        5231       5946    1.12  7.1e-16  bluestein
       880  2^4.5.11         chain3   replayed         965       1188    1.22  4.3e-16  
       881  881              prime    replayed        3459       5945    1.70  6.4e-16  rader
       882  2.3^2.7^2        flat     replayed        1344       1257    0.91  7.2e-16  
       883  883              prime    replayed        5183       5862    1.12  8.9e-16  bluestein
       884  2^2.13.17        chain3   replayed        1375       2290    1.65  7.3e-16  
       885  3.5.59           prime    replayed        5182       8675    1.43  8.6e-16  bluestein
       887  887              prime    replayed        5040       5867    1.16  6.8e-16  bluestein
       888  2^3.3.37         chain3   replayed        1696       4130    2.36  6.1e-16  
       889  7.127            prime    replayed        5254       6317    1.16  7.7e-16  bluestein
       891  3^4.11           chain3   replayed        1228       1424    1.11  6.3e-16  
       893  19.47            flat     replayed        2338       8010    3.32  9.9e-16  
       895  5.179            prime    replayed        5003       5878    1.16  8.7e-16  bluestein
       896  2^7.7            chain3   replayed         950       1154    1.21  3.7e-16  
       897  3.13.23          flat     replayed        1722       3131    1.58  1.1e-15  
       899  29.31            2p       replayed        2683       6725    2.47  5.6e-16  
       901  17.53            prime    replayed        5213       9053    1.65  1.2e-15  bluestein
       902  2.11.41          flat     replayed        2579       4916    1.89  9.8e-16  
       903  3.7.43           flat     replayed        2391       5991    2.34  1.9e-15  
       905  5.181            prime    replayed        5098       6016    1.15  6.9e-16  bluestein
       907  907              prime    replayed        5181       5953    1.11  1.1e-15  bluestein
       909  3^2.101          prime    replayed        5137       5960    1.12  8.3e-16  bluestein
       910  2.5.7.13         chain3   replayed        1188       1562    1.29  5.6e-16  
       911  911              prime    replayed        5229       5885    1.12  9.4e-16  bluestein
       912  2^4.3.19         chain3   replayed        1178       2397    2.00  7.2e-16  
       913  11.83            prime    replayed        5066      13904    2.73  8.1e-16  bluestein
       915  3.5.61           prime    replayed        5015       9534    1.87  8.8e-16  bluestein
       917  7.131            prime    replayed        5083       5936    1.12  1.3e-15  bluestein
       918  2.3^3.17         flat     replayed        1633       2368    1.39  9.6e-16  
       919  919              prime    replayed        5070       5966    1.12  9.0e-16  bluestein
       920  2^3.5.23         chain3   replayed        1656       2871    1.58  4.9e-16  
       921  3.307            prime    replayed        5018       5957    1.17  9.9e-16  bluestein
       923  13.71            prime    replayed        5120      11483    2.24  6.4e-16  bluestein
       924  2^2.3.7.11       chain3   replayed         990       1468    1.48  5.8e-16  
       925  5^2.37           2p       replayed        2299       5108    2.19  5.0e-16  
       927  3^2.103          prime    replayed        5414       5905    1.08  8.4e-16  bluestein
       928  2^5.29           2p       replayed        1299       3445    2.60  5.6e-16  
       929  929              prime    replayed        4833       5972    1.08  6.8e-16  rader
       930  2.3.5.31         chain3   replayed        2638       3626    1.04  6.5e-16  flips differ 1.32x
       931  7^2.19           flat     replayed        1539       2650    1.70  7.4e-16  
       933  3.311            prime    replayed        5112       6100    1.12  7.8e-16  bluestein
       935  5.11.17          flat     replayed        1391       2549    1.81  7.5e-16  
       936  2^3.3^2.13       chain3   replayed        1048       1425    1.34  8.7e-16  
       937  937              prime    replayed        3641       6001    1.61  8.9e-16  rader
       939  3.313            prime    replayed        5104       5927    1.16  7.2e-16  bluestein
       940  2^2.5.47         flat     replayed        1968       5684    2.81  5.2e-16  
       941  941              prime    replayed        5102       5982    1.16  8.8e-16  bluestein
       943  23.41            2p       replayed        2777       8016    2.85  4.9e-16  
       945  3^3.5.7          chain3   replayed        1373       1474    1.06  5.7e-16  
       946  2.11.43          flat     replayed        2864       5522    1.89  7.5e-16  
       947  947              prime    replayed        5202       5990    1.14  1.1e-15  bluestein
       949  13.73            prime    replayed        5030      12391    2.34  6.9e-16  bluestein
       950  2.5^2.19         flat     replayed        1668       2608    1.44  6.1e-16  
       951  3.317            prime    replayed        5118       5918    1.03  9.6e-16  bluestein
       952  2^3.7.17         flat     replayed        1279       2392    1.82  7.6e-16  
       953  953              prime    replayed        4152       6020    1.44  9.4e-16  rader
       955  5.191            prime    replayed        5051       5911    1.16  9.2e-16  bluestein
       957  3.11.29          chain3   replayed        2346       4049    1.72  5.4e-16  
       959  7.137            prime    replayed        5137       5934    1.15  9.7e-16  bluestein
       961  31^2             flat     replayed        2577       7389    2.33  9.8e-16  
       962  2.13.37          flat     replayed        2546       4919    1.92  9.1e-16  
       963  3^2.107          prime    replayed        5084       5959    1.16  1.2e-15  bluestein
       965  5.193            prime    replayed        5115       5974    1.16  9.7e-16  bluestein
       966  2.3.7.23         chain3   replayed        1720       3032    1.75  6.7e-16  
       967  967              prime    replayed        5099       5933    1.14  7.3e-16  bluestein
       968  2^3.11^2         chain3   replayed        1157       1450    1.25  5.6e-16  
       969  3.17.19          flat     replayed        1794       4125    2.27  1.0e-15  
       971  971              prime    replayed        5272       6006    1.11  1.1e-15  bluestein
       973  7.139            prime    replayed        5250       6036    1.14  8.0e-16  bluestein
       975  3.5^2.13         flat     replayed        1339       1494    1.05  9.1e-16  
       977  977              prime    replayed        5403       5992    1.10  1.1e-15  bluestein
       979  11.89            prime    replayed        5274      16944    3.21  1.1e-15  bluestein
       980  2^2.5.7^2        flat     replayed        1256       1405    1.11  5.5e-16  
       981  3^2.109          prime    replayed        5203       6021    1.11  9.3e-16  bluestein
       983  983              prime    replayed        5126       5953    1.15  9.6e-16  bluestein
       984  2^3.3.41         chain3   replayed        2062       5127    2.46  5.2e-16  
       985  5.197            prime    replayed        5395       6072    1.06  1.3e-15  bluestein
       986  2.17.29          flat     replayed        2754       4794    1.72  9.0e-16  
       987  3.7.47           2p       replayed        2143       7374    3.27  5.5e-16  
       988  2^2.13.19        chain3   replayed        1684       2799    1.38  6.4e-16  
       989  23.43            flat     replayed        2691       8677    2.73  1.3e-15  
       990  2.3^2.5.11       chain3   replayed        1196       1499    1.24  6.6e-16  
       991  991              prime    replayed        5129       5988    1.17  8.4e-16  bluestein
       992  2^5.31           flat     replayed        1990       3916    1.68  1.4e-15  
       993  3.331            prime    replayed        5056       6032    1.18  7.8e-16  bluestein
       995  5.199            prime    replayed        5075       5998    1.15  8.3e-16  bluestein
       997  997              prime    replayed        5092       6058    1.18  9.5e-16  bluestein
       999  3^3.37           2p       replayed        2800       5685    1.51  7.6e-16  flips differ 1.35x
      1001  7.11.13          flat     replayed        1490       1471    0.97  8.1e-16  
      1003  17.59            prime    replayed        5133      11281    2.08  7.1e-16  bluestein
      1005  3.5.67           prime    replayed        5246      11697    2.14  7.2e-16  bluestein
      1007  19.53            prime    replayed        5250      10465    1.90  8.6e-16  bluestein
      1008  2^4.3^2.7        chain3   replayed        1044       1317    1.24  7.6e-16  
      1009  1009             prime    replayed        5215       6041    1.15  1.1e-15  bluestein
      1011  3.337            prime    replayed        5086       5996    1.17  9.4e-16  bluestein
      1012  2^2.11.23        chain3   replayed        1853       3291    1.54  6.1e-16  
      1013  1013             prime    replayed        5419       6074    1.11  1.1e-15  bluestein
      1014  2.3.13^2         chain3   replayed        1446       1603    1.10  6.3e-16  
      1015  5.7.29           flat     replayed        1864       4358    1.96  7.4e-16  
      1017  3^2.113          prime    replayed        5236       6042    1.10  9.6e-16  bluestein
      1019  1019             prime    replayed        5189       6019    1.15  9.1e-16  bluestein
      1020  2^2.3.5.17       chain3   replayed        1437       2692    1.83  5.6e-16  
      1021  1021             prime    replayed        5159       6068    1.14  9.7e-16  bluestein
      1023  3.11.31          chain3   replayed        2295       4649    1.74  5.3e-16  
      1025  5^2.41           2p       replayed        2004       6571    3.10  6.0e-16  
      1026  2.3^3.19         chain3   replayed        1574       2902    1.82  5.4e-16  
      1027  13.79            prime    replayed       10385      14744    1.41  7.3e-16  bluestein
      1029  3.7^3            flat     replayed        1546       1564    0.99  5.9e-16  
      1031  1031             prime    replayed       10474       9899    0.93  1.1e-15  bluestein
      1032  2^3.3.43         chain3   replayed        2148       5614    2.52  6.0e-16  
      1033  1033             prime    replayed        8574       9978    1.14  9.1e-16  rader
      1034  2.11.47          flat     replayed        3306       6675    2.02  1.1e-15  
      1035  3^2.5.23         flat     replayed        1843       3690    1.98  6.1e-16  
      1036  2^2.7.37         flat     replayed        1893       4903    2.54  6.6e-16  
      1037  17.61            prime    replayed       10453      12266    1.17  8.1e-16  bluestein
      1039  1039             prime    replayed       10330       9882    0.95  9.9e-16  bluestein
      1040  2^4.5.13         chain3   replayed        1189       1490    1.24  6.1e-16  
      1041  3.347            prime    replayed       10407       9967    0.95  8.2e-16  bluestein
      1043  7.149            prime    replayed       10617       9907    0.93  8.0e-16  bluestein
      1044  2^2.3^2.29       chain3   replayed        2391       4071    1.42  6.8e-16  
      1045  5.11.19          flat     replayed        1845       3107    1.66  6.3e-16  
      1047  3.349            prime    replayed       10538       9915    0.93  7.3e-16  bluestein
      1049  1049             prime    replayed       10673       9993    0.93  9.0e-16  bluestein
      1050  2.3.5^2.7        chain3   replayed        1635       1489    0.89  7.3e-16  
      1051  1051             prime    replayed        5505       9925    1.74  1.0e-15  rader
      1053  3^4.13           chain3   replayed        1646       1820    1.10  9.5e-16  
      1054  2.17.31          flat     replayed        3029       5316    1.69  1.7e-15  
      1055  5.211            prime    replayed       10497       9939    0.94  7.4e-16  bluestein
      1056  2^5.3.11         chain3   replayed        1215       1631    1.33  7.2e-16  
      1057  7.151            prime    replayed       10552      10009    0.93  8.2e-16  bluestein
      1058  2.23^2           flat     replayed        2786       5090    1.54  1.3e-15  
      1059  3.353            prime    replayed       10361       9946    0.94  8.1e-16  bluestein
      1061  1061             prime    replayed       10536      10012    0.93  8.2e-16  bluestein
      1063  1063             prime    replayed       10550       9926    0.93  7.6e-16  bluestein
      1064  2^3.7.19         chain3   replayed        1408       2926    2.00  4.2e-16  
      1065  3.5.71           prime    replayed       10408      13290    1.23  7.2e-16  bluestein
      1066  2.13.41          flat     replayed        3211       5965    1.85  8.6e-16  
      1067  11.97            prime    replayed       10385       9923    0.95  1.1e-15  bluestein
      1069  1069             prime    replayed       10623       9998    0.93  1.3e-15  bluestein
      1071  3^2.7.17         flat     replayed        1822       3101    1.65  5.7e-16  
      1073  29.37            2p       replayed        2893       9315    3.14  5.9e-16  
      1075  5^2.43           2p       replayed        2271       7222    2.93  6.8e-16  
      1077  3.359            prime    replayed       10478      10000    0.93  7.2e-16  bluestein
      1078  2.7^2.11         flat     replayed        1852       1934    1.03  6.7e-16  
      1079  13.83            prime    replayed       10428      16807    1.59  1.1e-15  bluestein
      1081  23.47            2p       replayed        3829      10446    1.88  5.7e-16  flips differ 1.46x
      1083  3.19^2           chain3   replayed        2043       5000    2.08  5.0e-16  
      1085  5.7.31           chain3   replayed        3085       4989    1.27  5.7e-16  flips differ 1.29x
      1087  1087             prime    replayed       10389       9978    0.95  7.6e-16  bluestein
      1088  2^6.17           2p       replayed        1329       2647    1.99  5.4e-16  
      1089  3^2.11^2         chain3   replayed        1580       1732    1.09  5.4e-16  
      1091  1091             prime    replayed       10645       9949    0.93  8.0e-16  bluestein
      1092  2^2.3.7.13       chain3   replayed        1237       1816    1.45  6.8e-16  
      1093  1093             prime    replayed        5726      10018    1.74  1.1e-15  rader
      1095  3.5.73           prime    replayed       10379      14259    1.33  8.7e-16  bluestein
      1097  1097             prime    replayed       10735      10017    0.93  7.5e-16  bluestein
      1099  7.157            prime    replayed       10621       9969    0.94  1.1e-15  bluestein
      1100  2^2.5^2.11       chain3   replayed        1279       1720    1.32  4.0e-16  
      1101  3.367            prime    replayed       10634      10049    0.93  9.0e-16  bluestein
      1102  2.19.29          flat     replayed        3327       5519    1.60  1.1e-15  
      1103  1103             prime    replayed       10772       9973    0.91  7.6e-16  bluestein
      1104  2^4.3.23         chain3   replayed        1685       3459    1.47  6.9e-16  flips differ 1.40x
      1105  5.13.17          flat     replayed        1756       3195    1.78  8.0e-16  
      1107  3^3.41           2p       replayed        2413       7234    2.93  6.3e-16  
      1109  1109             prime    replayed       10486      10051    0.94  9.2e-16  bluestein
      1110  2.3.5.37         chain3   replayed        2283       5180    2.20  5.9e-16  
      1111  11.101           prime    replayed       10303       9963    0.95  9.3e-16  bluestein
      1113  3.7.53           prime    replayed       10683       9848    0.92  9.3e-16  bluestein
      1115  5.223            prime    replayed       10477       9999    0.92  8.1e-16  bluestein
      1116  2^2.3^2.31       flat     replayed        2116       4545    1.72  8.9e-16  flips differ 1.25x
      1117  1117             prime    replayed        8336      10195    1.19  9.9e-16  rader
      1118  2.13.43          flat     replayed        3371       6611    1.87  7.4e-16  
      1119  3.373            prime    replayed       10611       9993    0.94  1.1e-15  bluestein
      1120  2^5.5.7          chain3   replayed        1169       1593    1.35  5.0e-16  
      1121  19.59            prime    replayed       10839      12830    1.18  8.2e-16  bluestein
      1122  2.3.11.17        chain3   replayed        1718       2986    1.73  7.2e-16  
      1123  1123             prime    replayed        5433       9984    1.79  9.0e-16  rader
      1125  3^2.5^3          flat     replayed        1523       1873    1.18  5.9e-16  
      1127  7^2.23           chain3   replayed        2314       3878    1.57  8.1e-16  
      1128  2^3.3.47         flat     replayed        2760       6952    2.50  8.6e-16  
      1129  1129             prime    replayed       10562      10059    0.93  9.5e-16  bluestein
      1131  3.13.29          chain3   replayed        3082       5012    1.45  7.9e-16  
      1133  11.103           prime    replayed       10504      10060    0.95  6.6e-16  bluestein
      1134  2.3^4.7          chain3   replayed        1677       1736    1.01  6.5e-16  
      1135  5.227            prime    replayed       10448       9993    0.93  7.4e-16  bluestein
      1137  3.379            prime    replayed       10693      10070    0.94  8.3e-16  bluestein
      1139  17.67            prime    replayed       10653      14870    1.39  7.3e-16  bluestein
      1140  2^2.3.5.19       chain3   replayed        1528       3211    2.09  5.3e-16  
      1141  7.163            prime    replayed       10671      10070    0.94  9.0e-16  bluestein
      1143  3^2.127          prime    replayed       10509      10004    0.94  8.6e-16  bluestein
      1144  2^3.11.13        chain3   replayed        1456       1806    1.23  5.7e-16  
      1145  5.229            prime    replayed       10512      10065    0.94  7.6e-16  bluestein
      1147  31.37            flat     replayed        3182      10164    3.17  8.4e-16  
      1148  2^2.7.41         chain3   replayed        2355       5982    2.52  5.6e-16  
      1149  3.383            prime    replayed       10445      10087    0.96  7.0e-16  bluestein
      1150  2.5^2.23         chain3   replayed        2320       3732    1.48  6.0e-16  
      1151  1151             prime    replayed        7076      10007    1.40  1.0e-15  rader
      1153  1153             prime    replayed        4234      11566    2.71  1.0e-15  rader
      1155  3.5.7.11         chain3   replayed        1668       1838    1.07  4.6e-16  
      1156  2^2.17^2         chain3   replayed        1888       4196    2.17  9.6e-16  
      1157  13.89            prime    replayed       10645      20139    1.86  7.9e-16  bluestein
      1159  19.61            prime    replayed       10683      13977    1.31  8.3e-16  bluestein
      1160  2^3.5.29         flat     replayed        2583       4433    1.69  9.9e-16  
      1161  3^3.43           2p       replayed        2401       7946    3.21  5.8e-16  
      1163  1163             prime    replayed       10818      11501    1.06  7.7e-16  bluestein
      1165  5.233            prime    replayed       10481      11583    1.07  6.7e-16  bluestein
      1167  3.389            prime    replayed       10660      11491    1.07  8.3e-16  bluestein
      1169  7.167            prime    replayed       10676      11599    1.08  7.7e-16  bluestein
      1170  2.3^2.5.13       chain3   replayed        1471       1873    1.27  7.7e-16  
      1171  1171             prime    replayed        4820      11552    2.39  8.5e-16  rader
      1173  3.17.23          flat     replayed        2380       5698    2.22  1.1e-15  
      1175  5^2.47           flat     replayed        2753       8817    3.17  4.6e-16  
      1176  2^3.3.7^2        chain3   replayed        1500       1691    1.10  4.8e-16  
      1177  11.107           prime    replayed       10531      11595    1.08  7.0e-16  bluestein
      1178  2.19.31          flat     replayed        3079       6106    1.98  1.0e-15  
      1179  3^2.131          prime    replayed       10465      11523    1.07  6.0e-16  bluestein
      1181  1181             prime    replayed       10679      11559    1.08  6.9e-16  bluestein
      1183  7.13^2           chain3   replayed        1809       1926    1.06  8.3e-16  
      1184  2^5.37           2p       replayed        3623       5511    1.32  4.6e-16  
      1185  3.5.79           prime    replayed       10640      16963    1.58  7.9e-16  bluestein
      1187  1187             prime    replayed       10489      11525    1.08  9.2e-16  bluestein
      1188  2^2.3^3.11       chain3   replayed        1383       1942    1.40  5.5e-16  
      1189  29.41            2p       replayed        3463      11201    3.15  5.9e-16  
      1190  2.5.7.17         chain3   replayed        1678       3384    1.98  8.2e-16  
      1191  3.397            prime    replayed       10523      11540    1.08  9.2e-16  bluestein
      1193  1193             prime    replayed       10455      11587    1.09  8.6e-16  bluestein
      1195  5.239            prime    replayed       10700      11540    1.07  9.2e-16  bluestein
      1196  2^2.13.23        chain3   replayed        2077       3907    1.87  6.3e-16  
      1197  3^2.7.19         flat     replayed        2225       3652    1.50  1.3e-15  
      1199  11.109           prime    replayed       10499      11523    1.08  8.4e-16  bluestein
      1201  1201             prime    replayed        6843      11621    1.49  9.1e-16  rader
      1203  3.401            prime    replayed       10496      11567    1.08  6.4e-16  bluestein
      1204  2^2.7.43         flat     replayed        2517       6605    2.57  6.5e-16  
      1205  5.241            prime    replayed       10576      11571    1.09  7.1e-16  bluestein
      1207  17.71            prime    replayed       10495      16663    1.56  7.8e-16  bluestein
      1209  3.13.31          chain3   replayed        3105       5677    1.81  6.0e-16  
      1210  2.5.11^2         chain3   replayed        1608       2328    1.42  3.7e-16  
      1211  7.173            prime    replayed       10837      11547    1.06  8.9e-16  bluestein
      1213  1213             prime    replayed       10693      11649    1.08  7.9e-16  bluestein
      1215  3^5.5            chain3   replayed        1737       2114    1.18  8.5e-16  
      1216  2^6.19           2p       replayed        1529       3174    1.81  4.3e-16  
      1217  1217             prime    replayed        5158      11646    2.25  8.8e-16  rader
      1218  2.3.7.29         flat     replayed        3063       4626    1.49  6.3e-16  
      1219  23.53            prime    replayed       10541      13266    1.23  9.2e-16  bluestein
      1221  3.11.37          chain3   replayed        2864       6878    2.35  4.9e-16  
      1222  2.13.47          flat     replayed        3892       8001    1.97  9.4e-16  
      1223  1223             prime    replayed       10498      11577    1.08  9.0e-16  bluestein
      1224  2^3.3^2.17       chain3   replayed        1514       3266    2.13  7.6e-16  
      1225  5^2.7^2          flat     replayed        1784       1928    1.07  6.7e-16  
      1227  3.409            prime    replayed       10506      11558    1.08  8.8e-16  bluestein
      1229  1229             prime    replayed       10487      11588    1.10  7.5e-16  bluestein
      1230  2.3.5.41         chain3   replayed        2637       6337    2.33  5.8e-16  
      1231  1231             prime    replayed       10492      11567    1.10  9.7e-16  bluestein
      1232  2^4.7.11         chain3   replayed        1341       1727    1.28  5.1e-16  
      1233  3^2.137          prime    replayed       10563      11643    1.08  7.9e-16  bluestein
      1235  5.13.19          flat     replayed        2183       3768    1.68  7.9e-16  
      1237  1237             prime    replayed       10629      11628    1.09  7.6e-16  bluestein
      1239  3.7.59           prime    replayed       10745      12298    1.14  7.0e-16  bluestein
      1240  2^3.5.31         chain3   replayed        3116       4954    1.10  6.5e-16  flips differ 1.44x
      1241  17.73            prime    replayed       10471      17804    1.70  6.3e-16  bluestein
      1242  2.3^3.23         flat     replayed        2728       4050    1.43  8.1e-16  
      1243  11.113           prime    replayed       10512      11558    1.07  7.6e-16  bluestein
      1245  3.5.83           prime    replayed       10533      19223    1.78  6.4e-16  bluestein
      1247  29.43            flat     replayed        3872      12134    2.96  2.6e-15  
      1248  2^5.3.13         chain3   replayed        1340       1962    1.46  5.5e-16  
      1249  1249             prime    replayed        5353      11686    2.15  9.5e-16  rader
      1251  3^2.139          prime    replayed       10653      11611    1.09  9.5e-16  bluestein
      1253  7.179            prime    replayed       10664      11632    1.08  1.0e-15  bluestein
      1254  2.3.11.19        chain3   replayed        2243       3561    1.35  7.3e-16  
      1255  5.251            prime    replayed       10534      11590    1.08  8.2e-16  bluestein
      1257  3.419            prime    replayed       10741      11677    1.08  8.3e-16  bluestein
      1258  2.17.37          flat     replayed        3464       7181    2.06  1.2e-15  
      1259  1259             prime    replayed       10733      11601    1.08  7.2e-16  bluestein
      1260  2^2.3^2.5.7      chain3   replayed        1411       1896    1.34  5.8e-16  
      1261  13.97            prime    replayed       10722      11657    1.08  7.8e-16  bluestein
      1263  3.421            prime    replayed       10528      11614    1.08  8.4e-16  bluestein
      1265  5.11.23          flat     replayed        2514       4445    1.62  9.5e-16  
      1267  7.181            prime    replayed       10661      11621    1.07  9.6e-16  bluestein
      1269  3^3.47           flat     replayed        3450       9661    2.60  8.5e-16  
      1271  31.41            flat     replayed        3700      12339    3.20  1.3e-15  
      1273  19.67            prime    replayed       10644      16810    1.56  7.4e-16  bluestein
      1274  2.7^2.13         flat     replayed        2360       2317    0.97  7.8e-16  
      1275  3.5^2.17         flat     replayed        2239       3695    1.55  7.2e-16  
      1276  2^2.11.29        chain3   replayed        2798       4918    1.74  4.5e-16  
      1277  1277             prime    replayed        7960      11670    1.27  7.8e-16  rader
      1279  1279             prime    replayed       10520      11669    1.11  7.6e-16  bluestein
      1281  3.7.61           prime    replayed       10523      13495    1.26  8.9e-16  bluestein
      1283  1283             prime    replayed       10530      13529    1.27  1.5e-15  bluestein
      1285  5.257            prime    replayed       10749      13575    1.26  9.8e-16  bluestein
      1287  3^2.11.13        chain3   replayed        1955       2150    1.07  6.7e-16  
      1288  2^3.7.23         chain3   replayed        2158       4104    1.75  5.5e-16  
      1289  1289             prime    replayed        6868      13567    1.88  8.3e-16  rader
      1290  2.3.5.43         flat     replayed        3932       6993    1.75  8.6e-16  
      1291  1291             prime    replayed       10532      13473    1.26  8.2e-16  bluestein
      1292  2^2.17.19        chain3   replayed        2205       4906    2.17  6.4e-16  
      1293  3.431            prime    replayed       10567      13556    1.28  8.2e-16  bluestein
      1295  5.7.37           flat     replayed        2619       7303    2.78  8.6e-16  
      1297  1297             prime    replayed        8752      13599    1.50  9.4e-16  rader
      1299  3.433            prime    replayed       10542      13486    1.26  8.3e-16  bluestein
      1300  2^2.5^2.13       flat     replayed        1748       2056    1.17  5.0e-16  
      1301  1301             prime    replayed        7300      13574    1.61  1.0e-15  rader
      1302  2.3.7.31         chain3   replayed        3719       5195    1.17  6.9e-16  
      1303  1303             prime    replayed       10522      13523    1.26  8.2e-16  bluestein
      1305  3^2.5.29         chain3   replayed        2892       5780    1.99  6.7e-16  
      1307  1307             prime    replayed       10702      13498    1.26  7.7e-16  bluestein
      1309  7.11.17          flat     replayed        2402       3700    1.47  1.1e-15  
      1311  3.19.23          chain3   replayed        2818       6773    1.93  5.4e-16  
      1312  2^5.41           2p       replayed        2203       6748    3.06  5.8e-16  
      1313  13.101           prime    replayed       10739      13578    1.25  9.1e-16  bluestein
      1315  5.263            prime    replayed       10517      13496    1.26  8.1e-16  bluestein
      1316  2^2.7.47         chain3   replayed        4684       8098    1.30  6.0e-16  flips differ 1.33x
      1317  3.439            prime    replayed       10714      13597    1.26  9.0e-16  bluestein
      1319  1319             prime    replayed       10671      13503    1.26  7.4e-16  bluestein
      1320  2^3.3.5.11       chain3   replayed        1552       2121    1.37  6.1e-16  
      1321  1321             prime    replayed        5671      13584    2.36  7.6e-16  rader
      1323  3^3.7^2          flat     replayed        2133       2185    1.02  9.2e-16  
      1325  5^2.53           prime    replayed       10540      11731    1.09  6.2e-16  bluestein
      1326  2.3.13.17        chain3   replayed        1990       3599    1.78  7.6e-16  
      1327  1327             prime    replayed        6191      13512    2.18  1.1e-15  rader
      1329  3.443            prime    replayed       10488      13629    1.28  9.2e-16  bluestein
      1330  2.5.7.19         chain3   replayed        1976       4006    2.02  6.7e-16  
      1331  11^3             chain3   replayed        2054       2080    1.01  4.5e-16  
      1332  2^2.3^2.37       chain3   replayed        2541       6352    2.43  5.1e-16  
      1333  31.43            2p       replayed        4329      13313    2.45  7.7e-16  flips differ 1.26x
      1334  2.23.29          flat     replayed        3692       7176    1.57  1.1e-15  
      1335  3.5.89           prime    replayed       10650      23256    2.17  9.2e-16  bluestein
      1337  7.191            prime    replayed       10668      13603    1.27  7.4e-16  bluestein
      1339  13.103           prime    replayed       10565      13527    1.26  1.0e-15  bluestein
      1341  3^2.149          prime    replayed       10510      13623    1.26  8.4e-16  bluestein
      1343  17.79            prime    replayed       10548      20936    1.95  8.8e-16  bluestein
      1344  2^6.3.7          2p       replayed        1433       1868    1.25  5.3e-16  
      1345  5.269            prime    replayed       10524      13616    1.29  6.8e-16  bluestein
      1347  3.449            prime    replayed       10542      13543    1.25  8.4e-16  bluestein
      1349  19.71            prime    replayed       10525      18929    1.77  9.9e-16  bluestein
      1351  7.193            prime    replayed       10735      13535    1.26  8.1e-16  bluestein
      1352  2^3.13^2         chain3   replayed        1785       2177    1.21  7.1e-16  
      1353  3.11.41          chain3   replayed        3537       8721    2.43  5.3e-16  
      1355  5.271            prime    replayed       10522      13558    1.26  9.7e-16  bluestein
      1357  23.59            prime    replayed       11080      16236    1.45  6.4e-16  bluestein
      1359  3^2.151          prime    replayed       10950      13556    1.24  9.0e-16  bluestein
      1360  2^4.5.17         chain3   replayed        1741       3425    1.96  8.0e-16  
      1361  1361             prime    replayed        6065      13615    2.21  9.6e-16  rader
      1363  29.47            2p       replayed        5470      14277    2.19  5.3e-16  
      1364  2^2.11.31        flat     replayed        3572       5496    1.44  7.3e-16  
      1365  3.5.7.13         chain3   replayed        2250       2265    1.00  6.9e-16  
      1367  1367             prime    replayed       10538      13573    1.26  7.8e-16  bluestein
      1368  2^3.3^2.19       chain3   replayed        1828       3888    2.12  5.1e-16  
      1369  37^2             2p       replayed        4369      13552    2.37  5.1e-16  flips differ 1.31x
      1371  3.457            prime    replayed       10546      13593    1.29  8.2e-16  bluestein
      1372  2^2.7^3          flat     replayed        1910       2208    1.15  6.3e-16  
      1373  1373             prime    replayed       10577      13655    1.26  9.2e-16  bluestein
      1375  5^3.11           chain3   replayed        2288       2250    0.98  5.5e-16  
      1376  2^5.43           2p       replayed        2524       7422    2.79  5.4e-16  
      1377  3^4.17           chain3   replayed        2769       4106    1.48  9.2e-16  
      1379  7.197            prime    replayed       10538      13579    1.27  1.0e-15  bluestein
      1380  2^2.3.5.23       chain3   replayed        2075       4495    2.09  5.7e-16  
      1381  1381             prime    replayed        8571      13653    1.53  9.6e-16  rader
      1383  3.461            prime    replayed       10770      13580    1.26  6.9e-16  bluestein
      1385  5.277            prime    replayed       10430      13659    1.27  8.9e-16  bluestein
      1386  2.3^2.7.11       chain3   replayed        2183       2294    0.99  7.4e-16  
      1387  19.73            prime    replayed       10728      20211    1.88  8.7e-16  bluestein
      1389  3.463            prime    replayed       10493      13652    1.30  8.8e-16  bluestein
      1391  13.107           prime    replayed       10543      13595    1.26  6.7e-16  bluestein
      1392  2^4.3.29         flat     replayed        2472       5195    2.10  5.5e-16  
      1393  7.199            prime    replayed       10751      13665    1.27  8.6e-16  bluestein
      1394  2.17.41          flat     replayed        3924       8677    2.18  1.2e-15  
      1395  3^2.5.31         flat     replayed        2845       6525    1.98  8.5e-16  
      1397  11.127           prime    replayed       10690      13664    1.27  7.8e-16  bluestein
      1399  1399             prime    replayed       10548      13597    1.27  1.1e-15  bluestein
      1400  2^3.5^2.7        chain3   replayed        1849       2058    1.07  5.5e-16  
      1401  3.467            prime    replayed       10687      13657    1.27  8.8e-16  bluestein
      1403  23.61            prime    replayed       10729      17625    1.64  6.5e-16  bluestein
      1404  2^2.3^3.13       chain3   replayed        1711       2339    1.36  6.9e-16  
      1405  5.281            prime    replayed       10520      13666    1.28  1.2e-15  bluestein
      1406  2.19.37          flat     replayed        3903       8223    1.87  1.3e-15  
      1407  3.7.67           prime    replayed       10565      16396    1.55  8.3e-16  bluestein
      1408  2^7.11           chain3   replayed        1528       2149    1.41  4.2e-16  
      1409  1409             prime    replayed        5365      13670    2.53  7.7e-16  rader
      1410  2.3.5.47         flat     replayed        4546       8559    1.88  7.0e-16  
      1411  17.83            prime    replayed       10752      23575    2.19  6.9e-16  bluestein
      1413  3^2.157          prime    replayed       10686      13688    1.27  9.0e-16  bluestein
      1415  5.283            prime    replayed       10708      13617    1.27  1.0e-15  bluestein
      1417  13.109           prime    replayed       10603      13700    1.26  1.1e-15  bluestein
      1419  3.11.43          flat     replayed        3639       9623    2.58  6.0e-16  
      1421  7^2.29           chain3   replayed        3401       6141    1.78  6.6e-16  
      1423  1423             prime    replayed       10781      13622    1.26  9.6e-16  bluestein
      1425  3.5^2.19         flat     replayed        2219       4343    1.93  5.9e-16  
      1426  2.23.31          flat     replayed        5262       7931    1.44  1.8e-15  
      1427  1427             prime    replayed       10778      13607    1.26  8.4e-16  bluestein
      1428  2^2.3.7.17       chain3   replayed        1766       3933    2.08  6.4e-16  
      1429  1429             prime    replayed        8005      13686    1.69  1.3e-15  rader
      1430  2.5.11.13        chain3   replayed        1951       2795    1.43  6.0e-16  
      1431  3^3.53           prime    replayed       10881      12825    1.18  7.5e-16  bluestein
      1433  1433             prime    replayed       10786      13703    1.26  8.9e-16  bluestein
      1435  5.7.41           chain3   replayed        4177       9267    2.19  5.7e-16  
      1437  3.479            prime    replayed       10574      13709    1.27  1.0e-15  bluestein
      1439  1439             prime    replayed       10608      13669    1.26  9.0e-16  bluestein
      1441  11.131           prime    replayed       10600      13713    1.27  7.9e-16  bluestein
      1443  3.13.37          flat     replayed        3438       8310    2.22  8.7e-16  
      1444  2^2.19^2         chain3   replayed        2475       5672    2.28  5.1e-16  
      1445  5.17^2           chain3   replayed        2784       6091    2.16  6.9e-16  
      1447  1447             prime    replayed       10586      13642    1.28  1.2e-15  bluestein
      1449  3^2.7.23         flat     replayed        2595       5238    1.95  5.8e-16  
      1450  2.5^2.29         chain3   replayed        2829       5573    1.72  5.1e-16  
      1451  1451             prime    replayed       10800      13637    1.26  6.8e-16  bluestein
      1452  2^2.3.11^2       chain3   replayed        1729       2491    1.43  4.8e-16  
      1453  1453             prime    replayed        5879      13732    2.33  1.1e-15  rader
      1455  3.5.97           prime    replayed       10588      13649    1.27  8.4e-16  bluestein
      1456  2^4.7.13         chain3   replayed        1650       2081    1.24  6.8e-16  
      1457  31.47            2p       replayed        6145      15677    2.52  6.6e-16  
      1459  1459             prime    replayed        9318      13636    1.33  1.2e-15  rader
      1461  3.487            prime    replayed       10839      13722    1.26  8.0e-16  bluestein
      1462  2.17.43          flat     replayed        4680       9490    2.02  1.0e-15  
      1463  7.11.19          chain3   replayed        2516       4376    1.45  5.2e-16  
      1465  5.293            prime    replayed       10549      13734    1.30  8.5e-16  bluestein
      1467  3^2.163          prime    replayed       10783      13646    1.22  1.1e-15  bluestein
      1469  13.113           prime    replayed       10820      13733    1.26  1.0e-15  bluestein
      1470  2.3.5.7^2        chain3   replayed        2225       2247    0.97  5.9e-16  
      1471  1471             prime    replayed        7725      13648    1.76  1.2e-15  rader
      1472  2^6.23           2p       replayed        2582       4473    1.48  4.9e-16  
      1473  3.491            prime    replayed       10589      13741    1.27  7.5e-16  bluestein
      1475  5^2.59           prime    replayed       10808      14677    1.35  9.0e-16  bluestein
      1476  2^2.3^2.41       chain3   replayed        3102       7767    2.50  5.0e-16  
      1477  7.211            prime    replayed       10695      13721    1.27  8.8e-16  bluestein
      1479  3.17.29          flat     replayed        3675       8467    2.27  1.2e-15  
      1480  2^3.5.37         flat     replayed        3154       6971    2.15  6.6e-16  
      1481  1481             prime    replayed       10573      13722    1.27  9.7e-16  bluestein
      1482  2.3.13.19        chain3   replayed        2310       4271    1.83  7.4e-16  
      1483  1483             prime    replayed        7384      13659    1.75  1.0e-15  rader
      1485  3^3.5.11         chain3   replayed        2210       2542    1.14  6.5e-16  
      1487  1487             prime    replayed       10797      13662    1.26  1.1e-15  bluestein
      1488  2^4.3.31         chain3   replayed        3296       5813    1.61  5.5e-16  
      1489  1489             prime    replayed       10784      13746    1.27  9.4e-16  bluestein
      1491  3.7.71           prime    replayed       10746      18628    1.73  8.1e-16  bluestein
      1493  1493             prime    replayed       10869      13748    1.26  7.8e-16  bluestein
      1495  5.13.23          chain3   replayed        3026       5393    1.56  6.1e-16  
      1496  2^3.11.17        chain3   replayed        2071       3998    1.92  5.5e-16  
      1497  3.499            prime    replayed       10747      13755    1.28  1.2e-15  bluestein
      1499  1499             prime    replayed       10457      13677    1.29  1.1e-15  bluestein
      1501  19.79            prime    replayed       10585      23766    2.24  7.3e-16  bluestein
      1503  3^2.167          prime    replayed       10564      13670    1.28  9.5e-16  bluestein
      1504  2^5.47           2p       replayed        2809       9112    3.17  4.9e-16  
      1505  5.7.43           chain3   replayed        5542      10186    1.42  6.2e-16  flips differ 1.29x
      1507  11.137           prime    replayed       10831      13701    1.26  7.9e-16  bluestein
      1508  2^2.13.29        flat     replayed        3677       5823    1.57  9.1e-16  
      1509  3.503            prime    replayed       10621      13758    1.27  7.8e-16  bluestein
      1511  1511             prime    replayed       10628      13676    1.29  9.4e-16  bluestein
      1512  2^3.3^3.7        chain3   replayed        1888       2331    1.23  6.7e-16  
      1513  17.89            prime    replayed       10828      28337    2.61  6.8e-16  bluestein
      1515  3.5.101          prime    replayed       10855      13679    1.26  8.2e-16  bluestein
      1517  37.41            2p       replayed        5117      16319    2.48  8.5e-16  flips differ 1.29x
      1518  2.3.11.23        chain3   replayed        2736       4938    1.78  5.2e-16  
      1519  7^2.31           flat     replayed        4088       6933    1.68  7.5e-16  
      1520  2^4.5.19         chain3   replayed        2084       4086    1.95  5.1e-16  
      1521  3^2.13^2         chain3   replayed        2332       2823    1.18  7.1e-16  
      1523  1523             prime    replayed       10837      13714    1.26  7.1e-16  bluestein
      1525  5^2.61           prime    replayed       10613      16076    1.48  6.0e-16  bluestein
      1527  3.509            prime    replayed       10626      13708    1.26  9.1e-16  bluestein
      1529  11.139           prime    replayed       10842      13765    1.27  6.9e-16  bluestein
      1530  2.3^2.5.17       chain3   replayed        2051       4095    2.00  6.7e-16  
      1531  1531             prime    replayed        6922      13722    1.89  1.1e-15  rader
      1533  3.7.73           prime    replayed       10834      19998    1.84  6.9e-16  bluestein
      1535  5.307            prime    replayed       10679      13739    1.28  1.1e-15  bluestein
      1537  29.53            prime    replayed       10602      18178    1.68  8.1e-16  bluestein
      1539  3^4.19           chain3   replayed        3222       4954    1.53  7.1e-16  
      1540  2^2.5.7.11       flat     replayed        2119       2576    1.18  7.3e-16  
      1541  23.67            prime    replayed       10604      21152    1.96  9.9e-16  bluestein
      1543  1543             prime    replayed       10690      15534    1.42  6.1e-16  bluestein
      1545  3.5.103          prime    replayed       10583      15619    1.47  9.7e-16  bluestein
      1547  7.13.17          flat     replayed        2636       4524    1.68  9.1e-16  
      1548  2^2.3^2.43       chain3   replayed        3217       8494    2.54  6.3e-16  
      1549  1549             prime    replayed       10856      15593    1.44  7.8e-16  bluestein
      1550  2.5^2.31         chain3   replayed        3843       6246    1.47  7.5e-16  
      1551  3.11.47          chain3   replayed        5766      11695    2.03  6.7e-16  
      1553  1553             prime    replayed       10786      15638    1.43  7.5e-16  bluestein
      1554  2.3.7.37         flat     replayed        4547       7260    1.58  8.7e-16  
      1555  5.311            prime    replayed       10826      15539    1.43  1.3e-15  bluestein
      1557  3^2.173          prime    replayed       10843      15655    1.44  8.0e-16  bluestein
      1558  2.19.41          flat     replayed        4743       9909    2.08  1.3e-15  
      1559  1559             prime    replayed       10632      15559    1.44  9.0e-16  bluestein
      1560  2^3.3.5.13       chain3   replayed        1818       2545    1.40  6.9e-16  
      1561  7.223            prime    replayed       10603      15656    1.47  8.6e-16  bluestein
      1563  3.521            prime    replayed       10840      15557    1.43  8.7e-16  bluestein
      1564  2^2.17.23        chain3   replayed        2931       6622    2.25  6.6e-16  
      1565  5.313            prime    replayed       10630      15670    1.47  8.5e-16  bluestein
      1566  2.3^3.29         chain3   replayed        3533       6038    1.28  6.0e-16  flips differ 1.34x
      1567  1567             prime    replayed       10643      15573    1.44  8.8e-16  bluestein
      1568  2^5.7^2          chain3   replayed        1731       2335    1.33  5.1e-16  
      1569  3.523            prime    replayed       10878      15664    1.44  7.1e-16  bluestein
      1571  1571             prime    replayed       10639      15570    1.44  9.0e-16  bluestein
      1573  11^2.13          chain3   replayed        2437       2528    1.00  5.8e-16  
      1575  3^2.5^2.7        flat     replayed        2404       2761    1.15  8.2e-16  
      1577  19.83            prime    replayed       10639      26807    2.47  8.7e-16  bluestein
      1579  1579             prime    replayed       10646      15584    1.43  6.7e-16  bluestein
      1581  3.17.31          chain3   replayed        4900       9440    1.59  6.4e-16  
      1583  1583             prime    replayed       10884      15590    1.43  7.7e-16  bluestein
      1584  2^4.3^2.11       chain3   replayed        1802       2440    1.35  5.4e-16  
      1585  5.317            prime    replayed       10839      15672    1.44  7.7e-16  bluestein
      1587  3.23^2           flat     replayed        4849       8935    1.79  1.2e-15  
      1589  7.227            prime    replayed       10853      15707    1.45  9.1e-16  bluestein
      1591  37.43            2p       replayed        5955      17572    2.35  5.8e-16  flips differ 1.25x
      1593  3^3.59           prime    replayed       10856      16058    1.48  6.8e-16  bluestein
      1595  5.11.29          flat     replayed        4104       6981    1.60  1.0e-15  
      1596  2^2.3.7.19       chain3   replayed        2143       4663    2.16  5.5e-16  
      1597  1597             prime    replayed       10870      15690    1.44  8.9e-16  bluestein
      1598  2.17.47          flat     replayed        5362      11457    2.11  1.7e-15  
      1599  3.13.41          chain3   replayed        4069      10487    2.52  6.1e-16  
      1601  1601             prime    replayed        6331      12592    1.96  7.7e-16  rader
      1603  7.229            prime    replayed       10658      12398    1.16  7.7e-16  bluestein
      1605  3.5.107          prime    replayed       10881      12638    1.16  8.2e-16  bluestein
      1607  1607             prime    replayed       10694      12494    1.15  1.0e-15  bluestein
      1609  1609             prime    replayed       10643      12497    1.15  8.8e-16  bluestein
      1610  2.5.7.23         chain3   replayed        3251       5533    1.70  6.2e-16  
      1611  3^2.179          prime    replayed       10598      12403    1.17  9.0e-16  bluestein
      1612  2^2.13.31        flat     replayed        4764       6509    1.14  1.0e-15  
      1613  1613             prime    replayed       10900      12582    1.15  8.3e-16  bluestein
      1615  5.17.19          chain3   replayed        3525       7055    2.00  6.1e-16  
      1617  3.7^2.11         chain3   replayed        2762       2781    0.99  5.7e-16  
      1619  1619             prime    replayed       10797      12395    1.14  7.5e-16  bluestein
      1621  1621             prime    replayed       10667      12584    1.18  9.7e-16  bluestein
      1623  3.541            prime    replayed       10878      12450    1.14  7.1e-16  bluestein
      1624  2^3.7.29         flat     replayed        3660       6172    1.62  7.4e-16  
      1625  5^3.13           flat     replayed        2458       2770    1.11  7.6e-16  
      1627  1627             prime    replayed       10847      12440    1.15  6.8e-16  bluestein
      1628  2^2.11.37        chain3   replayed        4727       7699    1.60  6.3e-16  
      1629  3^2.181          prime    replayed       10824      12669    1.17  7.9e-16  bluestein
      1631  7.233            prime    replayed       10601      12419    1.14  1.1e-15  bluestein
      1632  2^5.3.17         chain3   replayed        1990       4346    2.16  5.2e-16  
      1633  23.71            prime    replayed       10655      23768    2.23  7.6e-16  bluestein
      1634  2.19.43          flat     replayed        4926      10861    2.01  1.0e-15  
      1635  3.5.109          prime    replayed       10664      12459    1.13  7.9e-16  bluestein
      1637  1637             prime    replayed       10822      12550    1.15  6.5e-16  bluestein
      1638  2.3^2.7.13       chain3   replayed        2752       2760    0.98  7.0e-16  
      1639  11.149           prime    replayed       10818      12454    1.15  9.9e-16  bluestein
      1640  2^3.5.41         flat     replayed        3565       8541    2.30  6.5e-16  
      1641  3.547            prime    replayed       10691      12545    1.15  7.4e-16  bluestein
      1643  31.53            prime    replayed       10885      19925    1.83  8.7e-16  bluestein
      1645  5.7.47           flat     replayed        4153      12411    2.90  5.6e-16  
      1647  3^3.61           prime    replayed       10738      17551    1.61  8.7e-16  bluestein
      1649  17.97            prime    replayed       10668      12625    1.18  1.0e-15  bluestein
      1650  2.3.5^2.11       chain3   replayed        2170       2633    1.20  5.7e-16  
      1651  13.127           prime    replayed       10676      12496    1.15  7.2e-16  bluestein
      1653  3.19.29          chain3   replayed        4154       9968    2.39  6.9e-16  
      1655  5.331            prime    replayed       10702      12516    1.14  8.6e-16  bluestein
      1656  2^3.3^2.23       chain3   replayed        2716       5415    1.88  6.1e-16  
      1657  1657             prime    replayed       10669      12653    1.15  1.1e-15  bluestein
      1659  3.7.79           prime    replayed       10868      23725    2.17  8.7e-16  bluestein
      1661  11.151           prime    replayed       10921      12569    1.15  7.0e-16  bluestein
      1663  1663             prime    replayed       10670      12464    1.14  8.0e-16  bluestein
      1664  2^7.13           chain3   replayed        1887       2584    1.36  6.3e-16  
      1665  3^2.5.37         chain3   replayed        4901       9620    1.37  6.5e-16  flips differ 1.43x
      1666  2.7^2.17         flat     replayed        3470       4821    1.38  7.7e-16  
      1667  1667             prime    replayed       10830      12638    1.16  9.3e-16  bluestein
      1669  1669             prime    replayed       10698      12628    1.18  9.0e-16  bluestein
      1671  3.557            prime    replayed       10704      12522    1.14  9.5e-16  bluestein
      1672  2^3.11.19        chain3   replayed        2418       4863    1.96  6.3e-16  
      1673  7.239            prime    replayed       10785      12606    1.15  7.8e-16  bluestein
      1674  2.3^3.31         flat     replayed        4314       6760    1.56  8.3e-16  
      1675  5^2.67           prime    replayed       10892      19593    1.80  9.7e-16  bluestein
      1677  3.13.43          chain3   replayed        4479      11588    2.57  7.2e-16  
      1679  23.73            prime    replayed       10833      25343    2.31  7.8e-16  bluestein
      1680  2^4.3.5.7        chain3   replayed        1912       2380    1.24  6.7e-16  
      1681  41^2             2p       replayed        5932      19561    2.53  4.9e-16  flips differ 1.30x
      1682  2.29^2           flat     replayed        5220      10038    1.50  1.7e-15  flips differ 1.28x
      1683  3^2.11.17        chain3   replayed        2752       5027    1.81  8.3e-16  
      1685  5.337            prime    replayed       10912      12678    1.16  9.7e-16  bluestein
      1687  7.241            prime    replayed       10845      12630    1.15  8.9e-16  bluestein
      1689  3.563            prime    replayed       10917      12592    1.15  7.4e-16  bluestein
      1690  2.5.13^2         chain3   replayed        2304       3488    1.48  7.4e-16  
      1691  19.89            prime    replayed       10705      32135    2.95  8.3e-16  bluestein
      1692  2^2.3^2.47       chain3   replayed        3711      10473    2.79  6.1e-16  
      1693  1693             prime    replayed       10883      12640    1.16  1.2e-15  bluestein
      1694  2.7.11^2         flat     replayed        3638       3320    0.89  7.9e-16  
      1695  3.5.113          prime    replayed       10886      12526    1.15  9.9e-16  bluestein
      1697  1697             prime    replayed       10870      12675    1.16  7.3e-16  bluestein
      1699  1699             prime    replayed       10580      12500    1.14  9.5e-16  bluestein
      1700  2^2.5^2.17       chain3   replayed        2380       4540    1.90  6.5e-16  
      1701  3^5.7            chain3   replayed        2553       3186    1.23  6.0e-16  
      1702  2.23.37          flat     replayed        5326      10600    1.90  1.6e-15  
      1703  13.131           prime    replayed       10669      12526    1.17  7.7e-16  bluestein
      1705  5.11.31          chain3   replayed        4937       7895    1.26  5.4e-16  flips differ 1.27x
      1707  3.569            prime    replayed       10604      12637    1.16  9.7e-16  bluestein
      1709  1709             prime    replayed       10771      12661    1.15  9.4e-16  bluestein
      1710  2.3^2.5.19       chain3   replayed        3143       4855    1.35  6.0e-16  
      1711  29.59            prime    replayed       10724      22134    2.02  8.5e-16  bluestein
      1713  3.571            prime    replayed       10868      12635    1.15  7.1e-16  bluestein
      1715  5.7^3            flat     replayed        2755       2851    0.99  7.0e-16  
      1716  2^2.3.11.13      chain3   replayed        2157       3042    1.40  7.5e-16  
      1717  17.101           prime    replayed       10935      12688    1.15  1.2e-15  bluestein
      1719  3^2.191          prime    replayed       10714      12581    1.14  8.5e-16  bluestein
      1720  2^3.5.43         chain3   replayed        3525       9339    2.59  5.4e-16  
      1721  1721             prime    replayed       10880      12647    1.16  8.1e-16  bluestein
      1722  2.3.7.41         chain3   replayed        3808       8910    2.31  5.4e-16  
      1723  1723             prime    replayed       10991      12537    1.14  8.3e-16  bluestein
      1725  3.5^2.23         chain3   replayed        3475       6294    1.81  6.2e-16  
      1727  11.157           prime    replayed       10705      12528    1.17  9.2e-16  bluestein
      1729  7.13.19          chain3   replayed        3237       5354    1.64  6.4e-16  
      1731  3.577            prime    replayed       10705      12532    1.16  8.8e-16  bluestein
      1733  1733             prime    replayed       10694      12667    1.16  9.9e-16  bluestein
      1734  2.3.17^2         chain3   replayed        3040       6274    1.95  4.7e-16  
      1735  5.347            prime    replayed       10718      12548    1.17  6.6e-16  bluestein
      1736  2^3.7.31         flat     replayed        4198       6912    1.64  8.6e-16  
      1737  3^2.193          prime    replayed       10765      12665    1.15  9.2e-16  bluestein
      1739  37.47            flat     replayed        5672      20560    3.39  1.9e-15  
      1740  2^2.3.5.29       flat     replayed        3374       6694    1.69  7.9e-16  
      1741  1741             prime    replayed       10780      12654    1.14  8.4e-16  bluestein
      1743  3.7.83           prime    replayed       10687      26923    2.45  6.1e-16  bluestein
      1745  5.349            prime    replayed       10704      12727    1.19  8.6e-16  bluestein
      1747  1747             prime    replayed       10984      12587    1.14  9.2e-16  bluestein
      1748  2^2.19.23        chain3   replayed        3333       7620    2.04  4.9e-16  
      1749  3.11.53          prime    replayed       10678      15565    1.42  7.1e-16  bluestein
      1750  2.5^3.7          chain3   replayed        3235       2850    0.88  6.1e-16  
      1751  17.103           prime    replayed       10769      12567    1.16  8.4e-16  bluestein
      1753  1753             prime    replayed       10929      12674    1.15  6.1e-16  bluestein
      1755  3^3.5.13         chain3   replayed        2753       3262    1.15  1.1e-15  
      1757  7.251            prime    replayed       10856      12686    1.16  9.8e-16  bluestein
      1759  1759             prime    replayed       10905      12555    1.15  9.2e-16  bluestein
      1760  2^5.5.11         chain3   replayed        1989       2842    1.42  4.7e-16  
      1761  3.587            prime    replayed       10706      12624    1.16  1.3e-15  bluestein
      1763  41.43            2p       replayed        6771      21081    3.08  6.6e-16  
      1764  2^2.3^2.7^2      chain3   replayed        2230       2930    1.31  7.3e-16  
      1765  5.353            prime    replayed       10661      12728    1.19  7.8e-16  bluestein
      1767  3.19.31          chain3   replayed        4533      11106    2.01  4.8e-16  
      1768  2^3.13.17        chain3   replayed        2548       4809    1.88  1.0e-15  
      1769  29.61            prime    replayed       10729      23950    2.23  1.1e-15  bluestein
      1771  7.11.23          flat     replayed        3907       6278    1.53  9.6e-16  
      1773  3^2.197          prime    replayed       10756      12652    1.15  9.0e-16  bluestein
      1775  5^2.71           prime    replayed       10801      22229    2.02  7.0e-16  bluestein
      1776  2^4.3.37         chain3   replayed        3263       8202    2.46  5.6e-16  
      1777  1777             prime    replayed       10985      12645    1.15  1.2e-15  bluestein
      1779  3.593            prime    replayed       10722      12583    1.15  6.1e-16  bluestein
      1781  13.137           prime    replayed       10735      12731    1.18  7.4e-16  bluestein
      1782  2.3^4.11         chain3   replayed        2952       3047    0.94  7.5e-16  
      1783  1783             prime    replayed       10936      12587    1.15  7.2e-16  bluestein
      1785  3.5.7.17         chain3   replayed        3003       5343    1.76  1.0e-15  
      1786  2.19.47          flat     replayed        6195      13092    2.00  1.2e-15  
      1787  1787             prime    replayed       10888      12639    1.15  8.5e-16  bluestein
      1789  1789             prime    replayed       10958      12737    1.16  9.7e-16  bluestein
      1791  3^2.199          prime    replayed       10829      12636    1.15  7.8e-16  bluestein
      1792  2^8.7            chain3   replayed        2035       2442    1.20  4.4e-16  
      1793  11.163           prime    replayed       10975      12703    1.16  5.6e-16  bluestein
      1794  2.3.13.23        chain3   replayed        3811       5932    1.40  7.3e-16  
      1795  5.359            prime    replayed       10943      12545    1.14  9.2e-16  bluestein
      1797  3.599            prime    replayed       10717      12754    1.18  8.1e-16  bluestein
      1798  2.29.31          flat     replayed        7224      11065    1.44  2.0e-15  
      1799  7.257            prime    replayed       10963      12581    1.14  8.9e-16  bluestein
      1801  1801             prime    replayed       10777      12686    1.16  8.0e-16  bluestein
      1803  3.601            prime    replayed       10978      12688    1.15  9.0e-16  bluestein
      1804  2^2.11.41        chain3   replayed        3837       9412    2.44  6.0e-16  
      1805  5.19^2           flat     replayed        3878       8451    2.15  1.3e-15  
      1806  2.3.7.43         chain3   replayed        8154       9917    1.12  6.8e-16  
      1807  13.139           prime    replayed       10993      12602    1.14  9.4e-16  bluestein
      1809  3^3.67           prime    replayed       10993      21354    1.94  7.3e-16  bluestein
      1811  1811             prime    replayed       10739      12625    1.17  7.0e-16  bluestein
      1813  7^2.37           chain3   replayed        4813      10277    2.12  6.6e-16  
      1815  3.5.11^2         chain3   replayed        2779       3245    1.15  6.1e-16  
      1817  23.79            prime    replayed       10713      29757    2.77  7.9e-16  bluestein
      1819  17.107           prime    replayed       10768      12597    1.15  8.4e-16  bluestein
      1820  2^2.5.7.13       flat     replayed        2580       3083    1.18  8.1e-16  
      1821  3.607            prime    replayed       10727      12727    1.16  1.0e-15  bluestein
      1823  1823             prime    replayed       10799      12619    1.14  7.4e-16  bluestein
      1824  2^5.3.19         chain3   replayed        2393       5182    2.14  5.0e-16  
      1825  5^2.73           prime    replayed       10989      23853    2.17  7.4e-16  bluestein
      1827  3^2.7.29         flat     replayed        4465       8207    1.82  6.7e-16  
      1829  31.59            prime    replayed       10722      24180    2.20  7.7e-16  bluestein
      1831  1831             prime    replayed       10961      12671    1.15  6.7e-16  bluestein
      1833  3.13.47          flat     replayed        5072      14005    2.74  8.6e-16  
      1835  5.367            prime    replayed       10841      12659    1.17  8.6e-16  bluestein
      1836  2^2.3^3.17       chain3   replayed        2421       5050    2.08  7.1e-16  
      1837  11.167           prime    replayed       10933      12757    1.16  8.1e-16  bluestein
      1839  3.613            prime    replayed       10901      12615    1.15  8.5e-16  bluestein
      1840  2^4.5.23         chain3   replayed        2794       5728    2.02  5.0e-16  
      1841  7.263            prime    replayed       10749      12721    1.16  1.1e-15  bluestein
      1843  19.97            prime    replayed       10749      12681    1.17  8.3e-16  bluestein
      1845  3^2.5.41         chain3   replayed        4673      12197    2.51  5.7e-16  
      1847  1847             prime    replayed       10846      12674    1.14  7.8e-16  bluestein
      1848  2^3.3.7.11       flat     replayed        3183       3032    0.92  9.1e-16  
      1849  43^2             flat     replayed        5885      22629    3.69  1.6e-15  
      1850  2.5^2.37         chain3   replayed        3864       8705    2.23  6.3e-16  
      1851  3.617            prime    replayed       10962      12601    1.15  7.9e-16  bluestein
      1853  17.109           prime    replayed       10936      12759    1.16  8.7e-16  bluestein
      1855  5.7.53           prime    replayed       10927      16524    1.51  6.7e-16  bluestein
      1856  2^6.29           2p       replayed        2839       6754    2.37  4.4e-16  
      1857  3.619            prime    replayed       10778      12737    1.16  9.3e-16  bluestein
      1859  11.13^2          chain3   replayed        3033       3311    1.08  8.5e-16  
      1860  2^2.3.5.31       chain3   replayed        3343       7479    1.74  8.9e-16  flips differ 1.29x
      1861  1861             prime    replayed       10777      12716    1.15  9.5e-16  bluestein
      1862  2.7^2.19         flat     replayed        4081       5691    1.28  7.0e-16  
      1863  3^4.23           chain3   replayed        3713       6933    1.67  8.0e-16  
      1865  5.373            prime    replayed       11003      12715    1.15  7.2e-16  bluestein
      1867  1867             prime    replayed       10980      12597    1.14  9.4e-16  bluestein
      1869  3.7.89           prime    replayed       11011      32624    2.95  7.0e-16  bluestein
      1870  2.5.11.17        chain3   replayed        2586       5642    2.10  5.0e-16  
      1871  1871             prime    replayed       10802      12646    1.17  8.0e-16  bluestein
      1872  2^4.3^2.13       chain3   replayed        2219       2931    1.30  8.0e-16  
      1873  1873             prime    replayed        7961      12790    1.60  1.2e-15  rader
      1875  3.5^4            flat     replayed        3408       3378    0.97  6.7e-16  
      1877  1877             prime    replayed       11180      12715    1.14  8.3e-16  bluestein
      1879  1879             prime    replayed       10940      12655    1.15  8.7e-16  bluestein
      1880  2^3.5.47         flat     replayed        4294      11526    2.55  6.5e-16  
      1881  3^2.11.19        chain3   replayed        3326       5930    1.77  7.6e-16  
      1883  7.269            prime    replayed       10994      12625    1.15  7.8e-16  bluestein
      1885  5.13.29          chain3   replayed        4460       8445    1.89  7.3e-16  
      1886  2.23.41          flat     replayed        6636      12715    1.67  1.6e-15  
      1887  3.17.37          chain3   replayed        4608      13365    2.64  7.8e-16  
      1889  1889             prime    replayed       10820      12800    1.16  1.1e-15  bluestein
      1890  2.3^3.5.7        chain3   replayed        2697       2969    1.06  7.3e-16  
      1891  31.61            prime    replayed       11034      26308    2.37  1.1e-15  bluestein
      1892  2^2.11.43        chain3   replayed        4107      10375    2.51  4.8e-16  
      1893  3.631            prime    replayed       11001      12722    1.16  7.8e-16  bluestein
      1895  5.379            prime    replayed       10794      12725    1.16  9.3e-16  bluestein
      1897  7.271            prime    replayed       10775      12744    1.18  7.4e-16  bluestein
      1899  3^2.211          prime    replayed       11043      12662    1.14  9.2e-16  bluestein
      1900  2^2.5^2.19       flat     replayed        3482       5389    1.53  6.3e-16  
      1901  1901             prime    replayed       10965      12776    1.16  1.2e-15  bluestein
      1903  11.173           prime    replayed       10948      12636    1.15  1.1e-15  bluestein
      1904  2^4.7.17         chain3   replayed        2430       4822    1.98  5.8e-16  
      1905  3.5.127          prime    replayed       11024      12776    1.16  7.6e-16  bluestein
      1907  1907             prime    replayed       11023      12664    1.15  9.0e-16  bluestein
      1909  23.83            prime    replayed       11031      33428    3.01  7.9e-16  bluestein
      1911  3.7^2.13         chain3   replayed        3109       3426    1.10  5.8e-16  
      1913  1913             prime    replayed       11011      12748    1.16  8.7e-16  bluestein
      1914  2.3.11.29        flat     replayed        5342       7362    1.21  2.0e-15  
      1915  5.383            prime    replayed       10945      12652    1.15  8.4e-16  bluestein
      1917  3^3.71           prime    replayed       10829      24253    2.19  8.5e-16  bluestein
      1919  19.101           prime    replayed       10800      12706    1.17  8.6e-16  bluestein
      1921  17.113           prime    replayed       11040      12837    1.16  8.0e-16  bluestein
      1922  2.31^2           flat     replayed        6209      12128    1.94  2.4e-15  
      1923  3.641            prime    replayed       11015      12711    1.15  1.0e-15  bluestein
      1924  2^2.13.37        chain3   replayed        3995       9087    2.25  5.8e-16  
      1925  5^2.7.11         flat     replayed        3173       3340    1.04  8.2e-16  
      1927  41.47            prime    replayed       10974      24473    2.22  8.5e-16  bluestein
      1929  3.643            prime    replayed       10995      12802    1.16  6.7e-16  bluestein
      1931  1931             prime    replayed       10785      12738    1.18  8.6e-16  bluestein
      1932  2^2.3.7.23       flat     replayed        3428       6463    1.73  1.1e-15  
      1933  1933             prime    replayed       10808      12817    1.18  9.6e-16  bluestein
      1935  3^2.5.43         flat     replayed        4484      13388    2.87  6.2e-16  
      1936  2^4.11^2         chain3   replayed        2285       3009    1.31  6.5e-16  
      1937  13.149           prime    replayed       10805      12800    1.18  8.7e-16  bluestein
      1938  2.3.17.19        chain3   replayed        3253       7315    2.23  8.8e-16  
      1939  7.277            prime    replayed       11049      12778    1.14  8.0e-16  bluestein
      1941  3.647            prime    replayed       10834      12803    1.16  7.7e-16  bluestein
      1943  29.67            prime    replayed       11057      28520    2.58  8.3e-16  bluestein
      1945  5.389            prime    replayed       11058      12832    1.15  8.2e-16  bluestein
      1947  3.11.59          prime    replayed       10880      19460    1.78  8.7e-16  bluestein
      1949  1949             prime    replayed       11039      12766    1.15  8.4e-16  bluestein
      1950  2.3.5^2.13       chain3   replayed        3107       3166    1.01  6.5e-16  
      1951  1951             prime    replayed       11062      12728    1.15  7.7e-16  bluestein
      1953  3^2.7.31         flat     replayed        5157       9242    1.64  8.5e-16  
      1955  5.17.23          chain3   replayed        5671       9644    1.68  5.5e-16  
      1957  19.103           prime    replayed       10834      12826    1.16  8.6e-16  bluestein
      1959  3.653            prime    replayed       10836      12651    1.15  8.8e-16  bluestein
      1960  2^3.5.7^2        flat     replayed        3030       3054    1.00  6.9e-16  
      1961  37.53            prime    replayed       10813      25811    2.33  8.2e-16  bluestein
      1963  13.151           prime    replayed       10833      12773    1.17  8.3e-16  bluestein
      1965  3.5.131          prime    replayed       10841      12834    1.16  1.0e-15  bluestein
      1967  7.281            prime    replayed       10839      12730    1.15  9.7e-16  bluestein
      1968  2^4.3.41         flat     replayed        4161      10060    2.39  7.7e-16  
      1969  11.179           prime    replayed       10813      12855    1.16  7.5e-16  bluestein
      1971  3^3.73           prime    replayed       10828      25990    2.39  1.0e-15  bluestein
      1972  2^2.17.29        flat     replayed        4634       9506    1.80  8.5e-16  
      1973  1973             prime    replayed       10856      12843    1.16  1.0e-15  bluestein
      1974  2.3.7.47         chain3   replayed        4587      12088    2.55  5.7e-16  
      1975  5^2.79           prime    replayed       10965      28340    2.56  6.5e-16  bluestein
      1976  2^3.13.19        chain3   replayed        2949       5694    1.87  8.6e-16  
      1977  3.659            prime    replayed       10792      12847    1.16  1.0e-15  bluestein
      1978  2.23.43          flat     replayed        6999      13827    1.67  1.7e-15  
      1979  1979             prime    replayed       11021      12737    1.15  5.5e-16  bluestein
      1980  2^2.3^2.5.11     chain3   replayed        2448       3338    1.36  6.9e-16  
      1981  7.283            prime    replayed       10839      12841    1.16  1.0e-15  bluestein
      1983  3.661            prime    replayed       11046      12850    1.15  8.7e-16  bluestein
      1984  2^6.31           2p       replayed        3167       7565    1.71  5.3e-16  flips differ 1.39x
      1985  5.397            prime    replayed       11074      12871    1.16  9.6e-16  bluestein
      1987  1987             prime    replayed       11064      12754    1.15  8.8e-16  bluestein
      1989  3^2.13.17        chain3   replayed        3441       6132    1.77  7.3e-16  
      1991  11.181           prime    replayed       11192      12782    1.13  1.1e-15  bluestein
      1993  1993             prime    replayed       10929      12856    1.17  1.1e-15  bluestein
      1995  3.5.7.19         chain3   replayed        3557       6292    1.77  5.7e-16  
      1997  1997             prime    replayed       10905      12838    1.16  8.2e-16  bluestein
      1998  2.3^3.37         chain3   replayed        7874       9438    1.20  6.6e-16  
      1999  1999             prime    replayed       11083      12712    1.14  9.1e-16  bluestein
      2001  3.23.29          chain3   replayed        6126      13056    1.63  6.4e-16  flips differ 1.31x
      2002  2.7.11.13        flat     replayed        4131       3968    0.90  7.9e-16  
      2003  2003             prime    replayed       11023      12750    1.15  1.1e-15  bluestein
      2005  5.401            prime    replayed       10866      12878    1.16  9.1e-16  bluestein
      2007  3^2.223          prime    replayed       10845      12758    1.17  8.8e-16  bluestein
      2009  7^2.41           chain3   replayed        6442      13026    1.52  5.2e-16  flips differ 1.33x
      2011  2011             prime    replayed       10879      12840    1.17  9.0e-16  bluestein
      2013  3.11.61          prime    replayed       11105      21312    1.92  7.7e-16  bluestein
      2015  5.13.31          flat     replayed        4657       9558    1.81  7.2e-16  
      2016  2^5.3^2.7        chain3   replayed        2255       3217    1.40  6.3e-16  
      2017  2017             prime    replayed       10817      12965    1.20  9.6e-16  bluestein
      2019  3.673            prime    replayed       10859      12863    1.16  8.6e-16  bluestein
      2021  43.47            prime    replayed       10887      26264    2.37  8.7e-16  bluestein
      2023  7.17^2           chain3   replayed        3551       8575    2.28  9.0e-16  
      2024  2^3.11.23        chain3   replayed        3299       6644    1.98  6.4e-16  
      2025  3^4.5^2          chain3   replayed        3559       3880    1.07  6.1e-16  
      2027  2027             prime    replayed       10826      12819    1.15  8.0e-16  bluestein
      2028  2^2.3.13^2       chain3   replayed        2651       3661    1.37  7.7e-16  
      2029  2029             prime    replayed        8779      12884    1.45  1.2e-15  rader
      2030  2.5.7.29         flat     replayed        5325       8166    1.36  6.2e-16  
      2031  3.677            prime    replayed       10957      12788    1.15  7.4e-16  bluestein
      2033  19.107           prime    replayed       11100      12844    1.16  7.7e-16  bluestein
      2035  5.11.37          chain3   replayed        6198      11655    1.88  5.7e-16  
      2037  3.7.97           prime    replayed       10886      12913    1.17  9.8e-16  bluestein
      2039  2039             prime    replayed       10884      12812    1.15  1.1e-15  bluestein
      2040  2^3.3.5.17       chain3   replayed        2906       5555    1.90  6.9e-16  
      2041  13.157           prime    replayed       10852      12876    1.18  8.4e-16  bluestein
      2043  3^2.227          prime    replayed       10855      12834    1.18  1.2e-15  bluestein
      2045  5.409            prime    replayed       11113      12882    1.16  8.8e-16  bluestein
      2046  2.3.11.31        chain3   replayed        5838       8243    1.03  6.1e-16  flips differ 1.37x
      2047  23.89            prime    replayed       11065      39938    3.59  8.2e-16  bluestein
      2050  2.5^2.41         chain3   replayed        6360      10656    1.12  6.1e-16  flips differ 1.50x
      2052  2^2.3^3.19       chain3   replayed        2918       6000    1.99  6.6e-16  
      2057  11^2.17          flat     replayed        4201       5991    1.43  1.3e-15  
      2058  2.3.7^3          flat     replayed        4041       3421    0.84  5.6e-16  
      2064  2^4.3.43         flat     replayed        4229      11037    2.42  7.1e-16  
      2068  2^2.11.47        chain3   replayed        7327      12736    1.74  5.8e-16  
      2070  2.3^2.5.23       chain3   replayed        4844       6745    1.37  7.8e-16  
      2072  2^3.7.37         chain3   replayed        6599       9718    1.10  6.0e-16  flips differ 1.34x
      2079  3^3.7.11         chain3   replayed        3288       3849    1.15  6.9e-16  
      2080  2^5.5.13         chain3   replayed        2487       3409    1.35  7.2e-16  
      2088  2^3.3^2.29       flat     replayed        4743       8076    1.64  9.2e-16  
      2090  2.5.11.19        chain3   replayed        2981       6661    2.15  6.5e-16  
      2091  3.17.41          chain3   replayed        5517      16531    2.88  6.7e-16  
      2093  7.13.23          flat     replayed        4245       7624    1.74  6.7e-16  
      2100  2^2.3.5^2.7      chain3   replayed        2876       3370    1.15  7.2e-16  
      2106  2.3^4.13         chain3   replayed        3846       3676    0.85  6.1e-16  
      2107  7^2.43           chain3   replayed        7176      14320    1.51  5.1e-16  flips differ 1.33x
      2108  2^2.17.31        flat     replayed        5874      10561    1.48  1.1e-15  
      2109  3.19.37          flat     replayed        5824      15450    2.59  8.0e-16  
      2112  2^6.3.11         chain3   replayed        2427       3307    1.36  6.3e-16  
      2115  3^2.5.47         chain3   replayed        7775      16256    1.58  7.2e-16  flips differ 1.33x
      2116  2^2.23^2         flat     replayed        5598      10079    1.25  1.6e-15  flips differ 1.44x
      2125  5^3.17           chain3   replayed        3660       6325    1.59  7.5e-16  
      2128  2^4.7.19         chain3   replayed        2779       5779    2.02  5.3e-16  
      2132  2^2.13.41        chain3   replayed        4782      11120    2.23  6.3e-16  
      2139  3.23.31          flat     replayed        6021      14462    1.85  8.6e-16  flips differ 1.30x
      2142  2.3^2.7.17       chain3   replayed        3403       5912    1.64  6.7e-16  
      2145  3.5.11.13        chain3   replayed        3434       4008    1.16  6.0e-16  
      2146  2.29.37          flat     replayed        6797      14646    2.13  2.7e-15  
      2150  2.5^2.43         chain3   replayed        5066      11744    2.27  6.8e-16  
      2156  2^2.7^2.11       flat     replayed        3211       3911    1.18  7.9e-16  
      2162  2.23.47          flat     replayed        8538      16602    1.92  1.3e-15  
      2166  2.3.19^2         chain3   replayed        3721       8488    2.25  1.0e-15  
      2170  2.5.7.31         chain3   replayed        6261       9131    1.10  6.4e-16  flips differ 1.32x
      2175  3.5^2.29         chain3   replayed        5781       9855    1.67  5.7e-16  
      2176  2^7.17           chain3   replayed        2719       5786    2.11  6.6e-16  
      2178  2.3^2.11^2       flat     replayed        4855       3927    0.79  1.0e-15  
      2184  2^3.3.7.13       chain3   replayed        3433       3657    1.05  7.5e-16  
      2185  5.19.23          chain3   replayed        4928      11513    2.31  7.0e-16  
      2193  3.17.43          chain3   replayed        8057      18089    1.70  7.2e-16  flips differ 1.33x
      2197  13^3             flat     replayed        4106       4123    1.00  1.0e-15  
      2200  2^3.5^2.11       flat     replayed        3188       3623    1.12  7.9e-16  
      2204  2^2.19.29        flat     replayed        5169      10927    1.83  8.4e-16  
      2205  3^2.5.7^2        chain3   replayed        3528       4110    1.12  8.0e-16  
      2208  2^5.3.23         chain3   replayed        3818       7222    1.82  6.1e-16  
      2209  47^2             flat     replayed        7533      30527    4.00  1.9e-15  
      2210  2.5.13.17        chain3   replayed        3868       6912    1.78  6.6e-16  
      2214  2.3^3.41         chain3   replayed        6577      11525    1.74  7.8e-16  
      2220  2^2.3.5.37       chain3   replayed        4240      10456    2.30  6.6e-16  
      2223  3^2.13.19        flat     replayed        4301       7229    1.60  1.2e-15  
      2232  2^3.3^2.31       flat     replayed        4795       9040    1.87  8.3e-16  
      2233  7.11.29          chain3   replayed        5194       9853    1.62  6.7e-16  
      2236  2^2.13.43        chain3   replayed        4961      12344    2.42  7.8e-16  
      2240  2^6.5.7          ztt      replayed        2053       3234    1.57  4.2e-16  
      2244  2^2.3.11.17      chain3   replayed        3002       6361    2.10  5.8e-16  
      2254  2.7^2.23         flat     replayed        5638       7866    1.36  6.2e-16  
      2255  5.11.41          chain3   replayed        5869      14732    2.44  6.0e-16  
      2256  2^4.3.47         chain3   replayed        4990      13605    2.61  6.3e-16  
      2261  7.17.19          flat     replayed        4441       9958    2.22  9.9e-16  
      2262  2.3.13.29        chain3   replayed        5325       8819    1.65  8.6e-16  
      2268  2^2.3^4.7        chain3   replayed        2928       3888    1.32  5.9e-16  
      2275  5^2.7.13         chain3   replayed        3778       4166    1.03  5.6e-16  
      2277  3^2.11.23        chain3   replayed        4759       8480    1.64  6.3e-16  
      2280  2^3.3.5.19       chain3   replayed        3009       6578    2.15  5.3e-16  
      2288  2^4.11.13        chain3   replayed        2840       3631    1.25  6.9e-16  
      2294  2.31.37          flat     replayed        7748      16041    1.74  1.9e-15  
      2295  3^3.5.17         chain3   replayed        3944       7230    1.76  8.8e-16  
      2296  2^3.7.41         flat     replayed        5434      11898    2.19  6.8e-16  
      2299  11^2.19          flat     replayed        4518       7032    1.36  1.0e-15  
      2300  2^2.5^2.23       chain3   replayed        4083       7492    1.70  6.4e-16  
      2303  7^2.47           chain3   replayed        8256      17424    2.10  7.6e-16  
      2310  2.3.5.7.11       chain3   replayed        3462       3944    1.06  7.2e-16  
      2312  2^3.17^2         chain3   replayed        4211       8474    1.90  8.5e-16  
      2320  2^4.5.29         flat     replayed        4380       8596    1.73  6.2e-16  
      2322  2.3^3.43         chain3   replayed        8073      12628    1.20  5.3e-16  flips differ 1.31x
      2325  3.5^2.31         chain3   replayed        6837      11113    1.27  8.4e-16  flips differ 1.28x
      2331  3^2.7.37         chain3   replayed        6838      13628    1.39  5.8e-16  flips differ 1.43x
      2337  3.19.41          chain3   replayed        7227      19122    2.57  7.5e-16  
      2340  2^2.3^2.5.13     chain3   replayed        3010       3981    1.32  7.6e-16  
      2346  2.3.17.23        chain3   replayed        4625       9863    2.13  8.7e-16  
      2349  3^4.29           chain3   replayed        5507      11019    1.45  7.9e-16  flips differ 1.38x
      2350  2.5^2.47         chain3   replayed       11417      14394    1.23  5.0e-16  
      2352  2^4.3.7^2        ztt      replayed        2278       3457    1.50  5.7e-16  
      2356  2^2.19.31        chain3   replayed        6454      12096    1.75  6.3e-16  
      2365  5.11.43          chain3   replayed        8135      16232    1.44  6.3e-16  flips differ 1.39x
      2366  2.7.13^2         flat     replayed        5633       4912    0.86  1.1e-15  
      2368  2^6.37           2p       replayed        7923      10630    1.18  4.8e-16  
      2375  5^3.19           flat     replayed        4031       7466    1.64  6.8e-16  
      2376  2^3.3^3.11       chain3   replayed        4059       4080    0.91  5.8e-16  
      2378  2.29.41          flat     replayed        9474      17441    1.83  2.6e-15  
      2380  2^2.5.7.17       flat     replayed        4179       6628    1.52  6.4e-16  
      2387  7.11.31          chain3   replayed        6852      11125    1.61  6.7e-16  
      2392  2^3.13.23        chain3   replayed        5586       7978    1.39  7.3e-16  
      2394  2.3^2.7.19       chain3   replayed        4234       7000    1.58  5.9e-16  
      2397  3.17.47          flat     replayed        8432      21539    2.47  1.6e-15  
      2401  7^4              flat     replayed        4597       4089    0.89  6.8e-16  
      2405  5.13.37          chain3   replayed        5787      14075    2.41  7.1e-16  
      2408  2^3.7.43         chain3   replayed        5170      13042    2.51  5.8e-16  
      2415  3.5.7.23         chain3   replayed        6449       8942    1.38  6.8e-16  
      2418  2.3.13.31        flat     replayed        6690       9853    1.23  1.2e-15  
      2420  2^2.5.11^2       flat     replayed        4155       4473    1.05  7.1e-16  
      2431  11.13.17         chain3   replayed        4231       7296    1.70  6.8e-16  
      2432  2^7.19           chain3   replayed        3222       6874    2.08  6.5e-16  
      2436  2^2.3.7.29       flat     replayed        5101       9602    1.70  5.9e-16  
      2442  2.3.11.37        chain3   replayed        5164      11506    2.16  4.9e-16  
      2444  2^2.13.47        flat     replayed        6306      15084    2.27  8.8e-16  
      2448  2^4.3^2.17       chain3   replayed        3291       6490    1.96  6.5e-16  
      2450  2.5^2.7^2        flat     replayed        4604       4473    0.96  8.4e-16  
      2451  3.19.43          flat     replayed        7526      20782    2.51  1.5e-15  
      2457  3^3.7.13         chain3   replayed        4193       4792    1.14  8.3e-16  
      2460  2^2.3.5.41       flat     replayed        5467      12779    2.33  6.9e-16  
      2464  2^5.7.11         chain3   replayed        2870       4124    1.42  6.0e-16  
      2465  5.17.29          flat     replayed        7754      14390    1.65  1.5e-15  
      2470  2.5.13.19        chain3   replayed        4448       8140    1.54  7.6e-16  
      2475  3^2.5^2.11       chain3   replayed        3900       4686    1.18  7.2e-16  
      2480  2^4.5.31         flat     replayed        4843       9626    1.57  4.8e-16  flips differ 1.26x
      2484  2^2.3^3.23       chain3   replayed        3818       8311    2.17  7.1e-16  
      2494  2.29.43          flat     replayed        8769      18843    2.09  2.8e-15  
      2496  2^6.3.13         chain3   replayed        2899       3956    1.36  6.7e-16  
      2499  3.7^2.17         chain3   replayed        4201       7557    1.78  7.2e-16  
      2508  2^2.3.11.19      flat     replayed        4892       7539    1.51  1.5e-15  
      2511  3^4.31           flat     replayed        5685      12329    2.14  7.8e-16  
      2516  2^2.17.37        chain3   replayed        5541      14269    2.58  7.7e-16  
      2520  2^3.3^2.5.7      chain3   replayed        3181       4093    1.25  6.5e-16  
      2523  3.29^2           flat     replayed        8516      18910    2.00  1.4e-15  
      2527  7.19^2           chain3   replayed        5054      11902    2.28  6.4e-16  
      2530  2.5.11.23        chain3   replayed        4352       9160    2.09  7.3e-16  
      2535  3.5.13^2         chain3   replayed        4202       5113    1.21  7.3e-16  
      2538  2.3^3.47         chain3   replayed        9190      15587    1.27  5.6e-16  flips differ 1.33x
      2541  3.7.11^2         flat     replayed        5299       4786    0.89  8.2e-16  
      2542  2.31.41          flat     replayed       10640      19054    1.29  2.6e-15  flips differ 1.38x
      2548  2^2.7^2.13       flat     replayed        3946       4671    1.14  7.8e-16  
      2550  2.3.5^2.17       chain3   replayed        3676       6887    1.86  8.0e-16  
      2552  2^3.11.29        chain3   replayed        5438       9938    1.14  5.2e-16  flips differ 1.61x
      2553  3.23.37          flat     replayed        7828      20059    2.48  8.7e-16  
      2565  3^3.5.19         chain3   replayed        4781       8407    1.55  9.9e-16  
      2574  2.3^2.11.13      flat     replayed        6363       4656    0.71  1.1e-15  
      2576  2^4.7.23         flat     replayed        4361       8095    1.76  8.2e-16  
      2580  2^2.3.5.43       flat     replayed        5879      13996    2.22  1.0e-15  
      2583  3^2.7.41         chain3   replayed        6371      17184    2.51  6.2e-16  
      2584  2^3.17.19        chain3   replayed        4179       9873    2.20  6.1e-16  
      2585  5.11.47          chain3   replayed       12762      19740    1.41  6.4e-16  
      2590  2.5.7.37         flat     replayed        7526      12697    1.66  6.4e-16  
      2600  2^3.5^2.13       flat     replayed        3942       4332    1.10  7.5e-16  
      2601  3^2.17^2         chain3   replayed        4581      11465    2.33  8.6e-16  
      2604  2^2.3.7.31       chain3   replayed        4719      10722    2.20  7.0e-16  
      2610  2.3^2.5.29       flat     replayed        6642      10057    1.50  6.6e-16  
      2618  2.7.11.17        flat     replayed        5912       8032    1.34  8.9e-16  
      2622  2.3.19.23        flat     replayed        7980      11417    1.26  1.2e-15  
      2624  2^6.41           2p       replayed        6901      13119    1.39  5.1e-16  flips differ 1.37x
      2625  3.5^3.7          flat     replayed        4813       4969    1.02  9.2e-16  
      2632  2^3.7.47         flat     replayed        6110      16135    2.60  5.8e-16  
      2635  5.17.31          flat     replayed        6712      16030    1.95  7.0e-16  
      2639  7.13.29          chain3   replayed        5996      11943    1.97  8.9e-16  
      2640  2^4.3.5.11       chain3   replayed        3229       4171    1.28  6.0e-16  
      2645  5.23^2           flat     replayed        7391      15146    1.67  1.8e-15  
      2646  2.3^3.7^2        chain3   replayed        4737       4568    0.96  7.6e-16  
      2652  2^2.3.13.17      chain3   replayed        4535       7739    1.69  7.6e-16  
      2660  2^2.5.7.19       flat     replayed        4902       7836    1.57  6.5e-16  
      2662  2.11^3           flat     replayed        6610       5560    0.83  7.0e-16  
      2664  2^3.3^2.37       flat     replayed        5663      12626    2.19  7.6e-16  
      2665  5.13.41          chain3   replayed        6751      17800    2.42  8.5e-16  
      2666  2.31.43          flat     replayed        9133      20597    2.08  3.7e-15  
      2668  2^2.23.29        chain3   replayed        6749      14345    1.82  5.9e-16  
      2673  3^5.11           chain3   replayed        4625       5305    1.00  7.4e-16  
      2679  3.19.47          chain3   replayed       10140      24750    1.90  9.5e-16  flips differ 1.28x
      2688  2^7.3.7          ztt      replayed        2486       4404    1.77  5.0e-16  
      2691  3^2.13.23        chain3   replayed        5365      10216    1.89  5.7e-16  
      2695  5.7^2.11         flat     replayed        4920       5031    1.02  6.0e-16  
      2697  3.29.31          flat     replayed       10927      20907    1.86  2.4e-15  
      2704  2^4.13^2         chain3   replayed        3529       4400    1.24  7.6e-16  
      2706  2.3.11.41        flat     replayed        9374      14066    1.47  9.2e-16  
      2709  3^2.7.43         chain3   replayed        9615      18858    1.54  7.8e-16  flips differ 1.28x
      2717  11.13.19         flat     replayed        5386       8623    1.59  9.6e-16  
      2720  2^5.5.17         chain3   replayed        3504       7498    2.13  7.4e-16  
      2726  2.29.47          flat     replayed       10885      22514    2.01  3.1e-15  
      2728  2^3.11.31        flat     replayed        6296      11096    1.75  8.1e-16  
      2730  2.3.5.7.13       flat     replayed        5565       4714    0.83  6.4e-16  
      2736  2^4.3^2.19       chain3   replayed        3742       7711    2.03  6.8e-16  
      2737  7.17.23          flat     replayed        5778      13643    2.10  1.4e-15  
      2738  2.37^2           flat     replayed        9235      20995    2.27  3.7e-15  
      2744  2^3.7^3          flat     replayed        4626       4487    0.95  6.8e-16  
      2750  2.5^3.11         chain3   replayed        4874       4879    1.00  7.5e-16  
      2752  2^6.43           2p       replayed        5142      14409    2.79  5.3e-16  
      2754  2.3^4.17         chain3   replayed        6269       7793    1.24  6.2e-16  
      2755  5.19.29          chain3   replayed        7265      16838    2.29  6.8e-16  
      2760  2^3.3.5.23       chain3   replayed        4252       9146    1.92  1.0e-15  
      2772  2^2.3^2.7.11     chain3   replayed        3674       5067    1.36  6.5e-16  
      2775  3.5^2.37         chain3   replayed        8566      16260    1.89  7.5e-16  
      2783  11^2.23          flat     replayed        6337      10054    1.55  1.4e-15  
      2784  2^5.3.29         flat     replayed        6334      10782    1.40  6.5e-16  
      2788  2^2.17.41        chain3   replayed        6476      17229    2.64  8.6e-16  
      2790  2.3^2.5.31       chain3   replayed        5379      11260    2.03  5.6e-16  
      2793  3.7^2.19         chain3   replayed        4773       8887    1.77  7.6e-16  
      2795  5.13.43          chain3   replayed        7523      19551    2.42  7.4e-16  
      2800  2^4.5^2.7        ztt      replayed        2715       4186    1.53  4.6e-16  
      2805  3.5.11.17        chain3   replayed        4987       8614    1.61  7.9e-16  
      2808  2^3.3^3.13       flat     replayed        5139       4904    0.93  1.2e-15  
      2812  2^2.19.37        chain3   replayed        6312      16324    2.55  6.1e-16  
      2816  2^8.11           chain3   replayed        3932       4341    1.09  5.7e-16  
      2820  2^2.3.5.47       chain3   replayed        6371      17304    2.71  9.5e-16  
      2821  7.13.31          chain3   replayed        6771      13450    1.67  7.5e-16  
      2829  3.23.41          flat     replayed        8348      24557    2.57  9.8e-16  
      2835  3^4.5.7          chain3   replayed        5067       5602    1.10  8.2e-16  
      2838  2.3.11.43        chain3   replayed        7621      15632    2.00  5.8e-16  
      2842  2.7^2.29         flat     replayed        8690      11631    1.18  2.0e-15  
      2849  7.11.37          flat     replayed        6552      16393    2.21  7.2e-16  
      2850  2.3.5^2.19       flat     replayed        6347       8169    1.28  8.6e-16  
      2852  2^2.23.31        chain3   replayed        8417      15792    1.30  9.5e-16  flips differ 1.45x
      2856  2^3.3.7.17       chain3   replayed        4568       8007    1.74  6.3e-16  
      2860  2^2.5.11.13      flat     replayed        4702       5250    1.11  7.7e-16  
      2870  2.5.7.41         flat     replayed        8751      15429    1.75  7.7e-16  
      2871  3^2.11.29        flat     replayed        8653      13126    1.44  8.2e-16  
      2873  13^2.17          chain3   replayed        5264       8931    1.69  9.1e-16  
      2875  5^3.23           chain3   replayed        6259      10749    1.68  7.3e-16  
      2883  3.31^2           flat     replayed        9230      23102    2.03  2.0e-15  
      2886  2.3.13.37        chain3   replayed        5998      13740    2.16  8.6e-16  
      2888  2^3.19^2         flat     replayed        6915      11478    1.29  1.8e-15  flips differ 1.29x
      2890  2.5.17^2         chain3   replayed        5371      11389    2.10  7.8e-16  
      2898  2.3^2.7.23       flat     replayed        6902       9808    1.33  8.2e-16  
      2900  2^2.5^2.29       flat     replayed        5837      11193    1.86  5.3e-16  
      2904  2^3.3.11^2       flat     replayed        5280       5385    1.02  9.1e-16  
      2907  3^2.17.19        flat     replayed        7720      13267    1.57  1.6e-15  
      2912  2^5.7.13         chain3   replayed        3500       4960    1.41  5.6e-16  
      2914  2.31.47          flat     replayed       10413      24555    2.01  1.9e-15  
      2924  2^2.17.43        chain3   replayed        6999      18814    2.52  7.2e-16  
      2925  3^2.5^2.13       chain3   replayed        5340       5861    1.08  7.7e-16  
      2926  2.7.11.19        flat     replayed        7250       9451    1.20  1.5e-15  
      2940  2^2.3.5.7^2      flat     replayed        4695       5431    1.15  6.8e-16  
      2944  2^7.23           chain3   replayed        5218       9632    1.69  5.6e-16  
      2945  5.19.31          chain3   replayed       10137      18836    1.83  6.2e-16  
      2952  2^3.3^2.41       chain3   replayed        5786      15450    2.66  6.1e-16  
      2958  2.3.17.29        flat     replayed        8783      14232    1.59  1.2e-15  
      2960  2^4.5.37         flat     replayed        5711      13528    2.28  5.9e-16  
      2961  3^2.7.47         flat     replayed        7321      22911    3.00  5.8e-16  
      2964  2^2.3.13.19      chain3   replayed        4416       9144    2.04  7.9e-16  
      2967  3.23.43          chain3   replayed       14328      26767    1.83  5.8e-16  
      2970  2.3^3.5.11       chain3   replayed        4711       5152    1.09  8.1e-16  
      2975  5^2.7.17         flat     replayed        5935       9178    1.50  7.4e-16  
      2976  2^5.3.31         chain3   replayed        5148      12049    1.89  6.5e-16  
      2990  2.5.13.23        flat     replayed        8134      11203    1.36  1.0e-15  
      2992  2^4.11.17        chain3   replayed        4107       8134    1.96  7.1e-16  
      2997  3^4.37           chain3   replayed        7798      17901    2.19  6.2e-16  
      3003  3.7.11.13        chain3   replayed        4619       5800    1.24  6.5e-16  
      3008  2^6.47           2p       replayed       13197      17900    1.29  5.4e-16  
      3010  2.5.7.43         chain3   replayed       13176      17007    1.20  6.1e-16  
      3016  2^3.13.29        flat     replayed        6809      11908    1.61  7.2e-16  
      3024  2^4.3^3.7        ztt      replayed        3207       4876    1.49  4.1e-16  
      3025  5^2.11^2         flat     replayed        5869       5903    0.95  9.1e-16  
      3034  2.37.41          flat     replayed       10335      24791    2.40  3.2e-15  
      3036  2^2.3.11.23      flat     replayed        5906      10431    1.75  6.7e-16  
      3038  2.7^2.31         flat     replayed        8345      12972    1.37  7.2e-16  
      3040  2^5.5.19         chain3   replayed        4103       8896    2.17  6.8e-16  
      3042  2.3^2.13^2       flat     replayed        7104       5834    0.79  1.2e-15  
      3045  3.5.7.29         flat     replayed        8143      13935    1.53  8.1e-16  
      3055  5.13.47          chain3   replayed       11552      23653    1.55  8.4e-16  flips differ 1.32x
      3059  7.19.23          flat     replayed        6872      16141    2.20  9.1e-16  
      3060  2^2.3^2.5.17     chain3   replayed        4290       8597    1.93  7.8e-16  
      3069  3^2.11.31        chain3   replayed        8964      14848    1.32  8.0e-16  
      3075  3.5^2.41         chain3   replayed        9794      20575    1.59  8.0e-16  flips differ 1.32x
      3078  2.3^4.19         chain3   replayed        6329       9223    1.36  6.8e-16  
      3080  2^3.5.7.11       flat     replayed        4838       5410    1.10  6.4e-16  
      3087  3^2.7^3          chain3   replayed        5256       6016    1.08  5.6e-16  
      3094  2.7.13.17        flat     replayed        7458       9843    1.27  1.2e-15  
      3096  2^3.3^2.43       chain3   replayed        6484      16962    2.50  7.0e-16  
      3100  2^2.5^2.31       flat     replayed        8086      12542    1.45  5.9e-16  
      3102  2.3.11.47        flat     replayed       10540      19028    1.77  1.0e-15  
      3105  3^3.5.23         chain3   replayed        7361      11932    1.62  7.3e-16  
      3108  2^2.3.7.37       chain3   replayed        5774      14908    2.54  7.6e-16  
      3116  2^2.19.41        chain3   replayed        7438      19667    2.43  5.4e-16  
      3120  2^4.3.5.13       chain3   replayed        4013       5009    1.24  7.2e-16  
      3128  2^3.17.23        chain3   replayed        7735      13310    1.46  7.0e-16  
      3132  2^2.3^3.29       chain3   replayed        6677      12390    1.80  6.1e-16  
      3135  3.5.11.19        chain3   replayed        5548      10119    1.81  8.8e-16  
      3136  2^6.7^2          ztt      replayed        3073       4759    1.53  4.0e-16  
      3145  5.17.37          chain3   replayed        8192      22613    2.58  7.1e-16  
      3146  2.11^2.13        flat     replayed        7317       6644    0.88  1.1e-15  
      3150  2.3^2.5^2.7      chain3   replayed        4676       5313    1.08  7.3e-16  
      3157  7.11.41          chain3   replayed        8450      20760    2.42  5.2e-16  
      3159  3^5.13           chain3   replayed        5721       6581    1.03  8.2e-16  
      3162  2.3.17.31        flat     replayed        9378      15790    1.44  1.4e-15  
      3168  2^5.3^2.11       chain3   replayed        3844       5570    1.43  6.3e-16  
      3174  2.3.23^2         chain3   replayed       10434      15208    1.45  8.6e-16  
      3179  11.17^2          flat     replayed        7005      13777    1.93  1.1e-15  
      3182  2.37.43          flat     replayed       11786      26749    2.23  2.3e-15  
      3185  5.7^2.13         flat     replayed        5518       6152    1.10  8.1e-16  
      3190  2.5.11.29        chain3   replayed        7523      13463    1.78  6.2e-16  
      3192  2^3.3.7.19       chain3   replayed        5312       9498    1.78  7.0e-16  
      3196  2^2.17.47        flat     replayed        8848      22752    2.57  1.1e-15  
      3198  2.3.13.41        chain3   replayed        7098      16842    2.36  8.4e-16  
      3211  13^2.19          flat     replayed        6503      10467    1.50  1.1e-15  
      3213  3^3.7.17         chain3   replayed        5724      10204    1.77  9.4e-16  
      3219  3.29.37          flat     replayed       10180      28454    2.78  2.2e-15  
      3220  2^2.5.7.23       flat     replayed        5705      10916    1.79  5.1e-16  
      3224  2^3.13.31        chain3   replayed        7507      13304    1.73  7.6e-16  
      3225  3.5^2.43         chain3   replayed       14614      22602    1.52  7.8e-16  
      3230  2.5.17.19        chain3   replayed        5886      13269    2.07  7.0e-16  
      3234  2.3.7^2.11       flat     replayed        6615       5918    0.89  7.7e-16  
      3243  3.23.47          chain3   replayed       16561      31611    1.85  6.3e-16  
      3248  2^4.7.29         chain3   replayed        7065      12197    1.72  5.8e-16  
      3249  3^2.19^2         flat     replayed        8953      15820    1.69  1.2e-15  
      3250  2.5^3.13         chain3   replayed        5032       5844    1.15  7.2e-16  
      3255  3.5.7.31         flat     replayed        8703      15691    1.79  8.0e-16  
      3256  2^3.11.37        flat     replayed        7420      15514    2.07  8.7e-16  
      3264  2^6.3.17         chain3   replayed        4380       8824    2.00  6.7e-16  
      3267  3^3.11^2         flat     replayed        7057       6640    0.94  6.1e-16  
      3268  2^2.19.43        flat     replayed        8859      21416    2.30  1.7e-15  
      3276  2^2.3^2.7.13     chain3   replayed        4844       6046    1.18  7.9e-16  
      3280  2^4.5.41         flat     replayed        6717      16664    2.19  4.4e-16  
      3289  11.13.23         chain3   replayed        7839      12286    1.52  9.7e-16  
      3290  2.5.7.47         flat     replayed       10523      20771    1.94  5.9e-16  
      3300  2^2.3.5^2.11     chain3   replayed        4499       5826    1.23  7.5e-16  
      3306  2.3.19.29        chain3   replayed       10892      16370    1.38  8.0e-16  
      3311  7.11.43          chain3   replayed       16515      22892    1.37  6.4e-16  
      3312  2^4.3^2.23       chain3   replayed        5581      10776    1.81  5.7e-16  
      3315  3.5.13.17        flat     replayed        6444      10507    1.58  1.3e-15  
      3321  3^4.41           chain3   replayed        8797      22598    2.30  7.4e-16  
      3325  5^2.7.19         chain3   replayed        6587      10802    1.36  6.2e-16  
      3328  2^8.13           chain3   replayed        5603       5197    0.91  6.9e-16  
      3330  2.3^2.5.37       chain3   replayed        9529      15719    1.64  6.9e-16  
      3332  2^2.7^2.17       flat     replayed        5619       9720    1.72  7.2e-16  
      3335  5.23.29          chain3   replayed       12538      22082    1.68  6.0e-16  
      3344  2^4.11.19        chain3   replayed        4679       9671    2.02  7.0e-16  
      3348  2^2.3^3.31       flat     replayed        7899      13856    1.75  7.4e-16  
      3354  2.3.13.43        chain3   replayed        7781      18674    2.40  6.9e-16  
      3360  2^5.3.5.7        chain3   replayed        4018       5589    1.32  6.6e-16  
      3364  2^2.29^2         chain3   replayed        9279      20084    1.50  6.0e-16  flips differ 1.44x
      3366  2.3^2.11.17      flat     replayed        8479       9800    1.15  8.0e-16  
      3367  7.13.37          chain3   replayed        8288      19823    2.27  7.2e-16  
      3380  2^2.5.13^2       flat     replayed        5744       6442    1.11  8.4e-16  
      3381  3.7^2.23         flat     replayed        8558      12645    1.43  8.2e-16  
      3384  2^3.3^2.47       flat     replayed        8119      20896    2.55  7.8e-16  
      3388  2^2.7.11^2       flat     replayed        5702       6352    1.11  9.4e-16  
      3393  3^2.13.29        flat     replayed        9556      15900    1.65  1.2e-15  
      3400  2^3.5^2.17       chain3   replayed        5412       9476    1.65  6.7e-16  
      3402  2.3^5.7          chain3   replayed        6401       6318    0.87  6.5e-16  
      3404  2^2.23.37        flat     replayed       10786      21171    1.78  1.2e-15  
      3410  2.5.11.31        flat     replayed        8998      15022    1.66  7.4e-16  
      3420  2^2.3^2.5.19     flat     replayed        6517      10192    1.53  5.5e-16  
      3430  2.5.7^3          flat     replayed        7358       6620    0.87  1.0e-15  
      3432  2^3.3.11.13      flat     replayed        6132       6408    1.03  8.0e-16  
      3440  2^4.5.43         chain3   replayed        7140      18271    2.54  5.4e-16  
      3441  3.31.37          flat     replayed       11554      31373    2.30  2.4e-15  
      3444  2^2.3.7.41       flat     replayed        7993      18236    2.15  9.0e-16  
      3450  2.3.5^2.23       chain3   replayed        8134      11416    1.38  9.0e-16  
      3451  7.17.29          chain3   replayed        8639      20251    1.90  6.6e-16  
      3458  2.7.13.19        flat     replayed        8033      11589    1.43  8.0e-16  
      3465  3^2.5.7.11       chain3   replayed        5567       6846    1.23  7.8e-16  
      3468  2^2.3.17^2       flat     replayed        7153      13488    1.84  1.3e-15  
      3472  2^4.7.31         flat     replayed        6911      13675    1.74  7.2e-16  
      3480  2^3.3.5.29       flat     replayed        8652      13635    1.43  1.0e-15  
      3483  3^4.43           chain3   replayed       13193      24769    1.31  7.8e-16  flips differ 1.43x
      3485  5.17.41          chain3   replayed       11074      27896    2.51  6.3e-16  
      3496  2^3.19.23        chain3   replayed        7124      15442    1.95  6.6e-16  
      3500  2^2.5^3.7        flat     replayed        5644       6351    1.12  8.1e-16  
      3509  11^2.29          chain3   replayed        8380      15774    1.74  5.9e-16  
      3510  2.3^3.5.13       chain3   replayed        6482       6172    0.88  8.0e-16  
      3515  5.19.37          chain3   replayed        9649      26038    2.69  6.9e-16  
      3519  3^2.17.23        flat     replayed        7825      18108    2.13  9.6e-16  
      3520  2^6.5.11         chain3   replayed        4253       5693    1.33  4.1e-16  
      3525  3.5^2.47         flat     replayed        8692      27366    3.13  8.2e-16  
      3526  2.41.43          flat     replayed       13104      31527    2.31  2.4e-15  
      3528  2^3.3^2.7^2      chain3   replayed        5569       6393    1.13  7.4e-16  
      3534  2.3.19.31        chain3   replayed       10662      18172    1.44  7.0e-16  
      3536  2^4.13.17        flat     replayed        6525       9824    1.50  9.6e-16  
      3542  2.7.11.23        flat     replayed        9550      13071    1.30  1.1e-15  
      3549  3.7.13^2         chain3   replayed        5834       7365    1.21  7.5e-16  
      3552  2^5.3.37         flat     replayed        8117      16828    1.96  7.2e-16  
      3553  11.17.19         chain3   replayed        6789      15967    2.31  8.5e-16  
      3564  2^2.3^4.11       chain3   replayed        5292       6807    1.23  8.9e-16  
      3565  5.23.31          flat     replayed       10686      24527    1.84  1.7e-15  
      3567  3.29.41          flat     replayed       13386      34470    2.47  2.3e-15  
      3570  2.3.5.7.17       chain3   replayed        7006      10168    1.44  5.7e-16  
      3572  2^2.19.47        chain3   replayed       17188      25935    1.50  6.8e-16  
      3575  5^2.11.13        chain3   replayed        6231       7191    1.03  6.9e-16  
      3584  2^9.7            ztt      replayed        3500       6219    1.76  4.2e-16  
      3588  2^2.3.13.23      chain3   replayed        6490      12954    1.87  6.9e-16  
      3591  3^3.7.19         chain3   replayed        7125      12010    1.68  6.1e-16  
      3596  2^2.29.31        chain3   replayed       13009      22135    1.32  6.8e-16  flips differ 1.29x
      3608  2^3.11.41        chain3   replayed        7712      18961    2.45  5.4e-16  
      3610  2.5.19^2         chain3   replayed        6765      15286    2.23  6.3e-16  
      3612  2^2.3.7.43       flat     replayed        9451      19965    2.10  7.0e-16  
      3619  7.11.47          chain3   replayed       13124      27765    1.64  7.9e-16  flips differ 1.29x
      3625  5^3.29           chain3   replayed       10063      16763    1.50  6.3e-16  
      3626  2.7^2.37         flat     replayed        9665      17974    1.76  6.9e-16  
      3627  3^2.13.31        flat     replayed        8735      17893    2.02  9.3e-16  
      3630  2.3.5.11^2       flat     replayed        7973       6953    0.87  9.1e-16  
      3640  2^3.5.7.13       flat     replayed        5681       6438    1.08  5.5e-16  
      3648  2^6.3.19         chain3   replayed        4816      10497    2.17  6.3e-16  
      3654  2.3^2.7.29       flat     replayed       10325      14571    1.26  8.2e-16  
      3655  5.17.43          chain3   replayed       11017      30449    2.65  5.8e-16  
      3663  3^2.11.37        flat     replayed       10228      21682    2.02  2.2e-15  
      3666  2.3.13.47        chain3   replayed        8703      22783    2.46  7.3e-16  
      3672  2^3.3^3.17       chain3   replayed        6159      10581    1.68  7.7e-16  
      3675  3.5^2.7^2        chain3   replayed        6490       7154    1.09  6.3e-16  
      3680  2^5.5.23         flat     replayed        6634      12374    1.85  5.7e-16  
      3689  7.17.31          chain3   replayed       13685      22568    1.55  7.3e-16  
      3690  2.3^2.5.41       chain3   replayed        7854      19229    2.30  6.6e-16  
      3696  2^4.3.7.11       chain3   replayed        4945       6136    1.23  5.9e-16  
      3698  2.43^2           flat     replayed       14115      33913    2.40  2.4e-15  
      3700  2^2.5^2.37       flat     replayed        7605      17486    2.10  6.4e-16  
      3703  7.23^2           chain3   replayed       11256      21375    1.90  7.0e-16  
      3705  3.5.13.19        chain3   replayed        6609      12362    1.82  8.3e-16  
      3712  2^7.29           flat     replayed        8687      14619    1.48  7.2e-16  
      3718  2.11.13^2        flat     replayed        9555       8286    0.83  1.0e-15  
      3720  2^3.3.5.31       chain3   replayed        6666      15262    2.02  8.1e-16  
      3724  2^2.7^2.19       flat     replayed        6710      11456    1.71  9.0e-16  
      3726  2.3^4.23         chain3   replayed       10031      12824    1.10  7.4e-16  
      3731  7.13.41          chain3   replayed        9617      25027    2.56  6.6e-16  
      3740  2^2.5.11.17      flat     replayed        6459      10925    1.65  6.4e-16  
      3741  3.29.43          flat     replayed       13884      37281    2.67  3.4e-15  
      3744  2^5.3^2.13       chain3   replayed        4759       6638    1.38  6.6e-16  
      3751  11^2.31          flat     replayed        9525      17804    1.85  8.3e-16  
      3757  13.17^2          chain3   replayed        6976      16793    2.40  8.0e-16  
      3760  2^4.5.47         chain3   replayed        8251      22672    2.68  5.1e-16  
      3762  2.3^2.11.19      flat     replayed        9326      11656    1.24  1.0e-15  
      3770  2.5.13.29        chain3   replayed        8795      16368    1.86  6.9e-16  
      3772  2^2.23.41        flat     replayed       11434      25409    2.15  2.4e-15  
      3773  7^3.11           flat     replayed        7304       7164    0.98  8.7e-16  
      3774  2.3.17.37        chain3   replayed        8346      21446    2.56  5.7e-16  
      3780  2^2.3^3.5.7      chain3   replayed        5316       6770    1.27  6.3e-16  
      3784  2^3.11.43        chain3   replayed        8194      21073    2.45  6.5e-16  
      3795  3.5.11.23        chain3   replayed        7848      14388    1.72  6.6e-16  
      3800  2^3.5^2.19       flat     replayed        6322      11253    1.77  7.6e-16  
      3807  3^4.47           chain3   replayed       14913      30010    2.01  8.5e-16  
      3808  2^5.7.17         chain3   replayed        5128      10839    2.06  6.8e-16  
      3813  3.31.41          flat     replayed       13273      37890    2.48  2.5e-15  
      3822  2.3.7^2.13       flat     replayed        8534       6987    0.80  8.2e-16  
      3825  3^2.5^2.17       chain3   replayed        6696      12225    1.76  1.0e-15  
      3828  2^2.3.11.29      chain3   replayed        6913      15494    2.23  6.3e-16  
      3844  2^2.31^2         chain3   replayed       10586      24361    1.85  7.1e-16  
      3848  2^3.13.37        chain3   replayed       11231      18552    1.22  7.6e-16  flips differ 1.35x
      3850  2.5^2.7.11       flat     replayed        7764       7641    0.98  6.3e-16  
      3857  7.19.29          chain3   replayed       11551      23648    2.01  7.0e-16  
      3861  3^3.11.13        flat     replayed        8748       8144    0.93  9.0e-16  
      3864  2^3.3.7.23       flat     replayed        7331      13162    1.77  7.8e-16  
      3870  2.3^2.5.43       chain3   replayed        8286      21370    2.58  6.9e-16  
      3872  2^5.11^2         chain3   replayed        5185       7278    1.34  4.6e-16  
      3875  5^3.31           chain3   replayed        9392      18867    1.72  7.8e-16  
      3876  2^2.3.17.19      chain3   replayed        6277      15747    2.48  7.2e-16  
      3885  3.5.7.37         chain3   replayed        9476      23004    2.41  5.4e-16  
      3887  13^2.23          flat     replayed        8088      14886    1.57  1.0e-15  
      3895  5.19.41          chain3   replayed       12305      32215    2.51  7.0e-16  
      3900  2^2.3.5^2.13     chain3   replayed        5655       6960    1.21  8.1e-16  
      3906  2.3^2.7.31       chain3   replayed        8662      16633    1.87  7.4e-16  
      3910  2.5.17.23        chain3   replayed        8461      17821    2.09  7.8e-16  
      3913  7.13.43          chain3   replayed       14027      27519    1.94  7.6e-16  
      3915  3^3.5.29         flat     replayed       11151      18482    1.64  7.2e-16  
      3920  2^4.5.7^2        ztt      replayed        4241       6700    1.56  4.2e-16  
      3927  3.7.11.17        chain3   replayed        7461      12204    1.63  7.3e-16  
      3933  3^2.19.23        chain3   replayed        8758      21542    2.44  7.4e-16  
      3936  2^5.3.41         chain3   replayed        7586      20621    2.71  5.7e-16  
      3944  2^3.17.29        flat     replayed        9755      19207    1.72  1.1e-15  
      3948  2^2.3.7.47       flat     replayed       12648      24754    1.91  7.0e-16  
      3952  2^4.13.19        chain3   replayed        5850      11667    1.99  6.3e-16  
      3956  2^2.23.43        chain3   replayed       10351      27794    2.34  6.7e-16  
      3960  2^3.3^2.5.11     flat     replayed        6731       7045    1.04  5.0e-16  
      3968  2^7.31           chain3   replayed        7619      16327    1.57  5.7e-16  flips differ 1.37x
      3969  3^4.7^2          chain3   replayed        7078       8302    1.17  8.6e-16  
      3971  11.19^2          chain3   replayed        9222      19122    2.06  6.9e-16  
      3978  2.3^2.13.17      flat     replayed        9720      12109    1.24  1.4e-15  
      3990  2.3.5.7.19       chain3   replayed        6584      12066    1.75  5.9e-16  
      3993  3.11^3           flat     replayed        8023       7668    0.95  9.1e-16  
      3995  5.17.47          chain3   replayed       15292      36308    2.36  7.1e-16  
      3996  2^2.3^3.37       chain3   replayed        7623      19268    2.52  6.6e-16  
      3999  3.31.43          chain3   replayed       18053      41029    2.20  8.3e-16  
      4002  2.3.23.29        flat     replayed       12254      21547    1.42  2.5e-15  
      4004  2^2.7.11.13      flat     replayed        6875       7638    1.09  8.2e-16  
      4018  2.7^2.41         flat     replayed       14356      21848    1.48  3.7e-15  
      4025  5^2.7.23         chain3   replayed        8477      15312    1.80  6.4e-16  
      4030  2.5.13.31        flat     replayed       13154      18313    1.39  8.1e-16  
      4032  2^6.3^2.7        ztt      replayed        4255       6881    1.58  4.0e-16  
      4046  2.7.17^2         flat     replayed       10484      16148    1.51  1.1e-15  
      4048  2^4.11.23        chain3   replayed        6357      13501    1.83  7.6e-16  
      4056  2^3.3.13^2       flat     replayed        7322       7794    1.06  9.7e-16  
      4059  3^2.11.41        chain3   replayed       12424      27376    1.62  6.1e-16  flips differ 1.36x
      4060  2^2.5.7.29       flat     replayed       10166      16245    1.46  6.6e-16  
      4070  2.5.11.37        chain3   replayed       16439      20687    1.24  5.4e-16  
      4080  2^4.3.5.17       chain3   replayed        5737      11189    1.95  5.2e-16  
      4085  5.19.43          flat     replayed       12856      35189    2.73  1.4e-15  
      4089  3.29.47          chain3   replayed       19274      43857    1.95  6.8e-16  
      4092  2^2.3.11.31      chain3   replayed        7634      17394    1.98  6.3e-16  
      4095  3^2.5.7.13       chain3   replayed        7535       8457    1.12  6.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime      759      0     60   1.08   1.25   2.35    1.40
 chain3     593      0     17   1.11   1.58   2.36    1.60
 flat       436      3     38   1.02   1.69   2.50    1.64
 2p         238      1      1   1.30   2.03   3.10    1.97
 mono        15      0      2   0.97   1.60   1.75    1.44
 ztt          9      0      0   1.49   1.56   1.77    1.59
 ALL       2050      4    118   1.09   1.48   2.48    1.57
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   1   1.57      0      0
 8..31                 18   1.40      3      1
 32..127               90   1.60      2      0
 128..511             378   1.56      5      0
 512..2047           1080   1.33     77      0
 2048..4095           483   1.68     31      3
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                         593   1.58     17   1.60
 composite with a prime >= 53 (prime cell)      457   1.24     37   1.39
 flat                                           436   1.69     38   1.64
 2p                                             238   2.03      1   1.97
 prime N, bluestein                             188   1.16     21   1.17
 prime N, rader                                 107   1.94      0   1.98
 mono                                            15   1.60      2   1.44
 ztt                                              9   1.56      0   1.59
 composite, prime cell by race                    7   1.26      2   1.37
```


flip agreement: our two readings more than 25% apart at 84 of 2050 cells.

worst 10: 15 (2p 0.52), 2574 (flat 0.71), 3042 (flat 0.79), 2178 (flat 0.79), 3822 (flat 0.80), 82 (prime 0.83), 2662 (flat 0.83), 3718 (flat 0.83), 2730 (flat 0.83), 168 (chain3 0.84)
best 5: 89 (prime 7.04), 79 (prime 5.22), 73 (prime 5.21), 71 (prime 4.85), 254 (prime 4.45)
