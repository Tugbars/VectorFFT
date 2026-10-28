# gauntlet report

run: `fftw_avx512_2026-10-22_1d`  contract file suffix: `_fftw`  cells: 4095 listed, 4095 benched, comparator: FFTW

control cell: 84 readings, 0.421..2.268 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     raced              6          7    0.84  0.0e+00  flips differ 1.50x
         3  3                mono     raced              8         10    1.14  1.5e-16  
         4  2^2              mono     raced              9          9    0.91  0.0e+00  
         5  5                mono     raced             11         13    1.09  1.5e-16  
         6  2.3              mono     raced             13         12    0.71  2.8e-16  
         7  7                mono     raced             16         15    0.92  2.6e-16  
         8  2^3              mono     raced             14         13    0.90  5.6e-17  
         9  3^2              mono     raced             20         17    0.85  4.2e-16  
        10  2.5              mono     raced             20         16    0.78  1.8e-16  
        11  11               mono     raced             26         20    0.75  1.6e-16  
        12  2^2.3            mono     raced             20         17    0.82  2.8e-16  
        13  13               mono     raced             34         24    0.67  6.0e-16  
        14  2.7              mono     raced             28         20    0.72  1.6e-16  
        15  3.5              2p       raced             24         23    0.95  3.3e-16  
        16  2^4              2p       raced             23         20    0.86  9.4e-17  
        17  17               mono     raced             51         74    1.43  2.4e-16  
        18  2.3^2            2p       raced             28         33    1.00  3.6e-16  
        19  19               mono     raced             64         86    1.35  3.9e-16  
        20  2^2.5            2p       raced             28         26    0.91  2.2e-16  
        21  3.7              2p       raced             35         50    1.13  4.1e-16  flips differ 1.26x
        22  2.11             mono     raced             46         42    0.90  2.5e-16  
        23  23               mono     raced             83        119    1.37  3.1e-16  
        24  2^3.3            2p       raced             28         53    1.89  3.3e-16  
        25  5^2              2p       raced             39         52    1.33  1.8e-16  
        26  2.13             mono     raced             61         48    0.71  4.4e-16  
        27  3^3              2p       raced             45         62    1.35  4.5e-16  
        28  2^2.7            2p       raced             40         63    1.55  1.8e-16  
        29  29               mono     raced            120        171    1.33  3.7e-16  
        30  2.3.5            2p       raced             38         45    1.15  3.0e-16  
        31  31               mono     raced            180        189    1.01  2.7e-16  
        32  2^5              2p       raced             30         37    1.21  1.3e-16  
        33  3.11             2p       raced             59         77    1.21  3.3e-16  
        34  2.17             flat     raced             87        174    1.86  3.7e-16  
        35  5.7              2p       raced             54         73    1.35  3.9e-16  
        36  2^2.3^2          2p       raced             45         46    0.79  2.4e-16  flips differ 1.31x
        37  37               mono     raced            245        249    0.96  3.6e-16  
        38  2.19             flat     raced            109        201    1.83  3.5e-16  
        39  3.13             2p       raced             68         87    1.28  5.3e-16  
        40  2^3.5            2p       raced             43         39    0.92  2.0e-16  
        41  41               mono     raced            301        298    0.97  4.1e-16  
        42  2.3.7            2p       raced             57         61    1.08  3.6e-16  
        43  43               mono     raced            343        327    0.93  3.6e-16  
        44  2^2.11           2p       raced             54         76    1.40  3.5e-16  
        45  3^2.5            2p       raced             60         94    1.56  4.6e-16  
        46  2.23             flat     raced            131        268    1.97  3.9e-16  
        47  47               mono     raced            478        378    0.76  4.0e-16  
        48  2^4.3            2p       raced             48         47    0.97  3.7e-16  
        49  7^2              2p       raced             76        101    1.32  2.5e-16  
        50  2.5^2            2p       raced             58         72    1.20  2.7e-16  
        51  3.17             2p       raced            102        264    2.55  3.5e-16  
        52  2^2.13           2p       raced             73         90    1.19  6.2e-16  
        53  53               prime    raced            319        500    1.45  5.8e-16  bluestein
        54  2.3^3            2p       raced             67         79    1.15  4.8e-16  
        55  5.11             2p       raced             79        116    1.42  4.2e-16  
        56  2^3.7            2p       raced             63         64    0.98  2.5e-16  
        57  3.19             2p       raced            136        304    2.22  5.4e-16  
        58  2.29             flat     raced            175        375    2.05  5.0e-16  
        59  59               prime    raced            285        520    1.62  4.9e-16  bluestein
        60  2^2.3.5          2p       raced             65         67    1.03  3.4e-16  
        61  61               prime    raced            285        611    2.12  3.7e-16  bluestein
        62  2.31             flat     raced            203        415    1.99  4.4e-16  
        63  3^2.7            2p       raced             83        125    1.51  3.5e-16  
        64  2^6              mono     raced             47         61    1.19  3.8e-16  
        65  5.13             2p       raced             93        130    1.39  4.7e-16  
        66  2.3.11           2p       raced             83        111    1.34  4.5e-16  
        67  67               prime    raced            350        727    2.01  9.2e-16  bluestein
        68  2^2.17           2p       raced             96        345    3.16  4.3e-16  
        69  3.23             2p       raced            205        407    1.98  4.1e-16  
        70  2.5.7            2p       raced             85         99    1.03  2.3e-16  
        71  71               prime    raced            370        975    2.51  4.5e-16  rader
        72  2^3.3^2          2p       raced             71        155    2.18  3.5e-16  
        73  73               prime    raced            340        864    2.50  4.8e-16  rader
        74  2.37             flat     raced            376        555    1.18  3.2e-16  
        75  3.5^2            2p       raced             91        140    1.52  2.5e-16  
        76  2^2.19           2p       raced            125        450    2.95  2.4e-16  
        77  7.11             2p       raced            108        147    1.25  3.6e-16  
        78  2.3.13           2p       raced            108        130    1.18  4.4e-16  
        79  79               prime    raced            411        736    1.73  7.4e-16  bluestein
        80  2^4.5            2p       raced             71         67    0.92  2.8e-16  
        81  3^4              2p       raced             97        155    1.55  4.3e-16  
        82  2.41             flat     raced            455        649    1.12  4.0e-16  flips differ 1.29x
        83  83               prime    raced            456       1120    2.38  5.6e-16  bluestein
        84  2^2.3.7          2p       raced             98         97    0.89  2.7e-16  
        85  5.17             2p       raced            138        443    3.16  4.0e-16  
        86  2.43             flat     raced            491        703    1.42  3.8e-16  
        87  3.29             flat     raced            299        567    1.89  5.0e-16  
        88  2^3.11           2p       raced             91        149    1.45  2.8e-16  
        89  89               prime    raced            454        887    1.93  4.2e-16  bluestein
        90  2.3^2.5          2p       raced            100        118    1.18  3.7e-16  
        91  7.13             2p       raced            130        173    1.21  5.3e-16  
        92  2^2.23           2p       raced            164        534    2.43  3.6e-16  flips differ 1.34x
        93  3.31             2p       raced            360        629    1.15  3.8e-16  flips differ 1.52x
        94  2.47             flat     raced            567        814    1.38  4.6e-16  
        95  5.19             2p       raced            173        504    2.54  3.0e-16  
        96  2^5.3            2p       raced             83         81    0.91  3.9e-16  
        97  97               prime    raced            422        870    1.90  5.4e-16  rader
        98  2.7^2            flat     raced            234        142    0.59  2.8e-16  
        99  3^2.11           2p       raced            122        195    1.50  4.1e-16  
       100  2^2.5^2          2p       raced             98        107    1.08  2.8e-16  
       101  101              prime    raced            489        929    1.89  4.5e-16  rader
       102  2.3.17           2p       raced            171        519    2.93  4.3e-16  
       103  103              prime    raced            589       1606    2.71  5.8e-16  bluestein
       104  2^3.13           2p       raced            110        166    1.46  5.6e-16  
       105  3.5.7            2p       raced            128        188    1.46  3.4e-16  
       106  2.53             prime    raced            564       1014    1.78  3.3e-16  bluestein
       107  107              prime    raced            566       1926    3.30  6.2e-16  bluestein
       108  2^2.3^3          2p       raced            108        203    1.28  4.2e-16  
       109  109              prime    raced            697       1169    1.48  6.4e-16  bluestein
       110  2.5.11           2p       raced            127        179    1.39  3.2e-16  
       111  3.37             flat     raced            792        826    1.03  4.0e-16  
       112  2^4.7            2p       raced            114        115    0.91  2.9e-16  
       113  113              prime    raced            536       1408    2.57  5.3e-16  rader
       114  2.3.19           2p       raced            201        599    2.52  3.9e-16  
       115  5.23             2p       raced            239        675    2.26  4.1e-16  flips differ 1.26x
       116  2^2.29           2p       raced            302        750    2.38  4.4e-16  
       117  3^2.13           2p       raced            145        218    1.26  5.2e-16  
       118  2.59             prime    raced            566       1215    2.09  2.8e-16  bluestein
       119  7.17             2p       raced            189        613    3.24  4.1e-16  
       120  2^3.3.5          2p       raced            118        100    0.84  3.7e-16  
       121  11^2             2p       raced            160        354    2.21  3.7e-16  
       122  2.61             prime    raced            571       1309    2.27  3.2e-16  bluestein
       123  3.41             2p       raced            713        983    1.37  4.2e-16  
       124  2^2.31           2p       raced            268        838    2.79  4.3e-16  
       125  5^3              2p       raced            147        288    1.86  4.8e-16  
       126  2.3^2.7          2p       raced            160        182    1.13  2.8e-16  
       127  127              prime    raced            573        976    1.69  3.8e-16  bluestein
       128  2^7              2p       raced             94        105    1.06  2.1e-16  
       129  3.43             2p       raced            824       1070    1.29  4.4e-16  
       130  2.5.13           2p       raced            157        215    1.34  3.8e-16  
       131  131              prime    raced            688       1475    2.11  5.0e-16  rader
       132  2^2.3.11         2p       raced            144        243    1.58  4.6e-16  
       133  7.19             2p       raced            244        721    2.88  4.2e-16  
       134  2.67             prime    raced            788       1549    1.94  7.2e-16  bluestein
       135  3^3.5            2p       raced            150        237    1.52  3.8e-16  
       136  2^3.17           2p       raced            172        691    3.98  6.0e-16  
       137  137              prime    raced            721       1389    1.88  6.6e-16  rader
       138  2.3.23           2p       raced            283        852    2.77  4.8e-16  
       139  139              prime    raced            777       1389    1.75  7.9e-16  bluestein
       140  2^2.5.7          chain3   raced            213        151    0.69  3.6e-16  
       141  3.47             2p       raced            788       1255    1.56  3.6e-16  
       142  2.71             prime    raced            797       1462    1.45  1.3e-15  flips differ 1.26x bluestein
       143  11.13            2p       raced            190        416    2.18  5.1e-16  
       144  2^4.3^2          2p       raced            121        315    2.58  2.9e-16  
       145  5.29             2p       raced            408        950    2.30  4.0e-16  
       146  2.73             prime    raced            881       1807    2.01  6.7e-16  bluestein
       147  3.7^2            2p       raced            195        396    2.02  3.5e-16  
       148  2^2.37           2p       raced            314       1090    3.33  4.2e-16  
       149  149              prime    raced            858       1499    1.71  5.1e-16  bluestein
       150  2.3.5^2          2p       raced            173        203    1.13  3.5e-16  
       151  151              prime    raced            803       1181    1.43  5.7e-16  rader
       152  2^3.19           2p       raced            221        795    3.57  4.1e-16  
       153  3^2.17           2p       raced            228        794    3.46  4.2e-16  
       154  2.7.11           flat     raced            380        371    0.97  2.8e-16  
       155  5.31             2p       raced            468       1056    2.21  3.6e-16  
       156  2^2.3.13         2p       raced            172        294    1.70  6.5e-16  
       157  157              prime    raced            757       1168    1.51  8.8e-16  rader
       158  2.79             prime    raced            874       1538    1.75  4.9e-16  bluestein
       159  3.53             prime    raced            862       1529    1.72  4.5e-16  bluestein
       160  2^5.5            2p       raced            137        131    0.96  3.0e-16  
       161  7.23             2p       raced            407        963    2.33  4.5e-16  
       162  2.3^4            2p       raced            207        304    1.46  3.8e-16  
       163  163              prime    raced           1095       1745    1.58  6.1e-16  bluestein
       164  2^2.41           2p       raced            365       1293    3.32  4.0e-16  
       165  3.5.11           2p       raced            207        350    1.58  3.8e-16  
       166  2.83             prime    raced           1064       2299    2.13  5.5e-16  bluestein
       167  167              prime    raced           1085       1983    1.44  4.6e-16  bluestein
       168  2^3.3.7          2p       raced            164        189    1.08  2.6e-16  
       169  13^2             2p       raced            225        499    1.60  6.0e-16  flips differ 1.38x
       170  2.5.17           2p       raced            229        952    3.16  7.2e-16  flips differ 1.40x
       171  3^2.19           2p       raced            278        917    2.96  4.7e-16  
       172  2^2.43           2p       raced            411       1476    3.56  7.6e-16  
       173  173              prime    raced           1162       1718    1.40  4.3e-16  bluestein
       174  2.3.29           2p       raced            371       1133    2.84  5.5e-16  
       175  5^2.7            2p       raced            238        361    1.24  4.5e-16  
       176  2^4.11           2p       raced            164        300    1.80  2.9e-16  
       177  3.59             prime    raced            926       1855    1.97  8.5e-16  bluestein
       178  2.89             prime    raced            944       1859    1.94  6.9e-16  bluestein
       179  179              prime    raced            938       1988    1.72  7.1e-16  bluestein
       180  2^2.3^2.5        2p       raced            175        427    2.44  3.6e-16  
       181  181              prime    raced            891       2119    2.37  6.1e-16  rader
       182  2.7.13           flat     raced            440        433    0.84  4.7e-16  
       183  3.61             prime    raced            953       1970    2.04  6.4e-16  bluestein
       184  2^3.23           2p       raced            346       1085    2.81  5.6e-16  
       185  5.37             2p       raced            617       1404    2.22  4.8e-16  
       186  2.3.31           2p       raced            416       1257    1.59  6.3e-16  flips differ 1.89x
       187  11.17            2p       raced            288       1158    3.87  6.0e-16  
       188  2^2.47           2p       raced            446       1639    3.24  4.5e-16  
       189  3^3.7            2p       raced            223        500    2.17  3.6e-16  
       190  2.5.19           2p       raced            307       1003    2.87  3.6e-16  
       191  191              prime    raced            946       1370    1.43  7.6e-16  bluestein
       192  2^6.3            2p       raced            149        155    1.02  3.4e-16  
       193  193              prime    raced            840       1644    1.93  5.2e-16  rader
       194  2.97             prime    raced           1167       1772    1.51  6.2e-16  bluestein
       195  3.5.13           2p       raced            246        417    1.69  5.7e-16  
       196  2^2.7^2          chain3   raced            326        342    1.02  2.8e-16  
       197  197              prime    raced           1168       1645    1.33  5.3e-16  bluestein
       198  2.3^2.11         chain3   raced            296        420    1.38  5.3e-16  
       199  199              prime    raced           1173       1617    1.37  4.7e-16  bluestein
       200  2^3.5^2          2p       raced            187        182    0.95  3.2e-16  
       201  3.67             prime    raced           1170       2316    1.93  4.8e-16  bluestein
       202  2.101            prime    raced           1167       2002    1.67  3.6e-16  bluestein
       203  7.29             2p       raced            530       1368    2.48  4.5e-16  
       204  2^2.3.17         2p       raced            258       1116    4.14  4.8e-16  
       205  5.41             2p       raced            732       1635    2.22  4.3e-16  
       206  2.103            prime    raced           1174       3542    2.99  5.6e-16  bluestein
       207  3^2.23           2p       raced            378       1238    2.62  4.7e-16  
       208  2^4.13           2p       raced            199        363    1.67  5.0e-16  
       209  11.19            2p       raced            361       1326    3.35  3.6e-16  
       210  2.3.5.7          2p       raced            242        283    1.16  4.1e-16  
       211  211              prime    raced           1184       3117    2.62  5.6e-16  bluestein
       212  2^2.53           prime    raced           1181       2039    1.61  3.8e-16  bluestein
       213  3.71             prime    raced           1172       2212    1.87  4.0e-16  bluestein
       214  2.107            prime    raced           1172       3876    2.81  6.1e-16  bluestein
       215  5.43             2p       raced            827       1786    2.13  5.2e-16  
       216  2^3.3^3          2p       raced            212        802    3.77  3.3e-16  
       217  7.31             2p       raced            740       1512    1.99  4.4e-16  
       218  2.109            prime    raced           1182       2538    1.83  4.8e-16  bluestein
       219  3.73             prime    raced           1197       2284    1.90  5.1e-16  bluestein
       220  2^2.5.11         chain3   raced            300        428    1.29  3.1e-16  
       221  13.17            2p       raced            332       1383    4.12  5.8e-16  
       222  2.3.37           2p       raced            561       1672    2.58  3.9e-16  
       223  223              prime    raced           1184       2398    1.86  6.9e-16  bluestein
       224  2^5.7            2p       raced            247        239    0.96  3.9e-16  
       225  3^2.5^2          2p       raced            260        467    1.73  5.6e-16  
       226  2.113            prime    raced           1179       2970    2.46  5.5e-16  bluestein
       227  227              prime    raced           1186       2052    1.63  5.6e-16  bluestein
       228  2^2.3.19         2p       raced            371       1311    3.44  5.5e-16  
       229  229              prime    raced           1208       2043    1.63  4.6e-16  bluestein
       230  2.5.23           2p       raced            392       1350    3.36  5.4e-16  
       231  3.7.11           2p       raced            291        585    1.99  3.5e-16  
       232  2^3.29           2p       raced            417       1527    3.35  3.5e-16  
       233  233              prime    raced           1184       2037    1.71  4.8e-16  bluestein
       234  2.3^2.13         chain3   raced            359        494    1.31  4.5e-16  
       235  5.47             2p       raced           1017       2072    2.00  4.8e-16  
       236  2^2.59           prime    raced           1199       2459    1.98  3.0e-16  bluestein
       237  3.79             prime    raced           1183       2406    2.00  6.0e-16  bluestein
       238  2.7.17           flat     raced            603       1348    2.23  5.5e-16  
       239  239              prime    raced           1192       2057    1.67  4.9e-16  bluestein
       240  2^4.3.5          2p       raced            208        208    0.97  3.4e-16  
       241  241              prime    raced           1184       2091    1.76  4.0e-16  bluestein
       242  2.11^2           flat     raced            566        827    1.41  3.3e-16  
       243  3^5              2p       raced            279        620    2.20  4.9e-16  
       244  2^2.61           prime    raced           1188       2615    1.66  3.6e-16  flips differ 1.32x bluestein
       245  5.7^2            chain3   raced            424        674    1.56  2.8e-16  
       246  2.3.41           2p       raced            794       1954    1.98  3.0e-16  
       247  13.19            2p       raced            422       1582    2.62  4.7e-16  flips differ 1.44x
       248  2^3.31           2p       raced            526       1687    3.19  5.1e-16  
       249  3.83             prime    raced           1186       3622    2.92  4.3e-16  bluestein
       250  2.5^3            2p       raced            308        421    1.27  4.4e-16  
       251  251              prime    raced           1193       1816    1.51  5.4e-16  bluestein
       252  2^2.3^2.7        2p       raced            264        678    2.54  3.9e-16  
       253  11.23            2p       raced            474       1748    3.33  5.3e-16  
       254  2.127            prime    raced           1190       2112    1.73  4.3e-16  bluestein
       255  3.5.17           2p       raced            344       1425    4.03  5.5e-16  
       256  2^8              2p       raced            177        215    1.14  2.5e-16  
       257  257              prime    raced           1087       2212    1.98  3.5e-16  rader
       258  2.3.43           2p       raced            839       2344    2.58  4.8e-16  
       259  7.37             2p       raced            773       1941    1.96  4.1e-16  flips differ 1.29x
       260  2^2.5.13         chain3   raced            366        487    1.32  4.3e-16  
       261  3^2.29           2p       raced            605       1733    2.80  4.2e-16  
       262  2.131            prime    raced           1856       3168    1.70  7.3e-16  bluestein
       263  263              prime    raced           1831       2877    1.40  6.2e-16  bluestein
       264  2^3.3.11         chain3   raced            357        534    1.45  4.8e-16  
       265  5.53             prime    raced           1806       2554    1.30  5.7e-16  bluestein
       266  2.7.19           flat     raced            778       1543    1.86  3.5e-16  
       267  3.89             prime    raced           1887       2785    1.42  8.4e-16  bluestein
       268  2^2.67           flat     raced           1750       3122    1.72  6.2e-16  
       269  269              prime    raced           1870       2842    1.50  8.0e-16  bluestein
       270  2.3^3.5          2p       raced            327        473    1.44  4.0e-16  
       271  271              prime    raced           1633       2833    1.64  7.0e-16  rader
       272  2^4.17           2p       raced            307       1443    4.56  4.6e-16  
       273  3.7.13           2p       raced            337        696    1.78  5.1e-16  
       274  2.137            prime    raced           1892       2859    1.40  5.4e-16  bluestein
       275  5^2.11           2p       raced            353        764    2.03  4.0e-16  
       276  2^2.3.23         2p       raced            466       1856    3.12  5.6e-16  
       277  277              prime    raced           1799       3988    1.85  6.2e-16  rader
       278  2.139            prime    raced           1863       2863    1.47  6.5e-16  bluestein
       279  3^2.31           2p       raced            689       1919    2.41  4.3e-16  
       280  2^3.5.7          chain3   raced            385        309    0.80  3.4e-16  
       281  281              prime    raced           1597       3978    2.47  5.2e-16  rader
       282  2.3.47           2p       raced            994       2464    2.33  4.3e-16  
       283  283              prime    raced           1826       2514    1.31  7.3e-16  bluestein
       284  2^2.71           flat     raced           1638       3485    1.72  5.5e-16  
       285  3.5.19           2p       raced            484       1647    3.00  4.3e-16  
       286  2.11.13          flat     raced            668        954    1.32  4.9e-16  
       287  7.41             2p       raced            953       2309    1.78  3.2e-16  flips differ 1.35x
       288  2^5.3^2          2p       raced            279        545    1.31  4.4e-16  
       289  17^2             2p       raced            492       2822    5.61  7.8e-16  
       290  2.5.29           2p       raced            545       1924    3.27  4.4e-16  
       291  3.97             prime    raced           2545       3124    1.22  4.7e-16  bluestein
       292  2^2.73           flat     raced           1546       3135    1.83  5.4e-16  
       293  293              prime    raced           2541       2617    1.01  4.6e-16  bluestein
       294  2.3.7^2          chain3   raced            524        553    1.04  4.9e-16  
       295  5.59             prime    raced           1896       3099    1.54  4.7e-16  bluestein
       296  2^3.37           2p       raced            582       2229    3.50  3.7e-16  
       297  3^3.11           2p       raced            396        729    1.81  4.7e-16  
       298  2.149            prime    raced           1952       3310    1.64  6.2e-16  bluestein
       299  13.23            2p       raced            555       2099    3.08  4.0e-16  
       300  2^2.3.5^2        2p       raced            322        369    1.13  5.1e-16  
       301  7.43             2p       raced           1446       2510    1.66  5.2e-16  
       302  2.151            prime    raced           1985       2518    1.24  5.7e-16  bluestein
       303  3.101            prime    raced           1926       2976    1.54  4.8e-16  bluestein
       304  2^4.19           2p       raced            421       1654    3.27  3.3e-16  
       305  5.61             prime    raced           1921       3280    1.60  4.8e-16  bluestein
       306  2.3^2.17         chain3   raced            510       1769    3.41  5.7e-16  
       307  307              prime    raced           1929       4779    2.46  6.9e-16  bluestein
       308  2^2.7.11         chain3   raced            465        690    1.47  3.4e-16  
       309  3.103            prime    raced           1964       4257    1.70  6.6e-16  bluestein
       310  2.5.31           2p       raced            708       2114    2.88  4.5e-16  
       311  311              prime    raced           1929       4727    2.34  5.6e-16  bluestein
       312  2^3.3.13         chain3   raced            464        631    1.31  4.9e-16  
       313  313              prime    raced           1933       4718    2.41  8.4e-16  rader
       314  2.157            prime    raced           1922       2487    1.27  5.1e-16  bluestein
       315  3^2.5.7          2p       raced            383        753    1.95  4.2e-16  
       316  2^2.79           flat     raced           2158       3133    1.37  6.4e-16  
       317  317              prime    raced           1928       2378    1.19  6.3e-16  bluestein
       318  2.3.53           prime    raced           1932       3050    1.54  5.9e-16  bluestein
       319  11.29            2p       raced            935       2405    2.50  4.5e-16  
       320  2^6.5            chain3   raced            326        275    0.74  2.5e-16  
       321  3.107            prime    raced           2245       4260    1.56  6.4e-16  bluestein
       322  2.7.23           flat     raced           1095       2091    1.88  4.0e-16  
       323  17.19            2p       raced            607       3208    4.69  5.8e-16  
       324  2^2.3^4          2p       raced            356        732    1.57  4.9e-16  
       325  5^2.13           2p       raced            417        869    1.98  6.4e-16  
       326  2.163            prime    raced           2243       3684    1.59  4.8e-16  bluestein
       327  3.109            prime    raced           2199       3854    1.57  6.6e-16  bluestein
       328  2^3.41           2p       raced            683       2587    3.62  3.4e-16  
       329  7.47             2p       raced           1262       2918    1.64  4.5e-16  flips differ 1.42x
       330  2.3.5.11         chain3   raced            490        676    1.30  4.5e-16  
       331  331              prime    raced           1990       4478    2.24  6.7e-16  rader
       332  2^2.83           prime    raced           2276       4613    2.00  7.0e-16  bluestein
       333  3^2.37           2p       raced            906       2507    2.69  4.5e-16  
       334  2.167            prime    raced           2211       4134    1.51  5.3e-16  bluestein
       335  5.67             prime    raced           2224       3954    1.76  5.0e-16  bluestein
       336  2^4.3.7          2p       raced            309        503    1.58  3.6e-16  
       337  337              prime    raced           1772       4516    2.54  5.8e-16  rader
       338  2.13^2           flat     raced            816       1163    1.12  5.3e-16  flips differ 1.30x
       339  3.113            prime    raced           2591       4421    1.66  4.6e-16  bluestein
       340  2^2.5.17         chain3   raced            523       1839    3.47  6.2e-16  
       341  11.31            2p       raced            956       2656    2.69  5.6e-16  
       342  2.3^2.19         chain3   raced            642       2080    2.74  5.1e-16  
       343  7^3              chain3   raced            667        951    1.42  2.8e-16  
       344  2^3.43           2p       raced            786       2906    3.15  4.4e-16  
       345  3.5.23           2p       raced            726       2246    3.05  4.4e-16  
       346  2.173            prime    raced           2514       4892    1.64  5.1e-16  bluestein
       347  347              prime    raced           2728       4784    1.74  6.5e-16  bluestein
       348  2^2.3.29         2p       raced            671       2383    3.54  5.5e-16  
       349  349              prime    raced           2372       3980    1.24  6.9e-16  rader
       350  2.5^2.7          chain3   raced            516        572    1.10  2.6e-16  
       351  3^3.13           2p       raced            446        866    1.87  5.7e-16  
       352  2^5.11           2p       raced            364        652    1.76  2.6e-16  
       353  353              prime    raced           1753       3935    1.81  4.9e-16  rader
       354  2.3.59           prime    raced           2564       3874    1.44  3.8e-16  bluestein
       355  5.71             prime    raced           2539       3830    1.50  4.5e-16  bluestein
       356  2^2.89           flat     raced           1907       3713    1.94  4.8e-16  
       357  3.7.17           2p       raced            510       2155    4.02  4.4e-16  
       358  2.179            prime    raced           2528       4264    1.41  5.7e-16  bluestein
       359  359              prime    raced           2519       3936    1.11  5.3e-16  bluestein
       360  2^3.3^2.5        chain3   raced            466        851    1.76  4.4e-16  
       361  19^2             2p       raced            763       3667    4.65  4.5e-16  
       362  2.181            prime    raced           2209       4592    1.42  6.8e-16  bluestein
       363  3.11^2           chain3   raced            601       2026    2.20  3.8e-16  
       364  2^2.7.13         chain3   raced            548        979    1.70  4.9e-16  
       365  5.73             prime    raced           2211       3906    1.67  6.6e-16  bluestein
       366  2.3.61           prime    raced           2131       3935    1.80  5.2e-16  bluestein
       367  367              prime    raced           2161       4026    1.60  7.5e-16  bluestein
       368  2^4.23           2p       raced            550       2233    3.31  3.8e-16  
       369  3^2.41           2p       raced           1127       3039    2.65  4.5e-16  
       370  2.5.37           2p       raced            944       2774    2.59  4.0e-16  
       371  7.53             prime    raced           2110       3594    1.65  6.1e-16  bluestein
       372  2^2.3.31         2p       raced            793       2629    3.24  5.3e-16  
       373  373              prime    raced           2207       4278    1.80  6.9e-16  bluestein
       374  2.11.17          flat     raced            988       2501    2.52  4.7e-16  
       375  3.5^3            2p       raced            480        909    1.88  4.5e-16  
       376  2^3.47           2p       raced            880       3309    3.71  4.6e-16  
       377  13.29            2p       raced            870       2880    3.29  4.9e-16  
       378  2.3^3.7          chain3   raced            539        736    1.22  4.4e-16  
       379  379              prime    raced           2088       2874    1.27  7.0e-16  bluestein
       380  2^2.5.19         chain3   raced            728       2103    2.73  4.0e-16  
       381  3.127            prime    raced           2162       3255    1.49  5.8e-16  bluestein
       382  2.191            prime    raced           2185       2966    1.31  7.8e-16  bluestein
       383  383              prime    raced           2134       2995    1.16  6.9e-16  bluestein
       384  2^7.3            2p       raced            327        316    0.96  3.5e-16  
       385  5.7.11           chain3   raced            619        996    1.58  4.3e-16  
       386  2.193            prime    raced           2592       3413    1.27  5.9e-16  bluestein
       387  3^2.43           2p       raced           1210       3285    2.58  4.5e-16  
       388  2^2.97           flat     raced           2600       3618    1.28  5.7e-16  
       389  389              prime    raced           2703       3377    1.15  4.5e-16  bluestein
       390  2.3.5.13         chain3   raced            610        836    1.33  6.2e-16  
       391  17.23            2p       raced            815       4107    5.03  6.4e-16  
       392  2^3.7^2          chain3   raced            624        901    1.04  2.9e-16  
       393  3.131            prime    raced           2670       4718    1.76  7.4e-16  bluestein
       394  2.197            prime    raced           2598       3742    1.27  6.4e-16  bluestein
       395  5.79             prime    raced           2548       3946    1.46  6.6e-16  bluestein
       396  2^2.3^2.11       chain3   raced            537        848    1.55  4.2e-16  
       397  397              prime    raced           2758       3438    1.23  6.0e-16  bluestein
       398  2.199            prime    raced           2564       3452    1.33  4.5e-16  bluestein
       399  3.7.19           2p       raced            722       2392    2.89  5.7e-16  
       400  2^4.5^2          2p       raced            367        386    0.98  2.9e-16  
       401  401              prime    raced           2074       3584    1.71  5.2e-16  rader
       402  2.3.67           prime    raced           2610       4736    1.77  4.3e-16  bluestein
       403  13.31            2p       raced            981       3154    2.24  6.4e-16  flips differ 1.44x
       404  2^2.101          flat     raced           3201       4258    1.24  5.9e-16  
       405  3^4.5            2p       raced            520        951    1.80  5.5e-16  
       406  2.7.29           flat     raced           1941       2845    1.42  5.6e-16  
       407  11.37            flat     raced           1336       3464    2.59  3.7e-16  
       408  2^3.3.17         chain3   raced            638       2433    3.47  5.3e-16  
       409  409              prime    raced           2591       5186    1.61  4.3e-16  flips differ 1.27x bluestein
       410  2.5.41           2p       raced           1127       3265    2.74  4.9e-16  
       411  3.137            prime    raced           2560       4399    1.68  6.2e-16  bluestein
       412  2^2.103          prime    raced           2559       5691    1.83  6.6e-16  bluestein
       413  7.59             prime    raced           3136       4364    1.38  7.5e-16  bluestein
       414  2.3^2.23         chain3   raced            884       2617    2.55  6.1e-16  
       415  5.83             prime    raced           2561       5801    1.88  4.7e-16  bluestein
       416  2^5.13           2p       raced            418        759    1.80  4.9e-16  
       417  3.139            prime    raced           2652       4372    1.34  4.8e-16  bluestein
       418  2.11.19          flat     raced           1337       2905    2.09  4.5e-16  
       419  419              prime    raced           2572       5180    1.59  4.8e-16  bluestein
       420  2^2.3.5.7        chain3   raced            563        671    1.18  4.3e-16  
       421  421              prime    raced           2555       5173    1.55  6.2e-16  bluestein
       422  2.211            prime    raced           2576       6440    2.45  5.9e-16  bluestein
       423  3^2.47           2p       raced           1480       3752    2.47  5.3e-16  
       424  2^3.53           flat     raced           1804       4076    2.25  6.7e-16  
       425  5^2.17           2p       raced            617       2507    3.86  5.9e-16  
       426  2.3.71           prime    raced           2588       4638    1.76  6.4e-16  bluestein
       427  7.61             flat     raced           2848       4635    1.54  4.3e-16  
       428  2^2.107          prime    raced           2877       5807    1.66  5.4e-16  bluestein
       429  3.11.13          chain3   raced            708       1509    2.05  5.2e-16  
       430  2.5.43           2p       raced           1215       3581    2.37  3.4e-16  
       431  431              prime    raced           2576       4165    1.41  4.3e-16  bluestein
       432  2^4.3^3          2p       raced            418       1175    2.73  4.2e-16  
       433  433              prime    raced           2325       5208    1.78  5.7e-16  rader
       434  2.7.31           flat     raced           1940       4284    1.89  4.6e-16  flips differ 1.32x
       435  3.5.29           2p       raced           1062       3069    2.57  4.8e-16  
       436  2^2.109          flat     raced           3189       5057    1.33  5.4e-16  
       437  19.23            2p       raced           1248       4700    2.91  4.7e-16  flips differ 1.30x
       438  2.3.73           prime    raced           2562       5393    2.07  4.6e-16  bluestein
       439  439              prime    raced           2673       4175    1.38  5.8e-16  bluestein
       440  2^3.5.11         chain3   raced            637        870    1.30  3.5e-16  
       441  3^2.7^2          2p       raced            625       1195    1.90  4.0e-16  
       442  2.13.17          flat     raced           1163       2961    2.48  6.1e-16  
       443  443              prime    raced           2568       5270    1.36  5.2e-16  bluestein
       444  2^2.3.37         2p       raced            846       3675    4.17  5.2e-16  
       445  5.89             prime    raced           2610       4709    1.74  4.1e-16  bluestein
       446  2.223            prime    raced           2584       5081    1.66  5.6e-16  bluestein
       447  3.149            prime    raced           2624       4910    1.82  5.5e-16  bluestein
       448  2^6.7            chain3   raced            503        539    1.03  2.1e-16  
       449  449              prime    raced           2317       5274    1.81  5.6e-16  rader
       450  2.3^2.5^2        chain3   raced            671        795    1.06  4.9e-16  
       451  11.41            2p       raced           1326       4051    3.04  5.0e-16  
       452  2^2.113          flat     raced           2679       5944    2.19  5.6e-16  
       453  3.151            prime    raced           2619       3829    1.25  4.7e-16  bluestein
       454  2.227            prime    raced           2565       4313    1.65  5.1e-16  bluestein
       455  5.7.13           chain3   raced            744       1170    1.56  6.0e-16  
       456  2^3.3.19         chain3   raced            783       3049    3.83  3.9e-16  
       457  457              prime    raced           2723       4007    1.45  7.3e-16  bluestein
       458  2.229            prime    raced           2590       4327    1.64  5.9e-16  bluestein
       459  3^3.17           2p       raced            699       3108    3.94  5.8e-16  
       460  2^2.5.23         chain3   raced            970       2890    2.79  4.3e-16  
       461  461              prime    raced           2614       4192    1.55  4.8e-16  bluestein
       462  2.3.7.11         chain3   raced            758       1018    1.32  3.6e-16  
       463  463              prime    raced           2599       3983    1.52  4.8e-16  bluestein
       464  2^4.29           2p       raced            867       3110    2.95  2.8e-16  
       465  3.5.31           2p       raced           1344       3729    2.73  5.2e-16  
       466  2.233            prime    raced           2597       4331    1.64  5.1e-16  bluestein
       467  467              prime    raced           2701       4117    1.28  5.4e-16  bluestein
       468  2^2.3^2.13       chain3   raced            701        991    1.40  4.9e-16  
       469  7.67             prime    raced           2585       5536    2.05  4.7e-16  bluestein
       470  2.5.47           2p       raced           1447       4158    2.56  4.4e-16  
       471  3.157            prime    raced           2580       3842    1.47  5.3e-16  bluestein
       472  2^3.59           prime    raced           2626       4934    1.76  3.7e-16  bluestein
       473  11.43            2p       raced           1481       4456    2.52  4.3e-16  
       474  2.3.79           prime    raced           2630       4921    1.77  4.7e-16  bluestein
       475  5^2.19           2p       raced            843       3251    3.53  4.2e-16  
       476  2^2.7.17         chain3   raced            802       2726    3.39  6.6e-16  
       477  3^2.53           prime    raced           2584       4660    1.70  4.5e-16  bluestein
       478  2.239            prime    raced           2617       4408    1.68  4.9e-16  bluestein
       479  479              prime    raced           2573       4064    1.48  4.2e-16  bluestein
       480  2^5.3.5          2p       raced            489        592    1.18  3.2e-16  
       481  13.37            2p       raced           1239       4125    3.21  5.8e-16  
       482  2.241            prime    raced           2580       4533    1.74  4.4e-16  bluestein
       483  3.7.23           2p       raced           1015       3178    3.09  5.3e-16  
       484  2^2.11^2         chain3   raced            768       1691    2.17  4.4e-16  
       485  5.97             prime    raced           2578       4578    1.70  3.6e-16  bluestein
       486  2.3^5            chain3   raced            715        972    1.35  4.3e-16  
       487  487              prime    raced           2612       4492    1.69  4.1e-16  bluestein
       488  2^3.61           flat     raced           2048       5247    2.47  4.1e-16  
       489  3.163            prime    raced           2563       5581    2.17  6.1e-16  bluestein
       490  2.5.7^2          chain3   raced            783        975    1.22  3.8e-16  
       491  491              prime    raced           2574       4424    1.58  5.4e-16  bluestein
       492  2^2.3.41         2p       raced           1080       4119    3.76  5.0e-16  
       493  17.29            2p       raced           1115       5515    4.55  5.6e-16  
       494  2.13.19          flat     raced           1367       3409    2.33  5.2e-16  
       495  3^2.5.11         chain3   raced            742       1256    1.65  5.0e-16  
       496  2^4.31           2p       raced            859       3501    4.04  4.5e-16  
       497  7.71             prime    raced           2593       6000    1.83  4.9e-16  bluestein
       498  2.3.83           prime    raced           2662       5542    1.81  7.1e-16  bluestein
       499  499              prime    raced           2826       4439    1.45  7.7e-16  bluestein
       500  2^2.5^3          chain3   raced            659        697    0.95  5.4e-16  
       501  3.167            prime    raced           2611       6284    1.98  5.7e-16  bluestein
       502  2.251            prime    raced           2590       3893    1.44  5.2e-16  bluestein
       503  503              prime    raced           2644       4126    1.31  4.0e-16  bluestein
       504  2^3.3^2.7        chain3   raced            660       1346    2.02  4.5e-16  
       505  5.101            prime    raced           2614       4989    1.64  5.5e-16  bluestein
       506  2.11.23          flat     raced           1464       3745    2.28  4.5e-16  
       507  3.13^2           chain3   raced            910       1772    1.78  4.6e-16  
       508  2^2.127          prime    raced           2696       4332    1.44  5.1e-16  bluestein
       509  509              prime    raced           2607       4089    1.35  4.4e-16  bluestein
       510  2.3.5.17         chain3   raced            897       2856    3.15  6.3e-16  
       511  7.73             prime    raced           2615       5357    1.77  5.3e-16  bluestein
       512  2^9              2p       raced            402        458    1.06  3.1e-16  
       513  3^3.19           2p       raced            919       3090    3.17  4.6e-16  
       514  2.257            prime    raced           3854       4673    1.14  7.1e-16  bluestein
       515  5.103            prime    raced           3936       7296    1.52  6.1e-16  bluestein
       516  2^2.3.43         2p       raced           1127       4455    3.86  4.9e-16  
       517  11.47            2p       raced           1819       5037    2.74  4.6e-16  
       518  2.7.37           flat     raced           2360       4076    1.51  5.0e-16  
       519  3.173            prime    raced           3836       6308    1.36  4.5e-16  bluestein
       520  2^3.5.13         chain3   raced            784        989    1.23  5.8e-16  
       521  521              prime    raced           3188       8047    2.48  9.7e-16  rader
       522  2.3^2.29         chain3   raced           1481       3643    2.29  4.2e-16  
       523  523              prime    raced           3936       6765    1.39  5.4e-16  bluestein
       524  2^2.131          flat     raced           3533       6262    1.71  6.1e-16  
       525  3.5^2.7          chain3   raced            749       1287    1.71  5.1e-16  
       526  2.263            prime    raced           3938       6272    1.46  6.8e-16  bluestein
       527  17.31            2p       raced           1449       6194    4.08  6.9e-16  
       528  2^4.3.11         chain3   raced            647       1075    1.57  3.8e-16  
       529  23^2             2p       raced           1311       5969    3.66  4.5e-16  
       530  2.5.53           flat     raced           2805       5125    1.81  6.4e-16  
       531  3^2.59           prime    raced           3843       5618    1.43  5.7e-16  bluestein
       532  2^2.7.19         chain3   raced            973       3128    2.70  5.8e-16  
       533  13.41            2p       raced           1537       4830    3.10  5.6e-16  
       534  2.3.89           flat     raced           3535       8623    2.05  5.4e-16  
       535  5.107            prime    raced           3986       9569    2.34  6.1e-16  bluestein
       536  2^3.67           flat     raced           2460       6179    2.47  5.1e-16  
       537  3.179            prime    raced           3871       6345    1.35  6.6e-16  bluestein
       538  2.269            prime    raced           3946       6066    1.51  6.1e-16  bluestein
       539  7^2.11           chain3   raced            923       1427    1.50  3.8e-16  
       540  2^2.3^3.5        chain3   raced            735       1574    2.13  4.0e-16  
       541  541              prime    raced           3711       6230    1.50  6.8e-16  bluestein
       542  2.271            prime    raced           3669       6885    1.52  8.5e-16  bluestein
       543  3.181            prime    raced           3724       6270    1.51  7.8e-16  bluestein
       544  2^5.17           2p       raced            685       2911    4.09  5.6e-16  
       545  5.109            prime    raced           3687       5952    1.56  6.9e-16  bluestein
       546  2.3.7.13         chain3   raced            915       1202    1.31  5.0e-16  
       547  547              prime    raced           3820       8488    2.18  7.9e-16  bluestein
       548  2^2.137          flat     raced           3263       5782    1.71  6.3e-16  
       549  3^2.61           flat     raced           3115       5962    1.85  6.2e-16  
       550  2.5^2.11         chain3   raced            843       1158    1.34  4.1e-16  
       551  19.29            2p       raced           1526       6262    3.92  4.0e-16  
       552  2^3.3.23         chain3   raced           1038       4288    3.82  3.5e-16  
       553  7.79             prime    raced           3709       5600    1.50  5.5e-16  bluestein
       554  2.277            prime    raced           3628       8225    2.24  7.3e-16  bluestein
       555  3.5.37           2p       raced           1355       4486    2.63  5.1e-16  flips differ 1.26x
       556  2^2.139          prime    raced           3654       5809    1.56  7.7e-16  bluestein
       557  557              prime    raced           3693       8389    2.25  7.9e-16  bluestein
       558  2.3^2.31         chain3   raced           1641       4029    2.45  6.6e-16  
       559  13.43            2p       raced           1692       5237    3.04  5.4e-16  
       560  2^4.5.7          chain3   raced            709        853    1.17  3.3e-16  
       561  3.11.17          chain3   raced           1005       3844    3.40  6.1e-16  
       562  2.281            prime    raced           3688       8204    2.19  8.1e-16  bluestein
       563  563              prime    raced           3667       8438    2.26  7.0e-16  bluestein
       564  2^2.3.47         2p       raced           1327       5384    3.86  3.8e-16  
       565  5.113            prime    raced           3713       7632    1.99  7.9e-16  bluestein
       566  2.283            prime    raced           3800       7219    1.79  6.9e-16  bluestein
       567  3^4.7            2p       raced            847       1532    1.60  6.9e-16  
       568  2^3.71           flat     raced           2438       6080    2.41  4.0e-16  
       569  569              prime    raced           3667       7984    2.11  8.2e-16  bluestein
       570  2.3.5.19         chain3   raced           1057       3259    2.69  4.7e-16  
       571  571              prime    raced           3747       6850    1.50  6.2e-16  bluestein
       572  2^2.11.13        chain3   raced            926       1955    2.10  5.3e-16  
       573  3.191            prime    raced           3702       4398    1.15  7.7e-16  bluestein
       574  2.7.41           flat     raced           2605       4849    1.67  6.9e-16  
       575  5^2.23           2p       raced           1132       3774    2.96  5.7e-16  
       576  2^6.3^2          chain3   raced            650       1045    1.12  4.3e-16  
       577  577              prime    raced           3027       5383    1.76  5.6e-16  rader
       578  2.17^2           flat     raced           1697       6037    3.51  6.4e-16  
       579  3.193            prime    raced           5684       5243    0.78  4.2e-16  bluestein
       580  2^2.5.29         chain3   raced           1475       4027    2.69  3.9e-16  
       581  7.83             prime    raced           5619       8181    1.45  5.0e-16  bluestein
       582  2.3.97           flat     raced           4203       5684    1.33  6.1e-16  
       583  11.53            flat     raced           3140       6181    1.93  6.5e-16  
       584  2^3.73           flat     raced           2282       6087    2.64  5.1e-16  
       585  3^2.5.13         chain3   raced            959       1602    1.62  5.7e-16  
       586  2.293            prime    raced           5658       5437    0.92  5.3e-16  bluestein
       587  587              prime    raced           5562       5398    0.95  4.8e-16  bluestein
       588  2^2.3.7^2        chain3   raced            824       1802    2.10  4.6e-16  
       589  19.31            2p       raced           1728       6885    3.75  5.1e-16  
       590  2.5.59           flat     raced           4851       6176    1.25  4.9e-16  
       591  3.197            prime    raced           5563       5373    0.94  5.8e-16  bluestein
       592  2^4.37           2p       raced           1080       4525    4.07  4.3e-16  
       593  593              prime    raced           4415       5431    1.12  6.2e-16  rader
       594  2.3^3.11         chain3   raced           1012       1356    1.27  4.9e-16  
       595  5.7.17           chain3   raced           1074       3470    3.21  5.1e-16  
       596  2^2.149          flat     raced           4534       6718    1.40  5.8e-16  
       597  3.199            prime    raced           5623       5585    0.92  3.9e-16  bluestein
       598  2.13.23          flat     raced           1761       4495    2.14  6.5e-16  
       599  599              prime    raced           5575       5432    0.97  5.8e-16  bluestein
       600  2^3.3.5^2        chain3   raced            817        787    0.93  3.5e-16  
       601  601              prime    raced           4266       5780    1.33  5.7e-16  bluestein
       602  2.7.43           flat     raced           3044       5374    1.67  4.0e-16  
       603  3^2.67           flat     raced           3772       7095    1.84  5.9e-16  
       604  2^2.151          flat     raced           4984       5218    0.87  8.3e-16  
       605  5.11^2           chain3   raced           1041       2258    2.09  5.0e-16  
       606  2.3.101          prime    raced           4378       6455    1.42  6.8e-16  bluestein
       607  607              prime    raced           4371       8598    1.93  8.0e-16  bluestein
       608  2^5.19           2p       raced            919       3366    3.64  3.8e-16  
       609  3.7.29           2p       raced           1240       4373    3.50  4.4e-16  
       610  2.5.61           flat     raced           3271       6619    1.98  7.7e-16  
       611  13.47            2p       raced           1980       6071    2.96  5.8e-16  
       612  2^2.3^2.17       chain3   raced           1037       3818    3.32  5.7e-16  
       613  613              prime    raced           4208       8246    1.80  6.3e-16  bluestein
       614  2.307            prime    raced           4271       9913    2.23  7.6e-16  bluestein
       615  3.5.41           2p       raced           1697       5269    3.10  4.0e-16  
       616  2^3.7.11         chain3   raced            977       1333    1.36  3.5e-16  
       617  617              prime    raced           3816       7878    2.04  6.2e-16  rader
       618  2.3.103          prime    raced           4310      10326    2.18  6.2e-16  bluestein
       619  619              prime    raced           4383       7724    1.74  8.3e-16  bluestein
       620  2^2.5.31         chain3   raced           1428       4439    2.56  5.6e-16  
       621  3^3.23           2p       raced           1174       4102    2.95  5.7e-16  
       622  2.311            prime    raced           4324      10052    2.32  6.4e-16  bluestein
       623  7.89             flat     raced           3924       9385    1.95  5.4e-16  
       624  2^4.3.13         chain3   raced            842       1298    1.52  4.4e-16  
       625  5^4              2p       raced            896       1790    1.93  4.4e-16  
       626  2.313            prime    raced           4362      10146    1.97  6.5e-16  bluestein
       627  3.11.19          chain3   raced           1223       4378    3.41  4.5e-16  
       628  2^2.157          flat     raced           4849       5179    0.97  6.4e-16  
       629  17.37            2p       raced           1710       7710    4.48  6.2e-16  
       630  2.3^2.5.7        chain3   raced            908       1134    1.21  4.8e-16  
       631  631              prime    raced           4748       5190    1.07  7.0e-16  bluestein
       632  2^3.79           flat     raced           3071       6380    2.04  7.0e-16  
       633  3.211            prime    raced           4710      10690    2.26  5.1e-16  bluestein
       634  2.317            prime    raced           4445       5124    1.13  5.2e-16  bluestein
       635  5.127            prime    raced           4346       5516    1.25  6.6e-16  bluestein
       636  2^2.3.53         flat     raced           2892       6384    2.17  6.3e-16  
       637  7^2.13           chain3   raced           1100       1664    1.50  6.1e-16  
       638  2.11.29          flat     raced           2730       5292    1.88  5.4e-16  
       639  3^2.71           flat     raced           3616       9643    2.35  6.1e-16  
       640  2^7.5            chain3   raced            717        679    0.87  2.8e-16  
       641  641              prime    raced           3202       5831    1.79  3.7e-16  rader
       642  2.3.107          prime    raced           4639       9719    1.94  6.1e-16  bluestein
       643  643              prime    raced           4590       7174    1.49  7.2e-16  bluestein
       644  2^2.7.23         chain3   raced           1450       4145    2.75  3.3e-16  
       645  3.5.43           2p       raced           2128       5760    2.58  4.3e-16  
       646  2.17.19          flat     raced           2013       6778    3.35  5.7e-16  
       647  647              prime    raced           4608       7294    1.49  7.4e-16  bluestein
       648  2^3.3^4          chain3   raced            894       1921    2.10  4.3e-16  
       649  11.59            prime    raced           4844       7016    1.40  6.3e-16  bluestein
       650  2.5^2.13         chain3   raced           1082       1391    1.16  5.0e-16  
       651  3.7.31           2p       raced           1475       4965    3.13  5.6e-16  
       652  2^2.163          flat     raced           5340       7426    1.17  6.5e-16  
       653  653              prime    raced           4596       8274    1.69  5.7e-16  bluestein
       654  2.3.109          flat     raced           5432       7728    1.27  6.9e-16  
       655  5.131            prime    raced           4673       8237    1.76  7.4e-16  bluestein
       656  2^4.41           2p       raced           1349       5425    3.84  4.6e-16  
       657  3^2.73           flat     raced           3435       7210    1.90  6.4e-16  
       658  2.7.47           flat     raced           3885       6142    1.56  5.8e-16  
       659  659              prime    raced           4659       7845    1.57  5.3e-16  bluestein
       660  2^2.3.5.11       chain3   raced            956       1385    1.43  5.8e-16  
       661  661              prime    raced           4642       8011    1.65  8.0e-16  bluestein
       662  2.331            prime    raced           4688       9353    1.89  6.7e-16  bluestein
       663  3.13.17          chain3   raced           1289       4547    3.28  5.9e-16  
       664  2^3.83           prime    raced           4751       7427    1.45  6.7e-16  bluestein
       665  5.7.19           chain3   raced           1304       3999    2.85  4.0e-16  
       666  2.3^2.37         chain3   raced           2325       5415    2.15  3.6e-16  
       667  23.29            2p       raced           1958       7969    4.01  5.4e-16  
       668  2^2.167          prime    raced           4646      11464    2.14  6.6e-16  bluestein
       669  3.223            prime    raced           4655       7648    1.62  7.7e-16  bluestein
       670  2.5.67           flat     raced           3872       7811    2.00  6.0e-16  
       671  11.61            flat     raced           3657       7902    2.04  5.0e-16  
       672  2^5.3.7          2p       raced            770       1076    1.35  4.1e-16  
       673  673              prime    raced           3576       7900    2.19  5.6e-16  rader
       674  2.337            prime    raced           5817       9253    1.55  5.8e-16  bluestein
       675  3^3.5^2          2p       raced            988       1639    1.64  5.1e-16  
       676  2^2.13^2         chain3   raced           1191       2325    1.89  7.1e-16  
       677  677              prime    raced           5275      10431    1.30  1.0e-15  rader
       678  2.3.113          flat     raced           4761       9741    1.78  3.9e-16  
       679  7.97             flat     raced           5188       6455    1.23  5.8e-16  
       680  2^3.5.17         chain3   raced           1082       3691    3.33  5.3e-16  
       681  3.227            prime    raced           5889       7055    1.19  8.7e-16  bluestein
       682  2.11.31          flat     raced           2359       6100    2.03  4.0e-16  
       683  683              prime    raced           5821      10628    1.38  4.8e-16  flips differ 1.29x bluestein
       684  2^2.3^2.19       chain3   raced           1352       4405    2.79  5.4e-16  
       685  5.137            flat     raced           5600       7669    1.19  5.5e-16  
       686  2.7^3            flat     raced           1741       1391    0.77  2.6e-16  
       687  3.229            prime    raced           6197       8231    0.62  4.2e-16  flips differ 2.04x bluestein
       688  2^4.43           2p       raced           1486       7066    2.71  4.5e-16  flips differ 2.00x
       689  13.53            flat     raced           3589       8173    1.83  7.3e-16  
       690  2.3.5.23         chain3   raced           1634       5657    2.71  5.1e-16  flips differ 1.47x
       691  691              prime    raced           5820       9354    1.07  6.6e-16  rader
       692  2^2.173          flat     raced           5636      10194    1.77  6.2e-16  
       693  3^2.7.11         chain3   raced           1085       2394    1.42  4.0e-16  flips differ 1.65x
       694  2.347            prime    raced           6277      11695    1.42  5.9e-16  bluestein
       695  5.139            prime    raced           5928       8911    1.10  5.3e-16  bluestein
       696  2^3.3.29         chain3   raced           1560       5848    1.68  4.5e-16  flips differ 2.39x
       697  17.41            2p       raced           2067       9722    3.53  6.6e-16  
       698  2.349            prime    raced           6157      10530    1.46  6.1e-16  bluestein
       699  3.233            prime    raced           5725       6810    1.09  4.4e-16  bluestein
       700  2^2.5^2.7        chain3   raced           1004       1160    1.07  4.1e-16  
       701  701              prime    raced           6577       7867    1.13  5.5e-16  bluestein
       702  2.3^3.13         chain3   raced           1184       1715    1.36  5.7e-16  
       703  19.37            2p       raced           2346       8737    3.42  5.8e-16  
       704  2^6.11           chain3   raced            843       1375    1.61  3.1e-16  
       705  3.5.47           2p       raced           2732       7094    2.30  5.3e-16  
       706  2.353            prime    raced           5821       7702    1.22  5.3e-16  bluestein
       707  7.101            flat     raced           6014       7069    1.14  5.8e-16  
       708  2^2.3.59         flat     raced           5312       8023    1.34  6.0e-16  
       709  709              prime    raced           5653       8798    1.22  5.0e-16  bluestein
       710  2.5.71           flat     raced           3768       8007    2.04  6.9e-16  
       711  3^2.79           flat     raced           4617       8279    1.75  6.9e-16  
       712  2^3.89           flat     raced           2942       7478    2.49  7.4e-16  
       713  23.31            2p       raced           2194       9117    3.17  6.0e-16  flips differ 1.28x
       714  2.3.7.17         chain3   raced           1263       4089    3.22  6.2e-16  
       715  5.11.13          chain3   raced           1215       2606    2.12  5.8e-16  
       716  2^2.179          prime    raced           5871      10155    1.52  4.9e-16  bluestein
       717  3.239            prime    raced           5978       6944    1.10  6.2e-16  bluestein
       718  2.359            prime    raced           5934       8256    1.14  5.1e-16  bluestein
       719  719              prime    raced           5855       8605    1.15  6.5e-16  bluestein
       720  2^4.3^2.5        chain3   raced            905       1363    1.08  3.3e-16  
       721  7.103            prime    raced           5932      12739    2.04  5.7e-16  bluestein
       722  2.19^2           flat     raced           2252       7937    2.90  5.9e-16  
       723  3.241            prime    raced           5802       7117    1.22  3.7e-16  bluestein
       724  2^2.181          flat     raced           5639       8047    1.27  5.8e-16  
       725  5^2.29           2p       raced           1510       5349    3.19  4.8e-16  
       726  2.3.11^2         chain3   raced           1230       2489    1.98  4.5e-16  
       727  727              prime    raced           5079      10108    1.96  8.7e-16  rader
       728  2^3.7.13         chain3   raced           1120       1578    1.38  6.1e-16  
       729  3^6              2p       raced           1106       1993    1.79  4.5e-16  
       730  2.5.73           flat     raced           3824       7711    1.97  4.9e-16  
       731  17.43            2p       raced           2361       9853    3.98  5.2e-16  
       732  2^2.3.61         flat     raced           3322       8187    2.45  6.2e-16  
       733  733              prime    raced           5831       9718    1.43  4.2e-16  bluestein
       734  2.367            prime    raced           5690       8548    1.47  6.3e-16  bluestein
       735  3.5.7^2          chain3   raced           1135       2002    1.51  4.6e-16  
       736  2^5.23           2p       raced           1182       4548    3.79  3.6e-16  
       737  11.67            flat     raced           4326       9649    2.12  6.9e-16  
       738  2.3^2.41         chain3   raced           2742       6514    1.82  4.5e-16  flips differ 1.25x
       739  739              prime    raced           5605       7700    1.34  5.1e-16  bluestein
       740  2^2.5.37         flat     raced           2084       5770    2.61  5.4e-16  
       741  3.13.19          chain3   raced           1578       5252    3.04  6.2e-16  
       742  2.7.53           flat     raced           3450       7578    2.14  5.5e-16  
       743  743              prime    raced           5507       9346    1.34  6.3e-16  bluestein
       744  2^3.3.31         chain3   raced           2000       5372    2.36  4.5e-16  
       745  5.149            prime    raced           5593       8113    1.42  7.5e-16  bluestein
       746  2.373            prime    raced           5756       8852    1.51  5.4e-16  bluestein
       747  3^2.83           prime    raced           5507      10664    1.87  6.1e-16  bluestein
       748  2^2.11.17        chain3   raced           1281       5262    3.94  6.8e-16  
       749  7.107            prime    raced           5604      10967    1.65  7.2e-16  bluestein
       750  2.3.5^3          chain3   raced           1103       1472    1.29  5.3e-16  
       751  751              prime    raced           5930       6893    1.10  5.5e-16  bluestein
       752  2^4.47           2p       raced           1696       6851    3.83  4.4e-16  
       753  3.251            prime    raced           5562       5921    1.03  5.4e-16  bluestein
       754  2.13.29          flat     raced           2774       6288    2.24  6.9e-16  
       755  5.151            prime    raced           5565       7128    1.21  6.9e-16  bluestein
       756  2^2.3^3.7        chain3   raced           1065       1823    1.23  4.4e-16  
       757  757              prime    raced           5477       6736    1.21  5.3e-16  bluestein
       758  2.379            prime    raced           5551       6106    1.09  5.0e-16  bluestein
       759  3.11.23          chain3   raced           1881       5715    2.82  5.4e-16  
       760  2^3.5.19         chain3   raced           1378       4649    3.20  3.6e-16  
       761  761              prime    raced           5516       6818    1.22  5.7e-16  bluestein
       762  2.3.127          flat     raced           6705       7003    1.00  7.6e-16  
       763  7.109            flat     raced           6505       8251    1.24  7.4e-16  
       764  2^2.191          prime    raced           5555       5877    1.05  6.3e-16  bluestein
       765  3^2.5.17         chain3   raced           1380       4514    3.25  7.0e-16  
       766  2.383            prime    raced           5562       6102    1.08  5.3e-16  bluestein
       767  13.59            prime    raced           5633       8870    1.57  4.7e-16  bluestein
       768  2^8.3            chain3   raced            817        737    0.89  2.8e-16  
       769  769              prime    raced           3681       7281    1.95  5.2e-16  rader
       770  2.5.7.11         chain3   raced           1291       1672    1.28  3.4e-16  
       771  3.257            prime    raced           5812       7164    1.07  5.8e-16  bluestein
       772  2^2.193          flat     raced           5473       6983    1.26  5.1e-16  
       773  773              prime    raced           5713       7483    1.24  7.6e-16  bluestein
       774  2.3^2.43         chain3   raced           3031       6783    1.87  4.9e-16  
       775  5^2.31           2p       raced           1803       5794    2.92  5.8e-16  
       776  2^3.97           flat     raced           3534       7298    2.00  3.3e-16  
       777  3.7.37           2p       raced           1951       6339    3.18  6.1e-16  
       778  2.389            prime    raced           5727       7274    1.25  5.5e-16  bluestein
       779  19.41            2p       raced           2492      10083    3.31  4.7e-16  
       780  2^2.3.5.13       chain3   raced           1255       1682    1.30  6.6e-16  
       781  11.71            flat     raced           4161       9012    2.16  5.2e-16  
       782  2.17.23          flat     raced           2532       8624    3.12  6.6e-16  
       783  3^3.29           2p       raced           1666       5690    2.91  5.0e-16  
       784  2^4.7^2          chain3   raced           1028       1321    1.28  3.9e-16  
       785  5.157            prime    raced           5740       6722    1.15  5.1e-16  bluestein
       786  2.3.131          flat     raced           5979       9523    1.56  7.5e-16  
       787  787              prime    raced           5712       7238    1.23  4.9e-16  bluestein
       788  2^2.197          flat     raced           5146       7026    1.35  6.0e-16  
       789  3.263            prime    raced           5664       9039    1.33  6.3e-16  bluestein
       790  2.5.79           flat     raced           4727       7901    1.66  7.2e-16  
       791  7.113            flat     raced           5438      10685    1.90  5.5e-16  
       792  2^3.3^2.11       chain3   raced           1156       2141    1.42  6.2e-16  
       793  13.61            flat     raced           3982       9436    2.35  5.7e-16  
       794  2.397            prime    raced           6001       7204    1.12  5.0e-16  bluestein
       795  3.5.53           flat     raced           3946       8213    2.01  6.7e-16  
       796  2^2.199          flat     raced           6413       7077    1.10  7.1e-16  
       797  797              prime    raced           5633       7343    1.20  5.4e-16  bluestein
       798  2.3.7.19         chain3   raced           1570       4647    2.74  3.9e-16  
       799  17.47            2p       raced           2668      10793    3.88  4.8e-16  
       800  2^5.5^2          2p       raced            977       1060    1.08  2.6e-16  
       801  3^2.89           flat     raced           4363       9048    2.05  7.8e-16  
       802  2.401            prime    raced           5783       7612    1.25  5.5e-16  bluestein
       803  11.73            flat     raced           4061       9188    2.25  6.2e-16  
       804  2^2.3.67         flat     raced           3909       9582    2.37  5.9e-16  
       805  5.7.23           chain3   raced           1745       5334    2.73  4.7e-16  
       806  2.13.31          flat     raced           2743       6762    2.38  5.0e-16  
       807  3.269            prime    raced           5703       9078    1.55  5.3e-16  bluestein
       808  2^3.101          flat     raced           4923       7971    1.50  7.4e-16  
       809  809              prime    raced           5955      10725    1.46  7.3e-16  bluestein
       810  2.3^4.5          chain3   raced           1208       1562    1.28  4.8e-16  
       811  811              prime    raced           5689       8440    1.39  5.0e-16  bluestein
       812  2^2.7.29         chain3   raced           1803       6153    2.60  4.9e-16  
       813  3.271            prime    raced           6060       9023    1.47  7.4e-16  bluestein
       814  2.11.37          flat     raced           3639       7264    1.98  4.7e-16  
       815  5.163            prime    raced           5782       9340    1.60  6.2e-16  bluestein
       816  2^4.3.17         chain3   raced           1337       4558    3.39  4.7e-16  
       817  19.43            2p       raced           2709      10937    3.15  5.3e-16  flips differ 1.26x
       818  2.409            prime    raced           5811      10710    1.49  6.4e-16  bluestein
       819  3^2.7.13         chain3   raced           1451       2147    1.48  6.6e-16  
       820  2^2.5.41         chain3   raced           3027       6875    2.25  3.6e-16  
       821  821              prime    raced           5667       8579    1.49  5.5e-16  bluestein
       822  2.3.137          prime    raced           5783       8720    1.43  5.1e-16  bluestein
       823  823              prime    raced           5788       8573    1.44  6.0e-16  bluestein
       824  2^3.103          flat     raced           4430      11498    2.05  7.7e-16  
       825  3.5^2.11         chain3   raced           1307       2321    1.72  3.9e-16  
       826  2.7.59           prime    raced           6046       9156    0.65  5.2e-16  flips differ 2.38x bluestein
       827  827              prime    raced           5964       9173    0.67  7.5e-16  flips differ 2.33x bluestein
       828  2^2.3^2.23       chain3   raced           1934       5261    2.65  4.8e-16  
       829  829              prime    raced           5922       9035    1.47  7.1e-16  bluestein
       830  2.5.83           prime    raced           6350      12240    1.90  5.2e-16  bluestein
       831  3.277            prime    raced           5919      13206    2.16  6.3e-16  bluestein
       832  2^6.13           chain3   raced           1140       1632    1.38  5.1e-16  
       833  7^2.17           chain3   raced           1498       4935    3.19  6.4e-16  
       834  2.3.139          prime    raced           5894       9009    1.49  5.6e-16  bluestein
       835  5.167            prime    raced           5696       9190    1.57  5.7e-16  bluestein
       836  2^2.11.19        chain3   raced           1563       5958    3.46  3.5e-16  
       837  3^3.31           2p       raced           1983       6286    2.74  6.4e-16  
       838  2.419            prime    raced           6126      10972    1.25  5.5e-16  bluestein
       839  839              prime    raced           5850       8516    1.43  5.8e-16  bluestein
       840  2^3.3.5.7        chain3   raced           1244       2270    1.77  3.5e-16  
       841  29^2             2p       raced           2645      10774    3.71  6.3e-16  
       842  2.421            prime    raced           5697       8709    1.45  5.7e-16  bluestein
       843  3.281            prime    raced           5702      10892    1.83  5.4e-16  bluestein
       844  2^2.211          flat     raced           7541      14044    1.83  4.7e-16  
       845  5.13^2           chain3   raced           1561       3035    1.87  5.7e-16  
       846  2.3^2.47         flat     raced           3767       8010    2.07  5.7e-16  
       847  7.11^2           chain3   raced           1556       3150    1.94  4.3e-16  
       848  2^4.53           flat     raced           3080       8440    2.68  5.9e-16  
       849  3.283            prime    raced           5974      10909    1.53  5.9e-16  bluestein
       850  2.5^2.17         chain3   raced           1501       4946    3.22  4.6e-16  
       851  23.37            flat     raced           3572      11223    3.09  5.0e-16  
       852  2^2.3.71         flat     raced           3891       9344    2.35  4.3e-16  
       853  853              prime    raced           6120       8811    1.44  6.1e-16  bluestein
       854  2.7.61           flat     raced           4022       9741    2.38  4.1e-16  
       855  3^2.5.19         chain3   raced           1771       5327    2.63  5.4e-16  
       856  2^3.107          prime    raced           5798      15128    2.59  5.9e-16  bluestein
       857  857              prime    raced           6083       8481    1.35  5.7e-16  bluestein
       858  2.3.11.13        chain3   raced           1575       2915    1.79  7.2e-16  
       859  859              prime    raced           5852       8536    1.17  6.9e-16  bluestein
       860  2^2.5.43         chain3   raced           2776       7461    2.68  5.7e-16  
       861  3.7.41           2p       raced           2316       7601    3.01  4.1e-16  
       862  2.431            prime    raced           5770      13412    2.06  6.0e-16  bluestein
       863  863              prime    raced           5685       8772    1.41  5.2e-16  bluestein
       864  2^5.3^3          2p       raced           1027       1841    1.23  4.1e-16  
       865  5.173            prime    raced           5891       9176    1.47  5.3e-16  bluestein
       866  2.433            prime    raced           5740       9725    1.48  5.7e-16  bluestein
       867  3.17^2           chain3   raced           1876       9126    4.71  5.8e-16  
       868  2^2.7.31         chain3   raced           2093       7356    3.49  4.7e-16  
       869  11.79            flat     raced           5411       9525    1.73  8.0e-16  
       870  2.3.5.29         chain3   raced           2424       6124    2.51  4.9e-16  
       871  13.67            flat     raced           4707      11259    2.33  7.4e-16  
       872  2^3.109          flat     raced           5166      11326    2.10  7.7e-16  
       873  3^2.97           flat     raced           5435       8423    1.51  6.7e-16  
       874  2.19.23          flat     raced           2803       9902    2.96  5.0e-16  
       875  5^3.7            chain3   raced           1308       2481    1.85  6.4e-16  
       876  2^2.3.73         flat     raced           3916       9420    2.37  5.2e-16  
       877  877              prime    raced           5835      10874    1.48  4.3e-16  bluestein
       878  2.439            prime    raced           5837      13042    2.17  5.6e-16  bluestein
       879  3.293            prime    raced           6042       8344    1.35  4.4e-16  bluestein
       880  2^4.5.11         chain3   raced           1166       1709    1.46  3.2e-16  
       881  881              prime    raced           4985      13006    2.60  5.0e-16  rader
       882  2.3^2.7^2        chain3   raced           1337       1754    1.30  4.8e-16  
       883  883              prime    raced           5798      10991    1.54  4.7e-16  bluestein
       884  2^2.13.17        chain3   raced           1633       6012    3.65  7.0e-16  
       885  3.5.59           prime    raced           5813      10050    1.71  4.6e-16  bluestein
       886  2.443            prime    raced           5907       8717    1.44  5.9e-16  bluestein
       887  887              prime    raced           5741       8736    1.50  5.1e-16  bluestein
       888  2^3.3.37         chain3   raced           2141       7069    3.20  5.5e-16  
       889  7.127            prime    raced           5903       7673    1.13  5.8e-16  bluestein
       890  2.5.89           flat     raced           4704      13563    2.43  7.2e-16  
       891  3^4.11           chain3   raced           1569       2366    1.50  5.9e-16  
       892  2^2.223          prime    raced           5862      10289    1.72  6.3e-16  bluestein
       893  19.47            flat     raced           3970      12203    2.98  5.9e-16  
       894  2.3.149          prime    raced           5884      10152    1.69  5.5e-16  bluestein
       895  5.179            prime    raced           5877      13170    2.19  5.1e-16  bluestein
       896  2^7.7            chain3   raced           1234       1209    0.96  3.0e-16  
       897  3.13.23          chain3   raced           2320       7073    2.93  7.2e-16  
       898  2.449            prime    raced           5847      13091    1.57  6.5e-16  flips differ 1.43x bluestein
       899  29.31            2p       raced           3102      11656    3.65  5.6e-16  
       900  2^2.3^2.5^2      chain3   raced           1318       1985    1.13  5.3e-16  
       901  17.53            flat     raced           4327      12939    2.97  7.8e-16  
       902  2.11.41          flat     raced           3630       8649    2.35  4.9e-16  
       903  3.7.43           2p       raced           2631       8110    3.04  5.7e-16  
       904  2^3.113          flat     raced           3960      12914    3.19  4.6e-16  
       905  5.181            prime    raced           5738       9432    1.63  5.3e-16  bluestein
       906  2.3.151          prime    raced           5743       7732    1.33  4.0e-16  bluestein
       907  907              prime    raced           5831      15807    2.64  6.7e-16  bluestein
       908  2^2.227          prime    raced           5740       9072    1.56  6.3e-16  bluestein
       909  3^2.101          prime    raced           5730       9099    1.58  4.2e-16  bluestein
       910  2.5.7.13         chain3   raced           1603       2000    1.24  4.1e-16  
       911  911              prime    raced           6906      15576    2.18  8.3e-16  bluestein
       912  2^4.3.19         chain3   raced           1598       5194    2.87  4.7e-16  
       913  11.83            prime    raced           5735      13724    2.35  4.0e-16  bluestein
       914  2.457            prime    raced           5899       8443    1.40  4.9e-16  bluestein
       915  3.5.61           flat     raced           4589      10501    2.29  5.0e-16  
       916  2^2.229          prime    raced           5739       8828    1.47  4.3e-16  bluestein
       917  7.131            prime    raced           5766      12251    2.10  6.7e-16  bluestein
       918  2.3^3.17         chain3   raced           1622       5255    3.23  6.5e-16  
       919  919              prime    raced           5774      15487    2.62  6.0e-16  bluestein
       920  2^3.5.23         chain3   raced           1758       5647    3.20  4.0e-16  
       921  3.307            prime    raced           5964      14746    2.45  6.9e-16  bluestein
       922  2.461            prime    raced           5865       8467    1.42  5.5e-16  bluestein
       923  13.71            flat     raced           4612      10820    2.28  5.9e-16  
       924  2^2.3.7.11       chain3   raced           1416       2769    1.54  4.6e-16  
       925  5^2.37           2p       raced           2185       7565    3.19  5.0e-16  
       926  2.463            prime    raced           5785       8478    1.40  7.3e-16  bluestein
       927  3^2.103          prime    raced           5946      15235    2.45  6.3e-16  bluestein
       928  2^5.29           2p       raced           1646       6359    3.60  4.5e-16  
       929  929              prime    raced           5914      15506    2.54  3.8e-16  bluestein
       930  2.3.5.31         chain3   raced           2392       6791    2.55  5.1e-16  
       931  7^2.19           chain3   raced           2076       5705    2.69  4.5e-16  
       932  2^2.233          prime    raced           5859       8874    1.50  5.8e-16  bluestein
       933  3.311            prime    raced           5940      19028    2.71  5.4e-16  bluestein
       934  2.467            prime    raced           5739       8503    1.45  4.6e-16  bluestein
       935  5.11.17          chain3   raced           1818       6414    3.48  5.9e-16  
       936  2^3.3^2.13       chain3   raced           1564       3242    2.01  5.4e-16  
       937  937              prime    raced           5784      15581    2.66  4.7e-16  bluestein
       938  2.7.67           flat     raced           4740      11355    2.34  6.3e-16  
       939  3.313            prime    raced           5929      15367    2.56  4.6e-16  bluestein
       940  2^2.5.47         flat     raced           3039       8556    2.75  5.2e-16  
       941  941              prime    raced           5930       8652    1.35  5.9e-16  bluestein
       942  2.3.157          prime    raced           5904       7801    1.32  5.6e-16  bluestein
       943  23.41            2p       raced           3125      12699    3.23  4.9e-16  flips differ 1.26x
       944  2^4.59           prime    raced           5891      10116    1.68  4.9e-16  bluestein
       945  3^3.5.7          chain3   raced           1454       2369    1.61  4.4e-16  
       946  2.11.43          flat     raced           3949       9344    2.25  4.2e-16  
       947  947              prime    raced           5873       8377    1.40  4.8e-16  bluestein
       948  2^2.3.79         flat     raced           4905       9871    2.00  6.2e-16  
       949  13.73            flat     raced           4517      11389    2.24  6.9e-16  
       950  2.5^2.19         chain3   raced           1975       6256    2.91  3.8e-16  
       951  3.317            prime    raced           5791       7786    1.20  5.9e-16  bluestein
       952  2^3.7.17         chain3   raced           1589       5608    3.51  6.4e-16  
       953  953              prime    raced           5886       8365    1.39  5.3e-16  bluestein
       954  2.3^2.53         flat     raced           4173      10023    2.38  6.5e-16  
       955  5.191            prime    raced           5794       7981    1.26  4.6e-16  bluestein
       956  2^2.239          prime    raced           5836       8763    1.49  5.7e-16  bluestein
       957  3.11.29          chain3   raced           2753       7853    2.78  4.6e-16  
       958  2.479            prime    raced           6286       8574    1.34  6.3e-16  bluestein
       959  7.137            prime    raced           5803      11533    1.97  5.7e-16  bluestein
       960  2^6.3.5          chain3   raced           1144       1216    1.05  4.4e-16  
       961  31^2             flat     raced           4457      12699    2.77  6.2e-16  
       962  2.13.37          flat     raced           3557       8641    2.41  6.9e-16  
       963  3^2.107          prime    raced           6117      13748    1.88  6.3e-16  bluestein
       964  2^2.241          flat     raced           5955       8971    1.49  6.0e-16  
       965  5.193            prime    raced           5915       8756    1.44  6.0e-16  bluestein
       966  2.3.7.23         chain3   raced           2051       6329    2.67  5.2e-16  
       967  967              prime    raced           6048      16704    2.06  6.3e-16  bluestein
       968  2^3.11^2         chain3   raced           1571       3311    2.02  3.7e-16  
       969  3.17.19          chain3   raced           2304      10229    4.00  6.6e-16  
       970  2.5.97           flat     raced           5566       9420    1.61  4.9e-16  
       971  971              prime    raced           5770      16617    2.02  5.7e-16  bluestein
       972  2^2.3^5          chain3   raced           1417       1967    1.37  4.9e-16  
       973  7.139            prime    raced           5803      11342    1.92  4.2e-16  bluestein
       974  2.487            prime    raced           5887       9402    1.53  4.5e-16  bluestein
       975  3.5^2.13         chain3   raced           1713       2828    1.60  5.7e-16  
       976  2^4.61           flat     raced           3501      11049    3.00  5.2e-16  
       977  977              prime    raced           6140       9461    1.42  5.1e-16  bluestein
       978  2.3.163          prime    raced           5975      13002    1.76  6.2e-16  bluestein
       979  11.89            flat     raced           5091      11438    2.20  6.8e-16  
       980  2^2.5.7^2        flat     raced           2329       2872    1.22  3.6e-16  
       981  3^2.109          flat     raced           7940      13285    1.57  7.6e-16  
       982  2.491            prime    raced           6000       9700    1.49  5.8e-16  bluestein
       983  983              prime    raced           5791       9054    1.54  6.1e-16  bluestein
       984  2^3.3.41         chain3   raced           2674       9462    3.44  4.6e-16  
       985  5.197            prime    raced           6027       9170    1.46  5.3e-16  bluestein
       986  2.17.29          flat     raced           3924      11560    2.89  5.6e-16  
       987  3.7.47           2p       raced           2975       9319    3.02  4.6e-16  
       988  2^2.13.19        chain3   raced           2005       7009    3.25  5.9e-16  
       989  23.43            2p       raced           3356      13779    3.37  6.7e-16  
       990  2.3^2.5.11       chain3   raced           1547       2203    1.40  6.0e-16  
       991  991              prime    raced           5761       9148    1.49  4.9e-16  bluestein
       992  2^5.31           2p       raced           2134       6977    2.81  5.4e-16  
       993  3.331            prime    raced           5794      14125    2.44  5.2e-16  bluestein
       994  2.7.71           flat     raced           4693      10979    2.28  5.4e-16  
       995  5.199            prime    raced           6055       8873    1.44  6.5e-16  bluestein
       996  2^2.3.83         prime    raced           6120      14412    2.29  4.9e-16  bluestein
       997  997              prime    raced           5790       9142    1.57  7.1e-16  bluestein
       998  2.499            prime    raced           5962       9236    1.22  6.2e-16  flips differ 1.26x bluestein
       999  3^3.37           2p       raced           2728       8220    3.01  5.9e-16  
      1000  2^3.5^3          chain3   raced           1351       1482    1.02  3.6e-16  
      1001  7.11.13          chain3   raced           1757       3658    2.02  6.1e-16  
      1002  2.3.167          prime    raced           5860      15902    2.62  6.0e-16  bluestein
      1003  17.59            prime    raced           5843      15238    2.42  3.9e-16  bluestein
      1004  2^2.251          prime    raced           5928       7852    1.17  4.8e-16  bluestein
      1005  3.5.67           flat     raced           5378      12294    2.23  5.8e-16  
      1006  2.503            prime    raced           5924       8747    1.39  4.9e-16  bluestein
      1007  19.53            flat     raced           4864      14817    2.95  6.7e-16  
      1008  2^4.3^2.7        chain3   raced           1340       2325    1.29  5.1e-16  
      1009  1009             prime    raced           5761      15439    2.65  6.8e-16  bluestein
      1010  2.5.101          flat     raced           6631      10037    1.30  5.5e-16  
      1011  3.337            prime    raced           5915      14124    2.35  6.1e-16  bluestein
      1012  2^2.11.23        chain3   raced           2058       7593    3.60  4.4e-16  
      1013  1013             prime    raced           6797      15318    2.22  5.8e-16  bluestein
      1014  2.3.13^2         chain3   raced           1843       3545    1.87  6.8e-16  
      1015  5.7.29           chain3   raced           2828       7398    2.51  4.5e-16  
      1016  2^3.127          flat     raced           6336       8846    1.28  6.0e-16  
      1017  3^2.113          flat     raced           6046      13704    2.24  5.3e-16  
      1018  2.509            prime    raced           5911       9184    1.40  5.0e-16  bluestein
      1019  1019             prime    raced           5894       8869    1.26  5.1e-16  bluestein
      1020  2^2.3.5.17       chain3   raced           1775       6524    3.67  5.6e-16  
      1021  1021             prime    raced           6728       8981    1.23  5.5e-16  bluestein
      1022  2.7.73           flat     raced           4658      11613    2.23  5.9e-16  
      1023  3.11.31          chain3   raced           2595       8605    2.83  4.7e-16  
      1024  2^10             2p       raced           1078       1108    1.01  3.0e-16  
      1025  5^2.41           2p       raced           2845       8938    3.06  5.0e-16  
      1026  2.3^3.19         chain3   raced           1961       6383    3.20  5.4e-16  
      1027  13.79            flat     raced           6073      11442    1.87  7.3e-16  
      1028  2^2.257          flat     raced           6811       9528    1.39  4.1e-16  
      1029  3.7^3            chain3   raced           1617       3482    2.14  3.2e-16  
      1030  2.5.103          flat     raced           6503      14169    1.78  7.3e-16  
      1031  1031             prime    raced           6869      14112    1.72  7.9e-16  bluestein
      1032  2^3.3.43         chain3   raced           2781       9483    2.95  4.6e-16  
      1033  1033             prime    raced           6899      10255    1.44  7.4e-16  bluestein
      1034  2.11.47          flat     raced           4952      10615    2.00  5.7e-16  
      1035  3^2.5.23         chain3   raced           2483       6938    2.73  4.7e-16  
      1036  2^2.7.37         flat     raced           2930       8239    2.76  3.8e-16  
      1037  17.61            flat     raced           4861      16335    3.13  6.2e-16  
      1038  2.3.173          prime    raced           6872      15358    2.22  6.8e-16  bluestein
      1039  1039             prime    raced           6890      12977    1.47  7.3e-16  bluestein
      1040  2^4.5.13         chain3   raced           1461       2046    1.37  6.6e-16  
      1041  3.347            prime    raced           6834      12531    1.50  7.2e-16  bluestein
      1042  2.521            prime    raced           6848      11730    1.52  6.5e-16  bluestein
      1043  7.149            prime    raced           6905      11533    1.56  6.0e-16  bluestein
      1044  2^2.3^2.29       chain3   raced           2334       7560    3.09  7.1e-16  
      1045  5.11.19          chain3   raced           2267       7363    3.20  4.4e-16  
      1046  2.523            prime    raced           6850      14245    1.72  6.1e-16  bluestein
      1047  3.349            prime    raced           6828      12521    1.50  7.3e-16  bluestein
      1048  2^3.131          flat     raced           5197      12935    2.47  8.3e-16  
      1049  1049             prime    raced           6873      10312    1.49  7.8e-16  bluestein
      1050  2.3.5^2.7        chain3   raced           1639       2064    1.15  4.2e-16  
      1051  1051             prime    raced           6834      10426    1.48  6.8e-16  bluestein
      1052  2^2.263          prime    raced           6907      12288    1.74  7.6e-16  bluestein
      1053  3^4.13           chain3   raced           1827       2867    1.50  6.9e-16  
      1054  2.17.31          flat     raced           3804      12923    3.05  8.4e-16  
      1055  5.211            prime    raced           6813      16861    2.43  7.6e-16  bluestein
      1056  2^5.3.11         chain3   raced           1333       2306    1.71  4.2e-16  
      1057  7.151            prime    raced           6832      10371    1.48  5.5e-16  bluestein
      1058  2.23^2           flat     raced           4268      12568    2.83  6.0e-16  
      1059  3.353            prime    raced           6871      14727    2.13  5.8e-16  bluestein
      1060  2^2.5.53         flat     raced           4771      10643    1.83  6.6e-16  
      1061  1061             prime    raced           6809      13151    1.55  8.2e-16  bluestein
      1062  2.3^2.59         prime    raced           7152      11835    1.58  5.3e-16  bluestein
      1063  1063             prime    raced           6887      12949    1.51  7.6e-16  bluestein
      1064  2^3.7.19         chain3   raced           1963       6388    2.96  3.6e-16  
      1065  3.5.71           flat     raced           5327      11835    2.20  5.6e-16  
      1066  2.13.41          flat     raced           4166      10245    2.20  5.3e-16  
      1067  11.97            flat     raced           6247      11061    1.70  5.6e-16  
      1068  2^2.3.89         flat     raced           4840      11674    2.36  6.1e-16  
      1069  1069             prime    raced           6916      10139    1.45  6.2e-16  bluestein
      1070  2.5.107          prime    raced           6835      17044    2.46  6.2e-16  bluestein
      1071  3^2.7.17         chain3   raced           1973       6945    3.48  4.6e-16  
      1072  2^4.67           flat     raced           4150      12675    3.01  5.1e-16  
      1073  29.37            flat     raced           3969      14735    3.29  4.9e-16  
      1074  2.3.179          prime    raced           6838      15436    1.99  7.4e-16  bluestein
      1075  5^2.43           2p       raced           3009       9717    3.05  4.8e-16  
      1076  2^2.269          prime    raced           6845      12713    1.76  6.0e-16  bluestein
      1077  3.359            prime    raced           6872      12553    1.50  6.3e-16  bluestein
      1078  2.7^2.11         flat     raced           2903       3235    1.08  5.5e-16  
      1079  13.83            prime    raced           6953      16261    1.97  5.6e-16  bluestein
      1080  2^3.3^3.5        chain3   raced           1544       1712    1.09  5.2e-16  
      1081  23.47            2p       raced           3961      15562    3.39  6.0e-16  
      1082  2.541            prime    raced           6897      13329    1.79  6.4e-16  bluestein
      1083  3.19^2           chain3   raced           2767      11694    3.61  3.8e-16  
      1084  2^2.271          prime    raced           6889      12040    1.55  8.0e-16  bluestein
      1085  5.7.31           chain3   raced           3173       8139    2.55  4.7e-16  
      1086  2.3.181          prime    raced           7087      12627    1.61  5.3e-16  bluestein
      1087  1087             prime    raced           6892      18912    2.69  6.6e-16  bluestein
      1088  2^6.17           2p       raced           1771       5915    3.31  6.2e-16  
      1089  3^2.11^2         chain3   raced           2028       4046    1.90  4.6e-16  
      1090  2.5.109          flat     raced           7577      15219    1.75  5.1e-16  
      1091  1091             prime    raced           7038      19953    2.50  7.5e-16  bluestein
      1092  2^2.3.7.13       chain3   raced           1761       2926    1.58  5.2e-16  
      1093  1093             prime    raced           6913      18722    2.68  9.5e-16  bluestein
      1094  2.547            prime    raced           7041      19129    2.52  6.5e-16  bluestein
      1095  3.5.73           flat     raced           5123      12927    2.32  6.1e-16  
      1096  2^3.137          flat     raced           5041      13097    2.31  7.2e-16  
      1097  1097             prime    raced           6923      13299    1.82  7.5e-16  bluestein
      1098  2.3^2.61         flat     raced           4912      12747    2.50  6.5e-16  
      1099  7.157            prime    raced           6920      10386    1.43  7.4e-16  bluestein
      1100  2^2.5^2.11       chain3   raced           1639       2395    1.38  3.5e-16  
      1101  3.367            prime    raced           6920      13301    1.88  7.5e-16  bluestein
      1102  2.19.29          flat     raced           3926      13374    3.40  6.6e-16  
      1103  1103             prime    raced           6884      13127    1.83  5.3e-16  bluestein
      1104  2^4.3.23         chain3   raced           2455       8506    3.37  5.5e-16  
      1105  5.13.17          chain3   raced           2153       7673    3.49  7.4e-16  
      1106  2.7.79           flat     raced           5731      12323    2.03  7.8e-16  
      1107  3^3.41           2p       raced           3421       9832    2.75  5.2e-16  
      1108  2^2.277          prime    raced           6923      16793    2.33  4.9e-16  bluestein
      1109  1109             prime    raced           6898      13434    1.89  8.0e-16  bluestein
      1110  2.3.5.37         chain3   raced           3436       8930    2.48  5.7e-16  
      1111  11.101           flat     raced           8322      12289    1.31  7.5e-16  
      1112  2^3.139          flat     raced           6698      12035    1.61  7.2e-16  
      1113  3.7.53           flat     raced           5303      11571    2.17  8.4e-16  
      1114  2.557            prime    raced           7032      19404    2.48  6.4e-16  bluestein
      1115  5.223            prime    raced           6885      13667    1.79  9.8e-16  bluestein
      1116  2^2.3^2.31       chain3   raced           2660       8091    2.86  5.0e-16  
      1117  1117             prime    raced           6908      13154    1.90  6.5e-16  bluestein
      1118  2.13.43          flat     raced           4401      11011    2.40  6.0e-16  
      1119  3.373            prime    raced           6920      13059    1.74  6.8e-16  bluestein
      1120  2^5.5.7          chain3   raced           1470       1705    1.16  4.3e-16  
      1121  19.59            prime    raced           7429      17306    2.02  6.2e-16  bluestein
      1122  2.3.11.17        chain3   raced           2056       7624    3.69  5.3e-16  
      1123  1123             prime    raced           7748      13031    1.67  6.0e-16  bluestein
      1124  2^2.281          flat     raced           7407      14733    1.96  6.6e-16  
      1125  3^2.5^3          chain3   raced           1815       3129    1.71  4.5e-16  
      1126  2.563            prime    raced           7412      17668    2.36  6.8e-16  bluestein
      1127  7^2.23           chain3   raced           2425       7553    2.71  5.8e-16  
      1128  2^3.3.47         chain3   raced           3332      11780    3.51  5.2e-16  
      1129  1129             prime    raced           7607      16338    2.14  5.9e-16  bluestein
      1130  2.5.113          flat     raced           6160      14999    2.42  4.8e-16  
      1131  3.13.29          chain3   raced           3348       9288    2.70  9.0e-16  
      1132  2^2.283          prime    raced           7542      14559    1.89  5.5e-16  bluestein
      1133  11.103           prime    raced           7420      20579    2.46  5.7e-16  bluestein
      1134  2.3^4.7          chain3   raced           1739       2381    1.24  6.5e-16  
      1135  5.227            prime    raced           7618      11117    1.27  5.4e-16  bluestein
      1136  2^4.71           flat     raced           4132      12176    2.86  4.8e-16  
      1137  3.379            prime    raced           7454       9718    1.26  6.8e-16  bluestein
      1138  2.569            prime    raced           7718      16536    2.12  4.3e-16  bluestein
      1139  17.67            flat     raced           5855      18924    2.92  7.6e-16  
      1140  2^2.3.5.19       chain3   raced           2154       6654    2.81  4.7e-16  
      1141  7.163            prime    raced           7424      13202    1.63  6.3e-16  bluestein
      1142  2.571            prime    raced           7705      19503    2.18  6.4e-16  bluestein
      1143  3^2.127          flat     raced           8995       9863    0.99  8.6e-16  
      1144  2^3.11.13        chain3   raced           1865       4736    2.54  6.2e-16  
      1145  5.229            prime    raced           7580      11041    1.41  5.1e-16  bluestein
      1146  2.3.191          prime    raced           7533       8984    1.17  5.4e-16  bluestein
      1147  31.37            flat     raced           4902      16376    3.16  5.1e-16  
      1148  2^2.7.41         chain3   raced           3441      11729    3.18  4.1e-16  
      1149  3.383            prime    raced           7450      10290    1.27  6.7e-16  bluestein
      1150  2.5^2.23         chain3   raced           2726       7312    2.52  5.1e-16  
      1151  1151             prime    raced           7516      13608    1.44  6.3e-16  bluestein
      1152  2^7.3^2          chain3   raced           1345       2813    2.08  3.9e-16  
      1153  1153             prime    raced           5939      11595    1.70  6.7e-16  rader
      1154  2.577            prime    raced          12252      12389    0.99  6.2e-16  bluestein
      1155  3.5.7.11         chain3   raced           1821       3487    1.83  3.3e-16  
      1156  2^2.17^2         chain3   raced           2509      12338    4.84  5.8e-16  
      1157  13.89            flat     raced           5698      13426    2.28  6.5e-16  
      1158  2.3.193          flat     raced           9442      11082    1.17  5.3e-16  
      1159  19.61            flat     raced           5743      18095    2.99  5.2e-16  
      1160  2^3.5.29         chain3   raced           2776       7980    2.67  4.6e-16  
      1161  3^3.43           2p       raced           3720      10457    2.76  5.8e-16  
      1162  2.7.83           flat     raced           9515      17106    1.70  4.8e-16  
      1163  1163             prime    raced          14148      10639    0.75  7.8e-16  bluestein
      1164  2^2.3.97         flat     raced           5666      11244    1.89  5.4e-16  
      1165  5.233            flat     raced          13079      11105    0.84  6.1e-16  
      1166  2.11.53          flat     raced           5005      12960    2.53  5.9e-16  
      1167  3.389            prime    raced          13035      10986    0.84  4.7e-16  bluestein
      1168  2^4.73           flat     raced           4033      12525    3.08  5.0e-16  
      1169  7.167            prime    raced          13323      18867    1.00  5.5e-16  bluestein
      1170  2.3^2.5.13       chain3   raced           2052       2650    1.27  7.1e-16  
      1171  1171             prime    raced           7698      10719    1.33  7.6e-16  rader
      1172  2^2.293          prime    raced          12671      11296    0.87  5.6e-16  bluestein
      1173  3.17.23          chain3   raced           2962      14254    4.76  7.3e-16  
      1174  2.587            prime    raced          14407      11504    0.74  5.9e-16  bluestein
      1175  5^2.47           2p       raced           3674      11445    3.01  4.4e-16  
      1176  2^3.3.7^2        chain3   raced           1671       2856    1.31  3.9e-16  
      1177  11.107           prime    raced          13659      17113    1.04  5.6e-16  bluestein
      1178  2.19.31          flat     raced           4945      18038    2.96  4.7e-16  
      1179  3^2.131          flat     raced           7578      14512    1.86  6.7e-16  
      1180  2^2.5.59         flat     raced           8335      12953    1.51  5.8e-16  
      1181  1181             prime    raced          13375      10737    0.64  5.7e-16  flips differ 1.27x bluestein
      1182  2.3.197          flat     raced           8922      11085    1.19  5.3e-16  
      1183  7.13^2           chain3   raced           2240       4383    1.89  7.8e-16  
      1184  2^5.37           2p       raced           2367       9197    3.67  4.6e-16  
      1185  3.5.79           flat     raced           6863      12623    1.83  7.9e-16  
      1186  2.593            prime    raced          12476      11538    0.90  5.5e-16  bluestein
      1187  1187             prime    raced          14602      10812    0.73  5.6e-16  bluestein
      1188  2^2.3^3.11       chain3   raced           1835       3027    1.60  5.5e-16  
      1189  29.41            2p       raced           4251      16932    3.56  5.3e-16  
      1190  2.5.7.17         chain3   raced           2175       7027    3.09  6.4e-16  
      1191  3.397            prime    raced          14239      11087    0.76  5.6e-16  bluestein
      1192  2^3.149          flat     raced           7176      13031    1.76  6.6e-16  
      1193  1193             prime    raced          13372      10679    0.79  6.0e-16  bluestein
      1194  2.3.199          flat     raced          10714      10665    0.97  6.3e-16  
      1195  5.239            flat     raced          11599      11113    0.92  6.5e-16  
      1196  2^2.13.23        chain3   raced           2849       9071    2.88  6.4e-16  
      1197  3^2.7.19         chain3   raced           2595       7370    2.81  3.8e-16  
      1198  2.599            prime    raced          12843      11775    0.88  5.2e-16  bluestein
      1199  11.109           flat     raced           8292      15361    1.68  7.3e-16  
      1200  2^4.3.5^2        chain3   raced           1608       1613    0.98  4.0e-16  
      1201  1201             prime    raced           8431      11613    1.24  6.8e-16  rader
      1202  2.601            prime    raced          12535      12120    0.95  5.8e-16  bluestein
      1203  3.401            prime    raced          12738      11708    0.90  3.8e-16  bluestein
      1204  2^2.7.43         flat     raced           3795      11973    2.98  6.0e-16  
      1205  5.241            flat     raced          10309      12212    1.03  6.1e-16  
      1206  2.3^2.67         flat     raced           5659      14684    2.54  6.8e-16  
      1207  17.71            flat     raced           5724      18323    3.15  7.3e-16  
      1208  2^3.151          flat     raced           8379      10220    1.21  6.6e-16  
      1209  3.13.31          chain3   raced           3952      10333    2.59  5.6e-16  
      1210  2.5.11^2         chain3   raced           2047       4231    2.00  3.3e-16  
      1211  7.173            prime    raced          12756      16433    1.08  6.4e-16  bluestein
      1212  2^2.3.101        flat     raced           7073      12305    1.72  6.1e-16  
      1213  1213             prime    raced          12706      17069    1.20  4.8e-16  bluestein
      1214  2.607            prime    raced          12865      17844    1.35  6.5e-16  bluestein
      1215  3^5.5            chain3   raced           1936       3127    1.60  3.9e-16  
      1216  2^6.19           2p       raced           2324       6821    2.83  3.4e-16  
      1217  1217             prime    raced           7719      14132    1.81  5.3e-16  rader
      1218  2.3.7.29         chain3   raced           2981       8858    2.55  4.4e-16  
      1219  23.53            flat     raced           6013      18764    3.09  8.0e-16  
      1220  2^2.5.61         flat     raced           5421      13437    2.45  4.4e-16  
      1221  3.11.37          chain3   raced           4125      11340    2.40  5.4e-16  
      1222  2.13.47          flat     raced           4977      12622    2.47  5.7e-16  
      1223  1223             prime    raced          12893      14035    1.06  5.3e-16  bluestein
      1224  2^3.3^2.17       chain3   raced           2090       6969    3.33  5.4e-16  
      1225  5^2.7^2          chain3   raced           1889       3475    1.77  5.0e-16  
      1226  2.613            prime    raced           7675      16485    2.12  5.4e-16  bluestein
      1227  3.409            prime    raced           7720      13522    1.57  6.8e-16  bluestein
      1228  2^2.307          prime    raced           7530      20540    2.63  4.5e-16  bluestein
      1229  1229             prime    raced           7628      14310    1.69  6.5e-16  bluestein
      1230  2.3.5.41         chain3   raced           3884      10774    2.70  4.4e-16  
      1231  1231             prime    raced           7698      14332    1.84  6.6e-16  bluestein
      1232  2^4.7.11         chain3   raced           1686       3598    1.70  4.1e-16  
      1233  3^2.137          flat     raced           7466      13185    1.68  1.0e-15  
      1234  2.617            prime    raced           7614      16589    2.12  6.4e-16  bluestein
      1235  5.13.19          chain3   raced           2591       8853    3.23  6.3e-16  
      1236  2^2.3.103        flat     raced           6803      17802    2.10  7.2e-16  
      1237  1237             prime    raced           7669      13829    1.75  5.7e-16  bluestein
      1238  2.619            prime    raced           7729      16024    1.98  6.7e-16  bluestein
      1239  3.7.59           prime    raced           7609      13864    1.81  5.2e-16  bluestein
      1240  2^3.5.31         chain3   raced           3007       8798    2.66  5.0e-16  
      1241  17.73            flat     raced           5500      18696    3.27  7.0e-16  
      1242  2.3^3.23         chain3   raced           3001       8112    2.67  5.8e-16  
      1243  11.113           flat     raced           6805      18026    2.62  5.4e-16  
      1244  2^2.311          prime    raced           8369      20013    2.34  7.6e-16  bluestein
      1245  3.5.83           prime    raced           7808      18760    2.12  4.6e-16  bluestein
      1246  2.7.89           flat     raced           5824      13541    2.31  5.6e-16  
      1247  29.43            2p       raced           4509      18287    3.94  4.9e-16  
      1248  2^5.3.13         chain3   raced           1828       2932    1.52  5.5e-16  
      1249  1249             prime    raced           7420      14508    1.94  7.6e-16  rader
      1250  2.5^4            chain3   raced           1960       2678    1.12  4.1e-16  
      1251  3^2.139          prime    raced           7771      13610    1.62  6.6e-16  bluestein
      1252  2^2.313          prime    raced           7660      20094    2.60  6.5e-16  bluestein
      1253  7.179            prime    raced           7645      14038    1.78  7.2e-16  bluestein
      1254  2.3.11.19        chain3   raced           2461       8760    3.49  5.0e-16  
      1255  5.251            prime    raced           8004      10620    1.28  5.9e-16  bluestein
      1256  2^3.157          flat     raced           6553      10333    1.55  7.2e-16  
      1257  3.419            prime    raced           7641      16374    1.75  7.6e-16  bluestein
      1258  2.17.37          flat     raced           4621      15855    3.21  6.1e-16  
      1259  1259             prime    raced           8003      11987    1.47  5.3e-16  bluestein
      1260  2^2.3^2.5.7      chain3   raced           1826       2959    1.22  4.4e-16  
      1261  13.97            flat     raced           6908      13198    1.88  5.4e-16  
      1262  2.631            prime    raced           7774      10977    1.39  4.6e-16  bluestein
      1263  3.421            prime    raced           7734      16448    1.72  6.0e-16  bluestein
      1264  2^4.79           flat     raced           5236      13000    2.46  6.9e-16  
      1265  5.11.23          chain3   raced           2752       9539    3.07  5.0e-16  
      1266  2.3.211          prime    raced           7585      17082    1.85  6.5e-16  bluestein
      1267  7.181            prime    raced           7815      14355    1.65  7.0e-16  bluestein
      1268  2^2.317          prime    raced           7915      10536    1.29  5.3e-16  bluestein
      1269  3^3.47           2p       raced           4365      12117    2.68  6.4e-16  
      1270  2.5.127          flat     raced           9467      10937    1.15  7.3e-16  
      1271  31.41            flat     raced           5321      18555    3.46  5.0e-16  
      1272  2^3.3.53         flat     raced           4854      13457    2.63  7.1e-16  
      1273  19.67            flat     raced           6541      21947    2.93  5.8e-16  
      1274  2.7^2.13         flat     raced           3624       3757    1.01  4.7e-16  
      1275  3.5^2.17         chain3   raced           2671       8046    3.00  5.9e-16  
      1276  2^2.11.29        chain3   raced           3042      10450    2.91  3.9e-16  
      1277  1277             prime    raced           7787      11911    1.51  4.7e-16  bluestein
      1278  2.3^2.71         flat     raced           5679      13959    2.38  6.4e-16  
      1279  1279             prime    raced           7616      11802    1.40  5.5e-16  bluestein
      1280  2^8.5            chain3   raced           1707       1454    0.84  4.3e-16  
      1281  3.7.61           flat     raced           6040      15068    2.35  4.7e-16  
      1282  2.641            prime    raced           8693      11912    1.36  6.0e-16  bluestein
      1283  1283             prime    raced           8977      18689    1.57  7.6e-16  bluestein
      1284  2^2.3.107        prime    raced           8681      17851    1.72  7.8e-16  bluestein
      1285  5.257            prime    raced           8748      12508    1.42  6.6e-16  bluestein
      1286  2.643            prime    raced           8876      15288    1.68  6.6e-16  bluestein
      1287  3^2.11.13        chain3   raced           2297       4889    2.05  5.3e-16  
      1288  2^3.7.23         chain3   raced           2571       8238    2.54  4.5e-16  flips differ 1.25x
      1289  1289             prime    raced           8727      17855    1.67  5.0e-16  bluestein
      1290  2.3.5.43         chain3   raced           4268      11335    2.51  5.8e-16  
      1291  1291             prime    raced           8752      17814    1.51  7.2e-16  bluestein
      1292  2^2.17.19        chain3   raced           3057      13631    4.32  5.8e-16  
      1293  3.431            prime    raced           8729      13314    1.41  6.8e-16  bluestein
      1294  2.647            prime    raced           9081      14992    1.62  6.6e-16  bluestein
      1295  5.7.37           chain3   raced           5010      10629    1.80  5.3e-16  
      1296  2^4.3^4          chain3   raced           1739       6016    3.37  5.2e-16  
      1297  1297             prime    raced           8825      16523    1.46  6.0e-16  bluestein
      1298  2.11.59          prime    raced           8740      15769    1.79  6.4e-16  bluestein
      1299  3.433            prime    raced           8755      14361    1.58  6.6e-16  bluestein
      1300  2^2.5^2.13       chain3   raced           2138       2908    1.34  5.4e-16  
      1301  1301             prime    raced           8713      22492    2.42  6.2e-16  bluestein
      1302  2.3.7.31         chain3   raced           3301       9716    2.58  5.2e-16  
      1303  1303             prime    raced           9221      18981    1.75  6.8e-16  bluestein
      1304  2^3.163          flat     raced           8475      14916    1.59  7.1e-16  
      1305  3^2.5.29         chain3   raced           4065      10293    2.49  6.7e-16  
      1306  2.653            prime    raced           9376      16253    1.60  7.2e-16  bluestein
      1307  1307             prime    raced           8730      21704    2.43  5.9e-16  bluestein
      1308  2^2.3.109        flat     raced           7529      16795    2.19  5.9e-16  
      1309  7.11.17          chain3   raced           2457       8992    3.63  6.1e-16  
      1310  2.5.131          flat     raced           7748      17498    2.05  7.6e-16  
      1311  3.19.23          chain3   raced           3416      15739    4.35  4.7e-16  
      1312  2^5.41           2p       raced           2816      10847    3.82  5.3e-16  
      1313  13.101           flat     raced           8300      14423    1.56  6.5e-16  
      1314  2.3^2.73         flat     raced           5600      14428    2.57  6.6e-16  
      1315  5.263            prime    raced           8684      15356    1.74  8.1e-16  bluestein
      1316  2^2.7.47         flat     raced           4316      12427    2.70  5.6e-16  
      1317  3.439            prime    raced           8681      13521    1.54  5.5e-16  bluestein
      1318  2.659            prime    raced           8713      16510    1.84  6.2e-16  bluestein
      1319  1319             prime    raced           8676      18594    1.74  6.7e-16  bluestein
      1320  2^3.3.5.11       chain3   raced           2035       2969    1.44  5.1e-16  
      1321  1321             prime    raced           8533      15751    1.80  6.1e-16  rader
      1322  2.661            prime    raced           8708      16508    1.86  5.8e-16  bluestein
      1323  3^3.7^2          chain3   raced           2146       4368    2.01  4.2e-16  
      1324  2^2.331          flat     raced           9428      19349    2.02  6.9e-16  
      1325  5^2.53           flat     raced           6115      14000    2.16  6.2e-16  
      1326  2.3.13.17        chain3   raced           2588       9113    3.45  6.2e-16  
      1327  1327             prime    raced           8691      21659    2.41  6.5e-16  bluestein
      1328  2^4.83           flat     raced           8430      18984    2.24  5.5e-16  
      1329  3.443            prime    raced           9050      16592    1.48  5.8e-16  bluestein
      1330  2.5.7.19         chain3   raced           2665       7896    2.69  4.3e-16  
      1331  11^3             chain3   raced           2571       6024    2.26  4.5e-16  
      1332  2^2.3^2.37       chain3   raced           3182      10624    3.20  5.5e-16  
      1333  31.43            flat     raced           5681      19739    3.37  5.1e-16  
      1334  2.23.29          flat     raced           5077      16620    2.71  5.5e-16  
      1335  3.5.89           flat     raced           6544      15034    2.00  8.4e-16  
      1336  2^3.167          prime    raced           8935      17814    1.39  5.8e-16  bluestein
      1337  7.191            prime    raced           8768      12527    1.38  6.0e-16  bluestein
      1338  2.3.223          prime    raced           8950      15615    1.72  7.6e-16  bluestein
      1339  13.103           flat     raced           8332      24738    2.91  7.8e-16  
      1340  2^2.5.67         flat     raced           6510      16012    2.42  5.7e-16  
      1341  3^2.149          prime    raced           9110      15017    1.57  6.3e-16  bluestein
      1342  2.11.61          flat     raced           5790      16522    2.74  5.7e-16  
      1343  17.79            flat     raced           7181      19875    2.69  7.4e-16  
      1344  2^6.3.7          chain3   raced           1698       1998    1.15  4.1e-16  
      1345  5.269            prime    raced          12892      15579    1.10  5.4e-16  bluestein
      1346  2.673            prime    raced          13326      16366    1.15  6.0e-16  bluestein
      1347  3.449            prime    raced          13597      13560    0.97  6.9e-16  bluestein
      1348  2^2.337          flat     raced          10308      19132    1.83  7.0e-16  
      1349  19.71            flat     raced           6377      26285    4.09  6.2e-16  
      1350  2.3^3.5^2        chain3   raced           2056       2655    1.28  5.0e-16  
      1351  7.193            flat     raced          11168      12366    1.09  5.6e-16  
      1352  2^3.13^2         chain3   raced           2390       4715    1.94  5.8e-16  
      1353  3.11.41          chain3   raced           4618      13087    2.51  4.0e-16  
      1354  2.677            prime    raced           9838      17875    1.43  7.4e-16  bluestein
      1355  5.271            prime    raced          10027      17240    1.37  6.8e-16  bluestein
      1356  2^2.3.113        flat     raced           6251      18743    2.91  4.6e-16  
      1357  23.59            prime    raced          10487      21924    1.95  6.6e-16  bluestein
      1358  2.7.97           flat     raced           6906      13410    1.93  5.0e-16  
      1359  3^2.151          flat     raced          11085      11741    0.95  1.1e-15  
      1360  2^4.5.17         chain3   raced           2223       9034    4.06  5.3e-16  
      1361  1361             prime    raced           9119      17135    0.83  6.5e-16  flips differ 1.77x rader
      1362  2.3.227          prime    raced           9854      13398    1.31  7.3e-16  bluestein
      1363  29.47            flat     raced           6221      20435    3.23  4.0e-16  
      1364  2^2.11.31        chain3   raced           3802      11487    2.90  5.4e-16  
      1365  3.5.7.13         chain3   raced           2277       4086    1.78  5.0e-16  
      1366  2.683            prime    raced           9669      22058    2.23  7.6e-16  bluestein
      1367  1367             prime    raced           9916      17068    1.35  7.6e-16  bluestein
      1368  2^3.3^2.19       chain3   raced           2960       8780    2.90  5.1e-16  
      1369  37^2             2p       raced           5436      20244    3.36  6.8e-16  
      1370  2.5.137          flat     raced           8009      14687    1.79  7.5e-16  
      1371  3.457            prime    raced          10296      13084    1.16  6.9e-16  bluestein
      1372  2^2.7^3          flat     raced           3268       3218    0.93  4.5e-16  
      1373  1373             prime    raced           9656      17047    1.37  8.3e-16  bluestein
      1374  2.3.229          prime    raced           9635      13212    1.27  5.5e-16  bluestein
      1375  5^3.11           chain3   raced           2200       4623    2.07  4.2e-16  
      1376  2^5.43           2p       raced           3075      12191    3.96  5.4e-16  
      1377  3^4.17           chain3   raced           2588       8423    3.17  8.2e-16  
      1378  2.13.53          flat     raced           5921      15478    2.52  7.3e-16  
      1379  7.197            prime    raced           9667      12538    1.27  6.7e-16  bluestein
      1380  2^2.3.5.23       chain3   raced           3226       8787    2.72  4.2e-16  
      1381  1381             prime    raced           9815      16899    1.35  7.8e-16  bluestein
      1382  2.691            prime    raced           9901      17901    1.49  6.8e-16  bluestein
      1383  3.461            prime    raced           9673      13038    1.31  6.3e-16  bluestein
      1384  2^3.173          flat     raced           8674      20119    2.29  6.0e-16  
      1385  5.277            prime    raced           9817      20716    2.09  7.1e-16  bluestein
      1386  2.3^2.7.11       chain3   raced           2231       3763    1.66  5.6e-16  
      1387  19.73            flat     raced           6320      21889    3.00  5.5e-16  
      1388  2^2.347          prime    raced           9623      19798    2.02  8.7e-16  bluestein
      1389  3.463            prime    raced           9662      12903    1.32  9.5e-16  bluestein
      1390  2.5.139          flat     raced          10256      14523    1.24  6.4e-16  
      1391  13.107           prime    raced           9820      19986    1.70  6.7e-16  bluestein
      1392  2^4.3.29         chain3   raced           3200      10239    3.18  4.0e-16  
      1393  7.199            prime    raced           9775      12520    1.26  6.4e-16  bluestein
      1394  2.17.41          flat     raced           5449      18693    3.29  5.4e-16  
      1395  3^2.5.31         chain3   raced           3995      10601    2.49  7.0e-16  
      1396  2^2.349          prime    raced          10646      20240    1.88  7.8e-16  bluestein
      1397  11.127           flat     raced          11092      13403    1.17  6.9e-16  
      1398  2.3.233          prime    raced          10386      13793    1.21  6.5e-16  bluestein
      1399  1399             prime    raced           9833      13451    1.36  7.7e-16  bluestein
      1400  2^3.5^2.7        chain3   raced           2118       2184    1.01  4.6e-16  
      1401  3.467            prime    raced           9742      12859    1.24  8.6e-16  bluestein
      1402  2.701            prime    raced           9787      18387    1.60  7.9e-16  bluestein
      1403  23.61            flat     raced           6927      22880    3.05  6.1e-16  
      1404  2^2.3^3.13       chain3   raced           2265       3529    1.53  5.5e-16  
      1405  5.281            prime    raced           9711      18452    1.79  7.1e-16  bluestein
      1406  2.19.37          flat     raced           5883      18337    2.97  5.0e-16  
      1407  3.7.67           flat     raced           7131      17300    2.42  6.5e-16  
      1408  2^7.11           chain3   raced           1713       2858    1.64  4.0e-16  
      1409  1409             prime    raced           7473      17080    1.77  6.0e-16  rader
      1410  2.3.5.47         chain3   raced           5552      13088    2.02  5.6e-16  
      1411  17.83            prime    raced           9746      26641    2.62  6.5e-16  bluestein
      1412  2^2.353          flat     raced           8856      19734    2.17  5.5e-16  
      1413  3^2.157          flat     raced          10361      11964    1.13  7.6e-16  
      1414  2.7.101          flat     raced           9113      14622    1.58  8.2e-16  
      1415  5.283            prime    raced           9814      21155    2.06  6.9e-16  bluestein
      1416  2^3.3.59         flat     raced           9112      16871    1.78  6.8e-16  
      1417  13.109           flat     raced           9980      19609    1.92  9.8e-16  
      1418  2.709            prime    raced           9643      22320    1.58  7.1e-16  flips differ 1.69x bluestein
      1419  3.11.43          flat     raced           5292      14774    2.70  4.7e-16  
      1420  2^2.5.71         flat     raced           6265      15684    2.43  5.3e-16  
      1421  7^2.29           chain3   raced           3424      13279    3.43  5.2e-16  
      1422  2.3^2.79         flat     raced           6881      14927    2.15  7.5e-16  
      1423  1423             prime    raced           9732      13361    1.31  7.6e-16  bluestein
      1424  2^4.89           flat     raced           5069      22025    3.69  4.7e-16  
      1425  3.5^2.19         chain3   raced           2871       9232    2.83  4.3e-16  
      1426  2.23.31          flat     raced           5466      18317    2.72  6.1e-16  
      1427  1427             prime    raced           9822      20728    2.07  6.3e-16  bluestein
      1428  2^2.3.7.17       chain3   raced           2550       8598    3.32  6.3e-16  
      1429  1429             prime    raced           9741      24801    2.13  7.2e-16  bluestein
      1430  2.5.11.13        chain3   raced           2523       5001    1.97  5.8e-16  
      1431  3^3.53           flat     raced           6686      14933    2.11  5.9e-16  
      1432  2^3.179          prime    raced          10154      19930    1.74  6.3e-16  bluestein
      1433  1433             prime    raced           9697      13768    1.36  7.7e-16  bluestein
      1434  2.3.239          prime    raced          10128      13600    1.21  7.7e-16  bluestein
      1435  5.7.41           chain3   raced           5578      12672    2.21  5.2e-16  
      1436  2^2.359          prime    raced           9703      16970    1.43  7.9e-16  bluestein
      1437  3.479            prime    raced           9681      13020    1.20  7.2e-16  bluestein
      1438  2.719            prime    raced           9912      17807    1.44  6.5e-16  bluestein
      1439  1439             prime    raced           9652      17593    1.50  7.2e-16  bluestein
      1440  2^5.3^2.5        chain3   raced           1877       3894    1.85  5.3e-16  
      1441  11.131           flat     raced           8994      18700    2.03  5.9e-16  
      1442  2.7.103          flat     raced           8085      25335    2.97  6.6e-16  
      1443  3.13.37          chain3   raced           4462      13253    2.81  5.3e-16  
      1444  2^2.19^2         chain3   raced           4035      16309    3.15  3.6e-16  flips differ 1.28x
      1445  5.17^2           chain3   raced           3313      15595    4.45  6.1e-16  
      1446  2.3.241          flat     raced          10594      14174    1.32  5.2e-16  
      1447  1447             prime    raced          14167      22453    1.35  7.5e-16  bluestein
      1448  2^3.181          flat     raced           7798      16448    1.85  9.0e-16  
      1449  3^2.7.23         chain3   raced           3157       9748    2.73  5.0e-16  
      1450  2.5^2.29         chain3   raced           3996      10950    2.54  5.1e-16  
      1451  1451             prime    raced          14091      18705    1.32  5.1e-16  bluestein
      1452  2^2.3.11^2       chain3   raced           2277       5532    2.19  4.0e-16  
      1453  1453             prime    raced           8768      22240    2.11  6.8e-16  rader
      1454  2.727            prime    raced          12786      20985    1.49  6.2e-16  bluestein
      1455  3.5.97           flat     raced           8154      16369    1.86  5.8e-16  
      1456  2^4.7.13         chain3   raced           2154       4860    1.77  5.8e-16  
      1457  31.47            2p       raced           8042      26983    2.74  6.6e-16  
      1458  2.3^6            chain3   raced           2565       4040    1.49  5.1e-16  
      1459  1459             prime    raced          13674      18526    1.10  5.4e-16  flips differ 1.27x bluestein
      1460  2^2.5.73         flat     raced           6393      15743    2.44  5.1e-16  
      1461  3.487            prime    raced          13090      14220    1.08  5.5e-16  bluestein
      1462  2.17.43          flat     raced           5768      19764    3.34  6.8e-16  
      1463  7.11.19          chain3   raced           3145      10331    3.25  4.2e-16  
      1464  2^3.3.61         flat     raced           5652      17930    3.00  5.8e-16  
      1465  5.293            prime    raced          12824      14208    1.07  5.7e-16  bluestein
      1466  2.733            prime    raced          12920      19558    1.23  6.5e-16  bluestein
      1467  3^2.163          flat     raced          15455      18467    1.13  8.3e-16  
      1468  2^2.367          prime    raced          12825      17280    1.34  6.9e-16  bluestein
      1469  13.113           flat     raced           7637      20973    2.74  7.3e-16  
      1470  2.3.5.7^2        chain3   raced           2293       2826    1.21  4.6e-16  
      1471  1471             prime    raced           9918      18552    1.51  7.1e-16  bluestein
      1472  2^6.23           2p       raced           2944       9288    3.05  5.4e-16  
      1473  3.491            prime    raced          10358      14241    1.32  6.6e-16  bluestein
      1474  2.11.67          flat     raced           6704      19398    2.83  5.2e-16  
      1475  5^2.59           prime    raced          10257      16479    1.55  6.1e-16  bluestein
      1476  2^2.3^2.41       chain3   raced           3932      12501    3.13  4.2e-16  
      1477  7.211            prime    raced          10805      24518    2.06  6.3e-16  bluestein
      1478  2.739            prime    raced           9786      15879    1.54  5.2e-16  bluestein
      1479  3.17.29          chain3   raced           4357      17894    3.70  5.3e-16  
      1480  2^3.5.37         chain3   raced           3678      11921    3.17  4.1e-16  
      1481  1481             prime    raced          10071      15262    1.42  5.7e-16  bluestein
      1482  2.3.13.19        chain3   raced           3458      10943    3.16  6.7e-16  
      1483  1483             prime    raced          10002      19654    1.68  5.9e-16  bluestein
      1484  2^2.7.53         flat     raced           6430      15185    2.35  6.8e-16  
      1485  3^3.5.11         chain3   raced           2462       4358    1.74  4.4e-16  
      1486  2.743            prime    raced           9968      22995    1.61  6.8e-16  bluestein
      1487  1487             prime    raced          10384      22531    1.94  5.4e-16  bluestein
      1488  2^4.3.31         chain3   raced           3202      12496    3.83  5.1e-16  
      1489  1489             prime    raced           9921      15117    1.41  6.7e-16  bluestein
      1490  2.5.149          prime    raced           9914      16234    1.56  7.2e-16  bluestein
      1491  3.7.71           flat     raced           7001      19125    2.23  5.8e-16  
      1492  2^2.373          prime    raced           9980      17267    1.70  6.8e-16  bluestein
      1493  1493             prime    raced           9897      18573    1.50  6.3e-16  bluestein
      1494  2.3^2.83         prime    raced          10308      22056    1.95  4.8e-16  bluestein
      1495  5.13.23          chain3   raced           3852      11399    2.87  6.1e-16  
      1496  2^3.11.17        chain3   raced           2611      10187    3.78  5.1e-16  
      1497  3.499            prime    raced           9926      14236    1.42  7.4e-16  bluestein
      1498  2.7.107          prime    raced          10225      20700    1.65  7.1e-16  bluestein
      1499  1499             prime    raced           9900      22937    2.19  6.3e-16  bluestein
      1500  2^2.3.5^3        chain3   raced           2504       3584    1.16  4.4e-16  
      1501  19.79            flat     raced           8627      22911    2.64  8.2e-16  
      1502  2.751            prime    raced           9908      14879    1.33  5.6e-16  bluestein
      1503  3^2.167          prime    raced          10201      22545    2.11  7.6e-16  bluestein
      1504  2^5.47           2p       raced           3729      13756    3.65  3.7e-16  
      1505  5.7.43           flat     raced           5785      13867    2.17  4.3e-16  
      1506  2.3.251          prime    raced          10086      11964    1.09  5.9e-16  bluestein
      1507  11.137           flat     raced           8969      17576    1.72  7.9e-16  
      1508  2^2.13.29        chain3   raced           4274      12427    2.81  4.7e-16  
      1509  3.503            prime    raced           9894      13528    1.29  6.1e-16  bluestein
      1510  2.5.151          flat     raced          11040      12918    1.07  6.0e-16  
      1511  1511             prime    raced          11172      16186    1.26  6.0e-16  bluestein
      1512  2^3.3^3.7        chain3   raced           2219       3739    1.20  5.3e-16  
      1513  17.89            flat     raced           6899      30274    3.79  8.5e-16  
      1514  2.757            prime    raced          10295      14402    1.39  6.2e-16  bluestein
      1515  3.5.101          flat     raced           9534      15989    1.53  6.4e-16  
      1516  2^2.379          flat     raced          12892      13050    0.87  6.9e-16  
      1517  37.41            flat     raced           6152      23352    3.64  5.1e-16  
      1518  2.3.11.23        chain3   raced           3286      11485    3.46  4.2e-16  
      1519  7^2.31           chain3   raced           3843      11595    2.52  5.3e-16  
      1520  2^4.5.19         chain3   raced           2679       8933    3.25  4.2e-16  
      1521  3^2.13^2         chain3   raced           2887       5665    1.95  7.1e-16  
      1522  2.761            prime    raced           9964      14453    1.39  6.8e-16  bluestein
      1523  1523             prime    raced           9874      14858    1.40  5.9e-16  bluestein
      1524  2^2.3.127        flat     raced          10518      14280    1.34  7.2e-16  
      1525  5^2.61           flat     raced           7359      17900    2.38  4.9e-16  
      1526  2.7.109          flat     raced           9200      19769    2.11  6.5e-16  
      1527  3.509            prime    raced           9978      13505    1.33  7.3e-16  bluestein
      1528  2^3.191          prime    raced          10618      11974    1.09  5.1e-16  bluestein
      1529  11.139           prime    raced          10363      18044    1.65  6.1e-16  bluestein
      1530  2.3^2.5.17       chain3   raced           2822       9166    3.15  5.8e-16  
      1531  1531             prime    raced           9922      15260    1.37  6.2e-16  bluestein
      1532  2^2.383          prime    raced          10198      13326    1.27  6.2e-16  bluestein
      1533  3.7.73           flat     raced           7132      17214    2.39  6.7e-16  
      1534  2.13.59          prime    raced          10211      18809    1.80  5.9e-16  bluestein
      1535  5.307            prime    raced          10422      26431    2.40  6.3e-16  bluestein
      1536  2^9.3            chain3   raced           2099       2227    0.94  4.2e-16  
      1537  29.53            flat     raced           7902      25149    3.09  8.5e-16  
      1538  2.769            prime    raced          10404      14765    1.37  7.5e-16  bluestein
      1539  3^4.19           chain3   raced           3283       9912    2.89  6.2e-16  
      1540  2^2.5.7.11       flat     raced           3966       4702    0.97  4.6e-16  
      1541  23.67            flat     raced           7781      26635    3.40  8.4e-16  
      1542  2.3.257          prime    raced          10109      15320    1.40  5.6e-16  bluestein
      1543  1543             prime    raced          10272      28859    2.53  5.6e-16  bluestein
      1544  2^3.193          flat     raced           7655      14791    1.82  5.1e-16  
      1545  3.5.103          flat     raced           9361      22863    1.92  7.9e-16  
      1546  2.773            prime    raced          10311      15733    1.48  5.1e-16  bluestein
      1547  7.13.17          chain3   raced           3120      10756    3.41  6.7e-16  
      1548  2^2.3^2.43       chain3   raced           4167      14088    3.35  5.4e-16  
      1549  1549             prime    raced           9902      28642    2.83  5.3e-16  bluestein
      1550  2.5^2.31         chain3   raced           4080      11634    2.65  6.9e-16  
      1551  3.11.47          flat     raced           6434      16268    2.49  6.6e-16  
      1552  2^4.97           flat     raced           6241      15171    2.38  4.7e-16  
      1553  1553             prime    raced          10321      31546    2.75  6.3e-16  bluestein
      1554  2.3.7.37         chain3   raced           5938      12873    1.60  4.8e-16  flips differ 1.35x
      1555  5.311            prime    raced           9916      26860    2.56  7.1e-16  bluestein
      1556  2^2.389          prime    raced          10011      14936    1.44  5.1e-16  bluestein
      1557  3^2.173          prime    raced           9980      26935    2.33  7.1e-16  bluestein
      1558  2.19.41          flat     raced           7438      21020    2.81  4.8e-16  
      1559  1559             prime    raced           9992      28823    2.80  6.2e-16  bluestein
      1560  2^3.3.5.13       chain3   raced           2503       4361    1.01  5.5e-16  flips differ 2.06x
      1561  7.223            prime    raced           9937      18280    1.83  6.6e-16  bluestein
      1562  2.11.71          flat     raced           6703      20230    2.95  6.2e-16  
      1563  3.521            prime    raced           9904      21386    1.78  7.5e-16  bluestein
      1564  2^2.17.23        chain3   raced           4234      17869    4.06  5.6e-16  
      1565  5.313            prime    raced           9960      25016    2.49  6.2e-16  bluestein
      1566  2.3^3.29         chain3   raced           4390      11219    2.53  4.9e-16  
      1567  1567             prime    raced           9950      15199    1.49  7.7e-16  bluestein
      1568  2^5.7^2          chain3   raced           2010       2743    1.35  3.8e-16  
      1569  3.523            prime    raced          10286      21325    1.66  5.9e-16  bluestein
      1570  2.5.157          flat     raced          10314      13180    1.27  7.0e-16  
      1571  1571             prime    raced          10350      15285    1.47  6.6e-16  bluestein
      1572  2^2.3.131        flat     raced           8001      19719    2.42  7.6e-16  
      1573  11^2.13          chain3   raced           2880       7062    2.44  5.4e-16  
      1574  2.787            prime    raced          10325      15133    1.40  5.7e-16  bluestein
      1575  3^2.5^2.7        chain3   raced           2608       4845    1.79  4.1e-16  
      1576  2^3.197          flat     raced           7522      14121    1.85  4.5e-16  
      1577  19.83            prime    raced          10333      30395    2.83  5.3e-16  bluestein
      1578  2.3.263          prime    raced          10666      19808    1.73  6.8e-16  bluestein
      1579  1579             prime    raced          10485      15011    1.42  4.4e-16  bluestein
      1580  2^2.5.79         flat     raced           7666      16582    2.10  6.0e-16  
      1581  3.17.31          chain3   raced           4824      19137    3.91  4.6e-16  
      1582  2.7.113          flat     raced           7710      21875    2.80  4.9e-16  
      1583  1583             prime    raced          10622      15116    1.41  6.3e-16  bluestein
      1584  2^4.3^2.11       chain3   raced           2760       6476    2.01  5.2e-16  
      1585  5.317            prime    raced          10358      14010    1.26  5.4e-16  bluestein
      1586  2.13.61          flat     raced           7094      19490    2.71  6.3e-16  
      1587  3.23^2           flat     raced           5387      18883    3.43  6.0e-16  
      1588  2^2.397          prime    raced          10547      15200    1.37  6.5e-16  bluestein
      1589  7.227            prime    raced          10589      15553    1.43  6.6e-16  bluestein
      1590  2.3.5.53         flat     raced           6743      16215    2.29  6.5e-16  
      1591  37.43            flat     raced           7303      24879    3.30  5.6e-16  
      1592  2^3.199          flat     raced           8979      14915    1.51  5.6e-16  
      1593  3^3.59           prime    raced          10444      18008    1.72  5.9e-16  bluestein
      1594  2.797            prime    raced          10515      15129    1.44  5.6e-16  bluestein
      1595  5.11.29          chain3   raced           4034      13283    3.27  4.0e-16  
      1596  2^2.3.7.19       chain3   raced           3148      10313    2.94  4.8e-16  
      1597  1597             prime    raced          10416      15354    1.43  6.4e-16  bluestein
      1598  2.17.47          flat     raced           6752      22595    3.28  5.6e-16  
      1599  3.13.41          chain3   raced           5326      15578    2.80  5.5e-16  
      1600  2^6.5^2          chain3   raced           2001       2151    1.07  2.7e-16  
      1601  1601             prime    raced           8977      15432    1.67  5.0e-16  rader
      1602  2.3^2.89         flat     raced           7247      18210    2.49  6.1e-16  
      1603  7.229            prime    raced          13627      17972    0.57  4.6e-16  flips differ 2.11x bluestein
      1604  2^2.401          flat     raced          13914      15439    1.08  5.4e-16  
      1605  3.5.107          prime    raced          14196      27149    1.32  6.6e-16  flips differ 1.46x bluestein
      1606  2.11.73          flat     raced           6612      20092    3.01  5.6e-16  
      1607  1607             prime    raced          14418      32415    1.43  7.1e-16  bluestein
      1608  2^3.3.67         flat     raced           6562      21112    3.19  5.6e-16  
      1609  1609             prime    raced          13130      29872    1.42  6.2e-16  bluestein
      1610  2.5.7.23         chain3   raced           3428      10413    2.60  4.7e-16  
      1611  3^2.179          prime    raced          13034      20016    1.30  5.6e-16  bluestein
      1612  2^2.13.31        chain3   raced           4038      13683    2.92  6.1e-16  
      1613  1613             prime    raced          13783      30229    1.34  5.8e-16  bluestein
      1614  2.3.269          prime    raced          14529      18491    1.23  6.4e-16  bluestein
      1615  5.17.19          chain3   raced           3860      17134    4.23  5.4e-16  
      1616  2^4.101          flat     raced           7696      16493    1.90  7.0e-16  
      1617  3.7^2.11         chain3   raced           2654       5274    1.86  4.8e-16  
      1618  2.809            prime    raced          13634      26430    1.87  6.4e-16  bluestein
      1619  1619             prime    raced          12703      29464    1.45  6.6e-16  bluestein
      1620  2^2.3^4.5        chain3   raced           2462       4967    2.01  6.0e-16  
      1621  1621             prime    raced          13092      19253    1.31  5.5e-16  bluestein
      1622  2.811            prime    raced          14124      17858    1.24  6.2e-16  bluestein
      1623  3.541            prime    raced          13833      19623    1.27  6.0e-16  bluestein
      1624  2^3.7.29         chain3   raced           3666      11916    2.63  4.4e-16  
      1625  5^3.13           chain3   raced           2930       5594    1.89  5.6e-16  
      1626  2.3.271          flat     raced          14943      21653    1.07  6.5e-16  
      1627  1627             prime    raced          14017      27278    1.93  8.0e-16  bluestein
      1628  2^2.11.37        chain3   raced           5441      14904    2.70  4.2e-16  
      1629  3^2.181          flat     raced          12017      18415    1.36  6.6e-16  
      1630  2.5.163          flat     raced          12260      18796    1.52  7.6e-16  
      1631  7.233            prime    raced          13523      15658    1.12  5.0e-16  bluestein
      1632  2^5.3.17         chain3   raced           2712       9362    3.40  5.9e-16  
      1633  23.71            flat     raced           8008      26124    3.12  5.9e-16  
      1634  2.19.43          flat     raced           7154      22216    2.88  4.5e-16  
      1635  3.5.109          flat     raced          10448      19759    1.70  6.3e-16  
      1636  2^2.409          flat     raced          13215      26429    1.97  7.0e-16  
      1637  1637             prime    raced          12825      27432    2.10  5.2e-16  bluestein
      1638  2.3^2.7.13       chain3   raced           2783       4439    1.56  5.5e-16  
      1639  11.149           prime    raced          12885      19537    1.48  5.4e-16  bluestein
      1640  2^3.5.41         chain3   raced           4309      13956    3.11  4.8e-16  
      1641  3.547            prime    raced          12845      26385    2.04  5.8e-16  bluestein
      1642  2.821            prime    raced          13948      17711    1.21  6.0e-16  bluestein
      1643  31.53            flat     raced           8415      26586    3.10  6.6e-16  
      1644  2^2.3.137        flat     raced           8120      18101    2.20  7.5e-16  
      1645  5.7.47           flat     raced           6847      15865    2.26  4.4e-16  
      1646  2.823            prime    raced          12843      17682    1.35  5.8e-16  bluestein
      1647  3^3.61           flat     raced           7706      19061    2.44  6.8e-16  
      1648  2^4.103          flat     raced           7503      23136    2.38  6.5e-16  
      1649  17.97            flat     raced           9008      23176    2.54  6.0e-16  
      1650  2.3.5^2.11       chain3   raced           2763       4016    1.31  3.9e-16  
      1651  13.127           flat     raced          11355      15902    1.29  7.3e-16  
      1652  2^2.7.59         flat     raced          11319      18228    1.60  4.9e-16  
      1653  3.19.29          chain3   raced           4827      19779    3.33  4.4e-16  
      1654  2.827            prime    raced          13475      21956    1.17  6.1e-16  flips differ 1.65x bluestein
      1655  5.331            prime    raced          12398      23705    1.67  6.6e-16  bluestein
      1656  2^3.3^2.23       chain3   raced           3938      12135    2.77  4.9e-16  
      1657  1657             prime    raced          12673      27316    1.99  7.5e-16  bluestein
      1658  2.829            prime    raced          13147      17626    1.32  5.1e-16  bluestein
      1659  3.7.79           flat     raced           8570      17926    2.06  7.2e-16  
      1660  2^2.5.83         flat     raced          11741      23997    2.01  7.0e-16  
      1661  11.151           flat     raced          13878      15816    1.10  7.6e-16  
      1662  2.3.277          prime    raced          13062      21489    1.31  5.9e-16  bluestein
      1663  1663             prime    raced          13929      27431    1.79  5.8e-16  bluestein
      1664  2^7.13           chain3   raced           2202       3720    1.65  4.4e-16  
      1665  3^2.5.37         chain3   raced           5274      13812    2.59  5.2e-16  
      1666  2.7^2.17         flat     raced           5261      10587    1.98  6.1e-16  
      1667  1667             prime    raced          12782      27564    2.12  5.9e-16  bluestein
      1668  2^2.3.139        flat     raced          10284      18401    1.55  6.7e-16  
      1669  1669             prime    raced          13561      27812    1.97  6.4e-16  bluestein
      1670  2.5.167          prime    raced          12833      25413    1.95  5.8e-16  bluestein
      1671  3.557            prime    raced          13130      26236    1.90  6.0e-16  bluestein
      1672  2^3.11.19        chain3   raced           3201      11749    3.30  5.0e-16  
      1673  7.239            prime    raced          14009      15735    1.11  4.7e-16  bluestein
      1674  2.3^3.31         chain3   raced           4337      12643    2.60  5.3e-16  
      1675  5^2.67           flat     raced           8702      20630    2.29  7.3e-16  
      1676  2^2.419          prime    raced          14020      22303    1.28  6.6e-16  bluestein
      1677  3.13.43          chain3   raced           5932      16760    2.74  6.8e-16  
      1678  2.839            prime    raced          12976      17810    1.33  5.8e-16  bluestein
      1679  23.73            flat     raced           8365      29199    3.46  6.1e-16  
      1680  2^4.3.5.7        chain3   raced           2347       2673    1.12  5.3e-16  
      1681  41^2             flat     raced           6937      26395    3.56  4.3e-16  
      1682  2.29^2           flat     raced           7591      22379    2.91  7.0e-16  
      1683  3^2.11.17        chain3   raced           3245      11824    3.62  8.0e-16  
      1684  2^2.421          flat     raced          14528      26718    1.79  7.1e-16  
      1685  5.337            prime    raced          12211      23859    1.92  6.9e-16  bluestein
      1686  2.3.281          flat     raced          12700      21864    1.71  5.9e-16  
      1687  7.241            flat     raced          12186      16519    1.30  6.2e-16  
      1688  2^3.211          flat     raced           9828      22430    1.87  6.7e-16  
      1689  3.563            prime    raced          12133      26651    2.13  6.0e-16  bluestein
      1690  2.5.13^2         chain3   raced           3240       5977    1.79  7.1e-16  
      1691  19.89            flat     raced           7976      34545    3.50  6.1e-16  
      1692  2^2.3^2.47       chain3   raced           4819      15980    3.09  4.8e-16  
      1693  1693             prime    raced          12151      25730    2.09  8.0e-16  bluestein
      1694  2.7.11^2         flat     raced           4768       6764    1.41  5.9e-16  
      1695  3.5.113          flat     raced           8626      23839    2.70  6.0e-16  
      1696  2^5.53           flat     raced           6790      16961    2.46  7.0e-16  
      1697  1697             prime    raced          12169      28626    2.11  7.1e-16  bluestein
      1698  2.3.283          prime    raced          12174      22345    1.44  6.7e-16  bluestein
      1699  1699             prime    raced          12118      21859    1.39  6.5e-16  bluestein
      1700  2^2.5^2.17       chain3   raced           3138       9935    3.08  6.3e-16  
      1701  3^5.7            chain3   raced           2789       5821    1.99  4.3e-16  
      1702  2.23.37          flat     raced           6790      23232    2.92  5.3e-16  
      1703  13.131           flat     raced           9975      22390    2.23  8.1e-16  
      1704  2^3.3.71         flat     raced           6572      18812    2.85  6.2e-16  
      1705  5.11.31          chain3   raced           5461      14493    2.17  4.8e-16  
      1706  2.853            prime    raced          12161      17738    1.45  7.0e-16  bluestein
      1707  3.569            prime    raced          12225      25400    2.06  7.0e-16  bluestein
      1708  2^2.7.61         flat     raced           7710      19259    2.48  4.2e-16  
      1709  1709             prime    raced          13012      21773    1.32  6.8e-16  bluestein
      1710  2.3^2.5.19       chain3   raced           3503      10223    2.63  6.0e-16  
      1711  29.59            prime    raced          12246      27007    2.20  6.6e-16  bluestein
      1712  2^4.107          prime    raced          14338      31053    2.14  5.4e-16  bluestein
      1713  3.571            prime    raced          12130      25795    2.04  7.1e-16  bluestein
      1714  2.857            prime    raced          12173      17568    1.43  7.6e-16  bluestein
      1715  5.7^3            flat     raced           4305       5991    1.38  3.8e-16  
      1716  2^2.3.11.13      chain3   raced           2858       6372    2.21  6.8e-16  
      1717  17.101           flat     raced          10716      25302    2.17  7.7e-16  
      1718  2.859            prime    raced          12137      17917    1.47  6.8e-16  bluestein
      1719  3^2.191          prime    raced          12133      13747    1.10  6.5e-16  bluestein
      1720  2^3.5.43         chain3   raced           4811      14830    3.06  4.6e-16  
      1721  1721             prime    raced          12124      17361    1.42  7.4e-16  bluestein
      1722  2.3.7.41         chain3   raced           5795      14817    2.23  4.7e-16  
      1723  1723             prime    raced          12284      22293    1.42  6.0e-16  bluestein
      1724  2^2.431          prime    raced          12240      22350    1.49  7.8e-16  bluestein
      1725  3.5^2.23         chain3   raced           4263      12378    2.83  5.8e-16  
      1726  2.863            prime    raced          12127      17846    1.43  6.7e-16  bluestein
      1727  11.157           flat     raced          11963      16391    1.34  1.0e-15  
      1728  2^6.3^3          chain3   raced           2192       2700    1.22  4.7e-16  
      1729  7.13.19          chain3   raced           3818      12382    2.88  5.2e-16  
      1730  2.5.173          prime    raced          11065      25419    2.01  7.2e-16  bluestein
      1731  3.577            prime    raced          11195      17342    1.46  6.5e-16  bluestein
      1732  2^2.433          prime    raced          10931      20660    1.70  7.0e-16  bluestein
      1733  1733             prime    raced          11024      21368    0.98  8.3e-16  flips differ 2.39x bluestein
      1734  2.3.17^2         chain3   raced           3804      21289    5.48  4.2e-16  
      1735  5.347            prime    raced          11027      17311    1.49  5.6e-16  bluestein
      1736  2^3.7.31         chain3   raced           4222      13767    2.72  5.2e-16  
      1737  3^2.193          flat     raced          12363      16617    1.33  6.2e-16  
      1738  2.11.79          flat     raced           8240      20138    2.42  7.9e-16  
      1739  37.47            flat     raced           7488      27723    3.67  6.0e-16  
      1740  2^2.3.5.29       chain3   raced           3918      12379    2.84  4.6e-16  
      1741  1741             prime    raced          11165      16980    1.44  5.8e-16  bluestein
      1742  2.13.67          flat     raced           7858      23246    2.78  6.1e-16  
      1743  3.7.83           prime    raced          11265      25937    2.21  5.4e-16  bluestein
      1744  2^4.109          flat     raced           9186      22572    2.42  7.1e-16  
      1745  5.349            prime    raced          11005      17418    1.38  7.5e-16  bluestein
      1746  2.3^2.97         flat     raced           8124      17181    2.07  5.1e-16  
      1747  1747             prime    raced          10965      25617    1.45  5.5e-16  bluestein
      1748  2^2.19.23        chain3   raced           5340      20215    3.67  4.7e-16  
      1749  3.11.53          flat     raced           7918      19878    2.38  6.6e-16  
      1750  2.5^3.7          chain3   raced           2877       4473    1.49  4.5e-16  
      1751  17.103           flat     raced          10301      38564    3.60  7.6e-16  
      1752  2^3.3.73         flat     raced           6468      19152    2.83  6.3e-16  
      1753  1753             prime    raced          11021      25509    2.31  5.1e-16  bluestein
      1754  2.877            prime    raced          10978      22789    1.66  5.6e-16  bluestein
      1755  3^3.5.13         chain3   raced           3058       5380    1.70  6.7e-16  
      1756  2^2.439          prime    raced          10984      26808    2.37  4.8e-16  bluestein
      1757  7.251            prime    raced          11015      14197    1.18  6.0e-16  bluestein
      1758  2.3.293          prime    raced          11025      17702    1.59  6.5e-16  bluestein
      1759  1759             prime    raced          11097      17385    1.55  5.3e-16  bluestein
      1760  2^5.5.11         chain3   raced           2748       3705    1.33  3.9e-16  
      1761  3.587            prime    raced          11060      17579    1.57  7.5e-16  bluestein
      1762  2.881            prime    raced          11004      18334    1.49  6.4e-16  bluestein
      1763  41.43            flat     raced           8219      28704    3.37  6.6e-16  
      1764  2^2.3^2.7^2      chain3   raced           2730       6069    2.20  5.1e-16  
      1765  5.353            prime    raced          10995      21226    1.57  8.4e-16  bluestein
      1766  2.883            prime    raced          11063      18411    1.61  6.8e-16  bluestein
      1767  3.19.31          chain3   raced           5395      21658    3.20  4.7e-16  flips differ 1.25x
      1768  2^3.13.17        chain3   raced           3315      12900    3.85  7.7e-16  
      1769  29.61            flat     raced           8311      30398    3.54  5.6e-16  
      1770  2.3.5.59         prime    raced          11456      19416    1.66  5.3e-16  bluestein
      1771  7.11.23          chain3   raced           3928      13679    3.40  5.7e-16  
      1772  2^2.443          prime    raced          11134      22638    1.65  6.5e-16  bluestein
      1773  3^2.197          prime    raced          11159      16019    1.21  7.0e-16  bluestein
      1774  2.887            prime    raced          11132      22573    1.65  5.9e-16  bluestein
      1775  5^2.71           flat     raced           9061      28837    3.17  7.0e-16  
      1776  2^4.3.37         chain3   raced           4060      14376    3.42  5.0e-16  
      1777  1777             prime    raced          11109      25348    2.24  1.0e-15  bluestein
      1778  2.7.127          flat     raced          12411      16261    1.29  7.3e-16  
      1779  3.593            prime    raced          11152      17463    1.56  6.7e-16  bluestein
      1780  2^2.5.89         flat     raced           7810      20092    2.53  5.7e-16  
      1781  13.137           flat     raced           9608      20879    2.12  7.6e-16  
      1782  2.3^4.11         chain3   raced           2880       5022    1.73  7.0e-16  
      1783  1783             prime    raced          10973      25438    2.27  6.4e-16  bluestein
      1784  2^3.223          prime    raced          11057      21578    1.86  7.2e-16  bluestein
      1785  3.5.7.17         chain3   raced           3530      11423    3.15  6.7e-16  
      1786  2.19.47          flat     raced           8180      25563    2.96  6.0e-16  
      1787  1787             prime    raced          11030      21010    1.50  7.0e-16  bluestein
      1788  2^2.3.149        prime    raced          11160      20346    1.75  6.1e-16  bluestein
      1789  1789             prime    raced          11031      21082    1.52  6.5e-16  bluestein
      1790  2.5.179          prime    raced          11208      21449    1.60  6.4e-16  bluestein
      1791  3^2.199          prime    raced          11000      16258    1.42  5.4e-16  bluestein
      1792  2^8.7            chain3   raced           2356       2799    1.16  2.4e-16  
      1793  11.163           flat     raced          15451      22465    1.45  5.1e-16  
      1794  2.3.13.23        chain3   raced           4517      13682    2.90  6.1e-16  
      1795  5.359            prime    raced          13077      22228    1.36  6.1e-16  bluestein
      1796  2^2.449          flat     raced          12432      26523    2.10  5.7e-16  
      1797  3.599            prime    raced          12912      18228    1.33  4.8e-16  bluestein
      1798  2.29.31          flat     raced           7880      24526    3.00  6.3e-16  
      1799  7.257            flat     raced          14480      16805    1.12  5.1e-16  
      1800  2^3.3^2.5^2      chain3   raced           2686       3832    1.02  4.2e-16  
      1801  1801             prime    raced          13214      20478    1.38  4.7e-16  flips differ 1.25x bluestein
      1802  2.17.53          flat     raced           7829      27557    3.35  8.1e-16  
      1803  3.601            prime    raced          14275      18190    1.21  5.3e-16  bluestein
      1804  2^2.11.41        chain3   raced           5977      17628    2.48  4.8e-16  
      1805  5.19^2           chain3   raced           4541      20388    3.86  5.0e-16  
      1806  2.3.7.43         chain3   raced           6615      16000    2.34  5.7e-16  
      1807  13.139           flat     raced          12729      21246    1.50  7.7e-16  
      1808  2^4.113          flat     raced           6664      24807    3.59  5.2e-16  
      1809  3^3.67           flat     raced           8984      22641    2.49  6.1e-16  
      1810  2.5.181          flat     raced          11950      21406    1.64  5.1e-16  
      1811  1811             prime    raced          13218      29865    2.05  5.7e-16  bluestein
      1812  2^2.3.151        flat     raced          11541      19261    1.55  6.8e-16  
      1813  7^2.37           chain3   raced           5997      15013    2.23  5.0e-16  
      1814  2.907            prime    raced          13121      32753    2.40  6.2e-16  bluestein
      1815  3.5.11^2         chain3   raced           3143       7523    2.29  5.0e-16  
      1816  2^3.227          prime    raced          13154      17700    1.27  5.5e-16  bluestein
      1817  23.79            flat     raced           9797      28310    2.71  7.2e-16  
      1818  2.3^2.101        flat     raced          10179      18774    1.83  6.6e-16  
      1819  17.107           prime    raced          13340      32660    2.07  5.8e-16  bluestein
      1820  2^2.5.7.13       flat     raced           4682       5913    0.87  6.1e-16  flips differ 1.27x
      1821  3.607            prime    raced          12673      26836    2.02  5.7e-16  bluestein
      1822  2.911            prime    raced          13168      31754    2.35  6.3e-16  bluestein
      1823  1823             prime    raced          12861      30771    2.33  7.4e-16  bluestein
      1824  2^5.3.19         chain3   raced           3757      13159    3.41  4.8e-16  
      1825  5^2.73           flat     raced           8769      20512    2.32  6.4e-16  
      1826  2.11.83          prime    raced          14443      28367    1.95  5.8e-16  bluestein
      1827  3^2.7.29         chain3   raced           5169      13493    2.53  6.2e-16  
      1828  2^2.457          prime    raced          12910      17563    1.34  5.7e-16  bluestein
      1829  31.59            prime    raced          13297      31009    1.04  6.2e-16  flips differ 2.22x bluestein
      1830  2.3.5.61         flat     raced           7855      20537    2.56  5.3e-16  
      1831  1831             prime    raced          12824      25746    1.97  5.5e-16  bluestein
      1832  2^3.229          flat     raced          12513      17896    1.39  6.2e-16  
      1833  3.13.47          chain3   raced           6926      19121    2.65  6.5e-16  
      1834  2.7.131          flat     raced           9563      23126    2.37  9.5e-16  
      1835  5.367            prime    raced          13253      22444    1.59  6.3e-16  bluestein
      1836  2^2.3^3.17       chain3   raced           3467      11438    3.12  6.3e-16  
      1837  11.167           prime    raced          13124      29523    2.16  7.7e-16  bluestein
      1838  2.919            prime    raced          14234      31658    2.19  5.4e-16  bluestein
      1839  3.613            prime    raced          13923      24574    1.67  5.8e-16  bluestein
      1840  2^4.5.23         chain3   raced           3497      11481    3.26  4.2e-16  
      1841  7.263            prime    raced          13145      21537    1.58  7.1e-16  bluestein
      1842  2.3.307          prime    raced          12858      30219    2.15  6.8e-16  bluestein
      1843  19.97            flat     raced           9647      26958    2.74  6.9e-16  
      1844  2^2.461          prime    raced          13511      17647    1.29  5.0e-16  bluestein
      1845  3^2.5.41         chain3   raced           6912      16423    2.33  5.4e-16  
      1846  2.13.71          flat     raced           7836      23932    3.03  9.0e-16  
      1847  1847             prime    raced          12940      25823    1.85  5.7e-16  bluestein
      1848  2^3.3.7.11       chain3   raced           3414       5839    1.37  5.8e-16  
      1849  43^2             flat     raced           8534      30424    3.28  5.0e-16  
      1850  2.5^2.37         chain3   raced           6420      14766    2.15  4.7e-16  
      1851  3.617            prime    raced          13516      25225    1.82  5.8e-16  bluestein
      1852  2^2.463          flat     raced          15745      17848    1.06  6.7e-16  
      1853  17.109           flat     raced          11718      30611    2.43  7.3e-16  
      1854  2.3^2.103        flat     raced          10490      32420    3.01  6.8e-16  
      1855  5.7.53           flat     raced           8460      19239    2.16  6.8e-16  
      1856  2^6.29           2p       raced           3697      12791    3.21  4.1e-16  
      1857  3.619            prime    raced          12856      25042    1.90  6.1e-16  bluestein
      1858  2.929            prime    raced          12804      31916    2.47  6.4e-16  bluestein
      1859  11.13^2          chain3   raced           3672       8417    2.27  7.6e-16  
      1860  2^2.3.5.31       chain3   raced           4372      14563    3.07  8.1e-16  
      1861  1861             prime    raced          12826      24280    1.77  7.5e-16  bluestein
      1862  2.7^2.19         flat     raced           5871      11964    2.01  5.1e-16  
      1863  3^4.23           chain3   raced           4611      12866    2.76  5.5e-16  
      1864  2^3.233          prime    raced          12958      17865    1.37  4.8e-16  bluestein
      1865  5.373            prime    raced          12851      22819    1.67  5.6e-16  bluestein
      1866  2.3.311          prime    raced          12960      30718    2.37  4.9e-16  bluestein
      1867  1867             prime    raced          12784      24085    1.73  5.5e-16  bluestein
      1868  2^2.467          prime    raced          14220      17855    1.15  6.4e-16  bluestein
      1869  3.7.89           flat     raced           8773      20891    2.33  6.0e-16  
      1870  2.5.11.17        chain3   raced           3550      12858    3.61  5.0e-16  
      1871  1871             prime    raced          13017      24606    1.79  7.2e-16  bluestein
      1872  2^4.3^2.13       chain3   raced           3057       5455    1.40  6.7e-16  
      1873  1873             prime    raced          11247      24855    1.71  8.4e-16  flips differ 1.30x rader
      1874  2.937            prime    raced          13295      32213    2.31  6.3e-16  bluestein
      1875  3.5^4            chain3   raced           3189       6041    1.36  4.8e-16  flips differ 1.37x
      1876  2^2.7.67         flat     raced           8695      23763    2.62  5.4e-16  
      1877  1877             prime    raced          12972      17526    1.32  6.5e-16  bluestein
      1878  2.3.313          prime    raced          13241      29641    2.22  7.6e-16  bluestein
      1879  1879             prime    raced          13170      17385    1.32  4.5e-16  bluestein
      1880  2^3.5.47         chain3   raced           5472      17071    3.09  3.9e-16  
      1881  3^2.11.19        chain3   raced           3955      13530    3.13  6.1e-16  
      1882  2.941            prime    raced          13083      17614    1.31  5.9e-16  bluestein
      1883  7.269            prime    raced          13072      21706    1.62  5.6e-16  bluestein
      1884  2^2.3.157        flat     raced          10207      16599    1.56  7.7e-16  
      1885  5.13.29          chain3   raced           4716      15691    3.21  6.9e-16  
      1886  2.23.41          flat     raced           8614      26543    2.87  4.8e-16  
      1887  3.17.37          chain3   raced           5829      24296    3.67  6.2e-16  
      1888  2^5.59           flat     raced          12256      20397    1.61  6.1e-16  
      1889  1889             prime    raced          13080      18900    1.38  7.5e-16  bluestein
      1890  2.3^3.5.7        chain3   raced           3041       3754    0.77  6.5e-16  flips differ 1.61x
      1891  31.61            flat     raced           9136      35192    2.31  6.3e-16  flips differ 1.73x
      1892  2^2.11.43        flat     raced           6211      19111    3.07  5.0e-16  
      1893  3.631            prime    raced          13264      21001    1.28  6.0e-16  bluestein
      1894  2.947            prime    raced          13115      17584    1.17  7.5e-16  bluestein
      1895  5.379            prime    raced          12996      16157    1.22  7.0e-16  bluestein
      1896  2^3.3.79         flat     raced           8042      20154    2.42  7.6e-16  
      1897  7.271            prime    raced          12970      21523    1.61  5.2e-16  bluestein
      1898  2.13.73          flat     raced           7953      23734    2.93  6.3e-16  
      1899  3^2.211          flat     raced          14953      29676    1.97  7.9e-16  
      1900  2^2.5^2.19       chain3   raced           3740      11075    2.63  4.7e-16  
      1901  1901             prime    raced          12960      17553    1.32  6.3e-16  bluestein
      1902  2.3.317          prime    raced          13165      16254    1.17  6.1e-16  bluestein
      1903  11.173           prime    raced          13716      26882    1.67  5.4e-16  bluestein
      1904  2^4.7.17         chain3   raced           3253      11596    3.34  5.5e-16  
      1905  3.5.127          flat     raced          13322      17709    1.30  6.2e-16  
      1906  2.953            prime    raced          13565      17782    1.19  4.5e-16  bluestein
      1907  1907             prime    raced          13592      17323    1.05  4.7e-16  bluestein
      1908  2^2.3^2.53       flat     raced           7946      19458    2.41  8.4e-16  
      1909  23.83            prime    raced          12861      38970    2.91  5.8e-16  bluestein
      1910  2.5.191          prime    raced          13345      16264    1.10  3.8e-16  bluestein
      1911  3.7^2.13         chain3   raced           3344       6242    1.80  5.8e-16  
      1912  2^3.239          flat     raced           9992      17969    1.61  8.3e-16  
      1913  1913             prime    raced          12954      17317    1.33  6.3e-16  bluestein
      1914  2.3.11.29        chain3   raced           6667      15731    2.19  5.7e-16  
      1915  5.383            prime    raced          13078      16327    1.20  5.4e-16  bluestein
      1916  2^2.479          prime    raced          13267      17589    1.25  5.8e-16  bluestein
      1917  3^3.71           flat     raced           8942      24730    2.24  6.8e-16  
      1918  2.7.137          flat     raced           9878      21336    2.10  6.0e-16  
      1919  19.101           flat     raced          11627      28872    2.18  7.0e-16  
      1920  2^7.3.5          chain3   raced           2391       2507    1.01  4.7e-16  
      1921  17.113           flat     raced           9309      35175    3.66  5.9e-16  
      1922  2.31^2           flat     raced           7800      26950    2.89  6.8e-16  
      1923  3.641            prime    raced          13257      19137    1.41  4.9e-16  bluestein
      1924  2^2.13.37        flat     raced           5661      17616    2.92  6.8e-16  
      1925  5^2.7.11         chain3   raced           3176       6377    1.88  4.5e-16  
      1926  2.3^2.107        prime    raced          13465      35548    2.20  4.8e-16  bluestein
      1927  41.47            flat     raced           8600      32118    3.71  6.3e-16  
      1928  2^3.241          flat     raced           8983      18056    1.95  4.9e-16  
      1929  3.643            prime    raced          12961      22829    1.36  5.7e-16  flips differ 1.26x bluestein
      1930  2.5.193          flat     raced          12279      17708    1.42  6.6e-16  
      1931  1931             prime    raced          13815      41817    2.95  7.2e-16  bluestein
      1932  2^2.3.7.23       chain3   raced           4039      14143    2.83  4.5e-16  
      1933  1933             prime    raced          13721      22956    1.66  6.4e-16  bluestein
      1934  2.967            prime    raced          13449      33731    1.85  8.3e-16  bluestein
      1935  3^2.5.43         chain3   raced           7563      17736    2.25  6.2e-16  
      1936  2^4.11^2         chain3   raced           2780       7248    2.57  4.2e-16  
      1937  13.149           prime    raced          13034      23258    1.60  6.4e-16  bluestein
      1938  2.3.17.19        chain3   raced           4588      20516    4.27  7.5e-16  
      1939  7.277            prime    raced          13589      24418    1.44  6.7e-16  bluestein
      1940  2^2.5.97         flat     raced           9113      19037    1.99  5.7e-16  
      1941  3.647            prime    raced          14361      22552    1.54  6.0e-16  bluestein
      1942  2.971            prime    raced          13401      43167    3.02  7.8e-16  bluestein
      1943  29.67            flat     raced           9649      34866    3.28  6.4e-16  
      1944  2^3.3^5          chain3   raced           2951       6537    2.21  4.0e-16  
      1945  5.389            prime    raced          12658      19273    1.40  5.2e-16  bluestein
      1946  2.7.139          flat     raced          12605      21570    1.60  6.5e-16  
      1947  3.11.59          prime    raced          12964      23698    1.77  5.5e-16  bluestein
      1948  2^2.487          prime    raced          13210      19394    1.29  6.6e-16  bluestein
      1949  1949             prime    raced          13069      20674    1.53  6.7e-16  bluestein
      1950  2.3.5^2.13       chain3   raced           3293       4948    1.48  7.2e-16  
      1951  1951             prime    raced          13229      21940    1.50  6.5e-16  bluestein
      1952  2^5.61           flat     raced           7895      23899    2.77  5.9e-16  
      1953  3^2.7.31         chain3   raced           5630      15256    2.67  6.8e-16  
      1954  2.977            prime    raced          13342      19040    1.33  6.8e-16  bluestein
      1955  5.17.23          chain3   raced           5185      21974    4.01  5.6e-16  
      1956  2^2.3.163        flat     raced          14331      24165    1.44  8.6e-16  
      1957  19.103           flat     raced          11994      38931    2.78  7.2e-16  
      1958  2.11.89          flat     raced           8477      29220    3.18  6.0e-16  
      1959  3.653            prime    raced          14366      24978    1.70  6.7e-16  bluestein
      1960  2^3.5.7^2        flat     raced           4413       3661    0.82  4.2e-16  
      1961  37.53            flat     raced           9861      33435    3.32  9.1e-16  
      1962  2.3^2.109        flat     raced          11250      24965    2.09  6.6e-16  
      1963  13.151           flat     raced          14672      18967    1.20  8.7e-16  
      1964  2^2.491          flat     raced          13553      19394    1.39  5.9e-16  
      1965  3.5.131          flat     raced          11068      26364    2.18  9.5e-16  
      1966  2.983            prime    raced          13675      19912    1.41  5.9e-16  bluestein
      1967  7.281            prime    raced          13676      26551    1.83  5.7e-16  bluestein
      1968  2^4.3.41         chain3   raced           4793      18287    3.13  5.1e-16  flips differ 1.29x
      1969  11.179           prime    raced          13499      26161    1.49  6.0e-16  bluestein
      1970  2.5.197          flat     raced          11563      18170    1.49  5.4e-16  
      1971  3^3.73           flat     raced           8843      22599    2.51  5.8e-16  
      1972  2^2.17.29        chain3   raced           5239      24310    4.47  6.4e-16  
      1973  1973             prime    raced          13898      21673    1.40  5.6e-16  bluestein
      1974  2.3.7.47         flat     raced           8937      18828    2.08  5.7e-16  
      1975  5^2.79           flat     raced          10178      21210    1.82  7.2e-16  
      1976  2^3.13.19        chain3   raced           4557      15135    3.19  7.1e-16  
      1977  3.659            prime    raced          13517      29489    1.98  6.2e-16  bluestein
      1978  2.23.43          flat     raced           9641      28376    2.78  5.3e-16  
      1979  1979             prime    raced          13710      21106    1.44  4.7e-16  bluestein
      1980  2^2.3^2.5.11     chain3   raced           3132       5151    1.57  4.1e-16  
      1981  7.283            prime    raced          14122      25927    1.82  7.1e-16  bluestein
      1982  2.991            prime    raced          14221      20278    1.08  5.1e-16  bluestein
      1983  3.661            prime    raced          14299      25253    1.62  6.9e-16  bluestein
      1984  2^6.31           chain3   raced           4223      14389    3.03  4.9e-16  
      1985  5.397            prime    raced          13483      18594    1.31  5.8e-16  bluestein
      1986  2.3.331          prime    raced          13525      30896    2.18  6.0e-16  bluestein
      1987  1987             prime    raced          13004      20825    1.59  5.9e-16  bluestein
      1988  2^2.7.71         flat     raced           8980      29900    3.28  4.6e-16  
      1989  3^2.13.17        chain3   raced           4046      14180    3.42  6.5e-16  
      1990  2.5.199          flat     raced          13307      17720    1.32  5.8e-16  
      1991  11.181           flat     raced          14504      24775    1.58  6.3e-16  
      1992  2^3.3.83         prime    raced          13875      29547    1.92  5.0e-16  bluestein
      1993  1993             prime    raced          13381      21034    1.54  6.2e-16  bluestein
      1994  2.997            prime    raced          14032      19450    1.37  5.1e-16  bluestein
      1995  3.5.7.19         chain3   raced           4222      13982    3.13  4.1e-16  
      1996  2^2.499          prime    raced          13042      20648    1.52  5.6e-16  bluestein
      1997  1997             prime    raced          13228      20814    1.53  6.3e-16  bluestein
      1998  2.3^3.37         chain3   raced           6538      16242    2.42  5.3e-16  
      1999  1999             prime    raced          13814      21468    1.52  6.6e-16  bluestein
      2000  2^4.5^3          chain3   raced           2821       2856    0.96  3.9e-16  
      2001  3.23.29          chain3   raced           6053      25460    3.54  4.6e-16  
      2002  2.7.11.13        flat     raced           5485       8030    1.43  7.2e-16  
      2003  2003             prime    raced          12950      27167    2.04  5.9e-16  bluestein
      2004  2^2.3.167        prime    raced          13395      30546    2.26  4.5e-16  bluestein
      2005  5.401            prime    raced          13985      19215    1.35  5.8e-16  bluestein
      2006  2.17.59          prime    raced          13355      31636    2.33  5.1e-16  bluestein
      2007  3^2.223          prime    raced          13111      23488    1.75  6.3e-16  bluestein
      2008  2^3.251          flat     raced          12688      16353    1.03  6.9e-16  
      2009  7^2.41           chain3   raced           6556      17747    1.93  4.1e-16  flips differ 1.40x
      2010  2.3.5.67         flat     raced           8982      24473    2.67  5.9e-16  
      2011  2011             prime    raced          13802      26812    1.91  5.8e-16  bluestein
      2012  2^2.503          prime    raced          13550      17983    1.28  6.2e-16  bluestein
      2013  3.11.61          flat     raced           9247      24993    2.57  6.6e-16  
      2014  2.19.53          flat     raced           9060      31176    3.15  7.7e-16  
      2015  5.13.31          chain3   raced           5286      17521    2.66  6.5e-16  flips differ 1.25x
      2016  2^5.3^2.7        chain3   raced           2650       6101    2.29  4.7e-16  
      2017  2017             prime    raced          14700      29410    1.61  6.0e-16  flips differ 1.25x bluestein
      2018  2.1009           prime    raced          13328      32117    2.38  6.4e-16  bluestein
      2019  3.673            prime    raced          14502      25441    0.51  7.0e-16  flips differ 3.41x bluestein
      2020  2^2.5.101        flat     raced          11424      20651    1.58  5.6e-16  
      2021  43.47            flat     raced           9536      34489    3.51  6.1e-16  
      2022  2.3.337          prime    raced          13232      28685    2.14  5.0e-16  bluestein
      2023  7.17^2           chain3   raced           4532      21238    4.61  7.4e-16  
      2024  2^3.11.23        chain3   raced           4915      15436    2.85  4.5e-16  
      2025  3^4.5^2          chain3   raced           3384       5907    1.71  4.6e-16  
      2026  2.1013           prime    raced          13337      32524    2.31  9.2e-16  bluestein
      2027  2027             prime    raced          13170      19462    1.46  7.3e-16  bluestein
      2028  2^2.3.13^2       chain3   raced           3616       7621    2.01  5.8e-16  
      2029  2029             prime    raced          13061      20145    1.49  5.9e-16  bluestein
      2030  2.5.7.29         chain3   raced           4871      14632    2.94  4.7e-16  
      2031  3.677            prime    raced          13059      32350    2.43  5.8e-16  bluestein
      2032  2^4.127          flat     raced          11537      17992    1.55  8.8e-16  
      2033  19.107           prime    raced          13810      47969    3.09  6.1e-16  bluestein
      2034  2.3^2.113        flat     raced           9144      27499    2.95  6.8e-16  
      2035  5.11.37          chain3   raced           7437      18588    2.45  5.0e-16  
      2036  2^2.509          prime    raced          13283      19232    1.43  6.2e-16  bluestein
      2037  3.7.97           flat     raced          10861      20425    1.86  6.9e-16  
      2038  2.1019           prime    raced          12925      19026    1.44  6.6e-16  bluestein
      2039  2039             prime    raced          13702      20359    1.47  6.3e-16  bluestein
      2040  2^3.3.5.17       chain3   raced           3762      11851    3.11  5.2e-16  
      2041  13.157           flat     raced          12881      18711    1.44  8.4e-16  
      2042  2.1021           prime    raced          13399      18771    1.37  6.1e-16  bluestein
      2043  3^2.227          prime    raced          13036      20717    1.50  6.6e-16  bluestein
      2044  2^2.7.73         flat     raced           8737      23279    2.44  6.1e-16  
      2045  5.409            prime    raced          13045      32667    2.44  5.9e-16  bluestein
      2046  2.3.11.31        chain3   raced           5341      17367    3.15  5.6e-16  
      2047  23.89            flat     raced           9782      42232    3.49  5.7e-16  
      2048  2^11             ztt      raced           2102       2627    1.23  3.1e-16  
      2049  3.683            prime    raced          28269      32145    1.09  5.8e-16  bluestein
      2050  2.5^2.41         chain3   raced           6521      18499    2.65  4.8e-16  
      2051  7.293            prime    raced          27343      19974    0.72  5.7e-16  bluestein
      2052  2^2.3^3.19       chain3   raced           4072      13918    2.90  5.3e-16  
      2053  2053             prime    raced          23134      27431    0.95  7.5e-16  rader
      2054  2.13.79          flat     raced           9430      26108    2.72  8.6e-16  
      2055  3.5.137          flat     raced          11009      23017    2.01  6.2e-16  
      2056  2^3.257          flat     raced          10515      19120    1.76  5.0e-16  
      2057  11^2.17          chain3   raced           4007      16138    3.83  6.3e-16  
      2058  2.3.7^3          flat     raced           5613       5105    0.88  4.2e-16  
      2059  29.71            flat     raced          11100      34493    2.97  5.9e-16  
      2060  2^2.5.103        flat     raced          11468      35330    3.07  8.4e-16  
      2061  3^2.229          flat     raced          17187      20112    1.07  7.0e-16  
      2062  2.1031           prime    raced          26750      26948    0.80  5.4e-16  bluestein
      2063  2063             prime    raced          27339      32719    1.19  6.9e-16  bluestein
      2064  2^4.3.43         chain3   raced           5056      19577    3.69  5.2e-16  
      2065  5.7.59           flat     raced          15035      23077    1.50  4.5e-16  
      2066  2.1033           prime    raced          27222      26816    0.79  5.0e-16  bluestein
      2067  3.13.53          flat     raced           9498      23398    2.40  7.2e-16  
      2068  2^2.11.47        flat     raced           6821      21711    3.10  5.8e-16  
      2069  2069             prime    raced          27835      28021    0.82  1.1e-15  flips differ 1.26x rader
      2070  2.3^2.5.23       chain3   raced           4404      13722    2.89  6.0e-16  
      2071  19.109           flat     raced          12909      34395    2.47  7.2e-16  
      2072  2^3.7.37         chain3   raced           5435      18920    3.41  4.7e-16  
      2073  3.691            prime    raced          26304      26914    0.81  6.2e-16  bluestein
      2074  2.17.61          flat     raced           9519      32917    3.31  5.6e-16  
      2075  5^2.83           flat     raced          16087      30545    1.77  5.2e-16  
      2076  2^2.3.173        flat     raced          14301      30774    2.11  7.6e-16  
      2077  31.67            flat     raced          10852      37686    3.33  7.5e-16  
      2078  2.1039           prime    raced          29076      21772    0.72  5.4e-16  bluestein
      2079  3^3.7.11         chain3   raced           3458       6735    1.94  4.9e-16  
      2080  2^5.5.13         chain3   raced           2975       4400    1.41  5.6e-16  
      2081  2081             prime    raced          12581      21994    1.73  8.2e-16  rader
      2082  2.3.347          prime    raced          28824      26628    0.76  6.3e-16  bluestein
      2083  2083             prime    raced          29187      27486    0.74  5.2e-16  bluestein
      2084  2^2.521          flat     raced          15744      29044    1.51  7.1e-16  
      2085  3.5.139          flat     raced          14809      23132    1.48  6.5e-16  
      2086  2.7.149          flat     raced          14779      23657    1.57  6.1e-16  
      2087  2087             prime    raced          29597      27736    0.74  5.8e-16  bluestein
      2088  2^3.3^2.29       chain3   raced           4744      16673    2.70  5.5e-16  
      2089  2089             prime    raced          17226      22204    1.26  8.0e-16  rader
      2090  2.5.11.19        chain3   raced           4640      14903    3.17  5.7e-16  
      2091  3.17.41          chain3   raced           7142      28022    3.87  6.2e-16  
      2092  2^2.523          flat     raced          23261      34087    1.46  1.4e-15  
      2093  7.13.23          chain3   raced           4803      16130    2.90  8.2e-16  
      2094  2.3.349          flat     raced          22679      26584    0.97  8.1e-16  
      2095  5.419            flat     raced          22762      33216    1.43  7.8e-16  
      2096  2^4.131          flat     raced           8760      26277    2.93  8.1e-16  
      2097  3^2.233          flat     raced          18425      20543    1.10  7.5e-16  
      2098  2.1049           prime    raced          26726      21477    0.78  6.7e-16  bluestein
      2099  2099             prime    raced          29292      21787    0.73  6.1e-16  bluestein
      2100  2^2.3.5^2.7      chain3   raced           3359       6728    1.89  4.5e-16  
      2101  11.191           flat     raced          15397      18466    1.12  6.3e-16  
      2102  2.1051           prime    raced          27293      34082    1.22  5.5e-16  bluestein
      2103  3.701            prime    raced          29289      32941    1.06  6.4e-16  bluestein
      2104  2^3.263          flat     raced          22934      24479    1.06  1.0e-15  
      2105  5.421            flat     raced          25464      27561    0.79  6.3e-16  
      2106  2.3^4.13         chain3   raced           3600       5881    1.61  6.1e-16  
      2107  7^2.43           flat     raced           7757      19362    2.46  4.1e-16  
      2108  2^2.17.31        chain3   raced           5955      25676    4.26  6.4e-16  
      2109  3.19.37          chain3   raced           6888      27354    3.39  5.1e-16  
      2110  2.5.211          flat     raced          15166      27573    1.42  6.2e-16  
      2111  2111             prime    raced          28814      27747    0.78  6.7e-16  bluestein
      2112  2^6.3.11         chain3   raced           2959       4854    1.63  3.5e-16  
      2113  2113             prime    raced          11583      27429    1.91  7.2e-16  rader
      2114  2.7.151          flat     raced          13808      19705    1.34  7.5e-16  
      2115  3^2.5.47         flat     raced           9559      20231    2.01  5.3e-16  
      2116  2^2.23^2         chain3   raced           6979      26369    3.63  4.9e-16  
      2117  29.73            flat     raced          10987      38047    3.29  7.2e-16  
      2118  2.3.353          flat     raced          15828      25180    1.29  6.4e-16  
      2119  13.163           flat     raced          15086      35627    1.72  1.1e-15  flips differ 1.57x
      2120  2^3.5.53         flat     raced           7940      22159    1.05  6.8e-16  flips differ 2.66x
      2121  3.7.101          flat     raced          13889      22319    1.59  6.9e-16  
      2122  2.1061           prime    raced          33299      26981    0.53  5.9e-16  bluestein
      2123  11.193           flat     raced          13997      23089    1.54  6.6e-16  
      2124  2^2.3^2.59       flat     raced          15833      24519    1.48  6.9e-16  
      2125  5^3.17           chain3   raced           4375      13962    3.17  6.4e-16  
      2126  2.1063           prime    raced          26397      26879    0.79  6.6e-16  bluestein
      2127  3.709            prime    raced          26600      38338    1.10  4.7e-16  bluestein
      2128  2^4.7.19         chain3   raced           3935      15420    3.83  4.0e-16  
      2129  2129             prime    raced          15051      27520    1.42  7.1e-16  rader
      2130  2.3.5.71         flat     raced          10330      23509    2.25  6.4e-16  
      2131  2131             prime    raced          24271      22560    0.89  9.2e-16  rader
      2132  2^2.13.41        flat     raced           6553      20736    2.97  5.9e-16  
      2133  3^3.79           flat     raced          10860      22988    1.94  9.1e-16  
      2134  2.11.97          flat     raced           9806      23347    2.36  3.9e-16  
      2135  5.7.61           flat     raced           9934      24508    2.46  4.6e-16  
      2136  2^3.3.89         flat     raced           8283      25580    2.99  6.1e-16  
      2137  2137             prime    raced          23141      27905    0.99  8.3e-16  rader
      2138  2.1069           prime    raced          26778      21582    0.40  7.2e-16  flips differ 2.06x bluestein
      2139  3.23.31          chain3   raced           7408      31351    3.57  6.2e-16  
      2140  2^2.5.107        flat     raced          18785      32362    1.60  9.3e-16  
      2141  2141             prime    raced          28765      22305    0.71  5.8e-16  bluestein
      2142  2.3^2.7.17       chain3   raced           4257      13196    2.95  7.0e-16  
      2143  2143             prime    raced          20382      31124    1.05  9.7e-16  rader
      2144  2^5.67           flat     raced           9311      25390    2.67  5.6e-16  
      2145  3.5.11.13        chain3   raced           3964       8800    2.19  6.0e-16  
      2146  2.29.37          flat     raced          10076      30915    2.93  5.8e-16  
      2147  19.113           flat     raced          11094      39638    3.52  6.4e-16  
      2148  2^2.3.179        flat     raced          20302      30982    1.49  7.0e-16  
      2149  7.307            flat     raced          23361      35534    1.43  8.4e-16  
      2150  2.5^2.43         chain3   raced           7043      19229    2.53  4.6e-16  
      2151  3^2.239          flat     raced          15077      20400    1.27  7.6e-16  
      2152  2^3.269          flat     raced          20712      25490    1.21  8.1e-16  
      2153  2153             prime    raced          27668      27670    0.76  5.8e-16  bluestein
      2154  2.3.359          prime    raced          26723      31171    1.16  6.4e-16  bluestein
      2155  5.431            prime    raced          29246      27962    0.75  5.2e-16  bluestein
      2156  2^2.7^2.11       flat     raced           5370       8653    1.35  4.8e-16  
      2157  3.719            prime    raced          26601      35224    1.21  4.9e-16  bluestein
      2158  2.13.83          flat     raced          16509      36082    1.88  7.9e-16  
      2159  17.127           flat     raced          15028      28692    1.86  7.5e-16  
      2160  2^4.3^3.5        chain3   raced           3048       5978    1.92  4.7e-16  
      2161  2161             prime    raced          15025      24205    1.43  6.0e-16  bluestein
      2162  2.23.47          flat     raced           9708      31949    3.24  6.1e-16  
      2163  3.7.103          flat     raced          12780      39005    3.03  8.7e-16  
      2164  2^2.541          flat     raced          19179      29431    1.52  7.0e-16  
      2165  5.433            prime    raced          15367      34921    2.25  5.8e-16  bluestein
      2166  2.3.19^2         chain3   raced           5511      23442    3.66  4.9e-16  
      2167  11.197           flat     raced          13358      21791    1.60  5.2e-16  
      2168  2^3.271          flat     raced          15542      25365    1.54  6.3e-16  
      2169  3^2.241          flat     raced          13147      20912    1.57  6.4e-16  
      2170  2.5.7.31         chain3   raced           6496      16272    2.31  5.6e-16  
      2171  13.167           prime    raced          15082      29927    1.67  7.3e-16  bluestein
      2172  2^2.3.181        flat     raced          12321      27489    2.22  5.8e-16  
      2173  41.53            flat     raced          10695      37689    3.12  6.9e-16  
      2174  2.1087           prime    raced          14912      38500    2.56  8.0e-16  bluestein
      2175  3.5^2.29         chain3   raced           5407      16915    2.57  4.6e-16  
      2176  2^7.17           chain3   raced           3442      12057    3.48  6.0e-16  
      2177  7.311            prime    raced          15542      35554    2.21  5.6e-16  bluestein
      2178  2.3^2.11^2       flat     raced           6309       8544    1.34  5.2e-16  
      2179  2179             prime    raced          14886      31090    2.04  8.8e-16  bluestein
      2180  2^2.5.109        flat     raced          13428      31252    2.12  7.0e-16  
      2181  3.727            prime    raced          14953      31629    2.09  8.9e-16  bluestein
      2182  2.1091           prime    raced          14917      39549    2.58  6.8e-16  bluestein
      2183  37.59            prime    raced          15135      38724    2.50  4.7e-16  bluestein
      2184  2^3.3.7.13       chain3   raced           3704       7160    1.48  6.0e-16  
      2185  5.19.23          chain3   raced           7139      25100    3.48  4.9e-16  
      2186  2.1093           prime    raced          14868      38931    2.53  6.8e-16  bluestein
      2187  3^7              chain3   raced           3624       7561    2.05  5.7e-16  
      2188  2^2.547          prime    raced          14934      35689    2.36  7.6e-16  bluestein
      2189  11.199           flat     raced          16073      23082    1.35  7.6e-16  
      2190  2.3.5.73         flat     raced           8872      30039    3.10  4.9e-16  
      2191  7.313            prime    raced          14981      35169    2.29  7.5e-16  bluestein
      2192  2^4.137          flat     raced           8900      24656    2.20  6.0e-16  
      2193  3.17.43          flat     raced           9690      29888    3.05  6.0e-16  
      2194  2.1097           prime    raced          15165      27159    1.74  5.5e-16  bluestein
      2195  5.439            prime    raced          14961      33288    2.03  6.7e-16  bluestein
      2196  2^2.3^2.61       flat     raced           9343      25865    2.64  5.3e-16  
      2197  13^3             chain3   raced           4646      10126    2.16  8.9e-16  
      2198  2.7.157          flat     raced          12430      19111    1.52  7.4e-16  
      2199  3.733            prime    raced          14880      34840    2.33  7.6e-16  bluestein
      2200  2^3.5^2.11       chain3   raced           3346       4931    1.40  4.6e-16  
      2201  31.71            flat     raced          10502      41772    3.80  6.8e-16  
      2202  2.3.367          prime    raced          15047      26966    1.73  5.2e-16  bluestein
      2203  2203             prime    raced          15038      34807    2.31  6.4e-16  bluestein
      2204  2^2.19.29        chain3   raced           6308      26855    3.56  4.2e-16  
      2205  3^2.5.7^2        chain3   raced           3797       6924    1.81  4.5e-16  
      2206  2.1103           prime    raced          14986      27186    1.80  6.0e-16  bluestein
      2207  2207             prime    raced          14924      30743    1.76  6.6e-16  bluestein
      2208  2^5.3.23         chain3   raced           4145      15196    3.63  5.0e-16  
      2209  47^2             flat     raced          10665      38338    3.42  5.0e-16  
      2210  2.5.13.17        chain3   raced           4469      15344    3.43  7.0e-16  
      2211  3.11.67          flat     raced          10657      29096    2.70  5.6e-16  
      2212  2^2.7.79         flat     raced          10569      23169    2.14  8.5e-16  
      2213  2213             prime    raced          15011      30134    1.69  6.4e-16  bluestein
      2214  2.3^3.41         chain3   raced           7407      19357    2.54  4.8e-16  
      2215  5.443            prime    raced          14900      33834    2.25  7.8e-16  bluestein
      2216  2^3.277          flat     raced          15793      29535    1.81  6.5e-16  
      2217  3.739            prime    raced          15031      25643    1.64  6.3e-16  bluestein
      2218  2.1109           prime    raced          15029      26760    1.76  7.2e-16  bluestein
      2219  7.317            prime    raced          15007      18805    1.25  6.2e-16  bluestein
      2220  2^2.3.5.37       chain3   raced           5308      17580    3.21  4.6e-16  
      2221  2221             prime    raced          14986      34954    2.20  6.0e-16  bluestein
      2222  2.11.101         flat     raced          12267      25929    1.88  7.0e-16  
      2223  3^2.13.19        chain3   raced           4894      16806    3.06  6.4e-16  
      2224  2^4.139          flat     raced          12873      23897    1.82  6.0e-16  
      2225  5^2.89           flat     raced          10282      42293    3.98  6.4e-16  
      2226  2.3.7.53         flat     raced           9705      23060    2.23  8.1e-16  
      2227  17.131           flat     raced          12021      37241    3.01  1.1e-15  
      2228  2^2.557          prime    raced          14960      36414    2.35  6.0e-16  bluestein
      2229  3.743            prime    raced          15040      35001    1.58  6.5e-16  bluestein
      2230  2.5.223          prime    raced          14956      26025    1.73  5.8e-16  bluestein
      2231  23.97            flat     raced          11561      33110    2.84  5.9e-16  
      2232  2^3.3^2.31       chain3   raced           5251      16908    3.14  6.1e-16  
      2233  7.11.29          chain3   raced           5474      18444    3.28  5.1e-16  
      2234  2.1117           prime    raced          15154      27629    1.79  6.4e-16  bluestein
      2235  3.5.149          prime    raced          15007      25984    1.66  5.1e-16  bluestein
      2236  2^2.13.43        chain3   raced           7617      23322    2.63  7.0e-16  
      2237  2237             prime    raced          15155      30428    1.68  5.9e-16  bluestein
      2238  2.3.373          prime    raced          15035      26158    1.72  8.1e-16  bluestein
      2239  2239             prime    raced          14896      35012    2.14  7.6e-16  bluestein
      2240  2^6.5.7          ztt      raced           2556       3535    1.36  2.7e-16  
      2241  3^3.83           prime    raced          15494      32803    2.04  4.2e-16  bluestein
      2242  2.19.59          prime    raced          15636      35384    2.24  4.9e-16  bluestein
      2243  2243             prime    raced          15520      35320    1.97  6.9e-16  bluestein
      2244  2^2.3.11.17      chain3   raced           4193      16027    3.78  4.9e-16  
      2245  5.449            prime    raced          15497      33857    2.12  6.2e-16  bluestein
      2246  2.1123           prime    raced          15502      27835    1.79  4.8e-16  bluestein
      2247  3.7.107          prime    raced          15483      36898    2.36  6.0e-16  bluestein
      2248  2^3.281          flat     raced          10798      29710    2.73  4.8e-16  
      2249  13.173           prime    raced          15517      25545    1.61  6.3e-16  bluestein
      2250  2.3^2.5^3        chain3   raced           3632       4511    1.18  6.6e-16  
      2251  2251             prime    raced          15553      34724    2.08  7.4e-16  bluestein
      2252  2^2.563          prime    raced          15486      35415    2.01  5.6e-16  bluestein
      2253  3.751            prime    raced          17313      25114    0.79  5.7e-16  flips differ 1.84x bluestein
      2254  2.7^2.23         flat     raced           7213      15813    2.13  4.6e-16  
      2255  5.11.41          flat     raced           9914      22429    2.01  6.7e-16  
      2256  2^4.3.47         chain3   raced           5972      21008    3.45  4.9e-16  
      2257  37.61            flat     raced          11482      40603    3.49  5.1e-16  
      2258  2.1129           prime    raced          15531      27894    1.42  5.1e-16  bluestein
      2259  3^2.251          flat     raced          18707      18665    0.95  6.7e-16  
      2260  2^2.5.113        flat     raced          10385      29959    2.80  7.2e-16  
      2261  7.17.19          chain3   raced           5827      24012    4.06  6.8e-16  
      2262  2.3.13.29        chain3   raced           6442      19104    2.56  6.9e-16  
      2263  31.73            flat     raced          10513      41814    3.74  7.2e-16  
      2264  2^3.283          prime    raced          15392      29660    1.85  7.1e-16  bluestein
      2265  3.5.151          flat     raced          16079      20373    1.15  7.3e-16  
      2266  2.11.103         flat     raced          11536      42959    3.69  8.8e-16  
      2267  2267             prime    raced          15824      34852    1.49  5.8e-16  bluestein
      2268  2^2.3^4.7        chain3   raced           3593      10657    2.93  5.0e-16  
      2269  2269             prime    raced          15498      34426    2.13  8.6e-16  bluestein
      2270  2.5.227          prime    raced          15779      22313    1.40  6.5e-16  bluestein
      2271  3.757            prime    raced          15586      22264    1.38  5.0e-16  bluestein
      2272  2^5.71           flat     raced           9176      28382    2.41  5.2e-16  
      2273  2273             prime    raced          15600      33853    2.15  6.7e-16  bluestein
      2274  2.3.379          prime    raced          15433      18981    1.20  6.1e-16  bluestein
      2275  5^2.7.13         chain3   raced           4011       7670    1.83  5.3e-16  
      2276  2^2.569          prime    raced          15460      46016    2.87  7.9e-16  bluestein
      2277  3^2.11.23        chain3   raced           5089      17897    3.03  4.9e-16  
      2278  2.17.67          flat     raced          10790      38424    3.45  7.2e-16  
      2279  43.53            flat     raced          11646      41431    3.26  7.8e-16  
      2280  2^3.3.5.19       chain3   raced           4434      15877    3.32  3.8e-16  
      2281  2281             prime    raced          15416      27955    1.40  5.9e-16  bluestein
      2282  2.7.163          flat     raced          14846      27003    1.78  6.9e-16  
      2283  3.761            prime    raced          16186      22058    1.29  6.3e-16  bluestein
      2284  2^2.571          prime    raced          15402      28324    1.51  6.3e-16  bluestein
      2285  5.457            prime    raced          15463      22814    1.45  6.8e-16  bluestein
      2286  2.3^2.127        flat     raced          15105      20895    1.37  6.6e-16  
      2287  2287             prime    raced          16272      28597    1.41  7.1e-16  bluestein
      2288  2^4.11.13        chain3   raced           3798       8219    2.14  5.2e-16  
      2289  3.7.109          flat     raced          14228      27577    1.63  8.1e-16  
      2290  2.5.229          flat     raced          16725      23059    1.32  5.8e-16  
      2291  29.79            flat     raced          13109      37281    2.83  8.8e-16  
      2292  2^2.3.191        flat     raced          15269      18428    1.20  5.9e-16  
      2293  2293             prime    raced          15901      28257    1.43  5.9e-16  bluestein
      2294  2.31.37          flat     raced          10431      32927    3.14  5.7e-16  
      2295  3^3.5.17         chain3   raced           4555      15258    3.21  6.2e-16  
      2296  2^3.7.41         chain3   raced           6353      19570    3.05  4.1e-16  
      2297  2297             prime    raced          15465      33870    2.14  5.5e-16  bluestein
      2298  2.3.383          prime    raced          15529      20916    1.34  5.5e-16  bluestein
      2299  11^2.19          chain3   raced           4919      18942    3.51  4.1e-16  
      2300  2^2.5^2.23       chain3   raced           4892      14904    2.57  6.0e-16  
      2301  3.13.59          prime    raced          15463      27964    1.80  5.4e-16  bluestein
      2302  2.1151           prime    raced          15478      28449    1.42  6.5e-16  bluestein
      2303  7^2.47           chain3   raced           9463      23398    2.35  5.3e-16  
      2304  2^8.3^2          chain3   raced           3177       6281    0.34  6.7e-16  flips differ 5.87x
      2305  5.461            prime    raced          17360      23161    1.29  7.9e-16  bluestein
      2306  2.1153           prime    raced          17156      23203    1.27  6.4e-16  bluestein
      2307  3.769            prime    raced          16693      21843    1.26  6.4e-16  bluestein
      2308  2^2.577          flat     raced          15119      23389    1.47  5.8e-16  
      2309  2309             prime    raced          16632      22425    1.32  6.1e-16  bluestein
      2310  2.3.5.7.11       chain3   raced           3774       6369    1.60  5.9e-16  
      2311  2311             prime    raced          16616      22103    1.26  8.0e-16  bluestein
      2312  2^3.17^2         chain3   raced           4951      24041    4.79  7.6e-16  
      2313  3^2.257          flat     raced          16094      22025    1.30  5.5e-16  
      2314  2.13.89          flat     raced           9998      40052    3.43  6.8e-16  
      2315  5.463            prime    raced          16614      21905    1.29  6.9e-16  bluestein
      2316  2^2.3.193        flat     raced          11930      21855    1.75  6.4e-16  
      2317  7.331            prime    raced          16598      33567    2.02  7.2e-16  bluestein
      2318  2.19.61          flat     raced          10181      37285    3.50  5.8e-16  
      2319  3.773            prime    raced          16607      22686    1.36  5.9e-16  bluestein
      2320  2^4.5.29         chain3   raced           4965      16121    2.98  4.6e-16  
      2321  11.211           flat     raced          17410      33600    1.50  5.9e-16  
      2322  2.3^3.43         chain3   raced           8742      20969    2.38  4.4e-16  
      2323  23.101           flat     raced          14117      35679    2.49  6.6e-16  
      2324  2^2.7.83         flat     raced          16156      33761    2.00  4.7e-16  
      2325  3.5^2.31         chain3   raced           6108      18614    2.60  6.6e-16  
      2326  2.1163           prime    raced          16758      22738    1.32  7.3e-16  bluestein
      2327  13.179           prime    raced          16860      30209    1.51  7.1e-16  bluestein
      2328  2^3.3.97         flat     raced           9704      23117    2.25  5.3e-16  
      2329  17.137           flat     raced          11772      37338    2.93  8.5e-16  
      2330  2.5.233          flat     raced          16777      22393    1.15  6.1e-16  
      2331  3^2.7.37         chain3   raced           6612      19440    2.86  5.8e-16  
      2332  2^2.11.53        flat     raced           9837      26610    2.68  6.5e-16  
      2333  2333             prime    raced          16647      22200    1.17  7.7e-16  bluestein
      2334  2.3.389          prime    raced          16678      22420    1.31  6.3e-16  bluestein
      2335  5.467            prime    raced          16694      22881    1.34  7.0e-16  bluestein
      2336  2^5.73           flat     raced           9195      26233    2.58  7.5e-16  
      2337  3.19.41          chain3   raced           8699      31473    3.18  5.0e-16  
      2338  2.7.167          prime    raced          16802      25641    1.48  7.0e-16  bluestein
      2339  2339             prime    raced          16651      23247    1.37  8.2e-16  bluestein
      2340  2^2.3^2.5.13     chain3   raced           4274       6021    1.34  7.6e-16  
      2341  2341             prime    raced          15036      22096    1.44  8.5e-16  rader
      2342  2.1171           prime    raced          16844      22475    1.33  6.6e-16  bluestein
      2343  3.11.71          flat     raced          10664      30509    2.77  6.3e-16  
      2344  2^3.293          prime    raced          16673      22542    1.33  8.0e-16  bluestein
      2345  5.7.67           flat     raced          11483      29114    2.51  5.1e-16  
      2346  2.3.17.23        chain3   raced           5993      27262    4.49  6.3e-16  
      2347  2347             prime    raced          16860      22195    1.28  6.9e-16  bluestein
      2348  2^2.587          prime    raced          16783      23376    1.39  6.7e-16  bluestein
      2349  3^4.29           chain3   raced           5832      17640    3.00  5.6e-16  
      2350  2.5^2.47         flat     raced          11005      21921    1.31  4.5e-16  flips differ 1.51x
      2351  2351             prime    raced          16690      26147    0.91  6.5e-16  flips differ 1.96x bluestein
      2352  2^4.3.7^2        chain3   raced           3517       4762    1.30  4.2e-16  
      2353  13.181           flat     raced          15170      30784    1.73  7.0e-16  
      2354  2.11.107         prime    raced          16871      45729    2.63  7.1e-16  bluestein
      2355  3.5.157          flat     raced          14662      20541    1.35  9.3e-16  
      2356  2^2.19.31        chain3   raced           7096      28739    3.37  4.6e-16  
      2357  2357             prime    raced          16626      22573    1.22  7.7e-16  bluestein
      2358  2.3^2.131        flat     raced          11633      29953    2.49  8.9e-16  
      2359  7.337            flat     raced          19811      33307    1.68  7.4e-16  
      2360  2^3.5.59         flat     raced          14814      27965    1.80  1.0e-15  
      2361  3.787            prime    raced          16778      22780    1.35  9.7e-16  bluestein
      2362  2.1181           prime    raced          16682      22730    1.34  7.2e-16  bluestein
      2363  17.139           flat     raced          15727      36150    2.21  8.7e-16  
      2364  2^2.3.197        flat     raced          12013      21747    1.75  5.5e-16  
      2365  5.11.43          flat     raced          10212      25205    2.34  5.2e-16  
      2366  2.7.13^2         flat     raced           6924       9553    1.35  7.1e-16  
      2367  3^2.263          prime    raced          16767      28025    1.65  7.5e-16  bluestein
      2368  2^6.37           2p       raced           5395      18838    3.42  5.5e-16  
      2369  23.103           flat     raced          14463      55621    3.62  7.6e-16  
      2370  2.3.5.79         flat     raced          11020      24754    2.11  8.3e-16  
      2371  2371             prime    raced          16621      22165    1.31  6.8e-16  bluestein
      2372  2^2.593          prime    raced          16901      24128    1.37  7.3e-16  bluestein
      2373  3.7.113          flat     raced          11073      34381    3.04  6.1e-16  
      2374  2.1187           prime    raced          17182      23443    1.29  6.9e-16  bluestein
      2375  5^3.19           chain3   raced           5147      15960    2.80  6.8e-16  
      2376  2^3.3^3.11       chain3   raced           3713       9177    2.46  4.6e-16  
      2377  2377             prime    raced          16896      22375    1.29  7.4e-16  bluestein
      2378  2.29.41          flat     raced          10475      34686    2.98  5.0e-16  
      2379  3.13.61          flat     raced          10942      29626    2.59  6.0e-16  
      2380  2^2.5.7.17       flat     raced           7190      17061    2.06  4.9e-16  
      2381  2381             prime    raced          16810      23210    1.37  6.7e-16  bluestein
      2382  2.3.397          prime    raced          16610      22499    1.34  7.3e-16  bluestein
      2383  2383             prime    raced          16723      22443    1.31  6.9e-16  bluestein
      2384  2^4.149          flat     raced          13130      26675    2.01  6.4e-16  
      2385  3^2.5.53         flat     raced          10546      26378    2.47  6.3e-16  
      2386  2.1193           prime    raced          16827      23123    1.32  6.9e-16  bluestein
      2387  7.11.31          chain3   raced           7181      21689    2.47  5.1e-16  
      2388  2^2.3.199        flat     raced          13838      22106    1.58  6.6e-16  
      2389  2389             prime    raced          16876      22604    1.15  6.1e-16  bluestein
      2390  2.5.239          flat     raced          15503      22787    1.41  6.6e-16  
      2391  3.797            prime    raced          16735      22905    1.34  7.3e-16  bluestein
      2392  2^3.13.23        chain3   raced           6032      18572    3.04  8.6e-16  
      2393  2393             prime    raced          16766      23319    1.27  6.3e-16  bluestein
      2394  2.3^2.7.19       chain3   raced           5167      15025    2.86  5.2e-16  
      2395  5.479            prime    raced          16737      22231    1.29  6.3e-16  bluestein
      2396  2^2.599          prime    raced          16958      23517    1.36  7.8e-16  bluestein
      2397  3.17.47          chain3   raced           9119      35092    3.72  5.5e-16  
      2398  2.11.109         flat     raced          13347      33640    2.33  6.1e-16  
      2399  2399             prime    raced          16620      22716    1.29  7.6e-16  bluestein
      2400  2^5.3.5^2        chain3   raced           3297       3483    0.99  4.4e-16  
      2401  7^4              flat     raced           6191       8561    1.32  4.1e-16  
      2402  2.1201           prime    raced          30474      25644    0.46  6.4e-16  flips differ 1.92x bluestein
      2403  3^3.89           flat     raced          11306      41492    3.05  6.4e-16  
      2404  2^2.601          flat     raced          25325      26060    1.00  7.9e-16  
      2405  5.13.37          chain3   raced           7284      23079    2.69  5.7e-16  
      2406  2.3.401          flat     raced          20820      22987    1.10  5.4e-16  
      2407  29.83            flat     raced          18951      49207    2.50  6.6e-16  
      2408  2^3.7.43         chain3   raced           6917      21367    2.89  5.4e-16  
      2409  3.11.73          flat     raced          10509      30424    2.89  6.9e-16  
      2410  2.5.241          flat     raced          13988      22820    1.61  7.4e-16  
      2411  2411             prime    raced          27021      40509    1.06  7.2e-16  bluestein
      2412  2^2.3^2.67       flat     raced          10848      30465    2.62  7.9e-16  
      2413  19.127           flat     raced          15457      32705    2.10  6.1e-16  
      2414  2.17.71          flat     raced          10404      39705    3.74  5.7e-16  
      2415  3.5.7.23         chain3   raced           6079      17194    2.68  4.4e-16  
      2416  2^4.151          flat     raced          13163      21070    1.41  6.6e-16  
      2417  2417             prime    raced          29337      29384    0.95  7.1e-16  bluestein
      2418  2.3.13.31        chain3   raced           7348      20586    2.78  5.5e-16  
      2419  41.59            flat     raced          18449      43805    2.08  5.6e-16  
      2420  2^2.5.11^2       flat     raced           6442       9266    1.32  4.4e-16  
      2421  3^2.269          prime    raced          27838      28842    0.99  5.8e-16  bluestein
      2422  2.7.173          flat     raced          18751      30618    1.35  6.1e-16  
      2423  2423             prime    raced          29376      40714    0.96  6.3e-16  bluestein
      2424  2^3.3.101        flat     raced          13473      28238    2.07  7.4e-16  
      2425  5^2.97           flat     raced          12305      24475    1.87  7.0e-16  
      2426  2.1213           prime    raced          27527      33227    1.16  5.2e-16  bluestein
      2427  3.809            prime    raced          29004      41122    1.27  6.3e-16  bluestein
      2428  2^2.607          prime    raced          28838      35689    1.21  4.8e-16  bluestein
      2429  7.347            prime    raced          28916      24508    0.79  6.5e-16  bluestein
      2430  2.3^5.5          chain3   raced           3964       5828    1.46  5.1e-16  
      2431  11.13.17         chain3   raced           5034      19309    3.75  5.7e-16  
      2432  2^7.19           chain3   raced           4287      14864    3.23  4.6e-16  
      2433  3.811            prime    raced          28465      28407    0.94  6.7e-16  bluestein
      2434  2.1217           prime    raced          26819      28747    0.97  6.4e-16  bluestein
      2435  5.487            flat     raced          37322      24278    0.63  8.4e-16  
      2436  2^2.3.7.29       chain3   raced           6233      21001    3.31  5.2e-16  
      2437  2437             prime    raced          26153      26259    0.98  5.7e-16  bluestein
      2438  2.23.53          flat     raced          11252      37883    3.08  6.8e-16  
      2439  3^2.271          flat     raced          20019      32019    1.29  6.5e-16  
      2440  2^3.5.61         flat     raced           9431      29135    3.08  4.1e-16  
      2441  2441             prime    raced          27321      27674    0.96  8.8e-16  rader
      2442  2.3.11.37        chain3   raced           7455      22405    2.93  3.8e-16  
      2443  7.349            flat     raced          29234      35357    1.11  7.4e-16  
      2444  2^2.13.47        flat     raced           8835      25535    2.82  8.8e-16  
      2445  3.5.163          flat     raced          19265      29616    1.53  8.0e-16  
      2446  2.1223           prime    raced          26360      29286    1.08  6.3e-16  bluestein
      2447  2447             prime    raced          27255      25762    0.92  5.6e-16  bluestein
      2448  2^4.3^2.17       chain3   raced           4396      14615    3.19  6.5e-16  
      2449  31.79            flat     raced          12616      40545    2.99  7.1e-16  
      2450  2.5^2.7^2        flat     raced           6648       5670    0.84  3.9e-16  
      2451  3.19.43          flat     raced          11949      34042    2.76  6.2e-16  
      2452  2^2.613          prime    raced          17447      33077    1.85  5.9e-16  bluestein
      2453  11.223           prime    raced          17497      31658    1.76  6.3e-16  bluestein
      2454  2.3.409          prime    raced          16637      39822    2.34  7.4e-16  bluestein
      2455  5.491            prime    raced          17067      24923    1.24  6.3e-16  bluestein
      2456  2^3.307          flat     raced          15724      40275    2.42  9.9e-16  
      2457  3^3.7.13         chain3   raced           4420       7851    1.76  6.2e-16  
      2458  2.1229           prime    raced          16888      29518    1.73  6.1e-16  bluestein
      2459  2459             prime    raced          16836      26070    1.43  6.1e-16  bluestein
      2460  2^2.3.5.41       chain3   raced           8321      23755    2.83  4.5e-16  
      2461  23.107           prime    raced          18244      46130    1.98  6.7e-16  bluestein
      2462  2.1231           prime    raced          17408      30237    1.71  6.2e-16  bluestein
      2463  3.821            prime    raced          17533      27467    1.51  6.6e-16  bluestein
      2464  2^5.7.11         chain3   raced           3411       7515    1.74  4.1e-16  
      2465  5.17.29          chain3   raced           6769      29203    3.95  6.5e-16  
      2466  2.3^2.137        flat     raced          11987      27238    2.22  7.6e-16  
      2467  2467             prime    raced          17208      28149    1.50  6.9e-16  bluestein
      2468  2^2.617          flat     raced          18039      33879    1.78  5.9e-16  
      2469  3.823            prime    raced          16767      26652    1.58  8.2e-16  bluestein
      2470  2.5.13.19        chain3   raced           5346      17534    2.89  5.4e-16  
      2471  7.353            prime    raced          16886      29752    1.46  5.8e-16  bluestein
      2472  2^3.3.103        flat     raced          11374      45596    3.93  8.4e-16  
      2473  2473             prime    raced          17480      25629    1.46  5.6e-16  bluestein
      2474  2.1237           prime    raced          16821      28959    1.71  7.3e-16  bluestein
      2475  3^2.5^2.11       chain3   raced           4632       8104    1.63  5.8e-16  
      2476  2^2.619          prime    raced          17026      32645    1.88  6.2e-16  bluestein
      2477  2477             prime    raced          16778      25401    1.48  6.5e-16  bluestein
      2478  2.3.7.59         prime    raced          17597      27429    1.55  5.7e-16  bluestein
      2479  37.67            flat     raced          12969      46534    3.49  6.7e-16  
      2480  2^4.5.31         chain3   raced           5558      19170    2.94  4.3e-16  
      2481  3.827            prime    raced          18261      27115    1.48  7.2e-16  bluestein
      2482  2.17.73          flat     raced          10598      40100    3.78  6.0e-16  
      2483  13.191           prime    raced          16851      22068    1.26  6.8e-16  bluestein
      2484  2^2.3^3.23       chain3   raced           5112      16877    3.29  6.8e-16  
      2485  5.7.71           flat     raced          11346      28087    2.40  6.3e-16  
      2486  2.11.113         flat     raced          10946      36574    3.32  5.6e-16  
      2487  3.829            prime    raced          16873      27107    1.59  6.7e-16  bluestein
      2488  2^3.311          prime    raced          17476      40304    2.24  7.3e-16  bluestein
      2489  19.131           flat     raced          14048      42100    2.94  6.8e-16  
      2490  2.3.5.83         prime    raced          16813      35904    2.10  5.6e-16  bluestein
      2491  47.53            flat     raced          12751      45348    3.21  8.3e-16  
      2492  2^2.7.89         flat     raced          11009      30931    2.71  4.2e-16  
      2493  3^2.277          prime    raced          17328      38855    2.16  6.4e-16  bluestein
      2494  2.29.43          flat     raced          11095      37350    2.92  6.5e-16  
      2495  5.499            prime    raced          16721      25006    1.46  7.4e-16  bluestein
      2496  2^6.3.13         chain3   raced           3497       5971    1.68  6.6e-16  
      2497  11.227           prime    raced          17538      26898    1.43  5.7e-16  bluestein
      2498  2.1249           prime    raced          16687      30250    1.70  6.2e-16  bluestein
      2499  3.7^2.17         chain3   raced           4995      16500    3.09  5.1e-16  
      2500  2^2.5^4          chain3   raced           4072       5224    1.22  4.7e-16  
      2501  41.61            flat     raced          12353      46395    3.68  4.9e-16  
      2502  2.3^2.139        flat     raced          14888      28725    1.56  8.2e-16  flips differ 1.28x
      2503  2503             prime    raced          16673      24262    1.40  6.4e-16  bluestein
      2504  2^3.313          flat     raced          13604      41964    3.02  8.1e-16  
      2505  3.5.167          prime    raced          17325      39608    2.21  5.8e-16  bluestein
      2506  2.7.179          prime    raced          17428      31146    1.43  6.5e-16  bluestein
      2507  23.109           flat     raced          16295      42208    2.41  6.7e-16  
      2508  2^2.3.11.19      chain3   raced           5012      18218    3.62  4.7e-16  
      2509  13.193           flat     raced          15336      25710    1.64  6.6e-16  
      2510  2.5.251          flat     raced          18513      20324    0.99  6.6e-16  
      2511  3^4.31           chain3   raced           6404      20539    2.49  6.3e-16  
      2512  2^4.157          flat     raced          11197      21577    1.89  5.8e-16  
      2513  7.359            prime    raced          16732      30205    1.52  5.4e-16  bluestein
      2514  2.3.419          prime    raced          16696      34656    1.70  7.5e-16  bluestein
      2515  5.503            prime    raced          16783      22370    1.33  5.6e-16  bluestein
      2516  2^2.17.37        chain3   raced           7794      32323    4.11  6.3e-16  
      2517  3.839            prime    raced          17606      26858    1.50  6.6e-16  bluestein
      2518  2.1259           prime    raced          16813      24663    1.45  5.9e-16  bluestein
      2519  11.229           prime    raced          17536      26937    1.51  6.3e-16  bluestein
      2520  2^3.3^2.5.7      chain3   raced           4234       8095    1.86  5.8e-16  
      2521  2521             prime    raced          16845      26258    1.53  5.6e-16  bluestein
      2522  2.13.97          flat     raced          11644      28124    2.12  5.7e-16  
      2523  3.29^2           chain3   raced           9094      34603    3.62  5.9e-16  
      2524  2^2.631          flat     raced          26657      24396    0.85  7.5e-16  
      2525  5^2.101          flat     raced          15522      27091    1.53  7.5e-16  
      2526  2.3.421          prime    raced          17221      40477    2.24  7.0e-16  bluestein
      2527  7.19^2           chain3   raced           7430      28596    3.63  4.7e-16  
      2528  2^5.79           flat     raced          11167      26108    2.23  7.5e-16  
      2529  3^2.281          flat     raced          16336      34184    1.96  6.0e-16  
      2530  2.5.11.23        chain3   raced           5568      19490    3.00  5.3e-16  
      2531  2531             prime    raced          16740      24428    1.42  6.1e-16  bluestein
      2532  2^2.3.211        flat     raced          15364      42642    2.46  8.4e-16  
      2533  17.149           prime    raced          16916      41142    2.37  6.8e-16  bluestein
      2534  2.7.181          flat     raced          14571      26488    1.65  5.8e-16  
      2535  3.5.13^2         chain3   raced           4933      10786    1.96  6.4e-16  
      2536  2^3.317          prime    raced          16802      20978    1.21  5.4e-16  bluestein
      2537  43.59            prime    raced          16727      46849    2.72  4.8e-16  bluestein
      2538  2.3^3.47         chain3   raced          10530      24109    2.26  5.4e-16  
      2539  2539             prime    raced          16656      24298    1.41  6.2e-16  bluestein
      2540  2^2.5.127        flat     raced          16026      22307    1.35  6.8e-16  
      2541  3.7.11^2         chain3   raced           4373      10907    2.43  4.1e-16  
      2542  2.31.41          flat     raced          11356      39054    3.37  7.1e-16  
      2543  2543             prime    raced          17334      24563    1.33  4.7e-16  bluestein
      2544  2^4.3.53         flat     raced           9851      26198    2.61  6.6e-16  
      2545  5.509            prime    raced          18118      25394    1.35  6.5e-16  bluestein
      2546  2.19.67          flat     raced          11886      43663    3.14  6.2e-16  
      2547  3^2.283          prime    raced          18026      33204    1.71  7.5e-16  bluestein
      2548  2^2.7^2.13       flat     raced           6430      10295    1.58  5.6e-16  
      2549  2549             prime    raced          17259      24378    1.38  6.5e-16  bluestein
      2550  2.3.5^2.17       chain3   raced           5143      14822    2.63  6.6e-16  
      2551  2551             prime    raced          17825      24675    1.32  6.3e-16  bluestein
      2552  2^3.11.29        chain3   raced           5824      20897    3.44  3.8e-16  
      2553  3.23.37          chain3   raced           8568      34560    3.27  5.1e-16  
      2554  2.1277           prime    raced          17204      26547    1.42  6.2e-16  bluestein
      2555  5.7.73           flat     raced          11485      28906    2.50  5.8e-16  
      2556  2^2.3^2.71       flat     raced          10721      32153    2.98  5.0e-16  
      2557  2557             prime    raced          17165      24137    1.40  5.7e-16  bluestein
      2558  2.1279           prime    raced          17406      25094    1.36  5.9e-16  bluestein
      2559  3.853            prime    raced          17172      27008    1.52  5.9e-16  bluestein
      2560  2^9.5            ztt      raced           2830       3675    1.29  4.4e-16  
      2561  13.197           flat     raced          14666      26073    1.73  7.1e-16  
      2562  2.3.7.61         flat     raced          12402      29298    2.32  4.9e-16  
      2563  11.233           prime    raced          19165      27279    1.37  5.7e-16  bluestein
      2564  2^2.641          flat     raced          16458      24273    1.38  5.0e-16  
      2565  3^3.5.19         chain3   raced           5667      17889    2.96  5.0e-16  
      2566  2.1283           prime    raced          19362      36297    1.54  7.6e-16  bluestein
      2567  17.151           flat     raced          17336      34712    1.85  8.9e-16  
      2568  2^3.3.107        prime    raced          19220      35640    1.50  7.2e-16  bluestein
      2569  7.367            prime    raced          19295      30327    1.56  6.4e-16  bluestein
      2570  2.5.257          flat     raced          16140      24292    1.42  4.8e-16  
      2571  3.857            prime    raced          19269      26942    1.39  7.6e-16  bluestein
      2572  2^2.643          prime    raced          19158      30741    1.58  7.9e-16  bluestein
      2573  31.83            prime    raced          19215      52660    2.71  6.8e-16  bluestein
      2574  2.3^2.11.13      flat     raced           7509       9937    1.29  5.5e-16  
      2575  5^2.103          flat     raced          15062      38860    2.16  8.2e-16  
      2576  2^4.7.23         chain3   raced           5330      20105    3.56  5.8e-16  
      2577  3.859            prime    raced          19325      26820    1.39  7.0e-16  bluestein
      2578  2.1289           prime    raced          19206      31757    1.60  7.7e-16  bluestein
      2579  2579             prime    raced          19153      54606    2.83  9.2e-16  bluestein
      2580  2^2.3.5.43       chain3   raced           6854      24333    3.18  5.0e-16  
      2581  29.89            flat     raced          12544      55908    3.57  5.8e-16  
      2582  2.1291           prime    raced          20154      48476    2.18  7.9e-16  bluestein
      2583  3^2.7.41         chain3   raced           8449      22953    2.68  6.5e-16  
      2584  2^3.17.19        chain3   raced           6087      30165    4.55  4.9e-16  
      2585  5.11.47          chain3   raced          12954      27322    2.02  5.2e-16  
      2586  2.3.431          prime    raced          19147      40022    2.07  5.8e-16  bluestein
      2587  13.199           flat     raced          17442      25928    1.38  8.5e-16  
      2588  2^2.647          prime    raced          19169      30106    1.55  7.0e-16  bluestein
      2589  3.863            prime    raced          19133      27295    1.40  8.0e-16  bluestein
      2590  2.5.7.37         chain3   raced           8369      21231    2.41  4.7e-16  
      2591  2591             prime    raced          19271      41659    1.50  6.9e-16  bluestein
      2592  2^5.3^4          chain3   raced           3532      11211    2.94  5.0e-16  
      2593  2593             prime    raced          17868      41186    2.26  7.5e-16  bluestein
      2594  2.1297           prime    raced          17886      34050    1.54  6.4e-16  bluestein
      2595  3.5.173          prime    raced          17769      33357    1.57  8.3e-16  bluestein
      2596  2^2.11.59        flat     raced          17177      31934    1.84  6.5e-16  
      2597  7^2.53           flat     raced          11126      27284    2.35  5.8e-16  
      2598  2.3.433          prime    raced          17997      31306    1.50  5.4e-16  bluestein
      2599  23.113           flat     raced          12916      49411    3.54  5.5e-16  
      2600  2^3.5^2.13       chain3   raced           4240       5945    1.33  5.5e-16  
      2601  3^2.17^2         chain3   raced           5899      27799    4.48  6.4e-16  
      2602  2.1301           prime    raced          17857      51245    2.48  7.0e-16  bluestein
      2603  19.137           flat     raced          13962      40584    2.85  6.9e-16  
      2604  2^2.3.7.31       chain3   raced           7326      23115    3.03  4.9e-16  
      2605  5.521            prime    raced          17825      36114    1.69  7.1e-16  bluestein
      2606  2.1303           prime    raced          17936      44409    2.45  7.5e-16  bluestein
      2607  3.11.79          flat     raced          12923      34234    2.57  8.5e-16  
      2608  2^4.163          flat     raced          13902      30515    1.78  7.4e-16  
      2609  2609             prime    raced          17793      42257    2.26  7.3e-16  bluestein
      2610  2.3^2.5.29       chain3   raced           6742      18777    2.37  5.3e-16  
      2611  7.373            prime    raced          18061      30839    1.70  7.4e-16  bluestein
      2612  2^2.653          prime    raced          18300      34832    1.85  6.4e-16  bluestein
      2613  3.13.67          flat     raced          12636      34915    2.75  4.8e-16  
      2614  2.1307           prime    raced          17840      44702    2.32  7.2e-16  bluestein
      2615  5.523            prime    raced          17960      37292    1.75  6.9e-16  bluestein
      2616  2^3.3.109        flat     raced          14477      34430    2.36  6.2e-16  
      2617  2617             prime    raced          17774      35730    1.66  7.1e-16  bluestein
      2618  2.7.11.17        flat     raced           7659      18729    2.30  5.8e-16  
      2619  3^3.97           flat     raced          13411      26595    1.95  6.3e-16  
      2620  2^2.5.131        flat     raced          12876      33278    2.51  7.9e-16  
      2621  2621             prime    raced          18197      42634    2.31  7.6e-16  bluestein
      2622  2.3.19.23        chain3   raced           6932      29969    3.56  3.9e-16  
      2623  43.61            flat     raced          13214      50721    3.65  7.4e-16  
      2624  2^6.41           chain3   raced           6798      21944    3.16  3.8e-16  
      2625  3.5^3.7          chain3   raced           4704       8629    1.80  4.3e-16  
      2626  2.13.101         flat     raced          14553      29899    1.87  7.5e-16  
      2627  37.71            flat     raced          12887      47069    3.59  7.1e-16  
      2628  2^2.3^2.73       flat     raced          10624      32673    3.07  5.3e-16  
      2629  11.239           prime    raced          17905      27097    1.49  6.3e-16  bluestein
      2630  2.5.263          prime    raced          17929      30558    1.69  7.0e-16  bluestein
      2631  3.877            prime    raced          18374      40821    2.16  5.2e-16  bluestein
      2632  2^3.7.47         chain3   raced           8033      24650    2.83  3.7e-16  
      2633  2633             prime    raced          17896      42346    2.35  7.9e-16  bluestein
      2634  2.3.439          prime    raced          18238      34823    1.55  8.9e-16  bluestein
      2635  5.17.31          chain3   raced           7928      32388    4.02  5.2e-16  
      2636  2^2.659          prime    raced          17773      33749    1.83  5.6e-16  bluestein
      2637  3^2.293          prime    raced          17793      26084    1.46  7.2e-16  bluestein
      2638  2.1319           prime    raced          18320      38494    1.76  6.3e-16  bluestein
      2639  7.13.29          chain3   raced           7017      22012    2.77  6.5e-16  
      2640  2^4.3.5.11       chain3   raced           3830       7948    1.64  7.1e-16  
      2641  19.139           prime    raced          17864      40947    2.27  6.4e-16  bluestein
      2642  2.1321           prime    raced          17976      32718    1.79  8.6e-16  bluestein
      2643  3.881            prime    raced          17740      41240    1.57  5.9e-16  bluestein
      2644  2^2.661          prime    raced          18118      33620    1.83  6.5e-16  bluestein
      2645  5.23^2           chain3   raced           7412      31631    4.23  5.4e-16  
      2646  2.3^3.7^2        chain3   raced           5064       6594    1.29  6.0e-16  
      2647  2647             prime    raced          18145      36301    1.57  6.6e-16  bluestein
      2648  2^3.331          flat     raced          14473      37772    2.53  6.7e-16  
      2649  3.883            prime    raced          17761      34348    1.57  6.5e-16  bluestein
      2650  2.5^2.53         flat     raced          12032      27269    2.12  6.3e-16  
      2651  11.241           flat     raced          15701      27689    1.72  5.3e-16  
      2652  2^2.3.13.17      chain3   raced           5241      20318    3.66  7.6e-16  
      2653  7.379            prime    raced          17804      22574    1.26  7.8e-16  bluestein
      2654  2.1327           prime    raced          17807      38759    1.84  7.3e-16  bluestein
      2655  3^2.5.59         prime    raced          17925      30659    1.69  6.6e-16  bluestein
      2656  2^5.83           flat     raced          18091      38515    2.10  6.0e-16  
      2657  2657             prime    raced          18147      36709    1.58  7.0e-16  bluestein
      2658  2.3.443          prime    raced          17867      34638    1.58  6.3e-16  bluestein
      2659  2659             prime    raced          18035      43704    2.33  7.5e-16  bluestein
      2660  2^2.5.7.19       flat     raced           8042      16407    1.97  5.2e-16  
      2661  3.887            prime    raced          18019      40616    2.20  6.7e-16  bluestein
      2662  2.11^3           flat     raced           8047      13371    1.64  4.2e-16  
      2663  2663             prime    raced          18176      29453    1.52  7.5e-16  bluestein
      2664  2^3.3^2.37       chain3   raced           6368      21432    3.21  5.8e-16  
      2665  5.13.41          chain3   raced           9086      25967    2.83  7.1e-16  
      2666  2.31.43          flat     raced          12461      40727    3.26  5.9e-16  
      2667  3.7.127          flat     raced          16552      24962    1.37  7.2e-16  
      2668  2^2.23.29        chain3   raced           9336      33794    3.54  5.6e-16  
      2669  17.157           flat     raced          15728      34127    2.15  6.5e-16  
      2670  2.3.5.89         flat     raced          11410      42171    3.17  6.0e-16  
      2671  2671             prime    raced          18174      36001    1.56  8.1e-16  bluestein
      2672  2^4.167          prime    raced          17948      40898    2.27  6.2e-16  bluestein
      2673  3^5.11           chain3   raced           4704       8937    1.84  8.0e-16  
      2674  2.7.191          flat     raced          17354      21634    0.94  7.1e-16  flips differ 1.32x
      2675  5^2.107          prime    raced          18122      39065    1.81  7.2e-16  bluestein
      2676  2^2.3.223        prime    raced          17907      31715    1.68  6.6e-16  bluestein
      2677  2677             prime    raced          18442      35620    1.54  7.0e-16  bluestein
      2678  2.13.103         flat     raced          13802      50795    3.50  6.0e-16  
      2679  3.19.47          chain3   raced          10428      38151    3.60  4.5e-16  
      2680  2^3.5.67         flat     raced          10816      31855    2.91  5.5e-16  
      2681  7.383            prime    raced          18157      22750    1.25  6.1e-16  bluestein
      2682  2.3^2.149        flat     raced          18535      31452    1.69  6.8e-16  
      2683  2683             prime    raced          18074      29353    1.57  8.0e-16  bluestein
      2684  2^2.11.61        flat     raced          11469      33112    2.87  4.5e-16  
      2685  3.5.179          prime    raced          18049      34896    1.54  6.0e-16  bluestein
      2686  2.17.79          flat     raced          12642      44348    3.49  7.6e-16  
      2687  2687             prime    raced          17996      29562    1.62  7.5e-16  bluestein
      2688  2^7.3.7          ztt      raced           3350       5972    1.27  3.4e-16  
      2689  2689             prime    raced          13958      36005    2.10  5.5e-16  rader
      2690  2.5.269          flat     raced          27724      31635    1.10  8.4e-16  
      2691  3^2.13.23        chain3   raced           6179      20974    2.97  5.9e-16  
      2692  2^2.673          flat     raced          22668      33993    1.46  6.7e-16  
      2693  2693             prime    raced          27267      28805    1.05  5.1e-16  bluestein
      2694  2.3.449          flat     raced          21872      39958    1.82  7.5e-16  
      2695  5.7^2.11         flat     raced           6836       9196    1.33  4.4e-16  
      2696  2^3.337          flat     raced          15977      38204    2.30  7.2e-16  
      2697  3.29.31          chain3   raced           9562      36465    3.67  6.2e-16  
      2698  2.19.71          flat     raced          12311      44409    3.57  6.1e-16  
      2699  2699             prime    raced          29048      29262    1.01  6.1e-16  bluestein
      2700  2^2.3^3.5^2      chain3   raced           4263       6758    1.20  5.0e-16  
      2701  37.73            flat     raced          12353      47280    3.79  7.8e-16  
      2702  2.7.193          flat     raced          14750      26468    1.73  5.1e-16  
      2703  3.17.53          flat     raced          12194      40108    3.18  8.6e-16  
      2704  2^4.13^2         chain3   raced           5173       9896    1.90  5.8e-16  
      2705  5.541            flat     raced          38288      36397    0.92  6.0e-16  
      2706  2.3.11.41        chain3   raced           8948      26140    2.40  5.2e-16  
      2707  2707             prime    raced          27207      49244    1.80  4.7e-16  bluestein
      2708  2^2.677          flat     raced          21686      42925    1.95  8.4e-16  
      2709  3^2.7.43         chain3   raced           8596      25029    2.88  5.4e-16  
      2710  2.5.271          flat     raced          22533      43374    1.86  7.2e-16  
      2711  2711             prime    raced          28435      49514    1.19  5.9e-16  flips differ 1.46x bluestein
      2712  2^3.3.113        flat     raced          10719      38151    3.42  4.6e-16  
      2713  2713             prime    raced          28963      55301    1.79  6.2e-16  bluestein
      2714  2.23.59          flat     raced          20085      44462    2.19  7.0e-16  
      2715  3.5.181          flat     raced          17487      35173    1.61  7.6e-16  
      2716  2^2.7.97         flat     raced          12944      27079    2.00  5.7e-16  
      2717  11.13.19         chain3   raced           6072      21945    3.28  5.6e-16  
      2718  2.3^2.151        flat     raced          16710      24325    1.30  7.7e-16  
      2719  2719             prime    raced          27619      52314    1.85  5.7e-16  bluestein
      2720  2^5.5.17         chain3   raced           4475      15894    3.43  5.8e-16  
      2721  3.907            prime    raced          28564      46919    1.63  7.3e-16  bluestein
      2722  2.1361           prime    raced          27162      49059    1.52  6.6e-16  bluestein
      2723  7.389            prime    raced          29180      26281    0.85  5.1e-16  bluestein
      2724  2^2.3.227        flat     raced          25974      27735    0.90  5.9e-16  
      2725  5^2.109          flat     raced          17890      32856    1.60  7.6e-16  
      2726  2.29.47          flat     raced          13065      41815    2.81  6.9e-16  
      2727  3^3.101          flat     raced          16217      30859    1.64  7.4e-16  
      2728  2^3.11.31        chain3   raced           6827      23647    3.32  5.1e-16  
      2729  2729             prime    raced          23656      50128    2.03  8.2e-16  rader
      2730  2.3.5.7.13       chain3   raced           4895       7367    1.49  6.2e-16  
      2731  2731             prime    raced          22323      49709    2.21  9.4e-16  rader
      2732  2^2.683          flat     raced          25960      45332    1.65  8.8e-16  
      2733  3.911            prime    raced          30136      48024    1.52  5.3e-16  bluestein
      2734  2.1367           prime    raced          31916      37234    0.89  6.6e-16  bluestein
      2735  5.547            prime    raced          29695      45072    1.48  6.5e-16  bluestein
      2736  2^4.3^2.19       chain3   raced           5261      17314    3.20  5.6e-16  
      2737  7.17.23          chain3   raced           7335      31010    3.98  6.4e-16  
      2738  2.37^2           flat     raced          11758      41556    3.49  5.4e-16  
      2739  3.11.83          flat     raced          20897      42968    1.97  4.8e-16  
      2740  2^2.5.137        flat     raced          14256      38532    2.68  7.1e-16  
      2741  2741             prime    raced          29538      49575    1.62  5.8e-16  bluestein
      2742  2.3.457          flat     raced          26939      26096    0.94  7.0e-16  
      2743  13.211           flat     raced          19060      45518    2.37  8.6e-16  
      2744  2^3.7^3          flat     raced           6314       8260    1.03  3.4e-16  
      2745  3^2.5.61         flat     raced          12384      32407    2.55  5.8e-16  
      2746  2.1373           prime    raced          21413      28697    1.33  7.1e-16  bluestein
      2747  41.67            flat     raced          14508      54040    3.64  5.8e-16  
      2748  2^2.3.229        flat     raced          17234      30607    1.72  6.1e-16  
      2749  2749             prime    raced          21108      49263    2.31  8.2e-16  bluestein
      2750  2.5^3.11         chain3   raced           4855       7678    1.43  5.5e-16  
      2751  3.7.131          flat     raced          14668      34894    2.30  6.7e-16  
      2752  2^6.43           chain3   raced           7295      24050    2.79  4.5e-16  
      2753  2753             prime    raced          21117      48904    2.25  6.9e-16  bluestein
      2754  2.3^4.17         chain3   raced           5545      17726    3.10  6.2e-16  
      2755  5.19.29          chain3   raced           8671      33390    3.78  5.6e-16  
      2756  2^2.13.53        flat     raced          11766      31258    2.62  8.2e-16  
      2757  3.919            prime    raced          21394      47629    2.17  9.6e-16  bluestein
      2758  2.7.197          flat     raced          14355      27934    1.88  6.0e-16  
      2759  31.89            flat     raced          12932      59160    3.73  6.6e-16  
      2760  2^3.3.5.23       chain3   raced           6692      18052    2.51  6.0e-16  
      2761  11.251           flat     raced          23369      26309    1.06  7.8e-16  
      2762  2.1381           prime    raced          21103      27759    1.30  6.9e-16  bluestein
      2763  3^2.307          prime    raced          20893      44744    2.09  8.9e-16  bluestein
      2764  2^2.691          prime    raced          20958      44540    2.07  8.6e-16  bluestein
      2765  5.7.79           flat     raced          14237      29874    2.06  7.4e-16  
      2766  2.3.461          prime    raced          20924      26102    1.19  7.0e-16  bluestein
      2767  2767             prime    raced          20952      49745    2.31  7.7e-16  bluestein
      2768  2^4.173          flat     raced          15839      40463    2.49  5.9e-16  
      2769  3.13.71          flat     raced          12484      35007    2.71  7.5e-16  
      2770  2.5.277          prime    raced          20865      36514    1.64  9.1e-16  bluestein
      2771  17.163           flat     raced          18837      45062    2.13  9.1e-16  
      2772  2^2.3^2.7.11     chain3   raced           4601       9239    1.63  4.9e-16  
      2773  47.59            flat     raced          21347      55027    2.50  8.4e-16  
      2774  2.19.73          flat     raced          12266      47067    3.58  6.1e-16  
      2775  3.5^2.37         chain3   raced           8057      24357    2.96  6.0e-16  
      2776  2^3.347          prime    raced          20913      40310    1.91  7.0e-16  bluestein
      2777  2777             prime    raced          20867      48784    2.21  6.5e-16  bluestein
      2778  2.3.463          prime    raced          21122      26177    1.22  7.1e-16  bluestein
      2779  7.397            prime    raced          21325      26320    1.23  6.9e-16  bluestein
      2780  2^2.5.139        flat     raced          16748      30696    1.81  6.8e-16  
      2781  3^3.103          flat     raced          16274      47633    2.92  7.6e-16  
      2782  2.13.107         prime    raced          21053      54379    2.55  7.6e-16  bluestein
      2783  11^2.23          chain3   raced           6445      23829    3.68  6.9e-16  
      2784  2^5.3.29         chain3   raced           6280      19760    2.74  5.0e-16  
      2785  5.557            prime    raced          20937      44174    2.06  7.3e-16  bluestein
      2786  2.7.199          flat     raced          16373      25639    1.53  7.5e-16  
      2787  3.929            prime    raced          21174      48390    2.24  7.2e-16  bluestein
      2788  2^2.17.41        flat     raced           9103      37415    3.87  7.4e-16  
      2789  2789             prime    raced          21079      49510    2.34  7.5e-16  bluestein
      2790  2.3^2.5.31       chain3   raced           7746      20883    2.53  4.6e-16  
      2791  2791             prime    raced          21226      49525    2.28  7.9e-16  bluestein
      2792  2^3.349          flat     raced          21710      40400    1.83  7.4e-16  
      2793  3.7^2.19         chain3   raced           6177      18593    2.78  5.7e-16  
      2794  2.11.127         flat     raced          16199      28335    1.65  7.2e-16  
      2795  5.13.43          chain3   raced          10032      27991    2.77  6.5e-16  
      2796  2^2.3.233        flat     raced          17575      27032    1.31  7.1e-16  
      2797  2797             prime    raced          20925      49249    2.26  8.1e-16  bluestein
      2798  2.1399           prime    raced          21276      35175    1.18  7.6e-16  bluestein
      2799  3^2.311          prime    raced          21044      45815    2.13  8.2e-16  bluestein
      2800  2^4.5^2.7        chain3   raced           4169       4791    1.06  3.5e-16  
      2801  2801             prime    raced          21044      48868    2.28  5.9e-16  bluestein
      2802  2.3.467          prime    raced          21444      26928    1.23  5.2e-16  bluestein
      2803  2803             prime    raced          21192      50119    2.28  6.2e-16  bluestein
      2804  2^2.701          flat     raced          25630      36703    0.99  7.4e-16  
      2805  3.5.11.17        chain3   raced           5933      21057    3.32  5.7e-16  
      2806  2.23.61          flat     raced          13078      46777    3.31  6.3e-16  
      2807  7.401            flat     raced          24886      27665    1.06  7.4e-16  
      2808  2^3.3^3.13       chain3   raced           4651      13726    1.88  5.8e-16  flips differ 1.45x
      2809  53^2             prime    raced          22285      53861    2.36  6.6e-16  bluestein
      2810  2.5.281          flat     raced          16725      43234    1.96  6.3e-16  flips differ 1.37x
      2811  3.937            prime    raced          22423      47941    2.10  7.4e-16  bluestein
      2812  2^2.19.37        chain3   raced           9849      37131    3.73  5.9e-16  
      2813  29.97            flat     raced          14099      43685    2.71  6.2e-16  
      2814  2.3.7.67         flat     raced          13029      33902    2.48  5.8e-16  
      2815  5.563            prime    raced          21150      45400    2.10  7.4e-16  bluestein
      2816  2^8.11           chain3   raced           4523       6745    1.49  4.2e-16  
      2817  3^2.313          flat     raced          20821      45105    2.12  6.0e-16  
      2818  2.1409           prime    raced          21303      35345    1.31  7.9e-16  bluestein
      2819  2819             prime    raced          21393      35605    1.33  5.9e-16  bluestein
      2820  2^2.3.5.47       chain3   raced           7784      28187    3.48  6.7e-16  
      2821  7.13.31          chain3   raced           8851      24115    2.65  7.1e-16  
      2822  2.17.83          prime    raced          21203      53848    2.48  6.3e-16  bluestein
      2823  3.941            prime    raced          21282      26842    1.25  6.6e-16  bluestein
      2824  2^3.353          flat     raced          13656      40039    2.89  6.7e-16  
      2825  5^2.113          flat     raced          13216      38818    2.85  6.8e-16  
      2826  2.3^2.157        flat     raced          15075      27394    1.62  7.4e-16  
      2827  11.257           flat     raced          18181      29375    1.56  4.8e-16  
      2828  2^2.7.101        flat     raced          16224      29377    1.65  6.3e-16  
      2829  3.23.41          flat     raced          12724      40627    2.60  6.1e-16  
      2830  2.5.283          prime    raced          31149      56782    1.73  6.9e-16  bluestein
      2831  19.149           flat     raced          27876      57285    1.64  7.7e-16  
      2832  2^4.3.59         flat     raced          17469      35320    1.91  6.5e-16  
      2833  2833             prime    raced          21198      37150    1.39  7.1e-16  bluestein
      2834  2.13.109         flat     raced          17056      37162    2.06  7.3e-16  
      2835  3^4.5.7          chain3   raced           5469       9659    1.63  6.5e-16  
      2836  2^2.709          prime    raced          24146      44388    1.53  7.6e-16  bluestein
      2837  2837             prime    raced          24622      35480    1.43  8.5e-16  bluestein
      2838  2.3.11.43        chain3   raced          11914      34972    2.56  5.5e-16  
      2839  17.167           prime    raced          21538      48697    1.67  7.3e-16  bluestein
      2840  2^3.5.71         flat     raced          10966      31530    2.63  6.0e-16  
      2841  3.947            prime    raced          21315      27098    1.27  6.4e-16  bluestein
      2842  2.7^2.29         flat     raced          11625      21688    1.82  5.4e-16  
      2843  2843             prime    raced          21856      36615    1.38  8.5e-16  bluestein
      2844  2^2.3^2.79       flat     raced          13168      30481    2.24  9.0e-16  
      2845  5.569            prime    raced          22109      59075    2.54  7.0e-16  bluestein
      2846  2.1423           prime    raced          21045      29724    1.36  6.6e-16  bluestein
      2847  3.13.73          flat     raced          12610      36247    2.82  6.5e-16  
      2848  2^5.89           flat     raced          11477      30705    2.65  5.7e-16  
      2849  7.11.37          chain3   raced           8734      26309    2.93  4.8e-16  
      2850  2.3.5^2.19       chain3   raced           5908      16963    2.53  5.3e-16  
      2851  2851             prime    raced          21127      28424    1.31  6.2e-16  bluestein
      2852  2^2.23.31        chain3   raced           8765      36768    3.37  4.5e-16  
      2853  3^2.317          prime    raced          21351      25623    1.17  6.8e-16  bluestein
      2854  2.1427           prime    raced          21292      27731    1.24  6.8e-16  bluestein
      2855  5.571            prime    raced          21155      36331    1.41  7.5e-16  bluestein
      2856  2^3.3.7.17       chain3   raced           5670      20142    3.53  5.9e-16  
      2857  2857             prime    raced          20959      35872    1.35  7.9e-16  bluestein
      2858  2.1429           prime    raced          21140      35862    1.36  6.6e-16  bluestein
      2859  3.953            prime    raced          21142      27106    1.24  7.6e-16  bluestein
      2860  2^2.5.11.13      flat     raced           8048      10565    1.30  6.0e-16  
      2861  2861             prime    raced          21367      29371    1.34  8.4e-16  bluestein
      2862  2.3^3.53         flat     raced          12710      29427    2.28  7.4e-16  
      2863  7.409            prime    raced          21298      40029    1.46  7.4e-16  bluestein
      2864  2^4.179          prime    raced          21083      40568    1.71  7.9e-16  bluestein
      2865  3.5.191          flat     raced          18831      23850    1.19  7.0e-16  
      2866  2.1433           prime    raced          21053      28559    1.35  7.2e-16  bluestein
      2867  47.61            flat     raced          14563      55417    3.77  5.6e-16  
      2868  2^2.3.239        flat     raced          18463      27877    1.48  1.0e-15  
      2869  19.151           flat     raced          19536      38527    1.74  8.0e-16  
      2870  2.5.7.41         chain3   raced           9423      24523    2.43  4.7e-16  
      2871  3^2.11.29        chain3   raced           7198      23928    2.83  5.8e-16  
      2872  2^3.359          prime    raced          21297      39769    1.85  7.2e-16  bluestein
      2873  13^2.17          chain3   raced           6332      22820    3.58  8.0e-16  
      2874  2.3.479          prime    raced          20952      26221    1.19  7.9e-16  bluestein
      2875  5^3.23           chain3   raced           6879      20756    2.65  5.5e-16  
      2876  2^2.719          prime    raced          21053      42973    2.03  7.1e-16  bluestein
      2877  3.7.137          flat     raced          14851      32582    2.15  6.9e-16  
      2878  2.1439           prime    raced          21050      35856    1.38  8.4e-16  bluestein
      2879  2879             prime    raced          21216      35869    1.39  8.3e-16  bluestein
      2880  2^6.3^2.5        ztt      raced           3807       7751    2.01  4.7e-16  
      2881  43.67            flat     raced          15455      56812    3.66  7.1e-16  
      2882  2.11.131         flat     raced          14057      39790    2.65  7.6e-16  
      2883  3.31^2           chain3   raced          11294      39944    3.38  6.0e-16  
      2884  2^2.7.103        flat     raced          14942      52071    3.39  7.0e-16  
      2885  5.577            flat     raced          26666      29479    1.03  8.3e-16  
      2886  2.3.13.37        chain3   raced           8747      27154    3.03  6.3e-16  
      2887  2887             prime    raced          27488      64870    2.18  6.9e-16  bluestein
      2888  2^3.19^2         flat     raced           8801      32261    3.20  5.3e-16  
      2889  3^3.107          prime    raced          28071      48641    1.72  6.1e-16  bluestein
      2890  2.5.17^2         chain3   raced           6502      30149    4.49  6.1e-16  
      2891  7^2.59           flat     raced          19724      32825    1.62  8.0e-16  
      2892  2^2.3.241        flat     raced          14456      28915    1.94  6.1e-16  
      2893  11.263           prime    raced          26388      36766    1.27  6.0e-16  bluestein
      2894  2.1447           prime    raced          28189      59672    1.87  6.8e-16  bluestein
      2895  3.5.193          flat     raced          17464      28650    1.60  5.9e-16  
      2896  2^4.181          flat     raced          12827      33089    2.21  5.4e-16  
      2897  2897             prime    raced          26465      50829    1.36  7.6e-16  bluestein
      2898  2.3^2.7.23       chain3   raced           8483      19811    2.31  6.5e-16  
      2899  13.223           flat     raced          25019      36835    1.45  8.1e-16  
      2900  2^2.5^2.29       chain3   raced           6821      24964    3.62  5.3e-16  
      2901  3.967            prime    raced          26550      37975    1.38  6.8e-16  bluestein
      2902  2.1451           prime    raced          26660      39305    1.43  6.2e-16  bluestein
      2903  2903             prime    raced          29062      37522    1.22  7.6e-16  bluestein
      2904  2^3.3.11^2       flat     raced           6842      11042    1.59  4.5e-16  
      2905  5.7.83           flat     raced          23326      42719    1.81  3.8e-16  
      2906  2.1453           prime    raced          27123      38291    1.41  6.0e-16  bluestein
      2907  3^2.17.19        chain3   raced           7299      31927    4.23  6.3e-16  
      2908  2^2.727          flat     raced          23736      43450    1.62  8.2e-16  
      2909  2909             prime    raced          28911      36988    1.25  7.3e-16  bluestein
      2910  2.3.5.97         flat     raced          12941      28650    2.03  6.0e-16  
      2911  41.71            flat     raced          14857      64911    4.23  6.2e-16  
      2912  2^5.7.13         chain3   raced           4197      10899    2.47  5.6e-16  
      2913  3.971            prime    raced          26496      51311    1.40  7.3e-16  bluestein
      2914  2.31.47          flat     raced          14229      45551    2.89  5.6e-16  
      2915  5.11.53          flat     raced          13647      33221    2.34  7.7e-16  
      2916  2^2.3^6          chain3   raced           4653       8653    1.46  6.5e-16  
      2917  2917             prime    raced          28351      30817    1.08  8.1e-16  bluestein
      2918  2.1459           prime    raced          26287      39000    1.07  6.6e-16  flips differ 1.63x bluestein
      2919  3.7.139          flat     raced          19004      32599    1.67  7.4e-16  
      2920  2^3.5.73         flat     raced          10876      36777    3.36  5.5e-16  
      2921  23.127           flat     raced          19524      41149    2.04  7.4e-16  
      2922  2.3.487          flat     raced          30763      29935    0.90  6.9e-16  
      2923  37.79            flat     raced          15673      50308    3.15  8.3e-16  
      2924  2^2.17.43        flat     raced          10050      40021    3.90  7.2e-16  
      2925  3^2.5^2.13       chain3   raced           6126       9570    1.48  4.9e-16  
      2926  2.7.11.19        flat     raced           9178      21802    2.35  5.3e-16  
      2927  2927             prime    raced          26506      31014    1.13  6.3e-16  bluestein
      2928  2^4.3.61         flat     raced          11469      33226    2.88  5.4e-16  
      2929  29.101           flat     raced          17141      47284    2.72  7.1e-16  
      2930  2.5.293          prime    raced          28997      28312    0.96  6.5e-16  bluestein
      2931  3.977            prime    raced          26692      29294    1.06  5.5e-16  bluestein
      2932  2^2.733          prime    raced          28739      40433    1.18  5.4e-16  bluestein
      2933  7.419            prime    raced          28117      38839    1.12  5.3e-16  bluestein
      2934  2.3^2.163        flat     raced          18543      35211    1.64  7.2e-16  
      2935  5.587            prime    raced          27629      29296    0.97  7.1e-16  bluestein
      2936  2^3.367          flat     raced          27405      34685    1.20  8.2e-16  
      2937  3.11.89          flat     raced          13241      58771    3.30  5.0e-16  
      2938  2.13.113         flat     raced          12932      46924    3.46  7.0e-16  
      2939  2939             prime    raced          28895      30787    1.04  5.5e-16  bluestein
      2940  2^2.3.5.7^2      flat     raced           6888       8093    0.89  4.3e-16  
      2941  17.173           prime    raced          21832      50873    2.07  6.3e-16  bluestein
      2942  2.1471           prime    raced          21701      40136    1.54  6.8e-16  bluestein
      2943  3^3.109          flat     raced          20304      43106    2.09  7.4e-16  
      2944  2^7.23           chain3   raced           6110      19642    3.13  5.8e-16  
      2945  5.19.31          chain3   raced           9266      36283    3.82  4.9e-16  
      2946  2.3.491          prime    raced          21610      28969    1.33  6.4e-16  bluestein
      2947  7.421            prime    raced          22009      46614    2.08  6.0e-16  bluestein
      2948  2^2.11.67        flat     raced          12959      40249    2.58  5.8e-16  
      2949  3.983            prime    raced          22493      29416    1.30  7.7e-16  bluestein
      2950  2.5^2.59         flat     raced          21019      34328    1.54  6.5e-16  
      2951  13.227           prime    raced          20933      33369    1.49  7.1e-16  bluestein
      2952  2^3.3^2.41       chain3   raced           8130      27463    3.21  5.5e-16  
      2953  2953             prime    raced          21127      31761    1.45  6.0e-16  bluestein
      2954  2.7.211          flat     raced          18214      50955    2.68  7.1e-16  
      2955  3.5.197          flat     raced          16727      28370    1.67  6.1e-16  
      2956  2^2.739          prime    raced          22482      39716    1.43  5.6e-16  bluestein
      2957  2957             prime    raced          21317      31382    1.43  7.9e-16  bluestein
      2958  2.3.17.29        chain3   raced           8332      35086    3.96  6.9e-16  
      2959  11.269           prime    raced          20847      36998    1.75  7.8e-16  bluestein
      2960  2^4.5.37         chain3   raced           6874      23074    3.31  4.7e-16  
      2961  3^2.7.47         chain3   raced          10083      28367    2.76  4.5e-16  
      2962  2.1481           prime    raced          22367      38131    1.37  6.2e-16  bluestein
      2963  2963             prime    raced          22536      30853    1.35  6.5e-16  bluestein
      2964  2^2.3.13.19      chain3   raced           7469      22476    2.69  7.3e-16  
      2965  5.593            prime    raced          21794      30819    1.37  5.8e-16  bluestein
      2966  2.1483           prime    raced          22791      43585    1.55  6.0e-16  bluestein
      2967  3.23.43          chain3   raced          11059      42648    3.30  5.6e-16  
      2968  2^3.7.53         flat     raced          11064      32418    2.74  6.2e-16  
      2969  2969             prime    raced          21130      31692    1.48  6.9e-16  bluestein
      2970  2.3^3.5.11       chain3   raced           5036       8294    1.64  6.1e-16  
      2971  2971             prime    raced          21803      30874    1.39  7.5e-16  bluestein
      2972  2^2.743          prime    raced          22489      44870    1.67  7.2e-16  bluestein
      2973  3.991            prime    raced          23058      30443    1.21  6.9e-16  bluestein
      2974  2.1487           prime    raced          22751      31658    1.37  6.5e-16  bluestein
      2975  5^2.7.17         chain3   raced           6065      19645    3.19  5.5e-16  
      2976  2^5.3.31         chain3   raced           6234      21837    3.09  6.8e-16  
      2977  13.229           flat     raced          21594      32719    1.42  8.2e-16  
      2978  2.1489           prime    raced          21817      40144    1.47  7.3e-16  bluestein
      2979  3^2.331          flat     raced          20684      43322    2.04  7.4e-16  
      2980  2^2.5.149        flat     raced          19304      33600    1.71  6.4e-16  
      2981  11.271           flat     raced          25664      49579    1.85  6.4e-16  
      2982  2.3.7.71         flat     raced          13880      34171    2.25  6.2e-16  
      2983  19.157           flat     raced          17862      40983    2.19  8.4e-16  
      2984  2^3.373          prime    raced          21078      35097    1.66  6.6e-16  bluestein
      2985  3.5.199          flat     raced          19695      28628    1.43  7.6e-16  
      2986  2.1493           prime    raced          21136      46111    2.07  7.7e-16  bluestein
      2987  29.103           flat     raced          17347      62243    3.06  8.7e-16  
      2988  2^2.3^2.83       flat     raced          22335      47126    1.97  7.3e-16  
      2989  7^2.61           flat     raced          13230      34716    2.52  5.0e-16  
      2990  2.5.13.23        chain3   raced           7914      23026    2.87  6.3e-16  
      2991  3.997            prime    raced          23075      28922    1.13  6.4e-16  bluestein
      2992  2^4.11.17        chain3   raced           5391      21530    3.96  6.2e-16  
      2993  41.73            flat     raced          14061      54583    3.73  6.8e-16  
      2994  2.3.499          prime    raced          20842      29978    1.35  6.4e-16  bluestein
      2995  5.599            prime    raced          21012      29308    1.38  6.0e-16  bluestein
      2996  2^2.7.107        prime    raced          21009      42544    1.70  6.5e-16  bluestein
      2997  3^4.37           chain3   raced           8425      25296    2.98  5.6e-16  
      2998  2.1499           prime    raced          22688      38784    1.39  7.7e-16  bluestein
      2999  2999             prime    raced          22354      31178    1.35  6.0e-16  bluestein
      3000  2^3.3.5^3        chain3   raced           4770       8732    1.80  4.6e-16  
      3001  3001             prime    raced          20779      31062    1.47  5.6e-16  bluestein
      3002  2.19.79          flat     raced          14539      50544    3.30  7.2e-16  
      3003  3.7.11.13        chain3   raced           5489      12637    2.27  5.8e-16  
      3004  2^2.751          flat     raced          36445      29518    0.78  7.5e-16  
      3005  5.601            prime    raced          22954      30530    1.31  5.7e-16  bluestein
      3006  2.3^2.167        prime    raced          21034      40110    0.86  7.2e-16  flips differ 1.87x bluestein
      3007  31.97            flat     raced          16718      48396    2.86  5.7e-16  
      3008  2^6.47           chain3   raced           8065      28755    3.46  4.0e-16  
      3009  3.17.59          prime    raced          22155      47482    2.01  6.2e-16  bluestein
      3010  2.5.7.43         chain3   raced          10146      27119    2.66  5.0e-16  
      3011  3011             prime    raced          23073      53081    2.20  7.8e-16  bluestein
      3012  2^2.3.251        flat     raced          19771      29054    1.44  6.5e-16  
      3013  23.131           flat     raced          16863      53180    3.07  8.1e-16  
      3014  2.11.137         flat     raced          14423      36516    2.40  7.4e-16  
      3015  3^2.5.67         flat     raced          14059      38223    2.60  5.4e-16  
      3016  2^3.13.29        chain3   raced           8444      24848    2.73  6.1e-16  
      3017  7.431            prime    raced          20938      46350    2.18  5.9e-16  bluestein
      3018  2.3.503          prime    raced          21740      28346    1.29  5.6e-16  bluestein
      3019  3019             prime    raced          22551      55597    2.44  7.5e-16  bluestein
      3020  2^2.5.151        flat     raced          18527      26590    1.43  7.8e-16  
      3021  3.19.53          flat     raced          14654      46182    3.13  6.8e-16  
      3022  2.1511           prime    raced          22849      31420    1.33  8.0e-16  bluestein
      3023  3023             prime    raced          22347      56890    2.19  7.6e-16  bluestein
      3024  2^4.3^3.7        chain3   raced           4532       8114    1.37  4.6e-16  
      3025  5^2.11^2         chain3   raced           5258      13299    2.49  4.9e-16  
      3026  2.17.89          flat     raced          12871      63255    4.40  8.0e-16  
      3027  3.1009           prime    raced          22363      48806    1.89  7.6e-16  bluestein
      3028  2^2.757          prime    raced          21113      31357    1.12  6.7e-16  flips differ 1.28x bluestein
      3029  13.233           flat     raced          24320      32887    1.09  6.5e-16  
      3030  2.3.5.101        flat     raced          18992      32012    1.62  6.5e-16  
      3031  7.433            prime    raced          22220      36670    1.54  8.6e-16  bluestein
      3032  2^3.379          flat     raced          22022      37176    1.12  6.9e-16  flips differ 1.94x
      3033  3^2.337          flat     raced          22679      45900    1.95  7.6e-16  
      3034  2.37.41          flat     raced          13953      51780    2.93  6.0e-16  
      3035  5.607            prime    raced          21963      47897    2.05  7.0e-16  bluestein
      3036  2^2.3.11.23      chain3   raced           7526      26535    3.11  5.6e-16  
      3037  3037             prime    raced          22198      58639    2.35  7.9e-16  bluestein
      3038  2.7^2.31         flat     raced          10876      25121    1.91  4.4e-16  
      3039  3.1013           prime    raced          21726      52997    2.25  7.3e-16  bluestein
      3040  2^5.5.19         chain3   raced           5822      18018    2.96  4.9e-16  
      3041  3041             prime    raced          21309      30607    1.28  7.2e-16  rader
      3042  2.3^2.13^2       flat     raced           9038      11912    1.29  6.5e-16  
      3043  17.179           prime    raced          21760      54384    2.11  6.9e-16  bluestein
      3044  2^2.761          prime    raced          21859      30866    1.29  6.3e-16  bluestein
      3045  3.5.7.29         chain3   raced           7673      25059    2.60  5.6e-16  
      3046  2.1523           prime    raced          22670      33045    1.29  5.7e-16  bluestein
      3047  11.277           prime    raced          22935      50522    1.83  6.7e-16  bluestein
      3048  2^3.3.127        flat     raced          17915      32559    1.78  7.7e-16  
      3049  3049             prime    raced          21750      31750    1.20  5.3e-16  bluestein
      3050  2.5^2.61         flat     raced          14361      36145    2.34  6.4e-16  
      3051  3^3.113          flat     raced          14286      42682    2.87  5.2e-16  
      3052  2^2.7.109        flat     raced          18405      41711    2.19  5.6e-16  
      3053  43.71            flat     raced          15358      61908    3.59  7.3e-16  
      3054  2.3.509          prime    raced          21108      28625    1.33  6.0e-16  bluestein
      3055  5.13.47          chain3   raced          11732      32334    2.69  6.7e-16  
      3056  2^4.191          flat     raced          16868      24568    1.37  6.0e-16  
      3057  3.1019           prime    raced          20779      28809    1.28  6.0e-16  bluestein
      3058  2.11.139         flat     raced          19837      36801    1.83  7.3e-16  
      3059  7.19.23          chain3   raced           8159      35143    4.22  6.1e-16  
      3060  2^2.3^2.5.17     chain3   raced           5942      20320    3.35  6.0e-16  
      3061  3061             prime    raced          23171      30433    1.30  6.8e-16  bluestein
      3062  2.1531           prime    raced          21437      31017    1.43  6.9e-16  bluestein
      3063  3.1021           prime    raced          22907      28540    1.10  6.3e-16  bluestein
      3064  2^3.383          prime    raced          22113      25719    1.14  6.4e-16  bluestein
      3065  5.613            prime    raced          22418      41904    1.84  7.1e-16  bluestein
      3066  2.3.7.73         flat     raced          13560      37593    2.55  6.2e-16  
      3067  3067             prime    raced          21147      31158    1.38  6.4e-16  bluestein
      3068  2^2.13.59        flat     raced          20968      37658    1.72  7.8e-16  
      3069  3^2.11.31        chain3   raced           8190      26827    2.90  5.7e-16  
      3070  2.5.307          prime    raced          21365      51190    2.31  7.3e-16  bluestein
      3071  37.83            prime    raced          21561      65200    3.01  5.5e-16  bluestein
      3072  2^10.3           ztt      raced           3626       4700    1.28  3.5e-16  
      3073  7.439            prime    raced          21538      47657    1.45  5.7e-16  bluestein
      3074  2.29.53          flat     raced          14438      50993    3.44  4.8e-16  
      3075  3.5^2.41         chain3   raced           9887      28849    2.61  5.1e-16  
      3076  2^2.769          flat     raced          20880      28938    1.38  5.0e-16  
      3077  17.181           flat     raced          19134      53325    2.75  6.7e-16  
      3078  2.3^4.19         chain3   raced           7008      20031    1.19  5.7e-16  flips differ 2.42x
      3079  3079             prime    raced          22999      49091    2.02  5.7e-16  bluestein
      3080  2^3.5.7.11       flat     raced           7415       8375    1.05  3.4e-16  
      3081  3.13.79          flat     raced          15392      39584    2.56  9.1e-16  
      3082  2.23.67          flat     raced          14897      54279    3.59  6.2e-16  
      3083  3083             prime    raced          21547      45364    1.98  5.5e-16  bluestein
      3084  2^2.3.257        flat     raced          15878      30056    1.84  4.3e-16  
      3085  5.617            prime    raced          21881      41080    1.87  6.0e-16  bluestein
      3086  2.1543           prime    raced          21777      58818    2.34  6.1e-16  bluestein
      3087  3^2.7^3          chain3   raced           5888      10910    1.79  4.7e-16  
      3088  2^4.193          flat     raced          13829      29512    2.12  6.5e-16  
      3089  3089             prime    raced          21447      44305    1.77  5.8e-16  bluestein
      3090  2.3.5.103        flat     raced          15785      44446    2.22  8.6e-16  
      3091  11.281           flat     raced          19220      41811    1.73  5.7e-16  
      3092  2^2.773          prime    raced          21438      31241    1.41  6.6e-16  bluestein
      3093  3.1031           prime    raced          21484      49178    2.12  5.9e-16  bluestein
      3094  2.7.13.17        flat     raced           9405      22359    2.36  7.0e-16  
      3095  5.619            prime    raced          21593      41899    1.91  6.2e-16  bluestein
      3096  2^3.3^2.43       chain3   raced           8325      31538    3.73  6.4e-16  
      3097  19.163           flat     raced          23545      51347    2.15  6.7e-16  
      3098  2.1549           prime    raced          21501      57735    2.64  6.1e-16  bluestein
      3099  3.1033           prime    raced          21485      40372    1.43  7.2e-16  bluestein
      3100  2^2.5^2.31       chain3   raced           8007      22891    2.55  4.8e-16  
      3101  7.443            prime    raced          21623      46747    2.15  6.2e-16  bluestein
      3102  2.3.11.47        flat     raced          14253      32613    2.18  5.0e-16  
      3103  29.107           prime    raced          21967      74416    1.81  5.5e-16  flips differ 1.90x bluestein
      3104  2^5.97           flat     raced          13346      30447    2.21  3.9e-16  
      3105  3^3.5.23         chain3   raced           7895      22621    2.78  5.3e-16  
      3106  2.1553           prime    raced          21518      57761    2.52  6.8e-16  bluestein
      3107  13.239           flat     raced          19311      32440    1.56  1.1e-15  
      3108  2^2.3.7.37       chain3   raced           7868      27907    3.20  6.0e-16  
      3109  3109             prime    raced          21463      48124    2.03  6.0e-16  bluestein
      3110  2.5.311          prime    raced          21777      50809    2.26  7.1e-16  bluestein
      3111  3.17.61          flat     raced          14790      49889    3.16  5.5e-16  
      3112  2^3.389          prime    raced          22135      29724    0.72  6.2e-16  flips differ 1.86x bluestein
      3113  11.283           prime    raced          60092      55450    0.77  8.4e-16  flips differ 1.38x bluestein
      3114  2.3^2.173        prime    raced          22915      40260    1.50  7.9e-16  bluestein
      3115  5.7.89           flat     raced          14459      38998    2.55  7.2e-16  
      3116  2^2.19.41        flat     raced          11610      44512    3.19  4.8e-16  
      3117  3.1039           prime    raced          23158      48987    1.66  6.7e-16  flips differ 1.28x bluestein
      3118  2.1559           prime    raced          21975      64091    2.74  6.3e-16  bluestein
      3119  3119             prime    raced          22289      64267    2.11  6.9e-16  bluestein
      3120  2^4.3.5.13       chain3   raced           5230       9338    1.44  6.2e-16  
      3121  3121             prime    raced          19928      44945    2.23  8.4e-16  rader
      3122  2.7.223          prime    raced          21974      37092    1.50  7.2e-16  bluestein
      3123  3^2.347          prime    raced          23383      38664    1.38  6.9e-16  bluestein
      3124  2^2.11.71        flat     raced          13687      40796    2.89  5.6e-16  
      3125  5^5              chain3   raced           6044      12104    1.96  4.4e-16  
      3126  2.3.521          prime    raced          21954      52936    2.32  7.6e-16  bluestein
      3127  53.59            prime    raced          23257      63056    2.12  6.8e-16  flips differ 1.26x bluestein
      3128  2^3.17.23        chain3   raced           8385      34827    4.13  7.0e-16  
      3129  3.7.149          prime    raced          21554      36226    1.68  7.4e-16  bluestein
      3130  2.5.313          flat     raced          20400      51018    2.46  9.3e-16  
      3131  31.101           flat     raced          19925      51153    2.55  6.2e-16  
      3132  2^2.3^3.29       chain3   raced           8066      25490    2.15  5.4e-16  flips differ 1.40x
      3133  13.241           flat     raced          17221      32904    1.88  6.9e-16  
      3134  2.1567           prime    raced          21752      47243    1.68  6.0e-16  bluestein
      3135  3.5.11.19        chain3   raced           7935      23758    2.77  5.5e-16  
      3136  2^6.7^2          ztt      raced           3719       7943    1.58  3.3e-16  
      3137  3137             prime    raced          16056      30800    1.90  1.1e-15  rader
      3138  2.3.523          prime    raced          21511      50780    2.31  6.8e-16  bluestein
      3139  43.73            flat     raced          15073      59083    3.71  6.8e-16  
      3140  2^2.5.157        flat     raced          16356      26457    1.59  8.0e-16  
      3141  3^2.349          prime    raced          21469      46067    2.10  8.8e-16  bluestein
      3142  2.1571           prime    raced          21555      31435    1.40  7.0e-16  bluestein
      3143  7.449            prime    raced          22546      32349    0.71  6.3e-16  flips differ 2.03x bluestein
      3144  2^3.3.131        flat     raced          13577      40873    2.90  7.5e-16  
      3145  5.17.37          chain3   raced          10743      40177    3.63  5.5e-16  
      3146  2.11^2.13        flat     raced           9507      15506    1.62  5.6e-16  
      3147  3.1049           prime    raced          21464      48551    2.21  7.3e-16  bluestein
      3148  2^2.787          prime    raced          22224      32068    1.37  7.0e-16  bluestein
      3149  47.67            flat     raced          16672      63605    3.76  6.8e-16  
      3150  2.3^2.5^2.7      chain3   raced           6395       6479    0.97  5.0e-16  
      3151  23.137           flat     raced          16992      51317    2.87  7.7e-16  
      3152  2^4.197          flat     raced          12794      29043    2.25  4.9e-16  
      3153  3.1051           prime    raced          21706      41533    1.49  6.6e-16  bluestein
      3154  2.19.83          prime    raced          21723      61542    2.78  5.7e-16  bluestein
      3155  5.631            prime    raced          23008      28109    1.17  6.5e-16  bluestein
      3156  2^2.3.263        prime    raced          21103      38247    1.70  6.8e-16  bluestein
      3157  7.11.41          chain3   raced          12003      30780    2.55  4.3e-16  
      3158  2.1579           prime    raced          21455      38906    1.47  6.3e-16  bluestein
      3159  3^5.13           chain3   raced           5690      10598    1.74  6.6e-16  
      3160  2^3.5.79         flat     raced          13228      32838    2.22  8.1e-16  
      3161  29.109           flat     raced          19714      54005    2.59  6.9e-16  
      3162  2.3.17.31        chain3   raced          10597      38381    3.49  9.1e-16  
      3163  3163             prime    raced          21350      33703    1.48  7.0e-16  bluestein
      3164  2^2.7.113        flat     raced          14062      43959    2.92  6.8e-16  
      3165  3.5.211          flat     raced          21027      43126    1.67  5.8e-16  
      3166  2.1583           prime    raced          21494      30967    1.44  5.2e-16  bluestein
      3167  3167             prime    raced          21152      30819    1.39  7.1e-16  bluestein
      3168  2^5.3^2.11       chain3   raced           4471      11825    2.63  5.5e-16  
      3169  3169             prime    raced          18967      53151    1.60  7.6e-16  flips differ 1.97x rader
      3170  2.5.317          prime    raced          31714      35804    0.90  4.6e-16  bluestein
      3171  3.7.151          flat     raced          20338      29305    1.37  6.9e-16  
      3172  2^2.13.61        flat     raced          13446      39975    2.83  6.6e-16  
      3173  19.167           prime    raced          21652      64706    0.97  6.9e-16  flips differ 3.04x bluestein
      3174  2.3.23^2         chain3   raced           8834      38218    3.63  5.4e-16  
      3175  5^2.127          flat     raced          19261      29273    1.29  7.4e-16  
      3176  2^3.397          flat     raced          16765      31310    1.81  6.7e-16  
      3177  3^2.353          flat     raced          20124      40355    1.64  7.9e-16  
      3178  2.7.227          prime    raced          21157      32751    1.46  6.6e-16  bluestein
      3179  11.17^2          chain3   raced           7530      36998    4.79  7.1e-16  
      3180  2^2.3.5.53       flat     raced          13489      33635    2.39  7.7e-16  
      3181  3181             prime    raced          21247      31953    1.41  6.2e-16  bluestein
      3182  2.37.43          flat     raced          15645      50950    3.23  5.8e-16  
      3183  3.1061           prime    raced          22107      36498    1.52  6.8e-16  bluestein
      3184  2^4.199          flat     raced          14873      29494    1.94  6.1e-16  
      3185  5.7^2.13         flat     raced           8216      10785    1.30  5.5e-16  
      3186  2.3^3.59         prime    raced          21496      35559    1.63  5.3e-16  bluestein
      3187  3187             prime    raced          21209      31404    1.46  5.4e-16  bluestein
      3188  2^2.797          prime    raced          21185      30710    1.41  5.7e-16  bluestein
      3189  3.1063           prime    raced          22485      48690    2.14  8.7e-16  bluestein
      3190  2.5.11.29        chain3   raced           7829      27520    2.61  4.2e-16  flips differ 1.28x
      3191  3191             prime    raced          22137      30833    1.36  7.0e-16  bluestein
      3192  2^3.3.7.19       chain3   raced           6777      21708    2.91  4.9e-16  
      3193  31.103           flat     raced          20329      74195    3.54  7.8e-16  
      3194  2.1597           prime    raced          24116      31011    1.26  6.6e-16  bluestein
      3195  3^2.5.71         flat     raced          13917      37923    2.65  6.5e-16  
      3196  2^2.17.47        flat     raced          11412      46404    3.91  7.5e-16  
      3197  23.139           flat     raced          20889      51923    2.45  8.6e-16  
      3198  2.3.13.41        chain3   raced          11089      30894    2.73  6.3e-16  
      3199  7.457            prime    raced          21740      31014    1.41  6.7e-16  bluestein
      3200  2^7.5^2          ztt      raced           3885       8173    2.08  3.4e-16  
      3201  3.11.97          flat     raced          15447      36590    2.27  5.7e-16  
      3202  2.1601           prime    raced          24333      32567    1.08  7.2e-16  bluestein
      3203  3203             prime    raced          24034      67965    2.64  8.5e-16  bluestein
      3204  2^2.3^2.89       flat     raced          13361      62387    4.65  6.4e-16  
      3205  5.641            prime    raced          23866      30650    1.28  7.0e-16  bluestein
      3206  2.7.229          flat     raced          20357      32710    1.60  6.5e-16  
      3207  3.1069           prime    raced          23913      41393    1.42  8.3e-16  bluestein
      3208  2^3.401          flat     raced          20878      31013    1.47  6.8e-16  
      3209  3209             prime    raced          23864      69730    2.84  9.9e-16  bluestein
      3210  2.3.5.107        prime    raced          23947      52525    2.12  7.5e-16  bluestein
      3211  13^2.19          chain3   raced           7380      25923    3.50  5.0e-16  
      3212  2^2.11.73        flat     raced          13215      41269    2.95  6.1e-16  
      3213  3^3.7.17         chain3   raced           6758      21043    1.06  5.8e-16  flips differ 2.94x
      3214  2.1607           prime    raced          32219      89916    2.48  8.0e-16  bluestein
      3215  5.643            prime    raced          23844      38714    1.33  8.4e-16  bluestein
      3216  2^4.3.67         flat     raced          12922      39335    3.04  5.8e-16  
      3217  3217             prime    raced          23893      52088    1.52  8.7e-16  bluestein
      3218  2.1609           prime    raced          23925      60781    1.61  7.0e-16  bluestein
      3219  3.29.37          chain3   raced          11592      45796    3.50  4.7e-16  
      3220  2^2.5.7.23       flat     raced          11372      25498    2.16  4.9e-16  
      3221  3221             prime    raced          24010      52090    1.43  8.2e-16  bluestein
      3222  2.3^2.179        prime    raced          23883      42363    1.56  6.9e-16  bluestein
      3223  11.293           prime    raced          24116      34757    1.32  7.0e-16  bluestein
      3224  2^3.13.31        chain3   raced           8140      28063    2.83  6.2e-16  
      3225  3.5^2.43         chain3   raced          10187      31336    2.91  6.7e-16  
      3226  2.1613           prime    raced          24075      83119    3.35  8.2e-16  bluestein
      3227  7.461            prime    raced          37239      31730    0.67  7.0e-16  flips differ 1.26x bluestein
      3228  2^2.3.269        prime    raced          37020      37928    0.84  6.6e-16  bluestein
      3229  3229             prime    raced          24146      53182    1.51  8.8e-16  flips differ 1.32x bluestein
      3230  2.5.17.19        chain3   raced           8402      34480    4.06  7.9e-16  
      3231  3^2.359          prime    raced          24385      38995    1.28  7.5e-16  bluestein
      3232  2^5.101          flat     raced          18597      32807    1.75  7.0e-16  
      3233  53.61            prime    raced          24436      65808    2.54  6.7e-16  bluestein
      3234  2.3.7^2.11       flat     raced           9421       9254    0.96  3.8e-16  
      3235  5.647            prime    raced          24025      37393    1.54  8.5e-16  bluestein
      3236  2^2.809          prime    raced          24108      53728    2.05  8.6e-16  bluestein
      3237  3.13.83          prime    raced          23831      50953    2.06  6.9e-16  bluestein
      3238  2.1619           prime    raced          24007      89922    3.71  8.4e-16  bluestein
      3239  41.79            flat     raced          17418      58033    3.27  9.5e-16  
      3240  2^3.3^4.5        chain3   raced           5281       8317    1.19  4.8e-16  
      3241  7.463            prime    raced          23887      30958    1.22  8.8e-16  bluestein
      3242  2.1621           prime    raced          24696      38565    1.48  9.6e-16  bluestein
      3243  3.23.47          flat     raced          15323      48270    3.10  5.6e-16  
      3244  2^2.811          prime    raced          23924      36742    1.49  7.4e-16  bluestein
      3245  5.11.59          flat     raced          23741      39475    1.55  6.0e-16  
      3246  2.3.541          prime    raced          23955      39759    1.47  7.3e-16  bluestein
      3247  17.191           flat     raced          21644      40727    1.83  6.3e-16  
      3248  2^4.7.29         chain3   raced           8260      24156    2.87  4.9e-16  
      3249  3^2.19^2         chain3   raced           8635      35742    4.11  7.3e-16  
      3250  2.5^3.13         chain3   raced           5741       8789    1.42  5.9e-16  
      3251  3251             prime    raced          24137      59698    2.41  9.4e-16  bluestein
      3252  2^2.3.271        flat     raced          20841      37838    1.57  8.8e-16  
      3253  3253             prime    raced          25476      58789    2.29  9.3e-16  bluestein
      3254  2.1627           prime    raced          28060      59966    1.05  7.7e-16  flips differ 2.18x bluestein
      3255  3.5.7.31         chain3   raced           9671      27049    1.59  6.0e-16  flips differ 1.73x
      3256  2^3.11.37        chain3   raced           8575      31262    3.59  4.0e-16  
      3257  3257             prime    raced          23907      60336    2.43  8.2e-16  bluestein
      3258  2.3^2.181        flat     raced          17411      38983    1.98  5.7e-16  
      3259  3259             prime    raced          23996      59448    2.46  1.0e-15  bluestein
      3260  2^2.5.163        flat     raced          22394      38557    1.68  8.1e-16  
      3261  3.1087           prime    raced          23846      58986    2.46  1.1e-15  bluestein
      3262  2.7.233          flat     raced          24326      32756    1.32  6.8e-16  
      3263  13.251           flat     raced          23513      29296    1.12  9.2e-16  
      3264  2^6.3.17         chain3   raced           5458      18980    3.34  5.7e-16  
      3265  5.653            prime    raced          24351      42639    1.68  6.7e-16  bluestein
      3266  2.23.71          flat     raced          14492      55834    3.54  7.0e-16  
      3267  3^3.11^2         chain3   raced           5816      14003    2.40  5.0e-16  
      3268  2^2.19.43        chain3   raced          11817      46154    3.20  5.2e-16  
      3269  7.467            prime    raced          24107      31579    1.15  6.3e-16  bluestein
      3270  2.3.5.109        flat     raced          18899      47588    2.32  6.0e-16  
      3271  3271             prime    raced          24028      58498    2.41  1.1e-15  bluestein
      3272  2^3.409          flat     raced          19047      53882    2.76  7.5e-16  
      3273  3.1091           prime    raced          23948      59724    2.41  8.5e-16  bluestein
      3274  2.1637           prime    raced          23972      56504    2.33  8.0e-16  bluestein
      3275  5^2.131          flat     raced          16707      42197    2.47  9.2e-16  
      3276  2^2.3^2.7.13     chain3   raced           5972      11764    1.63  5.8e-16  
      3277  29.113           flat     raced          15930      64083    3.66  5.3e-16  
      3278  2.11.149         flat     raced          21611      44607    2.03  7.1e-16  
      3279  3.1093           prime    raced          23867      58022    2.40  9.7e-16  bluestein
      3280  2^4.5.41         chain3   raced           8309      27224    3.21  4.4e-16  
      3281  17.193           flat     raced          18592      47615    2.49  6.8e-16  
      3282  2.3.547          prime    raced          23960      54946    2.24  9.0e-16  bluestein
      3283  7^2.67           flat     raced          14979      40270    2.60  7.5e-16  
      3284  2^2.821          prime    raced          24082      37179    1.53  6.9e-16  bluestein
      3285  3^2.5.73         flat     raced          14003      37588    2.62  6.9e-16  
      3286  2.31.53          flat     raced          15405      54375    3.20  5.9e-16  
      3287  19.173           prime    raced          24208      63503    2.60  8.3e-16  bluestein
      3288  2^3.3.137        flat     raced          13878      39280    2.76  5.8e-16  
      3289  11.13.23         chain3   raced           7840      28187    3.07  7.0e-16  
      3290  2.5.7.47         chain3   raced          12288      30706    1.96  7.1e-16  flips differ 1.27x
      3291  3.1097           prime    raced          23867      44712    1.67  7.5e-16  bluestein
      3292  2^2.823          prime    raced          23909      35958    1.45  7.1e-16  bluestein
      3293  37.89            flat     raced          15436      74841    4.24  6.7e-16  
      3294  2.3^3.61         flat     raced          14769      38032    2.46  5.9e-16  
      3295  5.659            prime    raced          24437      44412    1.76  8.8e-16  bluestein
      3296  2^5.103          flat     raced          16502      48299    2.53  6.9e-16  
      3297  3.7.157          flat     raced          18769      29941    1.54  7.6e-16  
      3298  2.17.97          flat     raced          15200      48618    3.13  7.8e-16  
      3299  3299             prime    raced          24371      46689    1.87  8.4e-16  bluestein
      3300  2^2.3.5^2.11     chain3   raced           5401      11870    2.12  5.3e-16  
      3301  3301             prime    raced          25341      51655    1.89  7.8e-16  rader
      3302  2.13.127         flat     raced          18899      33875    1.67  8.1e-16  
      3303  3^2.367          prime    raced          23998      39758    1.62  7.0e-16  bluestein
      3304  2^3.7.59         flat     raced          20049      36363    1.79  5.2e-16  
      3305  5.661            prime    raced          24094      42464    1.74  7.6e-16  bluestein
      3306  2.3.19.29        chain3   raced           9942      40263    3.98  5.1e-16  
      3307  3307             prime    raced          24111      45760    1.82  6.0e-16  bluestein
      3308  2^2.827          prime    raced          24279      36460    1.48  8.5e-16  bluestein
      3309  3.1103           prime    raced          24055      40880    1.69  6.8e-16  bluestein
      3310  2.5.331          flat     raced          21830      47528    2.14  7.6e-16  
      3311  7.11.43          chain3   raced          12225      33615    2.17  5.0e-16  flips differ 1.27x
      3312  2^4.3^2.23       chain3   raced           6718      24371    3.01  4.7e-16  flips differ 1.29x
      3313  3313             prime    raced          24174      45447    1.82  8.0e-16  bluestein
      3314  2.1657           prime    raced          23994      56794    2.26  8.0e-16  bluestein
      3315  3.5.13.17        chain3   raced           7834      24988    3.11  5.8e-16  
      3316  2^2.829          prime    raced          23956      37223    1.48  7.8e-16  bluestein
      3317  31.107           prime    raced          24165      73902    3.05  7.9e-16  bluestein
      3318  2.3.7.79         flat     raced          15800      34884    2.20  8.4e-16  
      3319  3319             prime    raced          24124      46468    1.90  8.2e-16  bluestein
      3320  2^3.5.83         flat     raced          21031      49537    2.30  5.7e-16  
      3321  3^4.41           chain3   raced          11469      29664    2.56  5.7e-16  
      3322  2.11.151         flat     raced          20144      32578    1.52  6.5e-16  
      3323  3323             prime    raced          24646      46738    1.87  9.0e-16  bluestein
      3324  2^2.3.277        flat     raced          22211      44747    1.39  7.8e-16  
      3325  5^2.7.19         chain3   raced           7474      22303    2.97  4.7e-16  
      3326  2.1663           prime    raced          24347      57087    2.32  9.3e-16  bluestein
      3327  3.1109           prime    raced          24366      41034    1.64  8.4e-16  bluestein
      3328  2^8.13           chain3   raced           5108       8436    1.59  5.5e-16  
      3329  3329             prime    raced          19554      45650    2.31  6.9e-16  rader
      3330  2.3^2.5.37       chain3   raced          11644      26979    2.23  5.1e-16  
      3331  3331             prime    raced          24609      47334    1.85  6.8e-16  bluestein
      3332  2^2.7^2.17       flat     raced           9891      24442    2.39  5.4e-16  
      3333  3.11.101         flat     raced          19261      38767    1.89  6.9e-16  
      3334  2.1667           prime    raced          23941      57582    1.47  7.8e-16  flips differ 1.60x bluestein
      3335  5.23.29          chain3   raced          11420      42683    3.31  5.5e-16  
      3336  2^3.3.139        flat     raced          19925      38781    1.73  8.5e-16  
      3337  47.71            flat     raced          16704      69239    3.69  7.0e-16  
      3338  2.1669           prime    raced          23970      56356    2.27  8.0e-16  bluestein
      3339  3^2.7.53         flat     raced          13551      35160    2.58  6.6e-16  
      3340  2^2.5.167        prime    raced          23951      45556    1.51  6.3e-16  bluestein
      3341  13.257           flat     raced          20384      34823    1.62  5.2e-16  
      3342  2.3.557          prime    raced          24076      52291    2.10  8.0e-16  bluestein
      3343  3343             prime    raced          23899      45915    1.63  7.6e-16  bluestein
      3344  2^4.11.19        chain3   raced           6700      23997    3.16  4.8e-16  
      3345  3.5.223          prime    raced          24101      40495    1.64  8.4e-16  bluestein
      3346  2.7.239          flat     raced          19815      36046    1.66  7.5e-16  
      3347  3347             prime    raced          24139      46748    1.86  7.1e-16  bluestein
      3348  2^2.3^3.31       chain3   raced           9907      30415    2.92  5.0e-16  
      3349  17.197           flat     raced          17797      45867    2.53  7.1e-16  
      3350  2.5^2.67         flat     raced          16043      41781    2.52  6.0e-16  
      3351  3.1117           prime    raced          26835      41169    1.49  6.5e-16  bluestein
      3352  2^3.419          flat     raced          23653      44281    1.51  7.3e-16  
      3353  7.479            prime    raced          23893      31172    1.00  6.1e-16  flips differ 1.29x bluestein
      3354  2.3.13.43        chain3   raced          11845      33852    2.70  8.3e-16  
      3355  5.11.61          flat     raced          15618      41617    2.65  5.1e-16  
      3356  2^2.839          prime    raced          23913      36086    1.48  8.5e-16  bluestein
      3357  3^2.373          prime    raced          23981      40199    1.66  7.6e-16  bluestein
      3358  2.23.73          flat     raced          14825      57913    3.62  6.6e-16  
      3359  3359             prime    raced          24175      69117    1.86  7.5e-16  bluestein
      3360  2^5.3.5.7        chain3   raced           4810       7800    1.12  4.3e-16  
      3361  3361             prime    raced          24580      45948    1.85  6.9e-16  rader
      3362  2.41^2           flat     raced          15663      54793    3.37  5.1e-16  
      3363  3.19.59          flat     raced          23835      53548    2.21  6.6e-16  
      3364  2^2.29^2         chain3   raced          11422      45483    3.88  4.7e-16  
      3365  5.673            prime    raced          26842      41832    1.53  5.7e-16  bluestein
      3366  2.3^2.11.17      flat     raced           9803      24311    2.28  5.5e-16  
      3367  7.13.37          chain3   raced          10776      31168    2.57  6.0e-16  
      3368  2^3.421          flat     raced          23462      53179    2.16  7.5e-16  
      3369  3.1123           prime    raced          27018      41270    1.46  6.6e-16  bluestein
      3370  2.5.337          flat     raced          22762      47728    1.97  5.5e-16  
      3371  3371             prime    raced          27783      46690    1.60  6.5e-16  bluestein
      3372  2^2.3.281        flat     raced          17139      45306    2.63  5.4e-16  
      3373  3373             prime    raced          27699      46530    1.63  4.8e-16  bluestein
      3374  2.7.241          flat     raced          17386      32870    1.89  6.2e-16  
      3375  3^3.5^3          chain3   raced           5861      10701    1.75  6.2e-16  
      3376  2^4.211          flat     raced          16973      45466    2.24  7.8e-16  
      3377  11.307           prime    raced          24724      61770    2.42  8.0e-16  bluestein
      3378  2.3.563          prime    raced          24749      52885    2.04  7.1e-16  bluestein
      3379  31.109           flat     raced          20375      60117    2.69  8.2e-16  
      3380  2^2.5.13^2       flat     raced           9150      12829    1.38  7.8e-16  
      3381  3.7^2.23         chain3   raced           8491      24739    2.60  4.7e-16  
      3382  2.19.89          flat     raced          15238      72442    4.09  5.6e-16  
      3383  17.199           flat     raced          21387      47012    2.16  8.5e-16  
      3384  2^3.3^2.47       chain3   raced           9911      32147    3.20  6.1e-16  
      3385  5.677            prime    raced          24878      54080    2.10  8.0e-16  bluestein
      3386  2.1693           prime    raced          24799      44592    1.43  6.3e-16  bluestein
      3387  3.1129           prime    raced          25015      51550    2.04  8.0e-16  bluestein
      3388  2^2.7.11^2       flat     raced           9226      14011    1.49  4.8e-16  
      3389  3389             prime    raced          25053      55117    1.64  8.3e-16  bluestein
      3390  2.3.5.113        flat     raced          14596      47629    3.25  6.6e-16  
      3391  3391             prime    raced          24957      53563    1.51  8.1e-16  bluestein
      3392  2^6.53           flat     raced          11820      33959    2.85  7.1e-16  
      3393  3^2.13.29        chain3   raced          10082      28832    2.80  6.1e-16  
      3394  2.1697           prime    raced          25167      36092    1.42  7.6e-16  bluestein
      3395  5.7.97           flat     raced          16820      34374    1.98  4.9e-16  
      3396  2^2.3.283        prime    raced          25571      45262    1.64  6.4e-16  bluestein
      3397  43.79            flat     raced          19077      61274    3.20  8.4e-16  
      3398  2.1699           prime    raced          24771      53510    2.15  8.1e-16  bluestein
      3399  3.11.103         flat     raced          18413      65548    3.51  9.1e-16  
      3400  2^3.5^2.17       chain3   raced           6542      19604    2.99  5.3e-16  
      3401  19.179           prime    raced          24814      57449    2.04  7.7e-16  bluestein
      3402  2.3^5.7          chain3   raced           6446       8952    1.34  4.8e-16  
      3403  41.83            prime    raced          24840      76663    2.72  6.0e-16  bluestein
      3404  2^2.23.37        chain3   raced          12641      45896    3.61  6.3e-16  
      3405  3.5.227          prime    raced          24844      35507    1.39  6.4e-16  bluestein
      3406  2.13.131         flat     raced          16568      46295    2.77  8.8e-16  
      3407  3407             prime    raced          25376      53099    1.46  7.2e-16  bluestein
      3408  2^4.3.71         flat     raced          12967      42951    3.25  6.9e-16  
      3409  7.487            prime    raced          24924      33722    1.34  8.1e-16  bluestein
      3410  2.5.11.31        chain3   raced           9617      33955    3.34  5.5e-16  
      3411  3^2.379          flat     raced          27920      29776    0.95  7.2e-16  
      3412  2^2.853          prime    raced          25339      37742    1.43  7.0e-16  bluestein
      3413  3413             prime    raced          25093      92096    1.47  7.3e-16  flips differ 1.86x bluestein
      3414  2.3.569          prime    raced          25201      68837    2.70  8.0e-16  bluestein
      3415  5.683            prime    raced          24840      45328    1.48  6.5e-16  bluestein
      3416  2^3.7.61         flat     raced          13028      38955    2.94  4.4e-16  
      3417  3.17.67          flat     raced          16182      57816    3.49  6.7e-16  
      3418  2.1709           prime    raced          25271      47553    1.40  7.1e-16  bluestein
      3419  13.263           prime    raced          24958      43582    1.73  8.4e-16  bluestein
      3420  2^2.3^2.5.19     chain3   raced           6990      20888    2.77  4.5e-16  
      3421  11.311           prime    raced          25739      58542    1.18  8.0e-16  flips differ 1.96x bluestein
      3422  2.29.59          flat     raced          25859      59038    2.20  6.7e-16  
      3423  3.7.163          flat     raced          22177      41459    1.83  8.2e-16  
      3424  2^5.107          prime    raced          25121      56372    2.12  7.6e-16  bluestein
      3425  5^2.137          flat     raced          18229      39285    2.11  8.3e-16  
      3426  2.3.571          prime    raced          24865      61577    2.19  6.5e-16  bluestein
      3427  23.149           prime    raced          24949      56831    2.24  8.7e-16  bluestein
      3428  2^2.857          prime    raced          24731      36181    1.39  7.5e-16  bluestein
      3429  3^3.127          flat     raced          23734      32227    1.30  7.9e-16  
      3430  2.5.7^3          flat     raced           9613       8862    0.92  4.1e-16  
      3431  47.73            flat     raced          16376      66437    3.92  6.7e-16  
      3432  2^3.3.11.13      flat     raced           7944      16649    2.06  5.3e-16  
      3433  3433             prime    raced          25273      61690    1.47  8.7e-16  bluestein
      3434  2.17.101         flat     raced          19088      51619    2.42  7.3e-16  
      3435  3.5.229          flat     raced          27143      36299    1.28  6.0e-16  
      3436  2^2.859          prime    raced          25041      36988    1.44  8.1e-16  bluestein
      3437  7.491            prime    raced          25473      33772    1.27  7.6e-16  bluestein
      3438  2.3^2.191        flat     raced          19750      28460    1.33  7.1e-16  
      3439  19.181           flat     raced          21212      59229    2.78  6.7e-16  
      3440  2^4.5.43         chain3   raced           8633      29886    3.38  5.4e-16  
      3441  3.31.37          chain3   raced          13755      50281    3.24  6.1e-16  
      3442  2.1721           prime    raced          25339      53282    1.43  6.5e-16  bluestein
      3443  11.313           flat     raced          24388      57768    2.36  6.8e-16  
      3444  2^2.3.7.41       chain3   raced           9124      34237    3.67  5.3e-16  
      3445  5.13.53          flat     raced          15732      39191    2.23  7.9e-16  
      3446  2.1723           prime    raced          25943      53384    2.01  7.3e-16  bluestein
      3447  3^2.383          prime    raced          25449      29385    1.11  5.4e-16  bluestein
      3448  2^3.431          flat     raced          24821      53072    2.10  7.0e-16  
      3449  3449             prime    raced          25547      36957    1.32  6.0e-16  bluestein
      3450  2.3.5^2.23       chain3   raced           8425      22727    2.41  6.1e-16  
      3451  7.17.29          chain3   raced           9912      41420    3.98  6.6e-16  
      3452  2^2.863          prime    raced          25017      36939    1.31  6.7e-16  bluestein
      3453  3.1151           prime    raced          25297      35861    1.39  7.6e-16  bluestein
      3454  2.11.157         flat     raced          17585      34273    1.90  5.9e-16  
      3455  5.691            prime    raced          25091      45618    1.48  6.8e-16  bluestein
      3456  2^7.3^3          chain3   raced           4760       9998    2.06  5.3e-16  
      3457  3457             prime    raced          18955      45732    1.97  5.8e-16  rader
      3458  2.7.13.19        flat     raced          11055      26530    2.35  6.3e-16  
      3459  3.1153           prime    raced          25276      34092    1.34  6.9e-16  bluestein
      3460  2^2.5.173        flat     raced          24111      50731    2.05  6.5e-16  
      3461  3461             prime    raced          24659      47169    1.49  6.7e-16  bluestein
      3462  2.3.577          flat     raced          27051      35590    1.25  8.7e-16  
      3463  3463             prime    raced          25256      54081    2.11  8.0e-16  bluestein
      3464  2^3.433          flat     raced          20060      45680    1.88  8.5e-16  flips differ 1.31x
      3465  3^2.5.7.11       chain3   raced           6246      11619    1.85  5.7e-16  
      3466  2.1733           prime    raced          24573      35479    1.37  7.5e-16  bluestein
      3467  3467             prime    raced          24681      45724    1.49  6.3e-16  bluestein
      3468  2^2.3.17^2       chain3   raced           7402      38705    4.53  6.9e-16  
      3469  3469             prime    raced          24983      37116    1.48  6.2e-16  bluestein
      3470  2.5.347          prime    raced          25114      43447    1.44  6.9e-16  bluestein
      3471  3.13.89          flat     raced          15539      63779    3.50  6.1e-16  
      3472  2^4.7.31         chain3   raced           8197      30372    3.63  5.4e-16  
      3473  23.151           flat     raced          25786      48725    1.80  7.1e-16  
      3474  2.3^2.193        flat     raced          17833      52620    2.58  6.0e-16  
      3475  5^2.139          flat     raced          24166      40889    1.58  9.2e-16  
      3476  2^2.11.79        flat     raced          15915      41988    1.96  8.2e-16  flips differ 1.39x
      3477  3.19.61          flat     raced          16255      56923    3.39  5.7e-16  
      3478  2.37.47          flat     raced          16936      58041    3.38  6.8e-16  
      3479  7^2.71           flat     raced          14814      53621    3.60  6.3e-16  
      3480  2^3.3.5.29       chain3   raced           9762      25956    2.60  5.2e-16  
      3481  59^2             prime    raced          24758      65566    2.62  6.7e-16  bluestein
      3482  2.1741           prime    raced          24604      34618    1.39  4.9e-16  bluestein
      3483  3^4.43           chain3   raced          11366      32446    2.39  6.6e-16  
      3484  2^2.13.67        flat     raced          15220      45977    2.76  6.8e-16  
      3485  5.17.41          chain3   raced          11776      46473    3.92  6.8e-16  
      3486  2.3.7.83         prime    raced          24785      51328    2.04  5.5e-16  bluestein
      3487  11.317           prime    raced          25177      34578    1.31  7.1e-16  bluestein
      3488  2^5.109          flat     raced          19940      46470    2.14  7.1e-16  
      3489  3.1163           prime    raced          24798      34734    1.30  6.1e-16  bluestein
      3490  2.5.349          prime    raced          24963      52008    1.71  6.4e-16  bluestein
      3491  3491             prime    raced          24928      41698    1.47  6.7e-16  bluestein
      3492  2^2.3^2.97       flat     raced          15388      34965    2.12  5.7e-16  
      3493  7.499            prime    raced          25268      36469    1.30  5.6e-16  bluestein
      3494  2.1747           prime    raced          25538      48343    1.29  5.7e-16  bluestein
      3495  3.5.233          flat     raced          24085      36285    1.47  7.4e-16  
      3496  2^3.19.23        chain3   raced           9119      40716    3.65  5.1e-16  
      3497  13.269           prime    raced          25790      46923    1.60  8.1e-16  bluestein
      3498  2.3.11.53        flat     raced          16542      41051    2.33  6.4e-16  
      3499  3499             prime    raced          26272      54708    2.03  6.0e-16  bluestein
      3500  2^2.5^3.7        flat     raced           9525       6850    0.69  3.9e-16  
      3501  3^2.389          prime    raced          25912      39760    1.49  6.4e-16  bluestein
      3502  2.17.103         flat     raced          18794      84619    4.31  8.1e-16  
      3503  31.113           flat     raced          17677      72009    3.94  6.0e-16  
      3504  2^4.3.73         flat     raced          13174      44598    3.29  5.8e-16  
      3505  5.701            prime    raced          27311      56007    1.96  6.1e-16  bluestein
      3506  2.1753           prime    raced          25580      37006    1.29  7.6e-16  bluestein
      3507  3.7.167          prime    raced          25763      56126    2.02  6.3e-16  bluestein
      3508  2^2.877          prime    raced          25076      59746    2.22  5.8e-16  bluestein
      3509  11^2.29          chain3   raced          10092      33653    3.27  5.3e-16  
      3510  2.3^3.5.13       chain3   raced           6895       9843    1.35  6.6e-16  
      3511  3511             prime    raced          25003      50972    1.84  8.3e-16  bluestein
      3512  2^3.439          prime    raced          24566      45776    1.52  8.1e-16  bluestein
      3513  3.1171           prime    raced          24731      34296    1.38  6.2e-16  bluestein
      3514  2.7.251          flat     raced          22994      30726    1.15  6.5e-16  
      3515  5.19.37          chain3   raced          14297      46023    3.17  4.7e-16  
      3516  2^2.3.293        prime    raced          24332      35383    1.37  7.2e-16  bluestein
      3517  3517             prime    raced          25090      45649    1.41  5.9e-16  bluestein
      3518  2.1759           prime    raced          25317      38363    1.39  5.2e-16  bluestein
      3519  3^2.17.23        chain3   raced           9192      40996    4.43  6.1e-16  
      3520  2^6.5.11         chain3   raced           5212       8048    1.51  4.5e-16  
      3521  7.503            prime    raced          24352      32372    1.31  5.3e-16  bluestein
      3522  2.3.587          prime    raced          24746      36283    1.46  6.1e-16  bluestein
      3523  13.271           flat     raced          28944      58704    2.01  9.2e-16  
      3524  2^2.881          flat     raced          27005      59625    2.01  6.4e-16  
      3525  3.5^2.47         chain3   raced          12280      35736    2.87  6.1e-16  
      3526  2.41.43          flat     raced          18535      58844    3.17  5.1e-16  
      3527  3527             prime    raced          27129      56748    1.47  7.9e-16  bluestein
      3528  2^3.3^2.7^2      chain3   raced           6546      12361    1.23  4.9e-16  flips differ 1.53x
      3529  3529             prime    raced          24468      36839    1.49  6.7e-16  bluestein
      3530  2.5.353          flat     raced          20965      44048    1.71  6.2e-16  
      3531  3.11.107         prime    raced          24892      70396    2.72  6.7e-16  bluestein
      3532  2^2.883          flat     raced          42972      46331    0.79  8.7e-16  
      3533  3533             prime    raced          25375      59126    1.86  7.1e-16  flips differ 1.27x bluestein
      3534  2.3.19.31        chain3   raced          10839      46067    3.71  5.4e-16  
      3535  5.7.101          flat     raced          20771      37589    1.79  5.8e-16  
      3536  2^4.13.17        chain3   raced           7669      25263    3.27  6.0e-16  
      3537  3^3.131          flat     raced          18534      45781    2.45  8.0e-16  
      3538  2.29.61          flat     raced          18719      63301    3.35  5.7e-16  
      3539  3539             prime    raced          25332      37070    1.39  5.8e-16  bluestein
      3540  2^2.3.5.59       flat     raced          23569      42708    1.80  5.6e-16  
      3541  3541             prime    raced          25245      54452    2.12  6.5e-16  bluestein
      3542  2.7.11.23        flat     raced          11635      28439    2.42  6.4e-16  
      3543  3.1181           prime    raced          24612      34088    1.37  7.1e-16  bluestein
      3544  2^3.443          flat     raced          22994      53324    2.30  1.0e-15  
      3545  5.709            prime    raced          24357      46065    1.48  7.9e-16  bluestein
      3546  2.3^2.197        flat     raced          17201      33620    1.92  6.3e-16  
      3547  3547             prime    raced          24210      56291    2.16  8.8e-16  bluestein
      3548  2^2.887          prime    raced          24948      46788    1.56  5.3e-16  bluestein
      3549  3.7.13^2         chain3   raced           6885      15502    2.22  7.5e-16  
      3550  2.5^2.71         flat     raced          17180      55082    3.14  6.0e-16  
      3551  53.67            prime    raced          24353      74551    3.04  5.9e-16  bluestein
      3552  2^5.3.37         chain3   raced           8268      33068    3.84  4.5e-16  
      3553  11.17.19         chain3   raced           9041      41452    4.26  6.3e-16  
      3554  2.1777           prime    raced          24990      44926    1.50  5.9e-16  bluestein
      3555  3^2.5.79         flat     raced          17013      38409    2.25  8.1e-16  
      3556  2^2.7.127        flat     raced          21229      32572    1.48  7.1e-16  
      3557  3557             prime    raced          25012      46766    1.57  5.3e-16  bluestein
      3558  2.3.593          prime    raced          24940      35910    1.41  7.2e-16  bluestein
      3559  3559             prime    raced          24450      37074    1.49  5.7e-16  bluestein
      3560  2^3.5.89         flat     raced          13426      38827    2.78  6.3e-16  
      3561  3.1187           prime    raced          25047      34529    1.36  6.2e-16  bluestein
      3562  2.13.137         flat     raced          17604      43739    2.39  6.9e-16  
      3563  7.509            prime    raced          25462      32464    1.13  7.7e-16  bluestein
      3564  2^2.3^4.11       chain3   raced           5895      14333    2.37  6.9e-16  
      3565  5.23.31          chain3   raced          11580      46157    3.61  7.1e-16  
      3566  2.1783           prime    raced          24402      53622    2.15  7.0e-16  bluestein
      3567  3.29.41          chain3   raced          13733      52601    3.40  6.1e-16  
      3568  2^4.223          flat     raced          22294      42483    1.90  5.8e-16  
      3569  43.83            prime    raced          24777      79626    3.09  5.5e-16  bluestein
      3570  2.3.5.7.17       chain3   raced           7242      22232    3.03  4.6e-16  
      3571  3571             prime    raced          25341      47170    1.45  7.8e-16  bluestein
      3572  2^2.19.47        chain3   raced          13739      51229    3.01  5.6e-16  
      3573  3^2.397          prime    raced          26243      34889    1.26  5.4e-16  bluestein
      3574  2.1787           prime    raced          25931      35518    1.33  6.8e-16  bluestein
      3575  5^2.11.13        chain3   raced           6557      15408    2.21  5.9e-16  
      3576  2^3.3.149        flat     raced          20372      45370    2.14  6.5e-16  
      3577  7^2.73           flat     raced          14853      41428    2.75  6.0e-16  
      3578  2.1789           prime    raced          25226      35107    1.38  7.2e-16  bluestein
      3579  3.1193           prime    raced          24470      35389    1.37  6.4e-16  bluestein
      3580  2^2.5.179        prime    raced          25059      51449    2.04  5.3e-16  bluestein
      3581  3581             prime    raced          25614      54393    2.10  7.2e-16  bluestein
      3582  2.3^2.199        flat     raced          19589      33658    1.68  7.6e-16  
      3583  3583             prime    raced          25092      37006    1.45  5.9e-16  bluestein
      3584  2^9.7            ztt      raced           4167       8413    1.47  3.2e-16  
      3585  3.5.239          flat     raced          22898      37225    1.50  6.4e-16  
      3586  2.11.163         flat     raced          21818      46232    1.93  6.4e-16  
      3587  17.211           flat     raced          24743      75857    3.00  6.8e-16  
      3588  2^2.3.13.23      chain3   raced           7997      28694    3.19  8.4e-16  
      3589  37.97            flat     raced          18171      59706    3.26  6.2e-16  
      3590  2.5.359          prime    raced          27043      51110    1.81  5.8e-16  bluestein
      3591  3^3.7.19         chain3   raced           8267      24479    2.78  4.4e-16  
      3592  2^3.449          flat     raced          18896      61275    3.23  5.6e-16  
      3593  3593             prime    raced          29211      46644    1.29  5.8e-16  bluestein
      3594  2.3.599          prime    raced          27538      35518    1.27  5.4e-16  bluestein
      3595  5.719            prime    raced          26768      54194    1.97  5.3e-16  bluestein
      3596  2^2.29.31        chain3   raced          12946      48950    3.69  6.2e-16  
      3597  3.11.109         flat     raced          21073      47051    1.86  7.7e-16  
      3598  2.7.257          flat     raced          21082      34637    1.63  5.0e-16  
      3599  59.61            prime    raced          26926      73472    2.71  6.3e-16  bluestein
      3600  2^4.3^2.5^2      chain3   raced           5427      10126    1.81  5.1e-16  
      3601  13.277           prime    raced          26748      48981    1.50  6.4e-16  bluestein
      3602  2.1801           prime    raced          27827      46665    1.63  6.1e-16  bluestein
      3603  3.1201           prime    raced          26927      36825    1.36  5.7e-16  bluestein
      3604  2^2.17.53        flat     raced          16222      54545    3.32  7.1e-16  
      3605  5.7.103          flat     raced          19593      61452    3.10  8.3e-16  
      3606  2.3.601          prime    raced          27423      36572    1.33  5.2e-16  bluestein
      3607  3607             prime    raced          27908      49596    1.76  6.0e-16  bluestein
      3608  2^3.11.41        chain3   raced           9995      34864    3.20  5.1e-16  
      3609  3^2.401          flat     raced          27438      35515    1.18  5.8e-16  
      3610  2.5.19^2         chain3   raced           9263      39293    3.55  6.0e-16  
      3611  23.157           flat     raced          21206      50365    2.37  7.4e-16  
      3612  2^2.3.7.43       chain3   raced           9892      37342    3.37  6.8e-16  
      3613  3613             prime    raced          29262      50736    1.59  6.1e-16  bluestein
      3614  2.13.139         flat     raced          20420      43755    1.81  7.9e-16  
      3615  3.5.241          flat     raced          19408      36273    1.78  5.4e-16  
      3616  2^5.113          flat     raced          14845      50458    3.36  6.0e-16  
      3617  3617             prime    raced          28165      50205    1.66  5.4e-16  bluestein
      3618  2.3^3.67         flat     raced          16759      44860    2.65  5.2e-16  
      3619  7.11.47          chain3   raced          14696      38670    2.29  4.9e-16  
      3620  2^2.5.181        flat     raced          19465      46998    2.40  5.7e-16  
      3621  3.17.71          flat     raced          16505      59411    3.59  6.4e-16  
      3622  2.1811           prime    raced          28351      61847    2.16  7.4e-16  bluestein
      3623  3623             prime    raced          27940      58999    1.95  7.3e-16  bluestein
      3624  2^3.3.151        flat     raced          19507      36675    1.74  7.2e-16  
      3625  5^3.29           chain3   raced          11621      28690    2.45  5.2e-16  
      3626  2.7^2.37         flat     raced          13732      31032    2.14  4.9e-16  
      3627  3^2.13.31        chain3   raced           9875      31482    3.13  5.1e-16  
      3628  2^2.907          prime    raced          27275      63456    2.31  6.2e-16  bluestein
      3629  19.191           flat     raced          23554      46170    1.80  7.0e-16  
      3630  2.3.5.11^2       flat     raced          10530      15338    1.34  4.9e-16  
      3631  3631             prime    raced          26909      50433    1.82  7.2e-16  bluestein
      3632  2^4.227          prime    raced          27021      35743    1.31  5.4e-16  bluestein
      3633  3.7.173          prime    raced          26982      54834    2.03  6.6e-16  bluestein
      3634  2.23.79          flat     raced          17943      64915    3.21  8.0e-16  
      3635  5.727            prime    raced          27385      55348    1.95  8.3e-16  bluestein
      3636  2^2.3^2.101      flat     raced          19456      37736    1.94  6.6e-16  
      3637  3637             prime    raced          37373      54344    0.89  7.0e-16  flips differ 1.69x bluestein
      3638  2.17.107         prime    raced          33954      81363    1.85  5.8e-16  flips differ 1.31x bluestein
      3639  3.1213           prime    raced          55360      59323    0.85  6.3e-16  flips differ 1.37x bluestein
      3640  2^3.5.7.13       flat     raced           8589      11571    1.05  5.9e-16  
      3641  11.331           flat     raced          24868      58082    0.81  8.3e-16  flips differ 2.78x
      3642  2.3.607          prime    raced          27580      54394    1.36  6.3e-16  flips differ 1.46x bluestein
      3643  3643             prime    raced          28129      50488    1.77  6.6e-16  bluestein
      3644  2^2.911          flat     raced          29243      64539    2.17  7.6e-16  
      3645  3^6.5            chain3   raced           6625      12305    1.84  4.9e-16  
      3646  2.1823           prime    raced          27097      63288    2.25  6.8e-16  bluestein
      3647  7.521            prime    raced          26848      50913    1.50  6.5e-16  bluestein
      3648  2^6.3.19         chain3   raced           7006      22814    3.21  4.2e-16  
      3649  41.89            flat     raced          17534      67295    3.80  6.3e-16  
      3650  2.5^2.73         flat     raced          17208      46867    2.69  6.3e-16  
      3651  3.1217           prime    raced          29385      44126    1.48  5.9e-16  bluestein
      3652  2^2.11.83        flat     raced          24268      57957    2.31  6.1e-16  
      3653  13.281           flat     raced          20805      58584    2.76  6.9e-16  
      3654  2.3^2.7.29       chain3   raced           9415      27601    2.71  6.1e-16  
      3655  5.17.43          chain3   raced          13214      50926    3.76  5.6e-16  
      3656  2^3.457          flat     raced          22984      50333    2.11  6.8e-16  
      3657  3.23.53          flat     raced          18300      58747    3.03  7.8e-16  
      3658  2.31.59          flat     raced          27192      63828    2.32  7.5e-16  
      3659  3659             prime    raced          34133      59865    1.43  6.0e-16  bluestein
      3660  2^2.3.5.61       flat     raced          17677      43597    2.38  6.5e-16  
      3661  7.523            prime    raced          27420      51170    1.54  6.2e-16  bluestein
      3662  2.1831           prime    raced          28032      61753    1.94  7.3e-16  bluestein
      3663  3^2.11.37        chain3   raced          11832      34001    2.66  5.0e-16  
      3664  2^4.229          flat     raced          21238      37595    1.74  7.1e-16  
      3665  5.733            prime    raced          28459      49642    1.44  5.7e-16  bluestein
      3666  2.3.13.47        chain3   raced          13995      38557    2.67  6.5e-16  
      3667  19.193           flat     raced          21275      51656    2.31  6.4e-16  
      3668  2^2.7.131        flat     raced          17868      48394    2.53  7.0e-16  
      3669  3.1223           prime    raced          28145      45341    1.58  6.3e-16  bluestein
      3670  2.5.367          prime    raced          27049      43342    1.53  5.7e-16  bluestein
      3671  3671             prime    raced          29424      59854    2.01  4.7e-16  bluestein
      3672  2^3.3^3.17       chain3   raced           6951      24465    3.10  6.7e-16  
      3673  3673             prime    raced          29128      52239    1.48  6.3e-16  bluestein
      3674  2.11.167         prime    raced          28115      44777    1.53  6.8e-16  bluestein
      3675  3.5^2.7^2        chain3   raced           7102      12029    1.66  5.0e-16  
      3676  2^2.919          prime    raced          26627      63839    2.37  8.0e-16  bluestein
      3677  3677             prime    raced          26808      60480    2.25  6.9e-16  bluestein
      3678  2.3.613          prime    raced          26912      49250    1.78  6.0e-16  bluestein
      3679  13.283           prime    raced          26806      50743    1.62  6.3e-16  bluestein
      3680  2^5.5.23         chain3   raced           7927      27823    3.43  5.2e-16  
      3681  3^2.409          flat     raced          30787      59419    1.87  7.4e-16  
      3682  2.7.263          prime    raced          27043      45309    1.65  6.7e-16  bluestein
      3683  29.127           flat     raced          26469      55335    2.07  9.3e-16  
      3684  2^2.3.307        flat     raced          23427      61396    2.54  8.4e-16  
      3685  5.11.67          flat     raced          18096      48672    2.68  5.0e-16  
      3686  2.19.97          flat     raced          17297      54136    3.01  6.8e-16  
      3687  3.1229           prime    raced          26915      43804    1.62  7.1e-16  bluestein
      3688  2^3.461          flat     raced          23674      35085    1.44  6.2e-16  
      3689  7.17.31          chain3   raced          11156      45217    4.00  6.3e-16  
      3690  2.3^2.5.41       chain3   raced          12891      31928    2.22  5.6e-16  
      3691  3691             prime    raced          27681      63500    2.21  4.8e-16  bluestein
      3692  2^2.13.71        flat     raced          15631      48286    3.02  6.0e-16  
      3693  3.1231           prime    raced          27413      44561    1.62  6.3e-16  bluestein
      3694  2.1847           prime    raced          26768      51085    1.84  7.1e-16  bluestein
      3695  5.739            prime    raced          28809      49595    0.79  5.3e-16  flips differ 1.79x bluestein
      3696  2^4.3.7.11       chain3   raced           5839      14551    1.63  4.2e-16  flips differ 1.53x
      3697  3697             prime    raced          35755      59888    1.22  7.3e-16  rader
      3698  2.43^2           flat     raced          19278      73126    3.42  7.3e-16  
      3699  3^3.137          flat     raced          19269      42333    2.06  7.9e-16  
      3700  2^2.5^2.37       chain3   raced          10407      30018    2.86  5.3e-16  
      3701  3701             prime    raced          27181      60141    2.16  6.4e-16  bluestein
      3702  2.3.617          prime    raced          28449      55481    1.30  6.6e-16  flips differ 1.57x bluestein
      3703  7.23^2           flat     raced          15844      45300    2.72  6.4e-16  
      3704  2^3.463          flat     raced          26684      36547    1.30  8.0e-16  
      3705  3.5.13.19        chain3   raced           8375      28410    3.33  6.2e-16  
      3706  2.17.109         flat     raced          23191      61765    1.90  6.7e-16  flips differ 1.43x
      3707  11.337           flat     raced          27805      59276    1.84  7.4e-16  
      3708  2^2.3^2.103      flat     raced          18282      65339    3.24  8.4e-16  
      3709  3709             prime    raced          27133      51672    1.56  5.5e-16  bluestein
      3710  2.5.7.53         flat     raced          17622      38901    2.20  6.1e-16  
      3711  3.1237           prime    raced          27308      44421    1.61  6.2e-16  bluestein
      3712  2^7.29           chain3   raced           7335      28858    3.23  5.7e-16  
      3713  47.79            flat     raced          20142      68923    3.41  6.4e-16  
      3714  2.3.619          prime    raced          27660      51711    1.43  8.1e-16  flips differ 1.26x bluestein
      3715  5.743            prime    raced          27212      63749    2.00  6.6e-16  bluestein
      3716  2^2.929          prime    raced          28179      67414    2.25  7.2e-16  bluestein
      3717  3^2.7.59         flat     raced          24375      42158    1.61  5.6e-16  
      3718  2.11.13^2        flat     raced          10912      18481    1.60  8.5e-16  
      3719  3719             prime    raced          27581      63072    2.15  6.0e-16  bluestein
      3720  2^3.3.5.31       chain3   raced           9141      29744    3.04  6.9e-16  
      3721  61^2             prime    raced          26753      80457    2.88  6.0e-16  bluestein
      3722  2.1861           prime    raced          28574      53337    1.66  7.6e-16  bluestein
      3723  3.17.73          flat     raced          16143      64072    3.82  6.7e-16  
      3724  2^2.7^2.19       flat     raced          12689      28035    2.19  4.8e-16  
      3725  5^2.149          flat     raced          25684      44103    1.66  6.9e-16  
      3726  2.3^4.23         chain3   raced           8390      25867    3.04  5.9e-16  
      3727  3727             prime    raced          28771      52772    1.42  9.1e-16  bluestein
      3728  2^4.233          flat     raced          23877      37037    1.50  7.6e-16  
      3729  3.11.113         flat     raced          17205      57469    3.15  5.0e-16  
      3730  2.5.373          prime    raced          26578      43424    1.58  8.5e-16  bluestein
      3731  7.13.41          chain3   raced          12700      36814    2.76  6.7e-16  
      3732  2^2.3.311        flat     raced          26778      61903    2.30  8.7e-16  
      3733  3733             prime    raced          29540      52321    1.40  6.2e-16  bluestein
      3734  2.1867           prime    raced          26573      50343    1.88  7.9e-16  bluestein
      3735  3^2.5.83         flat     raced          26857      55331    1.86  6.7e-16  
      3736  2^3.467          prime    raced          27270      35151    1.28  6.5e-16  bluestein
      3737  37.101           flat     raced          23561      65388    2.72  7.1e-16  
      3738  2.3.7.89         flat     raced          16384      72983    4.27  6.9e-16  
      3739  3739             prime    raced          26831      60315    2.00  8.7e-16  bluestein
      3740  2^2.5.11.17      flat     raced          10326      26423    2.31  6.4e-16  
      3741  3.29.43          chain3   raced          16224      56856    3.43  5.4e-16  
      3742  2.1871           prime    raced          26550      51345    1.91  7.6e-16  bluestein
      3743  19.197           flat     raced          21090      53593    2.44  5.8e-16  
      3744  2^5.3^2.13       chain3   raced           5436      11705    1.72  5.7e-16  
      3745  5.7.107          prime    raced          27216      61618    2.26  6.5e-16  bluestein
      3746  2.1873           prime    raced          27093      50038    1.85  6.1e-16  bluestein
      3747  3.1249           prime    raced          29320      43838    1.47  6.0e-16  bluestein
      3748  2^2.937          flat     raced          27924      64064    2.15  8.0e-16  
      3749  23.163           flat     raced          24823      63695    2.21  7.1e-16  
      3750  2.3.5^4          chain3   raced           6467       9631    1.45  4.6e-16  
      3751  11^2.31          chain3   raced          12778      35639    2.31  5.2e-16  
      3752  2^3.7.67         flat     raced          15260      45798    2.78  6.2e-16  
      3753  3^3.139          flat     raced          26259      41779    1.59  7.1e-16  
      3754  2.1877           prime    raced          26490      37587    1.36  6.5e-16  bluestein
      3755  5.751            prime    raced          26675      37041    1.29  5.5e-16  bluestein
      3756  2^2.3.313        flat     raced          20630      60723    2.81  7.5e-16  
      3757  13.17^2          chain3   raced           9033      43212    4.71  6.5e-16  
      3758  2.1879           prime    raced          26895      39170    1.36  7.0e-16  bluestein
      3759  3.7.179          prime    raced          26615      55187    1.95  7.0e-16  bluestein
      3760  2^4.5.47         chain3   raced          10070      36843    3.48  6.1e-16  
      3761  3761             prime    raced          26816      38690    1.43  7.2e-16  bluestein
      3762  2.3^2.11.19      flat     raced          13275      28546    2.10  5.6e-16  
      3763  53.71            prime    raced          26893      80667    2.65  6.8e-16  bluestein
      3764  2^2.941          prime    raced          26790      36107    1.34  5.7e-16  bluestein
      3765  3.5.251          flat     raced          27208      32028    1.08  7.3e-16  
      3766  2.7.269          prime    raced          28158      44995    1.59  5.5e-16  bluestein
      3767  3767             prime    raced          27007      38704    1.43  6.6e-16  bluestein
      3768  2^3.3.157        flat     raced          17067      34134    1.81  7.7e-16  
      3769  3769             prime    raced          27124      38661    1.39  6.1e-16  bluestein
      3770  2.5.13.29        chain3   raced           9671      31970    2.67  6.6e-16  flips differ 1.26x
      3771  3^2.419          prime    raced          26827      50460    1.54  6.7e-16  bluestein
      3772  2^2.23.41        flat     raced          13455      54283    3.44  6.5e-16  
      3773  7^3.11           flat     raced           9863      12991    1.29  4.4e-16  
      3774  2.3.17.37        chain3   raced          11802      48588    4.08  5.6e-16  
      3775  5^2.151          flat     raced          25426      34400    1.33  8.1e-16  
      3776  2^6.59           flat     raced          23812      40794    1.69  6.6e-16  
      3777  3.1259           prime    raced          27077      37916    1.39  6.3e-16  bluestein
      3778  2.1889           prime    raced          26777      37046    1.34  6.1e-16  bluestein
      3779  3779             prime    raced          29296      39216    1.33  5.9e-16  bluestein
      3780  2^2.3^3.5.7      chain3   raced           6517      10707    1.27  4.8e-16  
      3781  19.199           flat     raced          23734      52330    2.13  6.9e-16  
      3782  2.31.61          flat     raced          18819      67901    2.99  6.2e-16  
      3783  3.13.97          flat     raced          18367      41772    2.15  7.0e-16  
      3784  2^3.11.43        chain3   raced          10867      41898    3.49  4.9e-16  
      3785  5.757            prime    raced          26813      36989    1.17  6.1e-16  bluestein
      3786  2.3.631          prime    raced          27292      33818    1.17  6.5e-16  bluestein
      3787  7.541            prime    raced          26786      51029    1.72  6.4e-16  bluestein
      3788  2^2.947          prime    raced          27112      37753    1.33  6.0e-16  bluestein
      3789  3^2.421          flat     raced          32975      51504    1.28  7.5e-16  
      3790  2.5.379          flat     raced          28609      31754    1.03  6.0e-16  
      3791  17.223           prime    raced          28110      61430    2.15  5.2e-16  bluestein
      3792  2^4.3.79         flat     raced          15687      41928    2.54  6.9e-16  
      3793  3793             prime    raced          29041      40093    1.33  6.1e-16  bluestein
      3794  2.7.271          flat     raced          28395      43672    1.54  5.8e-16  
      3795  3.5.11.23        chain3   raced           8795      31492    3.54  6.1e-16  
      3796  2^2.13.73        flat     raced          15339      46613    3.01  7.9e-16  
      3797  3797             prime    raced          27603      39023    1.38  5.7e-16  bluestein
      3798  2.3^2.211        flat     raced          24080      61042    2.52  7.3e-16  
      3799  29.131           flat     raced          19268      69955    3.55  8.0e-16  
      3800  2^3.5^2.19       chain3   raced           7895      22714    2.85  4.8e-16  
      3801  3.7.181          flat     raced          22465      45284    1.79  6.3e-16  
      3802  2.1901           prime    raced          28483      37120    1.30  6.1e-16  bluestein
      3803  3803             prime    raced          26683      38680    1.36  8.1e-16  bluestein
      3804  2^2.3.317        prime    raced          27442      34368    1.12  6.2e-16  bluestein
      3805  5.761            prime    raced          27395      43781    1.41  6.4e-16  bluestein
      3806  2.11.173         flat     raced          26748      48202    1.64  6.5e-16  
      3807  3^4.47           chain3   raced          12939      39184    2.22  5.6e-16  flips differ 1.29x
      3808  2^5.7.17         chain3   raced           6899      28418    3.74  5.8e-16  
      3809  13.293           prime    raced          27714      43244    0.80  5.4e-16  flips differ 1.91x bluestein
      3810  2.3.5.127        flat     raced          21572      37960    1.53  7.5e-16  
      3811  37.103           flat     raced          21927      98155    3.38  8.1e-16  flips differ 1.30x
      3812  2^2.953          flat     raced          30747      36171    1.16  7.5e-16  
      3813  3.31.41          chain3   raced          18422      57503    1.29  7.3e-16  flips differ 2.45x
      3814  2.1907           prime    raced          27798      43556    1.29  6.4e-16  bluestein
      3815  5.7.109          flat     raced          22342      51608    2.23  6.7e-16  
      3816  2^3.3^2.53       flat     raced          14714      39276    2.64  6.7e-16  
      3817  11.347           prime    raced          27862      58735    1.45  6.2e-16  flips differ 1.46x bluestein
      3818  2.23.83          prime    raced          26649      76438    2.84  6.1e-16  bluestein
      3819  3.19.67          flat     raced          18399      70221    3.75  7.1e-16  
      3820  2^2.5.191        flat     raced          23630      30841    1.23  8.3e-16  
      3821  3821             prime    raced          29138      77393    1.33  6.5e-16  bluestein
      3822  2.3.7^2.13       flat     raced          12095      11579    0.91  5.6e-16  
      3823  3823             prime    raced          27394      50415    1.42  6.0e-16  bluestein
      3824  2^4.239          flat     raced          18102      36644    1.55  6.6e-16  flips differ 1.31x
      3825  3^2.5^2.17       chain3   raced           7952      25777    3.11  6.0e-16  
      3826  2.1913           prime    raced          27034      37094    1.36  6.5e-16  bluestein
      3827  43.89            flat     raced          19146     100569    2.12  6.6e-16  flips differ 2.44x
      3828  2^2.3.11.29      chain3   raced           9254      33152    3.48  6.3e-16  
      3829  7.547            prime    raced          28501      63778    2.17  6.8e-16  bluestein
      3830  2.5.383          prime    raced          27248      31913    0.93  7.0e-16  bluestein
      3831  3.1277           prime    raced          28023      38949    1.36  8.7e-16  bluestein
      3832  2^3.479          prime    raced          27877      35568    1.25  6.5e-16  bluestein
      3833  3833             prime    raced          27743      39150    1.38  6.3e-16  bluestein
      3834  2.3^3.71         flat     raced          17142      70729    3.26  7.2e-16  
      3835  5.13.59          prime    raced          27689      47461    1.49  5.7e-16  bluestein
      3836  2^2.7.137        flat     raced          18316      44318    2.33  5.8e-16  
      3837  3.1279           prime    raced          27391      39692    1.39  6.5e-16  bluestein
      3838  2.19.101         flat     raced          23487      62967    2.16  6.7e-16  
      3839  11.349           prime    raced          28522      50238    1.35  6.2e-16  bluestein
      3840  2^8.3.5          ztt      raced           5015      10113    1.95  4.1e-16  
      3841  23.167           prime    raced          27351      71864    2.39  6.4e-16  bluestein
      3842  2.17.113         flat     raced          16895      70418    4.01  6.7e-16  
      3843  3^2.7.61         flat     raced          15910      44701    2.77  5.5e-16  
      3844  2^2.31^2         chain3   raced          14949      53757    3.56  6.1e-16  
      3845  5.769            prime    raced          28898      36456    1.26  6.3e-16  bluestein
      3846  2.3.641          prime    raced          27238      37972    1.35  5.8e-16  bluestein
      3847  3847             prime    raced          27829      97909    3.23  6.2e-16  bluestein
      3848  2^3.13.37        chain3   raced          10895      36946    3.34  6.1e-16  
      3849  3.1283           prime    raced          27412      55995    1.55  5.3e-16  bluestein
      3850  2.5^2.7.11       flat     raced          12071      11863    0.73  5.2e-16  
      3851  3851             prime    raced          29666      62707    1.45  5.8e-16  flips differ 1.55x bluestein
      3852  2^2.3^2.107      prime    raced          28736      79078    2.62  6.9e-16  bluestein
      3853  3853             prime    raced          29625      70483    1.53  7.0e-16  bluestein
      3854  2.41.47          flat     raced          18763      67189    3.27  5.1e-16  
      3855  3.5.257          flat     raced          22873      38091    1.59  5.2e-16  
      3856  2^4.241          flat     raced          15389      36948    2.35  5.1e-16  
      3857  7.19.29          chain3   raced          13597      47236    3.39  6.5e-16  
      3858  2.3.643          prime    raced          27010      45986    1.67  6.5e-16  bluestein
      3859  17.227           prime    raced          28745      57120    1.67  4.7e-16  bluestein
      3860  2^2.5.193        flat     raced          19405      37033    1.86  5.6e-16  
      3861  3^3.11.13        chain3   raced           7270      16442    2.23  6.5e-16  
      3862  2.1931           prime    raced          27376      66300    1.73  6.5e-16  bluestein
      3863  3863             prime    raced          29215      53966    1.68  7.0e-16  bluestein
      3864  2^3.3.7.23       chain3   raced           9646      31296    3.13  5.0e-16  
      3865  5.773            prime    raced          27453      40499    1.43  6.3e-16  bluestein
      3866  2.1933           prime    raced          28396     114669    1.86  7.2e-16  flips differ 3.14x bluestein
      3867  3.1289           prime    raced          27134      71484    0.85  6.6e-16  flips differ 3.39x bluestein
      3868  2^2.967          prime    raced          27552      71597    1.47  6.6e-16  flips differ 1.29x bluestein
      3869  53.73            prime    raced          27517      80281    2.85  5.8e-16  bluestein
      3870  2.3^2.5.43       chain3   raced          15520      35117    2.25  6.9e-16  
      3871  7^2.79           flat     raced          18863      44318    2.28  7.4e-16  
      3872  2^5.11^2         chain3   raced           5753      14304    2.42  4.4e-16  
      3873  3.1291           prime    raced          27576      65110    2.34  6.4e-16  bluestein
      3874  2.13.149         flat     raced          25836      47940    1.84  7.0e-16  
      3875  5^3.31           chain3   raced          10832      31950    2.56  5.9e-16  
      3876  2^2.3.17.19      chain3   raced           9390      42948    4.23  5.7e-16  
      3877  3877             prime    raced          28817      64899    1.59  6.1e-16  bluestein
      3878  2.7.277          flat     raced          25803      52832    1.87  7.3e-16  
      3879  3^2.431          prime    raced          27494      51914    1.56  6.5e-16  bluestein
      3880  2^3.5.97         flat     raced          16216      37750    2.30  4.8e-16  
      3881  3881             prime    raced          27123      85565    2.99  7.2e-16  bluestein
      3882  2.3.647          prime    raced          26817      45237    1.64  6.4e-16  bluestein
      3883  11.353           flat     raced          24339      42097    1.65  6.4e-16  
      3884  2^2.971          prime    raced          28041      86574    3.08  7.1e-16  bluestein
      3885  3.5.7.37         chain3   raced          11241      34642    3.08  4.9e-16  
      3886  2.29.67          flat     raced          18560      71245    3.82  6.6e-16  
      3887  13^2.23          chain3   raced           9536      33712    3.11  8.1e-16  
      3888  2^4.3^5          chain3   raced           5995      17691    2.94  6.4e-16  
      3889  3889             prime    raced          29623      41671    1.34  5.7e-16  bluestein
      3890  2.5.389          prime    raced          27422      38633    1.37  5.8e-16  bluestein
      3891  3.1297           prime    raced          27599      61710    2.21  7.3e-16  bluestein
      3892  2^2.7.139        flat     raced          22973      43949    1.74  7.0e-16  
      3893  17.229           flat     raced          26189      56680    2.06  7.5e-16  
      3894  2.3.11.59        flat     raced          28512      46791    1.63  5.8e-16  
      3895  5.19.41          chain3   raced          16119      55475    3.08  5.8e-16  
      3896  2^3.487          flat     raced          32339      38824    1.18  7.0e-16  
      3897  3^2.433          flat     raced          29479      48299    1.48  7.6e-16  
      3898  2.1949           prime    raced          27531      43570    1.55  6.3e-16  bluestein
      3899  7.557            prime    raced          27548      62247    2.23  6.8e-16  bluestein
      3900  2^2.3.5^2.13     chain3   raced           6804       9893    1.41  7.3e-16  
      3901  47.83            prime    raced          28501      88882    3.07  5.8e-16  bluestein
      3902  2.1951           prime    raced          28322      44009    1.53  6.0e-16  bluestein
      3903  3.1301           prime    raced          28311      69335    2.43  7.9e-16  bluestein
      3904  2^6.61           flat     raced          13822      48119    3.46  4.2e-16  
      3905  5.11.71          flat     raced          17981      53273    2.88  6.6e-16  
      3906  2.3^2.7.31       chain3   raced          11440      32166    2.79  6.4e-16  
      3907  3907             prime    raced          27477      43675    1.53  8.6e-16  bluestein
      3908  2^2.977          prime    raced          27601      39708    1.41  6.3e-16  bluestein
      3909  3.1303           prime    raced          28291      82495    2.80  6.8e-16  bluestein
      3910  2.5.17.23        chain3   raced          11108      44243    3.86  6.7e-16  
      3911  3911             prime    raced          27647      59604    1.51  7.5e-16  bluestein
      3912  2^3.3.163        flat     raced          21192      46691    1.94  7.7e-16  
      3913  7.13.43          chain3   raced          14325      40076    2.74  5.4e-16  
      3914  2.19.103         flat     raced          20208     133101    4.33  6.6e-16  
      3915  3^3.5.29         chain3   raced          12158      30821    2.50  7.1e-16  
      3916  2^2.11.89        flat     raced          16367      55773    2.92  7.2e-16  
      3917  3917             prime    raced          27465      41670    1.48  7.4e-16  bluestein
      3918  2.3.653          prime    raced          27549      52359    1.86  6.2e-16  bluestein
      3919  3919             prime    raced          30155      44071    1.40  5.5e-16  bluestein
      3920  2^4.5.7^2        flat     raced           8828      12970    1.43  3.8e-16  
      3921  3.1307           prime    raced          27461      60722    1.79  4.6e-16  bluestein
      3922  2.37.53          flat     raced          20251      67036    3.30  8.1e-16  
      3923  3923             prime    raced          27857      52583    1.50  5.6e-16  bluestein
      3924  2^2.3^2.109      flat     raced          21323      48900    2.08  8.8e-16  
      3925  5^2.157          flat     raced          23220      37549    1.44  7.4e-16  
      3926  2.13.151         flat     raced          23983      40675    1.38  6.4e-16  flips differ 1.26x
      3927  3.7.11.17        chain3   raced           9027      29565    3.22  5.4e-16  
      3928  2^3.491          flat     raced          20077      39137    1.91  5.9e-16  
      3929  3929             prime    raced          29565      41713    1.35  5.6e-16  bluestein
      3930  2.3.5.131        flat     raced          18444      49747    2.59  8.2e-16  
      3931  3931             prime    raced          27652      53724    1.51  7.5e-16  flips differ 1.57x bluestein
      3932  2^2.983          prime    raced          27137      38938    1.25  5.2e-16  bluestein
      3933  3^2.19.23        chain3   raced          12821      45148    3.50  6.9e-16  
      3934  2.7.281          flat     raced          20401      54067    2.54  5.9e-16  
      3935  5.787            prime    raced          27251      40552    1.10  5.0e-16  flips differ 1.36x bluestein
      3936  2^5.3.41         chain3   raced           9698      33875    3.45  4.0e-16  
      3937  31.127           flat     raced          24085      60565    2.06  7.9e-16  
      3938  2.11.179         prime    raced          27149      52542    1.65  5.8e-16  bluestein
      3939  3.13.101         flat     raced          26667      46023    1.55  6.9e-16  
      3940  2^2.5.197        flat     raced          19639      36628    1.77  6.3e-16  
      3941  7.563            prime    raced          27452      64479    2.34  8.3e-16  bluestein
      3942  2.3^3.73         flat     raced          17379      51399    2.92  5.7e-16  
      3943  3943             prime    raced          30075      41603    1.36  6.1e-16  bluestein
      3944  2^3.17.29        chain3   raced          10506      48509    4.20  6.4e-16  
      3945  3.5.263          prime    raced          27604      49671    1.74  7.4e-16  bluestein
      3946  2.1973           prime    raced          27457      43264    1.55  5.9e-16  bluestein
      3947  3947             prime    raced          27305      41790    1.51  6.7e-16  bluestein
      3948  2^2.3.7.47       chain3   raced          11803      37742    3.18  5.2e-16  
      3949  11.359           prime    raced          27237      43506    1.53  6.0e-16  bluestein
      3950  2.5^2.79         flat     raced          20666      42641    1.97  1.0e-15  
      3951  3^2.439          prime    raced          27206      51046    1.53  7.5e-16  bluestein
      3952  2^4.13.19        chain3   raced           7981      28699    3.31  5.8e-16  
      3953  59.67            prime    raced          27303      82457    3.01  6.8e-16  bluestein
      3954  2.3.659          prime    raced          27416      54796    1.15  6.8e-16  flips differ 1.87x bluestein
      3955  5.7.113          flat     raced          20537      56469    1.51  5.3e-16  flips differ 1.83x
      3956  2^2.23.43        chain3   raced          17084      56923    3.26  6.0e-16  
      3957  3.1319           prime    raced          27324      67347    2.08  7.6e-16  bluestein
      3958  2.1979           prime    raced          27631      43930    1.03  7.5e-16  flips differ 1.60x bluestein
      3959  37.107           prime    raced          27059      91492    3.30  5.8e-16  bluestein
      3960  2^3.3^2.5.11     flat     raced           8993      14966    1.65  4.0e-16  
      3961  17.233           prime    raced          27489      56540    1.97  6.6e-16  bluestein
      3962  2.7.283          prime    raced          27452      70327    2.06  7.9e-16  bluestein
      3963  3.1321           prime    raced          27377      71089    2.55  6.0e-16  bluestein
      3964  2^2.991          prime    raced          27569      39255    1.30  5.6e-16  bluestein
      3965  5.13.61          flat     raced          18439      49760    2.69  6.2e-16  
      3966  2.3.661          prime    raced          27352      53635    1.83  8.1e-16  bluestein
      3967  3967             prime    raced          27705      42205    1.47  6.5e-16  bluestein
      3968  2^7.31           chain3   raced           8591      29186    3.38  5.2e-16  
      3969  3^4.7^2          chain3   raced           7707      13707    1.74  6.4e-16  
      3970  2.5.397          flat     raced          25540      41073    1.60  7.8e-16  
      3971  11.19^2          chain3   raced          10995      46922    3.75  4.6e-16  
      3972  2^2.3.331        flat     raced          22594      58483    2.45  5.8e-16  
      3973  29.137           flat     raced          20448      67425    2.97  9.0e-16  
      3974  2.1987           prime    raced          27777      43819    1.50  6.2e-16  bluestein
      3975  3.5^2.53         flat     raced          17618      46087    2.59  5.5e-16  
      3976  2^3.7.71         flat     raced          15208      45272    2.88  5.5e-16  
      3977  41.97            flat     raced          20205      70057    3.19  6.1e-16  
      3978  2.3^2.13.17      flat     raced          12536      28981    2.15  6.8e-16  
      3979  23.173           prime    raced          27032      79436    2.84  5.7e-16  bluestein
      3980  2^2.5.199        flat     raced          21682      37331    1.46  6.8e-16  
      3981  3.1327           prime    raced          27676      49072    1.72  7.6e-16  bluestein
      3982  2.11.181         flat     raced          20687      56185    2.58  6.6e-16  
      3983  7.569            prime    raced          28169      73847    2.19  5.5e-16  bluestein
      3984  2^4.3.83         flat     raced          24955      57291    2.22  5.9e-16  
      3985  5.797            prime    raced          28058      39052    1.37  6.2e-16  bluestein
      3986  2.1993           prime    raced          27796      43006    1.42  6.7e-16  bluestein
      3987  3^2.443          prime    raced          27357      53582    1.64  8.5e-16  bluestein
      3988  2^2.997          prime    raced          27382      40511    1.46  6.8e-16  bluestein
      3989  3989             prime    raced          27267      41665    1.50  7.0e-16  bluestein
      3990  2.3.5.7.19       chain3   raced           9556      32343    2.67  6.3e-16  
      3991  13.307           prime    raced          27411      69824    2.50  7.0e-16  bluestein
      3992  2^3.499          prime    raced          27267      38780    1.41  7.0e-16  bluestein
      3993  3.11^3           flat     raced          11321      20659    1.79  5.7e-16  
      3994  2.1997           prime    raced          27912      43637    1.55  6.8e-16  bluestein
      3995  5.17.47          chain3   raced          15157      56801    3.62  7.1e-16  
      3996  2^2.3^3.37       chain3   raced           9986      37878    3.76  5.1e-16  
      3997  7.571            prime    raced          28149      59433    1.45  7.1e-16  bluestein
      3998  2.1999           prime    raced          27278      44091    1.56  6.1e-16  bluestein
      3999  3.31.43          flat     raced          18597      61351    3.21  6.0e-16  
      4000  2^5.5^3          chain3   raced           6066       8467    1.03  4.3e-16  
      4001  4001             prime    raced          28033      86674    3.08  5.8e-16  bluestein
      4002  2.3.23.29        flat     raced          17681      53485    2.56  5.3e-16  
      4003  4003             prime    raced          28278      97894    3.13  6.3e-16  bluestein
      4004  2^2.7.11.13      flat     raced          10482      16097    1.41  5.7e-16  
      4005  3^2.5.89         flat     raced          17337      74912    3.19  5.6e-16  
      4006  2.2003           prime    raced          27139      54537    1.99  6.1e-16  bluestein
      4007  4007             prime    raced          27227      67290    1.77  6.3e-16  bluestein
      4008  2^3.3.167        prime    raced          28078      52531    1.56  5.6e-16  bluestein
      4009  19.211           flat     raced          28762      88295    2.95  6.9e-16  
      4010  2.5.401          flat     raced          31025      40826    1.24  6.4e-16  
      4011  3.7.191          flat     raced          25806      33965    1.23  8.1e-16  
      4012  2^2.17.59        prime    raced          27177      62991    0.74  5.0e-16  flips differ 3.16x bluestein
      4013  4013             prime    raced          28807      67024    1.64  7.6e-16  bluestein
      4014  2.3^2.223        prime    raced          27091      48994    1.78  5.9e-16  bluestein
      4015  5.11.73          flat     raced          18100      51490    2.84  6.1e-16  
      4016  2^4.251          flat     raced          24069      33913    1.37  7.2e-16  
      4017  3.13.103         flat     raced          21831      82289    3.34  1.1e-15  
      4018  2.7^2.41         flat     raced          16280      35794    2.16  5.6e-16  
      4019  4019             prime    raced          28010      49843    1.42  9.1e-16  flips differ 1.33x bluestein
      4020  2^2.3.5.67       flat     raced          17972      52640    2.84  5.7e-16  
      4021  4021             prime    raced          28169      67130    1.68  5.5e-16  bluestein
      4022  2.2011           prime    raced          27420      55026    1.95  7.7e-16  bluestein
      4023  3^3.149          prime    raced          27458      52499    1.47  5.1e-16  flips differ 1.45x bluestein
      4024  2^3.503          prime    raced          27436      40319    1.46  5.6e-16  bluestein
      4025  5^2.7.23         chain3   raced          10720      31950    1.31  5.7e-16  flips differ 2.39x
      4026  2.3.11.61        flat     raced          27407      67460    2.29  5.3e-16  
      4027  4027             prime    raced          39275      84248    1.71  6.1e-16  bluestein
      4028  2^2.19.53        flat     raced          22377      76845    3.30  7.5e-16  
      4029  3.17.79          flat     raced          22919      76243    3.06  8.2e-16  
      4030  2.5.13.31        chain3   raced          14265      42383    2.78  5.8e-16  
      4031  29.139           flat     raced          31436      78248    2.47  1.1e-15  
      4032  2^6.3^2.7        ztt      raced           6146      15598    2.42  3.1e-16  
      4033  37.109           flat     raced          26165      79343    3.00  7.2e-16  
      4034  2.2017           prime    raced          28089      54841    1.90  5.6e-16  bluestein
      4035  3.5.269          prime    raced          27981      49339    1.72  7.2e-16  bluestein
      4036  2^2.1009         flat     raced          37551      64502    1.70  7.5e-16  
      4037  11.367           prime    raced          27205      52515    1.88  6.3e-16  bluestein
      4038  2.3.673          prime    raced          27873      51286    1.77  6.2e-16  bluestein
      4039  7.577            prime    raced          27616      42387    1.40  6.5e-16  bluestein
      4040  2^3.5.101        flat     raced          20171      46731    2.27  6.3e-16  
      4041  3^2.449          flat     raced          27717      52382    1.57  7.7e-16  
      4042  2.43.47          flat     raced          21053      70583    3.27  5.9e-16  
      4043  13.311           prime    raced          28607      69104    2.36  5.6e-16  bluestein
      4044  2^2.3.337        flat     raced          23626      62039    2.50  6.3e-16  
      4045  5.809            prime    raced          28375      47314    1.59  6.2e-16  bluestein
      4046  2.7.17^2         flat     raced          13019      44280    3.32  8.5e-16  
      4047  3.19.71          flat     raced          18651      67016    3.47  7.3e-16  
      4048  2^4.11.23        chain3   raced           8559      31777    3.66  5.2e-16  
      4049  4049             prime    raced          29582      87270    2.87  6.7e-16  bluestein
      4050  2.3^4.5^2        chain3   raced           7095       9831    1.37  6.4e-16  
      4051  4051             prime    raced          27033      41863    1.52  6.1e-16  bluestein
      4052  2^2.1013         prime    raced          27127      66970    2.45  6.3e-16  bluestein
      4053  3.7.193          flat     raced          21618      39471    1.78  7.0e-16  
      4054  2.2027           prime    raced          28227      41801    1.43  6.9e-16  bluestein
      4055  5.811            prime    raced          27582      46043    1.65  5.7e-16  bluestein
      4056  2^3.3.13^2       flat     raced           9493      15288    1.57  6.8e-16  
      4057  4057             prime    raced          27504      42875    1.53  6.0e-16  bluestein
      4058  2.2029           prime    raced          28082      41203    1.32  6.8e-16  bluestein
      4059  3^2.11.41        chain3   raced          15191      40721    2.30  4.5e-16  
      4060  2^2.5.7.29       flat     raced          13679      32048    1.81  4.3e-16  
      4061  31.131           flat     raced          21565      76009    3.23  8.7e-16  
      4062  2.3.677          prime    raced          27230      66229    2.40  5.7e-16  bluestein
      4063  17.239           flat     raced          23210      57327    2.30  7.0e-16  
      4064  2^5.127          flat     raced          23806      35868    1.48  7.1e-16  
      4065  3.5.271          flat     raced          31907      48284    1.49  6.9e-16  
      4066  2.19.107         prime    raced          26979      78520    2.65  6.6e-16  bluestein
      4067  7^2.83           prime    raced          27284      61223    2.15  5.8e-16  bluestein
      4068  2^2.3^2.113      flat     raced          17806      60349    3.31  5.5e-16  
      4069  13.313           flat     raced          26503      70571    2.25  8.0e-16  
      4070  2.5.11.37        chain3   raced          12539      37431    2.68  5.4e-16  
      4071  3.23.59          prime    raced          27496      68165    2.41  5.4e-16  bluestein
      4072  2^3.509          prime    raced          27210      36504    1.29  6.4e-16  bluestein
      4073  4073             prime    raced          28048      41619    1.48  5.5e-16  bluestein
      4074  2.3.7.97         flat     raced          18894      40538    2.00  6.1e-16  
      4075  5^2.163          flat     raced          27747      49711    1.77  7.8e-16  
      4076  2^2.1019         prime    raced          28028      37808    1.29  6.7e-16  bluestein
      4077  3^3.151          flat     raced          26216      37290    1.41  7.2e-16  
      4078  2.2039           prime    raced          28358      41748    1.41  7.4e-16  bluestein
      4079  4079             prime    raced          27930      42170    1.45  6.7e-16  bluestein
      4080  2^4.3.5.17       chain3   raced           7937      23897    3.01  4.6e-16  
      4081  7.11.53          flat     raced          17928      46507    2.37  7.3e-16  
      4082  2.13.157         flat     raced          21133      40069    1.89  9.0e-16  
      4083  3.1361           prime    raced          27018      54212    1.55  5.5e-16  bluestein
      4084  2^2.1021         prime    raced          27621      38432    1.33  7.5e-16  bluestein
      4085  5.19.43          flat     raced          17018      57574    2.89  5.4e-16  
      4086  2.3^2.227        prime    raced          28135      41219    1.45  5.8e-16  bluestein
      4087  61.67            prime    raced          28201      95411    3.26  6.1e-16  bluestein
      4088  2^3.7.73         flat     raced          14897      45639    3.02  6.3e-16  
      4089  3.29.47          chain3   raced          16662      65833    3.44  5.8e-16  
      4090  2.5.409          flat     raced          29100      66848    2.29  8.8e-16  
      4091  4091             prime    raced          29269      41405    1.40  5.5e-16  bluestein
      4092  2^2.3.11.31      chain3   raced          12881      37125    2.75  5.2e-16  
      4093  4093             prime    raced          27459      42145    1.50  6.1e-16  bluestein
      4094  2.23.89          flat     raced          19837      68140    3.31  7.2e-16  
      4095  3^2.5.7.13       chain3   raced           7769      13592    1.74  6.8e-16  
      4096  2^12             ztt      raced           4741       6284    1.32  3.4e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     1698     38     80   1.18   1.53   2.31    1.59
 flat      1221      8     42   1.29   2.17   3.29    2.09
 chain3     841      4     16   1.31   2.69   3.66    2.40
 2p         298      1     18   1.13   2.57   3.83    2.23
 mono        25      7     17   0.71   0.91   1.35    0.94
 ztt         12      0      0   1.27   1.42   2.08    1.56
 ALL       4095     58    173   1.21   1.87   3.27    1.92
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   0.92      4      1
 8..31                 24   0.98     12      5
 32..127               96   1.46     14      3
 128..511             384   1.76     12      2
 512..2047           1536   1.89     51     14
 2048..4096          2049   1.94     80     33
```


## by family
```
 family                                       cells median   <1.0  gmean
 flat                                          1221   2.17     42   2.09
 composite with a prime >= 53 (prime cell)     1149   1.55     54   1.59
 chain3                                         841   2.69     16   2.40
 prime N, bluestein                             481   1.50     20   1.58
 2p                                             292   2.58     17   2.27
 prime N, rader                                  68   1.81      6   1.70
 mono                                            21   0.92     14   0.93
 pow2                                            12   1.06      4   1.05
 ztt                                             10   1.52      0   1.63
```


flip agreement: our two readings more than 25% apart at 139 of 4095 cells.

worst 10: 2304 (chain3 0.34), 2138 (prime 0.40), 2402 (prime 0.46), 2019 (prime 0.51), 2122 (prime 0.53), 1603 (prime 0.57), 98 (flat 0.59), 687 (prime 0.62), 2435 (flat 0.63), 1181 (prime 0.64)
best 5: 289 (2p 5.61), 1734 (chain3 5.48), 391 (2p 5.03), 1156 (chain3 4.84), 2312 (chain3 4.79)
