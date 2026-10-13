# gauntlet report

run: `c2c_1d_2_4096_fftw_20261013`  contract file suffix: `_fftw`  cells: 4095 listed, 4095 benched, comparator: FFTW

control cell: 84 readings, 0.825..1.443 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         2  2                mono     replayed           3          3    1.20  0.0e+00  
         3  3                mono     replayed           3          4    1.06  1.5e-16  
         4  2^2              mono     replayed           4          4    1.05  0.0e+00  
         5  5                mono     replayed           5          4    0.85  1.5e-16  
         6  2.3              mono     replayed           6          4    0.75  2.8e-16  
         7  7                mono     replayed           6          6    0.87  2.6e-16  
         8  2^3              mono     replayed           6          5    0.81  5.6e-17  
         9  3^2              mono     replayed           8          6    0.76  4.2e-16  
        10  2.5              mono     replayed           8          6    0.80  1.8e-16  
        11  11               mono     replayed          10          7    0.70  1.6e-16  
        12  2^2.3            2p       replayed          10          6    0.58  2.8e-16  
        13  13               mono     replayed          14         11    0.78  6.0e-16  
        14  2.7              mono     replayed          12          8    0.70  1.6e-16  
        15  3.5              2p       replayed          16         10    0.63  3.3e-16  
        16  2^4              ztt      replayed           9          9    0.99  7.5e-17  
        17  17               mono     replayed          18         41    2.24  2.4e-16  
        18  2.3^2            2p       replayed          14         14    1.02  3.8e-16  
        19  19               mono     replayed          22         50    2.23  3.9e-16  
        20  2^2.5            2p       replayed          13         19    1.41  1.2e-16  
        21  3.7              2p       replayed          17         21    1.18  3.3e-16  
        22  2.11             mono     replayed          23         19    0.86  2.5e-16  
        23  23               mono     replayed          30         72    2.36  3.1e-16  
        24  2^3.3            2p       replayed          17         14    0.82  2.2e-16  
        25  5^2              2p       replayed          19         22    1.12  1.8e-16  
        26  2.13             mono     replayed          29         21    0.73  4.1e-16  
        27  3^3              flat     replayed          28         28    0.99  3.8e-16  
        28  2^2.7            2p       replayed          19         16    0.84  2.0e-16  
        29  29               mono     replayed          48        111    2.30  3.7e-16  
        30  2.3.5            2p       replayed          20         20    0.94  2.6e-16  
        31  31               mono     replayed          58        133    2.30  2.7e-16  
        32  2^5              2p       replayed          15         24    1.11  1.1e-16  
        33  3.11             2p       replayed          32         33    1.04  3.3e-16  
        34  2.17             flat     replayed          53         93    1.71  5.2e-16  
        35  5.7              2p       replayed          25         32    1.27  3.3e-16  
        36  2^2.3^2          2p       replayed          22         22    1.00  2.3e-16  
        37  37               mono     replayed          75        185    2.46  3.6e-16  
        38  2.19             flat     replayed          63        113    1.76  3.5e-16  
        39  3.13             2p       replayed          33         43    1.32  4.2e-16  
        40  2^3.5            2p       replayed          22         21    0.97  1.4e-16  
        41  41               mono     replayed          94        225    2.39  4.1e-16  
        42  2.3.7            2p       replayed          27         29    1.06  3.4e-16  
        43  43               mono     raced            100        246    2.46  3.6e-16  
        44  2^2.11           2p       replayed          27         36    1.32  4.0e-16  
        45  3^2.5            2p       replayed          32         42    1.33  4.3e-16  
        46  2.23             flat     replayed          71        177    2.50  4.9e-16  
        47  47               mono     replayed         116        265    2.27  5.3e-16  
        48  2^4.3            2p       replayed          27         24    0.89  3.0e-16  
        49  7^2              2p       replayed          36         45    1.26  2.5e-16  
        50  2.5^2            2p       replayed          32         35    1.09  2.7e-16  
        51  3.17             2p       replayed          53        172    3.13  3.5e-16  
        52  2^2.13           2p       replayed          36         43    1.17  5.3e-16  
        53  53               prime    replayed         195        347    1.71  5.8e-16  rader
        54  2.3^3            2p       replayed          35         40    1.13  4.2e-16  
        55  5.11             2p       replayed          42         56    1.33  4.2e-16  
        56  2^3.7            2p       replayed          33         30    0.88  2.5e-16  
        57  3.19             2p       replayed          63        207    3.22  5.4e-16  
        58  2.29             flat     replayed          96        246    2.54  4.0e-16  
        59  59               prime    replayed         205        332    1.60  3.1e-16  bluestein
        60  2^2.3.5          2p       replayed          36         33    0.92  3.8e-16  
        61  61               prime    replayed         143        480    3.34  4.9e-16  rader
        62  2.31             flat     replayed         103        286    2.75  3.5e-16  
        63  3^2.7            2p       replayed          47         60    1.29  4.2e-16  
        64  2^6              2p       replayed          30         32    1.05  1.7e-16  
        65  5.13             2p       replayed          52         65    1.25  4.4e-16  
        66  2.3.11           2p       replayed          46         52    1.12  3.7e-16  
        67  67               prime    replayed         174        541    3.11  4.6e-16  rader
        68  2^2.17           2p       replayed          56        194    3.35  3.2e-16  
        69  3.23             2p       replayed          96        252    2.47  4.1e-16  
        70  2.5.7            2p       replayed          45         50    1.10  2.8e-16  
        71  71               prime    replayed         168        429    2.52  4.5e-16  rader
        72  2^3.3^2          2p       replayed          44         40    0.91  4.7e-16  
        73  73               prime    replayed         172        455    2.63  9.8e-16  rader
        74  2.37             flat     replayed         174        395    2.25  4.3e-16  
        75  3.5^2            2p       replayed          57         66    1.16  3.4e-16  
        76  2^2.19           2p       replayed          68        233    3.31  3.0e-16  
        77  7.11             2p       replayed          60         87    1.45  3.6e-16  
        78  2.3.13           2p       replayed          58         65    0.83  5.1e-16  flips differ 1.34x
        79  79               prime    replayed         279        455    1.61  9.3e-16  rader
        80  2^4.5            2p       replayed          45         41    0.91  4.1e-16  
        81  3^4              2p       replayed          60         87    1.43  4.3e-16  
        82  2.41             flat     raced            155        497    3.20  6.2e-16  
        83  83               prime    replayed         455        524    1.15  5.9e-16  bluestein
        84  2^2.3.7          2p       replayed          54         48    0.89  2.7e-16  
        85  5.17             2p       replayed          81        254    3.05  4.0e-16  
        86  2.43             flat     raced            164        516    3.12  1.3e-15  
        87  3.29             flat     replayed         150        368    2.28  5.5e-16  
        88  2^3.11           2p       replayed          59         78    1.31  3.3e-16  
        89  89               prime    replayed         217        525    2.39  4.4e-16  rader
        90  2.3^2.5          2p       replayed          60         66    1.09  4.3e-16  
        91  7.13             2p       replayed          74         91    1.22  5.0e-16  
        92  2^2.23           2p       replayed          87        321    3.65  4.1e-16  
        93  3.31             flat     replayed         167        462    2.52  4.4e-16  
        94  2.47             flat     raced            188        620    3.27  9.9e-16  
        95  5.19             2p       replayed         102        300    2.91  3.0e-16  
        96  2^5.3            2p       replayed          51         47    0.92  3.0e-16  
        97  97               prime    replayed         219        502    2.30  5.4e-16  rader
        98  2.7^2            flat     replayed         116         72    0.62  3.8e-16  
        99  3^2.11           2p       replayed          77         97    1.27  3.8e-16  
       100  2^2.5^2          2p       replayed          64         55    0.86  3.3e-16  
       101  101              prime    replayed         235        634    2.69  8.9e-16  rader
       102  2.3.17           2p       replayed          88        299    3.31  4.6e-16  
       103  103              prime    replayed         289        634    2.18  1.2e-15  rader
       104  2^3.13           2p       replayed          73         93    0.93  5.2e-16  flips differ 1.37x
       105  3.5.7            2p       replayed          85         91    1.07  5.0e-16  
       106  2.53             prime    replayed         453        790    1.74  4.1e-16  bluestein
       107  107              prime    replayed         450        636    1.41  6.2e-16  bluestein
       108  2^2.3^3          2p       replayed          76         80    1.04  4.2e-16  
       109  109              prime    replayed         268        878    3.26  8.3e-16  rader
       110  2.5.11           2p       replayed          77         97    1.24  3.7e-16  
       111  3.37             2p       raced            182        625    3.44  4.3e-16  
       112  2^4.7            2p       replayed          71         58    0.82  2.4e-16  
       113  113              prime    replayed         273        879    2.19  8.4e-16  flips differ 1.48x rader
       114  2.3.19           2p       replayed         104        349    3.24  3.9e-16  
       115  5.23             2p       replayed         146        416    2.72  3.2e-16  
       116  2^2.29           flat     replayed         164        524    2.76  5.5e-16  
       117  3^2.13           2p       replayed          97        125    1.25  5.2e-16  
       118  2.59             prime    replayed         455        861    1.88  4.2e-16  bluestein
       119  7.17             2p       replayed         112        380    3.36  4.1e-16  
       120  2^3.3.5          2p       replayed          77         64    0.82  4.7e-16  
       121  11^2             2p       replayed         100        164    1.25  3.7e-16  flips differ 1.31x
       122  2.61             prime    replayed         458        962    2.07  3.9e-16  bluestein
       123  3.41             2p       raced            219        714    3.27  4.0e-16  
       124  2^2.31           flat     replayed         181        567    2.49  5.9e-16  flips differ 1.26x
       125  5^3              2p       replayed         104        128    1.23  4.2e-16  
       126  2.3^2.7          2p       replayed          90        120    1.24  3.5e-16  
       127  127              prime    replayed         394        642    1.53  7.7e-16  rader
       128  2^7              2p       replayed          65         63    0.59  4.3e-16  flips differ 1.65x
       129  3.43             flat     replayed         246        818    3.33  4.4e-16  
       130  2.5.13           2p       replayed          97        123    1.25  4.2e-16  
       131  131              prime    replayed         361        848    2.33  6.3e-16  rader
       132  2^2.3.11         2p       replayed         100        121    1.20  4.6e-16  
       133  7.19             2p       replayed         139        448    2.32  3.0e-16  flips differ 1.39x
       134  2.67             prime    replayed         931       1227    1.05  4.5e-16  flips differ 1.26x bluestein
       135  3^3.5            2p       replayed         109        126    1.16  3.0e-16  
       136  2^3.17           2p       replayed         112        395    3.53  4.3e-16  
       137  137              prime    replayed         389        797    1.89  7.1e-16  rader
       138  2.3.23           2p       replayed         137        485    3.53  4.8e-16  
       139  139              prime    replayed         471        793    1.68  6.6e-16  rader
       140  2^2.5.7          chain3   replayed         110         78    0.70  2.7e-16  
       141  3.47             flat     replayed         286        922    3.22  3.9e-16  
       142  2.71             prime    replayed         966        923    0.95  5.2e-16  bluestein
       143  11.13            2p       replayed         123        212    1.71  4.4e-16  
       144  2^4.3^2          chain3   replayed         106         87    0.81  3.6e-16  
       145  5.29             2p       replayed         189        659    3.48  4.0e-16  
       146  2.73             prime    replayed         970        989    1.02  6.7e-16  bluestein
       147  3.7^2            chain3   replayed         135        179    1.31  2.6e-16  
       148  2^2.37           2p       replayed         183        781    4.26  2.8e-16  
       149  149              prime    replayed         737        867    1.16  5.4e-16  rader
       150  2.3.5^2          2p       replayed         110        118    1.06  3.5e-16  
       151  151              prime    replayed         400        829    2.06  6.5e-16  rader
       152  2^3.19           2p       replayed         138        468    3.17  4.1e-16  
       153  3^2.17           2p       replayed         143        475    3.29  4.2e-16  
       154  2.7.11           flat     replayed         190        168    0.85  4.2e-16  
       155  5.31             2p       replayed         260        731    2.23  4.3e-16  flips differ 1.26x
       156  2^2.3.13         2p       replayed         124        158    0.95  6.9e-16  flips differ 1.34x
       157  157              prime    replayed         426        837    1.95  9.9e-16  rader
       158  2.79             prime    replayed         957        962    0.83  4.1e-16  bluestein
       159  3.53             prime    replayed         955       1198    1.02  4.5e-16  bluestein
       160  2^5.5            2p       replayed          90         85    0.94  3.0e-16  
       161  7.23             2p       replayed         180        616    3.42  4.5e-16  
       162  2.3^4            2p       replayed         122        156    1.27  3.7e-16  
       163  163              prime    replayed         462       1043    2.23  6.5e-16  rader
       164  2^2.41           2p       replayed         297        949    1.72  4.5e-16  flips differ 1.86x
       165  3.5.11           2p       replayed         144        198    1.37  3.7e-16  
       166  2.83             prime    replayed         947       1429    1.30  5.5e-16  bluestein
       167  167              prime    replayed         967       1042    1.03  4.6e-16  bluestein
       168  2^3.3.7          chain3   replayed         125        106    0.80  2.6e-16  
       169  13^2             2p       replayed         157        248    1.54  6.4e-16  
       170  2.5.17           2p       replayed         142        516    3.33  7.1e-16  
       171  3^2.19           2p       replayed         183        543    2.80  5.5e-16  
       172  2^2.43           2p       replayed         238       1082    4.55  4.3e-16  
       173  173              prime    replayed         965       1036    1.07  4.6e-16  bluestein
       174  2.3.29           2p       replayed         196        789    4.03  5.5e-16  
       175  5^2.7            chain3   replayed         152        183    1.19  3.2e-16  
       176  2^4.11           2p       replayed         122        165    1.34  2.3e-16  
       177  3.59             prime    replayed         979       1411    1.38  5.3e-16  bluestein
       178  2.89             prime    replayed         968       1447    1.49  4.8e-16  bluestein
       179  179              prime    replayed         968       1046    1.08  6.3e-16  bluestein
       180  2^2.3^2.5        2p       replayed         140        137    0.98  4.0e-16  
       181  181              prime    replayed         524       1396    2.65  7.1e-16  rader
       182  2.7.13           flat     replayed         240        214    0.80  4.7e-16  
       183  3.61             prime    replayed         969       1503    1.55  5.1e-16  bluestein
       184  2^3.23           2p       replayed         210        649    2.38  4.4e-16  flips differ 1.30x
       185  5.37             2p       replayed         475        983    1.58  3.5e-16  flips differ 1.31x
       186  2.3.31           2p       replayed         215        858    3.99  6.3e-16  
       187  11.17            flat     replayed         246        630    2.42  6.0e-16  
       188  2^2.47           2p       replayed         415       1216    2.87  3.7e-16  
       189  3^3.7            chain3   replayed         168        232    1.38  4.4e-16  
       190  2.5.19           2p       replayed         171        596    3.40  4.6e-16  
       191  191              prime    replayed         587        995    1.60  7.0e-16  rader
       192  2^6.3            2p       replayed         109        106    0.96  2.7e-16  
       193  193              prime    replayed         471       1090    2.30  5.4e-16  rader
       194  2.97             prime    replayed         975       1314    1.34  5.6e-16  bluestein
       195  3.5.13           chain3   replayed         176        250    1.42  5.1e-16  
       196  2^2.7^2          chain3   replayed         167        171    1.02  2.5e-16  
       197  197              prime    replayed         580       1087    1.86  4.6e-16  rader
       198  2.3^2.11         chain3   replayed         171        201    1.16  5.3e-16  
       199  199              prime    replayed         620       1093    1.68  6.0e-16  rader
       200  2^3.5^2          chain3   replayed         140        114    0.81  3.2e-16  
       201  3.67             prime    replayed         986       1860    1.85  4.8e-16  bluestein
       202  2.101            prime    replayed         965       1922    1.97  4.2e-16  bluestein
       203  7.29             2p       replayed         261        872    3.33  4.5e-16  
       204  2^2.3.17         chain3   replayed         232        645    2.05  5.0e-16  flips differ 1.35x
       205  5.41             2p       replayed         329       1187    3.60  4.3e-16  
       206  2.103            prime    replayed        1000       1916    1.90  7.7e-16  bluestein
       207  3^2.23           2p       replayed         273        752    1.87  4.1e-16  flips differ 1.47x
       208  2^4.13           2p       replayed         146        212    1.43  5.7e-16  
       209  11.19            2p       replayed         217        730    3.14  4.3e-16  
       210  2.3.5.7          chain3   replayed         166        176    1.06  3.5e-16  
       211  211              prime    replayed         705       1321    1.81  7.4e-16  rader
       212  2^2.53           prime    replayed        1003       1526    1.50  4.5e-16  bluestein
       213  3.71             prime    replayed         999       1403    1.36  5.8e-16  bluestein
       214  2.107            prime    replayed        1000       1924    1.88  6.1e-16  bluestein
       215  5.43             flat     replayed         389       1358    3.49  4.5e-16  
       216  2^3.3^3          2p       replayed         160        174    0.94  3.9e-16  
       217  7.31             2p       replayed         467       1082    1.89  4.4e-16  
       218  2.109            prime    replayed         987       1886    1.90  5.4e-16  bluestein
       219  3.73             prime    replayed        1010       1506    1.49  5.6e-16  bluestein
       220  2^2.5.11         chain3   replayed         188        216    1.14  3.1e-16  
       221  13.17            2p       replayed         226        755    3.28  5.2e-16  
       222  2.3.37           2p       replayed         282       1249    4.42  3.9e-16  
       223  223              prime    replayed        1004       1612    1.60  6.1e-16  bluestein
       224  2^5.7            2p       replayed         146        136    0.93  2.6e-16  
       225  3^2.5^2          2p       replayed         201        248    1.18  4.8e-16  
       226  2.113            prime    replayed         987       1886    1.90  6.2e-16  bluestein
       227  227              prime    replayed         980       1455    1.48  5.6e-16  bluestein
       228  2^2.3.19         flat     replayed         328        745    2.16  1.1e-15  
       229  229              prime    replayed         947       1461    1.32  8.0e-16  rader
       230  2.5.23           2p       replayed         232        827    3.57  4.7e-16  
       231  3.7.11           chain3   replayed         214        273    1.27  3.5e-16  
       232  2^3.29           flat     replayed         337        983    2.43  5.3e-16  
       233  233              prime    replayed         980       1455    1.48  4.7e-16  bluestein
       234  2.3^2.13         chain3   replayed         231        246    1.06  5.1e-16  
       235  5.47             2p       replayed         876       1566    1.43  3.9e-16  flips differ 1.25x
       236  2^2.59           prime    replayed        1034       1741    1.67  4.8e-16  bluestein
       237  3.79             prime    replayed        1020       1494    1.41  5.8e-16  bluestein
       238  2.7.17           flat     replayed         348        744    1.92  1.2e-15  
       239  239              prime    replayed        1002       1464    1.39  4.9e-16  bluestein
       240  2^4.3.5          2p       replayed         171        141    0.82  3.4e-16  
       241  241              prime    replayed         654       1781    2.63  5.2e-16  rader
       242  2.11^2           flat     replayed         327        379    1.02  9.8e-16  
       243  3^5              2p       replayed         218        309    1.42  5.7e-16  
       244  2^2.61           prime    replayed        1016       2009    1.96  3.7e-16  bluestein
       245  5.7^2            chain3   replayed         228        309    1.35  2.8e-16  
       246  2.3.41           2p       replayed         339       1436    4.20  3.0e-16  
       247  13.19            2p       replayed         264        886    3.35  4.1e-16  
       248  2^3.31           flat     replayed         454       1233    2.14  4.8e-16  flips differ 1.27x
       249  3.83             prime    replayed        1008       2170    2.12  8.6e-16  bluestein
       250  2.5^3            chain3   replayed         190        210    1.10  4.1e-16  
       251  251              prime    replayed         696       1308    1.86  6.0e-16  rader
       252  2^2.3^2.7        chain3   replayed         208        209    1.00  4.8e-16  
       253  11.23            2p       replayed         302        995    3.29  5.3e-16  
       254  2.127            prime    replayed        1006       1438    1.19  4.4e-16  bluestein
       255  3.5.17           2p       replayed         255        819    3.09  5.5e-16  
       256  2^8              2p       replayed         134        136    1.01  2.5e-16  
       257  257              prime    replayed         655       1844    2.81  4.9e-16  rader
       258  2.3.43           2p       replayed         511       1558    1.71  4.3e-16  flips differ 1.79x
       259  7.37             2p       replayed         376       1474    3.91  4.1e-16  
       260  2^2.5.13         chain3   replayed         226        267    1.18  4.4e-16  
       261  3^2.29           2p       replayed         334       1133    3.38  4.2e-16  
       262  2.131            prime    replayed        2070       1779    0.83  6.1e-16  bluestein
       263  263              prime    replayed        2061       1857    0.90  5.4e-16  bluestein
       264  2^3.3.11         chain3   replayed         224        255    1.13  3.6e-16  
       265  5.53             prime    replayed        2057       1921    0.90  5.7e-16  bluestein
       266  2.7.19           flat     replayed         389        894    2.29  4.4e-16  
       267  3.89             prime    replayed        2051       2179    1.06  6.6e-16  bluestein
       268  2^2.67           prime    replayed        2267       2468    1.08  5.9e-16  bluestein
       269  269              prime    replayed        2113       1855    0.87  5.3e-16  bluestein
       270  2.3^3.5          chain3   replayed         223        257    1.14  4.9e-16  
       271  271              prime    replayed         791       1816    2.29  7.7e-16  rader
       272  2^4.17           2p       replayed         215        832    3.82  5.8e-16  
       273  3.7.13           2p       replayed         255        347    1.33  5.1e-16  
       274  2.137            prime    replayed        2195       2255    1.02  5.8e-16  bluestein
       275  5^2.11           chain3   replayed         250        347    1.37  3.8e-16  
       276  2^2.3.23         2p       replayed         289       1108    3.83  4.5e-16  
       277  277              prime    replayed        1053       1815    1.48  7.8e-16  rader
       278  2.139            prime    replayed        2250       2300    1.01  5.5e-16  bluestein
       279  3^2.31           2p       replayed         465       1331    2.18  4.9e-16  flips differ 1.31x
       280  2^3.5.7          chain3   replayed         210        172    0.82  3.4e-16  
       281  281              prime    replayed         814       1816    2.22  4.6e-16  rader
       282  2.3.47           2p       replayed         631       1836    1.83  4.3e-16  flips differ 1.59x
       283  283              prime    replayed        2162       1834    0.83  6.5e-16  bluestein
       284  2^2.71           prime    replayed        2139       1883    0.88  4.4e-16  bluestein
       285  3.5.19           2p       replayed         328        954    2.77  4.4e-16  
       286  2.11.13          flat     replayed         398        485    0.99  4.9e-16  
       287  7.41             2p       replayed         454       1689    3.70  3.2e-16  
       288  2^5.3^2          2p       replayed         198        213    1.08  3.8e-16  
       289  17^2             2p       replayed         339       1549    4.28  7.9e-16  
       290  2.5.29           2p       replayed         333       1251    3.76  4.4e-16  
       291  3.97             prime    replayed        2080       1994    0.94  5.9e-16  bluestein
       292  2^2.73           prime    replayed        2078       2002    0.94  5.4e-16  bluestein
       293  293              prime    replayed        2128       1849    0.70  6.9e-16  bluestein
       294  2.3.7^2          chain3   replayed         252        282    1.12  3.2e-16  
       295  5.59             prime    replayed        2081       2207    1.03  4.7e-16  bluestein
       296  2^3.37           2p       replayed         371       1663    4.47  3.7e-16  
       297  3^3.11           chain3   replayed         286        370    1.29  5.7e-16  
       298  2.149            prime    replayed        2165       2022    0.93  4.9e-16  bluestein
       299  13.23            2p       replayed         356       1206    3.38  5.3e-16  
       300  2^2.3.5^2        chain3   replayed         234        243    1.04  3.9e-16  
       301  7.43             flat     replayed         557       1928    3.45  3.9e-16  
       302  2.151            prime    replayed        2089       1844    0.88  4.2e-16  bluestein
       303  3.101            prime    replayed        2071       2844    1.35  5.8e-16  bluestein
       304  2^4.19           2p       replayed         287        980    2.89  2.7e-16  
       305  5.61             prime    replayed        2081       2581    1.22  4.4e-16  bluestein
       306  2.3^2.17         chain3   replayed         315        974    2.56  5.9e-16  
       307  307              prime    replayed        1201       2492    2.04  1.1e-15  rader
       308  2^2.7.11         chain3   replayed         275        329    1.13  3.4e-16  
       309  3.103            prime    replayed        2145       2880    1.34  6.6e-16  bluestein
       310  2.5.31           2p       replayed         676       1585    2.23  4.4e-16  
       311  311              prime    replayed        1501       2476    1.64  7.1e-16  rader
       312  2^3.3.13         chain3   replayed         277        320    1.15  4.7e-16  
       313  313              prime    replayed        1073       2482    2.04  8.4e-16  rader
       314  2.157            prime    replayed        2113       1838    0.86  4.3e-16  bluestein
       315  3^2.5.7          chain3   replayed         292        355    1.22  5.0e-16  
       316  2^2.79           prime    replayed        2160       1953    0.90  5.0e-16  bluestein
       317  317              prime    replayed        2150       1732    0.80  4.8e-16  bluestein
       318  2.3.53           prime    replayed        2105       2393    1.11  4.4e-16  bluestein
       319  11.29            2p       replayed         422       1505    3.56  4.8e-16  
       320  2^6.5            2p       replayed         205        189    0.73  2.1e-16  flips differ 1.27x
       321  3.107            prime    replayed        2101       2944    1.38  6.1e-16  bluestein
       322  2.7.23           flat     replayed         520       1219    2.33  4.9e-16  
       323  17.19            2p       replayed         400       1796    4.31  5.8e-16  
       324  2^2.3^4          chain3   replayed         319        309    0.90  4.3e-16  
       325  5^2.13           2p       replayed         318        431    1.33  7.2e-16  
       326  2.163            prime    replayed        2267       3098    1.34  5.4e-16  bluestein
       327  3.109            prime    replayed        2087       2839    1.34  5.5e-16  bluestein
       328  2^3.41           2p       replayed         608       1920    1.94  4.9e-16  flips differ 1.62x
       329  7.47             flat     replayed         638       2170    3.40  4.3e-16  
       330  2.3.5.11         chain3   replayed         301        343    1.12  5.3e-16  
       331  331              prime    replayed        1115       2632    2.32  6.9e-16  rader
       332  2^2.83           prime    replayed        2102       2854    1.35  5.4e-16  bluestein
       333  3^2.37           2p       replayed         975       1916    1.67  3.8e-16  
       334  2.167            prime    replayed        2174       3139    1.44  5.8e-16  bluestein
       335  5.67             prime    replayed        2279       3095    1.28  5.0e-16  bluestein
       336  2^4.3.7          2p       replayed         245        258    1.05  3.2e-16  
       337  337              prime    replayed        1061       2615    2.40  7.5e-16  rader
       338  2.13^2           flat     replayed         486        574    1.15  9.8e-16  
       339  3.113            prime    replayed        2232       2855    1.24  5.5e-16  bluestein
       340  2^2.5.17         chain3   replayed         338       1037    2.88  6.1e-16  
       341  11.31            flat     replayed         644       1717    2.47  9.0e-16  
       342  2.3^2.19         chain3   replayed         403       1125    2.46  3.9e-16  
       343  7^3              chain3   replayed         347        472    1.32  2.8e-16  
       344  2^3.43           2p       replayed         488       2174    4.44  4.4e-16  
       345  3.5.23           2p       replayed         474       1309    2.76  4.6e-16  
       346  2.173            prime    replayed        2170       3123    1.43  5.1e-16  bluestein
       347  347              prime    replayed        2104       2214    1.05  5.8e-16  bluestein
       348  2^2.3.29         2p       replayed         416       1517    3.63  5.5e-16  
       349  349              prime    replayed        1480       2221    1.29  9.9e-16  rader
       350  2.5^2.7          chain3   replayed         285        300    1.05  2.8e-16  
       351  3^3.13           chain3   replayed         360        469    1.07  4.6e-16  
       352  2^5.11           2p       replayed         258        345    1.34  2.6e-16  
       353  353              prime    replayed        1002       2225    2.14  4.9e-16  rader
       354  2.3.59           prime    replayed        2246       2825    1.22  4.3e-16  bluestein
       355  5.71             prime    replayed        2128       2349    1.10  5.7e-16  bluestein
       356  2^2.89           prime    replayed        2181       2282    1.03  4.6e-16  bluestein
       357  3.7.17           chain3   replayed         423       1136    2.37  4.8e-16  
       358  2.179            prime    replayed        2114       2260    1.06  5.2e-16  bluestein
       359  359              prime    replayed        2124       2227    1.03  5.7e-16  bluestein
       360  2^3.3^2.5        chain3   replayed         295        280    0.95  3.8e-16  
       361  19^2             2p       replayed         483       2088    3.77  4.5e-16  
       362  2.181            prime    replayed        2124       2980    1.39  5.7e-16  bluestein
       363  3.11^2           chain3   replayed         371        596    1.57  3.9e-16  
       364  2^2.7.13         chain3   replayed         337        409    1.21  4.9e-16  
       365  5.73             prime    replayed        2105       2559    1.21  6.3e-16  bluestein
       366  2.3.61           prime    replayed        2258       3009    1.33  3.6e-16  bluestein
       367  367              prime    replayed        2209       2681    1.20  5.9e-16  bluestein
       368  2^4.23           2p       replayed         473       1345    1.75  3.8e-16  flips differ 1.62x
       369  3^2.41           2p       replayed        1000       2289    2.29  4.5e-16  
       370  2.5.37           2p       replayed         484       2035    4.20  3.5e-16  
       371  7.53             prime    replayed        2189       2723    1.21  3.9e-16  bluestein
       372  2^2.3.31         flat     replayed         725       1898    2.54  4.0e-16  
       373  373              prime    replayed        2135       2686    1.25  4.9e-16  bluestein
       374  2.11.17          flat     replayed         557       1364    2.41  1.0e-15  
       375  3.5^3            chain3   replayed         348        444    1.12  4.7e-16  
       376  2^3.47           2p       replayed        1505       2459    1.63  3.7e-16  
       377  13.29            2p       replayed         955       1790    1.77  5.6e-16  
       378  2.3^3.7          chain3   replayed         336        404    1.17  4.0e-16  
       379  379              prime    replayed        1671       2256    1.31  8.1e-16  rader
       380  2^2.5.19         chain3   replayed         404       1249    2.91  3.7e-16  
       381  3.127            prime    replayed        2127       2141    1.00  5.3e-16  bluestein
       382  2.191            prime    replayed        2217       2191    0.96  5.5e-16  bluestein
       383  383              prime    replayed        2283       2207    0.96  6.0e-16  bluestein
       384  2^7.3            2p       replayed         250        224    0.89  3.5e-16  
       385  5.7.11           chain3   replayed         372        476    1.28  4.3e-16  
       386  2.193            prime    replayed        2295       2480    1.07  6.8e-16  bluestein
       387  3^2.43           2p       replayed         629       2387    3.74  4.5e-16  
       388  2^2.97           prime    replayed        2191       2627    1.20  5.0e-16  bluestein
       389  389              prime    replayed        2118       2486    1.16  6.1e-16  bluestein
       390  2.3.5.13         chain3   replayed         367        418    1.11  5.8e-16  
       391  17.23            2p       replayed         523       2347    4.24  6.2e-16  
       392  2^3.7^2          chain3   replayed         332        357    1.05  2.9e-16  
       393  3.131            prime    replayed        2227       2696    1.20  5.3e-16  bluestein
       394  2.197            prime    replayed        2131       2489    1.13  5.5e-16  bluestein
       395  5.79             prime    replayed        2120       2512    1.17  5.1e-16  bluestein
       396  2^2.3^2.11       chain3   replayed         371        421    1.13  4.2e-16  
       397  397              prime    replayed        1509       2489    1.64  7.9e-16  rader
       398  2.199            prime    replayed        2143       2517    1.14  4.4e-16  bluestein
       399  3.7.19           chain3   replayed         468       1441    1.64  5.7e-16  flips differ 1.87x
       400  2^4.5^2          2p       replayed         310        264    0.61  2.7e-16  flips differ 1.39x
       401  401              prime    replayed        1312       2833    2.04  7.6e-16  rader
       402  2.3.67           prime    replayed        2264       3713    1.57  5.7e-16  bluestein
       403  13.31            2p       replayed         716       2080    2.13  6.4e-16  flips differ 1.36x
       404  2^2.101          prime    replayed        2299       2734    1.18  6.9e-16  bluestein
       405  3^4.5            chain3   replayed         377        486    1.15  4.1e-16  
       406  2.7.29           flat     replayed         731       1822    2.49  5.6e-16  
       407  11.37            2p       replayed         604       2398    3.80  3.9e-16  
       408  2^3.3.17         chain3   replayed         440       1282    2.89  4.8e-16  
       409  409              prime    replayed        1630       2829    1.69  8.1e-16  rader
       410  2.5.41           2p       replayed         801       2550    1.76  3.3e-16  flips differ 1.80x
       411  3.137            prime    replayed        2277       3419    1.48  7.2e-16  bluestein
       412  2^2.103          prime    replayed        2235       2752    1.23  6.1e-16  bluestein
       413  7.59             prime    replayed        2227       3341    1.49  4.9e-16  bluestein
       414  2.3^2.23         chain3   replayed         513       1546    3.00  5.1e-16  
       415  5.83             prime    replayed        2285       3611    1.57  5.7e-16  bluestein
       416  2^5.13           2p       replayed         320        433    1.33  4.9e-16  
       417  3.139            prime    replayed        2270       3431    1.49  6.4e-16  bluestein
       418  2.11.19          flat     replayed         648       1545    2.37  9.5e-16  
       419  419              prime    replayed        2155       2821    1.31  6.9e-16  bluestein
       420  2^2.3.5.7        chain3   replayed         361        340    0.89  3.8e-16  
       421  421              prime    replayed        1830       2836    1.37  6.6e-16  rader
       422  2.211            prime    replayed        2268       2784    1.07  6.9e-16  bluestein
       423  3^2.47           flat     replayed         823       2820    3.39  6.0e-16  
       424  2^3.53           prime    replayed        2181       3084    1.41  4.9e-16  bluestein
       425  5^2.17           2p       replayed         457       1401    3.04  5.8e-16  
       426  2.3.71           prime    replayed        2150       2734    1.20  5.6e-16  bluestein
       427  7.61             prime    replayed        2158       3586    1.65  4.3e-16  bluestein
       428  2^2.107          prime    replayed        2321       2788    1.16  7.2e-16  bluestein
       429  3.11.13          chain3   replayed         452        739    1.61  5.6e-16  
       430  2.5.43           2p       replayed         630       2761    4.38  3.7e-16  
       431  431              prime    replayed        2157       2843    1.31  6.0e-16  bluestein
       432  2^4.3^3          2p       replayed         331        368    0.97  4.2e-16  
       433  433              prime    replayed        1505       3162    2.08  7.9e-16  rader
       434  2.7.31           flat     replayed         786       2159    2.73  4.6e-16  
       435  3.5.29           2p       replayed         584       1958    3.35  4.8e-16  
       436  2^2.109          prime    replayed        2280       3712    1.59  7.2e-16  bluestein
       437  19.23            2p       replayed         623       2726    4.21  4.7e-16  
       438  2.3.73           prime    replayed        2188       2895    1.29  7.1e-16  bluestein
       439  439              prime    replayed        2158       3174    1.46  5.8e-16  bluestein
       440  2^3.5.11         chain3   replayed         384        459    1.17  5.3e-16  
       441  3^2.7^2          chain3   replayed         443        590    1.33  4.9e-16  
       442  2.13.17          flat     replayed         668       1622    2.35  1.2e-15  
       443  443              prime    replayed        2209       3159    1.40  7.0e-16  bluestein
       444  2^2.3.37         2p       replayed         601       2561    4.14  5.2e-16  
       445  5.89             prime    replayed        2263       3689    1.60  4.6e-16  bluestein
       446  2.223            prime    replayed        2158       3387    1.56  5.6e-16  bluestein
       447  3.149            prime    replayed        2211       3070    1.36  6.4e-16  bluestein
       448  2^6.7            chain3   replayed         349        310    0.87  2.8e-16  
       449  449              prime    replayed        1438       3166    2.15  7.4e-16  rader
       450  2.3^2.5^2        chain3   replayed         390        433    1.10  4.6e-16  
       451  11.41            2p       replayed        1181       2805    1.64  3.7e-16  flips differ 1.45x
       452  2^2.113          prime    replayed        2173       3784    1.72  7.0e-16  bluestein
       453  3.151            prime    replayed        2172       2767    1.24  5.9e-16  bluestein
       454  2.227            prime    replayed        2166       3027    1.39  5.5e-16  bluestein
       455  5.7.13           chain3   replayed         459        601    1.31  5.2e-16  
       456  2^3.3.19         chain3   replayed         513       1649    3.03  3.9e-16  
       457  457              prime    replayed        1939       3004    1.52  7.3e-16  rader
       458  2.229            prime    replayed        2169       3054    1.41  5.9e-16  bluestein
       459  3^3.17           chain3   replayed         525       1573    2.94  4.7e-16  
       460  2^2.5.23         flat     replayed         644       1724    2.66  5.5e-16  
       461  461              prime    replayed        2172       2994    1.36  5.6e-16  bluestein
       462  2.3.7.11         chain3   replayed         457        513    1.06  5.0e-16  
       463  463              prime    replayed        1822       3003    1.52  6.4e-16  rader
       464  2^4.29           2p       replayed         855       2172    1.99  3.0e-16  flips differ 1.28x
       465  3.5.31           2p       replayed        1062       2420    2.14  5.6e-16  
       466  2.233            prime    replayed        2216       3057    1.35  5.0e-16  bluestein
       467  467              prime    replayed        2239       3006    1.34  5.4e-16  bluestein
       468  2^2.3^2.13       chain3   replayed         483        522    1.02  5.1e-16  
       469  7.67             prime    replayed        2250       4365    1.91  4.1e-16  bluestein
       470  2.5.47           2p       replayed        1064       3169    1.71  5.1e-16  flips differ 1.73x
       471  3.157            prime    replayed        2166       2782    1.26  6.2e-16  bluestein
       472  2^3.59           prime    replayed        2185       3775    1.66  5.5e-16  bluestein
       473  11.43            2p       replayed         784       3049    3.87  4.3e-16  
       474  2.3.79           prime    replayed        2245       2847    1.26  6.2e-16  bluestein
       475  5^2.19           2p       replayed         612       1656    2.68  6.7e-16  
       476  2^2.7.17         chain3   replayed         563       1533    2.72  5.9e-16  
       477  3^2.53           prime    replayed        2171       3649    1.67  4.7e-16  bluestein
       478  2.239            prime    replayed        2171       3048    1.40  7.0e-16  bluestein
       479  479              prime    replayed        2175       3012    1.38  6.1e-16  bluestein
       480  2^5.3.5          2p       replayed         376        359    0.94  4.1e-16  
       481  13.37            2p       replayed         725       2813    3.86  5.1e-16  
       482  2.241            prime    replayed        2197       4914    2.17  5.4e-16  bluestein
       483  3.7.23           2p       replayed         699       1842    2.40  5.5e-16  
       484  2^2.11^2         chain3   replayed         600        774    0.87  3.7e-16  flips differ 1.49x
       485  5.97             prime    replayed        2169       3453    1.54  3.9e-16  bluestein
       486  2.3^5            chain3   replayed         458        551    1.13  4.6e-16  
       487  487              prime    replayed        2257       3280    1.45  5.3e-16  bluestein
       488  2^3.61           prime    replayed        2171       4073    1.85  4.5e-16  bluestein
       489  3.163            prime    replayed        2256       4803    2.11  6.1e-16  bluestein
       490  2.5.7^2          chain3   replayed         451        486    1.04  3.8e-16  
       491  491              prime    replayed        1819       3453    1.79  6.2e-16  rader
       492  2^2.3.41         2p       replayed         953       2927    3.04  3.7e-16  
       493  17.29            2p       replayed         768       3453    4.45  5.6e-16  
       494  2.13.19          flat     replayed         781       1869    2.29  1.2e-15  
       495  3^2.5.11         flat     replayed         578        646    1.11  5.0e-16  
       496  2^4.31           flat     replayed         818       2370    2.69  6.6e-16  
       497  7.71             prime    replayed        2227       3349    1.48  5.5e-16  bluestein
       498  2.3.83           prime    replayed        2277       3363    1.47  5.3e-16  bluestein
       499  499              prime    replayed        2186       3298    1.46  7.4e-16  bluestein
       500  2^2.5^3          chain3   replayed         426        416    0.97  4.5e-16  
       501  3.167            prime    replayed        2182       4742    2.15  5.2e-16  bluestein
       502  2.251            prime    replayed        2258       2986    1.31  4.8e-16  bluestein
       503  503              prime    replayed        2188       2886    1.28  4.1e-16  bluestein
       504  2^3.3^2.7        chain3   replayed         468        449    0.91  4.5e-16  
       505  5.101            prime    replayed        2192       3507    1.58  6.3e-16  bluestein
       506  2.11.23          flat     replayed         861       2104    2.43  1.5e-15  
       507  3.13^2           chain3   replayed         705        863    1.18  5.4e-16  
       508  2^2.127          prime    replayed        2198       2872    1.29  5.9e-16  bluestein
       509  509              prime    replayed        2294       2898    1.24  4.3e-16  bluestein
       510  2.3.5.17         chain3   replayed         571       1639    2.72  5.3e-16  
       511  7.73             prime    replayed        2210       3562    1.56  7.1e-16  bluestein
       512  2^9              2p       replayed         296        300    1.01  3.1e-16  
       513  3^3.19           2p       replayed         577       1761    3.03  4.6e-16  
       514  2.257            prime    replayed        4736       4726    0.97  7.9e-16  bluestein
       515  5.103            prime    replayed        4729       3508    0.73  5.0e-16  bluestein
       516  2^2.3.43         flat     replayed         920       3208    3.48  4.5e-16  
       517  11.47            flat     replayed        1025       3634    3.40  5.5e-16  
       518  2.7.37           flat     replayed        1011       3059    2.88  6.2e-16  
       519  3.173            prime    replayed        4776       4717    0.99  4.0e-16  bluestein
       520  2^3.5.13         chain3   replayed         479        574    0.96  6.2e-16  flips differ 1.27x
       521  521              prime    replayed        2007       3865    1.69  1.1e-15  rader
       522  2.3^2.29         chain3   replayed         750       2507    3.29  3.9e-16  
       523  523              prime    replayed        3127       3823    1.15  7.1e-16  rader
       524  2^2.131          prime    replayed        4678       3637    0.58  5.7e-16  flips differ 1.35x bluestein
       525  3.5^2.7          2p       replayed         538        620    1.15  4.3e-16  
       526  2.263            prime    replayed        5477       4738    0.79  4.7e-16  bluestein
       527  17.31            2p       replayed         812       3800    4.50  6.1e-16  
       528  2^4.3.11         chain3   replayed         516        553    1.05  4.2e-16  
       529  23^2             2p       replayed         876       3546    2.58  4.5e-16  flips differ 1.57x
       530  2.5.53           prime    replayed        4702       3922    0.81  4.0e-16  bluestein
       531  3^2.59           prime    replayed        4691       4054    0.86  4.5e-16  bluestein
       532  2^2.7.19         flat     replayed         640       1802    2.80  4.6e-16  
       533  13.41            2p       replayed         871       3375    3.83  6.4e-16  
       534  2.3.89           prime    replayed        4698       3433    0.72  5.4e-16  bluestein
       535  5.107            prime    replayed        4737       3531    0.74  5.3e-16  bluestein
       536  2^3.67           prime    replayed        4725       4986    1.05  4.0e-16  bluestein
       537  3.179            prime    replayed        4760       4733    0.99  6.6e-16  bluestein
       538  2.269            prime    replayed        4850       4756    0.97  5.4e-16  bluestein
       539  7^2.11           flat     replayed         671        701    0.99  4.2e-16  
       540  2^2.3^3.5        chain3   replayed         494        520    1.05  4.8e-16  
       541  541              prime    replayed        2839       4853    1.55  6.2e-16  rader
       542  2.271            prime    replayed        4789       4430    0.80  7.1e-16  bluestein
       543  3.181            prime    replayed        4790       4526    0.82  4.4e-16  bluestein
       544  2^5.17           2p       replayed         554       1683    3.03  4.8e-16  
       545  5.109            prime    replayed        4994       5068    1.00  5.1e-16  bluestein
       546  2.3.7.13         chain3   replayed         588        636    0.98  5.9e-16  
       547  547              prime    replayed        2207       4861    2.17  9.5e-16  rader
       548  2^2.137          prime    replayed        4740       3502    0.72  4.5e-16  bluestein
       549  3^2.61           prime    replayed        4793       4675    0.96  6.2e-16  bluestein
       550  2.5^2.11         chain3   replayed         582        608    1.01  4.1e-16  
       551  19.29            2p       replayed         903       3986    3.82  4.0e-16  
       552  2^3.3.23         flat     replayed         779       2048    1.88  5.3e-16  flips differ 1.40x
       553  7.79             prime    replayed        4917       3470    0.69  5.2e-16  bluestein
       554  2.277            prime    replayed        4757       4407    0.93  4.6e-16  bluestein
       555  3.5.37           2p       replayed         825       3117    3.77  6.0e-16  
       556  2^2.139          prime    replayed        4777       3623    0.75  6.3e-16  bluestein
       557  557              prime    replayed        4753       4840    1.02  6.3e-16  bluestein
       558  2.3^2.31         chain3   replayed         845       2690    3.06  5.9e-16  
       559  13.43            2p       replayed         950       3672    3.85  6.4e-16  
       560  2^4.5.7          chain3   replayed         520        434    0.83  3.5e-16  
       561  3.11.17          flat     replayed         828       2056    2.46  1.2e-15  
       562  2.281            prime    replayed        4711       4386    0.91  5.3e-16  bluestein
       563  563              prime    replayed        4811       4857    1.00  6.2e-16  bluestein
       564  2^2.3.47         flat     replayed        1107       3766    3.40  4.1e-16  
       565  5.113            prime    replayed        4775       5119    1.07  5.0e-16  bluestein
       566  2.283            prime    replayed        4789       4389    0.91  6.1e-16  bluestein
       567  3^4.7            2p       replayed         598        790    1.32  6.0e-16  
       568  2^3.71           prime    replayed        4770       3747    0.78  3.7e-16  bluestein
       569  569              prime    replayed        4776       3941    0.81  5.8e-16  bluestein
       570  2.3.5.19         flat     replayed         844       1887    2.20  4.7e-16  
       571  571              prime    replayed        2573       4002    1.55  7.3e-16  rader
       572  2^2.11.13        chain3   replayed         727        960    1.30  4.5e-16  
       573  3.191            prime    replayed        4818       3271    0.67  6.0e-16  bluestein
       574  2.7.41           flat     replayed        1229       3635    2.91  2.1e-15  
       575  5^2.23           2p       replayed         848       2281    2.69  5.1e-16  
       576  2^6.3^2          2p       replayed         500        454    0.91  3.5e-16  
       577  577              prime    replayed        1913       4190    2.18  7.5e-16  rader
       578  2.17^2           flat     replayed         915       3278    3.42  8.0e-16  
       579  3.193            prime    replayed        4845       3734    0.60  4.2e-16  flips differ 1.29x bluestein
       580  2^2.5.29         chain3   replayed         922       2584    2.78  4.3e-16  
       581  7.83             prime    replayed        4751       5124    1.06  5.0e-16  bluestein
       582  2.3.97           prime    replayed        4753       3336    0.70  4.8e-16  bluestein
       583  11.53            prime    replayed        4832       4617    0.91  4.0e-16  bluestein
       584  2^3.73           prime    replayed        4784       4271    0.87  4.5e-16  bluestein
       585  3^2.5.13         chain3   replayed         778        809    1.02  5.1e-16  
       586  2.293            prime    replayed        4759       3891    0.80  6.6e-16  bluestein
       587  587              prime    replayed        4826       4233    0.87  5.5e-16  bluestein
       588  2^2.3.7^2        flat     replayed         618        553    0.65  4.2e-16  flips differ 1.39x
       589  19.31            2p       replayed         982       4495    3.15  5.1e-16  flips differ 1.45x
       590  2.5.59           prime    replayed        4860       4818    0.99  4.6e-16  bluestein
       591  3.197            prime    replayed        4778       3721    0.76  5.4e-16  bluestein
       592  2^4.37           2p       replayed         752       3215    3.82  4.3e-16  
       593  593              prime    replayed        3010       4253    1.41  7.2e-16  rader
       594  2.3^3.11         chain3   replayed         664        695    0.95  4.6e-16  
       595  5.7.17           chain3   replayed         874       1990    2.27  5.1e-16  
       596  2^2.149          prime    replayed        4764       3950    0.81  6.2e-16  bluestein
       597  3.199            prime    replayed        4809       3776    0.78  5.3e-16  bluestein
       598  2.13.23          flat     replayed        1017       2583    2.48  8.4e-16  
       599  599              prime    replayed        4865       4217    0.86  5.8e-16  bluestein
       600  2^3.3.5^2        chain3   replayed         607        483    0.76  4.2e-16  
       601  601              prime    replayed        2901       5334    1.78  7.6e-16  rader
       602  2.7.43           flat     replayed        1312       3973    2.91  6.8e-16  
       603  3^2.67           prime    replayed        4760       5658    1.19  3.9e-16  bluestein
       604  2^2.151          prime    replayed        4718       3732    0.74  4.8e-16  bluestein
       605  5.11^2           flat     replayed         744       1010    1.30  4.6e-16  
       606  2.3.101          prime    replayed        4745       4222    0.86  5.4e-16  bluestein
       607  607              prime    replayed        4821       5350    1.09  6.1e-16  bluestein
       608  2^5.19           2p       replayed         548       1970    3.59  3.8e-16  
       609  3.7.29           2p       replayed        1052       2757    1.59  4.5e-16  flips differ 1.65x
       610  2.5.61           prime    replayed        4746       5168    1.07  4.3e-16  bluestein
       611  13.47            flat     replayed        1227       4309    3.51  7.1e-16  
       612  2^2.3^2.17       chain3   replayed         888       1973    2.20  6.9e-16  
       613  613              prime    replayed        2684       4917    1.75  8.7e-16  rader
       614  2.307            prime    replayed        4724       5129    1.08  5.1e-16  bluestein
       615  3.5.41           2p       replayed         995       3912    3.93  4.2e-16  
       616  2^3.7.11         chain3   replayed         642        702    1.07  3.5e-16  
       617  617              prime    replayed        2340       4936    2.08  6.9e-16  rader
       618  2.3.103          prime    replayed        4789       4213    0.86  6.2e-16  bluestein
       619  619              prime    replayed        4809       5002    1.00  6.2e-16  bluestein
       620  2^2.5.31         flat     replayed        1179       2985    2.40  5.8e-16  
       621  3^3.23           2p       replayed         778       2414    3.10  5.7e-16  
       622  2.311            prime    replayed        4810       5201    1.07  5.6e-16  bluestein
       623  7.89             prime    replayed        4754       5169    1.08  5.4e-16  bluestein
       624  2^4.3.13         chain3   replayed         715        677    0.94  4.4e-16  
       625  5^4              2p       replayed         674        832    1.23  4.4e-16  
       626  2.313            prime    replayed        4943       5122    1.03  6.3e-16  bluestein
       627  3.11.19          flat     replayed         962       2364    2.44  6.9e-16  
       628  2^2.157          prime    replayed        4805       3775    0.77  5.1e-16  bluestein
       629  17.37            2p       replayed        1514       4968    2.23  7.8e-16  flips differ 1.47x
       630  2.3^2.5.7        chain3   replayed         705        638    0.87  4.8e-16  
       631  631              prime    replayed        2910       3962    1.33  7.7e-16  rader
       632  2^3.79           prime    replayed        4729       3911    0.81  5.0e-16  bluestein
       633  3.211            prime    replayed        5101       4260    0.82  4.5e-16  bluestein
       634  2.317            prime    replayed        4745       3640    0.75  4.7e-16  bluestein
       635  5.127            prime    replayed        4747       3612    0.76  4.6e-16  bluestein
       636  2^2.3.53         prime    replayed        4842       4736    0.97  3.7e-16  bluestein
       637  7^2.13           flat     replayed         906        993    1.02  8.4e-16  
       638  2.11.29          flat     replayed        1198       3147    2.50  6.5e-16  
       639  3^2.71           prime    replayed        4745       4223    0.89  6.1e-16  bluestein
       640  2^7.5            2p       replayed         502        416    0.79  3.5e-16  
       641  641              prime    replayed        1937       4813    2.45  5.9e-16  rader
       642  2.3.107          prime    replayed        4789       4223    0.86  5.2e-16  bluestein
       643  643              prime    replayed        4810       4821    0.98  6.5e-16  bluestein
       644  2^2.7.23         flat     replayed         989       2455    2.48  4.7e-16  
       645  3.5.43           2p       replayed        1079       4065    3.46  3.7e-16  
       646  2.17.19          flat     replayed        1085       3790    3.24  1.0e-15  
       647  647              prime    replayed        4919       4850    0.97  6.6e-16  bluestein
       648  2^3.3^4          chain3   replayed         633        656    1.03  5.4e-16  
       649  11.59            prime    replayed        4811       5034    1.04  3.8e-16  bluestein
       650  2.5^2.13         chain3   replayed         763        763    0.97  5.5e-16  
       651  3.7.31           2p       replayed        1317       3214    2.00  5.1e-16  
       652  2^2.163          prime    replayed        4804       4518    0.93  5.2e-16  bluestein
       653  653              prime    replayed        4784       5246    1.09  6.0e-16  bluestein
       654  2.3.109          prime    replayed        6291       5500    0.87  5.5e-16  bluestein
       655  5.131            prime    replayed        4831       4532    0.93  5.7e-16  bluestein
       656  2^4.41           2p       replayed         908       3892    4.28  3.9e-16  
       657  3^2.73           prime    replayed        4760       4621    0.96  6.4e-16  bluestein
       658  2.7.47           flat     replayed        1542       4498    2.83  5.8e-16  
       659  659              prime    replayed        4783       5268    1.10  6.7e-16  bluestein
       660  2^2.3.5.11       chain3   replayed         685        708    0.98  5.8e-16  
       661  661              prime    replayed        2963       5293    1.75  6.7e-16  rader
       662  2.331            prime    replayed        4950       5450    1.10  7.1e-16  bluestein
       663  3.13.17          flat     replayed        1030       2450    2.34  8.9e-16  
       664  2^3.83           prime    replayed        4785       4655    0.96  5.3e-16  bluestein
       665  5.7.19           flat     replayed         961       2305    2.05  4.0e-16  
       666  2.3^2.37         chain3   replayed        1239       3710    2.99  4.8e-16  
       667  23.29            2p       replayed        1289       4947    3.82  5.4e-16  
       668  2^2.167          prime    replayed        4791       4533    0.90  4.8e-16  bluestein
       669  3.223            prime    replayed        4834       5366    1.10  6.2e-16  bluestein
       670  2.5.67           prime    replayed        4749       6215    1.24  4.8e-16  bluestein
       671  11.61            prime    replayed        4880       5792    1.18  7.1e-16  bluestein
       672  2^5.3.7          2p       replayed         568        561    0.98  3.3e-16  
       673  673              prime    replayed        2507       5262    2.08  6.3e-16  rader
       674  2.337            prime    replayed        4807       5464    1.13  7.9e-16  bluestein
       675  3^3.5^2          2p       replayed         759        841    1.10  5.5e-16  
       676  2^2.13^2         chain3   replayed         958       1142    1.15  7.5e-16  
       677  677              prime    replayed        3008       4983    1.62  8.9e-16  rader
       678  2.3.113          prime    replayed        4749       5633    1.16  4.2e-16  bluestein
       679  7.97             prime    replayed        4735       4710    0.80  5.1e-16  bluestein
       680  2^3.5.17         flat     replayed         886       2163    2.40  7.4e-16  
       681  3.227            prime    replayed        4874       4592    0.94  6.2e-16  bluestein
       682  2.11.31          flat     replayed        1286       3588    2.65  5.8e-16  
       683  683              prime    replayed        4808       4920    1.02  4.8e-16  bluestein
       684  2^2.3^2.19       chain3   replayed         888       2295    2.46  4.7e-16  
       685  5.137            prime    replayed        4753       5751    1.20  5.5e-16  bluestein
       686  2.7^3            flat     replayed         955        743    0.77  4.1e-16  
       687  3.229            prime    replayed        4760       4611    0.94  5.8e-16  bluestein
       688  2^4.43           flat     replayed        1245       4508    3.56  5.8e-16  
       689  13.53            prime    replayed        4848       5364    1.10  5.7e-16  bluestein
       690  2.3.5.23         chain3   replayed        1127       2636    2.29  4.8e-16  
       691  691              prime    replayed        3477       4952    1.36  8.6e-16  rader
       692  2^2.173          prime    replayed        4772       4582    0.95  5.5e-16  bluestein
       693  3^2.7.11         flat     replayed         880        947    1.07  6.0e-16  
       694  2.347            prime    replayed        4859       4603    0.93  6.5e-16  bluestein
       695  5.139            prime    replayed        4861       4398    0.89  4.7e-16  bluestein
       696  2^3.3.29         flat     replayed        1191       3280    2.40  4.8e-16  
       697  17.41            2p       replayed        1285       5847    4.20  6.4e-16  
       698  2.349            prime    replayed        4848       4614    0.95  5.5e-16  bluestein
       699  3.233            prime    replayed        4857       4627    0.93  5.9e-16  bluestein
       700  2^2.5^2.7        chain3   replayed         703        605    0.82  3.2e-16  
       701  701              prime    replayed        3617       5018    1.32  8.3e-16  rader
       702  2.3^3.13         chain3   replayed         900        860    0.91  5.0e-16  
       703  19.37            2p       replayed        1228       5661    4.01  4.6e-16  
       704  2^6.11           2p       replayed         659        746    1.11  3.1e-16  
       705  3.5.47           2p       replayed        1249       4967    3.96  5.3e-16  
       706  2.353            prime    replayed        4764       4612    0.92  7.3e-16  bluestein
       707  7.101            prime    replayed        4807       4969    0.99  6.8e-16  bluestein
       708  2^2.3.59         prime    replayed        4899       5773    1.17  5.7e-16  bluestein
       709  709              prime    replayed        4877       5057    1.03  5.0e-16  bluestein
       710  2.5.71           prime    replayed        4798       4760    0.98  5.7e-16  bluestein
       711  3^2.79           prime    replayed        4849       4643    0.94  5.1e-16  bluestein
       712  2^3.89           prime    replayed        4806       4677    0.95  6.4e-16  bluestein
       713  23.31            2p       replayed        2177       5581    2.49  6.7e-16  
       714  2.3.7.17         chain3   replayed        1008       2336    2.17  6.2e-16  
       715  5.11.13          flat     replayed         927       1284    1.32  5.6e-16  
       716  2^2.179          prime    replayed        4834       4590    0.91  7.0e-16  bluestein
       717  3.239            prime    replayed        4870       4638    0.95  6.2e-16  bluestein
       718  2.359            prime    replayed        4801       4631    0.96  5.7e-16  bluestein
       719  719              prime    replayed        4769       5098    1.06  5.2e-16  bluestein
       720  2^4.3^2.5        chain3   replayed         775        603    0.77  4.8e-16  
       721  7.103            prime    replayed        4813       4949    1.02  6.3e-16  bluestein
       722  2.19^2           flat     replayed        1317       4378    3.30  1.0e-15  
       723  3.241            prime    replayed        4784       7444    1.55  5.6e-16  bluestein
       724  2^2.181          prime    replayed        4757       5953    1.24  5.8e-16  bluestein
       725  5^2.29           flat     replayed        1240       3387    2.66  5.7e-16  
       726  2.3.11^2         chain3   replayed         991       1187    1.20  4.8e-16  
       727  727              prime    replayed        3092       6440    2.08  7.1e-16  rader
       728  2^3.7.13         chain3   replayed         822        863    1.01  5.4e-16  
       729  3^6              2p       replayed         839       1077    1.28  4.5e-16  
       730  2.5.73           prime    replayed        4751       5036    1.05  6.1e-16  bluestein
       731  17.43            2p       replayed        1354       6302    4.63  5.2e-16  
       732  2^2.3.61         prime    replayed        4772       6168    1.29  4.5e-16  bluestein
       733  733              prime    replayed        4928       5899    1.18  5.1e-16  bluestein
       734  2.367            prime    replayed        4840       5579    1.15  5.6e-16  bluestein
       735  3.5.7^2          flat     replayed        1002        919    0.91  5.4e-16  
       736  2^5.23           2p       replayed         778       2731    3.49  4.1e-16  
       737  11.67            prime    replayed        4774       7125    1.48  5.5e-16  bluestein
       738  2.3^2.41         chain3   replayed        3025       4452    1.30  5.1e-16  
       739  739              prime    replayed        4787       5905    1.23  5.1e-16  bluestein
       740  2^2.5.37         flat     replayed        1308       4358    3.33  5.4e-16  
       741  3.13.19          chain3   replayed        1632       2846    1.34  6.2e-16  flips differ 1.30x
       742  2.7.53           prime    replayed        4864       5785    1.19  3.3e-16  bluestein
       743  743              prime    replayed        4882       5966    1.22  5.3e-16  bluestein
       744  2^3.3.31         flat     replayed        1333       3580    2.28  4.5e-16  
       745  5.149            prime    replayed        4775       5220    1.09  6.1e-16  bluestein
       746  2.373            prime    replayed        4867       5588    1.13  6.2e-16  bluestein
       747  3^2.83           prime    replayed        4931       5343    1.06  6.8e-16  bluestein
       748  2^2.11.17        flat     replayed        1209       2726    2.22  7.5e-16  
       749  7.107            prime    replayed        4867       6878    1.40  5.3e-16  bluestein
       750  2.3.5^3          chain3   replayed         944        792    0.83  4.0e-16  
       751  751              prime    replayed        4838       5077    1.02  5.0e-16  bluestein
       752  2^4.47           flat     replayed        1464       5021    3.41  5.8e-16  
       753  3.251            prime    replayed        4889       4468    0.91  5.1e-16  bluestein
       754  2.13.29          flat     replayed        1428       3710    2.53  9.7e-16  
       755  5.151            prime    replayed        4875       4716    0.97  4.7e-16  bluestein
       756  2^2.3^3.7        chain3   replayed         758        771    1.01  4.2e-16  
       757  757              prime    replayed        4147       5141    1.12  5.7e-16  rader
       758  2.379            prime    replayed        4832       4675    0.95  5.7e-16  bluestein
       759  3.11.23          flat     replayed        1393       3176    1.81  5.8e-16  flips differ 1.26x
       760  2^3.5.19         flat     replayed        1012       2541    2.42  4.9e-16  
       761  761              prime    replayed        3383       5080    1.47  8.0e-16  rader
       762  2.3.127          prime    replayed        4758       4503    0.93  4.7e-16  bluestein
       763  7.109            prime    replayed        4752       6690    1.39  5.4e-16  bluestein
       764  2^2.191          prime    replayed        4754       4335    0.89  4.4e-16  bluestein
       765  3^2.5.17         flat     replayed        1063       2545    2.33  6.0e-16  
       766  2.383            prime    replayed        4746       4639    0.97  4.8e-16  bluestein
       767  13.59            prime    replayed        4810       6543    1.33  3.8e-16  bluestein
       768  2^8.3            2p       replayed         620        518    0.83  3.8e-16  
       769  769              prime    replayed        2321       5310    2.21  6.4e-16  rader
       770  2.5.7.11         chain3   replayed         984        923    0.93  3.5e-16  
       771  3.257            prime    replayed        4857       7042    1.44  7.3e-16  bluestein
       772  2^2.193          prime    replayed        4750       4782    1.00  5.1e-16  bluestein
       773  773              prime    replayed        4810       5314    1.08  6.2e-16  bluestein
       774  2.3^2.43         chain3   replayed        1587       4903    2.93  4.9e-16  
       775  5^2.31           2p       replayed        1125       3922    3.47  5.8e-16  
       776  2^3.97           prime    replayed        4758       5414    1.12  3.8e-16  bluestein
       777  3.7.37           2p       replayed        1195       4607    3.84  6.1e-16  
       778  2.389            prime    replayed        4949       5186    0.96  7.3e-16  bluestein
       779  19.41            2p       replayed        1458       6848    4.69  5.0e-16  
       780  2^2.3.5.13       chain3   replayed         879        893    1.01  7.1e-16  
       781  11.71            prime    replayed        4762       5336    0.89  6.5e-16  flips differ 1.25x bluestein
       782  2.17.23          flat     replayed        1409       4965    3.51  1.9e-15  
       783  3^3.29           flat     replayed        1561       3597    2.19  5.0e-16  
       784  2^4.7^2          chain3   replayed         801        716    0.88  5.2e-16  
       785  5.157            prime    replayed        4846       4802    0.97  7.0e-16  bluestein
       786  2.3.131          prime    replayed        4820       5419    1.12  5.0e-16  bluestein
       787  787              prime    replayed        4904       5571    1.08  5.1e-16  bluestein
       788  2^2.197          prime    replayed        4875       4870    0.98  6.7e-16  bluestein
       789  3.263            prime    replayed        4934       7111    1.43  7.5e-16  bluestein
       790  2.5.79           prime    replayed        4807       5050    1.03  5.0e-16  bluestein
       791  7.113            prime    replayed        4841       6765    1.35  6.1e-16  bluestein
       792  2^3.3^2.11       chain3   replayed         848        935    1.09  6.2e-16  
       793  13.61            prime    replayed        4751       6954    1.43  4.8e-16  bluestein
       794  2.397            prime    replayed        4829       5295    1.07  5.8e-16  bluestein
       795  3.5.53           prime    replayed        4814       6020    1.25  4.2e-16  bluestein
       796  2^2.199          prime    replayed        4822       4887    1.01  5.8e-16  bluestein
       797  797              prime    replayed        4781       5229    1.07  4.2e-16  bluestein
       798  2.3.7.19         flat     replayed        1309       2709    2.07  5.0e-16  
       799  17.47            2p       replayed        1595       7296    3.99  5.6e-16  
       800  2^5.5^2          2p       replayed         770        652    0.84  3.1e-16  
       801  3^2.89           prime    replayed        4846       5485    1.12  6.9e-16  bluestein
       802  2.401            prime    replayed        4819       8229    1.70  4.9e-16  bluestein
       803  11.73            prime    replayed        4783       5657    1.17  6.2e-16  bluestein
       804  2^2.3.67         prime    replayed        4764       6959    1.44  5.9e-16  bluestein
       805  5.7.23           flat     replayed        1286       3448    2.57  5.6e-16  
       806  2.13.31          flat     replayed        1611       4315    2.62  9.9e-16  
       807  3.269            prime    replayed        5049       7177    1.37  6.2e-16  bluestein
       808  2^3.101          prime    replayed        4998       5654    1.12  6.7e-16  bluestein
       809  809              prime    replayed        4877       6417    1.31  6.1e-16  bluestein
       810  2.3^4.5          chain3   replayed        1075        915    0.84  5.1e-16  
       811  811              prime    replayed        4884       6420    1.10  5.9e-16  bluestein
       812  2^2.7.29         flat     replayed        1388       3928    2.55  4.9e-16  
       813  3.271            prime    replayed        4781       6563    1.34  5.6e-16  bluestein
       814  2.11.37          flat     replayed        1763       4959    2.80  8.1e-16  
       815  5.163            prime    replayed        4828       7890    1.62  6.2e-16  bluestein
       816  2^4.3.17         chain3   replayed        1264       2855    2.14  5.0e-16  
       817  19.43            flat     replayed        1701       7171    4.21  8.3e-16  
       818  2.409            prime    replayed        4785       5999    1.24  8.9e-16  bluestein
       819  3^2.7.13         flat     replayed        1112       1201    1.07  7.7e-16  
       820  2^2.5.41         flat     replayed        1455       5220    3.59  4.8e-16  
       821  821              prime    replayed        4852       6442    1.32  6.9e-16  bluestein
       822  2.3.137          prime    replayed        4838       5314    1.09  5.4e-16  bluestein
       823  823              prime    replayed        5154       6248    1.21  6.4e-16  bluestein
       824  2^3.103          prime    replayed        4866       5654    1.14  7.9e-16  bluestein
       825  3.5^2.11         chain3   replayed        1186       1231    0.98  4.4e-16  
       826  2.7.59           prime    replayed        4929       6410    1.29  4.5e-16  bluestein
       827  827              prime    replayed        4833       6448    1.33  5.9e-16  bluestein
       828  2^2.3^2.23       chain3   replayed        1252       3154    2.36  5.1e-16  
       829  829              prime    replayed        4216       6354    1.48  1.1e-15  rader
       830  2.5.83           prime    replayed        4844       5943    1.20  6.0e-16  bluestein
       831  3.277            prime    replayed        4774       6568    1.37  7.7e-16  bluestein
       832  2^6.13           2p       replayed         813        941    0.86  4.9e-16  flips differ 1.34x
       833  7^2.17           flat     replayed        1197       2983    1.61  6.4e-16  flips differ 1.55x
       834  2.3.139          prime    replayed        4843       5318    1.09  5.6e-16  bluestein
       835  5.167            prime    replayed        4992       7923    1.57  8.0e-16  bluestein
       836  2^2.11.19        flat     replayed        1333       3153    2.34  7.6e-16  
       837  3^3.31           flat     replayed        1776       4196    2.20  6.5e-16  
       838  2.419            prime    replayed        4881       5993    1.22  5.5e-16  bluestein
       839  839              prime    replayed        4803       6365    1.31  6.0e-16  bluestein
       840  2^3.3.5.7        chain3   replayed         879        758    0.84  4.0e-16  
       841  29^2             flat     replayed        1870       6888    3.38  1.2e-15  
       842  2.421            prime    replayed        4800       5971    1.23  6.8e-16  bluestein
       843  3.281            prime    replayed        4862       6611    1.36  5.9e-16  bluestein
       844  2^2.211          prime    replayed        4802       5675    1.16  5.2e-16  bluestein
       845  5.13^2           flat     replayed        1268       1504    1.17  8.0e-16  
       846  2.3^2.47         flat     replayed        2454       5720    2.32  5.7e-16  
       847  7.11^2           flat     replayed        1132       1501    1.30  5.4e-16  
       848  2^4.53           prime    replayed        4798       6323    1.31  4.4e-16  bluestein
       849  3.283            prime    replayed        4905       6591    1.30  5.0e-16  bluestein
       850  2.5^2.17         chain3   replayed        1373       2833    1.86  4.9e-16  
       851  23.37            2p       replayed        1705       7272    4.22  5.5e-16  
       852  2^2.3.71         prime    replayed        4883       5642    1.13  6.0e-16  bluestein
       853  853              prime    replayed        4890       6383    1.29  8.5e-16  bluestein
       854  2.7.61           prime    replayed        4902       7339    1.41  4.4e-16  bluestein
       855  3^2.5.19         flat     replayed        1293       2999    2.21  5.9e-16  
       856  2^3.107          prime    replayed        4858       5588    1.13  6.9e-16  bluestein
       857  857              prime    replayed        4927       6322    0.99  7.0e-16  flips differ 1.29x bluestein
       858  2.3.11.13        flat     replayed        1389       1521    1.09  9.6e-16  
       859  859              prime    replayed        3755       6430    1.71  8.5e-16  rader
       860  2^2.5.43         flat     replayed        1648       5411    3.26  6.3e-16  
       861  3.7.41           2p       replayed        1443       5529    3.79  4.1e-16  
       862  2.431            prime    replayed        4855       5947    1.20  6.0e-16  bluestein
       863  863              prime    replayed        4804       6383    1.31  7.8e-16  bluestein
       864  2^5.3^3          2p       replayed         814        794    0.96  4.7e-16  
       865  5.173            prime    replayed        4908       8081    1.62  5.6e-16  bluestein
       866  2.433            prime    replayed        4806       8318    1.69  5.2e-16  bluestein
       867  3.17^2           flat     replayed        1408       4915    3.48  1.1e-15  
       868  2^2.7.31         flat     replayed        1794       4259    2.00  5.3e-16  
       869  11.79            prime    replayed        4787       5682    1.18  4.6e-16  bluestein
       870  2.3.5.29         flat     replayed        1819       4220    2.30  5.6e-16  
       871  13.67            prime    replayed        4866       8316    1.71  6.1e-16  bluestein
       872  2^3.109          prime    replayed        4827       7598    1.54  5.5e-16  bluestein
       873  3^2.97           prime    replayed        4825       6138    1.25  5.5e-16  bluestein
       874  2.19.23          flat     replayed        1658       5762    3.20  1.1e-15  
       875  5^3.7            flat     replayed        1101       1181    1.07  5.1e-16  
       876  2^2.3.73         prime    replayed        4875       6387    1.07  4.5e-16  bluestein
       877  877              prime    replayed        4824       6775    1.38  4.5e-16  bluestein
       878  2.439            prime    replayed        4841       8228    1.68  5.3e-16  bluestein
       879  3.293            prime    replayed        4919       6235    1.26  4.4e-16  bluestein
       880  2^4.5.11         chain3   replayed         984        993    1.00  4.3e-16  
       881  881              prime    replayed        3401       6845    1.97  5.1e-16  rader
       882  2.3^2.7^2        flat     replayed        1366        974    0.71  6.0e-16  
       883  883              prime    replayed        4945       6811    1.36  5.5e-16  bluestein
       884  2^2.13.17        chain3   replayed        1511       3331    2.20  7.1e-16  
       885  3.5.59           prime    replayed        5159       7322    1.41  4.0e-16  bluestein
       886  2.443            prime    replayed        4930       8279    1.67  7.0e-16  bluestein
       887  887              prime    replayed        4794       6782    1.40  5.1e-16  bluestein
       888  2^3.3.37         chain3   replayed        1613       5229    3.21  5.5e-16  
       889  7.127            prime    replayed        4887       5207    1.05  5.2e-16  bluestein
       890  2.5.89           prime    replayed        4830       6013    1.24  7.0e-16  bluestein
       891  3^4.11           chain3   replayed        1285       1434    1.08  5.8e-16  
       892  2^2.223          prime    replayed        4778       6945    1.44  5.7e-16  bluestein
       893  19.47            flat     replayed        2037       8334    4.01  8.2e-16  
       894  2.3.149          prime    replayed        4906       5782    1.18  6.2e-16  bluestein
       895  5.179            prime    replayed        4834       7984    1.63  5.4e-16  bluestein
       896  2^7.7            chain3   replayed         944        671    0.70  4.0e-16  
       897  3.13.23          flat     replayed        1628       3877    2.34  9.6e-16  
       898  2.449            prime    replayed        4903       8294    1.69  7.6e-16  bluestein
       899  29.31            2p       replayed        2416       7712    2.67  5.0e-16  
       900  2^2.3^2.5^2      chain3   replayed         986        895    0.90  4.8e-16  
       901  17.53            prime    replayed        5037       8900    1.76  6.0e-16  bluestein
       902  2.11.41          flat     replayed        2122       5827    2.64  6.9e-16  
       903  3.7.43           flat     replayed        1945       5807    2.92  1.9e-15  
       904  2^3.113          prime    replayed        4931       7713    1.53  6.9e-16  bluestein
       905  5.181            prime    replayed        4826       7561    1.55  5.1e-16  bluestein
       906  2.3.151          prime    replayed        4809       5671    1.16  5.0e-16  bluestein
       907  907              prime    replayed        4996       8538    1.69  6.8e-16  bluestein
       908  2^2.227          prime    replayed        4843       6163    1.23  5.5e-16  bluestein
       909  3^2.101          prime    replayed        4959       6321    1.27  7.4e-16  bluestein
       910  2.5.7.13         chain3   replayed        1258       1163    0.92  4.4e-16  
       911  911              prime    replayed        4951       8561    1.73  5.2e-16  bluestein
       912  2^4.3.19         chain3   replayed        1325       3073    2.22  4.7e-16  
       913  11.83            prime    replayed        4863       6702    1.35  5.2e-16  bluestein
       914  2.457            prime    replayed        4835       6304    1.30  5.2e-16  bluestein
       915  3.5.61           prime    replayed        4799       7789    1.61  4.4e-16  bluestein
       916  2^2.229          prime    replayed        5022       6183    1.22  5.1e-16  bluestein
       917  7.131            prime    replayed        4897       6391    1.30  7.9e-16  bluestein
       918  2.3^3.17         flat     replayed        1512       3061    1.95  7.8e-16  
       919  919              prime    replayed        4830       8551    1.76  6.0e-16  bluestein
       920  2^3.5.23         chain3   replayed        1416       3827    2.66  3.7e-16  
       921  3.307            prime    replayed        4839       7806    1.60  6.2e-16  bluestein
       922  2.461            prime    replayed        4930       6249    1.25  6.6e-16  bluestein
       923  13.71            prime    replayed        4905       6397    1.30  4.9e-16  bluestein
       924  2^2.3.7.11       chain3   replayed        1007       1166    1.13  4.6e-16  
       925  5^2.37           2p       replayed        2288       5248    1.53  6.1e-16  flips differ 1.50x
       926  2.463            prime    replayed        5167       6311    1.06  6.0e-16  bluestein
       927  3^2.103          prime    replayed        5007       6402    1.27  6.3e-16  bluestein
       928  2^5.29           2p       replayed        1129       4458    3.43  4.5e-16  
       929  929              prime    replayed        4236       8587    1.79  4.9e-16  rader
       930  2.3.5.31         chain3   replayed        2597       4578    1.28  4.5e-16  flips differ 1.38x
       931  7^2.19           flat     replayed        1454       3506    2.41  6.2e-16  
       932  2^2.233          prime    replayed        4847       6291    1.27  5.8e-16  bluestein
       933  3.311            prime    replayed        4988       7859    1.56  5.7e-16  bluestein
       934  2.467            prime    replayed        4883       6331    1.28  5.7e-16  bluestein
       935  5.11.17          flat     replayed        1405       3484    2.44  6.2e-16  
       936  2^3.3^2.13       chain3   replayed        1082       1146    1.06  6.3e-16  
       937  937              prime    replayed        3700       8581    2.29  7.5e-16  rader
       938  2.7.67           prime    replayed        4929       8379    1.66  6.3e-16  bluestein
       939  3.313            prime    replayed        4943       7813    1.57  5.3e-16  bluestein
       940  2^2.5.47         flat     replayed        1823       6495    3.54  5.2e-16  
       941  941              prime    replayed        4932       6763    1.36  5.6e-16  bluestein
       942  2.3.157          prime    replayed        4950       5795    1.15  5.8e-16  bluestein
       943  23.41            2p       replayed        2055       8541    3.63  4.9e-16  
       944  2^4.59           prime    replayed        4852       7272    1.49  4.6e-16  bluestein
       945  3^3.5.7          chain3   replayed        1352       1288    0.90  5.5e-16  
       946  2.11.43          flat     replayed        2289       6367    2.74  7.5e-16  
       947  947              prime    replayed        5071       6732    1.31  6.7e-16  bluestein
       948  2^2.3.79         prime    replayed        4925       6292    1.27  4.6e-16  bluestein
       949  13.73            prime    replayed        4872       6808    1.37  8.1e-16  bluestein
       950  2.5^2.19         flat     replayed        1657       3300    1.94  4.3e-16  
       951  3.317            prime    replayed        4825       5903    1.20  6.0e-16  bluestein
       952  2^3.7.17         flat     replayed        1325       3115    2.28  8.3e-16  
       953  953              prime    replayed        4157       6750    1.36  8.2e-16  rader
       954  2.3^2.53         prime    replayed        5043       7487    1.48  5.9e-16  bluestein
       955  5.191            prime    replayed        4860       5498    1.12  5.2e-16  bluestein
       956  2^2.239          prime    replayed        4856       6208    1.26  5.9e-16  bluestein
       957  3.11.29          chain3   replayed        1834       4815    2.22  5.4e-16  
       958  2.479            prime    replayed        5150       6296    0.94  6.0e-16  flips differ 1.29x bluestein
       959  7.137            prime    replayed        5049       8150    1.61  5.8e-16  bluestein
       960  2^6.3.5          2p       replayed         952        850    0.85  4.4e-16  
       961  31^2             flat     replayed        2248       8707    3.48  8.2e-16  
       962  2.13.37          flat     replayed        2154       5866    2.68  9.1e-16  
       963  3^2.107          prime    replayed        4864       6460    1.31  6.9e-16  bluestein
       964  2^2.241          prime    replayed        4961       7538    1.46  6.9e-16  bluestein
       965  5.193            prime    replayed        4967       6252    1.19  5.7e-16  bluestein
       966  2.3.7.23         chain3   replayed        1570       3735    2.34  5.2e-16  
       967  967              prime    replayed        4945       8220    1.64  6.3e-16  bluestein
       968  2^3.11^2         chain3   replayed        1260       1618    1.28  3.7e-16  
       969  3.17.19          flat     replayed        1739       5745    3.23  8.3e-16  
       970  2.5.97           prime    replayed        4933       5920    1.15  5.5e-16  bluestein
       971  971              prime    replayed        4950       8056    1.62  8.0e-16  bluestein
       972  2^2.3^5          chain3   replayed        1030       1102    1.07  4.4e-16  
       973  7.139            prime    replayed        5017       6209    1.23  5.9e-16  bluestein
       974  2.487            prime    replayed        4868       8418    1.70  5.4e-16  bluestein
       975  3.5^2.13         flat     replayed        1368       1558    0.88  5.7e-16  flips differ 1.29x
       976  2^4.61           prime    replayed        4873       8507    1.42  4.7e-16  bluestein
       977  977              prime    replayed        4958       7289    1.41  5.4e-16  bluestein
       978  2.3.163          prime    replayed        4828       6807    1.12  7.4e-16  flips differ 1.25x bluestein
       979  11.89            prime    replayed        5027       6781    1.34  6.3e-16  bluestein
       980  2^2.5.7^2        flat     replayed        1232        927    0.75  4.2e-16  
       981  3^2.109          prime    replayed        5054       8516    1.68  7.0e-16  bluestein
       982  2.491            prime    replayed        4850       8402    1.70  5.8e-16  bluestein
       983  983              prime    replayed        4920       7289    1.45  6.7e-16  bluestein
       984  2^3.3.41         chain3   replayed        1902       5965    3.12  4.6e-16  
       985  5.197            prime    replayed        4932       6268    1.22  5.9e-16  bluestein
       986  2.17.29          flat     replayed        2023       6992    3.42  6.8e-16  
       987  3.7.47           2p       replayed        1807       6800    3.72  4.1e-16  
       988  2^2.13.19        chain3   replayed        2408       3833    1.51  5.5e-16  
       989  23.43            flat     replayed        2163       9216    4.09  1.0e-15  
       990  2.3^2.5.11       chain3   replayed        1279       1254    0.97  5.4e-16  
       991  991              prime    replayed        4851       7248    1.48  4.7e-16  bluestein
       992  2^5.31           flat     replayed        1800       5239    2.90  1.4e-15  
       993  3.331            prime    replayed        4819       8261    1.70  6.3e-16  bluestein
       994  2.7.71           prime    replayed        4931       6705    1.32  5.2e-16  bluestein
       995  5.199            prime    replayed        4824       6321    1.30  5.9e-16  bluestein
       996  2^2.3.83         prime    replayed        4934       7039    1.37  6.5e-16  bluestein
       997  997              prime    replayed        4854       7223    1.48  5.9e-16  bluestein
       998  2.499            prime    replayed        4856       8419    1.73  4.6e-16  bluestein
       999  3^3.37           2p       replayed        2543       5738    1.62  5.9e-16  flips differ 1.39x
      1000  2^3.5^3          chain3   replayed        1058        928    0.87  3.3e-16  
      1001  7.11.13          flat     replayed        1543       1892    1.22  5.6e-16  
      1002  2.3.167          prime    replayed        5014       6857    1.36  8.3e-16  bluestein
      1003  17.59            prime    replayed        4818      10714    2.20  5.6e-16  bluestein
      1004  2^2.251          prime    replayed        4855       5862    1.20  4.8e-16  bluestein
      1005  3.5.67           prime    replayed        4892       9666    1.96  5.1e-16  bluestein
      1006  2.503            prime    replayed        4869       6329    1.28  5.9e-16  bluestein
      1007  19.53            prime    replayed        4966      10112    2.00  5.4e-16  bluestein
      1008  2^4.3^2.7        chain3   replayed        1044        968    0.92  5.4e-16  
      1009  1009             prime    replayed        5015       9667    1.92  6.8e-16  bluestein
      1010  2.5.101          prime    replayed        4924       7041    1.25  4.7e-16  bluestein
      1011  3.337            prime    replayed        4852       8630    1.77  6.6e-16  bluestein
      1012  2^2.11.23        chain3   replayed        1693       4298    2.49  5.5e-16  
      1013  1013             prime    replayed        4955       9602    1.93  5.8e-16  bluestein
      1014  2.3.13^2         chain3   replayed        1538       1795    1.16  5.8e-16  
      1015  5.7.29           flat     replayed        2150       4704    2.15  5.0e-16  
      1016  2^3.127          prime    replayed        4846       5685    1.16  5.0e-16  bluestein
      1017  3^2.113          prime    replayed        4992       8533    1.71  5.8e-16  bluestein
      1018  2.509            prime    replayed        4922       6005    1.21  5.0e-16  bluestein
      1019  1019             prime    replayed        4998       6195    1.22  6.2e-16  bluestein
      1020  2^2.3.5.17       chain3   replayed        1540       3367    2.04  4.7e-16  
      1021  1021             prime    replayed        5000       6136    1.21  5.5e-16  bluestein
      1022  2.7.73           prime    replayed        4838       7115    1.45  6.6e-16  bluestein
      1023  3.11.31          chain3   replayed        1987       5505    2.74  5.4e-16  
      1024  2^10             ztt      replayed         717        801    1.10  2.3e-16  
      1025  5^2.41           2p       replayed        1891       6305    3.27  5.0e-16  
      1026  2.3^3.19         chain3   replayed        1626       3581    2.15  5.9e-16  
      1027  13.79            prime    replayed       10086       7026    0.68  4.4e-16  bluestein
      1028  2^2.257          prime    raced          10196       7935    0.78  4.6e-16  bluestein
      1029  3.7^3            flat     replayed        1514       1652    1.09  4.4e-16  
      1030  2.5.103          prime    raced          10069       7181    0.69  5.2e-16  bluestein
      1031  1031             prime    replayed       10041       8395    0.83  5.6e-16  bluestein
      1032  2^3.3.43         chain3   replayed        2069       6546    3.14  4.6e-16  
      1033  1033             prime    replayed        7294       8409    0.99  9.6e-16  rader
      1034  2.11.47          flat     replayed        2628       7374    2.77  8.6e-16  
      1035  3^2.5.23         flat     replayed        1661       4158    2.50  5.7e-16  
      1036  2^2.7.37         flat     replayed        1871       5843    3.10  5.7e-16  
      1037  17.61            prime    replayed       10076      11289    1.09  3.9e-16  bluestein
      1038  2.3.173          prime    raced          10181       6896    0.67  7.0e-16  bluestein
      1039  1039             prime    replayed       10013       8398    0.82  6.8e-16  bluestein
      1040  2^4.5.13         chain3   replayed        1237       1241    1.00  5.6e-16  
      1041  3.347            prime    replayed       10082       7009    0.69  5.1e-16  bluestein
      1042  2.521            prime    raced          10213       7975    0.68  5.4e-16  bluestein
      1043  7.149            prime    replayed       10191       7334    0.60  5.1e-16  bluestein
      1044  2^2.3^2.29       chain3   replayed        1787       4752    2.62  4.9e-16  
      1045  5.11.19          flat     replayed        1731       4029    2.26  5.5e-16  
      1046  2.523            prime    raced          10081       8024    0.78  5.5e-16  bluestein
      1047  3.349            prime    replayed       10098       7096    0.69  5.2e-16  bluestein
      1048  2^3.131          prime    raced          10079       7419    0.73  5.2e-16  bluestein
      1049  1049             prime    replayed       10122       8413    0.83  6.7e-16  bluestein
      1050  2.3.5^2.7        chain3   replayed        1604       1194    0.74  4.3e-16  
      1051  1051             prime    replayed        5112       8396    1.61  7.9e-16  rader
      1052  2^2.263          prime    raced          10086       7961    0.77  4.8e-16  bluestein
      1053  3^4.13           chain3   replayed        1647       1672    1.00  6.9e-16  
      1054  2.17.31          flat     replayed        2282       7938    3.45  1.6e-15  
      1055  5.211            prime    replayed       10084       7235    0.70  6.5e-16  bluestein
      1056  2^5.3.11         chain3   replayed        1257       1248    0.97  4.8e-16  
      1057  7.151            prime    replayed       10042       6673    0.66  4.9e-16  bluestein
      1058  2.23^2           flat     replayed        2248       7515    3.33  1.4e-15  
      1059  3.353            prime    replayed       10085       7079    0.69  4.8e-16  bluestein
      1060  2^2.5.53         prime    raced          10263       8057    0.78  5.2e-16  bluestein
      1061  1061             prime    replayed       10169       8421    0.82  8.2e-16  bluestein
      1062  2.3^2.59         prime    raced          10196       8240    0.80  4.2e-16  bluestein
      1063  1063             prime    replayed       10025       8425    0.82  6.5e-16  bluestein
      1064  2^3.7.19         chain3   replayed        1443       3665    2.35  4.2e-16  
      1065  3.5.71           prime    replayed       10094       7570    0.73  5.6e-16  bluestein
      1066  2.13.41          flat     replayed        2567       7029    2.72  8.6e-16  
      1067  11.97            prime    replayed       10103       6669    0.65  6.2e-16  bluestein
      1068  2^2.3.89         prime    raced          10183       9098    0.89  5.1e-16  bluestein
      1069  1069             prime    replayed       10233       8434    0.82  5.5e-16  bluestein
      1070  2.5.107          prime    raced          10100       7120    0.69  6.2e-16  bluestein
      1071  3^2.7.17         flat     replayed        1816       3714    2.02  5.3e-16  
      1072  2^4.67           prime    raced          10113       9882    0.98  3.2e-16  bluestein
      1073  29.37            2p       replayed        2524      10349    3.23  4.9e-16  flips differ 1.27x
      1074  2.3.179          prime    raced          10145       6922    0.67  6.4e-16  bluestein
      1075  5^2.43           2p       replayed        2029       6899    3.36  4.8e-16  
      1076  2^2.269          prime    raced          10118       7974    0.77  5.0e-16  bluestein
      1077  3.359            prime    replayed       10165       7041    0.68  5.0e-16  bluestein
      1078  2.7^2.11         flat     replayed        1791       1568    0.57  5.5e-16  flips differ 1.53x
      1079  13.83            prime    replayed       10131       7957    0.77  5.6e-16  bluestein
      1080  2^3.3^3.5        chain3   replayed        1164       1074    0.92  4.9e-16  
      1081  23.47            2p       replayed        3627      10664    1.99  5.0e-16  flips differ 1.48x
      1082  2.541            prime    raced          10115      12052    1.17  4.8e-16  bluestein
      1083  3.19^2           chain3   replayed        2423       6863    2.36  3.8e-16  
      1084  2^2.271          prime    raced          10026       8798    0.88  5.7e-16  bluestein
      1085  5.7.31           chain3   replayed        3130       5832    1.29  4.2e-16  flips differ 1.44x
      1086  2.3.181          prime    raced          10266       8855    0.86  5.3e-16  bluestein
      1087  1087             prime    replayed       10088      11231    1.11  7.6e-16  bluestein
      1088  2^6.17           2p       replayed        1251       3495    2.66  5.6e-16  
      1089  3^2.11^2         chain3   replayed        1662       1996    1.18  5.3e-16  
      1090  2.5.109          prime    raced          10048       9481    0.93  4.4e-16  bluestein
      1091  1091             prime    replayed       10124      11263    1.11  6.4e-16  bluestein
      1092  2^2.3.7.13       chain3   replayed        1273       1464    1.15  5.7e-16  
      1093  1093             prime    replayed        5746      11261    1.93  9.5e-16  rader
      1094  2.547            prime    raced          10106      10018    0.97  7.1e-16  bluestein
      1095  3.5.73           prime    replayed       10186       8062    0.79  6.9e-16  bluestein
      1096  2^3.137          prime    raced          10235       7068    0.68  6.1e-16  bluestein
      1097  1097             prime    replayed       10267       9728    0.94  5.3e-16  bluestein
      1098  2.3^2.61         prime    raced          10325       9388    0.91  4.3e-16  bluestein
      1099  7.157            prime    replayed       10113       6896    0.67  5.5e-16  bluestein
      1100  2^2.5^2.11       chain3   replayed        1340       1326    0.97  4.0e-16  
      1101  3.367            prime    replayed       10217       8825    0.86  5.4e-16  bluestein
      1102  2.19.29          flat     replayed        2506       7978    3.05  1.1e-15  
      1103  1103             prime    replayed       10328       9842    0.94  5.3e-16  bluestein
      1104  2^4.3.23         chain3   replayed        1607       4251    2.60  7.1e-16  
      1105  5.13.17          flat     replayed        1755       4193    2.38  6.7e-16  
      1106  2.7.79           prime    raced          10041       7505    0.71  5.2e-16  bluestein
      1107  3^3.41           2p       replayed        1906       6953    3.49  5.2e-16  
      1108  2^2.277          prime    raced          10124       8119    0.78  4.0e-16  bluestein
      1109  1109             prime    replayed       10111       9791    0.97  7.2e-16  bluestein
      1110  2.3.5.37         chain3   replayed        2180       6663    2.97  6.8e-16  
      1111  11.101           prime    replayed       10198       8256    0.60  6.8e-16  flips differ 1.37x bluestein
      1112  2^3.139          prime    raced          10046       7350    0.71  6.1e-16  bluestein
      1113  3.7.53           prime    replayed       10183       8514    0.82  3.4e-16  bluestein
      1114  2.557            prime    raced          10154      10058    0.97  4.9e-16  bluestein
      1115  5.223            prime    replayed       10310       8852    0.85  7.6e-16  bluestein
      1116  2^2.3^2.31       flat     replayed        2127       5535    2.13  5.3e-16  
      1117  1117             prime    replayed        7770       9761    1.24  1.1e-15  rader
      1118  2.13.43          flat     replayed        2829       7625    2.61  1.0e-15  
      1119  3.373            prime    replayed       10066       8671    0.85  6.2e-16  bluestein
      1120  2^5.5.7          chain3   replayed        1161       1078    0.92  4.3e-16  
      1121  19.59            prime    replayed       10261      11422    1.11  4.1e-16  bluestein
      1122  2.3.11.17        chain3   replayed        1804       4225    2.33  5.3e-16  
      1123  1123             prime    replayed        5642       9764    1.66  7.7e-16  rader
      1124  2^2.281          prime    raced          10148       8173    0.80  6.1e-16  bluestein
      1125  3^2.5^3          flat     replayed        1564       1675    1.05  4.7e-16  
      1126  2.563            prime    raced          10068      10008    0.98  6.2e-16  bluestein
      1127  7^2.23           chain3   replayed        2006       4795    2.37  5.2e-16  
      1128  2^3.3.47         flat     replayed        2575       7994    2.52  5.7e-16  
      1129  1129             prime    replayed       10279       8560    0.82  7.4e-16  bluestein
      1130  2.5.113          prime    raced          10263       9497    0.92  4.3e-16  bluestein
      1131  3.13.29          chain3   replayed        2256       5724    2.38  7.9e-16  
      1132  2^2.283          prime    raced          10085       8040    0.78  5.3e-16  bluestein
      1133  11.103           prime    replayed       10130       8081    0.78  5.7e-16  bluestein
      1134  2.3^4.7          chain3   replayed        1643       1371    0.80  5.4e-16  
      1135  5.227            prime    replayed       10149       7825    0.77  5.4e-16  bluestein
      1136  2^4.71           prime    raced          10148       8026    0.79  5.8e-16  bluestein
      1137  3.379            prime    replayed       10133       7530    0.73  5.5e-16  bluestein
      1138  2.569            prime    raced          10210       8300    0.80  4.3e-16  bluestein
      1139  17.67            prime    replayed       10150      13545    1.33  6.5e-16  bluestein
      1140  2^2.3.5.19       chain3   replayed        1552       4366    2.71  5.5e-16  
      1141  7.163            prime    replayed       10283      11076    1.07  5.3e-16  bluestein
      1142  2.571            prime    raced          10182       8194    0.79  4.9e-16  bluestein
      1143  3^2.127          prime    replayed       10404       6509    0.62  4.3e-16  bluestein
      1144  2^3.11.13        chain3   replayed        1520       2067    1.31  5.6e-16  
      1145  5.229            prime    replayed       10238       7822    0.76  4.7e-16  bluestein
      1146  2.3.191          prime    raced          10305       6573    0.63  5.6e-16  bluestein
      1147  31.37            flat     replayed        2544      11251    4.36  8.4e-16  
      1148  2^2.7.41         chain3   replayed        2259       7491    3.29  4.1e-16  
      1149  3.383            prime    replayed       10153       7070    0.50  5.4e-16  flips differ 1.39x bluestein
      1150  2.5^2.23         chain3   replayed        1876       4536    2.41  5.1e-16  
      1151  1151             prime    replayed        6197       8580    1.30  7.6e-16  rader
      1152  2^7.3^2          chain3   replayed        1313       1099    0.83  3.9e-16  
      1153  1153             prime    replayed        4229       9041    2.13  7.2e-16  rader
      1154  2.577            prime    raced          10090      10097    0.99  5.3e-16  bluestein
      1155  3.5.7.11         chain3   replayed        1688       1769    1.04  3.7e-16  
      1156  2^2.17^2         chain3   replayed        2106       6649    3.09  5.8e-16  
      1157  13.89            prime    replayed       10179       8010    0.77  4.3e-16  bluestein
      1158  2.3.193          prime    raced          10106       7640    0.75  4.9e-16  bluestein
      1159  19.61            prime    replayed       10237      12777    1.25  6.2e-16  bluestein
      1160  2^3.5.29         flat     replayed        2100       5360    2.49  8.9e-16  
      1161  3^3.43           2p       replayed        2086       7546    3.61  5.5e-16  
      1162  2.7.83           prime    raced          10293       8507    0.82  5.1e-16  bluestein
      1163  1163             prime    replayed       10505       8849    0.79  5.0e-16  bluestein
      1164  2^2.3.97         prime    raced          10244       8380    0.81  5.4e-16  bluestein
      1165  5.233            prime    replayed       10203       7823    0.75  5.1e-16  bluestein
      1166  2.11.53          prime    raced          10127       9268    0.89  4.5e-16  bluestein
      1167  3.389            prime    replayed       10185       7921    0.77  5.7e-16  bluestein
      1168  2^4.73           prime    raced          10096       8524    0.82  6.1e-16  bluestein
      1169  7.167            prime    replayed       10083      11245    1.11  7.7e-16  bluestein
      1170  2.3^2.5.13       chain3   replayed        1577       1580    1.00  6.0e-16  
      1171  1171             prime    replayed        4834       8904    1.82  7.7e-16  rader
      1172  2^2.293          prime    raced          10216       7975    0.78  5.6e-16  bluestein
      1173  3.17.23          flat     replayed        2398       7722    3.10  8.3e-16  
      1174  2.587            prime    raced          10395      10210    0.97  5.7e-16  bluestein
      1175  5^2.47           flat     replayed        2452       8126    3.28  5.6e-16  
      1176  2^3.3.7^2        chain3   replayed        1510       1243    0.69  3.4e-16  
      1177  11.107           prime    replayed       10150       8185    0.78  5.6e-16  bluestein
      1178  2.19.31          flat     replayed        2738       9096    3.23  1.2e-15  
      1179  3^2.131          prime    replayed       10136       8357    0.81  4.9e-16  bluestein
      1180  2^2.5.59         prime    raced          10260       9222    0.90  4.7e-16  bluestein
      1181  1181             prime    replayed       10195       9015    0.87  6.6e-16  bluestein
      1182  2.3.197          prime    raced          10198       7091    0.68  5.3e-16  bluestein
      1183  7.13^2           chain3   replayed        1929       2228    0.93  8.3e-16  
      1184  2^5.37           2p       replayed        2661       6619    1.69  3.7e-16  flips differ 1.47x
      1185  3.5.79           prime    replayed       10250       7917    0.65  4.9e-16  bluestein
      1186  2.593            prime    raced          10220      10031    0.97  5.5e-16  bluestein
      1187  1187             prime    replayed       10239       8918    0.85  6.5e-16  bluestein
      1188  2^2.3^3.11       chain3   replayed        1340       1557    1.13  5.0e-16  
      1189  29.41            2p       replayed        3146      11691    2.58  5.0e-16  flips differ 1.44x
      1190  2.5.7.17         chain3   replayed        1770       4059    2.28  6.0e-16  
      1191  3.397            prime    replayed       10408       8480    0.81  6.1e-16  bluestein
      1192  2^3.149          prime    raced          10163       8493    0.83  6.0e-16  bluestein
      1193  1193             prime    replayed       10106       8990    0.88  7.0e-16  bluestein
      1194  2.3.199          prime    raced          10306       7265    0.70  5.0e-16  bluestein
      1195  5.239            prime    replayed       10161       7860    0.76  4.6e-16  bluestein
      1196  2^2.13.23        chain3   replayed        2115       5230    2.46  5.4e-16  
      1197  3^2.7.19         flat     replayed        2219       4335    1.86  1.0e-15  
      1198  2.599            prime    raced          10225      10043    0.96  6.9e-16  bluestein
      1199  11.109           prime    replayed       10208      10738    1.04  5.7e-16  bluestein
      1200  2^4.3.5^2        chain3   replayed        1691       1186    0.70  5.0e-16  
      1201  1201             prime    replayed        6526      13031    1.74  9.6e-16  rader
      1202  2.601            prime    raced          10264      13109    1.28  5.8e-16  bluestein
      1203  3.401            prime    replayed       10223      12725    1.21  5.5e-16  bluestein
      1204  2^2.7.43         flat     replayed        2470       8062    3.21  6.5e-16  
      1205  5.241            prime    replayed       10281      12546    1.22  5.3e-16  bluestein
      1206  2.3^2.67         prime    raced          10261      11306    1.09  6.1e-16  bluestein
      1207  17.71            prime    replayed       10087      11378    1.11  5.7e-16  bluestein
      1208  2^3.151          prime    raced          10171       7793    0.74  4.4e-16  bluestein
      1209  3.13.31          chain3   replayed        2473       6680    2.67  5.6e-16  
      1210  2.5.11^2         chain3   replayed        1650       2105    1.27  3.3e-16  
      1211  7.173            prime    replayed       10269      11215    1.08  6.2e-16  bluestein
      1212  2^2.3.101        prime    raced          10236       8673    0.83  5.2e-16  bluestein
      1213  1213             prime    replayed       10151      10409    1.00  7.5e-16  bluestein
      1214  2.607            prime    raced          10237      11079    1.08  5.4e-16  bluestein
      1215  3^5.5            chain3   replayed        1739       1751    1.00  6.2e-16  
      1216  2^6.19           2p       replayed        1535       4138    2.38  3.4e-16  
      1217  1217             prime    replayed        5147      10444    2.01  7.3e-16  rader
      1218  2.3.7.29         flat     replayed        2650       5624    2.10  7.8e-16  
      1219  23.53            prime    replayed       10389      12853    1.24  5.2e-16  bluestein
      1220  2^2.5.61         prime    raced          10353      10468    1.01  4.9e-16  bluestein
      1221  3.11.37          chain3   replayed        2681       7559    2.74  4.7e-16  
      1222  2.13.47          flat     replayed        3221       8646    2.65  7.6e-16  
      1223  1223             prime    replayed       10407      10396    0.76  5.5e-16  flips differ 1.32x bluestein
      1224  2^3.3^2.17       chain3   replayed        1555       4101    2.60  5.4e-16  
      1225  5^2.7^2          flat     replayed        1801       1856    1.01  6.0e-16  
      1226  2.613            prime    raced          10253      10193    0.98  6.4e-16  bluestein
      1227  3.409            prime    replayed       10256       8942    0.86  6.8e-16  bluestein
      1228  2^2.307          prime    raced          10167      10464    1.02  4.6e-16  bluestein
      1229  1229             prime    replayed       10201      10393    1.02  4.7e-16  bluestein
      1230  2.3.5.41         chain3   replayed        2554       7636    2.97  4.4e-16  
      1231  1231             prime    replayed       10241      10408    0.76  5.0e-16  flips differ 1.33x bluestein
      1232  2^4.7.11         chain3   replayed        1390       1598    1.15  4.1e-16  
      1233  3^2.137          prime    replayed       10278       8119    0.57  5.0e-16  flips differ 1.39x bluestein
      1234  2.617            prime    raced          10278      10166    0.98  7.5e-16  bluestein
      1235  5.13.19          flat     replayed        2190       4934    2.25  6.7e-16  
      1236  2^2.3.103        prime    raced          10168       8695    0.83  4.8e-16  bluestein
      1237  1237             prime    replayed       10261      10390    1.01  5.7e-16  bluestein
      1238  2.619            prime    raced          10144      10174    1.00  5.7e-16  bluestein
      1239  3.7.59           prime    replayed       10198      10405    1.00  3.6e-16  bluestein
      1240  2^3.5.31         chain3   replayed        2148       6144    2.77  5.0e-16  
      1241  17.73            prime    replayed       10245      11876    1.14  5.9e-16  bluestein
      1242  2.3^3.23         flat     replayed        2525       5360    2.11  8.6e-16  
      1243  11.113           prime    replayed       10228      10680    1.01  5.4e-16  bluestein
      1244  2^2.311          prime    raced          10268      10600    1.03  5.7e-16  bluestein
      1245  3.5.83           prime    replayed       10253       9011    0.88  3.7e-16  bluestein
      1246  2.7.89           prime    raced          10216       8483    0.83  4.7e-16  bluestein
      1247  29.43            flat     replayed        3729      12594    3.36  2.8e-15  
      1248  2^5.3.13         chain3   replayed        1410       1578    0.94  5.1e-16  
      1249  1249             prime    replayed        5330      10347    1.93  8.6e-16  rader
      1250  2.5^4            chain3   replayed        2192       1573    0.71  5.1e-16  
      1251  3^2.139          prime    replayed       10278       8096    0.79  4.7e-16  bluestein
      1252  2^2.313          prime    raced          10300      10466    1.01  6.1e-16  bluestein
      1253  7.179            prime    replayed       10303      11227    1.09  6.2e-16  bluestein
      1254  2.3.11.19        chain3   replayed        2602       4908    1.78  5.4e-16  
      1255  5.251            prime    replayed       10204       7445    0.71  5.1e-16  bluestein
      1256  2^3.157          prime    raced          10303       7651    0.73  5.4e-16  bluestein
      1257  3.419            prime    replayed       10161       9102    0.88  5.7e-16  bluestein
      1258  2.17.37          flat     replayed        3025      10335    3.41  1.0e-15  
      1259  1259             prime    replayed       10249       9411    0.89  5.4e-16  bluestein
      1260  2^2.3^2.5.7      chain3   replayed        1404       1361    0.96  5.4e-16  
      1261  13.97            prime    replayed       10204       7988    0.76  3.9e-16  bluestein
      1262  2.631            prime    raced          10237       8258    0.80  4.6e-16  bluestein
      1263  3.421            prime    replayed       10211       8981    0.86  6.0e-16  bluestein
      1264  2^4.79           prime    raced          10412       8481    0.81  4.8e-16  bluestein
      1265  5.11.23          flat     replayed        2382       5489    2.21  8.0e-16  
      1266  2.3.211          prime    raced          10151       8588    0.83  7.3e-16  bluestein
      1267  7.181            prime    replayed       10295      10730    1.04  6.0e-16  bluestein
      1268  2^2.317          prime    raced          10160       7451    0.73  4.9e-16  bluestein
      1269  3^3.47           flat     replayed        3115       8936    2.86  6.4e-16  
      1270  2.5.127          prime    raced          10199       7236    0.70  5.5e-16  bluestein
      1271  31.41            flat     replayed        2961      13077    4.39  1.1e-15  
      1272  2^3.3.53         prime    raced          10191      10071    0.99  4.0e-16  bluestein
      1273  19.67            prime    replayed       10231      15324    1.49  4.6e-16  bluestein
      1274  2.7^2.13         flat     replayed        2281       2161    0.86  5.3e-16  
      1275  3.5^2.17         flat     replayed        2233       4537    1.90  6.4e-16  
      1276  2^2.11.29        chain3   replayed        2331       6487    2.36  4.1e-16  
      1277  1277             prime    replayed        7602       9380    1.10  7.0e-16  rader
      1278  2.3^2.71         prime    raced          10267       8686    0.84  5.5e-16  bluestein
      1279  1279             prime    replayed       10204       9433    0.91  6.4e-16  bluestein
      1280  2^8.5            chain3   replayed        1670       1081    0.60  4.3e-16  
      1281  3.7.61           prime    replayed       10195      11114    0.84  3.4e-16  flips differ 1.29x bluestein
      1282  2.641            prime    raced          10210      13454    1.29  6.0e-16  bluestein
      1283  1283             prime    replayed       10195      10833    1.05  7.1e-16  bluestein
      1284  2^2.3.107        prime    raced          10218       8902    0.85  5.3e-16  bluestein
      1285  5.257            prime    replayed       10211      11828    1.13  5.7e-16  bluestein
      1286  2.643            prime    raced          10208       9865    0.96  6.2e-16  bluestein
      1287  3^2.11.13        chain3   replayed        2070       2576    1.23  5.3e-16  
      1288  2^3.7.23         chain3   replayed        1909       5062    2.65  5.0e-16  
      1289  1289             prime    replayed        6578      10796    1.58  6.7e-16  rader
      1290  2.3.5.43         flat     replayed        3499       8686    2.48  7.3e-16  
      1291  1291             prime    replayed       10278      10842    1.05  6.9e-16  bluestein
      1292  2^2.17.19        chain3   replayed        2453       7701    2.86  4.9e-16  
      1293  3.431            prime    replayed       10170       9061    0.87  6.8e-16  bluestein
      1294  2.647            prime    raced          10187      10114    0.97  6.6e-16  bluestein
      1295  5.7.37           flat     replayed        2493       7894    3.12  7.7e-16  
      1296  2^4.3^4          chain3   replayed        1381       1433    1.01  5.4e-16  
      1297  1297             prime    replayed        7787      11123    1.29  8.4e-16  rader
      1298  2.11.59          prime    raced          10463       8707    0.82  6.1e-16  bluestein
      1299  3.433            prime    replayed       10168      12605    1.20  5.9e-16  bluestein
      1300  2^2.5^2.13       flat     replayed        1737       1669    0.72  5.8e-16  flips differ 1.33x
      1301  1301             prime    replayed        7585      11010    1.40  8.6e-16  rader
      1302  2.3.7.31         chain3   replayed        3693       6520    1.76  5.5e-16  
      1303  1303             prime    replayed       10251      11105    1.06  5.6e-16  bluestein
      1304  2^3.163          prime    raced          10252       9275    0.90  4.4e-16  bluestein
      1305  3^2.5.29         chain3   replayed        2584       6202    2.40  7.0e-16  
      1306  2.653            prime    raced          10384      10898    1.05  5.8e-16  bluestein
      1307  1307             prime    replayed       10472      11047    1.05  5.4e-16  bluestein
      1308  2^2.3.109        prime    raced          10278      11314    1.08  4.9e-16  bluestein
      1309  7.11.17          flat     replayed        2395       5084    2.08  9.4e-16  
      1310  2.5.131          prime    raced          10267       9169    0.89  5.4e-16  bluestein
      1311  3.19.23          chain3   replayed        3029       9002    2.82  5.6e-16  
      1312  2^5.41           2p       replayed        2074       7964    3.82  4.2e-16  
      1313  13.101           prime    replayed       10272       9681    0.71  6.1e-16  flips differ 1.32x bluestein
      1314  2.3^2.73         prime    raced          10442       9649    0.92  7.4e-16  bluestein
      1315  5.263            prime    replayed       10276      12119    1.15  6.1e-16  bluestein
      1316  2^2.7.47         chain3   replayed        4637       9131    1.34  4.9e-16  flips differ 1.46x
      1317  3.439            prime    replayed       10224      12683    1.23  5.2e-16  bluestein
      1318  2.659            prime    raced          10193      11261    1.04  4.8e-16  bluestein
      1319  1319             prime    replayed       10289      11187    1.08  6.2e-16  bluestein
      1320  2^3.3.5.11       chain3   replayed        1591       1614    0.97  4.6e-16  
      1321  1321             prime    replayed        5848      11227    1.87  6.1e-16  rader
      1322  2.661            prime    raced          10116      10909    1.07  6.7e-16  bluestein
      1323  3^3.7^2          flat     replayed        2005       2157    1.05  8.3e-16  
      1324  2^2.331          prime    raced          10266      11111    1.06  6.4e-16  bluestein
      1325  5^2.53           prime    replayed       10189      10194    1.00  4.1e-16  bluestein
      1326  2.3.13.17        chain3   replayed        2174       5078    2.25  6.4e-16  
      1327  1327             prime    replayed        6507      11166    1.66  9.2e-16  rader
      1328  2^4.83           prime    raced          10173       9528    0.94  4.8e-16  bluestein
      1329  3.443            prime    replayed       10185      12505    1.20  5.1e-16  bluestein
      1330  2.5.7.19         chain3   replayed        2088       4732    2.16  5.7e-16  
      1331  11^3             chain3   replayed        2155       2791    1.29  4.5e-16  
      1332  2^2.3^2.37       chain3   replayed        2461       8070    3.27  5.5e-16  
      1333  31.43            2p       replayed        4853      14050    2.83  6.3e-16  
      1334  2.23.29          flat     replayed        3406      10817    3.13  1.5e-15  
      1335  3.5.89           prime    replayed       10285      11725    1.11  5.8e-16  bluestein
      1336  2^3.167          prime    raced          10313       9163    0.88  5.0e-16  bluestein
      1337  7.191            prime    replayed       10326       7687    0.73  4.6e-16  bluestein
      1338  2.3.223          prime    raced          10194      10305    0.98  5.9e-16  bluestein
      1339  13.103           prime    replayed       10269       9811    0.95  5.6e-16  bluestein
      1340  2^2.5.67         prime    raced          10284      12470    1.21  6.3e-16  bluestein
      1341  3^2.149          prime    replayed       10488       9864    0.94  5.4e-16  bluestein
      1342  2.11.61          prime    raced          10260      11934    1.16  5.7e-16  bluestein
      1343  17.79            prime    replayed       10257      11998    1.14  6.0e-16  bluestein
      1344  2^6.3.7          2p       replayed        1388       1368    0.97  3.8e-16  
      1345  5.269            prime    replayed       10200      11982    1.15  5.1e-16  bluestein
      1346  2.673            prime    raced          10118      10872    1.06  6.3e-16  bluestein
      1347  3.449            prime    replayed       10209      12682    1.23  5.4e-16  bluestein
      1348  2^2.337          prime    raced          10222      11171    1.07  5.3e-16  bluestein
      1349  19.71            prime    replayed       10138      12888    1.23  7.3e-16  bluestein
      1350  2.3^3.5^2        chain3   replayed        1750       1701    0.94  5.5e-16  
      1351  7.193            prime    replayed       10426       8934    0.75  6.1e-16  bluestein
      1352  2^3.13^2         chain3   replayed        1922       2473    1.28  6.9e-16  
      1353  3.11.41          chain3   replayed        3141       8893    2.77  4.9e-16  
      1354  2.677            prime    raced          10197      10533    1.01  6.4e-16  bluestein
      1355  5.271            prime    replayed       10245      11122    1.06  5.8e-16  bluestein
      1356  2^2.3.113        prime    raced          10210      11389    1.08  6.0e-16  bluestein
      1357  23.59            prime    replayed       10568      15298    1.42  4.5e-16  bluestein
      1358  2.7.97           prime    raced          10255       9951    0.94  4.5e-16  bluestein
      1359  3^2.151          prime    replayed       10458       8734    0.83  5.3e-16  bluestein
      1360  2^4.5.17         chain3   replayed        1760       4555    2.40  6.2e-16  
      1361  1361             prime    replayed        6282      11079    1.43  8.4e-16  rader
      1362  2.3.227          prime    raced          10295       9362    0.91  5.1e-16  bluestein
      1363  29.47            2p       replayed        6338      14399    2.19  4.5e-16  
      1364  2^2.11.31        flat     replayed        3050       7445    2.42  7.3e-16  
      1365  3.5.7.13         chain3   replayed        2304       2228    0.96  5.2e-16  
      1366  2.683            prime    raced          10281      10472    1.01  6.6e-16  bluestein
      1367  1367             prime    replayed       10266      11062    1.08  6.0e-16  bluestein
      1368  2^3.3^2.19       chain3   replayed        1829       4781    2.39  5.9e-16  
      1369  37^2             2p       replayed        4158      14384    3.45  6.8e-16  
      1370  2.5.137          prime    raced          10287       8915    0.87  4.5e-16  bluestein
      1371  3.457            prime    replayed       10229       9596    0.94  6.9e-16  bluestein
      1372  2^2.7^3          flat     replayed        2597       1779    0.67  4.5e-16  
      1373  1373             prime    replayed       10238      11117    1.05  5.1e-16  bluestein
      1374  2.3.229          prime    raced          10365       9356    0.90  4.5e-16  bluestein
      1375  5^3.11           chain3   replayed        2357       2538    1.02  4.2e-16  
      1376  2^5.43           2p       replayed        2249       8679    3.79  4.7e-16  
      1377  3^4.17           chain3   replayed        2483       4921    1.73  6.9e-16  
      1378  2.13.53          prime    raced          10339      11065    1.07  4.6e-16  bluestein
      1379  7.197            prime    replayed       10208       8998    0.85  5.0e-16  bluestein
      1380  2^2.3.5.23       chain3   replayed        2014       5884    2.91  4.6e-16  
      1381  1381             prime    replayed        8001      11084    1.37  9.6e-16  rader
      1382  2.691            prime    raced          10351      10389    0.99  5.7e-16  bluestein
      1383  3.461            prime    replayed       10387       9590    0.92  7.2e-16  bluestein
      1384  2^3.173          prime    raced          10226       9288    0.89  7.1e-16  bluestein
      1385  5.277            prime    replayed       10261      11142    1.08  6.7e-16  bluestein
      1386  2.3^2.7.11       chain3   replayed        2167       1901    0.81  4.6e-16  
      1387  19.73            prime    replayed       10304      13513    1.31  5.9e-16  bluestein
      1388  2^2.347          prime    raced          10279       9488    0.92  5.8e-16  bluestein
      1389  3.463            prime    replayed       10234       9619    0.63  5.9e-16  flips differ 1.50x bluestein
      1390  2.5.139          prime    raced          10243       8879    0.86  5.0e-16  bluestein
      1391  13.107           prime    replayed       10404      10030    0.95  5.2e-16  bluestein
      1392  2^4.3.29         flat     replayed        2562       6845    2.33  5.7e-16  
      1393  7.199            prime    replayed       10300       8995    0.87  5.4e-16  bluestein
      1394  2.17.41          flat     replayed        3502      12183    3.38  1.2e-15  
      1395  3^2.5.31         flat     replayed        3206       7192    2.04  6.2e-16  
      1396  2^2.349          prime    raced          10179       9475    0.93  5.9e-16  bluestein
      1397  11.127           prime    replayed       10295       8045    0.77  5.2e-16  bluestein
      1398  2.3.233          prime    raced          10467       9424    0.90  5.4e-16  bluestein
      1399  1399             prime    replayed       10156      11114    1.07  5.8e-16  bluestein
      1400  2^3.5^2.7        chain3   replayed        1889       1480    0.78  4.8e-16  
      1401  3.467            prime    replayed       10308       9631    0.92  4.8e-16  bluestein
      1402  2.701            prime    raced          10303      10369    0.99  6.5e-16  bluestein
      1403  23.61            prime    replayed       10303      16126    1.56  4.3e-16  bluestein
      1404  2^2.3^3.13       chain3   replayed        1696       1939    1.14  5.7e-16  
      1405  5.281            prime    replayed       10242      11181    1.09  5.9e-16  bluestein
      1406  2.19.37          flat     replayed        3784      11887    3.07  1.5e-15  
      1407  3.7.67           prime    replayed       10250      12617    1.21  5.6e-16  bluestein
      1408  2^7.11           chain3   replayed        1574       1768    1.09  4.0e-16  
      1409  1409             prime    replayed        5513      11103    1.98  8.5e-16  rader
      1410  2.3.5.47         flat     replayed        3908       9759    2.49  5.6e-16  
      1411  17.83            prime    replayed       10342      13319    1.28  5.9e-16  bluestein
      1412  2^2.353          prime    raced          10388       9452    0.91  7.4e-16  bluestein
      1413  3^2.157          prime    replayed       10356       8490    0.82  6.3e-16  bluestein
      1414  2.7.101          prime    raced          10285      10238    0.97  5.5e-16  bluestein
      1415  5.283            prime    replayed       10345      11191    1.08  5.8e-16  bluestein
      1416  2^3.3.59         prime    raced          10435      11885    1.13  4.8e-16  bluestein
      1417  13.109           prime    replayed       10310      12613    1.20  7.4e-16  bluestein
      1418  2.709            prime    raced          10545      10364    0.97  7.5e-16  bluestein
      1419  3.11.43          flat     replayed        3357       9692    2.85  6.0e-16  
      1420  2^2.5.71         prime    raced          10317       9573    0.93  5.5e-16  bluestein
      1421  7^2.29           chain3   replayed        2905       6724    2.25  5.7e-16  
      1422  2.3^2.79         prime    raced          10224       9076    0.88  5.3e-16  bluestein
      1423  1423             prime    replayed       10265      11096    1.06  5.5e-16  bluestein
      1424  2^4.89           prime    raced          10233       9725    0.63  6.9e-16  flips differ 1.49x bluestein
      1425  3.5^2.19         flat     replayed        2304       5389    2.33  5.1e-16  
      1426  2.23.31          flat     replayed        3433      11719    3.27  1.5e-15  
      1427  1427             prime    replayed       10314      11104    0.78  4.6e-16  flips differ 1.39x bluestein
      1428  2^2.3.7.17       chain3   replayed        1885       5034    2.43  5.6e-16  
      1429  1429             prime    replayed        8106      11166    1.33  1.0e-15  rader
      1430  2.5.11.13        chain3   replayed        2008       2700    1.26  5.6e-16  
      1431  3^3.53           prime    replayed       10348      11065    1.05  4.5e-16  bluestein
      1432  2^3.179          prime    raced          10500       9704    0.79  4.9e-16  bluestein
      1433  1433             prime    replayed       10459      11124    1.06  5.4e-16  bluestein
      1434  2.3.239          prime    raced          10514       9463    0.90  5.4e-16  bluestein
      1435  5.7.41           chain3   replayed        3507       8976    2.47  5.2e-16  
      1436  2^2.359          prime    raced          10166       9519    0.93  6.9e-16  bluestein
      1437  3.479            prime    replayed       10257       9674    0.93  5.4e-16  bluestein
      1438  2.719            prime    raced          10286      10283    0.98  6.5e-16  bluestein
      1439  1439             prime    replayed       10514      11099    0.80  6.2e-16  flips differ 1.31x bluestein
      1440  2^5.3^2.5        chain3   replayed        1524       1508    0.98  4.4e-16  
      1441  11.131           prime    replayed       10188      10742    0.75  6.1e-16  flips differ 1.38x bluestein
      1442  2.7.103          prime    raced          10383      10410    1.00  5.6e-16  bluestein
      1443  3.13.37          flat     replayed        3212       8962    2.77  7.1e-16  
      1444  2^2.19^2         chain3   replayed        2724       8925    3.10  3.6e-16  
      1445  5.17^2           chain3   replayed        2860       8692    2.85  5.8e-16  
      1446  2.3.241          prime    raced          10204      11399    1.10  6.5e-16  bluestein
      1447  1447             prime    replayed       10321      13319    1.27  8.5e-16  bluestein
      1448  2^3.181          prime    raced          10256      12116    1.15  6.5e-16  bluestein
      1449  3^2.7.23         flat     replayed        2609       5939    2.16  5.4e-16  
      1450  2.5^2.29         chain3   replayed        2623       6816    2.56  4.5e-16  
      1451  1451             prime    replayed       10458      13250    1.06  6.4e-16  bluestein
      1452  2^2.3.11^2       chain3   replayed        1762       2792    1.57  6.4e-16  
      1453  1453             prime    replayed        5965      13174    1.99  7.5e-16  rader
      1454  2.727            prime    raced          10234      13289    1.29  5.8e-16  bluestein
      1455  3.5.97           prime    replayed       10244      10670    1.02  5.5e-16  bluestein
      1456  2^4.7.13         chain3   replayed        1674       2038    0.87  6.8e-16  flips differ 1.41x
      1457  31.47            2p       replayed        5147      16031    2.19  5.4e-16  flips differ 1.42x
      1458  2.3^6            chain3   replayed        2550       2164    0.75  5.5e-16  
      1459  1459             prime    replayed        8881      12245    1.22  1.1e-15  rader
      1460  2^2.5.73         prime    raced          10355      10300    0.98  5.5e-16  bluestein
      1461  3.487            prime    replayed       10281      12734    1.21  5.1e-16  bluestein
      1462  2.17.43          flat     replayed        3908      13214    3.33  9.4e-16  
      1463  7.11.19          chain3   replayed        2587       5831    2.24  3.7e-16  
      1464  2^3.3.61         prime    raced          10227      12593    1.22  3.7e-16  bluestein
      1465  5.293            prime    replayed       10205      10058    0.98  6.6e-16  bluestein
      1466  2.733            prime    raced          10263      12354    1.17  6.5e-16  bluestein
      1467  3^2.163          prime    replayed       10360      10716    1.03  6.3e-16  bluestein
      1468  2^2.367          prime    raced          10354      11458    1.10  5.2e-16  bluestein
      1469  13.113           prime    replayed       10298      12679    1.23  6.0e-16  bluestein
      1470  2.3.5.7^2        chain3   replayed        2386       1876    0.78  5.0e-16  
      1471  1471             prime    replayed        7519      12184    1.54  9.4e-16  rader
      1472  2^6.23           2p       replayed        2486       5699    1.45  4.5e-16  flips differ 1.58x
      1473  3.491            prime    replayed       10191      12687    1.22  5.3e-16  bluestein
      1474  2.11.67          prime    raced          10379      14247    0.82  4.4e-16  flips differ 1.67x bluestein
      1475  5^2.59           prime    replayed       10312      11702    1.12  5.1e-16  bluestein
      1476  2^2.3^2.41       chain3   replayed        2940       9185    3.10  4.5e-16  
      1477  7.211            prime    replayed       10381      10199    0.98  5.8e-16  bluestein
      1478  2.739            prime    raced          10275      12327    1.20  4.7e-16  bluestein
      1479  3.17.29          flat     replayed        3433      10864    3.06  1.1e-15  
      1480  2^3.5.37         flat     replayed        2926       8469    2.79  6.6e-16  
      1481  1481             prime    replayed       10294      12073    1.17  4.9e-16  bluestein
      1482  2.3.13.19        chain3   replayed        2443       5929    2.42  6.4e-16  
      1483  1483             prime    replayed        7868      12197    1.55  1.0e-15  rader
      1484  2^2.7.53         prime    raced          10261      11333    1.10  4.5e-16  bluestein
      1485  3^3.5.11         chain3   replayed        2238       2406    1.07  4.6e-16  
      1486  2.743            prime    raced          10280      12261    1.18  5.9e-16  bluestein
      1487  1487             prime    replayed       10494      12150    1.15  5.4e-16  bluestein
      1488  2^4.3.31         chain3   replayed        2519       7382    2.91  5.1e-16  
      1489  1489             prime    replayed       10260      12158    1.16  6.3e-16  bluestein
      1490  2.5.149          prime    raced          10372      10548    1.02  5.6e-16  bluestein
      1491  3.7.71           prime    replayed       10380      10217    0.98  5.1e-16  bluestein
      1492  2^2.373          prime    raced          10355      11528    0.80  6.0e-16  flips differ 1.39x bluestein
      1493  1493             prime    replayed       10481      12104    1.15  7.2e-16  bluestein
      1494  2.3^2.83         prime    raced          10271      10808    1.02  4.8e-16  bluestein
      1495  5.13.23          chain3   replayed        2851       6677    2.31  6.1e-16  
      1496  2^3.11.17        chain3   replayed        2223       5768    2.56  4.5e-16  
      1497  3.499            prime    replayed       10371      12776    1.23  5.9e-16  bluestein
      1498  2.7.107          prime    raced          10223      10481    1.01  6.6e-16  bluestein
      1499  1499             prime    replayed       10459      12129    0.81  5.9e-16  flips differ 1.42x bluestein
      1500  2^2.3.5^3        chain3   replayed        1853       1874    1.00  4.2e-16  
      1501  19.79            prime    replayed       10270      13682    1.30  6.4e-16  bluestein
      1502  2.751            prime    raced          10283      12745    1.23  5.4e-16  bluestein
      1503  3^2.167          prime    replayed       10252      10362    0.98  6.2e-16  bluestein
      1504  2^5.47           2p       replayed        2594      10217    3.89  4.1e-16  
      1505  5.7.43           chain3   replayed        5256       9813    1.86  4.6e-16  
      1506  2.3.251          prime    raced          10442       8681    0.82  6.7e-16  bluestein
      1507  11.137           prime    replayed       10269      10387    1.00  5.3e-16  bluestein
      1508  2^2.13.29        flat     replayed        3347       7667    1.64  1.1e-15  flips differ 1.40x
      1509  3.503            prime    replayed       10278       9429    0.88  5.7e-16  bluestein
      1510  2.5.151          prime    raced          10466       9693    0.92  4.4e-16  bluestein
      1511  1511             prime    replayed       10478      11517    1.10  5.5e-16  bluestein
      1512  2^3.3^3.7        chain3   replayed        1862       1813    0.96  5.3e-16  
      1513  17.89            prime    replayed       10494      13562    0.98  5.9e-16  flips differ 1.32x bluestein
      1514  2.757            prime    raced          10353      12862    1.23  5.3e-16  bluestein
      1515  3.5.101          prime    replayed       10445      11018    1.05  6.3e-16  bluestein
      1516  2^2.379          prime    raced          10292       9518    0.92  5.9e-16  bluestein
      1517  37.41            2p       replayed        4863      16774    2.57  5.9e-16  flips differ 1.34x
      1518  2.3.11.23        chain3   replayed        2559       6598    2.57  4.2e-16  
      1519  7^2.31           flat     replayed        3264       8253    2.52  7.1e-16  
      1520  2^4.5.19         chain3   replayed        2092       5333    2.47  4.4e-16  
      1521  3^2.13^2         chain3   replayed        2513       3013    1.19  8.2e-16  
      1522  2.761            prime    raced          10365      12857    1.22  5.3e-16  bluestein
      1523  1523             prime    replayed       10428      11595    1.11  5.5e-16  bluestein
      1524  2^2.3.127        prime    raced          10282       8519    0.81  5.4e-16  bluestein
      1525  5^2.61           prime    replayed       10362      13616    1.29  3.4e-16  bluestein
      1526  2.7.109          prime    raced          10544      13883    1.31  6.1e-16  bluestein
      1527  3.509            prime    replayed       10287       9194    0.87  6.4e-16  bluestein
      1528  2^3.191          prime    raced          10379       9007    0.86  4.3e-16  bluestein
      1529  11.139           prime    replayed       10296      10280    0.99  4.8e-16  bluestein
      1530  2.3^2.5.17       chain3   replayed        2198       5271    2.38  5.8e-16  
      1531  1531             prime    replayed        6970      11557    1.60  8.9e-16  rader
      1532  2^2.383          prime    raced          10288       9497    0.75  4.9e-16  bluestein
      1533  3.7.73           prime    replayed       10533      11570    1.09  5.6e-16  bluestein
      1534  2.13.59          prime    raced          10294      10687    1.03  5.7e-16  bluestein
      1535  5.307            prime    replayed       10408      13191    1.10  5.6e-16  bluestein
      1536  2^9.3            chain3   replayed        1608       1476    0.91  3.4e-16  
      1537  29.53            prime    replayed       10256      17382    1.64  5.4e-16  bluestein
      1538  2.769            prime    raced          10277      11399    1.08  5.7e-16  bluestein
      1539  3^4.19           chain3   replayed        3274       5721    1.74  5.8e-16  
      1540  2^2.5.7.11       flat     replayed        2078       2164    1.03  4.6e-16  
      1541  23.67            prime    replayed       10508      19423    1.84  5.4e-16  bluestein
      1542  2.3.257          prime    raced          10341      14358    1.35  7.3e-16  bluestein
      1543  1543             prime    replayed       10357      15815    1.52  5.0e-16  bluestein
      1544  2^3.193          prime    raced          10334      10166    0.97  5.1e-16  bluestein
      1545  3.5.103          prime    replayed       10393      11232    1.06  5.6e-16  bluestein
      1546  2.773            prime    raced          10504      11385    0.83  6.0e-16  flips differ 1.32x bluestein
      1547  7.13.17          flat     replayed        2606       6031    2.30  6.7e-16  
      1548  2^2.3^2.43       chain3   replayed        3224      10029    3.06  5.4e-16  
      1549  1549             prime    replayed       10428      15818    1.51  4.6e-16  bluestein
      1550  2.5^2.31         chain3   replayed        2906       7955    2.73  6.0e-16  
      1551  3.11.47          chain3   replayed        5768      11175    1.33  4.7e-16  flips differ 1.46x
      1552  2^4.97           prime    raced          10331       9591    0.91  6.1e-16  bluestein
      1553  1553             prime    replayed       10411      15940    1.52  5.7e-16  bluestein
      1554  2.3.7.37         flat     replayed        4127       8958    2.17  7.8e-16  
      1555  5.311            prime    replayed       10420      13196    1.26  7.1e-16  bluestein
      1556  2^2.389          prime    raced          10426      10930    1.04  6.2e-16  bluestein
      1557  3^2.173          prime    replayed       10437      10668    1.01  7.6e-16  bluestein
      1558  2.19.41          flat     replayed        4456      14450    3.22  1.4e-15  
      1559  1559             prime    replayed       10331      15856    1.51  5.5e-16  bluestein
      1560  2^3.3.5.13       chain3   replayed        1865       1991    1.05  5.5e-16  
      1561  7.223            prime    replayed       10361      12291    1.17  4.9e-16  bluestein
      1562  2.11.71          prime    raced          10353      11162    1.07  6.5e-16  bluestein
      1563  3.521            prime    replayed       10347      12229    1.15  6.6e-16  bluestein
      1564  2^2.17.23        chain3   replayed        2938      10091    3.37  5.8e-16  
      1565  5.313            prime    replayed       10319      13233    1.25  6.0e-16  bluestein
      1566  2.3^3.29         chain3   replayed        3633       7333    1.48  5.2e-16  flips differ 1.36x
      1567  1567             prime    replayed       10268      11615    1.13  6.3e-16  bluestein
      1568  2^5.7^2          chain3   replayed        1769       1750    0.98  3.8e-16  
      1569  3.523            prime    replayed       10385      12146    1.16  5.5e-16  bluestein
      1570  2.5.157          prime    raced          10375       9442    0.90  5.9e-16  bluestein
      1571  1571             prime    replayed       10353      11645    1.12  7.5e-16  bluestein
      1572  2^2.3.131        prime    raced          10364      11201    1.07  6.7e-16  bluestein
      1573  11^2.13          chain3   replayed        2580       3570    1.36  6.2e-16  
      1574  2.787            prime    raced          10437      11423    1.08  6.4e-16  bluestein
      1575  3^2.5^2.7        flat     replayed        2465       2460    0.97  4.9e-16  
      1576  2^3.197          prime    raced          10305       9863    0.93  4.5e-16  bluestein
      1577  19.83            prime    replayed       10355      15133    1.27  7.0e-16  bluestein
      1578  2.3.263          prime    raced          10348      11941    1.13  7.2e-16  bluestein
      1579  1579             prime    replayed       10377      11683    1.10  4.3e-16  bluestein
      1580  2^2.5.79         prime    raced          10287      10014    0.95  4.8e-16  bluestein
      1581  3.17.31          chain3   replayed        5848      12111    1.83  4.9e-16  
      1582  2.7.113          prime    raced          10312      13471    1.28  5.7e-16  bluestein
      1583  1583             prime    replayed       10569      11717    1.10  4.7e-16  bluestein
      1584  2^4.3^2.11       chain3   replayed        1839       2110    1.15  3.7e-16  
      1585  5.317            prime    replayed       10591       9406    0.88  5.1e-16  bluestein
      1586  2.13.61          prime    raced          10495      13918    0.97  5.4e-16  flips differ 1.37x bluestein
      1587  3.23^2           flat     replayed        3535      11434    2.93  1.4e-15  
      1588  2^2.397          prime    raced          10479      11186    1.05  6.5e-16  bluestein
      1589  7.227            prime    replayed       10354      11036    1.04  4.9e-16  bluestein
      1590  2.3.5.53         prime    raced          10323      12719    1.23  4.9e-16  bluestein
      1591  37.43            2p       replayed        5515      17962    2.51  5.0e-16  flips differ 1.29x
      1592  2^3.199          prime    raced          10328      10026    0.96  5.8e-16  bluestein
      1593  3^3.59           prime    replayed       10530      13530    1.28  5.1e-16  bluestein
      1594  2.797            prime    raced          10560      11524    0.90  7.4e-16  bluestein
      1595  5.11.29          flat     replayed        3571       8320    2.30  5.8e-16  
      1596  2^2.3.7.19       chain3   replayed        2179       6498    2.95  4.6e-16  
      1597  1597             prime    replayed       10487      11789    1.12  7.1e-16  bluestein
      1598  2.17.47          flat     replayed        4415      14647    3.24  1.7e-15  
      1599  3.13.41          chain3   replayed        3680      10627    2.85  6.4e-16  
      1600  2^6.5^2          chain3   replayed        1721       1652    0.95  3.0e-16  
      1601  1601             prime    replayed        6332      13709    2.15  6.2e-16  rader
      1602  2.3^2.89         prime    raced          10291      10875    1.03  5.3e-16  bluestein
      1603  7.229            prime    replayed       10312      11024    1.04  4.6e-16  bluestein
      1604  2^2.401          prime    raced          10456      12212    1.16  6.0e-16  bluestein
      1605  3.5.107          prime    replayed       10344      11327    1.07  6.6e-16  bluestein
      1606  2.11.73          prime    raced          10221      11755    1.14  7.4e-16  bluestein
      1607  1607             prime    replayed       10564      13698    1.28  5.8e-16  bluestein
      1608  2^3.3.67         prime    raced          10304      14210    1.35  4.8e-16  bluestein
      1609  1609             prime    replayed       10283      13702    1.29  6.6e-16  bluestein
      1610  2.5.7.23         chain3   replayed        2670       7051    2.64  4.9e-16  
      1611  3^2.179          prime    replayed       10283      10791    1.04  6.9e-16  bluestein
      1612  2^2.13.31        flat     replayed        4129       8984    2.15  7.9e-16  
      1613  1613             prime    replayed       10470      13649    1.30  4.9e-16  bluestein
      1614  2.3.269          prime    raced          10302      12013    1.16  6.4e-16  bluestein
      1615  5.17.19          chain3   replayed        3460       9849    2.19  5.4e-16  flips differ 1.30x
      1616  2^4.101          prime    raced          10415      11863    1.11  6.6e-16  bluestein
      1617  3.7^2.11         chain3   replayed        2813       2626    0.92  5.5e-16  
      1618  2.809            prime    raced          10406      13290    1.26  4.8e-16  bluestein
      1619  1619             prime    replayed       11986      13631    1.11  6.2e-16  bluestein
      1620  2^2.3^4.5        chain3   replayed        1896       2107    1.11  5.5e-16  
      1621  1621             prime    replayed       10316      16676    1.61  5.5e-16  bluestein
      1622  2.811            prime    raced          10399      13226    1.25  5.7e-16  bluestein
      1623  3.541            prime    replayed       10490      18033    1.71  5.6e-16  bluestein
      1624  2^3.7.29         flat     replayed        3045       7603    2.37  6.6e-16  
      1625  5^3.13           flat     replayed        2552       2975    1.15  6.6e-16  
      1626  2.3.271          prime    raced          10408      13275    1.27  4.8e-16  bluestein
      1627  1627             prime    replayed       10448      16638    1.58  6.8e-16  bluestein
      1628  2^2.11.37        chain3   replayed        4793      10079    1.51  4.2e-16  flips differ 1.39x
      1629  3^2.181          prime    replayed       10451      13636    1.30  5.4e-16  bluestein
      1630  2.5.163          prime    raced          10349      11592    1.10  5.9e-16  bluestein
      1631  7.233            prime    replayed       10411      11056    0.81  5.7e-16  flips differ 1.31x bluestein
      1632  2^5.3.17         chain3   replayed        2064       5475    2.65  4.5e-16  
      1633  23.71            prime    replayed       10345      16382    1.55  5.1e-16  bluestein
      1634  2.19.43          flat     replayed        4354      14866    3.30  8.3e-16  
      1635  3.5.109          prime    replayed       10294      14948    1.45  5.2e-16  bluestein
      1636  2^2.409          prime    raced          10551      12299    0.93  6.2e-16  flips differ 1.26x bluestein
      1637  1637             prime    replayed       10401      16638    1.59  5.1e-16  bluestein
      1638  2.3^2.7.13       chain3   replayed        2663       2371    0.87  5.1e-16  
      1639  11.149           prime    replayed       10417      11311    1.08  5.4e-16  bluestein
      1640  2^3.5.41         flat     replayed        3283      10737    3.25  6.5e-16  
      1641  3.547            prime    replayed       10361      15335    1.45  6.2e-16  bluestein
      1642  2.821            prime    raced          10540      13287    1.13  6.9e-16  bluestein
      1643  31.53            prime    replayed       10430      19323    1.84  5.0e-16  bluestein
      1644  2^2.3.137        prime    raced          10367      10802    1.01  4.9e-16  bluestein
      1645  5.7.47           flat     replayed        3804      11539    3.01  4.8e-16  
      1646  2.823            prime    raced          10339      13000    1.22  5.8e-16  bluestein
      1647  3^3.61           prime    replayed       10326      14819    1.43  4.6e-16  bluestein
      1648  2^4.103          prime    raced          10332      11856    1.12  6.3e-16  bluestein
      1649  17.97            prime    replayed       10346      15595    1.48  5.7e-16  bluestein
      1650  2.3.5^2.11       chain3   replayed        2215       2254    1.00  3.9e-16  
      1651  13.127           prime    replayed       10348       9882    0.93  4.9e-16  bluestein
      1652  2^2.7.59         prime    raced          10403      12989    1.23  4.2e-16  bluestein
      1653  3.19.29          chain3   replayed        3770      12821    2.98  5.1e-16  
      1654  2.827            prime    raced          10359      13130    1.27  5.7e-16  bluestein
      1655  5.331            prime    replayed       10372      14002    1.32  6.6e-16  bluestein
      1656  2^3.3^2.23       chain3   replayed        2528       6549    2.53  5.3e-16  
      1657  1657             prime    replayed       10426      16636    1.57  6.6e-16  bluestein
      1658  2.829            prime    raced          10331      13172    1.23  5.6e-16  bluestein
      1659  3.7.79           prime    replayed       10594      11230    1.06  5.4e-16  bluestein
      1660  2^2.5.83         prime    raced          10454      11879    1.01  6.1e-16  bluestein
      1661  11.151           prime    replayed       10487      10542    0.99  4.6e-16  bluestein
      1662  2.3.277          prime    raced          10349      11867    1.15  6.8e-16  bluestein
      1663  1663             prime    replayed       10574      16769    1.22  5.8e-16  flips differ 1.31x bluestein
      1664  2^7.13           chain3   replayed        1948       2164    1.11  4.6e-16  
      1665  3^2.5.37         chain3   replayed        4887      10367    2.10  5.2e-16  
      1666  2.7^2.17         flat     replayed        3314       5915    1.73  6.9e-16  
      1667  1667             prime    replayed       10378      16625    1.59  7.3e-16  bluestein
      1668  2^2.3.139        prime    raced          10443      10979    1.04  5.9e-16  bluestein
      1669  1669             prime    replayed       10394      16706    1.60  6.8e-16  bluestein
      1670  2.5.167          prime    raced          10570      11540    1.08  5.8e-16  bluestein
      1671  3.557            prime    replayed       10443      15271    1.43  5.6e-16  bluestein
      1672  2^3.11.19        chain3   replayed        2558       6701    2.60  5.4e-16  
      1673  7.239            prime    replayed       10483      11073    0.87  4.9e-16  bluestein
      1674  2.3^3.31         flat     replayed        3841       8512    2.21  8.3e-16  
      1675  5^2.67           prime    replayed       10431      14862    1.42  6.4e-16  bluestein
      1676  2^2.419          prime    raced          10372      12212    1.17  8.1e-16  bluestein
      1677  3.13.43          chain3   replayed        4056      11638    2.85  7.5e-16  
      1678  2.839            prime    raced          10331      13245    1.26  5.6e-16  bluestein
      1679  23.73            prime    replayed       10459      17213    1.64  6.1e-16  bluestein
      1680  2^4.3.5.7        chain3   replayed        1910       1833    0.96  4.8e-16  
      1681  41^2             2p       replayed        5688      19358    2.40  4.3e-16  flips differ 1.42x
      1682  2.29^2           flat     replayed        4311      14428    3.26  1.7e-15  
      1683  3^2.11.17        chain3   replayed        3038       6658    2.18  7.2e-16  
      1684  2^2.421          prime    raced          10468      12298    1.17  6.3e-16  bluestein
      1685  5.337            prime    replayed       10365      14094    1.33  6.1e-16  bluestein
      1686  2.3.281          prime    raced          10340      11790    1.13  5.4e-16  bluestein
      1687  7.241            prime    replayed       10411      17703    1.70  6.2e-16  bluestein
      1688  2^3.211          prime    raced          10523      11630    1.10  4.8e-16  bluestein
      1689  3.563            prime    replayed       10558      15266    1.19  5.3e-16  bluestein
      1690  2.5.13^2         chain3   replayed        2552       3258    1.27  5.9e-16  
      1691  19.89            prime    replayed       10340      15506    1.50  7.0e-16  bluestein
      1692  2^2.3^2.47       chain3   replayed        3570      11875    3.31  6.1e-16  
      1693  1693             prime    replayed       10454      13715    1.29  6.1e-16  bluestein
      1694  2.7.11^2         flat     replayed        3398       3327    0.97  9.5e-16  
      1695  3.5.113          prime    replayed       10530      14950    1.42  6.0e-16  bluestein
      1696  2^5.53           prime    raced          10337      12783    1.24  4.4e-16  bluestein
      1697  1697             prime    replayed       10391      13725    1.30  5.9e-16  bluestein
      1698  2.3.283          prime    raced          10550      11890    0.88  5.6e-16  flips differ 1.27x bluestein
      1699  1699             prime    replayed       10241      13487    1.30  6.3e-16  bluestein
      1700  2^2.5^2.17       chain3   replayed        2457       5832    2.34  5.3e-16  
      1701  3^5.7            chain3   replayed        2699       2987    1.10  4.2e-16  
      1702  2.23.37          flat     replayed        4561      15103    3.26  1.3e-15  
      1703  13.131           prime    replayed       10250      12666    1.21  6.5e-16  bluestein
      1704  2^3.3.71         prime    raced          10441      11550    1.10  4.5e-16  bluestein
      1705  5.11.31          chain3   replayed        6286       9404    1.40  4.6e-16  
      1706  2.853            prime    raced          10327      13044    1.23  6.3e-16  bluestein
      1707  3.569            prime    replayed       10281      12649    1.23  5.4e-16  bluestein
      1708  2^2.7.61         prime    raced          10463      14838    1.41  3.7e-16  bluestein
      1709  1709             prime    replayed       10651      13965    1.31  7.7e-16  bluestein
      1710  2.3^2.5.19       chain3   replayed        3022       6207    2.02  6.8e-16  
      1711  29.59            prime    replayed       10343      17323    1.64  5.1e-16  bluestein
      1712  2^4.107          prime    raced          10300      11961    1.16  5.9e-16  bluestein
      1713  3.571            prime    replayed       10519      12410    1.18  5.4e-16  bluestein
      1714  2.857            prime    raced          10339      13250    1.26  6.7e-16  bluestein
      1715  5.7^3            flat     replayed        2784       2938    1.04  5.3e-16  
      1716  2^2.3.11.13      chain3   replayed        2227       3517    1.30  6.0e-16  
      1717  17.101           prime    replayed       10518      16358    1.55  5.4e-16  bluestein
      1718  2.859            prime    raced          10568      13226    0.78  5.2e-16  flips differ 1.62x bluestein
      1719  3^2.191          prime    replayed       11227      10717    0.66  4.6e-16  flips differ 1.44x bluestein
      1720  2^3.5.43         chain3   replayed        3451      11167    2.75  4.6e-16  
      1721  1721             prime    replayed       10517      13724    1.28  5.9e-16  bluestein
      1722  2.3.7.41         chain3   replayed        3686      11282    2.99  5.4e-16  
      1723  1723             prime    replayed       10660      13731    1.26  5.3e-16  bluestein
      1724  2^2.431          prime    raced          10404      14925    1.19  7.2e-16  bluestein
      1725  3.5^2.23         chain3   replayed        3137       7347    2.33  5.4e-16  
      1726  2.863            prime    raced          10496      13296    1.25  5.4e-16  bluestein
      1727  11.157           prime    replayed       10355      10677    1.02  6.5e-16  bluestein
      1728  2^6.3^3          chain3   replayed        1858       1936    1.04  6.2e-16  
      1729  7.13.19          chain3   replayed        3345       7063    1.57  6.0e-16  flips differ 1.35x
      1730  2.5.173          prime    raced          10301      11611    1.11  5.6e-16  bluestein
      1731  3.577            prime    replayed       10345      15117    1.20  5.7e-16  bluestein
      1732  2^2.433          prime    raced          10511      16651    1.58  6.1e-16  bluestein
      1733  1733             prime    replayed       10351      14777    1.39  5.8e-16  bluestein
      1734  2.3.17^2         chain3   replayed        3540      10073    2.65  4.2e-16  
      1735  5.347            prime    replayed       10362      11911    1.15  5.0e-16  bluestein
      1736  2^3.7.31         flat     replayed        3568       8815    2.42  6.9e-16  
      1737  3^2.193          prime    replayed       10309      11133    1.06  5.1e-16  bluestein
      1738  2.11.79          prime    raced          10351      11746    1.11  6.0e-16  bluestein
      1739  37.47            flat     replayed        4783      20434    4.18  1.8e-15  
      1740  2^2.3.5.29       flat     replayed        3816       8645    2.12  5.4e-16  
      1741  1741             prime    replayed       10428      14818    1.39  5.8e-16  bluestein
      1742  2.13.67          prime    raced          10340      16992    1.64  4.6e-16  bluestein
      1743  3.7.83           prime    replayed       10367      16224    1.55  4.1e-16  bluestein
      1744  2^4.109          prime    raced          10546      15896    1.25  7.1e-16  bluestein
      1745  5.349            prime    replayed       10330      11893    1.15  6.0e-16  bluestein
      1746  2.3^2.97         prime    raced          10328      12706    1.20  4.8e-16  bluestein
      1747  1747             prime    replayed       10507      14645    1.36  5.9e-16  bluestein
      1748  2^2.19.23        chain3   replayed        3454      11695    3.03  5.0e-16  
      1749  3.11.53          prime    replayed       10355      13946    1.35  4.6e-16  bluestein
      1750  2.5^3.7          chain3   replayed        3075       2282    0.70  4.9e-16  
      1751  17.103           prime    replayed       10354      16317    1.57  6.8e-16  bluestein
      1752  2^3.3.73         prime    raced          10374      12984    0.89  6.7e-16  flips differ 1.40x bluestein
      1753  1753             prime    replayed       10617      14804    1.39  4.9e-16  bluestein
      1754  2.877            prime    raced          10384      14104    1.34  5.6e-16  bluestein
      1755  3^3.5.13         chain3   replayed        2837       3030    1.04  6.5e-16  
      1756  2^2.439          prime    raced          10356      13995    1.34  5.5e-16  bluestein
      1757  7.251            prime    replayed       10442      10484    1.00  4.9e-16  bluestein
      1758  2.3.293          prime    raced          10415      12105    1.16  6.9e-16  bluestein
      1759  1759             prime    replayed       10520      14721    1.37  6.4e-16  bluestein
      1760  2^5.5.11         chain3   replayed        1969       2329    1.15  3.7e-16  
      1761  3.587            prime    replayed       10248      15490    1.48  6.7e-16  bluestein
      1762  2.881            prime    raced          10578      14070    1.32  5.6e-16  bluestein
      1763  41.43            2p       replayed        6375      20773    3.24  5.8e-16  
      1764  2^2.3^2.7^2      chain3   replayed        2219       2337    1.04  6.0e-16  
      1765  5.353            prime    replayed       10523      11938    1.13  5.4e-16  bluestein
      1766  2.883            prime    raced          10371      14165    1.33  6.0e-16  bluestein
      1767  3.19.31          chain3   replayed        3950      13798    3.48  5.2e-16  
      1768  2^3.13.17        chain3   replayed        2667       6918    2.50  6.4e-16  
      1769  29.61            prime    replayed       10586      21706    2.03  5.6e-16  bluestein
      1770  2.3.5.59         prime    raced          10411      11687    1.10  4.0e-16  bluestein
      1771  7.11.23          flat     replayed        3736       7844    2.10  7.4e-16  
      1772  2^2.443          prime    raced          10449      13936    1.32  5.3e-16  bluestein
      1773  3^2.197          prime    replayed       10374      11252    1.08  5.6e-16  bluestein
      1774  2.887            prime    raced          10362      14013    1.34  5.5e-16  bluestein
      1775  5^2.71           prime    replayed       10477      12795    1.19  7.0e-16  bluestein
      1776  2^4.3.37         chain3   replayed        3117      10207    3.20  6.7e-16  
      1777  1777             prime    replayed       10403      14695    1.37  6.0e-16  bluestein
      1778  2.7.127          prime    raced          10558      11476    1.07  5.8e-16  bluestein
      1779  3.593            prime    replayed       10443      15221    1.44  5.4e-16  bluestein
      1780  2^2.5.89         prime    raced          10442      15434    1.11  5.7e-16  flips differ 1.32x bluestein
      1781  13.137           prime    replayed       10431      12395    1.17  5.3e-16  bluestein
      1782  2.3^4.11         chain3   replayed        3299       2596    0.78  6.1e-16  
      1783  1783             prime    replayed       10395      14906    1.41  6.4e-16  bluestein
      1784  2^3.223          prime    raced          10340      14017    1.09  6.3e-16  bluestein
      1785  3.5.7.17         chain3   replayed        2995       6526    2.09  6.5e-16  
      1786  2.19.47          flat     replayed        5029      16585    3.23  1.5e-15  
      1787  1787             prime    replayed       10448      14439    1.38  6.2e-16  bluestein
      1788  2^2.3.149        prime    raced          10256      12873    1.24  6.1e-16  bluestein
      1789  1789             prime    replayed       10395      14483    1.37  5.6e-16  bluestein
      1790  2.5.179          prime    raced          10455      11801    1.12  5.8e-16  bluestein
      1791  3^2.199          prime    replayed       10469      11289    1.06  5.4e-16  bluestein
      1792  2^8.7            chain3   replayed        2060       1900    0.90  2.9e-16  
      1793  11.163           prime    replayed       10529      13360    1.27  5.2e-16  bluestein
      1794  2.3.13.23        chain3   replayed        3499       8047    2.06  7.3e-16  
      1795  5.359            prime    replayed       10394      12064    1.15  6.1e-16  bluestein
      1796  2^2.449          prime    raced          10427      14028    1.32  7.3e-16  bluestein
      1797  3.599            prime    replayed       10396      15392    1.48  7.4e-16  bluestein
      1798  2.29.31          flat     replayed        4867      16125    3.29  2.0e-15  
      1799  7.257            prime    replayed       10384      17189    1.24  6.7e-16  flips differ 1.34x bluestein
      1800  2^3.3^2.5^2      chain3   replayed        2066       2126    1.02  4.6e-16  
      1801  1801             prime    replayed       10421      19943    1.88  6.0e-16  bluestein
      1802  2.17.53          prime    raced          10420      18229    1.71  5.0e-16  bluestein
      1803  3.601            prime    replayed       10521      19425    1.85  5.7e-16  bluestein
      1804  2^2.11.41        chain3   replayed        3744      11817    3.12  4.6e-16  
      1805  5.19^2           flat     replayed        3925      11708    2.79  1.3e-15  
      1806  2.3.7.43         chain3   replayed        8261      12278    1.32  5.7e-16  
      1807  13.139           prime    replayed       10556      12458    1.18  5.4e-16  bluestein
      1808  2^4.113          prime    raced          10403      15392    1.45  5.2e-16  bluestein
      1809  3^3.67           prime    replayed       10394      16183    1.52  5.4e-16  bluestein
      1810  2.5.181          prime    raced          10407      15133    1.45  5.0e-16  bluestein
      1811  1811             prime    replayed       10587      18709    1.47  5.7e-16  bluestein
      1812  2^2.3.151        prime    raced          10360      11772    1.00  5.4e-16  bluestein
      1813  7^2.37           chain3   replayed        4220      10898    2.50  5.0e-16  
      1814  2.907            prime    raced          10357      17638    1.70  6.3e-16  bluestein
      1815  3.5.11^2         chain3   replayed        2916       3674    1.26  4.7e-16  
      1816  2^3.227          prime    raced          10397      12626    1.20  5.5e-16  bluestein
      1817  23.79            prime    replayed       10439      17499    1.64  4.7e-16  bluestein
      1818  2.3^2.101        prime    raced          10639      13247    1.24  7.4e-16  bluestein
      1819  17.107           prime    replayed       10581      16618    1.56  6.1e-16  bluestein
      1820  2^2.5.7.13       flat     replayed        2606       2625    0.90  6.5e-16  
      1821  3.607            prime    replayed       10397      16752    1.58  6.3e-16  bluestein
      1822  2.911            prime    raced          10383      17721    1.67  6.3e-16  bluestein
      1823  1823             prime    replayed       10689      18792    1.75  5.9e-16  bluestein
      1824  2^5.3.19         chain3   replayed        2433       6422    2.42  4.8e-16  
      1825  5^2.73           prime    replayed       10399      12975    1.24  5.9e-16  bluestein
      1826  2.11.83          prime    raced          10430      13765    1.30  5.8e-16  bluestein
      1827  3^2.7.29         flat     replayed        3880       8822    2.27  6.4e-16  
      1828  2^2.457          prime    raced          10414      13005    1.23  4.8e-16  bluestein
      1829  31.59            prime    replayed       10393      22828    2.15  4.1e-16  bluestein
      1830  2.3.5.61         prime    raced          10530      16086    1.52  4.9e-16  bluestein
      1831  1831             prime    replayed       10597      16938    1.60  4.9e-16  bluestein
      1832  2^3.229          prime    raced          10521      12616    1.01  4.7e-16  bluestein
      1833  3.13.47          flat     replayed        4594      12774    2.76  8.1e-16  
      1834  2.7.131          prime    raced          10439      13307    1.25  6.0e-16  bluestein
      1835  5.367            prime    replayed       10428      14432    1.38  5.9e-16  bluestein
      1836  2^2.3^3.17       chain3   replayed        2412       6593    2.68  5.9e-16  
      1837  11.167           prime    replayed       10250      18112    1.17  6.1e-16  flips differ 1.52x bluestein
      1838  2.919            prime    raced          10408      17671    1.70  6.1e-16  bluestein
      1839  3.613            prime    replayed       10464      15578    1.48  5.8e-16  bluestein
      1840  2^4.5.23         chain3   replayed        2790       7248    2.58  5.4e-16  
      1841  7.263            prime    replayed       10281      16739    1.62  6.3e-16  bluestein
      1842  2.3.307          prime    raced          10537      16129    1.48  7.6e-16  bluestein
      1843  19.97            prime    replayed       10629      18021    1.69  5.7e-16  bluestein
      1844  2^2.461          prime    raced          10435      13039    1.24  5.0e-16  bluestein
      1845  3^2.5.41         chain3   replayed        4193      11848    2.57  5.2e-16  
      1846  2.13.71          prime    raced          10453      13297    1.26  5.8e-16  bluestein
      1847  1847             prime    replayed       10673      17043    1.59  6.0e-16  bluestein
      1848  2^3.3.7.11       flat     replayed        3113       2520    0.81  9.1e-16  
      1849  43^2             flat     replayed        4982      22215    4.40  1.5e-15  
      1850  2.5^2.37         chain3   replayed        3659      10791    2.84  5.4e-16  
      1851  3.617            prime    replayed       10602      15447    1.45  5.8e-16  bluestein
      1852  2^2.463          prime    raced          10439      13159    1.26  5.5e-16  bluestein
      1853  17.109           prime    replayed       10481      21137    2.00  5.7e-16  bluestein
      1854  2.3^2.103        prime    raced          10435      13032    1.24  5.3e-16  bluestein
      1855  5.7.53           prime    replayed       10496      14963    1.42  5.3e-16  bluestein
      1856  2^6.29           2p       replayed        2694       8581    3.17  4.1e-16  
      1857  3.619            prime    replayed       10626      15501    1.23  5.9e-16  bluestein
      1858  2.929            prime    raced          10458      17798    1.69  5.7e-16  bluestein
      1859  11.13^2          chain3   replayed        3235       4210    1.30  8.1e-16  
      1860  2^2.3.5.31       chain3   replayed        3104       9329    3.00  7.3e-16  
      1861  1861             prime    replayed       10401      17085    1.64  6.3e-16  bluestein
      1862  2.7^2.19         flat     replayed        4022       7012    1.74  7.0e-16  
      1863  3^4.23           chain3   replayed        3352       7793    2.29  6.2e-16  
      1864  2^3.233          prime    raced          10461      12672    1.21  4.8e-16  bluestein
      1865  5.373            prime    replayed       10536      14475    1.37  5.4e-16  bluestein
      1866  2.3.311          prime    raced          10406      15723    1.47  5.2e-16  bluestein
      1867  1867             prime    replayed       10639      17146    1.60  5.8e-16  bluestein
      1868  2^2.467          prime    raced          10632      13089    1.23  5.2e-16  bluestein
      1869  3.7.89           prime    replayed       10617      16362    1.53  6.6e-16  bluestein
      1870  2.5.11.17        chain3   replayed        2782       7308    2.61  4.7e-16  
      1871  1871             prime    replayed       10415      17441    1.64  7.7e-16  bluestein
      1872  2^4.3^2.13       chain3   replayed        2241       2623    1.14  6.3e-16  
      1873  1873             prime    replayed        7990      16975    1.85  1.1e-15  rader
      1874  2.937            prime    raced          10652      17700    1.66  6.3e-16  bluestein
      1875  3.5^4            flat     replayed        3396       3023    0.86  4.5e-16  
      1876  2^2.7.67         prime    raced          10441      16622    1.56  5.6e-16  bluestein
      1877  1877             prime    replayed       10696      14773    1.35  5.7e-16  bluestein
      1878  2.3.313          prime    raced          10626      15840    1.49  6.4e-16  bluestein
      1879  1879             prime    replayed       10501      14309    1.35  5.6e-16  bluestein
      1880  2^3.5.47         flat     replayed        4250      13026    3.06  5.5e-16  
      1881  3^2.11.19        chain3   replayed        4010       7656    1.89  4.9e-16  
      1882  2.941            prime    raced          10631      14043    1.31  6.6e-16  bluestein
      1883  7.269            prime    replayed       10458      16844    1.57  6.4e-16  bluestein
      1884  2^2.3.157        prime    raced          10438      11135    1.06  5.4e-16  bluestein
      1885  5.13.29          chain3   replayed        5985       9876    1.49  8.0e-16  
      1886  2.23.41          flat     replayed        5305      17583    3.30  1.5e-15  
      1887  3.17.37          chain3   replayed        4281      15708    3.50  6.2e-16  
      1888  2^5.59           prime    raced          10444      15053    1.42  4.8e-16  bluestein
      1889  1889             prime    replayed       10683      14392    1.34  7.0e-16  bluestein
      1890  2.3^3.5.7        chain3   replayed        2817       2561    0.89  4.9e-16  
      1891  31.61            prime    replayed       10452      24023    2.25  5.5e-16  bluestein
      1892  2^2.11.43        chain3   replayed        4039      12861    3.18  4.8e-16  
      1893  3.631            prime    replayed       10451      12706    1.17  5.3e-16  bluestein
      1894  2.947            prime    raced          10645      14235    1.33  5.9e-16  bluestein
      1895  5.379            prime    replayed       10403      12114    1.14  5.9e-16  bluestein
      1896  2^3.3.79         prime    raced          10502      12020    1.14  6.8e-16  bluestein
      1897  7.271            prime    replayed       10459      15703    1.48  5.2e-16  bluestein
      1898  2.13.73          prime    raced          10482      14138    1.33  5.6e-16  bluestein
      1899  3^2.211          prime    replayed       10465      13272    1.24  9.5e-16  bluestein
      1900  2^2.5^2.19       flat     replayed        3373       6754    1.98  5.1e-16  
      1901  1901             prime    replayed       10508      14331    1.36  6.9e-16  bluestein
      1902  2.3.317          prime    raced          10659      11309    1.05  6.3e-16  bluestein
      1903  11.173           prime    replayed       10500      18258    1.47  5.4e-16  bluestein
      1904  2^4.7.17         chain3   replayed        2485       6657    2.40  5.5e-16  
      1905  3.5.127          prime    replayed       10562      10960    1.03  6.0e-16  bluestein
      1906  2.953            prime    raced          10488      14041    1.11  5.2e-16  bluestein
      1907  1907             prime    replayed       10713      14527    1.35  5.0e-16  bluestein
      1908  2^2.3^2.53       prime    raced          10421      14758    1.41  5.5e-16  bluestein
      1909  23.83            prime    replayed       10430      19344    1.85  5.0e-16  bluestein
      1910  2.5.191          prime    raced          10617      11049    1.03  4.3e-16  bluestein
      1911  3.7^2.13         chain3   replayed        3119       3272    1.04  5.8e-16  
      1912  2^3.239          prime    raced          10483      12744    1.20  6.1e-16  bluestein
      1913  1913             prime    replayed       10437      14358    1.36  6.3e-16  bluestein
      1914  2.3.11.29        flat     replayed        4447       9933    2.22  2.0e-15  
      1915  5.383            prime    replayed       10504      12064    1.14  5.7e-16  bluestein
      1916  2^2.479          prime    raced          10456      13090    1.23  5.5e-16  bluestein
      1917  3^3.71           prime    replayed       10446      13285    1.25  6.1e-16  bluestein
      1918  2.7.137          prime    raced          10667      12872    1.20  5.3e-16  bluestein
      1919  19.101           prime    replayed       10441      18550    1.77  7.0e-16  bluestein
      1920  2^7.3.5          chain3   replayed        2089       2044    0.95  6.2e-16  
      1921  17.113           prime    replayed       10561      21304    2.02  5.6e-16  bluestein
      1922  2.31^2           flat     replayed        5102      18039    3.50  2.2e-15  
      1923  3.641            prime    replayed       10518      20402    1.90  6.6e-16  bluestein
      1924  2^2.13.37        chain3   replayed        3993      12139    3.02  4.8e-16  
      1925  5^2.7.11         flat     replayed        3019       3145    1.04  5.2e-16  
      1926  2.3^2.107        prime    raced          10627      13324    1.25  6.8e-16  bluestein
      1927  41.47            flat     raced           5265      23557    4.37  1.9e-15  
      1928  2^3.241          prime    raced          10640      15857    1.49  6.0e-16  bluestein
      1929  3.643            prime    replayed       10628      15103    1.40  6.4e-16  bluestein
      1930  2.5.193          prime    raced          10440      12399    1.17  6.5e-16  bluestein
      1931  1931             prime    replayed       10662      17155    1.60  8.6e-16  bluestein
      1932  2^2.3.7.23       flat     replayed        3759       7997    1.80  5.9e-16  
      1933  1933             prime    replayed       10429      17127    1.60  7.2e-16  bluestein
      1934  2.967            prime    raced          10641      16918    1.34  8.3e-16  bluestein
      1935  3^2.5.43         flat     replayed        4198      12769    3.01  5.5e-16  
      1936  2^4.11^2         chain3   replayed        2394       3743    1.55  4.6e-16  
      1937  13.149           prime    replayed       10522      14493    1.37  6.4e-16  bluestein
      1938  2.3.17.19        chain3   replayed        3604      11717    3.19  5.9e-16  
      1939  7.277            prime    replayed       10437      15642    1.47  6.4e-16  bluestein
      1940  2^2.5.97         prime    raced          10548      14052    1.33  6.5e-16  bluestein
      1941  3.647            prime    replayed       10647      14900    1.40  6.4e-16  bluestein
      1942  2.971            prime    raced          10422      16935    1.61  7.0e-16  bluestein
      1943  29.67            prime    replayed       10596      26100    2.45  5.3e-16  bluestein
      1944  2^3.3^5          chain3   replayed        2704       2622    0.94  5.1e-16  
      1945  5.389            prime    replayed       10573      13870    1.30  6.0e-16  bluestein
      1946  2.7.139          prime    raced          10670      12932    1.21  5.1e-16  bluestein
      1947  3.11.59          prime    replayed       10468      13263    1.25  5.5e-16  bluestein
      1948  2^2.487          prime    raced          10496      17133    1.62  6.6e-16  bluestein
      1949  1949             prime    replayed       10441      15514    1.46  5.8e-16  bluestein
      1950  2.3.5^2.13       chain3   replayed        3086       2804    0.88  5.2e-16  
      1951  1951             prime    replayed       10466      15493    1.47  6.5e-16  bluestein
      1952  2^5.61           prime    raced          10496      16733    1.59  5.2e-16  bluestein
      1953  3^2.7.31         flat     replayed        4479      10246    2.25  8.0e-16  
      1954  2.977            prime    raced          10507      16223    1.43  6.0e-16  bluestein
      1955  5.17.23          chain3   replayed        4958      13057    2.00  6.5e-16  flips differ 1.32x
      1956  2^2.3.163        prime    raced          10442      14087    1.35  6.6e-16  bluestein
      1957  19.103           prime    replayed       10688      18640    1.39  6.9e-16  flips differ 1.26x bluestein
      1958  2.11.89          prime    raced          10661      13894    1.30  6.8e-16  bluestein
      1959  3.653            prime    replayed       10679      16506    1.53  6.3e-16  bluestein
      1960  2^3.5.7^2        flat     replayed        3031       2293    0.75  6.1e-16  
      1961  37.53            prime    replayed       10430      24462    2.34  4.9e-16  bluestein
      1962  2.3^2.109        prime    raced          10694      17177    1.59  7.4e-16  bluestein
      1963  13.151           prime    replayed       10676      13442    1.24  7.2e-16  bluestein
      1964  2^2.491          prime    raced          10719      16921    1.57  5.0e-16  bluestein
      1965  3.5.131          prime    replayed       10453      14283    1.34  6.6e-16  bluestein
      1966  2.983            prime    raced          10463      15207    1.45  5.9e-16  bluestein
      1967  7.281            prime    replayed       10457      15731    1.47  5.0e-16  bluestein
      1968  2^4.3.41         flat     replayed        4064      12170    2.99  5.1e-16  
      1969  11.179           prime    replayed       10668      13527    1.26  6.3e-16  bluestein
      1970  2.5.197          prime    raced          10579      12280    1.13  5.4e-16  bluestein
      1971  3^3.73           prime    replayed       10422      14814    1.38  7.5e-16  bluestein
      1972  2^2.17.29        flat     replayed        4759      14332    2.90  8.5e-16  
      1973  1973             prime    replayed       10499      15544    1.48  6.4e-16  bluestein
      1974  2.3.7.47         chain3   replayed        4516      13963    3.04  4.3e-16  
      1975  5^2.79           prime    replayed       10515      13522    1.28  5.5e-16  bluestein
      1976  2^3.13.19        chain3   replayed        3171       8214    2.45  6.9e-16  
      1977  3.659            prime    replayed       10449      16494    1.58  6.0e-16  bluestein
      1978  2.23.43          flat     replayed        5536      19112    3.45  1.8e-15  
      1979  1979             prime    replayed       10511      15506    1.47  4.2e-16  bluestein
      1980  2^2.3^2.5.11     chain3   replayed        2442       2732    1.10  4.5e-16  
      1981  7.283            prime    replayed       10482      15736    1.47  6.3e-16  bluestein
      1982  2.991            prime    raced          10412      15085    1.44  5.5e-16  bluestein
      1983  3.661            prime    replayed       10669      16518    1.54  6.2e-16  bluestein
      1984  2^6.31           2p       replayed        2883      10676    3.69  4.9e-16  
      1985  5.397            prime    replayed       10450      14138    1.33  6.0e-16  bluestein
      1986  2.3.331          prime    raced          10683      16948    1.58  6.7e-16  bluestein
      1987  1987             prime    replayed       10462      15615    1.49  6.7e-16  bluestein
      1988  2^2.7.71         prime    raced          10692      13503    1.26  5.3e-16  bluestein
      1989  3^2.13.17        chain3   replayed        3692       7946    2.13  6.5e-16  
      1990  2.5.199          prime    raced          10603      12567    1.17  5.2e-16  bluestein
      1991  11.181           prime    replayed       10427      16958    1.60  6.3e-16  bluestein
      1992  2^3.3.83         prime    raced          10414      18313    1.75  5.4e-16  bluestein
      1993  1993             prime    replayed       10643      15548    1.11  6.2e-16  flips differ 1.32x bluestein
      1994  2.997            prime    raced          10390      15180    1.45  5.1e-16  bluestein
      1995  3.5.7.19         chain3   replayed        4157       8461    1.67  4.1e-16  
      1996  2^2.499          prime    raced          10409      14504    1.38  5.8e-16  bluestein
      1997  1997             prime    replayed       10512      15547    1.45  5.6e-16  bluestein
      1998  2.3^3.37         chain3   replayed        5639      11625    2.06  5.8e-16  
      1999  1999             prime    replayed       10672      15534    1.27  7.1e-16  bluestein
      2000  2^4.5^3          chain3   replayed        2671       2182    0.67  4.5e-16  
      2001  3.23.29          chain3   replayed        5691      15836    2.48  5.1e-16  
      2002  2.7.11.13        flat     replayed        4050       4309    0.76  7.2e-16  flips differ 1.41x
      2003  2003             prime    replayed       10571      18242    1.72  6.7e-16  bluestein
      2004  2^2.3.167        prime    raced          10505      14118    1.32  5.2e-16  bluestein
      2005  5.401            prime    replayed       10503      21208    1.99  6.8e-16  bluestein
      2006  2.17.59          prime    raced          10461      18015    1.69  6.6e-16  bluestein
      2007  3^2.223          prime    replayed       10469      15893    1.49  4.7e-16  bluestein
      2008  2^3.251          prime    raced          10687      12025    1.11  5.5e-16  bluestein
      2009  7^2.41           chain3   replayed        6459      12794    1.35  4.1e-16  flips differ 1.47x
      2010  2.3.5.67         prime    raced          10699      17692    1.65  6.9e-16  bluestein
      2011  2011             prime    replayed       10497      18164    1.53  6.5e-16  bluestein
      2012  2^2.503          prime    raced          10535      12535    1.16  5.8e-16  bluestein
      2013  3.11.61          prime    replayed       10500      17814    1.63  4.6e-16  bluestein
      2014  2.19.53          prime    raced          10469      20713    1.94  6.2e-16  bluestein
      2015  5.13.31          flat     replayed        4815      11402    2.18  5.8e-16  
      2016  2^5.3^2.7        chain3   replayed        2316       2408    1.03  5.1e-16  
      2017  2017             prime    replayed       10713      18168    1.69  6.8e-16  bluestein
      2018  2.1009           prime    raced          10476      19821    1.85  6.4e-16  bluestein
      2019  3.673            prime    replayed       10492      16526    1.54  6.2e-16  bluestein
      2020  2^2.5.101        prime    raced          10704      14601    1.35  5.6e-16  bluestein
      2021  43.47            flat     raced           5593      25212    4.50  2.0e-15  
      2022  2.3.337          prime    raced          10498      16876    1.58  5.9e-16  bluestein
      2023  7.17^2           chain3   replayed        3855      12445    3.20  6.7e-16  
      2024  2^3.11.23        chain3   replayed        3265       9101    2.75  5.6e-16  
      2025  3^4.5^2          chain3   replayed        3419       3403    0.99  4.9e-16  
      2026  2.1013           prime    raced          10523      19953    1.88  6.5e-16  bluestein
      2027  2027             prime    replayed       10482      14391    1.32  5.8e-16  bluestein
      2028  2^2.3.13^2       chain3   replayed        2766       4271    1.52  6.4e-16  
      2029  2029             prime    replayed        9033      14514    1.33  9.6e-16  rader
      2030  2.5.7.29         flat     replayed        4955      10350    2.06  5.1e-16  
      2031  3.677            prime    replayed       10571      15445    1.44  6.6e-16  bluestein
      2032  2^4.127          prime    raced          10461      11723    1.10  6.3e-16  bluestein
      2033  19.107           prime    replayed       10601      19097    1.80  6.1e-16  bluestein
      2034  2.3^2.113        prime    raced          10450      17202    1.60  5.9e-16  bluestein
      2035  5.11.37          chain3   replayed        6236      12690    1.52  5.7e-16  flips differ 1.34x
      2036  2^2.509          prime    raced          10492      12377    1.18  5.8e-16  bluestein
      2037  3.7.97           prime    replayed       10472      15069    1.43  6.1e-16  bluestein
      2038  2.1019           prime    raced          10549      13200    1.23  5.9e-16  bluestein
      2039  2039             prime    replayed       10439      14281    1.36  6.1e-16  bluestein
      2040  2^3.3.5.17       chain3   replayed        3021       7037    2.26  5.9e-16  
      2041  13.157           prime    replayed       10505      12807    1.21  5.8e-16  bluestein
      2042  2.1021           prime    raced          10595      12937    1.21  6.1e-16  bluestein
      2043  3^2.227          prime    replayed       10470      14311    1.36  7.1e-16  bluestein
      2044  2^2.7.73         prime    raced          10496      14401    1.34  6.5e-16  bluestein
      2045  5.409            prime    replayed       10476      15461    1.45  5.9e-16  bluestein
      2046  2.3.11.31        chain3   replayed        5897      11157    1.87  5.0e-16  
      2047  23.89            prime    replayed       10468      19835    1.86  6.4e-16  bluestein
      2048  2^11             ztt      replayed        1600       1908    1.18  3.1e-16  
      2049  3.683            prime    replayed       22518      15557    0.69  6.5e-16  bluestein
      2050  2.5^2.41         chain3   replayed        9591      13584    1.41  6.1e-16  
      2051  7.293            prime    replayed       22470      14569    0.65  6.5e-16  bluestein
      2052  2^2.3^3.19       chain3   replayed        2906       8353    2.80  4.9e-16  
      2053  2053             prime    replayed       14192      18051    1.10  1.1e-15  rader
      2054  2.13.79          prime    replayed       22157      13928    0.62  5.4e-16  bluestein
      2055  3.5.137          prime    replayed       21638      14064    0.65  4.3e-16  bluestein
      2056  2^3.257          prime    replayed       21639      19108    0.87  5.0e-16  bluestein
      2057  11^2.17          flat     replayed        3749       8711    2.30  9.5e-16  
      2058  2.3.7^3          flat     replayed        3945       2918    0.72  5.6e-16  
      2059  29.71            prime    replayed       22111      21592    0.97  5.8e-16  bluestein
      2060  2^2.5.103        prime    replayed       21977      14654    0.65  6.1e-16  bluestein
      2061  3^2.229          prime    replayed       22120      14279    0.64  6.2e-16  bluestein
      2062  2.1031           prime    replayed       22072      17520    0.79  5.4e-16  bluestein
      2063  2063             prime    replayed       21589      17890    0.81  5.6e-16  bluestein
      2064  2^4.3.43         flat     replayed        4142      13464    3.23  4.1e-16  
      2065  5.7.59           prime    replayed       21089      16576    0.78  3.3e-16  bluestein
      2066  2.1033           prime    replayed       21595      17419    0.79  6.2e-16  bluestein
      2067  3.13.53          prime    replayed       21960      16717    0.74  4.7e-16  bluestein
      2068  2^2.11.47        chain3   replayed        7339      14737    1.55  4.9e-16  flips differ 1.30x
      2069  2069             prime    replayed       21604      18136    0.82  7.1e-16  bluestein
      2070  2.3^2.5.23       chain3   replayed        5917       8518    1.18  6.3e-16  
      2071  19.109           prime    replayed       22515      23873    0.75  6.1e-16  flips differ 1.40x bluestein
      2072  2^3.7.37         chain3   replayed        6539      12026    1.40  4.7e-16  flips differ 1.32x
      2073  3.691            prime    replayed       21589      16011    0.57  5.5e-16  flips differ 1.29x bluestein
      2074  2.17.61          prime    replayed       21928      23421    1.06  4.9e-16  bluestein
      2075  5^2.83           prime    replayed       21642      19299    0.88  4.9e-16  bluestein
      2076  2^2.3.173        prime    replayed       21922      14120    0.64  5.8e-16  bluestein
      2077  31.67            prime    replayed       22165      28467    1.28  5.3e-16  bluestein
      2078  2.1039           prime    replayed       21666      17414    0.79  5.8e-16  bluestein
      2079  3^3.7.11         chain3   replayed        3435       3561    1.03  4.5e-16  
      2080  2^5.5.13         chain3   replayed        2463       2733    0.78  6.5e-16  flips differ 1.43x
      2081  2081             prime    replayed        9410      18161    1.87  8.2e-16  rader
      2082  2.3.347          prime    replayed       21571      14221    0.66  5.6e-16  bluestein
      2083  2083             prime    replayed       21540      18419    0.83  6.4e-16  bluestein
      2084  2^2.521          prime    replayed       22334      16306    0.73  5.3e-16  bluestein
      2085  3.5.139          prime    replayed       22537      14114    0.62  5.4e-16  bluestein
      2086  2.7.149          prime    replayed       21611      15417    0.70  5.1e-16  bluestein
      2087  2087             prime    replayed       22099      17896    0.81  5.9e-16  bluestein
      2088  2^3.3^2.29       flat     replayed        4006       9721    2.42  6.5e-16  
      2089  2089             prime    replayed       12080      18496    1.53  9.8e-16  rader
      2090  2.5.11.19        chain3   replayed        3275       8433    2.53  4.9e-16  
      2091  3.17.41          chain3   replayed        5038      18974    3.71  6.2e-16  
      2092  2^2.523          prime    replayed       21973      16409    0.74  6.4e-16  bluestein
      2093  7.13.23          flat     replayed        4105       9459    2.10  6.9e-16  
      2094  2.3.349          prime    replayed       22208      14469    0.65  6.3e-16  bluestein
      2095  5.419            prime    replayed       22009      15519    0.70  6.0e-16  bluestein
      2096  2^4.131          prime    replayed       21609      15246    0.70  5.4e-16  bluestein
      2097  3^2.233          prime    replayed       22027      14377    0.64  5.2e-16  bluestein
      2098  2.1049           prime    replayed       21586      17471    0.79  5.9e-16  bluestein
      2099  2099             prime    replayed       22548      17942    0.79  5.7e-16  bluestein
      2100  2^2.3.5^2.7      chain3   replayed        2901       2713    0.93  5.7e-16  
      2101  11.191           prime    replayed       22557      12885    0.41  5.0e-16  flips differ 1.38x bluestein
      2102  2.1051           prime    replayed       22018      17410    0.77  5.5e-16  bluestein
      2103  3.701            prime    replayed       22003      15782    0.70  6.9e-16  bluestein
      2104  2^3.263          prime    replayed       22047      16075    0.71  5.3e-16  bluestein
      2105  5.421            prime    replayed       22532      15449    0.68  4.6e-16  bluestein
      2106  2.3^4.13         chain3   replayed        3920       3156    0.79  6.1e-16  
      2107  7^2.43           chain3   replayed        9117      14257    1.38  4.1e-16  
      2108  2^2.17.31        flat     replayed        5139      16002    3.01  8.9e-16  
      2109  3.19.37          flat     replayed        5201      17974    3.45  7.3e-16  
      2110  2.5.211          prime    replayed       21615      14516    0.67  5.2e-16  bluestein
      2111  2111             prime    replayed       21995      17884    0.79  5.7e-16  bluestein
      2112  2^6.3.11         chain3   replayed        2430       2734    1.12  3.8e-16  
      2113  2113             prime    replayed        8791      18157    1.83  8.6e-16  rader
      2114  2.7.151          prime    replayed       22558      13205    0.58  5.2e-16  bluestein
      2115  3^2.5.47         chain3   replayed        7768      15412    1.37  5.0e-16  flips differ 1.45x
      2116  2^2.23^2         flat     replayed        4917      15081    3.00  1.4e-15  
      2117  29.73            prime    replayed       22139      23206    0.74  5.8e-16  flips differ 1.42x bluestein
      2118  2.3.353          prime    replayed       21984      14253    0.46  5.7e-16  flips differ 1.42x bluestein
      2119  13.163           prime    replayed       21651      15937    0.72  5.5e-16  bluestein
      2120  2^3.5.53         prime    replayed       21607      16251    0.73  4.6e-16  bluestein
      2121  3.7.101          prime    replayed       22059      15519    0.69  6.0e-16  bluestein
      2122  2.1061           prime    replayed       21922      17564    0.80  6.3e-16  bluestein
      2123  11.193           prime    replayed       22545      13897    0.61  6.6e-16  bluestein
      2124  2^2.3^2.59       prime    replayed       21633      16931    0.78  4.4e-16  bluestein
      2125  5^3.17           chain3   replayed        3810       8076    1.94  5.6e-16  
      2126  2.1063           prime    replayed       22548      17610    0.78  5.8e-16  bluestein
      2127  3.709            prime    replayed       21995      16062    0.71  5.3e-16  bluestein
      2128  2^4.7.19         chain3   replayed        2838       7913    2.66  4.0e-16  
      2129  2129             prime    replayed       10447      18488    1.71  7.6e-16  rader
      2130  2.3.5.71         prime    replayed       22001      15187    0.67  6.0e-16  bluestein
      2131  2131             prime    replayed       21603      18208    0.54  9.2e-16  flips differ 1.53x bluestein
      2132  2^2.13.41        chain3   replayed        4681      14191    3.00  6.3e-16  
      2133  3^3.79           prime    replayed       22539      14508    0.64  6.1e-16  bluestein
      2134  2.11.97          prime    replayed       21628      13672    0.61  4.8e-16  bluestein
      2135  5.7.61           prime    replayed       22040      18715    0.83  4.6e-16  bluestein
      2136  2^3.3.89         prime    replayed       21993      14407    0.65  4.8e-16  bluestein
      2137  2137             prime    replayed       21972      18190    0.81  7.5e-16  bluestein
      2138  2.1069           prime    replayed       21701      17491    0.79  5.8e-16  bluestein
      2139  3.23.31          flat     replayed        6665      18087    2.40  9.9e-16  
      2140  2^2.5.107        prime    replayed       22567      14895    0.62  5.3e-16  bluestein
      2141  2141             prime    replayed       21991      17993    0.81  6.5e-16  bluestein
      2142  2.3^2.7.17       chain3   replayed        3530       8451    2.39  7.0e-16  
      2143  2143             prime    replayed       12489      18197    1.44  1.1e-15  rader
      2144  2^5.67           prime    replayed       22156      19688    0.89  4.9e-16  bluestein
      2145  3.5.11.13        chain3   replayed        3444       4690    1.32  5.4e-16  
      2146  2.29.37          flat     replayed        5930      20595    3.47  2.9e-15  
      2147  19.113           prime    replayed       22010      24330    1.08  6.4e-16  bluestein
      2148  2^2.3.179        prime    replayed       21621      14308    0.66  5.6e-16  bluestein
      2149  7.307            prime    replayed       22506      18462    0.82  5.6e-16  bluestein
      2150  2.5^2.43         chain3   replayed        4673      14104    3.00  4.3e-16  
      2151  3^2.239          prime    replayed       22478      14443    0.64  5.8e-16  bluestein
      2152  2^3.269          prime    replayed       22207      16382    0.73  5.4e-16  bluestein
      2153  2153             prime    replayed       22030      17941    0.80  6.1e-16  bluestein
      2154  2.3.359          prime    replayed       22007      14400    0.64  6.6e-16  bluestein
      2155  5.431            prime    replayed       22039      15554    0.69  5.9e-16  bluestein
      2156  2^2.7^2.11       flat     replayed        3267       3184    0.85  6.3e-16  
      2157  3.719            prime    replayed       22173      15700    0.71  4.9e-16  bluestein
      2158  2.13.83          prime    replayed       22594      16335    0.72  6.0e-16  bluestein
      2159  17.127           prime    replayed       21636      18230    0.82  5.0e-16  bluestein
      2160  2^4.3^3.5        chain3   replayed        2489       2545    1.01  5.1e-16  
      2161  2161             prime    replayed        8421      21526    2.49  7.6e-16  rader
      2162  2.23.47          flat     replayed        6247      21176    3.19  1.2e-15  
      2163  3.7.103          prime    replayed       21687      15711    0.57  5.6e-16  flips differ 1.28x bluestein
      2164  2^2.541          prime    replayed       22525      24219    1.07  5.6e-16  bluestein
      2165  5.433            prime    replayed       22215      21306    0.95  4.5e-16  bluestein
      2166  2.3.19^2         chain3   replayed        4151      13623    3.11  5.9e-16  
      2167  11.197           prime    replayed       22360      14136    0.63  5.2e-16  bluestein
      2168  2^3.271          prime    replayed       22178      17921    0.81  6.0e-16  bluestein
      2169  3^2.241          prime    replayed       22011      17487    0.79  7.1e-16  bluestein
      2170  2.5.7.31         chain3   replayed        6205      11951    1.55  5.6e-16  
      2171  13.167           prime    replayed       22267      21927    0.97  5.1e-16  bluestein
      2172  2^2.3.181        prime    replayed       21675      17996    0.81  5.5e-16  bluestein
      2173  41.53            prime    replayed       22613      28162    1.16  4.5e-16  bluestein
      2174  2.1087           prime    replayed       21662      23166    1.04  7.3e-16  bluestein
      2175  3.5^2.29         chain3   replayed        4477      10866    2.20  5.0e-16  
      2176  2^7.17           chain3   replayed        2922       7302    2.35  4.5e-16  
      2177  7.311            prime    replayed       21683      18532    0.85  5.5e-16  bluestein
      2178  2.3^2.11^2       flat     replayed        4856       4415    0.83  6.5e-16  
      2179  2179             prime    replayed       21674      21594    0.66  8.0e-16  flips differ 1.52x bluestein
      2180  2^2.5.109        prime    replayed       21634      19145    0.88  5.5e-16  bluestein
      2181  3.727            prime    replayed       21692      20159    0.91  6.8e-16  bluestein
      2182  2.1091           prime    replayed       22035      23120    1.04  7.5e-16  bluestein
      2183  37.59            prime    replayed       22241      28809    1.29  3.7e-16  bluestein
      2184  2^3.3.7.13       chain3   replayed        3433       3182    0.78  6.8e-16  
      2185  5.19.23          chain3   replayed        6065      14779    2.19  4.7e-16  
      2186  2.1093           prime    replayed       22006      23177    1.02  6.8e-16  bluestein
      2187  3^7              chain3   replayed        3932       3957    0.95  5.4e-16  
      2188  2^2.547          prime    replayed       22579      20540    0.91  5.3e-16  bluestein
      2189  11.199           prime    replayed       21996      13932    0.62  5.6e-16  bluestein
      2190  2.3.5.73         prime    replayed       22042      16173    0.72  5.9e-16  bluestein
      2191  7.313            prime    replayed       21619      18689    0.68  6.8e-16  flips differ 1.28x bluestein
      2192  2^4.137          prime    replayed       21654      15124    0.68  4.8e-16  bluestein
      2193  3.17.43          chain3   replayed        7899      19831    1.74  6.0e-16  flips differ 1.44x
      2194  2.1097           prime    replayed       22389      20174    0.89  5.2e-16  bluestein
      2195  5.439            prime    replayed       21625      20998    0.94  5.4e-16  bluestein
      2196  2^2.3^2.61       prime    replayed       21625      19111    0.88  5.3e-16  bluestein
      2197  13^3             flat     replayed        4004       5187    1.22  1.2e-15  
      2198  2.7.157          prime    replayed       21652      14017    0.63  6.8e-16  bluestein
      2199  3.733            prime    replayed       22555      18650    0.82  6.1e-16  bluestein
      2200  2^3.5^2.11       flat     replayed        3203       2919    0.91  3.9e-16  
      2201  31.71            prime    replayed       21683      24729    1.11  5.4e-16  bluestein
      2202  2.3.367          prime    replayed       21659      17374    0.80  5.8e-16  bluestein
      2203  2203             prime    replayed       21474      19418    0.84  5.7e-16  bluestein
      2204  2^2.19.29        flat     replayed        5343      16162    2.72  9.6e-16  
      2205  3^2.5.7^2        chain3   replayed        3540       3570    0.96  5.8e-16  
      2206  2.1103           prime    replayed       22060      20258    0.90  6.0e-16  bluestein
      2207  2207             prime    replayed       21675      19378    0.87  5.2e-16  bluestein
      2208  2^5.3.23         chain3   replayed        3070       8779    2.85  4.7e-16  
      2209  47^2             flat     replayed        6246      27660    4.38  2.0e-15  
      2210  2.5.13.17        chain3   replayed        3790       8738    2.22  6.6e-16  
      2211  3.11.67          prime    replayed       21625      21492    0.97  5.2e-16  bluestein
      2212  2^2.7.79         prime    replayed       22008      15227    0.68  4.8e-16  bluestein
      2213  2213             prime    replayed       22009      19533    0.59  7.1e-16  flips differ 1.51x bluestein
      2214  2.3^3.41         chain3   replayed        6647      14632    1.60  4.8e-16  flips differ 1.38x
      2215  5.443            prime    replayed       22182      21135    0.94  6.4e-16  bluestein
      2216  2^3.277          prime    replayed       21635      16422    0.74  4.9e-16  bluestein
      2217  3.739            prime    replayed       22226      18578    0.83  5.1e-16  bluestein
      2218  2.1109           prime    replayed       22549      20124    0.89  6.2e-16  bluestein
      2219  7.317            prime    replayed       22039      13306    0.60  4.8e-16  bluestein
      2220  2^2.3.5.37       chain3   replayed        4022      12767    3.15  4.0e-16  
      2221  2221             prime    replayed       15653      19450    1.08  1.6e-15  rader
      2222  2.11.101         prime    replayed       22062      17119    0.75  6.3e-16  bluestein
      2223  3^2.13.19        flat     replayed        4424       9271    2.09  1.2e-15  
      2224  2^4.139          prime    replayed       22421      14985    0.66  4.3e-16  bluestein
      2225  5^2.89           prime    replayed       21658      19528    0.90  4.9e-16  bluestein
      2226  2.3.7.53         prime    replayed       22046      17854    0.81  4.4e-16  bluestein
      2227  17.131           prime    replayed       22223      21125    0.95  6.4e-16  bluestein
      2228  2^2.557          prime    replayed       22023      20507    0.91  6.6e-16  bluestein
      2229  3.743            prime    replayed       22126      18614    0.82  6.5e-16  bluestein
      2230  2.5.223          prime    replayed       22572      17634    0.78  4.5e-16  bluestein
      2231  23.97            prime    replayed       22565      22681    1.00  5.9e-16  bluestein
      2232  2^3.3^2.31       flat     replayed        5362      11290    1.96  7.2e-16  
      2233  7.11.29          chain3   replayed        4652      11728    2.49  4.8e-16  
      2234  2.1117           prime    replayed       22245      20177    0.90  6.4e-16  bluestein
      2235  3.5.149          prime    replayed       22029      15423    0.70  4.5e-16  bluestein
      2236  2^2.13.43        chain3   replayed        5035      15450    3.05  6.8e-16  
      2237  2237             prime    replayed       22043      19518    0.87  5.9e-16  bluestein
      2238  2.3.373          prime    replayed       21684      17398    0.80  5.9e-16  bluestein
      2239  2239             prime    replayed       22425      19481    0.87  4.7e-16  bluestein
      2240  2^6.5.7          ztt      replayed        2031       2539    1.23  3.4e-16  
      2241  3^3.83           prime    replayed       22388      16630    0.73  4.6e-16  bluestein
      2242  2.19.59          prime    replayed       22688      20507    0.90  4.6e-16  bluestein
      2243  2243             prime    replayed       21662      19515    0.88  6.9e-16  bluestein
      2244  2^2.3.11.17      chain3   replayed        3145       8874    2.81  5.2e-16  
      2245  5.449            prime    replayed       22407      21144    0.94  5.0e-16  bluestein
      2246  2.1123           prime    replayed       21676      20263    0.91  5.0e-16  bluestein
      2247  3.7.107          prime    replayed       22205      15742    0.70  5.8e-16  bluestein
      2248  2^3.281          prime    replayed       22183      16457    0.74  6.2e-16  bluestein
      2249  13.173           prime    replayed       21743      16096    0.72  6.6e-16  bluestein
      2250  2.3^2.5^3        chain3   replayed        3278       3108    0.81  5.1e-16  
      2251  2251             prime    replayed       13082      17750    1.30  1.2e-15  rader
      2252  2^2.563          prime    replayed       21685      20472    0.92  5.6e-16  bluestein
      2253  3.751            prime    replayed       21714      19380    0.87  5.7e-16  bluestein
      2254  2.7^2.23         flat     replayed        5467       9563    1.73  5.6e-16  
      2255  5.11.41          chain3   replayed        5305      15160    2.85  5.2e-16  
      2256  2^4.3.47         chain3   replayed        4730      15542    3.26  5.6e-16  
      2257  37.61            prime    replayed       21702      30318    1.36  3.8e-16  bluestein
      2258  2.1129           prime    replayed       22032      17856    0.80  5.5e-16  bluestein
      2259  3^2.251          prime    replayed       21658      13710    0.61  5.6e-16  bluestein
      2260  2^2.5.113        prime    replayed       22571      19060    0.84  5.8e-16  bluestein
      2261  7.17.19          flat     replayed        4148      14322    2.10  8.4e-16  flips differ 1.65x
      2262  2.3.13.29        chain3   replayed        5369      11633    1.47  7.4e-16  flips differ 1.47x
      2263  31.73            prime    replayed       21678      25936    1.19  5.8e-16  bluestein
      2264  2^3.283          prime    replayed       21989      16594    0.73  5.7e-16  bluestein
      2265  3.5.151          prime    replayed       22074      14151    0.63  5.3e-16  bluestein
      2266  2.11.103         prime    replayed       21995      16869    0.75  8.0e-16  bluestein
      2267  2267             prime    replayed       21644      17811    0.80  6.2e-16  bluestein
      2268  2^2.3^4.7        chain3   replayed        2870       3215    0.83  5.0e-16  flips differ 1.35x
      2269  2269             prime    replayed       15047      17727    1.07  8.3e-16  rader
      2270  2.5.227          prime    replayed       22239      15805    0.56  5.2e-16  flips differ 1.26x bluestein
      2271  3.757            prime    replayed       21727      19290    0.86  5.5e-16  bluestein
      2272  2^5.71           prime    replayed       22571      15599    0.69  5.9e-16  bluestein
      2273  2273             prime    replayed       21962      17894    0.81  5.4e-16  bluestein
      2274  2.3.379          prime    replayed       22076      14421    0.64  4.9e-16  bluestein
      2275  5^2.7.13         chain3   replayed        3901       4113    0.95  5.3e-16  
      2276  2^2.569          prime    replayed       22229      16932    0.52  5.9e-16  flips differ 1.49x bluestein
      2277  3^2.11.23        chain3   replayed        4451      10291    2.28  4.7e-16  
      2278  2.17.67          prime    replayed       22085      25696    1.16  5.7e-16  bluestein
      2279  43.53            prime    replayed       22055      30085    1.36  4.9e-16  bluestein
      2280  2^3.3.5.19       chain3   replayed        3082       8222    2.61  5.0e-16  
      2281  2281             prime    replayed       12386      17803    1.35  7.8e-16  rader
      2282  2.7.163          prime    replayed       22597      16675    0.74  6.1e-16  bluestein
      2283  3.761            prime    replayed       22326      19630    0.87  5.9e-16  bluestein
      2284  2^2.571          prime    replayed       22331      16925    0.75  4.9e-16  bluestein
      2285  5.457            prime    replayed       21679      16429    0.76  5.3e-16  bluestein
      2286  2.3^2.127        prime    replayed       22196      14255    0.64  4.8e-16  bluestein
      2287  2287             prime    replayed       21665      17818    0.80  5.7e-16  bluestein
      2288  2^4.11.13        chain3   replayed        2957       4564    1.54  6.4e-16  
      2289  3.7.109          prime    replayed       22580      21176    0.68  6.6e-16  flips differ 1.38x bluestein
      2290  2.5.229          prime    replayed       21663      15772    0.71  5.1e-16  bluestein
      2291  29.79            prime    replayed       22253      23049    0.83  5.1e-16  flips differ 1.25x bluestein
      2292  2^2.3.191        prime    replayed       22461      13924    0.61  4.7e-16  bluestein
      2293  2293             prime    replayed       21679      18082    0.81  4.9e-16  bluestein
      2294  2.31.37          flat     replayed        6589      23103    3.47  1.7e-15  
      2295  3^3.5.17         chain3   replayed        3989       8640    2.08  6.2e-16  
      2296  2^3.7.41         flat     replayed        5084      14485    2.70  5.5e-16  
      2297  2297             prime    replayed       22584      17922    0.79  5.0e-16  bluestein
      2298  2.3.383          prime    replayed       22070      14669    0.66  5.0e-16  bluestein
      2299  11^2.19          flat     replayed        4588      10022    1.82  1.3e-15  
      2300  2^2.5^2.23       chain3   replayed        3634       9258    2.53  5.4e-16  
      2301  3.13.59          prime    replayed       22046      15933    0.70  5.1e-16  bluestein
      2302  2.1151           prime    replayed       22209      18120    0.81  4.6e-16  bluestein
      2303  7^2.47           chain3   replayed        8184      16797    1.51  5.3e-16  flips differ 1.34x
      2304  2^8.3^2          ztt      replayed        2132       2554    1.19  3.7e-16  
      2305  5.461            prime    replayed       21674      16430    0.74  5.6e-16  bluestein
      2306  2.1153           prime    replayed       22016      18376    0.82  5.7e-16  bluestein
      2307  3.769            prime    replayed       21661      17745    0.79  5.1e-16  bluestein
      2308  2^2.577          prime    replayed       22228      20486    0.92  5.3e-16  bluestein
      2309  2309             prime    replayed       21736      18862    0.85  5.4e-16  bluestein
      2310  2.3.5.7.11       chain3   replayed        3511       3307    0.88  6.1e-16  
      2311  2311             prime    replayed       12235      18998    1.54  1.1e-15  rader
      2312  2^3.17^2         chain3   replayed        4219      13788    2.89  8.5e-16  
      2313  3^2.257          prime    replayed       22649      18216    0.57  5.5e-16  flips differ 1.41x bluestein
      2314  2.13.89          prime    replayed       21696      16664    0.75  5.4e-16  bluestein
      2315  5.463            prime    replayed       22074      16454    0.73  6.4e-16  bluestein
      2316  2^2.3.193        prime    replayed       21680      14717    0.68  6.4e-16  bluestein
      2317  7.331            prime    replayed       21592      19804    0.90  5.8e-16  bluestein
      2318  2.19.61          prime    replayed       21650      25455    1.16  5.0e-16  bluestein
      2319  3.773            prime    replayed       22602      17593    0.75  4.9e-16  bluestein
      2320  2^4.5.29         flat     replayed        4650      11638    2.40  5.3e-16  
      2321  11.211           prime    replayed       22550      16891    0.74  5.9e-16  bluestein
      2322  2.3^3.43         chain3   replayed        8125      15203    1.27  5.0e-16  flips differ 1.47x
      2323  23.101           prime    replayed       21660      23461    1.06  5.9e-16  bluestein
      2324  2^2.7.83         prime    replayed       21633      21576    0.96  4.7e-16  bluestein
      2325  3.5^2.31         chain3   replayed        6817      13263    1.60  5.7e-16  
      2326  2.1163           prime    replayed       22037      18614    0.83  5.8e-16  bluestein
      2327  13.179           prime    replayed       22209      16095    0.72  5.4e-16  bluestein
      2328  2^3.3.97         prime    replayed       22037      17436    0.78  5.3e-16  bluestein
      2329  17.137           prime    replayed       22623      20925    0.92  5.7e-16  bluestein
      2330  2.5.233          prime    replayed       21679      15871    0.73  4.9e-16  bluestein
      2331  3^2.7.37         chain3   replayed        6952      14043    2.02  5.8e-16  
      2332  2^2.11.53        prime    replayed       21692      18927    0.85  6.1e-16  bluestein
      2333  2333             prime    replayed       21629      18820    0.85  5.4e-16  bluestein
      2334  2.3.389          prime    replayed       22154      16355    0.74  5.0e-16  bluestein
      2335  5.467            prime    replayed       21665      16557    0.74  7.3e-16  bluestein
      2336  2^5.73           prime    replayed       22667      16895    0.73  6.4e-16  bluestein
      2337  3.19.41          chain3   replayed        6128      21038    3.13  5.0e-16  
      2338  2.7.167          prime    replayed       22256      16673    0.74  5.4e-16  bluestein
      2339  2339             prime    replayed       21681      18863    0.85  5.7e-16  bluestein
      2340  2^2.3^2.5.13     chain3   replayed        3089       3323    1.07  7.6e-16  
      2341  2341             prime    replayed       11114      18817    1.56  9.0e-16  rader
      2342  2.1171           prime    replayed       22619      18189    0.80  4.9e-16  bluestein
      2343  3.11.71          prime    replayed       22209      16797    0.75  5.6e-16  bluestein
      2344  2^3.293          prime    replayed       22038      16400    0.72  5.3e-16  bluestein
      2345  5.7.67           prime    replayed       22176      22552    0.94  4.0e-16  bluestein
      2346  2.3.17.23        chain3   replayed        4391      15510    3.38  5.8e-16  
      2347  2347             prime    replayed       14631      18822    1.20  1.0e-15  rader
      2348  2^2.587          prime    replayed       21669      18329    0.83  4.7e-16  bluestein
      2349  3^4.29           chain3   replayed        5573      11578    1.62  5.3e-16  flips differ 1.28x
      2350  2.5^2.47         chain3   replayed        8726      16776    1.29  4.5e-16  flips differ 1.49x
      2351  2351             prime    replayed       21736      18877    0.85  6.5e-16  bluestein
      2352  2^4.3.7^2        ztt      replayed        2289       3124    1.36  4.6e-16  
      2353  13.181           prime    replayed       21709      21113    0.97  5.6e-16  bluestein
      2354  2.11.107         prime    replayed       22475      17104    0.75  5.7e-16  bluestein
      2355  3.5.157          prime    replayed       22145      14344    0.64  5.3e-16  bluestein
      2356  2^2.19.31        chain3   replayed        5039      18246    3.27  5.3e-16  
      2357  2357             prime    replayed       21741      18816    0.84  5.1e-16  bluestein
      2358  2.3^2.131        prime    replayed       21709      16944    0.78  6.7e-16  bluestein
      2359  7.337            prime    replayed       21694      19820    0.89  6.1e-16  bluestein
      2360  2^3.5.59         prime    replayed       22285      19000    0.79  4.5e-16  bluestein
      2361  3.787            prime    replayed       21708      17219    0.79  5.9e-16  bluestein
      2362  2.1181           prime    replayed       22053      18256    0.81  6.3e-16  bluestein
      2363  17.139           prime    replayed       22672      21252    0.60  5.8e-16  flips differ 1.58x bluestein
      2364  2^2.3.197        prime    replayed       22062      14612    0.65  5.6e-16  bluestein
      2365  5.11.43          chain3   replayed        8198      16265    1.47  5.9e-16  flips differ 1.35x
      2366  2.7.13^2         flat     replayed        5397       4979    0.89  9.9e-16  
      2367  3^2.263          prime    replayed       21707      18105    0.65  7.5e-16  flips differ 1.29x bluestein
      2368  2^6.37           2p       replayed        8232      14435    1.56  4.9e-16  
      2369  23.103           prime    replayed       22047      23872    1.08  6.3e-16  bluestein
      2370  2.3.5.79         prime    replayed       22365      15294    0.67  5.6e-16  bluestein
      2371  2371             prime    replayed       21678      18823    0.85  5.7e-16  bluestein
      2372  2^2.593          prime    replayed       22121      18197    0.80  6.0e-16  bluestein
      2373  3.7.113          prime    replayed       22604      21193    0.94  6.4e-16  bluestein
      2374  2.1187           prime    replayed       22097      18553    0.82  5.5e-16  bluestein
      2375  5^3.19           flat     replayed        4033       9615    2.26  4.8e-16  
      2376  2^3.3^3.11       chain3   replayed        4015       3396    0.74  4.6e-16  
      2377  2377             prime    replayed       15723      18712    1.19  8.5e-16  rader
      2378  2.29.41          flat     replayed        6778      23935    3.52  2.5e-15  
      2379  3.13.61          prime    replayed       21713      21359    0.97  5.0e-16  bluestein
      2380  2^2.5.7.17       flat     replayed        4059       8439    2.05  5.4e-16  
      2381  2381             prime    replayed       21727      18830    0.85  5.6e-16  bluestein
      2382  2.3.397          prime    replayed       22062      16313    0.73  5.3e-16  bluestein
      2383  2383             prime    replayed       22044      18873    0.83  5.5e-16  bluestein
      2384  2^4.149          prime    replayed       21708      16400    0.74  6.4e-16  bluestein
      2385  3^2.5.53         prime    replayed       22664      18774    0.83  4.0e-16  bluestein
      2386  2.1193           prime    replayed       22066      18756    0.84  6.9e-16  bluestein
      2387  7.11.31          chain3   replayed        7020      13153    1.87  5.4e-16  
      2388  2^2.3.199        prime    replayed       22079      14709    0.64  5.3e-16  bluestein
      2389  2389             prime    replayed       21679      18884    0.87  6.4e-16  bluestein
      2390  2.5.239          prime    replayed       21670      15945    0.72  6.0e-16  bluestein
      2391  3.797            prime    replayed       22066      17440    0.77  5.8e-16  bluestein
      2392  2^3.13.23        chain3   replayed        4643      10777    2.27  7.9e-16  
      2393  2393             prime    replayed       13867      18882    1.26  9.1e-16  rader
      2394  2.3^2.7.19       chain3   replayed        4334       9086    1.83  6.5e-16  
      2395  5.479            prime    replayed       21724      16608    0.74  5.1e-16  bluestein
      2396  2^2.599          prime    replayed       22083      20413    0.92  6.5e-16  bluestein
      2397  3.17.47          flat     replayed        6944      23043    3.12  1.8e-15  
      2398  2.11.109         prime    replayed       21744      21994    0.66  6.1e-16  flips differ 1.52x bluestein
      2399  2399             prime    replayed       21770      18883    0.85  5.6e-16  bluestein
      2400  2^5.3.5^2        ztt      replayed        2421       2607    0.94  3.8e-16  
      2401  7^4              flat     replayed        5199       4500    0.85  5.4e-16  
      2402  2.1201           prime    replayed       21740      21532    0.97  7.1e-16  bluestein
      2403  3^3.89           prime    replayed       21710      21177    0.95  5.6e-16  bluestein
      2404  2^2.601          prime    replayed       22683      25905    1.14  7.5e-16  bluestein
      2405  5.13.37          chain3   replayed        5300      15290    2.85  5.9e-16  
      2406  2.3.401          prime    replayed       22355      18112    0.81  4.5e-16  bluestein
      2407  29.83            prime    replayed       22618      26016    1.15  5.9e-16  bluestein
      2408  2^3.7.43         chain3   replayed        4751      16497    3.46  4.8e-16  
      2409  3.11.73          prime    replayed       22102      17807    0.79  6.2e-16  bluestein
      2410  2.5.241          prime    replayed       22040      19572    0.85  7.4e-16  bluestein
      2411  2411             prime    replayed       21794      22407    1.03  5.8e-16  bluestein
      2412  2^2.3^2.67       prime    replayed       22109      21251    0.96  5.3e-16  bluestein
      2413  19.127           prime    replayed       22067      20706    0.93  4.0e-16  bluestein
      2414  2.17.71          prime    replayed       21712      23303    1.07  5.2e-16  bluestein
      2415  3.5.7.23         chain3   replayed        5562      10476    1.50  4.4e-16  flips differ 1.25x
      2416  2^4.151          prime    replayed       22268      15868    0.71  4.8e-16  bluestein
      2417  2417             prime    replayed       21291      22404    0.88  7.1e-16  bluestein
      2418  2.3.13.31        flat     replayed        6218      13447    2.14  1.1e-15  
      2419  41.59            prime    replayed       22612      33123    1.46  5.9e-16  bluestein
      2420  2^2.5.11^2       flat     replayed        4004       4748    1.18  6.6e-16  
      2421  3^2.269          prime    replayed       22011      18412    0.69  7.0e-16  bluestein
      2422  2.7.173          prime    replayed       22122      16946    0.74  6.1e-16  bluestein
      2423  2423             prime    replayed       22133      22483    1.01  7.0e-16  bluestein
      2424  2^3.3.101        prime    replayed       22148      17429    0.77  5.9e-16  bluestein
      2425  5^2.97           prime    replayed       22289      18144    0.80  7.0e-16  bluestein
      2426  2.1213           prime    replayed       21813      21581    0.97  7.0e-16  bluestein
      2427  3.809            prime    replayed       21834      20207    0.91  7.0e-16  bluestein
      2428  2^2.607          prime    replayed       22389      22492    1.00  4.8e-16  bluestein
      2429  7.347            prime    replayed       22092      16783    0.74  5.5e-16  bluestein
      2430  2.3^5.5          chain3   replayed        4378       3420    0.76  5.1e-16  
      2431  11.13.17         chain3   replayed        4496      10338    2.13  6.0e-16  
      2432  2^7.19           chain3   replayed        3588       9650    2.62  4.2e-16  
      2433  3.811            prime    replayed       21722      25039    1.11  9.4e-16  bluestein
      2434  2.1217           prime    replayed       22084      21368    0.97  5.1e-16  bluestein
      2435  5.487            prime    replayed       22695      21672    0.94  6.1e-16  bluestein
      2436  2^2.3.7.29       flat     replayed        5194      11949    1.51  6.5e-16  flips differ 1.53x
      2437  2437             prime    replayed       15968      21334    1.23  8.8e-16  rader
      2438  2.23.53          prime    replayed       21775      26404    1.18  4.5e-16  bluestein
      2439  3^2.271          prime    replayed       21786      18496    0.82  4.4e-16  bluestein
      2440  2^3.5.61         prime    replayed       22301      21063    0.94  4.2e-16  bluestein
      2441  2441             prime    replayed       22300      21264    0.95  8.1e-16  bluestein
      2442  2.3.11.37        chain3   replayed        5106      15279    2.94  4.4e-16  
      2443  7.349            prime    replayed       21755      16870    0.77  6.4e-16  bluestein
      2444  2^2.13.47        flat     replayed        6049      17077    2.58  8.8e-16  
      2445  3.5.163          prime    replayed       21772      17923    0.80  6.7e-16  bluestein
      2446  2.1223           prime    replayed       21729      21525    0.98  5.3e-16  bluestein
      2447  2447             prime    replayed       21760      21391    0.96  7.0e-16  bluestein
      2448  2^4.3^2.17       chain3   replayed        3182       8759    2.67  5.5e-16  
      2449  31.79            prime    replayed       22128      25537    1.15  5.3e-16  bluestein
      2450  2.5^2.7^2        flat     replayed        4625       3385    0.72  4.5e-16  
      2451  3.19.43          flat     replayed        7046      23080    3.27  1.2e-15  
      2452  2^2.613          prime    replayed       21765      20884    0.95  5.5e-16  bluestein
      2453  11.223           prime    replayed       22153      20160    0.90  4.9e-16  bluestein
      2454  2.3.409          prime    replayed       22078      18402    0.83  6.7e-16  bluestein
      2455  5.491            prime    replayed       22631      21582    0.94  5.0e-16  bluestein
      2456  2^3.307          prime    replayed       21765      21558    0.97  6.8e-16  bluestein
      2457  3^3.7.13         chain3   replayed        4442       4445    0.99  6.6e-16  
      2458  2.1229           prime    replayed       22307      21437    0.96  5.8e-16  bluestein
      2459  2459             prime    replayed       22137      21304    0.94  6.7e-16  bluestein
      2460  2^2.3.5.41       flat     replayed        5305      16171    3.05  4.8e-16  
      2461  23.107           prime    replayed       22136      24053    1.08  5.0e-16  bluestein
      2462  2.1231           prime    replayed       21800      21614    0.97  7.6e-16  bluestein
      2463  3.821            prime    replayed       21806      24689    1.10  6.6e-16  bluestein
      2464  2^5.7.11         chain3   replayed        2941       3422    1.14  4.8e-16  
      2465  5.17.29          flat     replayed        6040      18329    2.98  1.6e-15  
      2466  2.3^2.137        prime    replayed       22150      21708    0.96  5.7e-16  bluestein
      2467  2467             prime    replayed       21776      21298    0.97  6.2e-16  bluestein
      2468  2^2.617          prime    replayed       22137      20687    0.93  5.9e-16  bluestein
      2469  3.823            prime    replayed       21816      24751    1.13  8.2e-16  bluestein
      2470  2.5.13.19        chain3   replayed        5315      10234    1.88  5.7e-16  
      2471  7.353            prime    replayed       22099      16801    0.76  5.8e-16  bluestein
      2472  2^3.3.103        prime    replayed       22318      17829    0.80  6.0e-16  bluestein
      2473  2473             prime    replayed       21323      21345    0.83  6.6e-16  bluestein
      2474  2.1237           prime    replayed       22135      21895    0.97  5.8e-16  bluestein
      2475  3^2.5^2.11       chain3   replayed        4080       4358    0.95  6.1e-16  
      2476  2^2.619          prime    replayed       21831      20760    0.93  6.2e-16  bluestein
      2477  2477             prime    replayed       22325      21354    0.95  5.9e-16  bluestein
      2478  2.3.7.59         prime    replayed       22331      19758    0.88  5.5e-16  bluestein
      2479  37.67            prime    replayed       21785      35193    1.57  5.0e-16  bluestein
      2480  2^4.5.31         flat     replayed        4820      12504    2.15  4.3e-16  
      2481  3.827            prime    replayed       21812      24713    1.10  6.5e-16  bluestein
      2482  2.17.73          prime    replayed       21343      23905    1.10  5.4e-16  bluestein
      2483  13.191           prime    replayed       22123      15514    0.69  5.5e-16  bluestein
      2484  2^2.3^3.23       chain3   replayed        3654      10451    2.84  6.6e-16  
      2485  5.7.71           prime    replayed       22339      17409    0.78  5.0e-16  bluestein
      2486  2.11.113         prime    replayed       21829      21908    1.00  5.0e-16  bluestein
      2487  3.829            prime    replayed       22351      25841    1.10  5.6e-16  bluestein
      2488  2^3.311          prime    replayed       22173      21869    0.98  6.0e-16  bluestein
      2489  19.131           prime    replayed       21856      24303    1.09  6.1e-16  bluestein
      2490  2.3.5.83         prime    replayed       22080      18086    0.81  5.3e-16  bluestein
      2491  47.53            prime    replayed       22194      34072    1.50  5.5e-16  bluestein
      2492  2^2.7.89         prime    replayed       22212      21600    0.95  4.2e-16  bluestein
      2493  3^2.277          prime    replayed       21816      18489    0.83  6.4e-16  bluestein
      2494  2.29.43          flat     replayed        7485      25755    3.38  3.0e-15  
      2495  5.499            prime    replayed       21833      21365    0.98  5.4e-16  bluestein
      2496  2^6.3.13         chain3   replayed        2975       3430    1.13  6.7e-16  
      2497  11.227           prime    replayed       21825      18212    0.82  4.8e-16  bluestein
      2498  2.1249           prime    replayed       21750      21578    0.99  6.2e-16  bluestein
      2499  3.7^2.17         chain3   replayed        4046       9320    2.25  5.7e-16  
      2500  2^2.5^4          chain3   replayed        3621       3353    0.92  5.2e-16  
      2501  41.61            prime    replayed       22101      34829    1.53  4.3e-16  bluestein
      2502  2.3^2.139        prime    replayed       22330      21610    0.97  5.1e-16  bluestein
      2503  2503             prime    replayed       21778      20178    0.91  6.0e-16  bluestein
      2504  2^3.313          prime    replayed       21843      21642    0.98  5.1e-16  bluestein
      2505  3.5.167          prime    replayed       22126      18069    0.82  5.8e-16  bluestein
      2506  2.7.179          prime    replayed       21811      16804    0.75  5.9e-16  bluestein
      2507  23.109           prime    replayed       21736      30365    1.38  6.1e-16  bluestein
      2508  2^2.3.11.19      flat     replayed        4657      10244    2.11  1.4e-15  
      2509  13.193           prime    replayed       21830      16677    0.74  6.3e-16  bluestein
      2510  2.5.251          prime    replayed       21805      14873    0.67  4.8e-16  bluestein
      2511  3^4.31           flat     replayed        6427      13348    1.96  7.5e-16  
      2512  2^4.157          prime    replayed       21831      16451    0.74  4.7e-16  bluestein
      2513  7.359            prime    replayed       22134      16891    0.73  4.4e-16  bluestein
      2514  2.3.419          prime    replayed       21764      18746    0.84  8.1e-16  bluestein
      2515  5.503            prime    replayed       21843      15491    0.69  5.6e-16  bluestein
      2516  2^2.17.37        chain3   replayed        5716      20886    3.62  5.8e-16  
      2517  3.839            prime    replayed       21814      24636    1.13  6.6e-16  bluestein
      2518  2.1259           prime    replayed       22273      19455    0.87  4.8e-16  bluestein
      2519  11.229           prime    replayed       22647      18307    0.80  4.7e-16  bluestein
      2520  2^3.3^2.5.7      chain3   replayed        3184       3121    0.98  5.1e-16  
      2521  2521             prime    replayed       14757      20142    1.35  8.7e-16  rader
      2522  2.13.97          prime    replayed       21859      16353    0.73  5.4e-16  bluestein
      2523  3.29^2           flat     replayed        6545      21767    3.13  1.5e-15  
      2524  2^2.631          prime    replayed       22634      17027    0.75  5.5e-16  bluestein
      2525  5^2.101          prime    replayed       22409      18623    0.83  5.7e-16  bluestein
      2526  2.3.421          prime    replayed       22176      18632    0.84  6.4e-16  bluestein
      2527  7.19^2           chain3   replayed        5076      16440    3.18  4.7e-16  
      2528  2^5.79           prime    replayed       22189      16294    0.72  5.0e-16  bluestein
      2529  3^2.281          prime    replayed       21850      18490    0.82  7.0e-16  bluestein
      2530  2.5.11.23        chain3   replayed        4064      11359    2.74  6.0e-16  
      2531  2531             prime    replayed       15375      20065    1.30  1.0e-15  rader
      2532  2^2.3.211        prime    replayed       21762      17798    0.79  7.0e-16  bluestein
      2533  17.149           prime    replayed       21767      24603    1.11  6.2e-16  bluestein
      2534  2.7.181          prime    replayed       22190      21277    0.94  4.7e-16  bluestein
      2535  3.5.13^2         chain3   replayed        4329       5325    1.16  5.9e-16  
      2536  2^3.317          prime    replayed       22806      15405    0.68  6.5e-16  bluestein
      2537  43.59            prime    replayed       22054      35315    1.60  4.8e-16  bluestein
      2538  2.3^3.47         chain3   replayed        9109      18247    1.51  4.9e-16  flips differ 1.33x
      2539  2539             prime    replayed       21866      20151    0.92  4.5e-16  bluestein
      2540  2^2.5.127        prime    replayed       22523      14627    0.64  5.0e-16  bluestein
      2541  3.7.11^2         flat     replayed        5273       5264    0.99  7.0e-16  
      2542  2.31.41          flat     replayed        7609      26743    3.51  2.6e-15  
      2543  2543             prime    replayed       22698      20023    0.87  5.9e-16  bluestein
      2544  2^4.3.53         prime    replayed       21756      19433    0.89  5.3e-16  bluestein
      2545  5.509            prime    replayed       22125      16009    0.69  5.8e-16  bluestein
      2546  2.19.67          prime    replayed       22122      31449    1.40  4.4e-16  bluestein
      2547  3^2.283          prime    replayed       22480      18393    0.79  5.0e-16  bluestein
      2548  2^2.7^2.13       flat     replayed        3955       3961    1.00  6.3e-16  
      2549  2549             prime    replayed       21818      20142    0.90  5.4e-16  bluestein
      2550  2.3.5^2.17       chain3   replayed        3810       9085    2.36  7.3e-16  
      2551  2551             prime    replayed       12884      20131    1.49  8.8e-16  rader
      2552  2^3.11.29        chain3   replayed        4465      13263    2.96  4.1e-16  
      2553  3.23.37          flat     replayed        6647      22997    3.29  8.7e-16  
      2554  2.1277           prime    replayed       22758      19421    0.85  5.6e-16  bluestein
      2555  5.7.73           prime    replayed       21825      18306    0.81  6.3e-16  bluestein
      2556  2^2.3^2.71       prime    replayed       22155      18325    0.78  5.0e-16  bluestein
      2557  2557             prime    replayed       21362      20041    0.92  5.4e-16  bluestein
      2558  2.1279           prime    replayed       21803      19551    0.89  5.2e-16  bluestein
      2559  3.853            prime    replayed       21826      24864    1.12  5.9e-16  bluestein
      2560  2^9.5            ztt      replayed        2184       2836    1.29  3.2e-16  
      2561  13.197           prime    replayed       22200      16795    0.75  4.8e-16  bluestein
      2562  2.3.7.61         prime    replayed       22109      22337    1.01  4.9e-16  bluestein
      2563  11.233           prime    replayed       22229      18371    0.81  4.7e-16  bluestein
      2564  2^2.641          prime    replayed       22102      20505    0.89  6.3e-16  bluestein
      2565  3^3.5.19         chain3   replayed        4714      10110    2.10  5.0e-16  
      2566  2.1283           prime    replayed       21806      22338    1.02  7.4e-16  bluestein
      2567  17.151           prime    replayed       21848      22669    1.01  5.5e-16  bluestein
      2568  2^3.3.107        prime    replayed       22703      17966    0.78  7.9e-16  bluestein
      2569  7.367            prime    replayed       21842      20333    0.91  4.8e-16  bluestein
      2570  2.5.257          prime    replayed       22164      20080    0.89  4.8e-16  bluestein
      2571  3.857            prime    replayed       22137      24647    1.09  5.7e-16  bluestein
      2572  2^2.643          prime    replayed       22668      20491    0.90  6.6e-16  bluestein
      2573  31.83            prime    replayed       22280      28955    1.29  6.1e-16  bluestein
      2574  2.3^2.11.13      flat     replayed        6124       5364    0.74  7.3e-16  
      2575  5^2.103          prime    replayed       21848      18686    0.57  7.5e-16  flips differ 1.51x bluestein
      2576  2^4.7.23         flat     replayed        4371      10808    2.40  7.3e-16  
      2577  3.859            prime    replayed       21883      24762    1.11  6.7e-16  bluestein
      2578  2.1289           prime    replayed       22398      22396    0.68  6.7e-16  flips differ 1.48x bluestein
      2579  2579             prime    replayed       22132      22139    1.00  7.9e-16  bluestein
      2580  2^2.3.5.43       flat     replayed        5664      16896    2.96  5.0e-16  
      2581  29.89            prime    replayed       22071      26949    1.21  5.8e-16  bluestein
      2582  2.1291           prime    replayed       22130      22319    0.98  6.6e-16  bluestein
      2583  3^2.7.41         chain3   replayed        5892      16792    2.81  6.5e-16  
      2584  2^3.17.19        chain3   replayed        4370      16014    3.44  6.1e-16  
      2585  5.11.47          chain3   replayed       13519      18411    1.28  5.3e-16  
      2586  2.3.431          prime    replayed       22376      18724    0.82  5.3e-16  bluestein
      2587  13.199           prime    replayed       22678      16762    0.73  5.7e-16  bluestein
      2588  2^2.647          prime    replayed       22289      20348    0.91  5.5e-16  bluestein
      2589  3.863            prime    replayed       22686      24767    1.09  7.7e-16  bluestein
      2590  2.5.7.37         flat     replayed        7089      15287    2.14  6.4e-16  
      2591  2591             prime    replayed       26777      22185    0.79  7.7e-16  bluestein
      2592  2^5.3^4          ztt      replayed        2609       3423    1.28  3.7e-16  
      2593  2593             prime    replayed        9923      23190    2.32  8.8e-16  rader
      2594  2.1297           prime    replayed       22782      28256    1.23  6.4e-16  bluestein
      2595  3.5.173          prime    replayed       22308      18081    0.81  5.5e-16  bluestein
      2596  2^2.11.59        prime    replayed       21780      17725    0.81  6.5e-16  bluestein
      2597  7^2.53           prime    replayed       22722      20425    0.90  4.9e-16  bluestein
      2598  2.3.433          prime    replayed       22234      25143    1.13  5.4e-16  bluestein
      2599  23.113           prime    replayed       22697      30589    1.32  6.5e-16  bluestein
      2600  2^3.5^2.13       flat     replayed        3892       3663    0.91  7.2e-16  
      2601  3^2.17^2         chain3   replayed        4728      15593    3.05  7.7e-16  
      2602  2.1301           prime    replayed       22592      28519    1.25  6.3e-16  bluestein
      2603  19.137           prime    replayed       21843      24112    1.08  4.6e-16  bluestein
      2604  2^2.3.7.31       chain3   replayed        4241      14507    3.40  5.6e-16  
      2605  5.521            prime    replayed       22189      20657    0.92  6.5e-16  bluestein
      2606  2.1303           prime    replayed       22816      28167    1.23  6.1e-16  bluestein
      2607  3.11.79          prime    replayed       22353      17973    0.78  5.7e-16  bluestein
      2608  2^4.163          prime    replayed       22367      19163    0.62  4.9e-16  flips differ 1.39x bluestein
      2609  2609             prime    replayed       22187      23153    1.04  6.2e-16  bluestein
      2610  2.3^2.5.29       flat     replayed        6201      13463    2.16  5.9e-16  
      2611  7.373            prime    replayed       21870      20443    0.92  5.8e-16  bluestein
      2612  2^2.653          prime    replayed       22218      22206    1.00  6.7e-16  bluestein
      2613  3.13.67          prime    replayed       22213      25603    1.13  3.9e-16  bluestein
      2614  2.1307           prime    replayed       22198      22951    1.01  6.5e-16  bluestein
      2615  5.523            prime    replayed       22757      20828    0.91  6.9e-16  bluestein
      2616  2^3.3.109        prime    replayed       22898      22884    0.98  5.6e-16  bluestein
      2617  2617             prime    replayed       22777      23193    1.01  6.6e-16  bluestein
      2618  2.7.11.17        flat     replayed        5737      10819    1.87  8.8e-16  
      2619  3^3.97           prime    replayed       22139      19770    0.86  6.7e-16  bluestein
      2620  2^2.5.131        prime    replayed       26415      18998    0.60  6.3e-16  bluestein
      2621  2621             prime    replayed       22206      23283    1.03  6.4e-16  bluestein
      2622  2.3.19.23        flat     replayed        6437      17619    2.67  1.6e-15  
      2623  43.61            prime    replayed       22192      37080    1.67  5.3e-16  bluestein
      2624  2^6.41           2p       replayed        7028      16421    1.57  4.8e-16  flips differ 1.49x
      2625  3.5^3.7          flat     replayed        4791       4357    0.89  5.3e-16  
      2626  2.13.101         prime    replayed       22189      20069    0.89  6.3e-16  bluestein
      2627  37.71            prime    replayed       22235      30453    1.33  5.7e-16  bluestein
      2628  2^2.3^2.73       prime    replayed       22695      19527    0.84  6.3e-16  bluestein
      2629  11.239           prime    replayed       22095      18457    0.82  5.1e-16  bluestein
      2630  2.5.263          prime    replayed       22327      20244    0.90  5.5e-16  bluestein
      2631  3.877            prime    replayed       21783      21496    0.97  5.2e-16  bluestein
      2632  2^3.7.47         flat     replayed        6129      18609    3.00  5.8e-16  
      2633  2633             prime    replayed       22333      23173    1.03  6.7e-16  bluestein
      2634  2.3.439          prime    replayed       22611      20529    0.71  6.5e-16  flips differ 1.28x bluestein
      2635  5.17.31          flat     replayed        7389      20414    2.68  5.7e-16  
      2636  2^2.659          prime    replayed       22203      22235    0.97  4.8e-16  bluestein
      2637  3^2.293          prime    replayed       22383      18515    0.82  6.0e-16  bluestein
      2638  2.1319           prime    replayed       21872      22865    1.02  6.1e-16  bluestein
      2639  7.13.29          chain3   replayed        5258      14029    2.61  8.4e-16  
      2640  2^4.3.5.11       chain3   replayed        3344       3591    0.97  5.2e-16  
      2641  19.139           prime    replayed       22158      23968    1.07  5.7e-16  bluestein
      2642  2.1321           prime    replayed       21899      28330    1.27  7.2e-16  bluestein
      2643  3.881            prime    replayed       22858      21208    0.91  5.9e-16  bluestein
      2644  2^2.661          prime    replayed       21840      22506    1.01  6.5e-16  bluestein
      2645  5.23^2           flat     replayed        6315      19479    3.07  1.7e-15  
      2646  2.3^3.7^2        chain3   replayed        4732       3993    0.84  5.3e-16  
      2647  2647             prime    replayed       15019      23185    1.50  1.1e-15  rader
      2648  2^3.331          prime    replayed       21814      22616    1.03  6.3e-16  bluestein
      2649  3.883            prime    replayed       21788      21347    0.97  5.0e-16  bluestein
      2650  2.5^2.53         prime    replayed       21812      20749    0.93  4.3e-16  bluestein
      2651  11.241           prime    replayed       22619      22164    0.98  5.9e-16  bluestein
      2652  2^2.3.13.17      chain3   replayed        4522      10673    2.14  7.1e-16  
      2653  7.379            prime    replayed       21882      17094    0.78  5.4e-16  bluestein
      2654  2.1327           prime    replayed       22669      28218    1.24  6.3e-16  bluestein
      2655  3^2.5.59         prime    replayed       22114      21463    0.92  5.9e-16  bluestein
      2656  2^5.83           prime    replayed       22359      25163    1.12  5.3e-16  bluestein
      2657  2657             prime    replayed       21854      23186    0.89  5.6e-16  bluestein
      2658  2.3.443          prime    replayed       22244      25249    1.10  7.6e-16  bluestein
      2659  2659             prime    replayed       22727      23364    0.74  7.2e-16  flips differ 1.39x bluestein
      2660  2^2.5.7.19       flat     replayed        4756      10007    2.09  5.0e-16  
      2661  3.887            prime    replayed       22215      21302    0.95  6.7e-16  bluestein
      2662  2.11^3           flat     replayed        6091       6416    0.99  7.0e-16  
      2663  2663             prime    replayed       22869      23194    1.01  6.1e-16  bluestein
      2664  2^3.3^2.37       flat     replayed        5353      15442    2.77  7.6e-16  
      2665  5.13.41          chain3   replayed        6169      17871    2.88  8.5e-16  
      2666  2.31.43          flat     replayed        7647      29006    3.75  3.7e-15  
      2667  3.7.127          prime    replayed       22626      15530    0.68  5.2e-16  bluestein
      2668  2^2.23.29        chain3   replayed        5660      20918    3.68  4.7e-16  
      2669  17.157           prime    replayed       22481      23078    1.02  4.8e-16  bluestein
      2670  2.3.5.89         prime    replayed       21782      18254    0.82  6.5e-16  bluestein
      2671  2671             prime    replayed       22203      23325    1.03  6.7e-16  bluestein
      2672  2^4.167          prime    replayed       22393      19208    0.60  6.0e-16  flips differ 1.41x bluestein
      2673  3^5.11           chain3   replayed        5172       4748    0.90  8.0e-16  
      2674  2.7.191          prime    replayed       21804      16141    0.72  5.8e-16  bluestein
      2675  5^2.107          prime    replayed       22627      18821    0.82  6.5e-16  bluestein
      2676  2^2.3.223        prime    replayed       22824      21769    0.95  5.3e-16  bluestein
      2677  2677             prime    replayed       22221      23261    1.02  5.9e-16  bluestein
      2678  2.13.103         prime    replayed       22202      20302    0.91  5.0e-16  bluestein
      2679  3.19.47          chain3   replayed       10117      25286    1.61  5.9e-16  flips differ 1.57x
      2680  2^3.5.67         prime    replayed       21858      25185    1.13  6.1e-16  bluestein
      2681  7.383            prime    replayed       21856      17000    0.77  5.2e-16  bluestein
      2682  2.3^2.149        prime    replayed       22218      18189    0.80  5.9e-16  bluestein
      2683  2683             prime    replayed       22412      23301    1.03  6.4e-16  bluestein
      2684  2^2.11.61        prime    replayed       22433      23444    1.04  4.3e-16  bluestein
      2685  3.5.179          prime    replayed       21875      18388    0.82  5.4e-16  bluestein
      2686  2.17.79          prime    replayed       21829      23775    1.08  6.0e-16  bluestein
      2687  2687             prime    replayed       22381      23277    1.04  6.9e-16  bluestein
      2688  2^7.3.7          ztt      replayed        2498       3143    1.25  3.7e-16  
      2689  2689             prime    replayed        9758      23326    2.37  6.1e-16  rader
      2690  2.5.269          prime    replayed       22387      20415    0.79  6.0e-16  bluestein
      2691  3^2.13.23        chain3   replayed        4991      12567    2.50  5.6e-16  
      2692  2^2.673          prime    replayed       22455      22348    0.99  7.4e-16  bluestein
      2693  2693             prime    replayed       21897      23265    1.06  6.2e-16  bluestein
      2694  2.3.449          prime    replayed       21894      20520    0.92  6.2e-16  bluestein
      2695  5.7^2.11         flat     replayed        4847       4776    0.82  4.9e-16  
      2696  2^3.337          prime    replayed       22198      22809    1.02  6.5e-16  bluestein
      2697  3.29.31          flat     replayed        7086      24377    3.43  2.7e-15  
      2698  2.19.71          prime    replayed       21863      26459    1.18  6.3e-16  bluestein
      2699  2699             prime    replayed       22248      23206    1.01  6.7e-16  bluestein
      2700  2^2.3^3.5^2      flat     replayed        4020       3597    0.87  5.6e-16  
      2701  37.73            prime    replayed       21801      31807    1.46  6.5e-16  bluestein
      2702  2.7.193          prime    replayed       21858      17370    0.76  5.1e-16  bluestein
      2703  3.17.53          prime    replayed       21864      27751    1.25  5.5e-16  bluestein
      2704  2^4.13^2         chain3   replayed        3594       5500    1.52  7.3e-16  
      2705  5.541            prime    replayed       22450      30493    1.07  5.7e-16  flips differ 1.27x bluestein
      2706  2.3.11.41        flat     replayed        8076      17946    2.21  9.2e-16  
      2707  2707             prime    replayed       21821      29516    1.32  5.4e-16  bluestein
      2708  2^2.677          prime    replayed       22399      20790    0.93  6.0e-16  bluestein
      2709  3^2.7.43         chain3   replayed       13316      18222    1.30  5.4e-16  
      2710  2.5.271          prime    replayed       21825      20519    0.94  5.9e-16  bluestein
      2711  2711             prime    replayed       22245      29573    1.30  7.0e-16  bluestein
      2712  2^3.3.113        prime    replayed       22443      23099    0.99  5.2e-16  bluestein
      2713  2713             prime    replayed       22836      29635    1.29  6.2e-16  bluestein
      2714  2.23.59          prime    replayed       21871      26239    1.19  6.2e-16  bluestein
      2715  3.5.181          prime    replayed       22787      22883    1.00  6.1e-16  bluestein
      2716  2^2.7.97         prime    replayed       22403      19878    0.89  6.0e-16  bluestein
      2717  11.13.19         flat     replayed        5205      12072    2.16  1.1e-15  
      2718  2.3^2.151        prime    replayed       22395      18157    0.81  4.9e-16  bluestein
      2719  2719             prime    replayed       22236      29608    1.30  7.0e-16  bluestein
      2720  2^5.5.17         chain3   replayed        3554       9313    2.58  5.3e-16  
      2721  3.907            prime    replayed       21865      26772    1.19  7.3e-16  bluestein
      2722  2.1361           prime    replayed       21864      22790    1.02  6.0e-16  bluestein
      2723  7.389            prime    replayed       22414      19405    0.85  6.3e-16  bluestein
      2724  2^2.3.227        prime    replayed       21403      19204    0.88  5.0e-16  bluestein
      2725  5^2.109          prime    replayed       21929      25109    0.92  5.8e-16  bluestein
      2726  2.29.47          flat     replayed        8588      28562    3.30  3.6e-15  
      2727  3^3.101          prime    replayed       22424      20207    0.89  5.2e-16  bluestein
      2728  2^3.11.31        flat     replayed        5942      15009    2.01  7.6e-16  flips differ 1.26x
      2729  2729             prime    replayed       21878      29635    1.35  6.8e-16  bluestein
      2730  2.3.5.7.13       flat     replayed        5503       4097    0.73  7.0e-16  
      2731  2731             prime    replayed       14236      29661    2.03  9.4e-16  rader
      2732  2^2.683          prime    replayed       22801      21241    0.92  6.9e-16  bluestein
      2733  3.911            prime    replayed       22169      26775    1.17  5.6e-16  bluestein
      2734  2.1367           prime    replayed       21897      22906    1.02  7.9e-16  bluestein
      2735  5.547            prime    replayed       22354      25745    1.14  6.8e-16  bluestein
      2736  2^4.3^2.19       chain3   replayed        3846      10297    2.42  5.1e-16  
      2737  7.17.23          flat     replayed        6420      18513    2.79  1.4e-15  
      2738  2.37^2           flat     replayed        7809      29538    3.66  3.3e-15  
      2739  3.11.83          prime    replayed       22267      20655    0.61  4.4e-16  flips differ 1.50x bluestein
      2740  2^2.5.137        prime    replayed       22495      18372    0.69  4.9e-16  bluestein
      2741  2741             prime    replayed       22226      29650    1.30  6.9e-16  bluestein
      2742  2.3.457          prime    replayed       22255      19677    0.86  6.4e-16  bluestein
      2743  13.211           prime    replayed       21838      20139    0.90  6.6e-16  bluestein
      2744  2^3.7^3          flat     replayed        4480       3823    0.83  6.2e-16  
      2745  3^2.5.61         prime    replayed       22188      24758    1.09  4.5e-16  bluestein
      2746  2.1373           prime    replayed       21830      22888    1.02  6.3e-16  bluestein
      2747  41.67            prime    replayed       22367      40294    1.79  4.3e-16  bluestein
      2748  2^2.3.229        prime    replayed       22248      19260    0.84  5.7e-16  bluestein
      2749  2749             prime    replayed       22249      29627    1.33  6.6e-16  bluestein
      2750  2.5^3.11         chain3   replayed        5294       4397    0.82  5.5e-16  
      2751  3.7.131          prime    replayed       22239      20089    0.88  5.0e-16  bluestein
      2752  2^6.43           2p       replayed        4946      18883    3.68  4.5e-16  
      2753  2753             prime    replayed       22280      29575    1.29  6.4e-16  bluestein
      2754  2.3^4.17         chain3   replayed        5873      10205    1.62  5.4e-16  
      2755  5.19.29          chain3   replayed        7279      21815    1.97  4.9e-16  flips differ 1.52x
      2756  2^2.13.53        prime    replayed       21871      21519    0.97  4.7e-16  bluestein
      2757  3.919            prime    replayed       22443      26867    1.19  5.4e-16  bluestein
      2758  2.7.197          prime    replayed       22296      17791    0.77  6.0e-16  bluestein
      2759  31.89            prime    replayed       22459      29911    1.10  6.6e-16  bluestein
      2760  2^3.3.5.23       chain3   replayed        4042      11014    2.72  5.4e-16  
      2761  11.251           prime    replayed       22781      17714    0.78  5.5e-16  bluestein
      2762  2.1381           prime    replayed       22186      22967    1.03  6.0e-16  bluestein
      2763  3^2.307          prime    replayed       22840      24280    1.05  6.8e-16  bluestein
      2764  2^2.691          prime    replayed       22724      21294    0.92  6.0e-16  bluestein
      2765  5.7.79           prime    replayed       21879      19077    0.86  5.7e-16  bluestein
      2766  2.3.461          prime    replayed       23111      19674    0.83  5.1e-16  bluestein
      2767  2767             prime    replayed       22465      29675    1.32  5.9e-16  bluestein
      2768  2^4.173          prime    replayed       21879      19254    0.67  4.8e-16  flips differ 1.30x bluestein
      2769  3.13.71          prime    replayed       22258      20020    0.73  5.6e-16  bluestein
      2770  2.5.277          prime    replayed       22828      20303    0.89  5.2e-16  bluestein
      2771  17.163           prime    replayed       22281      26650    1.16  7.5e-16  bluestein
      2772  2^2.3^2.7.11     chain3   replayed        3863       4090    1.05  4.9e-16  
      2773  47.59            prime    replayed       22244      39932    1.75  4.8e-16  bluestein
      2774  2.19.73          prime    replayed       22403      26752    1.19  5.3e-16  bluestein
      2775  3.5^2.37         chain3   replayed        8486      17729    1.49  7.1e-16  flips differ 1.41x
      2776  2^3.347          prime    replayed       22827      19396    0.85  5.1e-16  bluestein
      2777  2777             prime    replayed       21828      29568    1.31  5.8e-16  bluestein
      2778  2.3.463          prime    replayed       22376      19581    0.87  5.7e-16  bluestein
      2779  7.397            prime    replayed       21896      19387    0.86  5.1e-16  bluestein
      2780  2^2.5.139        prime    replayed       22541      18574    0.82  5.4e-16  bluestein
      2781  3^3.103          prime    replayed       21947      19998    0.91  5.8e-16  bluestein
      2782  2.13.107         prime    replayed       22666      20378    0.89  5.4e-16  bluestein
      2783  11^2.23          flat     replayed        6105      13389    2.19  1.4e-15  
      2784  2^5.3.29         flat     replayed        5589      13145    2.28  5.6e-16  
      2785  5.557            prime    replayed       22151      25808    1.16  7.0e-16  bluestein
      2786  2.7.199          prime    replayed       22210      17445    0.78  6.0e-16  bluestein
      2787  3.929            prime    replayed       22532      26876    1.19  5.6e-16  bluestein
      2788  2^2.17.41        chain3   replayed        6538      24352    3.69  6.6e-16  
      2789  2789             prime    replayed       22293      29662    1.30  6.4e-16  bluestein
      2790  2.3^2.5.31       chain3   replayed        4942      14478    2.91  4.9e-16  
      2791  2791             prime    replayed       22403      29617    1.32  6.6e-16  bluestein
      2792  2^3.349          prime    replayed       21937      19416    0.88  5.9e-16  bluestein
      2793  3.7^2.19         chain3   replayed        4926      10985    2.21  4.4e-16  
      2794  2.11.127         prime    replayed       22293      17001    0.76  5.3e-16  bluestein
      2795  5.13.43          chain3   replayed        6865      19405    2.54  7.0e-16  
      2796  2^2.3.233        prime    replayed       22603      19323    0.81  5.9e-16  bluestein
      2797  2797             prime    replayed       21889      29627    1.32  6.2e-16  bluestein
      2798  2.1399           prime    replayed       22160      22932    1.03  5.8e-16  bluestein
      2799  3^2.311          prime    replayed       22721      24174    0.77  6.7e-16  flips differ 1.38x bluestein
      2800  2^4.5^2.7        ztt      replayed        2734       3229    1.02  2.9e-16  
      2801  2801             prime    replayed       10390      29585    2.82  5.9e-16  rader
      2802  2.3.467          prime    replayed       22510      19730    0.87  4.3e-16  bluestein
      2803  2803             prime    replayed       22808      29723    1.30  5.3e-16  bluestein
      2804  2^2.701          prime    replayed       22239      21474    0.94  6.1e-16  bluestein
      2805  3.5.11.17        chain3   replayed        5100      11363    2.19  6.5e-16  
      2806  2.23.61          prime    replayed       22791      32672    1.43  5.1e-16  bluestein
      2807  7.401            prime    replayed       22386      29882    1.33  5.5e-16  bluestein
      2808  2^3.3^3.13       flat     replayed        4980       4129    0.80  1.2e-15  
      2809  53^2             prime    replayed       22241      40534    1.78  4.8e-16  bluestein
      2810  2.5.281          prime    replayed       21866      20442    0.93  7.0e-16  bluestein
      2811  3.937            prime    replayed       22514      26839    1.12  5.7e-16  bluestein
      2812  2^2.19.37        chain3   replayed        6439      24836    3.74  6.4e-16  
      2813  29.97            prime    replayed       21904      27428    1.24  5.9e-16  bluestein
      2814  2.3.7.67         prime    replayed       21953      27438    1.22  5.2e-16  bluestein
      2815  5.563            prime    replayed       21902      25928    1.15  5.6e-16  bluestein
      2816  2^8.11           chain3   replayed        4040       3808    0.93  4.2e-16  
      2817  3^2.313          prime    replayed       22276      24307    0.77  4.8e-16  flips differ 1.41x bluestein
      2818  2.1409           prime    replayed       22795      22908    1.00  6.4e-16  bluestein
      2819  2819             prime    replayed       23109      23556    1.02  4.4e-16  bluestein
      2820  2^2.3.5.47       chain3   replayed        5789      19780    3.35  6.0e-16  
      2821  7.13.31          chain3   replayed        6008      15828    2.57  5.9e-16  
      2822  2.17.83          prime    replayed       22684      27091    1.18  5.8e-16  bluestein
      2823  3.941            prime    replayed       21921      21543    0.96  6.9e-16  bluestein
      2824  2^3.353          prime    replayed       22432      19533    0.86  5.8e-16  bluestein
      2825  5^2.113          prime    replayed       21811      24150    1.08  6.5e-16  bluestein
      2826  2.3^2.157        prime    replayed       22761      18189    0.80  5.4e-16  bluestein
      2827  11.257           prime    replayed       22782      27319    1.19  5.9e-16  bluestein
      2828  2^2.7.101        prime    replayed       22746      20767    0.91  5.2e-16  bluestein
      2829  3.23.41          flat     replayed        7717      26816    3.35  8.5e-16  
      2830  2.5.283          prime    replayed       21840      20364    0.66  5.6e-16  flips differ 1.42x bluestein
      2831  19.149           prime    replayed       22577      27613    1.21  6.5e-16  bluestein
      2832  2^4.3.59         prime    replayed       22202      18441    0.82  6.0e-16  bluestein
      2833  2833             prime    replayed       22126      23816    1.06  4.7e-16  bluestein
      2834  2.13.109         prime    replayed       22193      26064    1.16  6.8e-16  bluestein
      2835  3^4.5.7          chain3   replayed        4912       4897    0.99  6.5e-16  
      2836  2^2.709          prime    replayed       22389      21342    0.94  5.8e-16  bluestein
      2837  2837             prime    replayed       22557      23567    0.97  5.3e-16  bluestein
      2838  2.3.11.43        chain3   replayed        6600      19630    2.93  4.7e-16  
      2839  17.167           prime    replayed       21920      34485    1.04  6.1e-16  flips differ 1.51x bluestein
      2840  2^3.5.71         prime    replayed       22223      19399    0.85  6.6e-16  bluestein
      2841  3.947            prime    replayed       22435      21476    0.94  5.3e-16  bluestein
      2842  2.7^2.29         flat     replayed        6934      14231    1.97  2.0e-15  
      2843  2843             prime    replayed       21854      23573    1.07  6.5e-16  bluestein
      2844  2^2.3^2.79       prime    replayed       22230      19708    0.87  6.7e-16  bluestein
      2845  5.569            prime    replayed       22191      21308    0.95  5.7e-16  bluestein
      2846  2.1423           prime    replayed       22248      23162    0.82  6.3e-16  flips differ 1.28x bluestein
      2847  3.13.73          prime    replayed       22453      21251    0.94  6.8e-16  bluestein
      2848  2^5.89           prime    replayed       22196      25028    1.13  5.7e-16  bluestein
      2849  7.11.37          flat     replayed        6196      18075    2.70  6.0e-16  
      2850  2.3.5^2.19       flat     replayed        6330      10672    1.65  6.7e-16  
      2851  2851             prime    replayed       15433      23562    1.49  7.2e-16  rader
      2852  2^2.23.31        chain3   replayed        6250      23513    3.72  5.9e-16  
      2853  3^2.317          prime    replayed       22770      17415    0.76  6.1e-16  bluestein
      2854  2.1427           prime    replayed       22768      23072    1.01  6.2e-16  bluestein
      2855  5.571            prime    replayed       21893      21656    0.77  5.5e-16  flips differ 1.30x bluestein
      2856  2^3.3.7.17       chain3   replayed        4641      10407    1.75  6.0e-16  flips differ 1.27x
      2857  2857             prime    replayed       15852      23599    1.48  1.0e-15  rader
      2858  2.1429           prime    replayed       22404      23022    1.02  5.1e-16  bluestein
      2859  3.953            prime    replayed       22194      21356    0.95  6.3e-16  bluestein
      2860  2^2.5.11.13      flat     replayed        4629       5973    1.27  6.9e-16  
      2861  2861             prime    replayed       21898      23674    1.08  6.4e-16  bluestein
      2862  2.3^3.53         prime    replayed       22433      23073    0.89  5.4e-16  bluestein
      2863  7.409            prime    replayed       22443      21732    0.97  6.8e-16  bluestein
      2864  2^4.179          prime    replayed       22763      19519    0.85  5.5e-16  bluestein
      2865  3.5.191          prime    replayed       22317      17445    0.76  5.0e-16  bluestein
      2866  2.1433           prime    replayed       22263      23027    1.01  6.2e-16  bluestein
      2867  47.61            prime    replayed       22601      41992    1.85  5.6e-16  bluestein
      2868  2^2.3.239        prime    replayed       22231      19379    0.87  6.1e-16  bluestein
      2869  19.151           prime    replayed       22484      26066    1.16  5.7e-16  bluestein
      2870  2.5.7.41         flat     replayed        8217      18383    2.23  4.7e-16  
      2871  3^2.11.29        flat     replayed        6164      15487    2.15  7.3e-16  
      2872  2^3.359          prime    replayed       22448      19568    0.87  5.3e-16  bluestein
      2873  13^2.17          chain3   replayed        5593      12503    2.21  9.1e-16  
      2874  2.3.479          prime    replayed       22421      19740    0.88  6.5e-16  bluestein
      2875  5^3.23           chain3   replayed        6310      13010    2.04  5.8e-16  
      2876  2^2.719          prime    replayed       22652      21040    0.93  5.5e-16  bluestein
      2877  3.7.137          prime    replayed       22210      19482    0.87  5.7e-16  bluestein
      2878  2.1439           prime    replayed       22360      23079    1.02  6.5e-16  bluestein
      2879  2879             prime    replayed       21850      23522    1.05  6.4e-16  bluestein
      2880  2^6.3^2.5        ztt      replayed        3137       3277    1.04  4.4e-16  
      2881  43.67            prime    replayed       22409      43857    1.95  5.2e-16  bluestein
      2882  2.11.131         prime    replayed       22433      21958    0.98  6.2e-16  bluestein
      2883  3.31^2           flat     replayed        7566      27385    3.54  1.9e-15  
      2884  2^2.7.103        prime    replayed       22275      20737    0.91  6.7e-16  bluestein
      2885  5.577            prime    replayed       22270      25483    1.14  5.7e-16  bluestein
      2886  2.3.13.37        chain3   replayed        5958      18068    2.97  5.3e-16  
      2887  2887             prime    replayed       22779      27835    1.22  6.9e-16  bluestein
      2888  2^3.19^2         flat     replayed        7101      18267    2.56  1.6e-15  
      2889  3^3.107          prime    replayed       21909      20289    0.90  6.7e-16  bluestein
      2890  2.5.17^2         chain3   replayed        5106      17176    3.18  6.0e-16  
      2891  7^2.59           prime    replayed       22809      23196    1.02  5.3e-16  bluestein
      2892  2^2.3.241        prime    replayed       21932      23513    1.05  6.7e-16  bluestein
      2893  11.263           prime    replayed       21882      27523    1.25  6.0e-16  bluestein
      2894  2.1447           prime    replayed       22929      27086    1.11  7.4e-16  bluestein
      2895  3.5.193          prime    replayed       22679      18486    0.81  5.6e-16  bluestein
      2896  2^4.181          prime    replayed       22492      24557    0.90  5.6e-16  bluestein
      2897  2897             prime    replayed       21895      27892    1.24  7.0e-16  bluestein
      2898  2.3^2.7.23       flat     replayed        6634      12299    1.81  7.0e-16  
      2899  13.223           prime    replayed       21890      24251    1.10  6.1e-16  bluestein
      2900  2^2.5^2.29       flat     replayed        5910      13796    2.21  4.2e-16  
      2901  3.967            prime    replayed       22597      25491    1.11  7.1e-16  bluestein
      2902  2.1451           prime    replayed       22433      27238    1.20  7.9e-16  bluestein
      2903  2903             prime    replayed       22271      27840    1.22  8.2e-16  bluestein
      2904  2^3.3.11^2       flat     replayed        5099       5748    1.11  7.8e-16  
      2905  5.7.83           prime    replayed       22443      21445    0.95  4.3e-16  bluestein
      2906  2.1453           prime    replayed       21904      26687    1.19  8.4e-16  bluestein
      2907  3^2.17.19        flat     replayed        6966      18012    2.36  1.5e-15  
      2908  2^2.727          prime    replayed       22229      27262    1.22  7.5e-16  bluestein
      2909  2909             prime    replayed       21934      27902    1.24  6.7e-16  bluestein
      2910  2.3.5.97         prime    replayed       22314      18072    0.80  5.9e-16  bluestein
      2911  41.71            prime    replayed       22264      35125    1.54  5.4e-16  bluestein
      2912  2^5.7.13         chain3   replayed        3528       4312    1.22  5.6e-16  
      2913  3.971            prime    replayed       22544      25555    1.12  6.5e-16  bluestein
      2914  2.31.47          flat     replayed        8817      31741    3.40  2.0e-15  
      2915  5.11.53          prime    replayed       21913      23460    1.05  5.8e-16  bluestein
      2916  2^2.3^6          chain3   replayed        4015       4320    0.93  5.9e-16  
      2917  2917             prime    replayed       22457      25613    1.14  5.2e-16  bluestein
      2918  2.1459           prime    replayed       22455      32092    1.18  5.9e-16  bluestein
      2919  3.7.139          prime    replayed       22356      19648    0.86  5.1e-16  bluestein
      2920  2^3.5.73         prime    replayed       21825      21754    0.82  6.3e-16  bluestein
      2921  23.127           prime    replayed       22213      26218    1.18  5.5e-16  bluestein
      2922  2.3.487          prime    replayed       21983      21529    0.96  6.6e-16  bluestein
      2923  37.79            prime    replayed       22329      32751    1.44  5.0e-16  bluestein
      2924  2^2.17.43        chain3   replayed        7010      26382    3.51  7.2e-16  
      2925  3^2.5^2.13       chain3   replayed        5551       5492    0.99  5.7e-16  
      2926  2.7.11.19        flat     replayed        7030      12339    1.73  1.3e-15  
      2927  2927             prime    replayed       21911      25552    1.14  5.8e-16  bluestein
      2928  2^4.3.61         prime    replayed       22846      25855    1.13  5.2e-16  bluestein
      2929  29.101           prime    replayed       21967      31734    1.01  7.1e-16  flips differ 1.44x bluestein
      2930  2.5.293          prime    replayed       22585      20364    0.90  6.1e-16  bluestein
      2931  3.977            prime    replayed       22765      23079    1.00  5.1e-16  bluestein
      2932  2^2.733          prime    replayed       22472      25073    1.11  6.0e-16  bluestein
      2933  7.419            prime    replayed       22813      21720    0.95  5.9e-16  bluestein
      2934  2.3^2.163        prime    replayed       21945      21209    0.95  6.2e-16  bluestein
      2935  5.587            prime    replayed       21958      25586    1.14  6.1e-16  bluestein
      2936  2^3.367          prime    replayed       21899      23534    1.05  6.4e-16  bluestein
      2937  3.11.89          prime    replayed       21890      20656    0.94  4.1e-16  bluestein
      2938  2.13.113         prime    replayed       21926      26208    1.19  7.0e-16  bluestein
      2939  2939             prime    replayed       22268      25628    1.15  4.9e-16  bluestein
      2940  2^2.3.5.7^2      flat     replayed        4615       4141    0.87  4.9e-16  
      2941  17.173           prime    replayed       22840      27157    1.18  6.3e-16  bluestein
      2942  2.1471           prime    replayed       22267      25220    1.13  5.3e-16  bluestein
      2943  3^3.109          prime    replayed       22275      27362    1.20  7.4e-16  bluestein
      2944  2^7.23           chain3   replayed        4381      11864    2.67  4.4e-16  
      2945  5.19.31          chain3   replayed        6764      23471    3.08  4.5e-16  
      2946  2.3.491          prime    replayed       21929      21633    0.98  6.0e-16  bluestein
      2947  7.421            prime    replayed       22282      22313    0.98  6.0e-16  bluestein
      2948  2^2.11.67        prime    replayed       21942      27144    0.82  6.4e-16  flips differ 1.51x bluestein
      2949  3.983            prime    replayed       22452      23087    1.02  6.4e-16  bluestein
      2950  2.5^2.59         prime    replayed       21975      19517    0.87  4.8e-16  bluestein
      2951  13.227           prime    replayed       22371      21813    0.95  6.0e-16  bluestein
      2952  2^3.3^2.41       chain3   replayed        5521      19445    3.50  5.5e-16  
      2953  2953             prime    replayed       22234      25674    1.15  6.3e-16  bluestein
      2954  2.7.211          prime    replayed       22164      20986    0.95  5.9e-16  bluestein
      2955  3.5.197          prime    replayed       22273      18802    0.82  5.6e-16  bluestein
      2956  2^2.739          prime    replayed       22386      25060    0.87  6.2e-16  flips differ 1.28x bluestein
      2957  2957             prime    replayed       21841      25481    1.14  5.8e-16  bluestein
      2958  2.3.17.29        flat     replayed        8669      21529    2.46  1.2e-15  
      2959  11.269           prime    replayed       21906      27553    1.22  6.7e-16  bluestein
      2960  2^4.5.37         flat     replayed        5518      16983    3.05  4.7e-16  
      2961  3^2.7.47         flat     replayed        6900      21433    3.01  5.8e-16  
      2962  2.1481           prime    replayed       22375      32474    1.43  5.2e-16  bluestein
      2963  2963             prime    replayed       22219      25674    1.16  5.4e-16  bluestein
      2964  2^2.3.13.19      chain3   replayed        4507      12335    2.69  6.6e-16  
      2965  5.593            prime    replayed       22784      25725    1.12  6.9e-16  bluestein
      2966  2.1483           prime    replayed       22235      32275    1.44  5.2e-16  bluestein
      2967  3.23.43          chain3   replayed       11171      29816    2.67  5.1e-16  
      2968  2^3.7.53         prime    replayed       22420      22083    0.98  4.6e-16  bluestein
      2969  2969             prime    replayed       22790      25539    1.12  6.3e-16  bluestein
      2970  2.3^3.5.11       chain3   replayed        4784       4581    0.88  6.8e-16  
      2971  2971             prime    replayed       22267      25644    0.88  5.7e-16  flips differ 1.30x bluestein
      2972  2^2.743          prime    replayed       22273      25959    0.93  7.2e-16  bluestein
      2973  3.991            prime    replayed       22819      23016    1.00  6.3e-16  bluestein
      2974  2.1487           prime    replayed       22452      32394    1.17  5.4e-16  flips differ 1.25x bluestein
      2975  5^2.7.17         flat     replayed        5913      11385    1.92  5.5e-16  
      2976  2^5.3.31         chain3   replayed        4732      15287    3.14  6.2e-16  
      2977  13.229           prime    replayed       22221      21848    0.78  6.3e-16  flips differ 1.26x bluestein
      2978  2.1489           prime    replayed       22783      31953    1.40  7.0e-16  bluestein
      2979  3^2.331          prime    replayed       22411      26010    1.14  7.6e-16  bluestein
      2980  2^2.5.149        prime    replayed       21882      20460    0.93  5.8e-16  bluestein
      2981  11.271           prime    replayed       21876      22703    1.00  5.9e-16  bluestein
      2982  2.3.7.71         prime    replayed       22727      20481    0.89  5.1e-16  bluestein
      2983  19.157           prime    replayed       22491      26410    1.17  6.3e-16  bluestein
      2984  2^3.373          prime    replayed       22310      23495    1.03  5.9e-16  bluestein
      2985  3.5.199          prime    replayed       22802      18617    0.82  4.9e-16  bluestein
      2986  2.1493           prime    replayed       22466      25042    1.11  7.1e-16  bluestein
      2987  29.103           prime    replayed       21917      31988    1.42  6.0e-16  bluestein
      2988  2^2.3^2.83       prime    replayed       22329      27765    1.24  5.8e-16  bluestein
      2989  7^2.61           prime    replayed       21940      26525    1.21  4.7e-16  bluestein
      2990  2.5.13.23        flat     replayed        7622      13760    1.80  7.2e-16  
      2991  3.997            prime    replayed       22499      23175    0.82  6.1e-16  bluestein
      2992  2^4.11.17        chain3   replayed        4198      12108    2.51  5.6e-16  
      2993  41.73            prime    replayed       22492      37154    1.10  6.8e-16  flips differ 1.48x bluestein
      2994  2.3.499          prime    replayed       22430      25698    1.14  5.5e-16  bluestein
      2995  5.599            prime    replayed       22340      26103    1.14  6.3e-16  bluestein
      2996  2^2.7.107        prime    replayed       22643      21033    0.90  5.7e-16  bluestein
      2997  3^4.37           chain3   replayed        7034      18290    2.38  5.6e-16  
      2998  2.1499           prime    replayed       21885      25080    1.11  7.1e-16  bluestein
      2999  2999             prime    replayed       22308      25534    1.12  6.5e-16  bluestein
      3000  2^3.3.5^3        chain3   replayed        4492       3860    0.81  4.8e-16  
      3001  3001             prime    replayed       19048      32831    1.71  8.9e-16  rader
      3002  2.19.79          prime    replayed       21946      27102    1.20  5.6e-16  bluestein
      3003  3.7.11.13        chain3   replayed        5244       6728    1.26  5.0e-16  
      3004  2^2.751          prime    replayed       22819      26081    0.92  5.7e-16  flips differ 1.25x bluestein
      3005  5.601            prime    replayed       21917      33320    1.48  5.7e-16  bluestein
      3006  2.3^2.167        prime    replayed       22233      21367    0.94  6.0e-16  bluestein
      3007  31.97            prime    replayed       22263      30518    1.37  6.8e-16  bluestein
      3008  2^6.47           2p       replayed        9423      21245    2.25  4.2e-16  
      3009  3.17.59          prime    replayed       22256      27391    1.20  6.2e-16  bluestein
      3010  2.5.7.43         chain3   replayed        9977      20989    2.08  4.6e-16  
      3011  3011             prime    replayed       22258      32833    1.47  9.0e-16  bluestein
      3012  2^2.3.251        prime    replayed       22263      17614    0.79  5.9e-16  bluestein
      3013  23.131           prime    replayed       22836      30695    1.15  6.2e-16  bluestein
      3014  2.11.137         prime    replayed       21897      21625    0.98  4.9e-16  bluestein
      3015  3^2.5.67         prime    replayed       22227      29206    1.29  5.0e-16  bluestein
      3016  2^3.13.29        flat     replayed        5909      15684    2.45  6.4e-16  
      3017  7.431            prime    replayed       22252      21816    0.95  6.5e-16  bluestein
      3018  2.3.503          prime    replayed       22829      18712    0.80  5.1e-16  bluestein
      3019  3019             prime    replayed       22967      33089    1.34  6.7e-16  bluestein
      3020  2^2.5.151        prime    replayed       21913      18930    0.86  6.2e-16  bluestein
      3021  3.19.53          prime    replayed       22452      31538    1.40  4.3e-16  bluestein
      3022  2.1511           prime    replayed       22242      23920    1.07  5.6e-16  bluestein
      3023  3023             prime    replayed       22787      32762    1.43  5.5e-16  bluestein
      3024  2^4.3^3.7        ztt      replayed        3239       4164    1.27  4.1e-16  
      3025  5^2.11^2         flat     replayed        5875       6610    1.12  8.7e-16  
      3026  2.17.89          prime    replayed       22262      27848    1.22  7.4e-16  bluestein
      3027  3.1009           prime    replayed       21970      30291    1.35  7.2e-16  bluestein
      3028  2^2.757          prime    replayed       23016      22163    0.96  5.6e-16  bluestein
      3029  13.233           prime    replayed       22296      21810    0.95  5.3e-16  bluestein
      3030  2.3.5.101        prime    replayed       22840      22049    0.83  7.1e-16  bluestein
      3031  7.433            prime    replayed       21922      29622    1.35  6.8e-16  bluestein
      3032  2^3.379          prime    replayed       22823      19720    0.86  6.3e-16  bluestein
      3033  3^2.337          prime    replayed       22450      25699    1.14  5.6e-16  bluestein
      3034  2.37.41          flat     replayed        9110      34151    3.53  2.9e-15  
      3035  5.607            prime    replayed       21919      28253    1.28  7.6e-16  bluestein
      3036  2^2.3.11.23      flat     replayed        6167      13874    2.15  5.6e-16  
      3037  3037             prime    replayed       16988      32694    1.82  1.1e-15  rader
      3038  2.7^2.31         flat     replayed        7881      17517    2.21  5.5e-16  
      3039  3.1013           prime    replayed       22476      30190    1.34  6.7e-16  bluestein
      3040  2^5.5.19         chain3   replayed        4111      12008    2.91  4.9e-16  
      3041  3041             prime    replayed       15144      24870    1.62  8.3e-16  rader
      3042  2.3^2.13^2       flat     replayed        6753       6337    0.91  1.1e-15  
      3043  17.179           prime    replayed       21905      34922    1.59  6.1e-16  bluestein
      3044  2^2.761          prime    replayed       22504      26403    1.17  5.5e-16  bluestein
      3045  3.5.7.29         flat     replayed        7367      15365    1.72  6.7e-16  
      3046  2.1523           prime    replayed       22278      24009    1.06  5.4e-16  bluestein
      3047  11.277           prime    replayed       22309      25789    1.15  5.6e-16  bluestein
      3048  2^3.3.127        prime    replayed       21960      18605    0.83  5.5e-16  bluestein
      3049  3049             prime    replayed       22459      24781    1.10  5.3e-16  bluestein
      3050  2.5^2.61         prime    replayed       22815      27267    1.03  4.8e-16  bluestein
      3051  3^3.113          prime    replayed       22251      27626    1.24  6.8e-16  bluestein
      3052  2^2.7.109        prime    replayed       22363      26690    1.17  6.1e-16  bluestein
      3053  43.71            prime    replayed       22758      37536    1.64  5.8e-16  bluestein
      3054  2.3.509          prime    replayed       22717      18582    0.81  5.0e-16  bluestein
      3055  5.13.47          chain3   replayed       16777      21702    1.27  5.6e-16  
      3056  2^4.191          prime    replayed       22304      18359    0.81  5.4e-16  bluestein
      3057  3.1019           prime    replayed       21955      19622    0.88  4.9e-16  bluestein
      3058  2.11.139         prime    replayed       21963      21540    0.98  4.9e-16  bluestein
      3059  7.19.23          flat     replayed        6764      21426    3.05  8.2e-16  
      3060  2^2.3^2.5.17     chain3   replayed        4378      11042    2.47  6.6e-16  
      3061  3061             prime    replayed       15307      24888    1.55  9.6e-16  rader
      3062  2.1531           prime    replayed       22891      24078    0.77  6.0e-16  flips differ 1.37x bluestein
      3063  3.1021           prime    replayed       22288      19696    0.88  6.0e-16  bluestein
      3064  2^3.383          prime    replayed       22767      19566    0.76  5.3e-16  bluestein
      3065  5.613            prime    replayed       22537      26229    1.16  5.9e-16  bluestein
      3066  2.3.7.73         prime    replayed       22641      23057    1.01  5.5e-16  bluestein
      3067  3067             prime    replayed       21942      24892    0.88  5.4e-16  flips differ 1.29x bluestein
      3068  2^2.13.59        prime    replayed       22857      21112    0.92  5.4e-16  bluestein
      3069  3^2.11.31        chain3   replayed       10828      17513    1.52  6.1e-16  
      3070  2.5.307          prime    replayed       22536      26659    1.18  6.0e-16  bluestein
      3071  37.83            prime    replayed       21492      36537    1.68  6.1e-16  bluestein
      3072  2^10.3           ztt      replayed        2723       3489    1.26  4.0e-16  
      3073  7.439            prime    replayed       21988      29627    1.32  5.7e-16  bluestein
      3074  2.29.53          prime    replayed       22363      35455    1.24  3.9e-16  flips differ 1.28x bluestein
      3075  3.5^2.41         chain3   replayed        9808      21328    2.13  6.2e-16  
      3076  2^2.769          prime    replayed       22008      23944    1.03  5.4e-16  bluestein
      3077  17.181           prime    replayed       22464      33828    1.21  5.6e-16  bluestein
      3078  2.3^4.19         chain3   replayed        5963      12900    2.09  5.7e-16  
      3079  3079             prime    replayed       21975      29232    1.33  5.9e-16  bluestein
      3080  2^3.5.7.11       flat     replayed        4816       4444    0.92  5.9e-16  
      3081  3.13.79          prime    replayed       22358      21183    0.92  5.8e-16  bluestein
      3082  2.23.67          prime    replayed       22329      39710    1.44  5.0e-16  bluestein
      3083  3083             prime    replayed       21971      29337    1.31  6.3e-16  bluestein
      3084  2^2.3.257        prime    replayed       22261      29440    1.31  5.3e-16  bluestein
      3085  5.617            prime    replayed       22450      26098    1.16  6.0e-16  bluestein
      3086  2.1543           prime    replayed       21893      32655    1.48  5.1e-16  bluestein
      3087  3^2.7^3          chain3   replayed        5637       5606    0.95  4.7e-16  
      3088  2^4.193          prime    replayed       21876      19984    0.90  6.5e-16  bluestein
      3089  3089             prime    replayed       22480      29249    1.30  6.1e-16  bluestein
      3090  2.3.5.103        prime    replayed       22430      22053    0.98  6.4e-16  bluestein
      3091  11.281           prime    replayed       22458      26072    1.16  6.0e-16  bluestein
      3092  2^2.773          prime    replayed       21899      23085    1.05  5.5e-16  bluestein
      3093  3.1031           prime    replayed       21908      26458    1.17  6.5e-16  bluestein
      3094  2.7.13.17        flat     replayed        7103      12823    1.51  1.4e-15  
      3095  5.619            prime    replayed       22521      26283    1.17  6.8e-16  bluestein
      3096  2^3.3^2.43       chain3   replayed        6131      20206    3.29  7.0e-16  
      3097  19.163           prime    replayed       22471      30398    1.34  5.2e-16  bluestein
      3098  2.1549           prime    replayed       27228      32740    1.20  5.6e-16  bluestein
      3099  3.1033           prime    replayed       21907      26410    1.18  6.0e-16  bluestein
      3100  2^2.5^2.31       flat     replayed        6689      17143    2.55  5.3e-16  
      3101  7.443            prime    replayed       22857      29556    1.28  5.7e-16  bluestein
      3102  2.3.11.47        flat     replayed        9552      21902    2.25  7.8e-16  
      3103  29.107           prime    replayed       22330      32719    1.43  5.5e-16  bluestein
      3104  2^5.97           prime    replayed       22781      19398    0.85  5.3e-16  bluestein
      3105  3^3.5.23         chain3   replayed        8813      13699    1.33  4.4e-16  
      3106  2.1553           prime    replayed       21916      32395    1.44  6.8e-16  bluestein
      3107  13.239           prime    replayed       22290      22085    0.99  6.5e-16  bluestein
      3108  2^2.3.7.37       chain3   replayed        5528      18581    3.36  6.3e-16  
      3109  3109             prime    replayed       21928      29314    1.30  5.9e-16  bluestein
      3110  2.5.311          prime    replayed       22301      26808    1.17  6.0e-16  bluestein
      3111  3.17.61          prime    replayed       22279      35134    1.54  4.5e-16  bluestein
      3112  2^3.389          prime    replayed       22245      22298    0.80  6.2e-16  flips differ 1.25x bluestein
      3113  11.283           prime    replayed       22879      22957    0.99  6.6e-16  bluestein
      3114  2.3^2.173        prime    replayed       22507      21646    0.95  6.1e-16  bluestein
      3115  5.7.89           prime    replayed       21946      21547    0.98  7.2e-16  bluestein
      3116  2^2.19.41        chain3   replayed        7437      27743    3.57  4.3e-16  
      3117  3.1039           prime    replayed       22261      26396    1.18  5.6e-16  bluestein
      3118  2.1559           prime    replayed       22486      32611    1.44  5.6e-16  bluestein
      3119  3119             prime    replayed       22943      29333    1.27  6.7e-16  bluestein
      3120  2^4.3.5.13       chain3   replayed        4061       4440    1.07  7.5e-16  
      3121  3121             prime    replayed       15580      29225    1.87  9.0e-16  rader
      3122  2.7.223          prime    replayed       21955      25234    1.15  6.0e-16  bluestein
      3123  3^2.347          prime    replayed       22742      22179    0.97  6.9e-16  bluestein
      3124  2^2.11.71        prime    replayed       22352      22311    0.98  6.7e-16  bluestein
      3125  5^5              flat     replayed        5760       5665    0.97  6.1e-16  
      3126  2.3.521          prime    replayed       22013      24949    1.11  5.4e-16  bluestein
      3127  53.59            prime    replayed       21919      45490    2.07  5.0e-16  bluestein
      3128  2^3.17.23        chain3   replayed        7871      20753    2.01  6.4e-16  flips differ 1.31x
      3129  3.7.149          prime    replayed       21950      23192    1.05  7.4e-16  bluestein
      3130  2.5.313          prime    replayed       22333      26594    1.18  5.8e-16  bluestein
      3131  31.101           prime    replayed       21985      35277    1.60  5.5e-16  bluestein
      3132  2^2.3^3.29       chain3   replayed        5128      15408    3.00  5.4e-16  
      3133  13.241           prime    replayed       21934      34354    1.52  6.3e-16  bluestein
      3134  2.1567           prime    replayed       21916      24039    1.09  6.0e-16  bluestein
      3135  3.5.11.19        chain3   replayed        6669      13060    1.91  6.1e-16  
      3136  2^6.7^2          ztt      replayed        3073       3925    0.58  3.3e-16  flips differ 2.18x
      3137  3137             prime    replayed       12157      25464    2.09  1.0e-15  rader
      3138  2.3.523          prime    replayed       21965      24717    1.12  6.8e-16  bluestein
      3139  43.73            prime    replayed       22371      40439    1.77  6.8e-16  bluestein
      3140  2^2.5.157        prime    replayed       21960      20134    0.91  5.4e-16  bluestein
      3141  3^2.349          prime    replayed       22464      22342    0.99  6.6e-16  bluestein
      3142  2.1571           prime    replayed       22517      23942    1.06  5.8e-16  bluestein
      3143  7.449            prime    replayed       22404      29722    1.29  5.2e-16  bluestein
      3144  2^3.3.131        prime    replayed       22554      23042    1.02  6.7e-16  bluestein
      3145  5.17.37          chain3   replayed        7414      27604    3.57  6.0e-16  
      3146  2.11^2.13        flat     replayed        7091       7925    1.10  7.9e-16  
      3147  3.1049           prime    replayed       22514      26350    1.17  5.9e-16  bluestein
      3148  2^2.787          prime    replayed       21949      23350    1.03  5.6e-16  bluestein
      3149  47.67            prime    replayed       22488      49422    2.20  8.3e-16  bluestein
      3150  2.3^2.5^2.7      chain3   replayed        4861       4542    0.71  4.7e-16  flips differ 1.31x
      3151  23.137           prime    replayed       21975      30657    1.39  5.9e-16  bluestein
      3152  2^4.197          prime    replayed       22351      20033    0.87  4.6e-16  bluestein
      3153  3.1051           prime    replayed       21952      26474    1.17  7.2e-16  bluestein
      3154  2.19.83          prime    replayed       22326      31062    1.37  5.7e-16  bluestein
      3155  5.631            prime    replayed       22882      21193    0.66  6.5e-16  flips differ 1.40x bluestein
      3156  2^2.3.263        prime    replayed       21992      24047    1.06  5.7e-16  bluestein
      3157  7.11.41          chain3   replayed        7657      21062    2.73  5.4e-16  
      3158  2.1579           prime    replayed       22015      24136    1.07  6.0e-16  bluestein
      3159  3^5.13           chain3   replayed        5768       5891    0.93  6.4e-16  
      3160  2^3.5.79         prime    replayed       22032      21445    0.95  5.3e-16  bluestein
      3161  29.109           prime    replayed       21964      39254    1.24  6.6e-16  flips differ 1.44x bluestein
      3162  2.3.17.31        flat     replayed        8158      24245    2.86  1.3e-15  
      3163  3163             prime    replayed       22336      25290    1.11  7.0e-16  bluestein
      3164  2^2.7.113        prime    replayed       22035      27478    1.22  6.8e-16  bluestein
      3165  3.5.211          prime    replayed       22737      22411    0.98  5.8e-16  bluestein
      3166  2.1583           prime    replayed       22523      24066    1.06  5.8e-16  bluestein
      3167  3167             prime    replayed       22333      25381    1.13  5.9e-16  bluestein
      3168  2^5.3^2.11       chain3   replayed        3833       4492    1.16  5.8e-16  
      3169  3169             prime    replayed       14225      25291    1.75  8.8e-16  rader
      3170  2.5.317          prime    replayed       22481      19163    0.85  4.6e-16  bluestein
      3171  3.7.151          prime    replayed       21880      21124    0.96  4.9e-16  bluestein
      3172  2^2.13.61        prime    replayed       22514      27879    1.21  5.3e-16  bluestein
      3173  19.167           prime    replayed       22362      38776    1.69  4.9e-16  bluestein
      3174  2.3.23^2         chain3   replayed        8489      22879    2.20  6.3e-16  
      3175  5^2.127          prime    replayed       22002      19574    0.89  5.3e-16  bluestein
      3176  2^3.397          prime    replayed       22320      22328    1.00  6.1e-16  bluestein
      3177  3^2.353          prime    replayed       22994      22149    0.96  5.6e-16  bluestein
      3178  2.7.227          prime    replayed       22357      22676    0.82  5.5e-16  bluestein
      3179  11.17^2          flat     replayed        6185      19888    3.01  1.2e-15  
      3180  2^2.3.5.53       prime    replayed       22356      25361    1.13  5.6e-16  bluestein
      3181  3181             prime    replayed       22740      25436    1.11  6.2e-16  bluestein
      3182  2.37.43          flat     replayed       10228      36695    3.49  2.3e-15  
      3183  3.1061           prime    replayed       22765      26449    1.16  6.3e-16  bluestein
      3184  2^4.199          prime    replayed       22804      20531    0.88  5.1e-16  bluestein
      3185  5.7^2.13         flat     replayed        5634       6050    0.80  5.5e-16  flips differ 1.34x
      3186  2.3^3.59         prime    replayed       22995      25467    1.04  4.5e-16  bluestein
      3187  3187             prime    replayed       22309      25284    1.13  4.9e-16  bluestein
      3188  2^2.797          prime    replayed       22388      23401    1.04  7.1e-16  bluestein
      3189  3.1063           prime    replayed       22373      26505    1.16  6.6e-16  bluestein
      3190  2.5.11.29        chain3   replayed        7588      16904    1.77  4.5e-16  flips differ 1.26x
      3191  3191             prime    replayed       22323      25361    1.13  5.8e-16  bluestein
      3192  2^3.3.7.19       chain3   replayed        5318      12138    2.18  5.2e-16  
      3193  31.103           prime    replayed       23081      35533    1.54  5.4e-16  bluestein
      3194  2.1597           prime    replayed       22312      24136    0.89  5.8e-16  bluestein
      3195  3^2.5.71         prime    replayed       22340      22815    1.01  6.5e-16  bluestein
      3196  2^2.17.47        flat     replayed        8652      29073    3.33  7.5e-16  
      3197  23.139           prime    replayed       22561      30631    0.98  5.4e-16  flips differ 1.38x bluestein
      3198  2.3.13.41        chain3   replayed        7088      21498    2.99  6.8e-16  
      3199  7.457            prime    replayed       22659      23278    0.87  6.1e-16  bluestein
      3200  2^7.5^2          ztt      replayed        3031       3681    1.21  3.4e-16  
      3201  3.11.97          prime    replayed       22728      20563    0.80  6.3e-16  bluestein
      3202  2.1601           prime    replayed       22363      28082    1.22  8.4e-16  bluestein
      3203  3203             prime    replayed       22801      28908    1.26  6.4e-16  bluestein
      3204  2^2.3^2.89       prime    replayed       22495      28030    0.88  6.9e-16  flips differ 1.41x bluestein
      3205  5.641            prime    replayed       22299      34374    1.53  6.1e-16  bluestein
      3206  2.7.229          prime    replayed       21982      22685    1.03  5.4e-16  bluestein
      3207  3.1069           prime    replayed       22809      26444    1.15  8.9e-16  bluestein
      3208  2^3.401          prime    replayed       22584      34529    1.53  7.1e-16  bluestein
      3209  3209             prime    replayed       22294      28798    1.25  5.5e-16  bluestein
      3210  2.3.5.107        prime    replayed       22930      22494    0.97  7.0e-16  bluestein
      3211  13^2.19          flat     replayed        6294      14570    2.12  1.1e-15  
      3212  2^2.11.73        prime    replayed       21998      23937    1.07  7.3e-16  bluestein
      3213  3^3.7.17         chain3   replayed        5753      12291    2.11  7.3e-16  
      3214  2.1607           prime    replayed       22466      28536    0.85  5.7e-16  flips differ 1.48x bluestein
      3215  5.643            prime    replayed       22317      25604    1.13  7.2e-16  bluestein
      3216  2^4.3.67         prime    replayed       22570      28523    1.26  6.1e-16  bluestein
      3217  3217             prime    replayed       22306      29099    1.27  6.5e-16  bluestein
      3218  2.1609           prime    replayed       22489      28321    1.25  6.0e-16  bluestein
      3219  3.29.37          flat     replayed        9018      31354    3.47  2.0e-15  
      3220  2^2.5.7.23       flat     replayed        5717      13641    2.37  5.1e-16  
      3221  3221             prime    replayed       21955      29069    1.29  6.2e-16  bluestein
      3222  2.3^2.179        prime    replayed       21996      21758    0.97  5.9e-16  bluestein
      3223  11.293           prime    replayed       21985      23532    1.04  6.9e-16  bluestein
      3224  2^3.13.31        chain3   replayed        6083      18070    2.96  7.5e-16  
      3225  3.5^2.43         chain3   replayed       11184      23128    1.54  5.0e-16  flips differ 1.34x
      3226  2.1613           prime    replayed       22782      28072    1.21  7.1e-16  bluestein
      3227  7.461            prime    replayed       22523      23198    1.03  5.9e-16  bluestein
      3228  2^2.3.269        prime    replayed       21672      24670    1.12  5.6e-16  bluestein
      3229  3229             prime    replayed       22346      29074    1.27  6.1e-16  bluestein
      3230  2.5.17.19        chain3   replayed        6193      19911    2.52  6.7e-16  flips differ 1.27x
      3231  3^2.359          prime    replayed       21972      22357    1.01  5.9e-16  bluestein
      3232  2^5.101          prime    replayed       21926      23974    1.09  6.4e-16  bluestein
      3233  53.61            prime    replayed       22496      49700    2.21  4.9e-16  bluestein
      3234  2.3.7^2.11       flat     replayed        6617       4987    0.74  5.0e-16  
      3235  5.647            prime    replayed       21998      25781    1.15  8.5e-16  bluestein
      3236  2^2.809          prime    replayed       22319      27005    1.21  6.5e-16  bluestein
      3237  3.13.83          prime    replayed       21973      24673    1.12  6.1e-16  bluestein
      3238  2.1619           prime    replayed       22894      28305    0.84  6.3e-16  flips differ 1.47x bluestein
      3239  41.79            prime    replayed       21986      37585    1.67  5.3e-16  bluestein
      3240  2^3.3^4.5        chain3   replayed        5277       4421    0.82  5.9e-16  
      3241  7.463            prime    replayed       22507      23258    1.03  6.6e-16  bluestein
      3242  2.1621           prime    replayed       22345      37082    1.34  7.2e-16  bluestein
      3243  3.23.47          chain3   replayed       12620      33197    1.80  6.2e-16  flips differ 1.46x
      3244  2^2.811          prime    replayed       21997      32826    1.49  6.4e-16  bluestein
      3245  5.11.59          prime    replayed       22009      22289    0.98  5.4e-16  bluestein
      3246  2.3.541          prime    replayed       21974      36544    1.66  6.2e-16  bluestein
      3247  17.191           prime    replayed       22360      26887    1.19  5.1e-16  bluestein
      3248  2^4.7.29         chain3   replayed        5220      16001    3.06  5.8e-16  
      3249  3^2.19^2         flat     replayed        8726      20931    2.38  1.2e-15  
      3250  2.5^3.13         chain3   replayed        5942       5236    0.82  7.1e-16  
      3251  3251             prime    replayed       22936      37339    1.62  7.6e-16  bluestein
      3252  2^2.3.271        prime    replayed       22038      24483    0.90  6.5e-16  bluestein
      3253  3253             prime    replayed       22605      37582    1.66  7.6e-16  bluestein
      3254  2.1627           prime    replayed       22915      34114    1.49  6.6e-16  bluestein
      3255  3.5.7.31         flat     replayed        7882      18708    2.35  7.1e-16  
      3256  2^3.11.37        flat     replayed        6876      20566    2.97  6.0e-16  
      3257  3257             prime    replayed       21502      37319    1.73  7.2e-16  bluestein
      3258  2.3^2.181        prime    replayed       21979      28452    1.29  5.9e-16  bluestein
      3259  3259             prime    replayed       22000      37265    1.69  8.0e-16  bluestein
      3260  2^2.5.163        prime    replayed       22346      23575    1.01  6.2e-16  bluestein
      3261  3.1087           prime    replayed       21996      35284    1.60  7.7e-16  bluestein
      3262  2.7.233          prime    replayed       21548      22781    1.03  6.5e-16  bluestein
      3263  13.251           prime    replayed       21988      21118    0.77  6.3e-16  flips differ 1.25x bluestein
      3264  2^6.3.17         chain3   replayed        4321      11382    2.54  5.6e-16  
      3265  5.653            prime    replayed       22036      28063    1.25  6.1e-16  bluestein
      3266  2.23.71          prime    replayed       22591      32709    1.44  5.9e-16  bluestein
      3267  3^3.11^2         flat     replayed        7089       6950    0.87  5.3e-16  
      3268  2^2.19.43        flat     replayed        9001      29899    3.32  1.6e-15  
      3269  7.467            prime    replayed       22384      23211    1.01  5.2e-16  bluestein
      3270  2.3.5.109        prime    replayed       22999      28868    1.25  5.5e-16  bluestein
      3271  3271             prime    replayed       22951      37383    1.62  6.9e-16  bluestein
      3272  2^3.409          prime    replayed       22017      24797    1.10  6.5e-16  bluestein
      3273  3.1091           prime    replayed       21998      35266    1.56  7.5e-16  bluestein
      3274  2.1637           prime    replayed       22640      34087    1.23  6.3e-16  bluestein
      3275  5^2.131          prime    replayed       22578      24318    0.73  6.1e-16  flips differ 1.48x bluestein
      3276  2^2.3^2.7.13     chain3   replayed        5172       5097    0.97  6.1e-16  
      3277  29.113           prime    replayed       22006      40996    1.86  6.1e-16  bluestein
      3278  2.11.149         prime    replayed       22360      23227    1.00  6.0e-16  bluestein
      3279  3.1093           prime    replayed       22515      35288    1.55  7.9e-16  bluestein
      3280  2^4.5.41         flat     replayed        6509      21518    3.30  4.4e-16  
      3281  17.193           prime    replayed       22824      30056    1.30  5.4e-16  bluestein
      3282  2.3.547          prime    replayed       22970      30894    1.12  7.8e-16  bluestein
      3283  7^2.67           prime    replayed       22659      32760    1.44  5.4e-16  bluestein
      3284  2^2.821          prime    replayed       22885      33259    1.45  6.4e-16  bluestein
      3285  3^2.5.73         prime    replayed       21984      25112    1.11  6.4e-16  bluestein
      3286  2.31.53          prime    replayed       22004      39408    1.75  4.7e-16  bluestein
      3287  19.173           prime    replayed       22877      39223    1.71  5.9e-16  bluestein
      3288  2^3.3.137        prime    replayed       22675      28902    1.27  5.4e-16  bluestein
      3289  11.13.23         chain3   replayed        9364      16156    1.72  6.6e-16  
      3290  2.5.7.47         flat     replayed        9750      23463    2.36  7.1e-16  
      3291  3.1097           prime    replayed       21996      30395    1.08  6.9e-16  flips differ 1.28x bluestein
      3292  2^2.823          prime    replayed       21995      32881    1.46  6.6e-16  bluestein
      3293  37.89            prime    replayed       22391      38135    1.69  6.1e-16  bluestein
      3294  2.3^3.61         prime    replayed       22883      29600    1.29  5.4e-16  bluestein
      3295  5.659            prime    replayed       22015      28026    1.24  7.2e-16  bluestein
      3296  2^5.103          prime    replayed       22019      24035    1.06  5.8e-16  bluestein
      3297  3.7.157          prime    replayed       22047      21745    0.97  6.0e-16  bluestein
      3298  2.17.97          prime    replayed       22419      28376    1.24  6.5e-16  bluestein
      3299  3299             prime    replayed       22335      31374    1.40  7.0e-16  bluestein
      3300  2^2.3.5^2.11     chain3   replayed        4831       4608    0.83  5.3e-16  
      3301  3301             prime    replayed       22003      31403    1.42  5.6e-16  bluestein
      3302  2.13.127         prime    replayed       22022      20512    0.91  6.9e-16  bluestein
      3303  3^2.367          prime    replayed       22360      26582    1.15  6.5e-16  bluestein
      3304  2^3.7.59         prime    replayed       22570      28480    1.26  4.2e-16  bluestein
      3305  5.661            prime    replayed       22761      28794    1.24  5.7e-16  bluestein
      3306  2.3.19.29        chain3   replayed        8622      24515    2.24  6.3e-16  flips differ 1.27x
      3307  3307             prime    replayed       22512      31588    1.40  6.5e-16  bluestein
      3308  2^2.827          prime    replayed       22021      26314    1.19  6.4e-16  bluestein
      3309  3.1103           prime    replayed       22028      30431    1.35  7.9e-16  bluestein
      3310  2.5.331          prime    replayed       22994      28439    1.05  5.9e-16  bluestein
      3311  7.11.43          chain3   replayed       11468      22879    1.48  5.0e-16  flips differ 1.35x
      3312  2^4.3^2.23       chain3   replayed        4934      13772    2.78  4.9e-16  
      3313  3313             prime    replayed       18889      31527    1.60  1.1e-15  rader
      3314  2.1657           prime    replayed       22364      34467    1.21  5.7e-16  flips differ 1.28x bluestein
      3315  3.5.13.17        flat     replayed        7074      13598    1.73  1.4e-15  
      3316  2^2.829          prime    replayed       22585      33412    1.06  6.4e-16  flips differ 1.40x bluestein
      3317  31.107           prime    replayed       22413      36533    1.59  6.4e-16  bluestein
      3318  2.3.7.79         prime    replayed       22042      21381    0.95  5.2e-16  bluestein
      3319  3319             prime    replayed       22424      31545    1.37  6.9e-16  bluestein
      3320  2^3.5.83         prime    replayed       22459      23795    0.82  6.2e-16  flips differ 1.28x bluestein
      3321  3^4.41           chain3   replayed        8146      21897    2.41  5.4e-16  
      3322  2.11.151         prime    replayed       22052      22007    0.98  6.1e-16  bluestein
      3323  3323             prime    replayed       22425      31605    1.40  6.7e-16  bluestein
      3324  2^2.3.277        prime    replayed       22398      24528    1.07  6.8e-16  bluestein
      3325  5^2.7.19         chain3   replayed        8006      13380    1.65  3.9e-16  
      3326  2.1663           prime    replayed       22726      34173    1.41  7.3e-16  bluestein
      3327  3.1109           prime    replayed       22670      30450    1.09  6.0e-16  bluestein
      3328  2^8.13           chain3   replayed        6210       4796    0.63  5.5e-16  
      3329  3329             prime    replayed       16286      31713    1.94  7.9e-16  rader
      3330  2.3^2.5.37       chain3   replayed       12336      20766    1.65  5.5e-16  
      3331  3331             prime    replayed       22404      31517    1.41  5.3e-16  bluestein
      3332  2^2.7^2.17       flat     replayed        5780      12517    1.59  6.2e-16  flips differ 1.36x
      3333  3.11.101         prime    replayed       22415      26019    1.14  7.4e-16  bluestein
      3334  2.1667           prime    replayed       22298      34155    1.52  5.9e-16  bluestein
      3335  5.23.29          chain3   replayed        9191      26714    2.36  5.0e-16  
      3336  2^3.3.139        prime    replayed       22581      22825    1.00  5.8e-16  bluestein
      3337  47.71            prime    replayed       22571      42629    1.34  5.9e-16  flips differ 1.41x bluestein
      3338  2.1669           prime    replayed       22549      34340    1.51  6.9e-16  bluestein
      3339  3^2.7.53         prime    replayed       23052      26677    1.16  5.1e-16  bluestein
      3340  2^2.5.167        prime    replayed       21964      23537    1.06  5.8e-16  bluestein
      3341  13.257           prime    replayed       22033      27356    1.21  5.8e-16  bluestein
      3342  2.3.557          prime    replayed       22350      30931    1.34  6.3e-16  bluestein
      3343  3343             prime    replayed       22003      31495    1.43  6.5e-16  bluestein
      3344  2^4.11.19        chain3   replayed        4889      13818    2.80  4.8e-16  
      3345  3.5.223          prime    replayed       22322      28778    1.27  6.2e-16  bluestein
      3346  2.7.239          prime    replayed       22587      22953    1.01  5.4e-16  bluestein
      3347  3347             prime    replayed       22942      31589    1.38  5.8e-16  bluestein
      3348  2^2.3^3.31       flat     replayed        7509      18800    2.49  5.3e-16  
      3349  17.197           prime    replayed       22005      29978    1.36  5.7e-16  bluestein
      3350  2.5^2.67         prime    replayed       22352      29936    1.31  5.0e-16  bluestein
      3351  3.1117           prime    replayed       22914      30583    1.33  6.7e-16  bluestein
      3352  2^3.419          prime    replayed       22414      24857    1.08  6.3e-16  bluestein
      3353  7.479            prime    replayed       22692      23336    1.02  4.4e-16  bluestein
      3354  2.3.13.43        chain3   replayed        7626      23348    3.04  6.3e-16  
      3355  5.11.61          prime    replayed       22570      29656    1.30  5.1e-16  bluestein
      3356  2^2.839          prime    replayed       22933      32980    1.44  6.4e-16  bluestein
      3357  3^2.373          prime    replayed       22393      26689    1.19  6.5e-16  bluestein
      3358  2.23.73          prime    replayed       22088      34283    1.53  8.4e-16  bluestein
      3359  3359             prime    replayed       22395      31445    1.37  5.9e-16  bluestein
      3360  2^5.3.5.7        chain3   replayed        4140       4032    0.93  4.3e-16  
      3361  3361             prime    replayed       13769      31439    2.28  6.9e-16  rader
      3362  2.41^2           flat     raced          10567      40542    3.65  2.7e-15  
      3363  3.19.59          prime    replayed       22399      31154    1.36  7.4e-16  bluestein
      3364  2^2.29^2         chain3   replayed        7805      28836    3.68  5.0e-16  
      3365  5.673            prime    replayed       22626      28128    1.24  6.4e-16  bluestein
      3366  2.3^2.11.17      flat     replayed        8055      13525    1.66  9.0e-16  
      3367  7.13.37          chain3   replayed        7342      21216    2.80  7.5e-16  
      3368  2^3.421          prime    replayed       22661      24881    1.10  8.4e-16  bluestein
      3369  3.1123           prime    replayed       22027      30432    1.38  8.3e-16  bluestein
      3370  2.5.337          prime    replayed       22749      28684    1.25  5.5e-16  bluestein
      3371  3371             prime    replayed       22438      31495    1.08  6.5e-16  flips differ 1.29x bluestein
      3372  2^2.3.281        prime    replayed       22628      24260    0.89  5.6e-16  bluestein
      3373  3373             prime    replayed       23425      31509    1.30  4.9e-16  bluestein
      3374  2.7.241          prime    replayed       22415      27807    1.21  6.2e-16  bluestein
      3375  3^3.5^3          chain3   replayed        5177       5874    1.11  5.4e-16  
      3376  2^4.211          prime    replayed       22372      24271    1.07  5.6e-16  bluestein
      3377  11.307           prime    replayed       22571      31008    1.37  5.7e-16  bluestein
      3378  2.3.563          prime    replayed       22443      30991    1.35  5.9e-16  bluestein
      3379  31.109           prime    replayed       22557      43468    1.92  5.5e-16  bluestein
      3380  2^2.5.13^2       flat     replayed        5762       6988    1.10  6.7e-16  
      3381  3.7^2.23         flat     replayed        7934      15981    2.01  5.1e-16  
      3382  2.19.89          prime    replayed       22029      32211    1.46  5.7e-16  bluestein
      3383  17.199           prime    replayed       22393      30360    1.35  6.4e-16  bluestein
      3384  2^3.3^2.47       flat     replayed        7807      23763    2.95  7.8e-16  
      3385  5.677            prime    replayed       22028      26547    1.18  7.9e-16  bluestein
      3386  2.1693           prime    replayed       22377      27704    1.24  6.1e-16  bluestein
      3387  3.1129           prime    replayed       22084      27057    1.20  5.7e-16  bluestein
      3388  2^2.7.11^2       flat     replayed        5604       7079    1.25  6.0e-16  
      3389  3389             prime    replayed       22349      30677    1.28  6.6e-16  bluestein
      3390  2.3.5.113        prime    replayed       22054      28969    1.31  6.9e-16  bluestein
      3391  3391             prime    replayed       23051      29176    1.26  5.8e-16  bluestein
      3392  2^6.53           prime    replayed       22067      26445    1.17  4.6e-16  bluestein
      3393  3^2.13.29        flat     replayed        9128      18018    1.91  1.0e-15  
      3394  2.1697           prime    replayed       22616      27727    1.22  6.2e-16  bluestein
      3395  5.7.97           prime    replayed       22401      25343    0.95  6.0e-16  bluestein
      3396  2^2.3.283        prime    replayed       22014      24531    0.74  5.2e-16  flips differ 1.52x bluestein
      3397  43.79            prime    replayed       22409      40745    1.79  6.2e-16  bluestein
      3398  2.1699           prime    replayed       22579      28266    1.23  7.0e-16  bluestein
      3399  3.11.103         prime    replayed       22629      25455    1.12  6.8e-16  bluestein
      3400  2^3.5^2.17       chain3   replayed        5669      11923    1.87  4.8e-16  
      3401  19.179           prime    replayed       22452      39658    1.73  6.6e-16  bluestein
      3402  2.3^5.7          chain3   replayed        6384       5227    0.76  5.5e-16  
      3403  41.83            prime    replayed       22060      42144    1.86  5.4e-16  bluestein
      3404  2^2.23.37        flat     replayed        9007      30393    3.33  1.0e-15  
      3405  3.5.227          prime    replayed       22627      24361    1.07  6.0e-16  bluestein
      3406  2.13.131         prime    replayed       22034      26486    0.84  5.0e-16  flips differ 1.42x bluestein
      3407  3407             prime    replayed       22673      29186    1.28  6.1e-16  bluestein
      3408  2^4.3.71         prime    replayed       22647      24472    1.08  5.8e-16  bluestein
      3409  7.487            prime    replayed       22411      30198    1.32  5.9e-16  bluestein
      3410  2.5.11.31        flat     replayed        8453      19205    2.21  6.8e-16  
      3411  3^2.379          prime    replayed       22045      22144    1.00  5.8e-16  bluestein
      3412  2^2.853          prime    replayed       22420      33315    1.48  8.1e-16  bluestein
      3413  3413             prime    replayed       22489      29353    1.28  6.0e-16  bluestein
      3414  2.3.569          prime    replayed       22998      25608    0.76  5.7e-16  flips differ 1.47x bluestein
      3415  5.683            prime    replayed       22067      26267    0.79  6.1e-16  flips differ 1.51x bluestein
      3416  2^3.7.61         prime    replayed       22419      30943    1.34  5.2e-16  bluestein
      3417  3.17.67          prime    replayed       22083      38816    1.76  6.4e-16  bluestein
      3418  2.1709           prime    replayed       22615      28167    1.24  6.6e-16  bluestein
      3419  13.263           prime    replayed       22358      27432    1.23  6.6e-16  bluestein
      3420  2^2.3^2.5.19     flat     replayed        6231      12937    2.03  5.1e-16  
      3421  11.311           prime    replayed       22430      30684    1.06  6.0e-16  flips differ 1.28x bluestein
      3422  2.29.59          prime    replayed       22439      35288    1.31  6.7e-16  bluestein
      3423  3.7.163          prime    replayed       22726      25029    1.08  5.7e-16  bluestein
      3424  2^5.107          prime    replayed       22026      24255    1.07  5.7e-16  bluestein
      3425  5^2.137          prime    replayed       22973      30173    1.31  6.5e-16  bluestein
      3426  2.3.571          prime    replayed       22951      25472    1.10  5.5e-16  bluestein
      3427  23.149           prime    replayed       22677      35355    1.54  5.5e-16  bluestein
      3428  2^2.857          prime    replayed       22970      27466    1.19  7.5e-16  bluestein
      3429  3^3.127          prime    replayed       22951      21879    0.94  5.7e-16  bluestein
      3430  2.5.7^3          flat     replayed        7349       5299    0.72  6.2e-16  
      3431  47.73            prime    replayed       22854      45080    1.92  5.5e-16  bluestein
      3432  2^3.3.11.13      flat     replayed        5983       7037    1.12  6.0e-16  
      3433  3433             prime    replayed       22409      29226    1.30  6.7e-16  bluestein
      3434  2.17.101         prime    replayed       22853      33342    1.01  6.1e-16  flips differ 1.46x bluestein
      3435  3.5.229          prime    replayed       22060      24411    1.10  5.6e-16  bluestein
      3436  2^2.859          prime    replayed       22810      27208    1.19  6.6e-16  bluestein
      3437  7.491            prime    replayed       22637      30037    1.31  5.4e-16  bluestein
      3438  2.3^2.191        prime    replayed       22121      20788    0.93  6.0e-16  bluestein
      3439  19.181           prime    replayed       22994      38671    1.45  6.7e-16  bluestein
      3440  2^4.5.43         chain3   replayed        6825      23397    3.41  5.7e-16  
      3441  3.31.37          flat     replayed        9752      35065    3.57  2.1e-15  
      3442  2.1721           prime    replayed       22601      27708    1.22  6.0e-16  bluestein
      3443  11.313           prime    replayed       22433      30681    1.35  5.8e-16  bluestein
      3444  2^2.3.7.41       flat     replayed        7752      23366    3.01  6.2e-16  
      3445  5.13.53          prime    replayed       22434      28063    1.22  5.6e-16  bluestein
      3446  2.1723           prime    replayed       22915      27893    1.21  6.7e-16  bluestein
      3447  3^2.383          prime    replayed       23020      22177    0.95  5.8e-16  bluestein
      3448  2^3.431          prime    replayed       22450      24942    1.08  6.0e-16  bluestein
      3449  3449             prime    replayed       21614      29223    1.32  5.2e-16  bluestein
      3450  2.3.5^2.23       chain3   replayed        8119      14453    1.75  5.6e-16  
      3451  7.17.29          chain3   replayed        8486      25513    2.36  5.5e-16  flips differ 1.27x
      3452  2^2.863          prime    replayed       22445      27626    0.81  6.5e-16  flips differ 1.50x bluestein
      3453  3.1151           prime    replayed       22789      27108    1.18  6.5e-16  bluestein
      3454  2.11.157         prime    replayed       22493      22399    0.96  5.1e-16  bluestein
      3455  5.691            prime    replayed       22828      27096    1.12  5.6e-16  bluestein
      3456  2^7.3^3          ztt      replayed        3644       4201    1.15  4.0e-16  
      3457  3457             prime    replayed       13879      30681    2.18  7.4e-16  rader
      3458  2.7.13.19        flat     replayed        8039      14708    1.66  6.9e-16  
      3459  3.1153           prime    replayed       22655      33319    1.45  6.9e-16  bluestein
      3460  2^2.5.173        prime    replayed       22108      23946    1.04  5.9e-16  bluestein
      3461  3461             prime    replayed       22099      30474    1.37  6.7e-16  bluestein
      3462  2.3.577          prime    replayed       22691      30769    1.35  6.0e-16  bluestein
      3463  3463             prime    replayed       22399      30557    1.32  6.9e-16  bluestein
      3464  2^3.433          prime    replayed       22067      27716    1.25  5.8e-16  bluestein
      3465  3^2.5.7.11       chain3   replayed        5825       6172    0.91  5.0e-16  
      3466  2.1733           prime    replayed       22428      29678    1.29  6.2e-16  bluestein
      3467  3467             prime    replayed       22095      30270    1.34  5.9e-16  bluestein
      3468  2^2.3.17^2       flat     replayed        7066      20722    2.85  1.5e-15  
      3469  3469             prime    replayed       18853      30235    1.60  8.8e-16  rader
      3470  2.5.347          prime    replayed       22022      24088    1.06  5.8e-16  bluestein
      3471  3.13.89          prime    replayed       22927      24923    1.07  7.1e-16  bluestein
      3472  2^4.7.31         flat     replayed        6887      18579    2.10  4.8e-16  flips differ 1.28x
      3473  23.151           prime    replayed       27521      34080    1.21  5.7e-16  bluestein
      3474  2.3^2.193        prime    replayed       22053      22019    0.97  6.5e-16  bluestein
      3475  5^2.139          prime    replayed       22046      23327    1.05  6.9e-16  bluestein
      3476  2^2.11.79        prime    replayed       22061      24026    0.77  5.7e-16  flips differ 1.42x bluestein
      3477  3.19.61          prime    replayed       22042      40158    1.77  4.6e-16  bluestein
      3478  2.37.47          flat     raced          11231      40588    3.47  2.7e-15  
      3479  7^2.71           prime    replayed       22043      25844    1.15  6.9e-16  bluestein
      3480  2^3.3.5.29       flat     replayed        7164      16364    2.08  7.5e-16  
      3481  59^2             prime    replayed       22106      42054    1.84  5.3e-16  bluestein
      3482  2.1741           prime    replayed       22634      29726    1.31  5.9e-16  bluestein
      3483  3^4.43           chain3   replayed       14334      24849    1.40  5.4e-16  
      3484  2^2.13.67        prime    replayed       22402      32297    1.42  5.7e-16  bluestein
      3485  5.17.41          chain3   replayed       15173      32519    2.06  5.2e-16  
      3486  2.3.7.83         prime    replayed       22413      32563    1.45  5.4e-16  bluestein
      3487  11.317           prime    replayed       22111      22554    1.00  6.0e-16  bluestein
      3488  2^5.109          prime    replayed       22956      32419    1.40  7.1e-16  bluestein
      3489  3.1163           prime    replayed       22021      33031    1.49  6.6e-16  bluestein
      3490  2.5.349          prime    replayed       22898      24238    1.05  5.8e-16  bluestein
      3491  3491             prime    replayed       23013      30513    1.10  5.8e-16  bluestein
      3492  2^2.3^2.97       prime    replayed       22691      21597    0.94  5.4e-16  bluestein
      3493  7.499            prime    replayed       22730      30093    1.30  5.6e-16  bluestein
      3494  2.1747           prime    replayed       22462      30181    1.12  6.2e-16  bluestein
      3495  3.5.233          prime    replayed       22793      24404    1.06  4.9e-16  bluestein
      3496  2^3.19.23        chain3   replayed        6414      23610    3.60  5.1e-16  
      3497  13.269           prime    replayed       22587      27479    0.88  6.9e-16  flips differ 1.38x bluestein
      3498  2.3.11.53        prime    replayed       22721      27084    1.18  4.3e-16  bluestein
      3499  3499             prime    replayed       22434      30882    1.36  6.0e-16  bluestein
      3500  2^2.5^3.7        flat     replayed        5643       4709    0.83  5.6e-16  
      3501  3^2.389          prime    replayed       22441      25626    1.11  5.9e-16  bluestein
      3502  2.17.103         prime    replayed       22366      33671    1.50  5.1e-16  bluestein
      3503  31.113           prime    replayed       22064      45016    1.98  5.7e-16  bluestein
      3504  2^4.3.73         prime    replayed       22118      26074    1.16  5.8e-16  bluestein
      3505  5.701            prime    replayed       22489      26285    1.17  5.3e-16  bluestein
      3506  2.1753           prime    replayed       23005      29872    1.29  6.0e-16  bluestein
      3507  3.7.167          prime    replayed       22432      25341    1.11  6.1e-16  bluestein
      3508  2^2.877          prime    replayed       22084      28766    1.29  6.3e-16  bluestein
      3509  11^2.29          chain3   replayed        7421      19826    2.67  5.3e-16  
      3510  2.3^3.5.13       chain3   replayed        6402       5480    0.75  7.2e-16  
      3511  3511             prime    replayed       22085      30380    1.34  6.1e-16  bluestein
      3512  2^3.439          prime    replayed       22108      27794    1.25  6.0e-16  bluestein
      3513  3.1171           prime    replayed       22050      28216    1.27  5.7e-16  bluestein
      3514  2.7.251          prime    replayed       22475      22553    0.98  5.4e-16  bluestein
      3515  5.19.37          chain3   replayed        9233      30033    2.96  5.4e-16  
      3516  2^2.3.293        prime    replayed       22629      24711    1.08  5.9e-16  bluestein
      3517  3517             prime    replayed       23000      30703    1.33  5.9e-16  bluestein
      3518  2.1759           prime    replayed       22406      29968    1.33  5.7e-16  bluestein
      3519  3^2.17.23        flat     replayed        7598      23652    2.85  7.6e-16  
      3520  2^6.5.11         chain3   replayed        4267       4842    1.13  3.6e-16  
      3521  7.503            prime    replayed       22149      22038    0.96  6.1e-16  bluestein
      3522  2.3.587          prime    replayed       22128      30671    1.38  5.6e-16  bluestein
      3523  13.271           prime    replayed       22508      27775    1.19  6.0e-16  bluestein
      3524  2^2.881          prime    replayed       22971      28532    1.24  6.4e-16  bluestein
      3525  3.5^2.47         flat     replayed        8263      26372    3.17  5.4e-16  
      3526  2.41.43          flat     replayed       11540      42509    3.52  2.4e-15  
      3527  3527             prime    replayed       22446      30337    1.32  7.9e-16  bluestein
      3528  2^3.3^2.7^2      chain3   replayed        5582       5115    0.91  4.6e-16  
      3529  3529             prime    replayed       22158      30402    1.33  6.7e-16  bluestein
      3530  2.5.353          prime    replayed       22989      24270    0.77  6.2e-16  flips differ 1.38x bluestein
      3531  3.11.107         prime    replayed       22396      25918    1.15  6.7e-16  bluestein
      3532  2^2.883          prime    replayed       22074      28818    1.27  6.4e-16  bluestein
      3533  3533             prime    replayed       22431      30562    1.33  5.9e-16  bluestein
      3534  2.3.19.31        chain3   replayed       10725      27611    2.12  5.9e-16  
      3535  5.7.101          prime    replayed       22083      26276    1.17  6.3e-16  bluestein
      3536  2^4.13.17        flat     replayed        6411      14449    2.23  8.5e-16  
      3537  3^3.131          prime    replayed       22429      26056    1.15  7.2e-16  bluestein
      3538  2.29.61          prime    replayed       22053      43131    1.89  5.3e-16  bluestein
      3539  3539             prime    replayed       22443      30408    1.32  5.3e-16  bluestein
      3540  2^2.3.5.59       prime    replayed       22110      28564    1.26  4.5e-16  bluestein
      3541  3541             prime    replayed       21659      30524    1.37  6.6e-16  bluestein
      3542  2.7.11.23        flat     replayed        8960      16283    1.81  6.4e-16  
      3543  3.1181           prime    replayed       22078      33268    1.15  6.1e-16  flips differ 1.30x bluestein
      3544  2^3.443          prime    replayed       22618      27722    1.22  5.5e-16  bluestein
      3545  5.709            prime    replayed       22360      26801    1.18  6.9e-16  bluestein
      3546  2.3^2.197        prime    replayed       22466      22400    0.99  5.3e-16  bluestein
      3547  3547             prime    replayed       22433      30335    1.33  5.9e-16  bluestein
      3548  2^2.887          prime    replayed       22073      28781    1.30  5.2e-16  bluestein
      3549  3.7.13^2         chain3   replayed        6400       7659    1.18  7.0e-16  
      3550  2.5^2.71         prime    replayed       23053      25848    1.12  6.5e-16  bluestein
      3551  53.67            prime    replayed       23001      58419    2.52  5.4e-16  bluestein
      3552  2^5.3.37         flat     replayed        7636      20720    2.67  6.7e-16  
      3553  11.17.19         chain3   replayed        6839      23094    3.21  6.0e-16  
      3554  2.1777           prime    replayed       22057      29748    1.35  6.4e-16  bluestein
      3555  3^2.5.79         prime    replayed       22098      23989    1.05  5.9e-16  bluestein
      3556  2^2.7.127        prime    replayed       22461      20551    0.89  6.1e-16  bluestein
      3557  3557             prime    replayed       22124      30895    1.08  6.3e-16  flips differ 1.30x bluestein
      3558  2.3.593          prime    replayed       22065      30886    1.07  5.7e-16  flips differ 1.30x bluestein
      3559  3559             prime    replayed       22134      30536    1.35  5.7e-16  bluestein
      3560  2^3.5.89         prime    replayed       22621      31004    1.37  6.3e-16  bluestein
      3561  3.1187           prime    replayed       22075      28067    1.22  5.7e-16  bluestein
      3562  2.13.137         prime    replayed       22113      25427    1.12  4.9e-16  bluestein
      3563  7.509            prime    replayed       22145      22493    0.96  5.5e-16  bluestein
      3564  2^2.3^4.11       chain3   replayed        5049       5479    1.02  5.6e-16  
      3565  5.23.31          flat     replayed        8888      29525    3.28  1.6e-15  
      3566  2.1783           prime    replayed       22658      29969    1.32  6.5e-16  bluestein
      3567  3.29.41          flat     replayed       10141      36416    3.58  2.1e-15  
      3568  2^4.223          prime    replayed       23129      28797    1.24  5.6e-16  bluestein
      3569  43.83            prime    replayed       23073      45112    1.95  5.0e-16  bluestein
      3570  2.3.5.7.17       chain3   replayed        6412      13399    2.07  5.3e-16  
      3571  3571             prime    replayed       22126      30750    1.36  6.5e-16  bluestein
      3572  2^2.19.47        chain3   replayed       13311      33182    2.48  7.3e-16  
      3573  3^2.397          prime    replayed       22120      25696    1.14  4.9e-16  bluestein
      3574  2.1787           prime    replayed       22159      30081    1.33  6.8e-16  bluestein
      3575  5^2.11.13        chain3   replayed        6411       7822    1.00  5.9e-16  
      3576  2^3.3.149        prime    replayed       22711      26490    1.16  6.2e-16  bluestein
      3577  7^2.73           prime    replayed       22110      27072    1.22  6.6e-16  bluestein
      3578  2.1789           prime    replayed       22132      30072    1.34  7.2e-16  bluestein
      3579  3.1193           prime    replayed       22667      33009    1.45  6.4e-16  bluestein
      3580  2^2.5.179        prime    replayed       22127      24378    1.09  5.6e-16  bluestein
      3581  3581             prime    replayed       22139      30577    1.33  7.2e-16  bluestein
      3582  2.3^2.199        prime    replayed       22449      22438    0.99  5.5e-16  bluestein
      3583  3583             prime    replayed       22499      30628    1.32  6.1e-16  bluestein
      3584  2^9.7            ztt      replayed        3503       4368    1.09  3.2e-16  
      3585  3.5.239          prime    replayed       22409      24575    1.07  6.6e-16  bluestein
      3586  2.11.163         prime    replayed       22123      27482    1.24  7.3e-16  bluestein
      3587  17.211           prime    replayed       22110      33867    1.52  5.5e-16  bluestein
      3588  2^2.3.13.23      chain3   replayed        5819      16705    2.51  7.1e-16  
      3589  37.97            prime    replayed       22444      38905    1.73  6.3e-16  bluestein
      3590  2.5.359          prime    replayed       22135      24372    1.07  6.2e-16  bluestein
      3591  3^3.7.19         chain3   replayed        7352      14450    1.68  5.1e-16  
      3592  2^3.449          prime    replayed       22667      28092    1.24  5.6e-16  bluestein
      3593  3593             prime    replayed       22672      30587    1.34  6.1e-16  bluestein
      3594  2.3.599          prime    replayed       23035      26975    1.16  6.3e-16  bluestein
      3595  5.719            prime    replayed       22409      26804    1.18  5.8e-16  bluestein
      3596  2^2.29.31        chain3   replayed        8348      32255    3.86  5.6e-16  
      3597  3.11.109         prime    replayed       22039      33316    1.50  6.1e-16  bluestein
      3598  2.7.257          prime    replayed       23047      28986    1.25  6.2e-16  bluestein
      3599  59.61            prime    replayed       22070      53038    2.35  5.7e-16  bluestein
      3600  2^4.3^2.5^2      ztt      replayed        4626       4564    0.98  4.3e-16  
      3601  13.277           prime    replayed       22401      27448    1.18  7.5e-16  bluestein
      3602  2.1801           prime    replayed       22074      38286    1.69  6.7e-16  bluestein
      3603  3.1201           prime    replayed       22099      32512    1.44  6.9e-16  bluestein
      3604  2^2.17.53        prime    replayed       23100      35479    1.52  6.0e-16  bluestein
      3605  5.7.103          prime    replayed       22740      26418    1.16  5.3e-16  bluestein
      3606  2.3.601          prime    replayed       23358      39982    1.71  4.7e-16  bluestein
      3607  3607             prime    replayed       22092      35312    1.59  7.0e-16  bluestein
      3608  2^3.11.41        chain3   replayed        7390      24050    3.20  5.7e-16  
      3609  3^2.401          prime    replayed       22736      27708    1.21  4.9e-16  bluestein
      3610  2.5.19^2         chain3   replayed        7651      23191    2.24  6.0e-16  flips differ 1.36x
      3611  23.157           prime    replayed       21828      33775    1.51  5.8e-16  bluestein
      3612  2^2.3.7.43       flat     replayed        9220      24331    2.26  5.4e-16  
      3613  3613             prime    replayed       22454      35343    1.54  7.1e-16  bluestein
      3614  2.13.139         prime    replayed       22082      25740    1.16  5.3e-16  bluestein
      3615  3.5.241          prime    replayed       22419      29551    1.28  6.2e-16  bluestein
      3616  2^5.113          prime    replayed       22123      30904    1.07  7.6e-16  flips differ 1.30x bluestein
      3617  3617             prime    replayed       22078      35345    1.60  6.3e-16  bluestein
      3618  2.3^3.67         prime    replayed       22998      32343    1.40  6.1e-16  bluestein
      3619  7.11.47          chain3   replayed       17220      25821    1.32  4.4e-16  
      3620  2^2.5.181        prime    replayed       22082      30234    1.37  4.8e-16  bluestein
      3621  3.17.71          prime    replayed       22151      33813    1.20  5.7e-16  flips differ 1.28x bluestein
      3622  2.1811           prime    replayed       22120      38785    1.74  8.0e-16  bluestein
      3623  3623             prime    replayed       22119      36371    1.60  7.9e-16  bluestein
      3624  2^3.3.151        prime    replayed       22137      24255    1.09  6.4e-16  bluestein
      3625  5^3.29           chain3   replayed        7619      18445    2.38  5.4e-16  
      3626  2.7^2.37         flat     replayed        9482      22276    2.33  5.9e-16  
      3627  3^2.13.31        flat     replayed        9810      20733    2.01  9.3e-16  
      3628  2^2.907          prime    replayed       22552      35773    1.46  7.5e-16  bluestein
      3629  19.191           prime    replayed       22699      31160    1.19  6.6e-16  bluestein
      3630  2.3.5.11^2       flat     replayed        8012       7375    0.92  6.2e-16  
      3631  3631             prime    replayed       22500      35300    1.57  7.2e-16  bluestein
      3632  2^4.227          prime    replayed       23011      25906    1.12  6.1e-16  bluestein
      3633  3.7.173          prime    replayed       22088      25371    1.11  6.6e-16  bluestein
      3634  2.23.79          prime    replayed       22124      34614    1.56  6.2e-16  bluestein
      3635  5.727            prime    replayed       22539      34067    1.48  6.7e-16  bluestein
      3636  2^2.3^2.101      prime    replayed       22198      26745    1.19  6.1e-16  bluestein
      3637  3637             prime    replayed       22670      35325    1.43  6.8e-16  bluestein
      3638  2.17.107         prime    replayed       22718      34129    1.49  5.6e-16  bluestein
      3639  3.1213           prime    replayed       23136      32596    1.40  5.4e-16  bluestein
      3640  2^3.5.7.13       flat     replayed        5776       5565    0.96  5.5e-16  
      3641  11.331           prime    replayed       22152      32403    1.43  6.3e-16  bluestein
      3642  2.3.607          prime    replayed       22130      33836    1.52  7.1e-16  bluestein
      3643  3643             prime    replayed       22118      35273    1.59  7.7e-16  bluestein
      3644  2^2.911          prime    replayed       22155      35687    1.57  6.3e-16  bluestein
      3645  3^6.5            chain3   replayed        6869       6651    0.95  5.7e-16  
      3646  2.1823           prime    replayed       22160      38326    1.69  6.8e-16  bluestein
      3647  7.521            prime    replayed       22137      28997    1.28  7.1e-16  bluestein
      3648  2^6.3.19         chain3   replayed        4921      13558    2.43  5.2e-16  
      3649  41.89            prime    replayed       23168      43868    1.20  8.4e-16  flips differ 1.58x bluestein
      3650  2.5^2.73         prime    replayed       22185      25953    1.14  5.8e-16  bluestein
      3651  3.1217           prime    replayed       22182      32375    1.43  5.2e-16  bluestein
      3652  2^2.11.83        prime    replayed       22707      27788    1.20  5.1e-16  bluestein
      3653  13.281           prime    replayed       22560      27657    1.20  5.6e-16  bluestein
      3654  2.3^2.7.29       flat     replayed        9501      19290    2.02  7.6e-16  
      3655  5.17.43          chain3   replayed        9747      34003    3.37  5.6e-16  
      3656  2^3.457          prime    replayed       22909      26395    1.14  6.6e-16  bluestein
      3657  3.23.53          prime    replayed       22742      39902    1.75  5.9e-16  bluestein
      3658  2.31.59          prime    replayed       22279      39441    1.74  6.8e-16  bluestein
      3659  3659             prime    replayed       23082      35287    1.23  7.1e-16  bluestein
      3660  2^2.3.5.61       prime    replayed       23055      32494    1.41  7.6e-16  bluestein
      3661  7.523            prime    replayed       22120      29186    1.28  6.8e-16  bluestein
      3662  2.1831           prime    replayed       22643      34939    1.54  7.3e-16  bluestein
      3663  3^2.11.37        flat     replayed        9261      23432    2.45  2.3e-15  
      3664  2^4.229          prime    replayed       22481      26038    1.13  8.0e-16  bluestein
      3665  5.733            prime    replayed       22158      31383    1.38  5.0e-16  bluestein
      3666  2.3.13.47        chain3   replayed        8550      25855    2.95  8.2e-16  
      3667  19.193           prime    replayed       22876      34390    1.49  5.9e-16  bluestein
      3668  2^2.7.131        prime    replayed       23050      26888    1.16  5.5e-16  bluestein
      3669  3.1223           prime    replayed       22132      32587    1.46  6.2e-16  bluestein
      3670  2.5.367          prime    replayed       22161      29313    1.30  5.9e-16  bluestein
      3671  3671             prime    replayed       22607      35027    1.53  5.2e-16  bluestein
      3672  2^3.3^3.17       chain3   replayed        6224      13383    2.14  5.6e-16  
      3673  3673             prime    replayed       22300      35015    1.57  5.6e-16  bluestein
      3674  2.11.167         prime    replayed       22516      27415    1.18  7.3e-16  bluestein
      3675  3.5^2.7^2        chain3   replayed        7622       6399    0.80  5.3e-16  
      3676  2^2.919          prime    replayed       23013      36051    1.56  6.0e-16  bluestein
      3677  3677             prime    replayed       22119      34947    1.09  7.1e-16  flips differ 1.45x bluestein
      3678  2.3.613          prime    replayed       23052      31481    1.36  6.0e-16  bluestein
      3679  13.283           prime    replayed       22542      27631    1.22  6.3e-16  bluestein
      3680  2^5.5.23         flat     replayed        6263      14817    1.90  5.2e-16  
      3681  3^2.409          prime    replayed       22709      28402    1.25  6.9e-16  bluestein
      3682  2.7.263          prime    replayed       22134      29207    1.27  5.9e-16  bluestein
      3683  29.127           prime    replayed       22588      34653    1.09  6.0e-16  flips differ 1.40x bluestein
      3684  2^2.3.307        prime    replayed       22470      32390    1.42  5.8e-16  bluestein
      3685  5.11.67          prime    replayed       22700      35967    1.58  4.5e-16  bluestein
      3686  2.19.97          prime    replayed       22685      32270    1.41  6.3e-16  bluestein
      3687  3.1229           prime    replayed       23064      32283    1.40  6.1e-16  bluestein
      3688  2^3.461          prime    replayed       22154      26373    1.16  7.0e-16  bluestein
      3689  7.17.31          chain3   replayed       11451      29053    2.52  6.3e-16  
      3690  2.3^2.5.41       chain3   replayed        7545      24838    3.26  5.5e-16  
      3691  3691             prime    replayed       22547      35159    1.23  4.7e-16  flips differ 1.27x bluestein
      3692  2^2.13.71        prime    replayed       22483      26675    1.15  6.0e-16  bluestein
      3693  3.1231           prime    replayed       22601      32927    1.41  6.1e-16  bluestein
      3694  2.1847           prime    replayed       22538      34941    1.51  6.2e-16  bluestein
      3695  5.739            prime    replayed       22526      31647    1.35  6.4e-16  bluestein
      3696  2^4.3.7.11       chain3   replayed        5063       5516    1.06  4.0e-16  
      3697  3697             prime    replayed       22112      35217    1.23  6.7e-16  flips differ 1.30x bluestein
      3698  2.43^2           flat     replayed       11953      45550    3.74  2.5e-15  
      3699  3^3.137          prime    replayed       22144      33444    1.49  5.7e-16  bluestein
      3700  2^2.5^2.37       flat     replayed        7444      21959    2.94  6.4e-16  
      3701  3701             prime    replayed       22238      35019    1.53  7.1e-16  bluestein
      3702  2.3.617          prime    replayed       22219      31536    1.41  7.5e-16  bluestein
      3703  7.23^2           chain3   replayed        8349      27791    3.31  6.3e-16  
      3704  2^3.463          prime    replayed       22571      26616    1.17  5.0e-16  bluestein
      3705  3.5.13.19        chain3   replayed        6701      15872    2.35  5.7e-16  
      3706  2.17.109         prime    replayed       23114      41882    1.81  5.6e-16  bluestein
      3707  11.337           prime    replayed       22737      32643    1.43  6.9e-16  bluestein
      3708  2^2.3^2.103      prime    replayed       22579      26459    1.17  6.3e-16  bluestein
      3709  3709             prime    replayed       22200      35104    1.54  6.5e-16  bluestein
      3710  2.5.7.53         prime    replayed       23074      29213    1.26  4.8e-16  bluestein
      3711  3.1237           prime    replayed       22173      32475    1.44  6.3e-16  bluestein
      3712  2^7.29           flat     replayed        8241      19236    2.33  6.2e-16  
      3713  47.79            prime    replayed       22211      45678    2.05  5.5e-16  bluestein
      3714  2.3.619          prime    replayed       23203      31418    1.35  6.7e-16  bluestein
      3715  5.743            prime    replayed       22597      31422    1.36  7.7e-16  bluestein
      3716  2^2.929          prime    replayed       22227      35888    1.58  6.6e-16  bluestein
      3717  3^2.7.59         prime    replayed       23101      30248    1.30  5.1e-16  bluestein
      3718  2.11.13^2        flat     replayed        9063       9401    1.03  9.1e-16  
      3719  3719             prime    replayed       22575      35037    1.55  5.5e-16  bluestein
      3720  2^3.3.5.31       chain3   replayed        6062      18955    3.09  6.6e-16  
      3721  61^2             prime    replayed       22177      60421    2.68  5.0e-16  bluestein
      3722  2.1861           prime    replayed       22723      34736    1.53  7.0e-16  bluestein
      3723  3.17.73          prime    replayed       22576      35823    1.57  6.2e-16  bluestein
      3724  2^2.7^2.19       flat     replayed        6435      14785    2.23  7.9e-16  
      3725  5^2.149          prime    replayed       22649      25738    1.13  5.3e-16  bluestein
      3726  2.3^4.23         chain3   replayed       11497      16086    1.15  5.4e-16  
      3727  3727             prime    replayed       22539      35175    1.53  7.4e-16  bluestein
      3728  2^4.233          prime    replayed       22457      26060    1.15  6.6e-16  bluestein
      3729  3.11.113         prime    replayed       22451      33069    1.45  7.7e-16  bluestein
      3730  2.5.373          prime    replayed       22879      29393    1.28  6.3e-16  bluestein
      3731  7.13.41          chain3   replayed        8573      25247    2.90  6.7e-16  
      3732  2^2.3.311        prime    replayed       22588      32695    1.41  6.1e-16  bluestein
      3733  3733             prime    replayed       22576      35154    1.55  7.0e-16  bluestein
      3734  2.1867           prime    replayed       22665      35086    1.49  6.6e-16  bluestein
      3735  3^2.5.83         prime    replayed       22679      27838    1.22  7.9e-16  bluestein
      3736  2^3.467          prime    replayed       22513      26522    1.17  6.7e-16  bluestein
      3737  37.101           prime    replayed       22710      44951    1.97  7.1e-16  bluestein
      3738  2.3.7.89         prime    replayed       22599      26000    1.12  6.9e-16  bluestein
      3739  3739             prime    replayed       22562      35166    1.52  7.1e-16  bluestein
      3740  2^2.5.11.17      flat     replayed        6462      14998    2.29  6.4e-16  
      3741  3.29.43          flat     replayed       11096      38793    3.48  3.5e-15  
      3742  2.1871           prime    replayed       22688      35174    1.54  6.0e-16  bluestein
      3743  19.197           prime    replayed       22154      34443    1.52  5.8e-16  bluestein
      3744  2^5.3^2.13       chain3   replayed        4648       5517    1.18  5.7e-16  
      3745  5.7.107          prime    replayed       23117      26536    1.00  6.0e-16  bluestein
      3746  2.1873           prime    replayed       22711      34959    1.53  7.2e-16  bluestein
      3747  3.1249           prime    replayed       22558      32365    1.43  5.5e-16  bluestein
      3748  2^2.937          prime    replayed       22163      36142    1.63  6.6e-16  bluestein
      3749  23.163           prime    replayed       22727      38359    1.68  5.7e-16  bluestein
      3750  2.3.5^4          chain3   replayed        6404       5670    0.87  4.6e-16  
      3751  11^2.31          flat     replayed       10305      22380    2.06  8.3e-16  
      3752  2^3.7.67         prime    replayed       22187      33532    1.04  5.7e-16  flips differ 1.44x bluestein
      3753  3^3.139          prime    replayed       22533      33114    1.46  6.1e-16  bluestein
      3754  2.1877           prime    replayed       22607      30294    1.32  6.3e-16  bluestein
      3755  5.751            prime    replayed       22846      33429    1.35  5.5e-16  bluestein
      3756  2^2.3.313        prime    replayed       22566      32417    1.41  6.0e-16  bluestein
      3757  13.17^2          chain3   replayed        7360      23826    3.12  7.0e-16  
      3758  2.1879           prime    replayed       23112      29813    1.28  6.5e-16  bluestein
      3759  3.7.179          prime    replayed       22525      25569    1.13  6.2e-16  bluestein
      3760  2^4.5.47         chain3   replayed        7867      26085    3.07  5.1e-16  
      3761  3761             prime    replayed       22601      30747    1.33  6.5e-16  bluestein
      3762  2.3^2.11.19      flat     replayed        8687      15619    1.78  7.9e-16  
      3763  53.71            prime    replayed       22197      50949    2.24  7.0e-16  bluestein
      3764  2^2.941          prime    replayed       22766      28347    1.24  5.7e-16  bluestein
      3765  3.5.251          prime    replayed       22646      22612    0.98  5.9e-16  bluestein
      3766  2.7.269          prime    replayed       22580      28643    1.26  5.5e-16  bluestein
      3767  3767             prime    replayed       22188      30786    1.38  5.6e-16  bluestein
      3768  2^3.3.157        prime    replayed       22801      24434    1.06  7.7e-16  bluestein
      3769  3769             prime    replayed       22595      30800    0.91  6.1e-16  flips differ 1.50x bluestein
      3770  2.5.13.29        chain3   replayed       11643      19833    1.68  7.3e-16  
      3771  3^2.419          prime    replayed       22160      28436    1.28  7.8e-16  bluestein
      3772  2^2.23.41        flat     replayed        9793      35723    3.63  2.2e-15  
      3773  7^3.11           flat     replayed        7251       7218    0.77  6.5e-16  flips differ 1.31x
      3774  2.3.17.37        chain3   replayed        8299      31553    3.75  6.6e-16  
      3775  5^2.151          prime    replayed       22547      23799    1.05  5.9e-16  bluestein
      3776  2^6.59           prime    replayed       22563      25125    1.09  6.6e-16  bluestein
      3777  3.1259           prime    replayed       22570      29688    1.30  5.4e-16  bluestein
      3778  2.1889           prime    replayed       22226      29267    1.29  5.7e-16  bluestein
      3779  3779             prime    replayed       22569      30928    1.33  6.1e-16  bluestein
      3780  2^2.3^3.5.7      chain3   replayed        5280       5517    1.02  5.1e-16  
      3781  19.199           prime    replayed       22587      34644    1.53  4.9e-16  bluestein
      3782  2.31.61          prime    replayed       23104      48219    2.07  5.2e-16  bluestein
      3783  3.13.97          prime    replayed       22186      24828    1.09  6.0e-16  bluestein
      3784  2^3.11.43        chain3   replayed        7945      26148    3.27  5.4e-16  
      3785  5.757            prime    replayed       22214      32499    1.46  6.6e-16  bluestein
      3786  2.3.631          prime    replayed       22175      25378    1.12  5.4e-16  bluestein
      3787  7.541            prime    replayed       22189      43004    1.89  5.7e-16  bluestein
      3788  2^2.947          prime    replayed       22616      28502    1.22  6.3e-16  bluestein
      3789  3^2.421          prime    replayed       22498      28552    1.24  7.0e-16  bluestein
      3790  2.5.379          prime    replayed       22191      24868    1.12  4.9e-16  bluestein
      3791  17.223           prime    replayed       22510      39569    1.71  6.0e-16  bluestein
      3792  2^4.3.79         prime    replayed       22170      25746    1.15  6.4e-16  bluestein
      3793  3793             prime    replayed       22214      30817    1.35  6.1e-16  bluestein
      3794  2.7.271          prime    replayed       23511      28794    1.22  5.3e-16  bluestein
      3795  3.5.11.23        chain3   replayed        7226      17565    2.25  5.2e-16  
      3796  2^2.13.73        prime    replayed       22152      28205    1.27  6.5e-16  bluestein
      3797  3797             prime    replayed       22545      30733    1.36  5.0e-16  bluestein
      3798  2.3^2.211        prime    replayed       22211      26789    1.20  5.8e-16  bluestein
      3799  29.131           prime    replayed       22570      41495    1.80  5.7e-16  bluestein
      3800  2^3.5^2.19       flat     replayed        6460      13827    2.09  5.7e-16  
      3801  3.7.181          prime    replayed       22901      32203    1.40  6.3e-16  bluestein
      3802  2.1901           prime    replayed       22218      29659    1.30  6.8e-16  bluestein
      3803  3803             prime    replayed       22210      30827    1.36  6.3e-16  bluestein
      3804  2^2.3.317        prime    replayed       22212      23397    1.05  6.3e-16  bluestein
      3805  5.761            prime    replayed       23137      32661    1.41  8.2e-16  bluestein
      3806  2.11.173         prime    replayed       21709      27590    1.26  6.0e-16  bluestein
      3807  3^4.47           chain3   replayed       19030      29375    1.38  5.1e-16  
      3808  2^5.7.17         chain3   replayed        5116      14236    2.65  6.3e-16  
      3809  13.293           prime    replayed       22535      28165    1.05  5.7e-16  bluestein
      3810  2.3.5.127        prime    replayed       22562      21915    0.96  6.0e-16  bluestein
      3811  37.103           prime    replayed       22187      45188    1.57  7.1e-16  flips differ 1.30x bluestein
      3812  2^2.953          prime    replayed       22569      28331    1.22  6.0e-16  bluestein
      3813  3.31.41          flat     replayed       11252      40552    3.58  2.5e-15  
      3814  2.1907           prime    replayed       22207      29731    1.32  6.8e-16  bluestein
      3815  5.7.109          prime    replayed       22599      35916    1.56  5.4e-16  bluestein
      3816  2^3.3^2.53       prime    replayed       23105      29781    1.28  4.6e-16  bluestein
      3817  11.347           prime    replayed       22193      27976    1.26  6.2e-16  bluestein
      3818  2.23.83          prime    replayed       22227      39256    1.76  5.8e-16  bluestein
      3819  3.19.67          prime    replayed       22475      47780    2.11  6.0e-16  bluestein
      3820  2^2.5.191        prime    replayed       22251      22896    1.01  5.8e-16  bluestein
      3821  3821             prime    replayed       22247      30911    1.36  5.9e-16  bluestein
      3822  2.3.7^2.13       flat     replayed        8691       6157    0.57  6.1e-16  
      3823  3823             prime    replayed       22558      30920    1.34  5.9e-16  bluestein
      3824  2^4.239          prime    replayed       22603      26214    1.13  6.6e-16  bluestein
      3825  3^2.5^2.17       chain3   replayed        6999      14843    2.11  6.8e-16  
      3826  2.1913           prime    replayed       22757      29308    1.27  6.5e-16  bluestein
      3827  43.89            prime    replayed       22783      54469    2.38  5.7e-16  bluestein
      3828  2^2.3.11.29      chain3   replayed        6495      20535    3.16  6.1e-16  
      3829  7.547            prime    replayed       22217      36256    1.15  6.2e-16  flips differ 1.41x bluestein
      3830  2.5.383          prime    replayed       22614      24513    1.06  6.0e-16  bluestein
      3831  3.1277           prime    replayed       22802      29837    1.30  6.9e-16  bluestein
      3832  2^3.479          prime    replayed       22678      26959    1.19  6.3e-16  bluestein
      3833  3833             prime    replayed       22769      30863    1.35  6.3e-16  bluestein
      3834  2.3^3.71         prime    replayed       21800      27867    1.27  5.9e-16  bluestein
      3835  5.13.59          prime    replayed       22624      26539    1.17  6.3e-16  bluestein
      3836  2^2.7.137        prime    replayed       22247      25932    0.93  4.7e-16  bluestein
      3837  3.1279           prime    replayed       22757      29343    1.03  6.5e-16  flips differ 1.26x bluestein
      3838  2.19.101         prime    replayed       22248      38071    1.70  9.3e-16  bluestein
      3839  11.349           prime    replayed       23110      28012    1.21  6.7e-16  bluestein
      3840  2^8.3.5          ztt      replayed        4021       4309    1.04  4.1e-16  
      3841  23.167           prime    replayed       22621      38740    1.69  5.9e-16  bluestein
      3842  2.17.113         prime    replayed       22231      42161    1.89  6.4e-16  bluestein
      3843  3^2.7.61         prime    replayed       22575      34977    1.51  5.0e-16  bluestein
      3844  2^2.31^2         chain3   replayed        9088      36349    3.96  6.1e-16  
      3845  5.769            prime    replayed       22233      30894    1.37  6.8e-16  bluestein
      3846  2.3.641          prime    replayed       22254      39996    1.79  7.8e-16  bluestein
      3847  3847             prime    replayed       22612      36418    1.57  6.9e-16  bluestein
      3848  2^3.13.37        chain3   replayed       11333      24284    2.13  6.6e-16  
      3849  3.1283           prime    replayed       22485      33742    1.46  5.5e-16  bluestein
      3850  2.5^2.7.11       flat     replayed        7806       6116    0.78  5.8e-16  
      3851  3851             prime    replayed       22556      36471    1.62  7.2e-16  bluestein
      3852  2^2.3^2.107      prime    replayed       23143      26805    1.15  7.9e-16  bluestein
      3853  3853             prime    replayed       22548      36540    1.62  7.5e-16  bluestein
      3854  2.41.47          flat     raced          13119      46682    3.49  3.2e-15  
      3855  3.5.257          prime    replayed       22159      37515    1.68  7.2e-16  bluestein
      3856  2^4.241          prime    replayed       22708      31951    1.41  7.2e-16  bluestein
      3857  7.19.29          chain3   replayed        8808      28769    2.89  5.6e-16  
      3858  2.3.643          prime    replayed       22177      41268    1.81  6.7e-16  bluestein
      3859  17.227           prime    replayed       22718      36678    1.61  5.6e-16  bluestein
      3860  2^2.5.193        prime    replayed       22165      24618    1.08  6.4e-16  bluestein
      3861  3^3.11.13        flat     replayed        8530       8784    1.02  9.9e-16  
      3862  2.1931           prime    replayed       22799      35325    1.55  7.5e-16  bluestein
      3863  3863             prime    replayed       22191      36641    1.65  8.6e-16  bluestein
      3864  2^3.3.7.23       flat     replayed        8203      17785    2.15  7.0e-16  
      3865  5.773            prime    replayed       22149      29960    1.28  5.9e-16  bluestein
      3866  2.1933           prime    replayed       22574      35279    1.56  6.9e-16  bluestein
      3867  3.1289           prime    replayed       22649      33904    1.48  6.6e-16  bluestein
      3868  2^2.967          prime    replayed       22255      34344    1.50  7.2e-16  bluestein
      3869  53.73            prime    replayed       23111      53169    2.30  7.3e-16  bluestein
      3870  2.3^2.5.43       chain3   replayed        8127      25818    3.17  5.3e-16  
      3871  7^2.79           prime    replayed       22643      26792    1.15  6.5e-16  bluestein
      3872  2^5.11^2         chain3   replayed        5432       7632    1.40  3.9e-16  
      3873  3.1291           prime    replayed       23431      33897    1.44  6.1e-16  bluestein
      3874  2.13.149         prime    replayed       22492      28047    1.22  5.4e-16  bluestein
      3875  5^3.31           chain3   replayed        9179      21655    2.32  5.6e-16  
      3876  2^2.3.17.19      chain3   replayed        6561      23984    3.56  6.2e-16  
      3877  3877             prime    replayed       22295      36572    1.60  7.1e-16  bluestein
      3878  2.7.277          prime    replayed       22970      28750    1.24  5.9e-16  bluestein
      3879  3^2.431          prime    replayed       23002      28557    1.03  7.0e-16  bluestein
      3880  2^3.5.97         prime    replayed       22879      28359    1.22  5.7e-16  bluestein
      3881  3881             prime    replayed       23112      36546    1.58  7.4e-16  bluestein
      3882  2.3.647          prime    replayed       22759      31387    1.11  6.8e-16  bluestein
      3883  11.353           prime    replayed       22965      28062    1.22  6.0e-16  bluestein
      3884  2^2.971          prime    replayed       22519      34215    1.52  7.2e-16  bluestein
      3885  3.5.7.37         chain3   replayed        8494      23905    2.81  4.5e-16  
      3886  2.29.67          prime    replayed       22541      49407    2.17  6.1e-16  bluestein
      3887  13^2.23          flat     replayed        8289      19513    2.07  9.6e-16  
      3888  2^4.3^5          ztt      replayed        4475       5552    1.22  4.8e-16  
      3889  3889             prime    replayed       16214      32657    1.98  1.4e-15  rader
      3890  2.5.389          prime    replayed       22640      28351    1.20  4.9e-16  bluestein
      3891  3.1297           prime    replayed       22569      42512    1.88  7.3e-16  bluestein
      3892  2^2.7.139        prime    replayed       22294      26070    1.15  6.3e-16  bluestein
      3893  17.229           prime    replayed       22593      36638    1.58  5.7e-16  bluestein
      3894  2.3.11.59        prime    replayed       22927      26262    1.14  5.6e-16  bluestein
      3895  5.19.41          chain3   replayed       10586      35371    3.07  5.1e-16  
      3896  2^3.487          prime    replayed       22560      29094    1.29  6.0e-16  bluestein
      3897  3^2.433          prime    replayed       22922      39712    1.70  5.8e-16  bluestein
      3898  2.1949           prime    replayed       22226      32172    1.41  5.3e-16  bluestein
      3899  7.557            prime    replayed       22599      36309    1.60  6.8e-16  bluestein
      3900  2^2.3.5^2.13     chain3   replayed        5756       5736    0.97  7.5e-16  
      3901  47.83            prime    replayed       22244      51275    1.52  5.9e-16  flips differ 1.51x bluestein
      3902  2.1951           prime    replayed       22608      32005    1.40  7.4e-16  bluestein
      3903  3.1301           prime    replayed       22629      34519    1.08  7.9e-16  flips differ 1.40x bluestein
      3904  2^6.61           prime    replayed       22266      35365    1.55  4.2e-16  bluestein
      3905  5.11.71          prime    replayed       23039      28252    1.22  6.0e-16  bluestein
      3906  2.3^2.7.31       chain3   replayed        7940      20827    2.00  5.8e-16  flips differ 1.31x
      3907  3907             prime    replayed       22634      32394    1.41  6.7e-16  bluestein
      3908  2^2.977          prime    replayed       22595      30451    1.34  6.1e-16  bluestein
      3909  3.1303           prime    replayed       22249      43132    1.52  6.3e-16  flips differ 1.28x bluestein
      3910  2.5.17.23        chain3   replayed        8217      26015    3.15  6.1e-16  
      3911  3911             prime    replayed       22606      32679    1.42  5.9e-16  bluestein
      3912  2^3.3.163        prime    replayed       22775      28810    1.26  7.1e-16  bluestein
      3913  7.13.43          chain3   replayed       14057      27414    1.24  7.6e-16  flips differ 1.58x
      3914  2.19.103         prime    replayed       23140      38789    1.67  6.3e-16  bluestein
      3915  3^3.5.29         flat     replayed        9586      20075    2.09  6.1e-16  
      3916  2^2.11.89        prime    replayed       22819      27748    0.96  6.7e-16  flips differ 1.25x bluestein
      3917  3917             prime    replayed       22624      32666    0.96  5.9e-16  flips differ 1.50x bluestein
      3918  2.3.653          prime    replayed       22755      33478    1.06  5.7e-16  flips differ 1.39x bluestein
      3919  3919             prime    replayed       21698      32463    1.45  6.4e-16  bluestein
      3920  2^4.5.7^2        ztt      replayed        4220       5354    1.25  2.9e-16  
      3921  3.1307           prime    replayed       23171      34478    1.49  5.7e-16  bluestein
      3922  2.37.53          prime    replayed       22223      49760    2.24  5.3e-16  bluestein
      3923  3923             prime    replayed       22778      32458    1.42  4.8e-16  bluestein
      3924  2^2.3^2.109      prime    replayed       23129      35993    1.55  6.6e-16  bluestein
      3925  5^2.157          prime    replayed       22232      24003    1.05  5.2e-16  bluestein
      3926  2.13.151         prime    replayed       23161      26294    1.13  5.5e-16  bluestein
      3927  3.7.11.17        chain3   replayed        7739      16212    2.09  5.8e-16  
      3928  2^3.491          prime    replayed       22813      29278    0.87  5.9e-16  flips differ 1.47x bluestein
      3929  3929             prime    replayed       22528      32402    1.43  5.6e-16  bluestein
      3930  2.3.5.131        prime    replayed       22255      28788    1.29  7.1e-16  bluestein
      3931  3931             prime    replayed       22814      32452    1.42  6.3e-16  bluestein
      3932  2^2.983          prime    replayed       22661      30544    1.31  5.2e-16  bluestein
      3933  3^2.19.23        chain3   replayed        8216      27338    2.40  6.4e-16  flips differ 1.39x
      3934  2.7.281          prime    replayed       22249      29706    1.29  6.4e-16  bluestein
      3935  5.787            prime    replayed       22962      29816    1.28  5.9e-16  bluestein
      3936  2^5.3.41         chain3   replayed        7237      26090    3.54  4.7e-16  
      3937  31.127           prime    replayed       22257      39816    1.75  5.9e-16  bluestein
      3938  2.11.179         prime    replayed       22209      28187    1.27  6.3e-16  bluestein
      3939  3.13.101         prime    replayed       22289      30382    1.34  6.5e-16  bluestein
      3940  2^2.5.197        prime    replayed       22637      24699    1.06  6.3e-16  bluestein
      3941  7.563            prime    replayed       22238      36241    1.62  7.2e-16  bluestein
      3942  2.3^3.73         prime    replayed       22665      29782    1.28  5.7e-16  bluestein
      3943  3943             prime    replayed       22288      32260    1.41  7.1e-16  bluestein
      3944  2^3.17.29        flat     replayed        8314      28934    3.46  7.4e-16  
      3945  3.5.263          prime    replayed       23298      30526    1.31  6.0e-16  bluestein
      3946  2.1973           prime    replayed       22919      32148    1.40  5.7e-16  bluestein
      3947  3947             prime    replayed       22595      32928    1.42  6.3e-16  bluestein
      3948  2^2.3.7.47       flat     replayed       11579      29297    2.19  6.0e-16  
      3949  11.359           prime    replayed       22659      28303    1.21  6.5e-16  bluestein
      3950  2.5^2.79         prime    replayed       22580      26815    1.18  6.0e-16  bluestein
      3951  3^2.439          prime    replayed       22922      31593    1.38  6.8e-16  bluestein
      3952  2^4.13.19        chain3   replayed        5878      16793    2.44  4.9e-16  
      3953  59.67            prime    replayed       21831      58664    2.63  5.9e-16  bluestein
      3954  2.3.659          prime    replayed       23171      33551    1.44  5.9e-16  bluestein
      3955  5.7.113          prime    replayed       22246      35874    1.61  6.0e-16  bluestein
      3956  2^2.23.43        chain3   replayed        9833      38518    3.90  6.0e-16  
      3957  3.1319           prime    replayed       22646      42832    1.88  5.8e-16  bluestein
      3958  2.1979           prime    replayed       22284      32287    1.41  6.5e-16  bluestein
      3959  37.107           prime    replayed       22686      46281    2.02  5.8e-16  bluestein
      3960  2^3.3^2.5.11     flat     replayed        6785       5769    0.85  5.4e-16  
      3961  17.233           prime    replayed       22666      36927    1.59  5.7e-16  bluestein
      3962  2.7.283          prime    replayed       22281      29111    1.27  6.8e-16  bluestein
      3963  3.1321           prime    replayed       22918      42651    1.85  6.0e-16  bluestein
      3964  2^2.991          prime    replayed       22866      31415    1.13  6.0e-16  bluestein
      3965  5.13.61          prime    replayed       21867      35392    1.60  5.7e-16  bluestein
      3966  2.3.661          prime    replayed       22654      33759    1.46  6.1e-16  bluestein
      3967  3967             prime    replayed       22725      33138    1.43  6.7e-16  bluestein
      3968  2^7.31           chain3   replayed        6452      20904    3.19  4.6e-16  
      3969  3^4.7^2          chain3   replayed        7177       7552    1.05  6.1e-16  
      3970  2.5.397          prime    replayed       22344      28052    1.24  6.5e-16  bluestein
      3971  11.19^2          chain3   replayed        8105      26835    3.27  6.4e-16  
      3972  2^2.3.331        prime    replayed       22674      34557    1.48  5.4e-16  bluestein
      3973  29.137           prime    replayed       22646      41179    1.81  6.3e-16  bluestein
      3974  2.1987           prime    replayed       22846      32218    1.40  6.7e-16  bluestein
      3975  3.5^2.53         prime    replayed       22546      33159    1.47  4.6e-16  bluestein
      3976  2^3.7.71         prime    replayed       22632      29034    1.28  7.1e-16  bluestein
      3977  41.97            prime    replayed       27024      45111    1.55  6.1e-16  bluestein
      3978  2.3^2.13.17      flat     replayed        9149      16217    1.70  1.4e-15  
      3979  23.173           prime    replayed       21820      39268    1.79  5.7e-16  bluestein
      3980  2^2.5.199        prime    replayed       22637      24863    1.09  5.3e-16  bluestein
      3981  3.1327           prime    replayed       22967      42810    1.86  6.1e-16  bluestein
      3982  2.11.181         prime    replayed       22819      34855    1.52  6.1e-16  bluestein
      3983  7.569            prime    replayed       22593      29643    1.28  6.5e-16  bluestein
      3984  2^4.3.83         prime    replayed       22843      29147    1.18  6.3e-16  bluestein
      3985  5.797            prime    replayed       22248      29678    1.33  5.7e-16  bluestein
      3986  2.1993           prime    replayed       22250      32118    1.44  6.2e-16  bluestein
      3987  3^2.443          prime    replayed       22652      38607    1.65  6.9e-16  bluestein
      3988  2^2.997          prime    replayed       22237      30839    1.35  6.3e-16  bluestein
      3989  3989             prime    replayed       23550      32788    1.31  7.0e-16  bluestein
      3990  2.3.5.7.19       chain3   replayed        7034      15550    1.93  5.8e-16  
      3991  13.307           prime    replayed       22832      36498    1.59  6.2e-16  bluestein
      3992  2^3.499          prime    replayed       22226      29347    1.28  6.4e-16  bluestein
      3993  3.11^3           flat     replayed        7661       9692    1.25  8.9e-16  
      3994  2.1997           prime    replayed       22684      32309    1.39  6.0e-16  bluestein
      3995  5.17.47          chain3   replayed       15320      36632    2.38  6.1e-16  
      3996  2^2.3^3.37       chain3   replayed        7221      25265    3.48  5.4e-16  
      3997  7.571            prime    replayed       22629      30197    1.29  6.5e-16  bluestein
      3998  2.1999           prime    replayed       22837      32168    1.41  7.8e-16  bluestein
      3999  3.31.43          chain3   replayed       17240      43556    2.33  7.7e-16  
      4000  2^5.5^3          ztt      replayed        3993       4786    1.19  3.7e-16  
      4001  4001             prime    replayed       15217      37884    2.48  7.8e-16  rader
      4002  2.3.23.29        flat     replayed       10459      31560    3.01  2.1e-15  
      4003  4003             prime    replayed       22816      37685    1.65  7.4e-16  bluestein
      4004  2^2.7.11.13      flat     replayed        6721       8622    1.26  7.2e-16  
      4005  3^2.5.89         prime    replayed       22733      28003    1.21  6.0e-16  bluestein
      4006  2.2003           prime    replayed       22324      37507    1.31  6.6e-16  flips differ 1.29x bluestein
      4007  4007             prime    replayed       22323      37918    1.70  6.3e-16  bluestein
      4008  2^3.3.167        prime    replayed       22926      28799    1.24  5.4e-16  bluestein
      4009  19.211           prime    replayed       22632      38673    1.37  6.9e-16  bluestein
      4010  2.5.401          prime    replayed       22844      42945    1.84  7.0e-16  bluestein
      4011  3.7.191          prime    replayed       22623      24311    1.05  6.5e-16  bluestein
      4012  2^2.17.59        prime    replayed       23219      36371    1.56  5.3e-16  bluestein
      4013  4013             prime    replayed       23344      38031    1.43  7.6e-16  bluestein
      4014  2.3^2.223        prime    replayed       22666      32596    1.43  6.1e-16  bluestein
      4015  5.11.73          prime    replayed       22834      29599    1.29  5.8e-16  bluestein
      4016  2^4.251          prime    replayed       23277      24251    1.03  6.2e-16  bluestein
      4017  3.13.103         prime    replayed       22299      30280    1.33  6.8e-16  bluestein
      4018  2.7^2.41         flat     replayed       10862      26553    2.31  3.7e-15  
      4019  4019             prime    replayed       22709      37746    1.53  7.4e-16  bluestein
      4020  2^2.3.5.67       prime    replayed       22707      35593    1.53  6.2e-16  bluestein
      4021  4021             prime    replayed       22263      37816    1.69  6.4e-16  bluestein
      4022  2.2011           prime    replayed       22635      37226    1.60  8.8e-16  bluestein
      4023  3^3.149          prime    replayed       22604      30118    1.33  6.6e-16  bluestein
      4024  2^3.503          prime    replayed       22661      25636    1.08  5.1e-16  bluestein
      4025  5^2.7.23         chain3   replayed        7948      17970    2.25  5.3e-16  
      4026  2.3.11.61        prime    replayed       22857      35415    1.53  6.0e-16  bluestein
      4027  4027             prime    replayed       22842      37797    1.65  6.8e-16  bluestein
      4028  2^2.19.53        prime    replayed       22671      40399    1.77  6.4e-16  bluestein
      4029  3.17.79          prime    replayed       23193      36222    1.55  6.2e-16  bluestein
      4030  2.5.13.31        flat     replayed       11375      22842    2.00  6.3e-16  
      4031  29.139           prime    replayed       22643      41851    1.82  5.8e-16  bluestein
      4032  2^6.3^2.7        ztt      replayed        4242       5296    1.11  3.1e-16  
      4033  37.109           prime    replayed       23202      54579    2.35  7.8e-16  bluestein
      4034  2.2017           prime    replayed       22649      37333    1.60  7.4e-16  bluestein
      4035  3.5.269          prime    replayed       22680      30783    1.32  7.2e-16  bluestein
      4036  2^2.1009         prime    replayed       22612      40157    1.77  6.2e-16  bluestein
      4037  11.367           prime    replayed       22332      33477    1.20  6.9e-16  bluestein
      4038  2.3.673          prime    replayed       22307      33483    1.46  6.5e-16  bluestein
      4039  7.577            prime    replayed       22635      36430    1.60  6.4e-16  bluestein
      4040  2^3.5.101        prime    replayed       22623      29572    1.30  6.3e-16  bluestein
      4041  3^2.449          prime    replayed       22302      39759    1.75  6.6e-16  bluestein
      4042  2.43.47          flat     raced          13338      49919    3.72  2.8e-15  
      4043  13.311           prime    replayed       22325      36600    1.56  6.1e-16  bluestein
      4044  2^2.3.337        prime    replayed       22276      34702    1.52  6.3e-16  bluestein
      4045  5.809            prime    replayed       22677      34200    1.48  7.3e-16  bluestein
      4046  2.7.17^2         flat     replayed       10238      24516    2.39  9.0e-16  
      4047  3.19.71          prime    replayed       22294      38600    1.73  7.3e-16  bluestein
      4048  2^4.11.23        chain3   replayed        6352      18451    2.86  6.3e-16  
      4049  4049             prime    replayed       22621      37799    1.65  6.7e-16  bluestein
      4050  2.3^4.5^2        chain3   replayed        6609       6169    0.93  6.5e-16  
      4051  4051             prime    replayed       22671      30133    1.29  7.1e-16  bluestein
      4052  2^2.1013         prime    replayed       22283      40277    1.77  7.6e-16  bluestein
      4053  3.7.193          prime    replayed       22942      26218    1.13  6.0e-16  bluestein
      4054  2.2027           prime    replayed       22849      29964    1.30  7.9e-16  bluestein
      4055  5.811            prime    replayed       22295      41315    1.50  6.4e-16  bluestein
      4056  2^3.3.13^2       flat     replayed        7098       8402    1.17  7.7e-16  
      4057  4057             prime    replayed       22222      30182    1.31  6.5e-16  bluestein
      4058  2.2029           prime    replayed       22256      30042    1.33  6.8e-16  bluestein
      4059  3^2.11.41        chain3   replayed       17812      27408    1.30  5.4e-16  
      4060  2^2.5.7.29       flat     replayed        8927      20387    2.26  6.6e-16  
      4061  31.131           prime    replayed       22340      46040    2.01  6.8e-16  bluestein
      4062  2.3.677          prime    replayed       23236      32013    1.37  6.7e-16  bluestein
      4063  17.239           prime    replayed       23170      37181    1.60  6.1e-16  bluestein
      4064  2^5.127          prime    replayed       22303      23741    1.03  6.0e-16  bluestein
      4065  3.5.271          prime    replayed       22347      34865    1.53  6.4e-16  bluestein
      4066  2.19.107         prime    replayed       22334      38864    1.69  7.1e-16  bluestein
      4067  7^2.83           prime    replayed       22290      38865    1.34  5.3e-16  flips differ 1.30x bluestein
      4068  2^2.3^2.113      prime    replayed       23322      34717    1.49  6.5e-16  bluestein
      4069  13.313           prime    replayed       22340      38353    1.71  6.6e-16  bluestein
      4070  2.5.11.37        chain3   replayed       11962      26340    1.51  5.0e-16  flips differ 1.46x
      4071  3.23.59          prime    replayed       23217      39487    1.70  5.4e-16  bluestein
      4072  2^3.509          prime    replayed       22242      27261    1.21  5.9e-16  bluestein
      4073  4073             prime    replayed       22553      30028    1.32  6.9e-16  bluestein
      4074  2.3.7.97         prime    replayed       23258      25532    1.09  6.1e-16  bluestein
      4075  5^2.163          prime    replayed       22349      30062    1.34  6.1e-16  bluestein
      4076  2^2.1019         prime    replayed       22315      26442    1.15  5.7e-16  bluestein
      4077  3^3.151          prime    replayed       22686      27651    1.19  6.3e-16  bluestein
      4078  2.2039           prime    replayed       22648      29894    1.30  5.3e-16  bluestein
      4079  4079             prime    replayed       22680      32547    1.29  6.1e-16  bluestein
      4080  2^4.3.5.17       chain3   replayed        5816      14883    2.55  5.0e-16  
      4081  7.11.53          prime    replayed       22291      33059    1.45  5.2e-16  bluestein
      4082  2.13.157         prime    replayed       22576      26450    1.17  7.0e-16  bluestein
      4083  3.1361           prime    replayed       23219      34807    1.50  6.6e-16  bluestein
      4084  2^2.1021         prime    replayed       22670      26364    1.15  5.9e-16  bluestein
      4085  5.19.43          flat     replayed       12937      38312    2.94  1.4e-15  
      4086  2.3^2.227        prime    replayed       22282      29009    1.26  5.7e-16  bluestein
      4087  61.67            prime    replayed       22599      71088    3.06  5.6e-16  bluestein
      4088  2^3.7.73         prime    replayed       22718      30839    1.31  7.4e-16  bluestein
      4089  3.29.47          chain3   replayed       16723      42919    2.00  6.3e-16  flips differ 1.29x
      4090  2.5.409          prime    replayed       22344      31117    1.35  7.2e-16  bluestein
      4091  4091             prime    replayed       22678      30416    1.33  6.2e-16  bluestein
      4092  2^2.3.11.31      chain3   replayed        7119      23469    3.28  6.1e-16  
      4093  4093             prime    replayed       22719      30038    1.30  6.1e-16  bluestein
      4094  2.23.89          prime    replayed       22360      41124    1.81  6.2e-16  bluestein
      4095  3^2.5.7.13       chain3   replayed        7267       7830    1.07  5.5e-16  
      4096  2^12             ztt      replayed        3560       4931    1.37  3.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     2706    297    868   0.79   1.15   1.63    1.14
 chain3     632     25    122   0.91   1.75   3.10    1.68
 flat       447     21     56   0.92   2.35   3.45    2.12
 2p         262      6     40   0.93   2.19   3.91    2.00
 ztt         25      1      4   0.98   1.19   1.29    1.13
 mono        23      6     11   0.73   1.05   2.39    1.25
 ALL       4095    356   1101   0.81   1.24   2.72    1.35
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   6   0.96      3      1
 8..31                 24   0.90     15      7
 32..127               96   1.49     14      1
 128..511             384   1.45     47      8
 512..2047           1536   1.25    384    116
 2048..4096          2049   1.19    638    223
```


## by family
```
 family                                       cells median   <1.0  gmean
 composite with a prime >= 53 (prime cell)     2157   1.12    781   1.10
 chain3                                         632   1.75    122   1.68
 flat                                           447   2.35     56   2.12
 prime N, bluestein                             402   1.28     86   1.19
 2p                                             257   2.23     39   2.03
 prime N, rader                                 147   1.71      1   1.75
 ztt                                             21   1.19      3   1.13
 mono                                            20   0.97     10   1.29
 pow2                                            12   1.05      3   1.02
```


flip agreement: our two readings more than 25% apart at 242 of 4095 cells.

worst 10: 2101 (prime 0.41), 2118 (prime 0.46), 1149 (prime 0.50), 2276 (prime 0.52), 2131 (prime 0.54), 2270 (prime 0.56), 2575 (prime 0.57), 2163 (prime 0.57), 1233 (prime 0.57), 2313 (prime 0.57)
best 5: 779 (2p 4.69), 731 (2p 4.63), 172 (2p 4.55), 527 (2p 4.50), 2021 (flat 4.50)
