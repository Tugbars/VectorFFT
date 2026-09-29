# gauntlet report

run: `oddtail_1d_new_2026-09-29_0844`  contract file suffix: `(oop, T=1)`  cells: 508 listed, 508 benched, comparator: MKL

control cell: 14 readings, 1.066..1.135 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         3  3                mono     replayed           4         11    3.05  1.5e-16  
         5  5                mono     replayed           5         12    2.23  7.5e-17  
         7  7                mono     replayed           7         12    1.88  2.6e-16  
         9  3^2              mono     replayed           8         13    1.56  4.2e-16  
        11  11               mono     replayed          11         14    1.27  1.2e-16  
        13  13               mono     replayed          15         15    1.01  5.0e-16  
        15  3.5              2p       replayed          15         14    0.94  3.3e-16  
        17  17               mono     replayed          19         40    2.10  2.0e-16  
        19  19               mono     replayed          23         45    1.97  4.6e-16  
        21  3.7              2p       replayed          17         25    1.46  3.3e-16  
        23  23               mono     replayed          30         69    2.28  2.5e-16  
        25  5^2              2p       replayed          19         26    1.31  1.8e-16  
        27  3^3              flat     replayed          28         32    1.15  3.8e-16  
        29  29               mono     replayed          48        107    2.20  2.8e-16  
        31  31               mono     replayed          59        121    2.07  3.3e-16  
        33  3.11             2p       replayed          26         36    1.39  2.7e-16  
        35  5.7              2p       replayed          25         36    1.42  2.6e-16  
        37  37               mono     replayed          75        185    2.45  4.8e-16  
        39  3.13             2p       replayed          33         44    1.34  6.4e-16  
        41  41               mono     replayed          93        242    2.60  3.7e-16  
        45  3^2.5            2p       replayed          32         49    1.49  5.0e-16  
        47  47               mono     replayed         117        328    2.80  4.0e-16  
        49  7^2              2p       replayed          36         47    1.31  3.2e-16  
        51  3.17             2p       replayed          53        114    2.13  4.1e-16  
        55  5.11             2p       replayed          43         54    1.27  2.8e-16  
        57  3.19             2p       replayed          64        136    2.10  4.1e-16  
        63  3^2.7            2p       replayed          47         68    1.44  4.2e-16  
        65  5.13             2p       replayed          53         67    1.26  6.3e-16  
        69  3.23             2p       replayed          93        198    1.88  4.1e-16  
        75  3.5^2            2p       replayed          58         78    1.35  4.4e-16  
        77  7.11             2p       replayed          60         75    1.24  3.6e-16  
        81  3^4              2p       replayed          60         86    1.43  4.3e-16  
        85  5.17             2p       replayed          82        190    2.25  4.7e-16  
        87  3.29             flat     replayed         149        332    2.05  6.6e-16  
        91  7.13             2p       replayed          75         91    1.21  5.0e-16  
        93  3.31             flat     replayed         166        376    2.09  4.4e-16  
        95  5.19             2p       replayed          98        228    2.21  4.4e-16  
        99  3^2.11           2p       replayed          77        112    1.44  4.7e-16  
       105  3.5.7            2p       replayed          85        113    1.33  5.0e-16  
       115  5.23             2p       replayed         156        340    1.52  4.6e-16  flips differ 1.44x
       117  3^2.13           2p       replayed          98        139    1.40  5.2e-16  
       119  7.17             2p       replayed         112        286    2.46  3.7e-16  
       121  11^2             2p       replayed         101        128    1.26  3.7e-16  
       125  5^3              2p       replayed         103        134    1.29  4.8e-16  
       129  3.43             flat     replayed         245        803    3.27  4.1e-16  
       133  7.19             2p       replayed         148        327    2.20  4.5e-16  
       135  3^3.5            2p       replayed         109        156    1.43  3.2e-16  
       141  3.47             flat     replayed         285        985    3.45  5.4e-16  
       143  11.13            2p       replayed         123        149    1.21  4.8e-16  
       145  5.29             2p       replayed         190        556    2.92  4.4e-16  
       147  3.7^2            chain3   replayed         135        158    1.17  4.0e-16  
       153  3^2.17           2p       replayed         144        372    2.54  5.6e-16  
       155  5.31             2p       replayed         335        633    1.87  4.3e-16  
       161  7.23             2p       replayed         180        477    2.44  4.5e-16  
       165  3.5.11           2p       replayed         144        189    1.30  4.0e-16  
       169  13^2             2p       replayed         158        189    1.17  7.3e-16  
       171  3^2.19           2p       replayed         183        442    2.24  6.3e-16  
       175  5^2.7            chain3   replayed         152        219    1.30  3.7e-16  
       185  5.37             2p       replayed         468        957    1.67  3.2e-16  
       187  11.17            flat     replayed         244        451    1.83  6.5e-16  
       189  3^3.7            chain3   replayed         168        228    1.36  4.4e-16  
       195  3.5.13           chain3   replayed         175        237    1.34  7.1e-16  
       203  7.29             2p       replayed         261        783    2.99  4.5e-16  
       205  5.41             2p       replayed         329       1231    3.73  5.7e-16  
       207  3^2.23           2p       replayed         269        636    2.35  5.0e-16  
       209  11.19            2p       replayed         222        526    2.36  5.1e-16  
       215  5.43             flat     replayed         389       1367    3.44  5.2e-16  
       217  7.31             2p       replayed         356        905    1.86  5.4e-16  flips differ 1.34x
       221  13.17            2p       replayed         227        549    2.37  5.5e-16  
       225  3^2.5^2          2p       replayed         200        268    1.32  4.8e-16  
       231  3.7.11           chain3   replayed         215        277    1.28  3.5e-16  
       235  5.47             2p       replayed         877       1668    1.62  3.4e-16  
       243  3^5              2p       replayed         217        297    1.24  6.5e-16  
       245  5.7^2            chain3   replayed         229        282    1.23  3.0e-16  
       247  13.19            2p       replayed         268        648    2.38  5.2e-16  
       253  11.23            2p       replayed         294        764    2.60  7.1e-16  
       255  3.5.17           2p       replayed         253        640    2.50  6.1e-16  
       259  7.37             2p       replayed         376       1343    3.57  4.1e-16  
       261  3^2.29           2p       replayed         334       1065    3.17  4.2e-16  
       273  3.7.13           2p       replayed         255        334    1.29  6.8e-16  
       275  5^2.11           chain3   replayed         250        333    1.33  3.4e-16  
       279  3^2.31           2p       replayed         465       1195    2.52  4.3e-16  
       285  3.5.19           2p       replayed         330        759    2.29  4.4e-16  
       287  7.41             2p       replayed         451       1733    3.85  3.7e-16  
       289  17^2             2p       replayed         353       1115    3.15  7.9e-16  
       297  3^3.11           chain3   replayed         287        397    1.37  5.2e-16  
       299  13.23            2p       replayed         357        939    2.60  5.0e-16  
       301  7.43             flat     replayed         554       1909    3.44  4.6e-16  
       315  3^2.5.7          chain3   replayed         292        396    1.36  7.9e-16  
       319  11.29            2p       replayed         422       1259    2.93  5.6e-16  
       323  17.19            2p       replayed         405       1290    3.12  6.8e-16  
       325  5^2.13           2p       replayed         322        409    1.16  7.5e-16  
       329  7.47             flat     replayed         636       2344    3.68  4.3e-16  
       333  3^2.37           2p       replayed         768       1779    1.76  5.9e-16  flips differ 1.32x
       341  11.31            flat     replayed         625       1425    2.03  6.8e-16  
       343  7^3              chain3   replayed         346        401    1.16  4.9e-16  
       345  3.5.23           2p       replayed         550       1119    1.60  5.9e-16  flips differ 1.27x
       351  3^3.13           chain3   replayed         350        481    1.36  5.7e-16  
       357  3.7.17           chain3   replayed         389        918    2.35  5.6e-16  
       361  19^2             2p       replayed         504       1537    2.12  5.0e-16  flips differ 1.44x
       363  3.11^2           chain3   replayed         373        459    1.23  3.3e-16  
       369  3^2.41           2p       replayed         996       2281    1.61  5.4e-16  flips differ 1.42x
       375  3.5^3            chain3   replayed         349        475    1.35  6.8e-16  
       377  13.29            2p       replayed         619       1521    1.88  6.2e-16  flips differ 1.31x
       385  5.7.11           chain3   replayed         377        490    1.29  5.6e-16  
       387  3^2.43           2p       replayed         626       2512    3.98  5.4e-16  
       391  17.23            2p       replayed         528       1780    3.23  6.7e-16  
       399  3.7.19           chain3   replayed         467       1087    2.32  6.2e-16  
       403  13.31            2p       replayed         704       1724    2.45  6.4e-16  
       405  3^4.5            chain3   replayed         383        558    1.44  5.2e-16  
       407  11.37            2p       replayed         604       2175    3.54  4.1e-16  
       423  3^2.47           flat     replayed         821       3069    3.71  7.0e-16  
       425  5^2.17           2p       replayed         455       1082    2.22  6.4e-16  
       429  3.11.13          chain3   replayed         463        555    1.17  5.2e-16  
       435  3.5.29           2p       replayed         585       1784    3.01  5.5e-16  
       437  19.23            2p       replayed         649       2113    3.24  5.5e-16  
       441  3^2.7^2          chain3   replayed         447        627    1.29  4.9e-16  
       451  11.41            2p       replayed        1188       2760    1.71  5.0e-16  flips differ 1.35x
       455  5.7.13           chain3   replayed         462        586    1.25  6.0e-16  
       459  3^3.17           chain3   replayed         525       1207    1.74  5.1e-16  flips differ 1.32x
       465  3.5.31           2p       replayed         799       2016    1.80  5.2e-16  flips differ 1.40x
       473  11.43            2p       replayed         783       3071    3.87  6.0e-16  
       475  5^2.19           2p       replayed         530       1286    2.42  5.0e-16  
       481  13.37            2p       replayed         725       2578    3.55  5.5e-16  
       483  3.7.23           2p       replayed         699       1590    1.31  5.1e-16  flips differ 1.74x
       493  17.29            2p       replayed         743       2684    3.46  5.0e-16  
       495  3^2.5.11         flat     replayed         591        671    1.12  5.8e-16  
       507  3.13^2           chain3   replayed         594        720    1.13  6.2e-16  
       513  3^3.19           2p       replayed         572       1472    1.94  5.4e-16  flips differ 1.33x
       517  11.47            flat     replayed        1029       3709    3.55  6.4e-16  
       525  3.5^2.7          2p       replayed         553        720    1.16  6.4e-16  
       527  17.31            2p       replayed         816       2991    3.39  7.5e-16  
       529  23^2             2p       replayed         873       2793    2.10  6.5e-16  flips differ 1.52x
       533  13.41            2p       replayed         866       3306    3.80  6.4e-16  
       539  7^2.11           flat     replayed         674        725    1.07  5.4e-16  
       551  19.29            2p       replayed         906       3133    3.21  4.0e-16  
       555  3.5.37           2p       replayed         833       3016    3.61  6.0e-16  
       559  13.43            2p       replayed         943       3646    3.83  5.6e-16  
       561  3.11.17          flat     replayed         824       1515    1.80  1.2e-15  
       567  3^4.7            2p       replayed         597        903    1.26  6.4e-16  flips differ 1.28x
       575  5^2.23           2p       replayed         983       1891    1.30  5.8e-16  flips differ 1.48x
       585  3^2.5.13         chain3   replayed         771        866    1.09  5.6e-16  
       589  19.31            2p       replayed         951       3497    3.56  5.1e-16  
       595  5.7.17           chain3   replayed         873       1577    1.76  6.6e-16  
       605  5.11^2           flat     replayed         751        827    1.09  6.1e-16  
       609  3.7.29           2p       replayed        1040       2519    2.38  5.8e-16  
       611  13.47            flat     replayed        1290       4449    2.80  7.8e-16  
       615  3.5.41           2p       replayed        1006       3860    3.59  6.3e-16  
       621  3^3.23           2p       replayed         779       2092    2.64  6.7e-16  
       625  5^4              2p       replayed         728        871    1.03  4.1e-16  
       627  3.11.19          flat     replayed         975       1754    1.73  6.9e-16  
       629  17.37            2p       replayed        1530       4243    2.08  8.6e-16  flips differ 1.34x
       637  7^2.13           flat     replayed         909        873    0.89  6.3e-16  
       645  3.5.43           2p       replayed        1078       4241    3.88  5.5e-16  
       651  3.7.31           2p       replayed        1780       2851    1.50  5.2e-16  
       663  3.13.17          flat     replayed         994       1823    1.56  8.9e-16  
       665  5.7.19           flat     replayed         937       1869    1.97  5.1e-16  
       667  23.29            2p       replayed        1624       4115    2.19  6.1e-16  
       675  3^3.5^2          2p       replayed         750       1015    1.30  5.9e-16  
       693  3^2.7.11         flat     replayed         882       1001    1.12  7.0e-16  
       697  17.41            2p       replayed        1248       5303    4.03  8.6e-16  
       703  19.37            2p       replayed        1228       4923    2.78  4.6e-16  flips differ 1.44x
       705  3.5.47           2p       replayed        1258       5182    4.11  6.6e-16  
       713  23.31            2p       replayed        1486       4574    2.37  9.4e-16  flips differ 1.30x
       715  5.11.13          flat     replayed         961       1003    1.00  6.4e-16  
       725  5^2.29           flat     replayed        1439       3038    2.10  5.8e-16  
       729  3^6              2p       replayed         831       1090    1.31  5.1e-16  
       731  17.43            2p       replayed        1359       5780    3.89  6.0e-16  
       735  3.5.7^2          flat     replayed        1004       1038    0.95  6.1e-16  
       741  3.13.19          chain3   replayed        1568       2147    1.10  6.5e-16  
       759  3.11.23          flat     replayed        1364       2542    1.29  6.8e-16  flips differ 1.45x
       765  3^2.5.17         flat     replayed        1056       2128    1.98  6.0e-16  
       775  5^2.31           2p       replayed        1120       3428    3.06  6.3e-16  
       777  3.7.37           2p       replayed        1194       4253    3.53  5.5e-16  
       779  19.41            2p       replayed        1468       6109    2.79  4.7e-16  flips differ 1.49x
       783  3^3.29           flat     replayed        1483       3398    2.01  4.8e-16  
       799  17.47            2p       replayed        1586       6924    4.27  5.6e-16  
       805  5.7.23           flat     replayed        1269       2695    1.99  6.6e-16  
       817  19.43            flat     replayed        1674       6676    3.82  8.9e-16  
       819  3^2.7.13         flat     replayed        1110       1225    1.09  7.7e-16  
       825  3.5^2.11         chain3   replayed        1198       1218    1.01  4.4e-16  
       833  7^2.17           flat     replayed        1254       2234    1.74  7.7e-16  
       837  3^3.31           flat     replayed        1842       3818    1.82  7.1e-16  
       841  29^2             flat     replayed        1887       6073    3.06  1.2e-15  
       845  5.13^2           flat     replayed        1270       1266    0.94  9.8e-16  
       847  7.11^2           flat     replayed        1135       1197    0.99  4.3e-16  
       851  23.37            2p       replayed        1679       6380    3.22  6.1e-16  
       855  3^2.5.19         flat     replayed        1326       2549    1.88  8.2e-16  
       861  3.7.41           2p       replayed        1415       5427    3.26  4.6e-16  
       867  3.17^2           flat     replayed        1425       3524    2.38  1.1e-15  
       875  5^3.7            flat     replayed        1115       1290    1.15  4.5e-16  
       891  3^4.11           chain3   replayed        1307       1419    1.08  6.0e-16  
       893  19.47            flat     replayed        1923       8023    4.15  9.9e-16  
       897  3.13.23          flat     replayed        1617       3463    1.87  8.4e-16  
       899  29.31            2p       replayed        1939       6642    2.67  5.6e-16  flips differ 1.29x
       903  3.7.43           flat     replayed        1978       5971    2.89  1.9e-15  
       925  5^2.37           2p       replayed        2411       5090    1.66  4.4e-16  flips differ 1.27x
       931  7^2.19           flat     replayed        1497       2642    1.76  7.7e-16  
       935  5.11.17          flat     replayed        1469       2546    1.73  6.2e-16  
       943  23.41            2p       replayed        2419       7929    3.17  5.5e-16  
       945  3^3.5.7          chain3   replayed        1384       1455    1.05  6.1e-16  
       957  3.11.29          chain3   replayed        1868       4102    2.17  5.4e-16  
       961  31^2             flat     replayed        2169       7374    2.91  9.3e-16  
       969  3.17.19          flat     replayed        1754       4112    2.24  8.8e-16  
       975  3.5^2.13         flat     replayed        1337       1527    1.08  6.2e-16  
       987  3.7.47           2p       replayed        1823       7449    3.44  5.1e-16  
       989  23.43            flat     replayed        2165       8665    3.98  1.1e-15  
       999  3^3.37           2p       replayed        2515       5712    1.50  6.4e-16  flips differ 1.50x
      1001  7.11.13          flat     replayed        1495       1476    0.59  7.1e-16  flips differ 1.69x
      1015  5.7.29           flat     replayed        1971       4361    2.02  5.8e-16  
      1023  3.11.31          chain3   replayed        2033       4630    2.00  6.3e-16  
      1025  5^2.41           2p       replayed        1763       6565    3.58  5.0e-16  
      1029  3.7^3            flat     replayed        1497       1548    1.03  4.4e-16  
      1035  3^2.5.23         flat     replayed        1795       3736    2.00  6.7e-16  
      1045  5.11.19          flat     replayed        1741       3094    1.71  6.3e-16  
      1053  3^4.13           chain3   replayed        1644       1792    1.08  8.0e-16  
      1071  3^2.7.17         flat     replayed        1694       3111    1.77  5.7e-16  
      1073  29.37            2p       replayed        2538       9167    2.46  4.9e-16  flips differ 1.47x
      1075  5^2.43           2p       replayed        1911       7215    3.35  5.6e-16  
      1081  23.47            2p       replayed        3650      10279    2.79  5.5e-16  
      1083  3.19^2           chain3   replayed        2102       4939    2.29  4.7e-16  
      1085  5.7.31           chain3   replayed        3089       5033    1.59  5.5e-16  
      1089  3^2.11^2         chain3   replayed        1705       1731    1.01  6.2e-16  
      1105  5.13.17          flat     replayed        1829       3192    1.74  8.0e-16  
      1107  3^3.41           2p       replayed        1997       7208    3.61  5.2e-16  
      1125  3^2.5^3          flat     replayed        1576       1831    1.15  7.8e-16  
      1127  7^2.23           chain3   replayed        2048       3923    1.86  6.4e-16  
      1131  3.13.29          chain3   replayed        2292       4962    2.16  7.4e-16  
      1147  31.37            flat     replayed        2569      10131    3.91  8.4e-16  
      1155  3.5.7.11         chain3   replayed        1698       1828    1.05  4.2e-16  
      1161  3^3.43           2p       replayed        2223       8037    3.31  5.5e-16  
      1173  3.17.23          flat     replayed        2403       5693    2.21  1.1e-15  
      1175  5^2.47           flat     replayed        2503       8815    3.51  4.6e-16  
      1183  7.13^2           chain3   replayed        1897       1910    0.81  7.3e-16  
      1189  29.41            2p       replayed        3090      11159    3.58  5.0e-16  
      1197  3^2.7.19         flat     replayed        2330       3653    1.56  1.1e-15  
      1209  3.13.31          chain3   replayed        2543       5608    2.18  5.6e-16  
      1215  3^5.5            chain3   replayed        1763       2142    1.19  9.3e-16  
      1221  3.11.37          chain3   replayed        2633       6880    2.58  4.9e-16  
      1225  5^2.7^2          flat     replayed        1815       1954    0.82  4.5e-16  flips differ 1.32x
      1235  5.13.19          flat     replayed        2161       3760    1.58  7.9e-16  
      1247  29.43            flat     replayed        3577      12098    3.31  2.8e-15  
      1265  5.11.23          flat     replayed        2519       4441    1.31  9.0e-16  flips differ 1.35x
      1269  3^3.47           flat     replayed        3143       9635    2.97  7.1e-16  
      1271  31.41            flat     replayed        2964      12303    4.12  1.3e-15  
      1275  3.5^2.17         flat     replayed        2218       3689    1.58  7.2e-16  
      1287  3^2.11.13        chain3   replayed        2072       2145    1.03  5.9e-16  
      1295  5.7.37           flat     replayed        2496       7314    2.92  7.7e-16  
      1305  3^2.5.29         chain3   replayed        2555       5788    2.22  6.7e-16  
      1309  7.11.17          flat     replayed        2353       3739    1.55  1.1e-15  
      1311  3.19.23          chain3   replayed        3039       6753    1.78  6.0e-16  
      1323  3^3.7^2          flat     replayed        2063       2159    1.01  8.3e-16  
      1331  11^3             chain3   replayed        2150       2075    0.96  4.5e-16  
      1333  31.43            2p       replayed        3986      13318    2.67  6.7e-16  
      1353  3.11.41          chain3   replayed        3124       8730    2.45  4.9e-16  
      1363  29.47            2p       replayed        4772      14268    2.09  4.5e-16  flips differ 1.43x
      1365  3.5.7.13         chain3   replayed        2301       2277    0.98  6.0e-16  
      1369  37^2             2p       replayed        4160      13554    2.47  5.1e-16  flips differ 1.32x
      1375  5^3.11           chain3   replayed        2547       2260    0.88  4.2e-16  
      1377  3^4.17           chain3   replayed        2847       4101    1.33  8.2e-16  
      1395  3^2.5.31         flat     replayed        2806       6521    2.26  6.6e-16  
      1419  3.11.43          flat     replayed        3372       9576    2.81  6.0e-16  
      1421  7^2.29           chain3   replayed        2905       6210    2.11  5.7e-16  
      1425  3.5^2.19         flat     replayed        2216       4345    1.84  5.1e-16  
      1435  5.7.41           chain3   replayed        3490       9267    2.64  5.8e-16  
      1443  3.13.37          flat     replayed        3182       8284    2.21  6.5e-16  
      1445  5.17^2           chain3   replayed        2704       6174    2.24  6.4e-16  
      1449  3^2.7.23         flat     replayed        2824       5227    1.42  5.4e-16  flips differ 1.30x
      1457  31.47            2p       replayed        6965      15690    1.97  5.9e-16  
      1463  7.11.19          chain3   replayed        2573       4377    1.24  4.8e-16  flips differ 1.37x
      1479  3.17.29          flat     replayed        3363       8477    2.48  1.2e-15  
      1485  3^3.5.11         chain3   replayed        2260       2565    1.07  5.3e-16  
      1495  5.13.23          chain3   replayed        2824       5367    1.86  6.5e-16  
      1505  5.7.43           chain3   replayed        5348      10196    1.33  6.1e-16  flips differ 1.43x
      1517  37.41            2p       replayed        4882      16349    2.53  8.5e-16  flips differ 1.32x
      1519  7^2.31           flat     replayed        3300       6954    2.09  7.3e-16  
      1521  3^2.13^2         chain3   replayed        2577       2847    0.93  7.1e-16  
      1539  3^4.19           chain3   replayed        2834       4965    1.35  7.1e-16  flips differ 1.29x
      1547  7.13.17          flat     replayed        2848       4639    1.59  9.5e-16  
      1551  3.11.47          chain3   replayed        5807      11719    2.00  5.0e-16  
      1573  11^2.13          chain3   replayed        2610       2566    0.95  5.6e-16  
      1575  3^2.5^2.7        flat     replayed        2411       2750    0.88  6.6e-16  flips differ 1.29x
      1581  3.17.31          chain3   replayed        4786       9481    1.58  6.2e-16  flips differ 1.26x
      1587  3.23^2           flat     replayed        3598       9188    2.31  1.2e-15  
      1591  37.43            2p       replayed        5505      17961    3.21  5.8e-16  
      1595  5.11.29          flat     replayed        3608       6981    1.91  6.7e-16  
      1599  3.13.41          chain3   replayed        3659      10504    2.85  5.7e-16  
      1615  5.17.19          chain3   replayed        3428       7082    2.04  6.1e-16  
      1617  3.7^2.11         chain3   replayed        3314       2885    0.82  5.5e-16  
      1625  5^3.13           flat     replayed        2609       2798    1.00  6.6e-16  
      1645  5.7.47           flat     replayed        3781      12413    3.25  5.6e-16  
      1653  3.19.29          chain3   replayed        3986       9936    2.32  5.4e-16  
      1665  3^2.5.37         chain3   replayed        4900       9651    1.96  5.6e-16  
      1677  3.13.43          chain3   replayed        4109      11538    2.81  6.8e-16  
      1681  41^2             2p       replayed        5645      19551    2.63  5.1e-16  flips differ 1.32x
      1683  3^2.11.17        chain3   replayed        3037       5029    1.65  7.8e-16  
      1701  3^5.7            chain3   replayed        2570       3159    1.17  4.7e-16  
      1705  5.11.31          chain3   replayed        5016       7891    1.19  5.0e-16  flips differ 1.32x
      1715  5.7^3            flat     replayed        2750       2832    1.02  4.9e-16  
      1725  3.5^2.23         chain3   replayed        3217       6297    1.95  6.9e-16  
      1729  7.13.19          chain3   replayed        3324       5640    1.53  6.4e-16  
      1739  37.47            flat     replayed        4810      20580    3.69  1.9e-15  
      1755  3^3.5.13         chain3   replayed        2855       3277    1.14  8.8e-16  
      1763  41.43            2p       replayed        6319      21073    2.22  6.0e-16  flips differ 1.50x
      1767  3.19.31          chain3   replayed        4509      11074    2.13  4.8e-16  
      1771  7.11.23          flat     replayed        3448       6479    1.82  8.1e-16  
      1785  3.5.7.17         chain3   replayed        2987       5341    1.78  6.5e-16  
      1805  5.19^2           flat     replayed        4278       8438    1.76  1.3e-15  
      1813  7^2.37           chain3   replayed        4312      10281    2.38  5.0e-16  
      1815  3.5.11^2         chain3   replayed        2903       3255    1.11  6.1e-16  
      1827  3^2.7.29         flat     replayed        3853       8197    2.06  6.0e-16  
      1833  3.13.47          flat     replayed        4632      14012    3.00  7.7e-16  
      1845  3^2.5.41         chain3   replayed        4323      12512    2.71  5.9e-16  
      1849  43^2             flat     replayed        4983      22685    4.39  1.6e-15  
      1859  11.13^2          chain3   replayed        3167       3311    0.82  7.3e-16  flips differ 1.29x
      1863  3^4.23           chain3   replayed        3376       6932    2.04  6.2e-16  
      1875  3.5^4            flat     replayed        3352       3382    1.00  6.3e-16  
      1881  3^2.11.19        chain3   replayed        3447       5936    1.54  6.5e-16  
      1885  5.13.29          chain3   replayed        5877       8489    1.40  6.9e-16  
      1887  3.17.37          chain3   replayed        4297      13354    2.97  6.2e-16  
      1911  3.7^2.13         chain3   replayed        3127       3430    1.07  5.8e-16  
      1925  5^2.7.11         flat     replayed        3026       3346    1.08  6.7e-16  
      1935  3^2.5.43         flat     replayed        4204      13458    3.11  6.2e-16  
      1953  3^2.7.31         flat     replayed        4355       9252    2.07  7.6e-16  
      1955  5.17.23          chain3   replayed        5000       9683    1.91  6.2e-16  
      1989  3^2.13.17        chain3   replayed        3628       6122    1.68  6.9e-16  
      1995  3.5.7.19         chain3   replayed        3529       6295    1.21  5.1e-16  flips differ 1.48x
      2001  3.23.29          chain3   replayed        6321      13083    1.84  6.4e-16  
      2009  7^2.41           chain3   replayed        6491      13042    1.52  4.7e-16  flips differ 1.32x
      2015  5.13.31          flat     replayed        4858       9517    1.73  6.5e-16  
      2023  7.17^2           chain3   replayed        3779       8591    2.15  7.8e-16  
      2025  3^4.5^2          chain3   replayed        3723       3874    1.01  4.9e-16  
      2035  5.11.37          chain3   replayed        6238      11640    1.41  5.0e-16  flips differ 1.32x
      2057  11^2.17          flat     replayed        3797       6079    1.52  9.5e-16  
      2079  3^3.7.11         chain3   replayed        3239       3871    1.05  7.6e-16  
      2091  3.17.41          chain3   replayed        5095      16550    3.19  6.6e-16  
      2093  7.13.23          flat     replayed        4115       7852    1.71  7.1e-16  
      2107  7^2.43           chain3   replayed        7165      14334    1.54  5.1e-16  flips differ 1.30x
      2109  3.19.37          flat     replayed        5135      15504    2.91  7.3e-16  
      2115  3^2.5.47         chain3   replayed       10745      16274    1.38  5.0e-16  
      2125  5^3.17           chain3   replayed        3749       6588    1.64  7.5e-16  
      2139  3.23.31          flat     replayed        5942      14486    2.29  9.3e-16  
      2145  3.5.11.13        chain3   replayed        3397       4005    1.17  5.4e-16  
      2175  3.5^2.29         chain3   replayed        4531       9875    2.09  5.0e-16  
      2185  5.19.23          chain3   replayed        4575      11439    2.48  5.1e-16  
      2187  3^7              chain3   replayed        4026       4512    0.96  6.0e-16  
      2193  3.17.43          chain3   replayed        7863      18078    2.29  7.4e-16  
      2197  13^3             flat     replayed        4305       4111    0.81  1.2e-15  
      2205  3^2.5.7^2        chain3   replayed        3523       4117    1.10  5.3e-16  
      2209  47^2             flat     replayed        6372      30535    4.68  1.9e-15  
      2223  3^2.13.19        flat     replayed        4431       7213    1.26  1.2e-15  flips differ 1.29x
      2233  7.11.29          chain3   replayed        4637       9859    1.70  6.3e-16  flips differ 1.25x
      2255  5.11.41          chain3   replayed        5249      14735    2.72  6.0e-16  
      2261  7.17.19          flat     replayed        4556       9973    2.09  9.9e-16  
      2275  5^2.7.13         chain3   replayed        4126       4520    1.01  5.9e-16  
      2277  3^2.11.23        chain3   replayed        4303       9127    1.97  6.3e-16  
      2295  3^3.5.17         chain3   replayed        4173       8135    1.84  6.9e-16  
      2299  11^2.19          flat     replayed        4851       7350    1.46  1.2e-15  
      2303  7^2.47           chain3   replayed        8271      17435    1.61  6.0e-16  flips differ 1.31x
      2325  3.5^2.31         chain3   replayed        6796      11127    1.15  5.7e-16  flips differ 1.43x
      2331  3^2.7.37         chain3   replayed        6910      13627    1.46  5.4e-16  flips differ 1.35x
      2337  3.19.41          chain3   replayed        6012      19092    2.38  7.5e-16  flips differ 1.33x
      2349  3^4.29           chain3   replayed        5596      11034    1.38  6.6e-16  flips differ 1.43x
      2365  5.11.43          chain3   replayed        8233      16265    1.35  5.6e-16  flips differ 1.46x
      2375  5^3.19           flat     replayed        4016       7523    1.82  6.8e-16  
      2387  7.11.31          chain3   replayed        8513      11145    1.25  4.7e-16  
      2397  3.17.47          flat     replayed        7053      21578    3.03  1.6e-15  
      2401  7^4              flat     replayed        4620       4142    0.84  6.8e-16  
      2405  5.13.37          chain3   replayed        5397      14101    2.61  6.3e-16  
      2415  3.5.7.23         chain3   replayed        5588       8959    1.59  6.2e-16  
      2431  11.13.17         chain3   replayed        4346       7325    1.65  6.0e-16  
      2451  3.19.43          flat     replayed        7049      20871    2.57  1.4e-15  
      2457  3^3.7.13         chain3   replayed        4410       4833    1.08  8.3e-16  
      2465  5.17.29          flat     replayed        6198      14394    1.91  1.6e-15  
      2475  3^2.5^2.11       chain3   replayed        4094       4763    1.15  7.2e-16  
      2499  3.7^2.17         chain3   replayed        4118       7557    1.79  6.6e-16  
      2511  3^4.31           flat     replayed        5730      12355    1.94  7.5e-16  
      2523  3.29^2           flat     replayed        6465      18947    2.81  1.4e-15  
      2527  7.19^2           chain3   replayed        5328      11823    2.20  5.3e-16  
      2535  3.5.13^2         chain3   replayed        4440       5102    0.91  7.3e-16  flips differ 1.27x
      2541  3.7.11^2         flat     replayed        5304       4881    0.90  7.6e-16  
      2553  3.23.37          flat     replayed        6733      20074    2.98  8.7e-16  
      2565  3^3.5.19         chain3   replayed        5043       8561    1.42  8.1e-16  
      2583  3^2.7.41         chain3   replayed        5934      17172    2.89  6.2e-16  
      2585  5.11.47          chain3   replayed       13852      19731    1.42  5.3e-16  
      2601  3^2.17^2         chain3   replayed        4711      11455    2.37  7.7e-16  
      2625  3.5^3.7          flat     replayed        4759       4956    1.04  9.2e-16  
      2635  5.17.31          flat     replayed        7257      16016    2.15  8.0e-16  
      2639  7.13.29          chain3   replayed        5382      11933    2.21  7.5e-16  
      2645  5.23^2           flat     replayed        6271      15128    2.39  1.7e-15  
      2665  5.13.41          chain3   replayed        6185      17754    2.87  7.8e-16  
      2673  3^5.11           chain3   replayed        5130       5323    1.00  7.9e-16  
      2679  3.19.47          chain3   replayed       13552      24849    1.56  5.9e-16  
      2691  3^2.13.23        chain3   replayed        5054      10219    1.93  6.2e-16  
      2695  5.7^2.11         flat     replayed        4953       5020    0.78  5.5e-16  flips differ 1.30x
      2697  3.29.31          flat     replayed        7094      20920    2.94  2.8e-15  
      2709  3^2.7.43         chain3   replayed        9449      18868    1.16  6.0e-16  flips differ 1.72x
      2717  11.13.19         flat     replayed        5709       8596    1.39  8.5e-16  
      2737  7.17.23          flat     replayed        5891      13663    2.28  1.4e-15  
      2755  5.19.29          chain3   replayed        7228      16860    2.28  6.8e-16  
      2775  3.5^2.37         chain3   replayed        8482      16322    1.50  7.1e-16  flips differ 1.29x
      2783  11^2.23          flat     replayed        6104      10189    1.64  1.4e-15  
      2793  3.7^2.19         chain3   replayed        4927       9152    1.79  6.3e-16  
      2795  5.13.43          chain3   replayed        6739      19561    2.82  5.9e-16  
      2805  3.5.11.17        chain3   replayed        5088       8608    1.68  6.0e-16  
      2821  7.13.31          chain3   replayed        6050      13434    2.20  6.5e-16  
      2829  3.23.41          flat     replayed        7786      24553    3.12  8.5e-16  
      2835  3^4.5.7          chain3   replayed        4888       5585    1.12  7.2e-16  
      2849  7.11.37          flat     replayed        6598      16384    2.48  6.0e-16  
      2871  3^2.11.29        flat     replayed        6105      13123    2.13  8.2e-16  
      2873  13^2.17          chain3   replayed        5563       8889    1.34  8.2e-16  
      2875  5^3.23           chain3   replayed        5986      10775    1.68  6.7e-16  
      2883  3.31^2           flat     replayed        7546      23112    3.04  1.9e-15  
      2907  3^2.17.19        flat     replayed        8822      13342    1.50  1.6e-15  
      2925  3^2.5^2.13       chain3   replayed        5147       5824    1.04  6.6e-16  
      2945  5.19.31          chain3   replayed        6773      18676    2.09  5.4e-16  flips differ 1.32x
      2961  3^2.7.47         flat     replayed        6880      22911    3.30  6.3e-16  
      2967  3.23.43          chain3   replayed       11239      26786    1.85  5.6e-16  flips differ 1.29x
      2975  5^2.7.17         flat     replayed        5681       9176    1.53  6.8e-16  
      2997  3^4.37           chain3   replayed        6899      17919    2.56  5.1e-16  
      3003  3.7.11.13        chain3   replayed        5259       5847    1.06  6.6e-16  
      3025  5^2.11^2         flat     replayed        5973       5983    0.99  8.7e-16  
      3045  3.5.7.29         flat     replayed        7261      13972    1.91  7.1e-16  
      3055  5.13.47          chain3   replayed       11608      23667    2.04  6.2e-16  
      3059  7.19.23          flat     replayed        7068      16093    2.13  7.0e-16  
      3069  3^2.11.31        chain3   replayed       11239      14835    1.23  9.8e-16  
      3075  3.5^2.41         chain3   replayed        9912      20554    1.59  6.2e-16  flips differ 1.30x
      3087  3^2.7^3          chain3   replayed        5211       6038    1.07  5.0e-16  
      3105  3^3.5.23         chain3   replayed        8543      11931    1.37  5.3e-16  
      3125  5^5              flat     replayed        5780       6036    0.80  8.4e-16  flips differ 1.31x
      3135  3.5.11.19        chain3   replayed        8167      10134    1.11  7.4e-16  
      3145  5.17.37          chain3   replayed        7397      22789    3.06  6.0e-16  
      3157  7.11.41          chain3   replayed        7650      20784    2.39  5.2e-16  
      3159  3^5.13           chain3   replayed        5701       6549    0.98  7.2e-16  
      3179  11.17^2          flat     replayed        6528      13749    1.99  1.1e-15  
      3185  5.7^2.13         flat     replayed        5666       6177    1.06  6.7e-16  
      3211  13^2.19          flat     replayed        6341      10464    1.64  1.0e-15  
      3213  3^3.7.17         chain3   replayed        5768      10206    1.77  9.3e-16  
      3219  3.29.37          flat     replayed        9273      28431    3.01  2.1e-15  
      3225  3.5^2.43         chain3   replayed       11263      22538    1.34  5.6e-16  flips differ 1.49x
      3243  3.23.47          chain3   replayed       16411      31697    1.65  5.8e-16  
      3249  3^2.19^2         flat     replayed        8542      15731    1.76  1.1e-15  
      3255  3.5.7.31         flat     replayed        7893      15703    1.98  6.8e-16  
      3267  3^3.11^2         flat     replayed        7084       6660    0.94  5.6e-16  
      3289  11.13.23         chain3   replayed        8313      12239    1.19  8.5e-16  
      3311  7.11.43          chain3   replayed       11507      22857    1.99  5.7e-16  
      3315  3.5.13.17        flat     replayed        6444      10484    1.39  1.3e-15  
      3321  3^4.41           chain3   replayed        8285      22611    2.61  6.2e-16  
      3325  5^2.7.19         chain3   replayed        6706      10813    1.60  5.0e-16  
      3335  5.23.29          chain3   replayed        9297      22065    1.86  6.0e-16  flips differ 1.28x
      3367  7.13.37          chain3   replayed        7674      19796    2.31  6.3e-16  
      3375  3^3.5^3          chain3   replayed        5143       6913    1.27  5.9e-16  
      3381  3.7^2.23         flat     replayed        7759      12669    1.61  7.7e-16  
      3393  3^2.13.29        flat     replayed        8696      15936    1.68  1.2e-15  
      3441  3.31.37          flat     replayed        9928      31389    3.02  2.3e-15  
      3451  7.17.29          chain3   replayed        8387      20253    1.84  7.1e-16  flips differ 1.31x
      3465  3^2.5.7.11       chain3   replayed        5519       6880    1.17  6.1e-16  
      3483  3^4.43           chain3   replayed       14048      24759    1.34  6.5e-16  flips differ 1.32x
      3485  5.17.41          chain3   replayed       11201      27985    2.48  5.7e-16  
      3509  11^2.29          chain3   replayed        7414      15861    2.11  6.4e-16  
      3515  5.19.37          chain3   replayed        9362      26311    2.68  6.3e-16  
      3519  3^2.17.23        flat     replayed        8090      18103    2.23  8.1e-16  
      3525  3.5^2.47         flat     replayed        8170      27358    2.46  5.6e-16  flips differ 1.36x
      3549  3.7.13^2         chain3   replayed        6358       7329    1.14  7.0e-16  
      3553  11.17.19         chain3   replayed        7021      16055    1.98  6.0e-16  
      3565  5.23.31          flat     replayed        8806      24495    2.64  1.7e-15  
      3567  3.29.41          flat     replayed       10681      34440    3.22  2.3e-15  
      3575  5^2.11.13        chain3   replayed        6430       7212    0.91  7.9e-16  
      3591  3^3.7.19         chain3   replayed        8461      12013    1.10  6.3e-16  flips differ 1.29x
      3619  7.11.47          chain3   replayed       13232      27757    1.58  4.7e-16  flips differ 1.33x
      3625  5^3.29           chain3   replayed        7959      16813    2.05  5.4e-16  
      3627  3^2.13.31        flat     replayed        9814      17940    1.74  1.0e-15  
      3645  3^6.5            chain3   replayed        7066       7763    1.09  6.6e-16  
      3655  5.17.43          chain3   replayed        9827      30442    2.97  5.3e-16  
      3663  3^2.11.37        flat     replayed        9099      21714    2.37  2.3e-15  
      3675  3.5^2.7^2        chain3   replayed        6499       7147    1.08  5.1e-16  
      3689  7.17.31          chain3   replayed       11559      22589    1.39  6.8e-16  flips differ 1.41x
      3703  7.23^2           chain3   replayed        8394      21390    2.54  6.2e-16  
      3705  3.5.13.19        chain3   replayed        6821      13616    1.81  6.1e-16  
      3731  7.13.41          chain3   replayed        8766      25829    2.85  6.8e-16  
      3741  3.29.43          flat     replayed       11190      37248    3.32  3.5e-15  
      3751  11^2.31          flat     replayed        9280      17804    1.62  8.8e-16  
      3757  13.17^2          chain3   replayed        7367      16710    2.25  8.5e-16  
      3773  7^3.11           flat     replayed        7223       7188    0.91  7.6e-16  
      3795  3.5.11.23        chain3   replayed        6999      15923    2.06  6.2e-16  
      3807  3^4.47           chain3   replayed       19784      30008    1.40  6.4e-16  
      3813  3.31.41          flat     replayed       11946      38254    3.13  2.5e-15  
      3825  3^2.5^2.17       chain3   replayed        6800      12471    1.78  8.6e-16  
      3857  7.19.29          chain3   replayed        8809      23850    2.44  6.5e-16  
      3861  3^3.11.13        flat     replayed        8321       8160    0.94  9.9e-16  
      3875  5^3.31           chain3   replayed        8848      18926    2.11  7.1e-16  
      3885  3.5.7.37         chain3   replayed        8633      23030    2.62  5.0e-16  
      3887  13^2.23          flat     replayed        9309      14970    1.08  9.1e-16  flips differ 1.48x
      3895  5.19.41          chain3   replayed       10455      32487    2.01  5.1e-16  flips differ 1.56x
      3913  7.13.43          chain3   replayed       14074      28715    1.96  7.6e-16  
      3915  3^3.5.29         flat     replayed        9638      18498    1.92  6.5e-16  
      3927  3.7.11.17        chain3   replayed        7630      12207    1.59  6.2e-16  
      3933  3^2.19.23        chain3   replayed       10757      21507    1.95  7.4e-16  
      3969  3^4.7^2          chain3   replayed        6744       8282    1.14  8.0e-16  
      3971  11.19^2          chain3   replayed        8062      18972    1.64  6.4e-16  flips differ 1.43x
      3993  3.11^3           flat     replayed        8149       7696    0.85  8.3e-16  
      3995  5.17.47          chain3   replayed       15245      36342    2.23  6.7e-16  
      3999  3.31.43          chain3   replayed       14636      40999    2.80  7.3e-16  
      4025  5^2.7.23         chain3   replayed        7975      15455    1.92  5.7e-16  
      4059  3^2.11.41        chain3   replayed       12562      27377    2.15  5.4e-16  
      4085  5.19.43          flat     replayed       12361      35263    2.85  1.5e-15  
      4089  3.29.47          chain3   replayed       16784      43912    2.60  5.9e-16  
      4095  3^2.5.7.13       chain3   replayed        7135       8446    1.05  6.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 chain3     204      0     12   1.05   1.59   2.56    1.60
 flat       158      3     19   0.99   1.91   3.32    1.87
 2p         132      0      1   1.29   2.32   3.59    2.18
 mono        14      0      0   1.27   2.15   2.80    2.03
 ALL        508      3     32   1.06   1.86   3.22    1.83
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   3   2.23      0      0
 8..31                 12   1.51      1      0
 32..127               29   1.44      0      0
 128..511              84   2.07      0      0
 512..2047            205   1.95     17      1
 2048..4095           175   1.79     14      2
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                         204   1.59     12   1.60
 flat                                           158   1.91     19   1.87
 2p                                             132   2.32      1   2.18
 mono                                            14   2.15      0   2.03
```


flip agreement: our two readings more than 25% apart at 71 of 508 cells.

worst 10: 1001 (flat 0.59), 2695 (flat 0.78), 3125 (flat 0.80), 1183 (chain3 0.81), 2197 (flat 0.81), 1859 (chain3 0.82), 1617 (chain3 0.82), 1225 (flat 0.82), 2401 (flat 0.84), 3993 (flat 0.85)
best 5: 2209 (flat 4.68), 1849 (flat 4.39), 799 (2p 4.27), 893 (flat 4.15), 1271 (flat 4.12)
