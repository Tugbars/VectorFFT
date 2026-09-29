# gauntlet report

run: `oddtail_1d_old_2026-09-29_0844`  contract file suffix: `(oop, T=1)`  cells: 508 listed, 508 benched, comparator: MKL

control cell: 14 readings, 1.068..1.119 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         3  3                mono     replayed           4         11    2.93  1.5e-16  
         5  5                mono     replayed           5         12    2.22  7.5e-17  
         7  7                mono     replayed           7         12    1.89  2.6e-16  
         9  3^2              mono     replayed           8         13    1.59  4.2e-16  
        11  11               mono     replayed          10         14    1.32  1.2e-16  
        13  13               mono     replayed          14         15    1.08  5.0e-16  
        15  3.5              2p       replayed          15         14    0.95  3.3e-16  
        17  17               mono     replayed          22         40    1.77  2.0e-16  
        19  19               mono     replayed          27         45    1.70  4.6e-16  
        21  3.7              2p       replayed          17         26    1.47  3.3e-16  
        23  23               mono     replayed          38         67    1.74  2.5e-16  
        25  5^2              2p       replayed          20         26    1.32  1.8e-16  
        27  3^3              flat     replayed          28         36    1.15  3.8e-16  
        29  29               mono     replayed          66        107    1.61  2.8e-16  
        31  31               mono     replayed          71        121    1.71  3.3e-16  
        33  3.11             2p       replayed          26         36    1.37  2.7e-16  
        35  5.7              2p       replayed          25         36    1.41  2.6e-16  
        37  37               mono     replayed         108        185    1.71  4.8e-16  
        39  3.13             2p       replayed          33         45    1.33  6.4e-16  
        41  41               mono     replayed         131        242    1.84  3.7e-16  
        45  3^2.5            2p       replayed          32         49    1.48  5.0e-16  
        47  47               mono     replayed         185        326    1.76  4.0e-16  
        49  7^2              2p       replayed          36         47    1.32  3.2e-16  
        51  3.17             2p       replayed          56        114    2.04  4.1e-16  
        55  5.11             2p       replayed          42         54    1.28  2.8e-16  
        57  3.19             2p       replayed          63        136    2.16  4.1e-16  
        63  3^2.7            2p       replayed          46         68    1.46  4.2e-16  
        65  5.13             2p       replayed          53         67    1.26  6.3e-16  
        69  3.23             2p       replayed         101        198    1.51  4.1e-16  flips differ 1.30x
        75  3.5^2            2p       replayed          57         78    1.36  4.4e-16  
        77  7.11             2p       replayed          60         75    1.24  3.6e-16  
        81  3^4              2p       replayed          62         86    1.39  4.3e-16  
        85  5.17             2p       replayed          82        190    2.32  4.7e-16  
        87  3.29             flat     replayed         162        332    1.72  6.6e-16  
        91  7.13             2p       replayed          75         91    1.22  5.0e-16  
        93  3.31             flat     replayed         201        377    1.81  4.4e-16  
        95  5.19             2p       replayed          98        228    2.21  4.4e-16  
        99  3^2.11           2p       replayed          81        112    1.38  4.7e-16  
       105  3.5.7            2p       replayed          85        114    1.33  5.0e-16  
       115  5.23             2p       replayed         177        339    1.81  4.6e-16  
       117  3^2.13           2p       replayed         101        138    1.36  5.2e-16  
       119  7.17             2p       replayed         113        279    2.47  3.7e-16  
       121  11^2             2p       replayed         105        123    1.16  3.7e-16  
       125  5^3              2p       replayed         103        133    1.21  4.8e-16  
       129  3.43             flat     replayed         334        804    2.41  4.1e-16  
       133  7.19             2p       replayed         139        328    2.35  4.5e-16  
       135  3^3.5            2p       replayed         114        156    1.37  3.2e-16  
       141  3.47             flat     replayed         525        986    1.88  5.4e-16  
       143  11.13            2p       replayed         130        149    1.14  4.8e-16  
       145  5.29             2p       replayed         252        555    2.18  4.4e-16  
       147  3.7^2            chain3   replayed         135        159    1.18  4.0e-16  
       153  3^2.17           2p       replayed         145        372    2.53  5.6e-16  
       155  5.31             2p       replayed         340        631    1.82  4.3e-16  
       161  7.23             2p       replayed         192        480    2.31  4.5e-16  
       165  3.5.11           2p       replayed         144        189    1.29  4.0e-16  
       169  13^2             2p       replayed         161        189    1.13  7.3e-16  
       171  3^2.19           2p       replayed         189        442    2.34  6.3e-16  
       175  5^2.7            chain3   replayed         152        197    1.29  3.7e-16  
       185  5.37             2p       replayed         598        956    1.59  3.2e-16  
       187  11.17            flat     replayed         246        455    1.84  6.5e-16  
       189  3^3.7            chain3   replayed         168        229    1.32  4.4e-16  
       195  3.5.13           chain3   replayed         175        236    1.32  7.1e-16  
       203  7.29             2p       replayed         299        783    2.21  4.5e-16  
       205  5.41             2p       replayed         406       1232    3.00  5.7e-16  
       207  3^2.23           2p       replayed         277        635    2.29  5.0e-16  
       209  11.19            2p       replayed         218        526    2.40  5.1e-16  
       215  5.43             flat     replayed         493       1356    2.74  5.2e-16  
       217  7.31             2p       replayed         361        890    2.46  5.4e-16  
       221  13.17            2p       replayed         225        551    2.42  5.5e-16  
       225  3^2.5^2          2p       replayed         206        270    1.30  4.8e-16  
       231  3.7.11           chain3   replayed         214        276    1.28  3.5e-16  
       235  5.47             2p       replayed         878       1671    1.41  3.4e-16  flips differ 1.35x
       243  3^5              2p       replayed         224        297    1.32  6.5e-16  
       245  5.7^2            chain3   replayed         228        283    1.23  3.0e-16  
       247  13.19            2p       replayed         265        648    2.42  5.2e-16  
       253  11.23            2p       replayed         324        768    2.12  7.1e-16  
       255  3.5.17           2p       replayed         256        641    1.97  6.1e-16  flips differ 1.27x
       259  7.37             2p       replayed         435       1343    3.09  4.1e-16  
       261  3^2.29           2p       replayed         437       1067    2.44  4.2e-16  
       273  3.7.13           2p       replayed         263        334    1.27  6.8e-16  
       275  5^2.11           chain3   replayed         250        334    1.33  3.4e-16  
       279  3^2.31           2p       replayed         626       1194    1.82  4.3e-16  
       285  3.5.19           2p       replayed         346        759    1.96  4.4e-16  
       287  7.41             2p       replayed         539       1730    3.13  3.7e-16  
       289  17^2             2p       replayed         339       1115    3.10  7.9e-16  
       297  3^3.11           chain3   replayed         297        394    1.33  5.2e-16  
       299  13.23            2p       replayed         437        937    1.90  5.0e-16  
       301  7.43             flat     replayed         682       1904    2.71  4.6e-16  
       315  3^2.5.7          chain3   replayed         292        396    1.36  7.9e-16  
       319  11.29            2p       replayed         541       1255    2.18  5.6e-16  
       323  17.19            2p       replayed         451       1287    2.27  6.8e-16  flips differ 1.26x
       325  5^2.13           2p       replayed         329        411    1.25  7.5e-16  
       329  7.47             flat     replayed         880       2345    2.55  4.3e-16  
       333  3^2.37           2p       replayed         800       1780    2.22  5.9e-16  
       341  11.31            flat     replayed        1013       1426    1.35  6.8e-16  
       343  7^3              chain3   replayed         347        400    1.11  4.9e-16  
       345  3.5.23           2p       replayed         475       1118    2.06  5.9e-16  
       351  3^3.13           chain3   replayed         366        481    1.21  5.7e-16  
       357  3.7.17           chain3   replayed         394        916    2.22  5.6e-16  
       361  19^2             2p       replayed         532       1537    2.61  5.0e-16  
       363  3.11^2           chain3   replayed         384        459    1.19  3.3e-16  
       369  3^2.41           2p       replayed        1009       2283    1.74  5.4e-16  flips differ 1.30x
       375  3.5^3            chain3   replayed         349        472    1.02  6.8e-16  flips differ 1.33x
       377  13.29            2p       replayed         629       1522    1.83  6.2e-16  flips differ 1.32x
       385  5.7.11           chain3   replayed         373        480    1.24  5.6e-16  
       387  3^2.43           2p       replayed         803       2508    3.11  5.4e-16  
       391  17.23            2p       replayed         539       1760    2.72  6.7e-16  
       399  3.7.19           chain3   replayed         470       1084    2.21  6.2e-16  
       403  13.31            2p       replayed         714       1722    2.38  6.4e-16  
       405  3^4.5            chain3   replayed         379        550    1.45  5.2e-16  
       407  11.37            2p       replayed         680       2136    3.13  4.1e-16  
       423  3^2.47           flat     replayed        1131       3066    2.61  7.0e-16  
       425  5^2.17           2p       replayed         457       1083    2.37  6.4e-16  
       429  3.11.13          chain3   replayed         467        555    1.18  5.2e-16  
       435  3.5.29           2p       replayed         636       1794    2.62  5.5e-16  
       437  19.23            2p       replayed        1019       2109    2.02  5.5e-16  
       441  3^2.7^2          chain3   replayed         447        567    1.27  4.9e-16  
       451  11.41            2p       replayed        1702       2747    1.54  5.0e-16  
       455  5.7.13           chain3   replayed         458        584    1.25  6.0e-16  
       459  3^3.17           chain3   replayed         551       1207    2.08  5.1e-16  
       465  3.5.31           2p       replayed         815       2017    1.94  5.2e-16  flips differ 1.28x
       473  11.43            2p       replayed         956       3032    2.98  6.0e-16  
       475  5^2.19           2p       replayed         533       1287    2.40  5.0e-16  
       481  13.37            2p       replayed         804       2578    2.76  5.5e-16  
       483  3.7.23           2p       replayed         698       1590    2.27  5.1e-16  
       493  17.29            2p       replayed         972       2702    2.35  5.0e-16  
       495  3^2.5.11         flat     replayed         590        670    1.13  5.8e-16  
       507  3.13^2           chain3   replayed         628        716    1.12  6.2e-16  
       513  3^3.19           2p       replayed         578       1472    2.29  5.4e-16  
       517  11.47            flat     replayed        1329       3711    2.72  6.4e-16  
       525  3.5^2.7          2p       replayed         539        721    1.22  6.4e-16  
       527  17.31            2p       replayed         910       2993    2.45  7.5e-16  flips differ 1.34x
       529  23^2             2p       replayed         896       2815    2.33  6.5e-16  flips differ 1.34x
       533  13.41            2p       replayed        1018       3308    3.10  6.4e-16  
       539  7^2.11           flat     replayed         677        729    1.07  5.4e-16  
       551  19.29            2p       replayed         950       3123    3.28  4.0e-16  
       555  3.5.37           2p       replayed         917       3015    3.28  6.0e-16  
       559  13.43            2p       replayed        1163       3642    2.84  5.6e-16  
       561  3.11.17          flat     replayed         867       1479    1.65  1.2e-15  
       567  3^4.7            2p       replayed         595        838    1.40  6.4e-16  
       575  5^2.23           2p       replayed         847       1886    1.86  5.8e-16  
       585  3^2.5.13         chain3   replayed         767        856    1.05  5.6e-16  
       589  19.31            2p       replayed        1062       3487    2.65  5.1e-16  
       595  5.7.17           chain3   replayed         827       1579    1.77  6.6e-16  
       605  5.11^2           flat     replayed         759        828    0.99  6.1e-16  
       609  3.7.29           2p       replayed        1042       2522    2.40  5.8e-16  
       611  13.47            flat     replayed        1612       4445    2.65  7.8e-16  
       615  3.5.41           2p       replayed        1186       3855    3.15  6.3e-16  
       621  3^3.23           2p       replayed         887       2096    2.36  6.7e-16  
       625  5^4              2p       replayed         676        872    1.26  4.1e-16  
       627  3.11.19          flat     replayed         995       1752    1.74  6.9e-16  
       629  17.37            2p       replayed        1504       4263    2.03  8.6e-16  flips differ 1.39x
       637  7^2.13           flat     replayed         990        876    0.78  6.3e-16  
       645  3.5.43           2p       replayed        1292       4241    2.88  5.5e-16  
       651  3.7.31           2p       replayed        1216       2846    1.69  5.2e-16  flips differ 1.39x
       663  3.13.17          flat     replayed        1058       1809    1.63  8.9e-16  
       665  5.7.19           flat     replayed         919       1867    1.77  5.1e-16  
       667  23.29            2p       replayed        1434       4111    2.34  6.1e-16  
       675  3^3.5^2          2p       replayed         748        997    1.29  5.9e-16  
       693  3^2.7.11         flat     replayed         902       1000    1.09  7.0e-16  
       697  17.41            2p       replayed        1556       5299    3.38  8.6e-16  
       703  19.37            2p       replayed        1390       4923    3.37  4.6e-16  
       705  3.5.47           2p       replayed        1557       5179    3.19  6.6e-16  
       713  23.31            2p       replayed        1766       4571    2.19  9.4e-16  
       715  5.11.13          flat     replayed         936        999    1.07  6.4e-16  
       725  5^2.29           flat     replayed        1421       3030    2.11  5.8e-16  
       729  3^6              2p       replayed         835       1085    1.28  5.1e-16  
       731  17.43            2p       replayed        1657       5772    3.48  6.0e-16  
       735  3.5.7^2          flat     replayed        1005       1036    1.03  6.1e-16  
       741  3.13.19          chain3   replayed        1358       2141    1.14  6.5e-16  flips differ 1.39x
       759  3.11.23          flat     replayed        1310       2554    1.90  6.8e-16  
       765  3^2.5.17         flat     replayed        1113       2134    1.91  6.0e-16  
       775  5^2.31           2p       replayed        1456       3430    2.17  6.3e-16  
       777  3.7.37           2p       replayed        1353       4253    3.07  5.5e-16  
       779  19.41            2p       replayed        1776       6100    3.34  4.7e-16  
       783  3^3.29           flat     replayed        1478       3407    2.28  4.8e-16  
       799  17.47            2p       replayed        1950       6913    3.45  5.6e-16  
       805  5.7.23           flat     replayed        1388       2688    1.93  6.6e-16  
       817  19.43            flat     replayed        2189       6692    3.03  8.9e-16  
       819  3^2.7.13         flat     replayed        1125       1215    1.04  7.7e-16  
       825  3.5^2.11         chain3   replayed        1194       1220    1.02  4.4e-16  
       833  7^2.17           flat     replayed        1195       2233    1.85  7.7e-16  
       837  3^3.31           flat     replayed        2036       3814    1.72  7.1e-16  
       841  29^2             flat     replayed        2366       6005    2.45  1.2e-15  
       845  5.13^2           flat     replayed        1331       1264    0.79  9.8e-16  
       847  7.11^2           flat     replayed        1160       1196    0.98  4.3e-16  
       851  23.37            2p       replayed        2040       6385    2.72  6.1e-16  
       855  3^2.5.19         flat     replayed        1361       2502    1.82  8.2e-16  
       861  3.7.41           2p       replayed        1693       5428    3.14  4.6e-16  
       867  3.17^2           flat     replayed        1501       3518    2.06  1.1e-15  
       875  5^3.7            flat     replayed        1128       1289    0.84  4.5e-16  flips differ 1.37x
       891  3^4.11           chain3   replayed        1217       1424    1.15  6.0e-16  
       893  19.47            flat     replayed        2436       7951    3.19  9.9e-16  
       897  3.13.23          flat     replayed        1962       3101    1.45  8.4e-16  
       899  29.31            2p       replayed        2529       6637    2.21  5.6e-16  
       903  3.7.43           flat     replayed        2481       5964    2.31  1.9e-15  
       925  5^2.37           2p       replayed        2288       5092    2.19  4.4e-16  
       931  7^2.19           flat     replayed        1523       2646    1.71  7.7e-16  
       935  5.11.17          flat     replayed        1381       2549    1.38  6.2e-16  flips differ 1.33x
       943  23.41            2p       replayed        2688       7888    2.86  5.5e-16  
       945  3^3.5.7          chain3   replayed        1338       1458    0.96  6.1e-16  
       957  3.11.29          chain3   replayed        2576       4041    1.54  5.4e-16  
       961  31^2             flat     replayed        2568       7370    2.81  9.3e-16  
       969  3.17.19          flat     replayed        1781       4090    2.17  8.8e-16  
       975  3.5^2.13         flat     replayed        1357       1493    0.79  6.2e-16  flips differ 1.40x
       987  3.7.47           2p       replayed        2153       7349    3.40  5.1e-16  
       989  23.43            flat     replayed        2723       8655    3.12  1.1e-15  
       999  3^3.37           2p       replayed        3458       5668    1.60  6.4e-16  
      1001  7.11.13          flat     replayed        1498       1462    0.93  7.1e-16  
      1015  5.7.29           flat     replayed        1900       4355    1.92  5.8e-16  
      1023  3.11.31          chain3   replayed        2316       4649    1.82  6.3e-16  
      1025  5^2.41           2p       replayed        2083       6563    3.08  5.0e-16  
      1029  3.7^3            flat     replayed        1488       1546    1.03  4.4e-16  
      1035  3^2.5.23         flat     replayed        1703       3672    2.13  6.7e-16  
      1045  5.11.19          flat     replayed        1797       3082    1.70  6.3e-16  
      1053  3^4.13           chain3   replayed        1602       1798    1.12  8.0e-16  
      1071  3^2.7.17         flat     replayed        1862       3095    1.54  5.7e-16  
      1073  29.37            2p       replayed        2880       9168    3.17  4.9e-16  
      1075  5^2.43           2p       replayed        2328       7214    2.80  5.6e-16  
      1081  23.47            2p       replayed        3889      10270    1.80  5.5e-16  flips differ 1.47x
      1083  3.19^2           chain3   replayed        2054       4931    2.09  4.7e-16  
      1085  5.7.31           chain3   replayed        3839       4919    1.28  5.5e-16  
      1089  3^2.11^2         chain3   replayed        1600       1723    1.06  6.2e-16  
      1105  5.13.17          flat     replayed        1765       3191    1.16  8.0e-16  flips differ 1.55x
      1107  3^3.41           2p       replayed        2227       7216    2.92  5.2e-16  
      1125  3^2.5^3          flat     replayed        1508       1821    1.15  7.8e-16  
      1127  7^2.23           chain3   replayed        2304       3887    1.37  6.4e-16  
      1131  3.13.29          chain3   replayed        2501       4979    1.97  7.4e-16  
      1147  31.37            flat     replayed        3178      10135    3.02  8.4e-16  
      1155  3.5.7.11         chain3   replayed        1662       1828    1.07  4.2e-16  
      1161  3^3.43           2p       replayed        2397       7915    3.03  5.5e-16  
      1173  3.17.23          flat     replayed        2519       5686    2.13  1.1e-15  
      1175  5^2.47           flat     replayed        2771       8793    3.17  4.6e-16  
      1183  7.13^2           chain3   replayed        1828       1920    1.04  7.3e-16  
      1189  29.41            2p       replayed        3457      11167    2.75  5.0e-16  
      1197  3^2.7.19         flat     replayed        2422       3659    1.43  1.1e-15  
      1209  3.13.31          chain3   replayed        2771       5608    1.73  5.6e-16  
      1215  3^5.5            chain3   replayed        1752       2105    1.20  9.3e-16  
      1221  3.11.37          chain3   replayed        2931       6885    2.22  4.9e-16  
      1225  5^2.7^2          flat     replayed        1833       1931    1.05  4.5e-16  
      1235  5.13.19          flat     replayed        2126       3760    1.57  7.9e-16  
      1247  29.43            flat     replayed        3782      12105    3.02  2.8e-15  
      1265  5.11.23          flat     replayed        2466       4432    1.70  9.0e-16  
      1269  3^3.47           flat     replayed        3422       9638    2.68  7.1e-16  
      1271  31.41            flat     replayed        3695      12332    3.33  1.3e-15  
      1275  3.5^2.17         flat     replayed        2247       3680    1.60  7.2e-16  
      1287  3^2.11.13        chain3   replayed        1955       2135    1.08  5.9e-16  
      1295  5.7.37           flat     replayed        2663       7305    2.62  7.7e-16  
      1305  3^2.5.29         chain3   replayed        2913       5788    1.97  6.7e-16  
      1309  7.11.17          flat     replayed        2525       3705    1.47  1.1e-15  
      1311  3.19.23          chain3   replayed        3769       6710    1.71  6.0e-16  
      1323  3^3.7^2          flat     replayed        2009       2190    1.03  8.3e-16  
      1331  11^3             chain3   replayed        2087       2079    0.83  4.5e-16  
      1333  31.43            2p       replayed        5456      13304    2.42  6.7e-16  
      1353  3.11.41          chain3   replayed        3743       8739    2.13  4.9e-16  
      1363  29.47            2p       replayed        6390      14279    2.04  4.5e-16  
      1365  3.5.7.13         chain3   replayed        2247       2265    1.01  6.0e-16  
      1369  37^2             2p       replayed        4491      13548    3.01  5.1e-16  
      1375  5^3.11           chain3   replayed        2288       2249    0.90  4.2e-16  
      1377  3^4.17           chain3   replayed        2524       4105    1.47  8.2e-16  
      1395  3^2.5.31         flat     replayed        3258       6525    1.93  6.6e-16  
      1419  3.11.43          flat     replayed        3636       9609    2.32  6.0e-16  
      1421  7^2.29           chain3   replayed        3423       6155    1.79  5.7e-16  
      1425  3.5^2.19         flat     replayed        2186       4352    1.78  5.1e-16  
      1435  5.7.41           chain3   replayed        3986       9269    2.31  5.8e-16  
      1443  3.13.37          flat     replayed        3883       8275    2.13  6.5e-16  
      1445  5.17^2           chain3   replayed        2599       6071    2.14  6.4e-16  
      1449  3^2.7.23         flat     replayed        2832       5228    1.79  5.4e-16  
      1457  31.47            2p       replayed        5853      15687    2.00  5.9e-16  flips differ 1.35x
      1463  7.11.19          chain3   replayed        2518       4368    1.70  4.8e-16  
      1479  3.17.29          flat     replayed        4178       8476    1.93  1.2e-15  
      1485  3^3.5.11         chain3   replayed        2294       2552    1.10  5.3e-16  
      1495  5.13.23          chain3   replayed        3023       5362    1.44  6.5e-16  
      1505  5.7.43           chain3   replayed        5553      10215    1.83  6.1e-16  
      1517  37.41            2p       replayed        5268      16308    3.08  8.5e-16  
      1519  7^2.31           flat     replayed        3760       6940    1.69  7.3e-16  
      1521  3^2.13^2         chain3   replayed        2357       2852    1.19  7.1e-16  
      1539  3^4.19           chain3   replayed        2892       4966    1.56  7.1e-16  
      1547  7.13.17          flat     replayed        2664       4542    1.52  9.5e-16  
      1551  3.11.47          chain3   replayed        5796      11706    1.31  5.0e-16  flips differ 1.54x
      1573  11^2.13          chain3   replayed        2476       2535    1.02  5.6e-16  
      1575  3^2.5^2.7        flat     replayed        3118       2759    0.64  6.6e-16  flips differ 1.39x
      1581  3.17.31          chain3   replayed        4794       9456    1.66  6.2e-16  
      1587  3.23^2           flat     replayed        4774       8955    1.77  1.2e-15  
      1591  37.43            2p       replayed        5752      17601    2.40  5.8e-16  flips differ 1.28x
      1595  5.11.29          flat     replayed        4890       6983    1.35  6.7e-16  
      1599  3.13.41          chain3   replayed        4099      10499    2.51  5.7e-16  
      1615  5.17.19          chain3   replayed        3314       7079    1.98  6.1e-16  
      1617  3.7^2.11         chain3   replayed        2765       2776    0.99  5.5e-16  
      1625  5^3.13           flat     replayed        2468       2807    1.10  6.6e-16  
      1645  5.7.47           flat     replayed        4180      12411    2.71  5.6e-16  
      1653  3.19.29          chain3   replayed        4299       9954    1.85  5.4e-16  
      1665  3^2.5.37         chain3   replayed        4862       9642    1.50  5.6e-16  flips differ 1.32x
      1677  3.13.43          chain3   replayed        4600      11546    2.32  6.8e-16  
      1681  41^2             2p       replayed        5917      19788    3.24  5.1e-16  
      1683  3^2.11.17        chain3   replayed        2778       5019    1.68  7.8e-16  
      1701  3^5.7            chain3   replayed        2634       3167    1.19  4.7e-16  
      1705  5.11.31          chain3   replayed        4975       7917    1.32  5.0e-16  
      1715  5.7^3            flat     replayed        2780       2841    1.00  4.9e-16  
      1725  3.5^2.23         chain3   replayed        3691       6291    1.70  6.9e-16  
      1729  7.13.19          chain3   replayed        3225       5345    1.65  6.4e-16  
      1739  37.47            flat     replayed        5604      20560    3.37  1.9e-15  
      1755  3^3.5.13         chain3   replayed        2730       3263    1.18  8.8e-16  
      1763  41.43            2p       replayed        8801      21294    2.38  6.0e-16  
      1767  3.19.31          chain3   replayed        4367      11074    2.01  4.8e-16  flips differ 1.26x
      1771  7.11.23          flat     replayed        3778       6274    1.62  8.1e-16  
      1785  3.5.7.17         chain3   replayed        3053       5348    1.74  6.5e-16  
      1805  5.19^2           flat     replayed        4010       8365    2.08  1.3e-15  
      1813  7^2.37           chain3   replayed        4823      10281    2.13  5.0e-16  
      1815  3.5.11^2         chain3   replayed        2830       3238    1.14  6.1e-16  
      1827  3^2.7.29         flat     replayed        4567       8206    1.48  6.0e-16  
      1833  3.13.47          flat     replayed        5336      13999    2.53  7.7e-16  
      1845  3^2.5.41         chain3   replayed        4557      12187    2.66  5.9e-16  
      1849  43^2             flat     replayed        5827      22621    3.83  1.6e-15  
      1859  11.13^2          chain3   replayed        3044       3294    1.08  7.3e-16  
      1863  3^4.23           chain3   replayed        3690       6927    1.84  6.2e-16  
      1875  3.5^4            flat     replayed        3391       3393    0.99  6.3e-16  
      1881  3^2.11.19        chain3   replayed        3349       5927    1.32  6.5e-16  flips differ 1.34x
      1885  5.13.29          chain3   replayed        4481       8452    1.87  6.9e-16  
      1887  3.17.37          chain3   replayed        4835      13365    2.70  6.2e-16  
      1911  3.7^2.13         chain3   replayed        2955       3445    1.05  5.8e-16  
      1925  5^2.7.11         flat     replayed        3038       3330    0.84  6.7e-16  flips differ 1.31x
      1935  3^2.5.43         flat     replayed        4877      13409    2.49  6.2e-16  
      1953  3^2.7.31         flat     replayed        5140       9258    1.51  7.6e-16  
      1955  5.17.23          chain3   replayed        6107       9669    1.57  6.2e-16  
      1989  3^2.13.17        chain3   replayed        3446       6126    1.64  6.9e-16  
      1995  3.5.7.19         chain3   replayed        4150       6308    1.25  5.1e-16  
      2001  3.23.29          chain3   replayed        6110      13055    1.62  6.4e-16  flips differ 1.32x
      2009  7^2.41           chain3   replayed        6412      13054    2.03  4.7e-16  
      2015  5.13.31          flat     replayed        4479       9522    1.39  6.5e-16  flips differ 1.53x
      2023  7.17^2           chain3   replayed        3542       8602    2.42  7.8e-16  
      2025  3^4.5^2          chain3   replayed        3567       3872    1.04  4.9e-16  
      2035  5.11.37          chain3   replayed        6159      11659    1.89  5.0e-16  
      2057  11^2.17          flat     replayed        3824       5978    1.20  9.5e-16  flips differ 1.30x
      2079  3^3.7.11         chain3   replayed        3147       3838    1.17  7.6e-16  
      2091  3.17.41          chain3   replayed        5772      16551    2.87  6.6e-16  
      2093  7.13.23          flat     replayed        3995       7606    1.73  7.1e-16  
      2107  7^2.43           chain3   replayed        7039      14339    1.56  5.1e-16  flips differ 1.30x
      2109  3.19.37          flat     replayed        5799      15448    2.63  7.3e-16  
      2115  3^2.5.47         chain3   replayed        7815      16243    1.34  5.0e-16  flips differ 1.56x
      2125  5^3.17           chain3   replayed        3696       6330    1.71  7.5e-16  
      2139  3.23.31          flat     replayed        7438      14480    1.82  9.3e-16  
      2145  3.5.11.13        chain3   replayed        3452       4017    1.03  5.4e-16  
      2175  3.5^2.29         chain3   replayed        4978       9858    1.67  5.0e-16  
      2185  5.19.23          chain3   replayed        4997      11365    1.74  5.1e-16  flips differ 1.30x
      2187  3^7              chain3   replayed        3826       4514    1.16  6.0e-16  
      2193  3.17.43          chain3   replayed        8100      18069    1.75  7.4e-16  flips differ 1.28x
      2197  13^3             flat     replayed        4114       4130    0.87  1.2e-15  
      2205  3^2.5.7^2        chain3   replayed        3610       4145    1.12  5.3e-16  
      2209  47^2             flat     replayed        7424      30516    3.79  1.9e-15  
      2223  3^2.13.19        flat     replayed        4719       7230    1.53  1.2e-15  
      2233  7.11.29          chain3   replayed        5788       9876    1.68  6.3e-16  
      2255  5.11.41          chain3   replayed        5777      14761    2.55  6.0e-16  
      2261  7.17.19          flat     replayed        4482       9955    2.11  9.9e-16  
      2275  5^2.7.13         chain3   replayed        4103       4144    0.99  5.9e-16  
      2277  3^2.11.23        chain3   replayed        4755       8477    1.58  6.3e-16  
      2295  3^3.5.17         chain3   replayed        4180       7229    1.50  6.9e-16  
      2299  11^2.19          flat     replayed        4588       7038    1.30  1.2e-15  
      2303  7^2.47           chain3   replayed        8218      17425    2.10  6.0e-16  
      2325  3.5^2.31         chain3   replayed        8464      11118    1.30  5.7e-16  
      2331  3^2.7.37         chain3   replayed        6899      13615    1.41  5.4e-16  flips differ 1.40x
      2337  3.19.41          chain3   replayed        6863      19009    2.17  7.5e-16  flips differ 1.27x
      2349  3^4.29           chain3   replayed        7654      11034    1.35  6.6e-16  
      2365  5.11.43          chain3   replayed        8230      16193    1.51  5.6e-16  flips differ 1.30x
      2375  5^3.19           flat     replayed        4103       7469    1.77  6.8e-16  
      2387  7.11.31          chain3   replayed        6843      11146    1.29  4.7e-16  flips differ 1.26x
      2397  3.17.47          flat     replayed        7904      21773    2.53  1.6e-15  
      2401  7^4              flat     replayed        4592       4111    0.87  6.8e-16  
      2405  5.13.37          chain3   replayed        5695      14099    2.21  6.3e-16  
      2415  3.5.7.23         chain3   replayed        6858       9020    1.23  6.2e-16  
      2431  11.13.17         chain3   replayed        4213       7338    1.73  6.0e-16  
      2451  3.19.43          flat     replayed        8007      20809    2.59  1.4e-15  
      2457  3^3.7.13         chain3   replayed        4359       4827    1.02  8.3e-16  
      2465  5.17.29          flat     replayed        6482      14398    2.19  1.6e-15  
      2475  3^2.5^2.11       chain3   replayed        4003       4728    0.95  7.2e-16  
      2499  3.7^2.17         chain3   replayed        4046       7562    1.85  6.6e-16  
      2511  3^4.31           flat     replayed        5756      12373    1.86  7.5e-16  
      2523  3.29^2           flat     replayed        8564      18904    2.08  1.4e-15  
      2527  7.19^2           chain3   replayed        5203      11935    2.28  5.3e-16  
      2535  3.5.13^2         chain3   replayed        4217       5115    1.13  7.3e-16  
      2541  3.7.11^2         flat     replayed        5390       4809    0.72  7.6e-16  
      2553  3.23.37          flat     replayed        7642      20086    2.57  8.7e-16  
      2565  3^3.5.19         chain3   replayed        5400       8404    1.54  8.1e-16  
      2583  3^2.7.41         chain3   replayed        6569      17214    2.57  6.2e-16  
      2585  5.11.47          chain3   replayed        9510      19738    1.56  5.3e-16  flips differ 1.33x
      2601  3^2.17^2         chain3   replayed        4880      11471    2.32  7.7e-16  
      2625  3.5^3.7          flat     replayed        4821       4946    1.02  9.2e-16  
      2635  5.17.31          flat     replayed        7845      16007    2.00  8.0e-16  
      2639  7.13.29          chain3   replayed        5906      11917    2.00  7.5e-16  
      2645  5.23^2           flat     replayed        7334      15123    2.02  1.7e-15  
      2665  5.13.41          chain3   replayed        6808      17707    2.50  7.8e-16  
      2673  3^5.11           chain3   replayed        4785       5314    1.02  7.9e-16  
      2679  3.19.47          chain3   replayed       10202      24846    1.38  5.9e-16  flips differ 1.76x
      2691  3^2.13.23        chain3   replayed        5417      10219    1.60  6.2e-16  
      2695  5.7^2.11         flat     replayed        4668       5023    0.86  5.5e-16  flips differ 1.25x
      2697  3.29.31          flat     replayed        8782      20891    1.92  2.8e-15  
      2709  3^2.7.43         chain3   replayed       12211      18848    1.33  6.0e-16  
      2717  11.13.19         flat     replayed        5311       8589    1.61  8.5e-16  
      2737  7.17.23          flat     replayed        6891      13626    1.88  1.4e-15  
      2755  5.19.29          chain3   replayed        9936      19353    1.69  6.8e-16  flips differ 1.28x
      2775  3.5^2.37         chain3   replayed        8568      16265    1.89  7.1e-16  
      2783  11^2.23          flat     replayed        6433      10073    1.49  1.4e-15  
      2793  3.7^2.19         chain3   replayed        4761       8899    1.79  6.3e-16  
      2795  5.13.43          chain3   replayed        7554      19528    2.41  5.9e-16  
      2805  3.5.11.17        chain3   replayed        4969       8616    1.64  6.0e-16  
      2821  7.13.31          chain3   replayed        6845      13438    1.35  6.5e-16  flips differ 1.46x
      2829  3.23.41          flat     replayed        8363      24561    2.52  8.5e-16  
      2835  3^4.5.7          chain3   replayed        4905       5578    1.12  7.2e-16  
      2849  7.11.37          flat     replayed        6340      16398    2.56  6.0e-16  
      2871  3^2.11.29        flat     replayed        7011      13147    1.19  8.2e-16  flips differ 1.58x
      2873  13^2.17          chain3   replayed        5438       8900    1.62  8.2e-16  
      2875  5^3.23           chain3   replayed        7444      10797    1.43  6.7e-16  
      2883  3.31^2           flat     replayed        9251      23106    1.27  1.9e-15  flips differ 1.97x
      2907  3^2.17.19        flat     replayed        6856      13337    1.63  1.6e-15  
      2925  3^2.5^2.13       chain3   replayed        4992       5937    1.15  6.6e-16  
      2945  5.19.31          chain3   replayed        9346      18630    1.93  5.4e-16  
      2961  3^2.7.47         flat     replayed        7479      22909    2.65  6.3e-16  
      2967  3.23.43          chain3   replayed       12604      27832    1.90  5.6e-16  
      2975  5^2.7.17         flat     replayed        5876       9224    1.50  6.8e-16  
      2997  3^4.37           chain3   replayed        7278      17926    2.34  5.1e-16  
      3003  3.7.11.13        chain3   replayed        4925       5843    1.18  6.6e-16  
      3025  5^2.11^2         flat     replayed        5767       5890    0.95  8.7e-16  
      3045  3.5.7.29         flat     replayed        8367      14743    1.60  7.1e-16  
      3055  5.13.47          chain3   replayed       11524      25031    1.33  6.2e-16  flips differ 1.73x
      3059  7.19.23          flat     replayed        6887      16196    2.05  7.0e-16  
      3069  3^2.11.31        chain3   replayed        8929      14839    1.66  9.8e-16  
      3075  3.5^2.41         chain3   replayed       12950      20596    1.58  6.2e-16  
      3087  3^2.7^3          chain3   replayed        5184       6015    0.87  5.0e-16  flips differ 1.34x
      3105  3^3.5.23         chain3   replayed        7398      11956    1.59  5.3e-16  
      3125  5^5              flat     replayed        5812       6032    1.03  8.4e-16  
      3135  3.5.11.19        chain3   replayed        5750      10192    1.54  7.4e-16  
      3145  5.17.37          chain3   replayed        7919      22652    2.71  6.0e-16  
      3157  7.11.41          chain3   replayed        8386      20780    2.37  5.2e-16  
      3159  3^5.13           chain3   replayed        5461       6561    1.20  7.2e-16  
      3179  11.17^2          flat     replayed        6863      13787    1.95  1.1e-15  
      3185  5.7^2.13         flat     replayed        5541       6154    1.07  6.7e-16  
      3211  13^2.19          flat     replayed        6829      10496    1.21  1.0e-15  flips differ 1.27x
      3213  3^3.7.17         chain3   replayed        5746      10222    1.64  9.3e-16  
      3219  3.29.37          flat     replayed       10503      28465    2.28  2.1e-15  
      3225  3.5^2.43         chain3   replayed       14947      22592    1.36  5.6e-16  
      3243  3.23.47          chain3   replayed       14075      31672    1.83  5.8e-16  
      3249  3^2.19^2         flat     replayed        8775      15800    1.74  1.1e-15  
      3255  3.5.7.31         flat     replayed        8732      15720    1.79  6.8e-16  
      3267  3^3.11^2         flat     replayed        7082       6652    0.93  5.6e-16  
      3289  11.13.23         chain3   replayed        7775      12228    1.31  8.5e-16  
      3311  7.11.43          chain3   replayed       11465      22985    1.50  5.7e-16  flips differ 1.34x
      3315  3.5.13.17        flat     replayed        7320      10746    1.36  1.3e-15  
      3321  3^4.41           chain3   replayed        8862      22659    2.37  6.2e-16  
      3325  5^2.7.19         chain3   replayed        6702      10827    1.17  5.0e-16  flips differ 1.38x
      3335  5.23.29          chain3   replayed       12176      22085    1.61  6.0e-16  
      3367  7.13.37          chain3   replayed        8204      19763    2.36  6.3e-16  
      3375  3^3.5^3          chain3   replayed        5425       6901    1.04  5.9e-16  
      3381  3.7^2.23         flat     replayed        8013      12673    1.56  7.7e-16  
      3393  3^2.13.29        flat     replayed        7988      15848    1.90  1.2e-15  
      3441  3.31.37          flat     replayed       11645      31405    2.64  2.3e-15  
      3451  7.17.29          chain3   replayed        8542      20268    1.80  7.1e-16  flips differ 1.32x
      3465  3^2.5.7.11       chain3   replayed        5710       7089    1.21  6.1e-16  
      3483  3^4.43           chain3   replayed       13181      24789    1.69  6.5e-16  
      3485  5.17.41          chain3   replayed       11474      27919    1.62  5.7e-16  flips differ 1.51x
      3509  11^2.29          chain3   replayed        8413      15815    1.87  6.4e-16  
      3515  5.19.37          chain3   replayed       10074      26218    2.44  6.3e-16  
      3519  3^2.17.23        flat     replayed        7726      18162    2.10  8.1e-16  
      3525  3.5^2.47         flat     replayed        8584      27585    2.99  5.6e-16  
      3549  3.7.13^2         chain3   replayed        5961       7360    1.16  7.0e-16  
      3553  11.17.19         chain3   replayed        6741      15967    2.36  6.0e-16  
      3565  5.23.31          flat     replayed       13079      24494    1.72  1.7e-15  
      3567  3.29.41          flat     replayed       12045      34435    2.84  2.3e-15  
      3575  5^2.11.13        chain3   replayed        6561       7186    1.01  7.9e-16  
      3591  3^3.7.19         chain3   replayed        7155      11987    1.38  6.3e-16  
      3619  7.11.47          chain3   replayed       13152      27754    1.60  4.7e-16  flips differ 1.32x
      3625  5^3.29           chain3   replayed        8389      16851    1.62  5.4e-16  
      3627  3^2.13.31        flat     replayed        8675      17925    1.97  1.0e-15  
      3645  3^6.5            chain3   replayed        6943       7766    1.10  6.6e-16  
      3655  5.17.43          chain3   replayed       10612      30590    2.36  5.3e-16  
      3663  3^2.11.37        flat     replayed       10753      21733    1.99  2.3e-15  
      3675  3.5^2.7^2        chain3   replayed        6534       7144    0.94  5.1e-16  
      3689  7.17.31          chain3   replayed       13563      22614    1.61  6.8e-16  
      3703  7.23^2           chain3   replayed       12087      21403    1.74  6.2e-16  
      3705  3.5.13.19        chain3   replayed        6581      12360    1.84  6.1e-16  
      3731  7.13.41          chain3   replayed        9484      25053    2.61  6.8e-16  
      3741  3.29.43          flat     replayed       13657      37334    2.60  3.5e-15  
      3751  11^2.31          flat     replayed        9355      17802    1.75  8.8e-16  
      3757  13.17^2          chain3   replayed        7412      16761    2.19  8.5e-16  
      3773  7^3.11           flat     replayed        7266       7164    0.91  7.6e-16  
      3795  3.5.11.23        chain3   replayed        9019      14434    1.59  6.2e-16  
      3807  3^4.47           chain3   replayed       14931      30123    1.55  6.4e-16  flips differ 1.30x
      3813  3.31.41          flat     replayed       14357      37964    2.47  2.5e-15  
      3825  3^2.5^2.17       chain3   replayed        6646      12483    1.84  8.6e-16  
      3857  7.19.29          chain3   replayed        9640      23721    1.96  6.5e-16  flips differ 1.25x
      3861  3^3.11.13        flat     replayed        8685       8161    0.88  9.9e-16  
      3875  5^3.31           chain3   replayed       11132      18892    1.66  7.1e-16  
      3885  3.5.7.37         chain3   replayed        9166      23278    2.31  5.0e-16  
      3887  13^2.23          flat     replayed        8186      14928    1.37  9.1e-16  flips differ 1.33x
      3895  5.19.41          chain3   replayed       11288      32211    2.63  5.1e-16  
      3913  7.13.43          chain3   replayed       13868      27539    1.98  7.6e-16  
      3915  3^3.5.29         flat     replayed       11408      18503    1.22  6.5e-16  flips differ 1.33x
      3927  3.7.11.17        chain3   replayed        7523      12221    1.59  6.2e-16  
      3933  3^2.19.23        chain3   replayed        8862      21572    2.42  7.4e-16  
      3969  3^4.7^2          chain3   replayed        6705       8276    1.16  8.0e-16  
      3971  11.19^2          chain3   replayed        7932      18995    2.02  6.4e-16  
      3993  3.11^3           flat     replayed        8849       7691    0.77  8.3e-16  
      3995  5.17.47          chain3   replayed       15694      36323    1.80  6.7e-16  flips differ 1.29x
      3999  3.31.43          chain3   replayed       15358      41014    2.30  7.3e-16  
      4025  5^2.7.23         chain3   replayed        8664      15343    1.77  5.7e-16  
      4059  3^2.11.41        chain3   replayed       12487      27372    1.65  5.4e-16  flips differ 1.33x
      4085  5.19.43          flat     replayed       13042      35228    2.56  1.5e-15  
      4089  3.29.47          chain3   replayed       18922      43863    1.95  5.9e-16  
      4095  3^2.5.7.13       chain3   replayed        7279       8437    1.11  6.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 chain3     204      0      8   1.05   1.58   2.32    1.55
 flat       158      6     19   0.95   1.77   2.71    1.69
 2p         132      0      1   1.28   2.22   3.13    2.08
 mono        14      0      0   1.32   1.73   2.22    1.73
 ALL        508      6     28   1.07   1.74   2.72    1.72
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   3   2.22      0      0
 8..31                 12   1.53      1      0
 32..127               29   1.41      0      0
 128..511              84   2.00      0      0
 512..2047            205   1.80     14      4
 2048..4095           175   1.66     13      2
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                         204   1.58      8   1.55
 flat                                           158   1.77     19   1.69
 2p                                             132   2.22      1   2.08
 mono                                            14   1.73      0   1.73
```


flip agreement: our two readings more than 25% apart at 58 of 508 cells.

worst 10: 1575 (flat 0.64), 2541 (flat 0.72), 3993 (flat 0.77), 637 (flat 0.78), 845 (flat 0.79), 975 (flat 0.79), 1331 (chain3 0.83), 875 (flat 0.84), 1925 (flat 0.84), 2695 (flat 0.86)
best 5: 1849 (flat 3.83), 2209 (flat 3.79), 731 (2p 3.48), 799 (2p 3.45), 987 (2p 3.40)
