# gauntlet report

run: `oddtail_before_2026-09-29`  contract file suffix: `(oop, T=1)`  cells: 508 listed, 508 benched, comparator: MKL

control cell: 14 readings, 1.062..1.083 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         3  3                mono     replayed           4         11    3.01  1.5e-16  
         5  5                mono     replayed           5         12    2.25  7.5e-17  
         7  7                mono     replayed           7         12    1.89  2.6e-16  
         9  3^2              mono     replayed           8         13    1.58  4.2e-16  
        11  11               mono     replayed          10         14    1.34  1.2e-16  
        13  13               mono     replayed          14         15    1.08  5.0e-16  
        15  3.5              2p       replayed          15         14    0.94  3.3e-16  
        17  17               mono     replayed          22         40    1.80  2.0e-16  
        19  19               mono     replayed          27         45    1.69  4.6e-16  
        21  3.7              2p       replayed          17         25    1.47  3.3e-16  
        23  23               mono     replayed          39         67    1.74  2.5e-16  
        25  5^2              2p       replayed          19         26    1.36  1.8e-16  
        27  3^3              flat     replayed          28         32    1.15  3.8e-16  
        29  29               mono     replayed          65        107    1.61  2.8e-16  
        31  31               mono     replayed          71        121    1.71  3.3e-16  
        33  3.11             2p       replayed          26         36    1.36  2.7e-16  
        35  5.7              2p       replayed          25         36    1.43  2.6e-16  
        37  37               mono     replayed         107        185    1.71  4.8e-16  
        39  3.13             2p       replayed          33         45    1.34  6.4e-16  
        41  41               mono     replayed         130        242    1.85  3.7e-16  
        45  3^2.5            2p       replayed          32         49    1.51  5.0e-16  
        47  47               mono     replayed         185        327    1.76  4.0e-16  
        49  7^2              2p       replayed          36         47    1.32  3.2e-16  
        51  3.17             2p       replayed          53        114    2.06  4.1e-16  
        55  5.11             2p       replayed          42         54    1.01  2.8e-16  flips differ 1.29x
        57  3.19             2p       replayed          65        136    2.09  4.1e-16  
        63  3^2.7            2p       replayed          46         69    1.46  4.2e-16  
        65  5.13             2p       replayed          53         67    1.25  6.3e-16  
        69  3.23             2p       replayed          93        199    2.00  4.1e-16  
        75  3.5^2            2p       replayed          58         78    1.36  4.4e-16  
        77  7.11             2p       replayed          60         76    1.25  3.6e-16  
        81  3^4              2p       replayed          62         86    1.39  4.3e-16  
        85  5.17             2p       replayed          86        191    2.22  4.7e-16  
        87  3.29             flat     replayed         162        332    1.90  6.6e-16  
        91  7.13             2p       replayed          77         91    1.18  5.0e-16  
        93  3.31             flat     replayed         183        377    1.85  4.4e-16  
        95  5.19             2p       replayed          98        228    2.05  4.4e-16  
        99  3^2.11           2p       replayed          81        113    1.38  4.7e-16  
       105  3.5.7            2p       replayed          85        113    1.33  5.0e-16  
       115  5.23             2p       replayed         147        339    1.75  4.6e-16  flips differ 1.31x
       117  3^2.13           2p       replayed         102        138    1.31  5.2e-16  
       119  7.17             2p       replayed         112        281    2.42  3.7e-16  
       121  11^2             2p       replayed         106        125    0.86  3.7e-16  flips differ 1.34x
       125  5^3              2p       replayed         103        133    1.15  4.8e-16  
       129  3.43             flat     replayed         334        806    2.38  4.1e-16  
       133  7.19             2p       replayed         140        328    2.24  4.5e-16  
       135  3^3.5            2p       replayed         114        156    1.37  3.2e-16  
       141  3.47             flat     replayed         525        988    1.88  5.4e-16  
       143  11.13            2p       replayed         129        149    1.15  4.8e-16  
       145  5.29             2p       replayed         253        556    2.19  4.4e-16  
       147  3.7^2            chain3   replayed         135        159    1.18  4.0e-16  
       153  3^2.17           2p       replayed         146        372    2.21  5.6e-16  
       155  5.31             2p       replayed         416        634    1.49  4.3e-16  
       161  7.23             2p       replayed         191        480    2.23  4.5e-16  
       165  3.5.11           2p       replayed         144        187    1.30  4.0e-16  
       169  13^2             2p       replayed         161        189    1.15  7.3e-16  
       171  3^2.19           2p       replayed         189        445    2.35  6.3e-16  
       175  5^2.7            chain3   replayed         152        197    1.29  3.7e-16  
       185  5.37             2p       replayed         478        959    2.00  3.2e-16  
       187  11.17            flat     replayed         245        450    1.83  6.5e-16  
       189  3^3.7            chain3   replayed         168        230    1.36  4.4e-16  
       195  3.5.13           chain3   replayed         177        236    1.00  7.1e-16  flips differ 1.33x
       203  7.29             2p       replayed         298        783    2.56  4.5e-16  
       205  5.41             2p       replayed         402       1233    2.86  5.7e-16  
       207  3^2.23           2p       replayed         277        634    1.98  5.0e-16  
       209  11.19            2p       replayed         219        529    2.31  5.1e-16  
       215  5.43             flat     replayed         493       1356    2.62  5.2e-16  
       217  7.31             2p       replayed         357        892    1.91  5.4e-16  flips differ 1.31x
       221  13.17            2p       replayed         228        548    2.39  5.5e-16  
       225  3^2.5^2          2p       replayed         206        269    1.27  4.8e-16  
       231  3.7.11           chain3   replayed         216        281    1.30  3.5e-16  
       235  5.47             2p       replayed         878       1671    1.87  3.4e-16  
       243  3^5              2p       replayed         225        298    1.32  6.5e-16  
       245  5.7^2            chain3   replayed         229        283    1.23  3.0e-16  
       247  13.19            2p       replayed         264        651    2.35  5.2e-16  
       253  11.23            2p       replayed         309        767    2.17  7.1e-16  
       255  3.5.17           2p       replayed         256        642    2.27  6.1e-16  
       259  7.37             2p       replayed         434       1343    3.09  4.1e-16  
       261  3^2.29           2p       replayed         384       1067    2.41  4.2e-16  
       273  3.7.13           2p       replayed         263        335    1.27  6.8e-16  
       275  5^2.11           chain3   replayed         250        334    1.32  3.4e-16  
       279  3^2.31           2p       replayed         473       1196    1.51  4.3e-16  flips differ 1.67x
       285  3.5.19           2p       replayed         329        762    2.20  4.4e-16  
       287  7.41             2p       replayed         540       1730    3.06  3.7e-16  
       289  17^2             2p       replayed         339       1121    2.55  7.9e-16  flips differ 1.29x
       297  3^3.11           chain3   replayed         297        393    1.31  5.2e-16  
       299  13.23            2p       replayed         378        940    2.17  5.0e-16  
       301  7.43             flat     replayed         702       1908    2.68  4.6e-16  
       315  3^2.5.7          chain3   replayed         292        400    1.36  7.9e-16  
       319  11.29            2p       replayed         556       1256    2.25  5.6e-16  
       323  17.19            2p       replayed         414       1305    3.11  6.8e-16  
       325  5^2.13           2p       replayed         328        410    1.25  7.5e-16  
       329  7.47             flat     replayed         912       2342    2.51  4.3e-16  
       333  3^2.37           2p       replayed         802       1780    2.22  5.9e-16  
       341  11.31            flat     replayed         736       1431    1.57  6.8e-16  
       343  7^3              chain3   replayed         348        400    1.14  4.9e-16  
       345  3.5.23           2p       replayed         476       1123    1.75  5.9e-16  flips differ 1.36x
       351  3^3.13           chain3   replayed         365        482    1.30  5.7e-16  
       357  3.7.17           chain3   replayed         413        918    1.37  5.6e-16  flips differ 1.63x
       361  19^2             2p       replayed         493       1544    3.12  5.0e-16  
       363  3.11^2           chain3   replayed         382        462    1.20  3.3e-16  
       369  3^2.41           2p       replayed        1010       2285    2.26  5.4e-16  
       375  3.5^3            chain3   replayed         348        473    1.33  6.8e-16  
       377  13.29            2p       replayed         630       1523    2.41  6.2e-16  
       385  5.7.11           chain3   replayed         376        480    1.27  5.6e-16  
       387  3^2.43           2p       replayed         759       2514    3.30  5.4e-16  
       391  17.23            2p       replayed         642       1798    2.42  6.7e-16  
       399  3.7.19           chain3   replayed         473       1089    2.18  6.2e-16  
       403  13.31            2p       replayed         712       1731    1.45  6.4e-16  flips differ 1.68x
       405  3^4.5            chain3   replayed         378        549    1.43  5.2e-16  
       407  11.37            2p       replayed         684       2140    3.08  4.1e-16  
       423  3^2.47           flat     replayed        1193       3069    2.17  7.0e-16  
       425  5^2.17           2p       replayed         457       1104    2.25  6.4e-16  
       429  3.11.13          chain3   replayed         468        556    1.18  5.2e-16  
       435  3.5.29           2p       replayed         638       1788    2.36  5.5e-16  
       437  19.23            2p       replayed         813       2137    2.62  5.5e-16  
       441  3^2.7^2          chain3   replayed         449        568    1.26  4.9e-16  
       451  11.41            2p       replayed        1192       2753    2.28  5.0e-16  
       455  5.7.13           chain3   replayed         461        584    1.23  6.0e-16  
       459  3^3.17           chain3   replayed         551       1224    2.20  5.1e-16  
       465  3.5.31           2p       replayed         817       2022    2.46  5.2e-16  
       473  11.43            2p       replayed         956       3072    2.86  6.0e-16  
       475  5^2.19           2p       replayed         533       1291    2.41  5.0e-16  
       481  13.37            2p       replayed         812       2591    3.10  5.5e-16  
       483  3.7.23           2p       replayed         698       1591    2.26  5.1e-16  
       493  17.29            2p       replayed         985       2712    2.39  5.0e-16  
       495  3^2.5.11         flat     replayed         596        669    0.99  5.8e-16  
       507  3.13^2           chain3   replayed         610        714    1.10  6.2e-16  
       513  3^3.19           2p       replayed         579       1474    2.54  5.4e-16  
       517  11.47            flat     replayed        1338       3715    2.66  6.4e-16  
       525  3.5^2.7          2p       replayed         545        724    1.24  6.4e-16  
       527  17.31            2p       replayed         906       3011    2.86  7.5e-16  
       529  23^2             2p       replayed         896       2858    3.13  6.5e-16  
       533  13.41            2p       replayed        1014       3301    3.24  6.4e-16  
       539  7^2.11           flat     replayed         681        722    1.04  5.4e-16  
       551  19.29            2p       replayed         954       3171    3.31  4.0e-16  
       555  3.5.37           2p       replayed         921       3018    3.14  6.0e-16  
       559  13.43            2p       replayed        1184       3642    3.04  5.6e-16  
       561  3.11.17          flat     replayed         899       1484    1.61  1.2e-15  
       567  3^4.7            2p       replayed         637        834    1.23  6.4e-16  
       575  5^2.23           2p       replayed         846       1901    1.95  5.8e-16  
       585  3^2.5.13         chain3   replayed         773        855    1.10  5.6e-16  
       589  19.31            2p       replayed        1069       3516    2.80  5.1e-16  
       595  5.7.17           chain3   replayed         889       1578    1.36  6.6e-16  flips differ 1.30x
       605  5.11^2           flat     replayed         764        830    1.08  6.1e-16  
       609  3.7.29           2p       replayed        1338       2523    1.75  5.8e-16  
       611  13.47            flat     replayed        1696       4462    2.18  7.8e-16  
       615  3.5.41           2p       replayed        1134       3855    3.40  6.3e-16  
       621  3^3.23           2p       replayed         881       2107    2.33  6.7e-16  
       625  5^4              2p       replayed         674        875    1.26  4.1e-16  
       627  3.11.19          flat     replayed         993       1754    1.75  6.9e-16  
       629  17.37            2p       replayed        1507       4269    2.82  8.6e-16  
       637  7^2.13           flat     replayed         925        875    0.80  6.3e-16  
       645  3.5.43           2p       replayed        1293       4245    3.13  5.5e-16  
       651  3.7.31           2p       replayed        1438       2856    1.52  5.2e-16  flips differ 1.30x
       663  3.13.17          flat     replayed        1048       1821    1.66  8.9e-16  
       665  5.7.19           flat     replayed         932       1869    1.67  5.1e-16  
       667  23.29            2p       replayed        1607       4173    2.35  6.1e-16  
       675  3^3.5^2          2p       replayed         769        998    1.24  5.9e-16  
       693  3^2.7.11         flat     replayed         969       1005    0.79  7.0e-16  flips differ 1.31x
       697  17.41            2p       replayed        1573       5334    3.37  8.6e-16  
       703  19.37            2p       replayed        1425       4941    3.16  4.6e-16  
       705  3.5.47           2p       replayed        1560       5194    3.30  6.6e-16  
       713  23.31            2p       replayed        1739       4585    2.21  9.4e-16  
       715  5.11.13          flat     replayed         948       1001    1.05  6.4e-16  
       725  5^2.29           flat     replayed        1293       3049    2.27  5.8e-16  
       729  3^6              2p       replayed         834       1091    1.10  5.1e-16  
       731  17.43            2p       replayed        1666       5803    3.47  6.0e-16  
       735  3.5.7^2          flat     replayed        1080       1041    0.96  6.1e-16  
       741  3.13.19          chain3   replayed        1373       2159    1.51  6.5e-16  
       759  3.11.23          flat     replayed        1402       2557    1.80  6.8e-16  
       765  3^2.5.17         flat     replayed        1096       2134    1.84  6.0e-16  
       775  5^2.31           2p       replayed        1267       3430    1.90  6.3e-16  flips differ 1.42x
       777  3.7.37           2p       replayed        1309       4260    3.23  5.5e-16  
       779  19.41            2p       replayed        1849       6111    2.96  4.7e-16  
       783  3^3.29           flat     replayed        1523       3411    2.21  4.8e-16  
       799  17.47            2p       replayed        1905       6909    3.24  5.6e-16  
       805  5.7.23           flat     replayed        1271       2691    1.52  6.6e-16  flips differ 1.40x
       817  19.43            flat     replayed        2155       6694    3.02  8.9e-16  
       819  3^2.7.13         flat     replayed        1131       1216    1.07  7.7e-16  
       825  3.5^2.11         chain3   replayed        1162       1221    1.02  4.4e-16  
       833  7^2.17           flat     replayed        1395       2235    1.55  7.7e-16  
       837  3^3.31           flat     replayed        1836       3814    2.07  7.1e-16  
       841  29^2             flat     replayed        2446       6005    2.29  1.2e-15  
       845  5.13^2           flat     replayed        1324       1269    0.87  9.8e-16  
       847  7.11^2           flat     replayed        1167       1197    0.95  4.3e-16  
       851  23.37            2p       replayed        2006       6454    3.12  6.1e-16  
       855  3^2.5.19         flat     replayed        1347       2504    1.82  8.2e-16  
       861  3.7.41           2p       replayed        1660       5426    3.08  4.6e-16  
       867  3.17^2           flat     replayed        1584       3566    2.06  1.1e-15  
       875  5^3.7            flat     replayed        1107       1300    1.17  4.5e-16  
       891  3^4.11           chain3   replayed        1295       1433    1.09  6.0e-16  
       893  19.47            flat     replayed        2339       7980    3.21  9.9e-16  
       897  3.13.23          flat     replayed        1730       3149    1.64  8.4e-16  
       899  29.31            2p       replayed        2576       6642    2.38  5.6e-16  
       903  3.7.43           flat     replayed        2327       5962    2.37  1.9e-15  
       925  5^2.37           2p       replayed        2382       5093    2.13  4.4e-16  
       931  7^2.19           flat     replayed        1518       2646    1.71  7.7e-16  
       935  5.11.17          flat     replayed        1418       2551    1.79  6.2e-16  
       943  23.41            2p       replayed        2793       7914    2.51  5.5e-16  
       945  3^3.5.7          chain3   replayed        1342       1461    1.05  6.1e-16  
       957  3.11.29          chain3   replayed        2364       4073    1.63  5.4e-16  
       961  31^2             flat     replayed        3154       7376    2.23  9.3e-16  
       969  3.17.19          flat     replayed        1788       4097    2.16  8.8e-16  
       975  3.5^2.13         flat     replayed        1361       1539    1.10  6.2e-16  
       987  3.7.47           2p       replayed        2171       7374    3.20  5.1e-16  
       989  23.43            flat     replayed        2840       8701    3.02  1.1e-15  
       999  3^3.37           2p       replayed        2534       5683    2.23  6.4e-16  
      1001  7.11.13          flat     replayed        1493       1453    0.79  7.1e-16  
      1015  5.7.29           flat     replayed        1871       4378    2.30  5.8e-16  
      1023  3.11.31          chain3   replayed        2316       4638    1.72  6.3e-16  
      1025  5^2.41           2p       replayed        1994       6571    3.18  5.0e-16  
      1029  3.7^3            flat     replayed        1502       1543    1.01  4.4e-16  
      1035  3^2.5.23         flat     replayed        1764       3673    1.94  6.7e-16  
      1045  5.11.19          flat     replayed        1832       3096    1.61  6.3e-16  
      1053  3^4.13           chain3   replayed        1611       1795    1.06  8.0e-16  
      1071  3^2.7.17         flat     replayed        1831       3096    1.63  5.7e-16  
      1073  29.37            2p       replayed        2897       9196    2.98  4.9e-16  
      1075  5^2.43           2p       replayed        2359       7217    2.99  5.6e-16  
      1081  23.47            2p       replayed        3960      10286    2.08  5.5e-16  
      1083  3.19^2           chain3   replayed        2043       5036    2.43  4.7e-16  
      1085  5.7.31           chain3   replayed        3822       4943    1.26  5.5e-16  
      1089  3^2.11^2         chain3   replayed        1588       1732    1.09  6.2e-16  
      1105  5.13.17          flat     replayed        1829       3237    1.75  8.0e-16  
      1107  3^3.41           2p       replayed        2225       7224    3.24  5.2e-16  
      1125  3^2.5^3          flat     replayed        1546       1826    1.17  7.8e-16  
      1127  7^2.23           chain3   replayed        2332       3894    1.45  6.4e-16  
      1131  3.13.29          chain3   replayed        2473       5093    1.67  7.4e-16  flips differ 1.26x
      1147  31.37            flat     replayed        3157      10150    2.79  8.4e-16  
      1155  3.5.7.11         chain3   replayed        1695       1830    0.95  4.2e-16  
      1161  3^3.43           2p       replayed        2412       8540    3.29  5.5e-16  
      1173  3.17.23          flat     replayed        2647       5938    2.16  1.1e-15  
      1175  5^2.47           flat     replayed        2796       9060    3.14  4.6e-16  
      1183  7.13^2           chain3   replayed        1847       1921    1.00  7.3e-16  
      1189  29.41            2p       replayed        4346      11192    2.51  5.0e-16  
      1197  3^2.7.19         flat     replayed        2209       3648    1.50  1.1e-15  
      1209  3.13.31          chain3   replayed        2764       5664    1.75  5.6e-16  
      1215  3^5.5            chain3   replayed        1729       2110    1.18  9.3e-16  
      1221  3.11.37          chain3   replayed        2911       6892    2.29  4.9e-16  
      1225  5^2.7^2          flat     replayed        1810       1949    1.06  4.5e-16  
      1235  5.13.19          flat     replayed        2156       3773    1.62  7.9e-16  
      1247  29.43            flat     replayed        3804      12103    3.06  2.8e-15  
      1265  5.11.23          flat     replayed        2607       4448    1.57  9.0e-16  
      1269  3^3.47           flat     replayed        3426       9652    2.79  7.1e-16  
      1271  31.41            flat     replayed        3592      12306    2.94  1.3e-15  
      1275  3.5^2.17         flat     replayed        2211       3691    1.39  7.2e-16  
      1287  3^2.11.13        chain3   replayed        1951       2143    1.09  5.9e-16  
      1295  5.7.37           flat     replayed        2625       7365    2.75  7.7e-16  
      1305  3^2.5.29         chain3   replayed        3014       5827    1.74  6.7e-16  
      1309  7.11.17          flat     replayed        2523       3705    1.44  1.1e-15  
      1311  3.19.23          chain3   replayed        2805       7042    1.66  6.0e-16  flips differ 1.55x
      1323  3^3.7^2          flat     replayed        2059       2201    1.02  8.3e-16  
      1331  11^3             chain3   replayed        2048       2081    0.85  4.5e-16  
      1333  31.43            2p       replayed        5317      13301    2.43  6.7e-16  
      1353  3.11.41          chain3   replayed        3450       8741    2.53  4.9e-16  
      1363  29.47            2p       replayed        6791      14303    1.86  4.5e-16  
      1365  3.5.7.13         chain3   replayed        2231       2268    1.01  6.0e-16  
      1369  37^2             2p       replayed        4445      13557    2.38  5.1e-16  flips differ 1.28x
      1375  5^3.11           chain3   replayed        2279       2253    0.89  4.2e-16  
      1377  3^4.17           chain3   replayed        2516       4114    1.63  8.2e-16  
      1395  3^2.5.31         flat     replayed        2896       6545    1.85  6.6e-16  
      1419  3.11.43          flat     replayed        3825       9596    2.36  6.0e-16  
      1421  7^2.29           chain3   replayed        3530       6159    1.64  5.7e-16  
      1425  3.5^2.19         flat     replayed        2420       4356    1.74  5.1e-16  
      1435  5.7.41           chain3   replayed        3980       9290    2.23  5.8e-16  
      1443  3.13.37          flat     replayed        3488       8276    2.30  6.5e-16  
      1445  5.17^2           chain3   replayed        2779       6103    2.17  6.4e-16  
      1449  3^2.7.23         flat     replayed        2737       5248    1.80  5.4e-16  
      1457  31.47            2p       replayed        5916      15694    1.91  5.9e-16  flips differ 1.39x
      1463  7.11.19          chain3   replayed        2537       4365    1.72  4.8e-16  
      1479  3.17.29          flat     replayed        3693       8497    1.80  1.2e-15  flips differ 1.27x
      1485  3^3.5.11         chain3   replayed        2314       2540    1.09  5.3e-16  
      1495  5.13.23          chain3   replayed        3088       5393    1.56  6.5e-16  
      1505  5.7.43           chain3   replayed        5376      10186    1.82  6.1e-16  
      1517  37.41            2p       replayed        5135      16321    3.17  8.5e-16  
      1519  7^2.31           flat     replayed        3770       6944    1.29  7.3e-16  flips differ 1.43x
      1521  3^2.13^2         chain3   replayed        2422       2840    0.96  7.1e-16  
      1539  3^4.19           chain3   replayed        2913       4986    1.56  7.1e-16  
      1547  7.13.17          flat     replayed        2873       4543    1.56  9.5e-16  
      1551  3.11.47          chain3   replayed        5775      11727    1.56  5.0e-16  flips differ 1.31x
      1573  11^2.13          chain3   replayed        2508       2539    0.92  5.6e-16  
      1575  3^2.5^2.7        flat     replayed        2407       2755    1.12  6.6e-16  
      1581  3.17.31          chain3   replayed        4800       9486    1.96  6.2e-16  
      1587  3.23^2           flat     replayed        4156       9056    1.76  1.2e-15  
      1591  37.43            2p       replayed        7383      17583    2.06  5.8e-16  
      1595  5.11.29          flat     replayed        4682       7424    1.46  6.7e-16  
      1599  3.13.41          chain3   replayed        4188      10535    2.40  5.7e-16  
      1615  5.17.19          chain3   replayed        3332       7082    1.36  6.1e-16  flips differ 1.56x
      1617  3.7^2.11         chain3   replayed        2764       2791    0.98  5.5e-16  
      1625  5^3.13           flat     replayed        2557       2844    0.58  6.6e-16  flips differ 1.90x
      1645  5.7.47           flat     replayed        4089      12426    2.90  5.6e-16  
      1653  3.19.29          chain3   replayed        5194       9994    1.88  5.4e-16  
      1665  3^2.5.37         chain3   replayed        4895       9649    1.95  5.6e-16  
      1677  3.13.43          chain3   replayed        4527      11547    2.46  6.8e-16  
      1681  41^2             2p       replayed        6201      19607    2.38  5.1e-16  flips differ 1.33x
      1683  3^2.11.17        chain3   replayed        2968       5068    1.69  7.8e-16  
      1701  3^5.7            chain3   replayed        2514       3153    1.18  4.7e-16  
      1705  5.11.31          chain3   replayed        4913       7919    1.58  5.0e-16  
      1715  5.7^3            flat     replayed        2785       2839    0.98  4.9e-16  
      1725  3.5^2.23         chain3   replayed        3697       6312    1.61  6.9e-16  
      1729  7.13.19          chain3   replayed        3731       5409    1.41  6.4e-16  
      1739  37.47            flat     replayed        5555      20664    3.60  1.9e-15  
      1755  3^3.5.13         chain3   replayed        2758       3260    1.18  8.8e-16  
      1763  41.43            2p       replayed        6783      21016    2.47  6.0e-16  flips differ 1.25x
      1767  3.19.31          chain3   replayed        4514      11150    1.60  4.8e-16  flips differ 1.55x
      1771  7.11.23          flat     replayed        3994       6307    1.57  8.1e-16  
      1785  3.5.7.17         chain3   replayed        2921       5373    1.64  6.5e-16  
      1805  5.19^2           flat     replayed        3953       8525    1.76  1.3e-15  
      1813  7^2.37           chain3   replayed        4818      10305    2.14  5.0e-16  
      1815  3.5.11^2         chain3   replayed        2816       3251    1.15  6.1e-16  
      1827  3^2.7.29         flat     replayed        4756       8397    1.51  6.0e-16  
      1833  3.13.47          flat     replayed        5244      14057    2.39  7.7e-16  
      1845  3^2.5.41         chain3   replayed        4730      12279    2.38  5.9e-16  
      1849  43^2             flat     replayed        5922      22686    3.76  1.6e-15  
      1859  11.13^2          chain3   replayed        3052       3295    1.07  7.3e-16  
      1863  3^4.23           chain3   replayed        3773       6964    1.49  6.2e-16  
      1875  3.5^4            flat     replayed        3408       3383    0.99  6.3e-16  
      1881  3^2.11.19        chain3   replayed        4607       5926    1.01  6.5e-16  flips differ 1.27x
      1885  5.13.29          chain3   replayed        4480       8462    1.42  6.9e-16  flips differ 1.33x
      1887  3.17.37          chain3   replayed        4870      13408    2.63  6.2e-16  
      1911  3.7^2.13         chain3   replayed        2942       3480    1.11  5.8e-16  
      1925  5^2.7.11         flat     replayed        3006       3335    1.06  6.7e-16  
      1935  3^2.5.43         flat     replayed        4542      13481    2.77  6.2e-16  
      1953  3^2.7.31         flat     replayed        5185       9267    1.68  7.6e-16  
      1955  5.17.23          chain3   replayed        4945       9693    1.57  6.2e-16  
      1989  3^2.13.17        chain3   replayed        3829       6166    1.47  6.9e-16  
      1995  3.5.7.19         chain3   replayed        4033       6316    1.25  5.1e-16  flips differ 1.26x
      2001  3.23.29          chain3   replayed        6188      13188    1.71  6.4e-16  
      2009  7^2.41           chain3   replayed        6393      13067    1.38  4.7e-16  flips differ 1.48x
      2015  5.13.31          flat     replayed        4645       9547    1.98  6.5e-16  
      2023  7.17^2           chain3   replayed        3645       8674    2.23  7.8e-16  
      2025  3^4.5^2          chain3   replayed        3286       3889    1.06  4.9e-16  
      2035  5.11.37          chain3   replayed        6175      11674    1.51  5.0e-16  flips differ 1.25x
      2057  11^2.17          flat     replayed        3965       6007    1.42  9.5e-16  
      2079  3^3.7.11         chain3   replayed        3069       3840    1.24  7.6e-16  
      2091  3.17.41          chain3   replayed        5470      16647    2.92  6.6e-16  
      2093  7.13.23          flat     replayed        4423       7650    1.55  7.1e-16  
      2107  7^2.43           chain3   replayed        7165      14348    2.00  5.1e-16  
      2109  3.19.37          flat     replayed        5641      15545    2.49  7.3e-16  
      2115  3^2.5.47         chain3   replayed        7778      16332    1.46  5.0e-16  flips differ 1.44x
      2125  5^3.17           chain3   replayed        3723       6339    1.55  7.5e-16  
      2139  3.23.31          flat     replayed        6251      14618    2.01  9.3e-16  
      2145  3.5.11.13        chain3   replayed        3406       4000    1.15  5.4e-16  
      2175  3.5^2.29         chain3   replayed        5016       9885    1.65  5.0e-16  
      2185  5.19.23          chain3   replayed        4943      11505    1.84  5.1e-16  flips differ 1.27x
      2187  3^7              chain3   replayed        3798       4516    1.14  6.0e-16  
      2193  3.17.43          chain3   replayed        7970      18165    2.27  7.4e-16  
      2197  13^3             flat     replayed        4150       4135    1.00  1.2e-15  
      2205  3^2.5.7^2        chain3   replayed        3624       4098    0.91  5.3e-16  flips differ 1.25x
      2209  47^2             flat     replayed        8149      30585    3.44  1.9e-15  
      2223  3^2.13.19        flat     replayed        4469       7242    1.54  1.2e-15  
      2233  7.11.29          chain3   replayed        5403       9876    1.60  6.3e-16  
      2255  5.11.41          chain3   replayed        5814      14762    2.54  6.0e-16  
      2261  7.17.19          flat     replayed        4612      10019    2.11  9.9e-16  
      2275  5^2.7.13         chain3   replayed        4114       4155    0.81  5.9e-16  
      2277  3^2.11.23        chain3   replayed        5716       8486    1.47  6.3e-16  
      2295  3^3.5.17         chain3   replayed        3958       7230    1.51  6.9e-16  
      2299  11^2.19          flat     replayed        4514       7044    1.55  1.2e-15  
      2303  7^2.47           chain3   replayed        8254      17453    1.60  6.0e-16  flips differ 1.32x
      2325  3.5^2.31         chain3   replayed        6773      11141    1.64  5.7e-16  
      2331  3^2.7.37         chain3   replayed        9216      13683    1.47  5.4e-16  
      2337  3.19.41          chain3   replayed        6776      19107    2.53  7.5e-16  
      2349  3^4.29           chain3   replayed        7734      11056    1.29  6.6e-16  
      2365  5.11.43          chain3   replayed        8509      16237    1.53  5.6e-16  
      2375  5^3.19           flat     replayed        4122       7488    1.79  6.8e-16  
      2387  7.11.31          chain3   replayed        6895      11131    1.33  4.7e-16  
      2397  3.17.47          flat     replayed        7955      21588    2.52  1.6e-15  
      2401  7^4              flat     replayed        4572       4089    0.89  6.8e-16  
      2405  5.13.37          chain3   replayed        5829      14071    2.40  6.3e-16  
      2415  3.5.7.23         chain3   replayed        5871       8945    1.25  6.2e-16  
      2431  11.13.17         chain3   replayed        4201       7355    1.74  6.0e-16  
      2451  3.19.43          flat     replayed        7682      21080    2.29  1.4e-15  
      2457  3^3.7.13         chain3   replayed        4195       4805    1.03  8.3e-16  
      2465  5.17.29          flat     replayed        6705      14469    1.73  1.6e-15  
      2475  3^2.5^2.11       chain3   replayed        4416       4700    1.00  7.2e-16  
      2499  3.7^2.17         chain3   replayed        4058       7562    1.60  6.6e-16  
      2511  3^4.31           flat     replayed        5712      12353    1.78  7.5e-16  
      2523  3.29^2           flat     replayed        8877      19019    2.11  1.4e-15  
      2527  7.19^2           chain3   replayed        5084      11978    2.31  5.3e-16  
      2535  3.5.13^2         chain3   replayed        4189       5115    1.22  7.3e-16  
      2541  3.7.11^2         flat     replayed        5455       4795    0.88  7.6e-16  
      2553  3.23.37          flat     replayed        7630      20262    2.45  8.7e-16  
      2565  3^3.5.19         chain3   replayed        4701       8457    1.54  8.1e-16  
      2583  3^2.7.41         chain3   replayed        6282      17245    2.53  6.2e-16  
      2585  5.11.47          chain3   replayed        9501      19805    2.08  5.3e-16  
      2601  3^2.17^2         chain3   replayed        4635      11502    2.35  7.7e-16  
      2625  3.5^3.7          flat     replayed        4856       4958    1.01  9.2e-16  
      2635  5.17.31          flat     replayed        7764      16063    1.97  8.0e-16  
      2639  7.13.29          chain3   replayed        6026      11931    1.75  7.5e-16  
      2645  5.23^2           flat     replayed        7387      15247    1.88  1.7e-15  
      2665  5.13.41          chain3   replayed        6872      17852    2.51  7.8e-16  
      2673  3^5.11           chain3   replayed        4637       5335    1.04  7.9e-16  
      2679  3.19.47          chain3   replayed       10168      24994    1.80  5.9e-16  flips differ 1.36x
      2691  3^2.13.23        chain3   replayed        5384      10290    1.88  6.2e-16  
      2695  5.7^2.11         flat     replayed        4558       5036    1.05  5.5e-16  
      2697  3.29.31          flat     replayed        8861      20993    2.33  2.8e-15  
      2709  3^2.7.43         chain3   replayed        9504      18923    1.54  6.0e-16  flips differ 1.30x
      2717  11.13.19         flat     replayed        5847       8600    1.27  8.5e-16  
      2737  7.17.23          flat     replayed        7273      13792    1.78  1.4e-15  
      2755  5.19.29          chain3   replayed        7661      16898    1.83  6.8e-16  
      2775  3.5^2.37         chain3   replayed       11079      16392    1.39  7.1e-16  
      2783  11^2.23          flat     replayed        6470      10123    1.42  1.4e-15  
      2793  3.7^2.19         chain3   replayed        4776       8967    1.76  6.3e-16  
      2795  5.13.43          chain3   replayed        7863      19614    2.46  5.9e-16  
      2805  3.5.11.17        chain3   replayed        5221       8659    1.65  6.0e-16  
      2821  7.13.31          chain3   replayed        6881      13539    1.75  6.5e-16  
      2829  3.23.41          flat     replayed        8361      24636    2.88  8.5e-16  
      2835  3^4.5.7          chain3   replayed        5050       5582    1.10  7.2e-16  
      2849  7.11.37          flat     replayed        6536      16510    2.17  6.0e-16  
      2871  3^2.11.29        flat     replayed        7149      13188    1.62  8.2e-16  
      2873  13^2.17          chain3   replayed        5255       8950    1.70  8.2e-16  
      2875  5^3.23           chain3   replayed        7356      10893    1.34  6.7e-16  
      2883  3.31^2           flat     replayed       11625      23133    1.95  1.9e-15  
      2907  3^2.17.19        flat     replayed        6904      13411    1.76  1.6e-15  
      2925  3^2.5^2.13       chain3   replayed        4978       5808    1.08  6.6e-16  
      2945  5.19.31          chain3   replayed        9469      19070    1.99  5.4e-16  
      2961  3^2.7.47         flat     replayed        8044      22912    2.74  6.3e-16  
      2967  3.23.43          chain3   replayed       12936      26957    1.85  5.6e-16  
      2975  5^2.7.17         flat     replayed        5763       9226    1.54  6.8e-16  
      2997  3^4.37           chain3   replayed        7369      17949    2.20  5.1e-16  
      3003  3.7.11.13        chain3   replayed        4959       5837    1.17  6.6e-16  
      3025  5^2.11^2         flat     replayed        5920       5905    0.95  8.7e-16  
      3045  3.5.7.29         flat     replayed        8093      13988    1.71  7.1e-16  
      3055  5.13.47          chain3   replayed       11519      23661    1.54  6.2e-16  flips differ 1.33x
      3059  7.19.23          flat     replayed        6916      16326    2.33  7.0e-16  
      3069  3^2.11.31        chain3   replayed        8963      14868    1.66  9.8e-16  
      3075  3.5^2.41         chain3   replayed        9946      20600    1.53  6.2e-16  flips differ 1.36x
      3087  3^2.7^3          chain3   replayed        5181       6004    1.14  5.0e-16  
      3105  3^3.5.23         chain3   replayed        7403      11975    1.61  5.3e-16  
      3125  5^5              flat     replayed        5742       6064    1.05  8.4e-16  
      3135  3.5.11.19        chain3   replayed        5614      10193    1.58  7.4e-16  
      3145  5.17.37          chain3   replayed        8259      22763    2.69  6.0e-16  
      3157  7.11.41          chain3   replayed        8364      20792    2.34  5.2e-16  
      3159  3^5.13           chain3   replayed        5574       6594    1.04  7.2e-16  
      3179  11.17^2          flat     replayed        6527      13931    1.93  1.1e-15  
      3185  5.7^2.13         flat     replayed        5564       6141    1.05  6.7e-16  
      3211  13^2.19          flat     replayed        6480      10481    1.61  1.0e-15  
      3213  3^3.7.17         chain3   replayed        5757      10211    1.77  9.3e-16  
      3219  3.29.37          flat     replayed       11053      28696    2.40  2.1e-15  
      3225  3.5^2.43         chain3   replayed       11255      22615    1.53  5.6e-16  flips differ 1.31x
      3243  3.23.47          chain3   replayed       14019      31681    2.16  5.8e-16  
      3249  3^2.19^2         flat     replayed        9219      16024    1.70  1.1e-15  
      3255  3.5.7.31         flat     replayed        8608      15791    1.63  6.8e-16  
      3267  3^3.11^2         flat     replayed        7381       6665    0.88  5.6e-16  
      3289  11.13.23         chain3   replayed        7987      12324    1.54  8.5e-16  
      3311  7.11.43          chain3   replayed       11539      22951    1.36  5.7e-16  flips differ 1.46x
      3315  3.5.13.17        flat     replayed        6932      10514    1.35  1.3e-15  
      3321  3^4.41           chain3   replayed        8874      22709    2.55  6.2e-16  
      3325  5^2.7.19         chain3   replayed        6579      10813    1.63  5.0e-16  
      3335  5.23.29          chain3   replayed       10991      22361    1.67  6.0e-16  
      3367  7.13.37          chain3   replayed        8156      19885    2.42  6.3e-16  
      3375  3^3.5^3          chain3   replayed        5259       6898    1.27  5.9e-16  
      3381  3.7^2.23         flat     replayed        8345      12728    1.48  7.7e-16  
      3393  3^2.13.29        flat     replayed        7852      15956    1.29  1.2e-15  flips differ 1.58x
      3441  3.31.37          flat     replayed       13469      31697    2.24  2.3e-15  
      3451  7.17.29          chain3   replayed       10799      20385    1.87  7.1e-16  
      3465  3^2.5.7.11       chain3   replayed        5715       6892    1.13  6.1e-16  
      3483  3^4.43           chain3   replayed       13076      24880    1.88  6.5e-16  
      3485  5.17.41          chain3   replayed       11159      28136    1.90  5.7e-16  flips differ 1.33x
      3509  11^2.29          chain3   replayed        8173      15893    1.93  6.4e-16  
      3515  5.19.37          chain3   replayed        9607      26451    2.48  6.3e-16  
      3519  3^2.17.23        flat     replayed        7464      18352    2.41  8.1e-16  
      3525  3.5^2.47         flat     replayed        8731      27465    2.80  5.6e-16  
      3549  3.7.13^2         chain3   replayed        5940       7354    1.22  7.0e-16  
      3553  11.17.19         chain3   replayed        6762      16011    2.07  6.0e-16  
      3565  5.23.31          flat     replayed       10758      24646    1.67  1.7e-15  flips differ 1.37x
      3567  3.29.41          flat     replayed       12279      34591    2.81  2.3e-15  
      3575  5^2.11.13        chain3   replayed        6367       7217    1.13  7.9e-16  
      3591  3^3.7.19         chain3   replayed        7261      12068    1.41  6.3e-16  
      3619  7.11.47          chain3   replayed       17846      27836    1.34  4.7e-16  
      3625  5^3.29           chain3   replayed        8506      16858    1.59  5.4e-16  
      3627  3^2.13.31        flat     replayed       10398      17906    1.68  1.0e-15  
      3645  3^6.5            chain3   replayed        6931       7772    1.09  6.6e-16  
      3655  5.17.43          chain3   replayed       11304      30614    2.69  5.3e-16  
      3663  3^2.11.37        flat     replayed       10928      21797    1.99  2.3e-15  
      3675  3.5^2.7^2        chain3   replayed        6631       7153    0.94  5.1e-16  
      3689  7.17.31          chain3   replayed       13555      22679    1.64  6.8e-16  
      3703  7.23^2           chain3   replayed       10029      21650    1.78  6.2e-16  
      3705  3.5.13.19        chain3   replayed        7291      12416    1.59  6.1e-16  
      3731  7.13.41          chain3   replayed        9894      25254    2.40  6.8e-16  
      3741  3.29.43          flat     replayed       13994      37337    2.64  3.5e-15  
      3751  11^2.31          flat     replayed        9924      17992    1.78  8.8e-16  
      3757  13.17^2          chain3   replayed        6962      17316    2.15  8.5e-16  
      3773  7^3.11           flat     replayed        7282       7178    0.89  7.6e-16  
      3795  3.5.11.23        chain3   replayed        7814      14439    1.82  6.2e-16  
      3807  3^4.47           chain3   replayed       19829      30195    1.52  6.4e-16  
      3813  3.31.41          flat     replayed       15308      38114    2.47  2.5e-15  
      3825  3^2.5^2.17       chain3   replayed        6692      12325    1.77  8.6e-16  
      3857  7.19.29          chain3   replayed        9618      23925    2.44  6.5e-16  
      3861  3^3.11.13        flat     replayed        8900       8168    0.89  9.9e-16  
      3875  5^3.31           chain3   replayed        9309      18890    1.66  7.1e-16  
      3885  3.5.7.37         chain3   replayed        9275      23090    2.42  5.0e-16  
      3887  13^2.23          flat     replayed        8078      14982    1.56  9.1e-16  
      3895  5.19.41          chain3   replayed       11985      32660    2.61  5.1e-16  
      3913  7.13.43          chain3   replayed       13920      28237    2.00  7.6e-16  
      3915  3^3.5.29         flat     replayed       11337      18773    1.24  6.5e-16  flips differ 1.36x
      3927  3.7.11.17        chain3   replayed        7666      12322    1.53  6.2e-16  
      3933  3^2.19.23        chain3   replayed        8955      22020    2.03  7.4e-16  
      3969  3^4.7^2          chain3   replayed        6615       8316    1.17  8.0e-16  
      3971  11.19^2          chain3   replayed        8025      19308    2.27  6.4e-16  
      3993  3.11^3           flat     replayed        9040       7684    0.80  8.3e-16  
      3995  5.17.47          chain3   replayed       21748      36419    1.60  6.7e-16  
      3999  3.31.43          chain3   replayed       15421      41160    2.03  7.3e-16  flips differ 1.31x
      4025  5^2.7.23         chain3   replayed       10375      15306    1.34  5.7e-16  
      4059  3^2.11.41        chain3   replayed       12640      27449    1.66  5.4e-16  flips differ 1.32x
      4085  5.19.43          flat     replayed       13352      35224    2.39  1.5e-15  
      4089  3.29.47          chain3   replayed       19075      43864    1.71  5.9e-16  flips differ 1.35x
      4095  3^2.5.7.13       chain3   replayed        7211       8501    1.10  6.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 chain3     204      0     12   1.06   1.56   2.40    1.54
 flat       158      4     18   0.99   1.75   2.77    1.70
 2p         132      0      2   1.25   2.24   3.17    2.08
 mono        14      0      0   1.34   1.73   2.25    1.74
 ALL        508      4     32   1.06   1.72   2.79    1.72
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   3   2.25      0      0
 8..31                 12   1.53      1      0
 32..127               29   1.43      1      0
 128..511              84   2.17      2      0
 512..2047            205   1.75     16      3
 2048..4095           175   1.66     12      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                         204   1.56     12   1.54
 flat                                           158   1.75     18   1.70
 2p                                             132   2.24      2   2.08
 mono                                            14   1.73      0   1.74
```


flip agreement: our two readings more than 25% apart at 49 of 508 cells.

worst 10: 1625 (flat 0.58), 1001 (flat 0.79), 693 (flat 0.79), 3993 (flat 0.80), 637 (flat 0.80), 2275 (chain3 0.81), 1331 (chain3 0.85), 121 (2p 0.86), 845 (flat 0.87), 2541 (flat 0.88)
best 5: 1849 (flat 3.76), 1739 (flat 3.60), 731 (2p 3.47), 2209 (flat 3.44), 615 (2p 3.40)
