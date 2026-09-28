# gauntlet report

run: `oddtail_after_2026-09-29`  contract file suffix: `(oop, T=1)`  cells: 508 listed, 508 benched, comparator: MKL

control cell: 14 readings, 0.653..1.085 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
         3  3                mono     replayed           4         11    3.01  1.5e-16  
         5  5                mono     replayed           5         12    2.25  7.5e-17  
         7  7                mono     replayed           7         12    1.88  2.6e-16  
         9  3^2              mono     replayed           8         13    1.59  4.2e-16  
        11  11               mono     replayed          11         14    1.27  1.2e-16  
        13  13               mono     replayed          15         15    0.99  5.0e-16  
        15  3.5              2p       replayed          15         14    0.95  3.3e-16  
        17  17               mono     replayed          19         40    2.09  2.0e-16  
        19  19               mono     replayed          23         45    1.96  4.6e-16  
        21  3.7              2p       replayed          17         25    1.47  3.3e-16  
        23  23               mono     replayed          33         67    2.03  2.5e-16  
        25  5^2              2p       replayed          19         26    1.37  1.8e-16  
        27  3^3              flat     replayed          28         32    1.14  3.8e-16  
        29  29               mono     replayed          51        107    2.09  2.8e-16  
        31  31               mono     replayed          60        121    2.01  3.3e-16  
        33  3.11             2p       replayed          26         36    1.37  2.7e-16  
        35  5.7              2p       replayed          25         36    1.43  2.6e-16  
        37  37               mono     replayed          79        185    2.32  4.8e-16  
        39  3.13             2p       replayed          33         45    1.32  6.4e-16  
        41  41               mono     replayed          98        242    2.47  3.7e-16  
        45  3^2.5            2p       replayed          32         49    1.51  5.0e-16  
        47  47               mono     replayed         132        327    2.47  4.0e-16  
        49  7^2              2p       replayed          36         47    1.31  3.2e-16  
        51  3.17             2p       replayed          54        115    2.14  4.1e-16  
        55  5.11             2p       replayed          42         54    1.27  2.8e-16  
        57  3.19             2p       replayed          63        136    2.15  4.1e-16  
        63  3^2.7            2p       replayed          46         67    1.45  4.2e-16  
        65  5.13             2p       replayed          53         67    1.25  6.3e-16  
        69  3.23             2p       replayed          98        199    2.02  4.1e-16  
        75  3.5^2            2p       replayed          58         78    1.31  4.4e-16  
        77  7.11             2p       replayed          60         75    1.25  3.6e-16  
        81  3^4              2p       replayed          63         86    1.38  4.3e-16  
        85  5.17             2p       replayed          82        190    2.19  4.7e-16  
        87  3.29             flat     replayed         153        332    2.15  6.6e-16  
        91  7.13             2p       replayed          75         91    1.21  5.0e-16  
        93  3.31             flat     replayed         175        376    2.01  4.4e-16  
        95  5.19             2p       replayed         110        228    1.27  4.4e-16  flips differ 1.63x
        99  3^2.11           2p       replayed          81        113    1.38  4.7e-16  
       105  3.5.7            2p       replayed          85        113    1.33  5.0e-16  
       115  5.23             2p       replayed         146        339    1.95  4.6e-16  
       117  3^2.13           2p       replayed         101        138    1.36  5.2e-16  
       119  7.17             2p       replayed         112        278    2.22  3.7e-16  
       121  11^2             2p       replayed         106        123    1.16  3.7e-16  
       125  5^3              2p       replayed         103        133    1.29  4.8e-16  
       129  3.43             flat     replayed         253        801    3.16  4.1e-16  
       133  7.19             2p       replayed         144        326    2.23  4.5e-16  
       135  3^3.5            2p       replayed         114        156    1.35  3.2e-16  
       141  3.47             flat     replayed         293        986    3.18  5.4e-16  
       143  11.13            2p       replayed         130        162    1.17  4.8e-16  
       145  5.29             2p       replayed         221        588    2.44  4.4e-16  
       147  3.7^2            chain3   replayed         135        174    0.87  4.0e-16  flips differ 1.47x
       153  3^2.17           2p       replayed         146        372    2.46  5.6e-16  
       155  5.31             2p       replayed         344        631    1.49  4.3e-16  
       161  7.23             2p       replayed         194        478    2.00  4.5e-16  
       165  3.5.11           2p       replayed         144        189    1.30  4.0e-16  
       169  13^2             2p       replayed         169        191    0.92  7.3e-16  
       171  3^2.19           2p       replayed         199        444    2.22  6.3e-16  
       175  5^2.7            chain3   replayed         152        197    1.30  3.7e-16  
       185  5.37             2p       replayed         488        958    1.63  3.2e-16  
       187  11.17            flat     replayed         246        452    1.53  6.5e-16  
       189  3^3.7            chain3   replayed         168        230    1.37  4.4e-16  
       195  3.5.13           chain3   replayed         175        236    1.07  7.1e-16  flips differ 1.25x
       203  7.29             2p       replayed         371        783    1.91  4.5e-16  
       205  5.41             2p       replayed         349       1234    3.40  5.7e-16  
       207  3^2.23           2p       replayed         276        636    1.98  5.0e-16  
       209  11.19            2p       replayed         220        526    2.34  5.1e-16  
       215  5.43             flat     replayed         424       1360    3.17  5.2e-16  
       217  7.31             2p       replayed         468        890    1.63  5.4e-16  
       221  13.17            2p       replayed         229        550    2.36  5.5e-16  
       225  3^2.5^2          2p       replayed         206        268    1.29  4.8e-16  
       231  3.7.11           chain3   replayed         214        276    1.28  3.5e-16  
       235  5.47             2p       replayed         878       1669    1.65  3.4e-16  
       243  3^5              2p       replayed         225        308    1.32  6.5e-16  
       245  5.7^2            chain3   replayed         228        302    1.23  3.0e-16  
       247  13.19            2p       replayed         269        646    2.08  5.2e-16  
       253  11.23            2p       replayed         315        764    2.38  7.1e-16  
       255  3.5.17           2p       replayed         256        641    2.50  6.1e-16  
       259  7.37             2p       replayed         396       1342    3.39  4.1e-16  
       261  3^2.29           2p       replayed         384       1065    2.40  4.2e-16  
       273  3.7.13           2p       replayed         266        335    1.25  6.8e-16  
       275  5^2.11           chain3   replayed         249        333    1.32  3.4e-16  
       279  3^2.31           2p       replayed         474       1196    1.95  4.3e-16  flips differ 1.29x
       285  3.5.19           2p       replayed         329        761    1.81  4.4e-16  flips differ 1.28x
       287  7.41             2p       replayed         475       1734    3.60  3.7e-16  
       289  17^2             2p       replayed         356       1111    3.08  7.9e-16  
       297  3^3.11           chain3   replayed         297        405    1.32  5.2e-16  
       299  13.23            2p       replayed         380        934    2.41  5.0e-16  
       301  7.43             flat     replayed         572       1907    3.10  4.6e-16  
       315  3^2.5.7          chain3   replayed         292        396    1.36  7.9e-16  
       319  11.29            2p       replayed         483       1255    2.58  5.6e-16  
       323  17.19            2p       replayed         416       1294    2.91  6.8e-16  
       325  5^2.13           2p       replayed         331        410    1.24  7.5e-16  
       329  7.47             flat     replayed         649       2340    3.60  4.3e-16  
       333  3^2.37           2p       replayed         805       1779    1.62  5.9e-16  flips differ 1.37x
       341  11.31            flat     replayed         781       1425    1.81  6.8e-16  
       343  7^3              chain3   replayed         346        400    1.16  4.9e-16  
       345  3.5.23           2p       replayed         475       1115    2.15  5.9e-16  
       351  3^3.13           chain3   replayed         360        480    1.32  5.7e-16  
       357  3.7.17           chain3   replayed         409        921    2.19  5.6e-16  
       361  19^2             2p       replayed         483       1532    2.92  5.0e-16  
       363  3.11^2           chain3   replayed         386        457    1.18  3.3e-16  
       369  3^2.41           2p       replayed        1310       2278    1.53  5.4e-16  
       375  3.5^3            chain3   replayed         348        475    1.15  6.8e-16  
       377  13.29            2p       replayed         636       1522    1.47  6.2e-16  flips differ 1.63x
       385  5.7.11           chain3   replayed         374        481    1.28  5.6e-16  
       387  3^2.43           2p       replayed         677       2508    3.57  5.4e-16  
       391  17.23            2p       replayed         754       1779    2.24  6.7e-16  
       399  3.7.19           chain3   replayed         470       1086    2.19  6.2e-16  
       403  13.31            2p       replayed         924       1721    1.55  6.4e-16  
       405  3^4.5            chain3   replayed         376        548    1.44  5.2e-16  
       407  11.37            2p       replayed         639       2143    3.35  4.1e-16  
       423  3^2.47           flat     replayed         877       3070    3.42  7.0e-16  
       425  5^2.17           2p       replayed         471       1083    2.20  6.4e-16  
       429  3.11.13          chain3   replayed         512        558    0.82  5.2e-16  flips differ 1.32x
       435  3.5.29           2p       replayed         772       1783    2.29  5.5e-16  
       437  19.23            2p       replayed         654       2118    2.80  5.5e-16  
       441  3^2.7^2          chain3   replayed         456        567    1.24  4.9e-16  
       451  11.41            2p       replayed        1197       2749    2.29  5.0e-16  
       455  5.7.13           chain3   replayed         460        588    1.27  6.0e-16  
       459  3^3.17           chain3   replayed         566       1208    2.09  5.1e-16  
       465  3.5.31           2p       replayed         814       2016    1.91  5.2e-16  flips differ 1.30x
       473  11.43            2p       replayed         824       3032    3.40  6.0e-16  
       475  5^2.19           2p       replayed         535       1288    2.29  5.0e-16  
       481  13.37            2p       replayed         762       2582    3.39  5.5e-16  
       483  3.7.23           2p       replayed         699       1586    2.08  5.1e-16  
       493  17.29            2p       replayed         868       2672    3.07  5.0e-16  
       495  3^2.5.11         flat     replayed         596        667    0.97  5.8e-16  
       507  3.13^2           chain3   replayed         610        716    1.12  6.2e-16  
       513  3^3.19           2p       replayed         583       1474    2.45  5.4e-16  
       517  11.47            flat     replayed        1074       3714    3.20  6.4e-16  
       525  3.5^2.7          2p       replayed         579        720    1.19  6.4e-16  
       527  17.31            2p       replayed        1028       2996    2.89  7.5e-16  
       529  23^2             2p       replayed        1160       2800    2.13  6.5e-16  
       533  13.41            2p       replayed         979       3313    3.34  6.4e-16  
       539  7^2.11           flat     replayed         682        722    1.01  5.4e-16  
       551  19.29            2p       replayed         999       3132    3.11  4.0e-16  
       555  3.5.37           2p       replayed         877       3019    3.43  6.0e-16  
       559  13.43            2p       replayed        1012       3700    3.54  5.6e-16  
       561  3.11.17          flat     replayed         875       1481    1.67  1.2e-15  
       567  3^4.7            2p       replayed         597        835    1.39  6.4e-16  
       575  5^2.23           2p       replayed         845       1895    2.22  5.8e-16  
       585  3^2.5.13         chain3   replayed         768        858    1.10  5.6e-16  
       589  19.31            2p       replayed        1115       3500    2.87  5.1e-16  
       595  5.7.17           chain3   replayed         848       1576    1.48  6.6e-16  flips differ 1.25x
       605  5.11^2           flat     replayed         762        831    1.09  6.1e-16  
       609  3.7.29           2p       replayed        1162       2520    1.52  5.8e-16  flips differ 1.42x
       611  13.47            flat     replayed        1315       4457    3.31  7.8e-16  
       615  3.5.41           2p       replayed        1049       3856    3.64  6.3e-16  
       621  3^3.23           2p       replayed         928       2096    2.08  6.7e-16  
       625  5^4              2p       replayed         686        872    1.03  4.1e-16  
       627  3.11.19          flat     replayed        1030       1754    1.53  6.9e-16  
       629  17.37            2p       replayed        1964       4281    2.08  8.6e-16  
       637  7^2.13           flat     replayed         947        874    0.83  6.3e-16  
       645  3.5.43           2p       replayed        1137       4244    3.67  5.5e-16  
       651  3.7.31           2p       replayed        1781       2848    1.52  5.2e-16  
       663  3.13.17          flat     replayed        1051       1811    1.70  8.9e-16  
       665  5.7.19           flat     replayed         939       1903    1.97  5.1e-16  
       667  23.29            2p       replayed        1532       4113    2.36  6.1e-16  
       675  3^3.5^2          2p       replayed         749        995    1.02  5.9e-16  flips differ 1.30x
       693  3^2.7.11         flat     replayed         907        999    1.10  7.0e-16  
       697  17.41            2p       replayed        1326       5285    3.29  8.6e-16  
       703  19.37            2p       replayed        1286       4923    3.81  4.6e-16  
       705  3.5.47           2p       replayed        1316       5181    3.91  6.6e-16  
       713  23.31            2p       replayed        1785       4570    1.72  9.4e-16  flips differ 1.49x
       715  5.11.13          flat     replayed         957       1007    1.05  6.4e-16  
       725  5^2.29           flat     replayed        1262       3037    1.97  5.8e-16  
       729  3^6              2p       replayed         824       1089    1.31  5.1e-16  
       731  17.43            2p       replayed        1527       5775    3.73  6.0e-16  
       735  3.5.7^2          flat     replayed        1033       1040    0.99  6.1e-16  
       741  3.13.19          chain3   replayed        1373       2146    1.35  6.5e-16  
       759  3.11.23          flat     replayed        1402       2548    1.59  6.8e-16  
       765  3^2.5.17         flat     replayed        1084       2133    1.87  6.0e-16  
       775  5^2.31           2p       replayed        1243       3425    2.37  6.3e-16  
       777  3.7.37           2p       replayed        1329       4256    3.20  5.5e-16  
       779  19.41            2p       replayed        1541       6120    3.80  4.7e-16  
       783  3^3.29           flat     replayed        1741       3401    1.88  4.8e-16  
       799  17.47            2p       replayed        1748       6912    3.88  5.6e-16  
       805  5.7.23           flat     replayed        1262       2682    2.08  6.6e-16  
       817  19.43            flat     replayed        1818       6668    3.55  8.9e-16  
       819  3^2.7.13         flat     replayed        1122       1221    1.07  7.7e-16  
       825  3.5^2.11         chain3   replayed        1201       1220    0.97  4.4e-16  
       833  7^2.17           flat     replayed        1285       2235    1.73  7.7e-16  
       837  3^3.31           flat     replayed        1803       3817    1.80  7.1e-16  
       841  29^2             flat     replayed        2537       5999    2.21  1.2e-15  
       845  5.13^2           flat     replayed        1336       1263    0.94  9.8e-16  
       847  7.11^2           flat     replayed        1172       1201    0.98  4.3e-16  
       851  23.37            2p       replayed        1812       6386    3.46  6.1e-16  
       855  3^2.5.19         flat     replayed        1320       2504    1.83  8.2e-16  
       861  3.7.41           2p       replayed        1514       5426    3.37  4.6e-16  
       867  3.17^2           flat     replayed        1537       3522    2.19  1.1e-15  
       875  5^3.7            flat     replayed        1111       1294    1.16  4.5e-16  
       891  3^4.11           chain3   replayed        1236       1415    1.11  6.0e-16  
       893  19.47            flat     replayed        2213       7990    3.23  9.9e-16  
       897  3.13.23          flat     replayed        1766       3083    1.73  8.4e-16  
       899  29.31            2p       replayed        2248       6641    2.23  5.6e-16  flips differ 1.33x
       903  3.7.43           flat     replayed        2107       5983    2.40  1.9e-15  
       925  5^2.37           2p       replayed        3355       5101    1.50  4.4e-16  
       931  7^2.19           flat     replayed        1510       2647    1.71  7.7e-16  
       935  5.11.17          flat     replayed        1507       2546    1.66  6.2e-16  
       943  23.41            2p       replayed        2129       7896    3.70  5.5e-16  
       945  3^3.5.7          chain3   replayed        1330       1456    1.08  6.1e-16  
       957  3.11.29          chain3   replayed        2336       4044    1.67  5.4e-16  
       961  31^2             flat     replayed        2538       7371    2.24  9.3e-16  flips differ 1.30x
       969  3.17.19          flat     replayed        1787       4099    2.02  8.8e-16  
       975  3.5^2.13         flat     replayed        1360       1496    1.06  6.2e-16  
       987  3.7.47           2p       replayed        1912       7355    3.69  5.1e-16  
       989  23.43            flat     replayed        2906       8662    2.98  1.1e-15  
       999  3^3.37           2p       replayed        2585       5668    1.58  6.4e-16  flips differ 1.39x
      1001  7.11.13          flat     replayed        1524       1457    0.82  7.1e-16  
      1015  5.7.29           flat     replayed        1907       4358    2.11  5.8e-16  
      1023  3.11.31          chain3   replayed        2535       4630    1.80  6.3e-16  
      1025  5^2.41           2p       replayed        2019       6570    3.13  5.0e-16  
      1029  3.7^3            flat     replayed        1557       1543    0.89  4.4e-16  
      1035  3^2.5.23         flat     replayed        1916       3674    1.73  6.7e-16  
      1045  5.11.19          flat     replayed        1741       3086    1.72  6.3e-16  
      1053  3^4.13           chain3   replayed        1631       1796    1.08  8.0e-16  
      1071  3^2.7.17         flat     replayed        1729       3101    1.71  5.7e-16  
      1073  29.37            2p       replayed        3871       9209    2.25  4.9e-16  
      1075  5^2.43           2p       replayed        2071       7210    2.96  5.6e-16  
      1081  23.47            2p       replayed        4034      10269    1.96  5.5e-16  flips differ 1.30x
      1083  3.19^2           chain3   replayed        2374       4953    2.06  4.7e-16  
      1085  5.7.31           chain3   replayed        3101       4923    1.58  5.5e-16  
      1089  3^2.11^2         chain3   replayed        1590       1724    1.06  6.2e-16  
      1105  5.13.17          flat     replayed        1857       3200    1.57  8.0e-16  
      1107  3^3.41           2p       replayed        2039       7235    3.27  5.2e-16  
      1125  3^2.5^3          flat     replayed        1576       1797    1.13  7.8e-16  
      1127  7^2.23           chain3   replayed        2582       3904    1.41  6.4e-16  
      1131  3.13.29          chain3   replayed        2584       4968    1.79  7.4e-16  
      1147  31.37            flat     replayed        2901      10136    2.67  8.4e-16  flips differ 1.31x
      1155  3.5.7.11         chain3   replayed        1702       1838    1.08  4.2e-16  
      1161  3^3.43           2p       replayed        2396       7938    3.28  5.5e-16  
      1173  3.17.23          flat     replayed        2523       5685    2.13  1.1e-15  
      1175  5^2.47           flat     replayed        2557       8788    3.04  4.6e-16  
      1183  7.13^2           chain3   replayed        1856       1911    1.02  7.3e-16  
      1189  29.41            2p       replayed        4007      11181    2.52  5.0e-16  
      1197  3^2.7.19         flat     replayed        2260       3658    1.34  1.1e-15  
      1209  3.13.31          chain3   replayed        3231       5622    1.71  5.6e-16  
      1215  3^5.5            chain3   replayed        1738       2093    1.20  9.3e-16  
      1221  3.11.37          chain3   replayed        2810       6877    2.23  4.9e-16  
      1225  5^2.7^2          flat     replayed        1805       1931    1.01  4.5e-16  
      1235  5.13.19          flat     replayed        2166       3767    1.67  7.9e-16  
      1247  29.43            flat     replayed        3603      12098    2.87  2.8e-15  
      1265  5.11.23          flat     replayed        2477       4427    1.69  9.0e-16  
      1269  3^3.47           flat     replayed        3252       9629    2.73  7.1e-16  
      1271  31.41            flat     replayed        3408      12316    3.51  1.3e-15  
      1275  3.5^2.17         flat     replayed        2232       3693    1.59  7.2e-16  
      1287  3^2.11.13        chain3   replayed        2009       2138    1.06  5.9e-16  
      1295  5.7.37           flat     replayed        2582       7302    2.42  7.7e-16  
      1305  3^2.5.29         chain3   replayed        3154       5787    1.74  6.7e-16  
      1309  7.11.17          flat     replayed        2434       3708    1.17  1.1e-15  flips differ 1.31x
      1311  3.19.23          chain3   replayed        4200       6776    1.26  6.0e-16  flips differ 1.28x
      1323  3^3.7^2          flat     replayed        2060       2178    1.04  8.3e-16  
      1331  11^3             chain3   replayed        2063       2084    1.01  4.5e-16  
      1333  31.43            2p       replayed        4127      13306    3.22  6.7e-16  
      1353  3.11.41          chain3   replayed        3236       8723    2.26  4.9e-16  
      1363  29.47            2p       replayed        7060      14255    1.98  4.5e-16  
      1365  3.5.7.13         chain3   replayed        2249       2266    0.98  6.0e-16  
      1369  37^2             2p       replayed        4353      13555    2.23  5.1e-16  flips differ 1.40x
      1375  5^3.11           chain3   replayed        2546       2245    0.81  4.2e-16  
      1377  3^4.17           chain3   replayed        2849       4109    1.44  8.2e-16  
      1395  3^2.5.31         flat     replayed        2876       6519    2.01  6.6e-16  
      1419  3.11.43          flat     replayed        3440       9617    2.48  6.0e-16  
      1421  7^2.29           chain3   replayed        3623       6153    1.61  5.7e-16  
      1425  3.5^2.19         flat     replayed        2196       4363    1.93  5.1e-16  
      1435  5.7.41           chain3   replayed        3690       9273    2.51  5.8e-16  
      1443  3.13.37          flat     replayed        3441       8290    2.26  6.5e-16  
      1445  5.17^2           chain3   replayed        2825       6079    1.90  6.4e-16  
      1449  3^2.7.23         flat     replayed        2877       5257    1.79  5.4e-16  
      1457  31.47            2p       replayed        5493      15683    1.83  5.9e-16  flips differ 1.56x
      1463  7.11.19          chain3   replayed        3053       4369    0.93  4.8e-16  flips differ 1.54x
      1479  3.17.29          flat     replayed        3984       8473    2.09  1.2e-15  
      1485  3^3.5.11         chain3   replayed        2208       2545    1.09  5.3e-16  
      1495  5.13.23          chain3   replayed        3218       5381    1.64  6.5e-16  
      1505  5.7.43           chain3   replayed        5339      10208    1.90  6.1e-16  
      1517  37.41            2p       replayed        5025      16302    2.29  8.5e-16  flips differ 1.42x
      1519  7^2.31           flat     replayed        4143       6936    1.62  7.3e-16  
      1521  3^2.13^2         chain3   replayed        2425       2852    1.17  7.1e-16  
      1539  3^4.19           chain3   replayed        3144       4958    1.54  7.1e-16  
      1547  7.13.17          flat     replayed        2652       4544    1.31  9.5e-16  flips differ 1.31x
      1551  3.11.47          chain3   replayed        5768      11694    1.41  5.0e-16  flips differ 1.44x
      1573  11^2.13          chain3   replayed        2466       2532    1.00  5.6e-16  
      1575  3^2.5^2.7        flat     replayed        2418       2741    1.13  6.6e-16  
      1581  3.17.31          chain3   replayed        4776       9445    1.56  6.2e-16  flips differ 1.27x
      1587  3.23^2           flat     replayed        4621       8965    1.78  1.2e-15  
      1591  37.43            2p       replayed        5606      17558    2.35  5.8e-16  flips differ 1.33x
      1595  5.11.29          flat     replayed        3974       6993    1.60  6.7e-16  
      1599  3.13.41          chain3   replayed        3875      10481    2.66  5.7e-16  
      1615  5.17.19          chain3   replayed        3344       7054    1.82  6.1e-16  
      1617  3.7^2.11         chain3   replayed        2797       2783    0.85  5.5e-16  
      1625  5^3.13           flat     replayed        2541       2789    1.07  6.6e-16  
      1645  5.7.47           flat     replayed        4246      12410    2.86  5.6e-16  
      1653  3.19.29          chain3   replayed        4779       9923    2.07  5.4e-16  
      1665  3^2.5.37         chain3   replayed        4921       9643    1.52  5.6e-16  flips differ 1.29x
      1677  3.13.43          chain3   replayed        4153      11540    2.67  6.8e-16  
      1681  41^2             2p       replayed        5907      19589    2.58  5.1e-16  flips differ 1.28x
      1683  3^2.11.17        chain3   replayed        2727       5021    1.70  7.8e-16  
      1701  3^5.7            chain3   replayed        2569       3166    1.18  4.7e-16  
      1705  5.11.31          chain3   replayed        6174       7891    1.26  5.0e-16  
      1715  5.7^3            flat     replayed        2785       2828    1.00  4.9e-16  
      1725  3.5^2.23         chain3   replayed        3692       6303    1.69  6.9e-16  
      1729  7.13.19          chain3   replayed        3260       5335    1.42  6.4e-16  
      1739  37.47            flat     replayed        5538      20561    3.57  1.9e-15  
      1755  3^3.5.13         chain3   replayed        2768       3265    1.13  8.8e-16  
      1763  41.43            2p       replayed        6687      21033    2.51  6.0e-16  flips differ 1.25x
      1767  3.19.31          chain3   replayed        4826      11108    2.29  4.8e-16  
      1771  7.11.23          flat     replayed        4049       6285    1.53  8.1e-16  
      1785  3.5.7.17         chain3   replayed        3062       5352    1.68  6.5e-16  
      1805  5.19^2           flat     replayed        3940       8383    1.89  1.3e-15  
      1813  7^2.37           chain3   replayed        4480      10311    2.26  5.0e-16  
      1815  3.5.11^2         chain3   replayed        2795       3239    1.14  6.1e-16  
      1827  3^2.7.29         flat     replayed        4240       8209    1.72  6.0e-16  
      1833  3.13.47          flat     replayed        4872      14012    2.66  7.7e-16  
      1845  3^2.5.41         chain3   replayed        4410      12193    2.59  5.9e-16  
      1849  43^2             flat     replayed        5385      22671    3.78  1.6e-15  
      1859  11.13^2          chain3   replayed        3098       3303    1.06  7.3e-16  
      1863  3^4.23           chain3   replayed        4524       6946    1.50  6.2e-16  
      1875  3.5^4            flat     replayed        3408       3376    0.98  6.3e-16  
      1881  3^2.11.19        chain3   replayed        3371       5929    1.51  6.5e-16  
      1885  5.13.29          chain3   replayed        4515       8481    1.40  6.9e-16  flips differ 1.34x
      1887  3.17.37          chain3   replayed        4468      13358    2.89  6.2e-16  
      1911  3.7^2.13         chain3   replayed        2999       3456    0.89  5.8e-16  flips differ 1.29x
      1925  5^2.7.11         flat     replayed        3034       3365    1.10  6.7e-16  
      1935  3^2.5.43         flat     replayed        4333      13412    3.01  6.2e-16  
      1953  3^2.7.31         flat     replayed        5100       9250    1.74  7.6e-16  
      1955  5.17.23          chain3   replayed        5185       9678    1.44  6.2e-16  flips differ 1.29x
      1989  3^2.13.17        chain3   replayed        3513       6133    1.74  6.9e-16  
      1995  3.5.7.19         chain3   replayed        3530       6296    1.21  5.1e-16  flips differ 1.48x
      2001  3.23.29          chain3   replayed        7129      13056    1.45  6.4e-16  flips differ 1.26x
      2009  7^2.41           chain3   replayed        6509      13040    1.52  4.7e-16  flips differ 1.32x
      2015  5.13.31          flat     replayed        4558       9515    1.86  6.5e-16  
      2023  7.17^2           chain3   replayed        3875       8581    1.82  7.8e-16  
      2025  3^4.5^2          chain3   replayed        3380       3853    1.07  4.9e-16  
      2035  5.11.37          chain3   replayed        6153      11642    1.42  5.0e-16  flips differ 1.33x
      2057  11^2.17          flat     replayed        3921       5972    1.38  9.5e-16  
      2079  3^3.7.11         chain3   replayed        3060       3848    1.17  7.6e-16  
      2091  3.17.41          chain3   replayed        5438      16552    2.93  6.6e-16  
      2093  7.13.23          flat     replayed        5025       7595    1.39  7.1e-16  
      2107  7^2.43           chain3   replayed        9990      14333    1.38  5.1e-16  
      2109  3.19.37          flat     replayed        5428      15423    2.51  7.3e-16  
      2115  3^2.5.47         chain3   replayed        7759      16257    1.58  5.0e-16  flips differ 1.33x
      2125  5^3.17           chain3   replayed        3862       6330    1.54  7.5e-16  
      2139  3.23.31          flat     replayed        6881      14508    2.07  9.3e-16  
      2145  3.5.11.13        chain3   replayed        3394       4021    1.17  5.4e-16  
      2175  3.5^2.29         chain3   replayed        5514       9858    1.55  5.0e-16  
      2185  5.19.23          chain3   replayed        6509      11440    1.75  5.1e-16  
      2187  3^7              chain3   replayed        3859       4507    1.03  6.0e-16  
      2193  3.17.43          chain3   replayed        7989      18121    1.66  7.4e-16  flips differ 1.37x
      2197  13^3             flat     replayed        4244       4127    0.78  1.2e-15  flips differ 1.25x
      2205  3^2.5.7^2        chain3   replayed        3515       4129    1.14  5.3e-16  
      2209  47^2             flat     replayed        6950      30532    4.35  1.9e-15  
      2223  3^2.13.19        flat     replayed        4539       7223    1.49  1.2e-15  
      2233  7.11.29          chain3   replayed        5316       9864    1.64  6.3e-16  
      2255  5.11.41          chain3   replayed        5444      14773    2.55  6.0e-16  
      2261  7.17.19          flat     replayed        4575       9977    2.09  9.9e-16  
      2275  5^2.7.13         chain3   replayed        3777       4158    1.02  5.9e-16  
      2277  3^2.11.23        chain3   replayed        4957       8527    1.57  6.3e-16  
      2295  3^3.5.17         chain3   replayed        4166       7238    1.72  6.9e-16  
      2299  11^2.19          flat     replayed        4571       7037    1.54  1.2e-15  
      2303  7^2.47           chain3   replayed        8205      17438    2.12  6.0e-16  
      2325  3.5^2.31         chain3   replayed        6779      11131    1.64  5.7e-16  
      2331  3^2.7.37         chain3   replayed        6890      13634    1.96  5.4e-16  
      2337  3.19.41          chain3   replayed        6518      19140    2.71  7.5e-16  
      2349  3^4.29           chain3   replayed        5513      11056    2.00  6.6e-16  
      2365  5.11.43          chain3   replayed        8175      16262    1.53  5.6e-16  flips differ 1.30x
      2375  5^3.19           flat     replayed        4117       7473    1.75  6.8e-16  
      2387  7.11.31          chain3   replayed        8656      11146    1.14  4.7e-16  
      2397  3.17.47          flat     replayed        7287      21541    2.90  1.6e-15  
      2401  7^4              flat     replayed        4575       4113    0.88  6.8e-16  
      2405  5.13.37          chain3   replayed        5394      14061    2.59  6.3e-16  
      2415  3.5.7.23         chain3   replayed        6615       8963    1.23  6.2e-16  
      2431  11.13.17         chain3   replayed        4316       7303    1.64  6.0e-16  
      2451  3.19.43          flat     replayed        7691      20833    2.59  1.4e-15  
      2457  3^3.7.13         chain3   replayed        4438       4798    0.89  8.3e-16  
      2465  5.17.29          flat     replayed        7338      14404    1.85  1.6e-15  
      2475  3^2.5^2.11       chain3   replayed        4007       4708    1.17  7.2e-16  
      2499  3.7^2.17         chain3   replayed        4181       7566    1.79  6.6e-16  
      2511  3^4.31           flat     replayed        6970      12350    1.77  7.5e-16  
      2523  3.29^2           flat     replayed        7695      18989    1.62  1.4e-15  flips differ 1.53x
      2527  7.19^2           chain3   replayed        5230      11912    1.97  5.3e-16  
      2535  3.5.13^2         chain3   replayed        4375       5122    1.16  7.3e-16  
      2541  3.7.11^2         flat     replayed        5333       4804    0.88  7.6e-16  
      2553  3.23.37          flat     replayed        7611      20172    2.58  8.7e-16  
      2565  3^3.5.19         chain3   replayed        4669       8409    1.53  8.1e-16  
      2583  3^2.7.41         chain3   replayed        6050      17267    2.73  6.2e-16  
      2585  5.11.47          chain3   replayed        9492      19738    1.37  5.3e-16  flips differ 1.52x
      2601  3^2.17^2         chain3   replayed        5584      11473    1.94  7.7e-16  
      2625  3.5^3.7          flat     replayed        4773       4948    1.02  9.2e-16  
      2635  5.17.31          flat     replayed        7428      16045    2.14  8.0e-16  
      2639  7.13.29          chain3   replayed        6338      11928    1.75  7.5e-16  
      2645  5.23^2           flat     replayed        7101      15214    1.57  1.7e-15  flips differ 1.36x
      2665  5.13.41          chain3   replayed        6422      17781    2.75  7.8e-16  
      2673  3^5.11           chain3   replayed        4493       5305    1.03  7.9e-16  
      2679  3.19.47          chain3   replayed       10454      24824    2.37  5.9e-16  
      2691  3^2.13.23        chain3   replayed        5897      10244    1.46  6.2e-16  
      2695  5.7^2.11         flat     replayed        4640       5045    1.07  5.5e-16  
      2697  3.29.31          flat     replayed        9121      20888    1.85  2.8e-15  
      2709  3^2.7.43         chain3   replayed        9438      18860    1.37  6.0e-16  flips differ 1.46x
      2717  11.13.19         flat     replayed        5580       8590    1.53  8.5e-16  
      2737  7.17.23          flat     replayed        6203      13635    2.01  1.4e-15  
      2755  5.19.29          chain3   replayed        7420      16829    2.20  6.8e-16  
      2775  3.5^2.37         chain3   replayed        8529      16288    1.44  7.1e-16  flips differ 1.32x
      2783  11^2.23          flat     replayed        6102      10085    1.48  1.4e-15  
      2793  3.7^2.19         chain3   replayed        4874       8912    1.81  6.3e-16  
      2795  5.13.43          chain3   replayed        7130      19542    2.67  5.9e-16  
      2805  3.5.11.17        chain3   replayed        5016       8606    1.15  6.0e-16  flips differ 1.49x
      2821  7.13.31          chain3   replayed        7808      13432    1.72  6.5e-16  
      2829  3.23.41          flat     replayed        9105      24540    2.59  8.5e-16  
      2835  3^4.5.7          chain3   replayed        5003       5578    1.10  7.2e-16  
      2849  7.11.37          flat     replayed        6465      16414    2.16  6.0e-16  
      2871  3^2.11.29        flat     replayed        7080      13125    1.79  8.2e-16  
      2873  13^2.17          chain3   replayed        5259       8891    1.48  8.2e-16  
      2875  5^3.23           chain3   replayed        6643      10771    1.21  6.7e-16  flips differ 1.34x
      2883  3.31^2           flat     replayed       10599      23092    1.97  1.9e-15  
      2907  3^2.17.19        flat     replayed        7962      13286    1.54  1.6e-15  
      2925  3^2.5^2.13       chain3   replayed        5386       5844    0.93  6.6e-16  
      2945  5.19.31          chain3   replayed        8207      18714    2.25  5.4e-16  
      2961  3^2.7.47         flat     replayed        7901      22893    2.75  6.3e-16  
      2967  3.23.43          chain3   replayed       12750      26765    2.08  5.6e-16  
      2975  5^2.7.17         flat     replayed        5734       9176    1.59  6.8e-16  
      2997  3^4.37           chain3   replayed        7243      17936    2.36  5.1e-16  
      3003  3.7.11.13        chain3   replayed        5065       5839    1.15  6.6e-16  
      3025  5^2.11^2         flat     replayed        6053       5868    0.79  8.7e-16  
      3045  3.5.7.29         flat     replayed        8143      13944    1.62  7.1e-16  
      3055  5.13.47          chain3   replayed       15022      23667    1.55  6.2e-16  
      3059  7.19.23          flat     replayed        7011      16188    2.23  7.0e-16  
      3069  3^2.11.31        chain3   replayed        8936      14850    1.34  9.8e-16  
      3075  3.5^2.41         chain3   replayed       14819      20576    1.38  6.2e-16  
      3087  3^2.7^3          chain3   replayed        5302       6044    1.07  5.0e-16  
      3105  3^3.5.23         chain3   replayed        8476      11921    1.27  5.3e-16  
      3125  5^5              flat     replayed        5710       6041    1.05  8.4e-16  
      3135  3.5.11.19        chain3   replayed        5855      10119    1.58  7.4e-16  
      3145  5.17.37          chain3   replayed        7617      22652    2.88  6.0e-16  
      3157  7.11.41          chain3   replayed        7991      20773    2.52  5.2e-16  
      3159  3^5.13           chain3   replayed        5780       6570    1.01  7.2e-16  
      3179  11.17^2          flat     replayed        6462      13775    2.07  1.1e-15  
      3185  5.7^2.13         flat     replayed        5688       6155    1.07  6.7e-16  
      3211  13^2.19          flat     replayed        6525      10457    1.58  1.0e-15  
      3213  3^3.7.17         chain3   replayed        5704      10212    1.66  9.3e-16  
      3219  3.29.37          flat     replayed       10050      28439    2.21  2.1e-15  flips differ 1.28x
      3225  3.5^2.43         chain3   replayed       16221      22533    1.36  5.6e-16  
      3243  3.23.47          chain3   replayed       14540      31648    1.84  5.8e-16  
      3249  3^2.19^2         flat     replayed        8840      15774    1.78  1.1e-15  
      3255  3.5.7.31         flat     replayed        8545      15688    1.45  6.8e-16  flips differ 1.27x
      3267  3^3.11^2         flat     replayed        7114       6686    0.94  5.6e-16  
      3289  11.13.23         chain3   replayed        9724      12213    1.25  8.5e-16  
      3311  7.11.43          chain3   replayed       15868      22852    1.36  5.7e-16  
      3315  3.5.13.17        flat     replayed        6697      10502    1.49  1.3e-15  
      3321  3^4.41           chain3   replayed        8430      22605    2.62  6.2e-16  
      3325  5^2.7.19         chain3   replayed        6550      10795    1.08  5.0e-16  flips differ 1.52x
      3335  5.23.29          chain3   replayed       11016      22067    1.79  6.0e-16  
      3367  7.13.37          chain3   replayed        7872      19771    2.51  6.3e-16  
      3375  3^3.5^3          chain3   replayed        5829       6917    1.04  5.9e-16  
      3381  3.7^2.23         flat     replayed        8259      12632    1.53  7.7e-16  
      3393  3^2.13.29        flat     replayed        8761      15850    1.72  1.2e-15  
      3441  3.31.37          flat     replayed       11349      31410    2.76  2.3e-15  
      3451  7.17.29          chain3   replayed       10982      20229    1.77  7.1e-16  
      3465  3^2.5.7.11       chain3   replayed        5372       6904    1.26  6.1e-16  
      3483  3^4.43           chain3   replayed       16810      24826    1.33  6.5e-16  
      3485  5.17.41          chain3   replayed       14802      27926    1.87  5.7e-16  
      3509  11^2.29          chain3   replayed        8255      15787    1.90  6.4e-16  
      3515  5.19.37          chain3   replayed        9792      26210    2.55  6.3e-16  
      3519  3^2.17.23        flat     replayed        7624      18225    2.12  8.1e-16  
      3525  3.5^2.47         flat     replayed        8438      27369    3.23  5.6e-16  
      3549  3.7.13^2         chain3   replayed        5919       7360    1.23  7.0e-16  
      3553  11.17.19         chain3   replayed        7015      15979    2.22  6.0e-16  
      3565  5.23.31          flat     replayed       13901      24495    1.72  1.7e-15  
      3567  3.29.41          flat     replayed       11410      34465    2.48  2.3e-15  
      3575  5^2.11.13        chain3   replayed        7059       7207    0.98  7.9e-16  
      3591  3^3.7.19         chain3   replayed        8553      12007    1.00  6.3e-16  flips differ 1.40x
      3619  7.11.47          chain3   replayed       12927      27915    2.15  4.7e-16  
      3625  5^3.29           chain3   replayed        9356      16774    1.69  5.4e-16  
      3627  3^2.13.31        flat     replayed        9328      17904    1.90  1.0e-15  
      3645  3^6.5            chain3   replayed        6921       7771    1.09  6.6e-16  
      3655  5.17.43          chain3   replayed       10494      30491    2.90  5.3e-16  
      3663  3^2.11.37        flat     replayed       10282      21746    2.10  2.3e-15  
      3675  3.5^2.7^2        chain3   replayed        6568       7139    0.84  5.1e-16  flips differ 1.29x
      3689  7.17.31          chain3   replayed       11318      22575    1.99  6.8e-16  
      3703  7.23^2           chain3   replayed       11831      21364    1.77  6.2e-16  
      3705  3.5.13.19        chain3   replayed        6949      12353    1.63  6.1e-16  
      3731  7.13.41          chain3   replayed        9105      25064    2.68  6.8e-16  
      3741  3.29.43          flat     replayed       13914      37293    2.40  3.5e-15  
      3751  11^2.31          flat     replayed        9032      17797    1.36  8.8e-16  flips differ 1.45x
      3757  13.17^2          chain3   replayed        7197      16760    2.18  8.5e-16  
      3773  7^3.11           flat     replayed        7310       7178    0.98  7.6e-16  
      3795  3.5.11.23        chain3   replayed        8875      14407    1.50  6.2e-16  
      3807  3^4.47           chain3   replayed       15921      30019    1.60  6.4e-16  
      3813  3.31.41          flat     replayed       14623      37933    2.58  2.5e-15  
      3825  3^2.5^2.17       chain3   replayed        6669      12210    1.81  8.6e-16  
      3857  7.19.29          chain3   replayed       11199      23722    1.81  6.5e-16  
      3861  3^3.11.13        flat     replayed        9124       8152    0.89  9.9e-16  
      3875  5^3.31           chain3   replayed       10475      18910    1.73  7.1e-16  
      3885  3.5.7.37         chain3   replayed        8998      23036    2.54  5.0e-16  
      3887  13^2.23          flat     replayed        8472      15002    1.76  9.1e-16  
      3895  5.19.41          chain3   replayed       10832      32194    2.69  5.1e-16  
      3913  7.13.43          chain3   replayed       13956      27570    1.38  7.6e-16  flips differ 1.43x
      3915  3^3.5.29         flat     replayed       11014      18489    1.60  6.5e-16  
      3927  3.7.11.17        chain3   replayed        7627      12250    1.60  6.2e-16  
      3933  3^2.19.23        chain3   replayed        9398      21450    2.27  7.4e-16  
      3969  3^4.7^2          chain3   replayed        6594       8275    1.15  8.0e-16  
      3971  11.19^2          chain3   replayed       11629      18958    1.44  6.4e-16  
      3993  3.11^3           flat     replayed        8054       7702    0.95  8.3e-16  
      3995  5.17.47          chain3   replayed       20371      36328    1.60  6.7e-16  
      3999  3.31.43          chain3   replayed       14943      40996    2.12  7.3e-16  flips differ 1.30x
      4025  5^2.7.23         chain3   replayed        9010      15319    1.70  5.7e-16  
      4059  3^2.11.41        chain3   replayed       12437      27353    1.37  5.4e-16  flips differ 1.60x
      4085  5.19.43          flat     replayed       12979      35127    2.70  1.5e-15  
      4089  3.29.47          chain3   replayed       18437      43868    2.38  5.9e-16  
      4095  3^2.5.7.13       chain3   replayed        7280       8444    1.12  6.8e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 chain3     204      0     13   1.04   1.51   2.51    1.53
 flat       158      2     17   0.99   1.75   3.04    1.75
 2p         132      0      2   1.27   2.17   3.43    2.07
 mono        14      0      1   1.27   2.06   2.47    1.96
 ALL        508      2     33   1.06   1.72   2.93    1.74
```


## by size
```
 band               cells median   <1.0   <0.8
 2..7                   3   2.25      0      0
 8..31                 12   1.53      2      0
 32..127               29   1.38      0      0
 128..511              84   1.96      4      0
 512..2047            205   1.73     14      0
 2048..4095           175   1.64     13      2
```


## by family
```
 family                                       cells median   <1.0  gmean
 chain3                                         204   1.51     13   1.53
 flat                                           158   1.75     17   1.75
 2p                                             132   2.17      2   2.07
 mono                                            14   2.06      1   1.96
```


flip agreement: our two readings more than 25% apart at 58 of 508 cells.

worst 10: 2197 (flat 0.78), 3025 (flat 0.79), 1375 (chain3 0.81), 1001 (flat 0.82), 429 (chain3 0.82), 637 (flat 0.83), 3675 (chain3 0.84), 1617 (chain3 0.85), 147 (chain3 0.87), 2401 (flat 0.88)
best 5: 2209 (flat 4.35), 705 (2p 3.91), 799 (2p 3.88), 703 (2p 3.81), 779 (2p 3.80)
