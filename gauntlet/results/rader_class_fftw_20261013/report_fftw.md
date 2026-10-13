# gauntlet report

run: `rader_class_fftw_20261013`  contract file suffix: `_fftw`  cells: 1562 listed, 1562 benched, comparator: FFTW

control cell: 34 readings, 1.316..1.475 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
       106  2.53             prime    raced            434        728    1.68  4.4e-16  bluestein
       118  2.59             prime    raced            448        672    1.48  6.4e-16  bluestein
       122  2.61             prime    raced            443        996    2.24  4.2e-16  bluestein
       134  2.67             prime    raced            574        960    1.64  6.8e-16  bluestein
       142  2.71             prime    raced            572        901    1.56  8.7e-16  bluestein
       146  2.73             prime    raced            594       1013    1.69  7.6e-16  bluestein
       158  2.79             prime    raced            602        953    1.56  6.0e-16  bluestein
       159  3.53             prime    raced            603       1154    1.91  5.3e-16  bluestein
       166  2.83             prime    raced            747       1091    1.27  8.3e-16  bluestein
       177  3.59             prime    raced            753       1459    1.92  7.2e-16  bluestein
       178  2.89             prime    raced            742       1441    1.94  6.1e-16  bluestein
       183  3.61             prime    raced            747       1503    2.01  6.8e-16  bluestein
       194  2.97             prime    raced            985       1192    1.20  5.8e-16  bluestein
       201  3.67             prime    raced            966       1729    1.79  4.8e-16  bluestein
       202  2.101            prime    raced            964       1352    1.40  3.6e-16  bluestein
       206  2.103            prime    raced            966       1951    2.02  5.8e-16  bluestein
       212  2^2.53           flat     raced            607       1587    2.59  8.7e-16  
       213  3.71             prime    raced            974       1407    1.44  4.0e-16  bluestein
       214  2.107            prime    raced            971       1974    2.02  5.5e-16  bluestein
       218  2.109            prime    raced            975       1809    1.85  4.8e-16  bluestein
       219  3.73             prime    raced            974       1501    1.52  4.5e-16  bluestein
       226  2.113            prime    raced            987       1905    1.93  5.8e-16  bluestein
       236  2^2.59           prime    raced            982       1878    1.91  3.6e-16  bluestein
       237  3.79             prime    raced            981       1468    1.48  5.1e-16  bluestein
       244  2^2.61           flat     raced            746       1992    2.65  4.4e-16  
       249  3.83             prime    raced            988       2142    2.17  4.3e-16  bluestein
       254  2.127            prime    raced            990       1359    1.36  4.3e-16  bluestein
       262  2.131            prime    raced           1435       1784    1.24  7.3e-16  bluestein
       265  5.53             flat     raced           1048       1939    1.83  8.0e-16  
       267  3.89             prime    raced           1428       2193    1.53  8.4e-16  bluestein
       268  2^2.67           flat     raced            826       2338    2.81  8.3e-16  
       274  2.137            prime    raced           1403       1781    1.24  5.8e-16  bluestein
       278  2.139            prime    raced           1405       2255    1.55  6.9e-16  bluestein
       284  2^2.71           flat     raced            790       1857    2.31  5.5e-16  
       291  3.97             flat     raced           1886       2006    1.01  7.8e-16  
       292  2^2.73           flat     raced            788       1987    2.48  6.7e-16  
       295  5.59             prime    raced           1339       2434    1.80  5.7e-16  bluestein
       298  2.149            prime    raced           1343       1918    1.41  6.2e-16  bluestein
       302  2.151            prime    raced           1342       1755    1.27  4.5e-16  bluestein
       303  3.101            prime    raced           1340       2069    1.53  6.3e-16  bluestein
       305  5.61             flat     raced           1218       2531    2.05  5.5e-16  
       309  3.103            prime    raced           1343       3040    2.18  7.7e-16  bluestein
       314  2.157            prime    raced           1343       1858    1.38  5.6e-16  bluestein
       316  2^2.79           flat     raced            988       1943    1.95  1.0e-15  
       318  2.3.53           flat     raced           1210       2305    1.90  7.6e-16  
       321  3.107            prime    raced           1888       2940    1.45  6.1e-16  bluestein
       326  2.163            prime    raced           1912       2249    1.12  5.0e-16  bluestein
       327  3.109            flat     raced           2253       2742    1.17  8.8e-16  
       332  2^2.83           flat     raced           1716       2866    1.67  7.6e-16  
       334  2.167            prime    raced           1918       3126    1.61  5.5e-16  bluestein
       335  5.67             flat     raced           1351       2926    2.15  7.2e-16  
       339  3.113            flat     raced           1908       2884    1.50  6.4e-16  
       346  2.173            prime    raced           2108       3151    1.47  6.1e-16  bluestein
       354  2.3.59           flat     raced           1853       2825    1.49  6.5e-16  
       355  5.71             flat     raced           1278       2405    1.81  4.7e-16  
       356  2^2.89           flat     raced            999       2328    2.31  5.5e-16  
       358  2.179            prime    raced           2111       3170    1.48  5.2e-16  bluestein
       362  2.181            prime    raced           1676       2886    1.72  8.0e-16  bluestein
       365  5.73             flat     raced           1324       2511    1.89  7.4e-16  
       366  2.3.61           flat     raced           1469       3006    2.01  5.5e-16  
       371  7.53             flat     raced           1278       2721    2.06  6.6e-16  
       381  3.127            prime    raced           1678       2156    1.26  7.6e-16  bluestein
       382  2.191            prime    raced           1676       2126    1.25  8.6e-16  bluestein
       386  2.193            prime    raced           2153       2493    1.15  7.6e-16  bluestein
       388  2^2.97           flat     raced           1282       2616    2.04  5.7e-16  
       393  3.131            prime    raced           2137       2683    1.25  6.4e-16  bluestein
       394  2.197            prime    raced           2135       2332    1.08  5.5e-16  bluestein
       395  5.79             flat     raced           1654       2464    1.46  9.2e-16  
       398  2.199            prime    raced           2144       2518    1.16  4.4e-16  bluestein
       402  2.3.67           flat     raced           1610       3498    2.16  6.4e-16  
       404  2^2.101          flat     raced           1539       2738    1.59  7.9e-16  
       411  3.137            prime    raced           2139       3382    1.56  5.5e-16  bluestein
       412  2^2.103          flat     raced           1395       2756    1.93  1.1e-15  
       413  7.59             flat     raced           2028       3331    1.60  8.3e-16  
       415  5.83             prime    raced           2219       3626    1.61  6.2e-16  bluestein
       417  3.139            prime    raced           2153       3409    1.52  5.2e-16  bluestein
       422  2.211            prime    raced           2163       2768    1.26  5.9e-16  bluestein
       424  2^3.53           flat     raced           1177       3081    2.59  6.9e-16  
       426  2.3.71           flat     raced           1619       2721    1.68  4.8e-16  
       427  7.61             flat     raced           1514       3547    2.32  5.1e-16  
       428  2^2.107          prime    raced           2148       2767    1.25  5.4e-16  bluestein
       436  2^2.109          flat     raced           1907       3633    1.63  7.2e-16  
       438  2.3.73           flat     raced           1669       2886    1.72  5.7e-16  
       445  5.89             flat     raced           1698       3663    2.11  5.2e-16  
       446  2.223            prime    raced           2161       3019    1.40  5.6e-16  bluestein
       447  3.149            prime    raced           2189       3104    1.38  6.4e-16  bluestein
       452  2^2.113          flat     raced           1284       3749    2.77  5.1e-16  
       453  3.151            prime    raced           2248       2674    1.18  6.9e-16  bluestein
       454  2.227            prime    raced           2240       2947    1.29  5.5e-16  bluestein
       458  2.229            prime    raced           2241       2936    1.30  5.4e-16  bluestein
       466  2.233            prime    raced           2178       2958    1.35  5.0e-16  bluestein
       469  7.67             flat     raced           1708       4118    2.15  6.3e-16  
       471  3.157            prime    raced           2189       2798    1.23  6.2e-16  bluestein
       472  2^3.59           flat     raced           1801       3779    2.06  4.9e-16  
       474  2.3.79           flat     raced           1979       2850    1.42  9.4e-16  
       477  3^2.53           flat     raced           1521       3525    2.29  8.9e-16  
       478  2.239            prime    raced           2166       2981    1.37  5.3e-16  bluestein
       482  2.241            prime    raced           2179       4943    2.25  5.4e-16  bluestein
       485  5.97             flat     raced           2116       3110    1.40  4.2e-16  
       488  2^3.61           flat     raced           1278       4019    3.04  5.0e-16  
       489  3.163            prime    raced           2180       3424    1.56  6.9e-16  bluestein
       497  7.71             flat     raced           1666       3404    2.01  5.5e-16  
       498  2.3.83           prime    raced           2273       3363    1.47  5.3e-16  bluestein
       501  3.167            prime    raced           2205       4784    2.16  6.5e-16  bluestein
       502  2.251            prime    raced           2202       2762    1.21  5.9e-16  bluestein
       505  5.101            flat     raced           2447       3510    1.42  7.1e-16  
       508  2^2.127          flat     raced           1848       2802    1.50  8.6e-16  
       511  7.73             flat     raced           1679       3576    2.11  7.9e-16  
       514  2.257            prime    raced           3346       3849    1.15  7.9e-16  bluestein
       515  5.103            flat     raced           2334       5253    2.23  9.2e-16  
       519  3.173            prime    raced           3257       4750    1.41  6.1e-16  bluestein
       524  2^2.131          flat     raced           1692       3658    2.14  7.9e-16  
       526  2.263            prime    raced           3316       3855    1.14  6.8e-16  bluestein
       530  2.5.53           flat     raced           1685       3918    2.28  7.2e-16  
       531  3^2.59           flat     raced           2486       4321    1.71  6.0e-16  
       534  2.3.89           flat     raced           2161       4380    2.00  6.5e-16  
       535  5.107            prime    raced           3313       4937    1.44  6.6e-16  bluestein
       536  2^3.67           flat     raced           1412       4869    3.43  5.9e-16  
       537  3.179            prime    raced           3251       3457    1.04  6.6e-16  bluestein
       538  2.269            prime    raced           3299       3846    1.13  5.4e-16  bluestein
       542  2.271            prime    raced           3296       3842    1.14  7.1e-16  bluestein
       543  3.181            prime    raced           3336       4361    1.30  7.0e-16  bluestein
       545  5.109            flat     raced           2675       4577    1.56  6.2e-16  
       548  2^2.137          flat     raced           1841       3724    2.00  9.9e-16  
       549  3^2.61           flat     raced           1804       4599    2.55  5.8e-16  
       553  7.79             flat     raced           2097       3531    1.61  6.5e-16  
       554  2.277            prime    raced           3300       3870    1.14  7.5e-16  bluestein
       556  2^2.139          flat     raced           2192       4556    2.00  9.2e-16  
       562  2.281            prime    raced           3318       3854    1.15  9.1e-16  bluestein
       565  5.113            flat     raced           2156       5152    2.36  5.7e-16  
       566  2.283            prime    raced           3284       3863    1.16  7.1e-16  bluestein
       568  2^3.71           flat     raced           1463       3746    2.49  4.4e-16  
       573  3.191            prime    raced           3337       3197    0.94  8.5e-16  bluestein
       579  3.193            flat     raced           4407       3737    0.83  5.6e-16  
       581  7.83             flat     raced           3491       5126    1.45  6.9e-16  
       582  2.3.97           flat     raced           2570       3963    1.53  7.7e-16  
       583  11.53            flat     raced           1814       4465    2.37  6.5e-16  
       584  2^3.73           flat     raced           1429       4110    2.82  6.2e-16  
       586  2.293            prime    raced           4466       3864    0.83  7.6e-16  bluestein
       590  2.5.59           flat     raced           2682       4952    1.83  5.9e-16  
       591  3.197            flat     raced           3627       3806    1.03  6.2e-16  
       596  2^2.149          flat     raced           2427       4193    1.71  7.2e-16  
       597  3.199            flat     raced           4499       3745    0.81  7.9e-16  
       603  3^2.67           flat     raced           1998       4639    2.14  6.2e-16  
       604  2^2.151          flat     raced           2397       3526    1.40  9.1e-16  
       606  2.3.101          flat     raced           3091       4186    1.31  6.8e-16  
       610  2.5.61           flat     raced           2054       5100    2.46  6.0e-16  
       614  2.307            prime    raced           3730       5117    1.36  7.6e-16  bluestein
       618  2.3.103          flat     raced           2823       4189    1.48  8.3e-16  
       622  2.311            prime    raced           3706       5087    1.34  8.0e-16  bluestein
       623  7.89             flat     raced           2166       5234    2.38  6.1e-16  
       626  2.313            prime    raced           3721       5193    1.39  6.9e-16  bluestein
       628  2^2.157          flat     raced           2261       3828    1.67  8.9e-16  
       632  2^3.79           flat     raced           1810       4032    2.20  1.1e-15  
       633  3.211            prime    raced           3726       4247    1.12  5.6e-16  bluestein
       634  2.317            prime    raced           3715       3640    0.96  6.7e-16  bluestein
       635  5.127            flat     raced           3008       3708    1.21  8.5e-16  
       636  2^2.3.53         flat     raced           1639       4724    2.83  7.1e-16  
       639  3^2.71           flat     raced           1923       4370    2.20  6.1e-16  
       642  2.3.107          prime    raced           3967       4239    0.99  6.6e-16  bluestein
       649  11.59            flat     raced           2848       5450    1.85  6.3e-16  
       652  2^2.163          flat     raced           2704       4588    1.69  6.3e-16  
       654  2.3.109          flat     raced           3396       5508    1.49  6.9e-16  
       655  5.131            flat     raced           2728       4544    1.65  8.4e-16  
       657  3^2.73           flat     raced           1959       4514    2.22  9.4e-16  
       662  2.331            prime    raced           4015       5447    1.35  6.5e-16  bluestein
       664  2^3.83           flat     raced           2997       4639    1.53  7.3e-16  
       668  2^2.167          prime    raced           4023       6362    1.55  6.6e-16  bluestein
       669  3.223            prime    raced           3996       4810    1.18  8.5e-16  bluestein
       670  2.5.67           flat     raced           2228       6231    2.69  7.2e-16  
       671  11.61            flat     raced           2099       5790    2.67  6.0e-16  
       674  2.337            prime    raced           4770       5463    1.14  6.5e-16  bluestein
       678  2.3.113          flat     raced           2817       5569    1.96  4.4e-16  
       679  7.97             flat     raced           2695       4338    1.54  6.5e-16  
       681  3.227            prime    raced           4836       4603    0.93  5.8e-16  bluestein
       685  5.137            flat     raced           2924       6110    1.99  7.6e-16  
       687  3.229            flat     raced           5171       4526    0.87  7.8e-16  
       689  13.53            flat     raced           2028       5365    2.18  7.9e-16  
       692  2^2.173          flat     raced           2996       4748    1.55  6.2e-16  
       694  2.347            prime    raced           4863       4708    0.85  6.5e-16  bluestein
       695  5.139            flat     raced           3496       5793    1.62  7.1e-16  
       698  2.349            prime    raced           4780       4666    0.97  6.1e-16  bluestein
       699  3.233            prime    raced           4779       4537    0.95  4.7e-16  bluestein
       706  2.353            prime    raced           4809       4720    0.91  6.7e-16  bluestein
       707  7.101            flat     raced           3287       4959    1.48  7.5e-16  
       708  2^2.3.59         flat     raced           2645       4450    1.67  7.8e-16  
       710  2.5.71           flat     raced           2276       4777    2.08  5.9e-16  
       711  3^2.79           flat     raced           2475       4469    1.80  7.2e-16  
       712  2^3.89           flat     raced           1805       4739    2.45  8.2e-16  
       716  2^2.179          prime    raced           4777       4818    1.00  5.6e-16  bluestein
       717  3.239            flat     raced           4655       4506    0.95  9.3e-16  
       718  2.359            prime    raced           4769       4685    0.97  5.1e-16  bluestein
       721  7.103            flat     raced           2982       6921    2.29  1.0e-15  
       723  3.241            flat     raced           4478       7442    1.64  5.6e-16  
       724  2^2.181          flat     raced           2713       5812    2.13  7.9e-16  
       730  2.5.73           flat     raced           2369       5177    2.13  6.1e-16  
       732  2^2.3.61         flat     raced           2000       6207    3.04  6.0e-16  
       734  2.367            prime    raced           4807       5586    1.16  6.0e-16  bluestein
       737  11.67            flat     raced           2367       6934    2.81  8.3e-16  
       742  2.7.53           flat     raced           2218       5598    2.07  5.5e-16  
       745  5.149            flat     raced           3856       5202    1.34  6.5e-16  
       746  2.373            prime    raced           4824       5603    1.16  5.6e-16  bluestein
       747  3^2.83           flat     raced           4056       5284    1.29  8.2e-16  
       749  7.107            prime    raced           4861       7050    1.43  6.6e-16  bluestein
       753  3.251            flat     raced           5232       4254    0.81  6.8e-16  
       755  5.151            flat     raced           3791       4501    1.17  8.7e-16  
       758  2.379            prime    raced           4848       4659    0.95  6.3e-16  bluestein
       762  2.3.127          flat     raced           3701       4104    1.10  9.4e-16  
       763  7.109            flat     raced           3690       6491    1.60  7.1e-16  
       764  2^2.191          flat     raced           2757       4321    1.55  6.3e-16  
       766  2.383            prime    raced           4818       4617    0.96  4.2e-16  bluestein
       767  13.59            flat     raced           3292       6539    1.88  6.5e-16  
       771  3.257            flat     raced           5534       5846    1.04  8.8e-16  
       772  2^2.193          flat     raced           3011       4844    1.59  7.0e-16  
       776  2^3.97           flat     raced           2205       4931    2.13  3.8e-16  
       778  2.389            prime    raced           4842       5201    1.06  5.8e-16  bluestein
       781  11.71            flat     raced           2296       5337    2.29  6.5e-16  
       785  5.157            flat     raced           3798       4718    1.24  7.7e-16  
       786  2.3.131          flat     raced           3501       5433    1.51  1.0e-15  
       788  2^2.197          flat     raced           2547       4868    1.89  6.7e-16  
       789  3.263            prime    raced           4778       5880    1.22  6.3e-16  bluestein
       790  2.5.79           flat     raced           2815       4997    1.76  8.8e-16  
       791  7.113            flat     raced           2820       6806    2.33  4.9e-16  
       793  13.61            flat     raced           2421       6958    2.74  5.9e-16  
       794  2.397            prime    raced           4820       5236    1.08  5.7e-16  bluestein
       795  3.5.53           flat     raced           2257       5994    2.62  8.5e-16  
       796  2^2.199          flat     raced           3179       4972    1.54  7.7e-16  
       801  3^2.89           flat     raced           2554       5309    2.04  7.2e-16  
       802  2.401            prime    raced           4867       5830    0.94  5.5e-16  flips differ 1.28x bluestein
       803  11.73            flat     raced           2374       5936    2.45  7.2e-16  
       804  2^2.3.67         flat     raced           2139       7326    3.42  6.5e-16  
       807  3.269            prime    raced           4842       5888    1.21  5.8e-16  bluestein
       808  2^3.101          flat     raced           3078       5520    1.71  8.2e-16  
       813  3.271            prime    raced           4854       5859    1.20  5.6e-16  bluestein
       815  5.163            flat     raced           4318       5789    1.32  6.8e-16  
       818  2.409            prime    raced           4874       5884    1.20  5.1e-16  bluestein
       822  2.3.137          flat     raced           3734       6845    1.80  8.1e-16  
       824  2^3.103          flat     raced           2580       5622    2.13  1.1e-15  
       826  2.7.59           flat     raced           3516       7078    2.00  7.8e-16  
       830  2.5.83           flat     raced           4320       7402    1.70  8.1e-16  
       831  3.277            prime    raced           4848       5826    1.19  7.0e-16  bluestein
       834  2.3.139          flat     raced           4164       6931    1.66  8.0e-16  
       835  5.167            prime    raced           4850       7868    1.61  5.2e-16  bluestein
       838  2.419            prime    raced           4858       5901    1.21  5.5e-16  bluestein
       842  2.421            prime    raced           4862       5966    1.21  6.8e-16  bluestein
       843  3.281            prime    raced           4912       5879    1.19  4.8e-16  bluestein
       844  2^2.211          flat     raced           3042       5706    1.83  6.2e-16  
       848  2^4.53           flat     raced           2095       6312    2.96  7.1e-16  
       849  3.283            prime    raced           4898       6103    1.24  6.2e-16  bluestein
       852  2^2.3.71         flat     raced           2170       5689    2.60  5.3e-16  
       854  2.7.61           flat     raced           2712       7298    2.65  4.6e-16  
       856  2^3.107          flat     raced           4832       5644    1.14  1.3e-15  
       862  2.431            prime    raced           4897       6014    1.23  5.2e-16  bluestein
       865  5.173            flat     raced           4733       7954    1.67  6.9e-16  
       866  2.433            prime    raced           4843       6462    1.33  5.2e-16  bluestein
       869  11.79            flat     raced           2849       5953    2.02  8.0e-16  
       871  13.67            flat     raced           2672       8305    3.07  7.7e-16  
       872  2^3.109          flat     raced           3182       7342    2.01  7.7e-16  
       873  3^2.97           flat     raced           3048       6248    2.01  6.1e-16  
       876  2^2.3.73         flat     raced           2208       6331    2.82  6.7e-16  
       878  2.439            prime    raced           4833       6512    1.34  5.9e-16  bluestein
       879  3.293            prime    raced           4850       6148    1.26  3.8e-16  bluestein
       885  3.5.59           flat     raced           3903       5690    1.44  6.9e-16  
       886  2.443            prime    raced           4845       6551    1.35  5.9e-16  bluestein
       889  7.127            flat     raced           3794       5098    1.32  8.1e-16  
       890  2.5.89           flat     raced           3022       5902    1.93  7.7e-16  
       892  2^2.223          flat     raced           4094       6198    1.49  8.0e-16  
       894  2.3.149          flat     raced           4702       6259    1.32  7.2e-16  
       895  5.179            prime    raced           4864       5869    1.20  5.4e-16  bluestein
       898  2.449            prime    raced           4929       6545    1.31  7.0e-16  bluestein
       901  17.53            flat     raced           2592       8920    3.43  1.1e-15  
       904  2^3.113          flat     raced           2412       7567    2.95  6.9e-16  
       905  5.181            flat     raced           4416       7377    1.65  6.9e-16  
       906  2.3.151          flat     raced           4604       5343    1.14  7.5e-16  
       908  2^2.227          prime    raced           4884       6104    1.19  5.8e-16  bluestein
       909  3^2.101          flat     raced           3627       6417    1.66  8.7e-16  
       913  11.83            flat     raced           4747       6589    1.36  6.3e-16  
       914  2.457            prime    raced           4888       6335    1.30  5.8e-16  bluestein
       915  3.5.61           flat     raced           2713       7800    2.81  7.2e-16  
       916  2^2.229          flat     raced           3710       6031    1.52  7.1e-16  
       917  7.131            flat     raced           3492       6439    1.84  9.2e-16  
       921  3.307            prime    raced           4904       7726    1.57  6.3e-16  bluestein
       922  2.461            prime    raced           4882       6368    1.30  5.5e-16  bluestein
       923  13.71            flat     raced           2666       6432    2.36  5.9e-16  
       926  2.463            prime    raced           4849       6338    1.28  6.1e-16  bluestein
       927  3^2.103          flat     raced           3480       6356    1.81  8.6e-16  
       932  2^2.233          flat     raced           4077       6052    1.16  7.3e-16  flips differ 1.29x
       933  3.311            prime    raced           4950       7798    1.21  5.4e-16  flips differ 1.31x bluestein
       934  2.467            prime    raced           4820       6365    0.99  4.5e-16  flips differ 1.33x bluestein
       938  2.7.67           flat     raced           3041       8770    2.86  6.8e-16  
       939  3.313            prime    raced           4928       7825    1.58  5.1e-16  bluestein
       942  2.3.157          flat     raced           4547       5433    1.18  7.8e-16  
       944  2^4.59           flat     raced           3287       6051    1.70  6.6e-16  
       948  2^2.3.79         flat     raced           2694       5901    2.14  7.2e-16  
       949  13.73            flat     raced           2714       7175    2.51  8.0e-16  
       951  3.317            prime    raced           4836       5614    0.90  6.0e-16  flips differ 1.29x bluestein
       954  2.3^2.53         flat     raced           2749       6914    2.45  8.3e-16  
       955  5.191            flat     raced           4381       5471    1.18  6.9e-16  
       956  2^2.239          flat     raced           3520       6233    1.74  9.1e-16  
       958  2.479            prime    raced           4907       6362    1.28  6.9e-16  bluestein
       959  7.137            flat     raced           3703       8057    2.15  8.9e-16  
       963  3^2.107          prime    raced           4852       6534    1.02  6.6e-16  flips differ 1.35x bluestein
       964  2^2.241          flat     raced           3411       9904    2.88  6.3e-16  
       965  5.193            flat     raced           4942       6254    1.26  6.4e-16  
       970  2.5.97           flat     raced           3579       6851    1.85  6.6e-16  
       973  7.139            flat     raced           4445       8148    1.80  8.5e-16  
       974  2.487            prime    raced           4897       6954    1.41  5.7e-16  bluestein
       976  2^4.61           flat     raced           2539       8149    3.04  6.2e-16  
       978  2.3.163          flat     raced           5105       6933    1.27  8.7e-16  
       979  11.89            flat     raced           2988       8525    2.78  6.8e-16  
       981  3^2.109          flat     raced           4121       8365    1.81  9.3e-16  
       982  2.491            prime    raced           4965       6857    1.37  5.8e-16  bluestein
       985  5.197            flat     raced           4155       6263    1.47  5.9e-16  
       993  3.331            prime    raced           4893       8293    1.69  6.3e-16  bluestein
       994  2.7.71           flat     raced           3093       6707    2.13  5.2e-16  
       995  5.199            flat     raced           5261       6343    1.20  8.6e-16  
       996  2^2.3.83         flat     raced           4284       8976    2.09  6.5e-16  
       998  2.499            prime    raced           4853       6875    1.40  4.6e-16  bluestein
      1002  2.3.167          prime    raced           4895       6934    1.28  5.5e-16  bluestein
      1003  17.59            flat     raced           4142      10674    2.55  6.6e-16  
      1004  2^2.251          flat     raced           3961       5655    1.42  6.5e-16  
      1005  3.5.67           flat     raced           3015       9149    3.02  6.1e-16  
      1006  2.503            prime    raced           4889       5960    1.20  4.9e-16  bluestein
      1007  19.53            flat     raced           2908      10115    3.27  8.1e-16  
      1010  2.5.101          flat     raced           4306       7046    1.63  8.0e-16  
      1011  3.337            prime    raced           4899       8272    1.43  5.8e-16  bluestein
      1016  2^3.127          flat     raced           3392       5751    1.59  7.5e-16  
      1017  3^2.113          flat     raced           3212       8564    2.59  5.5e-16  
      1018  2.509            prime    raced           4900       6041    1.21  5.5e-16  bluestein
      1022  2.7.73           flat     raced           3168       7090    2.23  7.5e-16  
      1027  13.79            flat     raced           3253       6701    2.04  7.8e-16  
      1028  2^2.257          flat     raced           3874       7818    2.00  6.2e-16  
      1030  2.5.103          flat     raced           4325       7046    1.58  8.8e-16  
      1037  17.61            flat     raced           3022      11265    3.68  6.7e-16  
      1038  2.3.173          prime    raced           5893       7211    1.22  6.1e-16  bluestein
      1041  3.347            prime    raced           5866       7481    1.20  6.2e-16  bluestein
      1042  2.521            prime    raced           5815       8032    1.38  5.4e-16  bluestein
      1043  7.149            flat     raced           4851       7317    1.49  5.7e-16  
      1046  2.523            prime    raced           5951       7911    1.24  7.7e-16  bluestein
      1047  3.349            prime    raced           5839       7445    1.15  6.2e-16  bluestein
      1048  2^3.131          flat     raced           3035       7358    2.40  7.3e-16  
      1052  2^2.263          prime    raced           5838       8009    1.37  6.2e-16  bluestein
      1055  5.211            flat     raced           4912       7112    1.40  7.6e-16  
      1057  7.151            flat     raced           4909       6932    1.32  9.3e-16  
      1059  3.353            prime    raced           5816       7136    1.20  5.3e-16  bluestein
      1060  2^2.5.53         flat     raced           2575       8028    2.60  8.5e-16  
      1062  2.3^2.59         flat     raced           4332       6809    1.51  7.6e-16  
      1065  3.5.71           flat     raced           3041       7580    2.45  5.8e-16  
      1067  11.97            flat     raced           3586       7910    2.16  6.7e-16  
      1068  2^2.3.89         flat     raced           2863       7098    2.45  6.4e-16  
      1070  2.5.107          prime    raced           5837       7086    1.12  4.7e-16  bluestein
      1072  2^4.67           flat     raced           2639       9965    3.76  5.5e-16  
      1074  2.3.179          prime    raced           5933       7008    1.06  6.4e-16  bluestein
      1076  2^2.269          prime    raced           5882       7897    1.32  6.0e-16  bluestein
      1077  3.359            prime    raced           5814       7498    1.28  4.8e-16  bluestein
      1079  13.83            flat     raced           5407       7905    1.45  9.5e-16  
      1082  2.541            prime    raced           5820      10084    1.72  5.8e-16  bluestein
      1084  2^2.271          flat     raced           4529       7801    1.71  9.1e-16  
      1086  2.3.181          flat     raced           5461       8831    1.61  6.9e-16  
      1090  2.5.109          flat     raced           4777       9258    1.91  6.0e-16  
      1094  2.547            prime    raced           5812      10124    1.72  8.3e-16  bluestein
      1095  3.5.73           flat     raced           3032       8083    2.61  7.6e-16  
      1096  2^3.137          flat     raced           3293       9225    2.63  6.6e-16  
      1098  2.3^2.61         flat     raced           3316       9407    2.80  7.6e-16  
      1099  7.157            flat     raced           4728       6723    1.41  8.4e-16  
      1101  3.367            prime    raced           5840       8866    1.51  6.4e-16  bluestein
      1106  2.7.79           flat     raced           3759       7300    1.88  7.8e-16  
      1108  2^2.277          flat     raced           4765       7761    1.52  8.0e-16  
      1111  11.101           flat     raced           4400       8029    1.66  7.9e-16  
      1112  2^3.139          flat     raced           3833       9252    2.20  7.2e-16  
      1113  3.7.53           flat     raced           3221       8515    2.62  9.1e-16  
      1114  2.557            prime    raced           5804      10260    1.76  6.9e-16  bluestein
      1115  5.223            prime    raced           5859       7844    1.12  6.5e-16  bluestein
      1119  3.373            prime    raced           5848       8911    1.51  6.8e-16  bluestein
      1121  19.59            flat     raced           4798      12397    2.57  6.7e-16  
      1124  2^2.281          flat     raced           3818       7868    2.00  6.6e-16  
      1126  2.563            prime    raced           6059      10105    1.52  6.2e-16  bluestein
      1130  2.5.113          flat     raced           3950       9489    2.38  5.2e-16  
      1132  2^2.283          flat     raced           5959       7924    1.33  8.2e-16  
      1133  11.103           flat     raced           4091       8187    1.99  7.6e-16  
      1135  5.227            prime    raced           6019       7617    1.25  5.7e-16  bluestein
      1136  2^4.71           flat     raced           2794       7587    2.69  5.4e-16  
      1137  3.379            prime    raced           6078       7024    1.15  5.7e-16  bluestein
      1138  2.569            prime    raced           6023       8140    1.33  5.5e-16  bluestein
      1139  17.67            flat     raced           3378      12952    3.83  8.7e-16  
      1141  7.163            flat     raced           5587       8139    1.43  8.2e-16  
      1142  2.571            prime    raced           6076       7992    1.21  5.8e-16  bluestein
      1143  3^2.127          flat     raced           4431       6439    1.43  7.3e-16  
      1145  5.229            flat     raced           6074       7704    1.26  6.8e-16  
      1146  2.3.191          flat     raced           5569       6600    1.18  7.1e-16  
      1149  3.383            prime    raced           6031       7064    1.16  5.9e-16  bluestein
      1154  2.577            prime    raced           6736       8716    1.29  6.2e-16  bluestein
      1157  13.89            flat     raced           3474       8043    2.24  6.9e-16  
      1158  2.3.193          flat     raced           5831       7523    1.25  5.9e-16  
      1159  19.61            flat     raced           3417      12786    3.72  7.5e-16  
      1162  2.7.83           flat     raced           5726      10709    1.83  7.1e-16  
      1164  2^2.3.97         flat     raced           3300       8336    2.52  5.9e-16  
      1165  5.233            prime    raced           6615       7647    1.16  6.1e-16  bluestein
      1166  2.11.53          flat     raced           3492       9262    2.38  6.4e-16  
      1167  3.389            prime    raced           6676       8392    1.23  6.3e-16  bluestein
      1168  2^4.73           flat     raced           2789       8466    3.02  7.3e-16  
      1169  7.167            prime    raced           6637      11199    1.67  6.6e-16  bluestein
      1172  2^2.293          prime    raced           6635       7877    1.17  7.6e-16  bluestein
      1174  2.587            prime    raced           6635       8665    1.30  6.8e-16  bluestein
      1177  11.107           prime    raced           6713       8347    1.23  6.5e-16  bluestein
      1179  3^2.131          flat     raced           3996       8294    2.06  7.2e-16  
      1180  2^2.5.59         flat     raced           4054       9891    2.43  6.6e-16  
      1182  2.3.197          flat     raced           5467       7079    1.25  6.2e-16  
      1185  3.5.79           flat     raced           3650       7953    2.16  8.9e-16  
      1186  2.593            prime    raced           6856       8672    1.26  6.4e-16  bluestein
      1191  3.397            prime    raced           6830       8309    1.18  7.8e-16  bluestein
      1192  2^3.149          flat     raced           4331       8307    1.90  6.7e-16  
      1194  2.3.199          flat     raced           6062       7347    1.17  6.3e-16  
      1195  5.239            flat     raced           5546       7725    1.36  8.1e-16  
      1198  2.599            prime    raced           6773       8729    1.27  6.9e-16  bluestein
      1199  11.109           flat     raced           5128      10660    1.97  8.4e-16  
      1202  2.601            flat     raced          11742      11081    0.94  8.6e-16  
      1203  3.401            flat     raced           8608       8901    1.00  1.0e-15  
      1205  5.241            flat     raced           5279      12546    2.37  7.1e-16  
      1206  2.3^2.67         flat     raced           3779       9236    2.41  8.7e-16  
      1207  17.71            flat     raced           3301      11285    3.30  7.7e-16  
      1208  2^3.151          flat     raced           4105       7187    1.69  6.6e-16  
      1211  7.173            flat     raced           6167      11108    1.78  8.1e-16  
      1212  2^2.3.101        flat     raced           4120       8653    1.91  9.4e-16  
      1214  2.607            prime    raced          10169      10984    1.06  5.4e-16  bluestein
      1219  23.53            flat     raced           3539      13195    3.52  8.0e-16  
      1220  2^2.5.61         flat     raced           3079      10469    3.40  4.9e-16  
      1226  2.613            prime    raced           6363      10152    1.50  5.9e-16  bluestein
      1227  3.409            prime    raced           6337       8935    1.40  6.8e-16  bluestein
      1228  2^2.307          flat     raced           5429      10399    1.90  8.6e-16  
      1233  3^2.137          flat     raced           4416      10714    2.18  7.2e-16  
      1234  2.617            prime    raced           6356      10189    1.60  6.4e-16  bluestein
      1236  2^2.3.103        flat     raced           3793       8772    2.29  9.1e-16  
      1238  2.619            prime    raced           6361      10240    1.61  5.5e-16  bluestein
      1239  3.7.59           flat     raced           5157       8133    1.54  6.2e-16  
      1241  17.73            flat     raced           3454      11935    3.40  8.0e-16  
      1243  11.113           flat     raced           3781      10625    2.77  5.9e-16  
      1244  2^2.311          flat     raced           5420      10456    1.91  7.6e-16  
      1245  3.5.83           flat     raced           6104       8969    1.47  6.9e-16  
      1246  2.7.89           flat     raced           4103      10803    2.62  5.6e-16  
      1251  3^2.139          flat     raced           5052       8494    1.66  7.6e-16  
      1252  2^2.313          flat     raced           4724      10459    2.12  7.4e-16  
      1253  7.179            prime    raced           6372       8265    1.29  5.9e-16  bluestein
      1255  5.251            flat     raced           6334       7120    1.11  7.8e-16  
      1256  2^3.157          flat     raced           3929       7456    1.87  7.2e-16  
      1257  3.419            prime    raced           6354       9462    1.48  7.1e-16  bluestein
      1261  13.97            flat     raced           4148       9623    2.27  5.4e-16  
      1262  2.631            prime    raced           6347       8338    1.31  4.3e-16  bluestein
      1263  3.421            prime    raced           6343       8993    1.41  6.8e-16  bluestein
      1264  2^4.79           flat     raced           3211       8018    2.42  7.5e-16  
      1266  2.3.211          flat     raced           6274       8497    1.33  9.4e-16  
      1267  7.181            flat     raced           5440      10408    1.87  7.0e-16  
      1268  2^2.317          prime    raced           6378       7453    1.16  5.8e-16  bluestein
      1270  2.5.127          flat     raced           5407       7384    1.35  9.2e-16  
      1272  2^3.3.53         flat     raced           3256      10043    3.02  8.3e-16  
      1273  19.67            flat     raced           3702      14718    3.95  6.5e-16  
      1278  2.3^2.71         flat     raced           3867       8655    2.21  5.6e-16  
      1281  3.7.61           flat     raced           3835      11057    2.64  5.1e-16  
      1282  2.641            prime    raced           7238      10160    1.29  7.7e-16  bluestein
      1284  2^2.3.107        flat     raced           6683       8784    1.31  1.2e-15  
      1285  5.257            flat     raced           6277       9757    1.54  6.5e-16  
      1286  2.643            prime    raced           7206       9925    1.35  6.6e-16  bluestein
      1293  3.431            prime    raced           7248       8980    1.23  6.8e-16  bluestein
      1294  2.647            prime    raced           7426       9882    1.31  6.6e-16  bluestein
      1298  2.11.59          flat     raced           5236       8888    1.68  9.2e-16  
      1299  3.433            prime    raced           7200       9949    1.37  7.8e-16  bluestein
      1304  2^3.163          flat     raced           4691       9349    1.97  8.0e-16  
      1306  2.653            prime    raced           7307      10882    1.48  6.7e-16  bluestein
      1308  2^2.3.109        flat     raced           4788      11326    2.34  5.9e-16  
      1310  2.5.131          flat     raced           4952       9190    1.85  8.3e-16  
      1313  13.101           flat     raced           5120       9723    1.84  7.8e-16  
      1314  2.3^2.73         flat     raced           3849       9590    2.49  8.9e-16  
      1315  5.263            prime    raced           7355       9963    1.28  9.1e-16  bluestein
      1317  3.439            prime    raced           7247       9977    1.37  5.4e-16  bluestein
      1318  2.659            prime    raced           7198      10993    1.52  7.2e-16  bluestein
      1322  2.661            prime    raced           7252      10913    1.50  5.9e-16  bluestein
      1324  2^2.331          flat     raced           4957      11191    2.20  7.9e-16  
      1325  5^2.53           flat     raced           3617      10205    2.78  5.8e-16  
      1328  2^4.83           flat     raced           5422       9509    1.63  7.8e-16  
      1329  3.443            prime    raced           7221       9978    1.38  6.4e-16  bluestein
      1335  3.5.89           flat     raced           3837      11559    2.72  9.2e-16  
      1336  2^3.167          prime    raced           7332       9618    1.31  5.8e-16  bluestein
      1337  7.191            flat     raced           5649       7912    1.38  6.9e-16  
      1338  2.3.223          prime    raced           7245       9262    1.27  5.7e-16  bluestein
      1339  13.103           flat     raced           4877      13649    2.78  8.4e-16  
      1340  2^2.5.67         flat     raced           3270      12646    3.82  7.3e-16  
      1341  3^2.149          flat     raced           5794       9312    1.58  7.7e-16  
      1342  2.11.61          flat     raced           4288      12337    2.87  6.2e-16  
      1343  17.79            flat     raced           4077      11903    2.90  8.8e-16  
      1345  5.269            prime    raced           7749       9881    1.27  5.5e-16  bluestein
      1346  2.673            prime    raced           7760      10856    1.39  6.9e-16  bluestein
      1347  3.449            prime    raced           7755      10084    1.29  6.9e-16  bluestein
      1348  2^2.337          flat     raced           5206      11145    2.14  6.1e-16  
      1349  19.71            flat     raced           3748      12917    3.44  7.3e-16  
      1351  7.193            flat     raced           6145       9006    1.46  6.1e-16  
      1354  2.677            prime    raced           8419      10481    1.23  8.3e-16  bluestein
      1355  5.271            flat     raced           7101      10045    1.38  8.0e-16  
      1356  2^2.3.113        flat     raced           3548      11972    3.32  6.4e-16  
      1357  23.59            flat     raced           5473      15289    2.79  7.2e-16  
      1358  2.7.97           flat     raced           4878       9930    1.98  6.3e-16  
      1359  3^2.151          flat     raced           5733       8298    1.42  1.0e-15  
      1362  2.3.227          prime    raced           7725       9096    1.16  6.5e-16  bluestein
      1366  2.683            prime    raced           7824      10486    1.25  7.1e-16  bluestein
      1370  2.5.137          flat     raced           5402      11566    2.12  6.0e-16  
      1371  3.457            prime    raced           7753      10129    1.29  5.9e-16  bluestein
      1374  2.3.229          flat     raced           7493       9169    1.21  6.0e-16  
      1378  2.13.53          flat     raced           4097      11075    2.69  7.7e-16  
      1379  7.197            flat     raced           5208       8931    1.68  6.2e-16  
      1382  2.691            prime    raced           7772      10224    1.29  6.1e-16  bluestein
      1383  3.461            prime    raced           7831       9705    1.16  6.0e-16  bluestein
      1384  2^3.173          flat     raced           5530      12803    2.30  7.5e-16  
      1385  5.277            flat     raced           7744       9891    1.23  7.6e-16  
      1387  19.73            flat     raced           3829      13549    3.49  7.6e-16  
      1388  2^2.347          prime    raced           7748       9664    1.24  7.7e-16  bluestein
      1389  3.463            prime    raced           7764       9697    1.24  6.7e-16  bluestein
      1390  2.5.139          flat     raced           6124       9509    1.53  6.8e-16  
      1391  13.107           prime    raced           7846       9777    1.19  6.7e-16  bluestein
      1393  7.199            flat     raced           6282       9063    1.44  7.2e-16  
      1396  2^2.349          flat     raced           6737       9574    1.40  8.8e-16  
      1397  11.127           flat     raced           5272       8168    1.52  7.8e-16  
      1398  2.3.233          prime    raced           7765       9087    1.17  6.3e-16  bluestein
      1401  3.467            prime    raced           8398      10169    1.20  7.2e-16  bluestein
      1402  2.701            prime    raced           8463      10394    1.21  8.9e-16  bluestein
      1403  23.61            flat     raced           4198      16139    3.74  5.7e-16  
      1405  5.281            flat     raced           6388       9877    1.53  5.8e-16  
      1407  3.7.67           flat     raced           4099      13231    2.89  7.0e-16  
      1411  17.83            flat     raced           6888      13344    1.92  8.3e-16  
      1412  2^2.353          flat     raced           5015       9761    1.88  7.4e-16  
      1413  3^2.157          flat     raced           5376       8872    1.59  9.9e-16  
      1414  2.7.101          flat     raced           6209      10261    1.50  7.7e-16  
      1415  5.283            prime    raced           8506       9906    1.16  8.7e-16  bluestein
      1416  2^3.3.59         flat     raced           5221       9072    1.72  7.7e-16  
      1417  13.109           flat     raced           5782      12611    1.90  8.8e-16  
      1418  2.709            prime    raced           8465      10378    1.21  7.5e-16  bluestein
      1420  2^2.5.71         flat     raced           3313      10109    2.41  6.2e-16  flips differ 1.26x
      1422  2.3^2.79         flat     raced           4755       9446    1.97  8.4e-16  
      1424  2^4.89           flat     raced           3649      12367    3.38  6.0e-16  
      1431  3^3.53           flat     raced           4172      11540    2.76  6.1e-16  
      1432  2^3.179          flat     raced           8024       9655    1.17  8.3e-16  
      1434  2.3.239          flat     raced           7201       9144    1.26  8.9e-16  
      1436  2^2.359          prime    raced           8489       9621    1.13  7.4e-16  bluestein
      1437  3.479            prime    raced           8478       9723    1.14  7.6e-16  bluestein
      1438  2.719            prime    raced           8560      10385    1.13  7.0e-16  bluestein
      1441  11.131           flat     raced           4857      10527    2.14  7.7e-16  
      1442  2.7.103          flat     raced           5534      10234    1.82  8.8e-16  
      1446  2.3.241          flat     raced           7250      12051    1.66  7.8e-16  
      1448  2^3.181          flat     raced           4657      12011    2.54  8.0e-16  
      1454  2.727            prime    raced          10588      13305    0.92  6.2e-16  flips differ 1.36x bluestein
      1455  3.5.97           flat     raced           4506      10628    2.26  5.8e-16  
      1460  2^2.5.73         flat     raced           3433      10705    3.11  6.8e-16  
      1461  3.487            prime    raced           9185      10486    1.01  6.7e-16  bluestein
      1464  2^3.3.61         flat     raced           4004      12557    3.03  6.5e-16  
      1465  5.293            prime    raced           9050      10056    1.10  7.6e-16  bluestein
      1466  2.733            prime    raced           9031      12164    1.31  7.4e-16  bluestein
      1467  3^2.163          flat     raced           6426      10912    1.67  7.2e-16  
      1468  2^2.367          prime    raced           9042      11450    1.24  7.2e-16  bluestein
      1469  13.113           flat     raced           4419      12661    2.80  8.0e-16  
      1473  3.491            prime    raced           7901      10458    1.25  7.1e-16  bluestein
      1474  2.11.67          flat     raced           4473      11813    2.52  7.1e-16  
      1475  5^2.59           flat     raced           5714      12445    2.12  8.1e-16  
      1477  7.211            flat     raced           6345      10112    1.58  8.3e-16  
      1478  2.739            prime    raced           7864      12292    1.56  5.5e-16  bluestein
      1484  2^2.7.53         flat     raced           3635      11417    3.12  6.9e-16  
      1486  2.743            prime    raced           7855      12232    1.55  6.8e-16  bluestein
      1490  2.5.149          flat     raced           6958      10472    1.50  8.4e-16  
      1491  3.7.71           flat     raced           4188      10702    2.55  6.8e-16  
      1492  2^2.373          flat     raced           7134      11471    1.58  7.9e-16  
      1494  2.3^2.83         flat     raced           7872      13768    1.72  6.2e-16  
      1497  3.499            prime    raced           7911      10546    1.32  6.5e-16  bluestein
      1498  2.7.107          prime    raced           7877      10512    1.33  6.6e-16  bluestein
      1501  19.79            flat     raced           4708      13741    2.89  9.1e-16  
      1502  2.751            prime    raced           7941      10838    1.30  5.4e-16  bluestein
      1503  3^2.167          prime    raced           7888      14502    1.76  6.7e-16  bluestein
      1506  2.3.251          flat     raced           8124       8610    1.05  7.1e-16  
      1507  11.137           flat     raced           5100      10818    2.11  8.3e-16  
      1509  3.503            prime    raced           7887       9131    1.13  8.0e-16  bluestein
      1510  2.5.151          flat     raced           6925       9220    1.30  7.7e-16  
      1513  17.89            flat     raced           4296      16477    3.67  6.8e-16  
      1514  2.757            prime    raced           7944      10569    1.32  5.8e-16  bluestein
      1515  3.5.101          flat     raced           5627      11095    1.74  8.1e-16  
      1516  2^2.379          flat     raced           6531       9578    1.44  7.9e-16  
      1522  2.761            prime    raced           7899      10609    1.33  7.2e-16  bluestein
      1524  2^2.3.127        flat     raced           4900       8630    1.71  8.5e-16  
      1525  5^2.61           flat     raced           4274      13443    3.10  6.0e-16  
      1526  2.7.109          flat     raced           6866      13304    1.87  7.6e-16  
      1527  3.509            prime    raced           7879       8983    0.99  8.2e-16  bluestein
      1528  2^3.191          flat     raced           4976       8821    1.67  5.5e-16  
      1529  11.139           flat     raced           5978      10747    1.79  7.7e-16  
      1532  2^2.383          prime    raced           7908       9667    1.22  6.9e-16  bluestein
      1533  3.7.73           flat     raced           4336      11412    2.49  8.3e-16  
      1534  2.13.59          flat     raced           6582      10545    1.56  7.2e-16  
      1535  5.307            flat     raced           8741      13149    1.49  9.1e-16  
      1537  29.53            flat     raced           4697      17375    3.69  9.9e-16  
      1538  2.769            prime    raced           8635      11227    1.29  6.7e-16  bluestein
      1541  23.67            flat     raced           4617      18619    4.01  8.8e-16  
      1542  2.3.257          flat     raced           7911      11818    1.46  6.5e-16  
      1544  2^3.193          flat     raced           4705      10171    2.15  6.0e-16  
      1545  3.5.103          flat     raced           5300      11098    2.08  9.5e-16  
      1546  2.773            prime    raced           8589      11021    1.21  5.5e-16  bluestein
      1552  2^4.97           flat     raced           3956      11410    2.52  7.6e-16  
      1555  5.311            prime    raced           8591      13283    1.53  7.1e-16  bluestein
      1556  2^2.389          prime    raced           8582      10991    1.28  5.4e-16  bluestein
      1557  3^2.173          flat     raced           7424      10933    1.46  7.8e-16  
      1561  7.223            flat     raced           8324      11072    1.32  7.4e-16  
      1562  2.11.71          flat     raced           4598      11115    2.37  7.0e-16  
      1563  3.521            prime    raced           8643      12278    1.41  6.6e-16  bluestein
      1565  5.313            flat     raced           7844      14136    1.80  7.3e-16  
      1569  3.523            prime    raced           8593      12302    1.42  5.5e-16  bluestein
      1570  2.5.157          flat     raced           6315       9854    1.51  8.7e-16  
      1572  2^2.3.131        flat     raced           4365      11199    2.52  8.4e-16  
      1574  2.787            prime    raced           8583      11262    1.31  5.6e-16  bluestein
      1576  2^3.197          flat     raced           4359       9954    2.15  6.1e-16  
      1577  19.83            flat     raced           7550      18365    2.23  6.6e-16  
      1578  2.3.263          prime    raced           8615      12014    1.39  5.9e-16  bluestein
      1580  2^2.5.79         flat     raced           4150      10599    2.55  9.7e-16  
      1582  2.7.113          flat     raced           5647      14035    2.48  5.3e-16  
      1585  5.317            prime    raced           8701       9425    1.02  5.2e-16  bluestein
      1586  2.13.61          flat     raced           4927      14280    2.89  6.3e-16  
      1588  2^2.397          flat     raced           5371      11051    2.01  9.1e-16  
      1589  7.227            prime    raced           8615      11120    1.28  4.9e-16  bluestein
      1590  2.3.5.53         flat     raced           5011      12192    2.39  6.5e-16  
      1592  2^3.199          flat     raced           5195       9882    1.84  7.0e-16  
      1593  3^3.59           flat     raced           6477      13502    2.06  7.8e-16  
      1594  2.797            prime    raced           8658      11605    1.20  6.3e-16  bluestein
      1602  2.3^2.89         flat     raced           5090      10736    2.06  6.9e-16  
      1603  7.229            flat     raced           7500      10800    1.25  6.6e-16  
      1604  2^2.401          flat     raced           6932      11947    1.47  7.4e-16  
      1605  3.5.107          prime    raced          10302      11181    1.06  6.6e-16  bluestein
      1606  2.11.73          flat     raced           4918      12330    2.42  7.4e-16  
      1608  2^3.3.67         flat     raced           4306      14677    3.35  7.6e-16  
      1611  3^2.179          prime    raced          10445      10930    1.03  7.7e-16  bluestein
      1614  2.3.269          prime    raced          10220      11905    1.16  6.8e-16  bluestein
      1616  2^4.101          flat     raced           5046      11669    2.21  7.4e-16  
      1618  2.809            prime    raced          10274      13256    1.27  8.0e-16  bluestein
      1622  2.811            prime    raced          10281      13104    1.27  6.2e-16  bluestein
      1623  3.541            flat     raced          13030      15469    1.14  7.1e-16  
      1626  2.3.271          flat     raced           8884      11675    1.27  7.3e-16  
      1629  3^2.181          flat     raced           6368      13490    2.06  7.4e-16  
      1630  2.5.163          flat     raced           7756      11702    1.50  9.6e-16  
      1631  7.233            flat     raced           7802      10721    1.34  7.4e-16  
      1633  23.71            flat     raced           4609      16339    3.54  7.2e-16  
      1635  3.5.109          flat     raced           6456      14284    1.82  6.7e-16  
      1636  2^2.409          flat     raced           6822      12258    1.66  8.5e-16  
      1639  11.149           flat     raced           6979      12155    1.70  7.0e-16  
      1641  3.547            flat     raced          12005      15299    1.13  9.9e-16  
      1642  2.821            prime    raced          10309      13167    1.26  6.9e-16  bluestein
      1643  31.53            flat     raced           4988      19306    3.75  7.5e-16  
      1644  2^2.3.137        flat     raced           4947      14167    2.80  7.5e-16  
      1646  2.823            prime    raced          10202      13348    1.13  7.5e-16  bluestein
      1647  3^3.61           flat     raced           4958      14354    2.89  6.5e-16  
      1648  2^4.103          flat     raced           5023      11851    2.23  8.3e-16  
      1649  17.97            flat     raced           5164      15721    3.01  6.4e-16  
      1651  13.127           flat     raced           6107       9712    1.54  8.7e-16  
      1652  2^2.7.59         flat     raced           5891      14075    2.20  5.7e-16  
      1654  2.827            prime    raced          10284      13113    1.25  6.1e-16  bluestein
      1655  5.331            flat     raced           8116      14055    1.70  8.6e-16  
      1658  2.829            prime    raced          10385      13209    1.25  7.2e-16  bluestein
      1659  3.7.79           flat     raced           5352      11203    2.06  8.4e-16  
      1660  2^2.5.83         flat     raced           6950      15195    2.18  7.0e-16  
      1661  11.151           flat     raced           6631      10735    1.56  6.9e-16  
      1662  2.3.277          flat     raced           9551      11754    1.22  8.9e-16  
      1668  2^2.3.139        flat     raced           5646      11235    1.90  9.7e-16  
      1670  2.5.167          prime    raced          10277      11995    1.13  5.8e-16  bluestein
      1671  3.557            prime    raced          10276      15333    1.49  5.6e-16  bluestein
      1673  7.239            flat     raced           7266      10788    1.47  8.2e-16  
      1675  5^2.67           flat     raced           4715      15823    3.33  9.2e-16  
      1676  2^2.419          flat     raced           6765      12254    1.80  8.1e-16  
      1678  2.839            prime    raced          10325      13319    1.26  6.2e-16  bluestein
      1679  23.73            flat     raced           4699      17283    3.64  7.8e-16  
      1684  2^2.421          flat     raced           6895      12245    1.77  8.7e-16  
      1685  5.337            flat     raced           7887      14017    1.64  7.3e-16  
      1686  2.3.281          flat     raced           8257      11817    1.42  6.5e-16  
      1687  7.241            flat     raced           6581      17719    2.27  7.1e-16  
      1688  2^3.211          flat     raced           5581      11619    2.07  8.7e-16  
      1689  3.563            prime    raced           9981      15279    1.53  7.2e-16  bluestein
      1691  19.89            flat     raced           5014      18702    3.65  8.7e-16  
      1695  3.5.113          flat     raced           5086      14581    2.85  6.9e-16  
      1696  2^5.53           flat     raced           4362      12793    2.86  7.9e-16  
      1698  2.3.283          prime    raced          10044      11940    1.09  6.9e-16  bluestein
      1703  13.131           flat     raced           5468      12697    2.23  9.8e-16  
      1704  2^3.3.71         flat     raced           4482      12180    2.65  5.6e-16  
      1706  2.853            prime    raced          10682      13161    1.19  6.4e-16  bluestein
      1707  3.569            prime    raced          10046      12507    1.24  7.0e-16  bluestein
      1708  2^2.7.61         flat     raced           4430      14870    3.34  4.7e-16  
      1711  29.59            flat     raced           7302      17219    2.24  7.2e-16  
      1712  2^4.107          flat     raced           8592      11961    1.39  1.2e-15  
      1713  3.571            prime    raced          10034      12387    1.22  5.9e-16  bluestein
      1714  2.857            prime    raced          10077      13283    1.22  8.4e-16  bluestein
      1717  17.101           flat     raced           6704      16254    2.25  9.6e-16  
      1718  2.859            prime    raced          10293      13243    1.14  6.2e-16  bluestein
      1719  3^2.191          flat     raced           6601      10133    1.50  5.9e-16  
      1724  2^2.431          flat     raced           8641      12291    1.41  9.0e-16  
      1726  2.863            prime    raced          10104      13282    1.29  7.5e-16  bluestein
      1727  11.157           flat     raced           6364      11158    1.70  1.2e-15  
      1730  2.5.173          flat     raced           8835      12040    1.34  7.0e-16  
      1731  3.577            prime    raced           9906      13298    1.14  7.3e-16  bluestein
      1732  2^2.433          flat     raced           7751      13510    1.56  7.7e-16  
      1735  5.347            prime    raced           9752      12086    1.22  5.6e-16  bluestein
      1737  3^2.193          flat     raced           7049      11562    1.56  7.8e-16  
      1738  2.11.79          flat     raced           5639      11636    2.03  9.5e-16  
      1742  2.13.67          flat     raced           5658      14247    2.50  8.7e-16  
      1743  3.7.83           flat     raced           8328      12582    1.50  6.2e-16  
      1744  2^4.109          flat     raced           7278      15327    1.95  8.5e-16  
      1745  5.349            flat     raced          10313      12138    1.17  7.5e-16  
      1746  2.3^2.97         flat     raced           5769      12666    2.19  6.6e-16  
      1749  3.11.53          flat     raced           5230      13973    2.23  7.3e-16  
      1751  17.103           flat     raced           5846      21395    3.59  1.0e-15  
      1752  2^3.3.73         flat     raced           4499      12315    2.72  7.8e-16  
      1754  2.877            prime    raced           9778      14503    1.48  6.0e-16  bluestein
      1756  2^2.439          prime    raced          10327      13425    1.28  5.5e-16  bluestein
      1757  7.251            flat     raced           8156      10233    1.25  6.8e-16  
      1758  2.3.293          prime    raced           9772      11938    1.13  6.5e-16  bluestein
      1761  3.587            prime    raced           9872      13190    1.33  6.7e-16  bluestein
      1762  2.881            prime    raced           9767      14229    1.30  6.4e-16  bluestein
      1765  5.353            flat     raced           8054      12235    1.49  6.1e-16  
      1766  2.883            prime    raced           9917      14262    1.34  6.8e-16  bluestein
      1769  29.61            flat     raced           5582      21729    3.81  6.4e-16  
      1770  2.3.5.59         flat     raced           7449      14882    1.97  6.7e-16  
      1772  2^2.443          flat     raced           7335      13722    1.86  1.1e-15  
      1773  3^2.197          flat     raced           6142      11130    1.79  5.6e-16  
      1774  2.887            prime    raced           9805      14386    1.28  7.1e-16  bluestein
      1775  5^2.71           flat     raced           4757      12162    2.53  6.2e-16  
      1778  2.7.127          flat     raced           7309      10941    1.40  8.9e-16  
      1779  3.593            prime    raced           9774      13430    1.36  4.7e-16  bluestein
      1780  2^2.5.89         flat     raced           4458      12001    2.67  6.5e-16  
      1781  13.137           flat     raced           6329      12640    1.96  8.7e-16  
      1784  2^3.223          flat     raced           7337      12548    1.70  8.0e-16  
      1788  2^2.3.149        flat     raced           6724      12993    1.93  8.8e-16  
      1790  2.5.179          prime    raced           9872      12156    1.23  6.4e-16  bluestein
      1791  3^2.199          flat     raced           7209      11070    1.52  7.8e-16  
      1793  11.163           flat     raced           7609      13501    1.75  6.9e-16  
      1795  5.359            prime    raced          10375      12153    1.17  6.1e-16  bluestein
      1796  2^2.449          flat     raced           6713      13675    2.02  6.5e-16  
      1797  3.599            prime    raced          10400      13475    1.27  6.0e-16  bluestein
      1799  7.257            flat     raced           7746      14271    1.82  6.7e-16  
      1802  2.17.53          flat     raced           5512      18294    3.30  8.5e-16  
      1803  3.601            prime    raced          10533      16613    1.57  6.5e-16  bluestein
      1807  13.139           flat     raced           7011      13018    1.81  7.7e-16  
      1808  2^4.113          flat     raced           4397      16224    3.37  6.4e-16  
      1809  3^3.67           flat     raced           5576      14278    2.50  7.6e-16  
      1810  2.5.181          flat     raced           7625      14878    1.94  6.1e-16  
      1812  2^2.3.151        flat     raced           6275      11171    1.74  7.2e-16  
      1814  2.907            prime    raced          10358      17655    1.70  6.3e-16  bluestein
      1816  2^3.227          flat     raced          10472      12287    1.14  8.1e-16  
      1817  23.79            flat     raced           5690      17461    3.03  8.6e-16  
      1818  2.3^2.101        flat     raced           7907      13117    1.64  8.1e-16  
      1819  17.107           prime    raced          10450      16723    1.55  6.1e-16  bluestein
      1821  3.607            prime    raced          10324      16683    1.61  6.3e-16  bluestein
      1822  2.911            prime    raced          10463      17645    1.68  6.3e-16  bluestein
      1825  5^2.73           flat     raced           4858      12851    2.62  7.6e-16  
      1826  2.11.83          flat     raced           8698      13652    1.56  7.3e-16  
      1828  2^2.457          flat     raced           7745      13177    1.66  7.5e-16  
      1829  31.59            flat     raced           7626      19246    2.42  6.9e-16  
      1830  2.3.5.61         flat     raced           6030      15920    2.50  6.5e-16  
      1832  2^3.229          flat     raced           6101      12384    1.89  7.0e-16  
      1834  2.7.131          flat     raced           6632      13283    1.95  8.9e-16  
      1835  5.367            prime    raced          10614      14541    1.37  5.9e-16  bluestein
      1837  11.167           prime    raced          10468      13428    1.27  6.1e-16  bluestein
      1838  2.919            prime    raced          10421      17695    1.67  6.1e-16  bluestein
      1839  3.613            prime    raced          10619      15350    1.10  7.0e-16  flips differ 1.31x bluestein
      1841  7.263            prime    raced          10373      14185    1.36  6.3e-16  bluestein
      1842  2.3.307          flat     raced          10433      16790    1.58  1.1e-15  
      1843  19.97            flat     raced           5821      17988    3.06  7.3e-16  
      1844  2^2.461          flat     raced           7819      13171    1.68  6.7e-16  
      1846  2.13.71          flat     raced           5785      13269    2.28  9.3e-16  
      1851  3.617            prime    raced          10598      15429    1.45  5.8e-16  bluestein
      1852  2^2.463          flat     raced           7606      13403    1.76  7.9e-16  
      1853  17.109           flat     raced           7163      20474    2.49  8.0e-16  
      1854  2.3^2.103        flat     raced           6853      13044    1.85  9.8e-16  
      1855  5.7.53           flat     raced           5192      14479    2.75  8.5e-16  
      1857  3.619            prime    raced          10407      15464    1.45  5.2e-16  bluestein
      1858  2.929            prime    raced          10385      17697    1.67  5.9e-16  bluestein
      1864  2^3.233          flat     raced           7197      12393    1.68  6.2e-16  
      1865  5.373            prime    raced          10536      14510    1.37  5.3e-16  bluestein
      1866  2.3.311          prime    raced          10413      15759    1.49  4.8e-16  bluestein
      1868  2^2.467          prime    raced          10421      13206    1.24  6.4e-16  bluestein
      1869  3.7.89           flat     raced           5496      12812    2.31  7.7e-16  
      1874  2.937            prime    raced          10369      17656    1.67  7.1e-16  bluestein
      1876  2^2.7.67         flat     raced           4670      17342    3.56  7.3e-16  
      1878  2.3.313          flat     raced           9948      15699    1.56  1.1e-15  
      1882  2.941            prime    raced          10403      14088    1.33  6.6e-16  bluestein
      1883  7.269            prime    raced          10432      14184    1.36  6.4e-16  bluestein
      1884  2^2.3.157        flat     raced           5807      11289    1.94  8.1e-16  
      1888  2^5.59           flat     raced           6832      15486    2.23  6.0e-16  
      1891  31.61            flat     raced           5973      24018    3.94  7.6e-16  
      1893  3.631            flat     raced          16540      12567    0.73  8.2e-16  
      1894  2.947            prime    raced          10487      14076    1.34  6.3e-16  bluestein
      1895  5.379            flat     raced          10376      12078    1.16  1.1e-15  
      1896  2^3.3.79         flat     raced           5357      12615    2.35  8.9e-16  
      1897  7.271            flat     raced           8981      14302    1.56  7.4e-16  
      1898  2.13.73          flat     raced           5720      14074    2.39  8.9e-16  
      1899  3^2.211          flat     raced           7353      13075    1.76  1.3e-15  
      1902  2.3.317          prime    raced          10425      11329    1.08  6.3e-16  bluestein
      1903  11.173           flat     raced           8743      18176    2.07  6.7e-16  
      1905  3.5.127          flat     raced           6833      11035    1.53  8.3e-16  
      1906  2.953            prime    raced          10344      14130    1.33  5.0e-16  bluestein
      1908  2^2.3^2.53       flat     raced           4933      15247    3.09  8.8e-16  
      1909  23.83            flat     raced           9174      23149    2.51  7.0e-16  
      1910  2.5.191          flat     raced           8465      11022    1.26  5.7e-16  
      1912  2^3.239          flat     raced           6285      12428    1.95  9.3e-16  
      1915  5.383            prime    raced          10462      11972    1.12  5.0e-16  bluestein
      1916  2^2.479          prime    raced          10343      13261    1.28  6.5e-16  bluestein
      1917  3^3.71           flat     raced           5564      13914    2.49  6.8e-16  
      1918  2.7.137          flat     raced           7630      13424    1.73  7.3e-16  
      1919  19.101           flat     raced           7876      18532    2.28  8.6e-16  
      1921  17.113           flat     raced           5842      21456    3.64  5.9e-16  
      1923  3.641            prime    raced          10420      14999    1.43  5.9e-16  bluestein
      1926  2.3^2.107        prime    raced          10643      13503    1.25  5.5e-16  bluestein
      1928  2^3.241          flat     raced           6015      20182    3.31  6.0e-16  
      1929  3.643            prime    raced          10437      14891    1.42  6.0e-16  bluestein
      1930  2.5.193          flat     raced           8176      13008    1.58  6.4e-16  
      1934  2.967            prime    raced          10394      16773    1.58  8.3e-16  bluestein
      1937  13.149           flat     raced           8228      14498    1.74  7.2e-16  
      1939  7.277            flat     raced           9618      13879    1.41  8.8e-16  
      1940  2^2.5.97         flat     raced           5128      14198    2.71  6.9e-16  
      1941  3.647            prime    raced          10381      15236    1.45  8.3e-16  bluestein
      1942  2.971            prime    raced          10400      16830    0.91  7.8e-16  flips differ 1.79x bluestein
      1943  29.67            flat     raced           6033      25443    4.21  6.6e-16  
      1945  5.389            prime    raced          10592      14133    1.32  5.6e-16  bluestein
      1946  2.7.139          flat     raced           8361      13284    1.55  7.6e-16  
      1947  3.11.59          flat     raced           7818      13334    1.69  7.5e-16  
      1948  2^2.487          flat     raced           9731      14334    1.44  9.2e-16  
      1952  2^5.61           flat     raced           5341      16727    3.12  5.2e-16  
      1954  2.977            prime    raced          10685      15088    1.40  7.8e-16  bluestein
      1956  2^2.3.163        flat     raced           7337      14322    1.92  9.4e-16  
      1957  19.103           flat     raced           6693      18726    2.50  8.2e-16  
      1958  2.11.89          flat     raced           6332      13981    2.14  6.8e-16  
      1959  3.653            prime    raced          10443      16429    1.57  6.3e-16  bluestein
      1961  37.53            flat     raced           6135      24491    3.95  7.4e-16  
      1962  2.3^2.109        flat     raced           7996      17080    1.63  7.4e-16  flips differ 1.31x
      1963  13.151           flat     raced           7980      12708    1.57  8.5e-16  
      1964  2^2.491          flat     raced           6911      14169    2.04  5.3e-16  
      1965  3.5.131          flat     raced           6136      14226    2.31  9.1e-16  
      1966  2.983            prime    raced          10766      15317    1.05  5.4e-16  flips differ 1.35x bluestein
      1967  7.281            flat     raced           7719      14178    1.82  5.0e-16  
      1969  11.179           prime    raced          10382      18320    1.76  6.0e-16  bluestein
      1970  2.5.197          flat     raced           7598      12322    1.61  5.4e-16  
      1971  3^3.73           flat     raced           5684      14186    2.42  7.5e-16  
      1975  5^2.79           flat     raced           5887      13692    2.31  9.1e-16  
      1977  3.659            prime    raced          10417      16553    1.58  6.9e-16  bluestein
      1981  7.283            prime    raced          10403      13922    1.32  5.9e-16  bluestein
      1982  2.991            prime    raced          10441      15093    1.43  5.5e-16  bluestein
      1983  3.661            prime    raced          10458      16548    1.57  7.7e-16  bluestein
      1985  5.397            flat     raced           8886      13923    1.54  7.9e-16  
      1986  2.3.331          flat     raced          10363      16749    1.61  9.7e-16  
      1988  2^2.7.71         flat     raced           4973      13645    2.32  5.0e-16  
      1990  2.5.199          flat     raced           8793      12511    1.41  6.8e-16  
      1991  11.181           flat     raced           7526      17070    1.97  7.5e-16  
      1992  2^3.3.83         flat     raced           8746      14380    1.47  7.1e-16  
      1994  2.997            prime    raced          10489      15117    1.43  5.7e-16  bluestein
      1996  2^2.499          prime    raced          10444      14280    1.34  5.2e-16  bluestein
      2004  2^2.3.167        prime    raced          10500      19820    1.83  4.5e-16  bluestein
      2005  5.401            flat     raced          10503      15075    1.40  7.7e-16  
      2006  2.17.59          flat     raced           8346      18018    2.13  7.3e-16  
      2007  3^2.223          flat     raced           9837      14416    1.41  6.3e-16  
      2008  2^3.251          flat     raced           6978      11627    1.67  7.1e-16  
      2010  2.3.5.67         flat     raced           6491      15721    2.41  6.9e-16  
      2012  2^2.503          prime    raced          10514      12639    1.16  5.7e-16  bluestein
      2013  3.11.61          flat     raced           6140      17979    2.92  6.9e-16  
      2014  2.19.53          flat     raced           6623      20156    2.85  8.5e-16  
      2018  2.1009           prime    raced          10638      20058    1.86  6.4e-16  bluestein
      2019  3.673            flat     raced          15538      16557    1.06  8.6e-16  
      2020  2^2.5.101        flat     raced           7380      14807    1.76  7.8e-16  
      2022  2.3.337          flat     raced          10952      16748    1.50  6.5e-16  
      2026  2.1013           prime    raced          10477      19994    1.91  6.1e-16  bluestein
      2031  3.677            prime    raced          10480      15506    1.46  7.3e-16  bluestein
      2032  2^4.127          flat     raced           6386      11758    1.81  9.8e-16  
      2033  19.107           prime    raced          10447      19101    1.82  7.7e-16  bluestein
      2034  2.3^2.113        flat     raced           6710      17156    2.39  6.6e-16  
      2036  2^2.509          prime    raced          10660      12250    1.15  6.2e-16  bluestein
      2037  3.7.97           flat     raced           6739      14934    2.20  7.7e-16  
      2038  2.1019           prime    raced          10414      13095    1.12  5.5e-16  bluestein
      2041  13.157           flat     raced           7209      13570    1.79  8.1e-16  
      2042  2.1021           prime    raced          10654      13074    1.22  5.6e-16  bluestein
      2043  3^2.227          prime    raced          10660      13932    1.29  6.6e-16  bluestein
      2044  2^2.7.73         flat     raced           5047      15070    2.97  7.3e-16  
      2045  5.409            flat     raced          11038      15323    1.38  8.2e-16  
      2047  23.89            flat     raced           6145      23633    3.84  7.1e-16  
      2049  3.683            prime    raced          13625      15749    1.14  7.0e-16  bluestein
      2051  7.293            prime    raced          13439      14068    1.04  6.5e-16  bluestein
      2054  2.13.79          flat     raced           7101      13945    1.91  9.4e-16  
      2055  3.5.137          flat     raced           6795      18211    2.66  9.0e-16  
      2056  2^3.257          flat     raced           6571      16038    2.41  6.4e-16  
      2059  29.71            flat     raced           6209      21559    3.23  6.9e-16  
      2060  2^2.5.103        flat     raced           5957      14596    2.28  7.8e-16  
      2061  3^2.229          flat     raced           8595      14026    1.61  8.7e-16  
      2062  2.1031           prime    raced          14067      17451    1.22  6.7e-16  bluestein
      2065  5.7.59           flat     raced           8080      17612    2.15  5.2e-16  
      2066  2.1033           prime    raced          13603      17371    1.27  6.9e-16  bluestein
      2067  3.13.53          flat     raced           6234      16832    2.67  7.7e-16  
      2071  19.109           flat     raced           8107      23255    2.82  9.3e-16  
      2073  3.691            prime    raced          14446      16326    1.11  7.8e-16  bluestein
      2074  2.17.61          flat     raced           6628      23480    3.52  5.6e-16  
      2075  5^2.83           flat     raced           9535      14942    1.55  6.1e-16  
      2076  2^2.3.173        flat     raced           8464      14439    1.70  8.6e-16  
      2077  31.67            flat     raced           6530      27462    4.19  8.5e-16  
      2078  2.1039           prime    raced          13789      17375    1.26  7.6e-16  bluestein
      2082  2.3.347          prime    raced          13401      14512    0.97  6.3e-16  bluestein
      2084  2^2.521          flat     raced           8063      16404    2.01  8.5e-16  
      2085  3.5.139          flat     raced           7969      18306    2.26  8.7e-16  
      2086  2.7.149          flat     raced           9723      15065    1.53  8.6e-16  
      2092  2^2.523          flat     raced          11066      16470    1.48  9.6e-16  
      2094  2.3.349          flat     raced          12764      14487    1.13  7.9e-16  
      2095  5.419            flat     raced          10972      15419    1.34  7.5e-16  
      2096  2^4.131          flat     raced           5686      15307    2.41  6.9e-16  
      2097  3^2.233          flat     raced           9463      14206    1.32  8.4e-16  
      2098  2.1049           prime    raced          13450      17478    1.29  6.7e-16  bluestein
      2101  11.191           flat     raced           7857      12828    1.61  6.7e-16  
      2102  2.1051           prime    raced          13411      17433    1.28  6.9e-16  bluestein
      2103  3.701            prime    raced          13465      15955    1.17  7.2e-16  bluestein
      2104  2^3.263          flat     raced          12725      16345    1.28  1.2e-15  
      2105  5.421            flat     raced          10854      15513    1.38  6.9e-16  
      2110  2.5.211          flat     raced           9339      14516    1.54  7.8e-16  
      2114  2.7.151          flat     raced           9759      13168    1.33  9.0e-16  
      2117  29.73            flat     raced           6306      23198    3.65  8.3e-16  
      2118  2.3.353          flat     raced          10848      14527    1.32  7.1e-16  
      2119  13.163           flat     raced           8694      16112    1.83  7.7e-16  
      2120  2^3.5.53         flat     raced           5477      16204    2.95  8.9e-16  
      2121  3.7.101          flat     raced           8426      15449    1.68  8.2e-16  
      2122  2.1061           prime    raced          13452      17428    1.16  9.0e-16  bluestein
      2123  11.193           flat     raced           8007      14701    1.79  9.5e-16  
      2124  2^2.3^2.59       flat     raced           7805      13788    1.75  8.7e-16  
      2126  2.1063           prime    raced          13464      17485    1.26  7.3e-16  bluestein
      2127  3.709            prime    raced          13600      16030    1.13  7.0e-16  bluestein
      2130  2.3.5.71         flat     raced           6309      14460    2.22  7.5e-16  
      2133  3^3.79           flat     raced           6766      14535    2.09  1.1e-15  
      2134  2.11.97          flat     raced           7500      16274    2.16  5.4e-16  
      2135  5.7.61           flat     raced           6076      18762    2.93  5.6e-16  
      2136  2^3.3.89         flat     raced           5939      14481    2.43  6.5e-16  
      2138  2.1069           prime    raced          13636      17460    1.19  7.1e-16  bluestein
      2140  2^2.5.107        flat     raced          11089      14972    1.34  1.0e-15  
      2144  2^5.67           flat     raced           5740      16602    2.86  6.0e-16  
      2147  19.113           flat     raced           6325      24427    3.84  6.4e-16  
      2148  2^2.3.179        flat     raced          11742      14762    1.24  1.1e-15  
      2149  7.307            flat     raced          10882      19634    1.79  1.1e-15  
      2151  3^2.239          flat     raced           8104      14026    1.65  8.7e-16  
      2152  2^3.269          flat     raced          12008      16528    1.37  1.1e-15  
      2154  2.3.359          prime    raced          13454      14546    1.06  8.1e-16  bluestein
      2155  5.431            prime    raced          13668      15576    1.08  7.4e-16  bluestein
      2157  3.719            prime    raced          13622      15872    1.15  6.3e-16  bluestein
      2158  2.13.83          flat     raced          10587      16474    1.55  8.2e-16  
      2159  17.127           flat     raced           7820      18214    2.29  8.9e-16  
      2163  3.7.103          flat     raced           7464      15613    2.08  8.7e-16  
      2164  2^2.541          flat     raced          10029      20791    1.98  7.7e-16  
      2165  5.433            flat     raced          12532      17130    1.30  7.7e-16  
      2167  11.197           flat     raced           7260      13891    1.82  6.1e-16  
      2168  2^3.271          flat     raced           7807      16084    2.03  8.5e-16  
      2169  3^2.241          flat     raced           7854      18209    2.32  8.6e-16  
      2171  13.167           prime    raced          12931      16142    1.08  6.6e-16  bluestein
      2172  2^2.3.181        flat     raced           6870      18254    2.54  6.8e-16  
      2173  41.53            flat     raced           7035      28129    3.99  8.3e-16  
      2174  2.1087           prime    raced          12835      23088    1.79  7.3e-16  bluestein
      2177  7.311            flat     raced          10857      19791    1.80  8.3e-16  
      2180  2^2.5.109        flat     raced           7495      19092    2.43  7.8e-16  
      2181  3.727            prime    raced          12792      20105    1.56  6.8e-16  bluestein
      2182  2.1091           prime    raced          12669      23129    1.82  7.9e-16  bluestein
      2183  37.59            flat     raced           9320      28848    3.06  5.4e-16  
      2186  2.1093           prime    raced          12838      23239    1.80  6.8e-16  bluestein
      2188  2^2.547          flat     raced           9967      20620    2.05  9.1e-16  
      2189  11.199           flat     raced           8578      13974    1.62  7.3e-16  
      2190  2.3.5.73         flat     raced           6801      15316    2.23  7.1e-16  
      2191  7.313            flat     raced           9551      19762    2.02  9.0e-16  
      2192  2^4.137          flat     raced           6534      19059    2.87  6.6e-16  
      2194  2.1097           prime    raced          12765      19810    1.55  5.2e-16  bluestein
      2195  5.439            prime    raced          12709      17025    1.31  6.1e-16  bluestein
      2196  2^2.3^2.61       flat     raced           5981      19090    3.17  5.3e-16  
      2198  2.7.157          flat     raced           8750      14093    1.57  8.7e-16  
      2199  3.733            prime    raced          12740      18333    1.42  8.0e-16  bluestein
      2201  31.71            flat     raced           6638      23927    3.55  6.8e-16  
      2202  2.3.367          prime    raced          12718      17363    1.34  5.8e-16  bluestein
      2206  2.1103           prime    raced          12894      19798    1.53  6.6e-16  bluestein
      2211  3.11.67          flat     raced           6921      17973    2.56  7.0e-16  
      2212  2^2.7.79         flat     raced           6108      14301    2.06  8.7e-16  
      2215  5.443            flat     raced          11637      17197    1.47  1.0e-15  
      2216  2^3.277          flat     raced           8123      16076    1.91  8.4e-16  
      2217  3.739            prime    raced          12798      18541    1.44  5.7e-16  bluestein
      2218  2.1109           prime    raced          12655      19764    1.56  6.8e-16  bluestein
      2219  7.317            prime    raced          12815      13354    0.96  5.1e-16  bluestein
      2222  2.11.101         flat     raced           9813      16815    1.59  7.6e-16  
      2224  2^4.139          flat     raced           7341      19183    2.43  7.1e-16  
      2225  5^2.89           flat     raced           6245      15477    2.13  7.3e-16  
      2226  2.3.7.53         flat     raced           7140      17886    2.44  8.9e-16  
      2227  17.131           flat     raced           6966      21122    2.93  1.0e-15  
      2228  2^2.557          prime    raced          12676      20633    1.48  6.5e-16  bluestein
      2229  3.743            prime    raced          12773      18469    1.44  6.9e-16  bluestein
      2230  2.5.223          flat     raced          11890      15837    1.33  8.4e-16  
      2231  23.97            flat     raced           7243      22752    3.12  7.5e-16  
      2234  2.1117           prime    raced          13095      19785    1.35  7.1e-16  bluestein
      2235  3.5.149          flat     raced           9307      16667    1.77  7.4e-16  
      2238  2.3.373          flat     raced          13905      17320    1.23  9.2e-16  
      2241  3^3.83           flat     raced          10751      16322    1.51  7.0e-16  
      2242  2.19.59          flat     raced           9955      20503    2.05  6.5e-16  
      2245  5.449            flat     raced          10949      17022    1.54  6.2e-16  
      2246  2.1123           prime    raced          12863      19947    1.52  6.7e-16  bluestein
      2247  3.7.107          prime    raced          12893      15760    1.16  5.8e-16  bluestein
      2248  2^3.281          flat     raced           6461      16032    2.45  6.8e-16  
      2249  13.173           flat     raced          10261      16227    1.51  8.8e-16  
      2252  2^2.563          prime    raced          12891      20922    1.52  5.7e-16  bluestein
      2253  3.751            prime    raced          13498      16122    1.18  6.7e-16  bluestein
      2257  37.61            flat     raced           7556      30311    4.01  6.3e-16  
      2258  2.1129           prime    raced          12873      17487    1.35  6.4e-16  bluestein
      2259  3^2.251          flat     raced           9522      13223    1.38  9.1e-16  
      2260  2^2.5.113        flat     raced           5888      19129    3.24  6.1e-16  
      2263  31.73            flat     raced           6837      25981    3.78  8.7e-16  
      2264  2^3.283          flat     raced          10592      16304    1.50  7.8e-16  
      2265  3.5.151          flat     raced           8869      14406    1.62  8.9e-16  
      2266  2.11.103         flat     raced           8156      16905    2.04  9.6e-16  
      2270  2.5.227          prime    raced          12923      15278    1.17  6.2e-16  bluestein
      2271  3.757            prime    raced          13482      16305    1.15  6.1e-16  bluestein
      2272  2^5.71           flat     raced           5887      15646    2.57  6.1e-16  
      2274  2.3.379          flat     raced          12819      14454    1.10  7.9e-16  
      2276  2^2.569          prime    raced          13805      16442    1.17  6.3e-16  bluestein
      2278  2.17.67          flat     raced           7437      26536    3.51  8.0e-16  
      2279  43.53            flat     raced           7809      30684    3.73  8.4e-16  
      2282  2.7.163          flat     raced          10420      16903    1.60  7.9e-16  
      2283  3.761            prime    raced          13144      16176    1.23  7.4e-16  bluestein
      2284  2^2.571          flat     raced          11220      16834    1.46  8.0e-16  
      2285  5.457            flat     raced          12608      16513    1.30  8.1e-16  
      2286  2.3^2.127        flat     raced           9387      13054    1.38  7.2e-16  
      2289  3.7.109          flat     raced           9513      20173    2.11  9.6e-16  
      2290  2.5.229          flat     raced          10329      15299    1.47  7.7e-16  
      2291  29.79            flat     raced           7432      22936    3.08  9.1e-16  
      2292  2^2.3.191        flat     raced           7536      13559    1.77  6.1e-16  
      2298  2.3.383          prime    raced          12888      14269    1.10  7.5e-16  bluestein
      2301  3.13.59          flat     raced           9990      15950    1.59  8.2e-16  
      2302  2.1151           prime    raced          12891      17794    1.37  6.2e-16  bluestein
      2305  5.461            flat     raced          12606      16568    1.28  7.9e-16  
      2306  2.1153           prime    raced          14211      18580    1.30  5.7e-16  bluestein
      2307  3.769            flat     raced          15938      16950    1.02  7.7e-16  
      2308  2^2.577          flat     raced           8967      18049    1.97  5.8e-16  
      2313  3^2.257          flat     raced           9116      18143    1.98  6.9e-16  
      2314  2.13.89          flat     raced           7217      16781    2.30  1.0e-15  
      2315  5.463            flat     raced          12238      16584    1.35  8.8e-16  
      2316  2^2.3.193        flat     raced           7291      14900    2.02  7.1e-16  
      2317  7.331            flat     raced           9699      19649    2.01  8.3e-16  
      2318  2.19.61          flat     raced           8120      26188    3.18  6.7e-16  
      2319  3.773            prime    raced          14610      17114    0.87  5.4e-16  flips differ 1.35x bluestein
      2321  11.211           flat     raced           8968      16661    1.84  7.1e-16  
      2323  23.101           flat     raced           8946      23728    2.40  7.9e-16  
      2324  2^2.7.83         flat     raced           9699      21615    2.03  6.5e-16  
      2326  2.1163           prime    raced          14281      18612    1.07  6.7e-16  bluestein
      2327  13.179           prime    raced          14237      16468    1.12  8.1e-16  bluestein
      2328  2^3.3.97         flat     raced           6775      17051    2.51  5.9e-16  
      2329  17.137           flat     raced           7780      25793    3.27  1.0e-15  
      2330  2.5.233          flat     raced          11510      15543    1.13  8.0e-16  
      2332  2^2.11.53        flat     raced           6422      18636    2.85  7.1e-16  
      2334  2.3.389          prime    raced          14540      16646    1.14  5.6e-16  bluestein
      2335  5.467            prime    raced          14249      16559    1.12  5.9e-16  bluestein
      2336  2^5.73           flat     raced           6120      16805    2.69  8.2e-16  
      2338  2.7.167          prime    raced          15337      17154    1.03  6.7e-16  bluestein
      2342  2.1171           prime    raced          14662      18718    1.26  6.0e-16  bluestein
      2343  3.11.71          flat     raced           6875      16680    2.42  8.0e-16  
      2344  2^3.293          flat     raced          12482      16318    1.15  9.8e-16  
      2345  5.7.67           flat     raced           6756      21900    3.24  5.8e-16  
      2348  2^2.587          prime    raced          14560      18054    1.24  6.4e-16  bluestein
      2353  13.181           flat     raced           8824      20349    2.28  8.3e-16  
      2354  2.11.107         flat     raced          13507      17253    1.26  1.3e-15  
      2355  3.5.157          flat     raced           8123      15657    1.92  1.0e-15  
      2358  2.3^2.131        flat     raced           8348      17181    2.01  8.9e-16  
      2359  7.337            flat     raced          10191      19758    1.83  7.5e-16  
      2360  2^3.5.59         flat     raced           8541      15321    1.77  8.7e-16  
      2361  3.787            prime    raced          15058      17472    1.15  7.6e-16  bluestein
      2362  2.1181           prime    raced          14439      18630    1.27  7.2e-16  bluestein
      2363  17.139           flat     raced           8950      21600    2.40  1.1e-15  
      2364  2^2.3.197        flat     raced           6853      14895    2.15  6.1e-16  
      2367  3^2.263          prime    raced          14926      18065    1.20  8.6e-16  bluestein
      2369  23.103           flat     raced           8211      23782    2.81  1.0e-15  
      2370  2.3.5.79         flat     raced           7827      15234    1.93  8.5e-16  
      2372  2^2.593          flat     raced          10504      17979    1.66  7.1e-16  
      2373  3.7.113          flat     raced           7315      21487    2.91  6.4e-16  
      2374  2.1187           prime    raced          14486      18806    1.28  7.3e-16  bluestein
      2379  3.13.61          flat     raced           7287      21492    2.84  6.2e-16  
      2382  2.3.397          flat     raced          12003      17115    1.41  7.5e-16  
      2384  2^4.149          flat     raced           8613      17417    1.92  7.1e-16  
      2385  3^2.5.53         flat     raced           6775      18805    2.72  9.3e-16  
      2386  2.1193           prime    raced          14699      18674    1.27  7.6e-16  bluestein
      2388  2^2.3.199        flat     raced           7722      14610    1.88  8.6e-16  
      2390  2.5.239          flat     raced          10451      15492    1.48  7.3e-16  
      2391  3.797            prime    raced          14510      17981    1.17  8.0e-16  bluestein
      2395  5.479            prime    raced          14700      16868    1.12  6.3e-16  bluestein
      2396  2^2.599          flat     raced          11067      18097    1.59  8.5e-16  
      2398  2.11.109         flat     raced          11270      21765    1.81  6.5e-16  
      2402  2.1201           flat     raced          24310      21450    0.88  8.5e-16  
      2403  3^3.89           flat     raced           7277      16942    2.31  8.5e-16  
      2404  2^2.601          flat     raced          10875      22504    2.01  9.8e-16  
      2406  2.3.401          flat     raced          14070      18165    1.23  6.2e-16  
      2407  29.83            flat     raced          11753      26041    2.20  9.5e-16  
      2409  3.11.73          flat     raced           7088      19007    2.61  6.9e-16  
      2410  2.5.241          flat     raced           9925      25374    2.55  7.4e-16  
      2412  2^2.3^2.67       flat     raced           6422      22242    3.44  6.6e-16  
      2413  19.127           flat     raced           9403      20620    2.14  8.5e-16  
      2414  2.17.71          flat     raced           7779      23191    2.92  5.7e-16  
      2416  2^4.151          flat     raced           8039      15308    1.88  7.3e-16  
      2419  41.59            flat     raced          10606      33075    3.12  5.9e-16  
      2421  3^2.269          flat     raced          21391      18832    0.81  1.3e-15  
      2422  2.7.173          flat     raced          12128      17408    1.43  8.2e-16  
      2424  2^3.3.101        flat     raced           8645      17812    2.05  1.1e-15  
      2425  5^2.97           flat     raced           7348      18101    2.45  6.3e-16  
      2426  2.1213           prime    raced          21321      22428    1.02  7.0e-16  bluestein
      2427  3.809            prime    raced          21300      20220    0.93  7.0e-16  bluestein
      2428  2^2.607          prime    raced          21287      22514    0.73  6.0e-16  flips differ 1.45x bluestein
      2429  7.347            prime    raced          21638      17091    0.66  5.5e-16  bluestein
      2433  3.811            flat     raced          22636      20506    0.89  8.7e-16  
      2434  2.1217           prime    raced          16390      21384    1.30  7.7e-16  bluestein
      2435  5.487            flat     raced          15419      17870    1.10  8.2e-16  
      2438  2.23.53          flat     raced           8047      26350    3.25  7.0e-16  
      2439  3^2.271          flat     raced          10561      18005    1.67  7.4e-16  
      2440  2^3.5.61         flat     raced           6498      21027    2.94  5.3e-16  
      2443  7.349            flat     raced          13027      17138    1.25  8.0e-16  
      2445  3.5.163          flat     raced          10175      18255    1.77  8.9e-16  
      2446  2.1223           prime    raced          16472      21376    1.24  7.5e-16  bluestein
      2449  31.79            flat     raced           8239      25542    3.07  7.7e-16  
      2452  2^2.613          flat     raced           9850      20794    2.05  9.1e-16  
      2453  11.223           flat     raced          11655      18177    1.55  7.4e-16  
      2454  2.3.409          flat     raced          13670      18483    1.33  8.8e-16  
      2455  5.491            flat     raced          11739      17981    1.53  6.3e-16  
      2456  2^3.307          flat     raced           9205      21728    2.34  1.0e-15  
      2458  2.1229           prime    raced          13575      21484    1.57  7.4e-16  bluestein
      2461  23.107           prime    raced          13649      31657    2.25  5.6e-16  bluestein
      2462  2.1231           prime    raced          13650      21354    1.55  6.2e-16  bluestein
      2463  3.821            prime    raced          13808      19689    1.42  9.2e-16  bluestein
      2466  2.3^2.137        flat     raced           8730      16977    1.88  8.4e-16  
      2468  2^2.617          flat     raced           8880      20760    2.31  5.9e-16  
      2469  3.823            prime    raced          13867      20077    1.44  7.4e-16  bluestein
      2471  7.353            flat     raced          10336      17178    1.65  7.0e-16  
      2472  2^3.3.103        flat     raced           7876      17752    2.23  1.0e-15  
      2474  2.1237           prime    raced          13603      21348    1.56  7.6e-16  bluestein
      2476  2^2.619          prime    raced          13668      20860    1.52  7.6e-16  bluestein
      2478  2.3.7.59         flat     raced          11014      16287    1.46  7.6e-16  
      2479  37.67            flat     raced           8120      35183    4.30  7.1e-16  
      2481  3.827            prime    raced          13612      20067    1.45  8.6e-16  bluestein
      2482  2.17.73          flat     raced           7382      24385    3.22  6.6e-16  
      2483  13.191           flat     raced           9269      15330    1.64  8.9e-16  
      2485  5.7.71           flat     raced           6777      17203    2.45  5.7e-16  
      2486  2.11.113         flat     raced           8285      22043    2.60  5.8e-16  
      2487  3.829            prime    raced          13579      20430    1.49  6.1e-16  bluestein
      2488  2^3.311          flat     raced           9643      21897    2.20  9.1e-16  
      2489  19.131           flat     raced           7944      24378    3.01  7.1e-16  
      2490  2.3.5.83         flat     raced          11782      17921    1.51  6.2e-16  
      2491  47.53            flat     raced           8470      31685    3.70  8.5e-16  
      2492  2^2.7.89         flat     raced           6543      17079    2.59  6.4e-16  
      2493  3^2.277          flat     raced          11241      18508    1.60  8.9e-16  
      2495  5.499            prime    raced          13903      17975    1.29  5.8e-16  bluestein
      2497  11.227           prime    raced          13654      17756    1.30  5.7e-16  bluestein
      2498  2.1249           prime    raced          13737      21667    1.52  6.2e-16  bluestein
      2501  41.61            flat     raced           8837      35253    3.94  5.2e-16  
      2502  2.3^2.139        flat     raced          10310      17436    1.63  8.7e-16  
      2504  2^3.313          flat     raced           8182      21873    2.65  7.8e-16  
      2505  3.5.167          prime    raced          13675      25464    1.86  5.8e-16  bluestein
      2506  2.7.179          prime    raced          13654      17498    1.14  5.9e-16  bluestein
      2507  23.109           flat     raced          11151      30488    2.71  7.7e-16  
      2509  13.193           flat     raced           9299      17992    1.86  7.3e-16  
      2510  2.5.251          flat     raced          11837      14566    1.20  7.2e-16  
      2512  2^4.157          flat     raced           7268      15539    2.10  7.1e-16  
      2513  7.359            prime    raced          14097      17239    1.14  5.4e-16  bluestein
      2514  2.3.419          prime    raced          13877      18514    1.33  7.5e-16  bluestein
      2515  5.503            prime    raced          13631      15553    1.11  7.0e-16  bluestein
      2517  3.839            prime    raced          15890      19866    1.24  7.2e-16  bluestein
      2518  2.1259           prime    raced          13544      19607    1.44  5.9e-16  bluestein
      2519  11.229           flat     raced          10404      17921    1.66  6.9e-16  
      2522  2.13.97          flat     raced           8365      19411    2.26  7.1e-16  
      2524  2^2.631          flat     raced          13019      16868    1.29  8.6e-16  
      2525  5^2.101          flat     raced          10057      18673    1.71  1.0e-15  
      2526  2.3.421          flat     raced          14374      18539    1.28  7.3e-16  
      2528  2^5.79           flat     raced           7505      17316    2.30  8.3e-16  
      2529  3^2.281          flat     raced           9186      18139    1.96  6.7e-16  
      2532  2^2.3.211        flat     raced           8645      17882    2.06  9.8e-16  
      2533  17.149           flat     raced          10693      24222    2.25  9.6e-16  
      2534  2.7.181          flat     raced          10243      21405    2.04  6.0e-16  
      2536  2^3.317          prime    raced          13906      15354    1.10  5.8e-16  bluestein
      2537  43.59            flat     raced          11293      35311    3.11  7.0e-16  
      2540  2^2.5.127        flat     raced           8278      14673    1.74  8.3e-16  
      2544  2^4.3.53         flat     raced           6377      20299    3.12  7.2e-16  
      2545  5.509            prime    raced          14443      15295    1.05  5.8e-16  bluestein
      2546  2.19.67          flat     raced           8813      26617    2.98  8.4e-16  
      2547  3^2.283          prime    raced          13653      18102    1.32  5.9e-16  bluestein
      2554  2.1277           prime    raced          13560      19508    1.42  5.7e-16  bluestein
      2555  5.7.73           flat     raced           7108      18815    2.19  6.9e-16  
      2556  2^2.3^2.71       flat     raced           6656      17541    2.59  6.2e-16  
      2558  2.1279           prime    raced          13582      19444    1.43  6.1e-16  bluestein
      2559  3.853            prime    raced          13549      20578    1.50  7.1e-16  bluestein
      2561  13.197           flat     raced           8616      16809    1.91  6.0e-16  
      2562  2.3.7.61         flat     raced           8837      22481    2.40  6.1e-16  
      2563  11.233           flat     raced          11382      17923    1.37  8.0e-16  
      2564  2^2.641          flat     raced           9570      20908    2.10  7.6e-16  
      2566  2.1283           prime    raced          16069      22135    1.19  7.4e-16  bluestein
      2567  17.151           flat     raced           9814      22176    2.25  1.0e-15  
      2568  2^3.3.107        flat     raced          13750      17957    1.14  1.3e-15  
      2569  7.367            prime    raced          16041      20378    1.26  6.2e-16  bluestein
      2570  2.5.257          flat     raced          11281      20158    1.78  5.4e-16  
      2571  3.857            prime    raced          16048      20626    1.24  7.6e-16  bluestein
      2572  2^2.643          prime    raced          15791      21456    1.20  7.3e-16  bluestein
      2573  31.83            flat     raced          12562      28845    2.27  1.0e-15  
      2575  5^2.103          flat     raced           8494      19162    2.05  8.6e-16  
      2577  3.859            prime    raced          15670      20146    1.13  7.4e-16  bluestein
      2578  2.1289           prime    raced          16045      22391    1.24  7.3e-16  bluestein
      2581  29.89            flat     raced           8068      26673    3.26  7.8e-16  
      2582  2.1291           prime    raced          15758      22485    1.24  6.9e-16  bluestein
      2586  2.3.431          flat     raced          16578      18746    1.13  7.0e-16  
      2587  13.199           flat     raced           9861      17052    1.72  8.8e-16  
      2588  2^2.647          flat     raced          13810      20327    1.45  9.8e-16  
      2589  3.863            prime    raced          15812      20034    1.26  7.7e-16  bluestein
      2594  2.1297           prime    raced          15781      23069    1.45  7.3e-16  bluestein
      2595  3.5.173          flat     raced          11543      18462    1.59  8.3e-16  
      2596  2^2.11.59        flat     raced           9989      17535    1.73  7.1e-16  
      2597  7^2.53           flat     raced           7402      20678    2.13  6.7e-16  flips differ 1.31x
      2598  2.3.433          flat     raced          16346      20373    1.11  7.1e-16  
      2599  23.113           flat     raced           8031      31256    3.80  6.2e-16  
      2602  2.1301           prime    raced          16106      23004    1.40  7.6e-16  bluestein
      2603  19.137           flat     raced           8817      29676    3.14  8.4e-16  
      2605  5.521            flat     raced          13040      20687    1.58  9.1e-16  
      2606  2.1303           prime    raced          15813      22817    1.42  8.2e-16  bluestein
      2607  3.11.79          flat     raced           8350      17490    2.06  1.0e-15  
      2608  2^4.163          flat     raced           9252      19461    1.95  8.0e-16  
      2611  7.373            flat     raced          14143      20421    1.40  9.7e-16  
      2612  2^2.653          prime    raced          15728      22238    1.38  6.4e-16  bluestein
      2613  3.13.67          flat     raced           8215      21742    2.62  5.8e-16  
      2614  2.1307           prime    raced          15590      22750    1.45  7.5e-16  bluestein
      2615  5.523            flat     raced          17534      20773    1.18  9.3e-16  
      2616  2^3.3.109        flat     raced           9860      22925    2.00  7.5e-16  
      2619  3^3.97           flat     raced           8437      19476    2.31  8.0e-16  
      2620  2^2.5.131        flat     raced           8689      19014    2.15  8.2e-16  
      2623  43.61            flat     raced           8778      37092    4.09  6.3e-16  
      2626  2.13.101         flat     raced          10647      20091    1.66  9.5e-16  
      2627  37.71            flat     raced           8088      30526    3.74  8.1e-16  
      2628  2^2.3^2.73       flat     raced           6779      18358    2.69  6.9e-16  
      2629  11.239           flat     raced           9901      17920    1.80  7.9e-16  
      2630  2.5.263          prime    raced          15722      20068    1.26  6.7e-16  bluestein
      2631  3.877            prime    raced          16591      21814    1.31  6.1e-16  bluestein
      2634  2.3.439          prime    raced          15721      20455    1.29  7.4e-16  bluestein
      2636  2^2.659          flat     raced          14172      22263    1.56  7.3e-16  
      2637  3^2.293          prime    raced          15828      18379    1.10  7.2e-16  bluestein
      2638  2.1319           prime    raced          15838      22993    1.30  6.4e-16  bluestein
      2641  19.139           flat     raced          10692      29357    2.73  9.6e-16  
      2642  2.1321           prime    raced          15786      23077    1.42  9.4e-16  bluestein
      2643  3.881            prime    raced          15793      21722    1.35  5.9e-16  bluestein
      2644  2^2.661          flat     raced          10390      22268    2.10  6.2e-16  
      2648  2^3.331          flat     raced           8748      22652    2.46  7.0e-16  
      2649  3.883            prime    raced          16084      22285    1.24  7.7e-16  bluestein
      2650  2.5^2.53         flat     raced           8154      21533    2.61  6.3e-16  
      2651  11.241           flat     raced           9262      28803    3.01  7.9e-16  
      2653  7.379            flat     raced          12751      17156    1.33  9.4e-16  
      2654  2.1327           prime    raced          15764      23115    1.42  7.0e-16  bluestein
      2655  3^2.5.59         flat     raced          10446      18211    1.74  8.5e-16  
      2656  2^5.83           flat     raced          11452      19411    1.67  6.8e-16  
      2658  2.3.443          flat     raced          14942      20420    1.32  1.1e-15  
      2661  3.887            prime    raced          15796      21707    1.23  8.0e-16  bluestein
      2667  3.7.127          flat     raced           9659      16433    1.67  7.8e-16  
      2669  17.157           flat     raced           9165      23032    2.41  7.0e-16  
      2670  2.3.5.89         flat     raced           8828      18362    2.04  6.0e-16  
      2672  2^4.167          prime    raced          15772      19565    1.21  6.5e-16  bluestein
      2674  2.7.191          flat     raced          10927      16029    1.44  7.1e-16  
      2675  5^2.107          prime    raced          16614      19057    1.14  6.5e-16  bluestein
      2676  2^2.3.223        flat     raced          11268      19367    1.70  7.9e-16  
      2678  2.13.103         flat     raced           9865      20290    1.93  7.2e-16  
      2680  2^3.5.67         flat     raced           7190      20606    2.69  6.4e-16  
      2681  7.383            prime    raced          15799      17049    1.07  6.1e-16  bluestein
      2682  2.3^2.149        flat     raced          11747      19738    1.64  6.8e-16  
      2684  2^2.11.61        flat     raced           7824      24270    3.06  4.8e-16  
      2685  3.5.179          prime    raced          15807      18390    1.13  6.0e-16  bluestein
      2686  2.17.79          flat     raced           9277      23973    2.57  9.6e-16  
      2690  2.5.269          prime    raced          16294      20494    1.25  7.8e-16  bluestein
      2692  2^2.673          flat     raced          11616      22370    1.89  8.7e-16  
      2694  2.3.449          flat     raced          14219      20471    1.43  7.5e-16  
      2696  2^3.337          flat     raced           9329      23072    2.44  8.6e-16  
      2698  2.19.71          flat     raced           8818      25464    2.79  6.9e-16  
      2701  37.73            flat     raced           8524      32718    3.82  7.8e-16  
      2702  2.7.193          flat     raced          10989      17462    1.58  7.0e-16  
      2703  3.17.53          flat     raced           8438      27842    3.29  9.5e-16  
      2705  5.541            flat     raced          16027      26093    1.58  9.2e-16  
      2708  2^2.677          flat     raced          10767      23403    1.81  9.6e-16  
      2710  2.5.271          flat     raced          13266      20075    1.39  9.2e-16  
      2712  2^3.3.113        flat     raced           7820      23099    2.94  5.8e-16  
      2714  2.23.59          flat     raced          11617      26100    2.22  7.7e-16  
      2715  3.5.181          flat     raced           9630      22928    2.32  7.3e-16  
      2716  2^2.7.97         flat     raced           7634      19833    2.50  6.3e-16  
      2718  2.3^2.151        flat     raced          11576      16942    1.43  8.0e-16  
      2721  3.907            prime    raced          16464      26746    1.55  8.0e-16  bluestein
      2722  2.1361           prime    raced          16190      22756    1.39  7.0e-16  bluestein
      2723  7.389            prime    raced          16493      19445    1.17  6.3e-16  bluestein
      2724  2^2.3.227        flat     raced          15288      18578    1.21  7.8e-16  
      2725  5^2.109          flat     raced          12259      24082    1.86  7.0e-16  
      2727  3^3.101          flat     raced          10599      20240    1.88  8.3e-16  
      2732  2^2.683          flat     raced          13314      20860    1.56  8.3e-16  
      2733  3.911            prime    raced          16190      26796    1.64  6.7e-16  bluestein
      2734  2.1367           prime    raced          16404      22865    1.22  7.9e-16  bluestein
      2735  5.547            flat     raced          15052      25775    1.64  1.1e-15  
      2739  3.11.83          flat     raced          12993      20495    1.55  6.4e-16  
      2740  2^2.5.137        flat     raced           8058      18869    2.27  8.0e-16  
      2742  2.3.457          flat     raced          15691      19743    1.25  8.3e-16  
      2743  13.211           flat     raced          10501      20034    1.88  8.6e-16  
      2745  3^2.5.61         flat     raced           8111      24295    2.96  5.3e-16  
      2746  2.1373           prime    raced          16408      22410    1.35  5.8e-16  bluestein
      2747  41.67            flat     raced           9208      39490    4.27  7.7e-16  
      2748  2^2.3.229        flat     raced          10029      18703    1.85  6.9e-16  
      2751  3.7.131          flat     raced           8806      20160    2.24  7.4e-16  
      2756  2^2.13.53        flat     raced           7831      21652    2.71  7.6e-16  
      2757  3.919            prime    raced          16394      26816    1.63  5.7e-16  bluestein
      2758  2.7.197          flat     raced          10548      17391    1.64  6.6e-16  
      2759  31.89            flat     raced           8636      29875    3.44  9.2e-16  
      2761  11.251           flat     raced          11413      17110    1.49  8.6e-16  
      2762  2.1381           prime    raced          16182      22840    1.40  6.6e-16  bluestein
      2763  3^2.307          flat     raced          12601      24461    1.93  9.9e-16  
      2764  2^2.691          flat     raced          14246      20852    1.39  8.7e-16  
      2765  5.7.79           flat     raced           8492      18058    2.11  8.5e-16  
      2766  2.3.461          prime    raced          16425      19896    1.21  6.5e-16  bluestein
      2768  2^4.173          flat     raced          10866      19651    1.70  6.4e-16  
      2769  3.13.71          flat     raced           8229      19974    2.34  8.1e-16  
      2770  2.5.277          flat     raced          13318      19966    1.50  8.4e-16  
      2771  17.163           flat     raced          11395      27050    2.35  9.5e-16  
      2773  47.59            flat     raced          12434      38852    3.01  7.2e-16  
      2774  2.19.73          flat     raced           9293      26821    2.74  6.7e-16  
      2776  2^3.347          prime    raced          16456      19747    1.03  6.8e-16  bluestein
      2778  2.3.463          flat     raced          15760      20121    1.24  8.0e-16  
      2779  7.397            flat     raced          11020      19537    1.76  7.2e-16  
      2780  2^2.5.139        flat     raced           9398      24032    2.53  8.0e-16  
      2781  3^3.103          flat     raced           9867      20683    2.04  1.0e-15  
      2782  2.13.107         prime    raced          16242      20529    1.12  8.4e-16  bluestein
      2785  5.557            prime    raced          16251      26077    1.56  6.1e-16  bluestein
      2786  2.7.199          flat     raced          11277      17402    1.48  9.4e-16  
      2787  3.929            prime    raced          16218      26817    1.62  6.6e-16  bluestein
      2792  2^3.349          flat     raced          11231      19893    1.70  7.9e-16  
      2794  2.11.127         flat     raced          11149      17212    1.50  9.7e-16  
      2796  2^2.3.233        flat     raced          10918      18675    1.70  8.9e-16  
      2798  2.1399           prime    raced          16273      22999    1.40  7.1e-16  bluestein
      2799  3^2.311          flat     raced          12991      24384    1.87  9.9e-16  
      2802  2.3.467          prime    raced          18398      19956    1.08  5.0e-16  bluestein
      2804  2^2.701          flat     raced          12124      21239    1.61  8.2e-16  
      2806  2.23.61          flat     raced           9970      32892    3.28  6.3e-16  
      2807  7.401            flat     raced          13005      21223    1.62  8.2e-16  
      2809  53^2             prime    raced          18330      40559    2.21  8.4e-16  bluestein
      2810  2.5.281          flat     raced          11339      19911    1.74  7.6e-16  
      2811  3.937            prime    raced          18369      26945    1.46  7.4e-16  bluestein
      2813  29.97            flat     raced           9310      30702    3.29  7.1e-16  
      2814  2.3.7.67         flat     raced           9744      21960    2.24  6.9e-16  
      2815  5.563            prime    raced          18341      25963    1.40  7.1e-16  bluestein
      2817  3^2.313          flat     raced          11488      24582    2.02  7.4e-16  
      2818  2.1409           prime    raced          18416      22571    1.20  8.4e-16  bluestein
      2822  2.17.83          flat     raced          13461      27128    1.98  7.8e-16  
      2823  3.941            prime    raced          19437      21466    1.10  6.9e-16  bluestein
      2824  2^3.353          flat     raced           8661      19813    2.27  6.5e-16  
      2825  5^2.113          flat     raced           8148      25488    3.11  5.8e-16  
      2826  2.3^2.157        flat     raced          10196      18235    1.75  8.3e-16  
      2827  11.257           flat     raced          10768      22952    2.12  6.7e-16  
      2828  2^2.7.101        flat     raced          10333      20495    1.85  8.8e-16  
      2830  2.5.283          flat     raced          16780      20196    1.19  7.2e-16  
      2831  19.149           flat     raced          12153      27527    2.24  8.4e-16  
      2832  2^4.3.59         flat     raced           9904      18801    1.88  7.8e-16  
      2834  2.13.109         flat     raced          11438      25877    2.17  7.9e-16  
      2836  2^2.709          prime    raced          18241      21668    1.13  6.4e-16  bluestein
      2839  17.167           prime    raced          18348      26942    1.47  7.3e-16  bluestein
      2840  2^3.5.71         flat     raced           7305      19544    2.62  5.7e-16  
      2841  3.947            prime    raced          18636      21401    1.13  6.9e-16  bluestein
      2844  2^2.3^2.79       flat     raced           7985      19295    2.36  9.3e-16  
      2845  5.569            prime    raced          18458      20946    1.11  6.4e-16  bluestein
      2846  2.1423           prime    raced          18350      23048    1.25  6.5e-16  bluestein
      2847  3.13.73          flat     raced           8559      22366    2.60  7.9e-16  
      2848  2^5.89           flat     raced           8238      25007    3.02  6.8e-16  
      2853  3^2.317          prime    raced          17929      17401    0.94  7.3e-16  bluestein
      2854  2.1427           prime    raced          18364      22548    1.20  7.1e-16  bluestein
      2855  5.571            flat     raced          17640      21081    1.14  7.5e-16  
      2858  2.1429           prime    raced          18366      22900    1.23  8.2e-16  bluestein
      2859  3.953            prime    raced          18396      21457    1.13  7.6e-16  bluestein
      2862  2.3^3.53         flat     raced           9741      22333    2.14  7.7e-16  
      2863  7.409            flat     raced          13454      21783    1.58  9.5e-16  
      2864  2^4.179          flat     raced          15133      19657    1.29  1.0e-15  
      2865  3.5.191          flat     raced          10175      17524    1.68  7.0e-16  
      2866  2.1433           prime    raced          18584      22985    1.21  8.7e-16  bluestein
      2867  47.61            flat     raced           9893      41931    4.07  6.3e-16  
      2868  2^2.3.239        flat     raced           9313      18935    2.00  1.0e-15  
      2869  19.151           flat     raced          11487      25384    2.15  8.6e-16  
      2872  2^3.359          prime    raced          18377      19801    1.06  6.4e-16  bluestein
      2874  2.3.479          prime    raced          18403      20213    1.10  8.4e-16  bluestein
      2876  2^2.719          prime    raced          18384      21204    1.10  7.4e-16  bluestein
      2877  3.7.137          flat     raced           9763      25761    2.55  7.8e-16  
      2878  2.1439           prime    raced          18370      22859    1.24  8.3e-16  bluestein
      2881  43.67            flat     raced          10064      42117    4.18  7.4e-16  
      2882  2.11.131         flat     raced          10028      21882    2.14  8.4e-16  
      2884  2^2.7.103        flat     raced           8544      20653    2.40  9.8e-16  
      2885  5.577            flat     raced          14310      22638    1.57  7.1e-16  
      2889  3^3.107          prime    raced          18721      20611    1.09  7.4e-16  bluestein
      2891  7^2.59           flat     raced          11401      19316    1.68  9.4e-16  
      2892  2^2.3.241        flat     raced           9078      24378    2.67  8.1e-16  
      2893  11.263           prime    raced          18875      23023    1.22  7.2e-16  bluestein
      2894  2.1447           prime    raced          18723      27041    1.42  8.0e-16  bluestein
      2895  3.5.193          flat     raced          10236      18500    1.78  6.7e-16  
      2896  2^4.181          flat     raced           9118      24513    2.67  7.0e-16  
      2899  13.223           flat     raced          13575      21764    1.59  9.8e-16  
      2901  3.967            prime    raced          18844      25241    1.33  7.3e-16  bluestein
      2902  2.1451           prime    raced          18843      27162    1.39  6.7e-16  bluestein
      2905  5.7.83           flat     raced          13247      21681    1.59  5.8e-16  
      2906  2.1453           prime    raced          18867      27078    1.42  8.4e-16  bluestein
      2908  2^2.727          flat     raced          11324      26900    2.30  8.2e-16  
      2910  2.3.5.97         flat     raced           9982      22334    2.15  8.6e-16  
      2911  41.71            flat     raced           9706      35115    3.46  6.8e-16  
      2913  3.971            prime    raced          18872      25472    1.33  7.7e-16  bluestein
      2915  5.11.53          flat     raced           9005      23468    2.46  7.7e-16  
      2918  2.1459           prime    raced          18729      25076    1.33  7.2e-16  bluestein
      2919  3.7.139          flat     raced          11111      20040    1.79  8.0e-16  
      2920  2^3.5.73         flat     raced           7378      20904    2.79  7.4e-16  
      2921  23.127           flat     raced          10638      26356    2.45  9.2e-16  
      2922  2.3.487          flat     raced          18445      21444    1.16  8.8e-16  
      2923  37.79            flat     raced           9699      32610    3.21  9.8e-16  
      2928  2^4.3.61         flat     raced           7817      25185    3.21  5.7e-16  
      2929  29.101           flat     raced          12685      31712    2.44  9.5e-16  
      2930  2.5.293          flat     raced          18821      20125    1.06  1.2e-15  
      2931  3.977            prime    raced          18695      22911    1.21  7.7e-16  bluestein
      2932  2^2.733          prime    raced          18848      24677    1.30  7.8e-16  bluestein
      2933  7.419            flat     raced          13443      21865    1.58  8.2e-16  
      2934  2.3^2.163        flat     raced          12826      21887    1.69  7.2e-16  
      2935  5.587            prime    raced          18699      22584    1.20  7.6e-16  bluestein
      2936  2^3.367          flat     raced          16904      23646    1.39  9.9e-16  
      2937  3.11.89          flat     raced           9152      21136    2.27  5.5e-16  
      2938  2.13.113         flat     raced           9459      26258    2.72  8.0e-16  
      2941  17.173           flat     raced          13468      27312    1.99  9.5e-16  
      2942  2.1471           prime    raced          17567      25048    1.42  7.9e-16  bluestein
      2943  3^3.109          flat     raced          12189      26117    1.90  8.6e-16  
      2946  2.3.491          flat     raced          14860      21524    1.45  7.0e-16  
      2947  7.421            flat     raced          13556      21826    1.57  7.0e-16  
      2948  2^2.11.67        flat     raced           8503      23921    2.74  7.0e-16  
      2949  3.983            prime    raced          17193      22995    1.33  6.8e-16  bluestein
      2950  2.5^2.59         flat     raced          12510      25978    2.06  7.1e-16  
      2951  13.227           prime    raced          17555      21165    1.20  7.1e-16  bluestein
      2954  2.7.211          flat     raced          12620      20894    1.49  7.5e-16  
      2955  3.5.197          flat     raced           9399      20067    2.13  7.7e-16  
      2956  2^2.739          flat     raced          17243      25088    1.44  8.4e-16  
      2959  11.269           prime    raced          17028      23201    1.33  8.7e-16  bluestein
      2962  2.1481           prime    raced          17556      25007    1.42  6.0e-16  bluestein
      2965  5.593            flat     raced          16429      22812    1.35  6.9e-16  
      2966  2.1483           prime    raced          17509      25240    1.43  7.0e-16  bluestein
      2968  2^3.7.53         flat     raced           7902      24037    2.95  6.2e-16  
      2972  2^2.743          prime    raced          17596      24892    1.41  7.9e-16  bluestein
      2973  3.991            prime    raced          16944      22908    1.35  8.1e-16  bluestein
      2974  2.1487           prime    raced          17514      25097    1.36  6.5e-16  bluestein
      2977  13.229           flat     raced          12020      21315    1.77  1.0e-15  
      2978  2.1489           prime    raced          16899      25132    1.41  8.5e-16  bluestein
      2979  3^2.331          flat     raced          12067      25434    2.09  8.1e-16  
      2980  2^2.5.149        flat     raced          11142      21918    1.93  7.0e-16  
      2981  11.271           flat     raced          12590      22850    1.78  7.0e-16  
      2982  2.3.7.71         flat     raced          10005      21508    2.11  6.1e-16  
      2983  19.157           flat     raced          11066      26499    2.38  8.9e-16  
      2984  2^3.373          flat     raced          12351      23413    1.88  8.8e-16  
      2985  3.5.199          flat     raced          10951      18730    1.69  8.1e-16  
      2986  2.1493           prime    raced          16862      25264    1.49  7.7e-16  bluestein
      2987  29.103           flat     raced          10681      31961    2.99  9.0e-16  
      2988  2^2.3^2.83       flat     raced          12780      21511    1.66  7.3e-16  
      2989  7^2.61           flat     raced           8837      26550    2.71  5.9e-16  
      2991  3.997            prime    raced          16678      23155    1.38  6.6e-16  bluestein
      2993  41.73            flat     raced           9646      37655    3.85  7.4e-16  
      2994  2.3.499          prime    raced          17075      21593    1.26  5.9e-16  bluestein
      2995  5.599            prime    raced          16754      22636    1.33  6.6e-16  bluestein
      2996  2^2.7.107        flat     raced          15446      21294    1.35  1.0e-15  
      2998  2.1499           prime    raced          16900      25178    1.49  7.4e-16  bluestein
      3002  2.19.79          flat     raced          10898      27049    2.42  9.1e-16  
      3004  2^2.751          flat     raced          15826      21543    1.32  7.7e-16  
      3005  5.601            flat     raced          17809      28192    1.54  7.4e-16  
      3006  2.3^2.167        prime    raced          16918      29551    1.74  6.6e-16  bluestein
      3007  31.97            flat     raced           9823      34040    3.41  6.2e-16  
      3009  3.17.59          flat     raced          12609      26907    2.10  8.7e-16  
      3012  2^2.3.251        flat     raced          10935      17756    1.60  8.2e-16  
      3013  23.131           flat     raced           9710      30563    3.12  1.0e-15  
      3014  2.11.137         flat     raced          11449      22195    1.90  7.6e-16  
      3015  3^2.5.67         flat     raced           8591      23934    2.72  6.6e-16  
      3017  7.431            prime    raced          16870      22084    1.31  7.2e-16  bluestein
      3018  2.3.503          prime    raced          16717      19418    1.12  5.8e-16  bluestein
      3020  2^2.5.151        flat     raced          10321      18806    1.80  8.7e-16  
      3021  3.19.53          flat     raced           9649      31545    3.15  6.5e-16  
      3022  2.1511           prime    raced          16939      23547    1.38  6.3e-16  bluestein
      3026  2.17.89          flat     raced           9263      27760    2.97  7.4e-16  
      3027  3.1009           prime    raced          16895      30194    1.78  6.4e-16  bluestein
      3028  2^2.757          flat     raced          13796      21747    1.53  7.8e-16  
      3029  13.233           flat     raced          13449      21367    1.45  8.8e-16  
      3030  2.3.5.101        flat     raced          13873      22015    1.57  8.1e-16  
      3031  7.433            flat     raced          14852      23937    1.46  8.6e-16  
      3032  2^3.379          flat     raced          11956      19959    1.63  8.3e-16  
      3033  3^2.337          flat     raced          12100      25518    2.03  8.2e-16  
      3035  5.607            prime    raced          16903      28487    1.68  8.9e-16  bluestein
      3039  3.1013           prime    raced          16897      30194    1.78  7.4e-16  bluestein
      3043  17.179           prime    raced          16897      27887    1.65  7.8e-16  bluestein
      3044  2^2.761          flat     raced          13004      21761    1.63  6.3e-16  
      3046  2.1523           prime    raced          16958      23663    1.39  6.5e-16  bluestein
      3047  11.277           flat     raced          13163      23026    1.71  7.8e-16  
      3048  2^3.3.127        flat     raced          10586      17464    1.63  8.0e-16  
      3050  2.5^2.61         flat     raced           9894      26738    2.35  6.4e-16  
      3051  3^3.113          flat     raced           9578      26351    2.75  7.1e-16  
      3052  2^2.7.109        flat     raced          10944      26789    2.16  6.6e-16  
      3053  43.71            flat     raced          10064      37462    3.62  6.4e-16  
      3054  2.3.509          prime    raced          16896      18332    1.08  6.0e-16  bluestein
      3056  2^4.191          flat     raced           9644      18400    1.74  7.5e-16  
      3057  3.1019           prime    raced          16947      20029    1.16  6.6e-16  bluestein
      3058  2.11.139         flat     raced          12486      22208    1.76  7.6e-16  
      3062  2.1531           prime    raced          16911      23629    1.39  7.2e-16  bluestein
      3063  3.1021           prime    raced          16982      19916    1.16  6.9e-16  bluestein
      3064  2^3.383          prime    raced          16915      20297    1.18  5.9e-16  bluestein
      3065  5.613            flat     raced          15865      26185    1.65  9.4e-16  
      3066  2.3.7.73         flat     raced           9929      21651    2.11  6.9e-16  
      3068  2^2.13.59        flat     raced          11953      20837    1.72  9.6e-16  
      3070  2.5.307          flat     raced          15435      27140    1.74  9.6e-16  
      3071  37.83            flat     raced          15266      36464    2.38  6.6e-16  
      3073  7.439            prime    raced          19192      23921    1.24  6.2e-16  bluestein
      3074  2.29.53          flat     raced          10336      35400    3.14  7.4e-16  
      3076  2^2.769          flat     raced          12023      22072    1.82  6.8e-16  
      3077  17.181           flat     raced          11015      32961    2.94  7.7e-16  
      3081  3.13.79          flat     raced           9901      21490    2.15  1.2e-15  
      3082  2.23.67          flat     raced          10174      33508    3.28  7.5e-16  
      3084  2^2.3.257        flat     raced           9981      24218    2.42  6.4e-16  
      3085  5.617            flat     raced          14935      26014    1.74  7.5e-16  
      3086  2.1543           prime    raced          18944      32499    1.70  6.1e-16  bluestein
      3088  2^4.193          flat     raced           9544      20004    2.08  6.6e-16  
      3090  2.3.5.103        flat     raced          10885      22122    1.90  1.1e-15  
      3091  11.281           flat     raced          10876      23124    2.12  6.7e-16  
      3092  2^2.773          prime    raced          21516      23290    1.08  5.5e-16  bluestein
      3093  3.1031           prime    raced          19134      26401    1.22  7.6e-16  bluestein
      3095  5.619            prime    raced          18774      26156    1.37  6.8e-16  bluestein
      3097  19.163           flat     raced          13251      30853    2.29  7.2e-16  
      3098  2.1549           prime    raced          18773      32525    1.69  5.8e-16  bluestein
      3099  3.1033           prime    raced          19678      26311    1.32  7.2e-16  bluestein
      3101  7.443            flat     raced          14484      24119    1.66  1.1e-15  
      3103  29.107           prime    raced          18685      32719    1.71  6.6e-16  bluestein
      3104  2^5.97           flat     raced           8802      20900    2.34  4.4e-16  
      3106  2.1553           prime    raced          21276      32606    1.51  6.5e-16  bluestein
      3107  13.239           flat     raced          11405      21445    1.86  1.0e-15  
      3110  2.5.311          flat     raced          16151      28458    1.75  9.4e-16  
      3111  3.17.61          flat     raced           9998      35031    3.41  6.8e-16  
      3112  2^3.389          flat     raced          17855      22379    1.24  9.8e-16  
      3113  11.283           flat     raced          16680      23177    1.37  9.6e-16  
      3114  2.3^2.173        flat     raced          14689      21790    1.47  8.8e-16  
      3115  5.7.89           flat     raced           8845      21451    2.33  9.6e-16  
      3117  3.1039           prime    raced          18726      26387    1.39  6.2e-16  bluestein
      3118  2.1559           prime    raced          19172      32621    1.66  6.0e-16  bluestein
      3122  2.7.223          flat     raced          15572      22784    1.44  8.4e-16  
      3123  3^2.347          prime    raced          18798      22357    1.17  6.6e-16  bluestein
      3124  2^2.11.71        flat     raced           8823      22414    2.50  5.6e-16  
      3126  2.3.521          flat     raced          16684      24775    1.46  9.4e-16  
      3127  53.59            prime    raced          17993      48149    2.67  6.8e-16  bluestein
      3129  3.7.149          flat     raced          13246      22034    1.65  8.3e-16  
      3130  2.5.313          flat     raced          13436      26747    1.99  8.7e-16  
      3131  31.101           flat     raced          12269      35365    2.84  7.3e-16  
      3133  13.241           flat     raced          10773      34248    3.08  8.8e-16  
      3134  2.1567           prime    raced          18026      24156    1.33  5.8e-16  bluestein
      3138  2.3.523          prime    raced          17983      24819    1.37  7.9e-16  bluestein
      3139  43.73            flat     raced          10157      40196    3.91  7.4e-16  
      3140  2^2.5.157        flat     raced           9403      20383    2.16  8.9e-16  
      3141  3^2.349          flat     raced          15611      22349    1.42  8.0e-16  
      3142  2.1571           prime    raced          17986      23942    1.32  6.4e-16  bluestein
      3143  7.449            flat     raced          15618      24309    1.55  6.5e-16  
      3144  2^3.3.131        flat     raced           9673      22841    2.35  7.8e-16  
      3147  3.1049           prime    raced          18939      26396    1.39  7.0e-16  bluestein
      3148  2^2.787          prime    raced          18053      22923    1.26  7.3e-16  bluestein
      3149  47.67            flat     raced          10853      47794    4.40  7.1e-16  
      3151  23.137           flat     raced          10659      37026    3.44  8.3e-16  
      3152  2^4.197          flat     raced           8594      20069    2.18  5.5e-16  
      3153  3.1051           prime    raced          18027      26464    1.46  7.2e-16  bluestein
      3154  2.19.83          flat     raced          15439      30887    1.92  6.9e-16  
      3155  5.631            flat     raced          20095      21297    1.06  9.7e-16  
      3156  2^2.3.263        prime    raced          18997      24306    1.26  6.8e-16  bluestein
      3158  2.1579           prime    raced          18041      24091    1.33  6.9e-16  bluestein
      3160  2^3.5.79         flat     raced           8905      21511    2.33  8.2e-16  
      3161  29.109           flat     raced          12990      39369    2.71  9.4e-16  
      3164  2^2.7.113        flat     raced           8990      28618    3.16  6.4e-16  
      3165  3.5.211          flat     raced          11876      22410    1.88  7.7e-16  
      3166  2.1583           prime    raced          18015      23709    1.30  5.8e-16  bluestein
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 flat      1020      1     10   1.34   1.98   3.08    1.99
 prime      542      2     25   1.09   1.29   1.62    1.31
 ALL       1562      3     35   1.16   1.63   2.82    1.72
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                3   1.68      0      0
 128..511             105   1.60      0      0
 512..2047            770   1.59     25      1
 2048..3166           684   1.67     10      2
```


## by family
```
 family                                       cells median   <1.0  gmean
 flat                                          1020   1.98     10   1.99
 composite with a prime >= 53 (prime cell)      542   1.29     25   1.31
```


flip agreement: our two readings more than 25% apart at 15 of 1562 cells.

worst 10: 2429 (prime 0.66), 1893 (flat 0.73), 2428 (prime 0.73), 753 (flat 0.81), 597 (flat 0.81), 2421 (flat 0.81), 579 (flat 0.83), 586 (prime 0.83), 694 (prime 0.85), 2319 (prime 0.87)
best 5: 3149 (flat 4.40), 2479 (flat 4.30), 2747 (flat 4.27), 1943 (flat 4.21), 2077 (flat 4.19)
