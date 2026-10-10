# gauntlet report

run: `kfr_c2c_64_4096`  contract file suffix: `_kfr`  cells: 4033 listed, 4033 benched, comparator: KFR

control cell: 84 readings, 0.802..1.470 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        64  2^6              2p       replayed          30         39    1.28  1.5e-16  
        65  5.13             2p       replayed          53        103    1.91  6.3e-16  
        66  2.3.11           2p       replayed          46        118    2.58  3.0e-16  
        67  67               prime    replayed         173        421    2.42  4.9e-16  rader
        68  2^2.17           2p       replayed          56        148    2.46  3.7e-16  
        69  3.23             2p       replayed          93        181    1.95  4.1e-16  
        70  2.5.7            2p       replayed          45         73    1.61  2.3e-16  
        71  71               prime    replayed         173        477    2.56  6.4e-16  rader
        72  2^3.3^2          2p       replayed          44        224    5.06  3.0e-16  
        73  73               prime    replayed         173        492    2.06  8.7e-16  flips differ 1.38x rader
        74  2.37             flat     replayed         174        269    1.52  4.8e-16  
        75  3.5^2            2p       replayed          57         79    1.22  4.4e-16  
        76  2^2.19           2p       replayed          68        189    2.72  3.6e-16  
        77  7.11             2p       replayed          60        115    1.89  3.6e-16  
        78  2.3.13           2p       replayed          57        145    2.50  5.7e-16  
        79  79               prime    replayed         289        587    1.96  1.3e-15  rader
        80  2^4.5            2p       replayed          45        233    5.19  3.5e-16  
        81  3^4              2p       replayed          60        192    3.19  4.3e-16  
        82  2.41             flat     raced            155        320    1.98  6.2e-16  
        83  83               prime    replayed         453        641    1.41  5.4e-16  bluestein
        84  2^2.3.7          2p       replayed          54        233    4.30  3.1e-16  
        85  5.17             2p       replayed          82        160    1.88  4.7e-16  
        86  2.43             flat     raced            165        363    2.13  1.3e-15  
        87  3.29             flat     replayed         149        258    1.16  6.6e-16  flips differ 1.50x
        88  2^3.11           2p       replayed          60        309    5.17  2.8e-16  
        89  89               prime    replayed         216        735    3.39  3.9e-16  rader
        90  2.3^2.5          2p       replayed          60        229    3.80  4.0e-16  
        91  7.13             2p       replayed          75        143    1.92  5.0e-16  
        92  2^2.23           2p       replayed          88        269    3.06  5.7e-16  
        93  3.31             flat     replayed         185        310    1.64  4.4e-16  
        94  2.47             flat     raced            189        437    2.21  8.6e-16  
        95  5.19             2p       replayed          98        211    2.14  4.4e-16  
        96  2^5.3            2p       replayed          51        308    6.03  3.0e-16  
        97  97               prime    replayed         219        869    3.96  6.5e-16  rader
        98  2.7^2            flat     replayed         116        103    0.88  5.0e-16  
        99  3^2.11           2p       replayed          77        172    2.21  4.7e-16  
       100  2^2.5^2          2p       replayed          64        262    4.11  3.2e-16  
       101  101              prime    replayed         234        438    1.87  8.0e-16  rader
       102  2.3.17           2p       replayed          88        224    2.23  5.8e-16  
       103  103              prime    replayed         292        439    1.45  9.6e-16  rader
       104  2^3.13           2p       replayed          74        370    4.97  4.8e-16  
       105  3.5.7            2p       replayed          85        107    1.25  5.0e-16  
       106  2.53             prime    replayed         447        543    1.20  3.1e-16  bluestein
       107  107              prime    replayed         448        440    0.95  1.0e-15  bluestein
       108  2^2.3^3          2p       replayed          77        335    4.29  4.4e-16  
       109  109              prime    replayed         268        441    1.64  8.0e-16  rader
       110  2.5.11           2p       replayed          78        195    2.50  3.7e-16  
       111  3.37             2p       raced            182        416    2.29  4.8e-16  
       112  2^4.7            2p       replayed          71        320    4.52  2.9e-16  
       113  113              prime    replayed         273        440    1.60  5.3e-16  rader
       114  2.3.19           2p       replayed         105        285    2.65  4.8e-16  
       115  5.23             2p       replayed         147        308    1.48  4.6e-16  flips differ 1.41x
       116  2^2.29           flat     replayed         161        383    2.38  5.5e-16  
       117  3^2.13           2p       replayed          98        213    2.17  5.2e-16  
       118  2.59             prime    replayed         456        680    1.48  2.8e-16  bluestein
       119  7.17             2p       replayed         112        231    2.04  3.7e-16  
       120  2^3.3.5          2p       replayed          77        347    4.48  3.7e-16  
       121  11^2             2p       replayed         101        292    2.89  3.7e-16  
       122  2.61             prime    replayed         457        715    1.54  4.2e-16  bluestein
       123  3.41             2p       raced            218        492    2.25  4.7e-16  
       124  2^2.31           flat     replayed         218        450    2.04  5.3e-16  
       125  5^3              2p       replayed         103        128    1.23  4.8e-16  
       126  2.3^2.7          2p       replayed          90        311    3.44  3.1e-16  
       127  127              prime    replayed         404        447    1.11  5.4e-16  rader
       128  2^7              2p       replayed          65         73    1.12  3.2e-16  
       129  3.43             flat     replayed         246        557    2.26  4.1e-16  
       130  2.5.13           2p       replayed          99        240    2.44  4.6e-16  
       131  131              prime    replayed         356        910    2.53  5.7e-16  rader
       132  2^2.3.11         2p       replayed         100        444    4.40  4.1e-16  
       133  7.19             2p       replayed         139        307    2.08  4.5e-16  
       134  2.67             prime    replayed         936        875    0.94  3.6e-16  bluestein
       135  3^3.5            2p       replayed         109        317    2.86  3.2e-16  
       136  2^3.17           2p       replayed         114        538    4.65  4.3e-16  
       137  137              prime    replayed         393        913    2.22  7.4e-16  rader
       138  2.3.23           2p       replayed         137        407    2.96  4.8e-16  
       139  139              prime    replayed         521        914    1.53  7.5e-16  rader
       140  2^2.5.7          chain3   replayed         111        367    3.25  2.7e-16  
       141  3.47             flat     replayed         285        674    2.36  5.4e-16  
       142  2.71             prime    replayed         961        991    1.00  4.3e-16  bluestein
       143  11.13            2p       replayed         123        357    2.89  4.8e-16  
       144  2^4.3^2          chain3   replayed         106        448    4.21  3.6e-16  
       145  5.29             2p       replayed         190        462    2.32  4.4e-16  
       146  2.73             prime    replayed         972       1012    1.02  4.8e-16  bluestein
       147  3.7^2            chain3   replayed         135        160    1.14  4.0e-16  
       148  2^2.37           2p       replayed         183        592    3.19  3.2e-16  
       149  149              prime    replayed         608        919    1.24  5.8e-16  rader
       150  2.3.5^2          2p       replayed         111        370    3.31  3.5e-16  
       151  151              prime    replayed         405        920    2.26  6.5e-16  rader
       152  2^3.19           2p       replayed         132        656    4.75  3.6e-16  
       153  3^2.17           2p       replayed         144        332    2.22  5.6e-16  
       154  2.7.11           flat     replayed         190        277    1.45  3.5e-16  
       155  5.31             2p       replayed         258        520    2.01  4.3e-16  
       156  2^2.3.13         2p       replayed         124        548    4.40  5.6e-16  
       157  157              prime    replayed         430        923    2.14  9.9e-16  rader
       158  2.79             prime    replayed         956       1216    1.27  3.3e-16  bluestein
       159  3.53             prime    replayed         955        822    0.86  3.0e-16  bluestein
       160  2^5.5            2p       replayed          91        479    5.29  3.0e-16  
       161  7.23             2p       replayed         180        441    2.40  4.5e-16  
       162  2.3^4            2p       replayed         123        435    3.45  4.6e-16  
       163  163              prime    replayed         461        969    2.00  6.8e-16  rader
       164  2^2.41           2p       replayed         298        694    1.25  4.0e-16  flips differ 1.87x
       165  3.5.11           2p       replayed         144        288    1.91  4.0e-16  
       166  2.83             prime    replayed         948       1358    1.40  4.6e-16  bluestein
       167  167              prime    replayed         973        926    0.89  4.1e-16  bluestein
       168  2^3.3.7          chain3   replayed         126        476    3.79  2.7e-16  
       169  13^2             2p       replayed         157        439    2.78  7.3e-16  
       170  2.5.17           2p       replayed         144        382    2.59  7.8e-16  
       171  3^2.19           2p       replayed         186        428    2.20  6.3e-16  
       172  2^2.43           2p       replayed         238        790    3.30  5.2e-16  
       173  173              prime    replayed         962        929    0.96  3.0e-16  bluestein
       174  2.3.29           2p       replayed         196        594    3.02  4.9e-16  
       175  5^2.7            chain3   replayed         152        176    1.16  3.7e-16  
       176  2^4.11           2p       replayed         123        625    4.75  2.6e-16  
       177  3.59             prime    replayed         999       1022    1.02  5.1e-16  bluestein
       178  2.89             prime    replayed         965       1494    1.26  3.8e-16  bluestein
       179  179              prime    replayed         966        927    0.96  6.3e-16  bluestein
       180  2^2.3^2.5        2p       replayed         140        510    3.57  4.0e-16  
       181  181              prime    replayed         524        933    1.42  7.5e-16  flips differ 1.26x rader
       182  2.7.13           flat     replayed         236        343    1.44  4.7e-16  
       183  3.61             prime    replayed         964       1084    1.12  5.1e-16  bluestein
       184  2^3.23           2p       replayed         237        857    3.56  3.8e-16  
       185  5.37             2p       replayed         578        691    1.08  3.2e-16  
       186  2.3.31           2p       replayed         215        670    3.10  4.3e-16  
       187  11.17            flat     replayed         244        533    2.18  6.5e-16  
       188  2^2.47           2p       replayed         410        961    1.16  4.5e-16  flips differ 1.97x
       189  3^3.7            chain3   replayed         168        458    2.52  4.4e-16  
       190  2.5.19           2p       replayed         178        481    2.60  5.0e-16  
       191  191              prime    replayed         608        930    1.53  8.0e-16  rader
       192  2^6.3            2p       replayed         110        610    5.46  3.4e-16  
       193  193              prime    replayed         476        980    1.96  6.2e-16  rader
       194  2.97             prime    replayed         977       1773    1.76  4.7e-16  bluestein
       195  3.5.13           chain3   replayed         176        356    2.02  7.1e-16  
       196  2^2.7^2          chain3   replayed         166        519    3.10  3.1e-16  
       197  197              prime    replayed         576        936    1.62  5.3e-16  rader
       198  2.3^2.11         chain3   replayed         171        640    3.58  4.9e-16  
       199  199              prime    replayed         622        933    1.49  6.8e-16  rader
       200  2^3.5^2          chain3   replayed         139        560    4.00  2.5e-16  
       201  3.67             prime    replayed        1016       1320    1.28  4.8e-16  bluestein
       202  2.101            prime    replayed         966        933    0.96  3.6e-16  bluestein
       203  7.29             2p       replayed         261        629    2.38  4.5e-16  
       204  2^2.3.17         chain3   replayed         194        767    3.73  6.1e-16  
       205  5.41             2p       replayed         329        821    2.49  5.7e-16  
       206  2.103            prime    replayed        1014       1021    0.87  6.1e-16  bluestein
       207  3^2.23           2p       replayed         271        604    1.34  5.0e-16  flips differ 1.67x
       208  2^4.13           2p       replayed         149        757    5.01  6.7e-16  
       209  11.19            2p       replayed         217        662    3.04  5.1e-16  
       210  2.3.5.7          chain3   replayed         167        513    3.06  3.5e-16  
       211  211              prime    replayed         690        938    1.33  7.1e-16  rader
       212  2^2.53           prime    replayed        1025       1141    1.11  6.0e-16  bluestein
       213  3.71             prime    replayed         999       1494    1.49  4.0e-16  bluestein
       214  2.107            prime    replayed        1007        941    0.84  4.9e-16  bluestein
       215  5.43             flat     replayed         388        930    2.40  5.2e-16  
       216  2^3.3^3          2p       replayed         160        666    4.15  3.9e-16  
       217  7.31             2p       replayed         464        732    1.50  5.4e-16  
       218  2.109            prime    replayed         985       1021    0.94  3.6e-16  bluestein
       219  3.73             prime    replayed        1025       1531    1.44  5.5e-16  bluestein
       220  2^2.5.11         chain3   replayed         188        729    3.74  2.8e-16  
       221  13.17            2p       replayed         225        658    2.90  5.5e-16  
       222  2.3.37           2p       replayed         283        884    3.11  3.9e-16  
       223  223              prime    replayed        1005        941    0.94  8.4e-16  bluestein
       224  2^5.7            2p       replayed         146        657    4.42  2.6e-16  
       225  3^2.5^2          2p       replayed         207        536    2.40  4.8e-16  
       226  2.113            prime    replayed         993        941    0.93  5.8e-16  bluestein
       227  227              prime    replayed         979        942    0.96  8.4e-16  bluestein
       228  2^2.3.19         flat     replayed         327        931    2.83  1.4e-15  
       229  229              prime    replayed         732        951    1.29  6.3e-16  rader
       230  2.5.23           2p       replayed         233        684    2.90  5.4e-16  
       231  3.7.11           chain3   replayed         214        420    1.92  3.5e-16  
       232  2^3.29           flat     replayed         340       1187    2.96  5.1e-16  
       233  233              prime    replayed         984        943    0.95  5.8e-16  bluestein
       234  2.3^2.13         chain3   replayed         231        768    3.31  6.4e-16  
       235  5.47             2p       replayed        1055       1124    1.02  3.4e-16  
       236  2^2.59           prime    replayed        1029       1411    1.26  6.0e-16  bluestein
       237  3.79             prime    replayed        1033       1833    1.74  7.2e-16  bluestein
       238  2.7.17           flat     replayed         348        528    1.49  1.2e-15  
       239  239              prime    replayed        1048        949    0.88  8.5e-16  bluestein
       240  2^4.3.5          2p       replayed         169        710    4.18  4.7e-16  
       241  241              prime    replayed         655        946    1.29  5.0e-16  rader
       242  2.11^2           flat     replayed         327        649    1.94  8.7e-16  
       243  3^5              2p       replayed         217        600    2.54  6.5e-16  
       244  2^2.61           prime    replayed        1063       1500    1.38  4.4e-16  bluestein
       245  5.7^2            chain3   replayed         228        263    1.14  3.0e-16  
       246  2.3.41           2p       replayed         338       1045    3.06  3.2e-16  
       247  13.19            2p       replayed         270        810    2.77  5.2e-16  
       248  2^3.31           flat     replayed         477       1312    2.67  7.1e-16  
       249  3.83             prime    replayed        1022       2000    1.95  5.6e-16  bluestein
       250  2.5^3            chain3   replayed         189        560    2.94  3.7e-16  
       251  251              prime    replayed         703        948    1.22  7.2e-16  rader
       252  2^2.3^2.7        chain3   replayed         209        707    3.38  4.5e-16  
       253  11.23            2p       replayed         295        908    3.03  7.1e-16  
       254  2.127            prime    replayed        1004        959    0.95  4.3e-16  bluestein
       255  3.5.17           2p       replayed         255        562    2.19  6.1e-16  
       256  2^8              2p       replayed         135        144    1.07  0.0e+00  
       257  257              prime    replayed         659       2060    3.11  5.8e-16  rader
       258  2.3.43           2p       replayed         508       1187    2.32  4.6e-16  
       259  7.37             2p       replayed         376       1012    2.59  4.1e-16  
       260  2^2.5.13         chain3   replayed         226        889    3.93  4.1e-16  
       261  3^2.29           2p       replayed         335        857    2.56  4.2e-16  
       262  2.131            prime    replayed        2055       2076    1.00  5.5e-16  bluestein
       263  263              prime    replayed        2064       2068    1.00  5.4e-16  bluestein
       264  2^3.3.11         chain3   replayed         225        926    4.10  3.8e-16  
       265  5.53             prime    replayed        2064       1372    0.64  5.7e-16  bluestein
       266  2.7.19           flat     replayed         395        673    1.19  6.0e-16  flips differ 1.44x
       267  3.89             prime    replayed        2046       2261    1.10  4.9e-16  bluestein
       268  2^2.67           prime    replayed        2177       1817    0.82  6.9e-16  bluestein
       269  269              prime    replayed        2128       2073    0.97  6.2e-16  bluestein
       270  2.3^3.5          chain3   replayed         223        706    3.16  4.9e-16  
       271  271              prime    replayed         792       2077    2.61  7.2e-16  rader
       272  2^4.17           2p       replayed         218       1065    4.12  5.2e-16  
       273  3.7.13           2p       replayed         262        514    1.94  6.8e-16  
       274  2.137            prime    replayed        2188       2073    0.95  5.8e-16  bluestein
       275  5^2.11           chain3   replayed         252        488    1.91  3.4e-16  
       276  2^2.3.23         2p       replayed         290       1293    4.45  4.5e-16  
       277  277              prime    replayed         996       2075    1.95  7.5e-16  rader
       278  2.139            prime    replayed        2216       2071    0.93  4.9e-16  bluestein
       279  3^2.31           2p       replayed         607       1006    1.55  4.3e-16  
       280  2^3.5.7          chain3   replayed         210        775    3.68  3.9e-16  
       281  281              prime    replayed         816       2069    2.50  6.3e-16  rader
       282  2.3.47           2p       replayed         632       1425    2.23  4.3e-16  
       283  283              prime    replayed        2171       2076    0.94  5.4e-16  bluestein
       284  2^2.71           prime    replayed        2135       2056    0.95  3.3e-16  bluestein
       285  3.5.19           2p       replayed         328        715    2.15  4.4e-16  
       286  2.11.13          flat     replayed         408        792    1.94  5.6e-16  
       287  7.41             2p       replayed         451       1159    2.57  3.7e-16  
       288  2^5.3^2          2p       replayed         198        907    4.54  3.8e-16  
       289  17^2             2p       replayed         357        984    2.72  7.9e-16  
       290  2.5.29           2p       replayed         333        950    2.85  5.0e-16  
       291  3.97             prime    replayed        2070       2668    1.29  6.7e-16  bluestein
       292  2^2.73           prime    replayed        2079       2102    1.01  5.1e-16  bluestein
       293  293              prime    replayed        2132       2087    0.97  4.6e-16  bluestein
       294  2.3.7^2          chain3   replayed         253        679    2.64  3.7e-16  
       295  5.59             prime    replayed        2070       1710    0.82  4.4e-16  bluestein
       296  2^3.37           2p       replayed         372       1712    4.53  4.0e-16  
       297  3^3.11           chain3   replayed         287        907    3.09  5.2e-16  
       298  2.149            prime    replayed        2221       2082    0.67  6.5e-16  flips differ 1.39x bluestein
       299  13.23            2p       replayed         356       1105    3.07  5.0e-16  
       300  2^2.3.5^2        chain3   replayed         233        837    3.28  4.1e-16  
       301  7.43             flat     replayed         554       1315    2.37  4.6e-16  
       302  2.151            prime    replayed        2070       2084    0.94  5.6e-16  bluestein
       303  3.101            prime    replayed        2073       2090    1.00  4.1e-16  bluestein
       304  2^4.19           2p       replayed         307       1301    3.08  2.7e-16  flips differ 1.37x
       305  5.61             prime    replayed        2095       1799    0.85  5.5e-16  bluestein
       306  2.3^2.17         chain3   replayed         335       1079    3.17  6.0e-16  
       307  307              prime    replayed        1216       2092    1.67  9.1e-16  rader
       308  2^2.7.11         chain3   replayed         274       1042    3.58  4.5e-16  
       309  3.103            prime    replayed        2150       2092    0.97  6.0e-16  bluestein
       310  2.5.31           2p       replayed         667       1118    1.53  4.0e-16  
       311  311              prime    replayed        1292       2088    1.61  7.1e-16  rader
       312  2^3.3.13         chain3   replayed         276       1126    4.07  5.1e-16  
       313  313              prime    replayed        1076       2088    1.93  9.6e-16  rader
       314  2.157            prime    replayed        2082       2084    0.91  4.3e-16  bluestein
       315  3^2.5.7          chain3   replayed         292        728    2.45  7.9e-16  
       316  2^2.79           prime    replayed        2185       2586    1.15  4.4e-16  bluestein
       317  317              prime    replayed        2179       2090    0.95  6.3e-16  bluestein
       318  2.3.53           prime    replayed        2080       1719    0.82  4.7e-16  bluestein
       319  11.29            2p       replayed         422       1264    2.98  5.6e-16  
       320  2^6.5            2p       replayed         205        955    4.62  3.0e-16  
       321  3.107            prime    replayed        2104       2102    0.99  6.1e-16  bluestein
       322  2.7.23           flat     replayed         518        952    1.82  4.9e-16  
       323  17.19            2p       replayed         416       1202    2.87  6.8e-16  
       324  2^2.3^4          chain3   replayed         271       1001    3.31  4.6e-16  
       325  5^2.13           2p       replayed         318        606    1.88  7.5e-16  
       326  2.163            prime    replayed        2210       2085    0.94  5.4e-16  bluestein
       327  3.109            prime    replayed        2094       2099    1.00  6.6e-16  bluestein
       328  2^3.41           2p       replayed         611       1947    2.04  3.9e-16  flips differ 1.56x
       329  7.47             flat     replayed         634       1585    2.50  4.3e-16  
       330  2.3.5.11         chain3   replayed         301       1041    3.45  5.3e-16  
       331  331              prime    replayed        1109       2090    1.83  5.7e-16  rader
       332  2^2.83           prime    replayed        2086       2735    1.30  6.5e-16  bluestein
       333  3^2.37           2p       replayed         999       1327    1.18  5.9e-16  
       334  2.167            prime    replayed        2131       2089    0.98  5.3e-16  bluestein
       335  5.67             prime    replayed        2216       2211    0.98  5.9e-16  bluestein
       336  2^4.3.7          2p       replayed         245        974    3.73  3.2e-16  
       337  337              prime    replayed        1051       2104    1.94  8.1e-16  rader
       338  2.13^2           flat     replayed         487        974    1.97  9.8e-16  
       339  3.113            prime    replayed        2395       2092    0.83  6.3e-16  bluestein
       340  2^2.5.17         chain3   replayed         338       1277    3.52  6.3e-16  
       341  11.31            flat     replayed         598       1473    2.06  6.8e-16  
       342  2.3^2.19         chain3   replayed         380       1318    1.73  4.2e-16  flips differ 2.01x
       343  7^3              chain3   replayed         348        378    1.07  4.9e-16  
       344  2^3.43           2p       replayed         489       2149    4.37  4.4e-16  
       345  3.5.23           2p       replayed         477       1012    1.28  5.9e-16  flips differ 1.66x
       346  2.173            prime    replayed        2174       2096    0.94  5.1e-16  bluestein
       347  347              prime    replayed        2116       2101    0.98  5.9e-16  bluestein
       348  2^2.3.29         2p       replayed         417       1726    4.07  4.5e-16  
       349  349              prime    replayed        1451       2091    1.11  7.8e-16  flips differ 1.30x rader
       350  2.5^2.7          chain3   replayed         284        709    2.26  2.6e-16  
       351  3^3.13           chain3   replayed         348       1108    3.16  5.7e-16  
       352  2^5.11           2p       replayed         259       1266    4.43  2.1e-16  
       353  353              prime    replayed        1031       2109    2.03  6.5e-16  rader
       354  2.3.59           prime    replayed        2237       2128    0.95  5.4e-16  bluestein
       355  5.71             prime    replayed        2092       2506    1.19  4.7e-16  bluestein
       356  2^2.89           prime    replayed        2201       3099    1.40  5.5e-16  bluestein
       357  3.7.17           chain3   replayed         409        792    1.76  5.6e-16  
       358  2.179            prime    replayed        2113       2093    0.98  5.2e-16  bluestein
       359  359              prime    replayed        2108       2103    0.99  6.7e-16  bluestein
       360  2^3.3^2.5        chain3   replayed         294       1056    3.57  4.9e-16  
       361  19^2             2p       replayed         477       1466    2.29  5.0e-16  flips differ 1.34x
       362  2.181            prime    replayed        2116       2103    0.99  5.7e-16  bluestein
       363  3.11^2           chain3   replayed         371        979    2.60  3.3e-16  
       364  2^2.7.13         chain3   replayed         338       1253    2.96  5.6e-16  flips differ 1.25x
       365  5.73             prime    replayed        2102       2564    1.20  7.9e-16  bluestein
       366  2.3.61           prime    replayed        2308       2274    0.97  4.6e-16  bluestein
       367  367              prime    replayed        2207       2108    0.78  6.5e-16  bluestein
       368  2^4.23           2p       replayed         478       1735    3.19  3.8e-16  
       369  3^2.41           2p       replayed         999       1571    1.22  5.4e-16  flips differ 1.29x
       370  2.5.37           2p       replayed         484       1476    3.03  3.7e-16  
       371  7.53             prime    replayed        2192       1941    0.88  4.9e-16  bluestein
       372  2^2.3.31         flat     replayed         739       1969    2.51  5.3e-16  
       373  373              prime    replayed        2114       2106    0.99  6.9e-16  bluestein
       374  2.11.17          flat     replayed         553       1213    2.10  1.1e-15  
       375  3.5^3            chain3   replayed         349        822    2.31  6.8e-16  
       376  2^3.47           2p       replayed         827       2545    3.05  5.1e-16  
       377  13.29            2p       replayed         781       1539    1.97  6.2e-16  
       378  2.3^3.7          chain3   replayed         336        991    2.94  4.8e-16  
       379  379              prime    replayed        1674       2109    1.24  1.1e-15  rader
       380  2^2.5.19         chain3   replayed         408       1561    3.69  3.7e-16  
       381  3.127            prime    replayed        2112       2108    0.99  5.2e-16  bluestein
       382  2.191            prime    replayed        2114       2105    0.98  7.2e-16  bluestein
       383  383              prime    replayed        2254       2120    0.92  6.4e-16  bluestein
       384  2^7.3            2p       replayed         252       1231    4.88  3.0e-16  
       385  5.7.11           chain3   replayed         373        698    1.85  5.6e-16  
       386  2.193            prime    replayed        2272       2116    0.90  7.6e-16  bluestein
       387  3^2.43           2p       replayed         625       1774    2.83  5.4e-16  
       388  2^2.97           prime    replayed        2197       3668    1.66  4.7e-16  bluestein
       389  389              prime    replayed        2124       2111    0.96  6.1e-16  bluestein
       390  2.3.5.13         chain3   replayed         367       1280    3.44  6.2e-16  
       391  17.23            2p       replayed         547       1619    2.94  6.7e-16  
       392  2^3.7^2          chain3   replayed         331       1091    3.27  2.7e-16  
       393  3.131            prime    replayed        2266       2111    0.93  7.4e-16  bluestein
       394  2.197            prime    replayed        2142       2112    0.93  6.0e-16  bluestein
       395  5.79             prime    replayed        2129       3071    1.44  5.1e-16  bluestein
       396  2^2.3^2.11       chain3   replayed         372       1391    3.67  4.5e-16  
       397  397              prime    replayed        1533       2127    1.38  8.9e-16  rader
       398  2.199            prime    replayed        2160       2107    0.96  5.6e-16  bluestein
       399  3.7.19           chain3   replayed         466       1021    2.06  6.2e-16  
       400  2^4.5^2          2p       replayed         308       1150    3.72  3.6e-16  
       401  401              prime    replayed        1416       2114    1.45  7.6e-16  rader
       402  2.3.67           prime    replayed        2280       2751    1.17  5.9e-16  bluestein
       403  13.31            2p       replayed         701       1775    2.51  6.4e-16  
       404  2^2.101          prime    replayed        2317       2121    0.91  4.9e-16  bluestein
       405  3^4.5            chain3   replayed         378        975    2.52  5.2e-16  
       406  2.7.29           flat     replayed         717       1347    1.79  5.6e-16  
       407  11.37            2p       replayed         605       1908    3.14  4.1e-16  
       408  2^3.3.17         chain3   replayed         420       1611    3.66  5.1e-16  
       409  409              prime    replayed        1659       2116    1.27  8.1e-16  rader
       410  2.5.41           2p       replayed         791       1750    2.18  4.9e-16  
       411  3.137            prime    replayed        2367       2120    0.88  1.0e-15  bluestein
       412  2^2.103          prime    replayed        2254       2191    0.94  5.5e-16  bluestein
       413  7.59             prime    replayed        2224       2405    1.08  4.9e-16  bluestein
       414  2.3^2.23         chain3   replayed         511       1809    3.50  5.3e-16  
       415  5.83             prime    replayed        2276       3385    1.48  4.7e-16  bluestein
       416  2^5.13           2p       replayed         315       1542    4.73  5.4e-16  
       417  3.139            prime    replayed        2291       2123    0.92  6.4e-16  bluestein
       418  2.11.19          flat     replayed         658       1435    2.02  1.0e-15  
       419  419              prime    replayed        2177       2117    0.96  6.0e-16  bluestein
       420  2^2.3.5.7        chain3   replayed         357       1167    3.25  5.2e-16  
       421  421              prime    replayed        1915       2129    1.08  7.0e-16  rader
       422  2.211            prime    replayed        2231       2142    0.96  7.9e-16  bluestein
       423  3^2.47           flat     replayed         819       2129    2.57  7.0e-16  
       424  2^3.53           prime    replayed        2139       3045    1.40  4.4e-16  bluestein
       425  5^2.17           2p       replayed         455        950    2.07  6.4e-16  
       426  2.3.71           prime    replayed        2151       3168    1.37  5.2e-16  bluestein
       427  7.61             prime    replayed        2179       2565    1.17  4.3e-16  bluestein
       428  2^2.107          prime    replayed        2291       2137    0.92  5.4e-16  bluestein
       429  3.11.13          chain3   replayed         465       1205    2.57  5.2e-16  
       430  2.5.43           2p       replayed         631       2031    2.84  3.4e-16  
       431  431              prime    replayed        2148       2135    0.99  7.7e-16  bluestein
       432  2^4.3^3          2p       replayed         331       1364    4.08  5.1e-16  
       433  433              prime    replayed        1499       2123    1.37  8.8e-16  rader
       434  2.7.31           flat     replayed         789       1578    1.97  4.6e-16  
       435  3.5.29           2p       replayed         581       1444    2.48  5.5e-16  
       436  2^2.109          prime    replayed        2299       2134    0.91  5.4e-16  bluestein
       437  19.23            2p       replayed         621       1974    2.98  5.5e-16  
       438  2.3.73           prime    replayed        2150       3195    1.46  5.3e-16  bluestein
       439  439              prime    replayed        2148       2130    0.99  6.7e-16  bluestein
       440  2^3.5.11         chain3   replayed         381       1546    3.95  5.3e-16  
       441  3^2.7^2          chain3   replayed         444        815    1.80  4.9e-16  
       442  2.13.17          flat     replayed         666       1434    2.14  1.3e-15  
       443  443              prime    replayed        2153       2141    0.99  6.1e-16  bluestein
       444  2^2.3.37         2p       replayed         604       2524    4.15  6.7e-16  
       445  5.89             prime    replayed        2267       3814    1.68  4.5e-16  bluestein
       446  2.223            prime    replayed        2173       2134    0.93  8.6e-16  bluestein
       447  3.149            prime    replayed        2159       2137    0.98  7.3e-16  bluestein
       448  2^6.7            chain3   replayed         351       1340    3.80  2.2e-16  
       449  449              prime    replayed        1462       2129    1.41  8.3e-16  rader
       450  2.3^2.5^2        chain3   replayed         388       1187    3.06  5.1e-16  
       451  11.41            2p       replayed        1564       2239    1.28  5.0e-16  
       452  2^2.113          prime    replayed        2166       2137    0.97  7.0e-16  bluestein
       453  3.151            prime    replayed        2145       2133    0.99  6.9e-16  bluestein
       454  2.227            prime    replayed        2159       2165    0.99  7.6e-16  bluestein
       455  5.7.13           chain3   replayed         457        877    1.90  6.0e-16  
       456  2^3.3.19         chain3   replayed         496       1955    3.92  4.7e-16  
       457  457              prime    replayed        1916       2142    1.08  8.2e-16  rader
       458  2.229            prime    replayed        2146       2137    0.98  5.9e-16  bluestein
       459  3^3.17           chain3   replayed         527       1569    2.73  5.1e-16  
       460  2^2.5.23         flat     replayed         647       2078    3.19  6.3e-16  
       461  461              prime    replayed        2171       2145    0.96  7.4e-16  bluestein
       462  2.3.7.11         chain3   replayed         464       1396    2.98  4.8e-16  
       463  463              prime    replayed        1793       2143    1.18  6.4e-16  rader
       464  2^4.29           2p       replayed         652       2354    3.61  2.8e-16  
       465  3.5.31           2p       replayed         800       1692    2.07  5.2e-16  
       466  2.233            prime    replayed        2168       2153    0.99  8.3e-16  bluestein
       467  467              prime    replayed        2247       2133    0.95  6.2e-16  bluestein
       468  2^2.3^2.13       chain3   replayed         477       1689    3.54  4.9e-16  
       469  7.67             prime    replayed        2294       3121    1.36  3.9e-16  bluestein
       470  2.5.47           2p       replayed        1064       2381    2.21  4.4e-16  
       471  3.157            prime    replayed        2163       2147    0.98  7.7e-16  bluestein
       472  2^3.59           prime    replayed        2175       3647    1.66  4.7e-16  bluestein
       473  11.43            2p       replayed         777       2504    3.21  6.0e-16  
       474  2.3.79           prime    replayed        2241       3834    1.71  4.7e-16  bluestein
       475  5^2.19           2p       replayed         530       1210    2.23  5.0e-16  
       476  2^2.7.17         chain3   replayed         559       1700    2.95  5.1e-16  
       477  3^2.53           prime    replayed        2164       2586    1.19  5.1e-16  bluestein
       478  2.239            prime    replayed        2166       2144    0.98  6.5e-16  bluestein
       479  479              prime    replayed        2182       2150    0.98  6.1e-16  bluestein
       480  2^5.3.5          2p       replayed         376       1450    3.85  4.1e-16  
       481  13.37            2p       replayed         724       2319    3.18  5.5e-16  
       482  2.241            prime    replayed        2248       2146    0.94  7.3e-16  bluestein
       483  3.7.23           2p       replayed         764       1434    1.86  5.1e-16  
       484  2^2.11^2         chain3   replayed         589       1705    2.84  3.7e-16  
       485  5.97             prime    replayed        2176       4589    2.09  5.9e-16  bluestein
       486  2.3^5            chain3   replayed         479       1422    2.94  5.0e-16  
       487  487              prime    replayed        2254       2150    0.95  4.2e-16  bluestein
       488  2^3.61           prime    replayed        2185       3925    1.75  4.5e-16  bluestein
       489  3.163            prime    replayed        2260       2150    0.85  6.1e-16  bluestein
       490  2.5.7^2          chain3   replayed         447        783    1.62  6.6e-16  
       491  491              prime    replayed        1816       2146    1.17  7.1e-16  rader
       492  2^2.3.41         2p       replayed         954       2890    3.01  4.8e-16  
       493  17.29            2p       replayed         772       2296    2.88  5.0e-16  
       494  2.13.19          flat     replayed         780       1760    2.08  1.0e-15  
       495  3^2.5.11         flat     replayed         585       1521    2.54  5.8e-16  
       496  2^4.31           flat     replayed         957       2683    2.78  5.7e-16  
       497  7.71             prime    replayed        2183       3526    1.59  4.7e-16  bluestein
       498  2.3.83           prime    replayed        2332       4183    1.29  5.3e-16  flips differ 1.40x bluestein
       499  499              prime    replayed        2199       2152    0.98  8.3e-16  bluestein
       500  2^2.5^3          chain3   replayed         425       1361    3.16  3.8e-16  
       501  3.167            prime    replayed        2189       2159    0.97  7.5e-16  bluestein
       502  2.251            prime    replayed        2266       2158    0.95  5.2e-16  bluestein
       503  503              prime    replayed        2176       2156    0.98  5.6e-16  bluestein
       504  2^3.3^2.7        chain3   replayed         458       1470    3.05  4.9e-16  
       505  5.101            prime    replayed        2191       2160    0.98  6.3e-16  bluestein
       506  2.11.23          flat     replayed         814       1988    2.38  1.3e-15  
       507  3.13^2           chain3   replayed         589       1484    2.38  6.2e-16  
       508  2^2.127          prime    replayed        2188       2159    0.97  5.1e-16  bluestein
       509  509              prime    replayed        2295       2155    0.94  5.0e-16  bluestein
       510  2.3.5.17         chain3   replayed         543       1812    3.30  6.7e-16  
       511  7.73             prime    replayed        2184       3629    1.66  6.6e-16  bluestein
       512  2^9              2p       replayed         297        332    1.04  2.3e-16  
       513  3^3.19           2p       replayed         601       1948    3.24  5.4e-16  
       514  2.257            prime    replayed        4773       4499    0.94  6.9e-16  bluestein
       515  5.103            prime    replayed        4727       4503    0.94  6.1e-16  bluestein
       516  2^2.3.43         flat     replayed         922       3201    3.36  4.9e-16  
       517  11.47            flat     replayed        1033       2970    2.41  6.4e-16  
       518  2.7.37           flat     replayed        1007       2092    2.06  6.7e-16  
       519  3.173            prime    replayed        4780       4867    0.94  4.7e-16  bluestein
       520  2^3.5.13         chain3   replayed         487       1884    3.61  6.2e-16  
       521  521              prime    replayed        1958       4523    2.27  1.1e-15  rader
       522  2.3^2.29         chain3   replayed         740       2491    3.31  4.3e-16  
       523  523              prime    replayed        3250       4525    1.38  7.4e-16  rader
       524  2^2.131          prime    replayed        4746       4530    0.95  4.7e-16  bluestein
       525  3.5^2.7          2p       replayed         541       1181    2.11  6.4e-16  
       526  2.263            prime    replayed        4749       4521    0.94  6.8e-16  bluestein
       527  17.31            2p       replayed         808       2565    3.06  7.5e-16  
       528  2^4.3.11         chain3   replayed         495       1915    3.73  3.8e-16  
       529  23^2             2p       replayed         941       2618    2.39  6.5e-16  
       530  2.5.53           prime    replayed        4752       2891    0.61  6.4e-16  bluestein
       531  3^2.59           prime    replayed        4781       3211    0.67  4.5e-16  bluestein
       532  2^2.7.19         flat     replayed         631       2067    3.26  5.0e-16  
       533  13.41            2p       replayed         867       2709    3.11  6.4e-16  
       534  2.3.89           prime    replayed        4720       4822    0.76  5.4e-16  flips differ 1.33x bluestein
       535  5.107            prime    replayed        4779       4530    0.95  5.3e-16  bluestein
       536  2^3.67           prime    replayed        4747       4581    0.95  5.1e-16  bluestein
       537  3.179            prime    replayed        4734       4514    0.95  5.8e-16  bluestein
       538  2.269            prime    replayed        4777       4531    0.94  5.8e-16  bluestein
       539  7^2.11           flat     replayed         667        998    1.47  5.4e-16  
       540  2^2.3^3.5        chain3   replayed         481       1618    3.30  5.0e-16  
       541  541              prime    replayed        2891       4516    1.43  8.2e-16  rader
       542  2.271            prime    replayed        4771       4560    0.90  6.7e-16  bluestein
       543  3.181            prime    replayed        4739       4892    0.95  5.1e-16  bluestein
       544  2^5.17           2p       replayed         546       2220    3.99  4.0e-16  
       545  5.109            prime    replayed        5069       4522    0.89  5.5e-16  bluestein
       546  2.3.7.13         chain3   replayed         635       1738    2.63  5.9e-16  
       547  547              prime    replayed        2245       4539    2.02  1.0e-15  rader
       548  2^2.137          prime    replayed        4733       4520    0.95  5.2e-16  bluestein
       549  3^2.61           prime    replayed        4953       3411    0.66  4.6e-16  bluestein
       550  2.5^2.11         chain3   replayed         520       1595    2.99  3.9e-16  
       551  19.29            2p       replayed         871       2720    2.98  4.0e-16  
       552  2^3.3.23         flat     replayed         775       2660    3.35  5.3e-16  
       553  7.79             prime    replayed        4855       4375    0.89  4.6e-16  bluestein
       554  2.277            prime    replayed        4738       4527    0.95  5.1e-16  bluestein
       555  3.5.37           2p       replayed         831       2239    2.68  6.0e-16  
       556  2^2.139          prime    replayed        4726       4562    0.96  5.2e-16  bluestein
       557  557              prime    replayed        4729       4516    0.95  6.3e-16  bluestein
       558  2.3^2.31         chain3   replayed         851       2870    3.31  5.0e-16  
       559  13.43            2p       replayed         933       3019    3.21  5.6e-16  
       560  2^4.5.7          chain3   replayed         521       1609    3.02  3.4e-16  
       561  3.11.17          flat     replayed         825       1772    2.12  1.2e-15  
       562  2.281            prime    replayed        4697       4519    0.94  5.3e-16  bluestein
       563  563              prime    replayed        4817       4530    0.93  9.4e-16  bluestein
       564  2^2.3.47         flat     replayed        1099       3784    3.43  4.1e-16  
       565  5.113            prime    replayed        4772       4517    0.64  5.7e-16  flips differ 1.48x bluestein
       566  2.283            prime    replayed        4902       4546    0.87  7.6e-16  bluestein
       567  3^4.7            2p       replayed         596       1386    1.91  6.4e-16  
       568  2^3.71           prime    replayed        4810       5107    1.06  3.7e-16  bluestein
       569  569              prime    replayed        4759       4526    0.95  5.8e-16  bluestein
       570  2.3.5.19         flat     replayed         852       2249    2.50  4.7e-16  
       571  571              prime    replayed        2625       4533    1.71  7.5e-16  rader
       572  2^2.11.13        chain3   replayed         682       2036    2.89  5.3e-16  
       573  3.191            prime    replayed        4801       4595    0.94  6.8e-16  bluestein
       574  2.7.41           flat     replayed        1257       2485    1.96  2.2e-15  
       575  5^2.23           2p       replayed         848       1702    1.99  5.8e-16  
       576  2^6.3^2          2p       replayed         502       1864    3.67  3.5e-16  
       577  577              prime    replayed        1911       4570    2.37  7.9e-16  rader
       578  2.17^2           flat     replayed         920       2145    2.08  8.0e-16  
       579  3.193            prime    replayed        4780       4542    0.94  4.9e-16  bluestein
       580  2^2.5.29         chain3   replayed         910       2832    3.07  5.0e-16  
       581  7.83             prime    replayed        4739       4770    1.00  5.0e-16  bluestein
       582  2.3.97           prime    replayed        4828       5582    0.91  6.0e-16  flips differ 1.27x bluestein
       583  11.53            prime    replayed        4887       3598    0.68  5.1e-16  bluestein
       584  2^3.73           prime    replayed        4856       5287    1.02  4.4e-16  bluestein
       585  3^2.5.13         chain3   replayed         768       1887    2.42  5.6e-16  
       586  2.293            prime    replayed        4766       4534    0.94  6.0e-16  bluestein
       587  587              prime    replayed        4886       4529    0.90  6.6e-16  bluestein
       588  2^2.3.7^2        flat     replayed         620       1656    1.95  5.0e-16  flips differ 1.37x
       589  19.31            2p       replayed         950       3091    3.22  5.1e-16  
       590  2.5.59           prime    replayed        4840       3578    0.72  4.6e-16  bluestein
       591  3.197            prime    replayed        4891       4540    0.91  5.2e-16  bluestein
       592  2^4.37           2p       replayed         752       3464    4.55  3.6e-16  
       593  593              prime    replayed        3026       4549    1.25  7.0e-16  rader
       594  2.3^3.11         chain3   replayed         697       1993    2.73  5.1e-16  
       595  5.7.17           chain3   replayed         898       1352    1.48  6.6e-16  
       596  2^2.149          prime    replayed        4779       4582    0.94  5.4e-16  bluestein
       597  3.199            prime    replayed        4820       4537    0.94  4.9e-16  bluestein
       598  2.13.23          flat     replayed        1016       2382    2.04  9.1e-16  
       599  599              prime    replayed        4835       4533    0.93  5.6e-16  bluestein
       600  2^3.3.5^2        chain3   replayed         600       1740    2.81  4.9e-16  
       601  601              prime    replayed        3027       4529    1.43  8.9e-16  rader
       602  2.7.43           flat     replayed        1302       2806    2.08  6.5e-16  
       603  3^2.67           prime    replayed        4704       4163    0.87  3.9e-16  bluestein
       604  2^2.151          prime    replayed        4785       4544    0.93  6.0e-16  bluestein
       605  5.11^2           flat     replayed         742       1655    2.20  6.1e-16  
       606  2.3.101          prime    replayed        4737       4536    0.95  4.8e-16  bluestein
       607  607              prime    replayed        4842       4548    0.94  6.6e-16  bluestein
       608  2^5.19           2p       replayed         564       2641    4.65  3.6e-16  
       609  3.7.29           2p       replayed        1050       2070    1.56  5.8e-16  
       610  2.5.61           prime    replayed        4762       3790    0.78  6.2e-16  bluestein
       611  13.47            flat     replayed        1229       3576    2.89  7.8e-16  
       612  2^2.3^2.17       chain3   replayed         849       2499    2.70  6.2e-16  
       613  613              prime    replayed        2685       4619    1.60  1.0e-15  rader
       614  2.307            prime    replayed        4771       4537    0.94  6.2e-16  bluestein
       615  3.5.41           2p       replayed        1000       2660    2.63  6.3e-16  
       616  2^3.7.11         chain3   replayed         662       2195    3.22  3.5e-16  
       617  617              prime    replayed        2330       4553    1.95  7.3e-16  rader
       618  2.3.103          prime    replayed        4798       4548    0.94  6.2e-16  bluestein
       619  619              prime    replayed        4822       4651    0.94  6.0e-16  bluestein
       620  2^2.5.31         flat     replayed        1187       3246    2.71  6.2e-16  
       621  3^3.23           2p       replayed         781       2687    3.42  6.7e-16  
       622  2.311            prime    replayed        4840       4547    0.89  6.4e-16  bluestein
       623  7.89             prime    replayed        4801       5386    1.09  4.8e-16  bluestein
       624  2^4.3.13         chain3   replayed         710       2320    3.25  5.5e-16  
       625  5^4              2p       replayed         691       1214    1.70  4.1e-16  
       626  2.313            prime    replayed        4796       4539    0.94  6.3e-16  bluestein
       627  3.11.19          flat     replayed         979       2166    1.86  6.9e-16  
       628  2^2.157          prime    replayed        4857       4542    0.93  5.7e-16  bluestein
       629  17.37            2p       replayed        1522       3313    1.58  8.6e-16  flips differ 1.38x
       630  2.3^2.5.7        chain3   replayed         677       1655    2.29  4.5e-16  
       631  631              prime    replayed        2967       4538    1.50  8.9e-16  rader
       632  2^3.79           prime    replayed        4811       6220    1.28  5.4e-16  bluestein
       633  3.211            prime    replayed        5017       4621    0.91  4.2e-16  bluestein
       634  2.317            prime    replayed        4735       4747    0.96  4.9e-16  bluestein
       635  5.127            prime    replayed        4772       4546    0.94  5.7e-16  bluestein
       636  2^2.3.53         prime    replayed        4771       4513    0.94  3.7e-16  bluestein
       637  7^2.13           flat     replayed         908       1251    1.37  6.3e-16  
       638  2.11.29          flat     replayed        1190       2699    2.26  5.8e-16  
       639  3^2.71           prime    replayed        4800       4762    0.98  4.6e-16  bluestein
       640  2^7.5            2p       replayed         509       1983    3.90  4.2e-16  
       641  641              prime    replayed        1924       4557    2.31  7.5e-16  rader
       642  2.3.107          prime    replayed        4811       4562    0.94  5.2e-16  bluestein
       643  643              prime    replayed        4840       5442    0.93  6.3e-16  bluestein
       644  2^2.7.23         flat     replayed         932       2810    2.91  6.6e-16  
       645  3.5.43           2p       replayed        1074       2992    2.78  5.5e-16  
       646  2.17.19          flat     replayed        1069       2684    2.38  1.0e-15  
       647  647              prime    replayed        4801       4567    0.93  7.4e-16  bluestein
       648  2^3.3^4          chain3   replayed         623       2106    3.37  5.2e-16  
       649  11.59            prime    replayed        4810       4392    0.91  3.8e-16  bluestein
       650  2.5^2.13         chain3   replayed         801       2013    2.48  6.7e-16  
       651  3.7.31           2p       replayed        1190       2391    2.00  5.2e-16  
       652  2^2.163          prime    replayed        4827       4532    0.94  4.5e-16  bluestein
       653  653              prime    replayed        4807       4555    0.94  5.4e-16  bluestein
       654  2.3.109          prime    replayed        4830       4917    0.93  5.5e-16  bluestein
       655  5.131            prime    replayed        4957       4563    0.91  6.0e-16  bluestein
       656  2^4.41           2p       replayed         910       4002    3.81  5.3e-16  
       657  3^2.73           prime    replayed        4759       4890    1.02  6.0e-16  bluestein
       658  2.7.47           flat     replayed        1544       3358    2.17  6.5e-16  
       659  659              prime    replayed        4785       4564    0.95  6.0e-16  bluestein
       660  2^2.3.5.11       chain3   replayed         701       2346    3.32  4.4e-16  
       661  661              prime    replayed        3027       4533    1.42  7.4e-16  rader
       662  2.331            prime    replayed        4874       4558    0.93  7.8e-16  bluestein
       663  3.13.17          flat     replayed         991       2169    2.13  8.9e-16  
       664  2^3.83           prime    replayed        4851       6754    1.05  5.9e-16  flips differ 1.35x bluestein
       665  5.7.19           flat     replayed         919       1719    1.83  5.1e-16  
       666  2.3^2.37         chain3   replayed        1242       3640    2.91  4.2e-16  
       667  23.29            2p       replayed        1295       3599    2.28  6.1e-16  
       668  2^2.167          prime    replayed        4746       4568    0.94  5.4e-16  bluestein
       669  3.223            prime    replayed        4849       4937    0.95  5.4e-16  bluestein
       670  2.5.67           prime    replayed        4791       4662    0.96  4.8e-16  bluestein
       671  11.61            prime    replayed        4878       4645    0.94  7.1e-16  bluestein
       672  2^5.3.7          2p       replayed         586       2026    2.65  3.3e-16  flips differ 1.31x
       673  673              prime    replayed        2532       4574    1.72  7.7e-16  rader
       674  2.337            prime    replayed        4817       4557    0.92  7.2e-16  bluestein
       675  3^3.5^2          2p       replayed         751       1673    2.20  5.9e-16  
       676  2^2.13^2         chain3   replayed         984       2478    2.48  7.5e-16  
       677  677              prime    replayed        3015       4559    1.50  9.2e-16  rader
       678  2.3.113          prime    replayed        4798       4584    0.95  4.9e-16  bluestein
       679  7.97             prime    replayed        4859       6372    1.31  5.8e-16  bluestein
       680  2^3.5.17         flat     replayed         854       2681    3.08  5.9e-16  
       681  3.227            prime    replayed        4818       4561    0.93  6.6e-16  bluestein
       682  2.11.31          flat     replayed        1290       3139    2.40  6.9e-16  
       683  683              prime    replayed        4781       4565    0.95  4.8e-16  bluestein
       684  2^2.3^2.19       chain3   replayed         881       2936    3.26  5.8e-16  
       685  5.137            prime    replayed        4806       4556    0.93  6.2e-16  bluestein
       686  2.7^3            flat     replayed         946       1102    1.14  4.1e-16  
       687  3.229            prime    replayed        4882       4576    0.78  5.4e-16  bluestein
       688  2^4.43           flat     replayed        1236       4406    3.52  4.3e-16  
       689  13.53            prime    replayed        4855       4324    0.89  5.0e-16  bluestein
       690  2.3.5.23         chain3   replayed        1130       3046    2.62  4.8e-16  
       691  691              prime    replayed        3428       4570    1.29  8.0e-16  rader
       692  2^2.173          prime    replayed        4790       4570    0.95  4.8e-16  bluestein
       693  3^2.7.11         flat     replayed         891       1806    1.51  7.0e-16  flips differ 1.41x
       694  2.347            prime    replayed        4808       4560    0.94  6.5e-16  bluestein
       695  5.139            prime    replayed        4916       4576    0.90  5.9e-16  bluestein
       696  2^3.3.29         flat     replayed        1228       3579    2.56  5.3e-16  
       697  17.41            2p       replayed        1243       3886    3.02  8.6e-16  
       698  2.349            prime    replayed        4819       4570    0.94  6.1e-16  bluestein
       699  3.233            prime    replayed        4920       4601    0.91  5.9e-16  bluestein
       700  2^2.5^2.7        chain3   replayed         726       1915    2.63  3.7e-16  
       701  701              prime    replayed        3585       4562    1.26  9.5e-16  rader
       702  2.3^3.13         chain3   replayed         932       2496    2.64  5.7e-16  
       703  19.37            2p       replayed        1224       4000    3.13  4.6e-16  
       704  2^6.11           2p       replayed         655       2610    3.88  3.1e-16  
       705  3.5.47           2p       replayed        1259       3592    2.82  6.6e-16  
       706  2.353            prime    replayed        4802       4608    0.96  5.3e-16  bluestein
       707  7.101            prime    replayed        4837       4692    0.94  5.5e-16  bluestein
       708  2^2.3.59         prime    replayed        4835       5439    1.07  5.7e-16  bluestein
       709  709              prime    replayed        4871       4580    0.94  5.5e-16  bluestein
       710  2.5.71           prime    replayed        4801       5295    1.09  5.5e-16  bluestein
       711  3^2.79           prime    replayed        4827       5802    1.20  6.4e-16  bluestein
       712  2^3.89           prime    replayed        4824       7460    1.54  5.2e-16  bluestein
       713  23.31            2p       replayed        1500       4072    2.14  9.4e-16  flips differ 1.26x
       714  2.3.7.17         chain3   replayed        1064       2202    1.87  7.4e-16  
       715  5.11.13          flat     replayed         924       2036    2.19  6.4e-16  
       716  2^2.179          prime    replayed        4900       4578    0.93  6.3e-16  bluestein
       717  3.239            prime    replayed        4910       4567    0.92  6.2e-16  bluestein
       718  2.359            prime    replayed        4843       4577    0.94  5.1e-16  bluestein
       719  719              prime    replayed        4790       4560    0.95  5.4e-16  bluestein
       720  2^4.3^2.5        chain3   replayed         801       2231    2.72  4.4e-16  
       721  7.103            prime    replayed        4774       4581    0.95  5.1e-16  bluestein
       722  2.19^2           flat     replayed        1231       3177    2.05  1.0e-15  flips differ 1.26x
       723  3.241            prime    replayed        4895       4574    0.73  4.5e-16  flips differ 1.29x bluestein
       724  2^2.181          prime    replayed        4857       4604    0.93  7.2e-16  bluestein
       725  5^2.29           flat     replayed        1254       2438    1.67  5.8e-16  
       726  2.3.11^2         chain3   replayed        1018       2572    2.51  4.5e-16  
       727  727              prime    replayed        3058       4551    1.47  6.5e-16  rader
       728  2^3.7.13         chain3   replayed         844       2662    3.06  5.1e-16  
       729  3^6              2p       replayed         917       2146    2.28  5.1e-16  
       730  2.5.73           prime    replayed        4879       5440    0.86  4.3e-16  flips differ 1.29x bluestein
       731  17.43            2p       replayed        1365       4335    3.13  6.0e-16  
       732  2^2.3.61         prime    replayed        4771       5789    1.17  6.0e-16  bluestein
       733  733              prime    replayed        4911       4587    0.93  6.2e-16  bluestein
       734  2.367            prime    replayed        4834       4584    0.94  5.3e-16  bluestein
       735  3.5.7^2          flat     replayed        1003       1249    1.24  6.1e-16  
       736  2^5.23           2p       replayed         784       3578    4.17  3.3e-16  
       737  11.67            prime    replayed        4749       5610    1.15  5.5e-16  bluestein
       738  2.3^2.41         chain3   replayed        2272       4268    1.85  5.8e-16  
       739  739              prime    replayed        4754       4628    0.94  5.1e-16  bluestein
       740  2^2.5.37         flat     replayed        1311       4174    2.88  5.4e-16  
       741  3.13.19          chain3   replayed        1413       2703    1.70  6.5e-16  
       742  2.7.53           prime    replayed        4870       4106    0.84  4.4e-16  bluestein
       743  743              prime    replayed        4903       4559    0.92  6.3e-16  bluestein
       744  2^3.3.31         flat     replayed        1558       4041    1.99  5.1e-16  flips differ 1.30x
       745  5.149            prime    replayed        4845       4575    0.94  6.8e-16  bluestein
       746  2.373            prime    replayed        4849       4585    0.94  7.4e-16  bluestein
       747  3^2.83           prime    replayed        4943       6321    1.26  6.1e-16  bluestein
       748  2^2.11.17        flat     replayed        1208       2950    2.41  7.9e-16  
       749  7.107            prime    replayed        4889       4584    0.94  6.6e-16  bluestein
       750  2.3.5^3          chain3   replayed         974       2001    2.02  4.7e-16  
       751  751              prime    replayed        4848       4970    0.92  6.3e-16  bluestein
       752  2^4.47           flat     replayed        1473       5147    3.48  5.8e-16  
       753  3.251            prime    replayed        4888       4587    0.92  6.0e-16  bluestein
       754  2.13.29          flat     replayed        1429       3291    2.21  1.0e-15  
       755  5.151            prime    replayed        4890       4574    0.93  5.7e-16  bluestein
       756  2^2.3^3.7        chain3   replayed         758       2237    2.94  3.9e-16  
       757  757              prime    replayed        4262       4586    1.00  6.3e-16  rader
       758  2.379            prime    replayed        4911       4583    0.93  6.3e-16  bluestein
       759  3.11.23          flat     replayed        1368       2980    1.80  6.8e-16  
       760  2^3.5.19         flat     replayed        1003       3281    3.17  6.1e-16  
       761  761              prime    replayed        3449       4586    1.29  7.6e-16  rader
       762  2.3.127          prime    replayed        5023       5526    0.82  5.9e-16  bluestein
       763  7.109            prime    replayed        4761       4586    0.94  5.5e-16  bluestein
       764  2^2.191          prime    replayed        4810       4571    0.94  6.1e-16  bluestein
       765  3^2.5.17         flat     replayed        1071       2720    2.53  6.0e-16  
       766  2.383            prime    replayed        4841       4591    0.72  5.3e-16  flips differ 1.32x bluestein
       767  13.59            prime    replayed        4810       5297    1.09  5.9e-16  bluestein
       768  2^8.3            2p       replayed         622       2616    4.13  4.3e-16  
       769  769              prime    replayed        2332       4587    1.93  6.4e-16  rader
       770  2.5.7.11         chain3   replayed         992       2044    2.03  3.9e-16  
       771  3.257            prime    replayed        4854       4667    0.95  7.5e-16  bluestein
       772  2^2.193          prime    replayed        4917       4577    0.73  5.9e-16  flips differ 1.27x bluestein
       773  773              prime    replayed        4767       4585    0.95  6.4e-16  bluestein
       774  2.3^2.43         chain3   replayed        1631       4737    2.88  4.7e-16  
       775  5^2.31           2p       replayed        1120       2859    2.53  6.3e-16  
       776  2^3.97           prime    replayed        4799       8750    1.82  3.1e-16  bluestein
       777  3.7.37           2p       replayed        1191       3163    2.63  5.5e-16  
       778  2.389            prime    replayed        5008       4596    0.90  6.1e-16  bluestein
       779  19.41            2p       replayed        1453       4671    3.06  4.7e-16  
       780  2^2.3.5.13       chain3   replayed         897       2869    3.17  7.1e-16  
       781  11.71            prime    replayed        4814       6290    1.28  5.4e-16  bluestein
       782  2.17.23          flat     replayed        1418       3557    2.47  2.0e-15  
       783  3^3.29           flat     replayed        1458       3648    2.20  4.8e-16  
       784  2^4.7^2          chain3   replayed         801       2305    2.83  3.9e-16  
       785  5.157            prime    replayed        4946       4603    0.87  7.0e-16  bluestein
       786  2.3.131          prime    replayed        4880       4861    0.94  6.0e-16  bluestein
       787  787              prime    replayed        4853       4603    0.94  5.7e-16  bluestein
       788  2^2.197          prime    replayed        4977       4656    0.91  6.3e-16  bluestein
       789  3.263            prime    replayed        4879       4605    0.91  6.9e-16  bluestein
       790  2.5.79           prime    replayed        4864       6515    1.33  6.3e-16  bluestein
       791  7.113            prime    replayed        4877       4603    0.88  6.1e-16  bluestein
       792  2^3.3^2.11       chain3   replayed         843       2891    3.38  5.8e-16  
       793  13.61            prime    replayed        4782       5618    1.17  5.1e-16  bluestein
       794  2.397            prime    replayed        4879       4743    0.92  7.6e-16  bluestein
       795  3.5.53           prime    replayed        4816       4441    0.88  5.7e-16  bluestein
       796  2^2.199          prime    replayed        4856       4612    0.88  5.8e-16  bluestein
       797  797              prime    replayed        4794       4622    0.95  5.4e-16  bluestein
       798  2.3.7.19         flat     replayed        1306       2818    2.13  5.3e-16  
       799  17.47            2p       replayed        1645       5109    2.88  5.6e-16  
       800  2^5.5^2          2p       replayed         764       2430    3.15  3.1e-16  
       801  3^2.89           prime    replayed        4884       7122    1.44  7.2e-16  bluestein
       802  2.401            prime    replayed        4791       4601    0.96  4.9e-16  bluestein
       803  11.73            prime    replayed        4827       6497    1.33  5.0e-16  bluestein
       804  2^2.3.67         prime    replayed        4898       6876    1.39  4.1e-16  bluestein
       805  5.7.23           flat     replayed        1335       2429    1.80  6.6e-16  
       806  2.13.31          flat     replayed        1606       3814    2.37  1.1e-15  
       807  3.269            prime    replayed        5132       4599    0.88  5.0e-16  bluestein
       808  2^3.101          prime    replayed        4911       4602    0.94  6.0e-16  bluestein
       809  809              prime    replayed        4889       4625    0.86  4.9e-16  bluestein
       810  2.3^4.5          chain3   replayed        1079       2373    2.15  4.5e-16  
       811  811              prime    replayed        4852       4595    0.94  5.2e-16  bluestein
       812  2^2.7.29         flat     replayed        1359       3883    2.48  7.3e-16  
       813  3.271            prime    replayed        4823       4588    0.95  5.6e-16  bluestein
       814  2.11.37          flat     replayed        1738       4072    2.11  8.1e-16  
       815  5.163            prime    replayed        4886       4601    0.73  6.5e-16  flips differ 1.30x bluestein
       816  2^4.3.17         chain3   replayed        1248       3300    2.54  4.8e-16  
       817  19.43            flat     replayed        1723       5121    2.95  8.9e-16  
       818  2.409            prime    replayed        4923       4596    0.77  5.4e-16  bluestein
       819  3^2.7.13         flat     replayed        1121       2329    2.07  7.7e-16  
       820  2^2.5.41         flat     replayed        1453       4847    3.30  4.8e-16  
       821  821              prime    replayed        4889       4604    0.94  5.5e-16  bluestein
       822  2.3.137          prime    replayed        4843       4624    0.94  5.4e-16  bluestein
       823  823              prime    replayed        4993       4600    0.91  7.7e-16  bluestein
       824  2^3.103          prime    replayed        4864       5034    0.94  5.9e-16  bluestein
       825  3.5^2.11         chain3   replayed        1216       2459    1.66  4.4e-16  
       826  2.7.59           prime    replayed        4827       5127    1.04  5.2e-16  bluestein
       827  827              prime    replayed        4778       4602    0.73  6.2e-16  flips differ 1.32x bluestein
       828  2^2.3^2.23       chain3   replayed        1275       3930    2.93  4.8e-16  
       829  829              prime    replayed        4552       4668    1.02  1.1e-15  rader
       830  2.5.83           prime    replayed        4842       7065    1.45  6.7e-16  bluestein
       831  3.277            prime    replayed        4853       4605    0.94  7.0e-16  bluestein
       832  2^6.13           2p       replayed         811       3188    3.88  5.0e-16  
       833  7^2.17           flat     replayed        1211       1925    1.55  7.7e-16  
       834  2.3.139          prime    replayed        4867       4698    0.95  6.2e-16  bluestein
       835  5.167            prime    replayed        4874       4612    0.92  6.9e-16  bluestein
       836  2^2.11.19        flat     replayed        1319       3511    2.58  8.8e-16  
       837  3^3.31           flat     replayed        2074       4222    2.02  7.1e-16  
       838  2.419            prime    replayed        4847       4638    0.94  5.3e-16  bluestein
       839  839              prime    replayed        4798       4642    0.96  7.2e-16  bluestein
       840  2^3.3.5.7        chain3   replayed         890       2490    2.79  4.0e-16  
       841  29^2             flat     replayed        2042       4836    2.30  1.2e-15  
       842  2.421            prime    replayed        4903       4727    0.95  5.7e-16  bluestein
       843  3.281            prime    replayed        4845       4631    0.95  4.8e-16  bluestein
       844  2^2.211          prime    replayed        4847       5024    0.84  5.2e-16  flips differ 1.33x bluestein
       845  5.13^2           flat     replayed        1272       2514    1.91  9.8e-16  
       846  2.3^2.47         flat     replayed        2453       5591    2.23  6.3e-16  
       847  7.11^2           flat     replayed        1323       2363    1.74  4.3e-16  
       848  2^4.53           prime    replayed        4904       6216    0.98  4.1e-16  flips differ 1.29x bluestein
       849  3.283            prime    replayed        4780       4701    0.93  5.0e-16  bluestein
       850  2.5^2.17         chain3   replayed        1439       2491    1.65  5.1e-16  
       851  23.37            2p       replayed        1694       5254    2.69  6.1e-16  
       852  2^2.3.71         prime    replayed        4981       7825    1.54  5.0e-16  bluestein
       853  853              prime    replayed        4953       4666    0.93  6.7e-16  bluestein
       854  2.7.61           prime    replayed        4918       5462    1.07  4.3e-16  bluestein
       855  3^2.5.19         flat     replayed        1287       3301    2.51  8.2e-16  
       856  2^3.107          prime    replayed        4889       4641    0.94  6.6e-16  bluestein
       857  857              prime    replayed        4931       4647    0.94  6.4e-16  bluestein
       858  2.3.11.13        flat     replayed        1411       3121    2.19  7.8e-16  
       859  859              prime    replayed        3732       4664    1.18  9.7e-16  rader
       860  2^2.5.43         flat     replayed        1653       5368    3.24  7.6e-16  
       861  3.7.41           2p       replayed        1419       3786    2.64  4.6e-16  
       862  2.431            prime    replayed        4818       4734    0.86  7.2e-16  bluestein
       863  863              prime    replayed        4927       4636    0.94  5.8e-16  bluestein
       864  2^5.3^3          2p       replayed         811       2969    3.65  5.8e-16  
       865  5.173            prime    replayed        4883       4672    0.94  5.6e-16  bluestein
       866  2.433            prime    replayed        4943       4646    0.90  5.7e-16  bluestein
       867  3.17^2           flat     replayed        1508       3271    2.10  1.1e-15  
       868  2^2.7.31         flat     replayed        1536       4521    2.89  5.9e-16  
       869  11.79            prime    replayed        4876       7621    1.56  4.6e-16  bluestein
       870  2.3.5.29         flat     replayed        1815       4164    2.27  5.6e-16  
       871  13.67            prime    replayed        4884       6705    1.37  6.4e-16  bluestein
       872  2^3.109          prime    replayed        4862       5015    0.90  6.4e-16  bluestein
       873  3^2.97           prime    replayed        4864       8370    1.70  6.1e-16  bluestein
       874  2.19.23          flat     replayed        1706       4294    2.48  1.1e-15  
       875  5^3.7            flat     replayed        1147       1951    1.27  4.5e-16  flips differ 1.32x
       876  2^2.3.73         prime    replayed        4867       7919    1.60  3.9e-16  bluestein
       877  877              prime    replayed        4829       4655    0.95  4.3e-16  bluestein
       878  2.439            prime    replayed        4888       5012    0.95  5.9e-16  bluestein
       879  3.293            prime    replayed        4927       4675    0.93  5.3e-16  bluestein
       880  2^4.5.11         chain3   replayed        1024       3236    2.84  3.2e-16  
       881  881              prime    replayed        3439       4656    1.34  5.0e-16  rader
       882  2.3^2.7^2        flat     replayed        1344       2557    1.87  5.4e-16  
       883  883              prime    replayed        4921       4658    0.91  7.0e-16  bluestein
       884  2^2.13.17        chain3   replayed        1518       3596    2.08  6.3e-16  
       885  3.5.59           prime    replayed        5100       5490    1.04  5.1e-16  bluestein
       886  2.443            prime    replayed        4938       5592    0.94  7.0e-16  bluestein
       887  887              prime    replayed        4841       4868    0.96  5.7e-16  bluestein
       888  2^3.3.37         chain3   replayed        1604       5197    3.23  4.6e-16  
       889  7.127            prime    replayed        4897       4683    0.95  5.2e-16  bluestein
       890  2.5.89           prime    replayed        4776       7976    1.33  5.6e-16  flips differ 1.25x bluestein
       891  3^4.11           chain3   replayed        1278       2840    2.12  6.0e-16  
       892  2^2.223          prime    replayed        4812       4650    0.95  4.6e-16  bluestein
       893  19.47            flat     replayed        1930       6020    3.09  9.9e-16  
       894  2.3.149          prime    replayed        4913       4863    0.95  7.8e-16  bluestein
       895  5.179            prime    replayed        4860       4671    0.73  5.4e-16  flips differ 1.32x bluestein
       896  2^7.7            chain3   replayed         934       2890    3.08  4.0e-16  
       897  3.13.23          flat     replayed        1644       3616    2.19  8.4e-16  
       898  2.449            prime    replayed        4932       4678    0.94  5.6e-16  bluestein
       899  29.31            2p       replayed        1940       5466    1.77  5.6e-16  flips differ 1.59x
       900  2^2.3^2.5^2      chain3   replayed         979       2732    2.73  3.7e-16  
       901  17.53            prime    replayed        4959       6133    1.23  4.8e-16  bluestein
       902  2.11.41          flat     replayed        2131       4893    2.17  8.6e-16  
       903  3.7.43           flat     replayed        1957       4293    2.16  1.9e-15  
       904  2^3.113          prime    replayed        4945       5548    0.91  5.8e-16  bluestein
       905  5.181            prime    replayed        4858       4674    0.95  4.8e-16  bluestein
       906  2.3.151          prime    replayed        4830       4689    0.97  5.0e-16  bluestein
       907  907              prime    replayed        5045       4786    0.91  7.8e-16  bluestein
       908  2^2.227          prime    replayed        4845       5051    0.96  5.2e-16  bluestein
       909  3^2.101          prime    replayed        5060       4679    0.92  6.3e-16  bluestein
       910  2.5.7.13         chain3   replayed        1233       2565    2.05  5.9e-16  
       911  911              prime    replayed        4929       4685    0.94  6.2e-16  bluestein
       912  2^4.3.19         chain3   replayed        1201       4010    3.28  4.7e-16  
       913  11.83            prime    replayed        4858       8291    1.68  4.6e-16  bluestein
       914  2.457            prime    replayed        4815       4687    0.97  8.1e-16  bluestein
       915  3.5.61           prime    replayed        4854       5843    1.19  6.6e-16  bluestein
       916  2^2.229          prime    replayed        4989       4683    0.87  6.1e-16  bluestein
       917  7.131            prime    replayed        4917       4706    0.94  8.5e-16  bluestein
       918  2.3^3.17         flat     replayed        1539       3605    2.21  6.9e-16  
       919  919              prime    replayed        4907       4708    0.95  6.0e-16  bluestein
       920  2^3.5.23         chain3   replayed        1404       4377    3.04  4.0e-16  
       921  3.307            prime    replayed        4864       4705    0.96  6.8e-16  bluestein
       922  2.461            prime    replayed        4970       4726    0.92  7.1e-16  bluestein
       923  13.71            prime    replayed        4917       7557    1.52  4.9e-16  bluestein
       924  2^2.3.7.11       chain3   replayed        1008       3359    3.28  4.6e-16  
       925  5^2.37           2p       replayed        2289       3815    1.11  4.4e-16  flips differ 1.50x
       926  2.463            prime    replayed        5098       4711    0.91  5.3e-16  bluestein
       927  3^2.103          prime    replayed        4908       4707    0.93  5.8e-16  bluestein
       928  2^5.29           2p       replayed        1171       4849    3.84  4.5e-16  
       929  929              prime    replayed        4809       4711    0.91  5.3e-16  rader
       930  2.3.5.31         chain3   replayed        2687       4804    1.43  5.1e-16  
       931  7^2.19           flat     replayed        1480       2448    1.65  7.7e-16  
       932  2^2.233          prime    replayed        4907       4704    0.95  7.0e-16  bluestein
       933  3.311            prime    replayed        4901       4705    0.94  6.2e-16  bluestein
       934  2.467            prime    replayed        5038       4766    0.93  6.0e-16  bluestein
       935  5.11.17          flat     replayed        1421       3019    2.10  6.2e-16  
       936  2^3.3^2.13       chain3   replayed        1071       3557    3.29  6.9e-16  
       937  937              prime    replayed        3675       4726    1.28  7.3e-16  rader
       938  2.7.67           prime    replayed        5024       6652    1.27  7.5e-16  bluestein
       939  3.313            prime    replayed        4922       4708    0.95  4.4e-16  bluestein
       940  2^2.5.47         flat     replayed        1810       6338    3.32  5.2e-16  
       941  941              prime    replayed        4952       4825    0.95  7.1e-16  bluestein
       942  2.3.157          prime    replayed        4912       4709    0.96  5.6e-16  bluestein
       943  23.41            2p       replayed        2046       6035    2.83  5.5e-16  
       944  2^4.59           prime    replayed        4894       7572    1.53  5.8e-16  bluestein
       945  3^3.5.7          chain3   replayed        1364       2515    1.77  6.1e-16  
       946  2.11.43          flat     replayed        2305       5374    2.23  5.6e-16  
       947  947              prime    replayed        5039       4718    0.93  6.6e-16  bluestein
       948  2^2.3.79         prime    replayed        5098       9308    1.82  5.1e-16  bluestein
       949  13.73            prime    replayed        4865       7756    1.59  6.9e-16  bluestein
       950  2.5^2.19         flat     replayed        1664       3254    1.82  5.4e-16  
       951  3.317            prime    replayed        4887       4722    0.96  8.4e-16  bluestein
       952  2^3.7.17         flat     replayed        1286       3920    2.59  7.6e-16  
       953  953              prime    replayed        4223       4786    1.12  8.2e-16  rader
       954  2.3^2.53         prime    replayed        5086       6752    1.26  5.9e-16  bluestein
       955  5.191            prime    replayed        4981       4733    0.80  6.3e-16  bluestein
       956  2^2.239          prime    replayed        4860       4737    0.96  5.7e-16  bluestein
       957  3.11.29          chain3   replayed        1861       4143    2.22  5.4e-16  
       958  2.479            prime    replayed        5124       4728    0.76  6.3e-16  bluestein
       959  7.137            prime    replayed        4948       4720    0.93  7.8e-16  bluestein
       960  2^6.3.5          2p       replayed         938       3124    3.28  4.4e-16  
       961  31^2             flat     replayed        2336       6204    2.48  9.3e-16  
       962  2.13.37          flat     replayed        2160       5010    2.30  9.1e-16  
       963  3^2.107          prime    replayed        4816       4730    0.95  7.5e-16  bluestein
       964  2^2.241          prime    replayed        5082       4742    0.93  6.3e-16  bluestein
       965  5.193            prime    replayed        5025       5077    0.94  7.4e-16  bluestein
       966  2.3.7.23         chain3   replayed        1577       3917    2.42  5.2e-16  
       967  967              prime    replayed        4926       4735    0.96  5.2e-16  bluestein
       968  2^3.11^2         chain3   replayed        1254       4365    3.38  4.3e-16  
       969  3.17.19          flat     replayed        1810       4012    2.17  8.8e-16  
       970  2.5.97           prime    replayed        5062       9381    1.84  5.5e-16  bluestein
       971  971              prime    replayed        5042       5099    0.92  6.3e-16  bluestein
       972  2^2.3^5          chain3   replayed        1029       3307    3.14  5.4e-16  
       973  7.139            prime    replayed        4920       4822    0.93  5.9e-16  bluestein
       974  2.487            prime    replayed        4884       4743    0.95  6.5e-16  bluestein
       975  3.5^2.13         flat     replayed        1343       3129    2.32  6.2e-16  
       976  2^4.61           prime    replayed        4875       7903    1.32  5.2e-16  bluestein
       977  977              prime    replayed        4939       4753    0.93  6.5e-16  bluestein
       978  2.3.163          prime    replayed        4848       4715    0.74  6.2e-16  flips differ 1.31x bluestein
       979  11.89            prime    replayed        4954       9300    1.80  6.0e-16  bluestein
       980  2^2.5.7^2        flat     replayed        1234       2818    2.26  4.8e-16  
       981  3^2.109          prime    replayed        5058       4783    0.91  6.4e-16  bluestein
       982  2.491            prime    replayed        4932       4757    0.96  5.2e-16  bluestein
       983  983              prime    replayed        4976       4728    0.95  7.3e-16  bluestein
       984  2^3.3.41         chain3   replayed        1892       6033    3.15  5.2e-16  
       985  5.197            prime    replayed        4983       4746    0.93  7.0e-16  bluestein
       986  2.17.29          flat     replayed        2022       4953    2.42  6.8e-16  
       987  3.7.47           2p       replayed        1828       5180    2.47  5.1e-16  
       988  2^2.13.19        chain3   replayed        1794       4360    2.12  6.4e-16  
       989  23.43            flat     replayed        2248       6639    2.84  1.1e-15  
       990  2.3^2.5.11       chain3   replayed        1238       3470    2.80  6.0e-16  
       991  991              prime    replayed        4892       4748    0.95  6.5e-16  bluestein
       992  2^5.31           flat     replayed        1801       5527    3.04  1.4e-15  
       993  3.331            prime    replayed        4901       4736    0.96  6.3e-16  bluestein
       994  2.7.71           prime    replayed        4892       7542    1.49  4.7e-16  bluestein
       995  5.199            prime    replayed        4856       5127    0.95  7.1e-16  bluestein
       996  2^2.3.83         prime    replayed        4924      10110    2.02  5.6e-16  bluestein
       997  997              prime    replayed        4938       5612    0.96  6.5e-16  bluestein
       998  2.499            prime    replayed        4862       5112    0.97  5.1e-16  bluestein
       999  3^3.37           2p       replayed        3493       5412    1.54  6.4e-16  
      1000  2^3.5^3          chain3   replayed        1051       3043    2.87  3.7e-16  
      1001  7.11.13          flat     replayed        1514       2921    1.82  7.1e-16  
      1002  2.3.167          prime    replayed        5039       4761    0.93  7.1e-16  bluestein
      1003  17.59            prime    replayed        4849       7514    1.53  4.5e-16  bluestein
      1004  2^2.251          prime    replayed        4852       4754    0.98  5.8e-16  bluestein
      1005  3.5.67           prime    replayed        4924       7152    1.22  5.5e-16  bluestein
      1006  2.503            prime    replayed        4878       4751    0.96  5.9e-16  bluestein
      1007  19.53            prime    replayed        4949       7195    1.44  6.4e-16  bluestein
      1008  2^4.3^2.7        chain3   replayed        1047       3203    3.03  5.5e-16  
      1009  1009             prime    replayed        5021       4782    0.94  7.9e-16  bluestein
      1010  2.5.101          prime    replayed        4928       4979    0.97  5.0e-16  bluestein
      1011  3.337            prime    replayed        4952       4778    0.96  6.8e-16  bluestein
      1012  2^2.11.23        chain3   replayed        1685       4796    2.81  5.1e-16  
      1013  1013             prime    replayed        5042       4774    0.88  6.8e-16  bluestein
      1014  2.3.13^2         chain3   replayed        1531       3892    2.11  6.3e-16  
      1015  5.7.29           flat     replayed        1865       3497    1.64  5.8e-16  
      1016  2^3.127          prime    replayed        4829       4847    0.98  5.0e-16  bluestein
      1017  3^2.113          prime    replayed        4984       4792    0.95  6.7e-16  bluestein
      1018  2.509            prime    replayed        4913       4773    0.97  5.0e-16  bluestein
      1019  1019             prime    replayed        5067       4799    0.93  7.2e-16  bluestein
      1020  2^2.3.5.17       chain3   replayed        1516       4203    2.74  4.7e-16  
      1021  1021             prime    replayed        5049       4773    0.94  6.6e-16  bluestein
      1022  2.7.73           prime    replayed        4964       7763    1.20  6.6e-16  flips differ 1.30x bluestein
      1023  3.11.31          chain3   replayed        2005       4860    2.25  6.3e-16  
      1024  2^10             ztt      replayed         719        928    1.11  4.0e-16  
      1025  5^2.41           2p       replayed        1754       4564    2.60  5.0e-16  
      1026  2.3^3.19         chain3   replayed        1605       4406    2.72  5.1e-16  
      1027  13.79            prime    replayed       10091       9146    0.89  3.9e-16  bluestein
      1028  2^2.257          prime    raced          10142      10521    1.03  6.2e-16  bluestein
      1029  3.7^3            flat     replayed        1489       2663    1.70  4.4e-16  
      1030  2.5.103          prime    raced          10086      10508    1.02  4.2e-16  bluestein
      1031  1031             prime    replayed       10091      10541    1.03  6.7e-16  bluestein
      1032  2^3.3.43         chain3   replayed        2064       6782    3.23  4.4e-16  
      1033  1033             prime    replayed        7258      10543    1.24  8.2e-16  rader
      1034  2.11.47          flat     replayed        2665       6413    2.33  8.6e-16  
      1035  3^2.5.23         flat     replayed        1669       4552    2.29  6.7e-16  
      1036  2^2.7.37         flat     replayed        1900       5972    3.13  6.6e-16  
      1037  17.61            prime    replayed       10312       7863    0.76  4.5e-16  bluestein
      1038  2.3.173          prime    raced          10353      10494    1.01  5.7e-16  bluestein
      1039  1039             prime    replayed       10009      10554    0.77  6.2e-16  flips differ 1.37x bluestein
      1040  2^4.5.13         chain3   replayed        1199       4036    3.25  4.6e-16  
      1041  3.347            prime    replayed       10298      10504    1.02  6.2e-16  bluestein
      1042  2.521            prime    raced          10256      10522    1.02  5.2e-16  bluestein
      1043  7.149            prime    replayed       10186      10637    1.02  5.0e-16  bluestein
      1044  2^2.3^2.29       chain3   replayed        1775       5546    3.09  5.9e-16  
      1045  5.11.19          flat     replayed        1721       3838    2.10  6.3e-16  
      1046  2.523            prime    raced          10300      10486    1.02  6.8e-16  bluestein
      1047  3.349            prime    replayed       10277      10485    1.02  6.2e-16  bluestein
      1048  2^3.131          prime    raced          10064      10526    1.04  5.2e-16  bluestein
      1049  1049             prime    replayed       10417      10779    0.78  6.2e-16  flips differ 1.36x bluestein
      1050  2.3.5^2.7        chain3   replayed        1625       3082    1.89  4.2e-16  
      1051  1051             prime    replayed        5212      10509    1.99  7.9e-16  rader
      1052  2^2.263          prime    raced          10130      10599    1.01  5.6e-16  bluestein
      1053  3^4.13           chain3   replayed        1713       3639    2.10  8.0e-16  
      1054  2.17.31          flat     replayed        2313       5674    2.41  1.7e-15  
      1055  5.211            prime    replayed       10177      10520    1.02  6.5e-16  bluestein
      1056  2^5.3.11         chain3   replayed        1260       4053    3.17  4.8e-16  
      1057  7.151            prime    replayed       10122      10561    1.04  6.0e-16  bluestein
      1058  2.23^2           flat     replayed        2231       5781    2.52  1.3e-15  
      1059  3.353            prime    replayed       10054      10663    1.04  4.9e-16  bluestein
      1060  2^2.5.53         prime    raced          10182       7650    0.74  5.2e-16  bluestein
      1061  1061             prime    replayed       10235      10514    1.02  6.4e-16  bluestein
      1062  2.3^2.59         prime    raced          10211       8203    0.80  5.3e-16  bluestein
      1063  1063             prime    replayed       10128      10507    1.03  5.4e-16  bluestein
      1064  2^3.7.19         chain3   replayed        1450       4683    3.08  3.6e-16  
      1065  3.5.71           prime    replayed       10201       8110    0.79  5.0e-16  bluestein
      1066  2.13.41          flat     replayed        2597       5924    2.24  8.6e-16  
      1067  11.97            prime    replayed       10324      10949    1.06  5.6e-16  bluestein
      1068  2^2.3.89         prime    raced          10214      11321    1.10  4.1e-16  bluestein
      1069  1069             prime    replayed       10263      10522    0.89  5.2e-16  bluestein
      1070  2.5.107          prime    raced          10138      10661    1.04  4.4e-16  bluestein
      1071  3^2.7.17         flat     replayed        1714       3240    1.87  5.7e-16  
      1072  2^4.67           prime    raced          10101       9411    0.82  3.9e-16  bluestein
      1073  29.37            2p       replayed        3188       6969    2.18  4.9e-16  
      1074  2.3.179          prime    raced          10322      11051    1.02  5.4e-16  bluestein
      1075  5^2.43           2p       replayed        1896       5166    2.21  5.6e-16  
      1076  2^2.269          prime    raced          10113      10605    1.02  5.0e-16  bluestein
      1077  3.359            prime    replayed       10293      10613    1.02  4.5e-16  bluestein
      1078  2.7^2.11         flat     replayed        1840       2797    1.48  6.1e-16  
      1079  13.83            prime    replayed       10123       9904    0.97  4.4e-16  bluestein
      1080  2^3.3^3.5        chain3   replayed        1163       3551    3.05  5.2e-16  
      1081  23.47            2p       replayed        3671       7790    2.11  5.5e-16  
      1082  2.541            prime    raced          10118      10565    1.01  5.3e-16  bluestein
      1083  3.19^2           chain3   replayed        2114       5017    2.29  4.7e-16  
      1084  2^2.271          prime    raced          10036      10569    1.04  5.8e-16  bluestein
      1085  5.7.31           chain3   replayed        3952       4140    0.97  5.5e-16  
      1086  2.3.181          prime    raced          10211      10562    1.03  6.4e-16  bluestein
      1087  1087             prime    replayed       10103      10537    1.04  6.6e-16  bluestein
      1088  2^6.17           2p       replayed        1324       4696    1.90  6.2e-16  flips differ 1.89x
      1089  3^2.11^2         chain3   replayed        1653       3830    2.31  6.2e-16  
      1090  2.5.109          prime    raced          10335      10581    1.01  4.2e-16  bluestein
      1091  1091             prime    replayed       10254      10558    1.02  6.4e-16  bluestein
      1092  2^2.3.7.13       chain3   replayed        1266       4194    3.27  6.1e-16  
      1093  1093             prime    replayed        5824      10547    1.73  9.5e-16  rader
      1094  2.547            prime    raced          10063      10560    1.03  7.1e-16  bluestein
      1095  3.5.73           prime    replayed       10083       8323    0.82  7.4e-16  bluestein
      1096  2^3.137          prime    raced          10212      10559    0.90  5.0e-16  bluestein
      1097  1097             prime    replayed       10455      10591    0.91  5.3e-16  bluestein
      1098  2.3^2.61         prime    raced          10138       8718    0.84  5.4e-16  bluestein
      1099  7.157            prime    replayed       10133      10550    1.04  7.4e-16  bluestein
      1100  2^2.5^2.11       chain3   replayed        1309       4019    3.01  3.5e-16  
      1101  3.367            prime    replayed       10240      10549    1.03  5.9e-16  bluestein
      1102  2.19.29          flat     replayed        2534       5948    2.25  1.2e-15  
      1103  1103             prime    replayed       10331      10558    1.01  4.4e-16  bluestein
      1104  2^4.3.23         chain3   replayed        1572       5434    3.42  5.8e-16  
      1105  5.13.17          flat     replayed        1750       3783    2.11  8.0e-16  
      1106  2.7.79           prime    raced          10134       9221    0.89  6.2e-16  bluestein
      1107  3^3.41           2p       replayed        1902       6410    3.30  5.2e-16  
      1108  2^2.277          prime    raced          10358      11060    1.01  4.4e-16  bluestein
      1109  1109             prime    replayed       10141      10548    1.02  6.9e-16  bluestein
      1110  2.3.5.37         chain3   replayed        2195       6407    2.89  5.4e-16  
      1111  11.101           prime    replayed       10078      10575    1.04  7.3e-16  bluestein
      1112  2^3.139          prime    raced          10052      10541    1.02  6.1e-16  bluestein
      1113  3.7.53           prime    replayed       10117       6359    0.61  3.9e-16  bluestein
      1114  2.557            prime    raced          10064      10601    1.04  5.9e-16  bluestein
      1115  5.223            prime    replayed       10333      10566    1.02  6.0e-16  bluestein
      1116  2^2.3^2.31       flat     replayed        2112       6215    2.93  6.7e-16  
      1117  1117             prime    replayed        7539      10552    1.39  1.0e-15  rader
      1118  2.13.43          flat     replayed        2862       6573    2.29  8.3e-16  
      1119  3.373            prime    replayed       10088      10562    1.04  6.2e-16  bluestein
      1120  2^5.5.7          chain3   replayed        1156       3578    3.03  3.5e-16  
      1121  19.59            prime    replayed       10312       8707    0.84  5.1e-16  bluestein
      1122  2.3.11.17        chain3   replayed        1867       4565    2.39  5.8e-16  
      1123  1123             prime    replayed        5384      10625    1.63  8.6e-16  rader
      1124  2^2.281          prime    raced          10150      10571    0.85  6.1e-16  bluestein
      1125  3^2.5^3          flat     replayed        1566       3249    2.05  7.8e-16  
      1126  2.563            prime    raced          10149      10558    1.03  6.2e-16  bluestein
      1127  7^2.23           chain3   replayed        2065       3522    1.69  6.4e-16  
      1128  2^3.3.47         flat     replayed        2727       7907    2.84  6.3e-16  
      1129  1129             prime    replayed       10237      10548    1.01  7.1e-16  bluestein
      1130  2.5.113          prime    raced          10242      10537    1.02  4.3e-16  bluestein
      1131  3.13.29          chain3   replayed        2195       5150    2.33  7.4e-16  
      1132  2^2.283          prime    raced          10134      10576    1.04  6.3e-16  bluestein
      1133  11.103           prime    replayed       10137      10566    1.02  5.7e-16  bluestein
      1134  2.3^4.7          chain3   replayed        1702       3607    2.04  5.4e-16  
      1135  5.227            prime    replayed       10096      10531    1.04  5.9e-16  bluestein
      1136  2^4.71           prime    raced          10152      10463    1.01  4.3e-16  bluestein
      1137  3.379            prime    replayed       10126      10563    1.04  6.3e-16  bluestein
      1138  2.569            prime    raced          10143      10700    1.04  5.5e-16  bluestein
      1139  17.67            prime    replayed       10200       9445    0.92  5.4e-16  bluestein
      1140  2^2.3.5.19       chain3   replayed        1552       5120    3.14  4.7e-16  
      1141  7.163            prime    replayed       10304      10565    1.02  5.1e-16  bluestein
      1142  2.571            prime    raced          10136      10569    1.04  4.9e-16  bluestein
      1143  3^2.127          prime    replayed       10153      10562    1.01  5.6e-16  bluestein
      1144  2^3.11.13        chain3   replayed        1515       5319    3.46  5.9e-16  
      1145  5.229            prime    replayed       10274      10543    1.00  5.1e-16  bluestein
      1146  2.3.191          prime    raced          10406      11483    1.02  7.1e-16  bluestein
      1147  31.37            flat     replayed        2576       7817    3.02  8.4e-16  
      1148  2^2.7.41         chain3   replayed        2238       6957    3.06  4.1e-16  
      1149  3.383            prime    replayed       10345      10587    1.02  6.5e-16  bluestein
      1150  2.5^2.23         chain3   replayed        1879       4654    2.37  4.6e-16  
      1151  1151             prime    replayed        6351      10610    1.45  8.2e-16  rader
      1152  2^7.3^2          chain3   replayed        1298       4106    2.92  4.9e-16  
      1153  1153             prime    replayed        4208      10619    2.49  9.8e-16  rader
      1154  2.577            prime    raced          10131      10559    1.02  4.9e-16  bluestein
      1155  3.5.7.11         chain3   replayed        1705       3303    1.93  4.2e-16  
      1156  2^2.17^2         chain3   replayed        2081       5340    2.40  6.7e-16  
      1157  13.89            prime    replayed       10351      11179    0.90  4.3e-16  bluestein
      1158  2.3.193          prime    raced          10142      10567    1.02  4.9e-16  bluestein
      1159  19.61            prime    replayed       11732       9149    0.77  5.1e-16  bluestein
      1160  2^3.5.29         flat     replayed        2124       6048    2.82  1.1e-15  
      1161  3^3.43           2p       replayed        2216       7102    3.08  5.5e-16  
      1162  2.7.83           prime    raced          10153      10027    0.99  4.6e-16  bluestein
      1163  1163             prime    replayed       10474      10562    1.00  5.6e-16  bluestein
      1164  2^2.3.97         prime    raced          10252      13315    1.29  5.4e-16  bluestein
      1165  5.233            prime    replayed       10342      10557    1.02  6.4e-16  bluestein
      1166  2.11.53          prime    raced          10356       7765    0.75  4.0e-16  bluestein
      1167  3.389            prime    replayed       10361      10577    1.02  5.7e-16  bluestein
      1168  2^4.73           prime    raced          10124      10793    1.04  5.0e-16  bluestein
      1169  7.167            prime    replayed       10099      10653    1.03  6.0e-16  bluestein
      1170  2.3^2.5.13       chain3   replayed        1502       4375    2.88  6.6e-16  
      1171  1171             prime    replayed        4786      10556    2.17  8.2e-16  rader
      1172  2^2.293          prime    raced          10398      10577    1.01  6.0e-16  bluestein
      1173  3.17.23          flat     replayed        2293       5472    2.29  1.1e-15  
      1174  2.587            prime    raced          10109      10587    1.02  6.8e-16  bluestein
      1175  5^2.47           flat     replayed        2468       6280    2.47  4.6e-16  
      1176  2^3.3.7^2        chain3   replayed        1506       3739    1.90  3.9e-16  flips differ 1.31x
      1177  11.107           prime    replayed       10242      10581    1.02  5.6e-16  bluestein
      1178  2.19.31          flat     replayed        2712       6753    2.43  1.2e-15  
      1179  3^2.131          prime    replayed       10232      10570    1.02  4.8e-16  bluestein
      1180  2^2.5.59         prime    raced          10270       9377    0.91  5.2e-16  bluestein
      1181  1181             prime    replayed       10377      10658    1.02  4.9e-16  bluestein
      1182  2.3.197          prime    raced          10148      10589    1.04  6.7e-16  bluestein
      1183  7.13^2           chain3   replayed        1910       3683    1.58  7.3e-16  
      1184  2^5.37           2p       replayed        2797       7135    1.76  4.6e-16  flips differ 1.45x
      1185  3.5.79           prime    replayed       10321       9850    0.95  5.9e-16  bluestein
      1186  2.593            prime    raced          10366      10648    1.02  4.5e-16  bluestein
      1187  1187             prime    replayed       10229      10637    1.02  5.7e-16  bluestein
      1188  2^2.3^3.11       chain3   replayed        1376       4554    3.25  5.5e-16  
      1189  29.41            2p       replayed        3133       8054    2.12  5.0e-16  
      1190  2.5.7.17         chain3   replayed        1747       3721    2.08  7.9e-16  
      1191  3.397            prime    replayed       10373      10671    1.02  6.7e-16  bluestein
      1192  2^3.149          prime    raced          10102      10565    1.03  4.5e-16  bluestein
      1193  1193             prime    replayed       10279      10578    1.02  6.0e-16  bluestein
      1194  2.3.199          prime    raced          10282      10586    1.02  5.5e-16  bluestein
      1195  5.239            prime    replayed       10183      10640    1.02  5.6e-16  bluestein
      1196  2^2.13.23        chain3   replayed        2077       5910    2.79  5.6e-16  
      1197  3^2.7.19         flat     replayed        2196       4121    1.84  1.1e-15  
      1198  2.599            prime    raced          10350      10595    1.01  5.4e-16  bluestein
      1199  11.109           prime    replayed       10156      10576    1.01  5.2e-16  bluestein
      1200  2^4.3.5^2        chain3   replayed        1699       3887    2.10  4.5e-16  
      1201  1201             prime    replayed        6711      10579    1.47  8.0e-16  rader
      1202  2.601            prime    raced          10299      10630    1.02  5.3e-16  bluestein
      1203  3.401            prime    replayed       10170      10560    1.02  4.7e-16  bluestein
      1204  2^2.7.43         flat     replayed        2449       7745    3.10  7.0e-16  
      1205  5.241            prime    replayed       10199      10591    1.01  6.1e-16  bluestein
      1206  2.3^2.67         prime    raced          10298      10362    1.00  5.5e-16  bluestein
      1207  17.71            prime    replayed       10106      10547    1.01  5.2e-16  bluestein
      1208  2^3.151          prime    raced          10145      10595    1.04  4.4e-16  bluestein
      1209  3.13.31          chain3   replayed        2460       5959    2.41  5.6e-16  
      1210  2.5.11^2         chain3   replayed        1650       4408    2.61  3.7e-16  
      1211  7.173            prime    replayed       10309      11661    0.86  6.4e-16  flips differ 1.39x bluestein
      1212  2^2.3.101        prime    raced          10364      11173    1.01  3.9e-16  bluestein
      1213  1213             prime    replayed       10167      10587    1.04  4.7e-16  bluestein
      1214  2.607            prime    raced          10259      10565    1.02  6.5e-16  bluestein
      1215  3^5.5            chain3   replayed        1764       3601    2.02  9.3e-16  
      1216  2^6.19           2p       replayed        1523       5624    3.67  3.4e-16  
      1217  1217             prime    replayed        4924      10626    2.08  7.5e-16  rader
      1218  2.3.7.29         flat     replayed        2688       5707    2.00  5.8e-16  
      1219  23.53            prime    replayed       10416       9260    0.86  6.0e-16  bluestein
      1220  2^2.5.61         prime    raced          10165       9788    0.96  4.1e-16  bluestein
      1221  3.11.37          chain3   replayed        2675       6403    2.31  4.9e-16  
      1222  2.13.47          flat     replayed        3293       7806    2.35  6.6e-16  
      1223  1223             prime    replayed       10186      10599    1.02  5.1e-16  bluestein
      1224  2^3.3^2.17       chain3   replayed        1558       5200    3.32  5.9e-16  
      1225  5^2.7^2          flat     replayed        1803       3174    1.70  4.5e-16  
      1226  2.613            prime    raced          10168      10589    1.01  6.6e-16  bluestein
      1227  3.409            prime    replayed       10382      10689    1.03  7.9e-16  bluestein
      1228  2^2.307          prime    raced          10173      10599    1.04  4.7e-16  bluestein
      1229  1229             prime    replayed       10200      10590    1.02  4.7e-16  bluestein
      1230  2.3.5.41         chain3   replayed        2509       7309    2.77  4.9e-16  
      1231  1231             prime    replayed       10230      10664    1.03  5.8e-16  bluestein
      1232  2^4.7.11         chain3   replayed        1430       4728    2.93  4.1e-16  
      1233  3^2.137          prime    replayed       10371      10587    1.02  6.1e-16  bluestein
      1234  2.617            prime    raced          10449      10575    1.00  6.9e-16  bluestein
      1235  5.13.19          flat     replayed        2094       4754    2.10  7.9e-16  
      1236  2^2.3.103        prime    raced          10169      10575    1.04  5.7e-16  bluestein
      1237  1237             prime    replayed       10241      10660    1.03  5.5e-16  bluestein
      1238  2.619            prime    raced          10161      10644    1.03  5.6e-16  bluestein
      1239  3.7.59           prime    replayed       10235       7923    0.77  7.0e-16  bluestein
      1240  2^3.5.31         chain3   replayed        2153       6982    3.22  5.0e-16  
      1241  17.73            prime    replayed       10407      10756    1.03  5.0e-16  bluestein
      1242  2.3^3.23         flat     replayed        2527       5973    2.36  8.1e-16  
      1243  11.113           prime    replayed       10182      10720    1.02  7.9e-16  bluestein
      1244  2^2.311          prime    raced          10275      10597    1.03  6.6e-16  bluestein
      1245  3.5.83           prime    replayed       10180      10720    1.05  4.6e-16  bluestein
      1246  2.7.89           prime    raced          10416      11313    1.08  4.7e-16  bluestein
      1247  29.43            flat     replayed        3416       8845    2.54  2.8e-15  
      1248  2^5.3.13         chain3   replayed        1379       5021    3.64  5.4e-16  
      1249  1249             prime    replayed        5424      10604    1.95  8.1e-16  rader
      1250  2.5^4            chain3   replayed        1933       3599    1.54  4.6e-16  
      1251  3^2.139          prime    replayed       10285      10608    1.03  5.7e-16  bluestein
      1252  2^2.313          prime    raced          10300      10592    1.03  5.0e-16  bluestein
      1253  7.179            prime    replayed       10269      10635    1.03  5.6e-16  bluestein
      1254  2.3.11.19        chain3   replayed        2616       5555    2.05  5.8e-16  
      1255  5.251            prime    replayed       10224      10631    1.03  5.4e-16  bluestein
      1256  2^3.157          prime    raced          10268      10591    1.02  5.7e-16  bluestein
      1257  3.419            prime    replayed       10481      10707    1.01  5.0e-16  bluestein
      1258  2.17.37          flat     replayed        3032       7420    2.27  1.2e-15  
      1259  1259             prime    replayed       10182      10585    1.02  5.9e-16  bluestein
      1260  2^2.3^2.5.7      chain3   replayed        1404       4008    2.83  6.3e-16  
      1261  13.97            prime    replayed       10023      13116    1.29  4.6e-16  bluestein
      1262  2.631            prime    raced          10277      10593    1.02  4.6e-16  bluestein
      1263  3.421            prime    replayed       10141      10630    1.04  6.6e-16  bluestein
      1264  2^4.79           prime    raced          10169      12604    1.21  5.5e-16  bluestein
      1265  5.11.23          flat     replayed        2437       5249    2.15  9.0e-16  
      1266  2.3.211          prime    raced          10136      10611    1.04  5.8e-16  bluestein
      1267  7.181            prime    replayed       10181      10588    1.00  6.0e-16  bluestein
      1268  2^2.317          prime    raced          10318      10610    1.03  6.8e-16  bluestein
      1269  3^3.47           flat     replayed        3101       8329    2.67  7.1e-16  
      1270  2.5.127          prime    raced          10470      10593    0.78  5.5e-16  flips differ 1.29x bluestein
      1271  31.41            flat     replayed        3002       9084    3.01  1.3e-15  
      1272  2^3.3.53         prime    raced          10149       9387    0.90  5.8e-16  bluestein
      1273  19.67            prime    replayed       10454      10909    1.03  5.6e-16  bluestein
      1274  2.7^2.13         flat     replayed        2281       3577    1.56  6.2e-16  
      1275  3.5^2.17         flat     replayed        2267       4670    2.03  7.2e-16  
      1276  2^2.11.29        chain3   replayed        2333       6751    2.85  3.9e-16  
      1277  1277             prime    replayed        7512      10643    1.21  7.9e-16  rader
      1278  2.3^2.71         prime    raced          10281      11622    1.12  4.6e-16  bluestein
      1279  1279             prime    replayed       10145      10695    1.02  7.3e-16  bluestein
      1280  2^8.5            chain3   replayed        1468       4404    2.98  4.3e-16  
      1281  3.7.61           prime    replayed       10169       8468    0.81  4.3e-16  bluestein
      1282  2.641            prime    raced          10272      10595    1.01  7.0e-16  bluestein
      1283  1283             prime    replayed       10322      10613    1.02  6.1e-16  bluestein
      1284  2^2.3.107        prime    raced          10092      11157    1.03  5.8e-16  bluestein
      1285  5.257            prime    replayed       10158      10561    1.03  6.5e-16  bluestein
      1286  2.643            prime    raced          10363      10609    0.74  5.7e-16  flips differ 1.38x bluestein
      1287  3^2.11.13        chain3   replayed        2067       4803    2.31  5.9e-16  
      1288  2^3.7.23         chain3   replayed        1888       6382    3.29  5.0e-16  
      1289  1289             prime    replayed        6570      10611    1.50  5.9e-16  rader
      1290  2.3.5.43         flat     replayed        3479       8182    2.26  6.8e-16  
      1291  1291             prime    replayed       10272      10624    1.03  6.2e-16  bluestein
      1292  2^2.17.19        chain3   replayed        2373       6548    2.71  5.8e-16  
      1293  3.431            prime    replayed       10176      10853    0.80  5.8e-16  flips differ 1.33x bluestein
      1294  2.647            prime    raced          10383      10624    1.02  7.5e-16  bluestein
      1295  5.7.37           flat     replayed        2494       5598    2.24  7.7e-16  
      1296  2^4.3^4          chain3   replayed        1387       4635    3.32  5.5e-16  
      1297  1297             prime    replayed        7793      10711    1.26  8.1e-16  rader
      1298  2.11.59          prime    raced          10189       9526    0.93  6.1e-16  bluestein
      1299  3.433            prime    replayed       10286      10720    1.03  5.0e-16  bluestein
      1300  2^2.5^2.13       flat     replayed        1729       4968    2.85  5.8e-16  
      1301  1301             prime    replayed        7486      10611    1.34  9.6e-16  rader
      1302  2.3.7.31         chain3   replayed        4616       6437    1.31  6.5e-16  
      1303  1303             prime    replayed       10383      10767    1.02  5.1e-16  bluestein
      1304  2^3.163          prime    raced          10160      10642    1.03  4.9e-16  bluestein
      1305  3^2.5.29         chain3   replayed        2569       6355    2.45  6.7e-16  
      1306  2.653            prime    raced          10402      10680    1.02  5.8e-16  bluestein
      1307  1307             prime    replayed       10228      10596    1.01  5.5e-16  bluestein
      1308  2^2.3.109        prime    raced          10201      10631    1.02  4.5e-16  bluestein
      1309  7.11.17          flat     replayed        2311       4433    1.92  1.1e-15  
      1310  2.5.131          prime    raced          10286      10640    1.03  5.7e-16  bluestein
      1311  3.19.23          chain3   replayed        3002       6662    2.21  6.0e-16  
      1312  2^5.41           2p       replayed        2047       8296    4.04  4.3e-16  
      1313  13.101           prime    replayed       10198      10612    1.02  6.1e-16  bluestein
      1314  2.3^2.73         prime    raced          10387      12007    1.14  5.3e-16  bluestein
      1315  5.263            prime    replayed       10383      10603    1.02  6.1e-16  bluestein
      1316  2^2.7.47         chain3   replayed        4649       9122    1.31  4.6e-16  flips differ 1.50x
      1317  3.439            prime    replayed       10204      10600    1.04  5.2e-16  bluestein
      1318  2.659            prime    raced          10166      10632    1.02  6.0e-16  bluestein
      1319  1319             prime    replayed       10301      10634    1.03  5.4e-16  bluestein
      1320  2^3.3.5.11       chain3   replayed        1611       5103    3.10  5.7e-16  
      1321  1321             prime    replayed        6027      10612    1.73  6.1e-16  rader
      1322  2.661            prime    raced          10124      10606    1.03  5.5e-16  bluestein
      1323  3^3.7^2          flat     replayed        2109       3764    1.24  8.3e-16  flips differ 1.44x
      1324  2^2.331          prime    raced          10175      10591    1.02  5.9e-16  bluestein
      1325  5^2.53           prime    replayed       10374       7654    0.73  4.9e-16  bluestein
      1326  2.3.13.17        chain3   replayed        2162       5634    2.46  6.9e-16  
      1327  1327             prime    replayed        6440      10644    1.57  1.0e-15  rader
      1328  2^4.83           prime    raced          10138      13685    1.34  4.6e-16  bluestein
      1329  3.443            prime    replayed       10197      10642    1.03  6.4e-16  bluestein
      1330  2.5.7.19         chain3   replayed        2010       4620    2.22  5.7e-16  
      1331  11^3             chain3   replayed        2131       5139    2.32  4.5e-16  
      1332  2^2.3^2.37       chain3   replayed        2418       8068    3.30  5.1e-16  
      1333  31.43            2p       replayed        4017       9947    1.97  6.7e-16  flips differ 1.26x
      1334  2.23.29          flat     replayed        3246       8088    2.43  1.3e-15  
      1335  3.5.89           prime    replayed       10282      12129    1.17  6.8e-16  bluestein
      1336  2^3.167          prime    raced          10314      10606    1.03  5.2e-16  bluestein
      1337  7.191            prime    replayed       10262      10637    1.03  4.6e-16  bluestein
      1338  2.3.223          prime    raced          10289      10689    1.03  5.7e-16  bluestein
      1339  13.103           prime    replayed       10210      10619    0.78  6.5e-16  flips differ 1.33x bluestein
      1340  2^2.5.67         prime    raced          10233      11741    1.11  5.4e-16  bluestein
      1341  3^2.149          prime    replayed       10424      10651    1.01  5.4e-16  bluestein
      1342  2.11.61          prime    raced          10291      10090    0.83  5.7e-16  bluestein
      1343  17.79            prime    replayed       10415      12673    1.21  5.5e-16  bluestein
      1344  2^6.3.7          2p       replayed        1396       4551    3.20  4.4e-16  
      1345  5.269            prime    replayed       10440      10626    1.02  5.1e-16  bluestein
      1346  2.673            prime    raced          10142      10616    1.03  7.9e-16  bluestein
      1347  3.449            prime    replayed       10213      11130    1.01  6.9e-16  bluestein
      1348  2^2.337          prime    raced          11770      10635    0.84  5.3e-16  bluestein
      1349  19.71            prime    replayed       10174      12204    1.17  6.5e-16  bluestein
      1350  2.3^3.5^2        chain3   replayed        1763       4235    2.37  5.5e-16  
      1351  7.193            prime    replayed       10415      10670    1.02  7.6e-16  bluestein
      1352  2^3.13^2         chain3   replayed        1844       6569    2.42  5.6e-16  flips differ 1.47x
      1353  3.11.41          chain3   replayed        3212       7616    2.34  4.9e-16  
      1354  2.677            prime    raced          10371      10767    1.03  5.5e-16  bluestein
      1355  5.271            prime    replayed       10433      10755    1.02  5.8e-16  bluestein
      1356  2^2.3.113        prime    raced          10215      10621    1.04  5.5e-16  bluestein
      1357  23.59            prime    replayed       10732      11128    1.03  4.9e-16  bluestein
      1358  2.7.97           prime    raced          10271      13323    1.26  5.0e-16  bluestein
      1359  3^2.151          prime    replayed       10481      10660    1.00  7.0e-16  bluestein
      1360  2^4.5.17         chain3   replayed        1749       5833    3.22  5.3e-16  
      1361  1361             prime    replayed        6221      10752    1.62  7.7e-16  rader
      1362  2.3.227          prime    raced          10275      10666    1.03  5.2e-16  bluestein
      1363  29.47            2p       replayed        4809      10524    1.55  4.5e-16  flips differ 1.40x
      1364  2^2.11.31        flat     replayed        3030       7715    2.51  7.7e-16  
      1365  3.5.7.13         chain3   replayed        2266       4213    1.82  6.0e-16  
      1366  2.683            prime    raced          10208      10634    1.04  6.2e-16  bluestein
      1367  1367             prime    replayed       10193      10622    1.04  6.0e-16  bluestein
      1368  2^3.3^2.19       chain3   replayed        1833       6365    3.40  5.1e-16  
      1369  37^2             2p       replayed        4189       9934    1.66  5.1e-16  flips differ 1.43x
      1370  2.5.137          prime    raced          10206      10610    1.01  4.5e-16  bluestein
      1371  3.457            prime    replayed       10210      10607    1.02  5.9e-16  bluestein
      1372  2^2.7^3          flat     replayed        1909       4267    2.21  4.0e-16  
      1373  1373             prime    replayed       10248      10653    1.01  6.3e-16  bluestein
      1374  2.3.229          prime    raced          10409      10635    1.02  4.1e-16  bluestein
      1375  5^3.11           chain3   replayed        2317       4116    1.71  4.2e-16  
      1376  2^5.43           2p       replayed        2456       9224    3.72  4.5e-16  
      1377  3^4.17           chain3   replayed        2515       5430    1.83  8.2e-16  
      1378  2.13.53          prime    raced          10320       9455    0.91  4.6e-16  bluestein
      1379  7.197            prime    replayed       10139      10641    1.04  6.0e-16  bluestein
      1380  2^2.3.5.23       chain3   replayed        2005       6798    3.39  5.0e-16  
      1381  1381             prime    replayed        8046      10614    1.23  9.6e-16  rader
      1382  2.691            prime    raced          10379      10609    1.02  4.4e-16  bluestein
      1383  3.461            prime    replayed       10400      11222    1.03  5.4e-16  bluestein
      1384  2^3.173          prime    raced          10239      10624    1.01  5.6e-16  bluestein
      1385  5.277            prime    replayed       10393      10625    1.01  7.1e-16  bluestein
      1386  2.3^2.7.11       chain3   replayed        2291       5197    2.24  5.2e-16  
      1387  19.73            prime    replayed       10352      12598    1.20  5.0e-16  bluestein
      1388  2^2.347          prime    raced          10316      10651    1.03  5.3e-16  bluestein
      1389  3.463            prime    replayed       10373      10683    1.02  6.7e-16  bluestein
      1390  2.5.139          prime    raced          10213      10645    0.86  5.4e-16  bluestein
      1391  13.107           prime    replayed       10223      10635    1.00  4.5e-16  bluestein
      1392  2^4.3.29         flat     replayed        2565       7478    2.60  5.3e-16  
      1393  7.199            prime    replayed       10341      10685    0.90  6.3e-16  bluestein
      1394  2.17.41          flat     replayed        3567       8626    2.38  1.1e-15  
      1395  3^2.5.31         flat     replayed        2788       7285    2.58  6.6e-16  
      1396  2^2.349          prime    raced          10157      10701    1.03  6.8e-16  bluestein
      1397  11.127           prime    replayed       10299      10636    1.03  5.2e-16  bluestein
      1398  2.3.233          prime    raced          10232      10618    1.03  5.8e-16  bluestein
      1399  1399             prime    replayed       10468      10620    1.01  6.8e-16  bluestein
      1400  2^3.5^2.7        chain3   replayed        1867       4450    2.37  4.6e-16  
      1401  3.467            prime    replayed       10443      10837    1.03  6.7e-16  bluestein
      1402  2.701            prime    raced          10251      10634    1.03  8.4e-16  bluestein
      1403  23.61            prime    replayed       10325      11852    1.14  4.8e-16  bluestein
      1404  2^2.3^3.13       chain3   replayed        1696       5611    3.29  5.5e-16  
      1405  5.281            prime    replayed       10495      10689    1.02  5.3e-16  bluestein
      1406  2.19.37          flat     replayed        3639       8747    2.35  1.2e-15  
      1407  3.7.67           prime    replayed       10255      10215    0.97  7.4e-16  bluestein
      1408  2^7.11           chain3   replayed        1646       5657    3.11  4.0e-16  
      1409  1409             prime    replayed        5399      10747    1.96  6.0e-16  rader
      1410  2.3.5.47         flat     replayed        3905       9588    2.42  5.6e-16  
      1411  17.83            prime    replayed       10365      13807    1.32  5.5e-16  bluestein
      1412  2^2.353          prime    raced          10381      10628    1.02  5.5e-16  bluestein
      1413  3^2.157          prime    replayed       10359      10670    1.02  5.4e-16  bluestein
      1414  2.7.101          prime    raced          10256      10632    1.01  6.4e-16  bluestein
      1415  5.283            prime    replayed       10362      10653    1.03  5.8e-16  bluestein
      1416  2^3.3.59         prime    raced          10459      11340    1.08  5.5e-16  bluestein
      1417  13.109           prime    replayed       10304      10660    1.01  5.9e-16  bluestein
      1418  2.709            prime    raced          10608      10682    0.78  6.6e-16  flips differ 1.29x bluestein
      1419  3.11.43          flat     replayed        3313       8382    2.49  6.0e-16  
      1420  2^2.5.71         prime    raced          10309      13086    1.26  5.7e-16  bluestein
      1421  7^2.29           chain3   replayed        2886       5237    1.80  5.7e-16  
      1422  2.3^2.79         prime    raced          10413      13972    1.33  5.3e-16  bluestein
      1423  1423             prime    replayed       10192      10667    0.81  6.2e-16  flips differ 1.29x bluestein
      1424  2^4.89           prime    raced          10197      15266    1.46  5.1e-16  bluestein
      1425  3.5^2.19         flat     replayed        2253       5775    2.53  5.1e-16  
      1426  2.23.31          flat     replayed        3422       9005    2.51  1.8e-15  
      1427  1427             prime    replayed       10209      10662    0.92  5.4e-16  bluestein
      1428  2^2.3.7.17       chain3   replayed        1826       6091    3.27  6.1e-16  
      1429  1429             prime    replayed        8024      10634    1.28  1.2e-15  rader
      1430  2.5.11.13        chain3   replayed        2026       5411    2.65  4.8e-16  
      1431  3^3.53           prime    replayed       10288      10070    0.96  6.0e-16  bluestein
      1432  2^3.179          prime    raced          10229      10820    1.04  5.3e-16  bluestein
      1433  1433             prime    replayed       10191      10749    1.02  5.4e-16  bluestein
      1434  2.3.239          prime    raced          10327      10633    0.74  5.4e-16  flips differ 1.39x bluestein
      1435  5.7.41           chain3   replayed        3496       6779    1.92  5.8e-16  
      1436  2^2.359          prime    raced          10166      10648    1.02  6.4e-16  bluestein
      1437  3.479            prime    replayed       10346      10734    0.88  6.3e-16  bluestein
      1438  2.719            prime    raced          10247      10660    1.02  6.5e-16  bluestein
      1439  1439             prime    replayed       10229      10673    1.02  5.8e-16  bluestein
      1440  2^5.3^2.5        chain3   replayed        1529       4938    3.20  4.8e-16  
      1441  11.131           prime    replayed       10194      10634    1.04  4.6e-16  bluestein
      1442  2.7.103          prime    raced          10346      10673    1.02  4.8e-16  bluestein
      1443  3.13.37          flat     replayed        3231       7829    2.38  6.5e-16  
      1444  2^2.19^2         chain3   replayed        2950       7947    2.65  4.3e-16  
      1445  5.17^2           chain3   replayed        2661       5901    2.08  6.4e-16  
      1446  2.3.241          prime    raced          10196      10655    1.04  5.6e-16  bluestein
      1447  1447             prime    replayed       10331      10680    1.01  6.6e-16  bluestein
      1448  2^3.181          prime    raced          10195      10668    1.04  7.0e-16  bluestein
      1449  3^2.7.23         flat     replayed        2780       5865    2.09  5.4e-16  
      1450  2.5^2.29         chain3   replayed        2606       6839    2.49  5.8e-16  
      1451  1451             prime    replayed       10438      10823    1.02  5.7e-16  bluestein
      1452  2^2.3.11^2       chain3   replayed        1755       6878    3.91  4.8e-16  
      1453  1453             prime    replayed        5890      10640    1.78  7.9e-16  rader
      1454  2.727            prime    raced          10233      10632    1.01  4.8e-16  bluestein
      1455  3.5.97           prime    replayed       10223      14249    1.04  5.5e-16  flips differ 1.34x bluestein
      1456  2^4.7.13         chain3   replayed        1675       5842    3.33  6.0e-16  
      1457  31.47            2p       replayed        5179      11559    2.19  5.9e-16  
      1458  2.3^6            chain3   replayed        2548       5131    1.66  4.7e-16  
      1459  1459             prime    replayed        8783      10729    1.12  1.1e-15  rader
      1460  2^2.5.73         prime    raced          10435      13508    1.29  4.5e-16  bluestein
      1461  3.487            prime    replayed       10471      10701    1.02  5.9e-16  bluestein
      1462  2.17.43          flat     replayed        3923       9583    2.43  1.0e-15  
      1463  7.11.19          chain3   replayed        2562       5582    2.12  4.8e-16  
      1464  2^3.3.61         prime    raced          10192      11979    1.14  5.6e-16  bluestein
      1465  5.293            prime    replayed       10253      10661    1.03  7.6e-16  bluestein
      1466  2.733            prime    raced          10248      10643    1.02  6.5e-16  bluestein
      1467  3^2.163          prime    replayed       10387      10665    1.02  5.8e-16  bluestein
      1468  2^2.367          prime    raced          10418      10692    1.02  5.2e-16  bluestein
      1469  13.113           prime    replayed       10255      10743    1.02  5.5e-16  bluestein
      1470  2.3.5.7^2        chain3   replayed        2261       4638    2.01  5.5e-16  
      1471  1471             prime    replayed        7861      10932    1.35  9.4e-16  rader
      1472  2^6.23           2p       replayed        2505       7504    2.47  4.5e-16  
      1473  3.491            prime    replayed       10194      10649    1.04  7.1e-16  bluestein
      1474  2.11.67          prime    raced          10154      12022    1.18  4.4e-16  bluestein
      1475  5^2.59           prime    replayed       10268       9427    0.90  7.8e-16  bluestein
      1476  2^2.3^2.41       chain3   replayed        2936       9293    3.16  4.9e-16  
      1477  7.211            prime    replayed       10321      10659    1.02  5.8e-16  bluestein
      1478  2.739            prime    raced          10172      10718    1.02  5.5e-16  bluestein
      1479  3.17.29          flat     replayed        3390       7809    2.29  1.2e-15  
      1480  2^3.5.37         flat     replayed        2932       8830    2.90  5.8e-16  
      1481  1481             prime    replayed       10342      10744    0.74  5.5e-16  flips differ 1.39x bluestein
      1482  2.3.13.19        chain3   replayed        2471       6933    2.73  6.9e-16  
      1483  1483             prime    replayed        7977      10683    1.28  9.6e-16  rader
      1484  2^2.7.53         prime    raced          10348      10877    1.04  5.3e-16  bluestein
      1485  3^3.5.11         chain3   replayed        2298       5185    2.19  5.3e-16  
      1486  2.743            prime    raced          10248      10764    1.04  5.9e-16  bluestein
      1487  1487             prime    replayed       10263      10952    1.04  5.6e-16  bluestein
      1488  2^4.3.31         chain3   replayed        2480       8471    3.12  5.1e-16  
      1489  1489             prime    replayed       10242      10737    1.04  6.3e-16  bluestein
      1490  2.5.149          prime    raced          10310      10672    0.99  6.0e-16  bluestein
      1491  3.7.71           prime    replayed       10356      11463    1.10  5.4e-16  bluestein
      1492  2^2.373          prime    raced          10202      10691    1.05  6.0e-16  bluestein
      1493  1493             prime    replayed       10474      10663    0.91  5.0e-16  bluestein
      1494  2.3^2.83         prime    raced          10304      15140    1.45  4.8e-16  bluestein
      1495  5.13.23          chain3   replayed        2813       6469    2.19  6.5e-16  
      1496  2^3.11.17        chain3   replayed        2224       7638    3.43  4.7e-16  
      1497  3.499            prime    replayed       10486      10678    1.00  6.5e-16  bluestein
      1498  2.7.107          prime    raced          10192      10706    1.04  5.7e-16  bluestein
      1499  1499             prime    replayed       10220      10672    1.04  7.2e-16  bluestein
      1500  2^2.3.5^3        chain3   replayed        1845       4851    2.60  4.2e-16  
      1501  19.79            prime    replayed       10242      14633    1.42  4.8e-16  bluestein
      1502  2.751            prime    raced          10248      10985    1.04  5.8e-16  bluestein
      1503  3^2.167          prime    replayed       10517      10734    1.01  6.4e-16  bluestein
      1504  2^5.47           2p       replayed        2658      10670    3.99  4.5e-16  
      1505  5.7.43           chain3   replayed        5436       7576    1.37  6.1e-16  
      1506  2.3.251          prime    raced          10454      10690    1.01  5.0e-16  bluestein
      1507  11.137           prime    replayed       10517      10663    0.97  6.1e-16  bluestein
      1508  2^2.13.29        flat     replayed        3365       8400    2.42  1.1e-15  
      1509  3.503            prime    replayed       10293      10673    1.03  6.6e-16  bluestein
      1510  2.5.151          prime    raced          10326      10740    1.02  4.8e-16  bluestein
      1511  1511             prime    replayed       10301      10682    1.04  6.4e-16  bluestein
      1512  2^3.3^3.7        chain3   replayed        1920       5081    2.46  4.5e-16  
      1513  17.89            prime    replayed       10288      15374    1.47  4.7e-16  bluestein
      1514  2.757            prime    raced          10273      10735    1.02  5.6e-16  bluestein
      1515  3.5.101          prime    replayed       10424      10715    1.02  5.4e-16  bluestein
      1516  2^2.379          prime    raced          10289      10694    0.86  8.0e-16  bluestein
      1517  37.41            2p       replayed        4874      11406    1.65  8.5e-16  flips differ 1.42x
      1518  2.3.11.23        chain3   replayed        2611       7618    2.91  4.2e-16  
      1519  7^2.31           flat     replayed        3331       6087    1.77  7.3e-16  
      1520  2^4.5.19         chain3   replayed        2155       7011    2.92  4.2e-16  
      1521  3^2.13^2         chain3   replayed        2455       6029    2.42  7.1e-16  
      1522  2.761            prime    raced          10462      10756    1.02  5.9e-16  bluestein
      1523  1523             prime    replayed       10605      10827    1.01  5.5e-16  bluestein
      1524  2^2.3.127        prime    raced          10517      10847    1.02  7.2e-16  bluestein
      1525  5^2.61           prime    replayed       10254       9961    0.95  4.5e-16  bluestein
      1526  2.7.109          prime    raced          10297      10678    1.03  5.2e-16  bluestein
      1527  3.509            prime    replayed       10240      10694    1.04  5.7e-16  bluestein
      1528  2^3.191          prime    raced          10353      10670    1.03  4.8e-16  bluestein
      1529  11.139           prime    replayed       10256      10670    1.03  5.4e-16  bluestein
      1530  2.3^2.5.17       chain3   replayed        2281       6514    2.70  5.8e-16  
      1531  1531             prime    replayed        7179      10704    1.45  7.5e-16  rader
      1532  2^2.383          prime    raced          10303      10714    1.02  6.6e-16  bluestein
      1533  3.7.73           prime    replayed       10304      11784    1.13  4.8e-16  bluestein
      1534  2.13.59          prime    raced          10282      11507    1.11  4.2e-16  bluestein
      1535  5.307            prime    replayed       10332      10678    1.01  6.5e-16  bluestein
      1536  2^9.3            chain3   replayed        1627       5635    3.43  4.5e-16  
      1537  29.53            prime    replayed       10260      12258    1.16  6.8e-16  bluestein
      1538  2.769            prime    raced          10277      10696    1.02  5.9e-16  bluestein
      1539  3^4.19           chain3   replayed        2882       6537    2.25  7.1e-16  
      1540  2^2.5.7.11       flat     replayed        2097       5865    2.73  4.6e-16  
      1541  23.67            prime    replayed       10301      13968    1.33  5.4e-16  bluestein
      1542  2.3.257          prime    raced          10403      10783    1.02  6.7e-16  bluestein
      1543  1543             prime    replayed       10566      10775    1.01  4.8e-16  bluestein
      1544  2^3.193          prime    raced          10340      10685    1.02  6.0e-16  bluestein
      1545  3.5.103          prime    replayed       10282      10711    1.02  5.2e-16  bluestein
      1546  2.773            prime    raced          10467      10707    1.02  6.5e-16  bluestein
      1547  7.13.17          flat     replayed        2768       5606    1.98  9.5e-16  
      1548  2^2.3^2.43       chain3   replayed        3097      10299    3.25  6.3e-16  
      1549  1549             prime    replayed       10444      10733    1.02  5.6e-16  bluestein
      1550  2.5^2.31         chain3   replayed        2912       7919    2.70  5.3e-16  
      1551  3.11.47          chain3   replayed        7638       9883    1.18  5.0e-16  
      1552  2^4.97           prime    raced          10316      17774    1.68  5.7e-16  bluestein
      1553  1553             prime    replayed       10421      10703    1.03  6.3e-16  bluestein
      1554  2.3.7.37         flat     replayed        4127       8661    2.03  8.7e-16  
      1555  5.311            prime    replayed       10393      10671    1.02  6.0e-16  bluestein
      1556  2^2.389          prime    raced          10516      10706    1.02  6.6e-16  bluestein
      1557  3^2.173          prime    replayed       10447      10789    1.03  5.9e-16  bluestein
      1558  2.19.41          flat     replayed        4310      10297    2.35  1.2e-15  
      1559  1559             prime    replayed       10272      10688    1.01  5.1e-16  bluestein
      1560  2^3.3.5.13       chain3   replayed        1886       6271    3.27  5.9e-16  
      1561  7.223            prime    replayed       10496      10691    1.02  5.7e-16  bluestein
      1562  2.11.71          prime    raced          10232      13390    1.27  6.0e-16  bluestein
      1563  3.521            prime    replayed       10582      10725    0.74  5.8e-16  flips differ 1.36x bluestein
      1564  2^2.17.23        chain3   replayed        2986       8826    2.70  5.2e-16  
      1565  5.313            prime    replayed       10335      10702    1.01  6.2e-16  bluestein
      1566  2.3^3.29         chain3   replayed        3590       8197    1.78  6.0e-16  flips differ 1.28x
      1567  1567             prime    replayed       10288      10756    1.04  6.3e-16  bluestein
      1568  2^5.7^2          chain3   replayed        1757       5230    2.85  4.8e-16  
      1569  3.523            prime    replayed       10423      10675    0.83  6.3e-16  bluestein
      1570  2.5.157          prime    raced          10267      10677    1.04  6.5e-16  bluestein
      1571  1571             prime    replayed       10355      10690    1.01  6.6e-16  bluestein
      1572  2^2.3.131        prime    raced          10514      10754    1.02  7.6e-16  bluestein
      1573  11^2.13          chain3   replayed        2579       6249    2.37  5.6e-16  
      1574  2.787            prime    raced          10451      10699    1.02  6.0e-16  bluestein
      1575  3^2.5^2.7        flat     replayed        2422       4615    1.88  6.6e-16  
      1576  2^3.197          prime    raced          10368      10720    1.01  6.1e-16  bluestein
      1577  19.83            prime    replayed       10481      15748    1.50  5.9e-16  bluestein
      1578  2.3.263          prime    raced          10283      10695    1.02  5.9e-16  bluestein
      1579  1579             prime    replayed       10311      10695    1.02  4.6e-16  bluestein
      1580  2^2.5.79         prime    raced          10512      15719    1.49  4.7e-16  bluestein
      1581  3.17.31          chain3   replayed        4841       8711    1.35  6.2e-16  flips differ 1.33x
      1582  2.7.113          prime    raced          10300      10705    1.02  4.9e-16  bluestein
      1583  1583             prime    replayed       10408      10758    1.02  6.3e-16  bluestein
      1584  2^4.3^2.11       chain3   replayed        1805       6296    3.17  3.7e-16  
      1585  5.317            prime    replayed       10300      10733    1.02  5.4e-16  bluestein
      1586  2.13.61          prime    raced          10250      12146    1.18  5.4e-16  bluestein
      1587  3.23^2           flat     replayed        3871       8955    2.31  1.2e-15  
      1588  2^2.397          prime    raced          10492      10753    1.02  6.8e-16  bluestein
      1589  7.227            prime    replayed       10308      10685    1.04  5.8e-16  bluestein
      1590  2.3.5.53         prime    raced          10266      11464    1.11  4.9e-16  bluestein
      1591  37.43            2p       replayed        5553      12517    1.56  5.8e-16  flips differ 1.44x
      1592  2^3.199          prime    raced          10249      10724    1.04  5.8e-16  bluestein
      1593  3^3.59           prime    replayed       10334      12104    1.15  5.5e-16  bluestein
      1594  2.797            prime    raced          10287      10746    1.03  6.5e-16  bluestein
      1595  5.11.29          flat     replayed        3554       7409    2.00  6.7e-16  
      1596  2^2.3.7.19       chain3   replayed        2156       7374    3.35  4.8e-16  
      1597  1597             prime    replayed       10301      10753    1.04  6.0e-16  bluestein
      1598  2.17.47          flat     replayed        4487      11132    2.47  1.7e-15  
      1599  3.13.41          chain3   replayed        3649       9102    2.46  5.7e-16  
      1600  2^6.5^2          chain3   replayed        1727       5320    3.07  3.4e-16  
      1601  1601             prime    replayed        6270      10723    1.68  5.8e-16  rader
      1602  2.3^2.89         prime    raced          10268      16962    1.61  4.6e-16  bluestein
      1603  7.229            prime    replayed       10507      10738    0.92  5.0e-16  bluestein
      1604  2^2.401          prime    raced          10468      10728    1.02  4.0e-16  bluestein
      1605  3.5.107          prime    replayed       10290      10714    1.04  5.3e-16  bluestein
      1606  2.11.73          prime    raced          10197      13827    1.32  5.6e-16  bluestein
      1607  1607             prime    replayed       10340      10736    1.02  4.8e-16  bluestein
      1608  2^3.3.67         prime    raced          10396      14170    1.35  5.2e-16  bluestein
      1609  1609             prime    replayed       10329      10730    1.02  7.1e-16  bluestein
      1610  2.5.7.23         chain3   replayed        2597       6549    2.52  4.9e-16  
      1611  3^2.179          prime    replayed       10235      10735    1.04  6.0e-16  bluestein
      1612  2^2.13.31        flat     replayed        3738       9457    2.51  8.9e-16  
      1613  1613             prime    replayed       10488      10714    1.02  4.0e-16  bluestein
      1614  2.3.269          prime    raced          10312      10731    1.02  4.8e-16  bluestein
      1615  5.17.19          chain3   replayed        3435       7098    1.96  6.1e-16  
      1616  2^4.101          prime    raced          10355      10701    1.01  4.9e-16  bluestein
      1617  3.7^2.11         chain3   replayed        2852       5445    1.91  5.5e-16  
      1618  2.809            prime    raced          10382      10746    1.01  4.9e-16  bluestein
      1619  1619             prime    replayed       10395      10922    1.03  6.2e-16  bluestein
      1620  2^2.3^4.5        chain3   replayed        1887       5503    2.91  6.4e-16  
      1621  1621             prime    replayed       10321      10745    1.04  5.3e-16  bluestein
      1622  2.811            prime    raced          10409      10730    0.86  6.2e-16  bluestein
      1623  3.541            prime    replayed       10449      10745    1.03  5.6e-16  bluestein
      1624  2^3.7.29         flat     replayed        2967       8648    2.16  5.8e-16  flips differ 1.35x
      1625  5^3.13           flat     replayed        2565       5454    2.12  6.6e-16  
      1626  2.3.271          prime    raced          10386      11246    1.03  5.4e-16  bluestein
      1627  1627             prime    replayed       10489      10765    0.90  5.9e-16  bluestein
      1628  2^2.11.37        chain3   replayed        4730      10086    2.10  4.8e-16  
      1629  3^2.181          prime    replayed       10518      10736    1.02  5.8e-16  bluestein
      1630  2.5.163          prime    raced          10390      10753    0.75  5.5e-16  flips differ 1.39x bluestein
      1631  7.233            prime    replayed       10299      10782    1.03  5.3e-16  bluestein
      1632  2^5.3.17         chain3   replayed        2076       7195    3.44  4.8e-16  
      1633  23.71            prime    replayed       10509      15523    1.47  5.9e-16  bluestein
      1634  2.19.43          flat     replayed        4413      11279    2.52  8.7e-16  
      1635  3.5.109          prime    replayed       10299      10769    1.03  5.7e-16  bluestein
      1636  2^2.409          prime    raced          10294      10896    1.04  5.0e-16  bluestein
      1637  1637             prime    replayed       10380      10786    1.03  5.1e-16  bluestein
      1638  2.3^2.7.13       chain3   replayed        2686       6388    2.31  5.7e-16  
      1639  11.149           prime    replayed       10413      10799    1.03  5.2e-16  bluestein
      1640  2^3.5.41         flat     replayed        3269      10264    2.83  8.1e-16  
      1641  3.547            prime    replayed       10315      10696    1.03  5.8e-16  bluestein
      1642  2.821            prime    raced          10319      10787    1.04  5.2e-16  bluestein
      1643  31.53            prime    replayed       10495      13597    1.29  5.0e-16  bluestein
      1644  2^2.3.137        prime    raced          10343      10764    1.02  4.5e-16  bluestein
      1645  5.7.47           flat     replayed        3801       9005    1.98  5.6e-16  
      1646  2.823            prime    raced          10334      10711    1.02  5.6e-16  bluestein
      1647  3^3.61           prime    replayed       10334      12914    1.22  5.7e-16  bluestein
      1648  2^4.103          prime    raced          10330      10751    1.01  5.6e-16  bluestein
      1649  17.97            prime    replayed       10359      17936    1.71  5.7e-16  bluestein
      1650  2.3.5^2.11       chain3   replayed        2243       6195    2.56  4.3e-16  
      1651  13.127           prime    replayed       10547      10745    0.80  5.6e-16  flips differ 1.27x bluestein
      1652  2^2.7.59         prime    raced          10351      13115    1.26  4.2e-16  bluestein
      1653  3.19.29          chain3   replayed        3828       9121    1.69  5.4e-16  flips differ 1.41x
      1654  2.827            prime    raced          10398      10741    1.02  4.9e-16  bluestein
      1655  5.331            prime    replayed       10308      10823    1.05  5.6e-16  bluestein
      1656  2^3.3^2.23       chain3   replayed        2497       8324    3.28  5.7e-16  
      1657  1657             prime    replayed       10322      10768    1.04  5.5e-16  bluestein
      1658  2.829            prime    raced          10335      10750    1.01  5.6e-16  bluestein
      1659  3.7.79           prime    replayed       10509      13983    1.32  4.9e-16  bluestein
      1660  2^2.5.83         prime    raced          10422      16967    1.62  6.1e-16  bluestein
      1661  11.151           prime    replayed       10457      10784    1.03  5.3e-16  bluestein
      1662  2.3.277          prime    raced          10329      10708    1.03  5.7e-16  bluestein
      1663  1663             prime    replayed       10545      10776    1.02  4.5e-16  bluestein
      1664  2^7.13           chain3   replayed        1919       6845    3.56  4.6e-16  
      1665  3^2.5.37         chain3   replayed        4911       9277    1.88  5.6e-16  
      1666  2.7^2.17         flat     replayed        3388       5539    1.62  7.7e-16  
      1667  1667             prime    replayed       10454      10789    1.03  6.7e-16  bluestein
      1668  2^2.3.139        prime    raced          10423      10751    0.90  5.9e-16  bluestein
      1669  1669             prime    replayed       10569      11279    1.01  6.0e-16  bluestein
      1670  2.5.167          prime    raced          10363      10760    1.02  5.5e-16  bluestein
      1671  3.557            prime    replayed       10454      10710    1.02  4.7e-16  bluestein
      1672  2^3.11.19        chain3   replayed        2604       9056    3.46  5.4e-16  
      1673  7.239            prime    replayed       10596      10767    1.01  5.5e-16  bluestein
      1674  2.3^3.31         flat     replayed        3827       9376    2.37  7.4e-16  
      1675  5^2.67           prime    replayed       10389      12199    1.16  5.5e-16  bluestein
      1676  2^2.419          prime    raced          10339      10801    1.02  5.9e-16  bluestein
      1677  3.13.43          chain3   replayed        4073      10082    2.47  6.8e-16  
      1678  2.839            prime    raced          10289      10741    1.04  4.9e-16  bluestein
      1679  23.73            prime    replayed       10431      15915    1.50  5.2e-16  bluestein
      1680  2^4.3.5.7        chain3   replayed        1894       5558    2.92  5.3e-16  
      1681  41^2             2p       replayed        7330      13261    1.74  5.1e-16  
      1682  2.29^2           flat     replayed        4291      10741    2.45  1.7e-15  
      1683  3^2.11.17        chain3   replayed        2825       7238    2.53  7.8e-16  
      1684  2^2.421          prime    raced          10429      10771    1.03  5.1e-16  bluestein
      1685  5.337            prime    replayed       10332      10812    1.04  5.2e-16  bluestein
      1686  2.3.281          prime    raced          10294      10790    1.02  6.5e-16  bluestein
      1687  7.241            prime    replayed       10426      10739    1.03  6.2e-16  bluestein
      1688  2^3.211          prime    raced          10350      10783    1.04  5.2e-16  bluestein
      1689  3.563            prime    replayed       10586      10782    1.01  5.1e-16  bluestein
      1690  2.5.13^2         chain3   replayed        2440       6849    2.67  5.9e-16  
      1691  19.89            prime    replayed       10546      17707    1.66  5.9e-16  bluestein
      1692  2^2.3^2.47       chain3   replayed        3569      11949    3.30  5.0e-16  
      1693  1693             prime    replayed       10416      10919    1.04  6.5e-16  bluestein
      1694  2.7.11^2         flat     replayed        3431       6528    1.86  7.9e-16  
      1695  3.5.113          prime    replayed       10329      10748    1.04  6.9e-16  bluestein
      1696  2^5.53           prime    raced          10329      12776    1.23  5.3e-16  bluestein
      1697  1697             prime    replayed       10395      10768    1.03  5.8e-16  bluestein
      1698  2.3.283          prime    raced          10373      10774    1.01  6.0e-16  bluestein
      1699  1699             prime    replayed       10345      10798    1.04  5.5e-16  bluestein
      1700  2^2.5^2.17       chain3   replayed        2445       7241    2.78  6.1e-16  
      1701  3^5.7            chain3   replayed        2672       5136    1.87  4.7e-16  
      1702  2.23.37          flat     replayed        4511      11459    2.53  1.5e-15  
      1703  13.131           prime    replayed       10288      10775    1.04  6.5e-16  bluestein
      1704  2^3.3.71         prime    raced          10522      15921    1.50  5.4e-16  bluestein
      1705  5.11.31          chain3   replayed        6010       8449    1.40  5.0e-16  
      1706  2.853            prime    raced          10458      10750    1.01  6.3e-16  bluestein
      1707  3.569            prime    replayed       10326      10809    1.02  5.0e-16  bluestein
      1708  2^2.7.61         prime    raced          10365      13935    1.33  4.4e-16  bluestein
      1709  1709             prime    replayed       10409      10847    1.04  5.5e-16  bluestein
      1710  2.3^2.5.19       chain3   replayed        3037       7805    2.54  5.1e-16  
      1711  29.59            prime    replayed       10523      14723    1.40  5.9e-16  bluestein
      1712  2^4.107          prime    raced          10215      10778    1.01  6.8e-16  bluestein
      1713  3.571            prime    replayed       10445      10772    1.02  5.7e-16  bluestein
      1714  2.857            prime    raced          10602      10775    0.78  5.9e-16  flips differ 1.30x bluestein
      1715  5.7^3            flat     replayed        2729       4466    1.63  4.9e-16  
      1716  2^2.3.11.13      chain3   replayed        2161       8278    3.68  6.0e-16  
      1717  17.101           prime    replayed       10496      10774    1.02  5.4e-16  bluestein
      1718  2.859            prime    raced          10393      10787    1.03  4.8e-16  bluestein
      1719  3^2.191          prime    replayed       10368      10779    1.02  5.2e-16  bluestein
      1720  2^3.5.43         chain3   replayed        3398      11317    3.28  5.4e-16  
      1721  1721             prime    replayed       10395      10750    1.00  5.9e-16  bluestein
      1722  2.3.7.41         chain3   replayed        3625       9996    2.71  5.4e-16  
      1723  1723             prime    replayed       10412      10784    1.03  6.8e-16  bluestein
      1724  2^2.431          prime    raced          10319      10754    1.03  5.8e-16  bluestein
      1725  3.5^2.23         chain3   replayed        3063       7830    2.52  6.9e-16  
      1726  2.863            prime    raced          10480      10845    1.03  5.9e-16  bluestein
      1727  11.157           prime    replayed       10371      10816    1.02  6.9e-16  bluestein
      1728  2^6.3^3          chain3   replayed        1868       6344    3.39  5.6e-16  
      1729  7.13.19          chain3   replayed        3413       6896    1.79  6.4e-16  
      1730  2.5.173          prime    raced          10421      10824    1.03  5.6e-16  bluestein
      1731  3.577            prime    replayed       10323      10954    1.02  5.7e-16  bluestein
      1732  2^2.433          prime    raced          10495      10788    1.02  6.1e-16  bluestein
      1733  1733             prime    replayed       10352      10769    1.02  5.8e-16  bluestein
      1734  2.3.17^2         chain3   replayed        3162       8666    2.60  5.3e-16  
      1735  5.347            prime    replayed       10317      10771    1.02  4.9e-16  bluestein
      1736  2^3.7.31         flat     replayed        3579       9787    2.40  6.9e-16  
      1737  3^2.193          prime    replayed       10429      11089    0.79  5.5e-16  flips differ 1.39x bluestein
      1738  2.11.79          prime    raced          10334      16216    1.53  5.2e-16  bluestein
      1739  37.47            flat     replayed        4674      14463    3.01  1.9e-15  
      1740  2^2.3.5.29       flat     replayed        3921       9278    2.36  5.4e-16  
      1741  1741             prime    replayed       10649      11041    0.84  5.5e-16  bluestein
      1742  2.13.67          prime    raced          10547      14393    1.36  4.6e-16  bluestein
      1743  3.7.83           prime    replayed       10380      15122    1.43  4.1e-16  bluestein
      1744  2^4.109          prime    raced          10589      10789    1.02  7.1e-16  bluestein
      1745  5.349            prime    replayed       10347      10817    1.04  6.3e-16  bluestein
      1746  2.3^2.97         prime    raced          10579      19684    1.85  4.6e-16  bluestein
      1747  1747             prime    replayed       10603      10779    1.01  4.8e-16  bluestein
      1748  2^2.19.23        chain3   replayed        3350      10565    3.06  4.7e-16  
      1749  3.11.53          prime    replayed       10593      11793    0.85  4.6e-16  flips differ 1.30x bluestein
      1750  2.5^3.7          chain3   replayed        3002       5107    1.54  5.7e-16  
      1751  17.103           prime    replayed       10374      10772    1.04  5.1e-16  bluestein
      1752  2^3.3.73         prime    raced          10612      16295    1.12  5.9e-16  flips differ 1.36x bluestein
      1753  1753             prime    replayed       10359      10794    1.01  5.1e-16  bluestein
      1754  2.877            prime    raced          10352      10774    1.04  5.2e-16  bluestein
      1755  3^3.5.13         chain3   replayed        2909       6399    2.04  8.8e-16  
      1756  2^2.439          prime    raced          10546      10870    1.02  4.8e-16  bluestein
      1757  7.251            prime    replayed       10445      10842    1.03  4.5e-16  bluestein
      1758  2.3.293          prime    raced          10432      10792    1.03  5.3e-16  bluestein
      1759  1759             prime    replayed       10556      10819    1.02  4.9e-16  bluestein
      1760  2^5.5.11         chain3   replayed        2004       7016    3.47  4.7e-16  
      1761  3.587            prime    replayed       10271      10799    1.04  6.7e-16  bluestein
      1762  2.881            prime    raced          10604      10829    0.79  5.4e-16  flips differ 1.29x bluestein
      1763  41.43            2p       replayed        6411      14358    2.22  6.0e-16  
      1764  2^2.3^2.7^2      chain3   replayed        2195       5824    2.63  5.1e-16  
      1765  5.353            prime    replayed       10289      11316    1.04  4.8e-16  bluestein
      1766  2.883            prime    raced          10315      10853    1.02  5.1e-16  bluestein
      1767  3.19.31          chain3   replayed        4011      10430    2.33  4.8e-16  
      1768  2^3.13.17        chain3   replayed        2637       9281    3.36  6.8e-16  
      1769  29.61            prime    replayed       10354      15461    1.45  5.4e-16  bluestein
      1770  2.3.5.59         prime    raced          10383      13886    1.30  4.2e-16  bluestein
      1771  7.11.23          flat     replayed        3585       7625    2.06  8.1e-16  
      1772  2^2.443          prime    raced          10359      10948    1.04  6.1e-16  bluestein
      1773  3^2.197          prime    replayed       10374      10793    1.01  4.9e-16  bluestein
      1774  2.887            prime    raced          10477      10809    1.02  6.7e-16  bluestein
      1775  5^2.71           prime    replayed       10669      13726    1.28  5.4e-16  bluestein
      1776  2^4.3.37         chain3   replayed        3147      10798    3.42  5.0e-16  
      1777  1777             prime    replayed       10573      11327    1.02  8.6e-16  bluestein
      1778  2.7.127          prime    raced          10390      10791    1.02  6.4e-16  bluestein
      1779  3.593            prime    replayed       10430      10817    1.04  4.7e-16  bluestein
      1780  2^2.5.89         prime    raced          10386      19016    1.83  4.3e-16  bluestein
      1781  13.137           prime    replayed       10518      10824    1.03  6.1e-16  bluestein
      1782  2.3^4.11         chain3   replayed        3110       6842    2.06  5.7e-16  
      1783  1783             prime    replayed       10441      10786    1.01  5.6e-16  bluestein
      1784  2^3.223          prime    raced          10387      10781    1.02  5.6e-16  bluestein
      1785  3.5.7.17         chain3   replayed        3128       6023    1.90  6.5e-16  
      1786  2.19.47          flat     replayed        5158      13076    2.48  1.5e-15  
      1787  1787             prime    replayed       10526      10782    0.90  5.8e-16  bluestein
      1788  2^2.3.149        prime    raced          10266      10766    1.05  6.1e-16  bluestein
      1789  1789             prime    replayed       10550      10847    1.02  5.6e-16  bluestein
      1790  2.5.179          prime    raced          10504      10810    1.02  5.6e-16  bluestein
      1791  3^2.199          prime    replayed       10343      10804    1.01  5.7e-16  bluestein
      1792  2^8.7            chain3   replayed        2019       6321    2.87  2.9e-16  
      1793  11.163           prime    replayed       10517      10814    1.02  4.5e-16  bluestein
      1794  2.3.13.23        chain3   replayed        3451       9419    2.70  7.3e-16  
      1795  5.359            prime    replayed       10393      10811    1.02  6.5e-16  bluestein
      1796  2^2.449          prime    raced          10585      11334    1.02  6.1e-16  bluestein
      1797  3.599            prime    replayed       10375      10801    1.04  4.9e-16  bluestein
      1798  2.29.31          flat     replayed        4920      12029    2.36  2.0e-15  
      1799  7.257            prime    replayed       10429      10855    1.03  6.3e-16  bluestein
      1800  2^3.3^2.5^2      chain3   replayed        2056       5925    2.85  4.6e-16  
      1801  1801             prime    replayed       10408      10851    1.03  5.6e-16  bluestein
      1802  2.17.53          prime    raced          10349      13287    1.28  4.7e-16  bluestein
      1803  3.601            prime    replayed       10495      11718    1.03  5.3e-16  bluestein
      1804  2^2.11.41        chain3   replayed        3778      11744    3.05  5.2e-16  
      1805  5.19^2           flat     replayed        3878       8658    2.05  1.3e-15  
      1806  2.3.7.43         chain3   replayed        6275      11050    1.27  5.7e-16  flips differ 1.38x
      1807  13.139           prime    replayed       10303      10832    1.02  5.4e-16  bluestein
      1808  2^4.113          prime    raced          10340      10789    1.04  5.2e-16  bluestein
      1809  3^3.67           prime    replayed       10418      15373    1.45  5.2e-16  bluestein
      1810  2.5.181          prime    raced          10569      10874    1.02  5.4e-16  bluestein
      1811  1811             prime    replayed       10345      10803    0.80  5.0e-16  flips differ 1.31x bluestein
      1812  2^2.3.151        prime    raced          10355      10834    1.02  5.4e-16  bluestein
      1813  7^2.37           chain3   replayed        4213       8137    1.90  5.0e-16  
      1814  2.907            prime    raced          10566      10856    1.02  6.2e-16  bluestein
      1815  3.5.11^2         chain3   replayed        2882       7095    2.43  6.1e-16  
      1816  2^3.227          prime    raced          10349      11231    1.04  6.3e-16  bluestein
      1817  23.79            prime    replayed       10370      18685    1.58  5.0e-16  bluestein
      1818  2.3^2.101        prime    raced          10374      10812    1.04  5.9e-16  bluestein
      1819  17.107           prime    replayed       10397      10831    1.04  6.1e-16  bluestein
      1820  2^2.5.7.13       flat     replayed        2571       7151    2.76  8.1e-16  
      1821  3.607            prime    replayed       10371      10805    1.04  5.6e-16  bluestein
      1822  2.911            prime    raced          10373      10819    1.04  6.3e-16  bluestein
      1823  1823             prime    replayed       10579      10979    1.03  5.2e-16  bluestein
      1824  2^5.3.19         chain3   replayed        2576       8603    2.98  4.8e-16  
      1825  5^2.73           prime    replayed       10366      14116    1.33  5.2e-16  bluestein
      1826  2.11.83          prime    raced          10383      17572    1.65  5.4e-16  bluestein
      1827  3^2.7.29         flat     replayed        3945       8554    2.09  6.0e-16  
      1828  2^2.457          prime    raced          10635      11450    1.03  4.8e-16  bluestein
      1829  31.59            prime    replayed       10401      16312    1.18  4.8e-16  flips differ 1.33x bluestein
      1830  2.3.5.61         prime    raced          10524      14657    1.39  5.7e-16  bluestein
      1831  1831             prime    replayed       10571      10854    1.02  4.9e-16  bluestein
      1832  2^3.229          prime    raced          10446      10809    1.02  5.1e-16  bluestein
      1833  3.13.47          flat     replayed        4613      11879    2.49  7.7e-16  
      1834  2.7.131          prime    raced          10364      11147    1.03  6.9e-16  bluestein
      1835  5.367            prime    replayed       10398      10831    1.04  6.3e-16  bluestein
      1836  2^2.3^3.17       chain3   replayed        2459       7970    3.20  6.3e-16  
      1837  11.167           prime    replayed       10308      10888    1.04  7.7e-16  bluestein
      1838  2.919            prime    raced          10604      10815    1.02  6.1e-16  bluestein
      1839  3.613            prime    replayed       10451      10833    1.03  5.8e-16  bluestein
      1840  2^4.5.23         chain3   replayed        2732       9237    3.32  4.6e-16  
      1841  7.263            prime    replayed       10330      10826    1.02  6.3e-16  bluestein
      1842  2.3.307          prime    raced          10530      10943    1.02  5.3e-16  bluestein
      1843  19.97            prime    replayed       10354      20755    1.97  5.7e-16  bluestein
      1844  2^2.461          prime    raced          10515      10812    1.03  5.0e-16  bluestein
      1845  3^2.5.41         chain3   replayed        4170      10856    2.49  5.9e-16  
      1846  2.13.71          prime    raced          10547      16222    1.53  7.2e-16  bluestein
      1847  1847             prime    replayed       10476      10865    1.02  5.3e-16  bluestein
      1848  2^3.3.7.11       flat     replayed        3135       7308    2.32  9.1e-16  
      1849  43^2             flat     replayed        5046      15771    3.11  1.6e-15  
      1850  2.5^2.37         chain3   replayed        3725      10265    2.60  4.7e-16  
      1851  3.617            prime    replayed       10405      10844    1.02  5.8e-16  bluestein
      1852  2^2.463          prime    raced          10447      10836    1.02  5.5e-16  bluestein
      1853  17.109           prime    replayed       10481      10814    1.03  5.5e-16  bluestein
      1854  2.3^2.103        prime    raced          10484      10832    1.03  5.3e-16  bluestein
      1855  5.7.53           prime    replayed       10498      10862    1.02  5.5e-16  bluestein
      1856  2^6.29           2p       replayed        2652      10092    3.69  4.1e-16  
      1857  3.619            prime    replayed       10422      10825    1.02  5.6e-16  bluestein
      1858  2.929            prime    raced          10471      10857    1.03  5.0e-16  bluestein
      1859  11.13^2          chain3   replayed        3255       7625    2.33  7.3e-16  
      1860  2^2.3.5.31       chain3   replayed        3122      10575    3.35  6.1e-16  
      1861  1861             prime    replayed       10419      10814    1.02  6.7e-16  bluestein
      1862  2.7^2.19         flat     replayed        4071       6992    1.69  7.8e-16  
      1863  3^4.23           chain3   replayed        3414       8637    2.52  6.2e-16  
      1864  2^3.233          prime    raced          10512      10805    1.03  5.5e-16  bluestein
      1865  5.373            prime    replayed       10565      10869    1.02  4.6e-16  bluestein
      1866  2.3.311          prime    raced          10383      10807    1.02  5.2e-16  bluestein
      1867  1867             prime    replayed       10409      10838    1.01  5.9e-16  bluestein
      1868  2^2.467          prime    raced          10626      10820    1.01  5.6e-16  bluestein
      1869  3.7.89           prime    replayed       10426      17115    1.64  5.8e-16  bluestein
      1870  2.5.11.17        chain3   replayed        2812       8228    2.88  5.0e-16  
      1871  1871             prime    replayed       10376      10808    1.02  6.1e-16  bluestein
      1872  2^4.3^2.13       chain3   replayed        2247       7661    3.31  6.3e-16  
      1873  1873             prime    replayed        7967      10835    1.35  1.1e-15  rader
      1874  2.937            prime    raced          10632      11359    1.02  6.0e-16  bluestein
      1875  3.5^4            flat     replayed        3382       5417    1.58  6.3e-16  
      1876  2^2.7.67         prime    raced          10404      16468    1.55  4.2e-16  bluestein
      1877  1877             prime    replayed       10704      10840    1.01  6.1e-16  bluestein
      1878  2.3.313          prime    raced          10401      10826    1.02  6.4e-16  bluestein
      1879  1879             prime    replayed       10460      10824    1.03  5.3e-16  bluestein
      1880  2^3.5.47         flat     replayed        4315      13240    3.06  5.8e-16  
      1881  3^2.11.19        chain3   replayed        3475       8791    2.15  6.5e-16  
      1882  2.941            prime    raced          10477      10872    1.03  5.0e-16  bluestein
      1883  7.269            prime    replayed       10392      10816    1.04  6.4e-16  bluestein
      1884  2^2.3.157        prime    raced          10651      10857    1.01  6.9e-16  bluestein
      1885  5.13.29          chain3   replayed        4552       8924    1.26  6.9e-16  flips differ 1.55x
      1886  2.23.41          flat     replayed        5293      13164    2.43  1.5e-15  
      1887  3.17.37          chain3   replayed        4333      11162    2.22  6.2e-16  
      1888  2^5.59           prime    raced          10396      15316    1.29  4.8e-16  bluestein
      1889  1889             prime    replayed       10371      10829    1.04  7.7e-16  bluestein
      1890  2.3^3.5.7        chain3   replayed        2788       6245    2.18  5.3e-16  
      1891  31.61            prime    replayed       10633      17161    1.37  6.5e-16  bluestein
      1892  2^2.11.43        chain3   replayed        4049      13137    3.22  4.1e-16  
      1893  3.631            prime    replayed       10724      10851    0.79  5.7e-16  flips differ 1.28x bluestein
      1894  2.947            prime    raced          10411      10869    1.03  5.9e-16  bluestein
      1895  5.379            prime    replayed       10415      10820    1.03  6.4e-16  bluestein
      1896  2^3.3.79         prime    raced          10702      19038    1.78  5.9e-16  bluestein
      1897  7.271            prime    replayed       10396      10827    1.03  5.9e-16  bluestein
      1898  2.13.73          prime    raced          10395      16561    1.59  5.2e-16  bluestein
      1899  3^2.211          prime    replayed       10367      10843    1.02  5.9e-16  bluestein
      1900  2^2.5^2.19       flat     replayed        3341       8661    2.51  5.5e-16  
      1901  1901             prime    replayed       10498      10821    1.02  6.6e-16  bluestein
      1902  2.3.317          prime    raced          10432      10812    1.03  5.9e-16  bluestein
      1903  11.173           prime    replayed       10431      10823    1.03  4.7e-16  bluestein
      1904  2^4.7.17         chain3   replayed        2467       8207    3.08  5.1e-16  
      1905  3.5.127          prime    replayed       10653      10883    1.01  6.0e-16  bluestein
      1906  2.953            prime    raced          10424      10818    1.03  5.2e-16  bluestein
      1907  1907             prime    replayed       10516      10849    1.02  5.3e-16  bluestein
      1908  2^2.3^2.53       prime    raced          10373      14287    1.37  4.6e-16  bluestein
      1909  23.83            prime    replayed       10403      20083    1.89  5.7e-16  bluestein
      1910  2.5.191          prime    raced          10755      10824    1.01  4.9e-16  bluestein
      1911  3.7^2.13         chain3   replayed        3107       6687    2.09  5.8e-16  
      1912  2^3.239          prime    raced          10450      10821    1.03  5.5e-16  bluestein
      1913  1913             prime    replayed       10601      10887    0.81  6.3e-16  flips differ 1.27x bluestein
      1914  2.3.11.29        flat     replayed        4439      10575    2.37  2.1e-15  
      1915  5.383            prime    replayed       10488      10824    1.03  5.8e-16  bluestein
      1916  2^2.479          prime    raced          10446      10819    1.03  6.2e-16  bluestein
      1917  3^3.71           prime    replayed       10437      17314    1.63  6.4e-16  bluestein
      1918  2.7.137          prime    raced          10591      10837    1.02  4.6e-16  bluestein
      1919  19.101           prime    replayed       10388      10844    1.04  6.2e-16  bluestein
      1920  2^7.3.5          chain3   replayed        2087       6685    3.11  4.7e-16  
      1921  17.113           prime    replayed       10593      10864    1.00  5.3e-16  bluestein
      1922  2.31^2           flat     replayed        5045      13457    2.50  2.2e-15  
      1923  3.641            prime    replayed       10470      10836    1.02  5.2e-16  bluestein
      1924  2^2.13.37        chain3   replayed        4007      12249    2.97  5.1e-16  
      1925  5^2.7.11         flat     replayed        3084       6388    2.04  6.7e-16  
      1926  2.3^2.107        prime    raced          10465      10810    1.01  4.8e-16  bluestein
      1927  41.47            flat     raced           5346      16556    3.01  1.7e-15  
      1928  2^3.241          prime    raced          10420      10845    1.02  7.5e-16  bluestein
      1929  3.643            prime    replayed       10654      11369    0.79  5.7e-16  flips differ 1.28x bluestein
      1930  2.5.193          prime    raced          10422      10831    1.03  6.8e-16  bluestein
      1931  1931             prime    replayed       10614      10844    1.01  7.2e-16  bluestein
      1932  2^2.3.7.23       flat     replayed        3528       9686    2.42  5.9e-16  
      1933  1933             prime    replayed       10380      10845    1.04  8.0e-16  bluestein
      1934  2.967            prime    raced          10420      10866    1.04  6.2e-16  bluestein
      1935  3^2.5.43         flat     replayed        4204      12088    2.65  6.2e-16  
      1936  2^4.11^2         chain3   replayed        2322       9332    3.85  3.8e-16  
      1937  13.149           prime    replayed       10433      10882    1.03  6.0e-16  bluestein
      1938  2.3.17.19        chain3   replayed        3697      10350    2.75  6.2e-16  
      1939  7.277            prime    replayed       10416      10843    1.02  6.4e-16  bluestein
      1940  2^2.5.97         prime    raced          10549      22095    2.08  5.7e-16  bluestein
      1941  3.647            prime    replayed       10622      10851    1.02  6.0e-16  bluestein
      1942  2.971            prime    raced          10438      10857    1.03  5.5e-16  bluestein
      1943  29.67            prime    replayed       10582      18300    1.71  7.6e-16  bluestein
      1944  2^3.3^5          chain3   replayed        2957       7161    2.12  5.3e-16  
      1945  5.389            prime    replayed       10583      10838    1.02  6.0e-16  bluestein
      1946  2.7.139          prime    raced          10433      10887    1.04  5.1e-16  bluestein
      1947  3.11.59          prime    replayed       10430      14327    1.34  6.3e-16  bluestein
      1948  2^2.487          prime    raced          10464      10861    1.01  6.6e-16  bluestein
      1949  1949             prime    replayed       10473      10872    1.02  6.7e-16  bluestein
      1950  2.3.5^2.13       chain3   replayed        3137       7455    2.02  5.4e-16  
      1951  1951             prime    replayed       10449      10860    1.02  6.2e-16  bluestein
      1952  2^5.61           prime    raced          10532      16176    1.53  5.9e-16  bluestein
      1953  3^2.7.31         flat     replayed        4461       9668    2.13  7.6e-16  
      1954  2.977            prime    raced          10471      10872    1.02  5.3e-16  bluestein
      1955  5.17.23          chain3   replayed        5327       9456    1.66  6.2e-16  
      1956  2^2.3.163        prime    raced          10635      10839    1.02  6.6e-16  bluestein
      1957  19.103           prime    replayed       10633      10860    1.02  6.2e-16  bluestein
      1958  2.11.89          prime    raced          10465      19702    1.84  6.0e-16  bluestein
      1959  3.653            prime    replayed       10426      10893    1.03  4.9e-16  bluestein
      1960  2^3.5.7^2        flat     replayed        3015       6403    2.12  4.6e-16  
      1961  37.53            prime    replayed       10640      17146    1.60  5.8e-16  bluestein
      1962  2.3^2.109        prime    raced          10447      11023    1.02  6.0e-16  bluestein
      1963  13.151           prime    replayed       10444      10910    1.02  7.2e-16  bluestein
      1964  2^2.491          prime    raced          10513      10859    1.03  5.3e-16  bluestein
      1965  3.5.131          prime    replayed       10483      10911    1.02  6.6e-16  bluestein
      1966  2.983            prime    raced          10628      10904    1.02  5.6e-16  bluestein
      1967  7.281            prime    replayed       10466      10899    1.03  5.0e-16  bluestein
      1968  2^4.3.41         flat     replayed        4063      12523    3.07  6.0e-16  
      1969  11.179           prime    replayed       10462      10896    1.03  5.9e-16  bluestein
      1970  2.5.197          prime    raced          10580      10880    1.02  5.5e-16  bluestein
      1971  3^3.73           prime    replayed       10406      17703    1.65  5.2e-16  bluestein
      1972  2^2.17.29        flat     replayed        4515      12139    2.29  9.9e-16  
      1973  1973             prime    replayed       10465      10947    1.02  8.0e-16  bluestein
      1974  2.3.7.47         chain3   replayed        4515      12980    2.86  5.7e-16  
      1975  5^2.79           prime    replayed       10492      16626    1.57  4.8e-16  bluestein
      1976  2^3.13.19        chain3   replayed        3101      10940    3.42  7.7e-16  
      1977  3.659            prime    replayed       10488      10864    1.02  6.9e-16  bluestein
      1978  2.23.43          flat     replayed        5509      14528    2.59  1.7e-15  
      1979  1979             prime    replayed       10613      10890    1.00  4.5e-16  bluestein
      1980  2^2.3^2.5.11     chain3   replayed        2447       7932    3.23  5.5e-16  
      1981  7.283            prime    replayed       10458      10843    1.03  7.9e-16  bluestein
      1982  2.991            prime    raced          10493      10865    1.02  5.1e-16  bluestein
      1983  3.661            prime    replayed       10496      10867    1.02  6.1e-16  bluestein
      1984  2^6.31           2p       replayed        2905      11450    3.91  5.3e-16  
      1985  5.397            prime    replayed       10455      10867    1.02  5.8e-16  bluestein
      1986  2.3.331          prime    raced          10433      10877    1.02  6.3e-16  bluestein
      1987  1987             prime    replayed       10644      10886    0.92  6.4e-16  bluestein
      1988  2^2.7.71         prime    raced          10440      18447    1.73  4.6e-16  bluestein
      1989  3^2.13.17        chain3   replayed        3665       8871    2.37  6.9e-16  
      1990  2.5.199          prime    raced          10582      10877    1.03  5.2e-16  bluestein
      1991  11.181           prime    replayed       10427      10962    1.04  6.9e-16  bluestein
      1992  2^3.3.83         prime    raced          10544      20595    1.95  5.0e-16  bluestein
      1993  1993             prime    replayed       10558      10880    1.03  7.3e-16  bluestein
      1994  2.997            prime    raced          10486      10900    1.03  4.7e-16  bluestein
      1995  3.5.7.19         chain3   replayed        4160       7474    1.57  5.1e-16  
      1996  2^2.499          prime    raced          10473      10917    1.02  6.3e-16  bluestein
      1997  1997             prime    replayed       10456      11123    1.04  5.9e-16  bluestein
      1998  2.3^3.37         chain3   replayed        5623      11999    2.11  5.8e-16  
      1999  1999             prime    replayed       10626      10863    1.01  5.9e-16  bluestein
      2000  2^4.5^3          chain3   replayed        2761       6633    2.21  4.5e-16  
      2001  3.23.29          chain3   replayed        5664      12107    1.90  6.4e-16  
      2002  2.7.11.13        flat     replayed        4234       8020    1.55  6.5e-16  
      2003  2003             prime    replayed       10582      10907    1.03  5.9e-16  bluestein
      2004  2^2.3.167        prime    raced          10652      10870    1.01  5.2e-16  bluestein
      2005  5.401            prime    replayed       10478      10883    1.02  6.9e-16  bluestein
      2006  2.17.59          prime    raced          10463      16088    1.51  4.8e-16  bluestein
      2007  3^2.223          prime    replayed       10707      10893    1.01  5.6e-16  bluestein
      2008  2^3.251          prime    raced          10450      11802    1.04  5.5e-16  bluestein
      2009  7^2.41           chain3   replayed        6485       9442    1.03  4.7e-16  flips differ 1.41x
      2010  2.3.5.67         prime    raced          10448      17567    1.67  5.3e-16  bluestein
      2011  2011             prime    replayed       10439      10890    1.04  5.8e-16  bluestein
      2012  2^2.503          prime    raced          10501      10875    1.01  6.0e-16  bluestein
      2013  3.11.61          prime    replayed       10474      15216    1.43  5.0e-16  bluestein
      2014  2.19.53          prime    raced          10510      15534    1.47  6.2e-16  bluestein
      2015  5.13.31          flat     replayed        4663      10166    1.93  6.5e-16  
      2016  2^5.3^2.7        chain3   replayed        2213       6989    3.15  5.3e-16  
      2017  2017             prime    replayed       10703      10883    1.02  6.8e-16  bluestein
      2018  2.1009           prime    raced          10436      10884    1.02  6.1e-16  bluestein
      2019  3.673            prime    replayed       10564      11004    1.01  7.2e-16  bluestein
      2020  2^2.5.101        prime    raced          10444      10858    1.03  5.6e-16  bluestein
      2021  43.47            flat     raced           6007      18100    3.01  2.0e-15  
      2022  2.3.337          prime    raced          10499      10871    1.03  5.0e-16  bluestein
      2023  7.17^2           chain3   replayed        3668       8257    2.18  7.8e-16  
      2024  2^3.11.23        chain3   replayed        3285      11789    3.58  5.2e-16  
      2025  3^4.5^2          chain3   replayed        3668       6252    1.69  4.9e-16  
      2026  2.1013           prime    raced          10455      10855    1.04  6.9e-16  bluestein
      2027  2027             prime    replayed       10707      10882    1.01  6.5e-16  bluestein
      2028  2^2.3.13^2       chain3   replayed        2756      10141    3.62  6.3e-16  
      2029  2029             prime    replayed        8835      11021    1.23  9.6e-16  rader
      2030  2.5.7.29         flat     replayed        4976       9329    1.71  5.8e-16  
      2031  3.677            prime    replayed       10750      10899    1.01  6.4e-16  bluestein
      2032  2^4.127          prime    raced          10507      10901    1.02  5.5e-16  bluestein
      2033  19.107           prime    replayed       10632      10844    1.02  6.5e-16  bluestein
      2034  2.3^2.113        prime    raced          10481      10892    1.04  5.9e-16  bluestein
      2035  5.11.37          chain3   replayed        6247      10809    1.25  5.0e-16  flips differ 1.38x
      2036  2^2.509          prime    raced          10495      10910    1.03  7.8e-16  bluestein
      2037  3.7.97           prime    replayed       10496      20097    1.88  6.9e-16  bluestein
      2038  2.1019           prime    raced          10522      10907    1.03  5.5e-16  bluestein
      2039  2039             prime    replayed       10512      10880    1.02  6.1e-16  bluestein
      2040  2^3.3.5.17       chain3   replayed        3045       8870    2.82  6.0e-16  
      2041  13.157           prime    replayed       10540      10979    1.03  6.6e-16  bluestein
      2042  2.1021           prime    raced          10539      10974    1.04  5.4e-16  bluestein
      2043  3^2.227          prime    replayed       10522      10969    1.03  6.6e-16  bluestein
      2044  2^2.7.73         prime    raced          10494      19005    1.57  5.7e-16  bluestein
      2045  5.409            prime    replayed       10645      10986    1.02  5.3e-16  bluestein
      2046  2.3.11.31        chain3   replayed        6327      12056    1.67  5.4e-16  
      2047  23.89            prime    replayed       10689      22406    2.07  5.0e-16  bluestein
      2048  2^11             ztt      replayed        1606       2053    1.25  3.1e-16  
      2049  3.683            prime    replayed       22483      23892    1.05  5.8e-16  bluestein
      2050  2.5^2.41         chain3   replayed        8973      11995    1.23  5.4e-16  
      2051  7.293            prime    replayed       22557      23958    1.06  5.7e-16  bluestein
      2052  2^2.3^3.19       chain3   replayed        2862       9624    3.27  4.6e-16  
      2053  2053             prime    replayed       14370      23905    1.51  9.6e-16  rader
      2054  2.13.79          prime    replayed       22087      19480    0.88  5.4e-16  bluestein
      2055  3.5.137          prime    replayed       21622      24057    1.11  6.9e-16  bluestein
      2056  2^3.257          prime    replayed       22128      24912    1.08  6.4e-16  bluestein
      2057  11^2.17          flat     replayed        3982       8979    2.16  9.5e-16  
      2058  2.3.7^3          flat     replayed        3953       6736    1.66  5.6e-16  
      2059  29.71            prime    replayed       22161      20311    0.91  5.8e-16  bluestein
      2060  2^2.5.103        prime    replayed       21962      23923    1.09  6.7e-16  bluestein
      2061  3^2.229          prime    replayed       22094      23905    1.06  7.8e-16  bluestein
      2062  2.1031           prime    replayed       21594      23889    0.88  5.4e-16  flips differ 1.26x bluestein
      2063  2063             prime    replayed       21533      23944    1.10  4.9e-16  bluestein
      2064  2^4.3.43         flat     replayed        4229      13889    3.27  5.2e-16  
      2065  5.7.59           prime    replayed       21065      13397    0.52  3.4e-16  bluestein
      2066  2.1033           prime    replayed       21560      24062    1.09  6.2e-16  bluestein
      2067  3.13.53          prime    replayed       22478      14207    0.63  5.4e-16  bluestein
      2068  2^2.11.47        chain3   replayed       11013      15203    1.38  5.1e-16  
      2069  2069             prime    replayed       21538      23950    1.08  5.5e-16  bluestein
      2070  2.3^2.5.23       chain3   replayed        5814      10250    1.46  6.3e-16  
      2071  19.109           prime    replayed       21985      23867    1.05  6.3e-16  bluestein
      2072  2^3.7.37         chain3   replayed        6226      12465    1.93  4.9e-16  
      2073  3.691            prime    replayed       21980      23902    1.06  7.0e-16  bluestein
      2074  2.17.61          prime    replayed       21919      16910    0.77  3.8e-16  bluestein
      2075  5^2.83           prime    replayed       21580      18269    0.84  4.9e-16  bluestein
      2076  2^2.3.173        prime    replayed       21958      23914    1.09  5.9e-16  bluestein
      2077  31.67            prime    replayed       21567      20153    0.92  5.3e-16  bluestein
      2078  2.1039           prime    replayed       21594      23898    1.10  5.8e-16  bluestein
      2079  3^3.7.11         chain3   replayed        3186       7561    2.36  7.6e-16  
      2080  2^5.5.13         chain3   replayed        2499       8516    3.38  5.8e-16  
      2081  2081             prime    replayed        9416      23916    2.53  7.5e-16  rader
      2082  2.3.347          prime    replayed       22098      24874    1.08  5.1e-16  bluestein
      2083  2083             prime    replayed       22112      23864    0.85  5.8e-16  flips differ 1.26x bluestein
      2084  2^2.521          prime    replayed       22334      23865    1.07  5.5e-16  bluestein
      2085  3.5.139          prime    replayed       22514      23909    1.06  5.8e-16  bluestein
      2086  2.7.149          prime    replayed       21637      23958    1.08  6.5e-16  bluestein
      2087  2087             prime    replayed       25083      30156    1.11  5.8e-16  bluestein
      2088  2^3.3^2.29       flat     replayed        4038      11352    2.68  9.2e-16  
      2089  2089             prime    replayed       12078      23988    1.76  9.5e-16  rader
      2090  2.5.11.19        chain3   replayed        3282       9896    2.90  5.9e-16  
      2091  3.17.41          chain3   replayed        5094      12997    2.47  6.6e-16  
      2092  2^2.523          prime    replayed       22022      23978    1.06  6.4e-16  bluestein
      2093  7.13.23          flat     replayed        4441       9167    2.02  7.1e-16  
      2094  2.3.349          prime    replayed       21569      23930    1.08  6.3e-16  bluestein
      2095  5.419            prime    replayed       21890      23977    1.07  6.0e-16  bluestein
      2096  2^4.131          prime    replayed       21581      23972    1.08  5.4e-16  bluestein
      2097  3^2.233          prime    replayed       21986      23912    1.09  7.0e-16  bluestein
      2098  2.1049           prime    replayed       21631      23960    1.11  7.4e-16  bluestein
      2099  2099             prime    replayed       22549      23961    1.06  6.5e-16  bluestein
      2100  2^2.3.5^2.7      chain3   replayed        2944       6951    2.32  5.6e-16  
      2101  11.191           prime    replayed       22518      25386    1.06  6.7e-16  bluestein
      2102  2.1051           prime    replayed       21978      23886    1.09  5.5e-16  bluestein
      2103  3.701            prime    replayed       22477      23915    1.05  6.0e-16  bluestein
      2104  2^3.263          prime    replayed       21972      23997    1.09  6.0e-16  bluestein
      2105  5.421            prime    replayed       21996      23948    1.08  4.6e-16  bluestein
      2106  2.3^4.13         chain3   replayed        3984       8358    1.87  6.1e-16  
      2107  7^2.43           chain3   replayed        7233      10713    1.47  5.1e-16  
      2108  2^2.17.31        flat     replayed        5696      13717    2.30  9.6e-16  
      2109  3.19.37          flat     replayed        5091      13129    2.51  7.3e-16  
      2110  2.5.211          prime    replayed       21582      23981    1.08  4.7e-16  bluestein
      2111  2111             prime    replayed       22532      23962    1.06  5.7e-16  bluestein
      2112  2^6.3.11         chain3   replayed        2442       8566    3.44  4.1e-16  
      2113  2113             prime    replayed        8912      23948    2.68  7.5e-16  rader
      2114  2.7.151          prime    replayed       21980      27348    1.09  5.6e-16  bluestein
      2115  3^2.5.47         chain3   replayed        7701      14092    1.30  5.0e-16  flips differ 1.39x
      2116  2^2.23^2         flat     replayed        4870      13860    2.76  1.4e-15  
      2117  29.73            prime    replayed       21581      20868    0.94  4.7e-16  bluestein
      2118  2.3.353          prime    replayed       21995      23952    1.06  5.7e-16  bluestein
      2119  13.163           prime    replayed       21645      23871    1.08  5.2e-16  bluestein
      2120  2^3.5.53         prime    replayed       21880      15756    0.71  6.2e-16  bluestein
      2121  3.7.101          prime    replayed       22565      23878    1.06  6.7e-16  bluestein
      2122  2.1061           prime    replayed       21964      23962    1.07  6.3e-16  bluestein
      2123  11.193           prime    replayed       22554      23957    1.06  7.4e-16  bluestein
      2124  2^2.3^2.59       prime    replayed       21667      17275    0.77  4.7e-16  bluestein
      2125  5^3.17           chain3   replayed        3883       7876    1.94  7.5e-16  
      2126  2.1063           prime    replayed       22082      23923    1.06  6.6e-16  bluestein
      2127  3.709            prime    replayed       22574      24148    1.06  5.3e-16  bluestein
      2128  2^4.7.19         chain3   replayed        2825       9808    3.15  3.7e-16  
      2129  2129             prime    replayed       10465      23906    2.28  7.1e-16  rader
      2130  2.3.5.71         prime    replayed       21984      19580    0.87  5.6e-16  bluestein
      2131  2131             prime    replayed       21628      23872    1.08  6.2e-16  bluestein
      2132  2^2.13.41        chain3   replayed        4570      14242    3.10  5.3e-16  
      2133  3^3.79           prime    replayed       22008      20747    0.92  5.3e-16  bluestein
      2134  2.11.97          prime    replayed       22256      23213    1.04  4.8e-16  bluestein
      2135  5.7.61           prime    replayed       22603      14363    0.51  5.3e-16  bluestein
      2136  2^3.3.89         prime    replayed       21925      22998    1.04  5.0e-16  bluestein
      2137  2137             prime    replayed       22004      23951    1.05  6.6e-16  bluestein
      2138  2.1069           prime    replayed       21643      24020    1.08  5.4e-16  bluestein
      2139  3.23.31          flat     replayed        6750      13518    1.84  9.3e-16  
      2140  2^2.5.107        prime    replayed       22629      23925    0.97  4.6e-16  bluestein
      2141  2141             prime    replayed       22011      23902    1.07  5.5e-16  bluestein
      2142  2.3^2.7.17       chain3   replayed        3470       9198    2.54  6.3e-16  
      2143  2143             prime    replayed       12089      24167    1.99  1.0e-15  rader
      2144  2^5.67           prime    replayed       21641      19125    0.88  5.6e-16  bluestein
      2145  3.5.11.13        chain3   replayed        3442       8638    2.47  5.4e-16  
      2146  2.29.37          flat     replayed        5996      15123    2.51  2.9e-15  
      2147  19.113           prime    replayed       22018      24041    1.08  7.1e-16  bluestein
      2148  2^2.3.179        prime    replayed       21580      25273    1.11  6.2e-16  bluestein
      2149  7.307            prime    replayed       22576      23899    1.05  6.3e-16  bluestein
      2150  2.5^2.43         chain3   replayed        4716      13169    2.79  4.6e-16  
      2151  3^2.239          prime    replayed       22646      23925    1.05  6.1e-16  bluestein
      2152  2^3.269          prime    replayed       22212      24041    1.08  5.4e-16  bluestein
      2153  2153             prime    replayed       22016      24040    1.07  5.8e-16  bluestein
      2154  2.3.359          prime    replayed       22032      24008    1.06  7.2e-16  bluestein
      2155  5.431            prime    replayed       22569      24027    0.88  6.1e-16  bluestein
      2156  2^2.7^2.11       flat     replayed        3291       8323    2.51  7.1e-16  
      2157  3.719            prime    replayed       21632      23947    1.10  5.6e-16  bluestein
      2158  2.13.83          prime    replayed       22625      21036    0.93  7.6e-16  bluestein
      2159  17.127           prime    replayed       21648      23990    1.11  6.1e-16  bluestein
      2160  2^4.3^3.5        chain3   replayed        2508       7552    2.99  5.4e-16  
      2161  2161             prime    replayed        8316      24007    2.87  6.3e-16  rader
      2162  2.23.47          flat     replayed        6225      16771    2.68  1.1e-15  
      2163  3.7.103          prime    replayed       21647      23936    1.08  5.6e-16  bluestein
      2164  2^2.541          prime    replayed       22561      23928    1.06  6.6e-16  bluestein
      2165  5.433            prime    replayed       22212      23960    1.08  5.1e-16  bluestein
      2166  2.3.19^2         chain3   replayed        4018      12325    2.72  7.0e-16  
      2167  11.197           prime    replayed       22379      24001    1.01  5.2e-16  bluestein
      2168  2^3.271          prime    replayed       21682      23976    1.06  6.0e-16  bluestein
      2169  3^2.241          prime    replayed       22492      23936    1.06  6.4e-16  bluestein
      2170  2.5.7.31         chain3   replayed        6293      10899    1.45  5.6e-16  
      2171  13.167           prime    replayed       22185      24007    1.08  5.9e-16  bluestein
      2172  2^2.3.181        prime    replayed       21623      23996    1.10  5.5e-16  bluestein
      2173  41.53            prime    replayed       22615      19564    0.86  6.2e-16  bluestein
      2174  2.1087           prime    replayed       21657      24074    1.10  6.5e-16  bluestein
      2175  3.5^2.29         chain3   replayed        4531      10680    2.25  5.0e-16  
      2176  2^7.17           chain3   replayed        2847       9674    3.29  5.4e-16  
      2177  7.311            prime    replayed       21645      25323    1.11  6.5e-16  bluestein
      2178  2.3^2.11^2       flat     replayed        4650      10331    2.18  7.1e-16  
      2179  2179             prime    replayed       22008      23982    1.07  7.2e-16  bluestein
      2180  2^2.5.109        prime    replayed       21667      23936    1.08  6.1e-16  bluestein
      2181  3.727            prime    replayed       21646      23938    1.09  6.8e-16  bluestein
      2182  2.1091           prime    replayed       22568      23949    1.06  6.2e-16  bluestein
      2183  37.59            prime    replayed       22230      20393    0.89  4.6e-16  bluestein
      2184  2^3.3.7.13       chain3   replayed        3413       8840    2.49  6.0e-16  
      2185  5.19.23          chain3   replayed        4613      11452    2.48  5.1e-16  
      2186  2.1093           prime    replayed       21975      24050    1.07  6.8e-16  bluestein
      2187  3^7              chain3   replayed        3870       7520    1.87  6.0e-16  
      2188  2^2.547          prime    replayed       22591      23947    1.05  5.3e-16  bluestein
      2189  11.199           prime    replayed       22031      25190    1.06  6.9e-16  bluestein
      2190  2.3.5.73         prime    replayed       22001      20032    0.89  4.7e-16  bluestein
      2191  7.313            prime    replayed       22070      23943    1.06  7.5e-16  bluestein
      2192  2^4.137          prime    replayed       21650      23946    0.86  4.5e-16  flips differ 1.28x bluestein
      2193  3.17.43          chain3   replayed        8184      14310    1.27  7.4e-16  flips differ 1.38x
      2194  2.1097           prime    replayed       22345      23949    1.07  6.1e-16  bluestein
      2195  5.439            prime    replayed       22167      23913    1.08  5.7e-16  bluestein
      2196  2^2.3^2.61       prime    replayed       21648      18239    0.82  4.3e-16  bluestein
      2197  13^3             flat     replayed        4000       9249    2.29  1.2e-15  
      2198  2.7.157          prime    replayed       22167      24023    1.08  6.0e-16  bluestein
      2199  3.733            prime    replayed       22021      24041    1.08  7.6e-16  bluestein
      2200  2^3.5^2.11       flat     replayed        3355       8635    2.55  5.3e-16  
      2201  31.71            prime    replayed       22125      22341    1.01  5.4e-16  bluestein
      2202  2.3.367          prime    replayed       22167      23974    1.06  5.8e-16  bluestein
      2203  2203             prime    replayed       21872      24032    1.09  6.4e-16  bluestein
      2204  2^2.19.29        flat     replayed        5277      14205    2.68  8.4e-16  
      2205  3^2.5.7^2        chain3   replayed        3642       6821    1.87  5.3e-16  
      2206  2.1103           prime    replayed       22513      24078    1.06  5.6e-16  bluestein
      2207  2207             prime    replayed       21601      23930    1.08  5.8e-16  bluestein
      2208  2^5.3.23         chain3   replayed        3071      11307    3.64  4.6e-16  
      2209  47^2             flat     replayed        6319      20685    3.08  1.9e-15  
      2210  2.5.13.17        chain3   replayed        3793      10000    2.61  6.4e-16  
      2211  3.11.67          prime    replayed       21755      18183    0.83  5.2e-16  bluestein
      2212  2^2.7.79         prime    replayed       22025      22196    1.00  4.5e-16  bluestein
      2213  2213             prime    replayed       22562      23932    1.05  5.7e-16  bluestein
      2214  2.3^3.41         chain3   replayed        7763      13912    1.54  4.8e-16  
      2215  5.443            prime    replayed       21842      25371    1.06  5.7e-16  bluestein
      2216  2^3.277          prime    replayed       21637      23972    1.08  6.2e-16  bluestein
      2217  3.739            prime    replayed       21737      25957    0.85  5.1e-16  flips differ 1.50x bluestein
      2218  2.1109           prime    replayed       22029      24147    1.08  7.2e-16  bluestein
      2219  7.317            prime    replayed       22623      24198    1.07  7.0e-16  bluestein
      2220  2^2.3.5.37       chain3   replayed        4008      13476    3.30  4.3e-16  
      2221  2221             prime    replayed       15944      23990    1.28  1.7e-15  rader
      2222  2.11.101         prime    replayed       22590      23974    1.05  5.1e-16  bluestein
      2223  3^2.13.19        flat     replayed        4077      10847    2.56  1.2e-15  
      2224  2^4.139          prime    replayed       22367      23950    1.07  5.5e-16  bluestein
      2225  5^2.89           prime    replayed       22168      20398    0.66  4.3e-16  flips differ 1.39x bluestein
      2226  2.3.7.53         prime    replayed       22032      15576    0.70  5.9e-16  bluestein
      2227  17.131           prime    replayed       21688      23991    1.07  7.1e-16  bluestein
      2228  2^2.557          prime    replayed       22041      23982    1.08  7.2e-16  bluestein
      2229  3.743            prime    replayed       22043      23950    1.08  7.3e-16  bluestein
      2230  2.5.223          prime    replayed       22637      24102    1.06  5.2e-16  bluestein
      2231  23.97            prime    replayed       22611      26147    1.15  5.2e-16  bluestein
      2232  2^3.3^2.31       flat     replayed        5472      12814    2.22  5.7e-16  
      2233  7.11.29          chain3   replayed        4536      10379    2.25  6.3e-16  
      2234  2.1117           prime    replayed       22262      23976    1.00  5.7e-16  bluestein
      2235  3.5.149          prime    replayed       22484      23985    1.06  4.5e-16  bluestein
      2236  2^2.13.43        chain3   replayed        4964      15718    3.13  6.1e-16  
      2237  2237             prime    replayed       22070      23995    1.06  6.7e-16  bluestein
      2238  2.3.373          prime    replayed       22201      24091    1.08  5.9e-16  bluestein
      2239  2239             prime    replayed       22449      24055    1.07  6.0e-16  bluestein
      2240  2^6.5.7          ztt      replayed        2040       7594    3.70  3.4e-16  
      2241  3^3.83           prime    replayed       22385      22289    0.99  3.9e-16  bluestein
      2242  2.19.59          prime    replayed       22641      18932    0.80  4.9e-16  bluestein
      2243  2243             prime    replayed       22103      24062    1.08  6.2e-16  bluestein
      2244  2^2.3.11.17      chain3   replayed        3148      11673    3.69  5.2e-16  
      2245  5.449            prime    replayed       22388      24115    1.07  5.0e-16  bluestein
      2246  2.1123           prime    replayed       22561      24068    0.89  5.4e-16  bluestein
      2247  3.7.107          prime    replayed       21691      24001    1.08  5.2e-16  bluestein
      2248  2^3.281          prime    replayed       22125      23981    1.08  6.2e-16  bluestein
      2249  13.173           prime    replayed       21662      23999    1.11  7.3e-16  bluestein
      2250  2.3^2.5^3        chain3   replayed        3293       7468    2.18  7.3e-16  
      2251  2251             prime    replayed       12770      25030    1.87  1.1e-15  rader
      2252  2^2.563          prime    replayed       22300      24049    1.07  5.0e-16  bluestein
      2253  3.751            prime    replayed       21665      24062    1.06  7.1e-16  bluestein
      2254  2.7^2.23         flat     replayed        5450       9518    1.74  6.2e-16  
      2255  5.11.41          chain3   replayed        5350      12646    2.35  6.0e-16  
      2256  2^4.3.47         chain3   replayed        4715      16128    3.40  4.9e-16  
      2257  37.61            prime    replayed       21696      21360    0.98  5.7e-16  bluestein
      2258  2.1129           prime    replayed       22351      26782    1.07  4.9e-16  bluestein
      2259  3^2.251          prime    replayed       21816      23893    1.09  6.3e-16  bluestein
      2260  2^2.5.113        prime    replayed       22170      23964    1.05  6.5e-16  bluestein
      2261  7.17.19          flat     replayed        4482      10205    2.17  9.9e-16  
      2262  2.3.13.29        chain3   replayed        5320      12835    2.41  7.9e-16  
      2263  31.73            prime    replayed       21693      23065    1.03  6.5e-16  bluestein
      2264  2^3.283          prime    replayed       22229      24061    0.91  5.3e-16  bluestein
      2265  3.5.151          prime    replayed       22019      24012    1.06  6.2e-16  bluestein
      2266  2.11.103         prime    replayed       22079      24069    1.06  6.4e-16  bluestein
      2267  2267             prime    replayed       22537      24014    0.92  6.2e-16  bluestein
      2268  2^2.3^4.7        chain3   replayed        2895       7840    2.71  5.0e-16  
      2269  2269             prime    replayed       14962      24027    1.55  8.5e-16  rader
      2270  2.5.227          prime    replayed       21698      24055    1.10  5.9e-16  bluestein
      2271  3.757            prime    replayed       21749      24017    1.07  5.5e-16  bluestein
      2272  2^5.71           prime    replayed       22078      21341    0.93  4.6e-16  bluestein
      2273  2273             prime    replayed       21966      23982    1.07  6.7e-16  bluestein
      2274  2.3.379          prime    replayed       22085      24044    1.07  6.5e-16  bluestein
      2275  5^2.7.13         chain3   replayed        3909       7852    1.84  5.9e-16  
      2276  2^2.569          prime    replayed       22252      24176    1.08  6.8e-16  bluestein
      2277  3^2.11.23        chain3   replayed        4349      11701    2.64  6.3e-16  
      2278  2.17.67          prime    replayed       22046      20162    0.91  7.1e-16  bluestein
      2279  43.53            prime    replayed       22184      21257    0.94  5.3e-16  bluestein
      2280  2^3.3.5.19       chain3   replayed        3111      10891    3.49  5.0e-16  
      2281  2281             prime    replayed       12508      24006    1.83  7.8e-16  rader
      2282  2.7.163          prime    replayed       22638      24013    1.06  5.9e-16  bluestein
      2283  3.761            prime    replayed       22278      24047    1.07  5.6e-16  bluestein
      2284  2^2.571          prime    replayed       22419      24006    1.07  6.3e-16  bluestein
      2285  5.457            prime    replayed       21796      24065    1.08  5.4e-16  bluestein
      2286  2.3^2.127        prime    replayed       21720      24129    1.08  4.8e-16  bluestein
      2287  2287             prime    replayed       21810      24112    1.08  6.4e-16  bluestein
      2288  2^4.11.13        chain3   replayed        2978      11270    3.76  6.2e-16  
      2289  3.7.109          prime    replayed       22093      23996    1.07  8.1e-16  bluestein
      2290  2.5.229          prime    replayed       21717      24014    1.08  6.4e-16  bluestein
      2291  29.79            prime    replayed       21727      24429    1.10  5.8e-16  bluestein
      2292  2^2.3.191        prime    replayed       22933      23971    1.04  5.2e-16  bluestein
      2293  2293             prime    replayed       21672      24025    1.08  5.6e-16  bluestein
      2294  2.31.37          flat     replayed        6618      17110    2.56  1.9e-15  
      2295  3^3.5.17         chain3   replayed        4129       9348    2.24  6.9e-16  
      2296  2^3.7.41         flat     replayed        5082      14514    2.84  6.8e-16  
      2297  2297             prime    replayed       23033      24011    0.87  5.2e-16  bluestein
      2298  2.3.383          prime    replayed       22111      26036    0.88  5.3e-16  flips differ 1.44x bluestein
      2299  11^2.19          flat     replayed        4574      10792    2.22  1.2e-15  
      2300  2^2.5^2.23       chain3   replayed        3679      11423    3.08  6.0e-16  
      2301  3.13.59          prime    replayed       22153      17229    0.66  5.4e-16  bluestein
      2302  2.1151           prime    replayed       22242      24155    1.07  7.2e-16  bluestein
      2303  7^2.47           chain3   replayed       11398      12567    1.10  6.0e-16  
      2304  2^8.3^2          ztt      replayed        2175       8707    3.99  4.5e-16  
      2305  5.461            prime    replayed       21680      23999    1.10  5.2e-16  bluestein
      2306  2.1153           prime    replayed       22825      24084    1.04  4.8e-16  bluestein
      2307  3.769            prime    replayed       21690      24054    1.08  5.1e-16  bluestein
      2308  2^2.577          prime    replayed       21824      24052    1.08  6.3e-16  bluestein
      2309  2309             prime    replayed       21714      23956    1.10  7.1e-16  bluestein
      2310  2.3.5.7.11       chain3   replayed        3556       8828    2.33  5.3e-16  
      2311  2311             prime    replayed       12369      23945    1.89  1.0e-15  rader
      2312  2^3.17^2         chain3   replayed        4572      13501    2.38  7.8e-16  
      2313  3^2.257          prime    replayed       22999      24052    0.90  6.7e-16  bluestein
      2314  2.13.89          prime    replayed       22230      23517    1.05  5.4e-16  bluestein
      2315  5.463            prime    replayed       22027      24009    1.05  6.2e-16  bluestein
      2316  2^2.3.193        prime    replayed       21648      23981    1.10  5.9e-16  bluestein
      2317  7.331            prime    replayed       21605      23969    1.08  5.5e-16  bluestein
      2318  2.19.61          prime    replayed       22303      19739    0.88  4.7e-16  bluestein
      2319  3.773            prime    replayed       22556      24042    1.06  5.0e-16  bluestein
      2320  2^4.5.29         flat     replayed        4830      12655    2.56  4.6e-16  
      2321  11.211           prime    replayed       22706      24039    1.05  5.0e-16  bluestein
      2322  2.3^3.43         chain3   replayed        7963      15474    1.90  5.0e-16  
      2323  23.101           prime    replayed       21830      24010    0.75  5.9e-16  flips differ 1.46x bluestein
      2324  2^2.7.83         prime    replayed       22161      23895    1.02  4.7e-16  bluestein
      2325  3.5^2.31         chain3   replayed        6739      12297    1.47  5.7e-16  
      2326  2.1163           prime    replayed       22136      24035    1.08  6.5e-16  bluestein
      2327  13.179           prime    replayed       22247      24080    1.08  5.4e-16  bluestein
      2328  2^3.3.97         prime    replayed       22116      26671    1.18  5.3e-16  bluestein
      2329  17.137           prime    replayed       22070      23981    1.06  5.7e-16  bluestein
      2330  2.5.233          prime    replayed       21667      23976    1.08  6.1e-16  bluestein
      2331  3^2.7.37         chain3   replayed        8790      12882    1.35  5.4e-16  
      2332  2^2.11.53        prime    replayed       22210      18045    0.81  4.1e-16  bluestein
      2333  2333             prime    replayed       21692      25477    1.08  5.7e-16  bluestein
      2334  2.3.389          prime    replayed       22333      23991    1.07  5.6e-16  bluestein
      2335  5.467            prime    replayed       22351      24012    0.85  5.1e-16  flips differ 1.26x bluestein
      2336  2^5.73           prime    replayed       22816      22137    0.96  6.0e-16  bluestein
      2337  3.19.41          chain3   replayed        6111      15427    2.04  7.5e-16  
      2338  2.7.167          prime    replayed       22274      24013    1.06  6.7e-16  bluestein
      2339  2339             prime    replayed       21779      25390    1.07  5.7e-16  bluestein
      2340  2^2.3^2.5.13     chain3   replayed        3023       9605    3.17  7.6e-16  
      2341  2341             prime    replayed       11039      24024    2.17  9.9e-16  rader
      2342  2.1171           prime    replayed       22178      24009    1.06  5.7e-16  bluestein
      2343  3.11.71          prime    replayed       22266      20277    0.91  4.9e-16  bluestein
      2344  2^3.293          prime    replayed       22268      24147    1.07  7.0e-16  bluestein
      2345  5.7.67           prime    replayed       22570      17188    0.75  4.8e-16  bluestein
      2346  2.3.17.23        chain3   replayed        4476      13619    2.94  6.4e-16  
      2347  2347             prime    replayed       15922      25401    1.51  1.0e-15  rader
      2348  2^2.587          prime    replayed       21680      24006    1.08  6.7e-16  bluestein
      2349  3^4.29           chain3   replayed        5604      12113    1.58  6.6e-16  flips differ 1.37x
      2350  2.5^2.47         chain3   replayed        8657      15371    1.18  5.0e-16  flips differ 1.49x
      2351  2351             prime    replayed       22213      24004    1.08  5.9e-16  bluestein
      2352  2^4.3.7^2        ztt      replayed        2259       7966    3.51  4.2e-16  
      2353  13.181           prime    replayed       21684      24008    1.08  5.6e-16  bluestein
      2354  2.11.107         prime    replayed       22422      24000    1.06  5.4e-16  bluestein
      2355  3.5.157          prime    replayed       22302      25462    1.00  6.7e-16  bluestein
      2356  2^2.19.31        chain3   replayed        5343      16114    2.97  5.4e-16  
      2357  2357             prime    replayed       21774      23981    1.07  6.4e-16  bluestein
      2358  2.3^2.131        prime    replayed       22334      24014    1.06  6.7e-16  bluestein
      2359  7.337            prime    replayed       22128      24068    1.08  4.8e-16  bluestein
      2360  2^3.5.59         prime    replayed       22229      19033    0.85  4.4e-16  bluestein
      2361  3.787            prime    replayed       21772      24037    1.08  6.9e-16  bluestein
      2362  2.1181           prime    replayed       22081      23991    1.06  5.6e-16  bluestein
      2363  17.139           prime    replayed       22446      25395    1.07  6.5e-16  bluestein
      2364  2^2.3.197        prime    replayed       22213      24191    1.07  5.5e-16  bluestein
      2365  5.11.43          chain3   replayed        8448      14265    1.27  5.6e-16  flips differ 1.32x
      2366  2.7.13^2         flat     replayed        5434       9899    1.59  1.1e-15  
      2367  3^2.263          prime    replayed       21717      23972    1.07  6.7e-16  bluestein
      2368  2^6.37           2p       replayed        8388      14628    1.62  4.8e-16  
      2369  23.103           prime    replayed       22059      23988    1.08  5.0e-16  bluestein
      2370  2.3.5.79         prime    replayed       22545      23437    1.03  5.6e-16  bluestein
      2371  2371             prime    replayed       21730      24021    1.07  6.5e-16  bluestein
      2372  2^2.593          prime    replayed       22202      23965    1.05  6.6e-16  bluestein
      2373  3.7.113          prime    replayed       22085      23973    1.08  6.4e-16  bluestein
      2374  2.1187           prime    replayed       22064      23995    1.09  5.6e-16  bluestein
      2375  5^3.19           flat     replayed        4006       9573    2.37  6.8e-16  
      2376  2^3.3^3.11       chain3   replayed        4150       9544    2.25  4.7e-16  
      2377  2377             prime    replayed       16025      24031    1.44  9.2e-16  rader
      2378  2.29.41          flat     replayed        6778      17574    2.55  2.5e-15  
      2379  3.13.61          prime    replayed       21742      18308    0.84  5.7e-16  bluestein
      2380  2^2.5.7.17       flat     replayed        4751      10227    2.07  4.9e-16  
      2381  2381             prime    replayed       21710      24090    1.11  7.4e-16  bluestein
      2382  2.3.397          prime    replayed       22618      24043    0.95  6.7e-16  bluestein
      2383  2383             prime    replayed       22185      24089    1.08  6.2e-16  bluestein
      2384  2^4.149          prime    replayed       21792      23980    1.07  6.4e-16  bluestein
      2385  3^2.5.53         prime    replayed       22530      17051    0.74  5.3e-16  bluestein
      2386  2.1193           prime    replayed       22234      24100    1.05  5.6e-16  bluestein
      2387  7.11.31          chain3   replayed        8599      11924    1.36  4.7e-16  
      2388  2^2.3.199        prime    replayed       22687      24008    1.06  5.3e-16  bluestein
      2389  2389             prime    replayed       21842      24881    1.10  6.4e-16  bluestein
      2390  2.5.239          prime    replayed       21759      24055    1.07  4.8e-16  bluestein
      2391  3.797            prime    replayed       22138      23959    1.06  6.4e-16  bluestein
      2392  2^3.13.23        chain3   replayed        4657      14225    3.00  5.9e-16  
      2393  2393             prime    replayed       15023      24040    1.60  9.7e-16  rader
      2394  2.3^2.7.19       chain3   replayed        4437      10959    2.46  5.4e-16  
      2395  5.479            prime    replayed       22334      24031    1.07  5.1e-16  bluestein
      2396  2^2.599          prime    replayed       22955      24080    0.76  5.8e-16  flips differ 1.39x bluestein
      2397  3.17.47          flat     replayed        6988      16806    2.27  1.6e-15  
      2398  2.11.109         prime    replayed       21807      24052    1.07  4.9e-16  bluestein
      2399  2399             prime    replayed       21856      24064    1.07  5.6e-16  bluestein
      2400  2^5.3.5^2        ztt      replayed        2415       8246    3.38  3.8e-16  
      2401  7^4              flat     replayed        4561       6188    1.35  6.8e-16  
      2402  2.1201           prime    replayed       21833      25886    1.10  6.4e-16  bluestein
      2403  3^3.89           prime    replayed       22366      25128    1.12  4.9e-16  bluestein
      2404  2^2.601          prime    replayed       22851      24020    1.04  7.2e-16  bluestein
      2405  5.13.37          chain3   replayed        5360      13361    2.25  6.3e-16  
      2406  2.3.401          prime    replayed       22087      24136    1.07  5.6e-16  bluestein
      2407  29.83            prime    replayed       22107      26147    1.16  8.1e-16  bluestein
      2408  2^3.7.43         chain3   replayed        4740      15924    3.35  4.8e-16  
      2409  3.11.73          prime    replayed       22152      20809    0.92  4.7e-16  bluestein
      2410  2.5.241          prime    replayed       22242      24019    1.05  6.6e-16  bluestein
      2411  2411             prime    replayed       21452      24046    1.10  6.5e-16  bluestein
      2412  2^2.3^2.67       prime    replayed       22761      21698    0.75  5.3e-16  flips differ 1.27x bluestein
      2413  19.127           prime    replayed       22625      24135    1.06  5.1e-16  bluestein
      2414  2.17.71          prime    replayed       21932      22479    1.02  5.7e-16  bluestein
      2415  3.5.7.23         chain3   replayed        5712      10357    1.61  6.2e-16  
      2416  2^4.151          prime    replayed       22248      24043    1.07  6.0e-16  bluestein
      2417  2417             prime    replayed       21389      25493    1.11  5.7e-16  bluestein
      2418  2.3.13.31        flat     replayed        6155      14675    2.30  1.1e-15  
      2419  41.59            prime    replayed       22251      23455    1.03  5.0e-16  bluestein
      2420  2^2.5.11^2       flat     replayed        4016      11569    2.88  6.6e-16  
      2421  3^2.269          prime    replayed       22569      26060    0.90  6.5e-16  bluestein
      2422  2.7.173          prime    replayed       22336      24037    1.07  6.8e-16  bluestein
      2423  2423             prime    replayed       22323      24077    1.06  5.6e-16  bluestein
      2424  2^3.3.101        prime    replayed       22107      25510    1.09  6.7e-16  bluestein
      2425  5^2.97           prime    replayed       21722      24194    1.09  6.3e-16  bluestein
      2426  2.1213           prime    replayed       22316      24124    1.06  5.8e-16  bluestein
      2427  3.809            prime    replayed       21717      24045    1.08  5.9e-16  bluestein
      2428  2^2.607          prime    replayed       22955      26954    1.05  4.8e-16  bluestein
      2429  7.347            prime    replayed       22182      24092    1.08  5.9e-16  bluestein
      2430  2.3^5.5          chain3   replayed        4410       8392    1.65  5.4e-16  
      2431  11.13.17         chain3   replayed        4548      11053    2.33  6.0e-16  
      2432  2^7.19           chain3   replayed        3524      11590    3.15  3.9e-16  
      2433  3.811            prime    replayed       21750      23971    1.10  5.7e-16  bluestein
      2434  2.1217           prime    replayed       22116      24059    1.06  5.5e-16  bluestein
      2435  5.487            prime    replayed       22701      24018    1.06  5.8e-16  bluestein
      2436  2^2.3.7.29       flat     replayed        5203      13170    2.47  5.9e-16  
      2437  2437             prime    replayed       15969      24088    1.35  8.5e-16  rader
      2438  2.23.53          prime    replayed       21736      19962    0.92  4.8e-16  bluestein
      2439  3^2.271          prime    replayed       21729      24036    1.10  5.1e-16  bluestein
      2440  2^3.5.61         prime    replayed       21719      20033    0.92  4.7e-16  bluestein
      2441  2441             prime    replayed       22291      24804    1.08  6.3e-16  bluestein
      2442  2.3.11.37        chain3   replayed        5074      15339    2.94  4.9e-16  
      2443  7.349            prime    replayed       22290      23988    1.06  6.0e-16  bluestein
      2444  2^2.13.47        flat     replayed        6023      18232    3.01  8.8e-16  
      2445  3.5.163          prime    replayed       21703      27380    1.08  6.3e-16  bluestein
      2446  2.1223           prime    replayed       21678      24020    1.11  5.0e-16  bluestein
      2447  2447             prime    replayed       21615      27453    1.11  6.3e-16  bluestein
      2448  2^4.3^2.17       chain3   replayed        3228      10820    3.31  5.8e-16  
      2449  31.79            prime    replayed       22473      26673    1.18  4.7e-16  bluestein
      2450  2.5^2.7^2        flat     replayed        4630       7611    1.63  4.6e-16  
      2451  3.19.43          flat     replayed        7200      16959    2.32  1.4e-15  
      2452  2^2.613          prime    replayed       21706      28231    0.90  6.3e-16  flips differ 1.52x bluestein
      2453  11.223           prime    replayed       22065      24004    1.06  6.2e-16  bluestein
      2454  2.3.409          prime    replayed       22622      24047    1.04  6.1e-16  bluestein
      2455  5.491            prime    replayed       22984      26867    0.88  5.7e-16  flips differ 1.46x bluestein
      2456  2^3.307          prime    replayed       21726      24064    1.08  6.7e-16  bluestein
      2457  3^3.7.13         chain3   replayed        4376       9236    2.10  8.3e-16  
      2458  2.1229           prime    replayed       22290      24021    1.07  6.1e-16  bluestein
      2459  2459             prime    replayed       22074      24030    1.09  5.8e-16  bluestein
      2460  2^2.3.5.41       flat     replayed        5279      15678    2.96  6.0e-16  
      2461  23.107           prime    replayed       22068      24045    1.04  4.7e-16  bluestein
      2462  2.1231           prime    replayed       21743      23997    1.07  6.9e-16  bluestein
      2463  3.821            prime    replayed       21734      24033    1.08  5.7e-16  bluestein
      2464  2^5.7.11         chain3   replayed        2940       9887    3.35  4.4e-16  
      2465  5.17.29          flat     replayed        6456      12952    1.98  1.6e-15  
      2466  2.3^2.137        prime    replayed       22636      24044    1.04  5.1e-16  bluestein
      2467  2467             prime    replayed       21792      24154    1.08  6.2e-16  bluestein
      2468  2^2.617          prime    replayed       22645      24127    1.06  5.9e-16  bluestein
      2469  3.823            prime    replayed       22263      23990    1.06  6.8e-16  bluestein
      2470  2.5.13.19        chain3   replayed        5463      12086    2.20  7.2e-16  
      2471  7.353            prime    replayed       22124      24257    1.08  6.3e-16  bluestein
      2472  2^3.3.103        prime    replayed       22384      24057    1.07  5.8e-16  bluestein
      2473  2473             prime    replayed       21444      24089    1.11  7.0e-16  bluestein
      2474  2.1237           prime    replayed       22095      27040    0.89  8.0e-16  flips differ 1.51x bluestein
      2475  3^2.5^2.11       chain3   replayed        4117       8882    2.14  7.2e-16  
      2476  2^2.619          prime    replayed       21749      25747    1.10  5.9e-16  bluestein
      2477  2477             prime    replayed       21834      24198    1.09  6.3e-16  bluestein
      2478  2.3.7.59         prime    replayed       21707      18892    0.87  6.9e-16  bluestein
      2479  37.67            prime    replayed       21722      25184    1.16  7.1e-16  bluestein
      2480  2^4.5.31         flat     replayed        4846      14212    2.71  4.3e-16  
      2481  3.827            prime    replayed       21739      24065    1.06  6.7e-16  bluestein
      2482  2.17.73          prime    replayed       21435      23171    1.07  4.2e-16  bluestein
      2483  13.191           prime    replayed       22158      23996    1.04  5.8e-16  bluestein
      2484  2^2.3^3.23       chain3   replayed        3644      12718    3.42  5.9e-16  
      2485  5.7.71           prime    replayed       21821      19288    0.87  5.7e-16  bluestein
      2486  2.11.113         prime    replayed       21753      24096    1.08  5.0e-16  bluestein
      2487  3.829            prime    replayed       21757      24059    1.08  4.9e-16  bluestein
      2488  2^3.311          prime    replayed       22106      24016    1.06  6.2e-16  bluestein
      2489  19.131           prime    replayed       21758      25377    1.08  6.0e-16  bluestein
      2490  2.3.5.83         prime    replayed       22299      25411    1.13  5.6e-16  bluestein
      2491  47.53            prime    replayed       22101      24298    1.07  6.9e-16  bluestein
      2492  2^2.7.89         prime    replayed       22102      26773    1.21  4.2e-16  bluestein
      2493  3^2.277          prime    replayed       21704      24058    1.08  6.4e-16  bluestein
      2494  2.29.43          flat     replayed        7553      19183    2.49  2.9e-15  
      2495  5.499            prime    replayed       22332      25432    0.89  5.9e-16  bluestein
      2496  2^6.3.13         chain3   replayed        2958      10394    3.40  6.1e-16  
      2497  11.227           prime    replayed       22315      24101    0.93  5.1e-16  bluestein
      2498  2.1249           prime    replayed       21708      24003    1.08  5.5e-16  bluestein
      2499  3.7^2.17         chain3   replayed        4088       9545    2.15  6.6e-16  
      2500  2^2.5^4          chain3   replayed        3666       8255    2.22  5.6e-16  
      2501  41.61            prime    replayed       22082      24498    1.10  4.9e-16  bluestein
      2502  2.3^2.139        prime    replayed       21768      24116    1.10  5.4e-16  bluestein
      2503  2503             prime    replayed       21767      27493    1.11  6.4e-16  bluestein
      2504  2^3.313          prime    replayed       21732      24017    1.10  6.3e-16  bluestein
      2505  3.5.167          prime    replayed       22071      24028    1.08  5.8e-16  bluestein
      2506  2.7.179          prime    replayed       22328      24038    1.06  5.3e-16  bluestein
      2507  23.109           prime    replayed       22313      24061    1.08  5.8e-16  bluestein
      2508  2^2.3.11.19      flat     replayed        4847      13972    2.43  1.4e-15  
      2509  13.193           prime    replayed       21797      25049    1.10  6.3e-16  bluestein
      2510  2.5.251          prime    replayed       21762      24108    1.08  5.4e-16  bluestein
      2511  3^4.31           flat     replayed        5643      13596    2.04  7.5e-16  
      2512  2^4.157          prime    replayed       21749      24091    1.08  5.0e-16  bluestein
      2513  7.359            prime    replayed       22305      25464    1.01  4.6e-16  bluestein
      2514  2.3.419          prime    replayed       21752      24165    1.09  6.8e-16  bluestein
      2515  5.503            prime    replayed       21774      24073    1.08  7.0e-16  bluestein
      2516  2^2.17.37        chain3   replayed        5579      17444    2.74  7.0e-16  
      2517  3.839            prime    replayed       22242      24122    1.08  6.6e-16  bluestein
      2518  2.1259           prime    replayed       22218      24116    1.08  5.4e-16  bluestein
      2519  11.229           prime    replayed       22704      24132    1.06  5.7e-16  bluestein
      2520  2^3.3^2.5.7      chain3   replayed        3172       8496    2.64  6.5e-16  
      2521  2521             prime    replayed       14116      24268    1.71  9.0e-16  rader
      2522  2.13.97          prime    replayed       21799      27497    1.06  5.4e-16  bluestein
      2523  3.29^2           flat     replayed        6509      16028    2.37  1.4e-15  
      2524  2^2.631          prime    replayed       22513      24075    1.05  6.9e-16  bluestein
      2525  5^2.101          prime    replayed       22328      25555    1.08  8.0e-16  bluestein
      2526  2.3.421          prime    replayed       22679      24016    1.06  5.7e-16  bluestein
      2527  7.19^2           chain3   replayed        5089      12105    2.09  5.3e-16  
      2528  2^5.79           prime    replayed       22666      25565    1.11  5.0e-16  bluestein
      2529  3^2.281          prime    replayed       21769      24054    1.10  7.0e-16  bluestein
      2530  2.5.11.23        chain3   replayed        4064      13129    3.22  6.9e-16  
      2531  2531             prime    replayed       14359      24170    1.68  1.1e-15  rader
      2532  2^2.3.211        prime    replayed       22220      24066    1.08  6.4e-16  bluestein
      2533  17.149           prime    replayed       22323      24036    1.06  6.8e-16  bluestein
      2534  2.7.181          prime    replayed       22238      24089    1.06  5.2e-16  bluestein
      2535  3.5.13^2         chain3   replayed        4588      10512    2.29  7.3e-16  
      2536  2^3.317          prime    replayed       22424      24102    1.07  5.8e-16  bluestein
      2537  43.59            prime    replayed       22080      25345    1.13  4.8e-16  bluestein
      2538  2.3^3.47         chain3   replayed       12054      17983    1.49  5.2e-16  
      2539  2539             prime    replayed       21768      24075    1.07  6.2e-16  bluestein
      2540  2^2.5.127        prime    replayed       22520      24080    1.06  4.7e-16  bluestein
      2541  3.7.11^2         flat     replayed        5333      10809    1.84  7.6e-16  
      2542  2.31.41          flat     replayed        7734      19481    2.51  2.6e-15  
      2543  2543             prime    replayed       22659      24137    1.06  5.9e-16  bluestein
      2544  2^4.3.53         prime    replayed       22027      19305    0.70  5.6e-16  flips differ 1.26x bluestein
      2545  5.509            prime    replayed       22171      24059    1.08  7.2e-16  bluestein
      2546  2.19.67          prime    replayed       22132      23283    1.02  4.4e-16  bluestein
      2547  3^2.283          prime    replayed       22704      24197    1.06  6.2e-16  bluestein
      2548  2^2.7^2.13       flat     replayed        3921      10185    2.46  6.3e-16  
      2549  2549             prime    replayed       21770      24115    1.08  6.5e-16  bluestein
      2550  2.3.5^2.17       chain3   replayed        3806      10673    2.69  6.8e-16  
      2551  2551             prime    replayed       13633      24081    1.75  1.0e-15  rader
      2552  2^3.11.29        chain3   replayed        4497      15818    3.44  4.8e-16  
      2553  3.23.37          flat     replayed        6709      17088    2.44  8.7e-16  
      2554  2.1277           prime    replayed       22723      24121    1.06  6.2e-16  bluestein
      2555  5.7.73           prime    replayed       22373      20020    0.61  4.4e-16  flips differ 1.47x bluestein
      2556  2^2.3^2.71       prime    replayed       22170      24137    1.06  4.4e-16  bluestein
      2557  2557             prime    replayed       21861      25466    0.88  5.4e-16  flips differ 1.26x bluestein
      2558  2.1279           prime    replayed       21791      24103    1.08  6.5e-16  bluestein
      2559  3.853            prime    replayed       22279      24039    1.08  5.9e-16  bluestein
      2560  2^9.5            ztt      replayed        2185       9278    4.22  3.6e-16  
      2561  13.197           prime    replayed       22126      27459    0.91  6.0e-16  bluestein
      2562  2.3.7.61         prime    replayed       22145      20044    0.89  4.3e-16  bluestein
      2563  11.233           prime    replayed       22161      24133    1.09  5.2e-16  bluestein
      2564  2^2.641          prime    replayed       22138      24074    1.06  5.7e-16  bluestein
      2565  3^3.5.19         chain3   replayed        4725      11147    2.00  8.1e-16  
      2566  2.1283           prime    replayed       21808      24051    1.10  6.7e-16  bluestein
      2567  17.151           prime    replayed       21810      24103    1.08  6.9e-16  bluestein
      2568  2^3.3.107        prime    replayed       22624      24061    1.06  6.6e-16  bluestein
      2569  7.367            prime    replayed       22179      24042    1.06  4.9e-16  bluestein
      2570  2.5.257          prime    replayed       22093      24038    1.08  4.8e-16  bluestein
      2571  3.857            prime    replayed       22152      24075    1.06  6.3e-16  bluestein
      2572  2^2.643          prime    replayed       22670      24036    1.06  7.3e-16  bluestein
      2573  31.83            prime    replayed       22201      28793    1.03  7.1e-16  flips differ 1.26x bluestein
      2574  2.3^2.11.13      flat     replayed        6114      12515    1.94  7.3e-16  
      2575  5^2.103          prime    replayed       21711      24078    1.06  6.3e-16  bluestein
      2576  2^4.7.23         flat     replayed        4270      13160    2.03  6.5e-16  flips differ 1.51x
      2577  3.859            prime    replayed       22173      24060    1.08  5.4e-16  bluestein
      2578  2.1289           prime    replayed       21717      24133    1.06  5.3e-16  bluestein
      2579  2579             prime    replayed       22148      24069    1.06  6.6e-16  bluestein
      2580  2^2.3.5.43       flat     replayed        5694      17368    2.93  5.6e-16  
      2581  29.89            prime    replayed       22563      29195    1.28  6.4e-16  bluestein
      2582  2.1291           prime    replayed       22578      24067    1.06  6.6e-16  bluestein
      2583  3^2.7.41         chain3   replayed        5994      14679    2.44  6.2e-16  
      2584  2^3.17.19        chain3   replayed        4803      15725    3.26  6.1e-16  
      2585  5.11.47          chain3   replayed       12549      16614    1.28  5.3e-16  
      2586  2.3.431          prime    replayed       22386      24049    1.07  5.0e-16  bluestein
      2587  13.199           prime    replayed       22111      24024    1.08  7.8e-16  bluestein
      2588  2^2.647          prime    replayed       22359      26779    1.07  5.4e-16  bluestein
      2589  3.863            prime    replayed       22184      27484    1.07  6.3e-16  bluestein
      2590  2.5.7.37         flat     replayed        7048      13991    1.96  6.4e-16  
      2591  2591             prime    replayed       22084      24038    1.07  7.1e-16  bluestein
      2592  2^5.3^4          ztt      replayed        2600       9857    2.94  4.0e-16  flips differ 1.29x
      2593  2593             prime    replayed        9891      25465    2.43  6.3e-16  rader
      2594  2.1297           prime    replayed       22073      24134    1.06  6.4e-16  bluestein
      2595  3.5.173          prime    replayed       22334      24075    1.07  6.9e-16  bluestein
      2596  2^2.11.59        prime    replayed       21819      21861    0.98  4.5e-16  bluestein
      2597  7^2.53           prime    replayed       22624      15321    0.67  5.5e-16  bluestein
      2598  2.3.433          prime    replayed       22719      24027    1.06  4.4e-16  bluestein
      2599  23.113           prime    replayed       22703      24244    1.06  6.2e-16  bluestein
      2600  2^3.5^2.13       flat     replayed        3901      10530    2.66  7.5e-16  
      2601  3^2.17^2         chain3   replayed        4957      13029    2.22  7.7e-16  
      2602  2.1301           prime    replayed       22523      24064    1.06  5.1e-16  bluestein
      2603  19.137           prime    replayed       21759      24071    1.08  5.9e-16  bluestein
      2604  2^2.3.7.31       chain3   replayed        4265      14900    3.47  6.1e-16  
      2605  5.521            prime    replayed       22169      25837    1.08  7.1e-16  bluestein
      2606  2.1303           prime    replayed       22629      25631    1.08  5.4e-16  bluestein
      2607  3.11.79          prime    replayed       22162      24464    1.10  5.1e-16  bluestein
      2608  2^4.163          prime    replayed       22298      25502    1.08  6.2e-16  bluestein
      2609  2609             prime    replayed       22172      24113    1.06  6.6e-16  bluestein
      2610  2.3^2.5.29       flat     replayed        6227      13936    2.23  7.2e-16  
      2611  7.373            prime    replayed       21761      24114    1.07  7.7e-16  bluestein
      2612  2^2.653          prime    replayed       22147      24053    1.06  6.1e-16  bluestein
      2613  3.13.67          prime    replayed       22127      21806    0.96  4.9e-16  bluestein
      2614  2.1307           prime    replayed       22137      24047    1.07  6.1e-16  bluestein
      2615  5.523            prime    replayed       22198      24007    1.05  7.6e-16  bluestein
      2616  2^3.3.109        prime    replayed       22730      24045    1.05  5.6e-16  bluestein
      2617  2617             prime    replayed       22729      24043    1.06  6.0e-16  bluestein
      2618  2.7.11.17        flat     replayed        5629      11492    1.98  8.8e-16  
      2619  3^3.97           prime    replayed       22198      29255    1.29  6.8e-16  bluestein
      2620  2^2.5.131        prime    replayed       22170      24193    1.09  7.9e-16  bluestein
      2621  2621             prime    replayed       22139      24060    1.06  5.7e-16  bluestein
      2622  2.3.19.23        flat     replayed        6494      15961    2.37  1.6e-15  
      2623  43.61            prime    replayed       22214      26627    1.17  5.3e-16  bluestein
      2624  2^6.41           2p       replayed        6942      16880    2.41  4.5e-16  
      2625  3.5^3.7          flat     replayed        4805       7690    1.58  9.2e-16  
      2626  2.13.101         prime    replayed       22219      24114    1.06  5.1e-16  bluestein
      2627  37.71            prime    replayed       22735      27796    1.22  7.1e-16  bluestein
      2628  2^2.3^2.73       prime    replayed       22734      24602    1.08  5.0e-16  bluestein
      2629  11.239           prime    replayed       21830      24044    1.10  6.3e-16  bluestein
      2630  2.5.263          prime    replayed       22074      24129    1.08  5.7e-16  bluestein
      2631  3.877            prime    replayed       22249      24177    1.08  5.2e-16  bluestein
      2632  2^3.7.47         flat     replayed        6088      18624    2.43  6.3e-16  flips differ 1.26x
      2633  2633             prime    replayed       22369      24038    1.05  6.1e-16  bluestein
      2634  2.3.439          prime    replayed       22440      24095    1.06  6.2e-16  bluestein
      2635  5.17.31          flat     replayed        7945      14572    1.82  8.0e-16  
      2636  2^2.659          prime    replayed       22705      24015    1.05  5.6e-16  bluestein
      2637  3^2.293          prime    replayed       21858      24040    1.08  6.0e-16  bluestein
      2638  2.1319           prime    replayed       22263      25751    1.07  5.5e-16  bluestein
      2639  7.13.29          chain3   replayed        5385      12475    2.27  7.5e-16  
      2640  2^4.3.5.11       chain3   replayed        3297      10632    3.15  4.4e-16  
      2641  19.139           prime    replayed       22236      24003    1.06  6.1e-16  bluestein
      2642  2.1321           prime    replayed       21838      24072    1.07  7.2e-16  bluestein
      2643  3.881            prime    replayed       22758      24025    0.97  5.9e-16  bluestein
      2644  2^2.661          prime    replayed       22341      24028    1.06  5.3e-16  bluestein
      2645  5.23^2           flat     replayed        6265      14891    2.36  1.7e-15  
      2646  2.3^3.7^2        chain3   replayed        4741       8985    1.87  6.0e-16  
      2647  2647             prime    replayed       15202      24076    1.54  1.1e-15  rader
      2648  2^3.331          prime    replayed       21841      24038    1.07  6.7e-16  bluestein
      2649  3.883            prime    replayed       22376      24085    1.07  6.3e-16  bluestein
      2650  2.5^2.53         prime    replayed       21921      18509    0.83  4.6e-16  bluestein
      2651  11.241           prime    replayed       22604      26070    1.06  6.6e-16  bluestein
      2652  2^2.3.13.17      chain3   replayed        4486      14257    3.06  7.6e-16  
      2653  7.379            prime    replayed       21807      24072    1.07  6.4e-16  bluestein
      2654  2.1327           prime    replayed       22821      24162    1.05  5.7e-16  bluestein
      2655  3^2.5.59         prime    replayed       22216      20462    0.90  7.4e-16  bluestein
      2656  2^5.83           prime    replayed       21784      27750    1.27  5.3e-16  bluestein
      2657  2657             prime    replayed       22368      24104    1.08  5.9e-16  bluestein
      2658  2.3.443          prime    replayed       22228      25792    1.08  7.6e-16  bluestein
      2659  2659             prime    replayed       22212      25078    1.08  5.9e-16  bluestein
      2660  2^2.5.7.19       flat     replayed        4662      12274    2.62  5.5e-16  
      2661  3.887            prime    replayed       22204      24027    1.05  8.0e-16  bluestein
      2662  2.11^3           flat     replayed        6440      13788    2.13  7.0e-16  
      2663  2663             prime    replayed       22787      24031    1.05  6.5e-16  bluestein
      2664  2^3.3^2.37       flat     replayed        5379      16348    2.90  7.6e-16  
      2665  5.13.41          chain3   replayed        6148      15328    2.49  7.8e-16  
      2666  2.31.43          flat     replayed        7783      21368    2.74  3.7e-15  
      2667  3.7.127          prime    replayed       22572      24082    1.06  5.1e-16  bluestein
      2668  2^2.23.29        chain3   replayed        5695      18598    3.20  4.7e-16  
      2669  17.157           prime    replayed       22410      24076    1.07  5.4e-16  bluestein
      2670  2.3.5.89         prime    replayed       21794      28433    1.27  6.0e-16  bluestein
      2671  2671             prime    replayed       22193      24192    1.05  7.4e-16  bluestein
      2672  2^4.167          prime    replayed       22324      24032    1.07  6.2e-16  bluestein
      2673  3^5.11           chain3   replayed        4727       9654    2.03  7.9e-16  
      2674  2.7.191          prime    replayed       22376      24102    1.07  6.4e-16  bluestein
      2675  5^2.107          prime    replayed       22805      24034    1.02  5.9e-16  bluestein
      2676  2^2.3.223        prime    replayed       22837      25645    0.82  8.4e-16  flips differ 1.44x bluestein
      2677  2677             prime    replayed       22149      24074    1.08  5.9e-16  bluestein
      2678  2.13.103         prime    replayed       22749      24076    1.05  4.0e-16  bluestein
      2679  3.19.47          chain3   replayed       14593      19676    1.24  5.9e-16  
      2680  2^3.5.67         prime    replayed       21839      23809    1.09  4.9e-16  bluestein
      2681  7.383            prime    replayed       22356      24166    1.08  5.8e-16  bluestein
      2682  2.3^2.149        prime    replayed       22647      24075    1.05  5.9e-16  bluestein
      2683  2683             prime    replayed       21861      25315    1.10  6.2e-16  bluestein
      2684  2^2.11.61        prime    replayed       28107      22930    0.73  4.3e-16  bluestein
      2685  3.5.179          prime    replayed       22426      24075    0.84  6.0e-16  flips differ 1.27x bluestein
      2686  2.17.79          prime    replayed       21816      26968    1.23  5.7e-16  bluestein
      2687  2687             prime    replayed       21798      24075    1.10  6.2e-16  bluestein
      2688  2^7.3.7          ztt      replayed        2517       9556    2.90  5.0e-16  flips differ 1.30x
      2689  2689             prime    replayed        9761      24147    2.47  5.5e-16  rader
      2690  2.5.269          prime    replayed       21831      24087    1.08  6.6e-16  bluestein
      2691  3^2.13.23        chain3   replayed        5005      14053    2.77  6.2e-16  
      2692  2^2.673          prime    replayed       21862      24078    1.08  8.1e-16  bluestein
      2693  2693             prime    replayed       22351      24124    1.08  6.2e-16  bluestein
      2694  2.3.449          prime    replayed       21821      24219    1.08  7.5e-16  bluestein
      2695  5.7^2.11         flat     replayed        4650       8875    1.86  5.5e-16  
      2696  2^3.337          prime    replayed       22284      24090    1.06  6.5e-16  bluestein
      2697  3.29.31          flat     replayed        7080      17998    2.38  2.8e-15  
      2698  2.19.71          prime    replayed       21824      25987    1.18  5.0e-16  bluestein
      2699  2699             prime    replayed       22718      24094    1.05  6.7e-16  bluestein
      2700  2^2.3^3.5^2      flat     replayed        4031       9347    2.32  4.7e-16  
      2701  37.73            prime    replayed       22234      28638    1.28  4.7e-16  bluestein
      2702  2.7.193          prime    replayed       21865      25612    1.11  7.0e-16  bluestein
      2703  3.17.53          prime    replayed       21780      19929    0.90  5.8e-16  bluestein
      2704  2^4.13^2         chain3   replayed        3538      13699    3.85  7.4e-16  
      2705  5.541            prime    replayed       21771      24077    1.08  5.0e-16  bluestein
      2706  2.3.11.41        flat     replayed        7977      17741    1.88  8.0e-16  
      2707  2707             prime    replayed       21831      24114    1.07  5.6e-16  bluestein
      2708  2^2.677          prime    replayed       21764      24076    1.07  6.4e-16  bluestein
      2709  3^2.7.43         chain3   replayed       12238      16234    1.25  6.0e-16  
      2710  2.5.271          prime    replayed       22312      24301    1.09  6.2e-16  bluestein
      2711  2711             prime    replayed       22146      24108    1.06  5.7e-16  bluestein
      2712  2^3.3.113        prime    replayed       22159      24142    1.09  5.2e-16  bluestein
      2713  2713             prime    replayed       22718      24164    1.05  5.6e-16  bluestein
      2714  2.23.59          prime    replayed       21785      24030    1.10  5.5e-16  bluestein
      2715  3.5.181          prime    replayed       22185      24091    1.08  5.7e-16  bluestein
      2716  2^2.7.97         prime    replayed       21844      31254    1.43  6.3e-16  bluestein
      2717  11.13.19         flat     replayed        5257      13113    1.90  8.5e-16  flips differ 1.32x
      2718  2.3^2.151        prime    replayed       22410      24082    1.07  6.8e-16  bluestein
      2719  2719             prime    replayed       22192      24144    0.88  6.4e-16  bluestein
      2720  2^5.5.17         chain3   replayed        3540      12253    3.40  5.5e-16  
      2721  3.907            prime    replayed       21836      24036    1.07  6.6e-16  bluestein
      2722  2.1361           prime    replayed       22355      24098    0.86  7.0e-16  flips differ 1.26x bluestein
      2723  7.389            prime    replayed       22326      24070    1.07  5.7e-16  bluestein
      2724  2^2.3.227        prime    replayed       21450      24846    1.12  5.5e-16  bluestein
      2725  5^2.109          prime    replayed       21843      24083    0.85  5.8e-16  flips differ 1.30x bluestein
      2726  2.29.47          flat     replayed        8578      22148    2.55  3.3e-15  
      2727  3^3.101          prime    replayed       22453      24026    1.07  5.5e-16  bluestein
      2728  2^3.11.31        flat     replayed        7185      17860    2.32  7.6e-16  
      2729  2729             prime    replayed       22354      24070    1.08  6.0e-16  bluestein
      2730  2.3.5.7.13       flat     replayed        5525      10829    1.94  6.5e-16  
      2731  2731             prime    replayed       14319      24086    1.60  9.1e-16  rader
      2732  2^2.683          prime    replayed       22148      24024    1.08  6.3e-16  bluestein
      2733  3.911            prime    replayed       22775      24024    1.05  5.6e-16  bluestein
      2734  2.1367           prime    replayed       22453      24199    0.77  6.6e-16  flips differ 1.40x bluestein
      2735  5.547            prime    replayed       22358      24103    1.08  7.5e-16  bluestein
      2736  2^4.3^2.19       chain3   replayed        3878      13106    3.28  5.3e-16  
      2737  7.17.23          flat     replayed        6046      13254    2.06  1.4e-15  
      2738  2.37^2           flat     replayed        7970      21384    2.63  3.5e-15  
      2739  3.11.83          prime    replayed       22194      26462    1.16  4.8e-16  bluestein
      2740  2^2.5.137        prime    replayed       22149      24149    1.08  5.5e-16  bluestein
      2741  2741             prime    replayed       22216      26894    0.89  6.7e-16  flips differ 1.51x bluestein
      2742  2.3.457          prime    replayed       22666      24122    1.06  6.4e-16  bluestein
      2743  13.211           prime    replayed       21823      24134    0.78  6.6e-16  flips differ 1.42x bluestein
      2744  2^3.7^3          flat     replayed        4530       9220    2.03  5.1e-16  
      2745  3^2.5.61         prime    replayed       22740      21636    0.95  4.5e-16  bluestein
      2746  2.1373           prime    replayed       21848      24114    1.10  5.3e-16  bluestein
      2747  41.67            prime    replayed       22425      28708    1.28  6.2e-16  bluestein
      2748  2^2.3.229        prime    replayed       22202      24155    1.09  5.7e-16  bluestein
      2749  2749             prime    replayed       22704      24055    1.06  6.0e-16  bluestein
      2750  2.5^3.11         chain3   replayed        4214      10015    2.30  5.5e-16  
      2751  3.7.131          prime    replayed       22126      24056    1.05  5.6e-16  bluestein
      2752  2^6.43           2p       replayed        4939      18766    3.78  5.0e-16  
      2753  2753             prime    replayed       22153      24304    1.09  6.0e-16  bluestein
      2754  2.3^4.17         chain3   replayed        5709      11936    2.07  6.2e-16  
      2755  5.19.29          chain3   replayed        9124      15367    1.65  6.8e-16  
      2756  2^2.13.53        prime    replayed       21815      21684    0.97  4.7e-16  bluestein
      2757  3.919            prime    replayed       22402      24110    1.07  5.7e-16  bluestein
      2758  2.7.197          prime    replayed       22202      24115    1.08  6.3e-16  bluestein
      2759  31.89            prime    replayed       22349      32063    1.43  5.2e-16  bluestein
      2760  2^3.3.5.23       chain3   replayed        4008      14126    3.51  5.8e-16  
      2761  11.251           prime    replayed       22808      24185    1.05  6.5e-16  bluestein
      2762  2.1381           prime    replayed       22651      24097    1.05  6.0e-16  bluestein
      2763  3^2.307          prime    replayed       22825      24088    0.83  6.1e-16  flips differ 1.27x bluestein
      2764  2^2.691          prime    replayed       22507      24149    1.06  8.1e-16  bluestein
      2765  5.7.79           prime    replayed       21867      23570    1.05  4.5e-16  bluestein
      2766  2.3.461          prime    replayed       23027      24102    1.04  5.3e-16  bluestein
      2767  2767             prime    replayed       21961      25495    0.78  5.9e-16  flips differ 1.42x bluestein
      2768  2^4.173          prime    replayed       21838      24282    1.09  6.4e-16  bluestein
      2769  3.13.71          prime    replayed       22265      24257    1.06  6.3e-16  bluestein
      2770  2.5.277          prime    replayed       23133      24117    1.04  6.5e-16  bluestein
      2771  17.163           prime    replayed       22342      24072    1.05  6.8e-16  bluestein
      2772  2^2.3^2.7.11     chain3   replayed        3811      11284    2.95  4.9e-16  
      2773  47.59            prime    replayed       22252      28896    1.27  6.0e-16  bluestein
      2774  2.19.73          prime    replayed       21853      26667    1.21  4.9e-16  bluestein
      2775  3.5^2.37         chain3   replayed        8569      15612    1.30  7.1e-16  flips differ 1.40x
      2776  2^3.347          prime    replayed       22732      24264    0.84  5.7e-16  flips differ 1.27x bluestein
      2777  2777             prime    replayed       21852      24143    1.08  5.8e-16  bluestein
      2778  2.3.463          prime    replayed       21842      24113    1.08  6.0e-16  bluestein
      2779  7.397            prime    replayed       22362      25152    1.06  6.0e-16  bluestein
      2780  2^2.5.139        prime    replayed       22413      24153    1.07  5.7e-16  bluestein
      2781  3^3.103          prime    replayed       21501      24081    1.10  5.5e-16  bluestein
      2782  2.13.107         prime    replayed       22492      25501    0.81  6.6e-16  flips differ 1.48x bluestein
      2783  11^2.23          flat     replayed        5893      14386    2.35  1.4e-15  
      2784  2^5.3.29         flat     replayed        5531      15426    2.32  5.6e-16  
      2785  5.557            prime    replayed       22105      24230    1.05  5.5e-16  bluestein
      2786  2.7.199          prime    replayed       22756      24146    1.06  6.3e-16  bluestein
      2787  3.929            prime    replayed       22449      24089    1.07  6.0e-16  bluestein
      2788  2^2.17.41        chain3   replayed        6404      20108    3.07  6.6e-16  
      2789  2789             prime    replayed       22630      24088    1.06  6.5e-16  bluestein
      2790  2.3^2.5.31       chain3   replayed        5044      15801    3.12  4.9e-16  
      2791  2791             prime    replayed       21884      24216    1.09  6.6e-16  bluestein
      2792  2^3.349          prime    replayed       21875      24074    1.10  6.5e-16  bluestein
      2793  3.7^2.19         chain3   replayed        4972      11557    2.25  6.3e-16  
      2794  2.11.127         prime    replayed       22769      24079    1.06  6.0e-16  bluestein
      2795  5.13.43          chain3   replayed        6912      16945    2.32  5.9e-16  
      2796  2^2.3.233        prime    replayed       22491      25498    1.07  5.9e-16  bluestein
      2797  2797             prime    replayed       21826      24266    1.11  6.2e-16  bluestein
      2798  2.1399           prime    replayed       22366      24154    1.08  6.9e-16  bluestein
      2799  3^2.311          prime    replayed       22818      24105    0.83  6.6e-16  flips differ 1.27x bluestein
      2800  2^4.5^2.7        ztt      replayed        2713       9474    3.46  4.6e-16  
      2801  2801             prime    replayed       10464      24251    2.31  5.9e-16  rader
      2802  2.3.467          prime    replayed       22442      24128    1.07  4.3e-16  bluestein
      2803  2803             prime    replayed       22245      24120    1.08  5.1e-16  bluestein
      2804  2^2.701          prime    replayed       22250      24155    1.06  5.8e-16  bluestein
      2805  3.5.11.17        chain3   replayed        5053      12363    2.29  6.0e-16  
      2806  2.23.61          prime    replayed       22257      25167    1.10  6.3e-16  bluestein
      2807  7.401            prime    replayed       22404      24142    1.07  6.9e-16  bluestein
      2808  2^3.3^3.13       flat     replayed        5040      11748    2.31  1.2e-15  
      2809  53^2             prime    replayed       22199      28671    1.29  6.0e-16  bluestein
      2810  2.5.281          prime    replayed       22414      24091    1.07  7.3e-16  bluestein
      2811  3.937            prime    replayed       22500      24170    1.07  5.2e-16  bluestein
      2812  2^2.19.37        chain3   replayed        6337      20407    3.20  5.1e-16  
      2813  29.97            prime    replayed       22412      33874    1.51  5.9e-16  bluestein
      2814  2.3.7.67         prime    replayed       21847      24116    1.06  6.4e-16  bluestein
      2815  5.563            prime    replayed       22299      24139    1.08  6.3e-16  bluestein
      2816  2^8.11           chain3   replayed        3989      11648    2.90  4.5e-16  
      2817  3^2.313          prime    replayed       22779      24256    1.06  4.8e-16  bluestein
      2818  2.1409           prime    replayed       22236      24114    1.06  7.1e-16  bluestein
      2819  2819             prime    replayed       23033      25440    1.05  5.9e-16  bluestein
      2820  2^2.3.5.47       chain3   replayed        5843      20164    3.45  6.0e-16  
      2821  7.13.31          chain3   replayed        5985      14324    2.36  6.5e-16  
      2822  2.17.83          prime    replayed       22599      29148    1.29  4.9e-16  bluestein
      2823  3.941            prime    replayed       22404      24146    1.08  5.5e-16  bluestein
      2824  2^3.353          prime    replayed       22444      24154    1.07  6.2e-16  bluestein
      2825  5^2.113          prime    replayed       22439      24133    0.89  5.8e-16  bluestein
      2826  2.3^2.157        prime    replayed       22556      25076    1.06  6.5e-16  bluestein
      2827  11.257           prime    replayed       22223      24146    0.77  5.3e-16  flips differ 1.42x bluestein
      2828  2^2.7.101        prime    replayed       22725      24186    1.06  5.0e-16  bluestein
      2829  3.23.41          flat     replayed        7804      19800    2.48  8.5e-16  
      2830  2.5.283          prime    replayed       21819      24163    1.08  5.0e-16  bluestein
      2831  19.149           prime    replayed       22374      24119    1.06  6.5e-16  bluestein
      2832  2^4.3.59         prime    replayed       22208      23299    1.03  5.8e-16  bluestein
      2833  2833             prime    replayed       22673      27500    1.04  5.5e-16  bluestein
      2834  2.13.109         prime    replayed       22455      24130    1.05  5.2e-16  bluestein
      2835  3^4.5.7          chain3   replayed        5055       8769    1.46  7.2e-16  
      2836  2^2.709          prime    replayed       21850      24138    1.08  5.8e-16  bluestein
      2837  2837             prime    replayed       22376      25000    1.09  6.5e-16  bluestein
      2838  2.3.11.43        chain3   replayed        6643      19598    2.93  5.2e-16  
      2839  17.167           prime    replayed       21802      25433    1.11  6.1e-16  bluestein
      2840  2^3.5.71         prime    replayed       22199      26547    0.82  5.7e-16  flips differ 1.46x bluestein
      2841  3.947            prime    replayed       21817      24148    1.06  5.7e-16  bluestein
      2842  2.7^2.29         flat     replayed        6948      13097    1.88  2.0e-15  
      2843  2843             prime    replayed       21824      24147    1.08  6.5e-16  bluestein
      2844  2^2.3^2.79       prime    replayed       22134      28873    1.27  5.3e-16  bluestein
      2845  5.569            prime    replayed       22175      24125    1.05  5.8e-16  bluestein
      2846  2.1423           prime    replayed       22177      24174    1.07  6.9e-16  bluestein
      2847  3.13.73          prime    replayed       22147      25039    1.13  5.1e-16  bluestein
      2848  2^5.89           prime    replayed       22195      31030    1.35  4.5e-16  bluestein
      2849  7.11.37          flat     replayed        6238      15301    2.40  6.0e-16  
      2850  2.3.5^2.19       flat     replayed        6237      13118    1.47  5.3e-16  flips differ 1.43x
      2851  2851             prime    replayed       15547      24163    1.54  9.2e-16  rader
      2852  2^2.23.31        chain3   replayed        6173      20754    3.35  6.4e-16  
      2853  3^2.317          prime    replayed       22239      24155    1.06  6.1e-16  bluestein
      2854  2.1427           prime    replayed       22209      24091    1.06  6.8e-16  bluestein
      2855  5.571            prime    replayed       21837      24119    0.89  6.1e-16  bluestein
      2856  2^3.3.7.17       chain3   replayed        4616      12622    2.69  6.3e-16  
      2857  2857             prime    replayed       16090      24194    1.46  1.1e-15  rader
      2858  2.1429           prime    replayed       22457      24134    1.07  5.0e-16  bluestein
      2859  3.953            prime    replayed       22208      24088    1.04  6.3e-16  bluestein
      2860  2^2.5.11.13      flat     replayed        4694      13941    2.89  7.2e-16  
      2861  2861             prime    replayed       21856      24183    0.93  6.4e-16  bluestein
      2862  2.3^3.53         prime    replayed       21807      21413    0.98  6.7e-16  bluestein
      2863  7.409            prime    replayed       23956      24136    0.99  7.9e-16  bluestein
      2864  2^4.179          prime    replayed       22279      24132    1.05  6.1e-16  bluestein
      2865  3.5.191          prime    replayed       22228      24231    1.08  7.0e-16  bluestein
      2866  2.1433           prime    replayed       22867      24208    1.04  7.5e-16  bluestein
      2867  47.61            prime    replayed       22713      30286    1.33  5.6e-16  bluestein
      2868  2^2.3.239        prime    replayed       22214      24293    1.08  6.1e-16  bluestein
      2869  19.151           prime    replayed       21882      24128    1.08  6.3e-16  bluestein
      2870  2.5.7.41         flat     replayed        8137      16236    1.98  5.9e-16  
      2871  3^2.11.29        flat     replayed        6126      15846    2.55  8.2e-16  
      2872  2^3.359          prime    replayed       21870      25462    1.11  6.4e-16  bluestein
      2873  13^2.17          chain3   replayed        5569      13167    2.36  8.2e-16  
      2874  2.3.479          prime    replayed       22388      24111    0.86  7.8e-16  flips differ 1.26x bluestein
      2875  5^3.23           chain3   replayed        6158      12958    2.04  6.7e-16  
      2876  2^2.719          prime    replayed       22754      24250    0.83  5.3e-16  flips differ 1.28x bluestein
      2877  3.7.137          prime    replayed       22254      24145    1.06  5.7e-16  bluestein
      2878  2.1439           prime    replayed       22442      27525    1.08  6.0e-16  bluestein
      2879  2879             prime    replayed       22308      24114    1.07  6.7e-16  bluestein
      2880  2^6.3^2.5        ztt      replayed        3054      10571    3.36  4.3e-16  
      2881  43.67            prime    replayed       21906      31063    1.39  5.2e-16  bluestein
      2882  2.11.131         prime    replayed       21852      24190    1.08  6.2e-16  bluestein
      2883  3.31^2           flat     replayed        7805      20259    2.56  1.9e-15  
      2884  2^2.7.103        prime    replayed       22827      24116    1.06  4.9e-16  bluestein
      2885  5.577            prime    replayed       22276      24107    1.05  5.9e-16  bluestein
      2886  2.3.13.37        chain3   replayed        5982      18523    3.08  6.3e-16  
      2887  2887             prime    replayed       22779      24121    1.05  6.4e-16  bluestein
      2888  2^3.19^2         flat     replayed        7065      18570    2.44  1.8e-15  
      2889  3^3.107          prime    replayed       22552      24105    1.05  6.1e-16  bluestein
      2890  2.5.17^2         chain3   replayed        5351      14550    2.70  6.0e-16  
      2891  7^2.59           prime    replayed       22229      18792    0.82  6.7e-16  bluestein
      2892  2^2.3.241        prime    replayed       21909      24279    1.08  6.7e-16  bluestein
      2893  11.263           prime    replayed       21849      24121    1.08  6.6e-16  bluestein
      2894  2.1447           prime    replayed       22883      24114    1.05  6.2e-16  bluestein
      2895  3.5.193          prime    replayed       22515      24159    1.06  6.7e-16  bluestein
      2896  2^4.181          prime    replayed       22298      24260    1.08  5.9e-16  bluestein
      2897  2897             prime    replayed       21896      24183    1.07  6.4e-16  bluestein
      2898  2.3^2.7.23       flat     replayed        6700      14507    2.07  7.0e-16  
      2899  13.223           prime    replayed       21864      24195    1.07  7.0e-16  bluestein
      2900  2^2.5^2.29       flat     replayed        6554      15571    2.14  4.8e-16  
      2901  3.967            prime    replayed       22962      27463    1.05  6.5e-16  bluestein
      2902  2.1451           prime    replayed       22308      24114    1.07  5.6e-16  bluestein
      2903  2903             prime    replayed       22815      26938    0.89  7.6e-16  flips differ 1.47x bluestein
      2904  2^3.3.11^2       flat     replayed        5122      14221    2.72  7.8e-16  
      2905  5.7.83           prime    replayed       21946      25575    1.14  5.8e-16  bluestein
      2906  2.1453           prime    replayed       22369      24146    1.08  6.0e-16  bluestein
      2907  3^2.17.19        flat     replayed        6918      15459    2.02  1.6e-15  
      2908  2^2.727          prime    replayed       22777      24285    1.06  8.9e-16  bluestein
      2909  2909             prime    replayed       21889      24142    1.10  6.2e-16  bluestein
      2910  2.3.5.97         prime    replayed       22306      33205    1.45  6.0e-16  bluestein
      2911  41.71            prime    replayed       22241      31904    1.40  6.2e-16  bluestein
      2912  2^5.7.13         chain3   replayed        3536      12201    3.29  5.6e-16  
      2913  3.971            prime    replayed       22522      24087    1.06  5.8e-16  bluestein
      2914  2.31.47          flat     replayed        8909      24544    2.75  1.9e-15  
      2915  5.11.53          prime    replayed       21883      19766    0.88  6.4e-16  bluestein
      2916  2^2.3^6          chain3   replayed        4015      11298    2.78  7.0e-16  
      2917  2917             prime    replayed       21909      24128    0.77  6.9e-16  flips differ 1.42x bluestein
      2918  2.1459           prime    replayed       21926      24175    1.06  7.5e-16  bluestein
      2919  3.7.139          prime    replayed       22336      24153    1.05  5.7e-16  bluestein
      2920  2^3.5.73         prime    replayed       21388      27358    1.27  4.9e-16  bluestein
      2921  23.127           prime    replayed       22820      26814    0.88  7.4e-16  flips differ 1.47x bluestein
      2922  2.3.487          prime    replayed       22402      25155    1.08  6.1e-16  bluestein
      2923  37.79            prime    replayed       22236      33077    1.46  6.5e-16  bluestein
      2924  2^2.17.43        chain3   replayed        6953      22108    3.10  7.2e-16  
      2925  3^2.5^2.13       chain3   replayed        5190      10890    2.08  6.6e-16  
      2926  2.7.11.19        flat     replayed        7163      13926    1.93  1.4e-15  
      2927  2927             prime    replayed       22442      24116    1.07  6.9e-16  bluestein
      2928  2^4.3.61         prime    replayed       22608      24548    1.08  4.6e-16  bluestein
      2929  29.101           prime    replayed       21824      27519    1.10  5.9e-16  bluestein
      2930  2.5.293          prime    replayed       22511      24243    1.08  7.1e-16  bluestein
      2931  3.977            prime    replayed       22511      24123    1.07  5.8e-16  bluestein
      2932  2^2.733          prime    replayed       21905      24168    1.08  7.2e-16  bluestein
      2933  7.419            prime    replayed       22275      24145    1.06  6.0e-16  bluestein
      2934  2.3^2.163        prime    replayed       22451      24131    1.07  5.3e-16  bluestein
      2935  5.587            prime    replayed       21858      27556    1.08  9.5e-16  bluestein
      2936  2^3.367          prime    replayed       21871      24120    1.08  7.0e-16  bluestein
      2937  3.11.89          prime    replayed       21878      29808    1.33  3.7e-16  bluestein
      2938  2.13.113         prime    replayed       22389      24211    1.08  7.6e-16  bluestein
      2939  2939             prime    replayed       22840      24132    1.06  6.0e-16  bluestein
      2940  2^2.3.5.7^2      flat     replayed        4644      10099    2.14  4.9e-16  
      2941  17.173           prime    replayed       22246      24903    1.06  6.3e-16  bluestein
      2942  2.1471           prime    replayed       22217      24107    1.06  6.2e-16  bluestein
      2943  3^3.109          prime    replayed       22318      25581    1.06  6.1e-16  bluestein
      2944  2^7.23           chain3   replayed        4265      15396    3.46  5.6e-16  
      2945  5.19.31          chain3   replayed        6755      17356    2.50  5.4e-16  
      2946  2.3.491          prime    replayed       22453      24166    1.07  6.4e-16  bluestein
      2947  7.421            prime    replayed       22700      24168    1.06  5.0e-16  bluestein
      2948  2^2.11.67        prime    replayed       22343      27329    0.88  4.9e-16  flips differ 1.39x bluestein
      2949  3.983            prime    replayed       22313      24293    1.07  7.7e-16  bluestein
      2950  2.5^2.59         prime    replayed       21864      22415    0.99  6.1e-16  bluestein
      2951  13.227           prime    replayed       22280      24177    1.06  5.3e-16  bluestein
      2952  2^3.3^2.41       chain3   replayed        5495      19011    3.39  5.3e-16  
      2953  2953             prime    replayed       22251      24968    1.06  6.6e-16  bluestein
      2954  2.7.211          prime    replayed       22122      24179    1.08  5.9e-16  bluestein
      2955  3.5.197          prime    replayed       22267      24253    1.08  5.6e-16  bluestein
      2956  2^2.739          prime    replayed       22533      24204    1.07  6.5e-16  bluestein
      2957  2957             prime    replayed       21869      24322    1.08  5.6e-16  bluestein
      2958  2.3.17.29        flat     replayed        8700      18241    2.08  1.1e-15  
      2959  11.269           prime    replayed       21842      25850    1.10  8.1e-16  bluestein
      2960  2^4.5.37         flat     replayed        5579      18290    3.25  7.1e-16  
      2961  3^2.7.47         flat     replayed        6899      19147    2.77  6.3e-16  
      2962  2.1481           prime    replayed       22400      24185    0.84  5.3e-16  flips differ 1.28x bluestein
      2963  2963             prime    replayed       22207      24252    1.08  6.0e-16  bluestein
      2964  2^2.3.13.19      chain3   replayed        4482      16896    3.77  6.6e-16  
      2965  5.593            prime    replayed       22278      24150    0.91  6.9e-16  bluestein
      2966  2.1483           prime    replayed       22135      24215    1.09  6.4e-16  bluestein
      2967  3.23.43          chain3   replayed       11237      21776    1.36  5.6e-16  flips differ 1.42x
      2968  2^3.7.53         prime    replayed       22441      22361    0.79  4.5e-16  flips differ 1.27x bluestein
      2969  2969             prime    replayed       22321      24182    1.05  6.6e-16  bluestein
      2970  2.3^3.5.11       chain3   replayed        4784      11745    2.39  6.8e-16  
      2971  2971             prime    replayed       22281      24185    1.06  6.3e-16  bluestein
      2972  2^2.743          prime    replayed       22368      24207    1.08  6.5e-16  bluestein
      2973  3.991            prime    replayed       22235      24196    1.09  6.3e-16  bluestein
      2974  2.1487           prime    replayed       22405      24212    0.89  5.4e-16  bluestein
      2975  5^2.7.17         flat     replayed        5986      11358    1.59  6.8e-16  
      2976  2^5.3.31         chain3   replayed        4752      17605    3.64  6.2e-16  
      2977  13.229           prime    replayed       22392      24267    1.07  6.5e-16  bluestein
      2978  2.1489           prime    replayed       22795      24182    1.06  7.3e-16  bluestein
      2979  3^2.331          prime    replayed       21861      24167    1.08  5.8e-16  bluestein
      2980  2^2.5.149        prime    replayed       21874      24147    1.10  5.3e-16  bluestein
      2981  11.271           prime    replayed       21878      24191    0.84  7.0e-16  flips differ 1.31x bluestein
      2982  2.3.7.71         prime    replayed       22697      26744    1.15  4.7e-16  bluestein
      2983  19.157           prime    replayed       22451      24216    1.07  7.0e-16  bluestein
      2984  2^3.373          prime    replayed       22277      24206    1.07  7.5e-16  bluestein
      2985  3.5.199          prime    replayed       22203      24152    1.06  5.9e-16  bluestein
      2986  2.1493           prime    replayed       21869      24131    1.10  5.9e-16  bluestein
      2987  29.103           prime    replayed       21897      24244    1.08  6.0e-16  bluestein
      2988  2^2.3^2.83       prime    replayed       22220      31173    1.38  6.1e-16  bluestein
      2989  7^2.61           prime    replayed       21898      20041    0.91  6.2e-16  bluestein
      2990  2.5.13.23        flat     replayed        7698      15875    2.05  6.9e-16  
      2991  3.997            prime    replayed       22333      24171    1.08  7.0e-16  bluestein
      2992  2^4.11.17        chain3   replayed        4219      15843    3.75  5.9e-16  
      2993  41.73            prime    replayed       21912      32628    1.49  6.2e-16  bluestein
      2994  2.3.499          prime    replayed       22429      24251    1.08  5.8e-16  bluestein
      2995  5.599            prime    replayed       22267      24182    1.05  5.5e-16  bluestein
      2996  2^2.7.107        prime    replayed       22590      24269    1.06  4.6e-16  bluestein
      2997  3^4.37           chain3   replayed        7292      17517    2.26  5.1e-16  
      2998  2.1499           prime    replayed       22409      24245    1.08  7.7e-16  bluestein
      2999  2999             prime    replayed       22296      24199    1.08  7.0e-16  bluestein
      3000  2^3.3.5^3        chain3   replayed        3927      10402    2.61  5.7e-16  
      3001  3001             prime    replayed       17974      24190    1.26  8.6e-16  rader
      3002  2.19.79          prime    replayed       21875      31057    1.40  5.3e-16  bluestein
      3003  3.7.11.13        chain3   replayed        5023      13026    2.50  6.6e-16  
      3004  2^2.751          prime    replayed       22269      24146    1.08  5.7e-16  bluestein
      3005  5.601            prime    replayed       21861      29299    1.26  6.0e-16  bluestein
      3006  2.3^2.167        prime    replayed       22895      24200    0.88  6.0e-16  bluestein
      3007  31.97            prime    replayed       22267      37109    1.66  5.7e-16  bluestein
      3008  2^6.47           2p       replayed        9400      21919    1.71  4.8e-16  flips differ 1.37x
      3009  3.17.59          prime    replayed       22301      24165    1.06  7.0e-16  bluestein
      3010  2.5.7.43         chain3   replayed       10033      18060    1.79  5.5e-16  
      3011  3011             prime    replayed       22261      24179    1.06  6.5e-16  bluestein
      3012  2^2.3.251        prime    replayed       22293      24186    1.06  5.9e-16  bluestein
      3013  23.131           prime    replayed       22635      24196    1.06  7.5e-16  bluestein
      3014  2.11.137         prime    replayed       22472      24234    1.08  5.3e-16  bluestein
      3015  3^2.5.67         prime    replayed       22279      25905    1.14  4.5e-16  bluestein
      3016  2^3.13.29        flat     replayed        5874      19337    2.99  6.7e-16  
      3017  7.431            prime    replayed       22838      24135    0.84  5.4e-16  flips differ 1.26x bluestein
      3018  2.3.503          prime    replayed       22290      24154    1.04  5.1e-16  bluestein
      3019  3019             prime    replayed       22811      24142    1.00  6.1e-16  bluestein
      3020  2^2.5.151        prime    replayed       21946      24161    1.07  6.5e-16  bluestein
      3021  3.19.53          prime    replayed       22464      23302    1.04  4.8e-16  bluestein
      3022  2.1511           prime    replayed       22246      24183    1.08  5.8e-16  bluestein
      3023  3023             prime    replayed       22832      24200    1.06  5.8e-16  bluestein
      3024  2^4.3^3.7        ztt      replayed        3283      10819    3.29  4.1e-16  
      3025  5^2.11^2         flat     replayed        5867      12715    2.15  8.7e-16  
      3026  2.17.89          prime    replayed       22739      32472    1.42  6.8e-16  bluestein
      3027  3.1009           prime    replayed       21846      24168    1.10  7.2e-16  bluestein
      3028  2^2.757          prime    replayed       22810      24159    1.05  5.6e-16  bluestein
      3029  13.233           prime    replayed       22306      25180    1.08  6.0e-16  bluestein
      3030  2.3.5.101        prime    replayed       22282      27576    1.09  6.2e-16  bluestein
      3031  7.433            prime    replayed       22479      24215    1.06  7.1e-16  bluestein
      3032  2^3.379          prime    replayed       22285      25630    1.09  6.9e-16  bluestein
      3033  3^2.337          prime    replayed       21871      24201    1.10  5.6e-16  bluestein
      3034  2.37.41          flat     replayed        9166      24463    2.66  3.1e-15  
      3035  5.607            prime    replayed       21889      24190    1.08  8.0e-16  bluestein
      3036  2^2.3.11.23      flat     replayed        6415      18162    2.82  6.7e-16  
      3037  3037             prime    replayed       16928      24181    1.35  9.2e-16  rader
      3038  2.7^2.31         flat     replayed        7823      15013    1.89  7.7e-16  
      3039  3.1013           prime    replayed       21893      24235    1.10  7.3e-16  bluestein
      3040  2^5.5.19         chain3   replayed        4194      14802    3.48  5.5e-16  
      3041  3041             prime    replayed       15546      24176    1.55  6.9e-16  rader
      3042  2.3^2.13^2       flat     replayed        6854      15039    2.17  1.1e-15  
      3043  17.179           prime    replayed       21906      24270    1.08  5.8e-16  bluestein
      3044  2^2.761          prime    replayed       21949      24194    1.08  6.3e-16  bluestein
      3045  3.5.7.29         flat     replayed        7242      14120    1.92  7.1e-16  
      3046  2.1523           prime    replayed       22286      24318    1.08  6.5e-16  bluestein
      3047  11.277           prime    replayed       23252      24360    0.85  5.6e-16  bluestein
      3048  2^3.3.127        prime    replayed       22407      24186    1.08  5.0e-16  bluestein
      3049  3049             prime    replayed       22491      24340    1.08  5.8e-16  bluestein
      3050  2.5^2.61         prime    replayed       22836      23725    1.03  5.8e-16  bluestein
      3051  3^3.113          prime    replayed       22565      24234    1.07  6.8e-16  bluestein
      3052  2^2.7.109        prime    replayed       22221      24233    1.06  7.2e-16  bluestein
      3053  43.71            prime    replayed       22664      34276    1.50  7.0e-16  bluestein
      3054  2.3.509          prime    replayed       22475      25201    1.06  5.0e-16  bluestein
      3055  5.13.47          chain3   replayed       15229      19913    1.17  6.2e-16  
      3056  2^4.191          prime    replayed       22344      24190    0.90  7.2e-16  bluestein
      3057  3.1019           prime    replayed       21898      24160    1.10  6.6e-16  bluestein
      3058  2.11.139         prime    replayed       21935      24270    1.07  5.4e-16  bluestein
      3059  7.19.23          flat     replayed        7003      15906    2.22  7.0e-16  
      3060  2^2.3^2.5.17     chain3   replayed        4391      13748    3.07  7.1e-16  
      3061  3061             prime    replayed       15574      24244    1.44  1.0e-15  rader
      3062  2.1531           prime    replayed       22833      24278    1.06  6.0e-16  bluestein
      3063  3.1021           prime    replayed       22929      24217    1.04  7.6e-16  bluestein
      3064  2^3.383          prime    replayed       22662      24183    1.06  5.9e-16  bluestein
      3065  5.613            prime    replayed       21996      25529    1.08  8.2e-16  bluestein
      3066  2.3.7.73         prime    replayed       22787      27675    1.21  5.2e-16  bluestein
      3067  3067             prime    replayed       21935      24190    1.10  5.4e-16  bluestein
      3068  2^2.13.59        prime    replayed       22926      26246    1.14  5.1e-16  bluestein
      3069  3^2.11.31        chain3   replayed        9135      18062    1.48  9.8e-16  flips differ 1.33x
      3070  2.5.307          prime    replayed       21916      24233    1.10  7.2e-16  bluestein
      3071  37.83            prime    replayed       21458      35574    1.62  4.8e-16  bluestein
      3072  2^10.3           ztt      replayed        2728      12239    4.48  4.0e-16  
      3073  7.439            prime    replayed       21942      24184    1.08  5.7e-16  bluestein
      3074  2.29.53          prime    replayed       23268      26380    0.99  4.7e-16  bluestein
      3075  3.5^2.41         chain3   replayed        9890      18449    1.84  6.2e-16  
      3076  2^2.769          prime    replayed       21971      24184    1.07  6.6e-16  bluestein
      3077  17.181           prime    replayed       22261      24211    1.07  5.6e-16  bluestein
      3078  2.3^4.19         chain3   replayed        6893      14400    2.08  7.6e-16  
      3079  3079             prime    replayed       22495      24227    0.86  5.7e-16  flips differ 1.26x bluestein
      3080  2^3.5.7.11       flat     replayed        4719      12407    2.57  5.9e-16  
      3081  3.13.79          prime    replayed       22896      29308    1.09  5.5e-16  bluestein
      3082  2.23.67          prime    replayed       22351      29666    1.30  6.2e-16  bluestein
      3083  3083             prime    replayed       22421      24242    1.08  6.6e-16  bluestein
      3084  2^2.3.257        prime    replayed       22297      24149    1.06  6.0e-16  bluestein
      3085  5.617            prime    replayed       22406      24182    1.07  7.2e-16  bluestein
      3086  2.1543           prime    replayed       22397      26191    0.88  6.1e-16  flips differ 1.43x bluestein
      3087  3^2.7^3          chain3   replayed        5302       9807    1.74  5.0e-16  
      3088  2^4.193          prime    replayed       21857      27521    1.11  7.1e-16  bluestein
      3089  3089             prime    replayed       22514      24228    1.07  6.9e-16  bluestein
      3090  2.3.5.103        prime    replayed       21881      24243    1.08  6.4e-16  bluestein
      3091  11.281           prime    replayed       21836      26989    0.90  6.6e-16  flips differ 1.52x bluestein
      3092  2^2.773          prime    replayed       22408      24187    1.08  5.2e-16  bluestein
      3093  3.1031           prime    replayed       21886      24484    1.09  6.6e-16  bluestein
      3094  2.7.13.17        flat     replayed        7108      13912    1.95  1.3e-15  
      3095  5.619            prime    replayed       22459      26177    0.90  6.5e-16  flips differ 1.40x bluestein
      3096  2^3.3^2.43       chain3   replayed        6207      21056    3.38  7.0e-16  
      3097  19.163           prime    replayed       22448      24186    1.08  6.2e-16  bluestein
      3098  2.1549           prime    replayed       21927      25571    1.10  6.3e-16  bluestein
      3099  3.1033           prime    replayed       21901      25620    1.10  6.8e-16  bluestein
      3100  2^2.5^2.31       flat     replayed        6750      17711    2.50  5.9e-16  
      3101  7.443            prime    replayed       22710      24196    1.04  6.3e-16  bluestein
      3102  2.3.11.47        flat     replayed        9627      22863    2.35  8.9e-16  
      3103  29.107           prime    replayed       22268      24989    1.09  7.1e-16  bluestein
      3104  2^5.97           prime    replayed       22672      36121    1.59  4.4e-16  bluestein
      3105  3^3.5.23         chain3   replayed        7401      14827    1.94  5.3e-16  
      3106  2.1553           prime    replayed       21904      24234    1.08  6.8e-16  bluestein
      3107  13.239           prime    replayed       22333      24233    1.06  7.1e-16  bluestein
      3108  2^2.3.7.37       chain3   replayed        5526      19122    3.46  6.1e-16  
      3109  3109             prime    replayed       22497      24225    0.84  7.0e-16  flips differ 1.27x bluestein
      3110  2.5.311          prime    replayed       22927      24238    0.84  6.2e-16  flips differ 1.26x bluestein
      3111  3.17.61          prime    replayed       22273      25411    0.90  6.0e-16  flips differ 1.28x bluestein
      3112  2^3.389          prime    replayed       22559      24250    1.07  6.8e-16  bluestein
      3113  11.283           prime    replayed       22950      24229    1.05  6.3e-16  bluestein
      3114  2.3^2.173        prime    replayed       21938      27561    1.07  7.3e-16  bluestein
      3115  5.7.89           prime    replayed       21953      28632    1.27  6.0e-16  bluestein
      3116  2^2.19.41        chain3   replayed        7543      23671    3.10  5.4e-16  
      3117  3.1039           prime    replayed       22416      24979    1.07  6.2e-16  bluestein
      3118  2.1559           prime    replayed       21912      24208    1.10  6.0e-16  bluestein
      3119  3119             prime    replayed       22973      24251    1.05  8.1e-16  bluestein
      3120  2^4.3.5.13       chain3   replayed        4047      13157    3.17  6.2e-16  
      3121  3121             prime    replayed       15490      24250    1.48  9.9e-16  rader
      3122  2.7.223          prime    replayed       21894      24202    1.10  6.6e-16  bluestein
      3123  3^2.347          prime    replayed       22668      24244    1.07  7.5e-16  bluestein
      3124  2^2.11.71        prime    replayed       22326      30192    1.31  5.6e-16  bluestein
      3125  5^5              flat     replayed        5657       9199    1.59  8.4e-16  
      3126  2.3.521          prime    replayed       21924      24368    1.08  5.4e-16  bluestein
      3127  53.59            prime    replayed       22456      33970    1.51  7.4e-16  bluestein
      3128  2^3.17.23        chain3   replayed       10836      20347    1.70  6.4e-16  
      3129  3.7.149          prime    replayed       22440      24261    0.85  6.9e-16  flips differ 1.27x bluestein
      3130  2.5.313          prime    replayed       22421      26132    1.08  6.9e-16  bluestein
      3131  31.101           prime    replayed       21947      24299    1.08  6.2e-16  bluestein
      3132  2^2.3^3.29       chain3   replayed        5143      17324    3.36  5.1e-16  
      3133  13.241           prime    replayed       21978      24173    1.05  6.8e-16  bluestein
      3134  2.1567           prime    replayed       21968      25911    1.10  6.6e-16  bluestein
      3135  3.5.11.19        chain3   replayed        6596      14811    2.21  7.4e-16  
      3136  2^6.7^2          ztt      replayed        3066      11230    3.65  3.8e-16  
      3137  3137             prime    replayed       12122      24237    2.00  8.5e-16  rader
      3138  2.3.523          prime    replayed       21920      25902    1.11  5.8e-16  bluestein
      3139  43.73            prime    replayed       22278      35194    1.54  7.5e-16  bluestein
      3140  2^2.5.157        prime    replayed       21938      24215    1.07  6.0e-16  bluestein
      3141  3^2.349          prime    replayed       21975      24265    1.10  6.3e-16  bluestein
      3142  2.1571           prime    replayed       21950      24246    1.08  7.3e-16  bluestein
      3143  7.449            prime    replayed       22307      24235    1.06  7.1e-16  bluestein
      3144  2^3.3.131        prime    replayed       22555      24238    1.07  5.0e-16  bluestein
      3145  5.17.37          chain3   replayed        7393      18817    2.54  6.0e-16  
      3146  2.11^2.13        flat     replayed        7001      16620    2.33  9.0e-16  
      3147  3.1049           prime    replayed       21965      24225    1.06  7.0e-16  bluestein
      3148  2^2.787          prime    replayed       22010      24207    1.07  6.2e-16  bluestein
      3149  47.67            prime    replayed       21963      35398    1.61  5.9e-16  bluestein
      3150  2.3^2.5^2.7      chain3   replayed        4656      10800    2.22  5.6e-16  
      3151  23.137           prime    replayed       21543      25246    0.88  6.5e-16  flips differ 1.28x bluestein
      3152  2^4.197          prime    replayed       22317      24937    1.08  5.9e-16  bluestein
      3153  3.1051           prime    replayed       21968      24218    1.10  6.3e-16  bluestein
      3154  2.19.83          prime    replayed       22516      33480    1.48  5.7e-16  bluestein
      3155  5.631            prime    replayed       22358      24275    1.06  8.4e-16  bluestein
      3156  2^2.3.263        prime    replayed       22370      24304    1.07  6.1e-16  bluestein
      3157  7.11.41          chain3   replayed        7630      17838    2.29  5.2e-16  
      3158  2.1579           prime    replayed       22062      24182    1.07  6.3e-16  bluestein
      3159  3^5.13           chain3   replayed        5727      11938    1.73  7.2e-16  
      3160  2^3.5.79         prime    replayed       22413      31992    1.43  5.8e-16  bluestein
      3161  29.109           prime    replayed       21916      25930    1.10  7.2e-16  bluestein
      3162  2.3.17.31        flat     replayed        8458      20589    2.43  1.6e-15  
      3163  3163             prime    replayed       22819      24279    1.06  7.0e-16  bluestein
      3164  2^2.7.113        prime    replayed       22460      24281    1.08  6.8e-16  bluestein
      3165  3.5.211          prime    replayed       22693      24179    1.06  5.8e-16  bluestein
      3166  2.1583           prime    replayed       21934      25280    1.10  5.8e-16  bluestein
      3167  3167             prime    replayed       22338      24213    1.04  6.2e-16  bluestein
      3168  2^5.3^2.11       chain3   replayed        3850      13333    3.42  4.7e-16  
      3169  3169             prime    replayed       14574      24265    1.66  1.0e-15  rader
      3170  2.5.317          prime    replayed       22502      24201    1.07  5.5e-16  bluestein
      3171  3.7.151          prime    replayed       21934      24275    1.10  4.9e-16  bluestein
      3172  2^2.13.61        prime    replayed       22324      27670    1.21  4.8e-16  bluestein
      3173  19.167           prime    replayed       22313      24322    1.06  5.4e-16  bluestein
      3174  2.3.23^2         chain3   replayed        8298      20835    2.50  6.0e-16  
      3175  5^2.127          prime    replayed       21952      24259    1.08  5.3e-16  bluestein
      3176  2^3.397          prime    replayed       22250      24230    1.07  5.5e-16  bluestein
      3177  3^2.353          prime    replayed       22819      24329    1.06  5.9e-16  bluestein
      3178  2.7.227          prime    replayed       22379      24226    1.06  5.5e-16  bluestein
      3179  11.17^2          flat     replayed        6308      15717    2.27  1.1e-15  
      3180  2^2.3.5.53       prime    replayed       22889      24094    1.05  5.3e-16  bluestein
      3181  3181             prime    replayed       22693      24200    1.05  6.2e-16  bluestein
      3182  2.37.43          flat     replayed       10131      26832    2.63  2.3e-15  
      3183  3.1061           prime    replayed       22740      24234    1.06  6.7e-16  bluestein
      3184  2^4.199          prime    replayed       22617      24195    1.05  5.6e-16  bluestein
      3185  5.7^2.13         flat     replayed        5679      10809    1.86  6.7e-16  
      3186  2.3^3.59         prime    replayed       22985      25995    1.05  5.3e-16  bluestein
      3187  3187             prime    replayed       22334      24224    1.06  4.9e-16  bluestein
      3188  2^2.797          prime    replayed       22346      24233    1.06  6.0e-16  bluestein
      3189  3.1063           prime    replayed       22377      24265    1.08  5.9e-16  bluestein
      3190  2.5.11.29        chain3   replayed        7582      17725    1.75  6.2e-16  flips differ 1.33x
      3191  3191             prime    replayed       22303      24223    1.08  7.0e-16  bluestein
      3192  2^3.3.7.19       chain3   replayed        5445      15343    2.72  5.2e-16  
      3193  31.103           prime    replayed       22732      24394    1.06  5.5e-16  bluestein
      3194  2.1597           prime    replayed       22479      24270    1.07  6.9e-16  bluestein
      3195  3^2.5.71         prime    replayed       22870      28732    1.25  6.5e-16  bluestein
      3196  2^2.17.47        flat     replayed        8424      25693    3.02  8.1e-16  
      3197  23.139           prime    replayed       21956      24221    1.10  5.9e-16  bluestein
      3198  2.3.13.41        chain3   replayed        6974      21315    3.03  6.8e-16  
      3199  7.457            prime    replayed       22260      24214    1.08  7.4e-16  bluestein
      3200  2^7.5^2          ztt      replayed        3033      11657    3.37  3.4e-16  
      3201  3.11.97          prime    replayed       22817      34518    1.50  6.4e-16  bluestein
      3202  2.1601           prime    replayed       22351      25987    1.06  7.2e-16  bluestein
      3203  3203             prime    replayed       22692      24252    0.83  5.7e-16  flips differ 1.29x bluestein
      3204  2^2.3^2.89       prime    replayed       22025      34950    1.27  5.2e-16  flips differ 1.25x bluestein
      3205  5.641            prime    replayed       22252      24245    1.09  6.4e-16  bluestein
      3206  2.7.229          prime    replayed       22014      24285    1.08  5.7e-16  bluestein
      3207  3.1069           prime    replayed       22738      24263    1.03  7.1e-16  bluestein
      3208  2^3.401          prime    replayed       22486      24366    1.08  8.3e-16  bluestein
      3209  3209             prime    replayed       22345      25285    1.09  6.6e-16  bluestein
      3210  2.3.5.107        prime    replayed       22906      27672    0.85  5.4e-16  flips differ 1.59x bluestein
      3211  13^2.19          flat     replayed        6276      15767    2.29  1.0e-15  
      3212  2^2.11.73        prime    replayed       22488      31043    1.35  6.1e-16  bluestein
      3213  3^3.7.17         chain3   replayed        5780      13159    1.64  9.3e-16  flips differ 1.38x
      3214  2.1607           prime    replayed       21969      24272    1.08  8.0e-16  bluestein
      3215  5.643            prime    replayed       22834      24298    1.06  7.2e-16  bluestein
      3216  2^4.3.67         prime    replayed       22589      29267    1.29  5.8e-16  bluestein
      3217  3217             prime    replayed       22345      24305    1.09  5.7e-16  bluestein
      3218  2.1609           prime    replayed       21985      24258    1.10  6.0e-16  bluestein
      3219  3.29.37          flat     replayed        9092      22760    2.47  2.1e-15  
      3220  2^2.5.7.23       flat     replayed        5739      16223    2.80  5.1e-16  
      3221  3221             prime    replayed       21995      24370    1.07  6.8e-16  bluestein
      3222  2.3^2.179        prime    replayed       21952      24289    1.08  5.9e-16  bluestein
      3223  11.293           prime    replayed       22590      24243    0.77  8.1e-16  flips differ 1.40x bluestein
      3224  2^3.13.31        chain3   replayed        6039      21566    3.52  7.1e-16  
      3225  3.5^2.43         chain3   replayed       11164      20228    1.79  5.6e-16  
      3226  2.1613           prime    replayed       23019      24261    1.05  8.2e-16  bluestein
      3227  7.461            prime    replayed       21943      24361    1.10  7.0e-16  bluestein
      3228  2^2.3.269        prime    replayed       21764      26205    1.10  6.6e-16  bluestein
      3229  3229             prime    replayed       22333      24360    1.08  7.7e-16  bluestein
      3230  2.5.17.19        chain3   replayed        6102      17430    2.45  6.4e-16  
      3231  3^2.359          prime    replayed       22518      24274    1.07  6.4e-16  bluestein
      3232  2^5.101          prime    replayed       21945      24233    1.10  5.9e-16  bluestein
      3233  53.61            prime    replayed       21999      35527    1.61  6.3e-16  bluestein
      3234  2.3.7^2.11       flat     replayed        6681      12641    1.88  5.4e-16  
      3235  5.647            prime    replayed       21928      24261    1.08  7.3e-16  bluestein
      3236  2^2.809          prime    replayed       22351      24248    1.06  6.5e-16  bluestein
      3237  3.13.83          prime    replayed       21941      31679    1.41  6.8e-16  bluestein
      3238  2.1619           prime    replayed       22884      24260    0.91  6.3e-16  bluestein
      3239  41.79            prime    replayed       21943      37600    1.66  5.9e-16  bluestein
      3240  2^3.3^4.5        chain3   replayed        4508      11796    2.61  5.8e-16  
      3241  7.463            prime    replayed       22025      24293    0.89  7.6e-16  bluestein
      3242  2.1621           prime    replayed       22456      26721    1.09  6.2e-16  bluestein
      3243  3.23.47          chain3   replayed       12632      25198    1.98  5.8e-16  
      3244  2^2.811          prime    replayed       21974      24325    0.85  6.4e-16  flips differ 1.31x bluestein
      3245  5.11.59          prime    replayed       21983      24030    1.06  6.0e-16  bluestein
      3246  2.3.541          prime    replayed       21961      24322    1.11  7.3e-16  bluestein
      3247  17.191           prime    replayed       22368      24242    1.08  6.0e-16  bluestein
      3248  2^4.7.29         chain3   replayed        5317      17971    3.37  5.8e-16  
      3249  3^2.19^2         flat     replayed        8486      18527    2.12  1.1e-15  
      3250  2.5^3.13         chain3   replayed        5907      11624    1.94  6.8e-16  
      3251  3251             prime    replayed       22818      24304    1.06  7.0e-16  bluestein
      3252  2^2.3.271        prime    replayed       21971      24837    1.10  7.6e-16  bluestein
      3253  3253             prime    replayed       22609      24297    1.07  7.0e-16  bluestein
      3254  2.1627           prime    replayed       22381      27147    0.89  6.8e-16  flips differ 1.51x bluestein
      3255  3.5.7.31         flat     replayed        7942      16301    2.02  6.8e-16  
      3256  2^3.11.37        flat     replayed        7206      22693    3.07  6.0e-16  
      3257  3257             prime    replayed       21988      24319    1.10  6.2e-16  bluestein
      3258  2.3^2.181        prime    replayed       22503      24251    1.07  7.1e-16  bluestein
      3259  3259             prime    replayed       22607      24387    0.78  6.6e-16  flips differ 1.38x bluestein
      3260  2^2.5.163        prime    replayed       22928      24322    1.04  8.7e-16  bluestein
      3261  3.1087           prime    replayed       22010      24373    1.08  7.1e-16  bluestein
      3262  2.7.233          prime    replayed       21701      25706    1.12  6.8e-16  bluestein
      3263  13.251           prime    replayed       21989      24324    1.10  7.0e-16  bluestein
      3264  2^6.3.17         chain3   replayed        4296      15249    3.42  6.7e-16  
      3265  5.653            prime    replayed       22073      24478    1.08  7.8e-16  bluestein
      3266  2.23.71          prime    replayed       22641      33111    1.46  5.4e-16  bluestein
      3267  3^3.11^2         flat     replayed        7030      14911    2.10  5.6e-16  
      3268  2^2.19.43        flat     replayed        9001      25744    2.85  1.7e-15  
      3269  7.467            prime    replayed       22461      24319    0.77  6.3e-16  flips differ 1.41x bluestein
      3270  2.3.5.109        prime    replayed       23104      24283    1.04  6.0e-16  bluestein
      3271  3271             prime    replayed       23060      26668    1.14  6.9e-16  bluestein
      3272  2^3.409          prime    replayed       22044      24317    1.10  5.9e-16  bluestein
      3273  3.1091           prime    replayed       22583      24335    0.92  6.4e-16  bluestein
      3274  2.1637           prime    replayed       22567      24285    1.06  5.7e-16  bluestein
      3275  5^2.131          prime    replayed       22115      24357    1.08  6.6e-16  bluestein
      3276  2^2.3^2.7.13     chain3   replayed        5077      13783    2.62  6.1e-16  
      3277  29.113           prime    replayed       22589      24393    1.08  6.1e-16  bluestein
      3278  2.11.149         prime    replayed       22382      24377    1.06  6.0e-16  bluestein
      3279  3.1093           prime    replayed       21986      24410    1.11  7.3e-16  bluestein
      3280  2^4.5.41         flat     replayed        6505      21312    3.26  4.4e-16  
      3281  17.193           prime    replayed       22905      24318    1.06  7.4e-16  bluestein
      3282  2.3.547          prime    replayed       22417      24355    1.06  6.8e-16  bluestein
      3283  7^2.67           prime    replayed       22696      24132    0.99  5.4e-16  bluestein
      3284  2^2.821          prime    replayed       22961      24476    1.06  5.9e-16  bluestein
      3285  3^2.5.73         prime    replayed       22622      29784    1.30  6.9e-16  bluestein
      3286  2.31.53          prime    replayed       22013      29237    1.30  4.8e-16  bluestein
      3287  19.173           prime    replayed       22903      25498    1.08  7.1e-16  bluestein
      3288  2^3.3.137        prime    replayed       22616      24311    1.07  4.5e-16  bluestein
      3289  11.13.23         chain3   replayed        8158      17192    1.84  8.5e-16  
      3290  2.5.7.47         flat     replayed        9832      21304    1.78  7.1e-16  
      3291  3.1097           prime    replayed       22627      28214    0.90  8.0e-16  flips differ 1.48x bluestein
      3292  2^2.823          prime    replayed       22035      24399    1.10  6.6e-16  bluestein
      3293  37.89            prime    replayed       22475      39798    1.73  4.9e-16  bluestein
      3294  2.3^3.61         prime    replayed       22440      27317    1.19  5.4e-16  bluestein
      3295  5.659            prime    replayed       22059      25297    1.10  7.5e-16  bluestein
      3296  2^5.103          prime    replayed       22059      24346    1.10  4.2e-16  bluestein
      3297  3.7.157          prime    replayed       22080      24463    1.08  6.5e-16  bluestein
      3298  2.17.97          prime    replayed       22911      37834    1.65  6.2e-16  bluestein
      3299  3299             prime    replayed       22451      29159    0.95  7.0e-16  flips differ 1.51x bluestein
      3300  2^2.3.5^2.11     chain3   replayed        4531      13453    2.97  5.8e-16  
      3301  3301             prime    replayed       22088      25073    0.89  7.8e-16  flips differ 1.30x bluestein
      3302  2.13.127         prime    replayed       22601      24327    0.85  6.9e-16  flips differ 1.26x bluestein
      3303  3^2.367          prime    replayed       22947      24378    1.06  7.5e-16  bluestein
      3304  2^3.7.59         prime    replayed       22150      26988    1.18  5.2e-16  bluestein
      3305  5.661            prime    replayed       22940      24391    0.88  5.4e-16  bluestein
      3306  2.3.19.29        chain3   replayed        8607      21563    2.48  5.7e-16  
      3307  3307             prime    replayed       22575      24329    1.08  6.0e-16  bluestein
      3308  2^2.827          prime    replayed       22659      24323    0.85  6.4e-16  flips differ 1.26x bluestein
      3309  3.1103           prime    replayed       22655      24451    1.08  7.9e-16  bluestein
      3310  2.5.331          prime    replayed       22812      24359    1.03  8.2e-16  bluestein
      3311  7.11.43          chain3   replayed       11490      19746    1.19  5.7e-16  flips differ 1.45x
      3312  2^4.3^2.23       chain3   replayed        4913      17471    3.53  4.6e-16  
      3313  3313             prime    replayed       18848      25676    1.29  1.1e-15  rader
      3314  2.1657           prime    replayed       23054      27379    0.93  6.9e-16  flips differ 1.42x bluestein
      3315  3.5.13.17        flat     replayed        6921      15084    2.10  1.3e-15  
      3316  2^2.829          prime    replayed       22045      25344    1.10  6.7e-16  bluestein
      3317  31.107           prime    replayed       22423      24325    1.06  5.7e-16  bluestein
      3318  2.3.7.79         prime    replayed       22100      32156    1.40  5.7e-16  bluestein
      3319  3319             prime    replayed       23048      24478    1.06  7.0e-16  bluestein
      3320  2^3.5.83         prime    replayed       22416      34555    1.50  5.9e-16  bluestein
      3321  3^4.41           chain3   replayed        8094      20204    2.21  6.2e-16  
      3322  2.11.151         prime    replayed       22043      24374    1.08  6.5e-16  bluestein
      3323  3323             prime    replayed       22481      24337    1.05  6.2e-16  bluestein
      3324  2^2.3.277        prime    replayed       22480      24367    1.06  7.9e-16  bluestein
      3325  5^2.7.19         chain3   replayed        6549      13951    2.12  5.0e-16  
      3326  2.1663           prime    replayed       22644      24361    1.07  6.1e-16  bluestein
      3327  3.1109           prime    replayed       22579      24541    1.08  7.2e-16  bluestein
      3328  2^8.13           chain3   replayed        6503      18397    2.81  5.8e-16  
      3329  3329             prime    replayed       16287      25788    1.48  8.2e-16  rader
      3330  2.3^2.5.37       chain3   replayed       12592      20294    1.57  6.8e-16  
      3331  3331             prime    replayed       22392      24366    1.06  8.3e-16  bluestein
      3332  2^2.7^2.17       flat     replayed        5702      14760    2.55  6.2e-16  
      3333  3.11.101         prime    replayed       23037      25794    1.06  6.9e-16  bluestein
      3334  2.1667           prime    replayed       22525      25130    1.08  6.7e-16  bluestein
      3335  5.23.29          chain3   replayed        9206      20102    1.63  6.0e-16  flips differ 1.34x
      3336  2^3.3.139        prime    replayed       22609      24375    1.07  6.4e-16  bluestein
      3337  47.71            prime    replayed       22683      39064    1.22  7.0e-16  flips differ 1.41x bluestein
      3338  2.1669           prime    replayed       22098      24412    1.08  8.0e-16  bluestein
      3339  3^2.7.53         prime    replayed       23005      23256    1.00  6.6e-16  bluestein
      3340  2^2.5.167        prime    replayed       21549      24491    1.11  6.1e-16  bluestein
      3341  13.257           prime    replayed       22600      24312    1.06  6.3e-16  bluestein
      3342  2.3.557          prime    replayed       22425      24333    1.06  6.3e-16  bluestein
      3343  3343             prime    replayed       21993      24433    1.09  7.0e-16  bluestein
      3344  2^4.11.19        chain3   replayed        4874      19005    3.48  4.8e-16  
      3345  3.5.223          prime    replayed       22792      24451    0.90  6.7e-16  bluestein
      3346  2.7.239          prime    replayed       22787      24373    1.01  5.1e-16  bluestein
      3347  3347             prime    replayed       22891      24345    1.06  7.1e-16  bluestein
      3348  2^2.3^3.31       flat     replayed        7320      19663    2.66  5.3e-16  
      3349  17.197           prime    replayed       22034      25165    1.11  8.0e-16  bluestein
      3350  2.5^2.67         prime    replayed       22949      28649    0.84  5.5e-16  flips differ 1.48x bluestein
      3351  3.1117           prime    replayed       22746      24366    1.06  7.0e-16  bluestein
      3352  2^3.419          prime    replayed       22369      24321    1.04  6.3e-16  bluestein
      3353  7.479            prime    replayed       22722      26105    0.99  5.2e-16  bluestein
      3354  2.3.13.43        chain3   replayed        7593      23533    3.03  7.7e-16  
      3355  5.11.61          prime    replayed       22036      25569    0.97  5.6e-16  bluestein
      3356  2^2.839          prime    replayed       23339      27229    0.89  6.1e-16  flips differ 1.44x bluestein
      3357  3^2.373          prime    replayed       22456      24377    1.06  6.5e-16  bluestein
      3358  2.23.73          prime    replayed       22035      34017    1.52  5.1e-16  bluestein
      3359  3359             prime    replayed       22436      24310    1.06  6.0e-16  bluestein
      3360  2^5.3.5.7        chain3   replayed        4192      12164    2.63  4.3e-16  
      3361  3361             prime    replayed       13722      24360    1.73  7.9e-16  rader
      3362  2.41^2           flat     raced          10858      28343    2.58  2.8e-15  
      3363  3.19.59          prime    replayed       22407      28131    1.22  5.5e-16  bluestein
      3364  2^2.29^2         chain3   replayed        7781      24803    3.12  5.0e-16  
      3365  5.673            prime    replayed       22619      25378    1.07  6.4e-16  bluestein
      3366  2.3^2.11.17      flat     replayed        7981      17546    2.19  1.0e-15  
      3367  7.13.37          chain3   replayed        7446      18426    2.41  6.3e-16  
      3368  2^3.421          prime    replayed       22676      24359    1.07  8.4e-16  bluestein
      3369  3.1123           prime    replayed       22586      24340    1.07  6.6e-16  bluestein
      3370  2.5.337          prime    replayed       23018      24370    0.90  6.0e-16  bluestein
      3371  3371             prime    replayed       22412      25349    1.08  7.5e-16  bluestein
      3372  2^2.3.281        prime    replayed       22061      24385    1.08  6.9e-16  bluestein
      3373  3373             prime    replayed       23268      24341    1.04  5.1e-16  bluestein
      3374  2.7.241          prime    replayed       22447      24465    1.09  8.6e-16  bluestein
      3375  3^3.5^3          chain3   replayed        5497      10927    1.99  5.9e-16  
      3376  2^4.211          prime    replayed       22347      24565    1.10  6.1e-16  bluestein
      3377  11.307           prime    replayed       22020      24361    1.07  6.8e-16  bluestein
      3378  2.3.563          prime    replayed       22415      24433    1.07  6.1e-16  bluestein
      3379  31.109           prime    replayed       22025      24357    1.10  7.1e-16  bluestein
      3380  2^2.5.13^2       flat     replayed        5737      16902    2.89  7.6e-16  
      3381  3.7^2.23         flat     replayed        7745      15570    1.94  7.7e-16  
      3382  2.19.89          prime    replayed       22052      37318    1.62  6.1e-16  bluestein
      3383  17.199           prime    replayed       22944      24400    1.06  6.4e-16  bluestein
      3384  2^3.3^2.47       flat     replayed        7771      24547    3.15  7.8e-16  
      3385  5.677            prime    replayed       22076      24365    1.10  7.3e-16  bluestein
      3386  2.1693           prime    replayed       22910      24374    1.06  6.1e-16  bluestein
      3387  3.1129           prime    replayed       22091      24364    1.08  6.9e-16  bluestein
      3388  2^2.7.11^2       flat     replayed        5619      16313    2.87  6.4e-16  
      3389  3389             prime    replayed       22722      25847    0.98  6.6e-16  bluestein
      3390  2.3.5.113        prime    replayed       22054      24333    1.10  6.1e-16  bluestein
      3391  3391             prime    replayed       23081      24444    1.05  5.6e-16  bluestein
      3392  2^6.53           prime    replayed       22057      26407    1.16  5.1e-16  bluestein
      3393  3^2.13.29        flat     replayed        9093      19173    1.96  1.2e-15  
      3394  2.1697           prime    replayed       22029      24360    1.10  7.1e-16  bluestein
      3395  5.7.97           prime    replayed       22977      33841    1.47  7.7e-16  bluestein
      3396  2^2.3.283        prime    replayed       22077      24359    1.10  6.0e-16  bluestein
      3397  43.79            prime    replayed       22941      40683    1.77  5.6e-16  bluestein
      3398  2.1699           prime    replayed       22103      24539    1.11  7.0e-16  bluestein
      3399  3.11.103         prime    replayed       22709      24357    1.07  5.7e-16  bluestein
      3400  2^3.5^2.17       chain3   replayed        5623      15141    2.66  5.9e-16  
      3401  19.179           prime    replayed       22487      24389    1.06  7.1e-16  bluestein
      3402  2.3^5.7          chain3   replayed        6271      12212    1.75  5.1e-16  
      3403  41.83            prime    replayed       22684      40496    1.77  4.9e-16  bluestein
      3404  2^2.23.37        flat     replayed        8945      26340    2.91  1.2e-15  
      3405  3.5.227          prime    replayed       22693      24369    1.07  5.4e-16  bluestein
      3406  2.13.131         prime    replayed       22127      24456    1.08  5.1e-16  bluestein
      3407  3407             prime    replayed       22655      24369    1.07  8.8e-16  bluestein
      3408  2^4.3.71         prime    replayed       22409      32662    1.45  6.9e-16  bluestein
      3409  7.487            prime    replayed       22379      24334    1.07  6.5e-16  bluestein
      3410  2.5.11.31        flat     replayed        8356      20079    2.35  5.7e-16  
      3411  3^2.379          prime    replayed       22612      24360    1.07  6.7e-16  bluestein
      3412  2^2.853          prime    replayed       22901      24347    1.06  6.9e-16  bluestein
      3413  3413             prime    replayed       22998      24378    1.06  6.3e-16  bluestein
      3414  2.3.569          prime    replayed       23005      24338    1.06  6.9e-16  bluestein
      3415  5.683            prime    replayed       22011      24402    1.08  6.6e-16  bluestein
      3416  2^3.7.61         prime    replayed       22422      28457    1.24  5.7e-16  bluestein
      3417  3.17.67          prime    replayed       22054      30368    1.37  5.8e-16  bluestein
      3418  2.1709           prime    replayed       22633      24313    1.07  7.7e-16  bluestein
      3419  13.263           prime    replayed       22388      24458    1.09  6.6e-16  bluestein
      3420  2^2.3^2.5.19     flat     replayed        6251      16622    2.65  4.7e-16  
      3421  11.311           prime    replayed       22477      27165    0.89  6.0e-16  flips differ 1.51x bluestein
      3422  2.29.59          prime    replayed       23007      31616    1.36  6.7e-16  bluestein
      3423  3.7.163          prime    replayed       22723      24556    1.07  6.5e-16  bluestein
      3424  2^5.107          prime    replayed       22631      24325    1.05  6.5e-16  bluestein
      3425  5^2.137          prime    replayed       22455      25111    1.08  7.7e-16  bluestein
      3426  2.3.571          prime    replayed       23042      24425    1.04  6.0e-16  bluestein
      3427  23.149           prime    replayed       22740      25457    1.07  6.6e-16  bluestein
      3428  2^2.857          prime    replayed       23111      24377    0.90  6.4e-16  bluestein
      3429  3^3.127          prime    replayed       22401      24428    1.09  6.8e-16  bluestein
      3430  2.5.7^3          flat     replayed        7208      10906    1.47  7.2e-16  
      3431  47.73            prime    replayed       22675      40155    1.75  5.1e-16  bluestein
      3432  2^3.3.11.13      flat     replayed        5959      17275    2.84  6.5e-16  
      3433  3433             prime    replayed       22384      24391    1.06  6.7e-16  bluestein
      3434  2.17.101         prime    replayed       22887      24244    1.05  5.6e-16  bluestein
      3435  3.5.229          prime    replayed       22068      24526    1.09  6.6e-16  bluestein
      3436  2^2.859          prime    replayed       22761      24464    1.06  8.1e-16  bluestein
      3437  7.491            prime    replayed       22010      24469    1.08  6.5e-16  bluestein
      3438  2.3^2.191        prime    replayed       22035      27754    1.10  6.6e-16  bluestein
      3439  19.181           prime    replayed       22951      24324    1.06  7.8e-16  bluestein
      3440  2^4.5.43         chain3   replayed        6843      23524    3.37  4.8e-16  
      3441  3.31.37          flat     replayed        9781      25522    2.60  2.3e-15  
      3442  2.1721           prime    replayed       22047      26009    1.07  8.0e-16  bluestein
      3443  11.313           prime    replayed       22422      24347    1.06  6.3e-16  bluestein
      3444  2^2.3.7.41       flat     replayed        7686      22142    2.85  6.2e-16  
      3445  5.13.53          prime    replayed       22974      23861    1.04  7.6e-16  bluestein
      3446  2.1723           prime    replayed       22797      24376    1.07  6.8e-16  bluestein
      3447  3^2.383          prime    replayed       22464      24350    1.08  5.4e-16  bluestein
      3448  2^3.431          prime    replayed       23000      24413    1.06  6.5e-16  bluestein
      3449  3449             prime    replayed       21590      24334    1.10  5.2e-16  bluestein
      3450  2.3.5^2.23       chain3   replayed        8102      17189    1.82  6.8e-16  
      3451  7.17.29          chain3   replayed        8496      18109    2.11  7.1e-16  
      3452  2^2.863          prime    replayed       23004      24369    1.06  6.5e-16  bluestein
      3453  3.1151           prime    replayed       22755      24343    1.03  7.0e-16  bluestein
      3454  2.11.157         prime    replayed       22387      24419    1.06  5.8e-16  bluestein
      3455  5.691            prime    replayed       22787      26035    1.07  6.2e-16  bluestein
      3456  2^7.3^3          ztt      replayed        3597      14026    3.84  4.5e-16  
      3457  3457             prime    replayed       13675      24346    1.75  8.1e-16  rader
      3458  2.7.13.19        flat     replayed        7812      16785    1.98  6.9e-16  
      3459  3.1153           prime    replayed       22643      24317    1.07  6.9e-16  bluestein
      3460  2^2.5.173        prime    replayed       22032      24467    1.09  7.0e-16  bluestein
      3461  3461             prime    replayed       22099      24357    1.10  6.7e-16  bluestein
      3462  2.3.577          prime    replayed       22669      24423    1.08  6.5e-16  bluestein
      3463  3463             prime    replayed       22483      24422    1.08  6.9e-16  bluestein
      3464  2^3.433          prime    replayed       22592      24333    1.07  7.4e-16  bluestein
      3465  3^2.5.7.11       chain3   replayed        5842      12790    2.15  6.1e-16  
      3466  2.1733           prime    replayed       22478      24438    1.08  7.5e-16  bluestein
      3467  3467             prime    replayed       22624      24350    1.07  6.9e-16  bluestein
      3468  2^2.3.17^2       flat     replayed        7289      20464    2.75  1.5e-15  
      3469  3469             prime    replayed       18043      24384    1.29  8.8e-16  rader
      3470  2.5.347          prime    replayed       22054      24378    1.08  6.9e-16  bluestein
      3471  3.13.89          prime    replayed       22353      35433    1.54  5.1e-16  bluestein
      3472  2^4.7.31         flat     replayed        6959      20369    2.04  4.3e-16  flips differ 1.43x
      3473  23.151           prime    replayed       22454      24476    1.05  6.5e-16  bluestein
      3474  2.3^2.193        prime    replayed       22555      24355    1.08  7.6e-16  bluestein
      3475  5^2.139          prime    replayed       22026      24340    1.07  8.1e-16  bluestein
      3476  2^2.11.79        prime    replayed       22122      36301    1.58  5.7e-16  bluestein
      3477  3.19.61          prime    replayed       22933      29630    0.88  6.3e-16  flips differ 1.48x bluestein
      3478  2.37.47          flat     raced          11374      30863    2.69  2.8e-15  
      3479  7^2.71           prime    replayed       22016      26917    1.20  5.4e-16  bluestein
      3480  2^3.3.5.29       flat     replayed        7282      19217    2.63  8.6e-16  
      3481  59^2             prime    replayed       22084      39892    1.76  5.3e-16  bluestein
      3482  2.1741           prime    replayed       22710      24331    1.07  4.9e-16  bluestein
      3483  3^4.43           chain3   replayed       13140      22365    1.35  6.5e-16  flips differ 1.26x
      3484  2^2.13.67        prime    replayed       22382      32534    1.42  5.2e-16  bluestein
      3485  5.17.41          chain3   replayed       11148      21844    1.94  5.7e-16  
      3486  2.3.7.83         prime    replayed       23023      34954    1.51  5.8e-16  bluestein
      3487  11.317           prime    replayed       22625      24411    0.86  7.1e-16  flips differ 1.26x bluestein
      3488  2^5.109          prime    replayed       22977      24398    1.06  5.9e-16  bluestein
      3489  3.1163           prime    replayed       22084      24467    1.08  6.1e-16  bluestein
      3490  2.5.349          prime    replayed       22823      24346    1.07  7.4e-16  bluestein
      3491  3491             prime    replayed       22882      24361    1.06  6.7e-16  bluestein
      3492  2^2.3^2.97       prime    replayed       22570      40654    1.79  4.9e-16  bluestein
      3493  7.499            prime    replayed       22775      24328    1.06  7.1e-16  bluestein
      3494  2.1747           prime    replayed       23025      24336    1.04  7.2e-16  bluestein
      3495  3.5.233          prime    replayed       23072      25656    0.85  6.4e-16  bluestein
      3496  2^3.19.23        chain3   replayed        6590      24083    3.62  5.1e-16  
      3497  13.269           prime    replayed       22163      24386    1.06  9.2e-16  bluestein
      3498  2.3.11.53        prime    replayed       22723      27207    1.19  4.8e-16  bluestein
      3499  3499             prime    replayed       22515      24356    0.83  6.6e-16  flips differ 1.30x bluestein
      3500  2^2.5^3.7        flat     replayed        5637      11715    2.04  5.6e-16  
      3501  3^2.389          prime    replayed       22403      24383    1.09  6.4e-16  bluestein
      3502  2.17.103         prime    replayed       22648      24349    1.07  5.1e-16  bluestein
      3503  31.113           prime    replayed       22601      24353    1.08  6.2e-16  bluestein
      3504  2^4.3.73         prime    replayed       22002      33482    1.48  6.3e-16  bluestein
      3505  5.701            prime    replayed       22372      24326    1.05  6.1e-16  bluestein
      3506  2.1753           prime    replayed       22438      24337    1.06  6.5e-16  bluestein
      3507  3.7.167          prime    replayed       23005      24367    1.06  6.6e-16  bluestein
      3508  2^2.877          prime    replayed       22095      24523    1.10  7.3e-16  bluestein
      3509  11^2.29          chain3   replayed        7442      19145    2.57  6.4e-16  
      3510  2.3^3.5.13       chain3   replayed        6529      14281    1.98  6.9e-16  
      3511  3511             prime    replayed       22014      24424    1.07  6.6e-16  bluestein
      3512  2^3.439          prime    replayed       22057      24335    1.08  8.1e-16  bluestein
      3513  3.1171           prime    replayed       22038      26011    1.07  6.2e-16  bluestein
      3514  2.7.251          prime    replayed       22424      24372    1.08  6.5e-16  bluestein
      3515  5.19.37          chain3   replayed       11476      22203    1.85  6.3e-16  
      3516  2^2.3.293        prime    replayed       22143      24386    1.07  6.4e-16  bluestein
      3517  3517             prime    replayed       22420      24360    1.08  6.4e-16  bluestein
      3518  2.1759           prime    replayed       23051      24371    1.06  5.7e-16  bluestein
      3519  3^2.17.23        flat     replayed        7569      20371    2.53  8.1e-16  
      3520  2^6.5.11         chain3   replayed        4332      14979    3.40  3.6e-16  
      3521  7.503            prime    replayed       22095      24388    1.08  7.4e-16  bluestein
      3522  2.3.587          prime    replayed       22008      24363    1.10  7.1e-16  bluestein
      3523  13.271           prime    replayed       22380      24396    1.08  7.0e-16  bluestein
      3524  2^2.881          prime    replayed       23127      24437    1.05  6.4e-16  bluestein
      3525  3.5^2.47         flat     replayed        8175      23576    2.87  5.6e-16  
      3526  2.41.43          flat     replayed       11946      30789    2.56  2.4e-15  
      3527  3527             prime    replayed       22430      24435    1.06  6.8e-16  bluestein
      3528  2^3.3^2.7^2      chain3   replayed        5568      12519    1.61  5.3e-16  flips differ 1.39x
      3529  3529             prime    replayed       22602      25301    1.08  7.2e-16  bluestein
      3530  2.5.353          prime    replayed       22471      24360    1.06  7.2e-16  bluestein
      3531  3.11.107         prime    replayed       22396      24362    1.06  7.8e-16  bluestein
      3532  2^2.883          prime    replayed       22143      24477    1.09  7.5e-16  bluestein
      3533  3533             prime    replayed       22970      24419    1.06  7.4e-16  bluestein
      3534  2.3.19.31        chain3   replayed       10574      24202    2.25  6.5e-16  
      3535  5.7.101          prime    replayed       22081      25746    1.07  8.5e-16  bluestein
      3536  2^4.13.17        flat     replayed        6510      19388    2.80  8.0e-16  
      3537  3^3.131          prime    replayed       22384      24434    1.09  6.7e-16  bluestein
      3538  2.29.61          prime    replayed       22044      32892    1.49  5.8e-16  bluestein
      3539  3539             prime    replayed       22990      24409    1.04  9.6e-16  bluestein
      3540  2^2.3.5.59       prime    replayed       22079      28949    1.31  4.5e-16  bluestein
      3541  3541             prime    replayed       21737      24365    1.10  5.9e-16  bluestein
      3542  2.7.11.23        flat     replayed        8975      18428    2.05  7.0e-16  
      3543  3.1181           prime    replayed       22068      24374    1.06  6.6e-16  bluestein
      3544  2^3.443          prime    replayed       22047      25179    1.08  7.7e-16  bluestein
      3545  5.709            prime    replayed       22615      25384    1.08  6.8e-16  bluestein
      3546  2.3^2.197        prime    replayed       22473      24367    1.08  7.4e-16  bluestein
      3547  3547             prime    replayed       22560      24355    1.07  7.6e-16  bluestein
      3548  2^2.887          prime    replayed       22014      24403    1.09  7.0e-16  bluestein
      3549  3.7.13^2         chain3   replayed        6456      15713    2.34  7.0e-16  
      3550  2.5^2.71         prime    replayed       23019      31808    1.38  5.7e-16  bluestein
      3551  53.67            prime    replayed       22801      41421    1.80  5.9e-16  bluestein
      3552  2^5.3.37         flat     replayed        7785      22544    2.82  6.7e-16  
      3553  11.17.19         chain3   replayed        8057      18752    2.30  6.0e-16  
      3554  2.1777           prime    replayed       22058      25442    1.11  8.6e-16  bluestein
      3555  3^2.5.79         prime    replayed       22545      34788    1.11  6.5e-16  flips differ 1.39x bluestein
      3556  2^2.7.127        prime    replayed       22473      24474    1.07  6.1e-16  bluestein
      3557  3557             prime    replayed       22572      25394    1.08  5.8e-16  bluestein
      3558  2.3.593          prime    replayed       22536      24367    1.07  6.7e-16  bluestein
      3559  3559             prime    replayed       22633      25374    1.08  5.7e-16  bluestein
      3560  2^3.5.89         prime    replayed       22614      38727    1.71  5.7e-16  bluestein
      3561  3.1187           prime    replayed       22115      24388    1.10  6.2e-16  bluestein
      3562  2.13.137         prime    replayed       22742      24375    1.06  6.9e-16  bluestein
      3563  7.509            prime    replayed       22667      24365    1.07  6.0e-16  bluestein
      3564  2^2.3^4.11       chain3   replayed        5155      14826    2.74  6.1e-16  
      3565  5.23.31          flat     replayed        8833      22682    2.56  1.7e-15  
      3566  2.1783           prime    replayed       22070      24427    1.08  8.0e-16  bluestein
      3567  3.29.41          flat     replayed       10397      26215    2.44  2.3e-15  
      3568  2^4.223          prime    replayed       22972      24363    1.05  4.9e-16  bluestein
      3569  43.83            prime    replayed       23050      43721    1.89  6.0e-16  bluestein
      3570  2.3.5.7.17       chain3   replayed        6410      15431    2.20  4.6e-16  
      3571  3571             prime    replayed       22544      24357    1.05  7.6e-16  bluestein
      3572  2^2.19.47        chain3   replayed       13277      29808    1.45  6.8e-16  flips differ 1.55x
      3573  3^2.397          prime    replayed       22736      24448    0.86  6.6e-16  bluestein
      3574  2.1787           prime    replayed       22680      24346    1.07  9.1e-16  bluestein
      3575  5^2.11.13        chain3   replayed        6653      15363    2.13  7.9e-16  
      3576  2^3.3.149        prime    replayed       22105      24481    1.09  5.4e-16  bluestein
      3577  7^2.73           prime    replayed       22149      28116    1.22  6.0e-16  bluestein
      3578  2.1789           prime    replayed       22601      24322    0.89  6.2e-16  bluestein
      3579  3.1193           prime    replayed       22101      24389    1.08  7.4e-16  bluestein
      3580  2^2.5.179        prime    replayed       22109      24831    1.10  6.2e-16  bluestein
      3581  3581             prime    replayed       22123      24326    1.07  7.2e-16  bluestein
      3582  2.3^2.199        prime    replayed       23414      24414    0.85  7.8e-16  bluestein
      3583  3583             prime    replayed       22491      25804    1.09  7.6e-16  bluestein
      3584  2^9.7            ztt      replayed        3511      13990    3.98  3.2e-16  
      3585  3.5.239          prime    replayed       22403      25412    1.05  5.9e-16  bluestein
      3586  2.11.163         prime    replayed       22153      24311    1.07  6.9e-16  bluestein
      3587  17.211           prime    replayed       22130      24321    1.09  6.8e-16  bluestein
      3588  2^2.3.13.23      chain3   replayed        5757      22051    3.78  6.9e-16  
      3589  37.97            prime    replayed       23045      45819    1.99  7.9e-16  bluestein
      3590  2.5.359          prime    replayed       22129      27114    1.19  5.8e-16  bluestein
      3591  3^3.7.19         chain3   replayed        7272      15950    1.88  6.3e-16  
      3592  2^3.449          prime    replayed       22139      24385    1.08  5.6e-16  bluestein
      3593  3593             prime    replayed       22049      24371    1.07  7.1e-16  bluestein
      3594  2.3.599          prime    replayed       23038      24373    1.05  7.6e-16  bluestein
      3595  5.719            prime    replayed       22545      24521    1.06  7.7e-16  bluestein
      3596  2^2.29.31        chain3   replayed        8375      27653    3.27  5.6e-16  
      3597  3.11.109         prime    replayed       22098      24384    1.08  8.8e-16  bluestein
      3598  2.7.257          prime    replayed       22857      24578    1.06  6.7e-16  bluestein
      3599  59.61            prime    replayed       22077      41882    1.86  6.9e-16  bluestein
      3600  2^4.3^2.5^2      ztt      replayed        4518      13025    2.88  4.3e-16  
      3601  13.277           prime    replayed       23024      24385    1.06  7.5e-16  bluestein
      3602  2.1801           prime    replayed       22109      24322    1.08  6.1e-16  bluestein
      3603  3.1201           prime    replayed       22116      24443    1.09  8.0e-16  bluestein
      3604  2^2.17.53        prime    replayed       23049      30477    1.31  5.5e-16  bluestein
      3605  5.7.103          prime    replayed       22641      24379    1.07  6.9e-16  bluestein
      3606  2.3.601          prime    replayed       23370      24437    1.04  5.9e-16  bluestein
      3607  3607             prime    replayed       22045      24440    1.11  7.0e-16  bluestein
      3608  2^3.11.41        chain3   replayed        7399      26247    3.54  4.5e-16  
      3609  3^2.401          prime    replayed       22674      24352    1.07  5.6e-16  bluestein
      3610  2.5.19^2         chain3   replayed        7907      20682    2.11  6.0e-16  
      3611  23.157           prime    replayed       21921      26306    1.04  6.3e-16  bluestein
      3612  2^2.3.7.43       flat     replayed        9101      24465    2.63  6.5e-16  
      3613  3613             prime    replayed       22521      24393    1.05  7.1e-16  bluestein
      3614  2.13.139         prime    replayed       22636      24355    1.07  5.8e-16  bluestein
      3615  3.5.241          prime    replayed       22512      24384    1.08  6.4e-16  bluestein
      3616  2^5.113          prime    replayed       22093      24403    1.10  7.0e-16  bluestein
      3617  3617             prime    replayed       22147      25197    1.10  7.2e-16  bluestein
      3618  2.3^3.67         prime    replayed       23087      32315    1.38  5.2e-16  bluestein
      3619  7.11.47          chain3   replayed       17216      23077    1.24  4.7e-16  
      3620  2^2.5.181        prime    replayed       22055      24377    1.07  5.7e-16  bluestein
      3621  3.17.71          prime    replayed       22118      33688    1.48  4.7e-16  bluestein
      3622  2.1811           prime    replayed       22124      24377    1.10  8.6e-16  bluestein
      3623  3623             prime    replayed       22115      24445    1.08  7.9e-16  bluestein
      3624  2^3.3.151        prime    replayed       22095      24468    1.08  7.4e-16  bluestein
      3625  5^3.29           chain3   replayed        7519      17711    2.31  5.4e-16  
      3626  2.7^2.37         flat     replayed        9410      19453    2.04  5.9e-16  
      3627  3^2.13.31        flat     replayed        9166      21612    2.08  1.0e-15  
      3628  2^2.907          prime    replayed       22649      24930    1.07  6.5e-16  bluestein
      3629  19.191           prime    replayed       22149      24458    1.06  6.1e-16  bluestein
      3630  2.3.5.11^2       flat     replayed        8008      17182    2.00  6.6e-16  
      3631  3631             prime    replayed       23039      25386    1.06  7.7e-16  bluestein
      3632  2^4.227          prime    replayed       22508      24340    1.06  5.8e-16  bluestein
      3633  3.7.173          prime    replayed       22072      27767    1.10  6.6e-16  bluestein
      3634  2.23.79          prime    replayed       22194      39250    1.73  5.4e-16  bluestein
      3635  5.727            prime    replayed       22613      24988    1.05  6.7e-16  bluestein
      3636  2^2.3^2.101      prime    replayed       22096      26076    1.10  5.3e-16  bluestein
      3637  3637             prime    replayed       22733      24467    1.08  6.8e-16  bluestein
      3638  2.17.107         prime    replayed       22656      24391    0.85  5.2e-16  flips differ 1.26x bluestein
      3639  3.1213           prime    replayed       23166      24409    0.96  6.3e-16  bluestein
      3640  2^3.5.7.13       flat     replayed        5784      15175    2.58  5.9e-16  
      3641  11.331           prime    replayed       22182      24399    1.07  8.0e-16  bluestein
      3642  2.3.607          prime    replayed       22140      24419    1.10  7.1e-16  bluestein
      3643  3643             prime    replayed       22131      25110    1.07  7.7e-16  bluestein
      3644  2^2.911          prime    replayed       22111      24350    1.07  7.6e-16  bluestein
      3645  3^6.5            chain3   replayed        6854      12321    1.75  6.6e-16  
      3646  2.1823           prime    replayed       22155      24421    1.07  7.9e-16  bluestein
      3647  7.521            prime    replayed       22116      24348    1.08  7.6e-16  bluestein
      3648  2^6.3.19         chain3   replayed        4961      18409    3.56  4.6e-16  
      3649  41.89            prime    replayed       22849      45107    1.97  5.7e-16  bluestein
      3650  2.5^2.73         prime    replayed       22125      32875    1.45  5.8e-16  bluestein
      3651  3.1217           prime    replayed       22128      24460    1.10  7.4e-16  bluestein
      3652  2^2.11.83        prime    replayed       22709      39060    1.71  5.1e-16  bluestein
      3653  13.281           prime    replayed       22501      27195    0.89  7.4e-16  flips differ 1.51x bluestein
      3654  2.3^2.7.29       flat     replayed        9384      19772    2.02  7.6e-16  
      3655  5.17.43          chain3   replayed        9929      24090    2.42  5.3e-16  
      3656  2^3.457          prime    replayed       22848      24515    1.07  6.6e-16  bluestein
      3657  3.23.53          prime    replayed       22721      29900    1.31  5.8e-16  bluestein
      3658  2.31.59          prime    replayed       22119      34790    1.53  5.9e-16  bluestein
      3659  3659             prime    replayed       22488      25565    1.10  6.5e-16  bluestein
      3660  2^2.3.5.61       prime    replayed       22965      30556    1.32  6.5e-16  bluestein
      3661  7.523            prime    replayed       22681      24395    1.06  7.3e-16  bluestein
      3662  2.1831           prime    replayed       22156      24422    1.05  7.3e-16  bluestein
      3663  3^2.11.37        flat     replayed        9094      22966    2.51  2.3e-15  
      3664  2^4.229          prime    replayed       22480      24403    1.06  8.0e-16  bluestein
      3665  5.733            prime    replayed       22542      24422    0.93  6.7e-16  bluestein
      3666  2.3.13.47        chain3   replayed        8705      27471    3.11  8.2e-16  
      3667  19.193           prime    replayed       22915      25376    1.06  7.5e-16  bluestein
      3668  2^2.7.131        prime    replayed       23006      24438    1.06  5.0e-16  bluestein
      3669  3.1223           prime    replayed       22077      24700    1.11  6.7e-16  bluestein
      3670  2.5.367          prime    replayed       22132      26124    1.10  6.5e-16  bluestein
      3671  3671             prime    replayed       22497      24428    1.08  5.2e-16  bluestein
      3672  2^3.3^3.17       chain3   replayed        6419      16828    2.54  6.2e-16  
      3673  3673             prime    replayed       21899      24535    1.11  6.3e-16  bluestein
      3674  2.11.167         prime    replayed       22557      24447    1.08  7.3e-16  bluestein
      3675  3.5^2.7^2        chain3   replayed        6471      11196    1.50  5.1e-16  
      3676  2^2.919          prime    replayed       23172      24434    1.05  6.6e-16  bluestein
      3677  3677             prime    replayed       22770      24456    0.78  8.1e-16  flips differ 1.38x bluestein
      3678  2.3.613          prime    replayed       23124      25848    1.06  7.0e-16  bluestein
      3679  13.283           prime    replayed       22991      24437    1.06  7.3e-16  bluestein
      3680  2^5.5.23         flat     replayed        6334      19610    2.91  5.2e-16  
      3681  3^2.409          prime    replayed       22141      24349    1.07  5.9e-16  bluestein
      3682  2.7.263          prime    replayed       22753      25139    1.06  7.2e-16  bluestein
      3683  29.127           prime    replayed       22548      24389    1.08  6.6e-16  bluestein
      3684  2^2.3.307        prime    replayed       23125      26438    0.88  6.3e-16  flips differ 1.40x bluestein
      3685  5.11.67          prime    replayed       22154      30560    1.35  5.3e-16  bluestein
      3686  2.19.97          prime    replayed       22101      43409    1.96  6.8e-16  bluestein
      3687  3.1229           prime    replayed       23115      24445    1.04  7.1e-16  bluestein
      3688  2^3.461          prime    replayed       22161      24395    1.07  7.0e-16  bluestein
      3689  7.17.31          chain3   replayed       11401      20523    1.50  6.8e-16  
      3690  2.3^2.5.41       chain3   replayed        7574      23508    3.10  6.0e-16  
      3691  3691             prime    replayed       22477      24471    1.06  5.8e-16  bluestein
      3692  2^2.13.71        prime    replayed       22546      36063    1.53  5.0e-16  bluestein
      3693  3.1231           prime    replayed       22498      24364    1.05  6.6e-16  bluestein
      3694  2.1847           prime    replayed       22534      24440    1.06  7.3e-16  bluestein
      3695  5.739            prime    replayed       22529      24388    1.08  7.4e-16  bluestein
      3696  2^4.3.7.11       chain3   replayed        5047      15549    2.97  4.1e-16  
      3697  3697             prime    replayed       22104      24437    1.08  6.3e-16  bluestein
      3698  2.43^2           flat     replayed       12001      33448    2.67  2.4e-15  
      3699  3^3.137          prime    replayed       22097      24383    1.07  6.8e-16  bluestein
      3700  2^2.5^2.37       flat     replayed        7362      22592    3.05  6.4e-16  
      3701  3701             prime    replayed       22251      24399    1.07  7.5e-16  bluestein
      3702  2.3.617          prime    replayed       22156      25511    1.11  7.0e-16  bluestein
      3703  7.23^2           chain3   replayed        8488      20882    2.46  6.2e-16  
      3704  2^3.463          prime    replayed       22552      24399    1.08  6.5e-16  bluestein
      3705  3.5.13.19        chain3   replayed        6680      17994    2.66  6.1e-16  
      3706  2.17.109         prime    replayed       22495      24482    1.06  5.6e-16  bluestein
      3707  11.337           prime    replayed       22151      24412    1.10  8.5e-16  bluestein
      3708  2^2.3^2.103      prime    replayed       22482      24353    1.08  6.3e-16  bluestein
      3709  3709             prime    replayed       22156      24553    1.10  5.5e-16  bluestein
      3710  2.5.7.53         prime    replayed       22480      25443    1.13  5.6e-16  bluestein
      3711  3.1237           prime    replayed       22156      25388    1.10  6.8e-16  bluestein
      3712  2^7.29           flat     replayed        8259      21345    2.58  5.1e-16  
      3713  47.79            prime    replayed       22138      46047    1.99  5.1e-16  bluestein
      3714  2.3.619          prime    replayed       23033      24391    1.05  6.7e-16  bluestein
      3715  5.743            prime    replayed       22535      24407    1.04  7.2e-16  bluestein
      3716  2^2.929          prime    replayed       22151      24368    1.06  6.6e-16  bluestein
      3717  3^2.7.59         prime    replayed       22469      28074    1.21  6.1e-16  bluestein
      3718  2.11.13^2        flat     replayed        9009      19920    2.19  1.1e-15  
      3719  3719             prime    replayed       22555      25124    1.08  8.0e-16  bluestein
      3720  2^3.3.5.31       chain3   replayed        6098      21894    3.58  6.3e-16  
      3721  61^2             prime    replayed       22164      43698    1.97  7.1e-16  bluestein
      3722  2.1861           prime    replayed       22142      24404    1.07  7.0e-16  bluestein
      3723  3.17.73          prime    replayed       22514      34521    1.47  4.8e-16  bluestein
      3724  2^2.7^2.19       flat     replayed        6636      17451    2.61  7.9e-16  
      3725  5^2.149          prime    replayed       22775      25172    1.07  6.5e-16  bluestein
      3726  2.3^4.23         chain3   replayed        9851      19049    1.92  5.4e-16  
      3727  3727             prime    replayed       22422      24470    1.06  6.8e-16  bluestein
      3728  2^4.233          prime    replayed       23051      24448    1.06  7.1e-16  bluestein
      3729  3.11.113         prime    replayed       22496      25351    1.08  7.7e-16  bluestein
      3730  2.5.373          prime    replayed       23092      24478    1.06  7.4e-16  bluestein
      3731  7.13.41          chain3   replayed        8629      21397    2.45  6.8e-16  
      3732  2^2.3.311        prime    replayed       23102      24405    0.84  8.7e-16  flips differ 1.26x bluestein
      3733  3733             prime    replayed       23025      24421    1.06  9.9e-16  bluestein
      3734  2.1867           prime    replayed       22777      24380    1.07  6.9e-16  bluestein
      3735  3^2.5.83         prime    replayed       22713      37410    1.62  5.7e-16  bluestein
      3736  2^3.467          prime    replayed       22437      27896    1.09  6.5e-16  flips differ 1.26x bluestein
      3737  37.101           prime    replayed       22197      24458    1.10  8.7e-16  bluestein
      3738  2.3.7.89         prime    replayed       22590      38920    1.69  5.8e-16  bluestein
      3739  3739             prime    replayed       23254      24409    1.05  7.6e-16  bluestein
      3740  2^2.5.11.17      flat     replayed        6469      19688    2.77  6.4e-16  
      3741  3.29.43          flat     replayed       11128      28771    2.58  3.5e-15  
      3742  2.1871           prime    replayed       22414      25494    1.09  7.0e-16  bluestein
      3743  19.197           prime    replayed       22691      24415    1.06  7.0e-16  bluestein
      3744  2^5.3^2.13       chain3   replayed        4632      16384    3.39  5.7e-16  
      3745  5.7.107          prime    replayed       23110      24391    1.03  7.0e-16  bluestein
      3746  2.1873           prime    replayed       22786      24417    1.07  7.2e-16  bluestein
      3747  3.1249           prime    replayed       23072      24488    1.06  8.0e-16  bluestein
      3748  2^2.937          prime    replayed       22156      25356    1.07  6.6e-16  bluestein
      3749  23.163           prime    replayed       22157      24431    1.07  6.6e-16  bluestein
      3750  2.3.5^4          chain3   replayed        5832      12581    2.15  4.6e-16  
      3751  11^2.31          flat     replayed       10674      21787    1.98  8.8e-16  
      3752  2^3.7.67         prime    replayed       22666      33857    1.06  4.6e-16  flips differ 1.41x bluestein
      3753  3^3.139          prime    replayed       22568      24420    1.05  7.1e-16  bluestein
      3754  2.1877           prime    replayed       22571      26449    0.88  6.0e-16  flips differ 1.43x bluestein
      3755  5.751            prime    replayed       22712      25436    1.08  5.5e-16  bluestein
      3756  2^2.3.313        prime    replayed       23045      25469    1.04  7.5e-16  bluestein
      3757  13.17^2          chain3   replayed        7444      19246    2.54  8.5e-16  
      3758  2.1879           prime    replayed       22566      25453    1.09  7.5e-16  bluestein
      3759  3.7.179          prime    replayed       22640      24471    1.07  7.2e-16  bluestein
      3760  2^4.5.47         chain3   replayed        7948      27436    3.44  4.3e-16  
      3761  3761             prime    replayed       22616      25605    1.08  7.5e-16  bluestein
      3762  2.3^2.11.19      flat     replayed        8782      20899    2.37  8.9e-16  
      3763  53.71            prime    replayed       22098      45499    2.00  6.0e-16  bluestein
      3764  2^2.941          prime    replayed       22746      24513    1.07  5.2e-16  bluestein
      3765  3.5.251          prime    replayed       23102      25904    1.06  9.8e-16  bluestein
      3766  2.7.269          prime    replayed       22509      24414    1.08  5.5e-16  bluestein
      3767  3767             prime    replayed       22696      24438    1.06  5.6e-16  bluestein
      3768  2^3.3.157        prime    replayed       22755      24410    1.07  6.6e-16  bluestein
      3769  3769             prime    replayed       22551      25294    1.06  6.3e-16  bluestein
      3770  2.5.13.29        chain3   replayed        9009      21417    2.36  7.1e-16  
      3771  3^2.419          prime    replayed       22216      24397    1.07  8.9e-16  bluestein
      3772  2^2.23.41        flat     replayed        9716      30339    3.11  2.3e-15  
      3773  7^3.11           flat     replayed        7166      12008    1.65  7.6e-16  
      3774  2.3.17.37        chain3   replayed        8266      26134    3.13  5.6e-16  
      3775  5^2.151          prime    replayed       23103      24445    0.77  7.6e-16  flips differ 1.37x bluestein
      3776  2^6.59           prime    replayed       22561      31849    1.39  5.5e-16  bluestein
      3777  3.1259           prime    replayed       22594      24411    1.08  6.3e-16  bluestein
      3778  2.1889           prime    replayed       22196      24423    1.07  7.1e-16  bluestein
      3779  3779             prime    replayed       22538      24429    0.92  8.2e-16  bluestein
      3780  2^2.3^3.5.7      chain3   replayed        5282      13333    2.02  5.5e-16  
      3781  19.199           prime    replayed       22519      25197    1.06  5.9e-16  bluestein
      3782  2.31.61          prime    replayed       22565      36504    1.25  5.9e-16  flips differ 1.29x bluestein
      3783  3.13.97          prime    replayed       23113      41512    1.44  7.8e-16  bluestein
      3784  2^3.11.43        chain3   replayed        8109      28813    3.44  5.4e-16  
      3785  5.757            prime    replayed       22177      24495    1.05  7.1e-16  bluestein
      3786  2.3.631          prime    replayed       22742      25469    1.07  7.4e-16  bluestein
      3787  7.541            prime    replayed       22674      24419    1.07  5.7e-16  bluestein
      3788  2^2.947          prime    replayed       22532      25172    1.05  8.4e-16  bluestein
      3789  3^2.421          prime    replayed       22520      24395    1.08  6.0e-16  bluestein
      3790  2.5.379          prime    replayed       22107      24475    1.11  5.6e-16  bluestein
      3791  17.223           prime    replayed       22557      24443    1.08  6.5e-16  bluestein
      3792  2^4.3.79         prime    replayed       22140      39002    1.71  5.3e-16  bluestein
      3793  3793             prime    replayed       22210      24611    1.10  7.1e-16  bluestein
      3794  2.7.271          prime    replayed       23445      24417    1.04  4.6e-16  bluestein
      3795  3.5.11.23        chain3   replayed        7068      19569    2.72  6.2e-16  
      3796  2^2.13.73        prime    replayed       22122      37240    1.64  6.5e-16  bluestein
      3797  3797             prime    replayed       22521      29583    1.13  6.6e-16  bluestein
      3798  2.3^2.211        prime    replayed       22591      24546    1.08  5.8e-16  bluestein
      3799  29.131           prime    replayed       22509      24491    1.06  7.1e-16  bluestein
      3800  2^3.5^2.19       flat     replayed        6662      18163    2.63  6.7e-16  
      3801  3.7.181          prime    replayed       22760      24431    1.06  5.9e-16  bluestein
      3802  2.1901           prime    replayed       22213      24445    1.07  7.3e-16  bluestein
      3803  3803             prime    replayed       22216      24508    1.08  8.1e-16  bluestein
      3804  2^2.3.317        prime    replayed       22168      24496    1.10  8.9e-16  bluestein
      3805  5.761            prime    replayed       22510      24411    1.08  7.4e-16  bluestein
      3806  2.11.173         prime    replayed       21705      27379    1.13  7.0e-16  bluestein
      3807  3^4.47           chain3   replayed       14866      26122    1.25  6.4e-16  flips differ 1.41x
      3808  2^5.7.17         chain3   replayed        5089      17809    3.24  5.2e-16  
      3809  13.293           prime    replayed       22499      24517    1.05  6.6e-16  bluestein
      3810  2.3.5.127        prime    replayed       22502      24469    1.06  5.7e-16  bluestein
      3811  37.103           prime    replayed       22279      24422    1.08  7.1e-16  bluestein
      3812  2^2.953          prime    replayed       22594      24493    1.06  8.1e-16  bluestein
      3813  3.31.41          flat     replayed       11331      29213    2.40  2.5e-15  
      3814  2.1907           prime    replayed       22177      24445    1.08  6.4e-16  bluestein
      3815  5.7.109          prime    replayed       22523      24422    1.08  6.3e-16  bluestein
      3816  2^3.3^2.53       prime    replayed       23169      29383    1.26  5.1e-16  bluestein
      3817  11.347           prime    replayed       22710      24454    0.89  7.2e-16  bluestein
      3818  2.23.83          prime    replayed       22208      42364    1.83  4.8e-16  bluestein
      3819  3.19.67          prime    replayed       22461      35116    1.54  6.6e-16  bluestein
      3820  2^2.5.191        prime    replayed       22221      24472    1.10  8.3e-16  bluestein
      3821  3821             prime    replayed       22742      24492    1.07  8.6e-16  bluestein
      3822  2.3.7^2.13       flat     replayed        8461      15437    1.75  7.1e-16  
      3823  3823             prime    replayed       23151      24609    1.06  6.2e-16  bluestein
      3824  2^4.239          prime    replayed       23113      24383    0.87  7.1e-16  bluestein
      3825  3^2.5^2.17       chain3   replayed        7859      15611    1.95  8.6e-16  
      3826  2.1913           prime    replayed       22731      24476    1.07  8.7e-16  bluestein
      3827  43.89            prime    replayed       22214      48430    2.09  4.9e-16  bluestein
      3828  2^2.3.11.29      chain3   replayed        6490      24419    3.66  6.0e-16  
      3829  7.547            prime    replayed       22215      24524    1.06  8.2e-16  bluestein
      3830  2.5.383          prime    replayed       22616      24414    1.03  7.6e-16  bluestein
      3831  3.1277           prime    replayed       22757      24439    1.07  8.0e-16  bluestein
      3832  2^3.479          prime    replayed       22681      24515    1.07  5.7e-16  bluestein
      3833  3833             prime    replayed       22505      24451    1.07  7.3e-16  bluestein
      3834  2.3^3.71         prime    replayed       21775      36053    1.66  4.9e-16  bluestein
      3835  5.13.59          prime    replayed       22541      29122    1.29  5.7e-16  bluestein
      3836  2^2.7.137        prime    replayed       22182      24380    1.10  5.0e-16  bluestein
      3837  3.1279           prime    replayed       22812      24464    1.05  7.6e-16  bluestein
      3838  2.19.101         prime    replayed       22709      24366    1.07  6.2e-16  bluestein
      3839  11.349           prime    replayed       22600      24576    1.07  7.2e-16  bluestein
      3840  2^8.3.5          ztt      replayed        4098      15026    3.65  4.1e-16  
      3841  23.167           prime    replayed       22562      24511    1.06  7.2e-16  bluestein
      3842  2.17.113         prime    replayed       22805      27412    0.90  6.4e-16  flips differ 1.47x bluestein
      3843  3^2.7.61         prime    replayed       23108      29917    1.29  5.5e-16  bluestein
      3844  2^2.31^2         chain3   replayed        9117      30727    3.31  7.1e-16  
      3845  5.769            prime    replayed       22186      24415    1.10  7.4e-16  bluestein
      3846  2.3.641          prime    replayed       22156      27801    1.08  7.8e-16  bluestein
      3847  3847             prime    replayed       22525      25573    1.09  6.9e-16  bluestein
      3848  2^3.13.37        chain3   replayed       11404      27408    2.38  6.6e-16  
      3849  3.1283           prime    replayed       22491      24392    1.05  5.5e-16  bluestein
      3850  2.5^2.7.11       flat     replayed        7898      14503    1.83  5.8e-16  
      3851  3851             prime    replayed       22522      24424    1.06  7.8e-16  bluestein
      3852  2^2.3^2.107      prime    replayed       22960      24386    1.05  6.9e-16  bluestein
      3853  3853             prime    replayed       22516      24505    1.08  8.0e-16  bluestein
      3854  2.41.47          flat     raced          13132      35384    2.63  3.0e-15  
      3855  3.5.257          prime    replayed       22684      24508    1.07  7.2e-16  bluestein
      3856  2^4.241          prime    replayed       22187      25213    1.09  6.1e-16  bluestein
      3857  7.19.29          chain3   replayed        8632      21593    2.46  6.5e-16  
      3858  2.3.643          prime    replayed       22198      24463    1.10  7.2e-16  bluestein
      3859  17.227           prime    replayed       22173      24437    1.07  5.6e-16  bluestein
      3860  2^2.5.193        prime    replayed       23130      26549    0.89  7.4e-16  flips differ 1.38x bluestein
      3861  3^3.11.13        flat     replayed        8550      18058    2.05  9.9e-16  
      3862  2.1931           prime    replayed       22755      25875    1.07  7.5e-16  bluestein
      3863  3863             prime    replayed       22738      24507    1.07  9.7e-16  bluestein
      3864  2^3.3.7.23       flat     replayed        7846      20213    2.46  8.0e-16  
      3865  5.773            prime    replayed       22327      25833    1.08  8.6e-16  bluestein
      3866  2.1933           prime    replayed       22558      24569    1.08  6.9e-16  bluestein
      3867  3.1289           prime    replayed       22575      24348    0.87  6.6e-16  bluestein
      3868  2^2.967          prime    replayed       22251      24434    1.07  7.8e-16  bluestein
      3869  53.73            prime    replayed       22547      46738    2.07  5.8e-16  bluestein
      3870  2.3^2.5.43       chain3   replayed        8156      25883    3.15  6.0e-16  
      3871  7^2.79           prime    replayed       22556      32878    1.42  6.0e-16  bluestein
      3872  2^5.11^2         chain3   replayed        5385      19723    3.62  4.2e-16  
      3873  3.1291           prime    replayed       23485      25761    1.04  6.4e-16  bluestein
      3874  2.13.149         prime    replayed       22994      24434    1.04  6.4e-16  bluestein
      3875  5^3.31           chain3   replayed        8534      20144    2.21  7.1e-16  
      3876  2^2.3.17.19      chain3   replayed        6689      24057    3.44  6.2e-16  
      3877  3877             prime    replayed       22248      24473    1.07  7.1e-16  bluestein
      3878  2.7.277          prime    replayed       23180      24452    1.05  6.3e-16  bluestein
      3879  3^2.431          prime    replayed       22857      24437    0.85  6.8e-16  flips differ 1.26x bluestein
      3880  2^3.5.97         prime    replayed       22980      44966    1.95  5.3e-16  bluestein
      3881  3881             prime    replayed       22585      27205    0.88  7.4e-16  flips differ 1.51x bluestein
      3882  2.3.647          prime    replayed       22154      24465    1.08  6.4e-16  bluestein
      3883  11.353           prime    replayed       22971      24502    0.88  6.4e-16  bluestein
      3884  2^2.971          prime    replayed       22484      24466    1.09  6.7e-16  bluestein
      3885  3.5.7.37         chain3   replayed        8585      21021    2.38  5.0e-16  
      3886  2.29.67          prime    replayed       22569      38944    1.34  6.3e-16  flips differ 1.28x bluestein
      3887  13^2.23          flat     replayed        8652      20719    2.28  9.1e-16  
      3888  2^4.3^5          ztt      replayed        4472      15505    3.40  4.6e-16  
      3889  3889             prime    replayed       16030      24462    1.51  1.5e-15  rader
      3890  2.5.389          prime    replayed       23088      24479    1.05  6.2e-16  bluestein
      3891  3.1297           prime    replayed       22579      24491    1.04  7.3e-16  bluestein
      3892  2^2.7.139        prime    replayed       22157      24478    1.10  6.3e-16  bluestein
      3893  17.229           prime    replayed       22597      26781    1.14  6.6e-16  bluestein
      3894  2.3.11.59        prime    replayed       23001      32776    1.42  5.0e-16  bluestein
      3895  5.19.41          chain3   replayed       10481      25580    1.82  5.1e-16  flips differ 1.34x
      3896  2^3.487          prime    replayed       22596      24737    1.08  8.0e-16  bluestein
      3897  3^2.433          prime    replayed       22971      24482    1.06  6.1e-16  bluestein
      3898  2.1949           prime    replayed       22236      24441    0.77  6.7e-16  flips differ 1.43x bluestein
      3899  7.557            prime    replayed       22506      24439    1.08  8.0e-16  bluestein
      3900  2^2.3.5^2.13     chain3   replayed        5798      16449    2.81  7.0e-16  
      3901  47.83            prime    replayed       22149      49493    2.23  6.6e-16  bluestein
      3902  2.1951           prime    replayed       22583      24435    1.06  7.9e-16  bluestein
      3903  3.1301           prime    replayed       23140      24484    1.05  7.9e-16  bluestein
      3904  2^6.61           prime    replayed       22744      33590    1.47  4.4e-16  bluestein
      3905  5.11.71          prime    replayed       22932      33924    1.48  5.7e-16  bluestein
      3906  2.3^2.7.31       chain3   replayed        8184      22516    2.46  6.4e-16  
      3907  3907             prime    replayed       22557      27107    1.08  8.6e-16  bluestein
      3908  2^2.977          prime    replayed       22649      24556    1.05  8.1e-16  bluestein
      3909  3.1303           prime    replayed       22703      24427    1.07  5.5e-16  bluestein
      3910  2.5.17.23        chain3   replayed        8216      22692    2.26  7.2e-16  
      3911  3911             prime    replayed       23171      24490    1.05  7.5e-16  bluestein
      3912  2^3.3.163        prime    replayed       22251      25046    1.10  6.4e-16  bluestein
      3913  7.13.43          chain3   replayed       14172      23974    1.31  7.6e-16  flips differ 1.29x
      3914  2.19.103         prime    replayed       23163      24468    0.85  5.9e-16  bluestein
      3915  3^3.5.29         flat     replayed        9478      20299    2.05  6.5e-16  
      3916  2^2.11.89        prime    replayed       22148      43450    1.96  5.2e-16  bluestein
      3917  3917             prime    replayed       22586      24501    1.08  7.8e-16  bluestein
      3918  2.3.653          prime    replayed       22686      25451    1.08  6.7e-16  bluestein
      3919  3919             prime    replayed       21798      24538    1.10  6.4e-16  bluestein
      3920  2^4.5.7^2        ztt      replayed        4233      14061    3.31  3.8e-16  
      3921  3.1307           prime    replayed       22657      27317    0.89  6.6e-16  flips differ 1.50x bluestein
      3922  2.37.53          prime    replayed       22274      36385    1.08  5.1e-16  flips differ 1.51x bluestein
      3923  3923             prime    replayed       22217      25865    1.10  5.6e-16  bluestein
      3924  2^2.3^2.109      prime    replayed       22634      24443    1.08  6.7e-16  bluestein
      3925  5^2.157          prime    replayed       22252      26193    1.08  8.2e-16  bluestein
      3926  2.13.151         prime    replayed       22600      24450    1.08  5.9e-16  bluestein
      3927  3.7.11.17        chain3   replayed        7592      18578    1.97  6.2e-16  
      3928  2^3.491          prime    replayed       22333      24507    1.08  6.4e-16  bluestein
      3929  3929             prime    replayed       22543      24546    1.08  6.1e-16  bluestein
      3930  2.3.5.131        prime    replayed       22205      24455    1.07  6.1e-16  bluestein
      3931  3931             prime    replayed       22685      24524    1.07  7.5e-16  bluestein
      3932  2^2.983          prime    replayed       23168      25534    1.06  6.1e-16  bluestein
      3933  3^2.19.23        chain3   replayed        8119      24107    2.58  7.4e-16  
      3934  2.7.281          prime    replayed       22230      24559    1.10  7.5e-16  bluestein
      3935  5.787            prime    replayed       22830      24466    1.07  6.3e-16  bluestein
      3936  2^5.3.41         chain3   replayed        7188      26236    3.64  4.4e-16  
      3937  31.127           prime    replayed       22249      25506    1.10  6.1e-16  bluestein
      3938  2.11.179         prime    replayed       22303      24502    1.09  8.4e-16  bluestein
      3939  3.13.101         prime    replayed       22770      24494    1.07  6.5e-16  bluestein
      3940  2^2.5.197        prime    replayed       22625      26645    1.08  7.4e-16  bluestein
      3941  7.563            prime    replayed       22710      24479    1.07  7.2e-16  bluestein
      3942  2.3^3.73         prime    replayed       22568      37096    1.60  5.7e-16  bluestein
      3943  3943             prime    replayed       22838      24448    1.06  7.1e-16  bluestein
      3944  2^3.17.29        flat     replayed        8307      27424    3.12  1.1e-15  
      3945  3.5.263          prime    replayed       23218      24434    1.05  7.4e-16  bluestein
      3946  2.1973           prime    replayed       22812      24482    1.07  6.7e-16  bluestein
      3947  3947             prime    replayed       23231      24725    1.05  7.7e-16  bluestein
      3948  2^2.3.7.47       flat     replayed       11616      28564    2.45  7.0e-16  
      3949  11.359           prime    replayed       22628      24412    1.05  6.5e-16  bluestein
      3950  2.5^2.79         prime    replayed       22586      38440    1.65  6.5e-16  bluestein
      3951  3^2.439          prime    replayed       22808      25523    1.07  7.3e-16  bluestein
      3952  2^4.13.19        chain3   replayed        5800      22992    3.73  4.9e-16  
      3953  59.67            prime    replayed       22029      48517    2.17  5.9e-16  bluestein
      3954  2.3.659          prime    replayed       23227      24519    1.05  6.8e-16  bluestein
      3955  5.7.113          prime    replayed       22306      24428    1.07  8.0e-16  bluestein
      3956  2^2.23.43        chain3   replayed        9948      32926    3.30  6.9e-16  
      3957  3.1319           prime    replayed       22612      24441    0.83  6.7e-16  flips differ 1.29x bluestein
      3958  2.1979           prime    replayed       22297      25755    1.07  7.5e-16  bluestein
      3959  37.107           prime    replayed       22644      24530    1.06  7.8e-16  bluestein
      3960  2^3.3^2.5.11     flat     replayed        6750      16519    2.43  4.5e-16  
      3961  17.233           prime    replayed       22637      24465    1.05  6.1e-16  bluestein
      3962  2.7.283          prime    replayed       22242      24429    1.07  1.0e-15  bluestein
      3963  3.1321           prime    replayed       22836      24557    1.07  6.5e-16  bluestein
      3964  2^2.991          prime    replayed       22291      26322    1.08  6.0e-16  bluestein
      3965  5.13.61          prime    replayed       21910      30537    1.37  6.2e-16  bluestein
      3966  2.3.661          prime    replayed       22637      24493    1.06  8.1e-16  bluestein
      3967  3967             prime    replayed       22603      24520    1.08  7.5e-16  bluestein
      3968  2^7.31           chain3   replayed        6512      24210    3.66  5.2e-16  
      3969  3^4.7^2          chain3   replayed        6648      13023    1.81  8.0e-16  
      3970  2.5.397          prime    replayed       22676      24551    1.07  9.2e-16  bluestein
      3971  11.19^2          chain3   replayed        8471      22368    2.51  6.4e-16  
      3972  2^2.3.331        prime    replayed       23196      24446    0.76  6.3e-16  flips differ 1.39x bluestein
      3973  29.137           prime    replayed       23264      25424    1.05  7.4e-16  bluestein
      3974  2.1987           prime    replayed       22904      27228    1.07  6.7e-16  bluestein
      3975  3.5^2.53         prime    replayed       22719      28371    1.23  5.6e-16  bluestein
      3976  2^3.7.71         prime    replayed       22640      37582    1.39  6.1e-16  bluestein
      3977  41.97            prime    replayed       22608      51932    1.93  6.3e-16  bluestein
      3978  2.3^2.13.17      flat     replayed        9186      21289    2.22  1.4e-15  
      3979  23.173           prime    replayed       21936      24416    1.09  6.6e-16  bluestein
      3980  2^2.5.199        prime    replayed       22646      26195    1.08  8.1e-16  bluestein
      3981  3.1327           prime    replayed       22975      24519    1.06  6.1e-16  bluestein
      3982  2.11.181         prime    replayed       22642      25377    1.08  7.1e-16  bluestein
      3983  7.569            prime    replayed       22614      24446    1.05  8.0e-16  bluestein
      3984  2^4.3.83         prime    replayed       22769      42159    1.84  5.1e-16  bluestein
      3985  5.797            prime    replayed       22179      24467    1.10  6.6e-16  bluestein
      3986  2.1993           prime    replayed       22248      24507    1.07  7.7e-16  bluestein
      3987  3^2.443          prime    replayed       22718      24435    1.05  8.5e-16  bluestein
      3988  2^2.997          prime    replayed       22259      24576    1.10  7.3e-16  bluestein
      3989  3989             prime    replayed       23482      24497    1.04  6.0e-16  bluestein
      3990  2.3.5.7.19       chain3   replayed        7084      18601    2.61  5.3e-16  
      3991  13.307           prime    replayed       22246      24473    1.10  7.0e-16  bluestein
      3992  2^3.499          prime    replayed       22225      24427    1.07  7.5e-16  bluestein
      3993  3.11^3           flat     replayed        7726      20092    2.59  8.3e-16  
      3994  2.1997           prime    replayed       22665      24528    1.04  6.0e-16  bluestein
      3995  5.17.47          chain3   replayed       15196      28181    1.21  6.7e-16  flips differ 1.53x
      3996  2^2.3^3.37       chain3   replayed        7200      25206    3.48  5.6e-16  
      3997  7.571            prime    replayed       22642      24443    1.05  8.7e-16  bluestein
      3998  2.1999           prime    replayed       22819      24465    1.07  6.1e-16  bluestein
      3999  3.31.43          chain3   replayed       14682      32041    1.87  7.3e-16  
      4000  2^5.5^3          ztt      replayed        3986      14752    3.68  4.3e-16  
      4001  4001             prime    replayed       15197      24615    1.61  7.8e-16  rader
      4002  2.3.23.29        flat     replayed       10517      28002    2.65  2.3e-15  
      4003  4003             prime    replayed       22263      24546    1.08  8.4e-16  bluestein
      4004  2^2.7.11.13      flat     replayed        6857      19748    2.83  8.2e-16  
      4005  3^2.5.89         prime    replayed       23280      41741    1.77  6.5e-16  bluestein
      4006  2.2003           prime    replayed       22793      24506    1.06  5.7e-16  bluestein
      4007  4007             prime    replayed       22249      24519    0.92  9.7e-16  bluestein
      4008  2^3.3.167        prime    replayed       22884      25406    1.07  6.8e-16  bluestein
      4009  19.211           prime    replayed       23222      24549    1.05  6.9e-16  bluestein
      4010  2.5.401          prime    replayed       22606      24452    1.05  7.3e-16  bluestein
      4011  3.7.191          prime    replayed       23008      25010    1.06  8.7e-16  bluestein
      4012  2^2.17.59        prime    replayed       23031      36246    1.56  6.0e-16  bluestein
      4013  4013             prime    replayed       22970      24423    1.06  7.0e-16  bluestein
      4014  2.3^2.223        prime    replayed       22623      24467    1.06  7.2e-16  bluestein
      4015  5.11.73          prime    replayed       22281      35047    1.04  5.8e-16  flips differ 1.51x bluestein
      4016  2^4.251          prime    replayed       23234      24529    0.91  7.2e-16  bluestein
      4017  3.13.103         prime    replayed       22840      24468    1.07  7.9e-16  bluestein
      4018  2.7^2.41         flat     replayed       11174      22623    1.99  3.8e-15  
      4019  4019             prime    replayed       23235      26206    1.05  9.4e-16  bluestein
      4020  2^2.3.5.67       prime    replayed       23265      36399    1.56  5.7e-16  bluestein
      4021  4021             prime    replayed       22285      24467    1.08  6.4e-16  bluestein
      4022  2.2011           prime    replayed       22703      24432    1.05  7.7e-16  bluestein
      4023  3^3.149          prime    replayed       22716      26368    1.12  6.6e-16  bluestein
      4024  2^3.503          prime    replayed       22644      24470    1.05  7.5e-16  bluestein
      4025  5^2.7.23         chain3   replayed        7972      18342    2.14  5.7e-16  
      4026  2.3.11.61        prime    replayed       22789      34542    1.51  6.0e-16  bluestein
      4027  4027             prime    replayed       22802      24461    1.07  7.1e-16  bluestein
      4028  2^2.19.53        prime    replayed       22673      35243    1.52  6.2e-16  bluestein
      4029  3.17.79          prime    replayed       22647      40332    1.73  5.9e-16  bluestein
      4030  2.5.13.31        flat     replayed       11536      24331    2.10  6.3e-16  
      4031  29.139           prime    replayed       22691      25514    1.08  8.5e-16  bluestein
      4032  2^6.3^2.7        ztt      replayed        4337      15424    3.54  3.6e-16  
      4033  37.109           prime    replayed       23052      27816    1.14  7.2e-16  bluestein
      4034  2.2017           prime    replayed       23192      24480    1.05  7.4e-16  bluestein
      4035  3.5.269          prime    replayed       22619      24496    1.05  9.2e-16  bluestein
      4036  2^2.1009         prime    replayed       22577      24544    1.07  7.2e-16  bluestein
      4037  11.367           prime    replayed       22304      24448    1.07  7.4e-16  bluestein
      4038  2.3.673          prime    replayed       22324      24603    1.07  9.2e-16  bluestein
      4039  7.577            prime    replayed       22622      25283    1.06  7.5e-16  bluestein
      4040  2^3.5.101        prime    replayed       22668      24454    1.04  4.6e-16  bluestein
      4041  3^2.449          prime    replayed       22768      24526    1.07  7.7e-16  bluestein
      4042  2.43.47          flat     raced          13309      38431    2.79  2.8e-15  
      4043  13.311           prime    replayed       22315      25453    1.07  7.6e-16  bluestein
      4044  2^2.3.337        prime    replayed       22883      25390    1.07  8.8e-16  bluestein
      4045  5.809            prime    replayed       22618      24668    1.09  8.3e-16  bluestein
      4046  2.7.17^2         flat     replayed        9768      20334    1.62  9.0e-16  flips differ 1.29x
      4047  3.19.71          prime    replayed       22245      38978    1.75  6.8e-16  bluestein
      4048  2^4.11.23        chain3   replayed        6345      24807    3.87  6.7e-16  
      4049  4049             prime    replayed       22258      24513    1.07  7.6e-16  bluestein
      4050  2.3^4.5^2        chain3   replayed        7177      14200    1.67  6.5e-16  
      4051  4051             prime    replayed       23280      25578    1.05  8.1e-16  bluestein
      4052  2^2.1013         prime    replayed       22319      24873    1.08  7.4e-16  bluestein
      4053  3.7.193          prime    replayed       22945      25514    1.01  7.0e-16  bluestein
      4054  2.2027           prime    replayed       22270      25473    0.78  7.9e-16  flips differ 1.42x bluestein
      4055  5.811            prime    replayed       22273      25954    0.86  7.3e-16  flips differ 1.28x bluestein
      4056  2^3.3.13^2       flat     replayed        7387      20988    2.49  8.7e-16  
      4057  4057             prime    replayed       22350      24545    1.10  8.2e-16  bluestein
      4058  2.2029           prime    replayed       22638      24510    1.07  7.3e-16  bluestein
      4059  3^2.11.41        chain3   replayed       16898      26765    1.51  5.4e-16  
      4060  2^2.5.7.29       flat     replayed        8910      22042    2.47  6.2e-16  
      4061  31.131           prime    replayed       22242      24570    1.07  7.8e-16  bluestein
      4062  2.3.677          prime    replayed       22685      25443    1.08  7.7e-16  bluestein
      4063  17.239           prime    replayed       22672      24497    1.06  7.0e-16  bluestein
      4064  2^5.127          prime    replayed       22226      25542    1.10  6.0e-16  bluestein
      4065  3.5.271          prime    replayed       22323      24588    1.10  6.9e-16  bluestein
      4066  2.19.107         prime    replayed       22814      24563    1.07  6.1e-16  bluestein
      4067  7^2.83           prime    replayed       22287      35698    1.60  6.9e-16  bluestein
      4068  2^2.3^2.113      prime    replayed       23234      26454    1.05  6.0e-16  bluestein
      4069  13.313           prime    replayed       22243      24570    1.07  6.6e-16  bluestein
      4070  2.5.11.37        chain3   replayed       15646      25603    1.48  4.5e-16  
      4071  3.23.59          prime    replayed       22648      35937    1.54  4.5e-16  bluestein
      4072  2^3.509          prime    replayed       22931      24618    1.05  7.5e-16  bluestein
      4073  4073             prime    replayed       22623      25462    1.09  6.4e-16  bluestein
      4074  2.3.7.97         prime    replayed       23312      45373    1.94  6.1e-16  bluestein
      4075  5^2.163          prime    replayed       22704      25557    1.05  6.9e-16  bluestein
      4076  2^2.1019         prime    replayed       22246      24489    0.86  6.7e-16  flips differ 1.29x bluestein
      4077  3^3.151          prime    replayed       22724      27628    0.90  8.7e-16  flips differ 1.50x bluestein
      4078  2.2039           prime    replayed       22594      25501    1.08  6.0e-16  bluestein
      4079  4079             prime    replayed       22671      24676    1.07  7.7e-16  bluestein
      4080  2^4.3.5.17       chain3   replayed        5845      19040    3.00  5.2e-16  
      4081  7.11.53          prime    replayed       28556      27785    0.95  6.3e-16  bluestein
      4082  2.13.157         prime    replayed       23098      24491    1.05  8.0e-16  bluestein
      4083  3.1361           prime    replayed       23197      24942    1.06  7.1e-16  bluestein
      4084  2^2.1021         prime    replayed       22696      24600    1.06  7.5e-16  bluestein
      4085  5.19.43          flat     replayed       12369      28399    2.22  1.5e-15  
      4086  2.3^2.227        prime    replayed       22838      24672    1.07  6.6e-16  bluestein
      4087  61.67            prime    replayed       22682      50800    2.15  6.1e-16  bluestein
      4088  2^3.7.73         prime    replayed       22702      38662    1.44  6.1e-16  bluestein
      4089  3.29.47          chain3   replayed       16699      33267    1.41  5.9e-16  flips differ 1.42x
      4090  2.5.409          prime    replayed       22333      25274    0.85  7.2e-16  flips differ 1.30x bluestein
      4091  4091             prime    replayed       22559      25267    1.09  7.4e-16  bluestein
      4092  2^2.3.11.31      chain3   replayed        7166      27464    3.81  4.8e-16  
      4093  4093             prime    replayed       23246      24505    1.05  8.1e-16  bluestein
      4094  2.23.89          prime    replayed       22888      47045    2.03  6.7e-16  bluestein
      4095  3^2.5.7.13       chain3   replayed        7197      15610    2.06  6.8e-16  
      4096  2^12             ztt      replayed        3556       4718    1.32  3.9e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     2703     79    581   0.91   1.06   1.42    1.08
 chain3     632      0      1   1.65   2.62   3.45    2.50
 flat       441      0      1   1.78   2.32   2.90    2.28
 2p         233      0      0   1.56   2.77   4.40    2.66
 ztt         24      0      0   1.32   3.43   3.99    3.09
 ALL       4033     79    583   0.94   1.09   2.93    1.42
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127               64   2.15      2      0
 128..511             384   1.94     88      3
 512..2047           1536   1.04    298     43
 2048..4096          2049   1.08    195     33
```


## by family
```
 family                                       cells median   <1.0  gmean
 composite with a prime >= 53 (prime cell)     2157   1.06    464   1.07
 chain3                                         632   2.62      1   2.50
 flat                                           441   2.32      1   2.28
 prime N, bluestein                             401   1.04    115   1.01
 2p                                             229   2.78      0   2.70
 prime N, rader                                 145   1.55      2   1.63
 ztt                                             21   3.51      0   3.53
 pow2                                             7   1.12      0   1.17
```


flip agreement: our two readings more than 25% apart at 215 of 4033 cells.

worst 10: 2135 (prime 0.51), 2065 (prime 0.52), 530 (prime 0.61), 2555 (prime 0.61), 1113 (prime 0.61), 2067 (prime 0.63), 565 (prime 0.64), 265 (prime 0.64), 2301 (prime 0.66), 549 (prime 0.66)
best 5: 96 (2p 6.03), 192 (2p 5.46), 160 (2p 5.29), 80 (2p 5.19), 88 (2p 5.17)
