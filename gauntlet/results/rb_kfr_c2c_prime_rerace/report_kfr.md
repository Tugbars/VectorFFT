# gauntlet report

run: `rb_kfr_c2c_prime_rerace`  contract file suffix: `_kfr`  cells: 583 listed, 583 benched, comparator: KFR

control cell: 14 readings, 1.314..1.342 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        98  2.7^2            flat     raced            116        102    0.64  5.0e-16  flips differ 1.37x
       107  107              prime    raced            435        439    0.94  8.8e-16  bluestein
       134  2.67             prime    raced            569        874    1.52  7.2e-16  bluestein
       159  3.53             prime    raced            600        819    1.36  4.5e-16  bluestein
       167  167              prime    raced            751        954    1.23  4.6e-16  bluestein
       173  173              prime    raced            813        928    1.13  4.8e-16  bluestein
       179  179              prime    raced            732        928    1.26  6.3e-16  bluestein
       202  2.101            prime    raced            964        947    0.82  3.6e-16  bluestein
       206  2.103            prime    raced            968        941    0.78  5.7e-16  bluestein
       214  2.107            prime    raced            971        992    0.90  6.1e-16  bluestein
       218  2.109            prime    raced            978        937    0.96  4.2e-16  bluestein
       223  223              prime    raced            968        944    0.74  8.4e-16  flips differ 1.32x rader
       226  2.113            prime    raced            977        941    0.96  5.5e-16  bluestein
       227  227              prime    raced            980        941    0.96  7.0e-16  bluestein
       233  233              prime    raced            904        954    0.91  7.0e-16  rader
       239  239              prime    raced            988        951    0.96  7.3e-16  bluestein
       254  2.127            prime    raced            992        953    0.88  3.8e-16  bluestein
       265  5.53             prime    raced           1395       1387    0.95  6.3e-16  bluestein
       268  2^2.67           prime    raced           1383       1855    1.30  8.3e-16  bluestein
       269  269              prime    raced           1396       2095    1.48  7.7e-16  bluestein
       274  2.137            prime    raced           1402       2078    1.46  5.8e-16  bluestein
       278  2.139            prime    raced           1408       2079    1.44  7.0e-16  bluestein
       283  283              prime    raced           1402       2089    1.48  7.5e-16  bluestein
       284  2^2.71           prime    raced           1398       2076    1.45  6.6e-16  bluestein
       293  293              prime    raced           2107       2093    0.98  5.8e-16  bluestein
       295  5.59             prime    raced           1339       1704    1.26  5.1e-16  bluestein
       298  2.149            prime    raced           1332       2113    1.58  5.9e-16  bluestein
       302  2.151            prime    raced           1338       2083    1.55  5.6e-16  bluestein
       305  5.61             prime    raced           1336       1802    1.34  6.1e-16  bluestein
       309  3.103            prime    raced           1342       2109    1.56  6.9e-16  bluestein
       314  2.157            prime    raced           1343       2081    1.54  5.3e-16  bluestein
       317  317              prime    raced           1347       2099    1.55  8.9e-16  bluestein
       318  2.3.53           prime    raced           1354       1719    1.27  5.9e-16  bluestein
       321  3.107            prime    raced           1922       2085    1.08  5.1e-16  bluestein
       326  2.163            prime    raced           1916       2094    1.07  5.2e-16  bluestein
       327  3.109            prime    raced           1912       2089    1.07  7.5e-16  bluestein
       334  2.167            prime    raced           1896       2095    1.08  5.8e-16  bluestein
       335  5.67             prime    raced           1896       2206    1.11  5.0e-16  bluestein
       339  3.113            prime    raced           2173       2097    0.95  5.1e-16  bluestein
       346  2.173            prime    raced           2174       2095    0.94  7.1e-16  bluestein
       347  347              prime    raced           2183       2100    0.96  4.8e-16  bluestein
       354  2.3.59           prime    raced           2176       2138    0.83  4.1e-16  bluestein
       358  2.179            prime    raced           2099       2099    1.00  5.2e-16  bluestein
       359  359              prime    raced           2118       2099    0.99  6.7e-16  bluestein
       362  2.181            prime    raced           1663       2105    1.25  8.5e-16  bluestein
       366  2.3.61           prime    raced           1663       2265    1.36  5.0e-16  bluestein
       367  367              prime    raced           1670       2119    1.26  7.1e-16  bluestein
       371  7.53             prime    raced           1683       1931    1.13  6.3e-16  bluestein
       373  373              prime    raced           1887       2107    1.11  9.1e-16  rader
       381  3.127            prime    raced           1676       2114    1.25  6.7e-16  bluestein
       382  2.191            prime    raced           1694       2119    1.23  7.3e-16  bluestein
       383  383              prime    raced           1707       2108    1.16  6.9e-16  bluestein
       386  2.193            prime    raced           2151       2113    0.93  6.5e-16  bluestein
       389  389              prime    raced           2192       2122    0.97  5.2e-16  bluestein
       393  3.131            prime    raced           2125       2116    0.98  8.0e-16  bluestein
       394  2.197            prime    raced           2127       2109    0.98  6.0e-16  bluestein
       398  2.199            prime    raced           2131       2130    0.99  5.6e-16  bluestein
       404  2^2.101          prime    raced           2139       2126    0.99  4.9e-16  bluestein
       411  3.137            prime    raced           2152       2118    0.98  7.2e-16  bluestein
       412  2^2.103          prime    raced           2132       2123    0.93  5.5e-16  bluestein
       417  3.139            prime    raced           2121       2120    0.98  6.4e-16  bluestein
       419  419              prime    raced           2132       2129    0.99  6.0e-16  bluestein
       422  2.211            prime    raced           2146       2129    0.98  6.9e-16  bluestein
       428  2^2.107          prime    raced           2143       2136    0.99  5.4e-16  bluestein
       431  431              prime    raced           2144       2134    0.99  7.7e-16  bluestein
       436  2^2.109          prime    raced           2151       2150    0.98  5.4e-16  bluestein
       439  439              prime    raced           2151       2136    0.98  6.7e-16  bluestein
       443  443              prime    raced           2138       2141    0.99  6.1e-16  bluestein
       446  2.223            prime    raced           2163       2129    0.77  8.6e-16  flips differ 1.28x bluestein
       447  3.149            prime    raced           2227       2143    0.96  7.3e-16  bluestein
       452  2^2.113          prime    raced           2642       2149    0.81  5.6e-16  bluestein
       453  3.151            prime    raced           2154       2136    0.99  6.9e-16  bluestein
       454  2.227            prime    raced           2144       2133    0.94  7.6e-16  bluestein
       458  2.229            prime    raced           2192       2138    0.94  5.9e-16  bluestein
       461  461              prime    raced           2170       2142    0.98  7.4e-16  bluestein
       466  2.233            prime    raced           2167       2144    0.98  8.3e-16  bluestein
       467  467              prime    raced           2155       2142    0.99  7.0e-16  bluestein
       471  3.157            prime    raced           2165       2148    0.99  7.7e-16  bluestein
       478  2.239            prime    raced           2151       2145    0.99  6.5e-16  bluestein
       479  479              prime    raced           2246       2148    0.79  6.1e-16  bluestein
       482  2.241            prime    raced           2200       2156    0.96  7.3e-16  bluestein
       487  487              prime    raced           2263       2166    0.94  3.6e-16  bluestein
       489  3.163            prime    raced           2178       2155    0.96  6.9e-16  bluestein
       499  499              prime    raced           2193       2164    0.70  8.3e-16  flips differ 1.42x bluestein
       501  3.167            prime    raced           2195       2169    0.98  7.5e-16  bluestein
       502  2.251            prime    raced           2205       2177    0.97  5.9e-16  bluestein
       503  503              prime    raced           2194       2159    0.98  5.6e-16  bluestein
       505  5.101            prime    raced           2217       2171    0.96  6.3e-16  bluestein
       508  2^2.127          prime    raced           2198       2159    0.98  5.1e-16  bluestein
       509  509              prime    raced           2182       2156    0.97  7.1e-16  bluestein
       514  2.257            prime    raced           3307       4541    1.37  8.8e-16  bluestein
       515  5.103            prime    raced           3318       4550    1.37  5.7e-16  bluestein
       519  3.173            prime    raced           3272       4542    1.31  5.6e-16  bluestein
       524  2^2.131          prime    raced           3323       4514    1.35  5.7e-16  bluestein
       526  2.263            prime    raced           3306       4571    1.36  7.3e-16  bluestein
       530  2.5.53           prime    raced           3316       2898    0.87  6.4e-16  bluestein
       531  3^2.59           prime    raced           3287       3239    0.96  5.7e-16  bluestein
       534  2.3.89           prime    raced           3353       4741    1.38  6.1e-16  bluestein
       535  5.107            prime    raced           3311       4511    1.36  6.6e-16  bluestein
       536  2^3.67           prime    raced           3311       4591    1.38  5.9e-16  bluestein
       537  3.179            prime    raced           3509       4833    1.32  7.8e-16  bluestein
       538  2.269            prime    raced           3364       4634    1.29  6.1e-16  bluestein
       542  2.271            prime    raced           3297       4536    1.37  8.9e-16  bluestein
       543  3.181            prime    raced           3315       4811    1.36  6.6e-16  bluestein
       545  5.109            prime    raced           3324       4519    1.33  6.2e-16  bluestein
       548  2^2.137          prime    raced           3319       4547    1.36  7.5e-16  bluestein
       549  3^2.61           prime    raced           3314       3423    1.02  5.0e-16  bluestein
       553  7.79             prime    raced           3320       4366    1.30  4.6e-16  bluestein
       554  2.277            prime    raced           3319       4516    1.33  7.1e-16  bluestein
       556  2^2.139          prime    raced           3325       4534    1.36  7.1e-16  bluestein
       557  557              prime    raced           3322       4624    1.36  7.1e-16  bluestein
       562  2.281            prime    raced           3309       4523    1.35  7.3e-16  bluestein
       563  563              prime    raced           3344       4520    1.34  7.8e-16  bluestein
       565  5.113            prime    raced           3323       4584    1.33  7.9e-16  bluestein
       566  2.283            prime    raced           3304       4510    1.35  7.3e-16  bluestein
       569  569              prime    raced           3310       4528    1.34  9.2e-16  bluestein
       573  3.191            prime    raced           3326       4538    1.35  8.5e-16  bluestein
       579  3.193            prime    raced           4676       4594    0.96  7.1e-16  bluestein
       582  2.3.97           prime    raced           4412       5593    1.21  7.2e-16  bluestein
       583  11.53            prime    raced           4490       3574    0.76  6.5e-16  bluestein
       586  2.293            prime    raced           4422       4522    1.00  7.6e-16  bluestein
       587  587              prime    raced           4498       4549    0.96  6.2e-16  bluestein
       590  2.5.59           prime    raced           4744       3595    0.75  4.3e-16  bluestein
       591  3.197            prime    raced           4751       4539    0.94  5.8e-16  bluestein
       596  2^2.149          prime    raced           4756       4530    0.95  6.2e-16  bluestein
       597  3.199            prime    raced           4804       4551    0.94  4.9e-16  bluestein
       599  599              prime    raced           4793       4551    0.94  5.2e-16  bluestein
       603  3^2.67           prime    raced           3704       4159    1.12  5.2e-16  bluestein
       604  2^2.151          prime    raced           3735       4571    1.22  6.4e-16  bluestein
       606  2.3.101          prime    raced           3668       4641    1.25  5.4e-16  bluestein
       607  607              prime    raced           3710       4543    1.22  6.1e-16  bluestein
       610  2.5.61           prime    raced           3704       3803    1.01  1.0e-15  bluestein
       614  2.307            prime    raced           4806       4542    0.94  5.8e-16  bluestein
       618  2.3.103          prime    raced           3713       4553    1.22  6.2e-16  bluestein
       619  619              prime    raced           3785       4549    1.18  6.0e-16  bluestein
       622  2.311            prime    raced           3733       4624    1.22  6.0e-16  bluestein
       626  2.313            prime    raced           3705       4544    1.22  6.4e-16  bluestein
       628  2^2.157          prime    raced           3705       4560    1.23  5.7e-16  bluestein
       633  3.211            prime    raced           3703       4547    1.20  4.5e-16  bluestein
       634  2.317            prime    raced           3704       4555    1.23  4.3e-16  bluestein
       635  5.127            prime    raced           3711       4628    1.24  6.0e-16  bluestein
       636  2^2.3.53         prime    raced           3658       4532    1.22  6.0e-16  bluestein
       639  3^2.71           prime    raced           3713       4751    1.27  7.7e-16  bluestein
       642  2.3.107          prime    raced           3985       4573    1.13  5.2e-16  bluestein
       643  643              prime    raced           4027       4610    1.14  8.0e-16  bluestein
       647  647              prime    raced           3997       4552    1.11  7.4e-16  bluestein
       649  11.59            prime    raced           4026       4389    1.09  5.7e-16  bluestein
       652  2^2.163          prime    raced           4026       4618    1.12  5.6e-16  bluestein
       653  653              prime    raced           4022       4563    1.13  7.2e-16  bluestein
       654  2.3.109          prime    raced           4030       4623    1.13  6.9e-16  bluestein
       655  5.131            prime    raced           4000       4609    1.14  7.0e-16  bluestein
       659  659              prime    raced           3994       4554    1.13  7.0e-16  bluestein
       662  2.331            prime    raced           4036       4666    1.14  7.9e-16  bluestein
       668  2^2.167          prime    raced           4035       4636    1.12  5.6e-16  bluestein
       669  3.223            prime    raced           4046       4636    1.09  7.9e-16  bluestein
       670  2.5.67           prime    raced           4021       4668    1.16  6.8e-16  bluestein
       671  11.61            prime    raced           4053       4640    1.14  6.4e-16  bluestein
       674  2.337            prime    raced           4808       4572    0.95  5.8e-16  bluestein
       678  2.3.113          prime    raced           4877       4635    0.93  4.9e-16  bluestein
       681  3.227            prime    raced           4794       4558    0.94  7.0e-16  bluestein
       683  683              prime    raced           4806       4639    0.53  6.0e-16  flips differ 1.87x bluestein
       685  5.137            prime    raced           4864       4574    0.94  6.9e-16  bluestein
       687  3.229            prime    raced           4794       4557    0.93  5.4e-16  bluestein
       689  13.53            prime    raced           4883       4325    0.88  5.0e-16  bluestein
       692  2^2.173          prime    raced           4804       4565    0.95  5.5e-16  bluestein
       694  2.347            prime    raced           4807       4660    0.95  6.5e-16  bluestein
       695  5.139            prime    raced           4809       4623    0.96  4.7e-16  bluestein
       698  2.349            prime    raced           4816       4563    0.94  7.6e-16  bluestein
       699  3.233            prime    raced           4807       4575    0.94  4.7e-16  bluestein
       706  2.353            prime    raced           4841       5434    0.94  5.3e-16  bluestein
       707  7.101            prime    raced           4847       4760    0.64  5.5e-16  flips differ 1.61x bluestein
       709  709              prime    raced           4833       4569    0.94  4.4e-16  bluestein
       716  2^2.179          prime    raced           4806       4566    0.95  7.0e-16  bluestein
       717  3.239            prime    raced           4851       4603    0.94  6.4e-16  bluestein
       718  2.359            prime    raced           4756       4616    0.96  5.1e-16  bluestein
       719  719              prime    raced           4829       4580    0.94  4.5e-16  bluestein
       721  7.103            prime    raced           4900       4562    0.93  5.7e-16  bluestein
       723  3.241            prime    raced           4798       4569    0.95  4.5e-16  bluestein
       724  2^2.181          prime    raced           4796       4553    0.95  5.8e-16  bluestein
       730  2.5.73           prime    raced           4800       5458    1.13  6.1e-16  bluestein
       733  733              prime    raced           4806       4607    0.95  6.2e-16  bluestein
       734  2.367            prime    raced           4795       4593    0.95  5.0e-16  bluestein
       739  739              prime    raced           4866       4581    0.93  6.1e-16  bluestein
       742  2.7.53           prime    raced           4759       4093    0.84  4.1e-16  bluestein
       743  743              prime    raced           4830       4594    0.94  5.1e-16  bluestein
       745  5.149            prime    raced           4857       4606    0.95  5.5e-16  bluestein
       746  2.373            prime    raced           4810       4943    0.95  5.3e-16  bluestein
       749  7.107            prime    raced           4830       4584    0.94  5.6e-16  bluestein
       751  751              prime    raced           3828       4578    1.17  8.2e-16  rader
       753  3.251            prime    raced           4781       4580    0.95  5.4e-16  bluestein
       755  5.151            prime    raced           4811       4590    0.95  6.0e-16  bluestein
       757  757              prime    raced           4155       4588    0.96  6.3e-16  rader
       758  2.379            prime    raced           4806       4654    0.95  5.0e-16  bluestein
       762  2.3.127          prime    raced           4793       4639    0.95  5.9e-16  bluestein
       763  7.109            prime    raced           4915       4599    0.93  4.9e-16  bluestein
       764  2^2.191          prime    raced           4872       4596    0.93  6.9e-16  bluestein
       766  2.383            prime    raced           4765       4596    0.96  5.3e-16  bluestein
       771  3.257            prime    raced           4826       4604    0.94  7.3e-16  bluestein
       772  2^2.193          prime    raced           4839       4586    0.86  5.7e-16  bluestein
       773  773              prime    raced           4824       4633    0.95  6.9e-16  bluestein
       778  2.389            prime    raced           4796       4597    0.96  5.5e-16  bluestein
       785  5.157            prime    raced           4835       4596    0.94  6.1e-16  bluestein
       786  2.3.131          prime    raced           4841       4616    0.95  6.3e-16  bluestein
       787  787              prime    raced           4855       4626    0.94  5.7e-16  bluestein
       788  2^2.197          prime    raced           4837       4632    0.95  7.3e-16  bluestein
       789  3.263            prime    raced           4842       4611    0.95  6.6e-16  bluestein
       791  7.113            prime    raced           4891       4615    0.94  6.1e-16  bluestein
       794  2.397            prime    raced           4794       4643    0.96  7.6e-16  bluestein
       795  3.5.53           prime    raced           4820       4415    0.70  5.7e-16  flips differ 1.30x bluestein
       796  2^2.199          prime    raced           4806       4648    0.94  7.8e-16  bluestein
       797  797              prime    raced           4832       4608    0.95  5.4e-16  bluestein
       802  2.401            prime    raced           4875       4595    0.93  6.1e-16  bluestein
       807  3.269            prime    raced           4840       4595    0.94  5.0e-16  bluestein
       808  2^3.101          prime    raced           4819       4603    0.95  5.3e-16  bluestein
       809  809              prime    raced           4913       4600    0.93  6.1e-16  bluestein
       811  811              prime    raced           4823       4623    0.94  5.1e-16  bluestein
       813  3.271            prime    raced           4865       4608    0.94  5.6e-16  bluestein
       815  5.163            prime    raced           4803       4588    0.74  6.5e-16  flips differ 1.29x bluestein
       818  2.409            prime    raced           4901       4960    0.88  5.4e-16  bluestein
       821  821              prime    raced           4851       4616    0.95  5.5e-16  bluestein
       822  2.3.137          prime    raced           4919       4607    0.93  6.7e-16  bluestein
       823  823              prime    raced           4870       4603    0.93  6.1e-16  bluestein
       824  2^3.103          prime    raced           4815       4652    0.95  5.2e-16  bluestein
       827  827              prime    raced           4854       4598    0.94  5.6e-16  bluestein
       831  3.277            prime    raced           4860       4628    0.95  5.6e-16  bluestein
       834  2.3.139          prime    raced           4847       4646    0.95  6.2e-16  bluestein
       835  5.167            prime    raced           4923       4990    0.88  5.7e-16  bluestein
       838  2.419            prime    raced           4840       4676    0.96  5.3e-16  bluestein
       839  839              prime    raced           4843       4658    0.94  7.2e-16  bluestein
       842  2.421            prime    raced           4865       4624    0.94  5.1e-16  bluestein
       843  3.281            prime    raced           4882       4637    0.95  4.8e-16  bluestein
       844  2^2.211          prime    raced           4836       4616    0.93  5.2e-16  bluestein
       848  2^4.53           prime    raced           4895       6247    1.26  4.7e-16  bluestein
       849  3.283            prime    raced           4915       4639    0.94  5.0e-16  bluestein
       853  853              prime    raced           4945       4986    0.94  4.9e-16  bluestein
       856  2^3.107          prime    raced           4877       4653    0.94  5.2e-16  bluestein
       857  857              prime    raced           4843       4640    0.95  6.4e-16  bluestein
       862  2.431            prime    raced           4855       4638    0.95  5.1e-16  bluestein
       863  863              prime    raced           4908       4641    0.94  5.8e-16  bluestein
       865  5.173            prime    raced           4904       4656    0.94  6.7e-16  bluestein
       866  2.433            prime    raced           4893       4670    0.95  5.7e-16  bluestein
       872  2^3.109          prime    raced           4823       4641    0.96  6.4e-16  bluestein
       877  877              prime    raced           4857       4661    0.94  4.3e-16  bluestein
       878  2.439            prime    raced           4823       4657    0.96  5.9e-16  bluestein
       879  3.293            prime    raced           4836       4644    0.95  4.2e-16  bluestein
       883  883              prime    raced           4566       4650    1.00  6.5e-16  rader
       886  2.443            prime    raced           4880       4677    0.95  5.9e-16  bluestein
       887  887              prime    raced           4849       4806    0.97  6.0e-16  bluestein
       889  7.127            prime    raced           4920       4756    0.93  6.4e-16  bluestein
       892  2^2.223          prime    raced           5006       4827    0.76  4.6e-16  flips differ 1.26x bluestein
       894  2.3.149          prime    raced           4845       4793    0.96  7.8e-16  bluestein
       895  5.179            prime    raced           4827       4683    0.95  5.4e-16  bluestein
       898  2.449            prime    raced           4851       4680    0.96  6.5e-16  bluestein
       904  2^3.113          prime    raced           4961       4680    0.93  5.8e-16  bluestein
       905  5.181            prime    raced           4905       4675    0.82  5.3e-16  bluestein
       906  2.3.151          prime    raced           4785       4675    0.97  5.0e-16  bluestein
       907  907              prime    raced           4930       5611    0.95  6.8e-16  bluestein
       908  2^2.227          prime    raced           4900       4701    0.95  5.2e-16  bluestein
       909  3^2.101          prime    raced           4838       4711    0.95  5.3e-16  bluestein
       911  911              prime    raced           3874       4675    1.17  8.8e-16  rader
       914  2.457            prime    raced           4850       4711    0.96  8.1e-16  bluestein
       916  2^2.229          prime    raced           4846       4708    0.97  6.1e-16  bluestein
       917  7.131            prime    raced           4862       4711    0.96  7.2e-16  bluestein
       919  919              prime    raced           4117       5562    1.14  1.2e-15  rader
       921  3.307            prime    raced           4863       4699    0.96  5.9e-16  bluestein
       922  2.461            prime    raced           4929       4714    0.95  5.5e-16  bluestein
       926  2.463            prime    raced           4896       4715    0.95  6.1e-16  bluestein
       927  3^2.103          prime    raced           4888       4697    0.96  4.7e-16  bluestein
       929  929              prime    raced           4167       4923    1.12  5.3e-16  rader
       932  2^2.233          prime    raced           4858       4715    0.96  7.0e-16  bluestein
       933  3.311            prime    raced           4881       4711    0.95  5.2e-16  bluestein
       934  2.467            prime    raced           4865       4711    0.95  6.0e-16  bluestein
       939  3.313            prime    raced           4892       4772    0.96  4.6e-16  bluestein
       941  941              prime    raced           4902       4769    0.96  5.9e-16  bluestein
       942  2.3.157          prime    raced           4904       4727    0.96  4.5e-16  bluestein
       947  947              prime    raced           4901       4720    0.96  6.1e-16  bluestein
       951  3.317            prime    raced           4888       4736    0.96  8.4e-16  bluestein
       955  5.191            prime    raced           4980       4700    0.74  6.3e-16  flips differ 1.28x bluestein
       956  2^2.239          prime    raced           4879       4720    0.96  5.7e-16  bluestein
       958  2.479            prime    raced           4937       4711    0.95  5.1e-16  bluestein
       959  7.137            prime    raced           4876       4743    0.96  5.7e-16  bluestein
       963  3^2.107          prime    raced           4917       4793    0.96  6.3e-16  bluestein
       964  2^2.241          prime    raced           4934       4704    0.95  5.7e-16  bluestein
       965  5.193            prime    raced           4866       4727    0.97  5.7e-16  bluestein
       967  967              prime    raced           4902       4742    0.96  5.2e-16  bluestein
       971  971              prime    raced           4898       4723    0.96  6.0e-16  bluestein
       973  7.139            prime    raced           4890       4753    0.96  4.7e-16  bluestein
       974  2.487            prime    raced           4939       4739    0.96  6.5e-16  bluestein
       977  977              prime    raced           5575       4744    0.74  5.4e-16  bluestein
       978  2.3.163          prime    raced           4871       4730    0.96  6.2e-16  bluestein
       981  3^2.109          prime    raced           4902       4755    0.97  6.1e-16  bluestein
       982  2.491            prime    raced           4897       4765    0.97  7.0e-16  bluestein
       983  983              prime    raced           4883       4781    0.97  6.1e-16  bluestein
       985  5.197            prime    raced           4877       4761    0.98  5.9e-16  bluestein
       991  991              prime    raced           4077       4760    1.15  6.4e-16  rader
       993  3.331            prime    raced           4877       4763    0.93  6.3e-16  bluestein
       995  5.199            prime    raced           4882       4758    0.97  6.5e-16  bluestein
       997  997              prime    raced           4941       4756    0.96  5.3e-16  bluestein
       998  2.499            prime    raced           4875       4754    0.97  5.1e-16  bluestein
      1002  2.3.167          prime    raced           4893       4754    0.97  5.4e-16  bluestein
      1004  2^2.251          prime    raced           4822       4762    0.97  5.8e-16  bluestein
      1006  2.503            prime    raced           4857       4792    0.98  5.9e-16  bluestein
      1009  1009             prime    raced           4906       5702    0.82  5.7e-16  bluestein
      1010  2.5.101          prime    raced           4868       4776    0.97  5.0e-16  bluestein
      1011  3.337            prime    raced           4885       4760    0.96  6.8e-16  bluestein
      1013  1013             prime    raced           4871       4785    0.76  5.3e-16  flips differ 1.29x bluestein
      1016  2^3.127          prime    raced           4877       4797    0.75  5.0e-16  flips differ 1.31x bluestein
      1017  3^2.113          prime    raced           4881       5850    1.05  4.8e-16  bluestein
      1018  2.509            prime    raced           4951       4769    0.96  6.0e-16  bluestein
      1019  1019             prime    raced           4879       4749    0.97  6.7e-16  bluestein
      1021  1021             prime    raced           4914       4790    0.97  5.7e-16  bluestein
      1027  13.79            prime    raced           6455       9147    1.36  6.8e-16  bluestein
      1037  17.61            prime    raced           5895       7816    1.32  5.3e-16  bluestein
      1039  1039             prime    raced           5838      10653    1.82  6.2e-16  bluestein
      1049  1049             prime    raced           5964      10625    1.77  7.3e-16  bluestein
      1060  2^2.5.53         prime    raced           5837       7663    1.31  6.9e-16  bluestein
      1062  2.3^2.59         prime    raced           5791       8285    1.41  5.3e-16  bluestein
      1065  3.5.71           prime    raced           5842       8141    1.38  6.7e-16  bluestein
      1069  1069             prime    raced           5872      10577    1.80  6.8e-16  bluestein
      1072  2^4.67           prime    raced           5839       9482    1.62  3.9e-16  bluestein
      1079  13.83            prime    raced           5913       9869    1.54  5.6e-16  bluestein
      1085  5.7.31           chain3   raced           2235       4134    1.84  5.1e-16  
      1095  3.5.73           prime    raced           5836       8386    1.31  6.4e-16  bluestein
      1096  2^3.137          prime    raced           5898      10675    1.79  5.1e-16  bluestein
      1097  1097             prime    raced           5804      10964    1.78  6.4e-16  bluestein
      1098  2.3^2.61         prime    raced           5841       8768    1.49  7.6e-16  bluestein
      1106  2.7.79           prime    raced           5836       9330    1.45  5.7e-16  bluestein
      1113  3.7.53           prime    raced           5835       6382    1.09  4.5e-16  bluestein
      1121  19.59            prime    raced           6027       8756    1.45  5.4e-16  bluestein
      1124  2^2.281          prime    raced           6003      10561    1.75  6.6e-16  bluestein
      1139  17.67            prime    raced           6066       9376    1.54  5.7e-16  bluestein
      1157  13.89            prime    raced           6731      11162    1.64  5.2e-16  bluestein
      1159  19.61            prime    raced           6625       9169    1.34  5.4e-16  bluestein
      1162  2.7.83           prime    raced           6785      10083    1.40  7.1e-16  bluestein
      1166  2.11.53          prime    raced           6779       7773    1.14  5.9e-16  bluestein
      1180  2^2.5.59         prime    raced           6742       9369    1.25  7.3e-16  bluestein
      1185  3.5.79           prime    raced           6793       9896    1.45  7.4e-16  bluestein
      1211  7.173            prime    raced          10334      10573    0.85  6.4e-16  bluestein
      1219  23.53            prime    raced          10987       9244    0.64  6.0e-16  flips differ 1.30x bluestein
      1220  2^2.5.61         prime    raced          10113       9839    0.95  4.1e-16  bluestein
      1239  3.7.59           prime    raced           6434       7938    1.22  7.9e-16  bluestein
      1270  2.5.127          prime    raced           6377      10639    1.66  5.5e-16  bluestein
      1272  2^3.3.53         prime    raced           6382       9376    1.47  6.7e-16  bluestein
      1281  3.7.61           prime    raced           7252       8421    1.14  5.1e-16  bluestein
      1286  2.643            prime    raced           7405      10596    1.41  6.6e-16  bluestein
      1293  3.431            prime    raced           7217      11338    1.47  6.8e-16  bluestein
      1298  2.11.59          prime    raced           7215       9560    1.32  5.6e-16  bluestein
      1325  5^2.53           prime    raced           7273       7703    0.99  6.2e-16  bluestein
      1339  13.103           prime    raced           7298      11178    1.46  6.5e-16  bluestein
      1342  2.11.61          prime    raced           7353      10069    1.36  5.3e-16  bluestein
      1348  2^2.337          prime    raced           7851      10687    1.36  7.0e-16  bluestein
      1378  2.13.53          prime    raced           7746       9481    1.21  6.4e-16  bluestein
      1390  2.5.139          prime    raced           7800      10695    1.36  7.3e-16  bluestein
      1393  7.199            prime    raced           7766      10732    1.36  6.3e-16  bluestein
      1407  3.7.67           prime    raced           8524      10242    1.20  7.4e-16  bluestein
      1418  2.709            prime    raced           8488      10764    1.01  9.4e-16  bluestein
      1423  1423             prime    raced           8447      10663    1.25  7.5e-16  bluestein
      1427  1427             prime    raced           8474      10652    1.25  6.1e-16  bluestein
      1431  3^3.53           prime    raced           8466       9982    1.17  6.4e-16  bluestein
      1434  2.3.239          prime    raced           8485      10664    1.25  8.1e-16  bluestein
      1437  3.479            prime    raced           8435      10680    1.26  7.3e-16  bluestein
      1475  5^2.59           prime    raced           7882       9485    1.07  9.0e-16  bluestein
      1481  1481             prime    raced           8258      10675    1.26  6.1e-16  bluestein
      1490  2.5.149          prime    raced           7926      10699    1.25  6.5e-16  bluestein
      1493  1493             prime    raced           7886      10662    1.34  7.2e-16  bluestein
      1507  11.137           prime    raced           7867      10645    1.35  6.6e-16  bluestein
      1516  2^2.379          prime    raced           7842      10696    1.27  7.9e-16  bluestein
      1525  5^2.61           prime    raced           7852       9996    1.27  5.2e-16  bluestein
      1563  3.521            prime    raced           8636      10701    1.24  6.6e-16  bluestein
      1569  3.523            prime    raced           8681      10697    1.23  6.3e-16  bluestein
      1603  7.229            prime    raced          10203      10736    0.89  7.7e-16  bluestein
      1622  2.811            prime    raced          10303      10716    1.03  6.6e-16  bluestein
      1627  1627             prime    raced          10305      10766    1.02  5.5e-16  bluestein
      1630  2.5.163          prime    raced          10483      10752    1.00  5.3e-16  bluestein
      1651  13.127           prime    raced          10305      10818    1.04  5.6e-16  bluestein
      1668  2^2.3.139        prime    raced          10432      10751    0.99  7.6e-16  bluestein
      1714  2.857            prime    raced          10043      10826    1.05  7.6e-16  bluestein
      1721  1721             prime    raced           9991      10848    1.08  5.9e-16  bluestein
      1737  3^2.193          prime    raced           9781      10789    1.10  5.1e-16  bluestein
      1741  1741             prime    raced           9755      10793    1.10  4.6e-16  bluestein
      1749  3.11.53          prime    raced           9989      11742    1.17  4.6e-16  bluestein
      1762  2.881            prime    raced           9923      10829    1.09  6.0e-16  bluestein
      1787  1787             prime    raced           9930      10842    0.99  7.8e-16  bluestein
      1811  1811             prime    raced          10540      10822    1.02  5.7e-16  bluestein
      1893  3.631            prime    raced          10484      10858    0.86  5.7e-16  bluestein
      1913  1913             prime    raced          10567      10845    1.02  7.1e-16  bluestein
      1929  3.643            prime    raced          10541      10819    0.72  6.0e-16  flips differ 1.42x bluestein
      1987  1987             prime    raced          10588      11068    1.02  7.3e-16  bluestein
      2054  2.13.79          prime    raced          13588      19539    1.42  7.7e-16  bluestein
      2059  29.71            prime    raced          13627      20269    1.47  7.3e-16  bluestein
      2062  2.1031           prime    raced          13386      23930    1.76  7.1e-16  bluestein
      2065  5.7.59           prime    raced          13393      13403    0.99  6.0e-16  bluestein
      2067  3.13.53          prime    raced          13646      14257    1.04  5.7e-16  bluestein
      2074  2.17.61          prime    raced          13386      17076    1.27  6.2e-16  bluestein
      2075  5^2.83           prime    raced          13659      18125    1.31  6.7e-16  bluestein
      2077  31.67            prime    raced          13382      20177    1.48  7.9e-16  bluestein
      2083  2083             prime    raced          13815      25222    1.73  6.1e-16  bluestein
      2117  29.73            prime    raced          13397      20831    1.53  7.2e-16  bluestein
      2120  2^3.5.53         prime    raced          13638      15801    1.16  7.0e-16  bluestein
      2124  2^2.3^2.59       prime    raced          13476      18582    1.13  7.3e-16  flips differ 1.30x bluestein
      2130  2.3.5.71         prime    raced          13643      19556    1.41  7.5e-16  bluestein
      2133  3^3.79           prime    raced          13395      20568    1.53  7.6e-16  bluestein
      2135  5.7.61           prime    raced          13434      14291    0.99  6.6e-16  bluestein
      2140  2^2.5.107        prime    raced          13622      23898    1.75  7.0e-16  bluestein
      2144  2^5.67           prime    raced          13498      19236    1.41  6.8e-16  bluestein
      2155  5.431            prime    raced          13394      23965    1.76  7.3e-16  bluestein
      2158  2.13.83          prime    raced          13621      21084    1.54  7.6e-16  bluestein
      2173  41.53            prime    raced          12854      19598    1.52  6.4e-16  bluestein
      2183  37.59            prime    raced          12906      20529    1.58  4.6e-16  bluestein
      2190  2.3.5.73         prime    raced          12888      20108    1.55  4.8e-16  bluestein
      2192  2^4.137          prime    raced          12724      23905    1.88  6.6e-16  bluestein
      2196  2^2.3^2.61       prime    raced          12734      18287    1.42  6.0e-16  bluestein
      2211  3.11.67          prime    raced          12718      18145    1.33  6.1e-16  bluestein
      2217  3.739            prime    raced          12862      24053    1.86  6.3e-16  bluestein
      2225  5^2.89           prime    raced          12818      20302    1.35  4.6e-16  bluestein
      2226  2.3.7.53         prime    raced          12648      15795    1.24  5.9e-16  bluestein
      2241  3^3.83           prime    raced          12936      22362    1.72  5.4e-16  bluestein
      2242  2.19.59          prime    raced          12996      18637    1.43  5.4e-16  bluestein
      2246  2.1123           prime    raced          12891      27379    1.86  6.0e-16  bluestein
      2257  37.61            prime    raced          12949      21475    1.66  5.1e-16  bluestein
      2264  2^3.283          prime    raced          12950      23967    1.85  6.7e-16  bluestein
      2267  2267             prime    raced          12878      23995    1.85  7.0e-16  bluestein
      2272  2^5.71           prime    raced          12935      21323    1.64  5.9e-16  bluestein
      2278  2.17.67          prime    raced          12867      20214    1.56  5.7e-16  bluestein
      2279  43.53            prime    raced          12907      21285    1.49  6.8e-16  bluestein
      2297  2297             prime    raced          12970      24081    1.80  6.6e-16  bluestein
      2298  2.3.383          prime    raced          12883      24088    1.86  5.7e-16  bluestein
      2301  3.13.59          prime    raced          12906      17290    1.27  5.4e-16  bluestein
      2313  3^2.257          prime    raced          14295      24595    1.68  8.3e-16  bluestein
      2318  2.19.61          prime    raced          14304      19875    1.34  5.8e-16  bluestein
      2323  23.101           prime    raced          14274      24643    1.67  6.9e-16  bluestein
      2332  2^2.11.53        prime    raced          14553      18061    1.23  6.8e-16  bluestein
      2335  5.467            prime    raced          14654      24768    1.67  7.7e-16  bluestein
      2336  2^5.73           prime    raced          14497      22093    1.35  7.5e-16  bluestein
      2343  3.11.71          prime    raced          14660      20368    1.38  7.0e-16  bluestein
      2345  5.7.67           prime    raced          14807      17409    1.06  5.8e-16  bluestein
      2360  2^3.5.59         prime    raced          14738      19154    1.13  7.3e-16  bluestein
      2379  3.13.61          prime    raced          14753      18267    1.23  6.3e-16  bluestein
      2382  2.3.397          prime    raced          14499      24151    1.65  7.3e-16  bluestein
      2385  3^2.5.53         prime    raced          14577      16873    1.15  7.3e-16  bluestein
      2396  2^2.599          prime    raced          14642      24144    1.64  7.1e-16  bluestein
      2409  3.11.73          prime    raced          21984      20807    0.90  4.4e-16  bluestein
      2412  2^2.3^2.67       prime    raced          22206      21806    0.97  5.3e-16  bluestein
      2421  3^2.269          prime    raced          21766      24071    1.04  5.6e-16  bluestein
      2438  2.23.53          prime    raced          16424      20051    1.19  7.7e-16  bluestein
      2440  2^3.5.61         prime    raced          16487      20117    1.21  6.6e-16  bluestein
      2452  2^2.613          prime    raced          13596      24075    1.77  5.5e-16  bluestein
      2455  5.491            prime    raced          14086      24231    1.56  6.3e-16  bluestein
      2474  2.1237           prime    raced          13636      24197    1.48  6.9e-16  bluestein
      2478  2.3.7.59         prime    raced          14418      18977    1.31  5.5e-16  bluestein
      2485  5.7.71           prime    raced          13578      19313    1.25  5.7e-16  bluestein
      2495  5.499            prime    raced          13641      24365    1.43  5.8e-16  flips differ 1.26x bluestein
      2497  11.227           prime    raced          13614      24027    1.76  4.8e-16  bluestein
      2544  2^4.3.53         prime    raced          13696      19210    1.40  7.9e-16  bluestein
      2555  5.7.73           prime    raced          13730      20105    1.23  5.0e-16  bluestein
      2557  2557             prime    raced          13719      24359    1.77  6.7e-16  bluestein
      2561  13.197           prime    raced          15803      24336    1.49  6.7e-16  bluestein
      2562  2.3.7.61         prime    raced          15652      20260    1.27  7.3e-16  bluestein
      2596  2^2.11.59        prime    raced          15803      21892    1.21  5.2e-16  bluestein
      2597  7^2.53           prime    raced          15900      15429    0.81  6.1e-16  bluestein
      2613  3.13.67          prime    raced          16135      22045    1.34  5.4e-16  bluestein
      2643  3.881            prime    raced          15647      26224    1.56  7.1e-16  bluestein
      2650  2.5^2.53         prime    raced          15830      18532    1.14  5.8e-16  bluestein
      2655  3^2.5.59         prime    raced          15657      20442    1.23  7.4e-16  bluestein
      2676  2^2.3.223        prime    raced          15774      24160    1.50  7.1e-16  bluestein
      2684  2^2.11.61        prime    raced          15779      23005    1.31  4.8e-16  bluestein
      2685  3.5.179          prime    raced          16651      25901    1.43  7.2e-16  bluestein
      2703  3.17.53          prime    raced          16210      20068    1.21  6.3e-16  bluestein
      2719  2719             prime    raced          16599      24181    1.32  8.3e-16  bluestein
      2722  2.1361           prime    raced          16422      24174    1.46  7.3e-16  bluestein
      2725  5^2.109          prime    raced          16220      25517    1.40  7.0e-16  bluestein
      2734  2.1367           prime    raced          16238      24153    1.48  9.2e-16  bluestein
      2741  2741             prime    raced          16645      24162    1.41  9.0e-16  bluestein
      2743  13.211           prime    raced          16277      24247    1.48  8.6e-16  bluestein
      2745  3^2.5.61         prime    raced          16281      21633    1.33  6.8e-16  bluestein
      2756  2^2.13.53        prime    raced          16446      21612    1.22  5.8e-16  bluestein
      2763  3^2.307          prime    raced          16239      24178    1.48  6.8e-16  bluestein
      2767  2767             prime    raced          16216      24125    1.46  7.1e-16  bluestein
      2776  2^3.347          prime    raced          18025      24130    1.34  6.5e-16  bluestein
      2782  2.13.107         prime    raced          16566      24653    1.30  7.2e-16  bluestein
      2799  3^2.311          prime    raced          16440      24439    1.47  7.2e-16  bluestein
      2825  5^2.113          prime    raced          17981      24602    1.33  7.8e-16  bluestein
      2827  11.257           prime    raced          17924      24169    1.33  7.4e-16  bluestein
      2840  2^3.5.71         prime    raced          18062      26711    1.41  6.6e-16  bluestein
      2855  5.571            prime    raced          17982      24444    1.36  7.7e-16  bluestein
      2861  2861             prime    raced          17890      24323    1.35  7.3e-16  bluestein
      2862  2.3^3.53         prime    raced          17902      21449    1.18  7.1e-16  bluestein
      2863  7.409            prime    raced          18102      25635    1.34  8.7e-16  bluestein
      2874  2.3.479          prime    raced          17949      27813    1.33  7.8e-16  bluestein
      2876  2^2.719          prime    raced          18025      24320    1.35  8.1e-16  bluestein
      2891  7^2.59           prime    raced          21943      18804    0.83  6.7e-16  bluestein
      2903  2903             prime    raced          18712      24166    1.27  6.3e-16  bluestein
      2915  5.11.53          prime    raced          18838      19796    1.04  6.8e-16  bluestein
      2917  2917             prime    raced          18709      24380    1.28  8.1e-16  bluestein
      2921  23.127           prime    raced          18811      24283    1.28  7.7e-16  bluestein
      2948  2^2.11.67        prime    raced          16645      27314    1.56  5.8e-16  bluestein
      2950  2.5^2.59         prime    raced          16851      22716    1.35  8.9e-16  bluestein
      2962  2.1481           prime    raced          16882      24193    1.43  6.0e-16  bluestein
      2965  5.593            prime    raced          16921      24162    1.34  6.9e-16  bluestein
      2968  2^3.7.53         prime    raced          16684      22358    1.33  5.6e-16  bluestein
      2974  2.1487           prime    raced          17401      24259    1.34  6.5e-16  bluestein
      2981  11.271           prime    raced          17085      24186    1.35  8.0e-16  bluestein
      2989  7^2.61           prime    raced          16654      20115    1.00  6.8e-16  bluestein
      3006  2.3^2.167        prime    raced          17635      24253    1.32  7.2e-16  bluestein
      3017  7.431            prime    raced          16910      24254    1.43  6.5e-16  bluestein
      3019  3019             prime    raced          16878      24239    1.43  7.1e-16  bluestein
      3047  11.277           prime    raced          16911      24193    1.43  6.7e-16  bluestein
      3056  2^4.191          prime    raced          17052      24346    1.42  6.9e-16  bluestein
      3074  2.29.53          prime    raced          19050      26296    1.38  4.8e-16  bluestein
      3079  3079             prime    raced          18872      24287    1.28  7.2e-16  bluestein
      3086  2.1543           prime    raced          18866      24315    1.27  5.6e-16  bluestein
      3091  11.281           prime    raced          18788      24239    1.23  7.2e-16  bluestein
      3095  5.619            prime    raced          18683      24303    1.26  6.8e-16  bluestein
      3109  3109             prime    raced          19143      24301    1.22  7.3e-16  bluestein
      3110  2.5.311          prime    raced          18934      26197    1.28  7.7e-16  bluestein
      3111  3.17.61          prime    raced          19213      25466    1.30  6.0e-16  bluestein
      3129  3.7.149          prime    raced          17984      24252    1.35  7.4e-16  bluestein
      3151  23.137           prime    raced          18158      24254    1.26  7.1e-16  bluestein
      3203  3203             prime    raced          21401      24339    1.11  9.5e-16  bluestein
      3210  2.3.5.107        prime    raced          21611      24280    0.93  5.4e-16  bluestein
      3223  11.293           prime    raced          21850      25342    1.11  8.1e-16  bluestein
      3238  2.1619           prime    raced          21373      24338    0.97  9.5e-16  bluestein
      3241  7.463            prime    raced          22134      24928    0.96  7.6e-16  bluestein
      3244  2^2.811          prime    raced          21845      24275    1.11  8.5e-16  bluestein
      3254  2.1627           prime    raced          21918      24215    0.90  8.2e-16  bluestein
      3259  3259             prime    raced          22045      24303    1.10  7.1e-16  bluestein
      3269  7.467            prime    raced          21457      24299    1.10  6.3e-16  bluestein
      3273  3.1091           prime    raced          21366      24287    1.11  7.5e-16  bluestein
      3283  7^2.67           prime    raced          21420      24004    1.11  5.9e-16  bluestein
      3291  3.1097           prime    raced          22028      24441    1.08  8.0e-16  bluestein
      3299  3299             prime    raced          21576      24326    1.11  9.3e-16  bluestein
      3301  3301             prime    raced          18958      24441    1.18  9.5e-16  rader
      3302  2.13.127         prime    raced          22037      24461    1.10  6.9e-16  bluestein
      3305  5.661            prime    raced          21315      24377    1.11  8.7e-16  bluestein
      3308  2^2.827          prime    raced          21984      25761    1.10  6.4e-16  bluestein
      3314  2.1657           prime    raced          21826      24354    1.11  9.1e-16  bluestein
      3345  3.5.223          prime    raced          21297      25748    1.10  1.0e-15  bluestein
      3350  2.5^2.67         prime    raced          21580      28550    1.25  5.0e-16  bluestein
      3353  7.479            prime    raced          21526      24330    1.09  6.1e-16  bluestein
      3355  5.11.61          prime    raced          21490      25595    1.17  7.9e-16  bluestein
      3356  2^2.839          prime    raced          21476      24395    1.13  6.4e-16  bluestein
      3370  2.5.337          prime    raced          21762      24413    1.12  6.0e-16  bluestein
      3389  3389             prime    raced          20710      24392    1.11  7.2e-16  bluestein
      3421  11.311           prime    raced          20838      24333    1.16  7.0e-16  bluestein
      3428  2^2.857          prime    raced          20841      24334    1.15  6.4e-16  bluestein
      3477  3.19.61          prime    raced          22962      29518    1.23  5.2e-16  bluestein
      3487  11.317           prime    raced          20355      24364    1.19  8.3e-16  bluestein
      3495  3.5.233          prime    raced          21607      24358    1.10  7.4e-16  bluestein
      3499  3499             prime    raced          20236      24437    1.19  6.0e-16  bluestein
      3573  3^2.397          prime    raced          23133      24347    1.05  6.9e-16  bluestein
      3578  2.1789           prime    raced          20285      25814    1.11  8.3e-16  bluestein
      3582  2.3^2.199        prime    raced          20342      24386    1.19  6.6e-16  bluestein
      3638  2.17.107         prime    raced          22703      24405    1.07  5.2e-16  bluestein
      3639  3.1213           prime    raced          22504      24431    1.06  6.3e-16  bluestein
      3653  13.281           prime    raced          22172      24466    1.10  6.4e-16  bluestein
      3665  5.733            prime    raced          22102      24464    1.10  5.7e-16  bluestein
      3677  3677             prime    raced          21725      24419    1.10  7.6e-16  bluestein
      3684  2^2.3.307        prime    raced          21662      24443    0.88  7.3e-16  flips differ 1.28x bluestein
      3732  2^2.3.311        prime    raced          22180      24381    1.07  8.7e-16  bluestein
      3754  2.1877           prime    raced          21785      24487    1.11  7.0e-16  bluestein
      3775  5^2.151          prime    raced          21790      25434    1.13  7.6e-16  bluestein
      3779  3779             prime    raced          22165      24536    1.07  8.2e-16  bluestein
      3817  11.347           prime    raced          21724      25494    1.10  7.2e-16  bluestein
      3824  2^4.239          prime    raced          22631      24815    0.75  6.1e-16  flips differ 1.47x bluestein
      3842  2.17.113         prime    raced          23043      24470    1.06  6.7e-16  bluestein
      3860  2^2.5.193        prime    raced          22806      24572    1.07  7.4e-16  bluestein
      3867  3.1289           prime    raced          22639      27383    1.08  6.6e-16  bluestein
      3879  3^2.431          prime    raced          22181      24542    1.10  6.5e-16  bluestein
      3881  3881             prime    raced          22140      25003    1.11  7.4e-16  bluestein
      3883  11.353           prime    raced          21635      24652    1.12  6.9e-16  bluestein
      3898  2.1949           prime    raced          28256      28928    0.83  6.3e-16  bluestein
      3914  2.19.103         prime    raced          22369      24498    1.09  5.9e-16  bluestein
      3921  3.1307           prime    raced          21968      24483    1.10  5.7e-16  bluestein
      3957  3.1319           prime    raced          22318      25403    1.09  6.7e-16  bluestein
      3972  2^2.3.331        prime    raced          22390      24504    1.07  6.3e-16  bluestein
      4007  4007             prime    raced          22935      25805    1.09  8.2e-16  bluestein
      4016  2^4.251          prime    raced          22772      24508    1.06  6.2e-16  bluestein
      4054  2.2027           prime    raced          22681      24591    0.87  6.9e-16  bluestein
      4055  5.811            prime    raced          22366      25672    1.10  7.3e-16  bluestein
      4076  2^2.1019         prime    raced          22984      24528    1.06  6.7e-16  bluestein
      4077  3^3.151          prime    raced          22380      24569    1.07  8.7e-16  bluestein
      4081  7.11.53          prime    raced          22423      27841    1.22  6.3e-16  bluestein
      4090  2.5.409          prime    raced          21775      24522    1.10  7.2e-16  bluestein
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime      581     19    238   0.93   1.10   1.48    1.12
 flat         1      1      1   0.64   0.64   0.64    0.64
 chain3       1      0      0   1.84   1.84   1.84    1.84
 ALL        583     20    239   0.93   1.10   1.48    1.12
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                2   0.79      2      1
 128..511              88   0.98     56      5
 512..2047            298   0.97    166     13
 2048..4090           195   1.27     15      1
```


## by family
```
 family                                       cells median   <1.0  gmean
 composite with a prime >= 53 (prime cell)      464   1.11    182   1.13
 prime N, bluestein                             106   1.00     53   1.09
 prime N, rader                                  11   1.12      3   1.05
 flat                                             1   0.64      1   0.64
 chain3                                           1   1.84      0   1.84
```


flip agreement: our two readings more than 25% apart at 18 of 583 cells.

worst 10: 683 (prime 0.53), 707 (prime 0.64), 98 (flat 0.64), 1219 (prime 0.64), 499 (prime 0.70), 795 (prime 0.70), 1929 (prime 0.72), 955 (prime 0.74), 977 (prime 0.74), 223 (prime 0.74)
best 5: 2192 (prime 1.88), 2217 (prime 1.86), 2246 (prime 1.86), 2298 (prime 1.86), 2267 (prime 1.85)
