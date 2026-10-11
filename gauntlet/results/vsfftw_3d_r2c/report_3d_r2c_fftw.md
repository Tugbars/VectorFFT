# gauntlet report (3D)

run: `vsfftw_3d_r2c`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 328 listed, 328 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 10 readings, 1.090..1.192


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       4x4x4  2^2          child  raced             56         47    0.84     17.2  0.0e+00
       4x4x8  2^2          child  raced             70         79    1.13     32.2  2.7e-16
      4x4x16  2^2          payonce raced            141        164    1.15     36.2  2.0e-16
      4x4x32  2^2          child  raced            236        364    1.50     48.7  2.6e-16
      4x4x64  2^2          child  raced            549        872    1.40     46.6  2.1e-16
     4x4x128  2^2          payonce raced           1271       1282    1.00     44.3  3.1e-16
     4x4x256  2^2          child  raced           2237       2551    1.08     54.9  3.0e-16
       4x8x4  2^2          payonce raced            121         89    0.73     18.4  1.5e-16
       4x8x8  2^2          child  raced            122        158    1.29     42.0  1.4e-16
      4x8x16  2^2          child  raced            238        352    1.25     48.4  2.5e-16
      4x8x32  2^2          child  raced            489        752    1.52     52.3  3.1e-16
      4x8x64  2^2          child  raced           1133       1728    1.52     49.7  2.6e-16
     4x8x128  2^2          child  raced           2315       2676    1.15     53.1  3.1e-16
     4x8x256  2^2          child  raced           5157       5482    1.02     51.6  3.1e-16
      4x16x4  2^2          child  raced            158        192    1.21     32.4  1.7e-16
      4x16x8  2^2          child  raced            253        338    1.33     45.5  2.5e-16
     4x16x16  2^2          child  raced            496        709    1.42     51.6  1.9e-16
     4x16x32  2^2          child  raced           1046       1574    1.47     53.9  2.5e-16
     4x16x64  2^2          child  raced           2488       3738    1.47     49.4  2.9e-16
    4x16x128  2^2          child  raced           5152       5780    1.12     51.7  2.9e-16
    4x16x256  2^2          child  raced          11051      11537    1.04     51.9  2.4e-16
      4x32x4  2^2          child  raced            336        475    1.30     34.3  2.7e-16
      4x32x8  2^2          child  raced            540        679    1.25     47.4  2.7e-16
     4x32x16  2^2          child  raced           1098       1443    1.31     51.3  2.4e-16
     4x32x32  2^2          child  raced           2635       3409    1.28     46.6  2.6e-16
     4x32x64  2^2          child  raced           5620       7889    1.39     47.4  3.2e-16
    4x32x128  2^2          child  raced          11168      12360    1.10     51.3  3.6e-16
    4x32x256  2^2          child  raced          24105      24388    1.01     51.0  3.8e-16
      4x64x4  2^2          child  raced            771       1002    1.25     33.2  2.9e-16
      4x64x8  2^2          child  raced           1247       1471    1.15     45.2  2.2e-16
     4x64x16  2^2          child  raced           2476       3314    1.32     49.6  3.1e-16
     4x64x32  2^2          child  raced           6464       7197    1.10     41.2  3.5e-16
     4x64x64  2^2          child  raced          12210      15218    1.23     47.0  3.6e-16
    4x64x128  2^2          child  raced          24336      25819    1.05     50.5  3.9e-16
    4x64x256  2^2          child  raced          50187      52471    1.03     52.2  4.1e-16
     4x128x4  2^2          child  raced           1795       1810    0.99     31.4  3.5e-16
     4x128x8  2^2          child  raced           3269       3675    1.11     37.6  2.6e-16
    4x128x16  2^2          child  raced           5447       7449    1.33     48.9  2.9e-16
    4x128x32  2^2          child  raced          14802      15389    1.03     38.7  4.4e-16
    4x128x64  2^2          child  raced          26256      34252    1.30     46.8  4.2e-16
   4x128x128  2^2          child  raced          52069      54929    1.05     50.3  3.3e-16
   4x128x256  2^2          child  raced         109800     114176    1.03     50.7  4.0e-16
     4x256x4  2^2          child  raced           4572       5252    1.09     26.9  3.3e-16
     4x256x8  2^2          child  raced           7836       9514    1.21     34.0  3.9e-16
    4x256x16  2^2          child  raced          14432      17770    1.23     39.7  4.1e-16
    4x256x32  2^2          child  raced          32316      36347    1.12     38.0  3.5e-16
    4x256x64  2^2          child  raced          63879      76639    1.20     41.0  3.5e-16
   4x256x128  2^2          child  raced         133833     135525    1.01     41.6  4.4e-16
   4x256x256  2^2          payonce raced         295313     324483    1.03     39.9  4.6e-16
       8x4x4  2^3          child  raced            125        100    0.81     18.0  1.4e-16
       8x4x8  2^3          child  raced            139        180    1.20     36.9  1.8e-16
      8x4x16  2^3          payonce raced            277        339    1.20     41.6  2.8e-16
      8x4x32  2^3          child  raced            505        747    1.47     50.7  3.4e-16
      8x4x64  2^3          child  raced           1162       1794    1.51     48.5  2.2e-16
     8x4x128  2^3          child  raced           2335       2722    1.15     52.6  2.9e-16
     8x4x256  2^3          child  raced           5275       5481    1.03     50.5  3.1e-16
       8x8x4  2^3          payonce raced            233        182    0.78     22.0  4.0e-16
       8x8x8  2^3          child  raced            254        357    1.30     45.4  2.7e-16
      8x8x16  2^3          payonce raced            558        712    1.27     45.9  2.4e-16
      8x8x32  2^3          child  raced           1030       1531    1.48     54.7  2.3e-16
      8x8x64  2^3          child  raced           2427       3630    1.48     50.6  3.6e-16
     8x8x128  2^3          child  raced           4902       5608    1.14     54.3  3.7e-16
     8x8x256  2^3          child  raced          10532      11221    1.02     54.4  3.2e-16
      8x16x4  2^3          child  raced            334        412    1.23     34.5  1.9e-16
      8x16x8  2^3          child  raced            531        706    1.33     48.2  2.6e-16
     8x16x16  2^3          child  raced           1231       1469    0.99     45.7  3.0e-16
     8x16x32  2^3          child  raced           2472       3297    1.32     49.7  2.7e-16
     8x16x64  2^3          child  raced           5262       7579    1.41     50.6  3.0e-16
    8x16x128  2^3          child  raced          10518      11926    1.13     54.5  3.6e-16
    8x16x256  2^3          child  raced          22258      23779    1.04     55.2  3.3e-16
      8x32x4  2^3          child  raced            701        918    1.30     36.5  2.5e-16
      8x32x8  2^3          child  raced           1136       1493    1.30     49.6  2.8e-16
     8x32x16  2^3          child  raced           2377       3206    1.34     51.7  3.2e-16
     8x32x32  2^3          child  raced           6119       7093    1.15     43.5  2.8e-16
     8x32x64  2^3          child  raced          11401      14862    1.29     50.3  3.2e-16
    8x32x128  2^3          child  raced          22477      25104    1.11     54.7  2.9e-16
    8x32x256  2^3          child  raced          47146      50104    1.06     55.6  3.8e-16
      8x64x4  2^3          child  raced           1631       2050    1.22     34.5  2.9e-16
      8x64x8  2^3          child  raced           2620       3519    1.33     46.9  3.6e-16
     8x64x16  2^3          child  raced           5203       7404    1.40     51.2  3.7e-16
     8x64x32  2^3          child  raced          13439      15004    1.11     42.7  3.5e-16
     8x64x64  2^3          child  raced          24548      33429    1.35     50.1  3.4e-16
    8x64x128  2^3          child  raced          49707      52854    1.06     52.7  4.0e-16
    8x64x256  2^3          child  raced         103210     107313    1.02     54.0  4.2e-16
     8x128x4  2^3          child  raced           3948       4808    1.19     31.1  2.8e-16
     8x128x8  2^3          child  raced           6605       8298    1.24     40.3  3.1e-16
    8x128x16  2^3          child  raced          11226      15512    1.34     51.1  3.1e-16
    8x128x32  2^3          child  raced          29225      31469    1.07     42.0  4.0e-16
    8x128x64  2^3          child  raced          52902      65441    1.23     49.6  3.8e-16
   8x128x128  2^3          child  raced         108677     112351    1.03     51.3  5.1e-16
   8x128x256  2^3          child  raced         236833     252660    1.06     49.8  4.6e-16
     8x256x4  2^3          child  raced          12894      11389    0.86     20.6  3.1e-16
     8x256x8  2^3          payonce raced          19424      19496    1.00     29.5  3.8e-16
    8x256x16  2^3          child  raced          29204      36043    1.22     42.1  4.2e-16
    8x256x32  2^3          child  raced          64720      74539    1.14     40.5  4.0e-16
    8x256x64  2^3          child  raced         128390     154353    1.20     43.4  3.4e-16
   8x256x128  2^3          child  raced         277647     288170    1.02     42.5  3.9e-16
   8x256x256  2^3          child  raced         593625     669188    1.11     42.0  4.9e-16
      16x4x4  2^4          child  raced            234        200    0.85     21.9  1.5e-16
      16x4x8  2^4          child  raced            299        348    1.15     38.6  2.6e-16
     16x4x16  2^4          child  raced            563        695    1.23     45.5  3.0e-16
     16x4x32  2^4          child  raced           1255       1590    1.25     44.9  2.5e-16
     16x4x64  2^4          child  raced           2539       3720    1.46     48.4  3.1e-16
    16x4x128  2^4          payonce raced           5928       6020    1.00     44.9  3.0e-16
    16x4x256  2^4          child  raced          10748      11625    1.07     53.4  3.3e-16
      16x8x4  2^4          payonce raced            486        382    0.78     23.7  1.9e-16
      16x8x8  2^4          child  raced            574        726    1.22     44.6  3.3e-16
     16x8x16  2^4          child  raced           1102       1491    1.34     51.1  3.4e-16
     16x8x32  2^4          child  raced           2303       3358    1.45     53.4  2.7e-16
     16x8x64  2^4          child  raced           5221       7685    1.46     51.0  3.0e-16
    16x8x128  2^4          child  raced          10447      11915    1.14     54.9  3.0e-16
    16x8x256  2^4          child  raced          23114      23900    1.02     53.2  3.4e-16
     16x16x4  2^4          child  raced            720        862    1.16     35.5  2.8e-16
     16x16x8  2^4          child  raced           1157       1550    1.33     48.7  2.8e-16
    16x16x16  2^4          child  raced           2300       3243    1.26     53.4  2.6e-16
    16x16x32  2^4          child  raced           6525       7010    1.06     40.8  2.6e-16
    16x16x64  2^4          payonce raced          14189      14979    0.81     40.4  3.3e-16
   16x16x128  2^4          child  raced          22183      25160    1.12     55.4  3.3e-16
   16x16x256  2^4          child  raced          46379      50276    1.08     56.5  3.7e-16
     16x32x4  2^4          child  raced           1906       1931    0.99     29.5  2.3e-16
     16x32x8  2^4          child  raced           2610       3567    1.33     47.1  2.6e-16
    16x32x16  2^4          child  raced           5166       7279    1.36     51.5  3.4e-16
    16x32x32  2^4          child  raced          13121      15290    1.14     43.7  3.6e-16
    16x32x64  2^4          child  raced          24101      33445    1.38     51.0  3.3e-16
   16x32x128  2^4          child  raced          47385      53319    1.12     55.3  3.8e-16
   16x32x256  2^4          child  raced         101027     108278    1.06     55.1  3.7e-16
     16x64x4  2^4          child  raced           3526       5337    1.48     34.8  3.2e-16
     16x64x8  2^4          child  raced           5811       8290    1.41     45.8  3.1e-16
    16x64x16  2^4          child  raced          11439      16060    1.40     50.1  3.6e-16
    16x64x32  2^4          child  raced          28417      31451    1.10     43.2  4.2e-16
    16x64x63  2^4          payonce raced          85937     127156    1.47     30.0  5.3e-16
    16x64x64  2^4          child  raced          51595      64964    1.24     50.8  3.8e-16
   16x64x128  2^4          child  raced         106757     112606    1.03     52.2  3.2e-16
   16x64x256  2^4          child  raced         228613     248150    1.07     51.6  4.6e-16
    16x128x4  2^4          child  raced           8425      10781    1.28     31.6  3.6e-16
    16x128x8  2^4          child  raced          13943      17703    1.24     41.1  2.9e-16
   16x128x16  2^4          child  raced          24130      32753    1.32     50.9  3.9e-16
   16x128x32  2^4          child  raced          61123      66083    1.08     42.9  4.1e-16
   16x128x64  2^4          child  raced         111617     138175    1.23     49.9  3.8e-16
  16x128x128  2^4          child  raced         237400     252550    1.05     49.7  4.6e-16
  16x128x256  2^4          child  raced         500187     534500    1.06     49.8  4.7e-16
    16x256x4  2^4          child  raced          24573      29931    1.17     23.3  3.4e-16
    16x256x8  2^4          child  raced          40599      50404    1.22     30.3  3.0e-16
   16x256x16  2^4          child  raced          74385      92958    1.24     35.2  3.8e-16
   16x256x32  2^4          child  raced         169327     188493    1.11     32.9  4.1e-16
   16x256x64  2^4          payonce raced         326293     398970    1.19     36.2  4.1e-16
  16x256x128  2^4          payonce raced         664238     737856    1.10     37.5  3.7e-16
  16x256x256  2^4          payonce raced        1415887    1666674    1.12     37.0  4.2e-16
     27x9x15  3^3          child  raced           5281       5028    0.95     20.4  5.2e-16
      32x4x4  2^5          child  raced            481        403    0.84     23.9  2.4e-16
      32x4x8  2^5          payonce raced            794        719    0.88     32.2  2.4e-16
     32x4x16  2^5          child  raced           1171       1465    1.21     48.1  2.7e-16
     32x4x32  2^5          child  raced           2393       3444    1.40     51.4  3.1e-16
     32x4x64  2^5          child  raced           5544       7887    1.41     48.0  3.0e-16
    32x4x128  2^5          child  raced          10690      12718    1.18     53.6  3.5e-16
    32x4x256  2^5          child  raced          23183      24845    1.02     53.0  3.4e-16
      32x8x4  2^5          payonce raced           1015        821    0.80     25.2  2.1e-16
      32x8x8  2^5          child  raced           1360       1494    1.10     41.4  2.7e-16
     32x8x16  2^5          child  raced           2436       3290    1.29     50.4  2.9e-16
     32x8x32  2^5          child  raced           5157       7204    1.39     51.6  2.6e-16
     32x8x64  2^5          payonce raced          13118      15252    1.15     43.7  4.3e-16
    32x8x128  2^5          child  raced          23085      25835    1.10     53.2  3.7e-16
    32x8x256  2^5          child  raced          48202      52019    1.03     54.4  3.9e-16
     32x16x4  2^5          child  raced           1549       1856    1.18     36.4  2.0e-16
     32x16x8  2^5          child  raced           2549       3801    1.46     48.2  2.5e-16
    32x16x16  2^5          child  raced           5078       7314    1.23     52.4  3.1e-16
    32x16x32  2^5          child  raced          12805      15419    1.19     44.8  3.6e-16
    32x16x64  2^5          child  raced          25398      34220    1.32     48.4  3.5e-16
   32x16x128  2^5          child  raced          49336      54788    1.11     53.1  3.5e-16
   32x16x256  2^5          child  raced         104200     112645    1.05     53.5  3.6e-16
     32x32x4  2^5          child  raced           4381       5242    1.18     28.0  2.7e-16
     32x32x8  2^5          child  raced           5712       8313    1.45     46.6  2.8e-16
    32x32x16  2^5          child  raced          11480      16061    1.26     50.0  3.0e-16
    32x32x27  2^5          child  raced          32365      45903    1.41     31.5  4.4e-16
    32x32x32  2^5          child  raced          28867      32780    1.13     42.6  3.9e-16
    32x32x64  2^5          child  raced          53418      67235    1.25     49.1  3.9e-16
   32x32x128  2^5          child  raced         108427     116923    1.07     51.4  4.2e-16
   32x32x256  2^5          child  raced         240493     261423    1.07     49.1  4.8e-16
     32x64x4  2^5          child  raced          10251      12044    1.12     26.0  4.1e-16
     32x64x8  2^5          child  raced          16373      17961    1.06     35.0  4.6e-16
    32x64x16  2^5          child  raced          30586      33954    1.03     40.2  3.0e-16
    32x64x32  2^5          child  raced          62087      67620    1.08     42.2  4.1e-16
    32x64x64  2^5          child  raced         115593     141247    1.21     48.2  4.3e-16
   32x64x128  2^5          child  raced         245967     258806    1.04     48.0  4.9e-16
   32x64x256  2^5          child  raced         503200     538937    1.06     49.5  4.4e-16
    32x128x4  2^5          child  raced          22733      28225    1.24     25.2  3.4e-16
    32x128x8  2^5          child  raced          36921      46458    1.19     33.3  3.9e-16
   32x128x16  2^5          child  raced          63210      84292    1.30     41.5  4.0e-16
   32x128x32  2^5          child  raced         162323     171418    1.05     34.3  4.9e-16
   32x128x64  2^5          payonce raced         313320     364033    1.16     37.6  3.6e-16
  32x128x128  2^5          child  raced         650712     661419    0.94     38.3  3.8e-16
  32x128x256  2^5          payonce raced        1424025    1378137    0.96     36.8  4.6e-16
    32x256x4  2^5          child+strips raced          53264      60852    1.12     23.1  3.6e-16
    32x256x8  2^5          child+strips raced          89302     103390    1.15     29.4  3.6e-16
   32x256x16  2^5          child+strips raced         166090     193428    1.16     33.5  3.8e-16
   32x256x32  2^5          child+strips raced         366120     404323    1.10     32.2  3.6e-16
   32x256x64  2^5          payonce+strips raced         677562     823556    1.19     36.8  3.8e-16
  32x256x128  2^5          payonce+strips raced        1542325    1517806    0.97     34.0  3.9e-16
    36x20x28  2^2.3^2      child  raced          27935      30032    1.07     25.8  4.1e-16
      64x4x4  2^6          child  raced            971        843    0.84     26.4  3.1e-16
      64x4x8  2^6          child  raced           1394       1545    1.07     40.4  3.5e-16
     64x4x16  2^6          child  raced           2597       3295    1.19     47.3  3.5e-16
     64x4x32  2^6          child  raced           5274       7658    1.41     50.5  3.0e-16
     64x4x64  2^6          child  raced          13382      15860    1.18     42.9  2.6e-16
    64x4x128  2^6          child  raced          25907      27389    1.04     47.4  3.8e-16
    64x4x256  2^6          child  raced          51216      53943    1.05     51.2  3.4e-16
      64x8x4  2^6          payonce raced           2387       1801    0.72     23.6  2.6e-16
      64x8x8  2^6          child  raced           3236       3643    1.10     38.0  2.5e-16
     64x8x16  2^6          child  raced           5428       7492    1.36     49.1  3.3e-16
     64x8x32  2^6          child  raced          12583      15757    1.25     45.6  3.8e-16
     64x8x64  2^6          child  raced          25480      35057    1.37     48.2  3.8e-16
    64x8x128  2^6          child  raced          53862      55541    1.03     48.7  4.0e-16
    64x8x256  2^6          child  raced         127167     112496    0.88     43.8  3.6e-16
     64x16x4  2^6          child  raced           3412       4946    1.43     36.0  2.3e-16
     64x16x8  2^6          child  raced           5836       8588    1.46     45.6  3.4e-16
    64x16x16  2^6          child  raced          12818      15843    0.98     44.7  3.6e-16
    64x16x32  2^6          child  raced          30140      32848    0.97     40.8  3.5e-16
    64x16x64  2^6          child  raced          59220      68782    1.16     44.3  3.2e-16
   64x16x128  2^6          child  raced         126890     116768    0.91     43.9  3.2e-16
   64x16x256  2^6          child  raced         280367     259910    0.91     42.1  4.2e-16
     64x32x4  2^6          child  raced           9183      11742    1.26     29.0  3.1e-16
     64x32x8  2^6          child  raced          12719      18578    1.44     45.1  4.2e-16
    64x32x16  2^6          child  raced          25470      33829    1.31     48.2  3.8e-16
    64x32x32  2^6          payonce raced          70310      68919    0.98     37.3  3.2e-16
    64x32x64  2^6          child  raced         134793     142200    1.05     41.3  3.8e-16
   64x32x128  2^6          child  raced         295160     262706    0.83     40.0  4.7e-16
   64x32x256  2^6          child  raced         614563     553706    0.88     40.5  3.9e-16
     64x64x4  2^6          child+strips raced          23243      29827    1.27     24.7  3.3e-16
     64x64x8  2^6          child  raced          43830      45236    1.01     28.0  3.5e-16
    64x64x15  2^6          child  raced          93586      76749    0.81     26.1  4.1e-16
    64x64x16  2^6          child  raced          87082      84813    0.82     30.1  3.9e-16
    64x64x32  2^6          child  raced         156740     166857    1.04     35.5  3.3e-16
    64x64x64  2^6          child  raced         303120     355716    1.17     38.9  4.0e-16
   64x64x128  2^6          payonce raced         626737     648287    1.00     39.7  4.8e-16
   64x64x256  2^6          payonce raced        1319462    1315006    0.92     39.7  4.4e-16
    64x128x4  2^6          child  raced          57553      58936    0.98     21.4  3.1e-16
    64x128x8  2^6          child  raced          99444      95347    0.96     26.4  4.0e-16
   64x128x16  2^6          child  raced         143503     179311    1.22     38.8  4.3e-16
   64x128x32  2^6          payonce raced         354327     369836    1.03     33.3  3.4e-16
   64x128x64  2^6          child  raced         647788     763056    1.08     38.4  5.0e-16
  64x128x128  2^6          child+strips raced        1376875    1371500    0.99     38.1  5.0e-16
    64x256x4  2^6          child  raced         108611     125142    1.12     24.1  3.2e-16
    64x256x8  2^6          child+strips raced         189953     218506    1.14     29.3  3.2e-16
   64x256x16  2^6          child+strips raced         361653     426120    1.16     32.6  3.8e-16
   64x256x32  2^6          payonce+strips raced         832263     848137    1.02     29.9  4.0e-16
   64x256x64  2^6          payonce+strips raced        1466025    1706113    1.11     35.8  4.7e-16
     128x4x4  2^7          child  raced           2338       1816    0.76     24.1  2.1e-16
     128x4x8  2^7          child  raced           3553       3871    1.08     34.6  3.1e-16
    128x4x16  2^7          child  raced           6427       7490    1.16     41.4  3.5e-16
    128x4x32  2^7          child  raced          12990      15750    1.20     44.1  2.5e-16
    128x4x64  2^7          child  raced          28660      35271    1.22     42.9  3.4e-16
   128x4x128  2^7          payonce raced          55941      57568    1.02     46.9  3.6e-16
   128x4x256  2^7          child  raced         129357     114396    0.88     43.1  3.4e-16
     128x8x4  2^7          payonce raced           5509       4881    0.87     22.3  2.8e-16
     128x8x8  2^7          child  raced           6958       8400    1.20     38.3  3.4e-16
    128x8x16  2^7          child  raced          12992      16134    1.23     44.1  3.4e-16
    128x8x32  2^7          child  raced          26489      32577    1.19     46.4  3.4e-16
    128x8x64  2^7          child  raced          58084      68340    1.17     45.1  3.8e-16
   128x8x128  2^7          child  raced         125090     117716    0.93     44.5  3.3e-16
   128x8x256  2^7          payonce raced         288187     261930    0.90     40.9  3.7e-16
    128x16x4  2^7          child  raced           9390      11167    1.18     28.4  3.0e-16
    128x16x8  2^7          child  raced          16233      18122    1.11     35.3  2.8e-16
   128x16x16  2^7          child  raced          30359      33349    1.09     40.5  3.0e-16
   128x16x32  2^7          child  raced          66477      68489    1.02     39.4  3.5e-16
   128x16x64  2^7          child  raced         133697     143105    1.06     41.7  3.4e-16
  128x16x128  2^7          child  raced         283447     261243    0.92     41.6  3.5e-16
  128x16x256  2^7          child  raced         582638     559337    0.95     42.7  4.1e-16
    128x32x4  2^7          child  raced          20575      28741    1.37     27.9  2.9e-16
    128x32x8  2^7          child  raced          34179      45178    1.30     36.0  2.6e-16
   128x32x16  2^7          child  raced          64338      83004    1.24     40.7  3.8e-16
   128x32x32  2^7          child+strips raced         160697     167943    1.04     34.7  2.9e-16
   128x32x64  2^7          child+strips raced         312880     358170    1.14     37.7  4.0e-16
  128x32x128  2^7          child+strips raced         647513     646375    0.99     38.5  4.8e-16
  128x32x256  2^7          child+strips raced        1348350    1324712    0.97     38.9  4.3e-16
    128x64x4  2^7          child  raced          46453      62874    1.35     26.5  3.4e-16
    128x64x8  2^7          child  raced          73795      97653    1.32     35.5  4.4e-16
   128x64x16  2^7          child+strips raced         148577     184658    1.19     37.5  4.4e-16
   128x64x32  2^7          child+strips raced         352413     374960    1.06     33.5  4.2e-16
   128x64x64  2^7          child+strips raced         662125     766893    1.15     37.6  4.4e-16
  128x64x128  2^7          child+strips raced        1395625    1389956    0.99     37.6  4.6e-16
   128x128x4  2^7          child  raced         100261     122947    1.22     26.1  3.8e-16
   128x128x8  2^7          child+strips raced         179713     205726    1.13     31.0  3.6e-16
  128x128x16  2^7          child+strips raced         323220     412630    1.25     36.5  3.9e-16
  128x128x32  2^7          child+strips raced         753050     810494    1.05     33.1  4.4e-16
  128x128x64  2^7          child+strips raced        1409287    1614118    1.14     37.2  5.5e-16
   128x256x4  2^7          child+strips raced         248297     286708    1.15     22.4  3.4e-16
   128x256x8  2^7          child  raced         436633     501497    1.14     27.0  3.5e-16
  128x256x16  2^7          child+strips raced         760675     941281    1.24     32.7  4.0e-16
  128x256x32  2^7          payonce+strips raced        1794875    1832044    1.00     29.2  3.9e-16
     256x4x4  2^8          child  raced           4993       5808    1.14     24.6  3.0e-16
     256x4x8  2^8          child  raced           7365      10075    1.36     36.2  2.6e-16
    256x4x16  2^8          child  raced          13412      19216    1.40     42.8  4.2e-16
    256x4x32  2^8          child  raced          26970      39506    1.45     45.6  2.8e-16
    256x4x64  2^8          child  raced          59339      88999    1.50     44.2  3.8e-16
   256x4x128  2^8          child  raced         127483     155576    1.03     43.7  3.3e-16
   256x4x256  2^8          child  raced         282567     338840    1.18     41.7  4.9e-16
     256x8x4  2^8          payonce raced          12104      13164    1.07     22.0  3.5e-16
     256x8x8  2^8          child  raced          14587      21684    1.24     39.3  2.6e-16
    256x8x16  2^8          child  raced          31216      41045    1.30     39.4  3.3e-16
    256x8x32  2^8          child  raced          63272      90562    1.43     41.4  3.7e-16
    256x8x64  2^8          child  raced         130700     177473    1.34     42.6  5.2e-16
   256x8x128  2^8          payonce raced         298500     354456    1.18     39.5  4.0e-16
   256x8x256  2^8          child  raced         587600     751862    1.28     42.4  4.0e-16
    256x16x4  2^8          child  raced          20862      30917    1.47     27.5  2.6e-16
    256x16x8  2^8          child  raced          34270      49862    1.45     35.9  3.7e-16
   256x16x16  2^8          child+strips raced          68534      92254    1.34     38.2  3.4e-16
   256x16x32  2^8          child  raced         152373     184828    1.17     36.6  5.1e-16
   256x16x64  2^8          child  raced         305593     406970    1.23     38.6  4.4e-16
  256x16x128  2^8          child+strips raced         649913     761324    1.14     38.3  4.8e-16
  256x16x256  2^8          child+strips raced        1321762    1579381    1.17     39.7  5.7e-16
    256x32x4  2^8          child+strips raced          46448      67226    1.43     26.5  3.2e-16
    256x32x8  2^8          child+strips raced          76823     105654    1.37     34.1  3.4e-16
   256x32x16  2^8          child  raced         148433     198660    1.31     37.5  3.6e-16
   256x32x32  2^8          child+strips raced         358873     423190    1.17     32.9  4.7e-16
   256x32x64  2^8          child+strips raced         665587     875168    1.30     37.4  4.2e-16
  256x32x128  2^8          child+strips raced        1386212    1570706    1.13     37.8  4.3e-16
    256x64x4  2^8          child+strips raced         104136     140729    1.35     25.2  3.1e-16
    256x64x8  2^8          child+strips raced         170850     223201    1.29     32.6  3.4e-16
   256x64x16  2^8          child+strips raced         337880     454750    1.32     34.9  4.9e-16
   256x64x32  2^8          child+strips raced         749875     892393    1.19     33.2  4.3e-16
   256x64x64  2^8          child+strips raced        1418725    1779419    1.25     37.0  4.1e-16
   256x128x4  2^8          child+strips raced         230033     299893    1.29     24.2  4.2e-16
   256x128x8  2^8          child+strips raced         403060     526183    1.30     29.3  4.1e-16
  256x128x16  2^8          child+strips raced         677862    1037250    1.51     36.7  4.5e-16
  256x128x32  2^8          child+strips raced        1599450    1865812    1.14     32.8  4.5e-16
   256x256x4  2^8          child+strips raced         542487     715333    1.28     21.7  4.9e-16
   256x256x8  2^8          child  raced         922112    1210043    1.30     27.0  3.9e-16
  256x256x16  2^8          child+strips raced        1619687    2241550    1.37     32.4  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                          252      1     27   0.99   1.18   1.41    1.17
 child+strips                    41      0      4   1.04   1.16   1.35    1.19
 payonce                         30      4     15   0.78   1.00   1.20    0.99
 payonce+strips                   5      0      1   0.97   1.02   1.19    1.06
 ALL                            328      5     47   0.97   1.16   1.40    1.15
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    326      5     46   0.97   1.16   1.40    1.15
 odd column                       1      0      1   0.95   0.95   0.95    0.95
 even column                      1      0      0   1.07   1.07   1.07    1.07
 ALL                            328      5     47   0.97   1.16   1.40    1.15
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                    146      0     11   1.02   1.18   1.41    1.18
 > 65536 points                  97      0     19   0.93   1.10   1.29    1.10
 1025..4096                      50      2      7   0.99   1.24   1.48    1.21
 257..1024                       25      1      5   0.84   1.25   1.47    1.19
 <= 256 points                   10      2      5   0.78   0.99   1.29    0.98
 ALL                            328      5     47   0.97   1.16   1.40    1.15
```


worst 10: 64x8x4 (payonce 0.72), 4x8x4 (payonce 0.73), 128x4x4 (child 0.76), 8x8x4 (payonce 0.78), 16x8x4 (payonce 0.78), 32x8x4 (payonce 0.80), 8x4x4 (child 0.81), 16x16x64 (payonce 0.81), 64x64x15 (child 0.81), 64x64x16 (child 0.82)
best 5: 4x8x32 (child 1.52), 4x8x64 (child 1.52), 8x4x64 (child 1.51), 256x128x16 (child+strips 1.51), 4x4x32 (child 1.50)
