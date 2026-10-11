# gauntlet report (3D)

run: `fftw_3d_r2c_real`  contract: 3D r2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 328 listed, 328 benched, comparator: FFTW 3D (out of place, MEASURE)

control cell 64x64x64: 10 readings, 1.100..1.191


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
       4x4x4  2^2          child  replayed          82         47    0.56     11.8  0.0e+00
       4x4x8  2^2          child  replayed          93         79    0.84     24.1  2.7e-16
      4x4x16  2^2          payonce replayed         159        163    1.02     32.2  2.0e-16
      4x4x32  2^2          child  replayed         263        388    1.39     43.8  2.6e-16
      4x4x64  2^2          child  replayed         563        856    1.52     45.5  2.1e-16
     4x4x128  2^2          payonce replayed        1257       1421    1.11     44.8  3.1e-16
     4x4x256  2^2          child  replayed        2450       2522    0.80     50.2  3.0e-16
       4x8x4  2^2          payonce replayed         128         89    0.69     17.5  1.5e-16
       4x8x8  2^2          child  replayed         143        158    1.09     35.8  1.4e-16
      4x8x16  2^2          child  replayed         261        338    1.29     44.2  2.5e-16
      4x8x32  2^2          child  replayed         512        748    1.46     50.0  3.1e-16
      4x8x64  2^2          child  replayed        1159       1765    1.51     48.6  2.6e-16
     4x8x128  2^2          child  replayed        2323       2692    1.15     52.9  3.1e-16
     4x8x256  2^2          child  replayed        5425       5652    0.86     49.1  3.1e-16
      4x16x4  2^2          child  replayed         159        192    1.16     32.2  1.7e-16
      4x16x8  2^2          child  replayed         260        346    1.31     44.3  2.5e-16
     4x16x16  2^2          child  replayed         493        692    1.40     51.9  1.9e-16
     4x16x32  2^2          child  replayed        1076       1535    1.40     52.3  2.5e-16
     4x16x64  2^2          child  replayed        2496       3634    1.43     49.2  2.9e-16
    4x16x128  2^2          child  replayed        5162       5845    1.12     51.6  2.9e-16
    4x16x256  2^2          child  replayed       10779      11939    1.09     53.2  3.0e-16
      4x32x4  2^2          child  replayed         336        397    1.13     34.3  2.7e-16
      4x32x8  2^2          child  replayed         547        714    1.25     46.8  2.7e-16
     4x32x16  2^2          child  replayed        1117       1421    0.84     50.4  2.4e-16
     4x32x32  2^2          child  replayed        2687       3349    1.18     45.7  2.6e-16
     4x32x64  2^2          child  replayed        5705       7749    1.33     46.7  3.2e-16
    4x32x128  2^2          child  replayed       11172      12261    1.09     51.3  3.6e-16
    4x32x256  2^2          child  replayed       24075      24506    1.00     51.0  3.8e-16
      4x64x4  2^2          child  replayed         766        816    1.04     33.4  2.9e-16
      4x64x8  2^2          child  replayed        1222       1453    1.17     46.1  2.2e-16
     4x64x16  2^2          child  replayed        2513       3379    1.29     48.9  3.1e-16
     4x64x32  2^2          child  replayed        6543       7200    1.03     40.7  3.5e-16
     4x64x64  2^2          child  replayed       11968      16263    1.30     47.9  3.0e-16
    4x64x128  2^2          child  replayed       24572      25803    1.03     50.0  3.9e-16
    4x64x256  2^2          child  replayed       51182      53276    0.99     51.2  4.0e-16
     4x128x4  2^2          child  replayed        1851       1900    1.00     30.4  3.5e-16
     4x128x8  2^2          child  replayed        3292       3772    1.12     37.3  2.6e-16
    4x128x16  2^2          child  replayed        5666       7449    1.30     47.0  2.9e-16
    4x128x32  2^2          child  replayed       14434      15379    1.03     39.7  4.4e-16
    4x128x64  2^2          child  replayed       26108      34423    1.31     47.1  4.2e-16
   4x128x128  2^2          child  replayed       52152      54775    1.04     50.3  3.3e-16
   4x128x256  2^2          child  replayed      109913     115161    1.04     50.7  4.0e-16
     4x256x4  2^2          child  replayed        4581       5245    1.14     26.8  3.3e-16
     4x256x8  2^2          child  replayed        7854       9448    1.20     33.9  3.9e-16
    4x256x16  2^2          child  replayed       14625      17814    1.01     39.2  4.1e-16
    4x256x32  2^2          child  replayed       32357      36745    1.13     38.0  3.5e-16
    4x256x64  2^2          child  replayed       63356      76232    1.20     41.4  3.5e-16
   4x256x128  2^2          child  replayed      132547     135015    1.01     42.0  2.9e-16
   4x256x256  2^2          payonce replayed      294693     325037    1.10     40.0  4.6e-16
       8x4x4  2^3          child  replayed         178        101    0.56     12.6  1.4e-16
       8x4x8  2^3          child  replayed         192        167    0.86     26.6  1.8e-16
      8x4x16  2^3          payonce replayed         303        337    1.11     38.0  2.8e-16
      8x4x32  2^3          child  replayed         560        751    1.33     45.7  3.4e-16
      8x4x64  2^3          child  replayed        1181       1763    1.49     47.7  2.2e-16
     8x4x128  2^3          child  replayed        2355       2723    1.15     52.2  2.9e-16
     8x4x256  2^3          child  replayed        5151       5482    1.02     51.7  3.1e-16
       8x8x4  2^3          payonce replayed         247        181    0.70     20.7  4.0e-16
       8x8x8  2^3          child  replayed         304        331    1.07     37.9  2.7e-16
      8x8x16  2^3          payonce replayed         578        716    1.22     44.3  2.4e-16
      8x8x32  2^3          child  replayed        1082       1529    1.40     52.0  2.3e-16
      8x8x64  2^3          child  replayed        2494       3682    1.47     49.3  3.6e-16
     8x8x128  2^3          child  replayed        4950       5660    1.06     53.8  3.7e-16
     8x8x256  2^3          child  replayed       10460      11207    1.07     54.8  3.2e-16
      8x16x4  2^3          child  replayed         339        445    1.19     34.0  1.9e-16
      8x16x8  2^3          child  replayed         535        710    1.29     47.9  2.6e-16
     8x16x16  2^3          child  replayed        1043       1449    1.36     54.0  3.0e-16
     8x16x32  2^3          child  replayed        2540       3319    1.29     48.4  2.7e-16
     8x16x64  2^3          child  replayed        5263       7621    1.42     50.6  3.0e-16
    8x16x128  2^3          child  replayed       10567      12013    1.13     54.3  3.6e-16
    8x16x256  2^3          child  replayed       22278      23862    1.06     55.2  3.3e-16
      8x32x4  2^3          child  replayed         702        802    1.13     36.5  2.5e-16
      8x32x8  2^3          child  replayed        1135       1452    1.25     49.6  2.8e-16
     8x32x16  2^3          child  replayed        2499       3177    1.25     49.2  3.2e-16
     8x32x32  2^3          child  replayed        6231       7093    1.11     42.7  2.8e-16
     8x32x64  2^3          child  replayed       11566      15914    1.36     49.6  3.3e-16
    8x32x128  2^3          child  replayed       22539      25099    1.09     54.5  3.1e-16
    8x32x256  2^3          child  replayed       47066      50565    1.07     55.7  3.8e-16
      8x64x4  2^3          child  replayed        1605       1715    1.04     35.1  2.9e-16
      8x64x8  2^3          child  replayed        2602       3562    1.33     47.2  3.6e-16
     8x64x16  2^3          child  replayed        5255       7314    1.36     50.7  3.7e-16
     8x64x32  2^3          child  replayed       13341      15040    1.10     43.0  3.5e-16
     8x64x64  2^3          child  replayed       25074      33375    1.31     49.0  3.4e-16
    8x64x128  2^3          child  replayed       49084      53150    1.08     53.4  4.0e-16
    8x64x256  2^3          child  replayed      105677     108976    1.02     52.7  4.2e-16
     8x128x4  2^3          child  replayed        4059       4991    1.22     30.3  2.8e-16
     8x128x8  2^3          child  replayed        6724       8638    1.26     39.6  4.1e-16
    8x128x16  2^3          child  replayed       11684      15420    1.31     49.1  3.1e-16
    8x128x32  2^3          child  replayed       29311      31848    0.88     41.9  4.0e-16
    8x128x64  2^3          child  replayed       52941      65730    1.23     49.5  3.8e-16
   8x128x128  2^3          child  replayed      109103     113611    1.04     51.1  5.1e-16
   8x128x256  2^3          child  replayed      236187     251126    1.05     49.9  4.6e-16
     8x256x4  2^3          child  replayed       12849      11389    0.87     20.7  3.1e-16
     8x256x8  2^3          payonce replayed       19412      19225    0.99     29.5  3.8e-16
    8x256x16  2^3          child  replayed       29069      36289    1.24     42.3  4.2e-16
    8x256x32  2^3          child  replayed       64759      74667    1.15     40.5  4.0e-16
    8x256x64  2^3          child  replayed      128963     156000    1.21     43.2  3.4e-16
   8x256x128  2^3          child  replayed      276660     284393    1.02     42.6  3.9e-16
   8x256x256  2^3          child  replayed      597525     684413    1.12     41.7  4.9e-16
      16x4x4  2^4          child  replayed         331        201    0.60     15.5  1.5e-16
      16x4x8  2^4          child  replayed         394        343    0.87     29.2  2.6e-16
     16x4x16  2^4          child  replayed         563        706    1.23     45.5  3.0e-16
     16x4x32  2^4          child  replayed        1373       1538    1.11     41.0  2.5e-16
     16x4x64  2^4          child  replayed        2577       3724    1.44     47.7  3.1e-16
    16x4x128  2^4          payonce replayed        5860       5884    0.95     45.4  3.0e-16
    16x4x256  2^4          child  replayed       10855      11698    1.01     52.8  3.3e-16
      16x8x4  2^4          payonce replayed         520        417    0.74     22.2  1.9e-16
      16x8x8  2^4          child  replayed         660        702    1.05     38.8  3.3e-16
     16x8x16  2^4          child  replayed        1186       1496    1.26     47.5  3.4e-16
     16x8x32  2^4          child  replayed        2427       3344    1.36     50.6  2.7e-16
     16x8x64  2^4          child  replayed        5365       7749    1.43     49.6  3.0e-16
    16x8x128  2^4          child  replayed       10541      11944    1.13     54.4  3.0e-16
    16x8x256  2^4          child  replayed       22487      23902    1.01     54.6  3.4e-16
     16x16x4  2^4          child  replayed         721        862    1.19     35.5  2.8e-16
     16x16x8  2^4          child  replayed        1199       1539    1.13     47.0  2.8e-16
    16x16x16  2^4          child  replayed        2305       3264    1.41     53.3  2.6e-16
    16x16x32  2^4          child  replayed        5912       7064    1.19     45.0  2.6e-16
    16x16x64  2^4          payonce replayed       14244      16109    1.11     40.3  3.3e-16
   16x16x128  2^4          child  replayed       22407      25713    1.13     54.8  3.3e-16
   16x16x256  2^4          child  replayed       46472      50923    1.09     56.4  3.7e-16
     16x32x4  2^4          child  replayed        1925       1727    0.88     29.3  2.3e-16
     16x32x8  2^4          child  replayed        2526       3637    1.42     48.6  2.6e-16
    16x32x16  2^4          child  replayed        5189       7188    1.36     51.3  3.4e-16
    16x32x32  2^4          child  replayed       13531      15204    1.11     42.4  3.2e-16
    16x32x64  2^4          child  replayed       24656      33245    1.32     49.8  3.3e-16
   16x32x128  2^4          child  replayed       47685      52757    1.10     55.0  3.8e-16
   16x32x256  2^4          child  replayed      101817     108748    1.05     54.7  3.7e-16
     16x64x4  2^4          child  replayed        3495       4755    1.25     35.2  3.2e-16
     16x64x8  2^4          child  replayed        5712       8331    1.42     46.6  3.1e-16
    16x64x16  2^4          child  replayed       11601      15744    1.28     49.4  3.6e-16
    16x64x32  2^4          child  replayed       28248      31894    1.11     43.5  4.2e-16
    16x64x63  2^4          payonce replayed       85274     127949    1.45     30.2  5.2e-16
    16x64x64  2^4          child  replayed       51703      65181    1.25     50.7  3.8e-16
   16x64x128  2^4          child  replayed      106253     113413    1.05     52.4  3.2e-16
   16x64x256  2^4          child  replayed      230233     253106    1.08     51.2  4.6e-16
    16x128x4  2^4          child  replayed        8570      11088    1.27     31.1  2.8e-16
    16x128x8  2^4          child  replayed       14087      18166    1.27     40.7  3.5e-16
   16x128x16  2^4          child  replayed       24334      32845    1.29     50.5  3.9e-16
   16x128x32  2^4          child  replayed       60874      66231    1.08     43.1  4.1e-16
   16x128x64  2^4          child  replayed      110547     138221    1.24     50.4  3.8e-16
  16x128x128  2^4          child  replayed      237727     253033    1.05     49.6  4.6e-16
  16x128x256  2^4          child  replayed      500138     536256    1.04     49.8  4.7e-16
    16x256x4  2^4          child  replayed       24586      30032    1.21     23.3  3.4e-16
    16x256x8  2^4          child  replayed       41497      50477    1.13     29.6  3.0e-16
   16x256x16  2^4          child  replayed       75538      92791    1.22     34.7  3.8e-16
   16x256x32  2^4          child  replayed      170007     188730    1.10     32.8  4.1e-16
   16x256x64  2^4          payonce replayed      324467     401290    1.21     36.4  4.1e-16
  16x256x128  2^4          payonce replayed      665212     736937    1.10     37.4  3.7e-16
  16x256x256  2^4          payonce replayed     1483675    1664018    1.10     35.3  4.2e-16
     27x9x15  3^3          child  replayed        5361       5130    0.94     20.1  5.2e-16
      32x4x4  2^5          child  replayed         679        404    0.58     17.0  2.4e-16
      32x4x8  2^5          payonce replayed         867        709    0.81     29.5  2.4e-16
     32x4x16  2^5          child  replayed        1185       1517    0.93     47.5  2.7e-16
     32x4x32  2^5          child  replayed        2686       3473    1.25     45.7  3.1e-16
     32x4x64  2^5          child  replayed        5581       7932    1.41     47.7  3.0e-16
    32x4x128  2^5          child  replayed       11025      12536    1.13     52.0  3.5e-16
    32x4x256  2^5          child  replayed       23458      24949    1.06     52.4  3.4e-16
      32x8x4  2^5          payonce replayed        1079        816    0.74     23.7  2.1e-16
      32x8x8  2^5          child  replayed        1521       1481    0.96     37.0  2.7e-16
     32x8x16  2^5          child  replayed        2655       3376    1.20     46.3  2.9e-16
     32x8x32  2^5          child  replayed        5478       7181    1.30     48.6  2.6e-16
     32x8x64  2^5          payonce replayed       12858      16352    1.22     44.6  2.9e-16
    32x8x128  2^5          child  replayed       23485      25816    1.08     52.3  3.7e-16
    32x8x256  2^5          child  replayed       49213      52996    1.05     53.3  3.9e-16
     32x16x4  2^5          child  replayed        1601       1849    1.12     35.2  3.0e-16
     32x16x8  2^5          child  replayed        2554       3777    1.44     48.1  2.5e-16
    32x16x16  2^5          child  replayed        5126       7279    1.41     51.9  3.1e-16
    32x16x32  2^5          child  replayed       13648      15462    1.00     42.0  3.6e-16
    32x16x64  2^5          child  replayed       26059      34455    1.32     47.2  3.5e-16
   32x16x128  2^5          child  replayed       49569      55155    1.11     52.9  3.5e-16
   32x16x256  2^5          child  replayed      104503     112205    1.05     53.3  3.6e-16
     32x32x4  2^5          child  replayed        4405       4791    1.07     27.9  2.7e-16
     32x32x8  2^5          child  replayed        5676       8363    1.47     46.9  2.8e-16
    32x32x16  2^5          child  replayed       11579      15848    1.10     49.5  3.0e-16
    32x32x27  2^5          child  replayed       32212      46227    1.42     31.7  4.3e-16
    32x32x32  2^5          child  replayed       29425      32552    1.10     41.8  3.9e-16
    32x32x64  2^5          child  replayed       53931      67233    1.24     48.6  3.9e-16
   32x32x128  2^5          child  replayed      109460     116505    1.05     50.9  4.2e-16
   32x32x256  2^5          child  replayed      240747     259033    1.06     49.0  4.8e-16
     32x64x4  2^5          child  replayed       10267      10836    1.02     25.9  4.1e-16
     32x64x8  2^5          child  replayed       16781      18006    1.05     34.2  4.6e-16
    32x64x16  2^5          child  replayed       32357      33965    1.03     38.0  3.0e-16
    32x64x32  2^5          child  replayed       62107      67820    1.09     42.2  4.1e-16
    32x64x64  2^5          child  replayed      114127     141141    1.20     48.8  4.3e-16
   32x64x128  2^5          child  replayed      245893     258943    1.04     48.0  4.9e-16
   32x64x256  2^5          child  replayed      502725     545212    1.07     49.5  4.4e-16
    32x128x4  2^5          child  replayed       22609      29033    1.25     25.4  3.4e-16
    32x128x8  2^5          child  replayed       37198      47362    1.24     33.0  3.9e-16
   32x128x16  2^5          child  replayed       66552      83901    1.19     39.4  4.0e-16
   32x128x32  2^5          child  replayed      162130     173103    1.06     34.4  4.9e-16
   32x128x64  2^5          payonce replayed      320407     362323    1.13     36.8  3.6e-16
  32x128x128  2^5          child  replayed      650438     658969    1.01     38.3  3.8e-16
  32x128x256  2^5          payonce replayed     1411325    1393394    0.93     37.1  4.6e-16
    32x256x4  2^5          child+strips replayed       54470      61252    1.12     22.6  3.6e-16
    32x256x8  2^5          child+strips replayed       89718     103196    1.12     29.2  3.6e-16
   32x256x16  2^5          child+strips replayed      166917     191227    1.14     33.4  3.8e-16
   32x256x32  2^5          child+strips replayed      365953     404030    1.10     32.2  3.6e-16
   32x256x64  2^5          payonce+strips replayed      675175     829781    1.22     36.9  3.8e-16
  32x256x128  2^5          payonce+strips replayed     1547175    1508575    0.95     33.9  4.3e-16
    36x20x28  2^2.3^2      child  replayed       27995      29864    1.06     25.7  4.1e-16
      64x4x4  2^6          child  replayed        1366        845    0.59     18.7  3.1e-16
      64x4x8  2^6          child  replayed        1798       1508    0.82     31.3  3.5e-16
     64x4x16  2^6          child  replayed        2577       3278    1.21     47.7  3.5e-16
     64x4x32  2^6          child  replayed        5969       7399    1.23     44.6  3.0e-16
     64x4x64  2^6          child  replayed       13421      16662    1.24     42.7  3.3e-16
    64x4x128  2^6          child  replayed       26104      27290    1.04     47.1  3.8e-16
    64x4x256  2^6          child  replayed       50277      54050    1.03     52.1  3.4e-16
      64x8x4  2^6          payonce replayed        2522       1799    0.70     22.3  2.6e-16
      64x8x8  2^6          child  replayed        3661       3623    0.99     33.6  2.5e-16
     64x8x16  2^6          child  replayed        5803       7611    1.30     45.9  3.3e-16
     64x8x32  2^6          child  replayed       13228      15821    1.18     43.4  3.8e-16
     64x8x64  2^6          child  replayed       26007      34371    1.31     47.2  3.8e-16
    64x8x128  2^6          child  replayed       54270      55687    1.02     48.3  4.0e-16
    64x8x256  2^6          child  replayed      124043     112645    0.88     44.9  3.6e-16
     64x16x4  2^6          child  replayed        3435       5050    1.46     35.8  2.3e-16
     64x16x8  2^6          child  replayed        5890       8667    1.42     45.2  3.4e-16
    64x16x16  2^6          child  replayed       12857      15941    1.11     44.6  3.6e-16
    64x16x32  2^6          child  replayed       31773      32878    1.00     38.7  3.5e-16
    64x16x64  2^6          child  replayed       59767      68424    1.11     43.9  3.9e-16
   64x16x128  2^6          child  replayed      127903     118608    0.92     43.6  3.4e-16
   64x16x256  2^6          child  replayed      282380     269313    0.86     41.8  3.6e-16
     64x32x4  2^6          child  replayed        9483      10807    1.13     28.1  3.1e-16
     64x32x8  2^6          child  replayed       12836      18135    1.36     44.7  4.2e-16
    64x32x16  2^6          child  replayed       25111      33822    1.33     48.9  3.8e-16
    64x32x32  2^6          payonce replayed       71908      68359    0.94     36.5  3.9e-16
    64x32x64  2^6          child  replayed      136640     143087    1.04     40.8  3.8e-16
   64x32x128  2^6          child  replayed      299027     259120    0.81     39.5  4.1e-16
   64x32x256  2^6          child  replayed      624087     556875    0.85     39.9  3.9e-16
     64x64x4  2^6          child+strips replayed       23258      27337    1.17     24.7  3.3e-16
     64x64x8  2^6          child  replayed       46572      45322    0.97     26.4  3.5e-16
    64x64x15  2^6          child  replayed       94634      76983    0.81     25.8  4.1e-16
    64x64x16  2^6          child  replayed       85343      87078    0.98     30.7  3.9e-16
    64x64x32  2^6          child  replayed      156380     167625    1.07     35.6  3.3e-16
    64x64x64  2^6          child  replayed      305487     356320    1.16     38.6  4.0e-16
   64x64x128  2^6          payonce replayed      634050     644862    1.01     39.3  4.8e-16
   64x64x256  2^6          payonce replayed     1360888    1348112    0.98     38.5  4.4e-16
    64x128x4  2^6          child  replayed       63103      59224    0.92     19.5  3.8e-16
    64x128x8  2^6          child  replayed       97485      97405    0.98     26.9  3.6e-16
   64x128x16  2^6          child  replayed      144953     180870    1.23     38.4  4.3e-16
   64x128x32  2^6          payonce replayed      353213     374966    1.04     33.4  3.4e-16
   64x128x64  2^6          child  replayed      644487     763144    1.18     38.6  5.0e-16
  64x128x128  2^6          child+strips replayed     1382088    1383894    0.98     37.9  5.0e-16
    64x256x4  2^6          child  replayed      113407     127278    1.11     23.1  3.2e-16
    64x256x8  2^6          child+strips replayed      192887     215755    1.11     28.9  3.2e-16
   64x256x16  2^6          child+strips replayed      363133     433443    1.18     32.5  3.8e-16
   64x256x32  2^6          payonce+strips replayed      834912     845637    1.00     29.8  4.0e-16
   64x256x64  2^6          payonce+strips replayed     1451675    1694593    1.15     36.1  4.7e-16
     128x4x4  2^7          child  replayed        3096       1859    0.59     18.2  2.1e-16
     128x4x8  2^7          child  replayed        4318       3745    0.86     28.5  3.1e-16
    128x4x16  2^7          child  replayed        6454       7464    1.14     41.3  3.5e-16
    128x4x32  2^7          child  replayed       14103      15674    1.10     40.7  2.5e-16
    128x4x64  2^7          child  replayed       28014      35354    1.25     43.9  3.4e-16
   128x4x128  2^7          payonce replayed       56166      57125    1.01     46.7  3.6e-16
   128x4x256  2^7          child  replayed      130513     116073    0.88     42.7  3.4e-16
     128x8x4  2^7          payonce replayed        5655       4845    0.85     21.7  2.8e-16
     128x8x8  2^7          child  replayed        7663       8526    1.11     34.7  3.4e-16
    128x8x16  2^7          child  replayed       13724      16200    1.15     41.8  3.4e-16
    128x8x32  2^7          child  replayed       28526      32636    1.14     43.1  3.4e-16
    128x8x64  2^7          child  replayed       58934      68256    1.14     44.5  3.8e-16
   128x8x128  2^7          child  replayed      124427     119152    0.91     44.8  3.3e-16
   128x8x256  2^7          payonce replayed      291860     257576    0.88     40.4  3.7e-16
    128x16x4  2^7          child  replayed        9364      10988    1.15     28.4  2.8e-16
    128x16x8  2^7          child  replayed       17005      18174    1.04     33.7  2.8e-16
   128x16x16  2^7          child  replayed       30185      33093    1.06     40.7  3.0e-16
   128x16x32  2^7          child  replayed       67618      68619    0.99     38.8  3.5e-16
   128x16x64  2^7          child  replayed      135063     142530    1.05     41.2  3.4e-16
  128x16x128  2^7          child  replayed      299453     264930    0.87     39.4  3.7e-16
  128x16x256  2^7          child  replayed      584800     563262    0.95     42.6  4.1e-16
    128x32x4  2^7          child  replayed       20932      27297    1.29     27.4  4.0e-16
    128x32x8  2^7          child  replayed       34558      45203    1.30     35.6  2.6e-16
   128x32x16  2^7          child  replayed       65470      82688    1.23     40.0  3.4e-16
   128x32x32  2^7          child+strips replayed      163687     167060    1.01     34.0  2.9e-16
   128x32x64  2^7          child+strips replayed      314933     355836    1.11     37.5  4.0e-16
  128x32x128  2^7          child+strips replayed      644300     645925    0.99     38.7  4.8e-16
  128x32x256  2^7          child+strips replayed     1367650    1329231    0.94     38.3  4.3e-16
    128x64x4  2^7          child  replayed       46180      58512    1.24     26.6  3.4e-16
    128x64x8  2^7          child  replayed       73198      97156    1.27     35.8  4.4e-16
   128x64x16  2^7          child+strips replayed      148990     183565    1.22     37.4  4.4e-16
   128x64x32  2^7          child+strips replayed      354793     377243    1.06     33.2  4.2e-16
   128x64x64  2^7          child+strips replayed      666363     770700    1.15     37.4  4.4e-16
  128x64x128  2^7          child+strips replayed     1393750    1393843    0.91     37.6  4.6e-16
   128x128x4  2^7          child  replayed      102566     130187    1.20     25.6  3.1e-16
   128x128x8  2^7          child+strips replayed      180130     216210    1.18     30.9  4.2e-16
  128x128x16  2^7          child+strips replayed      326420     414953    1.17     36.1  3.9e-16
  128x128x32  2^7          child+strips replayed      765050     816781    0.99     32.6  4.4e-16
  128x128x64  2^7          child+strips replayed     1414425    1627831    1.07     37.1  5.5e-16
   128x256x4  2^7          child+strips replayed      251147     287185    1.10     22.2  3.4e-16
   128x256x8  2^7          child  replayed      439327     501390    1.12     26.9  3.5e-16
  128x256x16  2^7          child+strips replayed      762375     943818    1.23     32.7  4.0e-16
  128x256x32  2^7          payonce+strips replayed     1803000    1838869    1.02     29.1  3.9e-16
     256x4x4  2^8          child  replayed        6643       5820    0.84     18.5  3.0e-16
     256x4x8  2^8          child  replayed        9127      10097    1.09     29.2  2.6e-16
    256x4x16  2^8          child  replayed       13750      18920    1.36     41.7  4.2e-16
    256x4x32  2^8          child  replayed       33439      39831    1.18     36.7  2.8e-16
    256x4x64  2^8          child  replayed       59259      85257    1.42     44.2  3.8e-16
   256x4x128  2^8          child  replayed      126437     154160    1.22     44.1  3.3e-16
   256x4x256  2^8          child  replayed      284893     336796    1.15     41.4  4.9e-16
     256x8x4  2^8          payonce replayed       12901      13234    1.00     20.6  3.5e-16
     256x8x8  2^8          child  replayed       16502      21613    1.28     34.7  2.6e-16
    256x8x16  2^8          child  replayed       32544      41568    1.26     37.8  3.3e-16
    256x8x32  2^8          child  replayed       65882      90882    1.27     39.8  3.7e-16
    256x8x64  2^8          child  replayed      129277     176658    1.31     43.1  3.9e-16
   256x8x128  2^8          payonce replayed      293440     351300    1.19     40.2  4.0e-16
   256x8x256  2^8          child  replayed      585675     809956    1.35     42.5  4.0e-16
    256x16x4  2^8          child  replayed       20960      30567    1.45     27.4  2.6e-16
    256x16x8  2^8          child  replayed       33770      49979    1.48     36.4  3.7e-16
   256x16x16  2^8          child+strips replayed       76020      91761    1.07     34.5  3.4e-16
   256x16x32  2^8          child  replayed      156257     183830    1.17     35.6  3.4e-16
   256x16x64  2^8          child  replayed      310067     405550    1.24     38.0  4.4e-16
  256x16x128  2^8          child+strips replayed      654313     779500    1.18     38.1  4.0e-16
  256x16x256  2^8          child+strips replayed     1343225    1584843    1.17     39.0  4.6e-16
    256x32x4  2^8          child+strips replayed       46647      62615    1.23     26.3  3.7e-16
    256x32x8  2^8          child+strips replayed       76892     106307    1.37     34.1  3.4e-16
   256x32x16  2^8          child  replayed      147763     199995    1.33     37.7  4.1e-16
   256x32x32  2^8          child+strips replayed      367027     422907    1.15     32.1  4.4e-16
   256x32x64  2^8          child+strips replayed      669188     893856    1.32     37.2  4.6e-16
  256x32x128  2^8          child+strips replayed     1382038    1566231    1.11     37.9  4.3e-16
    256x64x4  2^8          child+strips replayed      104164     130768    1.25     25.2  3.1e-16
    256x64x8  2^8          child+strips replayed      173497     226758    1.29     32.1  3.4e-16
   256x64x16  2^8          child+strips replayed      335547     449350    1.17     35.2  4.9e-16
   256x64x32  2^8          child+strips replayed      752213     901825    1.18     33.1  3.9e-16
   256x64x64  2^8          child+strips replayed     1424988    1774825    1.23     36.8  4.1e-16
   256x128x4  2^8          child+strips replayed      231473     297708    1.27     24.1  4.2e-16
   256x128x8  2^8          child+strips replayed      403380     533897    1.30     29.2  3.9e-16
  256x128x16  2^8          child+strips replayed      700875    1046294    1.47     35.5  4.5e-16
  256x128x32  2^8          child+strips replayed     1598462    1870256    1.14     32.8  4.5e-16
   256x256x4  2^8          child+strips replayed      527087     714453    1.32     22.4  3.7e-16
   256x256x8  2^8          child  replayed      929513    1215856    1.29     26.8  3.5e-16
  256x256x16  2^8          child+strips replayed     1629338    2285125    1.38     32.2  4.4e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 child                          252      7     39   0.91   1.13   1.39    1.13
 child+strips                    41      0      5   0.99   1.17   1.32    1.16
 payonce                         30      5     14   0.74   1.01   1.22    0.98
 payonce+strips                   5      0      1   0.95   1.02   1.22    1.06
 ALL                            328     12     59   0.88   1.13   1.36    1.11
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                    326     12     58   0.88   1.13   1.36    1.12
 odd column                       1      0      1   0.94   0.94   0.94    0.94
 even column                      1      0      0   1.06   1.06   1.06    1.06
 ALL                            328     12     59   0.88   1.13   1.36    1.11
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 4097..65536                    146      0     15   1.00   1.13   1.36    1.16
 > 65536 points                  97      0     18   0.92   1.10   1.29    1.09
 1025..4096                      50      3     13   0.84   1.17   1.44    1.13
 257..1024                       25      4      6   0.74   1.19   1.40    1.08
 <= 256 points                   10      5      7   0.56   0.77   1.16    0.78
 ALL                            328     12     59   0.88   1.13   1.36    1.11
```


worst 10: 4x4x4 (child 0.56), 8x4x4 (child 0.56), 32x4x4 (child 0.58), 64x4x4 (child 0.59), 128x4x4 (child 0.59), 16x4x4 (child 0.60), 4x8x4 (payonce 0.69), 8x8x4 (payonce 0.70), 64x8x4 (payonce 0.70), 16x8x4 (payonce 0.74)
best 5: 4x4x64 (child 1.52), 4x8x64 (child 1.51), 8x4x64 (child 1.49), 256x16x8 (child 1.48), 256x128x16 (child+strips 1.47)
