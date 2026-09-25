# gauntlet report (3D)

run: `3d_mt8_team_2026-09-25`  contract: 3D c2c interleaved, natural, out of place, K=1_mt8  cells: 76 listed, 76 benched, comparator: MKL DFTI 3D (out of place)

control cell 64x64x64: 4 readings, 1.120..1.236

threaded plans that ran serial (engaged = 0 at both flips): 0


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
  2x128x4096  2            chain  raced        2018075    1873625    0.65     52.0  3.2e-15
  2x128x8192  2            chain  raced        5234937    4465731    0.76     42.1  5.3e-15
  2x256x4096  2            chain  raced        5581712    4530400    0.59     39.5  5.6e-15
  2x256x8192  2            chain  raced       11978837   12439674    0.84     38.5  6.8e-15
  2x512x2048  2            chain  raced        4886538    4501581    0.89     45.1  5.2e-15
  2x512x4096  2            chain  raced       12128313   10606549    0.85     38.0  6.1e-15
 2x1024x2048  2            chain  raced       11627762   11747738    0.94     39.7  6.1e-15
  2x2048x512  2            chain  raced        5224675    5020294    0.63     42.1  5.3e-15
  2x4096x256  2            chain  raced        5736900    5195369    0.73     38.4  5.6e-15
   2x8192x64  2            chain  raced        2593575    2446150    0.71     40.4  3.0e-15
  2x8192x128  2            chain  raced        6301987    5540925    0.86     34.9  5.1e-15
   4x32x8192  2^2          chain  raced         924012     796906    0.83    113.5  2.6e-15
   4x64x8192  2^2          chain  raced        3214688    1956956    0.59     68.5  2.9e-15
  4x128x2048  2^2          chain  raced         928138     851075    0.83    113.0  2.4e-15
  4x128x4096  2^2          chain  raced        4348713    2157656    0.47     50.6  2.6e-15
  4x128x8192  2^2          chain  raced        9483812    6291793    0.62     48.6  7.5e-15
  4x256x2048  2^2          chain  raced        3435787    2207862    0.51     64.1  2.5e-15
  4x256x4096  2^2          chain  raced        9574713    6734581    0.64     48.2  7.4e-15
   4x512x512  2^2          chain  raced         999600    1002431    0.91    104.9  2.4e-15
  4x512x1024  2^2          chain  raced        3781213    2438125    0.57     58.2  2.7e-15
  4x512x2048  2^2          chain  raced        8898225    6989750    0.75     51.9  6.6e-15
  4x1024x256  2^2          chain  raced        1012375    1041456    0.87    103.6  2.3e-15
  4x1024x512  2^2          chain  raced        3369613    2377581    0.61     65.3  2.8e-15
 4x1024x1024  2^2          chain  raced        8434200    6519831    0.74     54.7  7.7e-15
  4x2048x128  2^2          chain  raced        1188213    1107493    0.93     88.2  2.5e-15
  4x2048x256  2^2          chain  raced        3270975    2437781    0.74     67.3  2.4e-15
  4x2048x512  2^2          chain  raced        8463862    7108262    0.81     54.5  7.4e-15
   4x4096x64  2^2          chain  raced        1253263    1100294    0.68     83.7  2.5e-15
  4x4096x128  2^2          chain  raced        3449163    2728112    0.75     63.8  2.8e-15
  4x4096x256  2^2          chain  raced        9881000    7466931    0.73     46.7  7.3e-15
   4x8192x32  2^2          chain  raced        1378025    1159031    0.76     76.1  2.6e-15
   4x8192x64  2^2          chain  raced        4826588    2866543    0.59     45.6  2.6e-15
  4x8192x128  2^2          chain  raced       11376450    8177218    0.71     40.6  6.6e-15
   8x32x8192  2^3          chain  raced        2884637    1362487    0.40     76.3  2.0e-15
   8x64x4096  2^3          chain  raced        2667287    1283162    0.37     82.6  2.0e-15
   8x64x8192  2^3          chain  raced        7235700    5329993    0.71     63.8  3.5e-15
  8x128x1024  2^3          chain  raced         509200     620493    0.95    205.9  1.4e-15
  8x128x2048  2^3          chain  raced        2997887    1241881    0.41     73.5  2.0e-15
  8x128x4096  2^3          chain  raced        7358962    4763812    0.59     62.7  3.2e-15
   8x256x512  2^3          chain  raced         614750     498868    0.79    170.6  1.3e-15
  8x256x1024  2^3          chain  raced        3487113    1445381    0.34     63.1  2.1e-15
  8x256x2048  2^3          chain  raced        7909975    5998319    0.73     58.3  3.5e-15
   8x512x256  2^3          chain  raced         603412     553912    0.50    173.8  1.3e-15
   8x512x512  2^3          chain  raced        3278188    1616731    0.46     67.2  2.0e-15
  8x512x1024  2^3          chain  raced        7661175    4983375    0.63     60.2  3.7e-15
  8x1024x128  2^3          chain  raced         626700     537481    0.81    167.3  1.4e-15
  8x1024x256  2^3          chain  raced        2771737    1545187    0.53     79.4  2.1e-15
  8x1024x512  2^3          chain  raced        6935437    5879181    0.83     66.5  3.4e-15
   8x2048x64  2^3          chain  raced         640538     505044    0.77    163.7  1.3e-15
  8x2048x128  2^3          chain  raced        2748200    1611143    0.57     80.1  2.0e-15
  8x2048x256  2^3          chain  raced        7166988    5226250    0.71     64.4  3.6e-15
   8x4096x32  2^3          chain  raced         674100     640543    0.76    155.6  1.3e-15
   8x4096x64  2^3          chain  raced        2967425    1742206    0.49     74.2  1.9e-15
  8x4096x128  2^3          chain  raced        7957587    6286987    0.75     58.0  3.3e-15
   8x8192x16  2^3          chain  raced         768225     681562    0.81    136.5  1.3e-15
   8x8192x32  2^3          chain  raced        3080175    2022993    0.58     71.5  2.0e-15
   8x8192x64  2^3          chain  raced        8687813    6085399    0.69     53.1  3.8e-15
  16x16x8192  2^4          chain  raced        1254413    1365106    1.08    175.5  1.5e-15
  16x32x4096  2^4          chain  raced        1263037    1213550    0.86    174.3  1.7e-15
  16x32x8192  2^4          chain  raced        6393363    4801206    0.73     72.2  3.4e-15
  16x64x4096  2^4          chain  raced        6793275    4801106    0.62     67.9  3.4e-15
  16x128x512  2^4          chain  raced         443225     592600    1.26    236.6  1.1e-15
 16x128x1024  2^4          chain  raced        2113000    1389225    0.60    104.2  1.5e-15
 16x128x2048  2^4          chain  raced        8371538    4498075    0.52     55.1  3.5e-15
  16x256x512  2^4          chain  raced        2267988    1671706    0.64     97.1  1.7e-15
 16x256x1024  2^4          chain  raced        8750000    5795712    0.60     52.7  3.5e-15
  16x512x256  2^4          chain  raced        2175200    1579956    0.70    101.2  1.7e-15
  16x512x512  2^4          chain  raced        6944175    6248250    0.87     66.4  3.2e-15
 16x1024x128  2^4          chain  raced        2021237    1591412    0.76    108.9  1.4e-15
 16x1024x256  2^4          chain  raced        6434175    6590700    1.00     71.7  3.6e-15
  16x2048x64  2^4          chain  raced        2138750    1575837    0.73    103.0  1.5e-15
 16x2048x128  2^4          chain  raced        6685413    6363469    0.91     69.0  3.6e-15
  16x4096x32  2^4          chain  raced        2173237    1877306    0.80    101.3  1.5e-15
  16x4096x64  2^4          chain  raced        7512788    6212643    0.77     61.4  3.2e-15
  16x8192x16  2^4          chain  raced        2566675    1830306    0.68     85.8  1.4e-15
  16x8192x32  2^4          chain  raced        7479100    6203325    0.82     61.7  3.4e-15
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                           76     54     74   0.50   0.73   0.91    0.69
 ALL                             76     54     74   0.50   0.73   0.91    0.69
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 pow2 column                     76     54     74   0.50   0.73   0.91    0.69
 ALL                             76     54     74   0.50   0.73   0.91    0.69
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  76     54     74   0.50   0.73   0.91    0.69
 ALL                             76     54     74   0.50   0.73   0.91    0.69
```


worst 10: 8x256x1024 (chain 0.34), 8x64x4096 (chain 0.37), 8x32x8192 (chain 0.40), 8x128x2048 (chain 0.41), 8x512x512 (chain 0.46), 4x128x4096 (chain 0.47), 8x4096x64 (chain 0.49), 8x512x256 (chain 0.50), 4x256x2048 (chain 0.51), 16x128x2048 (chain 0.52)
best 5: 16x128x512 (chain 1.26), 16x16x8192 (chain 1.08), 16x1024x256 (chain 1.00), 8x128x1024 (chain 0.95), 2x1024x2048 (chain 0.94)
