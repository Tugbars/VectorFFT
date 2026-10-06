# gauntlet report (2D)

run: `real300_2d_r2c_fftw`  contract: 2D c2c interleaved, natural, out of place, K=1_r2c_fftw  cells: 292 listed, 196 benched, comparator: FFTW 2D (out of place, MEASURE)

control cell 64x64: 8 readings, 1.366..1.449


## every shape

```
       shape  N1 factors   route  served       ours ns     cmp ns       x   GFLOPS   rt err
         8x8  2^3          chain  raced             31         36    1.14     30.7  1.8e-16
        8x16  2^3          chain  raced             48         77    1.57     46.2  1.3e-16
         9x9  3^2          mono   raced              -          -       -        - not benched
        9x64  3^2          chain  raced            347        467    1.34     38.1  4.7e-16
       12x12  2^2.3        chain  raced             82         83    1.01     31.4  4.4e-16
       12x48  2^2.3        chain  raced            335        436    1.29     39.4  3.5e-16
       14x14  2.7          chain  raced            175        146    0.83     21.3  2.5e-16
       15x15  3.5          2p     raced              -          -       -        - not benched
       15x45  3.5          2p     raced              -          -       -        - not benched
       15x64  3.5          chain  raced            551        682    1.24     43.2  3.6e-16
     15x4096  3.5          chain  raced          53223      49790    0.92     45.9  4.4e-16
        16x8  2^4          chain  raced             78         80    1.03     28.9  2.0e-16
       16x16  2^4          chain  raced            114        155    0.86     44.8  2.1e-16
       16x32  2^4          chain  raced            255        353    1.36     45.2  2.5e-16
       16x64  2^4          chain  raced            567        719    1.06     45.2  2.9e-16
      16x256  2^4          chain  raced           2354       2369    1.00     52.2  2.6e-16
     16x1024  2^4          chain  raced          12135      11053    0.91     47.3  2.8e-16
     16x4096  2^4          chain  raced          57054      52531    0.92     45.9  3.8e-16
       18x18  2.3^2        chain  raced            413        504    1.20     16.4  4.0e-16
       21x21  3.7          2p     raced              -          -       -        - not benched
      21x128  3.7          chain  raced           1893       3169    1.66     40.4  4.8e-16
       24x24  2^3.3        chain  raced            484        779    1.60     27.3  4.7e-16
       24x96  2^3.3        chain  raced           1656       2536    1.49     38.8  4.9e-16
       25x25  5^2          2p     raced              -          -       -        - not benched
       27x27  3^3          2p     raced              -          -       -        - not benched
       27x32  3^3          chain  raced            535       1049    1.95     39.4  4.0e-16
       27x81  3^3          2p     raced              -          -       -        - not benched
       28x28  2^2.7        chain  raced            824        982    1.19     22.9  3.4e-16
      28x112  2^2.7        chain  raced           2686       3381    1.25     33.9  3.2e-16
       32x16  2^5          chain  raced            234        319    1.31     49.2  3.2e-16
       32x27  2^5          2p     raced              -          -       -        - not benched
       32x32  2^5          chain  raced            514        729    1.40     49.8  3.2e-16
       32x64  2^5          chain  raced           1190       1725    1.40     47.3  2.8e-16
      32x128  2^5          chain  raced           2512       2626    1.03     48.9  2.7e-16
      32x512  2^5          chain  raced          11520      11178    0.97     49.8  3.4e-16
     32x2048  2^5          chain  raced          51482      50631    0.98     50.9  5.1e-16
       33x33  3.11         2p     raced              -          -       -        - not benched
       36x36  2^2.3^2      chain  raced           1029       1542    1.47     32.6  3.5e-16
      36x144  2^2.3^2      chain  raced           4508       5451    1.21     35.5  5.3e-16
       45x15  3^2.5        2p     raced              -          -       -        - not benched
       45x45  3^2.5        2p     raced              -          -       -        - not benched
       45x64  3^2.5        chain  raced           2183       3747    1.71     37.9  5.0e-16
     45x2048  3^2.5        chain  raced          89981     115669    1.28     42.2  5.2e-16
       48x12  2^4.3        chain  raced            307        516    1.67     43.0  2.8e-16
       48x48  2^4.3        chain  raced           1663       2467    1.48     38.7  3.9e-16
       50x50  2.5^2        chain  raced           2329       2977    1.26     30.3  4.1e-16
       54x54  2.3^3        chain  raced           2729       3469    1.27     30.8  6.5e-16
       56x56  2^3.7        chain  raced           2928       3228    1.10     31.1  3.4e-16
       60x60  2^2.3.5      chain  raced           3158       3822    1.21     33.7  4.8e-16
      60x240  2^2.3.5      chain  raced          12930      14943    1.15     38.5  6.0e-16
       63x63  3^2.7        2p     raced              -          -       -        - not benched
       63x64  3^2.7        chain  raced           3298       5258    1.58     36.6  4.6e-16
        64x9  2^6          mono   raced              -          -       -        - not benched
       64x15  2^6          2p     raced              -          -       -        - not benched
       64x32  2^6          chain  raced           1117       1492    1.33     50.4  3.4e-16
       64x45  2^6          2p     raced              -          -       -        - not benched
       64x63  2^6          2p     raced              -          -       -        - not benched
       64x64  2^6          chain  raced           2562       3606    1.36     48.0  3.0e-16
      64x128  2^6          chain  raced           5278       5758    1.08     50.4  3.9e-16
      64x256  2^6          chain  raced          11061      11706    1.04     51.8  3.4e-16
     64x1024  2^6          chain  raced          49920      50035    1.00     52.5  4.1e-16
     64x4096  2^6          chain  raced         264413     258483    0.98     44.6  4.0e-16
       72x72  2^3.3^2      chain  raced           4606       5426    1.17     34.7  4.4e-16
       75x75  3.5^2        2p     raced              -          -       -        - not benched
       81x27  3^4          2p     raced              -          -       -        - not benched
       81x81  3^4          2p     raced              -          -       -        - not benched
      81x128  3^4          chain  raced           8643      12166    1.40     40.0  4.7e-16
      81x512  3^4          chain  raced          40352      50459    1.25     39.4  6.0e-16
       96x24  2^5.3        chain  raced           2264       2966    1.30     28.4  5.2e-16
       96x96  2^5.3        chain  raced           7910       9511    1.20     38.4  5.2e-16
      96x160  2^5.3        chain  raced          12628      15413    1.21     42.3  4.6e-16
       97x99  97           2p     raced              -          -       -        - not benched
       99x97  3^2.11       prime  raced              -          -       -        - not benched
       99x99  3^2.11       2p     raced              -          -       -        - not benched
     100x100  2^2.5^2      chain  raced          10037      10473    1.04     33.1  4.9e-16
     105x105  3.5.7        2p     raced              -          -       -        - not benched
     105x256  3.5.7        chain  raced          23824      30141    1.25     41.5  5.1e-16
      112x28  2^4.7        chain  raced           3922       3790    0.96     23.2  3.3e-16
     112x112  2^4.7        chain  raced          11871      12893    1.08     36.0  3.5e-16
     120x120  2^3.3.5      chain  raced          13338      16797    1.25     37.3  5.4e-16
     120x480  2^3.3.5      chain  raced          59141      62553    1.05     38.5  4.9e-16
     125x125  5^3          2p     raced              -          -       -        - not benched
     125x625  5^3          2p     raced              -          -       -        - not benched
     127x127  127          prime  raced              -          -       -        - not benched
     127x256  127          blu    raced          72774      96060    1.31     16.7  5.4e-16
     127x405  127          chain3 raced              -          -       -        - not benched
      128x21  2^7          2p     raced              -          -       -        - not benched
      128x64  2^7          chain  raced           5732       7239    1.25     46.4  4.3e-16
      128x81  2^7          2p     raced              -          -       -        - not benched
     128x128  2^7          chain  raced          11599      12258    1.05     49.4  3.6e-16
     128x189  2^7          2p     raced              -          -       -        - not benched
     128x243  2^7          2p     raced              -          -       -        - not benched
     128x256  2^7          chain  raced          24654      24341    0.98     49.8  3.4e-16
     128x512  2^7          chain  raced          51628      50920    0.98     50.8  4.5e-16
     135x135  3^3.5        2p     raced              -          -       -        - not benched
     135x256  3^3.5        chain  raced          31973      40065    1.24     40.7  5.0e-16
     135x405  3^3.5        chain3 raced              -          -       -        - not benched
    135x2048  3^3.5        chain  raced         325693     403846    1.17     38.4  5.5e-16
      144x36  2^4.3^2      chain  raced           5427       6422    1.18     29.5  6.5e-16
     144x144  2^4.3^2      chain  raced          18515      23805    1.28     40.2  4.7e-16
      160x96  2^5.5        chain  raced          13529      16599    1.22     39.5  4.2e-16
     162x162  2.3^4        chain  raced          26299      36977    1.40     36.6  6.7e-16
     165x165  3.5.11       2p     raced              -          -       -        - not benched
     189x128  3^3.7        chain  raced          22273      32039    1.41     39.5  5.3e-16
     189x189  3^3.7        2p     raced              -          -       -        - not benched
     192x192  2^6.3        chain  raced          33282      38242    1.09     42.0  5.4e-16
     192x320  2^6.3        chain  raced          55940      69139    1.22     43.7  4.6e-16
     224x224  2^5.7        chain  raced          49994      55042    1.09     39.2  4.5e-16
     224x896  2^5.7        chain  raced         224895     260315    1.13     39.3  5.2e-16
     225x225  3^2.5^2      chain3 raced              -          -       -        - not benched
     225x256  3^2.5^2      chain  raced          53901      73087    1.32     42.2  4.5e-16
      240x60  2^4.3.5      chain  raced          14475      16269    1.10     34.4  5.9e-16
     240x240  2^4.3.5      chain  raced          56229      65022    1.15     40.5  5.0e-16
     243x128  3^5          chain  raced          30028      41296    1.38     38.6  5.2e-16
     243x243  3^5          2p     raced              -          -       -        - not benched
     243x729  3^5          2p     raced              -          -       -        - not benched
    243x1024  3^5          chain  raced         315488     395600    1.21     35.3  5.7e-16
     250x250  2.5^3        chain  raced          65924      86961    1.31     37.8  5.2e-16
     251x251  251          prime  raced              -          -       -        - not benched
     251x512  251          blu    raced         371223     448238    1.21     14.7  5.4e-16
       256x8  2^8          chain  raced           1733       1676    0.95     32.5  2.6e-16
     256x105  2^8          2p     raced              -          -       -        - not benched
     256x127  2^8          prime  raced              -          -       -        - not benched
     256x128  2^8          chain  raced          28498      30041    1.04     43.1  3.2e-16
     256x135  2^8          chain3 raced              -          -       -        - not benched
     256x225  2^8          chain3 raced              -          -       -        - not benched
     256x256  2^8          chain  raced          61089      69307    1.13     42.9  4.1e-16
     256x375  2^8          chain3 raced              -          -       -        - not benched
     256x512  2^8          chain  raced         146857     158168    1.05     37.9  4.1e-16
    256x1024  2^8          chain  raced         315793     325183    1.00     37.4  4.0e-16
     288x288  2^5.3^2      chain  raced          88529     103205    1.16     38.3  5.8e-16
     315x315  3^2.5.7      chain3 raced              -          -       -        - not benched
     315x512  3^2.5.7      chain  raced         201888     242406    1.19     34.5  5.1e-16
     320x192  2^6.5        chain  raced          56180      67296    1.16     43.5  4.6e-16
     360x360  2^3.3^2.5    chain  raced         153987     184005    1.18     35.7  5.5e-16
    360x1440  2^3.3^2.5    chain  raced         751700     800968    1.06     32.7  6.7e-16
     375x256  3.5^3        chain  raced         101851     120435    1.18     39.0  4.7e-16
     375x375  3.5^3        chain3 raced              -          -       -        - not benched
    375x1125  3.5^3        flat   raced              -          -       -        - not benched
     384x384  2^7.3        chain  raced         162230     182485    1.11     39.0  4.3e-16
     384x640  2^7.3        chain  raced         314494     320206    1.02     35.0  5.3e-16
     405x127  3^4.5        prime  raced              -          -       -        - not benched
     405x135  3^4.5        2p     raced              -          -       -        - not benched
     405x405  3^4.5        chain3 raced              -          -       -        - not benched
     405x512  3^4.5        chain  raced         264232     325439    1.19     34.7  5.0e-16
     432x432  2^4.3^3      chain  raced         239338     283862    1.18     34.1  5.7e-16
     448x448  2^6.7        chain  raced         249395     268218    1.04     35.4  5.8e-16
     480x120  2^5.3.5      chain  raced          55161      74917    1.35     41.3  5.4e-16
     480x480  2^5.3.5      chain  raced         293535     323805    1.05     35.0  5.2e-16
     486x486  2.3^5        chain  raced         321812     432965    1.31     32.8  7.5e-16
     495x495  3^2.5.11     chain3 raced              -          -       -        - not benched
     495x512  3^2.5.11     chain  raced         331953     446913    1.31     34.3  5.5e-16
     500x500  2^2.5^3      chain  raced         374200     401426    1.06     29.9  5.9e-16
      509x64  509          blu    raced          85325     117475    1.37     14.3  6.6e-16
     509x509  509          prime  raced              -          -       -        - not benched
    509x1024  509          blu    raced        1853850    1986581    1.05     13.3  6.5e-16
       512x8  2^9          chain  raced           3875       4130    1.06     31.7  2.9e-16
      512x16  2^9          chain  raced           6909       8366    1.21     38.5  3.0e-16
      512x81  2^9          2p     raced              -          -       -        - not benched
     512x251  2^9          prime  raced              -          -       -        - not benched
     512x256  2^9          chain  raced         143120     147115    1.02     38.9  3.9e-16
     512x315  2^9          chain3 raced              -          -       -        - not benched
     512x405  2^9          chain3 raced              -          -       -        - not benched
     512x495  2^9          chain3 raced              -          -       -        - not benched
     512x512  2^9          chain  raced         346860     352673    1.00     34.0  5.1e-16
     512x625  2^9          flat   raced              -          -       -        - not benched
     512x945  2^9          flat   raced              -          -       -        - not benched
    512x1024  2^9          chain  raced         658438     745593    1.12     37.8  4.3e-16
    512x2048  2^9          chain  raced        1366925    1647387    1.11     38.4  4.6e-16
     567x567  3^4.7        2p     raced              -          -       -        - not benched
    567x1024  3^4.7        chain  raced         729100    1023074    1.29     38.1  4.9e-16
     576x576  2^6.3^2      chain  raced         402733     498416    1.16     37.8  5.6e-16
     625x125  5^4          2p     raced              -          -       -        - not benched
     625x512  5^4          chain  raced         427067     552712    1.18     34.3  5.0e-16
     625x625  5^4          flat   raced              -          -       -        - not benched
     640x384  2^7.5        chain  raced         299575     341734    1.11     36.7  4.6e-16
     675x675  3^3.5^2      2p     raced              -          -       -        - not benched
     720x720  2^4.3^2.5    chain  raced         742262     822581    1.10     33.1  6.1e-16
    720x2880  2^4.3^2.5    chain  raced        3743375    4653837    1.20     29.1  6.2e-16
      729x16  3^6          chain  raced          12037      20105    1.65     32.7  5.5e-16
     729x243  3^6          2p     raced              -          -       -        - not benched
     729x729  3^6          2p     raced              -          -       -        - not benched
    729x1024  3^6          chain  raced         964837    1391975    1.41     37.7  6.7e-16
     768x768  2^8.3        chain  raced         792738     905675    1.08     35.7  5.0e-16
    768x1280  2^8.3        chain  raced        1366525    1468356    1.04     35.8  5.1e-16
     896x224  2^7.7        chain  raced         245068     291381    1.14     36.1  4.5e-16
     896x896  2^7.7        chain  raced        1197637    1236856    0.96     32.9  5.1e-16
     945x512  3^3.5.7      chain  raced         719150     858093    1.18     31.8  5.6e-16
     945x945  3^3.5.7      flat   raced              -          -       -        - not benched
     960x960  2^6.3.5      chain  raced        1296975    1596719    1.15     35.2  5.6e-16
     1021x16  1021         blu    raced          53331      68445    1.28     10.7  5.7e-16
   1021x1021  1021         prime  raced              -          -       -        - not benched
   1021x1024  1021         blu    raced        4626525    4620806    0.94     11.3  6.6e-16
      1024x8  2^10         chain  raced           8519      12640    1.46     31.3  2.8e-16
     1024x16  2^10         chain  raced          16533      22146    1.33     34.7  3.4e-16
     1024x32  2^10         chain  raced          35591      44291    1.22     34.5  3.9e-16
     1024x63  2^10         2p     raced              -          -       -        - not benched
    1024x243  2^10         chain3 raced              -          -       -        - not benched
    1024x509  2^10         prime  raced              -          -       -        - not benched
    1024x512  2^10         chain  raced         712763     753737    1.03     34.9  3.5e-16
    1024x567  2^10         2p     raced              -          -       -        - not benched
    1024x729  2^10         2p     raced              -          -       -        - not benched
   1024x1021  2^10         prime  raced              -          -       -        - not benched
   1024x1024  2^10         chain  raced        1398875    1553156    1.10     37.5  4.5e-16
   1024x1125  2^10         flat   raced              -          -       -        - not benched
   1024x1215  2^10         flat   raced              -          -       -        - not benched
   1024x2025  2^10         chain3 raced              -          -       -        - not benched
   1024x2048  2^10         chain  raced        3636175    4135081    1.06     30.3  4.5e-16
   1024x4093  2^10         prime  raced              -          -       -        - not benched
    1125x375  3^2.5^3      chain3 raced              -          -       -        - not benched
   1125x1024  3^2.5^3      chain  raced        1737275    2189656    1.16     33.4  5.5e-16
   1125x1125  3^2.5^3      flat   raced              -          -       -        - not benched
   1152x1152  2^7.3^2      chain  raced        2158850    2473018    1.12     31.3  5.5e-16
     1200x16  2^4.3.5^2    chain  raced          20071      28038    1.38     34.0  4.0e-16
   1200x1200  2^4.3.5^2    chain  raced        2522200    2986288    1.04     29.2  5.8e-16
     1215x64  3^5.5        chain  raced          84731     135371    1.59     37.3  5.9e-16
   1215x1024  3^5.5        chain  raced        1822888    2392287    1.29     34.5  5.8e-16
   1215x1215  3^5.5        chain3 raced              -          -       -        - not benched
   1250x1250  2.5^4        chain  raced        2942888    3412168    1.16     27.3  7.5e-16
    1280x768  2^8.5        chain  raced        1376663    1734350    1.23     35.5  5.2e-16
     1440x64  2^5.3^2.5    chain  raced          97809     137828    1.33     38.8  4.5e-16
    1440x360  2^5.3^2.5    chain  raced         736037     882625    1.18     33.4  6.6e-16
   1440x1440  2^5.3^2.5    chain  raced        4035050    4452393    1.08     27.0  7.9e-16
   1458x1458  2.3^6        chain  raced        4863163    6007781    1.21     23.0  8.1e-16
   1536x1536  2^9.3        chain  raced        4647625    4796125    1.01     26.9  5.7e-16
   1536x2560  2^9.3        chain  raced        9045288    9784650    1.07     23.8  5.1e-16
   1701x1701  3^5.7        chain3 raced              -          -       -        - not benched
   1701x2048  3^5.7        chain  raced        7405112   11082369    1.49     25.6  7.4e-16
   1792x1792  2^8.7        chain  raced        6948050    7764312    1.11     25.0  5.5e-16
   1800x1800  2^3.3^2.5^2  chain  raced        7343612    8880606    1.21     23.9  8.3e-16
   1920x1920  2^7.3.5      chain  raced        8357262    9274600    1.10     24.1  5.0e-16
     2025x32  3^4.5^2      chain  raced          83426     113782    1.35     31.0  5.0e-16
   2025x1024  3^4.5^2      chain  raced        3739162    5684356    1.48     29.1  6.6e-16
   2025x2025  3^4.5^2      chain3 raced              -          -       -        - not benched
     2039x32  2039         blu    raced         220457     295363    1.34     11.8  5.9e-16
   2039x2039  2039         prime  raced              -          -       -        - not benched
   2039x2048  2039         blu    raced       30028675   23836318    0.79      7.6  5.9e-16
      2048x8  2^11         chain  raced          20622      26473    1.28     27.8  3.3e-16
     2048x16  2^11         chain  raced          33753      46775    1.36     36.4  3.7e-16
     2048x32  2^11         chain  raced          79162      91538    1.15     33.1  3.8e-16
     2048x45  2^11         2p     raced              -          -       -        - not benched
     2048x64  2^11         chain  raced         153457     198300    1.28     36.3  4.2e-16
    2048x135  2^11         2p     raced              -          -       -        - not benched
   2048x1024  2^11         chain  raced        4352512    4437456    1.02     25.3  4.3e-16
   2048x1701  2^11         flat   raced              -          -       -        - not benched
   2048x2039  2^11         prime  raced              -          -       -        - not benched
   2048x2048  2^11         chain  raced        9412025   11141806    1.17     24.5  5.6e-16
   2048x3375  2^11         chain3 raced              -          -       -        - not benched
   2048x4096  2^11         chain  raced       19651912   22811437    1.16     24.5  7.5e-16
     2187x16  3^7          chain  raced          39965      65075    1.63     33.0  5.9e-16
   2187x2187  3^7          flat   raced              -          -       -        - not benched
   2187x4096  3^7          chain  raced       22132050   38238268    1.71     23.4  8.1e-16
   2400x2400  2^5.3.5^2    chain  raced       14230725   18565593    1.30     22.7  7.2e-16
   2500x2500  2^2.5^4      chain  raced       19382000   24934312    1.27     18.2  7.4e-16
   2560x1536  2^9.5        chain  raced        8975687   11798643    1.30     24.0  5.5e-16
   2835x2835  3^4.5.7      chain3 raced              -          -       -        - not benched
    2880x720  2^6.3^2.5    chain  raced        3843925    5310887    1.36     28.3  7.1e-16
   3000x3000  2^3.3.5^3    chain  raced       23222712   31513350    1.35     22.4  7.0e-16
   3375x2048  3^3.5^3      chain  raced       16138163   24017469    1.47     24.3  7.2e-16
   3375x3375  3^3.5^3      chain3 raced              -          -       -        - not benched
     3600x16  2^4.3^2.5^2  chain  raced          63667      88630    1.39     35.8  5.4e-16
   3600x3600  2^4.3^2.5^2  chain  raced       34230450   43913462    1.28     22.4  6.7e-16
     3645x32  3^6.5        chain  raced         167547     221864    1.29     29.3  6.5e-16
     4093x16  4093         blu    raced         254839     320888    1.25     10.3  6.3e-16
   4093x1024  4093         blu    raced       31383312   26076231    0.83      7.3  7.5e-16
      4096x8  2^12         chain  raced          49055      56382    1.14     25.0  3.5e-16
     4096x15  2^12         2p     raced              -          -       -        - not benched
     4096x16  2^12         chain  raced          83305     106720    1.27     31.5  3.8e-16
     4096x32  2^12         chain  raced         192033     207715    1.06     29.0  4.0e-16
     4096x64  2^12         chain  raced         359107     464963    1.27     32.9  3.2e-16
    4096x128  2^12         chain  raced         720250     888694    1.23     34.6  5.0e-16
   4096x2048  2^12         chain  raced       20545750   24922887    1.21     23.5  4.5e-16
   4096x2187  2^12         chain3 raced              -          -       -        - not benched
   4096x4096  2^12         chain  raced       42555100   55573481    1.30     23.7  8.8e-16
     4800x64  2^6.3.5^2    chain  raced         390769     545684    1.39     35.8  5.3e-16
     6561x16  3^8          chain  raced         135389     224684    1.65     32.3  7.0e-16
     7200x32  2^5.3^2.5^2  chain  raced         327412     412841    1.26     31.3  5.7e-16
     8192x16  2^13         chain  raced         198643     222375    1.12     28.0  3.8e-16
     8192x21  2^13         2p     raced              -          -       -        - not benched
     8192x27  2^13         2p     raced              -          -       -        - not benched
     8192x32  2^13         chain  raced         423753     477663    1.11     27.8  4.1e-16
     8192x64  2^13         chain  raced         789338     968300    1.22     31.6  3.7e-16
    8192x128  2^13         chain  raced        1702788    1892769    1.10     30.8  4.0e-16
   8192x4096  2^13         chain  raced       91410400  124268600    1.34     22.9  1.2e-15
     16384x9  2^14         mono   raced              -          -       -        - not benched
    16384x16  2^14         chain  raced         455907     542073    1.17     25.9  4.0e-16
    16384x32  2^14         chain  raced         941100    1112431    1.17     26.5  4.3e-16
    16384x64  2^14         chain  raced        1642412    2342318    1.41     31.9  4.5e-16
   16384x128  2^14         chain  raced        4653937    5297318    1.13     23.7  4.5e-16
    32768x16  2^15         chain  raced         983137    1621919    1.58     25.3  4.5e-16
    32768x32  2^15         chain  raced        2050638    3183781    1.55     25.6  4.2e-16
    65536x16  2^16         chain  raced        2505550    4005068    1.55     20.9  4.7e-16
```


## by route (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 chain                          186      0     14   1.01   1.19   1.48    1.21
 blu                             10      1      3   0.83   1.23   1.37    1.12
 ALL                            196      1     17   1.00   1.20   1.47    1.20
```


## by column class (N1) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 even column                     81      0      3   1.04   1.18   1.38    1.19
 pow2 column                     68      0     10   0.98   1.13   1.40    1.15
 odd column                      37      0      1   1.18   1.32   1.66    1.36
 prime column                    10      1      3   0.83   1.23   1.37    1.12
 ALL                            196      1     17   1.00   1.20   1.47    1.20
```


## by size (points) (worse of the two flips)
```
                              cells   <0.8   <1.0    p10    med    p90   gmean
 > 65536 points                  97      1      5   1.02   1.17   1.41    1.18
 4097..65536                     61      0      7   0.98   1.22   1.39    1.20
 1025..4096                      20      0      3   1.00   1.29   1.66    1.27
 257..1024                       12      0      0   1.19   1.32   1.67    1.36
 <= 256 points                    6      0      2   0.83   1.02   1.57    1.05
 ALL                            196      1     17   1.00   1.20   1.47    1.20
```


worst 10: 2039x2048 (blu 0.79), 4093x1024 (blu 0.83), 14x14 (chain 0.83), 16x16 (chain 0.86), 16x1024 (chain 0.91), 16x4096 (chain 0.92), 15x4096 (chain 0.92), 1021x1024 (blu 0.94), 256x8 (chain 0.95), 896x896 (chain 0.96)
best 5: 27x32 (chain 1.95), 2187x4096 (chain 1.71), 45x64 (chain 1.71), 48x12 (chain 1.67), 21x128 (chain 1.66)
