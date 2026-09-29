# gauntlet report

run: `tail_p2odd_2026-09-29`  contract file suffix: `(oop, T=1)`  cells: 64 listed, 64 benched, comparator: MKL

control cell: 4 readings, 1.071..1.078 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
        48  2^4.3            2p       replayed          27         44    1.63  2.7e-16  
        80  2^4.5            2p       replayed          45         70    1.55  3.5e-16  
        96  2^5.3            2p       replayed          51         97    1.89  3.0e-16  
       112  2^4.7            2p       replayed          71         94    1.33  2.9e-16  
       144  2^4.3^2          chain3   replayed         106        143    1.34  3.6e-16  
       160  2^5.5            2p       replayed          91        164    1.80  3.0e-16  
       192  2^6.3            2p       replayed         110        183    1.65  3.4e-16  
       224  2^5.7            2p       replayed         146        226    1.42  2.6e-16  
       240  2^4.3.5          2p       replayed         169        250    1.48  4.7e-16  
       288  2^5.3^2          2p       replayed         198        338    1.69  3.8e-16  
       320  2^6.5            2p       replayed         205        323    1.35  3.0e-16  
       384  2^7.3            2p       replayed         251        437    1.74  3.0e-16  
       400  2^4.5^2          2p       replayed         308        440    1.41  3.6e-16  
       432  2^4.3^3          2p       replayed         332        513    1.27  5.1e-16  
       448  2^6.7            chain3   replayed         350        444    1.25  2.2e-16  
       480  2^5.3.5          2p       replayed         387        584    1.42  4.1e-16  
       576  2^6.3^2          2p       replayed         499        713    1.43  3.5e-16  
       640  2^7.5            2p       replayed         501        823    1.63  4.2e-16  
       720  2^4.3^2.5        chain3   replayed         809        940    1.14  4.4e-16  
       768  2^8.3            2p       replayed         631        905    1.43  4.3e-16  
       800  2^5.5^2          2p       replayed         787       1096    1.07  3.1e-16  flips differ 1.31x
       864  2^5.3^3          2p       replayed         828       1199    1.26  5.8e-16  
       896  2^7.7            chain3   replayed         922       1155    1.21  4.0e-16  
       960  2^6.3.5          2p       replayed         937       1278    1.36  4.4e-16  
      1152  2^7.3^2          chain3   replayed        1327       1675    1.26  4.9e-16  
      1280  2^8.5            chain3   replayed        1477       1736    1.15  4.3e-16  
      1440  2^5.3^2.5        chain3   replayed        1549       2249    1.24  4.8e-16  
      1536  2^9.3            chain3   replayed        1620       2402    1.48  4.5e-16  
      1600  2^6.5^2          chain3   replayed        1723       2326    1.35  3.4e-16  
      1728  2^6.3^3          chain3   replayed        1865       2758    1.46  5.6e-16  
      1792  2^8.7            chain3   replayed        2068       2442    0.92  2.9e-16  flips differ 1.28x
      1920  2^7.3.5          chain3   replayed        2109       3189    1.50  4.7e-16  
      2304  2^8.3^2          ztt      replayed        2133       3784    1.75  4.5e-16  
      2560  2^9.5            ztt      replayed        2184       4462    2.04  3.6e-16  
      2880  2^6.3^2.5        ztt      replayed        3150       4950    1.57  4.3e-16  
      3072  2^10.3           ztt      replayed        2723       5231    1.92  4.0e-16  
      3200  2^7.5^2          ztt      replayed        3049       6099    1.72  3.4e-16  
      3456  2^7.3^3          ztt      replayed        3606       6487    1.77  4.5e-16  
      3584  2^9.7            ztt      replayed        3493       6214    1.77  3.2e-16  
      3840  2^8.3.5          ztt      replayed        4020       6739    1.56  4.1e-16  
      4608  2^9.3^2          ztt      raced           4581       8587    1.84  4.1e-16  
      5120  2^10.5           ztt      raced           5033       9482    1.88  3.6e-16  
      5760  2^7.3^2.5        ztt      raced           6782      11467    1.63  4.8e-16  
      6144  2^11.3           ztt      raced           6007      11854    1.96  3.1e-16  
      6400  2^8.5^2          ztt      raced           6489      12068    1.86  4.8e-16  
      6912  2^8.3^3          ztt      raced           7715      13515    1.72  4.5e-16  
      7168  2^10.7           ztt      raced           7563      13356    1.76  3.3e-16  
      7680  2^9.3.5          ztt      raced           9055      15287    1.69  4.2e-16  
      9216  2^10.3^2         ztt      raced          10335      18188    1.44  3.6e-16  
     10240  2^11.5           ztt      raced          10850      21202    1.95  3.9e-16  
     11520  2^8.3^2.5        ztt      raced          14228      23943    1.68  4.7e-16  
     12800  2^9.5^2          ztt      raced          13742      27786    1.75  4.5e-16  
     13824  2^9.3^3          ztt      raced          16368      30373    1.84  4.0e-16  
     14336  2^11.7           ztt      raced          16088      30549    1.88  4.0e-16  
     15360  2^10.3.5         ztt      raced          19293      31908    1.61  3.7e-16  
     18432  2^11.3^2         ztt      raced          21532      40243    1.85  3.8e-16  
     23040  2^9.3^2.5        ztt      raced          32516      52912    1.59  4.6e-16  
     25600  2^10.5^2         ztt      raced          30955      55612    1.78  4.6e-16  
     27648  2^10.3^3         ztt      raced          36918      62534    1.69  4.4e-16  
     30720  2^11.3.5         ztt      raced          39828      70172    1.75  4.5e-16  
     46080  2^10.3^2.5       ztt      raced          69412     112136    1.61  5.0e-16  
     51200  2^11.5^2         ztt      raced          67659     131290    1.91  3.9e-16  
     55296  2^11.3^3         ztt      raced          80183     145886    1.80  4.5e-16  
     92160  2^11.3^2.5       ztt      raced         169929     327521    1.84  4.0e-16  
```


## by route (worse of the two flips)
```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 ztt         32      0      0   1.59   1.77   1.92    1.76
 2p          20      0      0   1.27   1.43   1.80    1.48
 chain3      12      0      1   1.14   1.26   1.48    1.26
 ALL         64      0      1   1.25   1.63   1.88    1.56
```


## by size
```
 band               cells median   <1.0   <0.8
 32..127                4   1.59      0      0
 128..511              12   1.42      0      0
 512..2047             16   1.30      1      0
 2048..8191            16   1.77      0      0
 8192..32767           12   1.75      0      0
 32768..92160           4   1.82      0      0
```


## by family
```
 family                                       cells median   <1.0  gmean
 ztt                                             32   1.77      0   1.76
 2p                                              20   1.43      0   1.48
 chain3                                          12   1.26      1   1.26
```


flip agreement: our two readings more than 25% apart at 2 of 64 cells.

worst 10: 1792 (chain3 0.92), 800 (2p 1.07), 720 (chain3 1.14), 1280 (chain3 1.15), 896 (chain3 1.21), 1440 (chain3 1.24), 448 (chain3 1.25), 1152 (chain3 1.26), 864 (2p 1.26), 432 (2p 1.27)
best 5: 2560 (ztt 2.04), 6144 (ztt 1.96), 10240 (ztt 1.95), 3072 (ztt 1.92), 51200 (ztt 1.91)
