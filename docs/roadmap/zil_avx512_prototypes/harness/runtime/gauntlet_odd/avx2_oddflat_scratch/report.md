# gauntlet report

run: `avx2_oddflat_scratch`  contract file suffix: `(oop, T=1)`  cells: 7 listed, 7 benched, comparator: none (absolute numbers)

control cell: 4 readings, 0.000..0.000 (a run is internally comparable when the first and the last agree)


## every cell

```
         N  factors          route    served       ours ns     cmp ns       x   rt err  note
      1125  3^2.5^3          chain3   raced           2969          -       -  1.1e-15  
      1215  3^5.5            chain3   raced           2662          -       -  1.5e-15  flips differ 1.29x
      1575  3^2.5^2.7        chain3   raced           3786          -       -  1.3e-15  
      2025  3^4.5^2          chain3   raced           5868          -       -  1.2e-15  
      2187  3^7              chain3   raced           6407          -       -  1.5e-15  
      2401  7^4              flat     raced           8028          -       -  1.5e-15  
      2835  3^4.5.7          chain3   raced           6608          -       -  1.4e-15  flips differ 1.31x
```


(no comparator: the ratio tables need MKL; see the ns and GFLOPS columns)
