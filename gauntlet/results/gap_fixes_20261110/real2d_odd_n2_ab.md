# 2D r2c at an odd N2 (bench --2drealnat --realfwd): the tree before the fixes against the fixed tree, 9 rounds

| cell | plan | old ns | new ns | new/old median | min..max | MKL/ours old | MKL/ours new |
|---|---|---:|---:|---:|---|---:|---:|
| r2c 16x61 | z:r2c2d nat-oop | 1657 | 1653 | 1.004 | 0.976..1.099 | 5.747 | 5.753 |
| r2c 24x31 | z:r2c2d nat-oop | 1093 | 1098 | 1.005 | 0.987..1.016 | 2.705 | 2.690 |
| r2c 128x225 | z:r2c2d nat-oop | 27462 | 27654 | 1.008 | 0.844..1.070 | 1.761 | 1.748 |
