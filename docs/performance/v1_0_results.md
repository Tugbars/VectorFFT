# VectorFFT v1.0 — performance results

## 1. vs MKL — 1D C2C

```
Category              Cells    Min   Median    Max   Mean
─────────────────────────────────────────────────────────
Small (N≤128)            15   2.60×   4.28× 15.33×  5.98×
Power-of-2               29   1.10×   1.86×  3.04×  1.96×
Composite                43   1.62×   2.85×  4.51×  2.93×
Odd composite            26   2.26×   3.47×  5.16×  3.36×
Mixed deep               35   1.66×   2.71×  5.78×  2.89×
Prime powers             25   1.67×   2.69×  4.16×  2.76×
Genfft (R=11/13)         17   1.65×   2.79×  3.75×  2.63×
Rader primes             24   1.29×   2.34×  3.85×  2.36×
Bluestein primes         24   1.02×   1.55×  3.52×  1.74×
─────────────────────────────────────────────────────────
OVERALL                 238   1.02×   2.64× 15.33×  2.83×

Wins vs MKL: 238/238 (100%)
```

### Arbitrary K — odd / non-multiple-of-VW batch (single-thread)

```
 N      plan (calibrated)    K=32 rem0   K=33 rem1   K=31 rem3   | CSV baked K=32
──────────────────────────────────────────────────────────────────────────────────
 64     8x8/DIT              3.43×       2.69×       2.65×       | 3.03×
 128    4x32/DIF             1.92×*      2.30×       (noisy)     | 3.26×
 256    4x8x8/DIT            1.31×       2.27×       (noisy)     | 3.04×
 512    4x4x32/DIF           2.09×       1.78×       2.46×       | 1.98×
 1024   4x4x8x8/DIT          1.19×       1.63×       1.64×       | 2.75×
 4096   4x4x4x8x8/DIT        2.70×*      1.48×       1.52×       | 2.57×
```

### Multi-threaded — vs MKL at T=8

```
 N      K    dag-T8 (ns)  MKL-T8 (ns)  dag/MKL
─────────────────────────────────────────────
 8      256         571       23,906   41.90×
 64     256       8,140       46,266    5.68×
 256    256      45,560      128,653    2.82×
 1024   256     224,288      694,963    3.10×
 4096   256     696,100    3,387,937    4.87×
 256    32        8,715       19,634    2.25×
 1024   32       50,375       66,834    1.33×
 4096   32      233,233      405,020    1.74×
```

### Multi-threaded — INTERLEAVED transform-contiguous batch at T=8

```
 N      K    ours-T8   ours-T1    MKL-T8    MKL-T1 | vs MKL best  our scale  MKL scale
────────────────────────────────────────────────────────────────────────────────────
 256    4        607       614     3,573       673 |     1.11×      1.01×      0.19×
 512    4      1,146     2,271     6,626     1,419 |     1.24×      1.98×      0.21×
 1024   4      1,667     4,036     8,378     3,415 |     2.05×      2.42×      0.41×
 4096   4      4,973    16,199    31,433    15,266 |     3.07×      3.26×      0.49×
 16384  4     20,013    80,390   114,647    89,490 |     4.47×      4.02×      0.78×
 65536  4    137,438   494,375   568,275   581,750 |     4.13×      3.60×      1.02×
 256    8      1,012     1,170    19,933     1,268 |     1.25×      1.16×      0.06×
 512    8      1,080     4,879    23,064     2,911 |     2.69×      4.52×      0.13×
 1024   8      2,457     8,339    25,125     6,713 |     2.73×      3.39×      0.27×
 4096   8      5,138    33,352    60,036    31,823 |     6.19×      6.49×      0.53×
 16384  8     20,593   173,900   223,613   203,280 |     9.87×      8.44×      0.91×
 65536  8    113,550   919,550 1,127,175 1,042,487 |     9.18×      8.10×      0.92×
 256    32     1,673     4,704    25,034     4,991 |     2.98×      2.81×      0.20×
 512    32     2,758    17,056    32,020    11,190 |     4.06×      6.18×      0.35×
 1024   32     6,449    33,274    46,897    27,995 |     4.34×      5.16×      0.60×
 4096   32    18,240   153,947   188,880   170,467 |     9.35×      8.44×      0.90×
 16384  32    85,187   757,650   869,825   875,550 |    10.21×      8.89×      1.01×
 65536  32 1,182,062 4,126,425 7,977,025 8,091,300 |     6.75×      3.49×      1.01×
────────────────────────────────────────────────────────────────────────────────────
ns/call. "scale" = that engine's OWN T1/T8 (8.00 = perfect on 8 cores).
```

### Natural order — in-place (single-thread)

```
 N      K    mode     vfft ns   dag/MKL   chain / note
────────────────────────────────────────────────────────────────────────
 64     4    FREE         141     2.28×    radix-64 leaf — zero reorder
 100    4    PSWAP        295     2.72×    10·10  (odd-radix palindrome)
 128    4    PURE         508     1.27×    8·16   (cycle-follow reorder)
 250    4    PURE       1,079     1.84×    10·25
 256    4    PSWAP        899     1.35×    16·16  palindrome
 512    4    PSWAP      1,892     1.68×    8·8·8  palindrome
 1024   4    PSWAP      6,033     1.23×    4·64·4 injected — MKL's best-tuned cell (pow2)
────────────────────────────────────────────────────────────────────────
 128    32   PURE       2,751     2.09×    4·32
 512    32   PURE      15,554     1.61×    4·4·32
────────────────────────────────────────────────────────────────────────
 median               ~1.68×    (9/9 win)
```

### Order × placement — the K=1 INTERLEAVED grid

| order × placement | sub-2048 (mono ≤64 · il2p/il3p 128–1024) | ≥2048 (cascade tier) |
|---|---|---|
| **NATURAL · in-place** | native — `VFFT_NAT_ILP`: il2p/il3p aliased, raced vs convert at create, banked `@nat` | native — `VFFT_NAT_ZCASC`: `stfn` natural-terminator cascade, **no reorder pass**, raced vs tape, banked `@nat` |
| **NATURAL · OOP** | native — same IL engines, `z_in → z_out` | native — natord cascade via `@natoop` verdict + create race |
| **SCRAMBLED · in-place** | native — **identity rule**: served by the natural-native engines (the identity permutation is contract-legal; bits identical to natural, gated `IDENT`) | native — ZTURN-S digit-scrambled comb (kind-4 verdict); the comb is a REAL permutation (A3-gated) |
| **SCRAMBLED · OOP** | native — identity rule, same engines (`scr==nat` EXACT, gated) | native — kind-4 cascade attaches to the OOP handle, matched-permutation roundtrip |

```
  N       NATURAL in-place   NATURAL OOP   SCRAMBLED in-place
──────────────────────────────────────────────────────────────
  128        0.91            1.05         (= NAT bits)
  256       0.85–0.86       1.00         (= NAT bits)
  512       0.78–0.80      0.98–1.00      (= NAT bits)
  1024      0.91–0.95      ~0.95–parity     (= NAT bits)
  2048      1.09–1.16       0.99–1.11         1.15–1.18
  4096      0.96–0.99       0.91–0.94         1.02–1.04
  8192      1.00–1.03       0.95–0.98         1.05–1.06
  16384     1.02–1.03       0.94–0.98         1.05–1.08
  32768     0.94–0.97       0.88–0.91         1.00–1.02
──────────────────────────────────────────────────────────────
register-file rule (a monolithic R≥32 body holds ~40–64 live values against
AVX2's 16 registers), not a per-cell race. Same-run A/B against the
monolithic arm, pair pinned by wisdom so only the kernels vary, 8 alternating
arms on a pinned core: 1024 −27% (0.65→0.92), 512 −8% (0.70→0.78).
the heuristic balanced pair (16,16) with the measured (8,32), which puts
R=32 in the leaf slot, which the structural rule then blocks — 0.76→0.85.
Fully blocking a (16,16) pair instead was raced and LOST by 4.4%, so radix
choice dominates form choice here.
([tangent_scaled_butterflies.md](tangent_scaled_butterflies.md)) + the wing32
R32 forms + the TURNED store-edge axis
([store_edge_taxonomy.md](store_edge_taxonomy.md) defines T128/T256/M-128 and
the full set), all dp-raced (`calibrate_k1` over the
full form pool; verdicts banked as pair + `il_kv`). Winning rows: 128 = pair
4×32, kv 64 (mono mid + T256 wing32 leaf, 65 ns vs MKL 69); 256 = 16×16, kv 51
(tangent both slots, 136 vs 136); 512 = 16×32, kv 67 (tangent mid + T256 wing32
leaf, 296–297 vs 291–295). Canonical bench `--k1noop`, both flip orders,
cross-engine correctness 3–4e-16. The 2026-08-12 era below (-history) first
crossed 128/256 with tangent-only forms; the wing32 leaf then solved the R32
slot (the old "+32% killed leaf" was an edge×interior interaction, not the
tangent construction) and the TURNED axis picked the store edge per cell.
Historical  numbers: 128 1.04 (8×16, kv 51) · 256 1.01–1.02 · 512 0.87–0.88
(kv 35, pre-wing32) — superseded by the rows above.
848/889/893/901/978/987 ns vs MKL 833/841/854/860/864/873 — **median ratio
0.96, best-vs-best 0.98, one rep at 1.02**. The route is CLASSIC blocked 32×32
**re-raced against the complete form pool** (tangent, wing32, both TURNED
edges) and re-confirmed by the dp with the nearest challenger +4.5% behind:
1024's regime is memory (L1 pending misses ~19× the 512 level, store-latency
bound), where the tangent family's port/instruction levers buy nothing. The
cell's variance is dominated by MKL's own in-place shadow-plane placement
(its floor alone spans 833–873 here; historically 812–905), so the honest
datum is **~0.95-to-parity** — do not quote a third digit. (The 2026-08-12
figure 0.83–0.90 came from a noisier 5-rep set on the same route.)
the in-place figure no longer describes what ships. Sub-2048 in-place and OOP
run the SAME IL engines (see the grid above), so it is expected to track the
OOP column — but it has not been re-measured, and is not quoted as if it had.
```

  | | MKL col32 | ours `n1tb48` |
  |---|--:|--:|
  | instructions | **460** | 563 |
  | fma / bare mul | **68 / 0** | 36 / 20 |
  | naked add+sub | 118 (**63%**) | 152 (73%) |
  | shuffle + xor | **54 + 15** | 82 + 29 |
  | stack ops | **78 (0.42/fp-op)** | 48 (0.23/fp-op) |

### Out-of-place — vs MKL (single-thread)

```
 N       K     kind     plan          dag/MKL
──────────────────────────────────────────────
 8       32    LEAF     —              10.78×
 8       256   MODEB    8               5.67×
 16      32    BAILEY2  4×4             5.97×
 64      256   MODEB    4,4,4           2.11×   (carryover sweep mis-picked BAILEY2 → 0.77×)
 256     256   MODEB    4,4,16          2.09×
 1024    256   MODEB    4,4,4,4,4       1.57×
 4096    32    MODEB    4,4,4,8,8       1.63×
 65536   256   MODEB    4,4,8,16,32     1.40×
──────────────────────────────────────────────
 Min 1.37×   Median 2.01×   Max 10.78×   Mean 2.49×   Wins 31/31
```

### Out-of-place — vs MKL at T=8

```
 N       K     kind     dag/MKL-T8   note
──────────────────────────────────────────────
 8       256   MODEB      38.53×     MKL can't thread tiny batch
 16      32    BAILEY2    45.80×     dag-ST vs MKL-8T
 64      256   MODEB       5.26×
 256     256   MODEB       2.80×
 1024    256   MODEB       2.74×
 4096    256   MODEB       4.86×
 65536   256   MODEB       3.10×
 1024    32    MODEB       1.24×     (min)
──────────────────────────────────────────────
 Min 1.24×   Median 2.80×   Max 45.80×   Wins 31/31
```

### Out-of-place — arbitrary K (odd / non-multiple-of-8 batch)

```
 N      K    rem  kind     order      dag/MKL
─────────────────────────────────────────────
 8      31   3    LEAF     natural    6.32×
 8      33   1    LEAF     natural    6.26×
 16     31   3    BAILEY2  natural    2.99×
 16     33   1    BAILEY2  natural    2.82×
 64     31   3    BAILEY2  natural    1.79×
 64     33   1    BAILEY2  natural    2.06×
 256    31   3    BAILEY2  natural    1.32×
 256    33   1    MODEB    scrambled  1.40×
 1024   31   3    BAILEY2  natural    1.30×
 1024   33   1    BAILEY2  natural    1.36×
```

### K=1 INTERLEAVED — N = 2^a·odd, the ZTURN-T odd band vs MKL (2026-09-15)

```
 N        OOP vfft   OOP MKL   MKL/vfft     IP vfft   IP MKL   MKL/vfft   err
─────────────────────────────────────────────────────────────────────────────
 3072        2,892     5,354     1.85×        3,114    5,259     1.69×   5e-16
 6144        6,208    11,895     1.92×        6,507   11,852     1.82×   4e-16
 12288      13,104    25,080     1.91×       13,983   24,946     1.78×   4e-16
 24576      28,419    55,110     1.94×       29,338   58,179     1.98×   5e-16
 61440      97,606   169,222     1.73×      106,819  156,569     1.47×   6e-16
 245760    633,712 1,102,950     1.74×      615,938 1,112,100    1.81×   6e-16
```

### K=1 INTERLEAVED — every power of two 2..2^22 vs MKL, T=1 and T=8 (2026-09-22)

```
         N  serial verdict on the row       ours ns     MKL ns      x |   ours T=8    MKL T=8      x  T=8 verdict
         2  mono                                  4         11   2.89 |          4         11   2.84  serial
         4  mono                                  5         11   2.35 |          4         11   2.53  serial
         8  mono                                  7         12   1.61 |          7         12   1.71  serial
        16  ZTURN-T 4.4                          10         12   1.20 |         10         13   1.27  serial (the threaded arm lost)
        32  pair 4.8                             17         16   0.97 |         17         17   1.00  serial
        64  pair 4.16                            31         31   0.98 |         31         31   0.98  serial
       128  pair 4.32                            66         69   1.03 |         66         68   1.04  serial
       256  pair 16.16                          135        138   1.01 |        135        137   1.01  serial
       512  pair 16.32                          314        293   0.94 |        297        291   0.98  serial
      1024  ZTURN-T 4.8.8.4                     720        841   1.11 |        713        848   1.19  serial (the threaded arm lost)
      2048  ZTURN-T 8.8.8.4 tile 1024          1620       2032   1.25 |       1652       2162   1.31  serial (the threaded arm lost)
      4096  ZTURN-T 8.8.8.8 tile 2048          3593       3846   1.06 |       3679       3825   1.04  serial (the threaded arm lost)
      8192  ZTURN-T 8.4.8.4.8 tile 1024        8048       8477   1.05 |       4783       8352   1.70  threaded, 3949 MT passes counted
     16384  ZTURN-T 8.4.4.4.4.8 tile 2048      17189      19177   1.10 |      10804      14645   1.15  threaded, 1661 MT passes counted
     32768  ZTURN-T 8.4.4.8.8.4 tile 2048      38546      36893   0.95 |      20289      21434   0.98  threaded, 1229 MT passes counted
     65536  ZTURN-T 8.4.8.4.8.8 tile 1024      89247     109473   1.21 |      25213      42267   1.54  threaded, 487 MT passes counted
    131072  ZTURN-T 8.4.4.4.4.8.8 tile 2048     222413     259253   1.17 |      51707      80993   1.49  threaded, 255 MT passes counted
    262144  ZTURN-T 4.8.8.4.4.8.8 tile 2048     541912     710563   1.31 |     103363     185388   1.52  threaded, 220 MT passes counted
    524288  four-step 512.1024              1374012    1484350   1.04 |     271150     363462   1.34  threaded, four-step split 2048
   1048576  four-step 2048.512              3167088    4175437   1.28 |     578787     893788   1.54  threaded, four-step split 2048
   2097152  four-step 512.4096              8241637   10632675   1.27 |    1726562    2589850   1.50  threaded, four-step split 2048
   4194304  four-step 2048.2048            18851825   24110238   1.28 |    6328113    6966025   1.10  threaded, four-step split 4096
```

### K=1 INTERLEAVED — PRIME N, the prime cell's own raced method and inner (2026-09-19)

```
 N        N-1                 banked verdict                      ours (ns)    MKL (ns)   vs MKL
──────────────────────────────────────────────────────────────────────────────────────────────────
 31       2.3.5               RADER     il2p 3.10                        95          134   1.40 / 1.75
 131      2.5.13              RADER     il2p 13.10                      379         1066   2.81 / 2.58
 257      2^8                 RADER     il2p 4.64                       686         2159   3.15 / 3.32
 521      2^3.5.13            RADER     il3p 8.5.13                    2023         4573   2.26 / 2.14
 1021     2^2.3.5.17          BLUESTEIN ZTURN-T 8.8.4.8               5211         6013   1.15 / 1.16
 2053     2^2.3^3.19          RADER     il3p 4.19.27                  16160        22006   1.36 / 1.44
 4001     2^5.5^3             RADER     ZTURN-T 8.5.5.5.4             15731        27663   1.76 / 1.66
 4099     2.3.683             BLUESTEIN ZTURN-T 8.8.8.8.4 @ 2048     48222        54510   1.13 / 1.16
 8191     2.3^2.5.7.13        BLUESTEIN ZTURN-T 8.8.8.8.4 @ 2048     51420        57630   1.12 / 1.17
 12289    2^12.3              RADER     ZTURN-T 8.3.8.4.4.4 @ 2048    50803       112664   2.22 / 2.28
 40961    2^13.5              RADER     ZTURN-T 8.5.8.4.8.4 @ 1024   219487       795296   3.62 / 3.73
 65537    2^16                RADER     ZTURN-T 4.4.4.4.4.8.8       376547      1815207   4.82 / 4.23
 131071   2.3.5.17.257        BLUESTEIN ZTURN-T 8.8.8.8.4.4.4 @ 2048 1780033     1899607   1.07 / 1.19
```

### K=1 INTERLEAVED — the flat DIT's radix pool reaches 17 and 19 (2026-09-19)

```
 N                      before                        after
 6545  = 5.7.11.17      0.27 / 0.42   prime cell      1.26 / 1.21   flat DIT 5.11.17.7
 12155 = 5.11.13.17     0.34 / 0.40   prime cell      1.02 / 1.20   flat DIT 13.17.11.5
```

### K=1 INTERLEAVED — every N from 2 to 2048 vs MKL, the gauntlet (2026-09-21)

```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     1180     13    145   0.97   1.23   2.28    1.36
 chain3     346      2     34   1.00   1.29   1.70    1.29
 2p         256      0      6   1.18   1.54   2.23    1.58
 flat       240      2     23   1.00   1.27   1.70    1.27
 mono        22      0      1   1.16   1.65   2.34    1.61
 ztt          3      0      0   1.18   1.22   1.26    1.22
 ALL       2047     17    209   1.00   1.27   2.10    1.36
```

```
 size band     cells   median   <1.0   <0.8
 2..64             63     1.52      5      0
 65..256          192     1.53      6      1
 257..512         256     1.37     11      1
 513..1024        512     1.26     62      1
 1025..2048      1024     1.20    125     14
```

```
 family                                     cells   median   <1.0   what decides it
 composite with a prime >= 53 (prime cell)    878     1.23    109   no kernel above 47: whole-N Bluestein vs MKL's direct radix-p stage, whose cost climbs with p (parity from p ~ 53, 2x by 89)
 composite, prime cell BY RACE (primes <= 47)     7     1.24      2   the chain of large radices lost to whole-N Bluestein in its own cell's race (43.47, 43.43, 2.43): a chain's cost is the sum of its radices', the convolution's is flat
 composite whose largest prime is 29..47      293     1.31     12   kernels at every IL kind since 2026-09-21; the direct conjugate-pair form beats MKL's radix-p stage 1.45-1.63x where one large stage suffices
 prime, Bluestein banked                      189     1.15     32   Rader's inner N-1 is not a buildable length
 prime, Rader banked                          107     1.81      2   37 and 41 among them: Rader over a smooth N-1 beat the direct radix-37/41 solo kernel in the race
 prime, solo kernel                            14     1.77      1   2..47, where the solo kernel won its race
 chain3                                       346     1.29     34   cost per point and pass tracks the LARGEST radix in the chain; 13 is the one radix where the direct form trails MKL (0.92x); the route's two-speed readings were the SMT sibling (see the bench finding)
 flat (incl. the 2-led chains)                240     1.27     23   a radix-2 leaf when N/2 is odd; the tiny 2 x prime cells are solos now
 pow2 32..512                                   5     0.98      3   engine at parity with MKL inside the race; the door's bound K=1 fast path (2026-09-21) returned 2-3 ns of the 4-5 ns fixed cost per call
```

### K=1 INTERLEAVED — the cells 2..512 IN PLACE, raced in place (2026-09-21)

```
 route    cells   <0.8   <1.0    p10    med    p90   gmean   ours ip/oop   MKL ip/oop
 prime      269     36     57   0.73   1.25   2.62   1.38       0.99          1.00
 2p         126      0      7   1.15   1.47   2.17   1.53       1.00          1.00
 chain3      78      0      1   1.22   1.38   2.11   1.45       0.99          1.00
 flat        22      2      7   0.89   1.11   1.48   1.14       1.00          1.00
 mono        16      1      4   0.82   1.26   1.73   1.22       1.00          0.98
 ALL        511     39     76   0.85   1.33   2.35   1.41       1.00          1.00
```

## 2. vs MKL — 2D C2C

```
 N1×N2     dag/MKL   order
──────────────────────────────────
 64×64     ~1.6×*    scrambled
 128×128    1.41×    scrambled
 256×256    1.26×    scrambled
 512×512    1.29×    scrambled
──────────────────────────────────
 median    ~1.35×    (4/4 win)
```

### K=1 INTERLEAVED — every N from 2049 to 4096 vs MKL (2026-09-22)

```
 route    cells   <0.8   <1.0    p10    med    p90   gmean
 prime     1538     34    259   0.96   1.15   1.79    1.24
 chain3     286      1     22   1.01   1.27   1.74    1.30
 flat       198     10     38   0.86   1.17   1.58    1.18
 ztt         22      0      0   1.45   1.59   1.83    1.62
 2p           4      0      0   1.05   1.28   1.67    1.30
 ALL       2048     45    319   0.96   1.16   1.76    1.24
```

```
 size band     cells   median   <1.0   <0.8
 2049..2560       512     1.07    182     22
 2561..3072       512     1.31     39      6
 3073..3584       512     1.15     51      8
 3585..4096       512     1.14     47      9
```

```
 family                                          cells   median   <1.0   what decides it
 composite with a prime >= 53 (whole-N Bluestein)  1279     1.15    221   no kernel above 47; M = the next pow2 >= 2N-1 is 8192 for the whole range, 3.2-4.0x N below 2560 -- the weak band
 chain3                                           286     1.27     22   the raced chains keep their margin above 2048
 prime N, Bluestein banked                        214     1.12     38   Rader's inner N-1 not buildable
 flat                                             198     1.17     38   up to 10 stages; the 7^3 cells (2744, 3430) trail MKL
 prime N, Rader banked                             41     1.66      0
 2^a.odd in ZTURN-T's odd band (ZTT_ODD)           21     1.60      0   the staged odd-radix ZTURN-T, 1.45-1.83x
 pair                                               4     1.28      0
 composite, prime cell by race (primes <= 47)       4     1.45      0
 pow2 (ZTURN-T)                                     1     1.08      0   the control cell
```

### 2D C2C — vs MKL at T=8

```
 N1×N2     dag-T8 (ns)  MKL-T8 (ns)  dag/MKL   dag self-scale ST->T8
──────────────────────────────────────────────────────────────────
 64×64           5,641       48,751   8.64×     0.86× (overhead)
 128×128        23,996       87,069   3.63×     ~1.0×
 256×256        70,210      214,597   3.06×     1.88×
 512×512       500,200    1,307,575   2.61×     1.57×
──────────────────────────────────────────────────────────────────
 median                              ~3.34×     (4/4 win)
```

### 2D C2C — the NATIVE INTERLEAVED tier vs MKL CCE (standing as of 2026-09-15)

```
 N1×N2      O-NATIVE (ns)  MKL-CCE (ns)  vs MKL-CCE   rows
─────────────────────────────────────────────────────────────────
 128×128          19,849        31,538      1.59×      pairs
 256×256          85,603       130,823      1.53×      pairs
 512×512         447,463       927,763      2.07×     pairs
 1024×1024     2,206,587     4,809,862      2.18×     ZTURN-T
 16×4096          90,763       135,127      1.49×     ZTURN-T
 32×1024          40,002        69,593      1.74×     ZTURN-T
 64×256           17,955        32,263      1.80×     pairs
 4096×64         646,288       904,375      1.40×      pairs
 8192×64       1,650,375     2,066,875      1.25×*     pairs
 16384×64      3,704,962     5,062,325      1.37×      pairs
 32768×64      9,140,738    15,197,088      1.66×      pairs
─────────────────────────────────────────────────────────────────
                                    11/11 win, median ~1.59×
```

### 2D C2C — the native tier MULTITHREADED (2026-08-27)

```
 N1×N2       fwd      bwd      raced verdict
──────────────────────────────────────────────
 256×256    1.02×    1.00×    serial (wl=N1: one band, no MT axis —
                              wl re-race at T is a sweep item)
 512×512    4.04×    3.93×    threaded (4 bands, 4 band-workers)
 1024×1024  7.25×    4.89×    threaded
 4096×64    5.74×    5.03×    threaded
 8192×64    7.73×    7.60×    threaded
 64×1024    2.14×    1.93×    threaded (strip+slab shape)
```

### 2D C2C — every power-of-two plane up to 4M points, the pow2 grid vs MKL DFTI 2D (2026-09-23)

Every plane N1 × N2 with both sides a power of two from 2 to 8,192 and at most 2^22
points: 159 planes, complex-to-complex, interleaved, natural order, out of place, K=1,
single thread, through the front door on a scratch store (every plane raced and banked,
then benched: the gauntlet's 2D contract). Speedup = MKL time / ours, the worse of the
two engine orders. The record is `gauntlet/results/gauntlet_2d-pow2grid7/`.

```
 plane size          planes   median   at/above parity   best
──────────────────────────────────────────────────────────────────────
 up to 256 points        28    4.80×        96%          12.84× (4×2)
 257..4,096              38    1.63×        82%           4.78× (2×256)
 4,097..65,536           48    1.38×        98%           2.56× (2×4096)
 65,537..4M              45    1.22×        96%           1.90× (8192×512)
──────────────────────────────────────────────────────────────────────
 all                    159    1.41×        93%          12.84× (4×2)
```

Routes the planner served: the column chain with the per-row child (53), with the rows
batched through one kernel call (8) or through the row child's two stages batched (19);
the skewed column pass (29, plus 12 and 11 with the batched rows); the turn route for the
tall narrow planes (27). Below parity: 128×16 (0.76), 32×64 (0.79), 16×128 (0.90), 64×32
(0.92), and seven planes at 0.93–0.99 within a run-to-run swing (16×16, 32×16, 32×32,
16×64, 128×256, 512×2048, 4096×128). The design is `docs/design/il2d_c2c_strategy.md`.

### 3D C2C — the NATIVE INTERLEAVED tier vs MKL CCE (standing as of 2026-09-15)

```
 cell          SCRAMBLED T=1              NATURAL T=1               SCRAMBLED T=8            NATURAL T=8
               ours        MKL    ratio   ours        MKL    ratio  ours      MKL    ratio   ours      MKL    ratio
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 16³             6,463     7,099  1.10~     7,119     7,273  1.02~    4,251    3,026  0.71     3,086    3,197  1.04~
 32³            51,580    62,587  1.21     67,115    64,095  0.96    20,857   13,620  0.65    17,920   14,020  0.78~
 64³           512,775   773,225  1.51    732,362   833,963  1.14~   76,450  110,662  1.45~   94,687  114,650  1.21~
 128³        6,404,700 9,679,712  1.51  7,178,363 11,625,088 1.62 1,058,363 1,416,163 1.34 1,503,938 1,606,863 1.07~
 32×16×64       46,313    50,956  1.10     59,495    51,962  0.87    22,515   14,770  0.66    15,095   17,675  1.17~
 64×128×32     551,712   784,125  1.42    754,975   873,313  1.16~   76,113  111,013  1.46~  119,525  145,450  1.22~
 256×64×16     618,525   691,013  1.12~   724,700   706,163  0.97~  146,687  108,037  0.74~  109,138  107,800  0.99~
 27×9×15         8,693     9,090  1.05~     9,261     9,118  0.98~    5,282    5,819  1.10~    4,659    6,635  1.42
 36×20×28       33,499    60,281  1.80     34,423    59,174  1.72    20,422   17,035  0.83    10,199   19,236  1.89
 45³           188,238   273,976  1.46    230,914   269,348  1.17~   38,386   43,386  1.13~   37,895   47,919  1.26
 81×27×27      127,694   170,148  1.33    142,282   170,112  1.20~   33,124   32,615  0.98~   23,848   34,888  1.46
 16×16×4096  2,864,400 3,726,400  1.30  3,681,000 3,649,475  0.99~  333,563  449,512  1.35~  517,000  527,112  1.02~
 8×16×12288  4,395,350 7,356,538  1.67  4,907,763 7,489,638  1.53   636,350  896,312  1.41~  866,687 1,016,262 1.17~
 32×32×4096 13,603,162 18,541,325 1.36 15,484,150 18,167,375 1.17 4,916,088 4,963,975 1.01~ 5,036,162 5,212,275 1.03~
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 T=1: scrambled 14/14 (10 outside the spread); natural 4 win, 8 tie, 2 loss
      (32³, 32×16×64: small cubes where MKL's natural output is cheap).
 T=8: MKL's arm spreads 50–770% at most cells (ours as wide at some); the
      natural class wins or ties at 14/14; the scrambled class loses the
      four small cells (16³, 32³, 32×16×64, 36×20×28) — the two-phase
      fork-join floor against MKL's one parallel region — and wins or ties
      the rest.
```

```
 cell         SCRAMBLED: serial      MT   speedup  verdict     | NATURAL: cycle    strip    gain   verdict
────────────────────────────────────────────────────────────────────────────────────────────────────────────
 16³               5,422      4,589   1.2×  plane/flat   |          5,411     4,476   1.21×  plane/child/strip
 32³              48,109     21,998   2.2×  plane/flat   |         16,433        —      —    plane/flat (cycle)
 64³             445,698     79,119   5.6×  plane/flat   |        122,056   103,400   1.18×  plane/child/strip
 128³          4,950,267    599,433   8.3×  band/child   |      1,011,767   883,867   1.15×  plane/flat/strip
 32×16×64         42,897     19,117   2.2×  band/child   |         22,421    18,047   1.24×  plane/flat/strip
 64×128×32       466,626     76,728   6.1×  band/child   |        116,982   128,200     —    plane/flat (cycle)
 256×64×16       536,168    138,811   3.9×  plane/child  |        145,823   115,930   1.26×  plane/flat/strip
 27×9×15           7,107      4,651   1.5×  plane/flat   |          5,070     4,046   1.25×  plane/child/strip
 36×20×28         34,066     18,563   1.8×  band/child   |         16,463    10,026   1.64×  plane/child/strip
 45³             160,273     40,234   4.0×  plane/child  |         51,017    37,934   1.35×  plane/child/strip
 81×27×27        113,915     28,342   4.0×  band/child   |         48,646    31,458   1.55×  plane/flat/strip
 16×16×4096    2,217,756    306,711   7.2×  plane/flat   |        420,500        —      —    plane/child (cycle)
 8×16×12288    3,286,267    431,700   7.6×  plane/child  |        516,225        —      —    plane/flat (cycle)
 32×32×4096   11,396,650  3,345,450   3.4×  plane/flat   |      5,055,800 3,066,350   1.65×  plane/flat/strip
────────────────────────────────────────────────────────────────────────────────────────────────────────────
 The strip form won the natural threaded race at 10 of 14 cells, on two
 quiet runs: the strips give every worker independent in-scratch work with
 no cold scattered plane writes. Strip width matters at the long cells
 (one page visit per plane per strip): the arm times fall to 256–512
 columns; the pool runs 8..1024 under the L2 budget.
```

### 3D C2C — every power-of-two volume up to 4M points, the pow2 grid vs MKL DFTI 3D (2026-09-24)

Every volume N1 × N2 × N3 with each side a power of two from 2 to 8,192 and at most 2^22
points: 1,288 volumes, complex-to-complex, interleaved, natural order, out of place, K=1,
single thread, through the front door on a scratch store (every volume raced and banked,
then benched: the gauntlet's 3D contract, built this day). Speedup = MKL time / ours, the
worse of the two engine orders. The run took two hours; nothing was refused. The record is
`gauntlet/results/3d-pow2_2026-09-24/`.

```
 volume              volumes   median   at/above parity
───────────────────────────────────────────────────────
 up to 4,096 points      220    4.54×        99%
 4,097..65,536           337    1.94×        99%
 65,537..1M              478    1.38×        98%
 1M..4M                  253    1.26×        99%
───────────────────────────────────────────────────────
 all                   1,288    1.55×        98%      (gmean 1.87×, p10 1.12×, p90 4.53×)
```

The tall volumes, N1 from 256 to 8,192 over 2 to 16 lanes, are among the best cells:
36 of them, median 5.57×, none below 3.6×. The 20 volumes below parity are 2×2×2
(0.66), the tiny cubes within a swing of parity, and one class: a short first axis
(N1 = 4 or 8) over a tall plane (N2 ≥ 1,024, N3 ≤ 64), 8×2048×32 at 0.78 and 4×2048×32
at 0.83, the rest at 0.92–0.99. Decomposed against the 2D record, the per-plane
transforms cost the same as MKL's; the deficit is the first-axis sweep of the volume,
which streams at 56 GB/s where the machine copies at 81 and MKL's sweep runs at the
non-temporal store rate of about 90. A fused slab form (first and second axis per
cache-resident strip of lanes) was prototyped and refuted: at 4 to 8 lanes the narrow
column passes cost more than the residency saves.

## 3. vs MKL — 1D R2C

### Single-thread — the packing tax

```
 N      K     path    dag/MKL    note
──────────────────────────────────────────────
 256    8     rfft     1.07×     JIT-wired rfft, low-K win
 256    16    rfft     1.15×
 256    256   stride   1.04×
 512    8     rfft     1.17×
 1024   8     rfft     0.64×     large-N rfft plane = L2-bound
 1024   256   stride   0.80×     decoupled-r2c structural gap
──────────────────────────────────────────────
 18 cells: 6 win.  Median 0.79×, range 0.46–1.17×.
```

### Multi-threaded (T=8) — the layout payoff

```
 N      K     path    dag/MKL-T8   dag self-scale ST→T8
──────────────────────────────────────────────────────
 256    8     rfft      21.75×     ~1.0× (rfft is ST)
 256    256   stride     5.30×     2.79×
 512    256   stride     4.47×     4.87×
 1024   256   stride     3.65×     3.72×
 1024   16    rfft       1.74×     ~1.0×
──────────────────────────────────────────────────────
 18 cells: 18 win.  Median ~4.7×, range 1.74–21.75×.
```

### 1D C2R (backward) — the natural split path

#### Single-thread — the packing tax (again)
```
 N      K     path      dag/MKL    note
──────────────────────────────────────────────
 256    8     natural    0.92×     ≈parity — packed-speed on split input
 256    16    natural    0.74×
 256    64    natural    0.55×     mid-K: MKL compute-bound / L1-resident
 256    128   natural    0.55×
──────────────────────────────────────────────
 natural ≈ 2× the old forced-stride path; reaches MKL parity only at K=8.
```

#### Multi-threaded (T=8) — the layout payoff (again)
```
 N      K     path      dag/MKL-T8   dag self-scale ST→T8
──────────────────────────────────────────────────────
 256    8     natural    ~17×        ~1.0× (K<16: lane-split floor)
 256    32    natural    ~7.9×       ~1.4×
 256    64    natural    3.9×        1.9×
 256    128   natural    3.0×        2.2×
 256    256   natural     —          2.8×   (MKL-T8 crashes at N·K≥131072)
 512    256   natural     —          3.6×
 1024   256   natural     —          2.8×
```

### 1D INTERLEAVED r2c/c2r, K=1 — the D2 zr2c route (like-for-like vs MKL's home layout)

```
 N        r2c OOP   r2c IN-PLACE | c2r OOP   c2r IN-PLACE   <- as SHIPPED (wisdom picks the route)
--------------------------------------------------------------------------------
 512       1.40x       1.50x     |  1.30x       1.49x
 1024      1.22x       1.24x     |  1.03x       1.03x
 2048      1.21x       1.32x     |  0.97x       1.02x
 4096      1.08x       1.04x     |  0.66x       0.94x
 8192      1.20x       1.21x     |  0.81x       1.10x
 16384     1.23x       1.24x     |  0.93x       1.06x
 65536     1.01x       1.06x     |  0.88x       1.05x
--------------------------------------------------------------------------------
 r2c WINS EVERY CELL, both placements. In-place >= out-of-place everywhere
 except 4096. c2r wins at the small end and in-place from 8192 up.

 THE c2r OOP COLUMN IS A KNOWN-BAD ROUTE PICK, not an engine result. All 37
 shipped kind-5 route rows are src=migrated with no ns= - nothing was ever
 raced; a structural rule (place=oop -> route 0) was written down once. Where
 the race disagrees, forcing route 1 gives:
     2048   0.97x -> 1.12x      4096   0.66x -> 0.82x
     8192   0.81x -> 1.01x      65536  0.88x -> 0.88x
 i.e. the banked pick costs up to 27-35% on c2r OOP. Seeding those rows and
 re-racing is the open item (audit G3); until then read the c2r OOP column as
 "what the stale verdict serves", not as the engine's reach.

 UNRESOLVED: front-door route-0 OOP c2r runs ~22% slower than the identically
 shaped hand-built arm (2048: ~1680 vs ~1373 ns, reproducible). Same algorithm
 on paper, so the difference is buffer placement - the plan's internal scratch
 vs the bench's, or 4KB aliasing. Do not bank a front-door c2r OOP number
 until that is explained.
```

### 1D ODD c2c — the K=1 IL tier for odd N (2026-09-06)

```
 N        route (banked)                          vfft (µs)   MKL (µs)   vs MKL
───────────────────────────────────────────────────────────────────────────────
 405      chain3 9·9·5                               0.39       0.55      1.42×
 1215     chain3 15·9·9                              2.00       2.11      1.06×
 3125     flat 5·5·5·5·5 t.t.t.o                     5.92       6.13      1.04×
 4095     chain3 21·13·15                            6.77       8.40      1.24×
 6561     flat 9·9·9·9 t.t.o tw729                  13.82      15.73      1.14×
 15625    flat 5·5·5·5·5·5 t.t.t.t.o tw625          32.40      35.72      1.10×
 16807    flat 7·7·7·7·7 t.t.t.o                    35.70      38.53      1.08×
 19683    flat 9·3·9·9·9 t.t.t.o tw729              43.94      52.69      1.20×
 59049    flat 9·9·3·9·3·9 t.t.t.t.n               183.5      185.7       1.01×
 78125    flat 5·5·5·5·5·5·5 t.t.t.t.t.o tw625     215.4      277.3       1.29×
 98415    flat 9·9·5·9·9·3 t.m.t.n.o               295.7      353.9       1.20×
 117649   flat 6 stages m.t.t.t.o                  326.3      370.5       1.14×
 137781   flat 6 stages t.t.m.t.o                  447.6      537.5       1.20×
 177147   flat 9·9·9·3·9·9 t.t.t.t.o tw19683       540.1      699.7       1.30×
 194481   flat 6 stages t.t.t.t.o                  623.4      802.2       1.29×
```

```
 N        natural (µs)   scrambled (µs)   natural/scrambled
 1215         2.4             2.3            1.04×
 4095         9.6             8.6            1.12×
 6561        14.0            13.0            1.08×
 19683       53.7            50.3            1.07×
 59049      154.8           137.9            1.12×
 78125      244.6           214.7            1.14×
 98415      346.3           309.0            1.12×
```

```
 N        plane    widest span   natural   scrambled
 98415    1.5 MB   10935          0.98      0.99   (fits L2: the race banks 0)
 177147   2.7 MB   19683          0.86      0.91
 194481   3.0 MB   21609          0.85      0.83
 245025   3.7 MB   27225          0.95      0.93
```

### 1D ODD c2c — the K=1 IL tier MULTITHREADED (2026-09-07)

```
 N         serial (ns)   MT (ns)   speedup   verdict
──────────────────────────────────────────────────────
 6561          14,759     8,890     1.7×    tiles/tw729
 15625         38,053    17,167     2.2×    tiles/tw625
 16807         38,134    16,461     2.3×    tiles/tw2401
 19683         46,438    16,786     2.8×    tiles/tw729
 59049        160,497    39,382     4.1×    tiles/tw729
 78125        280,064    89,413     3.1×    tiles/tw125
 98415        287,983    54,429     5.3×    tiles/tw1215
 117649       353,283    59,150     6.0×    tiles/tw16807
 137781       389,016    83,332     4.7×    tiles/tw1701
 177147       593,023   100,606     5.9×    tiles/tw2187
 194481       594,571   118,306     5.0×    tiles/tw3087
```

```
 N          O-NATIVE T=8 (ns)   MKL T=8 (ns)   vs MKL   elementwise
──────────────────────────────────────────────────────────────────────
 6561                 8,125          11,908    1.47×    1.1e-15
 15625               16,247          19,790    1.22×    6.9e-16
 16807               13,997          25,336    1.81×    1.3e-15
 19683               16,982          24,141    1.42×    1.5e-15
 59049               40,424          55,958    1.38×    1.5e-15
 78125               77,712          62,168    0.80×    1.1e-15
 98415               51,840         104,995    2.03×    1.6e-15
 117649              59,569         118,100    1.98×    1.1e-15
 137781              70,507         121,657    1.73×    1.5e-15
 177147              90,027         159,173    1.77×    1.4e-15
 194481              97,440         195,170    2.00×    1.3e-15
──────────────────────────────────────────────────────────────────────
                                              10/11 win, median ~1.73×
```

### 1D ODD/PRIME r2c/c2r — full coverage, priced vs MKL (2026-08-27)

```
 N      class       r2c vs MKL   c2r vs MKL   serving
────────────────────────────────────────────────────────
 101    prime          1.67×        2.00×     bridge
 1021   prime          1.27×        1.33×     bridge
 129    3·43           0.90×        0.90×     bridge
 255    smooth         1.75×        1.60×     bridge (raced in —
                                              was 0.42× on rfft)
 63     smooth         ~par         ~par      bridge (raced in)
 1215   3⁵·5           0.67×        0.12×     rfft / bridge-only
 4095   smooth         0.30×        0.31×     bridge (raced in)
```

### 2D NATURAL order — native tier, both transforms, multithreaded (2026-09-04)

```
 cell            transform   ST (µs)   MT (µs)   speedup   verdict
──────────────────────────────────────────────────────────────────────
 1024x256        r2c          381.6      62.5      6.11×    threaded
 512x128         r2c           81.7      19.6      4.17×    threaded
 256x64          r2c           21.9      14.3      1.53×    threaded
 63x64 (odd)     r2c            4.2        —         —      serial (race)
 1024x128        c2c          302.7     135.3      2.24×    threaded (strips)
 256x64          c2c           37.8      15.2      2.49×    threaded (blocks)
 63x64 (odd)     c2c            7.8       4.3      1.81×    threaded (strips)
 512x64          c2c           61.1      55.0      1.11×    threaded (blocks)
```

## 4. vs MKL — 2D R2C

### Single-thread

```
 N1×N2      dag/MKL    order
──────────────────────────────────
 64×64       0.86×     scrambled
 128×128     0.85×     scrambled
 256×256     0.80×     scrambled
 512×512     0.89×     scrambled
──────────────────────────────────
 median     ~0.85×     (best-of-3)
```

### Multi-threaded (T=8)

```
 N1×N2     dag-T8 (ns)   dag self-scale ST→T8
──────────────────────────────────────────────
 64×64          6,734    0.78×  (overhead)
 128×128       23,271    0.96×
 256×256       70,010    1.71×
 512×512      415,188    1.38×
```

### 2D C2R (backward)

```
 N1×N2     dag/MKL   order
──────────────────────────────────
 64×64      0.84×    scrambled
 128×128    0.95×    scrambled
 256×256    0.75×    scrambled
 512×512    0.95×    scrambled
──────────────────────────────────
 median    ~0.89×    (single-thread)
```

#### 2D C2R — multi-threaded (T=8)

```
 N1×N2     dag-T8 (ns)   dag self-scale ST→T8
──────────────────────────────────────────────
 64×64          6,007    0.78×  (overhead)
 128×128       22,144    0.91×  (overhead)
 256×256       66,169    1.59×
 512×512      328,031    1.53×
```

### 2D R2C/C2R — the NATIVE INTERLEAVED tier vs MKL CCE (2026-08-26)

```
 N1×N2        r2c nat/MKL   c2r nat/MKL
────────────────────────────────────────
 64×64           1.46×         1.09×
 256×256         1.37×         1.50×
 512×512         2.22×         2.18×
 1024×1024       1.66×         2.25×
 16×4096         1.29×         1.13×
 4096×16         1.03×         1.02×
 32×1024         1.49×         1.07×
 64×256          1.68×         1.33×
 4096×64         1.83×         1.66×
 8192×64         2.20×         1.90×
────────────────────────────────────────
 20/20 rows ≥ parity vs MKL.
 Median r2c ~1.5×, c2r ~1.4×.
```

### 2D R2C/C2R — the native tier MULTITHREADED (2026-08-27)

```
 N1×N2       r2c      c2r      raced column verdict
────────────────────────────────────────────────────
 128×64     1.45×    1.64×    threaded (marginal cell)
 512×32     1.54×    1.31×    serial — banked "no"
 256×256    1.55×    1.19×    marginal, run-dependent
 512×512    4.19×    6.34×    threaded
 1024×1024  7.69×    7.70×    threaded
```

### 1D C2C — full sweep

```
Category       Cells    Min   Median    Max    Mean
─────────────────────────────────────────────────────
Small (N≤128)    15   1.86×   4.10×   8.70×   4.60×
Power-of-2       30   1.34×   3.08×  15.89×   4.28×
Composite        33   1.82×   3.45×  15.07×   4.93×
Odd composite    18   1.38×   3.67×   6.29×   3.72×
Mixed deep       18   1.50×   5.28×  11.38×   5.11×
Prime powers     30   1.37×   5.09×  17.79×   6.85×
Genfft (R=11/13) 15   1.85×   3.25×  10.94×   4.52×
Rader primes     24   1.07×   2.23×   4.05×   2.38×
Bluestein primes 24   0.92×   1.15×   1.74×   1.22×
─────────────────────────────────────────────────────
OVERALL         207   0.92×   3.21×  17.79×   4.25×

Wins vs FFTW3: 202/207 (97.6%)
```

| Cell | Factors | Ratio |
|------|---------|------:|
| N=390625 (5^8) K=256 | 5×5×5×5×5×5×25 | **17.79×** |
| N=78125 (5^7) K=256 | 5×5×5×25×5×5 | 17.51× |
| N=65536 K=256 | 4×4×8×16×32 | 15.89× |
| N=131072 K=256 | 4×4×4×4×4×4×32 | 15.57× |
| N=100000 K=256 | 4×25×5×8×25 | 15.07× |

| Cell | Ratio (pre-wisdom) |
|------|------:|
| N=179 K=256 (Bluestein) | 0.92× (FFTW wins) |
| N=59 K=256 (Bluestein) | 0.93× (FFTW wins) |
| N=59 K=32 (Bluestein) | 0.96× (within noise) |

### DCT-II (REDFT10) — `bench_dct2_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 1024 (JPEG) | 2,300 | 3,400 | **1.48×** |
| 8 | 4096 | 9,500 | 11,100 | 1.17× |
| 16 | 1024 | 12,400 | 39,200 | 3.16× |
| 32 | 1024 | 32,200 | 81,100 | 2.52× |
| 64 | 1024 | 71,200 | 173,800 | 2.44× |
| 128 | 256 | 28,900 | 88,300 | 3.06× |

### DCT-III (REDFT01) — `bench_dct3_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 1024 (JPEG) | 2,500 | 2,900 | 1.16× |
| **8** | **4096** | **17,200** | **10,400** | **0.60× (FFTW wins)** |
| 16 | 1024 | 13,700 | 41,100 | 3.00× |
| 32 | 1024 | 34,100 | 84,800 | 2.49× |
| 64 | 1024 | 75,200 | 178,100 | 2.37× |
| 256 | 256 | 65,900 | 203,300 | 3.08× |
| 1024 | 256 | 416,000 | 1,495,500 | **3.59×** |

### DCT-IV (REDFT11) — `bench_dct4_vs_fftw`

| N | K | vfft ns | fftw ns | ratio |
|--:|--:|--------:|--------:|------:|
| 8 | 256 | 800 | 2,700 | 3.38× |
| 8 | 1024 | 4,300 | 9,400 | 2.19× |
| 8 | 4096 | 17,600 | 36,900 | 2.10× |
| 16 | 1024 | 8,900 | 35,900 | **4.03×** |
| 32 | 1024 | 28,300 | 74,200 | 2.62× |
| 64 | 1024 | 60,800 | 161,800 | 2.66× |
| 256 | 256 | 59,500 | 186,000 | 3.13× |
| 1024 | 256 | 354,200 | 1,482,100 | **4.18×** |

### DST-II / DST-III (RODFT10 / RODFT01) — `bench_dst23_vs_fftw`

| Variant | N | K | vfft ns | fftw ns | ratio |
|---------|--:|--:|--------:|--------:|------:|
| DST-II | 8 | 256 | 600 | 2,400 | **4.00×** |
| DST-II | 16 | 1024 | 16,100 | 38,900 | 2.42× |
| DST-II | 32 | 1024 | 39,100 | 78,500 | 2.01× |
| DST-II | 64 | 1024 | 90,800 | 173,600 | 1.91× |
| DST-II | 256 | 256 | 82,600 | 198,600 | 2.40× |
| DST-II | 1024 | 256 | 553,900 | 1,484,500 | 2.68× |
| DST-III | 8 | 256 | 700 | 2,900 | **4.14×** |
| DST-III | 16 | 1024 | 21,400 | 40,800 | 1.91× |
| DST-III | 32 | 1024 | 41,100 | 83,200 | 2.02× |
| DST-III | 64 | 1024 | 94,700 | 176,700 | 1.87× |
| DST-III | 256 | 256 | 84,300 | 207,100 | 2.46× |
| DST-III | 1024 | 256 | 544,900 | 1,507,000 | 2.77× |

### Headline (r2r vs FFTW3, T=1)

| Family | Ratio range | Cells | Wins |
|--------|:-----------:|:-----:|:----:|
| DCT-II | 1.17–3.16× | 6 | 6/6 |
| DCT-III | 0.60–3.59× | 7 | 6/7 |
| DCT-IV | 1.85–4.18× | 11 | 11/11 |
| DST-II | 1.91–4.00× | 6 | 6/6 |
| DST-III | 1.87–4.14× | 6 | 6/6 |
| DHT | ~1.9–2.8× (summary) | — | — |

## 5. Multi-threaded scaling

### DCT-II / DCT-III / DCT-IV / DST-II/III / DHT (wrapper MT, new in v1.0)

```
Transform   Cell           T=1 ns   T=2 (×)    T=4 (×)    T=8 (×)
──────────────────────────────────────────────────────────────────
DCT-II      N=256  K=1024   482000  1.04   1.95   2.60
DCT-IV      N=256  K=1024   452200  1.12   1.77   2.09
DST-II      N=256  K=1024   620900  1.17   2.06   2.49
DHT         N=256  K=1024   452900  0.97   1.55   1.85
DCT-II      N=1024 K=1024  2297700  1.08   1.55   2.35
DCT-IV      N=1024 K=1024  2682900  1.16   1.77   2.65
DST-II      N=1024 K=1024  2713900  0.97   1.41   2.11
DHT         N=1024 K=1024  2047300  0.88   1.23   1.67
DCT-II      N=4096 K=1024 13911400  1.11   1.65   2.11
DCT-IV      N=4096 K=1024 16838200  1.20   1.58   2.14
DST-II      N=4096 K=1024 19109400  1.13   1.62   2.20
DHT         N=4096 K=1024 13426100  1.06   1.49   1.83
DCT-II      N=4096 K=4096 58493200  1.12   1.61   2.14
DCT-IV      N=4096 K=4096 72842000  1.22   1.44   1.62
DST-II      N=4096 K=4096 80495400  1.13   1.63   2.20
DHT         N=4096 K=4096 59296500  1.06   1.55   1.87
```

### Why not 8× at T=8?

```
Pass 1: pre-permute / pre-twiddle    — bandwidth-bound
Pass 2: inner FFT (R2C or C2C)       — has its own MT
Pass 3: post-process / post-twiddle  — compute + memory mix
```

### Where the 8× comes back: v1.1 fused codelets

| Generation | Memory traffic / call | T=8 ceiling |
|-----------|----------------------|:-----------:|
| Pre-v1.0 (sequential wrappers) | 3 × N·K·16 bytes | ~1.4× |
| **v1.0 (parallel wrappers, current)** | **3 × N·K·16 bytes** | **~2.6×** |
| v1.1 (fused codelets) | 1 × N·K·16 bytes | ~5× projected |

## 6. Per-codelet performance (VTune-grade)

| Radix | Retiring (% of pipeline slots) | Bottleneck |
|------|:-----:|------|
| R=4  | 86% | compute-peak (port 0/1 at 96/91%) |
| R=8  | 72% | DFT-8 critical path dependency chains |
| R=10 | 63% | radix-5 + radix-2 FMA chains |
| R=11 | 59% | Winograd, machine-clears flagged |
| R=12 | 57% | radix-3 + radix-4 FMA chains |
| R=13 | 60% | Winograd + Sethi-Ullman |
| R=16 | 25% | store-bound + L1 latency (post-prefetch) |
| R=20 | 54% | radix-5 FMA chains |
| R=25 | 50% | hybrid compute/store |
| R=32 | 34% | L1 store-DTLB overflow (~80 pages) |
| R=64 | 27% | load + store DTLB overflow (~160 pages) |

## See also
