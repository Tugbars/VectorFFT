# The prime engine's inner is raced, not borrowed (design, 2026-09-17)

Owner's ruling, 2026-09-17: **"prime cells should have their own inner race."**

## What a prime cell does today

A prime N cannot be split into stages, so `il_prime.h` turns it into a
convolution and does the convolution with an FFT of a friendlier length M:
Rader's M = N - 1, Bluestein's M = the next power of two >= max(16, 2N - 1).
The METHOD (Rader vs Bluestein) is already raced and banked
(`_ilprime_create_banked`: `vw2_prime_method_lookup` replays, else
`vfft_ilprime_create_method(N, 0)` runs the house race body over both arms,
min-of-3, and `vw2_prime_method_bank` writes `eng= ran=1 src=race` plus a
signpost to the inner's row).

The INNER is not raced. It is BORROWED, in three layers:

| layer | where | what it does |
| --- | --- | --- |
| 1 | `k1_commit.h:210` (`_ilprime_inner_from_wisdom`) | reads the length-M cell's **scrambled K=1 row**, UNGUARDED by recalibrate; if it names ZTURN-T, that chain and tile serve |
| 2 | `k1_commit.h:234` | `if (M > 4096) return 0;` -- above that, layer 1 is the ONLY source; below it, `_k1_il_candidate(W, cfg, M, ...)` asks the K=1 tier for a pair or chain3 at M (this one does honor recalibrate) |
| 3 | `il_prime.h:122` (`_ilprime_inner_make`, when the provider returns 0) | the engine's "structural" inner: the **most balanced pair** by `abs(R1 - R2)`, else `vfft_il3p_default_chain` -- two heuristics, never measured |

So the inner is chosen by proxy (M standalone is not M inside a convolution:
forward, pointwise multiply, backward, in place, any order acceptable), the
caller's `recalibrate` cannot reach layer 1 without racing M's cell nested
inside N's create -- which `_k1_il_dp_busy` forbids -- and layer 3 is
exactly the class of rule the owner's law excludes.

## The contract

A prime cell owns its inner verdict the way every other cell owns its plan:
raced at create on a miss or under `recalibrate`, banked on **the prime cell's
own row**, replayed thereafter. The K=1 tier's row at M is never consulted
for the inner again. Nothing structural remains: the balanced-pair rule and
the default chain become CANDIDATES in the race, not answers.

## The race

- **When**: inside `_ilprime_create_banked`, on a miss or under
  `recalibrate`. ONE race over every buildable (method, inner) pair -- Rader's
  pool at M = N - 1 beside Bluestein's at the next power of two >= 2N - 1 --
  so the method is decided WITH its best inner, not with a borrowed one. A
  row from before 2026-09-18 (an `eng=` with no inner) replays its method and
  races that method's inners only.
- **Candidates** (built DIRECTLY, no planner call, so the re-entrancy lock is
  never touched -- the four-step's `_k1fs_sb_chains` is the model):
  every legal `il2p` pair at M (the pool `_ilprime_inner_make` already
  enumerates to pick the balanced one), the `il3p` default chain at M, and
  at a power-of-two M every ZTURN-T registry chain at M, untiled and at each
  legal tile of the ladder `{1024, 2048}` -- the K=1 planner's own pool
  (`_il_dp_enumerate_ztt_ord`), scrambled class, since the convolution is a
  matched roundtrip in any order. The pool is COMPLETE (the K=1 sink's
  law: a capped pool is banked as "the best of a prefix"); its size at the
  gate's cells: 9, 12, 19, 63, 108 arms.
  The pool is a pool; nothing in it is a default.
- **What is timed**: the whole convolution -- `vfft_ilprime_execute` forward
  on the assembled prime plan with that inner installed -- not the inner
  standalone. That is the workload the verdict serves.
- **How**: `support/race.h`, the one race body (min-of-3, alternated, one
  warm pass), in HEATS: the race body takes 32 arms and a prime plan at
  M = 262144 costs ~22 MB, so the pool lives as DESCRIPTORS
  (`_ilprime_inner_desc_t`: kind, shape, tile), each heat of 16 is built,
  raced same-run, and reduced to its winner (the rest destroyed), and the
  heat winners meet in one same-run final. A candidate that does not build
  is dropped, never substituted: the descriptor provider flags a failed
  build so the create's structural fallback is discarded rather than served
  under the candidate's name.
- **What banks**: on the prime-method row, beside `eng=`: `in=2p|3p|ztt`
  (the inner's kind), `in_sh=R1.R2 | R2.A.B | the chain`, `in_tw=` (ZTURN-T's
  tile, 0 = untiled). The signpost `ref=` that pointed at M's row is retired
  with the borrowing it served.

## What moves, and what does not

- `_ilprime_inner_from_wisdom` and its context struct are DELETED -- the
  read of M's row, the `M > 4096` cliff, the call into `_k1_il_candidate`.
  In their place `_ilprime_inner_from_desc` builds exactly the descriptor it
  is handed. Layer 3's enumeration survives as `_ilprime_inner_cands`; its
  "pick the most balanced" line does not. `_ilprime_inner_make`'s structural
  rule still stands behind `vfft_ilprime_create` (no store) only.
- `vw2_prime_method_bank/lookup` grow the inner tokens (L10 territory: the
  reader, the writer, the serializer -- one row, no new row family).
- `il_prime.h`'s standalone entry (`vfft_ilprime_create`, no wisdom) keeps
  building SOMETHING for callers with no store, but that is the split
  library's / the bare API's concern and is out of this design's scope.

## Gate

`benches/ilprime_inner_gate.c`: for a spread of primes whose M spans the
bands (Rader and Bluestein, M below and above 4096), cold store:
- the cold create logs the inner race and banks `in= in_sh=` on the prime
  row;
- forward matches the naive DFT (N <= 4099);
- the warm create logs a replay, no race, and is BITWISE the cold forward;
- `recalibrate=1` on the warm store logs the race again -- the flag reaching
  the prime cell is the claim, and the race line under the flag is its
  direct proof (a poisoned legal token would be overwritten by the same
  line; latency is not consulted).
Watched to FAIL before it counted (2026-09-18, the bank disabled): every
cell failed twice -- `prime row has no in= in_sh= tokens`, `warm create
raced 1 / replayed 0`.

Evidence (2026-09-18): 31, 127, 257, 4099, 65537 ALL PASS, 5.3 s for the
five cells x three creates. Verdicts: 31 Rader `2p 3.10`; 127 Rader
`2p 6.21`; 257 Rader `2p 4.64`; 4099 Bluestein `ztt 8.8.8.8.4 @ 2048`; 65537
Bluestein `ztt` (Rader's M = 65536 lost to Bluestein's 262144 with its own
pool). The first build capped the pool at 24 arms and the first method's
arms could fill it before the second ran -- 13-28 candidates dropped at
M >= 16384, loudly; the heats replaced the cap. One thing seen, not a
defect of this design: at 65537 two cold races banked two chains
(`8.8.8.4.8.4.4 @ 1024`, then `8.8.8.8.8.8 @ 2048`) -- the one-sample cold
race, `policy_survey_defects.md` section F.

## Against MKL (2026-09-19)

`build_tuned/prime_vs_mkl.sh` and the canonical bench, `--k1noop` (natural,
out of place, T = 1, pace 300 ms, core 2 + HIGH, MKL single-threaded), one
process per cell on a scratch copy of the shipped store, the cold race banked
by a front-door create beforehand, both engine orders shown.

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

The split between the two methods is not written anywhere. Rader turns a
prime into a cyclic convolution of length N - 1, so it can only be built when
the inner pool can express N - 1, and it is offered as race arms exactly then.
Below 4096 the pool draws on every il2p pair and the il3p chain, so any radix
with a codelet counts -- which is why 521 rides `8.5.13`. Above 4096 the inner
must be a ZTURN-T chain, whose odd mids are 3, 5, 7, 9 and 15, so a prime
whose N - 1 carries 13, 17, 683 or 257 offers Rader nothing and Bluestein
takes the cell unopposed and correctly. That is the textbook rule -- smooth
predecessors favour Rader, rough ones favour Bluestein -- arrived at by
construction rather than by a threshold.

Three defects were found on the way there, all on 2026-09-19, and the table
above is after all three:

- **Rader carried a hard ceiling**, `if (nm1 > 4096) return 0;`, a vestige of
  the STRUCTURAL inner: that rule could only build a balanced pair, and a pair
  of radices <= 64 tops out at 4096. Since the inner became a raced pool the
  ceiling was obsolete, and it made every larger prime bank Bluestein without
  a contest. At 65537 Rader offered 108 inners and built NONE, so Bluestein
  convolved at 262144 where Rader convolves at 65536: 1.04x vs MKL became
  4.82x, our own time 1.71 ms to 0.38 ms.
- **The inner pool was a hand-rolled copy** of the registry walk and the tile
  ladder, and it knew only power-of-two chains. It now calls the K=1 planner's
  own two enumerators, `_il_dp_enumerate_ztt_ord` and
  `_il_dp_enumerate_ztt_odd` -- one law, one place -- which is what gives
  Rader the smooth non-power-of-two predecessors: 4001, 12289 and 40961 all
  flipped from Bluestein to Rader, at 1.76x, 2.22x and 3.62x.
- **The banked METHOD used to filter the race.** A binary that could not build
  that method was left with an empty race and REFUSED the cell -- a store
  calibrated against a wider pool, or an engine whose reach changed, turned
  into "no interleaved engine". Reaching the race at all means the row could
  not be replayed, so its method is a preference, not a measurement of this
  build: a verdict that cannot be built is a miss, and a miss races.

And one non-defect, recorded so it is not re-derived: 131071 read 0.88x once,
against 1.07-1.24x on three fresh races that all picked the same chain. A
single cold race is one sample (`policy_survey_defects.md` section F); it was
not a regression.

The method axis now obeys the no-silent-caps law: the race logs each method's
pool and how much of it built, and warns when a method was offered arms and
built none -- the exact shape of the 65537 defect.

## Cost, honestly

A race per prime cell, once: about a second at 65537 (108 arms in eight
heats plus the final), well under at the small primes. The wisdom row format
changes (additive tokens; old rows without `in=` replay their method and
race the inner once). The K=1 tier's rows at M lose one consumer and nothing
else. The serializer is generic (`vw2_rec_set`), so the risk sat in the
create; the negative test and the full sweep covered it.

## Checklist

- [x] 1. This design.
- [x] 2. `_ilprime_inner_cands` (2026-09-18): every legal pair, the default chain, ZTURN-T's registry chains untiled and at each legal tile.
- [x] 3. The race in `_ilprime_create_banked`; `in= in_sh= in_tw=` on the prime row (`vw2_prime_method_bank` / `vw2_prime_inner_lookup`).
- [x] 4. Layers 1-2 deleted; `_ilprime_inner_from_desc` builds exactly one inner from a descriptor and flags a failed build so the structural fallback is never served in its place.
- [x] 5. The gate, watched to fail (2026-09-18).
- [ ] 6. Full sweep; records (`design_contracts.md` section 6, `docs/roadmap/policy_survey_defects.md`, memory).
