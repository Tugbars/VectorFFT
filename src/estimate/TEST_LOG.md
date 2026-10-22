# src/estimate -- test log (2026-10-22)

This log covers the build-and-test pass over the 1D estimate (`estimate_1d.h`, `estimate_build.h`,
`estimate_tool.c`, `estimate_selftest.c`).

The pass checked correctness only. Nothing was timed or benchmarked, and no gauntlet run or gate
was used. Every plan was built from the glue's private in-memory store. Scratch material (lists,
logs, scripts) is in the session scratchpad under `estimate_impl/test/`.

**Machine:** the tool's own read is `lanes 4 (avx2 build), L1d 49152, L2 2097152, L3 37748736,
cores 24` (i9-14900KF). The 8-lane rules could not be built here (open decision 7).

## Result

(This is the first pass. The second pass, after the review fixes, is the last section, "Review
fixes": 29,367 cells, 0 measured / rewritten / wrong / crashed.)

On the first pass's final binary, every cell is one of two things:
- built and correct;
- refused with the one reason the task intends: the in-place scrambled prime cell at T = 1. Its
  route row and its prime method row share one key (`wisdom2_oop_il.h:515-527, 545-551`).

**Totals:** 27,967 cells, 27,055 built and correct, 912 refused for that reason. **0 measured, 0
wrong, 0 crashed, 0 glue refusals.** The worst relative error is 1.1e-14 (c2c N = 65537 T = 8,
Rader). Every other sweep's worst error is at most 5.2e-15. The tolerance is 1e-11.

All 14 routes and engines were built: mono, 2p, chain3, flat, ztt, prime (Rader and Bluestein;
2p, 3p, ztt and fs inners), fs, zrm, zrp, zttr, zr2c, zfsr, zrf and zrb.

## Builds

| what | command | result |
|---|---|---|
| formulas self-test | `gcc -std=c11 -Wall -Wextra -pedantic -O2 -o <scratch>/estimate_selftest.exe src/estimate/estimate_selftest.c -lm` (C:/mingw152/mingw64/bin/gcc.exe 15.2.0) | clean, no warnings |
| tool | `python gauntlet/build.py --compile --src src/estimate/estimate_tool.c` (no `--vfft`) | OK, about 63 s per compile. The only warnings build.py prints are in library code (`il2d_tier.h:1021`). |
| tool, extra warnings | the same compile command with `-Wall -Wextra` added, into a scratch object | 3 warnings in our files, all fixed (bug 4 below). The rest are library code. |

The tool runs with or without `C:/mingw152/mingw64/bin` on `PATH`.

## Sweeps (final binary; T = 1 unless stated)

All sweeps use `src/estimate/estimate_tool.exe check ...`.

| # | cells | command | cells | built, correct | refused (intended) | measured / wrong / crashed | worst error |
|---|---|---|---|---|---|---|---|
| F1 | c2c natural out of place, every N 2..4096 | `check 2-4096 -t c2c -o nat -p oop` | 4095 | 4095 | 0 | 0 / 0 / 0 | 4.7e-15 |
| F2 | c2c, every N 2..512, all four order/placement cells | `check 2-512 -t c2c -o nat,scr -p oop,ip` | 2044 | 1854 | 190 | 0 / 0 / 0 | 2.2e-15 |
| F3 | c2c powers of two 2..2^22, all four cells | `check 2-4194304:pow2 -t c2c -o nat,scr -p oop,ip` | 88 | 88 | 0 | 0 / 0 / 0 | 2.0e-15 |
| F4a | c2c, 100 random composites in 4097..2^20, all four cells | `check <list_c2c_comp> -t c2c -o nat,scr -p oop,ip` | 400 | 305 | 95 | 0 / 0 / 0 | 3.9e-15 |
| F4b | c2c, 100 random 2^a x odd in 4097..2^20 | the same with `list_c2c_pow2odd` | 400 | 340 | 60 | 0 / 0 / 0 | 4.0e-15 |
| F4c | c2c, 100 random composites with a prime factor > 47 | the same with `list_c2c_bigp` | 400 | 300 | 100 | 0 / 0 / 0 | 4.2e-15 |
| F4d | c2c, 100 primes up to 2^20 (20 below 4097) | the same with `list_c2c_primes` | 400 | 301 | 99 | 0 / 0 / 0 | 3.8e-15 |
| F5 | r2c and c2r, every N 2..4096, both placements | `check 2-4096 -t r2c,c2r -p oop,ip` | 16380 | 16380 | 0 | 0 / 0 / 0 | 5.1e-15 |
| F6 | r2c and c2r, 100 random N up to 2^20, both placements | `check <list_real_r100> -t r2c,c2r -p oop,ip` | 400 | 400 | 0 | 0 / 0 / 0 | 5.0e-15 |
| F7 | T = 8: 20 c2c N (all four cells), 20 r2c N and 20 c2r N (both placements) | `check <list_t8_*> ... -T 8` | 160 | 160 | 0 | 0 / 0 / 0 | 1.1e-14 |
| F8 | regression: the glue writer's set, 400 random N in 2049..262144, every transform and cell | `check 2049-262144:r400` | 3200 | 2832 | 368 | 0 / 0 / 0 | 5.2e-15 |
| X | extra: the F7 lists at T = 4, natural, out of place | `check <list_t8_*> -o nat -p oop -T 4` | 60 | 60 | 0 | 0 / 0 / 0 | -- |

**What the refusals were.** Every refusal (912 of 912) is the formulas' refusal of the in-place
scrambled prime cell at T = 1. No sweep logged a door re-write note.

**What the lists cover.** The lists are drawn with fixed seeds (`gen_lists.py`, `gen_lists2.py`
in the scratch folder).
- F6 holds:
  - 25 even N;
  - 10 even N with a 47-smooth child;
  - 10 even N with a prime-cell child;
  - 15 odd smooth N (zrf);
  - 10 odd composite prime cells and 15 odd primes (zrb, several with a four-step inner);
  - the powers of two 2^13, 2^15, 2^17, 2^19 and 2^20 (zttr; zfsr at 2^20);
  - 10 N of any kind.
- F7 holds these hand-picked N:
  - 233073, the zrf case the glue writer reported;
  - 251621, a zrb with a four-step inner 768.512;
  - non-power-of-two four-steps (360000, 483153);
  - Rader and Bluestein primes (7919, 65537, 100003, 300007, 1000003);
  - zfsr at 2^20 to 2^22;
  - zr2c children of the mono, flat, ZTURN-T, chain3, prime and four-step routes;
  - zttr, and zrf at the serial bound.

**The formulas' own law pass** (`estimate_selftest -sweep`, after the fixes): 0 refusals and 0 law
breaks at each of these settings:
- c2c 2..300000 at T = 1;
- c2c scrambled 2..300000 at T = 8;
- r2c and c2r 2..400000 at T = 8;
- c2c 2..300000 at 8 lanes;
- r2c 2..400000 at 8 lanes.

## First pass (before the fixes)

F1, F2, F3, F5, F6 and F7 were clean on the binary this pass started with. F4 was not:
- F4a: 9 glue refusals (three N, three cells each);
- F4b: 81 glue refusals (27 N, three cells each).

All 90 had the same reason: "the four-step's column chain at N1=... carries an odd radix and its
2D row no N1-arm verdict fs_blu=0". Each one is a non-power-of-two four-step with N1 = 576, 640,
768, 957, 1331 and so on, in the natural and scrambled out-of-place cells and the scrambled
in-place cell. The natural in-place cell takes the flat DIT (sketch 3.6). Nothing measured: the
glue's pre-flight refused these cells before any create.

## Bugs found and fixed (all in src/estimate/)

1. **Formulas: the four-step child had no N1-arm verdict (`fs_blu`).**
   - **Problem:** the 2D tier races a column chain that carries an odd radix against a Bluestein
     column unless the 2D row says which one won (`il2d_tier.h:3607-3650`; the verdict is read by
     `vw2_ilcol_chain_lookup`, `wisdom2_2d_il_reader.h:135`). Table 5.7 of the sketch does not
     list the token.
   - **Fix:** `est_fs_child` (`estimate_1d.h`) writes `fs_blu=0` (the chain won, which is the
     sketch's own `fs_chain`) whenever a column stage is odd. This is the spelling a race banks;
     see the stored 2D rows, e.g. `chain=5.3.9 ... blu=0`.
   - **Reach:** it covers the c2c four-step, the zr2c four-step child, zfsr, and the four-step
     inner of a prime cell or of zrb.
   - **Result:** the 90 F4 refusals became 90 built and correct cells. 0 measured.
2. **Formulas: no `fs_forms` for a column stage with rival forms.**
   - **Problem:** a stage whose radix has several column forms races them unless the row carries
     `forms=` (`il2d_tier.h:3218-3250`). The rivals come from `vfft_il2p_col_forms`
     (`il2p.h:481-487`): radix 32 has b48 and b84, every other radix one form.
   - **Fix:** `est_fs_child` writes `fs_forms` when a stage is 32. It spells every stage by its
     incumbent, as the race does (`il2d_tier.h:2483, 2597-2598`): b48 for 32, b88 for 64, `-` for
     the others.
   - **Reach:** no N that the sweeps or a sampled scan of 262145..2^24 covered needs it. The column
     chain folds to 32 only past four stages, which needs N1 well above the balanced split
     (`find_forms.c` in the scratch folder: 0 cells, against 670,425 cells with `fs_blu`). It is
     kept so that the rule is complete. The glue's check remains.
3. **Formulas: zrf LEVELS (`mt=2`) was emitted where it cannot bind.**
   - **Problem:** `vfft_zrf_mt_bind` declines LEVELS unless the first level is tiled with at least
     one tile per worker (`zrf_mt.h:133-156`). A race banks only an arm that bound: the stored
     T = 8 zrf rows carry mt=2, mt=1 or no mt. The glue writer found N = 233073 (9.19.29.47,
     budget 1024) refused as silently serial: no stage span fits the budget.
   - **Fix:** `est_real` computes the first level's tile as the code does. The tile is the first
     span R_s·D_s (s = 1..K-2, D_s > 4) within the budget (`zrf.h:255-262`). The tile count is
     h·D/tw (`zrf.h:329`). `mt=2` is emitted only when that count is at least T; otherwise the row
     stays serial.
   - **Compromise (flagged):** where LEVELS would decline but FIRST (`mt=1`) would bind, the race
     may have banked mt=1. The sketch names no rule for it, so estimate stays serial there.
   - **Result:** r2c and c2r N = 233073 at T = 8 built and correct with mt=0. 0 measured.
4. **Tool and glue: compiler warnings under `-Wall -Wextra`.**
   - `est__fs_row_recipe` (`estimate_build.h`): its prefix buffers were 16 bytes and could cut
     `fs_rp_<int>_`. They are now 24.
   - `est_store_load` (`estimate_tool.c`): it copied a 1024-byte path into the store's 512-byte
     `dir`. It now refuses a path that does not fit, rather than naming another folder.

`README.md` is updated to match. Its "Found while building" section now lists these fixes and the
library behaviour.

## Library limitations and behaviour (recorded, not patched)

- **The in-place scrambled prime cell at T = 1 cannot replay.** Its route row and its prime
  method row share one key: `vw2_oop_bank_k1_il` keeps the method row
  (`wisdom2_oop_il.h:515-527`, key `:545-551`), and the door's scan takes only eng=k1 rows
  (`wisdom2_oop_rows.h:103`). So the door races it (`k1_commit.h:993-1004`). Estimate refuses
  the cell (912 refusals above). At T > 1 the cell builds (F7).
- **A misleading message.** The four-step replay prints "did not replay -- it raced" whenever its
  2D child re-wrote its own row, even when nothing raced (`k1_commit.h:1555-1560`,
  `k1_fourstep.h:610-612`). The glue's `rp_` spelling avoids that path. No sweep showed it.
- **What the store banks for real rows.** Every stored r2c row has `il_bkv=0` (2000/2000) and
  every stored c2r row has `il_kv=0` (997/997): a race banks only the forms its direction runs
  (`wisdom2_real_il.h:378-386`). Estimate writes the child's computed forms in both slots, as
  sketch section 6 says for c2r. Plans are unaffected. Only the whole-recipe match moves: 98
  stored r2c rows differ from estimate's in `il_bkv` alone. This is reported, not changed.
- **8 lanes.** This machine builds AVX2 only. The 8-lane rules (open decision 7) were checked only
  by the formulas' law pass, not built.

## Compare against the stored winners (read only; section 8 of the sketch)

Store: `src/wisdom/14900KF`, read with the library's shard loader (22425 rows). The formulas plan
for its `@meta` machine, which is this one. "Recipe" is every payload token of every row the glue
banks.

| cells | compared | route | recipe | sketch section 8 (i9) |
|---|---|---|---|---|
| c2c natural out of place, 2..4096 | 3798 | **91.3%** (3466) | **32.4%** (1232) | route 91.6%, whole 33.3% (all 3813 rows) |
| c2c natural out of place, every stored row | 3827 | 91.3% (3493) | 32.4% (1239) | the same |
| r2c 2..4096, both placements | 3037 | **99.3%** (3015) | **29.3%** (891) | engine 99.3% (3046/3067), whole 37.0% |
| r2c 2..4096, out of place | 3030 | 99.4% (3012) | 29.4% (891) | the same |
| r2c, every stored out-of-place row | 3067 | 99.3% (3046) | 29.1% (892) | the same |
| c2r, every stored out-of-place row | 2069 | 99.3% (2055) | 22.5% (465) | engine 99.3% (2055/2069), whole 27.5% |

The stored in-place rows hold only 7 r2c rows in 2..4096.

Per stored route, c2c natural out of place 2..4096 (cells / route / recipe):
- mono 22 / 22 / 22;
- 2p 260 / 260 / 140;
- ztt 25 / 25 / 14;
- flat 438 / 148 / 35 (the sketch's weak spot 1: flat wins over pair and chain3);
- prime 2421 / 2409 / 815;
- chain3 632 / 602 / 206.

Per stored engine, r2c 2..4096 (cells / route / recipe):
- zrm 20 / 19 / 19;
- zr2c 2002 / 1995 / 599;
- zrf 314 / 314 / 137;
- zrp 16 / 16 / 10;
- zrb 658 / 658 / 122;
- zttr 27 / 13 / 4.

**Reading the numbers.**
- The c2c route and recipe rates match section 8 to within 1 point.
- The real engine rates match section 8 exactly.
- The real whole-recipe rate is 6-8 points under the model's. The leading first differences are
  the prime child's inner shape (`il_prime_sh` 639, `in_sh` 212) and `il_bkv` on r2c rows (201;
  see "What the store banks for real rows"). Section 8's figures come from the Python model,
  which predates the text's corrections.

## Hygiene

- `git -C C:/Users/Tugbars/Desktop/highSpeedFFT status --short` shows only ` M .gitignore`.
- `src/wisdom` is unchanged: an md5 of all 24 files before and after matches (the files include
  the ignored `14900KF/wisdom2.lock`, which was already there).
- `compare` opens the shards with `"rb"` only. It does not call `vw2_open`.
- No create in this pass read the clock: 0 cells measured. Unlike the earlier glue step, this pass
  started no aborted races.

## Review fixes (2026-10-22, second pass)

An independent review raised 12 findings. Each was checked against the sketch's text and the
library code, then fixed in `src/estimate/` or answered below. Nothing was timed. Scratch material
is in the session scratchpad under `estimate_impl/fix/`: the formula diffs (`diff_*.txt`), the sweep
logs, `unit_fix.c` and its output.

### The findings

| # | finding | verdict | what changed |
|---|---|---|---|
| 1 | The c2c prime cell's T > 1 arm read the serial bound at the inner's M | **Fixed.** Section 7 (line 306) reads the bound at N for c2c. | `est_inner_mt` takes the bound's length apart from M. The c2c cell passes N. The zr2c prime child keeps its M, as section 7's zr2c row words it; this is flagged to the owner. On this machine the two never differ for a zr2c child. |
| 2 | The scrambled ZTURN-T chain found its tiled part by "prefix product == tile" | **Fixed.** | The natural chain now reports its tiled part (the ingest 8 plus fill(s-3)), and exactly that part is moved. At s1 = 11 nothing changes. At L1d 64 KiB, 8192 is now 8.8.4.8.4 and 65536 is 8.8.8.4.8.4. At 16 KiB, 2048 is 8.8.4.8, 16384 is 8.8.8.4.8 and 131072 is 8.8.8.8.4.8. |
| 3 | zrf LEVELS only where it binds | **Kept, as a compromise for the owner.** The T in the test is the library's own bind law (`_zrf_mt_level_ok`, `zrf_mt.h:133-136`). Written literally, those cells would get a silently serial plan, which the glue refuses. | Comment and README say so. |
| 4 | `fs_blu` / `fs_forms` are not in table 5.7 | **No code change.** The tokens are needed: without them the 2D tier races at create. | Comment and README record them as additions the sketch should take. The sketch is a tracked file and was not edited. |
| 5 | zfsr at T > 1 filled a gap of the sketch | **Fixed:** the gap is reported, not filled. | The formulas refuse zfsr at T > 1 with the reason. A race at T picks the split and the 2D child at T (`zrp_build.h:316-318, 518-530`), and the text gives neither. |
| 6 | The contract said odd real N past (3N-1)/2 <= 2^24 is refused | **Fixed** (a comment). | Only an odd prime cell (zrb) is refused there. Odd smooth N (zrf) has no ceiling but its 10 stages. |
| 7 | The environment guard matched upper case only | **Fixed.** Confirmed: on Windows `getenv` ignores case. | Names are matched without case on Windows and exactly elsewhere. `vfft_il_kv=0` and `Vfft_Zrp=1` are now refused. A lower-case `vfft_wisdom_dir` stays harmless. |
| 8 | A scrambled c2c forward above 4096 was never compared with a DFT | **Fixed.** | Each of the 32 sampled reference bins must be found among the N outputs. A NaN anywhere now fails the cell: the old worst-of could hide a NaN Parseval behind a finite bin error. |
| 9 | The contract named `vfft_child_store_raced` as a detector that the code did not use | **Fixed.** | Every "own" row the door re-banks is compared with the row the glue wrote. The same payload is a note. A changed payload is the new status `EST_REWRITTEN`: the plan is destroyed, and the tool prints `rewritten(FAIL)`. The contract now says this, and the file names it gave are corrected. |
| 10 | The pre-flight did not check the outer engine's inner-length law for a four-step inner | **Fixed.** | The checks are now: Rader M = N - 1, Bluestein an even M >= 2N - 1 (prime cell and zr2c prime child), and zrb M >= (3N-1)/2. A wrong M is refused before any create. |
| 11 | The default-store detector sees only a first load | **Confirmed latent; documented, not changed.** | No IL nested create reaches `_default_wisdom()` today, and the tool never loads the default store. A stronger detector would swap the library's default-store state around the create, and on a hit that re-opens the user's folder from disk. Recorded in the contract, the glue header and README. |
| 12 | Unchecked allocations in the tool | **Fixed.** | `est_xmalloc` / `est_xrealloc` end the run cleanly, as `est_alloc` does. `strdup` is gone. |

### Builds

- **Tool:** `python gauntlet/build.py --compile --src src/estimate/estimate_tool.c` -- OK in 90 s.
  The only warning is in library code (`il2d_tier.h:1021`).
- **Tool with `-Wall -Wextra`:** compiled into a scratch object from build.py's own argument list
  (`fix/warn_build.py`). No warning in `src/estimate/`.
- **Formulas self-test:** gcc 15.2.0 with `-std=c11 -Wall -Wextra -pedantic`. Clean.
- **`unit_fix.c`** (scratch; it includes the tool's TU). 12/12 passed:
  - the sampled scrambled check passes a permuted DFT (9e-17) and fails a rotated forward (1e-2)
    and an all-NaN forward;
  - a re-bank with the same payload (reordered, plus a measurement token) reads the same, and a
    changed `il_kv` reads changed;
  - wrong inner lengths are refused before any create, with 0 clock reads: a Bluestein fs inner
    (N = 1000003), a Rader fs inner (N = 15417851), a zrb fs inner (r2c 251621) and a zr2c prime
    child's fs inner (r2c 600014);
  - the environment guard refuses a lower-case `vfft_il_kv` and passes a lower-case
    `vfft_wisdom_dir`.

### What the formula changes touch

Every cell (all transforms, orders and placements) of N = 2..300000 was dumped before and after
the change and diffed (`fix/dump.c`):

| setting | rows changed |
|---|---|
| T = 1, this machine | 0 |
| T = 8 | 12504: every c2c prime cell with N in 2049..6143 (3126 N x 4 cells), il_mt 2/1 -> 0. Nothing else. |
| L1d 64 KiB | 4: scrambled 8192 and 65536 |
| L1d 16 KiB | 6: scrambled 2048, 16384, 131072 |
| 8 lanes, T = 8 | 12516: the c2c prime cells' il_mt only |

zfsr starts at N = 2^20, past the dump. Its T > 1 refusal is in F10 below.

### Sweeps (final binary)

| # | cells | built, correct | refused | measured / rewritten / wrong / crashed | worst error |
|---|---|---|---|---|---|
| F1 c2c nat oop 2..4096 | 4095 | 4095 | 0 | 0 / 0 / 0 / 0 | 4.7e-15 |
| F2 c2c all four cells 2..512 | 2044 | 1854 | 190 | 0 / 0 / 0 / 0 | 2.2e-15 |
| F3 c2c pow2 2..2^22, all four | 88 | 88 | 0 | 0 / 0 / 0 / 0 | 2.0e-15 |
| F4a comp | 400 | 305 | 95 | 0 / 0 / 0 / 0 | 3.9e-15 |
| F4b pow2odd | 400 | 340 | 60 | 0 / 0 / 0 / 0 | 4.0e-15 |
| F4c bigp | 400 | 300 | 100 | 0 / 0 / 0 / 0 | 4.2e-15 |
| F4d primes | 400 | 301 | 99 | 0 / 0 / 0 / 0 | 3.8e-15 |
| F5 real 2..4096 | 16380 | 16380 | 0 | 0 / 0 / 0 / 0 | 5.1e-15 |
| F6 real r100 | 400 | 400 | 0 | 0 / 0 / 0 / 0 | 5.0e-15 |
| F7 T = 8 c2c / r2c / c2r | 80 / 40 / 40 | 80 / 36 / 36 | 0 / 4 / 4 | 0 / 0 / 0 / 0 | 1.1e-14 |
| F8 2049..262144 r400, every cell | 3200 | 2832 | 368 | 0 / 0 / 0 / 0 | 5.2e-15 |
| X T = 4, the F7 lists (c2c / r2c / c2r) | 20 / 20 / 20 | 20 / 18 / 18 | 0 / 2 / 2 | 0 / 0 / 0 / 0 | 1.1e-14 |
| F9 new: T = 8 c2c 2049..6144 r200, all four | 792 | 792 | 0 | 0 / 0 / 0 / 0 | 2.3e-15 |
| F9b new: T = 8 c2c 6145..8192 r50, all four | 200 | 200 | 0 | 0 / 0 / 0 / 0 | 2.4e-15 |
| F10 new: zfsr N = 2^20..2^25, r2c/c2r, both placements, T = 1, 8 | 48 | 24 | 24 | 0 / 0 / 0 / 0 | 9.7e-16 |
| F11 new: c2c scrambled 4097..2e6 r150, both placements | 300 | 153 | 147 | 0 / 0 / 0 / 0 | 4.3e-15 |

**Totals:** 29,367 cells. 28,272 built and correct. 1,095 refused by the formulas:
- 1,059 in-place scrambled prime cells at T = 1;
- 36 zfsr cells at T > 1 (finding 5): the zfsr N of the F7 and X real lists (r2c 2^20 and 2^21,
  c2r 2^20 and 2^22) and all of F10's T = 8 cells.

**0 measured, 0 rewritten, 0 wrong, 0 crashed, 0 glue refusals.** No sweep logged a re-bank note
or a library "did not replay" message.

- 1,343 scrambled c2c cells above N = 4096 are now gated on their forward. The worst sampled-bin
  error there is 4.3e-15.
- F9: 632 prime cells at T = 8 below the bound, all built serial. F9b: 172 prime cells above it,
  all built with their threaded arm.

### The formulas' law pass

`estimate_selftest -sweep`: 0 refusals and 0 law breaks in each of these settings:
- c2c 2..300000 at T = 1;
- c2c scrambled 2..300000 at T = 8;
- r2c and c2r 2..400000 at T = 8;
- c2c 2..300000 at 8 lanes;
- r2c 2..400000 at 8 lanes;
- c2c 2..300000 at T = 8;
- new: c2c scrambled 2..300000 at L1d 64 KiB, at 16 KiB, and at 8 lanes with 64 KiB.

### Compare (read only, `src/wisdom/14900KF`)

The old and the new binary give identical rates. No changed cell has a stored row: the store
holds no T = 8 prime row and no T > 1 zfsr row.

| cells | compared | route | recipe | sketch section 8 |
|---|---|---|---|---|
| c2c nat oop 2..4096 | 3798 | 91.3% | 32.4% | route 91.6%, whole 33.3% |
| c2c nat oop, every stored row at T = 1 | 3827 | 91.3% | 32.4% | the same |
| r2c 2..4096, both placements | 3037 | 99.3% | 29.3% | engine 99.3%, whole 37.0% |
| r2c, every stored oop row at T = 1 | 3067 | 99.3% | 29.1% | the same |
| c2r, every stored oop row at T = 1 | 2069 | 99.3% | 22.5% | engine 99.3%, whole 27.5% |
| every stored T = 8 row | 191 | 61.3% | 13.1% | -- (the route misses are stored flat wins: weak spot 1) |
| every stored row, every T | 14587 | 92.7% | 30.2% | -- |

### Hygiene

- `git -C C:/Users/Tugbars/Desktop/highSpeedFFT status --short` shows only ` M .gitignore`.
- `src/wisdom` is unchanged: the md5 of all 24 files was taken before this pass and after it, and
  the two match.
- Since the build, only comments changed (`estimate_build.h:45`, README, this log).

## Third pass (2026-10-22): the review fixes verified

This pass checked the same 12 review findings against the code as it now stands. Every fix of the
second pass is in place, and no code changed in this pass. Nothing was timed. Scratch material is
in the session scratchpad under `estimate_impl/finish/`: `run_finish.sh`, the sweep logs `s_*.txt`,
the compare logs `c_*.txt`, `summary.txt` and `selftest_sweeps.txt`.

### What each finding looks like now

| # | finding | checked how | state |
|---|---|---|---|
| 1 | c2c prime cell's T > 1 bound read at M | `estimate_selftest -T 8 4099 6143 6151`: il_mt 0, 0, 2 | fixed |
| 2 | scrambled ZTURN-T tiled-part search | `-l1d 65536 -o scr 8192 65536` gives 8.8.4.8.4 / 8.8.8.4.8.4; `-l1d 16384 -o scr 2048 16384` gives 8.8.4.8 / 8.8.8.4.8 | fixed |
| 3 | zrf LEVELS only where it binds | r2c 13125 and 233073 at T = 8 are serial (mt=0). 15625, 253125, 390625, 455625, 531441 and 1594323 take mt=2 and build (F12) | kept, a compromise for the owner |
| 4 | `fs_blu` / `fs_forms` not in table 5.7 | comment and README | documented as additions for the sketch |
| 5 | zfsr at T > 1 filled a gap | r2c 2^20 at T = 8 is refused, with the reason | fixed: a gap reported, not filled |
| 6 | contract: the odd real ceiling | `estimate_recipe.h:172-175`; r2c 29296875 plans zrf | fixed (a comment) |
| 7 | env guard matched upper case only | the tool refuses `vfft_il_kv=0` and `Vfft_Zrp=1`; it builds with `vfft_wisdom_dir` set | fixed |
| 8 | scrambled c2c above 4096 had no forward check | `check 8192 -o scr` prints `bins 4.2e-16`; no c2c cell in any sweep prints `bins -` | fixed |
| 9 | the contract named a detector the code did not run | `estimate_build.h:1246-1277`: an own row is compared by payload (EST_REWRITTEN) | fixed |
| 10 | inner-length law not pre-checked | unit test: 4 wrong-M recipes refused, 0 clock reads | fixed |
| 11 | default-store detector sees only a first load | `estimate_build.h:18-26`, contract step 3 | latent; documented |
| 12 | unchecked allocations in the tool | `est_xmalloc` / `est_xrealloc`; `est_add_N` checks its realloc | fixed |

### Builds

- **Tool:** rebuilt with `python gauntlet/build.py --compile --src src/estimate/estimate_tool.c`. OK
  in 66 s. The only warning is in library code (`il2d_tier.h:1021`). The new binary has the same
  code as the second pass's binary: `cmp` finds only 4 differing bytes, all in the header.
- **Formulas self-test:** gcc 15.2.0 with `-std=c11 -Wall -Wextra -pedantic -O2`. Clean.
- **`unit_fix.exe`** (second pass, scratch): re-run, 12/12 passed.

### Sweeps (rebuilt binary)

F1-F11 and X are the same commands as in "Review fixes", and they gave the same counts and the same
worst errors. Two sweeps are new:

| # | cells | built, correct | refused | measured / rewritten / wrong / crashed | worst error |
|---|---|---|---|---|---|
| F12 new: zrf at T = 8, the LEVELS law (13125 15625 233073 253125 390625 455625 531441 1594323), r2c/c2r, both placements | 32 | 32 | 0 | 0 / 0 / 0 / 0 | 4.6e-15 |
| F13 new: c2c scrambled 2^13..2^24, both placements, T = 1, 8 | 48 | 48 | 0 | 0 / 0 / 0 / 0 | 2.0e-15 |

**Totals:** 29,447 cells. 28,352 were built and correct. 1,095 were refused by the formulas:
- 1,059 in-place scrambled prime cells at T = 1;
- 36 zfsr cells at T > 1.

**0 measured, 0 rewritten, 0 wrong, 0 crashed, 0 glue refusals.** No log has a re-bank note or a
library "did not replay" message. All 1,391 scrambled c2c cells above N = 4096 were gated on their
forward.

### The formulas' law pass

`estimate_selftest -sweep` showed 0 law breaks in each of 13 settings:
- the 10 settings of the second pass;
- c2c scrambled in place at T = 1. It refuses exactly the 282,438 prime cells, as intended;
- c2c natural in place at T = 8;
- c2r at 8 lanes and T = 8.

The other 12 settings refuse nothing.

### Compare (read only, `src/wisdom/14900KF`)

The rates are the same as in the second pass:
- c2c natural out of place, 2..4096: route 91.3%, recipe 32.4% (3,798 cells);
- c2c, every stored row at T = 1: route 91.3%, recipe 32.4% (3,827);
- r2c 2..4096: route 99.3%, recipe 29.3% (3,037);
- r2c, every stored out-of-place row at T = 1: route 99.3%, recipe 29.1% (3,067);
- c2r: route 99.3%, recipe 22.5% (2,069);
- every T = 8 row: route 61.3%, recipe 13.1% (191);
- every row: route 92.7%, recipe 30.2% (14,587).

Sketch section 8 gives route 91.6% / whole 33.3% for c2c, and engine 99.3% with whole 37.0% (r2c)
and 27.5% (c2r) for real.

### Hygiene

- `git -C C:/Users/Tugbars/Desktop/highSpeedFFT status --short` shows only ` M .gitignore`.
- `src/wisdom`: the md5 of all 24 files is the same at the start and at the end of this pass. It
  is also the same as the 13:10 baseline (`estimate_impl/glue/wisdom_before.md5`).
