# src/estimate -- the 1D estimate planning method (UNRELEASED)

This folder is git-ignored (`.gitignore:87`, `/src/estimate/`). Nothing in it is part of the
library, the build, the gates or the gauntlet.

Estimate is a planning method the user selects. It is never a fallback. From N, the request and
the machine's cache sizes it computes the recipe a race would have banked: the route and every
parameter. It does this without measuring. The plan is then built through the library's own
replay code.

- Estimate never reads the user's wisdom store. The raced rows are only the evidence the formulas
  were fitted to.
- Specification: `docs/design/estimate_1d_sketch.md`. Where its text and the Python model under
  `docs/research/estimate_1d/synth/` differ, the text wins.
- Build reference: `docs/research/estimate_1d/impl/REPLAY_API.md`.

## Files

| file | what it is |
|---|---|
| `estimate_recipe.h` | The interface. It defines the machine (`est_machine_t`), the request (`est_request_t`, the row's key) and the recipe (`est_recipe_t`, the row's payload). It declares the three contracts. Plain C, with no library type. |
| `estimate_1d.h` | The formulas: `est_plan_1d` and `est_recipe_print`. They are pure: no library, no clock, no store, no allocation. |
| `estimate_build.h` | The glue: `est_build_1d`, and `est_build_1d_ex`, which also returns a report. It turns a recipe into a public `vfft_plan` with no measurement, or refuses with the reason. It compiles only inside a TU that includes `vfft.c`. |
| `estimate_tool.c` | The tool: `show`, `check`, `compare` (below). The TU includes `src/core/vfft.c` itself. |
| `estimate_selftest.c` | A library-free printer for the formulas (`estimate_1d.h`), with its own law pass. |
| `TEST_LOG.md` | Every test sweep, every bug found, and the review fixes. |

## How the glue builds a plan (`estimate_build.h`)

1. **Refuse what estimate does not serve.** This covers a transform other than c2c/r2c/c2r, a
   scrambled real request, and the in-place scrambled prime cell at T = 1. That cell's route row
   and prime method row share one key (`wisdom2_oop_il.h:515-527,545-551`), so the library would
   race it. The glue also refuses whenever any `VFFT_*` environment variable other than the
   logging ones or the store-directory ones is set (REPLAY_API 1.3). On Windows the names are
   matched without case, because `getenv` there ignores case (`vfft_il_kv` is `VFFT_IL_KV`).
2. **Pre-flight.** The door's own constructor sequence runs at T = 1, with no clock, and what it
   built is destroyed. A recipe the door would mark stale, and then race, is refused here instead.
   - A four-step child (the c2c four-step, zfsr, a four-step inner of the prime cell or of zrb, a
     zr2c four-step child) is a nested create, so it is not built here. It is still checked
     without a create:
     - the outer engine's inner-length law: Rader M = N - 1 (`il_prime.h:299-318`,
       `k1_commit.h:288-289`), Bluestein an even M >= 2N - 1 (`il_prime.h:249`), zrb
       M >= (3N - 1)/2 (`zrb.h:59,87`);
     - its column chain, with the 2D tier's own resolver;
     - for a chain with an odd radix, the N1-arm verdict `fs_blu=0`;
     - for a stage with several column forms, `fs_forms`;
     - its row plan at N2, by this same pre-flight.
3. **Rows into a private store.** The recipe is written as the rows a race would bank, into a
   private store held in memory (`vfft_child_store_new`), using the library's own writers. Those
   rows are:
   - the c2c route row;
   - the T row's `il_mt` / `il_mt_tw` / `il_mtsb`;
   - the prime method row;
   - the `dir=bwd` row;
   - the real row.

   A four-step child is written the way a race banks it today. Its row plan rides inside the 2D
   row as `fs_rp_*` (`wisdom2_child.h:5-21`). The recipe's `fs_row_*` spelling is converted to
   that form. Without the conversion, zfsr would re-race its own replay (`zrp_build.h:330-334`).
4. **The library's own create.** `_vfft_create_cfg` (`vfft.c:2509`) runs against that store, with
   howmany = 1. It runs nested (the create depth is raised) and with `wisdom_write = 0`. The user's
   store is never opened, read or written.
5. **No-measurement proof.** These are read before the pre-flight and after the create:
   - every library clock read (`VFFT_CLOCK_PROBE`);
   - `_vfft_create_race_count`;
   - `vfft_measure_scopes()`;
   - the private store's row count;
   - whether the library's default store or a directory store was opened.

   If any of them changed, the plan is destroyed and the status is `EST_MEASURED`.

   A row the door re-banked at an existing key (an "own" row, `wisdom2.h:189`) is compared with
   the row the glue wrote. The same payload is only reported. A changed payload means the plan is
   no longer the recipe, even though nothing measured: the plan is destroyed and the status is
   `EST_REWRITTEN`. The door can re-bank without a clock read in two places
   (`k1_commit.h:1558-1564`; `il2d_tier.h:4413-4424`, cmt=0 when the 2D MT arm cannot engage).
6. **The recipe's engine and threaded arms.** The handle must be the recipe's engine. Every
   threaded arm the recipe asked for must have taken. A silently serial plan is refused
   (`EST_SERIAL`). A T > 1 plan gets the caller pin that `vfft_create` gives it
   (`vfft_api.h:207-213`).

## Build and run

Build from the repository root. Do not pass `--vfft`: the TU includes `vfft.c` itself. It takes
about 90 s.

```
python gauntlet/build.py --compile --src src/estimate/estimate_tool.c
```

The formulas' self-test needs no library:

```
gcc -std=c11 -Wall -Wextra -pedantic -O2 -o <scratch>/estimate_selftest.exe src/estimate/estimate_selftest.c -lm
<scratch>/estimate_selftest.exe -all 1024 4099 2^20
<scratch>/estimate_selftest.exe -o scr -T 8 -sweep 2 300000
```

The exe lands beside the source. Run it from the Bash tool, where the mingw DLLs are on `PATH`.

```
src/estimate/estimate_tool.exe show 1024 1009 2^20 -p oop -o nat
src/estimate/estimate_tool.exe check 2-4096 -q
src/estimate/estimate_tool.exe check 262144-16777216:pow2 -T 1,8
src/estimate/estimate_tool.exe check 2049-262144:r400 -t r2c,c2r -q
src/estimate/estimate_tool.exe compare -q
src/estimate/estimate_tool.exe compare 2-4096 -t c2c -o nat -p oop -v
```

**N:** `1024`, `2^20`, `A-B` (every N), `A-B:pow2`, `A-B:sK` (step K), or `A-B:rK` (K values drawn
with a fixed seed).

**Options:**
- `-t c2c,r2c,c2r`: the transforms.
- `-o nat,scr`: the orders. Real is natural only.
- `-p oop,ip`: the placements.
- `-T 1,8`: the thread counts. `compare` with no `-T` takes every stored T.
- `--tol 1e-11`: the relative error a check accepts.
- `--isolate`: run every cell in its own process.
- `-q` / `-v`: quieter / louder output.
- `--store DIR`: the store folder for `compare`.
- `--l1d B --l2 B --l3 B --lanes 4|8`: plan for another machine (`show` and `compare` only).

### `check`

Each cell is built through `est_build_1d` and reported as one of:
- built, with its errors;
- `refused(formula: ...)` or `refused(glue: ...)`;
- `measured(FAIL)`;
- `rewritten(FAIL)`: the door replaced a recipe row without measuring;
- `wrong(FAIL)`.

There is no timing.

What a built cell is checked against:
- **The forward.** It is compared with a long-double DFT:
  - N <= 4096: every bin (scrambled: the sorted multiset of the outputs);
  - above 4096: 32 bins by direct long-double sums. Natural: each at its position. Scrambled:
    each must be found among the N outputs (the scrambled contract is any permutation,
    `vfft.c:1922-1928`, so a round trip alone cannot gate it).
- **A Parseval check.**
- **A forward-backward round trip.**
  - c2c: on the plan itself.
  - Real: the cell's r2c plan feeds the c2r plan. This is not done when the r2c recipe holds a
    nested create.

A NaN in any of the three fails the cell.

**What ends the process.** A create that reads the library's clock has started a race. The probe
ends the process at that first read (exit code 3), and the cell is reported as `measured(FAIL)`.
A cell whose recipe holds a nested create always runs in its own process, so the sweep goes on.

### `compare`

This mode reads this CPU's folder under `src/wisdom/`, READ ONLY. It names the folder with the
library's own folder finder and loads the shards with the library's shard loader. It never calls
`vw2_open`, which sweeps temporary files.

The formulas plan for the machine named in the folder's `@meta` stamp. For each cell, the rows the
glue would bank are compared with the stored rows at the same keys.

It reports:
- route matches (`il_route`, or the real `eng`);
- recipe matches (every payload token of every row, with the four-step child's two spellings
  read as one);
- a per-route table.

With no N given, every stored winner row of the asked transforms is a cell.

## What passed (2026-10-22, after the review fixes; re-verified in the third pass)

On the final binary every check cell is built and correct, or refused with an intended reason:
**0 measured, 0 rewritten, 0 wrong, 0 crashed, 0 glue refusals.** The full tables are in
`TEST_LOG.md` ("Review fixes" and "Third pass").

- Every sweep (F1-F11, X, plus F12-F13 of the third pass): 29,447 cells, 28,352 built and correct,
  1,095 refused by the formulas (1,059 in-place scrambled prime cells at T = 1, 36 zfsr cells at
  T > 1). The worst relative error is 1.1e-14 (c2c N = 65537, T = 8, Rader); the tolerance is
  1e-11. All 1,391 scrambled c2c cells above N = 4096 are gated on their forward.
- The formulas' law pass (`estimate_selftest -sweep`): 0 law breaks in all 13 settings, including
  L1d = 16 KiB and 64 KiB and 8 lanes. The only refusals are the in-place scrambled T = 1 setting's
  282,438 prime cells, which are intended.
- A scratch unit test of the fixes passed 12/12 (TEST_LOG.md).
- `compare` against `src/wisdom/14900KF` (read only), against sketch section 8:

  | cells | route | recipe | sketch section 8 |
  |---|---|---|---|
  | c2c natural out of place, T = 1, every stored row (3827) | 91.3% | 32.4% | route 91.6%, whole 33.3% |
  | r2c out of place, T = 1 (3067) | 99.3% | 29.1% | engine 99.3%, whole 37.0% |
  | c2r out of place, T = 1 (2069) | 99.3% | 22.5% | engine 99.3%, whole 27.5% |
  | every stored row, every T (14587) | 92.7% | 30.2% | -- |

## Defaults taken for the sketch's open decisions (section 12)

Each is marked "open decision N, default" in the code that relies on it.

1. **One transform only** (howmany = 1). Batches are refused. The request has no batch field, and
   the glue always creates howmany = 1.
2. **Any T > 1 uses the T = 8 fitted rules.** They are marked unfitted for T != 8. The glue writes
   the T row the recipe gives, and checks that its threaded arm took.
3. **The ZTURN-T tile is clamped** to the planner's raced ladder {1024, 2048}
   (`dp_planner_il.h:1455`) when the L1 formula leaves it. This is done in the formulas.
4. **The threaded four-step carries the threaded split's own child.** The glue seeds the child
   under the threaded split (`il_mt`) and T, as the door replays it (`k1_commit.h:1538-1552`).
5. **The prime-pool members of 12.5 are never emitted.** The primes the sketch refuses stay
   refused. The glue rejects a flat recipe that carries a Rader-stage inner (`il_rd`).
6. **The in-place four-step reuses the out-of-place split.**
7. **The 8-lane rules are written as specified** (legality only, unfitted). On an 8-lane build
   the pre-flight refuses whatever the build lacks (zrp, ZTT-r).

Also, the in-place scrambled prime cell at T = 1 is refused with its reason (step 1 above).

## For the owner: gaps, additions and one compromise

Gaps in the text are reported, not filled:
- **zfsr at T > 1 is refused.** Section 7 has no zfsr row. A race at T picks the split at T
  (`zrp_build.h:518-530`) and seeds the 2D child under T (`:316-318`), so neither follows from the
  text. The cells are r2c/c2r with N/2 a power of two in 2^19..2^24, at T > 1.
- **The zr2c prime child's serial bound** is read at the child's M, as section 7's zr2c row words
  it ("at the child's M"). The c2c prime cell reads it at N (section 7, line 306). On this machine
  the two never differ for a zr2c child. Which one is meant is the owner's call.

Additions the sketch should record (without them the 2D tier races at create):
- **`fs_blu=0`** on a four-step child whose column chain carries an odd radix (e.g. N1 = 768 =
  8.8.4.3). Without the 2D row's N1-arm verdict, the 2D tier races the chain against a Bluestein
  column (`il2d_tier.h:3607-3650`). 0 = the chain won, which is the sketch's `fs_chain`.
- **`fs_forms`** on a four-step child with a radix-32 column stage, spelled with the incumbent
  forms a race banks (`b48`; `il2d_tier.h:3218-3250, 2483`). No N that the sweeps sampled needs it.

A compromise to confirm:
- **zrf LEVELS (`mt=2`) only where the library can bind it.** Section 7 says mt=2 above the bound.
  The library's LEVELS needs the first level tiled with a tile per worker (`_zrf_mt_level_ok`,
  `zrf_mt.h:133-136`), so the rule depends on T through the library's own law. Where it cannot
  bind (e.g. r2c 13125 or 233073 at T = 8), estimate leaves the row serial. Written literally,
  those cells would build silently serial plans, which the glue refuses.

## Known library limitations (reported, not patched)

- **The in-place scrambled prime cell at T = 1 cannot replay.** Its route row and its prime
  method row share one key: `vw2_oop_bank_k1_il` keeps the method row
  (`wisdom2_oop_il.h:515-527`, key `:545-551`), and the door's scan takes only eng=k1 rows
  (`wisdom2_oop_rows.h:103`). So the door races it (`k1_commit.h:993-1004`). Estimate refuses it.
- **A misleading message.** The four-step replay prints "did not replay -- it raced" whenever its
  2D child re-wrote its own row, even when nothing raced (`k1_commit.h:1558-1564`,
  `k1_fourstep.h:600-612`). The glue's `rp_` spelling avoids that path.
- **The default-store detector sees a first load only.** Once a process has loaded its default
  store (`vfft.c:495-506`), a nested create that fell to `_default_wisdom()` (`vfft.c:2276`) would
  read it unseen. No IL nested create does today (each sets `.wisdom`), and the tool never loads
  the default store. A library caller of `est_build_1d` relies on that code fact.
- **What the store banks for real rows.** Every stored r2c row has `il_bkv=0` and every stored c2r
  row `il_kv=0`: a race banks only the forms its direction runs (`wisdom2_real_il.h:378-386`).
  Estimate writes both slots (sketch section 6). Plans are unaffected; only the recipe match rate
  moves.
- **Env pins beat banked rows silently.** `VFFT_IL_KV` and the engine pins (`k1_commit.h:85, 146`,
  `zrp_build.h:971-1027`, `odd_build.h:678-697`) override a replayed row without marking it stale.
  This is why the glue refuses every steering `VFFT_*` name.

## What is unfitted

- Every rule at T other than 1 and 8 (open decision 2). The thread rules rest on one CPU at T = 8.
- Every 8-lane rule (open decision 7). This machine builds AVX2 only, so 8-lane recipes were checked
  by the formulas' law pass, not built.
- Every L1d outside 32-63 KiB (s1 other than 11). The tile is clamped to {1024, 2048} (open
  decision 3); the scrambled chain at those sizes was checked by the law pass only.
- The in-place four-step split (open decision 6) and the threaded four-step's child (open
  decision 4; `fs_wl` 1/4 in the sketch).
- zrf's LEVELS test on T (the compromise above).
