# src/estimate -- the 1D estimate planning method: the real transforms and the tools

This folder is tracked but not wired: nothing in the library, the build, the gates or the gauntlet
includes or builds it. The library refuses `VFFT_FLAG_ESTIMATE` on R2C and C2R (`_vfft_api_translate`,
`src/core/vfft_api.h`). Lifting that refusal and wiring this folder's real build is left to the user.

Estimate is a planning method the user selects. It is never a fallback. From N, the request and
the machine's cache sizes it computes the recipe a race would have banked: the route and every
parameter. It does this without measuring. The plan is then built through the library's own
replay code.

**The 1D complex part is released (2026-10-22) as `VFFT_FLAG_ESTIMATE` (`include/vfft.h`):**

| tracked file | what it holds |
|---|---|
| `src/core/il/estimate/estimate_il.h` | the machine `est_machine_t`, the request `est_request_t`, the c2c recipe `est_recipe_t`; the c2c formulas `est_plan_1d` and every helper they use; the c2c row writer `est_put_rows` |
| `src/core/il/estimate/c2c_estimate_il.h` | the c2c build `est_build_c2c` (pre-flight, private store, the door's replay, the detectors `est__meter_read` / `est__replay`, the plan check), the environment guard `est_env_names` / `est_env_steers`, and the front door's `_vfft_create_estimate` |
| `src/core/il/estimate/estimate_nd_il.h` | rank 2 and 3: the recipes `est_plan2d_t` / `est_plan3d_t`, the formulas `est_plan_2d` / `est_plan_3d` (the column chain from the door's pool, the N-arm, the bands, the 2D row route, the 3D natural form; the 1D children by `est_plan_1d`), the row writers `est_put_rows_2d` / `est_put_rows_3d` |
| `src/core/il/estimate/c2c_estimate_nd_il.h` | rank 2 and 3: the build `est_build_nd` (pre-flight, rows, the door's replay, the plan check) and `_vfft_create_estimate_nd` |
| `src/core/vfft_api.h` | the flag's refusals (`_vfft_api_translate`) and the dispatch (`vfft_create`) |
| `src/tools/gates/estimate_gate.c` | the release gate (registered in `run_gates.py`) |

The flag serves one interleaved c2c transform (howmany 1): 1D in any order, placement and thread count;
rank 2 and 3 (2026-10-22, `docs/design/estimate_2d3d_sketch.md`) in natural order on one thread, either
placement. It refuses real, real-to-real, rank 4, howmany > 1, SPLIT, scrambled order or nthreads > 1 at
rank 2 and 3, the flag with `VFFT_FLAG_RECALIBRATE` or `VFFT_FLAG_PATIENT`, and any create while an
environment variable that steers a plan is set (`est_env_names`). At rank 2 and 3 the formulas also
refuse an axis of length 1, a last axis with no 1D engine, and a column with no chain whose Bluestein
length the column pool cannot chain (above 2^24). It never opens `wisdom_dir`.

**This folder keeps what is not released:** the r2c / c2r formulas and build, built on top of the
tracked c2c part, and the tools.

- Estimate never reads the user's wisdom store. The raced rows are only the evidence the formulas
  were fitted to.
- Specification: `docs/design/estimate_1d_sketch.md`.

## Files

| file | what it is |
|---|---|
| `estimate_recipe.h` | The recipe of any request, `est_any_recipe_t`: the library's c2c recipe as its member `c`, plus the real engines' fields. It declares this folder's contracts. It needs the library's types: include it only after `vfft.c`. |
| `estimate_1d.h` | The real formulas and the contracts `est_plan_any` (c2c: the library's `est_plan_1d`) and `est_recipe_print`. Pure: no clock, no store, no allocation. |
| `estimate_build.h` | The build of any request: `est_build_any`, and `est_build_any_ex`, which also returns a report. c2c goes through the library's `est_build_c2c`; real through the real build here, on the library's detectors. It compiles only inside a TU that includes `vfft.c`. |
| `estimate_tool.c` | The tool: `show`, `check`, `compare` (below). The TU includes `src/core/vfft.c` itself. |
| `estimate_selftest.c` | The recipe printer with its own law pass. Now a library TU too: the c2c formulas read the library's facts. |
| `estimate_race.c` | Timed: the banked plans against the estimate plans (E through the public flag). |
| `estimate_race_nd.c` | The same at rank 2 and 3: 2D and 3D power-of-two cells, the same protocol. Built, never run by its author: the owner times on a quiet machine. |
| `estimate_store.c` | Writes the c2c estimate plans as a store on disk, for the gauntlet. |
| `TEST_LOG.md` | Every test sweep before the release, every bug found, and the review fixes. |

## How a plan is built

**c2c (the library, `src/core/il/estimate/c2c_estimate_il.h`):**

1. **Refuse what estimate does not serve.** The front door refuses the requests the flag does not cover.
2. **Pre-flight.** The door's own constructor sequence runs at T = 1, and what it built is destroyed.
   A recipe the door would mark stale, and then race, is refused here instead. A four-step child (the
   four-step, a four-step inner of the prime cell) is a nested create, so it is not built here: its
   column chain is checked with the 2D tier's resolver (with `fs_blu=0` / `fs_forms` where the tier
   needs them) and its row plan at N2 by this same pre-flight.
3. **Rows into a private store.** The recipe is written as the rows a race would bank, into a private
   store held in memory (`vfft_child_store_new`), with the library's own writers (`est_put_rows`): the
   prime method row, the c2c route row, the T row's `il_mt` / `il_mt_tw` / `il_mtsb`, the four-step's
   child in the `fs_rp_*` spelling a race banks today, the `dir=bwd` row.
4. **The library's own create.** `_vfft_create_cfg` runs against that store, nested (the whole build
   runs inside a create) and with `wisdom_write = 0`.
5. **No-measurement proof.** Read before the pre-flight and after the create: `_vfft_create_race_count`,
   `vfft_measure_scopes()` (the build runs inside a create, so any clock read enters a scope), the
   private store's row count, `_def_loaded` and `_vfft_api_nstores`. Any change: the plan is destroyed
   (`EST_MEASURED`). A row the door re-banked with another payload: destroyed (`EST_REWRITTEN`).
6. **The recipe's engine and threaded arm.** Else the plan is destroyed (`EST_SERIAL` / `EST_NOPLAN`).
   `vfft_create` arms the caller's pin of a T > 1 plan.

**This folder's additions (`estimate_build.h`), for c2c and real alike:** the clock probe (every library
clock read, `VFFT_CLOCK_PROBE`: a read during a build is `EST_MEASURED`), the environment guard (any
create-steering `VFFT_*` variable refuses the build; REPLAY_API 1.3), and the caller's pin.

**Real** follows the same steps on the library's pieces (`est__meter_read`, `est__replay`,
`est__fs_preflight`, `est__fs_text`): the real pre-flight builds the door's real builders at T = 1 against
an empty private store; the rows are the real writers' (`vw2_real_il_bank_*`); a four-step child (zfsr,
zrb's four-step inner, a zr2c four-step child) is checked without a create, as above.

## Build and run

Build from the repository root. Do not pass `--vfft`: each TU includes `vfft.c` itself. Each build
takes about two minutes.

```
python gauntlet/build.py --compile --src src/estimate/estimate_tool.c
python gauntlet/build.py --compile --src src/estimate/estimate_selftest.c
```

The exe lands beside the source. Run it from the Bash tool, where the mingw DLLs are on `PATH`.

```
src/estimate/estimate_selftest.exe -all 1024 4099 2^20
src/estimate/estimate_selftest.exe -o scr -T 8 -sweep 2 300000
src/estimate/estimate_tool.exe show 1024 1009 2^20 -p oop -o nat
src/estimate/estimate_tool.exe check 2-4096 -t c2c -o nat -p oop -q
src/estimate/estimate_tool.exe check 2-2048 -t r2c,c2r -q
src/estimate/estimate_tool.exe check 262144-16777216:pow2 -T 1,8
src/estimate/estimate_tool.exe compare -q
```

The formulas follow the build's lanes (`est_plan_1d` refuses a machine of another count), so the selftest's
`-vw` and the tool's `--lanes` must name the build's: an 8-lane law pass needs an AVX-512 build
(`VFFT_ISA=avx512`) and an AVX-512 host or Intel SDE.

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
- `--l1d B --l2 B --l3 B`: plan for another machine's caches (`show` and `compare` only).

### `check`

Each cell is built through `est_build_any` and reported as one of:
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

## What passed before the release (2026-10-22, after the review fixes; re-verified in the third pass)

These runs were made on the pre-release code. The release moved the c2c part into the library with
no change of recipe: a dump of every recipe (c2c four cells and real, T = 1 and 8, N = 2..20000, the
powers of two and their neighbours to 2^25, 3000 random N) at L1d 16, 48 and 64 KiB is identical
before and after (2,606,190 lines).

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

- **A misleading message.** The four-step replay prints "did not replay -- it raced" whenever its
  2D child re-wrote its own row, even when nothing raced (`k1_commit.h:1558-1564`,
  `k1_fourstep.h:600-612`). The glue's `rp_` spelling avoids that path.
- **The default-store detector sees a first load only.** Once a process has loaded its default
  store (`vfft.c:495-506`), a nested create that fell to `_default_wisdom()` (`vfft.c:2276`) would
  read it unseen. No IL nested create does today (each sets `.wisdom`), and the tool never loads
  the default store. The library's est_build_c2c relies on that code fact (stated in c2c_estimate_il.h).
- **What the store banks for real rows.** Every stored r2c row has `il_bkv=0` and every stored c2r
  row `il_kv=0`: a race banks only the forms its direction runs (`wisdom2_real_il.h:378-386`).
  Estimate writes both slots (sketch section 6). Plans are unaffected; only the recipe match rate
  moves.
- **Env pins beat banked rows silently.** `VFFT_IL_KV` and the engine pins (`k1_commit.h:85, 146`,
  `zrp_build.h:971-1027`, `odd_build.h:678-697`) override a replayed row without marking it stale.
  This is why the estimate refuses while a steering `VFFT_*` name is set: this folder's build, and
  the released `VFFT_FLAG_ESTIMATE` since 2026-10-22 (`est_env_names`, c2c_estimate_il.h).

## What is unfitted

- Every rule at T other than 1 and 8 (open decision 2). The thread rules rest on one CPU at T = 8.
- Every 8-lane rule (open decision 7). This machine builds AVX2 only, so 8-lane recipes were checked
  by the formulas' law pass, not built.
- Every L1d outside 32-63 KiB (s1 other than 11). The tile is clamped to {1024, 2048} (open
  decision 3); the scrambled chain at those sizes was checked by the law pass only.
- The in-place four-step split (open decision 6) and the threaded four-step's child (open
  decision 4; `fs_wl` 1/4 in the sketch).
- zrf's LEVELS test on T (the compromise above).
