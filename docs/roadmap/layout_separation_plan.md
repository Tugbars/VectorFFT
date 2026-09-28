# Separating the interleaved (IL) machinery from the split machinery in `src/core`

Status (2026-09-27): phases 0-3 DONE, each gated at avx2 and avx512 by
`src/tools/baseline/step_gate.py`:
- phase 0: the gate. Self-checked on the unchanged tree: every rung PASS at both ISAs.
- phase 1: dead code deleted (commit "core: phase 1"). Semantics identical;
  code changed only where the deletion reaches.
- phases 2-3: 102 files moved into `common/`, `split/`, `il/` by
  `docs/roadmap/layout_separation/moves_phase2_3.map`. **Byte-identical** at both
  ISAs: the `vfft.c` objects at -O2 and at -O3 -march=native, every codelet
  object and library, and the harness binaries.

The remaining mixed files sit in the old folders; `python src/tools/baseline/hygiene.py`
lists the 12 includes that still cross the layout line, which is the work of
phases 4-6.

Originally: plan only, no code moved. The research was done by reading the code (four
read-only audits: `oop/`+`engine/`+`primes/`; `transforms/`; `planning/`+`support/`+
`wisdom2/`+top level; and the regression tooling). Line numbers are as of branch
`claude/upbeat-carson-qmcclb` at the time of writing and will drift; the seams they point
at will not.

## Status (2026-09-27)

**Phases 0-7 are done**, each step gated at avx2 and avx512 (R5 identical
throughout); from phase 7.3 the gate runs with `--enforce-deps` and the dependency
rules are green. The tree as built is described in `src/core/README.md`. Where it
differs from section 4:

- the two sides of the fork are `split/split_create.h` + `split/split_execute.h` and
  `il/il_create.h` + `il/il_execute.h` (unique basenames, rule 6);
- `bridge/` holds `real_bridge.h` + `real_bridge_exec.h` (B1/B3 and the smooth-odd
  race; temporary per D1). B4 (`nat_ilp.h`) is gone with D2. B2 (the IL 2D
  real ROWSPLIT reaching into a split child) is still inside `il2d_tier.h`: a call,
  not an include, so the checker does not see it;
- `vfft_internal.h` (the plan struct) is in `common/plan/`, shared by both layouts
  while D3 is deferred (phase 8 skipped for now);
- `wisdom2/` stays at the front: the legacy kind-3 reader and migration, the OOP
  codec aggregator and two gates span both layouts.

Sections 1-3 are the pre-separation survey and keep their original paths.

## 0. Summary

- `src/core` is 123 files, about 64,000 lines. Most files already belong to one layout:
  - **SPLIT only:** all of `engine/` and `primes/`, the split OOP files in `oop/`, the
    stride planners, and most of `transforms/`.
  - **IL only:** the K=1 IL engines in `oop/`, `dp_planner_il.h`, the il2d files and
    `fftnd_il.h`.
  - **NEUTRAL:** most of `support/` and the wisdom2 store core.
- The two layouts are tangled in about **20 mixed files**. The tangle has five sources:
  1. **Create tiers.** Five create tiers hold both layouts in one function:
     - `c2c_oop_create.h`
     - `c2c_ip_create.h`
     - `fft2d_create.h`
     - `fftnd_create.h`
     - `real_create.h`
  2. **Front door.** `vfft.c` and `vfft_execute.h` re-test `layout` at every tier, and
     there is no single dispatch point.
  3. **Plan struct.** `struct vfft_plan_s` has about 15 common fields, about 40 split
     fields and about 330 IL fields, all interleaved.
  4. **Shared vocabulary in the wrong home.** Examples: the 11-arg ABI typedef, the K=1
     route ids of both layouts, the IL wisdom kinds and the IL natural modes. These live
     in split files (`oop_leaf_registry.h`, `oop_plan.h`, `wisdom_reader.h`).
  5. **Wisdom codecs.** They put both layouts' rows in one file:
     - `wisdom2_oop.h`
     - `wisdom2_oop_reader.h`
     - `wisdom2_2d_reader.h`
     - the IL rows parked in `wisdom2_stride_reader.h` and `wisdom2_real_reader.h`
- **No engine of one layout calls an engine of the other through its internals, with
  three deliberate exceptions.** These are the owner's "IL at the boundary, split inside"
  composites, all in the real transforms and 2D real:
  - IL 1D real at K>1 is served by the split r2c/c2r engines through their `_z` doors.
  - The IL 2D real ROWSPLIT route runs a split child plus the `_rowz` doors.
  - The odd-N real bridge builds an IL c2c child that also serves split callers.

  They stay, but become a declared interface instead of struct reach-through.
- **Target:** `core/common/`, `core/split/` and `core/il/`, each organized by rank and
  engine, behind a thin front door that dispatches once on layout.
  - **Dependency rule:** `split` and `il` depend on `common` and never on each other.
    `common` depends on neither.
  - Only the front door (and a small, named `bridge/` for the three composites) sees
    both.
- **Migration:** 9 phases. Each step is gated by a refurbished `src/tools/baseline/`
  (section 7). Most steps are pure moves, so their proof is **byte identity** of the
  compiled library. Only the steps that change code (splitting functions, the struct
  split, de-duplication) fall back to object-level and semantic equivalence.

## 1. File map

Class key: **S** split, **IL** interleaved, **N** neutral, **M** mixed (seams in
section 2), **dead** (not compiled, or no callers).

### Top level

| file | lines | class | note |
|---|---|---|---|
| `vfft.c` | 2324 | M | Front door. Holds the create dispatcher, split helpers (`_build_2d`, `_calibrate_c2c`, `_pad_ladder`, `_exec_c2c_inplace`, the natorder MT), IL helpers (TC wrapper `_tc_*`, the ZTT MT probe, IL counters), the both-layout odd-real bridge `_oddr_build`, the wisdom bundle, and the fingerprint body. |
| `vfft_internal.h` | 516 | M | `vfft_wisdom_s` (split legacy tables plus the neutral `vw2`) and `vfft_plan_s` (section 2.1). |
| `vfft_execute.h` | 1463 | M | `_vfft_sig_bad`, `vfft_execute` (IL arms tested first, split falls through), the IL bind `_vfft_k1_bind_exec`, and `vfft_destroy`. |
| `vfft_batch.h` | 300 | S | Padded batch. IL is refused at line 210. |
| `vfft_fingerprint.h` | 84 | N | Declarations only. The body in `vfft.c` prints the fields of both layouts. |

### `engine/` — S (all)

- `executor.h`, `executor_generic.h`, `mt_execute.h`, `planner.h`, `twiddle.h`,
  `proto_stride_compat.h`, `plan.h`.
- Neutral code inside:
  - the aligned allocator (`plan.h:11-29`);
  - `STRIDE_ALIGNED_ALLOC` (`proto_stride_compat.h:24-43`);
  - `twiddle.h` defines its own `M_PI`.
- `compat.h` is **dead** (an empty placeholder).

### `primes/` — S (all)

- `bluestein.h`, `rader.h`, `prime_dispatch.h`, `bluestein_wisdom.h`,
  `bluestein_calibrator.h`.
- Neutral code inside: number theory in `rader.h:73-125` and `prime_dispatch.h:29-47`.
- `il/prime` (`il_prime.h:50-88`) duplicates the number theory on purpose, to avoid the
  coupling.

### `oop/`

| file | class | note |
|---|---|---|
| `oop_execute.h`, `oop_auto.h`, `oop_dp.h`, `oop_mt.h` | S | Split OOP (MODEB, the pair tuner, DP champions, lane-slice MT). |
| `oop_plan.h` | M | The split `vfft_oop_plan_t`, plus the route-id enums of **both** layouts and the IL wisdom kind `ZR2C`. |
| `oop_leaf_registry.h` | M | The neutral 11-arg typedef, split resolvers, and the IL mono/solo resolvers (which `#include "il_isa.h"`). |
| `k1_commit.h` | M | IL replay/race/bank, plus the split `_bank_nat_1d` and the bridge `_ilp_ref_of`. |
| `c2c_ip_create.h` | M | IL in-place create, split owned-batch arm, split stride arm. |
| `c2c_oop_create.h` | M | One K=1 block for both layouts, then the classic split OOP. |
| `il_isa.h`, `avx512/vtw_avx512.h` | IL | The IL ISA surface and the AVX-512 twiddle builders. |
| `il2p.h`, `il_flatdit*.h`, `il_prime.h`, `ztt.h`, `ztt_mt.h`, `k1_fourstep*.h`, `il2d_proto.h` | IL | The K=1 IL engines. `il2d_proto.h` is gate-only. |
| `ztt_qw16384.h` | IL | Neutral data, but only IL uses it. |
| `tw_exact.h` | N | Once-rounded cos/sin. It lives in `oop/`, and only IL uses it today. |

### `planning/`

| file | class | note |
|---|---|---|
| `dp_planner.h`, `exhaustive_plan.h`, `measure.h`, `pad_calibrate.h`, `adopt_wisdom.h` | S | Stride-engine search. |
| `wisdom_reader.h` | S* | Legacy stride codec. Its `VFFT_NAT_*` enum carries the IL modes ZCASC/ILP/CONV. |
| `dp_planner_split_oop.h` | S* | Includes `dp_planner_il.h` for no IL symbol (it only needs the wisdom2 OOP reader). |
| `dp_planner_il.h`, `il_slot_probe.h` | IL | The K=1 IL race. `dp_planner_il.h` includes `oop_plan.h` only for the IL route ids. |
| `policy.h` | M | L4 order, the cell and L8 cache laws are neutral. L9, L10 and the family/band/admit laws are IL. |

### `support/`

- N: `build_isa.h`, `cpu_cache.h`, `diag.h`, `race.h`, `race_timing.h`, `ref.h`
  (gate-side), `slot_check.h`, `threads.h`, `zalloc.h`.
- M: `env.h`. Part 1 is N. Part 2, the depth and prune knobs, is S (used by
  `exhaustive_plan.h`).
- S: `strided_codelets.h`, a split kernel registry type that sits in `support/` by
  accident.

### `wisdom2/`

| file | class | note |
|---|---|---|
| `wisdom2.h` | N | The store: key (with the `vw2_lay_t` axis), records, shards, lookup, save. |
| `wisdom2_selftest.h`, `wisdom2_migrate.h` | N | Store gates. The legacy migrator writes `lay=ANY`. |
| `wisdom2_oop.h` | M | `vfft_oop_wisdom_entry_t` carries both layouts' kind-3 fields. It includes `oop_plan.h` (a split plan type) for the enums. |
| `wisdom2_oop_reader.h` | M | Split classic kinds 0/1/2 and `lay=split` K=1 rows; IL `lay=il` K=1, bwd, prime method and zr2c rows. |
| `wisdom2_2d_reader.h` | M | Split 2D/3D codecs about lines 1-500; IL `ilcol`/`ilnd`/`forms` codecs about 500-1057. |
| `wisdom2_stride_reader.h` | S* | Hosts the IL TC-MT verdict rows (552-612). |
| `wisdom2_real_reader.h` | S* | Hosts the IL oddr row (244-290). |
| `wisdom2_fftnd.h` | S | Rank ≥ 2 split entries and builders. |
| `wisdom2_2d_gate.h`, `wisdom2_real_gate.h` | M (test) | Gates. |

### `transforms/`

| file | class | note |
|---|---|---|
| `fft2d/fft2d.h`, `fft2d_c2c_planner.h`, `fft2d_c2r_planner.h`, `fft2d_r2c.h`, `fft2d_r2c_planner.h`, `strided_tw.h` | S | Split 2D. |
| `fft2d/transpose.h` | N by nature, S by use | Double-plane transposes. |
| `fft2d/plane_queue.h` | N | The howmany > 1 queue over any plan. Its only layout use is the wisdom key. |
| `fft2d/il2d_col.h`, `il2d_cols.h`, `il2d_tier.h`, `fft2d_real_il.h` | IL | The IL 2D tier. `il2d_tier.h` holds the ROWSPLIT bridge (section 3). |
| `fft2d/fft2d_create.h` | M | One 1,230-line function holding both layouts (section 2.4). |
| `fft3d/fft3d.h`, `strided_rows.h` | S | Split 3D. |
| `fftnd/fftnd.h` | S | Rank-general split c2c. |
| `fftnd/fftnd_r2c.h` | S | Carries a **dead** IL bridge: `il_out`, `stride_plan_nd_r2c_il`. |
| `fftnd/fftnd_il.h` | IL | Rank-3 IL c2c. |
| `fftnd/fftnd_create.h` | M | IL peel at lines 53-62; split below it. |
| `fftnd/fftnd_natorder.h`, `fftnd_planner.h`, `fftnd_wisdom.h` | dead | Orphans: nothing includes them, except each other. |
| `natorder/*` | S | Except the index math in `natorder_perm.h` (`mk_perm`, `inv_perm`, cycles, pairs), which is N. |
| `real/r2c.h`, `rfft.h`, `c2r.h`, `r2c_dispatch.h`, `c2r_dispatch.h` | S + IL doors | Split engines whose `zo`/`zi`/rowz modes are fused into pack/unpack, exported as `_z`/`_rowz` doors. |
| `real/real_route_race.h`, `rfft_calibrate.h` | S | The route race is layout-parameterized through `as_z`. |
| `real/zr2c.h`, `zr2c_build.h` | IL | Hermitian folds, even N, K=1. |
| `real/real_create.h` | M | The dispatcher: odd bridge, zr2c (IL), split engines (which also serve IL through the doors). |
| `real/real_dispatch_config.h` | N | Knob declarations. |
| `conv/il_layout.h` | N | The converters `vfft_il2sp`/`sp2il`. It also holds a **dead** `stride_il_t` (a split-plan-behind-IL wrapper that contradicts the "never bridge" law) and unused padded converters. |
| `conv/conv.h` | dead | Orphan. |
| `trig/*` | S | Real-to-real only, and IL is refused above this tier. `trig_codelets.h` and `dct2`/`dct3_n8_avx2.h` are N. |

## 2. Seams inside the mixed files

### 2.1 `struct vfft_plan_s` (`vfft_internal.h:94-486`)

| group | fields |
|---|---|
| common | `transform`, `placement`, `layout`, `N..N4`, `K`, `nthreads`, `own_batch`. The bridge children `oddr_child`/`oddr_buf` serve both layouts. |
| split | `cplan`, `oplan`, `k1sp`, `k1_sp_route`, JIT `k1_jit*`, `rplan`, `c2rdisp`, `tplan`, `rfft_row`, `c2r_row`, `exec_fwd/bwd`, `padded`, `exec_me`, `mt_unsafe`, `nat_*`, `nat2d_*` |
| IL | `k1_on`, `k1_il_route`, `k1il2p`, `k1il3p`, `k1ilpr`, `k1ilfd`, `k1ztt`, `k1fs`, `k1_exec`, TC `tcb*`/`tc_mt`, `k1_mono_ilf/ilb`, `ilnd`, plane queue `pq_*`, the whole `il2d_*` block (282-456), `zr2c_*` |
| quirks | `k1_mono` is the split mono but is set on IL handles (`c2c_oop_create.h:528`). `rplan`/`c2rdisp` are split engines that serve IL through doors. `il2d_rows` holds a split handle inside the IL tier. |

Target: `struct vfft_plan_s { <common header>; struct vfft_split_part sp; struct
vfft_il_part il; }`. It is a struct, not a union, until the bridge fields are settled.
`vfft_destroy` and the fingerprint walker become per-part.

### 2.2 Create tiers

- **`c2c_oop_create.h`**
  - The K=1 block (73-561) interleaves both layouts:
    - IL: 86-121, 184-230, 271-420, 475-482.
    - Split: 148-183, 231-258, 429-463.
    - Shared handle build: 501-548.
  - Line 128 already enforces "two libraries": `want_il` zeroes the other axis.
  - Split it into `_k1_create_il` and `_k1_create_split`.
  - Drop the vestigial `iR1 = sR1` (201-203) and the split mono on IL handles (528).
  - The classic split path (599-716) and the IL refusal (584) follow.
- **`c2c_ip_create.h`**
  - One layout switch at 342-345.
  - IL create at 86-216. Its `vfft_proto_registry_t *reg` is unused (`(void)reg` at 126).
  - `_c2c_ip_finish` (218-239) mixes the split `_c2c_mt_safe` with the IL ZTT/FS MT
    replay. It becomes one finish per layout.
  - The race context at 38-84 is dead.
- **`fft2d/fft2d_create.h`**
  - IL: 56-61, 203-865, 898-1083.
  - Split: 866-879, 897, 1084-1272.
  - Neutral: the plane queue arm (70-201) and the handle header (880-896).
  - The two halves share no locals except `N1`/`N2`.
  - The Bluestein hook at 56-61 runs even for split callers.
- **`fftnd/fftnd_create.h`**: the IL peel at 53-62 moves to the rank dispatcher; the rest
  is pure S.
- **`real/real_create.h`**:
  - neutral finish/arm types at 44-58;
  - odd bridge at 68-83;
  - IL zr2c at 118-144 and 344-367;
  - split engines at 149-219 and 368-410, which **also serve IL K>1 through the doors**;
  - IL-only odd race at 224-301.

### 2.3 Vocabulary and registry headers

- **`oop_leaf_registry.h`**
  - The typedef `vfft_oop11_fn` (40-43) is identical to `vfft_il2p_fn` (`il2p.h:57`). It
    goes to common.
  - The IL block (320-412, incl. `#include "il_isa.h"` at 324) goes to `il/`.
  - Consequence today: every split OOP consumer transitively pulls in the IL ISA surface,
    the AVX-512 twiddle builders and the 4,116-line quarter-wave table.
  - Its ISA test is `__AVX512F__` plus `VFFT_OOP_FORCE_AVX2`, not `build_isa.h`. Two ISA
    rules. See decision D4.
- **`oop_plan.h`**: the route ids `VFFT_K1_SP_*` (60-75) and `VFFT_K1_IL_*` (76-129), plus
  the wisdom kinds incl. `ZR2C` (131-160). They go to a common ids header. The rest is S.
- **`wisdom_reader.h:52-67`**: the `VFFT_NAT_*` modes mix split modes with IL
  ZCASC/ILP/CONV.
- **`policy.h`**: `policy_core.h` (N) gets L4 `vfft_policy_ord`, `vfft_cell_t`, L8
  `fits_l2`/`exceeds_l3`; `policy_il.h` gets the rest.
  - Note: `policy.h` needs `vfft_ztt_band`/`vfft_k1fs_band` in scope first
    (`vfft.c:241-243`). The IL half should include those itself.

### 2.4 Front door and execute

- **`_vfft_create_inner`** (`vfft.c:1496-1886`)
  - Order today: validation → TC wrapper (IL) → refusals → rank dispatch → C2C ip / oop
    → real → trig.
  - Layout is decided inside each tier.
- **`vfft_execute`** (`vfft_execute.h:482`)
  - The IL `k1_exec` fast path goes first. Then pq/oddr/tcb, the 2D/ND block (IL at 638,
    split falls through at 937), C2C ip/oop, real, trig.
  - It tests `h->layout == IL` at every arm.
- **Stale comments** name `_exec_zcascade`, `_exec_c2c_interleaved` and
  `_exec_c2c_oop_convert`, which no longer exist (`vfft_execute.h:44,93`,
  `vfft.c:1974-1977`).

### 2.5 Wisdom codecs

| file | split side | IL side |
|---|---|---|
| `wisdom2_oop.h` | `kind`, `R1`, `R2`, `t1p_variant`, `nf`, `factors`, `variants`, `k1_sp_route`, `cc_*` | `k1_il_route`, `il_*`, `ord_scr`, `zr_kv` |
| `wisdom2_oop_reader.h` | classic 418-495; `vw2_oop_rec_k1_lay` split branch at 967 | the IL branch at 1000; bwd 347-420; prime 808; zr2c 1105-1218 |

- **`wisdom2_oop.h`**: the common fields are `N`, `K`, `place_ip`, `nthreads`, `role`,
  `ns`. The per-layout record builder at 947 already writes the two layouts separately.
- **`wisdom2_2d_reader.h`**: split below about line 500, IL `ilcol`/`ilnd`/`forms` above.
- **`wisdom2_stride_reader.h`**: the TC-MT rows (552-612) go to IL.
- **`wisdom2_real_reader.h`**: the oddr row (244-290) goes to the bridge.
- **Shared helpers**: the tokenizer is copied three times (`vw2__oop_split_ints`,
  `vw2__stride_split_ints`, `vw2__2d_split_ints`). It goes to the store core once.

The on-disk format does not change. The `lay` key already separates the rows. Only the
C code that reads and writes them is regrouped.

## 3. Cross-layout dependencies

### Real sharing — goes to `common/`, used by both, never duplicated

- **The 11-arg codelet ABI typedef.** Used at `c2c_ip_create.h:50/123`,
  `c2c_oop_create.h:533`, `k1_commit.h:1315`, `vfft_internal.h:224` and
  `dp_planner_il.h:330`.
- **The K=1 route-id and wisdom-kind namespace.** It is persisted, and both layouts'
  rows name it.
- **The worker pool.** It sits in `support/threads.h` under a `stride_` name; the name is
  an accident.
- **The race body.** `support/race.h` is used by both layouts.
- **Platform support:** `cpu_cache.h`, `env.h` part 1, `diag.h`, `race_timing.h`.
- **`tw_exact.h`.** Candidates to use it later are the split twiddles (`r2c.h:185`,
  `strided_tw.h:92`) and `zr2c.h:84`. That would change their bits, so it is a separate
  decision, D5.
- **Number theory.** Three copies:
  - `rader.h:73-125`
  - `prime_dispatch.h:29-47`
  - `il_prime.h:50-88`
- **Aligned allocation.** Five copies:
  - `plan.h`
  - `proto_stride_compat.h`
  - `oop_auto.h`
  - `il2p.h`
  - `ztt.h`
  - plus `support/zalloc.h`
- **PI.** Four copies: `twiddle.h:13`, `oop_plan.h:54`, `ztt.h:65`, `il2p.h`.
- **Digit-reversal permutations.** `natorder_perm.h` and `_il2d_nat_perm`
  (`il2d_cols.h:447`). Their orientation may differ; check before unifying.
- **Data-movement kernels:**
  - `transpose.h`
  - `_vfft_k1_transpose[_perm]`, which lives in the split `oop_plan.h:203` and is used by
    the IL 2D tier
  - `_k1fs_transpose_range`
  - `_il2d_transpose_zip`/`_unzip_transpose`
  - the `vfft_il2sp`/`sp2il` converters
- **Order and cache laws** (policy L4, L8), the cell, and `_vw2_lay_of`.
- **The plane queue** (`plane_queue.h`) and the wisdom2 store core.

### Placement accidents — fixed by moving, no interface needed

- **Includes that pull in the other layout for nothing:**
  - `dp_planner_split_oop.h:69` includes `dp_planner_il.h` but uses none of it.
  - `dp_planner_il.h:46` includes the split `oop_plan.h` for the route ids.
  - `wisdom2_oop.h:36` includes `oop_plan.h` for the enums.
- **Code in the wrong home:**
  - `k1_commit.h:1333-1356` `_bank_nat_1d` is split code in the IL commit file.
  - `support/strided_codelets.h` is a split kernel type.
  - `env.h` part 2 holds split planner knobs.
- **Misfiled wisdom rows:**
  - the IL TC-MT rows in the split stride codec;
  - the IL oddr row in the split real codec.
- **Unused split parameter.** The split registry pointer `vfft_proto_registry_t *reg` sits
  in the IL create signatures: `c2c_ip_create.h:113`, `fft2d_create.h`, `fftnd_il.h:1375`.
- **Split vocabulary on IL data:**
  - `k1_mono` (split) is set on IL handles;
  - the `VFFT_NAT_*` enum carries IL modes.

### Deliberate coupling — becomes a declared interface

- **B1. IL real 1D at K>1 (and the zr2c fallback) is served by split engines.**
  - The engines are the split `rplan`/`c2rdisp`, called through
    `vfft_r2c_execute_fwd_z` / `vfft_c2r_disp_execute_z`.
  - Their `zo`/`zi` modes are fused into the split engines' pack/unpack for speed. Cutting
    them out loses the fusion.
  - Keep them as the split side's published "IL doors".
- **B2. IL 2D real ROWSPLIT.**
  - `il2d_tier.h:1515-1545` builds a split child through the public `vfft_create`, which
    is clean.
  - It then reaches into the child's `rplan`/`c2rdisp` for the `_rowz` doors
    (`il2d_tier.h:160/211`), which is not clean.
- **B3. The odd-N real bridge.**
  - `_oddr_build` (`vfft.c:1032`) builds an IL c2c child (`rc.layout = INTERLEAVED` at
    1044).
  - That child serves both layouts' real odd N.
- **B4. Wisdom signpost `_ilp_ref_of`** (`k1_commit.h:957-975`).
  - A split `@nat` row points to the IL recipe (`VFFT_NAT_ILP`). The IL in-place door
    reads and banks split stride `@nat` rows (`c2c_ip_create.h:474`,
    `k1_commit.h:1354`).
- **Dead, delete rather than bridge:**
  - `stride_il_t` (`il_layout.h:79-121`);
  - the `fftnd_r2c.h` IL output (`il_out`, `stride_plan_nd_r2c_il`), whose setter in
    `vfft_execute.h:963/971` is unreachable because rank ≥ 3 IL real is refused.

## 4. Target layout

```
src/core/
  vfft.c                  front door: config validation, refusals, ONE layout dispatch,
                          wisdom bundle, fingerprint. Includes common/, split/, il/, bridge/.
  vfft_internal.h         vfft_plan_s = common header + split part + IL part
  vfft_execute.h          sig check, then ONE layout branch -> split_execute / il_execute
  common/                 depends on nothing in core but itself
    abi/                  codelet_abi.h (the 11-arg fn type), route_ids.h (K1 route ids,
                          wisdom kinds, natural modes)
    support/              build_isa, cpu_cache, diag, env (part 1), race, race_timing,
                          slot_check, threads (the pool), zalloc (ONE aligned allocator),
                          ref (gate-side)
    math/                 tw_exact.h, numtheory.h, pi, digit-reversal perms
    move/                 transposes, IL<->split converters, zip/unzip transposes
    policy/               policy_core.h (L4 order, cell, L8 cache laws)
    wisdom/               wisdom2.h store core, shared tokenizer, selftest, migrate
    plane_queue.h         howmany>1 over any plan
  split/                  depends on common/ only
    engine/               (today's engine/)
    primes/               (today's primes/)
    planning/             dp_planner, exhaustive_plan (+ env part 2 knobs), measure,
                          pad_calibrate, adopt_wisdom, wisdom_reader, dp_planner_split_oop
    oop/                  oop_plan (split kinds only), oop_leaf_registry (split half),
                          oop_execute, oop_auto, oop_dp, oop_mt, strided_codelets
    rank1/                c2c_ip (split arms), c2c_oop (split K=1 + classic), natorder/
    rank2/                fft2d, planners, fft2d_r2c, strided_tw, natorder_2d
    rank3/                fft3d, strided_rows, fftnd, fftnd_r2c
    real/                 r2c, rfft, c2r, dispatch (incl. the published IL doors), route
                          race, rfft_calibrate
    trig/                 (today's trig/)
    wisdom/               split codecs: oop classic + lay=split K1, stride, 2d/3d, fftnd,
                          real routes
    create.h, execute.h   the split create tiers and execute arms
  il/                     depends on common/ only
    isa/                  il_isa.h, avx512/vtw_avx512.h, ztt_qw16384.h, the IL mono/solo
                          resolvers (today in oop_leaf_registry.h)
    rank1/                il2p, il_flatdit(+mt, race), ztt(+mt), il_prime, k1_fourstep,
                          k1_commit (IL part), c2c_ip (IL), c2c_oop (IL K=1), TC wrapper
    rank2/                il2d_col, il2d_cols, il2d_tier, fft2d_real_il
    rank3/                fftnd_il
    real/                 zr2c, zr2c_build
    planning/             dp_planner_il, il_slot_probe, policy_il.h
    wisdom/               IL codecs: lay=il K1, bwd, prime, zr2c, ilcol/ilnd/forms, TC-MT
    create.h, execute.h   the IL create tiers and execute arms
  bridge/                 the ONLY place besides vfft.c allowed to include both split/ and il/
    real_doors.h          B1/B2: the contract IL uses to call split real engines
    oddr.h                B3: the odd-N real bridge (_oddr_build + its wisdom row)
    nat_ilp.h             B4: the @nat -> IL recipe signpost (until D2 retires it)
```

Rationale:

- **Organized by rank, then engine, inside each layout.** This mirrors how both create
  paths and both execute paths branch: transform → rank → placement.
  - `rank1/` holds the 1D engines and create tiers.
  - `rank2/` and `rank3/` hold the multi-dimensional tiers.
- **Real and trig stay separate from rank.** Their dispatch is by transform, not rank.
- **One front door, one fork.** `vfft_create` validates, applies the layout-independent
  refusals, then calls `vfft_split_create(cfg)` or `vfft_il_create(cfg)`.
  - Each side owns its transform × rank × placement tiers.
  - `vfft_execute` does the same after `_vfft_sig_bad`. The IL side keeps its `k1_exec`
    fast path as the first thing it tests.
  - The TC wrapper (1D IL howmany > 1) is an IL tier, so it moves under the IL create.
- **Bridges are named, few, and one-way.** The IL side calls split real engines only
  through prototypes declared in `bridge/real_doors.h`. It never reaches into a split
  handle's fields.

### Dependency rules (enforced mechanically, section 7 R0)

1. `common/**` includes only `common/**` and system headers.
2. `split/**` includes `common/**` and `split/**`, never `il/**` or `bridge/**`.
3. `il/**` includes `common/**` and `il/**`, never `split/**`.
   - For the doors it includes `bridge/real_doors.h`, which declares prototypes and
     types only and contains no split header.
   - Implementation option: the door types are opaque (`struct vfft_r2c_plan_s;`). The
     split side defines them.
4. `bridge/**` may include both. It is kept small; adding to it is a design decision,
   not a convenience.
5. `vfft.c` and the three front-door headers may include everything.
6. No two headers under `src/core` share a basename. The build puts every directory on
   `-I`, so a duplicate would silently change which file resolves.

Note on the single translation unit: all headers still land in one TU (`vfft.c`). The
rules govern the include graph, which is what a reader and a maintainer follow. They do
not govern linkage. The checker reads the `#include` lines.

## 5. Decisions for the owner (flagged, not assumed)

- **D1. Bridges B1/B2 (the IL doors on split real engines).** Keep them as a declared
  interface (recommended; the fusion is the reason they exist). The alternative is an IL
  real engine for K>1, which is new work.
  - **Owner (2026-09-27): an IL real engine will be built; IL and split must not cross.**
    The existing wrappers that make IL requests run on split machinery (IL real K>1 and
    the zr2c out-of-place fall-through onto the split CCE engines, the odd-real bridge,
    the smooth-odd r2c race) move to `bridge/` in phase 7 as a TEMPORARY holding place,
    to be removed once the IL real engine lands. No new crossing may be added.
- **D2. B4, the `@nat` → IL recipe signpost.** Keep it in `bridge/` for now (no format
  change). Or give the IL in-place door its own `lay=il` row and retire `VFFT_NAT_ILP`,
  `ZCASC` and `CONV` from the split enum. That changes the wisdom format, so it needs a
  migration and is out of scope for a pure restructure.
  - **Owner (2026-09-27): must do.** The IL in-place door gets its own `lay=il` row;
    `VFFT_NAT_ILP`, `ZCASC` and `CONV` leave the split enum; old files migrate; B4
    (`bridge/nat_ilp.h`) goes away.
  - **Done (2026-09-28).**
    - The bug it fixed: five lay-less `eng=stride mode=ilp` rows in the shipped
      `wisdom2_oop.txt` made the split in-place NATURAL K=1 create skip its reorder
      tape, so N = 128, 256, 255, 512 and 1024 came back in scrambled order (rel. err
      1.3-1.5 against a long-double DFT).
    - `common/abi/nat_modes.h`: the enum ends at `VFFT_NAT_PSWAP` (`VFFT_NAT_MAX`);
      6-8 are retired and never reused. The IL in-place door already banks on its own
      kind-3 `lay=il` row (since 2026-09-21); its handle marker is `VFFT_IL_NAT_K1`
      (7, the old value, so plan fingerprints read the same).
    - `split/wisdom/wisdom2_stride_reader.h`: a `@nat` row is a split verdict only
      (`eng=stride`, mode 1-5). `ref=` signposts, `eng=zturn|k1` and unknown modes
      read as absent. Deleted: the `zr` banked-loss marker (`raced`), the ILP
      signpost encoder (`ref_ilp`, `ref_comp`) and the `@scrmode` lookups and banks,
      which had no callers.
    - `_bank_nat_1d` moved into `split/rank1/c2c_ip_create_split.h` and no longer
      reads IL wisdom; `bridge/nat_ilp.h` is deleted. The natural arm also ignores a
      mode above 5 from the frozen spike table (read only under the retired kill
      switch).
    - Spike (owner: its machinery is fixed later; do not break its wiring):
      `spike_wisdom.txt` and its loader (`vfft_proto_wisdom_load`) are untouched.
      The spike -> wisdom2 migrator's codec refuses modes 6-8, so its 15 retired rows
      (5 ZCASC and 5 ILP `@nat`, 5 ZCASC `@natoop`) go to quarantine verbatim
      (`unknown-nat-mode`). The migrator's reader gate and the export gate apply
      their existing rule, "a codec-refused row must miss", to the natural tables.
    - Migration: `vw2_migrate_drop_retired_nat(dir)` (`wisdom2/wisdom2_migrate.h`)
      and `src/tools/wisdom2_drop_retired_nat.c` delete the retired rows from a store
      directory (row order kept, idempotent; the save writes the current `@vw2`
      header). The shipped files lost exactly 16 rows: `wisdom2_oop.txt` 6,
      `wisdom2_scr.txt` 1, `Zen4/wisdom2_oop.txt` 6, `Zen4/wisdom2_scr.txt` 3. They
      were deleted as text, which matches the tool's output below the header (the
      Zen4 files keep `@vw2 1.2`).
    - Proof, the order contract: a probe (split in-place NATURAL K=1 at N = 128, 256,
      255, 512, 1024, 64 and 2048 against a long-double DFT) gives rel. err at most
      6e-16 everywhere with the shipped store, an empty store, and a copy of the
      pre-D2 store (its old rows read as absent). Before: five cells scrambled.
    - Proof, spike: the stride migrator gate and the export gate on the shipped
      spike file FAILED before D2 (10 mismatches each: the ILP `@nat` and the ZCASC
      `@natoop` rows never round-tripped) and pass after. The migrated store and the
      export differ from the pre-D2 ones only by the 15 retired rows; the scrambled
      shard and the rfft export are byte-identical.
    - Proof, build: both ISAs compile with the baseline warnings (20 at O2, 5 at
      O3); the dependency rules are clean.
    - The baseline harnesses' natural cell (`harness_golden.c`
      `c2c.split.ip.natural`, `fp_sweep.c` `c2c.split.ip.nat`) moves from K=1 to
      K=4, where the split natural verdict is banked (the shipped store banks it at
      q=4 and q=32 only). At K=1 the cell was served by a `mode=ilp` row: the
      recorded natural digests are the scrambled cell's, byte for byte (in
      `src/tools/baseline/reference/golden_bits.txt` as well). With that row gone,
      K=1 has no banked verdict and races, which the purity assert refuses. That
      line of every recorded golden/fp reference changes by design; re-record it.
    - R5 at avx2 and avx512 against the step reference: every difference is D2's.
      - `roundtrip`: only the two edited shards' hashes; both still save identical
        to what they loaded.
      - `wisdom_replay`: exactly the sampled deleted rows are gone (247 -> 241
        specs); nothing else moved, except one avx512 cell that flips on its own
        (the open item below).
      - `api_sweep`: its two split in-place NATURAL K=1 cells. N=256 was `nat=7`
        with no tape; it now races (nothing is banked for it) and deploys a tape
        (PSWAP at avx2, PURE_CYCLE at avx512). N=1024 takes the clock-free
        opportunistic PSWAP.
      - `golden_bits` / `fp_replay`: only the natural cell (now K=4: pure, the
        same digests at both ISAs). In every reference, the recorded one and this
        VM's, the old natural digests equal the scrambled and default cells': the
        bug, recorded as the baseline.
    - Open, not D2 (found by this check): the avx512 replay cell `t=c2c n=256 q=2
      ord=nat place=oop` flips between two outputs from run to run with the same
      binary and store, the pre-D2 binary and store included. Both are correct (rel.
      err 3.2e-16 and 3.4e-16 against a long-double DFT), the fingerprint is the
      same, and the race counter reads 0: something outside the counter picks.
- **D3. The plan struct.** Recommended: a common header plus `sp`/`il` sub-structs. This
  touches every `h->il2d_*` access (a mechanical rename) and changes field offsets.
  - A cheaper alternative keeps the field names by using C11 anonymous sub-structs, so no
    access sites change. Offsets still change once the fields are regrouped.
  - **Owner (2026-09-27): leave it for now.** Phase 8 is deferred; the flat struct stays,
    and the separation is enforced by the include graph (hygiene.py) alone. The split 2D
    handle keeps writing the four IL column fields until then.
- **D4. Two ISA rules.** **Owner (2026-09-27): do it; the owner tests it on their
  machines.** The split OOP registry picks the ISA from `__AVX512F__` and
  `VFFT_OOP_FORCE_AVX2`; the IL side uses `build_isa.h`. Unifying them on `build_isa.h`
  can change which split kernels an AVX-512 build binds.
  - Following the rule "the user selects the ISA, avx2 is never a fallback", `build_isa.h`
    should win.
  - That is a behaviour change, so it gets its own step and a race/gauntlet check, not a
    silent part of a move.
- **D5. De-duplication that changes bits.** Examples: routing split twiddles through
  `tw_exact.h`, and unifying the two digit-reversal permutations. These are
  improvements, not restructuring. Do them after the tree is split, each with its own
  proof.
  - **Owner (2026-09-27), twiddles: `tw_exact.h` is the canonical path for every
    twiddle table; split must comply.** The split tables built as `cos/sin` of a
    double angle (`r2c.h`, `strided_tw.h`, and the rest found by a sweep) move to
    `vfft_cs2pi_exact`. Split output bits change by design: the proof is an accuracy
    report against a high-precision reference (must improve or hold at every gauntlet
    size) with plan choices unchanged, then the golden bits are re-baselined.
  - **Owner (2026-09-27), digit-reversal permutations: do not touch.** Split and IL are
    different libraries; the owner revisits the split side after IL is finished.
  - **Owner (2026-09-27), duplicated utilities: unify them.** Done in steps:
    - pi (A1): one `common/math/pi.h` (`VFFT_PI`, `VFFT_PI_L`) replaces eight `M_PI`
      fallbacks, `VFFT_IL2P_PI` / `VFFT_ZR2C_PI` / `VFFT_NATORDER_PI` and the inline
      literals; it keeps one `M_PI` fallback (the bare literal) for the benches.
    - the clock (A1): `vfft_proto_now_ns` (split planner) folded into `vfft_now_ns`
      (`common/support/race_timing.h`, QPC on Windows, `clock_gettime` elsewhere).
    - A1 proof: the two vfft.c objects at avx2 and avx512 are `obj_equiv
      --strict-data` EQUIVALENT to the previous commit's under the clock rename map.
    - number theory (A2): `common/math/numtheory.h` (`vfft_is_prime`,
      `vfft_is_radix_smooth`, `vfft_powmod`, `vfft_primitive_root`) replaces the copies
      in `prime_dispatch.h`, `bluestein_calibrator.h`, `rader.h` and `il_prime.h`. The
      split Rader root search (N-1 factored over {2..19} only) gives way to the full
      factorization; `vfft_is_radix_smooth(0)` no longer loops. Proof: an exhaustive
      value test, old copies vs new (primes and smoothness to 2^22 / 2^25, the
      primitive root of all 295,946 primes to 2^22 and of the 2,841 split-Rader ones):
      0 mismatches. The objects change only in the helpers' callers and in compiler
      churn in functions whose source is untouched (register allocation, inlining).
    - the remaining clocks (A3): `_sp_now_ns`, `_bcal_now_ns`, `_il_dp_now_ns` (the
      same computation as `vfft_now_ns` on Linux), the profiling clocks `_r2c_prof_now`,
      `_rfft_now` and `_f2d_now` (now a microsecond wrapper), and the twelve inline
      `clock_gettime` pairs in the split 2D / rank-N real races all read `vfft_now_ns`;
      `race_timing.h` is the only file in core that reads a clock. `__rdtsc` in the
      split OOP tuner stays: it banks cycles (`units=cyc`), a different unit. The race
      intervals are exact below 2^53 ns of uptime (~104 days), a couple of ns after.
    - the allocators (B): `vfft_aligned_alloc(bytes)` / `vfft_aligned_free(p)`
      (`common/support/zalloc.h`, 64-byte aligned, `VFFT_ALIGNMENT`) replace eleven
      families: `VFFT_ZS_ALLOC/FREE`, `stride_alloc/free`,
      `vfft_proto_posix_memalign`/`vfft_proto_aligned_free`, `STRIDE_ALIGNED_ALLOC/FREE`
      (two copies), `VFFT_OOP_AALLOC/AFREE`, `RFFT_ALIGNED_ALLOC/FREE`,
      `_vfft_proto_dp_aligned_alloc`, `VFFT_IL2P_ALLOC/FREE`, `VFFT_ZTT_ALLOC/FREE`. All
      were 64-byte aligned, `_aligned_malloc`/`_aligned_free` on Windows and
      `free()`-compatible elsewhere, so every pairing maps onto the one pair. 30
      posix-style calls became `p = vfft_aligned_alloc(n)` (statement) or
      `!(p = vfft_aligned_alloc(n))` (condition: non-zero on failure, as before).
      The huge-page pair (`stride_alloc_huge` / `stride_free_huge`, DTLB motivation,
      never called) stays by owner decision.
    - huge pages (side job, owner: "keep that feature, restructure it if needed"):
      `common/support/hugepage.h` (`vfft_alloc_huge` / `vfft_free_huge`,
      `VFFT_HUGEPAGE_THRESHOLD`) replaces the env.h pair. The backing is a function of
      (platform, size) alone -- heap below 64 KB; `mmap` (hugetlb, else THP) or
      `VirtualAlloc` (large, else ordinary pages) at and above -- so the free no longer
      guesses by 2 MB alignment (an ordinary `mmap` is only 4 KB aligned; the old free
      could hand `mmap` memory to `free()`). Still not wired in; included by nothing.
- **D6. Dead code.** Delete it (recommended) rather than move it:
  - `conv/conv.h`, `fftnd_natorder.h`, `fftnd_planner.h`, `fftnd_wisdom.h`;
  - `engine/compat.h`;
  - `stride_il_t` and the padded converters in `il_layout.h`;
  - the `fftnd_r2c.h` `il_out` path;
  - the dead race context in `c2c_ip_create.h`;
  - the stale comments.
- **D7. Naming.** The `stride_pool_*`, `vfft_proto_posix_memalign` and `_il_ab_now`
  names outlive their layout. Rename them in common, or keep the names and only move the
  files. Renames are cheap to gate: R3 with a rename map.
  - **Owner (2026-09-27): rename every split-era name defined in `common/`**, docs
    included; done as one mechanical commit. The pool module is `thread_pool`:
    `stride_pool_run` -> `thread_pool_run`, `stride_pool_workers_for` ->
    `thread_pool_workers_for`, `stride_get_num_threads` -> `thread_pool_size`,
    `stride_set_num_threads` -> `thread_pool_resize`, `STRIDE_POOL_MAX_DISPATCH` ->
    `THREAD_POOL_MAX_DISPATCH`, the internals `_stride_*` -> `_thread_pool_*`
    (`_stride_num_threads` -> `_thread_pool_nthreads`, `_stride_pool_size` ->
    `_thread_pool_nworkers`). `env.h`: `stride_env_init/restore`,
    `stride_get_num_cores`, `stride_pin_thread/unpin_thread`, `stride_print_info`,
    `stride_get/set_verbose` -> `vfft_*` (`vfft_num_cores`); `STRIDE_VERSION_*` ->
    `VFFT_VERSION_*`, `STRIDE_ISA_NAME` -> `VFFT_ISA_NAME` (values unchanged).
    `move/transpose.h`: `stride_transpose*` -> `vfft_transpose*`. The clock
    `_il_ab_now` -> `vfft_now_ns`. Include guards -> `VFFT_COMMON_*_H`.
  - Not renamed here: the allocators (`stride_alloc*`, `stride_free*`,
    `STRIDE_ALIGNMENT`, `STRIDE_HUGEPAGE_THRESHOLD`, `vfft_proto_posix_memalign`) and
    the second clock `vfft_proto_now_ns` get their final names in the utilities merge.
    The recorded gate data under `src/tools/baseline/reference/` keeps the old names.

## 6. Migration

Every step lands as its own commit and passes the gate in section 7 before the next one
starts. "Byte-identical" means the `vfft.c` object, every codelet object, both
libraries and every gauntlet/gate executable, at avx2 and avx512.

| # | phase | what | expected proof |
|---|---|---|---|
| 0 | gate | Refurbish `src/tools/baseline/` (section 7). Capture Linux references for avx2 and avx512, for both CMake and `build.py`. | The references exist; one self-comparison of HEAD vs HEAD is all green. |
| 1 | dead code | D6 deletions and the stale comments. Remove the unused IL include from `dp_planner_split_oop.h`. | R1 bytes, or R3-strict if a dead static body or struct field was emitted; R5 identical. |
| 2 | common, pure moves | `git mv` the neutral files (support/*, `tw_exact.h`, `plane_queue.h`, `real_dispatch_config.h`, `trig_codelets.h`, the wisdom2 store core, selftest, migrate) into `common/`. Update path-qualified includes only. | **Byte-identical**: include order is unchanged because each file keeps its include site. |
| 3 | pure-layout moves | `git mv` every file that is already S into `split/` and every IL file into `il/`, per section 4. | **Byte-identical.** |
| 4 | carve neutral out of mixed | New `common/abi/codelet_abi.h` (the typedef), `common/abi/route_ids.h` (both route enums, wisdom kinds, NAT modes), `common/policy/policy_core.h`. Cut-and-paste the definitions verbatim, included at the same point. | Byte-identical if the definition order is kept (it can be: the new header is included where the old text was). Otherwise R2 `--allow-reorder` + R3-strict + R5. |
| 5 | split mixed headers | Move `oop_leaf_registry.h`'s IL half to `il/isa/`; `_bank_nat_1d` to split; `policy_il.h`; the wisdom codecs per section 2.5 (with the shared tokenizer once); TC-MT rows to IL; the oddr row to bridge; `env.h` part 2 and `strided_codelets.h` to split. One header per commit. | R2 (permutation) + R3-strict + R4 + R5. |
| 6 | split mixed create tiers | One function per commit: `c2c_oop_create` K=1 → two functions; `c2c_ip_create` → IL/split plus two finish hooks; `fft2d_create` → `_2d_il`/`_2d_split` plus a neutral dispatcher; the `fftnd_create` peel; `real_create` (bridge B1 stays). Drop the unused `reg` parameter from IL signatures, `k1_mono` on IL handles, and the vestigial `iR1 = sR1`. | R3-strict (bodies change) + R4 + **R5 identical**: `fp_replay` and the API sweep prove the same routes and the same bits. |
| 7 | the front door | `_vfft_create_inner` and `vfft_execute` get one layout fork into `split/create.h` / `il/create.h` and `split/execute.h` / `il/execute.h`. Bridges B1-B4 behind `bridge/`. The dependency checker is turned on as a hard R0 rule from here on. | R5 identical; R0 dependency rules green. |
| 8 | plan struct (D3) | Regroup `vfft_plan_s` into common + `sp` + `il`; per-part destroy and fingerprint. | `layout.txt` changes (expected, recorded); R5 identical; ASan build of the gates clean. |
| 9 | optional clean-ups (D4, D5, D7) | One per commit, each with its own evidence: renames via an R3 rename map; the ISA unification and bit-changing de-duplication by gauntlet verify plus a race check. | As stated per item. |

Notes:

- **Include ORDER matters to the object, not only to the reader** (measured while
  building the gate): swapping two independent includes in `vfft.c` (`dct4.h`
  after `dht.h`) changed GCC's inlining in `_fft2d_destroy` and `_fft3d_destroy`.
  The code is still correct, but it is no longer the same code. So a step can
  expect byte identity only if every moved or carved-out definition lands at the
  same point in the translation unit. A step that changes the order is proven
  at R3 (the listed functions only) plus R5, never at R1.
- **Phases 2-3 are nearly free and give the reader the new map immediately.** They are
  pure `git mv` plus include-path edits, so byte identity is the expected outcome.
- **Phase 6 is the real work.** The five create tiers hold about 3,400 lines and are
  where layout is tested ad hoc. Each split is behaviour-preserving by construction: the
  code paths are already disjoint and gated by `want_il` / `layout == IL`. The proof is
  semantic identity at zero tolerance.
- **Build systems.** Both add every `src/core` subdirectory to `-I` automatically (CMake
  `CMakeLists.txt:95-106`, `build.py` `build_includes()`). A new directory needs a CMake
  **re-configure**, since the globs are evaluated at configure time. `build.py`'s object
  cache may not notice an `-I` change, so gate builds are clean builds.
- **Gates, benches and tools that include core by path must follow each move** (about 36
  core files use path-qualified includes, plus the gates that include `oop/ztt.h`,
  `planning/policy.h`, `../../src/core/...`). A build failure there is loud. Phase 0 adds
  "all gates built" as a separate precondition, so a build break is not scored as a
  failed gate.
- **The ZTT MT split and the AVX-512 tail work continue on the new tree.** They touch
  only `il/`, which is the point.

> **WARNING (owner, 2026-09-27): the kind-3 wisdom record split is the most
> dangerous change of the whole separation.** Decision: option (a), split
> `vfft_oop_wisdom_entry_t` and its reader into a split record/codec and an
> interleaved record/codec. It is necessary, and it touches the code that
> reads, races and banks every K=1 verdict of both layouts (about 400 field
> accesses in 8 files: `wisdom2_oop.h`, `wisdom2_oop_reader.h`, `k1_commit.h`,
> `c2c_oop_create.h`, `dp_planner_il.h`, `dp_planner_split_oop.h`,
> `wisdom2_migrate.h`, `zr2c_build.h`).
>
> The cloud gate covers it on this VM only: output bits, fingerprints, the
> 271-cell API sweep, the wisdom replay and the store round trip, at both
> ISAs. **After this split, everything has to be tested thoroughly on the
> owner's own machines** (the i9-14900KF AVX2 box and the Zen 4 AVX-512
> laptop) before the branch is trusted:
> - the gate battery (`build_tuned/run_gates.py`) at both ISAs;
> - cold creates that RACE and BANK into a scratch store, then a replay of
>   that store (the banking path is what the split rewrote, and the VM's
>   replay mostly exercises lookups);
> - the gauntlet `verify` groups (pow2, primes, 2d-small, 3d-pow2) and a
>   calibrate-and-race of a few K=1 cells of each engine family (pair,
>   chain3, flat DIT, ZTURN-T, four-step, prime, mono), in place and out of
>   place, natural and scrambled;
> - the wisdom migrator on an old-format store;
> - the owner's usual performance runs, against the pre-separation build.

## 7. Regression gate: refurbishing `src/tools/baseline/`

### What exists

`src/tools/baseline/` was built for the earlier "`vfft.c` into headers" migration. It
has four parts:

- `obj_equiv.py` compares per-function disassembly with addresses and hex normalized
  away.
- `sym_census.py` lists defined, undefined and mutable symbols.
- `race_census.py` statically keys every timing/race site so it survives moves.
- `capture_baseline.py` + `harness_golden.c` + `fp_sweep.c` produce output-bit digests
  and create-time plan fingerprints, one process per cell against a seeded scratch store.

`slice_ladder.py` chains these into five rungs. The method is right.

### Why it cannot be used as it stands (on Linux)

- **The reference is Windows-bound.** It is MinGW gcc 15.2: `vfft_baseline.o` is PE/COFF,
  and the symbol lists carry PE names. It is out of contract for Linux gcc 13.
- **The baseline SHA is not in this clone.**
- **Hardcoded Windows tool paths:**
  - `C:\mingw152` in `slice_ladder.py:46`, `obj_equiv.py:107`, `sym_census.py:310`;
  - `capture_baseline.py:209` expects `harness_golden.exe`, so it aborts on Linux.
- **`reference/build_all_gates.sh` is broken.** It globs a `benches/` folder that no
  longer exists there.
- **`race_census.py` silently shrinks when files move.** Its input is a fixed list of 13
  header paths filtered by `os.path.exists`. Moving `k1_commit.h` would drop its races
  without an error. This is the most dangerous gap for this restructure.
- **`obj_equiv.py` hides constants, immediates and displacements.** A 0.97 → 0.96
  threshold change passed as equivalent. So a struct field reorder, which is exactly
  what phase 8 does, is invisible to it.
- **Coverage is narrow.** Only the one avx2 `-O2` `vfft.c` object is covered: no avx512,
  no codelet objects, no libraries, no executables.

### Additions (small)

1. **`toolchain.py`** (a shared module)
   - Resolves `CC`/`NM`/`OBJDUMP`/`READELF` from env, then `PATH`, with the MinGW paths as
     the Windows fallback.
   - Provides the exe suffix, the object format, and a *flags key*: compiler version,
     host CPU model, ISA, flags.
   - All four scripts use it.
2. **`capture_ref.py --isa avx2|avx512`** writes `reference/<key>/`, e.g.
   `linux-gcc13-avx512/`, next to the Windows reference. It captures:
   - (a) `vfft_O2.o` and `vfft_O3native.o`, with the include set **generated** by the
     same recursive walk the build uses (not the frozen `include_flags.txt`);
   - (b) `vfft.i` (`gcc -E -P`), its sorted top-level-declaration form, and `-dM`
     macros;
   - (c) SHA-256 of every codelet `.o`, both `.a` files and every gauntlet/harness/gate
     binary, one manifest per build system;
   - (d) the `sym_census` trio;
   - (e) the race census over a **glob** of all core headers, plus a per-basename site
     count;
   - (f) `golden_bits.txt` and `fp_replay.txt` with repeats;
   - (g) `gates.txt` (pass/fail per gate from `build_tuned/run_gates.py`, no timings);
   - (h) the API sweep;
   - (i) the wisdom replay;
   - (j) `layout.txt`: `sizeof`/`offsetof` of `vfft_plan_s` and of every struct that
     moves;
   - (k) the `VFFT_WARN` warnings.
3. **`step_gate.py --ref reference/<key>/ [--allow-reorder] [--rename-map F]`** runs the
   rungs below in order and stops at the first red. It writes one table plus
   `step_gate.json`. `--restamp` is allowed only after an all-green run, and only for
   the artifacts a step is allowed to move.
4. **`both_builds.sh --isa X`** does clean builds with CMake (fresh dir, `-DVFFT_ISA=X`)
   and with `build.py`.
   - Each build system is compared only against its own reference, since their flags
     differ.
   - It runs for avx2 and avx512. This VM has AVX-512, so both are checked here.
5. **`api_sweep.c`** exercises the public API only:
   - transforms: c2c, r2c/c2r, trig;
   - both layouts; ranks 1-4; order DEFAULT/NATURAL/SCRAMBLED; in place / out of place;
   - `nthreads` 1/2/4; `howmany` > 1;
   - the refusal matrix (e.g. rank-3/4 IL real must be refused).

   It emits the refusal decision and an output-bit digest per cell, one process per
   cell against a seeded scratch store.
   - Tolerance is **zero** (bits, not error norms); the error vs a reference DFT is
     printed for information only.
   - A threaded cell that is not digest-stable is recorded as NONDETERMINISTIC. Any
     change in that status is red.
6. **`wisdom_replay.py`** replays every row of every shipped wisdom2 shard: it creates the
   plan against a seeded scratch copy with `VFFT_FINGERPRINT`.
   - It requires `races=0` and an identical fingerprint. This proves "same route from
     the same store" through every codec that phases 5-6 split.
   - It also checks that load → save round-trips byte-identically.
7. **Fixes to the existing scripts:**
   - `capture_baseline.py`: the exe suffix; the scratch dir via `tempfile`.
   - `race_census.py`: the input is the glob, and a missing path is a hard error.
   - `slice_ladder.py`: use `toolchain.py` and the generated include set.
   - `build_all_gates.sh`: delete it, or point it at `build_tuned/`.
   - `wisdom_store.sha256`: re-derive it for today's `src/wisdom`.
   - Record the git SHA with a check that it is reachable.

### The rungs

| rung | checks | pass rule |
|---|---|---|
| **R0 hygiene** | No duplicate header basenames under `src/core`. Every path-qualified include in core, gates, benches and tools resolves. The race census lost no site without the same count reappearing under another basename. No new warnings. All gates built. From phase 7 on, the **dependency rules** of section 4 (a small include-graph checker over `common/`, `split/`, `il/`, `bridge/`). | all hold |
| **R1 bytes** | All codelet `.o` and `libdagcodelets.a` (`ar` deterministic mode, or compare members); the `vfft` library; the executables; `vfft_O2.o` and `vfft_O3native.o`. | Codelets always identical. If `vfft.c` objects are identical ⇒ skip to R5. |
| **R2 preprocessed** | `vfft.i` identical, or with `--allow-reorder` the sorted declarations and `-dM` macros identical. R2b: a `strings` diff of `.rodata` must be empty (non-ASan builds embed no paths: there is no `-g`, no `assert`, and `__FILE__` only in the gate-side `ref.h`). | as stated |
| **R3 objects** | `obj_equiv.py --strict-data`, new: keep immediates; resolve RIP-relative displacements through relocations (`objdump -dr`) to `symbol+addend`; compare each referenced data object (`.rodata`, `.data`, `CSWTCH`, constant pools) **by content**; an optional rename map. | 0 changed, 0 gone, 0 new (split steps: the listed functions only) |
| **R4 censuses** | undefined, mutable and normalized defined symbols; race-census keys with `fn=` masked; `layout.txt`. | identical (phase 8: `layout.txt` diff expected and recorded) |
| **R5 semantics** | `golden_bits`, `fp_replay`, `api_sweep`, `wisdom_replay`, `gates.txt`. | byte-identical after LF normalization |
| **R6 milestone** | `gauntlet.py verify` on pow2 / primes / 2d-small / 3d-pow2 against a scratch store; precision columns compared exactly. | identical; run at the end of phases 3, 6, 7 and 8 |

### Risks the gate is built around

- **Nondeterministic cells.** `c2c.split.ip.nat` is known to flap. Capture the
  reference with a high repeat count and treat any change in NONDETERMINISTIC status as
  red.
- **`-march=native` ties the O3 object to the VM's CPU model.** The model is in the key,
  and the `-O2` object is the portable anchor.
- **Header splits can reorder `static` initializers, tentative definitions or redefined
  macros.** `--allow-reorder` at R2 is never enough on its own; R3-strict and R5 must
  also be green.
- **A moved header included twice duplicates a `static const` table.** Only R3-strict
  and the defined census catch that.
- **The owner's machines need their own references.** They are Windows/WSL and the i9 /
  Zen 4 laptop. A reference from this VM proves nothing about them, and the `.o` byte
  compare is valid only within one toolchain.

## 8. Effort and order of work

- **Phase 0 (the gate) comes first.** No file moves until HEAD-vs-HEAD is green at both
  ISAs.
- **Phases 1-3 are mechanical,** each a single sitting, with byte identity as the proof.
- **Phases 4-5 are about a dozen small commits.**
- **Phases 6-7 are the substantive part.** They need one careful commit per create
  tier.
- **Phase 8 is a large mechanical rename, or none with D3's anonymous-struct option.**
- **Nothing here changes a plan, a route, a wisdom row on disk or a bit of output,** except
  the explicitly separate D4/D5 items.

## 9. Deferred findings (fix after the new layout is integrated)

- **F1. Split out-of-place creates re-measure on every call and bank nothing.**
  Found by the phase-0 self-check (2026-09-27). The split OOP tuner (`oop_dp.h`,
  `oop_auto.h`) times its candidates with `__rdtsc` at every create, never
  writes a verdict to the store, and never increments the create-race counter.
  So the plan, and the output bits, can differ from one process to the next.
  Examples:
  - c2c split oop N=45 K=4 gave one fingerprint and three output patterns in
    eight runs;
  - a replay row (c2c N=32768 K=4 scrambled) gave six in six.

  The cost is a create-time measurement on every plan and non-reproducible
  plans. The owner's decision: fix it after the separation lands, in `split/`.
  Until then the gate tolerates it through the variant-set protocol
  (`src/tools/baseline/api_sweep.py`): only keys the reference saw flip get the
  structure-only waiver, and every deterministic key stays bit-exact.

- **F2. Split K=1 routes.** Split vectorizes across the K lanes (4 per vector
  at AVX2), so a K=1 split transform leaves most of each vector idle. Yet split
  K=1 is wired today:
  - `VFFT_K1_SP_*` routes in `k1sp`, `vfft_oop_plan_create_k1`;
  - 14 `lay=split` K=1 rows in `src/wisdom/wisdom2_oop.txt`;
  - the API sweep's split out-of-place N=256/1024/4096 K=1 cells build it.

  The owner noted the point on 2026-09-27. Retire or keep: decide after the
  separation, measuring the classic split out-of-place path at those N first.

- **F3. An interleaved K=1 request treats a cell that holds only a SPLIT kind-3
  row as "present".** This was found while splitting the kind-3 record. The
  pre-split reader filled one entry from the `lay=il`, `lay=split` and vintage
  rows, and returned "found" when either axis decoded. The interleaved
  consumers test that pointer:
  - `k1_commit.h`'s scrambled writer-band early return;
  - the per-T row copy;
  - `c2c_oop_create.h`'s T-row copy.

  So an IL request at a split-only cell sees a present entry whose IL axis is
  unraced. To keep behaviour identical, `vw2_oop_lookup_k1_il_cell` preserves
  this through `vw2_oop_k1_row_present` (common). It tests existence, not
  decode: the one difference from the old reader is a cell whose rows are all
  undecodable, which the old reader called absent.

  Decide after the separation whether IL should ignore split rows entirely.
  That is likely the right semantics: two libraries, two verdicts.

- **F4. The migrator's kind-3 reader check cannot find what it migrated.**
  `vw2_migrate_oop_gate` banks legacy kind-3 lines as wildcard rows
  (`q=* ord=* place=*`). Its reader check then looks them up through the
  kind-3 scan, which matches placement exactly (`key.pl == VW2_PL_OOP`), so
  every kind-3 cell reports MISSED.
  - Measured with `src/tools/baseline/mig_probe.c`: 4 of 4 kind-3 cells missed
    on the pre-split code, and identically on the split code.
  - The migrated rows are correct: both builds write byte-identical stores.
  - Pre-existing and not caused by the separation. Fix the check (or the
    scan's wildcard handling) after it.
