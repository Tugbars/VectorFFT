# Refactor-safety ladder, run OFF-HOST on Zen 4 — 2026-09-03

Not a re-stamp. `build_tuned/baseline/` is untouched: it is bound to the
14900KF at gcc 15.2 and nothing captured here may be diffed against it.
This is the ladder run **within one flags key on this host** (gcc 16.2.0
UCRT64, Ryzen 5 PRO 8640HS), as evidence for the 14900KF run that decides
whether the step lands and the baseline re-stamps.

## Step declaration

**Step A — harness-neutral (tooling, benches, docs, wisdom).** `build.py`
(CC search, subprocess PATH, `-mno-avx512f` clamp unless `VFFT_ISA=avx512`),
race-on-miss in `bench_1d_vs_mkl.c` / `bench_1d_vs_fftw.c`, three `zen4_*`
drivers, host-derived fixture in `oop_width_gate.h`, `obj_equiv.py` /
`sym_census.py` gained `NM` / `OBJDUMP` env overrides, `.gitignore` negation
for `generated/wisdom/*/*.txt`, `docs/design/cpu_discovery.md`,
`docs/performance/v1_0_results.md` §10, `generated/wisdom/Zen4/` (new store;
`generated/*.txt` untouched — `wisdom_store.sha256` still holds).

**Step B — library, class MERGE.** `src/core/support/cpu_cache.h` (vendor
detection, AMD `0x8000001D/1E` leaves, OS tier, `vfft_cpu_host_tag`,
`VFFT_CPU_DISABLE_CPUID` guard; `VFFT_L1D_DISCOVER` default unchanged) and
the `@meta` host stamp at `vw2_open` in `src/core/vfft.c`.

Expected diffs, declared before the run: `_bundle_load` and
`_vfft_cpu_cache_fill` bodies; +2 mutable objects (`tag`, `done`);
+`GetLogicalProcessorInformationEx` undefined; golden bits and fingerprint
replay byte-identical; race census identical; zero new warnings.

## "Before" tree

No archive existed. The pristine tree was reconstructed in scratch by
reversing every Step B edit with the exact old/new strings; each reversal
matched its anchor (a mismatch would have refused), so the reconstruction
is exact. Both trees then built with the same compiler.

## Results (pristine → today, same key)

| rung | result |
|---|---|
| 2.2 warnings, identity flags `-O2 -mavx2 -mfma` | 2 → 2, identical text |
| 2.2 warnings, `VFFT_WARN` key | 19 → 19, identical text |
| 2.3 `obj_equiv` (MERGE, informational) | 5 bodies changed: `_bundle_load`, `_vfft_cpu_cache_fill.constprop.0`, `_il2d_axis_race`, `_il2d_real_rowrace`, `_k1z_race_and_bank.part.0` (the last three inline the cache fill) |
| 2.3 symbols | 1027 → 1026: `bluestein_wisdom_load.isra.0` gone — **inlined into `_bundle_load`** together with `vfft_c2r_path_load` (both out-of-line calls vanish from `_bundle_load`, its indirect file-op calls go 5 → 9). Source untouched; inliner re-decided on the enlarged caller. NOT in the declaration — recorded as an explained diff. |
| 2.4 race census | identical (13 lines) |
| 2.5 defined | +`tag.N`, +`done.N`, −the isra clone |
| 2.5 undefined | +`__imp_GetLogicalProcessorInformationEx`, +`__imp_GetLastError` (the second was not listed; same OS-tier call) |
| 2.5 mutable | 48 → 50 (`tag.N`, `done.N`) |
| **2.7 golden bits** | **BYTE-IDENTICAL**, 35 cells, repeat 3 |
| **2.8 fingerprint replay** | **BYTE-IDENTICAL**, 45 cells, repeat 3 |
| 11 gates touched | `zturn_wisdom_width_gate` 6/6 · `wisdom2_2d_gate`, `wisdom2_g0_gate`, `wisdom2_real_gate` all pass · `vfft_ilp_front_gate` pass |

## What remains for the 14900KF

1. Commit A, then B with the declaration above (add `GetLastError` and the
   inliner note to it).
2. Run the ladder against `8588720d` with gcc 15.2. Golden bits and
   fingerprint must be byte-identical there too; the isra clone may or may
   not vanish under that compiler — either is fine, both are explained.
3. Re-stamp with `capture_baseline.py --out build_tuned/baseline --repeat 5`
   and record the new SHA. Truncate, never append.
4. `warnings_baseline.txt` is compiler-specific (2 on gcc 15.2, 19 on
   gcc 16.2 for the SAME source); keep it on the baseline compiler.

Scratch artifacts (this session only): identity objects, census files,
`cap_pristine/`, `cap_today/`, the reconstructed tree.
