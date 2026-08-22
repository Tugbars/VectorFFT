# FFTW arm for the canonical bench — design of record

**Status:** 🟡 **DESIGN APPROVED FOR IMPLEMENTATION — NO CODE HAS LANDED.**
Research complete (recon ×5, three competing proposals, two judgments, three
adversarial verifications, ~40 compiled probes). Every claim below is labeled
**[M]** measured/compiled on this host, **[D]** read from source or the FFTW
manual, **[I]** inferred. Nothing in `src/`, `build_tuned/benches/bench_1d_vs_mkl.c`,
`build.py` or `run_bench.py` was modified to produce it.

**Decision trail (gitignored, `docs/research/fftw_bench/`):** `SYNTHESIS.md` is the
recommendation this document supersedes; `verify/VERIFY_{correctness,feasibility,coverage}.md`
are the three adversarial passes whose refutations are folded in here; `recon/01..05`
are the fact bases; probes under `recon/probe/`, `probes/`, `probe/`, `verify/probe/`,
`scratch/vfeas/`.

🔎 **Completeness-critic pass (2026-08-22), folded in.** Every number in this document was
recomputed from the tree; five did not reconcile and are corrected in place, each marked 🔎:
the DLL export count (77 → **123**), the codelet genus histogram (94/94/`_avx_128` →
**376/372/`avx2_128`**), the blocked-configuration count (7 → **6**, plus a wider `--mt`
footgun), the remaining-planning-cost cell count (~157 → **92**) and the `PRESERVE_INPUT`
tax (**not monotonic in N**; re-measured median-of-15). Two of the document's own open
questions were closed with new probes under `docs/research/fftw_bench/critic/`: the P0.75
wisdom fix now verified for split **r2c**, split **c2r** and **rank-2** split
(`c1_p075.c`), and the rank-2 split real `fftw_iodim` triples for rows 4/5 written down and
checked elementwise (`c2_2dsplit.c`). One new gap was opened: the arm-order scheduler is
specified for 3 arms while `--kzb` has 4 and `--ilmt` has 5 (§4.4).

**Related:** `docs/performance/v1_0_results.md:901-974` (the standing FFTW liability this
campaign retracts) · memory node `canonical_mkl_bench.md` (the law) ·
`feedback_fftw_measure.md` (FFTW_MEASURE is law).

🔴 **Read §6.1 before §5.** The single largest finding of the campaign was produced by
the *verification* pass, not the design pass: **FFTW's wisdom key hashes the split-plane
pointer delta**, which makes the whole reproducibility story conditional on a plane-layout
change the design had ruled out. That conflict is resolved here, and the resolution
changes what P5 costs.

---

## 1 · The decision

### 1.1 One paragraph

`bench_1d_vs_mkl.c` keeps its name, its file boundary and its 8 positional arguments, and
gains one **modifier flag `--ref=mkl|fftw|both`**. The reference library sits behind a
small vtable (`refbackend/ref.h` + `ref_time.h` + `ref_mkl.h` + `ref_fftw.h` — headers, not
translation units, see §5.0) so that one timing core replaces the 21 hand-copied
warmup-10/best-of-5 idioms and every arm — ours, MKL's, FFTW's — is the same C shape.
**FFTW is bound at runtime** (`LoadLibraryA` + `GetProcAddress`, no `fftw3.lib` on the link
line). `--ref=mkl` is byte-for-byte today's schedule, so the banked corpus stays valid and
`run_bench.py` is untouched until it is deliberately extended.

### 1.2 The canonical-bench law: satisfied, no override requested

The law reads: *bench_1d_vs_mkl.c is always the canonical bench against a reference library;
never invent a new harness; add a `--mode`.* This design adds no harness, no binary, no
filename — and does not even add a mode. `--ref=` occupies exactly the grammatical category
`--mt` already occupies (parsed in the same argv-shifting loop at `:3423`, combined with
`--oop`/`--2d`/`--2dr2c`/`--2dc2r`/`--r2c` at `:138-142`) and `--tcut=`/`--tcuttw=` after it.
The request's wording — *"an FFTW version of bench_1d_vs_mkl.c"* — is honoured **by coverage,
not by copying**: the FFTW arm reaches the same mode surface at the same protocol quality,
inside the canonical file.

🔴 **The sibling file was already tried, and deleting it is why the law exists.**
`build_tuned/benches/bench_1d_vs_fftw.c` existed (456 lines). Its own header read
*"Companion to bench_1d_vs_mkl.c. Same 207-cell grid…"*. It was deleted at `6f9681af`
(2026-07-29) while the MKL bench grew to 4,494 lines — and its output, *"beats FFTW3
202/207, median 3.21×"*, is **still published** at `docs/performance/v1_0_results.md:901-974`
behind a dead link (`:910`), produced on the dead stride-fft engine, best-of-5, no
cachebust/pace/flip/control, lane-major split only at K∈{4,32,256} — precisely the shape
where a quick-look measures FFTW's adapter tax at 3.1–9.3× [M, unpaced]. **That number is
mostly the descriptor, not the kernel** [I]. Retracting it is P0.5 and it ships before any
new FFTW number does.

**What the sibling proposal got right is kept.** The shared timing core is adopted; only
its file boundary is rejected. And the law's own premise had to be corrected to make the
argument honest: *"the canonical bench is the only harness that implements the full
protocol"* **is not true today** [M] — the warmup-10/best-of-5 idiom is hand-copied 21×,
only `_zr2c_med5:2483` + `ZR2C_TIME:2493` obey the house MEDIANS law, `flip` is one bit with
no N-arm scheduler, only `--ilmt` has a control arm, and four modes ignore `argv[2]` and
hardcode CSV paths (`:3629`, `:3815`, `:3873`, `:4046`) — one of them writing the
**git-tracked** `src/dag-fft-compiler/generator/generated/c2r_path.txt` with `"w"`. The
shared core is what turns that premise from convention into invariant, **inside** the
canonical file where the law wants it.

### 1.3 🔴 Why runtime binding, not a link-order fix

`mkl_rt.2.dll` exports **191** `fftw*` symbols and `mkl_rt.lib` defines 92 `T fftw_*`
symbols — a complete FFTW-3.3.4 wrapper including `fftw_plan_guru_split_dft`,
`fftw_init_threads` and `fftw_plan_with_nthreads`. `build.py:277` appends `mkl_rt.lib`;
`build.py:282` appends `fftw3.lib`; MinGW `ld` resolves left-to-right. **Both link orders
were actually built** [M]:

| link order | `fftw_version` | `fftw_sprint_plan` | `fftw_flops` |
|---|---|---|---|
| `mkl_rt` first (today's `build.py`) | `"FFTW 3.3.4 wrappers to  Intel oneMKL"` | `(null)` | `-1` |
| `fftw3` first | `"fftw-3.3.10-avx-avx2-avx2_128"` | real plan string | real |

⇒ **an in-file FFTW arm on today's link line would measure MKL against MKL.** Moving the
link block is a *convention*; runtime binding is *structural*, and it strictly dominates:
under the link fix, any FFTW symbol **absent** from `fftw3.lib` still silently resolves to
MKL's wrapper, and `fftw_init_threads` is exactly such a symbol. The shim omits the thread
functions from its bind list, so an MT FFTW arm becomes physically unwritable rather than
falsely green.

**The shim is verified, not assumed** [M] — `docs/research/fftw_bench/scratch/vfeas/v4_shim.c`:
`__typeof__(f) *p_##f` compiles against the `dllimport`-declared prototypes; `GetProcAddress`
works on the `fftw_version` **data** export; the PE import table shows **zero** fftw imports;
and `v4_shim_bothlibs.exe` (shim + `mkl_rt.lib` + `fftw3.lib` on one line) still has zero
fftw imports and reports genuine FFTW with MKL live in-process. 🟢 **`build.py` therefore
needs no link change at all**, and the `find_fftw()` edit becomes optional hygiene rather
than a prerequisite. One rule: `LoadLibraryA` must take an **absolute** path (default
`C:/vcpkg/installed/x64-windows/bin/fftw3.dll`, overridable by `VFFT_FFTW_DLL`), or it
inherits the exe-dir-first search that lets the pre-staged `build_tuned/benches/fftw3.dll`
win silently.

**Permanent guard, in the banner and in every CSV row:** print `fftw_version`, and refuse to
run if it contains `oneMKL` or `Intel`.

### 1.4 Coverage today

19 leading flags (`:3421-3553`) = 17 explicit mode flags + 2 emitter modifiers
(`--tcut=`, `--tcuttw=`), plus the implicit no-flag default = **18 modes**. `--mt` is
additionally a cross-cutting modifier.

| | count | modes |
|---|---:|---|
| **armed** | **15** | default, `--oop`, `--2d`, `--2dr2c`, `--2dc2r`, `--r2c`, `--c2r`, `--zr2c`, `--pad`, `--padr2c`, `--k1dir`, `--k1nat`, `--k1noop`, `--k1zip`, `--kzb` |
| **partially armed** | **1** | `--ilmt` — a single-threaded FFTW column only (§2 row 13) |
| **blocked** | **1** | `--mt`, **plus its 5 modifier combinations = 6 blocked banked configurations** (§3.4) |
| **no meaningful counterpart** | **1** | `--c2rcalib` (no MKL arm either) |

Two qualifications inside the armed 15, both of which the first coverage claim missed [M]:
`--zr2c` is armed on `run_zr2c_cell` only — its second cell runner `run_zr2c_fd_cell:2647`
is a **12-arm race in which every arm is ours** and MKL appears only as an untimed gate
oracle (`:2673-2683`), so there is no reference timing slot to fill; and the default mode
needs **two** FFTW configurations, not one, because it writes three structurally different
passes into one CSV (§2 rows 1a/1b).

🔎 **"Armed" is also not "at parity", and the count above hides that.** Re-partitioned by
*"does this row yield a ratio you could print beside the MKL ratio without a paragraph of
caveats"*: **8 rows at parity** (1b, `--k1dir`, `--k1nat`, `--k1noop`, `--k1zip`, `--zr2c` 9a,
`--kzb`, `--r2c`) · **5 armed-but-not-equivalent** (1a and `--oop` are roundtrip-gated *and*
cell-limited by `max_NK`; `--pad`'s `Kp` arm has no MKL counterpart; `--c2r`'s verdict column
uses `PRESERVE_INPUT`, a different transform contract from FFTW's default, so it is not FFTW's
floor; `--padr2c`) · **3 conditional** (`--2d`, `--2dr2c`, `--2dc2r` — and `--2dc2r`'s
`reps_per_sample=1` forces a new CSV stem, so it is not comparable to the banked 2D corpus at
all) · **4 not armed** (`--mt`+combos, `--ilmt`'s MT column, `--c2rcalib`, `--zr2c` 9b).
Publish that partition, not "15 armed", the first time an FFTW table ships.

🔴 **The `max_NK` ceiling is an undefined coverage hole, not just a cost knob.** It converts
rows 1a, 2 and 10 from *armed* to *armed for whichever cells fall under it* — and since 92 of
the 190 default cells were never planned (§3.2), **nobody currently knows what fraction of the
biggest mode the FFTW column will actually cover.** Pin the ceiling with a number before P5.

🔴 **"Armed" is not "cheap".** After §3.2, the largest `N·K` cells of default/`--oop`/`--pad`
cost minutes of `FFTW_MEASURE` planning each — a single measured cell at N=60060 K=256 took
**643.6 s** [M]. Those cells print `fftw_ns=n/a reason=plan-cost` above a published ceiling
unless the deterministic-plane fix lands.

---

## 2 · 🥇 THE MODE MAPPING TABLE

This table is the deliverable. Everything else supports it.

**Layout shorthand** (all verified against a naive reference DFT, maxerr ≈2.4e-14 [M]):

- **LM-split** = `fftw_plan_guru_split_dft(rank=1, dims{{N,K,K}}, hrank=1, howmany{{K,1,1}},
  ri,ii,ro,io, flags)` — 🔴 **strides in units of `double`, and there is no ×2 factor for our
  layout** (`reference.texi:1349-1355` [D]; our planes are independent, so the element stride
  is K, not 2K [M]). `guru_split_dft` **has no `sign` argument**: backward = swap `(ii,ri)`
  and `(io,ro)` [D + M, verified correct at 7.105e-15].
- **LM-il** = `fftw_plan_guru_dft`, same dims/howmany on `fftw_complex` — strides in
  **complex** units.
- **TC-il** (FFTW's home) = `fftw_plan_many_dft(1,&N,K, in,NULL,1,N, out,NULL,1,N, sign, f)`.
- **CCE-il** = `fftw_plan_{many_,}dft_r2c/c2r`, N/2+1 interleaved complex — **byte-identical
  to MKL's CCE output**, maxerr 7.2e-15 [M].

**Gate classes**, named in every row and published as a CSV column:
**X** cross-engine elementwise per direction · **R** elementwise vs a naive O(N²) DFT
(reuse `_r2c_ref_check:2128-2148`, cap N≤4096) · **N** each engine gets its own roundtrip,
row carries the `two-roundtrips-are-not-the-same-spectrum` caveat · **S** ref-vs-ref
self-check (FFTW mirror output == FFTW home output after the layout transpose), the
`--kzb:946-990` "mirror strides suspect" shape.

🔴 **Hard constraint #4 answered: no FFTW arm is ever roundtrip-gated.** FFTW is natural in
every API, so even in class-N modes — where *our* forward is digit-reversed and cannot be
compared elementwise to anything — the FFTW arm carries `fftw_gate=xmkl`, a real elementwise
number against MKL. That is strictly *stronger* than what the MKL arm gets in those modes
today. It is mandatory, not optional: a missed pointer swap in a split backward plan is a
conjugation error that a roundtrip survives silently ⇒ **every backward split arm is class X
or R, never N**.

**Buffer / alignment handling is uniform for all rows** and is stated once here rather than
repeated per row: `alloc_d` → `posix_memalign(64)` planes are kept for the vfft and MKL arms
(`:98-107`, `src/core/engine/plan.h:21-29`); **every FFTW split arm gets a per-cell
FFTW-owned plane pair carved from ONE block at a fixed size-derived offset** (§3.1); FFTW
alignment class is 0 everywhere; plans are built on the **exact** pointers that will be
timed (`REF_ASSERT_BOUND`); new-array execute everywhere; plans are built **before** the
cachebust barrier (`REF_ASSERT_PREBARRIER`) and followed by `pace(cool_ms)`; setup order is
**`alloc → plan → fill → gate → warmup → time`**. All plans:
`FFTW_MEASURE | FFTW_WISDOM_ONLY` first, plain `FFTW_MEASURE` on miss. `FFTW_ESTIMATE` is
unrepresentable in the type; `FFTW_UNALIGNED` is banned.

### 2.1 The table

| # | Mode (cells) | FFTW plan fn — verdict arm | Diagnostic arms | Flags | Layout (verdict / diag) | Placement | Gate | Verdict |
|---|---|---|---|---|---|---|---|---|
| **1a** | *(default)* 1D c2c K-walk + primes — 67 cells at K=4, **190 at `run_bench.py`'s default `--K 4,32,256`** [M] | `fftw_plan_guru_split_dft(1,{{N,K,K}},1,{{K,1,1}},ri,ii,ri,ii,f)` | `home`: `plan_many_dft` TC-il on a transposed copy; `loop`: our K=1 engine ×K on FFTW's memory | MEASURE | LM-split / TC-il | **in-place** (`ro==ri`) — mirrors `mkl_make:291-310` (`DFTI_REAL_REAL`, `STRIDES={0,K}`, `DISTANCE=1`, INPLACE) | **N** (ours) + `fftw_gate=xmkl` + **S** | ✅ armed — 🔴 **plan-cost ceiling applies** (§3.2) |
| **1b** | *(default)* the **K=1 kind-4 sub-pass**, `:4141-4177` → `run_k1z_cell`, ~10 cells **into the same CSV** | `fftw_plan_dft_1d(N, zi, zo, FFTW_FORWARD, f)` | — | MEASURE | il, K=1 | **OOP** — matches `k1z_time_mkl:529` (`DFTI_COMPLEX`, `DFTI_NOT_INPLACE`, no strides, no `NUMBER_OF_TRANSFORMS`) | **N** + xmkl | ✅ armed — 🔴 **a separate row; applying 1a's config here would race split-LM-in-place-K=4 against interleaved-K=1-OOP** [M] |
| **2** | `--oop` — **51 cells** (the wisdom walk filtered to pow2 `N>=8` and `K%8==0`, `:4194-4195`; counted from `spike_wisdom.txt` [M]) | as 1a, with distinct `ro,io` planes | `home`, `loop` | MEASURE | LM-split / TC-il | OOP — mirrors `mkl_make_oop:1349` | **X** for LEAF/BAILEY2 natural cells; **N**+xmkl for MODEB (`:1524-1525`) | ✅ armed |
| **3** | `--2d` c2c 64²…512² (4) | `fftw_plan_guru_split_dft(2,{{N1,N2,N2},{N2,1,1}},0,NULL,…)` | `home`: `plan_dft_2d` | MEASURE | split rank-2 / il rank-2 | **OOP** — matches `bench_mkl_2d:1647` (ours runs in-place at `time_2d:1627`) | **N** + xmkl + **S** | ✅ armed. 🔴 the earlier justification *"rank-2 split in-place plans NULL"* is **false [M]** — it plans (0.303 s at 512²). Keep OOP because it mirrors the MKL arm, not because in-place is illegal |
| **4** | `--2dr2c` (4) | `fftw_plan_guru_split_dft_r2c(2, dims{{N1,N2,N2/2+1},{N2,1,1}}, 0, NULL, xr, ro, io, f)` | `home`: `plan_dft_r2c_2d` (CCE-il) | MEASURE | split / CCE-il | OOP | **N** + xmkl; **R** at the smallest cell | ✅ armed — 🔎 **geometry SETTLED by the completeness critic** (`docs/research/fftw_bench/critic/c2_2dsplit.c`): the triple above **plans under MEASURE and matches `fftw_plan_dft_r2c_2d` elementwise at maxerr 1.137e-13** [M] at 64². `dims` are the **real** array's extents; the last dimension's *output* stride is in complex units (`N2/2+1`), the input stride in reals (`N2`). Neither engine has an adapter here (`:1936`/`:1940`), so `mir`≡`home` on the input side |
| **5** | `--2dc2r` (4) | `fftw_plan_guru_split_dft_c2r(2, dims{{N1,N2/2+1,N2},{N2,1,1}}, 0, NULL, ro, io, yr, f)` | `home`: `plan_dft_c2r_2d` | MEASURE — 🔴 **`PRESERVE_INPUT` unavailable at rank 2 — for the interleaved AND the split path** [M, confirmed three times; the split leg is new, `critic/c2_2dsplit.c`: `guru_split_dft_c2r(2,…)+PRESERVE_INPUT` → **NULL**, plain → OK, roundtrip 3.183e-12] | split / CCE-il | OOP, **destructive + untimed pristine restore, `reps_per_sample=1`, identical sample shape for every arm** | **R** at the smallest cell else **N**+xmkl, on a fresh untimed execute | ⚠️ armed, geometry settled (as row 4), **and labelled unfair** — held behind auditing the existing MKL one-descriptor-both-directions bug (`:2081`/`:2090`), and `reps_per_sample=1` **changes `vfft_ns`/`mkl_ns`** away from banked values ⇒ new CSV stem |
| **6** | `--r2c` 1D, K lanes (15) | 🔴 **`fftw_plan_many_dft_r2c(1,&N,K, in,NULL,1,N, out,NULL,1,N/2+1, f)` — FFTW's HOME layout, on the transpose the bench already builds untimed at `:2211-2213`** | `mir`: `guru_split_dft_r2c(1,{{N,K,K}},1,{{K,1,1}},…)`; `loop` | MEASURE | CCE-il TC (verdict) / LM-split (diag) | OOP | **R** (N≤4096) + **X** + **S** | ✅ armed — **policy corrected**, see §4.2 |
| **7** | `--c2r` 1D (18) | `fftw_plan_many_dft_c2r(…)` with **`MEASURE\|FFTW_PRESERVE_INPUT`** (home layout, as row 6) | `destroy` column + a **refill-only arm** so the memcpy tax is visible and never subtracted; `mir` via `guru_split_dft_c2r` | §3.3 | CCE-il TC / LM-split | OOP | **R** / bwd vs `N·x`, per direction, on an **untimed** execute | ✅ armed — 🔴 fix the 7-column header (`:3901`) under 8-field rows (`:2458`) in the writer first |
| **8** | `--c2rcalib` | **none** | — | — | — | — | — | ❌ **no meaningful counterpart.** The mode has no MKL arm either (no `DftiCreateDescriptor` in the branch, `:3873-3894`); it calibrates our *two internal* c2r paths and writes `c2r_path.txt`. `--ref=fftw` must be **rejected with an error**, not silently ignored. 🔴 that file is git-tracked and opened `"w"` — pre-existing, own ticket |
| **9a** | `--zr2c` → `run_zr2c_cell` (5: N∈{512,2048,8192,16384,65536}) | `fftw_plan_dft_r2c_1d(N, x, (fftw_complex*)x, f)` **in-place on the N+2 plane** (`:2757`) + a **separate** `fftw_plan_dft_c2r_1d` twin | — none needed | MEASURE | **CCE-il — FFTW home == MKL home == our D2 output; nobody adapts** | **in-place** — mirrors MKL's `DFTI_REAL`+CCE+INPLACE, which `:2473` calls *"its BEST arm"* | **X** fwd elementwise (verified vs MKL at 1.1e-16…3.2e-16 [M]); bwd vs `N·x`; per direction | ✅ 🥇 **PILOT** — but re-cost it: 🔎 **up to 8 timed arms today, 10 with FFTW** (`tmf`,`tmb` MKL · `tf`,`tb` ours · conditional CASCADE `tcf`,`tcb` `:2964` · conditional NATORDER-IP `tnf`,`tnb` `:3047` — 8 distinct `ZR2C_TIME` destinations, not 10; the cited `:2960`/`:2993` are off by 4 and 54 lines). Even so the pilot is a **10-arm** row, which is exactly why §4.4's 3-arm scheduler has to be generalised before P2, not after |
| **9b** | `--zr2c` → `run_zr2c_fd_cell:2647` (always also runs, `:4055-4070`) | **none** | — | — | — | — | — | ❌ **no reference timing slot** — 12 arms, all ours (`{r2c,c2r}×{OOP,IP}×{r0,r1,W}`, `:2688-2695`); MKL is an untimed gate oracle only. FFTW may optionally serve as a *second* oracle. 🔴 its CSV `:4046` is opened `"w"` unconditionally ⇒ truncated on every isolated launch (pre-existing) |
| **10** | `--pad` K∈{7,11,15,19,23} (15) | as 1a, at the **true** K and at the padded `Kp` (2 plans/cell) — FFTW has no padding concept | `home` | MEASURE | LM-split | in-place | **N** + xmkl | ✅ armed — 🔴 the `Kp` plan measures an extent **MKL is never measured at** (`:3194`, `reps_for(totP)` vs `reps_for(totK)`) ⇒ it is a `role=DIAGNOSTIC` column, never a verdict. Mode ignores `argv[2]` and hardcodes its CSV (`:3629`) ⇒ must route through `csv_for(ref)`. 🔎 **`--pad` is also nested entirely inside `#ifdef VFFT_HAS_MKL` (`:3624-3660`)** — with `--ref=fftw` the flag is a no-op unless MKL is still linked, so v1 must keep `--mkl` on the line (§7 already rejects dropping it) or un-nest the branch |
| **11** | `--padr2c` (15) | as row 6 (home CCE-il), padded/tight twin | `mir` pair | MEASURE | CCE-il TC / LM-split | OOP | **N** + xmkl — 🔴 **not R**: the mode gates pad-vs-tight only (`:3332-3345`) and `_r2c_ref_check:2128` hardcodes stride `K`, not `Kp` | ✅ armed once the gate class is corrected; hardcodes its CSV (`:3815`); **also nested inside `#ifdef VFFT_HAS_MKL` (`:3801-3846`)**, same consequence as row 10 |
| **12** | `--mt` (146 cells) **+ the 6 modifier combinations** `--oop --mt`, `--2d --mt`, `--2dr2c --mt`, `--2dc2r --mt`, `--r2c --mt`, plain `--1d --mt` (`:3560-3579` enumerates six distinct banked CSVs) | — | — | — | — | — | — | 🔴 **BLOCKED — no threaded FFTW in this build.** Unblock condition in §3.4. Rows print `fftw_ns=n/a reason=no-threads-in-build build=fftw-3.3.10-avx-avx2-avx2_128` |
| **13** | `--ilmt` (18) | 🔴 **PARTIAL: a single-threaded arm is buildable today** — `fftw_plan_many_dft(1,&N,K,in,NULL,1,N,out,NULL,1,N,FFTW_FORWARD,f)` on the mode's own `VFFT_BATCH_TRANSFORM_CONTIGUOUS` interleaved natural OOP memory (`:1211-1219`): byte-identical memory on both sides, FFTW's absolute home turf | — | MEASURE | TC-il | OOP | **X** | ⚠️ **partially armed.** `run_ilmt_cell:1273-1310` already times `mst = ilmt_time_mkl(…,1)` — MKL **single-threaded** — and `:1319-1322` makes `mbest=min(mmt,mst)` the headline denominator, so an ST reference arm is already a first-class citizen here. Publish `fftw_st_ns` as **its own column** with `fftw_nthreads=1`; **never** fold it into `mbest` (that would conflate two libraries in one number) and never ratio it against `omt`. `fftw_mt_ns=n/a` |
| **14** | `--k1dir` | `fftw_plan_dft_1d(N, z, z, FFTW_FORWARD, f)` **plus a separate backward plan** | — | MEASURE | il, K=1 | **in-place** | 🔴 **X**, not N | ✅ armed — **corrected [M]:** `:3462-3468` sets `g_k1zip=1; g_k1nat=1; g_k1dir=1`, so `run_k1z_cell:606` sets `VFFT_ORDER_NATURAL` and the cross-engine elementwise block at `:652-691` runs. Row 14 = row 15 plus a backward plan; nothing else differs |
| **15** | `--k1nat` | `fftw_plan_dft_1d(N, z, z, FFTW_FORWARD, f)` | — | MEASURE | il, K=1 | 🔴 **in-place**, not OOP | **X** | ✅ armed — **corrected [M]:** `:3476-3487` sets `g_k1zip=1` as well as `g_k1nat=1`, so `run_k1z_cell:604` → `VFFT_INPLACE` and `k1z_time_mkl:545` → `DFTI_INPLACE`. An OOP FFTW plan here would be a third placement in a two-in-place cell, systematically flattering FFTW, and the elementwise gate passes it trivially |
| **16** | `--k1noop` | `fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, f)` | — | MEASURE | il, K=1 — **FFTW's absolute home turf and ours** | OOP | **X** | ✅ armed 🥇 the fairest cells in the arsenal |
| **17** | `--k1zip` | `fftw_plan_dft_1d(N, z, z, FFTW_FORWARD, f)` | — | MEASURE | il, K=1 | in-place | **N** + xmkl | ✅ armed |
| **18** | `--kzb` K∈{2,3,4} (24) | `mir`: `fftw_plan_guru_dft(1,{{N,K,K}},1,{{K,1,1}},…)` — strides in **complex** units | `home`: `plan_many_dft`, `idist=N`, on the **existing** `z0h` buffer (`:932-940`); `loop`: `kzb_time_loop:857`, unchanged | MEASURE | LM-il / TC-il | OOP | **X** + **S** | ✅ armed 🥇 **the mirror/home/loop TEMPLATE** (`:753-758`, `:855-860`) |
| **19** | `--tcut=` / `--tcuttw=` | not modes — emitter modifiers of row 1; the FFTW column rides along unchanged | inherits | inherits | inherits | inherits | inherits | ✅ |

### 2.2 What every armed row publishes

`ref_id` (the `fftw_version` string) · `ref_layout` · `ref_role` ∈ {MIRROR, HOME, LOOP, OURS,
CONTROL} · `mkl_role` · `ref_flags` · `plan_src` ∈ {wisdom, cold} · `plan_id` (FNV-1a of
`fftw_sprint_plan`) · `plan_ms` · `wis` ∈ {hit, miss} · `gate_class` · `timer` · `stat` ·
`sched` · and for real modes `fftw_c2r_input`. **A number without its layout label is not
publishable.**

🔴 **`plan_id` is diagnosis, not mitigation.** Eight isolated protocol-shaped launches of one
cell (N=1024 K=4, core 2, HIGH, median of 15) produced **5 distinct plans and 8823→9652 ns =
9.4 % spread**, grouped by plan rather than by launch order, with within-process spread ~1 %
[M]. A table showing five `plan_id`s in one mode is a table to **discard**, not to interpret.
The cure is §3.1's deterministic plane layout, which gave 6/6 identical `plan_id` across
isolated processes [M].

🔴 **`free()` the `sprint_plan` string — never `fftw_free()`.** `api/print-plan.c:33` uses
plain `malloc`; `kernel/kalloc.c:109-111,141` routes `fftw_free` to `_aligned_free`. 200
cycles: `fftw_free` → `0xC0000374 STATUS_HEAP_CORRUPTION`; libc `free` → clean [M,
reproduced by the critic]. 🔎 **The `free()` half is CRT-contingent, and the doc must say so:**
it is safe here only because `fftw3.dll` imports `api-ms-win-crt-heap-l1-1-0.dll` (UCRT) and so
does the mingw152 exe — one heap, verified with `objdump -p` [M]. Under a msvcrt-based mingw
or an icx/MSVC build the two heaps split and `free()` becomes the corrupting call. **Make the
code assert the shared heap or simply leak the string** (≈400 B per plan, a few dozen plans per
process) — it is diagnosis output, not a hot path. It did
**not** reproduce in a single-shot smoke test and only became reliable at loop scale — the
exact class of bug that passes a pilot and fails a sweep. `fftw_free` remains correct for
`fftw_malloc`'d **buffers**; the two rules apply to different pointers and the code must say
which is which.

🔴 **A static stride assertion is required before every `plan()` call.** `FFTW_MEASURE`
selects by *executing candidate plans on the caller's arrays*, so a wrong stride triple is an
**out-of-bounds write at plan time** — reproduced as `0xC0000005` inside `fftw_plan_…` by
handing `guru_split_dft_c2r` the r2c triple [M]. Gate class S runs after `plan()` returns and
therefore cannot be the backstop it was billed as. Assert statically that `n*is` and `n*os`
lie inside the declared extent of the buffer each pointer names.

---

## 3 · The four hazard decisions

### 3.1 Alignment vs the 64 B skew — **the feared hazard does not exist; a different one does**

**The brief's central fear is refuted, four ways [M]:**

1. **There is no arena and no 64 B skew in this bench.** Every plane is an independent
   `alloc_d` → `posix_memalign(64)` (`:98-107`). The "ONE arena +64 B skew" protocol lives in
   `benches/tangent_hand/*.c` and `benches/smallre_mkl_probe.c:32,82` — never here.
2. **FFTW's alignment class is `p % 16`** (`kernel/align.c:25`, `ALGN==16`; its AVX layer uses
   `_mm256_loadu_p`/`_mm256_storeu_p`, which is why it is 16 and not 32). Offsets
   0/16/32/64/128/4096/4160 → class 0; offset 8 → class 8. Every 64 B-aligned plane is class
   0 — *stricter* than `fftw_malloc`, which returned a `ptr%64==32` pointer.
3. **Decisive:** `fftw_sprint_plan()` is **byte-identical** between plans built on
   `fftw_malloc` buffers and on 64 B-skewed buffers at N=128/1024/4096/65536, verified
   independently twice.
4. Where it differs it **favours FFTW**: N=65536 ratio **0.868** — the skew dodges 4 KB
   aliasing that `fftw_malloc` planes walk into.

**Ruling:** keep `alloc_d` for the vfft and MKL arms; introduce no global arena (one would
perturb every banked MKL number for no benefit); ban `FFTW_UNALIGNED` (2.08–3.59× penalty
[M]; `reference.texi:518-527` *"no SIMD may be used"*); use **new-array execute everywhere**
(`fftw_execute_dft`, `_split_dft`, `_dft_r2c`, `_dft_c2r`, `_split_dft_r2c`, `_split_dft_c2r`)
because `fftw_execute(plan)` is unusable in nearly every mode — the timed buffers are not the
plan buffers (`measure_ab:341/:355`, `k1z_time_mkl:543`).

🔴 **The real hazard is the split plane delta, and it is not a correctness hazard — it is a
reproducibility hazard.** `dft/problem.c:37-38` hashes `p->ii - p->ri` and `p->io - p->ro`
into the wisdom key; `rdft/problem2.c:38-39` does the same for r2c/c2r [D]. For an
**interleaved** plan the delta is the constant 1 and the key is portable. For a **split** plan
it is whatever our independently-malloc'd planes happen to be. Measured [M]: same delta,
different base → HIT; delta different by 8 doubles (still 64 B-aligned, still class 0) →
**MISS**; interleaved with a wholly different buffer → HIT. Deltas observed across launches:
4104, 4112, 24128, 348024 — and 264/264/264/1032/1128 in five launches of a two-allocation
probe. **6/6 MISS with real `_aligned_malloc(…,64)` planes.**

**Ruling — a per-cell, FFTW-owned plane pair carved from ONE block at a fixed size-derived
offset.** `scratch/vfeas/v9_fix.c` uses `stride = roundup4K(N*K*8) + 64` — which also
*preserves* the house's 64 B skew — and gets **6/6 WISDOM_ONLY HIT with one identical
`plan_id` across isolated processes** [M]. This does not touch `alloc_d`, does not perturb any
banked MKL number, satisfies `REF_ASSERT_BOUND`, and cures the 9.4 % plan-drift term at the
same time. It costs one CSV column, `ref_planes=contiguous`, recording that the FFTW arm's
two planes are contiguous where ours are not — and one open experiment (§6.2), because
contiguous planes roll different 4 KB dice than independent ones.

```c
/* refbackend/ref_fftw.h — the FFTW arm's own planes, per cell */
static size_t ref_plane_stride(size_t bytes)          /* deterministic, size-derived */
{   return ((bytes + 4095) & ~(size_t)4095) + 64; }   /* 4 KB pitch + the house 64 B skew */

typedef struct { double *re, *im; void *blk; size_t stride; } ref_planes_t;

static ref_planes_t ref_planes_alloc(size_t n_doubles)
{   size_t s = ref_plane_stride(n_doubles * sizeof(double));
    void *blk = NULL; vfft_proto_posix_memalign(&blk, 64, 2 * s);
    ref_planes_t p = { (double*)blk, (double*)((char*)blk + s), blk, s };
    return p;                    /* ii - ri is now a pure function of (N,K) */
}
```

### 3.2 FFTW_MEASURE, planning order, and wisdom

**MEASURE is law.** The historical catastrophe that motivated the law is now explained: the
N=1000 guru `split_dft` "errors of 60+, sometimes 1e+299" was an **ESTIMATE artefact, not a
size artefact** — under MEASURE the same shape gates clean at **1.96e-11** [M]. And the
mechanism is pinned: ESTIMATE plans a *structurally different library* — at N=512 it picks
the 128-bit codelet family (`t3fv_16_avx2_128`/`n2fv_32_avx2_128`) where MEASURE picks
256-bit (`t2fv_64_avx2`/`n2fv_8_avx2`) [M]. **`FFTW_ESTIMATE` is unrepresentable in
`ref_flags_t`.** PATIENT/EXHAUSTIVE are diagnostic knobs only (three different plans and a
2.2× spread at one cell [D]), always with `fftw_set_timelimit`.

**Ordering rules, mechanically enforced:**

- **`alloc → plan → fill → gate → warmup → time`.** MEASURE clobbers **both** input and
  output at plan time (64/64 doubles; ESTIMATE 0/64) [M]. Several cells fill before planning
  today (`run_r2c_cell:2202-2209`). Enforced by **poison-on-plan**
  (`-DREF_POISON_ON_PLAN` makes even `ref_mkl` scribble the exemplar buffers at plan time),
  so the ordering bug is loud on the MKL-only build, in CI, before FFTW compiles.
- **Plans always before the cachebust barrier**, enforced by `REF_ASSERT_PREBARRIER` (a
  counter bumped by `cachebust()`, recorded at plan birth, asserted before the race). The file
  has two contradictory conventions today [M]: `bench_mkl_2d/2dr2c/2dc2r/r2c/c2r` plan
  **before**; `bench_mkl_oop`/`k1z_time_mkl`/`kzb_time_mkl`/`ilmt_time_mkl` plan **after**.
  For MKL that is ~10 µs. For FFTW it is up to **4.07 s of MEASURE planning after the
  cachebust** — fast, plausible, meaningless, with no other symptom.
- 🔴 **`pace(cool_ms)` *after* planning, and stamp `plan_ms`.** A multi-second cold MEASURE
  immediately before arm 1 is a **thermal preconditioner** paid only under `--ref=fftw|both`,
  which biases the *vfft* column between CSVs. This is a requirement the barrier assertion
  alone does not cover.
- **Separate plan objects per direction, always.** `guru_split_dft` has no `sign`; a reused
  plan with the wrong pointer order is a silent conjugation. `dir` is a `plan()` parameter for
  both backends, which makes `--2dc2r`'s surviving one-descriptor-both-directions bug
  (`:2081`/`:2090`) *structurally unwritable* — same class as the 2026-08-13 `--c2r`
  heap-OOB that voided every c2r ratio banked before it.

**Wisdom — two-phase, per-cell shards, never exported from a timed process:**

```
Phase P (once per host+build):  --ref=fftw --wisdom-prime [--mode …]
    builds every plan in the grid in ONE process →
    build_tuned/results/fftw_wisdom/<fftw_version>/master.wis
Phase R (per cell, isolated):   import master.wis; FFTW_MEASURE|FFTW_WISDOM_ONLY first
    HIT  → ~41 us, row stamped wis=hit,  plan_src=wisdom
    MISS → plain FFTW_MEASURE, row stamped wis=miss, plan_src=cold; the new wisdom
           goes to a PER-CELL SHARD .../shards/<mode>_N<N>_K<K>.wis
Phase M (runner, after the sweep): merge shards, re-run every wis=miss cell.
```

`wis=hit|miss` is a column; **a banked table with any `wis=miss` row is not final**;
`VFFT_FFTW_STRICT=1` makes a MISS fatal for banking runs. Wisdom is **not banked in-repo** —
its header hashes encode build+CPU config
(`(fftw-3.3.10 fftw_wisdom #xc6eb5a5a #x7ec1cde2 #x991a6133 #xefc8d223)` [M]), exactly as our
own wisdom is host-specific; it lives under `build_tuned/results/`, keyed by the
`fftw_version` string so a vcpkg upgrade cannot silently reuse stale plans, gitignored, with
hits and misses in the banner. 🔴 **This scheme only works at all because of §3.1** — without
the deterministic plane pair, Phase P and Phase R have different allocation histories, every
split row misses, and Phase M is a fixed point that never converges.

🔴 **Honest planning cost [M]** — measured, not estimated. The earlier "3–8 min for the whole
grid" figure is refuted by more than an order of magnitude:

| shape | cold `FFTW_MEASURE` |
|---|---:|
| 1D c2c, 14 sizes 128…1048576, oop+inplace = 28 plans | **19.06 s** total, worst cell 4.07 s at N=262144 |
| default-mode `guru_split_dft` in-place, all 67 K=4 cells | **46.32 s**, worst 8.06 s at N=823543 |
| N=30030 K=256 — **one plan** | **246.9 s** |
| N=60060 K=256 — **one plan** | **643.6 s (10.7 min)** |
| N=390625 K=32 / N=161051 K=32 / N=117649 K=32 | 17.8 / 9.8 / 9.4 s |
| every real-1D and 2D shape measured | ≤ 1.2 s each — **the cost lives entirely in large-`N·K` split c2c** |
| the same plans re-planned from imported wisdom | **0.003 s total**, all HIT, every `sprint_plan` string identical to the cold run |

`30030 = 2·3·5·7·11·13` and `60060 = 2²·3·5·7·11·13`: six distinct prime factors ⇒ a huge
search space in which every trial executes a 7.7 M / 15.4 M-point transform. 🔎 **Scope of the measurement, stated honestly:** 98 of the 190 default-mode cells were
actually planned (dedup total **17.8 min**); **92 remain unmeasured — 36 of them at K=256,
the most expensive class** — and the two ten-minute outliers are already inside the 17.8 min.
So *the default mode's mirror arm alone is ≈20–30 min of cold planning, ≈35–40 min per sweep
with the mandatory `home` arm* is an **[I] extrapolation, not an [M] measurement**, and it is
more likely low than high. `run_bench.py:114-117` `p.wait()`s with no timeout, so this
presents as a hang — the house law *"long create LOOKS LIKE A HANG — log on entry"* applies
to FFTW here exactly as it does to our own planner.

**Ruling:** (a) §3.1's deterministic planes so wisdom actually pins these plans and the cost
is paid **once**; and (b) a published, wisdom-primed `max_NK` ceiling above which the FFTW
column prints `fftw_ns=n/a reason=plan-cost` — an honest empty cell, in the same spirit as
the threads `n/a`. 🔴 **`fftw_set_timelimit` is explicitly rejected as the lever for MEASURE**:
it makes the chosen plan a function of how fast the machine was at plan time, on a box whose
first memory law is *THERMALLY NOISY*. That converts a 10-minute stall into a
thermally-correlated plan lottery — worse than the disease.

### 3.3 c2r input destruction

**Facts [M], four independent probes:** 1D c2r destroys its input by default — N≤32 → 0
doubles changed; **N≥64 → exactly N doubles clobbered per execute** (it *hides* at small N).
`FFTW_PRESERVE_INPUT` plans fine for rank 1 at every N tested and is expensive — but 🔎 **the
tax is NOT monotonic in N, and the earlier per-N figures were stitched together from two
different probe series**. Re-measured in one series, median of 15, core 2, HIGH
(`docs/research/fftw_bench/critic/c3_c2rtax.c`): N=1024 **+32.3 %** · N=4096 **+19.2 %** ·
N=16384 **+40.4 %** · N=65536 **+28.2 %** · N=262144 **+76.1 %**. The earlier claims
(+35.4 / +21.3 / **+59.5** / +72.5, "grows with N") over-tighten the 16384 cell by 19 points
and assert a trend the data does not show. **Quote the range 1.2–1.8×, per cell, never a trend.**
**Rank-2 c2r + `PRESERVE_INPUT` returns NULL** (`reference.texi:504-514`). **MKL's 1D OOP c2r
does NOT destroy its input** — 0 of 66…65,544 doubles changed after 1 and 10 executes,
N∈{64…16384} × K∈{1,4} [M] — and neither does our engine.

**Ruling — the mirror/home doctrine applied on the *semantic* axis instead of the memory axis:**

| Mode | Verdict column | Diagnostic columns | `fftw_c2r_input=` |
|---|---|---|---|
| `--c2r` (1D) | `MEASURE\|FFTW_PRESERVE_INPUT` — **semantics match MKL's and ours** | `MEASURE` alone with input refilled per rep, **plus a refill-only arm so the memcpy tax is visible and never subtracted** | `preserve` / `destroy+refill` |
| `--padr2c` | forward only — no destruction issue | — | — |
| `--zr2c` c2r | `MEASURE`, in-place, no refill | — | `inplace-destroy(both engines)` |
| `--2dc2r` | `MEASURE` destructive + **untimed pristine restore between samples, `reps_per_sample=1`, identical sample shape for every arm** | — | `destroy+restore(preserve-unavailable-rank2)` |
| `--c2rcalib` | no reference arm at all | — | — |

**Both columns or neither** — the same publication rule as mirror/home.

**Enforcement:** `ref_race` **refuses** to run an arm whose plan has `caps.destroys_input` and
no `refill`/`restore`, unless `--unsound-destroy` stamps the row `UNSOUND`. A destructive
timing loop cannot be created by omission — it has been created by omission **twice already in
this repo**: `bench_fft2d_r2c_vs_fftw.c:54,65-71` timed 11 `DESTROY_INPUT` rounds with no
refill; `bench_dct2_vs_fftw.c:145-147` refilled `src` **from `out_fftw`**, so FFTW transformed
DCT-of-DCT-of-DCT from round 2 *and* paid an extra NK write our arm never paid [M]. Every
destructive arm's correctness is a **separate, untimed execute** on a freshly written
spectrum; a destroyed timing loop can never contribute to a gate.

🔴 **Two corrections to what was nearly shipped as settled:**

1. **`--zr2c` is NOT "fair by construction", and the pilot is not exempt from
   `--c2r-repsens`.** The memcpy at `:2865` does sit outside `ZR2C_TIME` at `:2866`, so MKL
   reps on decaying junk — but `:2878 _zr2c_fold_bwd(bsrc, Zc, …)` is **inside** `ZR2C_TIME`
   and `bsrc` is `const double* = cref` (`:2810`), so **our** arm regenerates a pristine finite
   input every rep. With `reps_for(1024)=1951`, MKL's values reach max|v| 9.6e21…3.2e47 within
   20 reps. The conclusion survives on new evidence rather than on the old argument: both
   libraries are **data-oblivious** — MKL c2r inf/nrm 0.988–1.021, nan/nrm 0.958–1.004; FFTW
   c2c inf/nrm 0.975–1.003, nan/nrm 0.988–1.005 (core 2, HIGH, FTZ+DAZ, median of 5) [M]. But
   our arm still carries an extra (N+2)-double read stream (512 KB at N=65536) that neither
   reference pays, so `--c2r-repsens` (time the same cell at `reps ∈ {8,64,512}`; ns/rep must
   be flat) **extends to `--zr2c` and `--2dc2r`**, not just `--c2r`.
2. **FTZ/DAZ *is* enabled** — `:49` includes `env.h`, `:3603` calls `stride_env_init()` →
   `env.h:81 _mm_setcsr(old | 0x8040)` [M]. Two of the three proposals published the opposite
   as a 🔴 [M] finding. Consequence: a destructive loop must be justified by **±Inf being
   full-rate on x86**, which remains a *hypothesis until `--c2r-repsens` runs*; it can no
   longer borrow denormal flushing as its justification. Make the state explicit: set MXCSR in
   the bench itself and print `mxcsr=0x%04x` in the banner rather than inheriting it from a
   library header.

### 3.4 The missing threads library

**Triple-proven [M]:** a real build gives
`undefined reference to __imp_fftw_init_threads / __imp_fftw_plan_with_nthreads /
__imp_fftw_cleanup_threads`; `objdump -p fftw3.dll` → **123** `fftw_*` exports (the DLL's *entire* export table is
`fftw_*`; the "77" in the recon notes is unreproducible), **zero** thread symbols;
`fftw3.lib` → the same 123, none threading. 🔎 *re-verified by the completeness critic.* `fftw3.h:318-319` **declares** them
anyway (inside `FFTW_DEFINE_API`) ⇒ **an MT FFTW arm compiles and dies at LINK, not at
compile.** 🔴 **Add `--mkl` to that link and it succeeds and runs — via MKL's wrapper.** Any
threads probe must be FFTW-only or it produces a false positive. The runtime shim removes the
class entirely: the thread functions are **not in the bind list**.

**Scope of the block is wider than two modes:** `--mt` is also a modifier, so
`--oop --mt`, `--2d --mt`, `--2dr2c --mt`, `--2dc2r --mt`, `--r2c --mt` and the plain
`--1d --mt` all lose their FFTW arm — **6 blocked configurations** (5 modifier combinations
+ plain `--mt`), enumerated one-for-one by the six `*_mt.csv` names in the default ternary at
`:3560-3579`. 🔎 *the earlier "7" double-counted plain `--mt` as both the mode and a
combination.* 🔴 **The blast radius is wider than the CSV list**: `:3619` runs
`mkl_set_num_threads(mt ? g_mt : 1)` **globally**, so `--mt` bolted onto ANY mode
(`--zr2c --mt`, `--kzb --mt`, `--c2r --mt`, `--k1nat --mt`, …) threads the MKL arm while
writing the mode's **single-threaded CSV path** — a pre-existing footgun the FFTW arm must
not inherit. `--ref=fftw` must refuse `--mt` for every mode, not only the six named ones. And `--ilmt` is **over-blocked**: its ST arm is buildable
today (§2 row 13).

**Unblock condition — one command, deliberately not taken in v1:**
`vcpkg install "fftw3[core,avx,avx2,threads]:x64-windows" --recurse`, ~2–5 min (a previous
3-precision install measured 75 s wall). `ports/fftw3/portfile.cmake:26-27` maps feature
`threads` → `ENABLE_THREADS` **and `WITH_COMBINED_THREADS`**, folding the threads code
*inside* `fftw3.lib`/`fftw3.dll` ⇒ **no `build.py` link change** [D]. Back up
`bin/lib/fftw3.*` first; vcpkg removes the current port before reinstalling. A private CMake
build of the on-host source `C:/Users/Tugbars/fftw-3.3.10` into a side prefix is the
alternative and additionally recovers the missing SSE2 codelets (FFTW has a Win32 threads
path at `threads/threads.c:159`; pthreads not required).

**Why `n/a` for v1 anyway.** The resulting DLL is a **different binary**, so mixing it into
the table makes the FFTW column non-comparable *to itself* — forcing a full re-race of every
single-threaded cell, not just two new modes. And the standing MT law bites the moment a
threaded arm exists: **≥200 ms pacing is INVALID for threaded arms** (the worker team parks
against `KMP_BLOCKTIME` and you measure thread-wake — this manufactured a fake "MKL MT loses"
result once), and **a single-threaded control is not a control for a threaded arm**. Threaded
FFTW is therefore a *measurement design*, not a build change, and it gets its own campaign.

---

## 4 · The fairness doctrine

### 4.1 It is the house's own, promoted from a comment to a type

`bench_1d_vs_mkl.c:753-758` and `:855-860` already say it, for MKL, in `--kzb`:

1. **`mir`** — the reference in **our** layout: *"the routing-verdict number"*. The question
   the project asks is *"does routing work X through our engine beat sending it to library
   Y"*, and X is our layout.
2. **`home`** — the reference in **its own** native layout: *"a DIAGNOSTIC column for
   positioning (different memory contract, **never the verdict input**)"*.
3. **`loop`** — **our** K=1 engine called K times over the reference's transform-contiguous
   memory: *"the honest like-for-like against MKL's HOME arm: both engines see the same
   transform-contiguous memory"*. Where none exists the row emits
   `r_loop=n/a,reason=no-loop-arm` — never a bare `r_mir`.
4. **`r_mir` and `r_home`/`r_loop` publish together or not at all.** A mirror-only FFTW table
   is *exactly* the shape of the `v1_0_results.md:901-974` liability.
5. Every row carries the label set of §2.2. **A number without its layout label is not
   publishable.**
6. `fftw_version` is **asserted, not assumed**, at process start.
7. **State where fairness is impossible; do not paper over it.** Three places: `--2dc2r`
   (FFTW cannot preserve at rank 2, so it is measured on its fastest path while we are not —
   the 1D measurement puts that advantage at 1.2–1.7×); class-N modes (two passing roundtrips
   do not prove the same spectrum — the caveat already exists at `:249-251` and now applies to
   a third engine); `--mt`/`--ilmt`.
8. **Expect and publish losses.** The opponent is real: `fftw-3.3.10-avx-avx2-avx2_128`, MSVC
   14.42, genuine AVX2 codelets (`t2fv_32_avx2`, `n2fv_32_avx2`), DLL genus histogram — distinct
   codelet symbol names ending in each genus — `_avx` **376** / `_avx2` **372** /
   `_avx2_128` **376** / **`avx512` 0 / `sse2` 0** ⇒ **ISA-symmetric with the i9-14900KF**
   [M, recounted by the critic; the earlier `94 / 94 / _avx_128 94` line was both
   mis-counted and mis-named — this build has no `_avx_128` genus, the third family is
   `avx2_128`, exactly as `fftw_version` spells it].
   Historical corroboration: an AVX2-only FFTW **beat us in 6/7 cells by 1.12–2.0× on its home
   layout** [D]. This is why the K=1 IL family races early (P3) — *shipping the losses first
   is what makes the eventual wins credible*.

The mechanism: `ref_role_t` ∈ {MIRROR, HOME, LOOP, OURS, CONTROL}, with `ref_ratio()`
aborting on role HOME. That turns `--kzb:757`'s comment into a type error.

### 4.2 🔴 The doctrine is scoped, not universal — the correction that changes five rows

`ref_role_t` was designed for the FFTW arms and never retro-applied to the MKL arms, which
would have shipped `--ref=both` rows that mix roles silently. Read the code [M]:

- `run_r2c_cell:2211-2213` transposes our lane-major input to transform-contiguous
  **outside the timed region**, then sets `DFTI_INPUT_DISTANCE=N`,
  `DFTI_OUTPUT_DISTANCE=halfN+1` (`:2215-2223`). `bench_mkl_r2c:2170` times only the
  transform. ⇒ **MKL's arm here already IS a home arm, and it already IS the verdict**
  (`speedup = mns/vns`, `:2255`).
- `run_c2r_cell:2381-2383` does the same into `xtm`; the forward at `:2408` is explicitly
  untimed (*"the c2r input (not timed)"*); only `bench_mkl_c2r(hb, cce, mout, total)` is timed
  (`:2450`).
- The same holds for `--padr2c`, `--2dr2c`, `--2dc2r`, and `--zr2c` (`:2473` calls MKL's
  CCE+INPLACE descriptor *"its BEST arm"*).

**So five existing modes already chose home-is-verdict for the reference.** Handing FFTW a
lane-major mirror as *its* verdict in those rows would put
`ratio_vs_mkl = mkl_home/vfft_lanemajor` beside
`ratio_vs_fftw = fftw_lanemajor/vfft_lanemajor` in one row — at K∈{7,8,15,16,17}, where the
lane-major tax quick-looks at 3.1–9.3×. **That is the `v1_0_results.md` liability re-created
inside the canonical file, with the sign flipped.**

**Ruling: the doctrine is per-mode, and it follows the MKL arm.**

- Modes where the MKL arm is a **mirror** (default 1a, `--oop`, `--pad`, `--kzb`, the `--k1*`
  family, `--2d`): FFTW's verdict arm is the **mirror**, `home` is the mandatory diagnostic.
- Modes where the MKL arm is a **home** arm with an untimed adapter (`--r2c`, `--c2r`,
  `--padr2c`, `--2dr2c`, `--2dc2r`, `--zr2c`): **FFTW gets the same deal** — verdict = home
  layout with the same untimed adapter, `mir` is the diagnostic. Otherwise the two reference
  columns are not commensurable.
- **Every row publishes `mkl_role=` alongside `ref_role=`**, so a reader can see which regime
  the row is in without reading the source.
- `ref_ratio()`'s abort-on-HOME is therefore **scoped to mirror-regime modes**. Unscoped it
  would refuse to write `--kzb`'s own existing `rhom`/`rloop` columns (`:1046-1049`) — it
  would break the row nominated as the template.

### 4.3 The caveat block — verbatim, for any published FFTW table

🔴 This block asserts only what the harness actually delivers **at the phase that emits it**.
The first version of it claimed ">=15 rounds", "one control arm per cell" and medians while
the code shipped 5 trials, one control arm in one mode, and best-of-5 — the same failure class
as the headline being retracted, one level up. The `stat=`, `rounds=` and `control=` tokens
below are **printed from the values actually used**, never hardcoded.

```
FFTW ARM — READ BEFORE QUOTING ANY RATIO

1. BUILD.  fftw-3.3.10-avx-avx2-avx2_128 (vcpkg x64-windows, MSVC 14.42), features avx+avx2
   only: no AVX-512 (ISA-symmetric with the i9-14900KF) and NO SSE2 codelets.  The exact
   string is in every row as ref_id.  Numbers from a different FFTW build are NOT comparable
   to these, including a future threads-enabled build.

2. THREADS.  This build exports no threading symbols.  --mt and its five mode-combinations
   have no FFTW arm; those rows read  fftw_ns=n/a reason=no-threads-in-build.  --ilmt carries
   a SINGLE-THREADED FFTW column only (fftw_nthreads=1, fftw_mt_ns=n/a); it is never folded
   into the mode's best-MKL denominator.  No FFTW number in this table is threaded.

3. LAYOUT.  Two reference columns per K>1 cell, and the verdict role FOLLOWS THE MKL ARM
   (mkl_role= and ref_role= are in every row).  In mirror-regime modes fftw_mir_ns is the
   routing verdict and fftw_home_ns is a positioning DIAGNOSTIC, never a verdict input.  In
   home-regime modes (r2c, c2r, padr2c, 2dr2c, 2dc2r, zr2c) the bench hands BOTH references
   their native layout with the adapter outside the timed region, so fftw_home_ns is the
   verdict and fftw_mir_ns is the diagnostic.  Where a loop arm exists (our K=1 engine run K
   times on the reference's own memory) it is the honest like-for-like.  A mirror-only ratio
   at K>=4 is a LAYOUT measurement, not a kernel measurement, and must never be quoted alone.

4. PLANNING.  FFTW_MEASURE everywhere, plans built outside the timed region, before the
   cachebust barrier, followed by a pace, pinned by imported wisdom (wis=hit in every banked
   row; any wis=miss row means this table is NOT FINAL).  plan_ms is the planning cost, not
   measured work.  FFTW_ESTIMATE is never used: it plans a structurally different library.
   PATIENT is never used: it is non-deterministic.  plan_id identifies the exact plan chosen;
   a changed plan_id invalidates the comparison even at an unchanged N.  Cells above the
   published max_NK ceiling read fftw_ns=n/a reason=plan-cost.

5. c2r INPUT.  FFTW's default c2r DESTROYS its input; MKL's and ours do not.  In --c2r the
   verdict column uses FFTW_PRESERVE_INPUT (semantics matched, costing FFTW +21%..+72%,
   growing with N) and the destructive column is published beside it as FFTW's true floor.
   --2dc2r cannot use PRESERVE_INPUT at all (the plan returns NULL) and is labelled
   destroy+restore, with the restore untimed and applied identically to every arm.

6. ORDERING.  gate_class is in every row.  X = cross-engine elementwise per direction.
   R = elementwise vs a naive DFT.  N = each engine's own roundtrip.  A class-N row proves
   each engine is self-consistent; it does NOT prove the two produced the same spectrum.
   FFTW is never gated by roundtrip alone: it is natural in every API, so class-N rows still
   carry a real elementwise FFTW-vs-MKL number (fftw_gate=xmkl).

7. PROTOCOL, as actually run for this table.  Core 2, HIGH priority, FTZ+DAZ (mxcsr printed
   in the banner), QPC, rounds=<printed>, stat=<printed>, control=<printed>, arm order
   permuted per the scheduler named in sched=, >=200 ms pace between arms, cachebust between
   arms, spread reported.  A delta inside the control spread is NOT a result.
```

### 4.4 Scheduling and the estimator

- `--ref=mkl` — **literally today's code path**: 2 arms, 1 flip bit, existing CSV stem. The
  banked corpus stays valid and resumable; this is what `run_bench.py` invokes.
- `--ref=fftw` — the same 2-arm shape with FFTW in MKL's slot, into `*_fftw.csv`.
- `--ref=both` — the three-way race, into `*.ref3.csv`.
- The paired cross-process control survives as a **check** on the trusted number, not its
  source: `vfft_drift = |v_mkl − v_fftw| / min(v_mkl, v_fftw)`; if `> 0.03` the three-way row
  is flagged, because a cell whose *own engine* is not reproducible across processes is not a
  cell whose reference ratio should be quoted.

🔴 **Arm-order neutralisation must be the full symmetric group on the arms the row actually
has — and that is rarely 3.** 🔎 *Critic correction:* §4.4 was written for a 3-arm row, but
`--kzb` already times **four** arms today (`vns`, `mns` mirror, `hns` home, `lns` loop,
`:1013-1045`) and `--ilmt` times **five** (`omt`, `ost`, `mmt`, `mst`, `octl`, `:1276-1312`);
adding FFTW `mir`+`home` makes them **six**. `S₃`/`rot%6` is therefore a *special case*, not the
rule. Specify the scheduler as `rot % n!` for n ≤ 4 and a seeded random permutation with a
recorded seed (`sched=rand:<seed>`) for n ≥ 5, where 120–720 cells per permutation is not
reachable. The `sched=` column already exists to carry this. The original argument stands
unchanged for why a *cycle* is not enough: `flip` is a **reversal**,
which is the *complete* permutation group for 2 arms. A 3-cycle over 3 arms covers 3 of 6
permutations and **preserves every adjacency pair** — i.e. it is invariant on exactly the
"arm inherits its predecessor's cache/thermal state" bias that `cachebust` and `pace` exist to
fight. Use `rot % 6` over the full symmetric group with a **rotating control position**, or
demote `--ref=both` to quick-look. (The precedent that was cited for rotation is weaker than
described: `--ilmt:1276-1312` is a hand-written two-branch **flip**, and `octl` is the 5th arm
in *both* branches, so `ctl_spread` conflates repeatability with position bias and cannot
detect the effect it was cited to guard.)

🔴 **One estimator per row, and emit both statistics from the same pass.** Dividing a
median-of-5 FFTW arm by a best-of-5 vfft arm inflates `ratio_vs_fftw` in our favour in every
row of every mode — the exact shape of the headline being retracted. But the reason for
deferring medians entirely does **not** hold: five samples contain both statistics, so the
core emits `*_min` **and** `*_med` from one pass. Banked mins stay bit-comparable; the house
MEDIANS law lands in P1; P7 collapses from a re-race to a reporting change. (`REF_STAT_MIN5`
is also not ratio-safe across libraries — min-of-n favours whichever engine has the longer
downward tail, and three engines have three tail shapes.)

---

## 5 · The phased plan

### 5.0 Two build-path facts that gate P1

🔴 **`ref_time.c` / `ref_mkl.c` / `ref_fftw.c` as translation units do not compile today**
[M]: `build.py:385-389` hardcodes `extra_srcs` to exactly one file under `--vfft` and has no
`--extra-src` flag; `run_bench.py:102-103` invokes
`build.py --src bench_1d_vs_mkl.c --mkl --jit --vfft --compile`. The bench's 29 includes are
**all headers, zero `.c`**. ⇒ **ship the backend as headers** (`ref.h`, `ref_time.h`,
`ref_mkl.h`, `ref_fftw.h`), which is the repo's own convention, and no build-script change is
needed.

🔎 🔴 **The interface itself is still missing from this document.** `ref_role_t`, `ref_flags_t`,
`ref_shape_t`, `ref_caps_t`, `ref_race()`, `ref_ratio()`, `REF_ASSERT_BOUND`,
`REF_ASSERT_PREBARRIER`, `REF_STAT_MIN5`, `REF_TIME_INLINE` and `csv_for()` are named
**40+ times between them and defined zero times** — the document contains exactly one code
block, `ref_planes_alloc` (§3.1). Everything the P1a implementer would type is [I]. **Write
`ref.h` — the enums, the `caps` bitfield, and the exact `plan(dir, shape, planes, flags)` /
`execute(plan, in, out)` / `ref_race(arms[], n, sched)` signatures — as the first commit of
P1a and paste it back here**, or the next reader re-derives it and the vocabulary drifts the
way the campaign's own inventory numbers did.

🔴 **The trusted isolated mode exists for exactly ONE mode** [M]: `run_bench.py` is 217 lines
with no mode flag at all (`--K`, `--core`, `--cool-ms`, `--csv`, `--wisdom`, `--max-nk`,
`--no-primes`, `--fresh`, `--skip-build`, `--dry-run`) and always drives the default 1D c2c
mode. So *"isolated one-cell-per-process is the TRUSTED mode"* is true only there; every other
mode — including the `--zr2c` pilot — runs today as a single in-process sweep, which the
protocol calls **quick-look only**. **P2 must therefore also extend `run_bench.py`** (a mode
flag, `--ref` passthrough, per-ref CSV) or its output is not publishable. Silver lining, and
it is real: because `run_bench.py` never passes `--fftw` and never passes `--ref`, the
existing MKL sweep is structurally untouched by everything in this campaign.

### 5.1 Phases

| Phase | Content | Ships alone | Produces |
|---|---|:--:|---|
| **P0 — bind, time nothing** | `ref_fftw.h` runtime shim (absolute DLL path + `VFFT_FFTW_DLL`); `--ref` parsing; `csv_for(ref)` for the four hardcoded CSV paths; `--c2rcalib` rejects `--ref=fftw`; banner with `fftw_version` + `mxcsr` + wisdom status; the `oneMKL`/`Intel` assert. Optional hygiene: `build.py:201-202` `find_fftw()` → `<vcpkg>/include` (today it returns **Intel MKL's FFTW-wrapper dir** `<vcpkg>/include/fftw`) | ✅ | Proof that genuine FFTW binds and MKL cannot hijack it — the only way to know the campaign is measuring FFTW at all |
| **P0.5 — retract** | Re-caption or retract `docs/performance/v1_0_results.md:901-974` ("beats FFTW3 202/207, median **3.21×**") and fix the dead link at `:910` | ✅ | Removes the standing liability *before* honest numbers land and get read as a regression against a fictional baseline |
| **P0.75 — deterministic planes + wisdom smoke test** | `ref_planes_alloc` (§3.1) exercised on **one default-mode split cell**, three isolated launches: compare `wis`, `plan_id`, `delta`. 🟢 The rank-1 split c2c/r2c/c2r and rank-2 split c2c legs are **already green** (`critic/c1_p075.c`, 3/3 isolated HIT, identical `plan_id`). Remaining: prime, **reboot**, re-run | ✅ | 🔴 **The campaign's real gate.** If the delta is not reproducible for r2c/c2r/rank-2 or across a reboot, P5/P6 lose wisdom and the `max_NK` ceiling has to be published instead |
| **P1a — the core, behind a flag, MKL only** | `flip`→`rot` rename as a separate zero-behaviour commit; `ref.h` + `ref_time.h` + `ref_mkl.h`; convert `--zr2c`; emit `*_min` **and** `*_med`; `REF_ASSERT_PREBARRIER`; poison-on-plan; `plan_id`; `plan_ms`; the static stride assert; `dir` as a plan parameter | ✅ — **worth shipping even if FFTW never lands** | One timing core instead of 21 copies; an N-arm scheduler; a control arm available to every mode; `--2dc2r`'s direction-reuse bug rendered unwritable in converted modes |
| **P1b — calling-convention measurement** | `ref_race` (indirect call) vs `REF_TIME_INLINE` vs the legacy inline idiom, N∈{128,512} × K∈{4,32}, full protocol | ✅ | Sets the default calling convention. **Merged behind a flag, then measured** — measuring a throwaway core would measure the wrong codegen |
| **P1c — the merge gate** | Number preservation: the **5 `--zr2c` cells × ≥4 repeated launches**, pinned `VFFT_REPS`, sign test on the paired deltas; plus three banked default-mode cells re-run under `--ref=mkl` and compared byte-for-byte | blocking | 🔴 The original ">=20 cells across >=6 modes" gate is **arithmetically unsatisfiable** beside "land P1 on `--zr2c` only" — `--zr2c` is 5 cells in 1 mode (`:4061`). The ≥6-mode form defers to whichever phase first converts ≥6 modes |
| **P2 — FFTW pilot** | `ref_fftw.h` execute paths; `--ref=fftw` and `--ref=both` on `--zr2c`; **extend `run_bench.py` with a mode flag and `--ref` passthrough**; wisdom Phase P/R/M for the pilot | ✅ | **First publishable FFTW cells.** Exit criterion additionally includes P0.75's split smoke test — the pilot is interleaved-only and by itself proves nothing about the split modes that carry most of the grid |
| **P3 — K=1 IL family** | `--k1noop`, `--k1nat`, `--k1zip`, `--k1dir` (all in-place except `--k1noop`; all class X except `--k1zip`) | ✅ each | FFTW on its home turf and ours — our hardest and fairest test, raced early on purpose |
| **P4 — batched IL + real 1D** | `--kzb` (the mirror/home/loop showcase), then `--r2c`, `--c2r` (with `--c2r-repsens` first), `--padr2c` | ✅ each | The template validated before the harder layouts |
| **P5 — split lane-major c2c** | default 1a + 1b, `--oop`, `--pad`. Requires P0.75 green, the protocol-grade mirror/home re-measurement of §6.2, and the published `max_NK` ceiling | ✅ each | The biggest fairness gap and the whole planning-cost problem |
| **P6 — 2D** | `--2d`, `--2dr2c`, then `--2dc2r` | ✅ each | 🔴 `--2dc2r` held until its one-descriptor-both-directions MKL bug (`:2081`/`:2090`) is audited; its `reps_per_sample=1` needs a new CSV stem |
| **P7 — estimator switchover** | Promote `*_med` to the headline for **all** arms of **all** modes at once, its own commit, new CSV stem | ✅ | Closes the 19-of-21 best-of-5 violation without ever mixing estimators inside one row |
| **Deferred / blocked** | threaded FFTW for the `--mt`/`--ilmt` MT columns (§3.4); a global arena behind `BENCH_ARENA=1` | separate campaigns | — |

**Why `--zr2c` is the pilot** — unanimous across every proposal and judgment, and the decisive
reason is protocol, not convenience: it is the **only mode already obeying the house MEDIANS
law** (`ZR2C_TIME:2493` + `_zr2c_med5:2483`), so the pilot needs **zero new timing idiom**.
Supporting: all three engines share one layout (interleaved CCE in-place — nobody needs an
adapter), the N+2 padded plane is already allocated (`:2757`), per-direction gates already
exist, and its wisdom key is portable — 5/5 isolated launches HIT with one `plan_id` [M].
🔴 **That last property is also its weakness as a de-risking exercise**: it is exactly the
property the split modes lack, which is why P0.75 exists and why it runs *before* the core
refactor merges rather than after.

---

## 6 · Open questions and honest costs

### 6.1 What the adversarial passes refuted (all folded in above; listed so nothing is quietly dropped)

| # | Refuted claim | Where it now lives |
|---|---|---|
| 1 | *"Wisdom pins the plans"* — false for every split mode; the key hashes `ii−ri`/`io−ro` | §3.1 — cured by the per-cell contiguous plane pair, 6/6 HIT [M] |
| 2 | *"`plan_id` mitigates plan drift"* — it diagnoses; drift moves the number **9.4 %** across launches | §2.2, §3.1 — same cure |
| 3 | *"Phase P ≈ 3–8 min"* — one cell measured at **10.7 min**; default mode ≈35–40 min per sweep for both arms | §3.2 — published `max_NK` ceiling; `fftw_set_timelimit` rejected |
| 4 | *"Release the `sprint_plan` string with `fftw_free`"* — **backwards**; causes `STATUS_HEAP_CORRUPTION` at loop scale | §2.2 — use libc `free` |
| 5 | *"Rank-2 split in-place plans NULL"* — false; it plans (0.303 s at 512²) | §2 row 3 — keep OOP for a different reason |
| 6 | *"Gate class S is the backstop for a bad stride triple"* — MEASURE writes out of bounds **at plan time** | §2.2 — static stride assertion |
| 7 | *"`--zr2c` is fair by construction, so the pilot is exempt from repsens"* — our arm refills inside the timed region | §3.3 — repsens extends to `--zr2c` and `--2dc2r` |
| 8 | *"FTZ/DAZ does not exist in this bench"* — it does (`:49` → `:3603` → `env.h:81`) | §3.3 — destructive loops need ±Inf justification, not denormal flushing |
| 9 | The caveat block asserted ≥15 rounds / control arms / medians the harness does not deliver | §4.3 — tokens printed from actuals |
| 10 | A 3-cycle rotation over 3 arms is a **weakening** of `flip`, not an upgrade | §4.4 — full S₃ or quick-look |
| 11 | Mirror-is-always-verdict — **five existing modes already chose home-is-verdict** | §4.2 — the doctrine follows the MKL arm, per mode |
| 12 | `--k1nat` is OOP (it is in-place); `--k1dir` is scrambled/class N (it is natural/class X) | §2 rows 14, 15 |
| 13 | Row 1 is one configuration (it is three passes needing two configurations) | §2 rows 1a, 1b |
| 14 | Row 9 covers `--zr2c` (it covers one of two cell runners) | §2 rows 9a, 9b |
| 15 | `--ilmt` is blocked (its ST arm is buildable today) | §2 row 13 |
| 16 | Two blocked modes (there are **6 blocked banked configurations**; "7" double-counted plain `--mt`) | §3.4 |
| 17 | `ref_*.c` translation units (the build path takes headers only) | §5.0 |
| 18 | *"Isolated one-cell-per-process is the trusted mode"* (true for one mode only) | §5.0 |
| 19 | The P1 merge gate (≥20 cells / ≥6 modes vs `--zr2c`-only) was unsatisfiable | §5.1 P1c |
| 20 | `REF_STAT_MIN5` with medians deferred to P7 — five samples contain both statistics | §4.4 |

### 6.2 Open questions, each with the experiment that settles it

| Open | Experiment |
|---|---|
| ~~and for `guru_split_dft_r2c`/`_c2r` and rank 2~~ — 🟢 **CLOSED [M] by the completeness critic**, `critic/c1_p075.c`: with `ref_plane_stride`, three isolated `WISDOM_ONLY` launches HIT with a byte-identical `plan_id` for rank-1 split c2c, **rank-1 split r2c**, **rank-1 split c2r** and **rank-2 split c2c** (deltas 1032 / 1032 / 1032 / 65544 at N=256 K=4 and 256²). Only the **reboot** leg remains open | **P0.75 is now one experiment, not four:** prime `critic/c1.wis`, reboot, re-run `c1_p075.exe 256 4`; the four `plan_id`s above are the expected values. 🔴 Also still open: whether the deterministic *stride* is stable when the OS returns a differently-based block — it is by construction (`ii-ri` is a pure function of `N·K`), but that is [I] until the reboot run |
| Does a **contiguous** FFTW plane pair change FFTW's *time*? It rolls different 4 KB dice than independent planes — worth up to 13 % at N=65536 [M] | Protocol-grade `mir` at N∈{1024,4096,65536} × K∈{4,32}, contiguous-pair vs independent planes, ≥15 rounds, control arm |
| **Is the FFTW column a layout measurement or a kernel measurement at K≥4?** The 3.1–9.3× tax and the 0.99–1.00 K=1 result are **unpaced best-of-5 quick-looks** and may not be quoted as results | Protocol-grade `mir` vs `home` vs `loop` at N=1024, K∈{1,4,32}, before P5. If the tax reproduces, the ruling is **not** to hide it but to make the **`loop` arm the headline** for those modes and demote `r_mir` to a routing number never quoted alone |
| Full-grid cold planning cost | Run the **92** remaining default cells (36 at K=256, 56 at K=32 — recomputed from `cells.txt` vs the measured set; the earlier "~157" reconciles with nothing) plus every other armed mode. The default mode alone is 35–40 min for both arms; budget **hours** for the grid |
| Does the per-rep indirect call inflate small-N `vfft_ns`? Predicted 3–6 % at 30–65 ns cells; the 128-pt Bailey champion is 65 ns | **P1b**, sign test over ≥20 cells. If converted cells are slower more often than chance, `REF_TIME_INLINE` becomes the default; if the effect exceeds ~5 % even inline, the core becomes a macro and the vtable is used only for plan construction, never for the timed body |
| Is `--ref=mkl` byte-identical after the `alloc → plan → fill` reorder? | **P1c.** If not, the *"banked corpus stays valid"* claim collapses and P1 is re-scoped to leave those modes alone. Cannot be answered by reading |
| Is our own c2r input-preserving? FFTW is taxed 1.2–1.7× to match a property never probed on our side (`time_c2r:2440` reuses `in_a`/`in_b` with no refill) | A 20-line probe, before P4 fixes the verdict column |
| Does the missing SSE2 codelet family handicap any cell? | Directly readable in `sprint_plan` — the codelet names are in the string. Nobody has looked |
| The K=1024 non-monotonicity in the 2026-06-24 prior art (our AVX2 at 0.38 → 0.96 → 0.72 for K=64/1024/8192, ~64 KB L2-resident) | Unexplained by the DRAM/TLB story; its own investigation |
| Would a self-built threads-enabled FFTW reproduce the vcpkg single-thread numbers? | It must, or the two builds can never share a table. Blocking for the deferred MT campaign |
| `--2dr2c`/`--2dc2r` `fftw_iodim` triples | Write them out and plan them at all four cells before P6; rank-2 split real is where NULL plans hide |
| `--2dc2r`'s descriptor-reuse bug may already have voided banked 2D c2r ratios | P1 removes the bug; it does not retroactively validate the data. Audit before P6 |

### 6.3 Evidence quality — what is measured and what is design-only

**Compiled and run on this host [M]:** the MKL hijack in both link orders · the runtime shim
including `__typeof__`-on-`dllimport`, `GetProcAddress` on the `fftw_version` data export, and
the zero-fftw-imports PE table with both libs on the line · the threads link failure and the
`--mkl` false positive · alignment class granularity and `sprint_plan` byte-identity across
skews · the wisdom split-delta miss/hit matrix and the deterministic-delta fix · plan drift
(5 plans / 9.4 % over 8 launches) · planning cost at 100+ cells including the 643.6 s outlier ·
`fftw_free` heap corruption at 200 cycles · the stride-triple plan-time OOB · c2r destruction
counts and the `PRESERVE_INPUT` tax at four sizes · MKL c2r input preservation · rank-2
`PRESERVE_INPUT` → NULL · rank-2 split in-place planning · in-place `guru_split_dft` forward
and backward-by-pointer-swap at 7.105e-15 · pilot in-place r2c/c2r vs MKL CCE at
1.1e-16…3.2e-16 · FFTW r2c == MKL CCE at 7.2e-15 · data-obliviousness of both libraries under
Inf/NaN · N=1000 split under MEASURE at 1.96e-11 · the DLL export census and genus histogram ·
every flag, placement, order and gate-class correction in §2 (read from the working tree at
4,494 lines).

**Read from source or the manual [D]:** the wisdom-key hash sites · split strides in doubles
with no `sign` argument · `reference.texi` on `PRESERVE_INPUT` and rank-2 c2r · the vcpkg
portfile's `threads` → `WITH_COMBINED_THREADS` mapping · PATIENT non-determinism.

**Added by the completeness critic, compiled and run [M]** (`docs/research/fftw_bench/critic/`):
the deterministic-plane wisdom fix extended to `guru_split_dft_r2c`, `guru_split_dft_c2r` and
rank-2 `guru_split_dft` (3/3 isolated `WISDOM_ONLY` HIT, identical `plan_id`) · the rank-2 split
r2c triple checked elementwise against `plan_dft_r2c_2d` (1.137e-13) and the split c2r roundtrip
(3.183e-12) · rank-2 **split** c2r + `PRESERVE_INPUT` → NULL · the `PRESERVE_INPUT` tax
re-measured as one median-of-15 series at five sizes · the `fftw_free`-vs-`free` corruption
reproduced, and both binaries shown to share the UCRT heap · the 123-symbol DLL export table ·
the codelet genus recount · the 190/67/51/146/10 cell counts recomputed from
`spike_wisdom.txt` and `cells.txt` · the 98-of-190 planning-cost coverage.

**Design-only, never compiled [I]:** `ref.h`'s vtable shape and every `REF_ASSERT_*` ·
`csv_for(ref)` · the two-phase wisdom runner and shard merge · the `--ref=both` S₃ scheduler ·
`--c2r-repsens` · the `max_NK` ceiling · the `mkl_role=` retro-labelling · every phase estimate.

🔴 **No competitive FFTW-vs-vfft number exists yet at protocol quality, anywhere in this repo.**
The 3.21× in `v1_0_results.md` is void, the layout-tax numbers are quick-looks, and nothing in
this document may be quoted as a result.

---

## 7 · Rejected alternatives

**A — a third arm bolted into each mode, no shared core.** The most conservative option: leave
the 21 hand-copied timing idioms alone and add an FFTW arm beside the MKL arm in each one. It
contributed the two best mechanisms in the campaign — the **runtime shim** and the
**never-export-wisdom-from-a-timed-process** rule, both adopted — but its structure loses on
marginal cost and on statistics. Every new mode costs two edits instead of one, the 21 idioms
stay 21, and the assertions that make the whole design safe (`REF_ASSERT_PREBARRIER`,
poison-on-plan, `ref_role_t`, per-direction plan objects) have no home to live in, so each one
would have to be re-implemented per site or dropped. Its sinker was a proposal to time the FFTW
arms with median-of-5 beside best-of-5 vfft/MKL arms: `ratio_vs_fftw` would then divide one
estimator by another and be **systematically inflated in our favour in every row of every mode,
indefinitely** — precisely the failure shape of the headline this campaign exists to retract.
It also lacked a ref-vs-ref stride self-check, and *"the single most likely way to publish a
fast, wrong FFTW number"* is a wrong `guru` stride triple.

**C — a sibling `bench_1d_vs_fftw.c` sharing a `bench_core.h`.** The most literal reading of the
request, and the source of the campaign's best *facts*: the FTZ/DAZ correction (the one fact
both rivals got backwards), `--c2r-repsens`, the two-phase wisdom design, the
number-preservation merge gate, and gate class S — all adopted. It loses on **history, not
taste**: this exact file existed, said *"Companion to bench_1d_vs_mkl.c"*, and was deleted at
`6f9681af` while the MKL bench grew to 4,494 lines — and its output is still published behind a
dead link as a 3.21× headline that is mostly an adapter tax. A sibling drifts from the canonical
harness one commit at a time, and every protocol upgrade must then be made twice. C's own
fallback conceded the point — *"the core is the load-bearing idea, the file boundary is what I'd
trade away"* — which lands its `bench_core.h` almost exactly on the adopted `ref_time.h`. Also
rejected from C: shipping v1 with no MKL link and FFTW as the sole oracle, which would cost six
class-N modes their `fftw_gate=xmkl` cross-engine number.

**Rejected mechanisms, from all three.** `FFTW_UNALIGNED` (2.08–3.59× penalty; *"no SIMD may be
used"*). `FFTW_PATIENT`/`EXHAUSTIVE` as a default (three plans, 2.2× spread at one cell).
`FFTW_ESTIMATE` anywhere, gates included. The **r2r halfcomplex** family (`FFTW_R2HC`/`HC2R`) as
a "fix" for `--r2c` — a third real layout no mode races, whose use would silently change what is
measured; `ref_shape_t` deliberately cannot express it, and `fftw_plan_r2r_1d` **is** exported,
so accidental use is physically possible. A global 64 B-skewed arena (it would perturb every
banked MKL number; the FFTW-owned per-cell plane pair of §3.1 gets the wisdom benefit without
touching `alloc_d`). Exporting wisdom from `fini()` (≈361 isolated processes racing one file).
A cross-CSV join as the *primary* three-way mechanism (it survives only as the cross-process
drift control). And `fftw_set_timelimit` as the answer to planning cost — it trades a
reproducible stall for a thermally-correlated plan lottery on a machine whose first law is that
it is thermally noisy.
