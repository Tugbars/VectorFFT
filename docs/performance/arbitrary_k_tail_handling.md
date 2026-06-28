# Arbitrary-K Tail Handling — how a batch size that isn't a multiple of the SIMD width stays correct *and* fast

> The library vectorizes **across the batch axis**: a codelet processes `VW`
> independent transforms per vector register (`VW = 4` for AVX2, `8` for AVX-512)
> in the split-complex lane-batched layout `data[e*K + lane]`. For years the
> generated butterfly loop was `for (k = 0; k < me; k += VW)` with **no remainder
> handling**, so any batch size `K` not a multiple of `VW` (odd `K=7`, tiny `K=1`)
> silently corrupted the leftover lanes.
>
> This doc describes the fix that shipped on branch `dev/arbitraryTail`
> (commit `06e40ed0`): a **rem-aware hybrid tail**, *generated through the DAG
> compiler* (no hand-edited codelets), that covers the `1..VW-1` leftover lanes —
> bit-exactly, and while holding **1.5–2.1× over MKL**.
>
> Companion to the design notes in
> [`docs/roadmap/arbitrary_k_vectorization.md`](../roadmap/arbitrary_k_vectorization.md)
> and the de-risking record in
> [`docs/roadmap/arbitrary_k_scalartail_experiment.md`](../roadmap/arbitrary_k_scalartail_experiment.md).

---

## 1. The problem in one picture

The batch index `k` runs `0 .. me-1` (`me` = the per-group lane count, = the batch
`K` at the leaf). The old codelet stepped `k += VW` and stopped at `k < me`, so for
`me = 7, VW = 4` it ran `k = 0` and `k = 4` — and the second iteration's vector load
`_mm256_loadu_pd(&rio_re[... + 4])` reads lanes `4,5,6,7`, writing **lane 7 out of
bounds** (or into the next group), while lanes computed past `me` are garbage. The
dispatchers "solved" this by refusing: `K % VW != 0 → return NULL`.

We want the opposite: accept every `K`, and pay only for the lanes that exist.

## 2. The solution — a rem-aware hybrid tail

The generated codelet now splits into a **bulk loop** over whole vectors and a
**tail** over the `rem = me - floor(me/VW)*VW` leftover lanes:

```c
size_t k = 0;
for (; k + VW <= me; k += VW) {
    ... full-width body: loadu / arithmetic / storeu ...   // UNCHANGED hot path
}
if (k < me) {
    const size_t rem = me - k;
    if (rem == 1) {
        ... scalar single lane (plain double, __builtin_fma) ...
    } else {                 // rem in 2..VW-1
        ... ONE masked vector pass (maskload / maskstore) ...
    }
}
```

Two facts make this the right shape:

* **`rem == 1` → scalar.** A single leftover lane is one transform. Computing it
  1-wide is the cheapest option and is on-par with our peak margin at the extreme.
  (Note: this "scalar" is SSE-1-wide — `vmovsd`/`vaddsd`/`__builtin_fma`, **zero
  x87** — the bottom rung of a SIMD-width cascade, same instruction class MKL
  floors at. See [`batched_smallN_vs_mkl_fftw.md`](batched_smallN_vs_mkl_fftw.md).)
* **`rem >= 2` → one masked pass.** Scalar cost grows linearly with `rem` (each
  lane a separate ~4×-slower transform), so by `rem = 2` a *single* masked vector
  pass — fixed cost regardless of `rem` — wins and stays **flat**. This is the
  catch from the experiment: a pure scalar tail erodes the MKL margin as `rem`
  grows (1.77→1.61→1.29 across rem 1/2/3); the masked pass holds it flat.

On an **aligned** `K` (`me % VW == 0`) the bulk loop iterates exactly as the old
`k < me` loop did and the `if (k < me)` is a single never-taken branch — so the
fast path is untouched (verified byte-identical under the kill-switch, §7).

## 3. In-place safety — why *masked* store, not a full overlapping store

The transform is in-place: input and output share `rio_re`/`rio_im`. A tempting
"cheap" tail is to back up by `VW - rem` and do one *full* vector store covering the
last `VW` lanes. **That corrupts the overlap.** The `VW - rem` lanes it re-covers
were already transformed by the bulk loop; a full store re-runs the butterfly on
already-transformed data and writes garbage back.

The masked pass avoids it: it operates on lanes `[k, k+rem) = [me-rem, me)` only.
`maskload` (avx2 `_mm256_maskload_pd` / avx512 `_mm512_maskz_loadu_pd`) reads only
the active lanes — **masked-off lanes never touch memory**, so there is no
out-of-bounds fault even when `k + VW > me` — and `maskstore` writes only those
lanes. The region is disjoint from everything the bulk already wrote. In-place safe.
(`_mm256_maskload_pd` compiles to `VMASKMOVPD`, which has **no alignment
requirement** — the unaligned `&rio_re[j*ios+k]` addresses are fine.)

## 4. The three codelet families — one masking rule covers all of them

The library has three twiddle families per stage, and they touch twiddles
differently — but the tail handles all three with **zero per-family branching**,
because of where the masking hook sits:

| family | twiddle access in the body | masked in the tail? |
|---|---|---|
| **t1s** (broadcast) | `_mm256_set1_pd(tw_re[2])` — a scalar broadcast, **not** indexed by `k` | **rio only** (twiddle is lane-independent) |
| **t1 / FLAT** (per-lane) | `_mm256_loadu_pd(&tw_re[j*me + k])` — indexed by `k` | **rio + twiddles** |
| **log3** | `_mm256_loadu_pd(&tw_re[j*me + k])` (same as FLAT) + hoisted `set1_pd(0.95…)` trig **constants** | rio + twiddles; constants stay unmasked |

The rule is simply **"mask whatever is indexed by the loop variable `k`."** Broadcast
twiddles and hoisted constants go through `set1_pd`, which *never* calls the
load helper, so they are unmasked automatically; per-lane rio and twiddle reads go
through the load helper, so threading the mask there masks exactly them. The
split-complex layout is what makes this clean: re/im are decoupled from the lane
axis, so the complex multiply is shuffle-free and the only `k`-indexed memory ops
are plain strided loads/stores.

> Note: the design notes once worried `log3` used a "VW-blocked `(R-1)*(me/VW)`
> twiddle table" that wouldn't mask cleanly. The generated code shows otherwise —
> `log3` addresses twiddles as `tw_re[j*me + k]`, identical to FLAT, so it masks
> like FLAT. The `set1_pd(0.95…)` lines are hoisted *trig constants*, not twiddles.

## 5. How it's generated — one schedule, rendered three ways

This is **not** hand-written per codelet. The DAG compiler emits all three renderings
of the *same scheduled DAG* off a single schedule, so the bulk / masked / scalar
bodies can never drift apart. Three pieces, all in the generator:

**`generator/lib/isa.ml`** — a load/store *mode*:
```ocaml
type ls_mode = LS_vector | LS_masked of string   (* the string = the C mask var, "_m" *)
let loadu_pd  ?(mode = LS_vector) isa addr      = ...   (* maskload when LS_masked *)
let storeu_pd ?(mode = LS_vector) isa addr value = ...  (* maskstore when LS_masked *)
```
`mode` defaults to `LS_vector`, so every existing positional caller is byte-identical.
The width-1 `scalar` ISA (already present) ignores `mode` — a lone lane is always
active — and renders plain-`double` ops, which *is* the scalar tail with no extra code.

**`generator/lib/emit_c.ml`** — a module ref `current_ls_mode` is consulted by
`render_load` (Input + per-lane Twiddle) and `emit_store`. The whole codelet body is
factored into one closure:
```ocaml
let emit_body (isa : Isa.t) () = (* render_output_addr + emit_store + the scheduler/spill match *)
```
`isa` shadows the outer ISA, so `render_node_def ~isa` and `emit_store` pick up the
per-pass width. It is called three times off the **same** schedule:

| pass | call | `current_ls_mode` |
|---|---|---|
| bulk | `emit_body isa ()` (outer = avx2/avx512) | `LS_vector` |
| scalar tail (`rem==1`) | `emit_body Isa.scalar ()` | `LS_vector` (ignored at width 1) |
| masked tail (`rem>=2`) | `emit_body isa ()` | `LS_masked "_m"` |

Because the schedule is deterministic and ISA/width-agnostic (`schedule.ml` records
slots/passes as abstract counts with no `vec_width` dependency), the three passes
share the exact node order and spill recipe — no re-scheduling, no duplicated spill
machinery. The avx2 lane mask is a file-scope table:
```c
static const long long _vfft_masklo[VW+1][VW] = { {0,..},{-1,0,..}, ... };
const __m256i _m = _mm256_loadu_si256((const __m256i *)_vfft_masklo[rem]);
```
AVX-512 needs no table: `const __mmask8 _m = (__mmask8)((1u << rem) - 1u);`.

## 6. Scope gate — which codelets get the tail

```ocaml
let anyk_tail       = in_place && (env VFFT_NO_ANYK_TAIL unset)
let tail_scalar_rem1 = (spill = None)   (* rem==1 path: scalar vs masked *)
```

* **`in_place`** — every c2c batch codelet (`rio_re/rio_im/ios/me` signature), both
  **monolithic** (`spill = None`) and **composite / CT-blocked** (`spill = Some`). The
  strided two-pass codelets (a separate quadrant, for 2D row batches) take a different
  signature branch and are excluded automatically.
* **`rem == 1` routing** — monolithic codelets use the **scalar** single lane
  (cheapest at the extreme). Composite codelets route `rem == 1` through the **masked**
  pass instead: their cross-pass scratch `spill_re[]`/`spill_im[]` is declared
  `__m256d`, so a width-1 scalar pass would store `double` into it (type clash) and
  would need the scalar shims the AVX2 preamble doesn't emit. The masked pass needs
  neither — the scratch stores/loads use the raw `isa.storeu_pd`/`isa.loadu_pd` field
  (full-width; masked-off lanes are scratch garbage that never reach `rio`), and only
  the `rio` reads/writes are masked. `mask = 1` active lane handles `rem == 1`.

All **324** in-place AVX2 codelets now carry the tail (170 monolithic + 154 composite);
the regen showed 0 anomalies.

## 7. Kill-switch

`VFFT_NO_ANYK_TAIL=1` at generation time reverts the codelet to the legacy
`for (k = 0; k < me; k += VW)` form — verified **byte-identical** to the pre-tail
committed codelet. The fast path is provably untouched; the switch exists as an
A/B and a safety hatch.

## 8. Validation

End-to-end through the executor (`build_tuned/test/test_anyk_correct.c`): run the
plan `N=1024 [4,4,4,4,4] T1S` at batch `K`, and again at `Kp = roundup(K, 8)` with
the extra lanes zero-padded. Batch lanes are independent, so lanes `0..K-1` must be
**bit-identical** between the two runs (the `Kp` run is all-bulk; the `K` run
exercises bulk + tail).

```
K      rem4  tail_vs_pad  roundtrip      ...  all 21 cells:
1      1     0.00e+00     3.33e-16       ALL BIT-EXACT (21/21)
7      3     0.00e+00     3.89e-16       K = 1,2,3,4,5,6,7,8,9,12,15,16,17,
31     3     0.00e+00     4.44e-16           23,24,31,32,33,63,64,65
33     1     0.00e+00     4.44e-16       tail_vs_pad = 0.0 for every K
```

`tail_vs_pad = 0.0` exactly (not just ε) at every `K`, including all odd `K` and
every `rem ∈ {1,2,3}`. **`K=1`** (the pure scalar single-lane extreme) is exact.
The 484-codelet static lib also compiles clean.

**Composite radixes (r8/r16/r32, blocked two-pass CT) — bit-exact too, bulk *and*
tail.** A diagnostic that splits the error by lane class (`bulk_err` = lanes the
full-vector loop handled, `tail_err` = lanes the tail handled) over nine plans —
`[8,8]`, `[4,8,8]`, `[8,8,8]`, `[16,16]`, `[4,4,16]`, `[32,8]` (T1S **and** FLAT) —
reports `bulk_err = 0.0` **and** `tail_err = 0.0` at every odd `K`. This retires a
long-standing worry: the experiment's "blocked r8 corrupts bulk lanes at odd K" was
a **hand-splice artifact**, not a real executor/seam bug. The generated masked tail
is correct through the spill machinery; the scratch staying full-width is the key.

**Margin vs MKL** (`bench_oddk_tail`, canonical `measure_ab`: best-of-5 min,
cachebust + cool between engines, order-flip averaged; `flip0 ≈ flip1` is the
fairness proof). All cells `corr = 0.0`:

| K | rem | ratio (vfft faster than MKL) |
|---|---|---|
| 32 | 0 | **2.04×** |
| 33 | 1 | **1.72×** |
| 31 | 3 | **1.66×** |
| 24 | 0 | **2.10×** |
| 16 | 0 | **1.87×** |
| 23 | 3 | **1.64×** |
| 17 | 1 | **1.53×** |
| 15 | 3 | **1.50×** |

The aligned cells run ~1.9–2.1× over MKL; the tail costs ~15–25% versus the nearest
aligned cell but **stays comfortably above 1.5×**. This confirms the hand-edited
experiment's ~1.6–1.8×, on the *generated* codelets.

## 9. Coverage and what's deferred

**Done — all in-place AVX2 c2c codelets:** monolithic (r2–r5, r7, primes…) *and*
composite / blocked two-pass CT (r6, r8, r16, r32, r64…), both T1S and FLAT/log3.
The "blocked two-pass odd-K seam bug" turned out to be a hand-splice artifact (§8) —
there is no executor/seam bug; the generated tail is bit-exact through the spill
machinery.

**Deferred (phase-2):**

* **Strided codelets** (the separate 2D-row-batch quadrant, `strided` signature) —
  not yet covered; different signature branch, separate work.
* **AVX-512** — the emit path is in place (`maskz_loadu`/`mask_storeu` + `(1<<rem)-1`
  mask), but the host is AVX2-only so it's untested.
* **Front-door guards** — OOP / real-FFT / auto-dispatch still fail-closed on
  `K % 8 != 0`; relax to `K != 0` once their codelets carry the tail. (The bare c2c
  `plan_create` path already has **no** K guard — odd K works today, monolithic and
  composite.)
* **MT FLAT/log3** — the per-block twiddle staging (`_stride_cmul`) needs its own
  remainder handling. T1S MT is already fine (broadcast twiddles, no staging).

## 10. Reproduce

```sh
# build the generator (WSL: opam 5.2.0 / dune 3.23 — emits platform-agnostic .c)
cd src/dag-fft-compiler/generator && opam exec -- dune build

# regenerate the monolithic in-place AVX2 codelets with the tail
_build/default/bin/gen_set.exe --root ../codelets inplace-avx2

# correctness (no MKL): build + run
cd ../../../build_tuned && python build.py --src test/test_anyk_correct.c --compile
./test/test_anyk_correct.exe

# margin vs MKL: build with --mkl (needs oneAPI on PATH), run one cell per process
python build.py --src benches/bench_oddk_tail.c --mkl --compile
benches/bench_oddk_tail.exe 31 0 80     # <K> <flip> <cool_ms>
```
