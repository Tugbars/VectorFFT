# Arbitrary-K padding — the full design record (2026-06-30)

> **The decision, in one paragraph.** The **SSE2/scalar tail** stays the universal
> default — it makes every K correct on any buffer, and it shipped this session.
> **Padding** (round the batch to `Kp = roundup(K,VW)`, run pure full-SIMD, discard the
> junk lanes) is added as an **opt-in fast path** behind a **padded allocator** (the
> FFTW-`fftw_malloc` model). When the caller used the padded allocator, the **planner
> measures tail-vs-pad per cell** — jointly with the factorization — and records the
> winner in wisdom; when the caller brings a tight buffer, it always tails. The design
> is **uniform across every feature** (no bespoke per-feature padding paths) and
> **correctness-gated** (per-codelet bit-exact test + per-cell fallback + a
> stride-match invariant), so the worst case for any cell is "use the tail," never
> corruption. Measured win where padding applies: **+25–34% on small odd batches.**

This document records *every* decision we reached and *why*, so the reasoning isn't
lost. It supersedes the earlier "admissibility gate" framing in
[[arbitrary_k_pad_vs_tail]] with a cleaner, uniform model.

---

## 0. Implementation status & two refinements (2026-07-01)

Phase 0 gate **PASSED** and **Phase 1 c2c is built + validated end-to-end through the
public API** (`vfft.h`). Two design points were refined *during* implementation; they
supersede the matching bullets in §§9–12 and are the authoritative version:

**Refinement 1 — padded wisdom is a SEPARATE FILE, not a shared-format mutation.**
§10.1 originally proposed extending `wisdom_reader.h`'s tight format with an *optional
padded factorization* trailing field. Rejected in favor of a **parallel file
`spike_wisdom_padded.txt` in the *same* v6 format**: `factors[]` = the **padded-mode**
factorization (`factKp` when pad won, `factK` when the tail won even on the padded
buffer), `exec_me` = `Kp` (run full-SIMD) or `K` (run tail). Rationale: **zero risk to
the 198-cell tight wisdom** (its format is untouched), it needs **no new struct fields**
(the v6 `exec_me` already added is sufficient — a padded cell is fully specified by
`(factors, exec_me)`), and it matches the established per-feature-file pattern (§11).
The tight path never reads it; padded mode reads it and falls back to the tail on a miss.
Wired in `vfft.c`: `struct vfft_wisdom_s.c2c_pad` + `path_c2c_pad`, loaded in
`_bundle_load`, freed in `vfft_wisdom_free`.

**Refinement 2 — the opaque handle (§9.B) binds via `config.batch`.** The allocator
returns the opaque `vfft_batch`; it enters the plan through a new **`vfft_config_t.batch`
field** (NULL = tight, the default drop-in path; non-NULL = opt-in padded). One create
entry point, consistent with the descriptor front door; the handle *is* the padded
signal (no raw-pointer + separate bool, per §9.B). `vfft_create` rejects `config.batch`
for any feature other than **1D c2c in-place** (returns NULL) so a padded buffer can
never be silently strided wrong through an unsupported path — closing the last
silent-corruption gap the design set out to eliminate.

**The measurement lives in DEV TOOLING, not the production runtime — HARD RULE (decided
twice).** §10.2/§12-Phase-1 said "fold the measurement into live `_calibrate_c2c`." Rejected:
`vfft.c`/`vfft.h` is a **clean FFT library** that only **reads the padded wisdom and
dispatches** (padded batch → build at Kp, run the wisdom's `exec_me`; on a MISS → the
always-correct SSE2/scalar **tail fallback**). The pad-vs-tail **A/B benchmark machinery is
NEVER wired into the production API** — it is a **separate concern** that lives in the inner
calibrator (`build_tuned/benches/calibrator/calibrate_pad.c` + `calibrate_pad.py`, one cell
per process, thermal-paced) which writes `spike_wisdom_padded.txt` **offline**. That calibrator
(and the vs-MKL bench) are **shippable to library users as separate tools** they run to unlock
padding — deliberately kept OUT of the drop-in library surface. Rationale (user, decided twice):
separate bulk-calibration / vs-MKL-benchmarking from the FFT-library tool users utilize; don't
burden the API with machinery that isn't of interest to a plain caller. (A `_calibrate_pad`
runtime function was briefly added to `vfft.c` and REVERTED — do not re-add.) NOTE: tight c2c
still calibrates-on-miss in `vfft.c` via `_calibrate_c2c`/`_r2c_bakeoff` — that is the *existing*
design and out of scope here; the NEW padded path is wisdom-read + tail-fallback only.

**Done + tested (all through `vfft.h`):**
- **Allocator** (§10.3) — `vfft_alloc_batch`/`free_batch`/`re`/`im`/`stride`, opaque
  `Kp`-wide **zeroed** handle. Smoke test green.
- **Per-codelet bit-exact gate** (§10.5) — `build_tuned/test/test_pad_gate.c`: tight(`K`)
  vs padded(`Kp`) at `me=Kp`, fwd+bwd, isolated radixes r2…r32 + odd/prime composites
  (`[3,4,4] [5,4,4] [6,8] [7,8] [5,5,5] [7,7] [3,3,3,3]` …) — **23 plans, all bit-exact**
  (`bulk=0 tail=0`). (`test_anyk_correct.c` independently confirms the curated pow2 set.)
- **Dispatch** (§10.4) — `config.batch` → build at `Kp` + run `exec_me`; tail fallback on
  a padded-wisdom miss; **stride-match invariant** assert; **wrinkle-C JIT gating** (only
  the aligned pad leg `me=Kp` resolves the baked/JIT executor; the odd tail leg stays on
  the generic tail-capable executor). `build_tuned/test/test_padded_dispatch.c` drives
  both branches (seeded PAD cell `exec_me=Kp`; TAIL-fallback cell): **bit-exact vs the
  tight reference + roundtrip ~7e-15**.
- **Adversarial review** (4-lens find → verify): 5 confirmed (all nit/low), 2 plausible
  (low), 7 refuted. No critical/high survived; tight path untouched; no memory-safety or
  wrinkle-C issues. Fixes applied: the cross-feature `config.batch` guard (above),
  skip the wasted tight-calibrate when the padded cell already carries a factorization,
  `product(factors)==N` backstop before building from the trusted padded file, and the
  stale allocator comment.

**Pending (deferred — heavy CPU):** `calibrate_pad.c`/`calibrate_pad.py` to populate the
real `spike_wisdom_padded.txt`; the vs-MKL padded-win bench (the Phase-1 perf gate); then
Phases 2–4.

---

## 1. The problem

We process a **batch of K transforms** at once and the SIMD unit does `VW` lanes at a
time (AVX2 `VW=4`). When `K mod VW ≠ 0`, the leftover `1..VW-1` batch lanes don't fill
a SIMD group. Two ways to finish them:

- **Tail** — do the leftover lanes at narrower width (SSE2 width-2 + a scalar
  straggler). Correct, slightly slower per lane.
- **Padding** — pad the batch to `Kp`, run pure full-width SIMD on `Kp` "transforms"
  (the extra `Kp-K` are zeros, results discarded), no tail.

## 2. What already shipped — the tail (universal default)

The rem-aware **SSE2/scalar tail** is generated through the DAG compiler for **every
AVX2 family** (in-place c2c incl. composite, OOP, rfft, c2r, trig). Bit-exact
validated. AVX-512 keeps its masked tail (architectural). This is the correctness
floor: **any K works on any buffer.** Padding is therefore a pure *optimization*
competing against a working tail — decided per cell. See [[arbitrary_k_vectorization]],
[[arbitrary_k_scalartail_experiment]].

## 3. The padding finding (the motivation)

Calibrated, robust interleaved-median A/B (`build_tuned/benches/bench_pad_vs_tail.c`),
rem=3, N=256…4096 — **% padding is faster than the tail**:

| K | N=256 | N=512 | N=1024 | N=2048 | N=4096 |
|---|---|---|---|---|---|
| **7** | +33% | +1% | +22% | +24% | +28% |
| **11** | +15% | +9% | **+34%** | +33% | +25% |
| 15 | −3% | +3% | +5% | +18% | −6% |
| 19–31 | mixed | mixed | mostly + | mixed | mixed |

- **Small K (7, 11): padding wins everywhere, decisively (+25…+34%)** — the narrow tail
  is a huge fraction of the work there, so trading it for one wasted full-width
  transform is a big net win.
- **rem=1 / large K:** tail wins or ties (padding wastes too much, or the tail is
  already well-amortized).
- **Crossover rule:** padding wins when wasted-fraction `(Kp−K)/K` < the tail's overhead
  fraction → **small K, rem ≥ 2.**

We **conceded** the win is *not* marginal — the clean cells are a third faster.

## 4. The core constraint — padding needs `Kp` memory *in the right layout*

The expensive part of padding is **not** the extra memory — it's the **layout**. Our
batch-packed split layout puts element `e` of transform `b` at `re[e*K + b]`: the batch
axis `b` is **contiguous**, elements are at stride `K`. Padding widens the stride to
`Kp`, and the spare columns are **interleaved between every element-row**, not appended:

```
tight  (stride 3): [ t0e0 t1e0 t2e0 | t0e1 t1e1 t2e1 ]
padded (stride 4): [ t0e0 t1e0 t2e0 0 | t0e1 t1e1 t2e1 0 ]
                                    ^                  ^   spares woven between rows
```

So "pad" means **the data must physically live at stride `Kp`.** Getting tight (`K`)
data into a `Kp` layout is a **re-thread** — a copy of the whole batch (every element
past row 0 moves). On a memory-bound FFT that copy can be ~4 extra passes vs the FFT's
~`2·nf`, so **transparently copying-to-pad loses** — it costs more than the tail it
saves. Padding is free only when the data is **born** at stride `Kp` (contract) or the
re-thread **rides a write that's already happening** (OOP).

## 5. Where padding is free vs cursed (and the FFTW comparison)

- **OOP:** the stage-0 codelet already reads `src` and writes a fresh `dst`. Set
  `os=Kp` and the re-thread rides that write — **free.**
- **Strided / AOS (2D row codelet):** each transform is contiguous, so padding *appends*
  rows — no re-thread. (Moot for the 2D *production* path, which already handles odd
  dims via the transpose's scalar edges + the vertical tail — see
  [[strided_2d_tail_no_hard_tail]].)
- **In-place:** no free write exists; the data just sits there tight → padding needs a
  contract or a losing copy. **This is the only genuinely "cursed" case.**

**FFTW/MKL don't have this problem at all:** they use *interleaved* complex and
vectorize *within* a transform, looping over the batch — so the SIMD width never has to
divide the batch count; there's no batch remainder (it's why MKL is flat-in-K). We have
the remainder *because* our split + batch-packed layout vectorizes *across* the batch —
the same choice that wins us MT scaling, batch streaming, and the odd-radix/scrambled
blind-spot lead. **Padding-vs-tail is the maintenance cost of a layout that wins
everywhere else** — not a feature we lack relative to FFTW.

## 6. The converged design — opt-in padded allocator + measured tail-vs-pad

We do **not** force a layout (that would break drop-in interop and just relocate the
re-thread onto callers whose data arrives tight). Instead, the **FFTW model**: provide a
**padded allocator**; callers who use it get padding, everyone else gets the tail.

### The key realization that makes it uniform and footgun-free

On a **padded buffer** (stride `Kp`), we build **one `Kp`-strided plan**, and *both*
choices are valid for the K real transforms:

- **`me = Kp`** → pure full-SIMD, no tail, computes the junk columns too → **pad**.
- **`me = K`** → bulk loop + SSE2/scalar tail for `K mod VW`, skips junk → **tail**.

Same buffer, same plan, just a different batch count. So the wisdom's pad-vs-tail
verdict is simply **"which `me`"** — and it is **always honorable**, because both are
correct and free on a padded buffer. That dissolves the deepest danger (a wisdom
"pad" verdict the runtime can't legally honor): **pad is only ever consulted in padded
mode, where it is always legal.**

### The uniform rule (every feature)

> **Padded buffer → build the plan at `Kp`, run at the wisdom's `exec_me` (`Kp`=pad or
> `K`=tail). Tight buffer → build at `K`, run `me=K` with the tail.**

No new codelets, no new executor modes — the codelets already have both the full-SIMD
bulk *and* the tail; `plan_create_ex(N, W, …)` already takes the width as a parameter.
All the bespoke per-feature padding mechanics (OOP riding the write, MODEB
scrambled-backward, r2c-pack-riding, in-place fusion) were only needed to *manufacture*
`Kp` memory on a tight buffer — the padded allocator obviates all of them.

## 7. The verified technical model (from the code investigation)

A 6-agent code investigation + adversarial verification (709K tokens) confirmed:

- **`ios` (batch stride) is baked into the plan** (`st->stride = K·∏inner factors`,
  `src/core/engine/twiddle.h:62-71`); the runtime `K` arg flows **only** to `me`/
  `slice_K`. ⇒ a bare `me=Kp` against a `K`-built plan **over-runs into the next
  transform** — *wrong*. Padding **requires a `Kp`-built plan** (`plan_create_ex(N,Kp,…)`).
- **Twiddle ANGLES are K-independent** (`twiddle.h:251-296`; K is only a
  replication/stride count) ⇒ a `Kp` rebuild reuses the same factorization+variants and
  just re-lays-out geometry — **cheap to construct.**
- **Codelet bodies are stride-agnostic** (`rio[leg*ios + k]`) — widening `ios` K→Kp is
  the only change to addressing; the body is unchanged.
- **The leg-0 cf0 cmul / bwd conj sweep reads `slice_K` contiguous doubles**
  (`executor_generic.h:77-113`), so the **batch axis must be physically contiguous and
  ≥ the run width** — which the padded buffer (`Kp`-wide, batch contiguous) satisfies.
- **OOP is stage-0-only**; no last-stage-OOP mode exists — irrelevant under the allocator
  design (we don't fuse), but it's why the in-place *fusion* alternative was "large."
- **Calibrator buffer-overflow gotcha (real):** `vfft_proto_dp_bench_explicit` benches
  `N·K_eff` but the dp context buffers are sized `max_N·ctx->K`, and the `K_eff` plumbing
  is **dead** (every caller passes `ctx->K`). Benching at `Kp>K` **overflows** → the pad
  leg needs a **second dp context sized at `Kp`** (or self-alloc).
- **Wisdom trailing-field scheme is forward/backward compatible** (`wisdom_reader.h:88-91`
  load stops after the `nf` variants; old binaries ignore extra tokens, new binaries
  default a missing token).
- *(Citation note: the runtime c2c path is `src/core/engine/{executor_generic.h,
  twiddle.h, stride_executor.h}`; some investigator citations drifted into the
  `dag-fft-compiler` generated tree — substance held, line refs corrected here.)*

## 8. Every danger, and how it is resolved

1. **"Pad" is a conditional strategy the wisdom can't guarantee at runtime.**
   → **Resolved.** Pad is only recorded/consulted in **padded mode**, where the same
   `Kp`-plan runs at `me=Kp` *or* `me=K` — both legal. Tight mode never reads it.
2. **"Padding for every feature" = N bespoke paths.**
   → **Resolved.** The padded-allocator path is **uniform** (build at `Kp`, run at
   `exec_me`); the per-feature complexity was only for free-padding on tight buffers,
   which we don't do.
3. **Marginal/noisy win.**
   → **Withdrawn** — it's +25…+34% on the clean cells. (The only residue: under MT the
   batch is sliced into blocks, so "round up to VW" is per **`block_K`**, not K — a
   wiring detail, addressed by padding per block or shipping ST-first.)
4. **In-place specifics.** The scary ones (re-thread copy, fusion executor mode) are
   **gone** — the caller brings `Kp` memory. Residuals, each tackled:
   - *Plan-stride must equal buffer-stride* → the dispatch builds the plan at the
     buffer's actual stride (read from the handle); **invariant:** a plan built at width
     `W` only ever runs at `me ≤ W` on a `W`-strided buffer; one assert at the execute
     boundary.
   - *Junk lanes must be in-bounds + zeroed* → the **allocator zeroes** the pad columns
     (avoids denormal/NaN slowdown of junk lanes; correctness is already safe by lane
     independence).
   - *"Junk can't contaminate real lanes" is asserted, not proven* → the **validation
     gate** (below).
   - *`Kp`-geometry edge cases (log3/t1p tables, composite spill sized at `Kp`)* →
     covered by the gate; any failing cell **falls back to `exec_me=K`** (the validated
     tail).

## 9. The three wrinkles — decided

- **A — factorization optimality. DECIDED: joint search.** The K-optimal factorization
  is biased by the tail penalty (toward fewer stages); run padded, that bias is gone, so
  the **`Kp`-optimal factorization can differ.** **The planner finds the best
  factorization for that specific K considering BOTH the tail path and the full-SIMD
  path** — i.e. the calibrator runs a DP search at `K` (tail) *and* at `Kp` (pad), and
  records the globally-best `(factorization, exec_me)` for the padded mode (separately
  from the tight-mode `K`-factorization). Don't leave the few % on the table.
- **B — API binding. DECIDED: opaque handle.** The padded allocator returns a handle (or
  buffer+stride pair) that the execute path reads — the buffer **carries its own
  padded-ness**, à la an FFTW plan binding its buffer assumptions. A raw pointer + a
  separate "is it padded?" boolean is forbidden (a padded buffer passed through the tight
  path would build a `K`-plan and corrupt).
- **C — JIT/baked unlock. DECIDED: bench on the runtime path.** The baked/JIT executor is
  gated to `K%VW==0`; odd K falls to the generic executor. But `me=Kp` **is** VW-aligned,
  so **padding makes the odd-K batch eligible for the baked fast path.** Padding is
  therefore not only "skip the tail," it can "upgrade to the fast executor." The
  calibrator must bench `me=Kp` on the path the runtime will use (baked, if available),
  or it **under-rates** padding. Free upside if captured; a measurement bias if not.

## 10. Implementation plan (file-by-file)

1. **Wisdom** (`src/core/planning/wisdom_reader.h`): add trailing tokens — `exec_me` and
   an optional **padded factorization** (written only when it differs from the tight
   one). Forward/backward compatible: absent ⇒ `exec_me=K` ⇒ today's behavior. Written
   **only for misaligned K** (aligned cells stay blank — "K=4: nothing; K=7: a code").
2. **Calibrator** (`_calibrate_c2c`, `src/core/vfft.c:226-272`): for misaligned cells,
   run **DP@K** (tail, existing) **and DP@Kp** (pad), each best-of-min; an **interleaved
   A/B** between the two winners (no paired-interleave primitive exists in `measure.h` —
   add it; fixed-order A/B is thermally biased); **3% hysteresis toward the tail** on
   near-ties (mirrors `_r2c_bakeoff`, `vfft.c:347`). Bench `me=Kp` **on the baked path**
   when the runtime would use it (wrinkle C). Set `exec_me` in **both** the DP and the
   exhaustive branch. **Use a second dp context sized at `Kp`** for the pad leg (the
   overflow fix).
3. **Padded allocator + handle** (new, front door): returns a `Kp`-wide, **zeroed**
   buffer + the stride, as an opaque handle. Matching free (respect
   `_aligned_malloc`/`posix_memalign` ↔ `aligned_free`).
4. **Runtime dispatch** (`src/core/vfft.c`, `planner.h:233-234`): padded handle ⇒ build
   plan at `Kp` (the padded factorization) and run at `exec_me`; tight ⇒ build at `K`,
   run `me=K` with the tail. Plan/wisdom keyed on `(N,K)` but **built at the buffer's
   actual stride**.
5. **Validation gate** (the safety net): per-**codelet** bit-exact — for every radix ×
   variant × {n1,t1} × direction, build the `Kp` plan, run `me=Kp`, require the K real
   lanes bit-exact vs the `K`-plan `me=K`. Per-codelet granularity covers all
   factorizations structurally. Nothing merges until green.
6. **Stride-match invariant assert** at the execute boundary (cheap, catches any wiring
   slip into silent corruption).

## 11. The per-feature wisdom fragmentation (the honest scope)

The design is **uniform at the EXECUTION layer** (build at `Kp`, run `exec_me` — same
codelets, same mechanism, no new executor) but the **wisdom + calibration layer is
per-feature**: each feature persists its own wisdom in its **own file + own
parser/format**. So `exec_me` is **not a one-place add** — it must be added to each
feature's format, and each feature's calibrate path must run its own Step-0 measurement.

| feature | wisdom format / file |
|---|---|
| **1D c2c** | `src/core/planning/wisdom_reader.h` → `spike_wisdom.txt` / `vfft_wisdom_tuned` ← **reference impl** |
| OOP c2c | `src/core/oop/oop_wisdom.h` → `oop_wisdom.txt` |
| r2c | `src/core/transforms/real/r2c_dispatch.h` (own format) |
| c2r | `src/core/transforms/real/c2r_dispatch.h` (own format) |
| 2D c2c | `src/core/transforms/fft2d/fft2d_c2c_wisdom.h` |
| 2D r2c | `src/core/transforms/fft2d/fft2d_r2c_wisdom.h` |
| prime N | `src/core/primes/bluestein_wisdom.h` |
| MT variants | separate `*_mt.csv` files |

So **"padding for every feature" = the same execution mechanism replicated across ~5
independent wisdom formats + calibrators** — some of which (2D already handles odd dims;
odd-K r2c routes to the rfft cascade and ST-loses anyway) may not even *want* it. **c2c
is the reference implementation; the rest follow it one feature at a time.** Each row
above that we pursue needs its own `exec_me` field + Step-0 measurement noted as an
explicit follow-up (Phases 2–4 below). (A standing roadmap item,
`wisdom_bridge_retirement`, would consolidate these formats and make `exec_me` a
one-place add — optional; don't block padding on it.)

## 12. Rollout — phased & gated, feature-by-feature

Never build features speculatively: c2c proves the pattern, and two gates decide whether
to go further.

### Phase 0 — c2c measurement
*Trustworthy per-cell pad-vs-tail verdicts for 1D c2c, capturing the JIT advantage.*
- ✅ `exec_me` in `wisdom_reader.h` (struct + load default-K + save, `@version 6`,
  forward/backward compatible).
- ✅ Step-0 driver `build_tuned/benches/step0_pad_vs_tail_calib.c` — joint DP@K / DP@Kp,
  interleaved-median A/B on the `Kp` buffer, 3% hysteresis toward tail, writes a v6
  side-file. Producing sane verdicts (pad wins most cells +4…+25%; `factKp ≠ factK` in
  ~half — the joint search is live).
- ☐ **Wire the JIT/baked path into the pad leg** (wrinkle C: `vfft_proto_plan_jit_fwd(pP)`;
  tail stays generic) — else we under-rate pad on the generic-only floor.
- ☐ Run the full sweep → table + side-file. Read: how often / by how much pad wins *with
  JIT*, which near-ties flip.
- **🚦 GATE:** is the JIT-included pad win big enough to build the runtime? If marginal
  once the aligned-stride tail + JIT are in, **stop — the tail is already great.**

### Phase 1 — c2c execution (the real win)
- ☐ **Padded allocator** — `vfft_alloc_batch(N,K)` → opaque handle (zeroed `Kp` buffer +
  stride); matching free.
- ☐ **Store the pad (`Kp`) factorization** in wisdom when `exec_me=Kp` (reserved
  second-factorization trailing field) — the runtime needs the `Kp`-optimal plan, not the
  tail's.
- ☐ **Fold the measurement into live `_calibrate_c2c`** (`src/core/vfft.c:226-272`) — write
  `exec_me` + pad factorization into the production wisdom; set it in **both** branches;
  use a **second `Kp`-sized dp context** (the verified overflow fix).
- ☐ **Dispatch** — padded handle → build `Kp`-plan (pad factorization) + run `me=exec_me`;
  tight → `K`-plan + tail. **Stride-match invariant assert** at the execute boundary.
- ☐ **Validation gate** — per-codelet bit-exact `me=Kp` vs `me=K`, fwd+bwd. Non-negotiable.
- **🚦 GATE:** bit-exact green **+** a vs-MKL bench showing the padded win on real cells.
  Only this justifies the per-feature rollout.

### Phase 2 — OOP c2c (easiest next)
`exec_me` in `oop_wisdom.h`; Step-0 in the OOP calibrator; dispatch (OOP **rides the
write** — the allocator pads the output, free); validation. Lowest-friction feature.

### Phase 3 — r2c / c2r (conditional)
`exec_me` in their own formats; handle the known nuances — **odd K routes to the rfft
cascade** (uncalibratable), **MT pad axis = `block_K`** not K. Lower priority (r2c ST
structurally loses); gate hard on Phase-1 results.

### Phase 4 — trig / 2D (as needed)
- **2D:** skip — already handles odd dims (transpose scalar edges + vertical tail).
- **trig:** Makhoul stride path; only on demand.

### Cross-cutting (later, within each feature)
- **MT padding** — pad per `block_K`; follow-up once ST works.

### Immediate next steps
1. JIT-path fix in the Step-0 driver (wrinkle C).
2. Run the full c2c sweep.
3. Review verdicts → decide Phase 1 at the gate.

## 13. Why it is robust (the backstops)

- **Correctness:** three independent guards mean **no path to silent corruption** — the
  **stride-match invariant** (catches plan/buffer mismatch), the **per-codelet bit-exact
  gate** (proves `me=Kp` ≡ `me=K` on the real lanes), and **per-cell `exec_me`
  fallback** (any failing or losing cell degrades to the already-validated tail). The
  worst possible outcome for any cell is "padding didn't help here, use the tail."
- **Performance:** the decision is **measured per cell** (jointly with factorization, on
  the real runtime path), so we capture the +25–34% small-odd-K win wherever the buffer
  allows it, and never regress (hysteresis + fallback default to the tail).
- **Scope:** **uniform** — one mechanism for every feature, no bespoke per-feature
  padding paths, no new executor modes, no new codelets.

## 14. What we explicitly are NOT doing (and why)

- **No transparent in-place copy-to-pad** — the re-thread copy (~4 passes) loses to the
  tail on a memory-bound kernel.
- **No in-place full-fusion executor mode** — the padded allocator makes `Kp` memory
  available without it, so the new-executor-mode work is unjustified.
- **No mandatory padded layout** — it would break drop-in interop and merely relocate the
  re-thread onto callers with tight external data; the opt-in allocator captures the win
  from cooperative callers while everyone else gets the tail.
- **No measured padding on tight buffers** — pad is inadmissible there; tail only.

## See also
- [[arbitrary_k_pad_vs_tail]] — the bench + the earlier planner-pivot (superseded by this).
- [[arbitrary_k_tail_strategy]] — scrambled→SSE2 / natural→transpose.
- [[strided_2d_tail_no_hard_tail]] — 2D already handles odd dims; no hard tail remains.
- [[arbitrary_k_vectorization]] / [[arbitrary_k_scalartail_experiment]] — the shipped tail.
- [[memory_bound_thesis]] — why the copy-to-pad path loses.
- [[mkl_blind_spot_positioning]] — why our layout (and thus the tail/pad cost) is the win.
