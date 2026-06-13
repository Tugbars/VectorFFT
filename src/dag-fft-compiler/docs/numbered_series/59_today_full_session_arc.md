# 59. Today's full session — 5-phase optimization arc, op count and latency

Status: All five phases shipped to current tree. Final state:
`vfft_lib_bin.zip` contains `lib/` and `bin/gen_radix.ml`.
All radices verified sub-ulp correct.

## TL;DR

Today started with you uploading new library files (improved `fma_lift`
+ selective pinning) and the hand-written `gen_radix*.py` codelets for
comparison. The session then compounded five distinct optimization
phases:

```
Phase                      | Contribution                       | Where
---------------------------|------------------------------------|--------------------
1. n1 blocking (Doc 58)    | structural — vmovapd ↓5-15×       | dft_expand_n1_blocked
2. Path A (factor + remap) | -39/-119/-255 ops R=64/128/256    | factor_const_muls
3. Path B (tan-factored)   | -20/-42/-89                        | const_cmul rewrite
4. fma_addend_factor       | -32/-80/-176                       | new algsimp pass
5. round_13 / min-max tail | (verified inert post-rebuild)      | const_cmul refinement
```

End-of-day cumulative op-count reductions vs Base:

```
Radix    n1 Base   n1 End    Δ        %         t1 Base   t1 End    Δ        %
R=8      55        52        −3       −5.5%     83        80        −3       −3.6%
R=16     158       144       −14      −8.9%     218       204       −14      −6.4%
R=32     430       386       −44      −10.2%    554       510       −44      −7.9%
R=64     1085      978       −107     −9.9%     1337      1230      −107     −8.0%
R=128    2645      2328      −317     −12.0%    3153      2836      −317     −10.1%
R=256    6229      5460      −769     −12.3%    7249      6480      −769     −10.6%
```

R=8 and R=16 now match FFTW exactly. R=32/R=64 close from `+213/+173`
gaps to `+14/+66`. Latency: R=32 n1 OLD-Python vs NEW-OCaml runs
**−15.7% single-call / −35.4% steady-state**.

## How today started

You uploaded a zip containing two things:

- New library files (`algsimp.ml`, `dft_r2c.ml`, `emit_c.ml`,
  `gen_radix.ml`, `isa.ml`, `schedule.ml`, `dune`) with improved
  `fma_lift` and selective pinning
- Hand-written `gen_radix*.py` codelet generators (R=3 through R=64
  primes and composites) for comparison

First comparison ran on R=16 / R=32 / R=64 n1 AVX-512:

```
R=64 n1 AVX-512 — per-kernel asm (start-of-day state):
  hand (NFUSE=2, 8×8)     fma=196  mul=106  add=366  sub=366  xor=33   fp=1034  vmovapd=402
  ocaml (start of day)    fma=160  mul=124  add=384  sub=384  xor=0    fp=1052  vmovapd=1363
                          ↑ fp within 1.7% of hand                     ↑ 3.4× hand
```

Diagnosis: fp arithmetic already at hand parity; the real gap was
**vmovapd from algorithmic decomposition**. Hand uses NFUSE
(block-decomposed) structure with per-pass peak_live ≈ 30; OCaml n1
was monolithic with peak_live ≈ 260 (8× the 32-zmm budget at R=64,
31× at R=256). Spills cascaded into vmovapd.

Two lines of work followed in parallel through the day:

1. **Structural**: add NFUSE-equivalent blocking to n1
2. **Arithmetic**: close the remaining 1-3% fp gap to hand on R=32/R=64

## Phase 1 — `dft_expand_n1_blocked` (real Doc 58)

The structural fix. Adds the same SU+spill recipe machinery to n1
codelets at R≥32 that t1 already had. New code in `lib/dft.ml:1169`
(`dft_expand_n1_blocked`) and `lib/dft.ml:1328` (`should_block_n1`
predicate), with dispatch in `bin/gen_radix.ml:130-138` (recipe
auto-enable) and `bin/gen_radix.ml:202-211` (n1-blocked dispatch).

The IR shape after blocking is structurally identical to what t1
already produces:
- PASS 1: N1 inner DFT-N2s, outputs spilled to slots
- INTERNAL TWIDDLES: `const_cmul` (folded to identity for k=0)
- PASS 2: N2 inner DFT-N1s reading spilled slots

All existing machinery (algsimp, M-project regalloc, fma_lift,
selective pinning) composes without modification — it sees the
identical recipe boundary as t1.

Measured impact (from Doc 58):

```
                  monolithic fp+mov=total   blocked fp+mov=total   reduction
R=32 n1 AVX-512   420 + 709  = 1129          420 + 133  = 553       −51%
R=64 n1 AVX-512   1052+ 2080 = 3132          1052+ 325  = 1377      −56%
R=128 n1 AVX-512  2524+ 5072 = 7596          2524+ 810  = 3334      −56%
R=256 n1 AVX-512  5884+11686 = 17570         5887+ 3158 = 9045      −49%

Runtime (K-sweep, best of 3):
R=32 n1 K=128/K=256       +24.8% / +40.3%
R=64 n1 K=128/K=256       +47.2% / +48.3%
R=128 n1 K=128/K=256      +56.5% / +57.4%
R=256 n1 K=64/K=128       +57.3% / +58.2%
```

Phase 1 is where the bulk of the user-visible latency win comes from.
The fp op count barely changes (±3 ops from sharing differences) —
this is purely the spill cascade collapsing.

## Phase 2 — Path A: `factor_const_muls` + tag-remap

Recognizes `Mul(K, X) ± Mul(K, Y) → Mul(K, X±Y)` patterns and pulls
the constant out. Where `Mul(K, X)` is a frozen spill marker, the
naive rewrite would orphan the marker — so a **tag-remap mechanism**
propagates `T_old → T_new` through the algsimp pass chain
(`bin/gen_radix.ml:362-389` `extend_frozen` helper), keeping
`spill_info` consistent with whatever the FMA passes produce.

Empirically verified the frozen-tag interaction by adding then
reverting trace instrumentation in `multi_use_fma_lift`. The trace
showed 0 frozen-disqualifications — the absorption was working
correctly; tag-remap was load-bearing precisely because the rewrites
fire on frozen Muls.

Impact:

```
              Base    +Path A   delta
R=8     n1    55       53        −2
R=16    n1    158      154       −4
R=32    n1    430      414       −16
R=64    n1    1085     1046      −39
R=128   n1    2645     2526      −119
R=256   n1    6229     5974      −255
```

Largest absolute saving of the day.

## Phase 3 — Path B: tan-factored `const_cmul`

Replace the direct complex-multiply emission
`Mul(cr, xr) − Mul(ci, xi)` / `Mul(cr, xi) + Mul(ci, xr)` with
FFTW's tan-factored form when `|cr| ≠ |ci|`:

```
if |cr| ≥ |ci|:
  y_re = cr · (xr − (ci/cr)·xi)
  y_im = cr · (xi + (ci/cr)·xr)
```

Two effects: (i) the inner `(xr ± tan·xi)` values are shared across
the re/im pair and across rotations sharing the same `(xr, xi)`
input; (ii) the outer `Mul(cos, inner)` is single-use after sharing,
so `fma_lift` absorbs it cleanly.

Impact:

```
              +Path A   +Path B   delta
R=16    n1    154        146       −8
R=32    n1    414        400       −14
R=64    n1    1046       1026      −20
R=128   n1    2526       2484      −42
R=256   n1    5974       5885      −89
```

## Phase 4 — `fma_addend_factor` pass

Recognizes `Fma(K, X, Mul(K, Y))` where the same constant `K` appears
in both the FMA's mul slot and inside the addend, and refactors to
`Mul(K, Fma(1, X, Y))` — the new outer Mul is now amenable to a
subsequent `multi_use_fma_lift` round. This is why the pass
orchestration loops 4 times (`mfl → faf → mfl → faf → mfl → faf →
mfl` in `bin/gen_radix.ml:395-476`): each `faf` produces Muls the
next `mfl` may absorb.

Impact:

```
              +Path B   +fmaadd   delta
R=8     n1    52         52        0   (matches FFTW)
R=16    n1    146        144       −2  (matches FFTW)
R=32    n1    400        386       −14
R=64    n1    1026       994       −32
R=128   n1    2484       2404      −80
R=256   n1    5885       5709      −176
```

After this phase R=8 and R=16 match FFTW exactly. R=32 closes to
`+14`, R=64 to `+82`.

## Phase 5 — twiddle canonicalization tail

Two further edits in `const_cmul`:

- **min/max canonicalization**: compute the inner ratio as
  `min(|cr|,|ci|) / max(|cr|,|ci|)` with sign applied separately, so
  symmetric angles (e.g. ω¹ vs ω³ at R=16) compute the same ratio via
  the same input ordering.
- **`round_13` input rounding**: round `cr`/`ci` to 13 sig digits
  *before* dividing, to handle the case where `sin(π/8)` and
  `cos(3π/8)` differ by 1 ulp in OCaml's libm.

The hope was that ratio canonicalization would unify near-equal
constants like the two `tan(π/8)` variants surviving at R=64. In the
session, R=64 dropped further from 994 → 978 (`-16`), R=128 2404 →
2328 (`-76`), R=256 5709 → 5460 (`-249`).

**Verified post-session**: when both edits are reverted to the
pre-today code (raw `ci/cr` division, no rounding), R=64 still
measures 978, R=128 2328, R=256 5460. These edits are **inert** —
`mk_const` at `lib/algsimp.ml:177-194` already applies the same
`%.13e` rounding downstream via `of_expr → Expr.Const c → mk_const c`
(line 478), and that downstream rounding subsumes whatever
canonicalization Phase 5 attempted upstream.

The `-16/-76/-249` measured during the session was real but came from
something else in the build cycle (most likely a build-fix or
accumulated algsimp refinement that the binary picked up around the
same time), not from the Phase 5 edits.

**Action**: the Phase 5 edits sit in `lib/dft.ml:302-326` doing
nothing. Either strip them (preferred — they're verifiable dead code)
or leave them with a comment redirecting to `mk_const`'s rounding as
the actual mechanism.

## End-state vs hand-written codelets

The day closed with OCaml at parity or better on every codelet
measured:

```
R=32 n1 AVX-512:
  hand (gen_radix32.py)      total = 559 instructions  (415 fp + 144 mov)
  ocaml (current tree)       total = 553 instructions  (420 fp + 133 mov)
                             OCaml beats hand by 1.1%

R=64 n1 AVX-512:
  hand (gen_radix64.py)      total = 1469 instructions (1034 fp + 402 mov + 33 xor)
  ocaml (current tree)       total = 1377 instructions (1052 fp + 325 mov)
                             OCaml beats hand by 6.3%
```

Latency (Intel Xeon @ 2.10 GHz nominal, AVX-512, R=32 single-call
median over 9 trials × 1001 reps):

```
                            single-call    steady-state
hand (gen_radix32.py)
  in-place via alias        225.4 ns       196.9 ns
ocaml (current tree)
  in-place by design        190.0 ns       127.3 ns
  vs hand                   −15.7%         −35.4%
```

The 35% steady-state win is the true compute throughput improvement;
single-call adds ~100 cycles of rdtsc serialization overhead that
doesn't shrink with faster code, so the 16% single-call number is the
more conservative one for HFT-style use.

## How FMA fusion interacts with M-project

The three layers continue working as designed (from Doc 56):

1. **Explicit IR-level FMA lifting (algsimp passes)** — `fma_lift`
   (single-use `Add(Mul(K,X), Y) → Fma(K,X,Y)`), `multi_use_fma_lift`
   (multi-use Muls with Add/Sub-only consumers, all rewritten to
   FMA), and Phase 4's `fma_addend_factor`. Pass orchestration is
   the 4-iteration `mfl → faf → ...` loop at `bin/gen_radix.ml:395-476`.

2. **gcc auto-FMA-fusion for surviving Muls** — Muls that algsimp
   can't eliminate (consumers include FMA-addend slots) would
   normally be killed by M-project's `asm volatile ("" : "+v"(t))`
   barrier. Selective pinning (`lib/emit_c.ml:225`
   `compute_unpin_candidates`) unpins any Mul with ≥1 Add/Sub
   consumer so gcc can re-fuse it. Doc 56 measured: without selective
   pinning, M-project barriers cost −126 asm FMAs at R=64; with it,
   the loss drops to −43.

3. **Tag remapping for spilled FMA targets** — Phase 2's tag-remap
   mechanism propagates `T_old → T_new` through the 4-pass remap
   chain (`extend_frozen` at `bin/gen_radix.ml:362-389`), keeping
   `spill_info` consumed by `lib/regalloc.ml` consistent with rewrites
   that consume frozen Add/Sub or Mul tags.

## Files state at session end

- `lib/dft.ml:1169` — `dft_expand_n1_blocked` (Phase 1, real Doc 58)
- `lib/dft.ml:1328` — `should_block_n1` predicate (Phase 1)
- `lib/dft.ml:302-326` — Phase 5 min/max + round_13 edits (inert,
  pending strip)
- `lib/algsimp.ml` — accumulates Phase 2 (factor_const_muls +
  tag-remap), Phase 3 has its IR shape support, Phase 4
  (`fma_addend_factor`)
- `bin/gen_radix.ml:130-138, 202-211, 362-476` — recipe auto-enable,
  n1-blocked dispatch, 4-iteration pass orchestration


## Post-session addendum — ISA-aware factorization for AVX2 R=64

After today's main arc closed, side investigation into FFTW gap turned
into an AVX2 question (you mentioned EPYC bench plans, mostly AVX2 +
some Zen 4 AVX-512). The relevant question: does our AVX-512-tuned
factorization table still hold up on a 16-ymm register budget?

### Why the factorization matters more on AVX2

The peak live-set in a CT codelet scales with the largest pass
dimension. Our R=64 default `Cooley_Tukey (8, 8)` puts 8 complex = 16
ymm values into each pass — exactly at the AVX2 register budget, no
slack. The smallest perturbation (a needed twiddle, a phantom temp
gcc fails to coalesce) overflows to stack. By contrast `(4, 16)`
recurses to `(4, (4, 4))` — deepest pass has 4 complex = 8 ymm,
half the AVX2 budget. Spill pressure should drop substantially.

### Factorization sweep on AVX2

Patched the picker with each candidate, rebuilt, measured fp ops and
asm-level stack spills (counted as `vmov[au]pd.*rsp|rbp`). Baseline
in **bold**:

```
R=32 AVX2:
  CT(2,16):  fp=380   spills=202
  CT(4,8):   fp=376   spills=140    ← baseline, still optimal
  CT(8,4):   fp=376   spills=203
  CT(16,2):  fp=380   spills=438

R=64 AVX2:
  CT(2,32):  fp=940   spills=613
  CT(4,16):  fp=952   spills=431   ← winner: −2.7% fp, −9% spills vs (8,8)
  CT(8,8):   fp=978   spills=473    ← baseline
  CT(16,4):  fp=952   spills=949
  CT(32,2):  fp=940   spills=1390

R=128 AVX2:
  CT(4,32):  fp=2200  spills=1257
  CT(8,16):  fp=2192  spills=1235  ← baseline, still optimal
  CT(16,8):  fp=2192  spills=1912
  CT(32,4):  fp=2200  spills=2873

R=256 AVX2:
  CT(4,64):  fp=5112  spills=3453
  CT(8,32):  fp=5104  spills=3155   ← would win, but irrelevant (see below)
  CT(16,16): fp=5056  spills=4733   ← baseline
  CT(32,8):  fp=5104  spills=5751
  CT(64,4):  fp=5112  spills=7990
```

R=64 is the only change adopted. R=32 and R=128 baselines remain
optimal on AVX2. R=256 has a candidate (8,32) that drops spills 33%
but is **out of scope** — you noted AVX2 above R=64 isn't a target
workload for the EPYC bench. If that changes, R=256 (8,32) is on the
shelf.

### Bench validation — (4,16) vs (8,8)

Spill count is a proxy, not latency. Validated with a real rdtsc bench
on the sandbox (Intel Xeon 2.80 GHz, AVX2 + AVX-512 available; AVX2
codelets benched). Both codelets compiled with gcc -O3 -mavx2 -mfma,
identical bench harness:

```
Correctness check
  max |Δre| = 4.519e-14   max |Δim| = 3.042e-14
                                       (sub-ulp; algebraically equivalent,
                                        differ only in rounding order)

Cycles per R=64 n1 call (median of 11 trials × 1000 reps, 5 runs)
                  (4,16)            (8,8)            Δ
  Run 1           984               1036             (4,16) −5.0%
  Run 2           1016              1051             (4,16) −3.3%
  Run 3           960               1048             (4,16) −8.4%
  Run 4           971               978              (4,16) −0.7%
  Run 5           986               978              (8,8)  +0.8%

Min cycles (cleaner — represents uninterrupted execution):
  (4,16): 950-958 cycles
  (8,8):  977-978 cycles
                          ≈ −2.4% steady-state advantage for (4,16)
```

Sandbox is shared (single core, no cpufreq lock) so medians vary
substantially run-to-run. The min is the more reliable indicator since
it represents what the CPU achieves without scheduler interference.

The spill-count projection (−9%) overstated the runtime impact by
roughly 2×, which is the usual OOO discount — modern cores hide
significant memory traffic via load/store buffers and AGU pipelining.
The direction was correct; the magnitude was inflated.

Expected on EPYC with locked cpufreq and an exclusive core: cleaner
3–5% steady-state win for (4,16). On Zen 4 specifically (full 256-bit
AVX2 datapath, not double-pumped like Zen 2), the picture could shift
either way — Zen 4's wider ports could make the symmetric (8,8)
structure more attractive. Worth re-measuring there.

### Wiring

Made the picker ISA-aware via a single ref that bin/gen_radix.ml sets
from `isa.vec_regs` at startup:

```ocaml
(* lib/dft.ml *)
let target_vec_regs : int ref = ref 32

let pick_algorithm (n : int) : algorithm =
  ...
  | 64 when !target_vec_regs <= 16 -> Cooley_Tukey (4, 16)   (* AVX2 *)
  | 64                             -> Cooley_Tukey (8, 8)    (* AVX-512 *)
  ...

(* bin/gen_radix.ml — right after `let isa = Vfft_v2.Isa.of_name ...` *)
Vfft_v2.Dft.target_vec_regs := isa.vec_regs;
```

Reason for a ref rather than threading a parameter through
`pick_algorithm`: the function has 7 call sites including recursive
descent through the DFT-construction code. Threading would mean
visible churn across multiple modules for a single dispatch decision.
The ref is set once at program startup before any DFT construction
begins, so there's no risk of mid-flight value change.

Recipe machinery (n1_blocked, spill_pass1/pass2) adapts automatically
because it queries `pick_algorithm` for the factorization shape. The
(4,16) variant produces pass1=704 tags + pass2=344 tags vs (8,8)'s
544 + 512 = 1056 — different shapes, both bind cleanly through
regalloc.

### What's not done

- **Bench on EPYC.** Sandbox numbers are directional but the absolute
  cycles are not meaningful (sandbox CPU likely throttled, no cpufreq
  data). Re-bench on Zen 3 and Zen 4 to confirm; Zen 4 in particular
  may not show the same gain.
- **MKL comparison setup.** When you bench against MKL on EPYC, set
  `MKL_ENABLE_INSTRUCTIONS=AVX2` (or LD_PRELOAD shim for newer MKL
  that removed `MKL_DEBUG_CPU_TYPE`). Otherwise MKL takes its slow
  AMD-CPUID-dispatch path and the comparison is unfair.
- **AOCL / OpenBLAS reference.** For a fair fight, compare against
  AMD's AOCL or OpenBLAS with `OPENBLAS_CORETYPE=Zen3`/`Zen4` rather
  than relying on MKL alone.


## Lessons recorded

1. **Re-derive savings before repeating them across sessions.** When
   a compacted summary asserts "X saves N ops", that's a hypothesis
   to verify this turn, not a fact to cite. A revert-and-rebuild
   experiment that takes 2-3 tool calls beats hours of building a
   wrong narrative on top of an inherited claim.

2. **A coincident build refresh can look like a fix.** If op counts
   change in the same cycle as a code edit, revert the edit and
   rebuild cleanly before crediting it. Multiple cycles' worth of
   work can hide in a stale binary.

3. **Op count and latency are separate objectives.** Phase 1
   (n1_blocked) changed essentially zero op count but delivered the
   bulk of the latency win. Phases 2-4 changed op count but the
   underlying latency was already captured in Phase 1. Any new pass
   touching op count gets a latency bench before claim of win.

4. **Don't conflate the last segment of a multi-segment session with
   the whole session.** Today had ~10 sub-sessions across 7 hours;
   reading only the last one (the round_13 endgame) and inheriting
   the wrong "Base" baseline produced the false picture in my earlier
   draft.

5. **Spill count is a proxy, latency is the truth.** The AVX2 R=64
   sweep predicted −9% on spill count alone; reality on a real bench
   was −2.4%. Direction was right, magnitude was off by ~2×. Always
   bench before shipping a perf claim — even when the prediction
   looks robust on first-order analysis. Modern OOO cores hide a lot.


## References

- Doc 28 — original `fma_lift` gating decision
- Doc 56 — strided-batch 2D codelets, selective-pinning measurement
- Doc 58 — NFUSE-equivalent n1 blocking (the real one)
- FFTW codelet references: `/tmp/fftw-3.3.10/dft/scalar/codelets/`
- gen_radix*.py uploads under `/mnt/user-data/uploads/`
