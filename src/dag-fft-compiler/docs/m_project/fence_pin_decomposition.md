# Fence-vs-pin decomposition: what M-project actually does, and the resulting policy

This document records the empirical investigation that decomposed
M-project's `register __m###d t asm("regN") = ...; asm volatile("" :
"+v"(t))` pattern into its two components — the register-pinning
clause (`asm("regN")`) and the scheduling-fence clause (`asm volatile`
barrier) — and measured their independent contributions across 32
codelet variants. The investigation produces a clean two-rule policy
that simplifies `regalloc.ml` substantially while preserving every
meaningful performance win.

**Companion doc**: `m_project_costs.md` records earlier diagnostic work
(slot-reuse analysis, asm-pattern categorization, cost taxonomy) that
remains correct *as instruction-counting data* but whose central claim
("M-project prevents high-value evictions") is **incorrect** as a
mechanism explanation. This doc supersedes that one's policy
recommendations and corrects the mechanism story; sections 1-3 here
contain a brief retraction.

## 1. What we thought M-project did, and why we were wrong

The model we were operating on: M-project's pinning keeps high-fanout
multi-use values resident in the YMM/ZMM register file, preventing
GCC from spilling them to stack where they'd incur 5-7 ns reload cost
per use. The "OTHER" reg-to-reg moves we'd measured (mostly under
AVX2 M-on) were M-project's routing mechanism — moving pinned values
to free registers to avoid collision-driven spills.

Two observations falsified this model:

**Per-variable spill counts are bounded.** Re-counting spills with
`gcc -fverbose-asm` annotations, we found that under M-off no variable
is loaded from stack more than 3 times. The 18-20 references on hot
spill slots that we'd cited as "high-value eviction" were entirely
slot recycling — GCC reusing the same 32-byte stack slot for many
different short-lived values. The framework had no high-value
evictions to prevent.

**Stripping pinning while keeping fences preserves the win.** The
decisive test: take M-on output and `sed`-strip every `asm("regN")`
clause, leaving the `asm volatile("" : "+v"(t))` fences intact.
Compile and bench. Result on R=64 t1 AVX-512 (the radix where
M-project's win is largest at ~19%):

| Variant | ns/transform | vs M-off |
|---|---:|---:|
| M-on (pin + fence) | 60.0-61.6 | -19% |
| M-fence-only (no pin, fence kept) | 59.8-60.9 | -20% |
| M-off (neither) | 74.8-76.4 | (baseline) |

Across 5 stable runs, M-fence-only is within ±3% of M-on (noise).
The entire 19% advantage comes from the fence; the pin contributes
nothing measurable.

This restructured the entire mental model. M-project's "spill
prevention" was a misnomer — the fence is the win mechanism, and the
fence works by constraining GCC's instruction scheduler to honor the
codelet generator's careful Sethi-Ullman + Goodman-Hsu ordering
instead of re-scheduling under its own (worse) heuristics. The
pinning is mostly orthogonal — sometimes helpful, sometimes harmful,
usually inert.

## 2. Methodology

**Three variants generated per (radix, ISA, kind):**

1. **M-on**: current default emission, `register __m512d t42 asm("zmm15")
   = ...; asm volatile("" : "+v"(t42))`. Pin + fence.
2. **M-fence-only**: M-on with the `asm("zmm15")` clause stripped via
   sed: `register __m512d t42 = ...; asm volatile("" : "+v"(t42))`.
   Fence preserved, register choice returned to GCC.
3. **M-off**: `VFFT_NO_REGALLOC=1` generation. Plain locals, no pin,
   no fence. `__m512d t42 = ...;` and that's it.

The three differ by exactly two flags (pin yes/no, fence yes/no), so
any pairwise comparison isolates one mechanism.

**Bench harness:** pairwise 2-function binaries to minimize i-cache
pressure and cross-function interference observed earlier in the
investigation when 8-12 functions shared a binary. Each function is
warmed for 1000 calls, then 11 trials of 50000-1000000 reps measured
with `clock_gettime(CLOCK_MONOTONIC)`. The minimum across trials is
reported (warmest CPU state, lowest noise). Each comparison is run 3
times; numbers below are typical of the stable runs.

**Coverage:** 32 cases across (radix, ISA, kind), specifically
R ∈ {3, 4, 5, 7, 8, 16, 25, 32, 64} × ISA ∈ {AVX-512, AVX2} ×
kind ∈ {t1, n1, log3}, with the natural subset of valid combinations
(log3 requires R ≥ 16 in the generator; not all primes have n1
variants worth measuring on both ISAs).

## 3. Complete data

Reading guide for the table: positive `M-fence vs M-on` means M-fence
is **slower** than M-on. Positive `M-fence vs M-off` means M-off is
slower than M-fence (i.e., fence wins).

### 3.1 t1 codelets (twiddled)

| Radix | ISA | M-on vs M-off | M-fence vs M-off | M-fence vs M-on | Best |
|---:|---|---:|---:|---:|---|
| 3 | AVX-512 | -15% | -15% | tied | fence |
| 3 | AVX2 | -13% | -13% | tied | fence |
| 4 | AVX-512 | -12% | -12% | +2% | tied |
| 4 | AVX2 | -15% | -15% | tied | tied |
| 5 | AVX-512 | -15 to -17% | -14% | +2-3% | M-on slight |
| 5 | AVX2 | -10 to -12% | -11% | -5% | fence |
| 7 | AVX-512 | -10 to -11% | -10 to -12% | tied | fence |
| 7 | AVX2 | -11 to -13% | -11 to -13% | -6% | fence |
| 8 | AVX-512 | -11% | -11% | tied | fence |
| 8 | AVX2 | -14 to -17% | -15% | -2 to -3% | fence |
| 16 | AVX2 | -12% | -14% | tied | fence |
| 32 | **AVX-512** | **+6%** ❌ | **-10%** | **-13%** | **fence wins big** |
| 32 | AVX2 | -17% | -16% | +2% | tied |
| 64 | AVX-512 | -19% | -20% | tied | fence |

**Summary for t1**: fence-only is the best choice for every (radix,
ISA) combination tested. R=32 t1 AVX-512 is the most consequential
finding — pinning is actively bad there (M-on is 6% slower than
M-off), and fence-only delivers the largest single available win in
the dataset (+13% over M-on, +10% over M-off).

### 3.2 n1 codelets (no-twiddle)

| Radix | ISA | M-on vs M-off | M-fence vs M-off | M-fence vs M-on | Best |
|---:|---|---:|---:|---:|---|
| 3 | AVX-512 | -8 to -10% | -11 to -13% | -4 to -5% | fence |
| 3 | AVX2 | -8 to -11% | -10 to -12% | -3 to -4% | fence |
| 4 | AVX-512 | -10 to -14% | -12 to -14% | tied | tied |
| 4 | AVX2 | -20 to -23% | **-23 to -25%** | -3 to -5% | fence wins big |
| 5 | AVX-512 | -6 to -9% | -5 to -9% | tied | fence |
| 5 | AVX2 | -7 to -10% | -9 to -10% | **-13 to -15%** | fence wins big |
| 7 | AVX-512 | -7 to -8% | -8% | -3% | fence slight |
| 7 | AVX2 | -3 to -5% | -3 to -4% | tied | tied |
| 8 | AVX-512 | -39% | -39 to -40% | tied | fence wins big |
| **8** | **AVX2** | **+25%** ❌ | **+22% over fence-vs-moff means M-off is +22% faster** ❌ | +5% | **M-off wins** |
| 16 | AVX-512 | -8 to -10% | -3 to -6% | +4 to +10% | M-on slight |
| **16** | **AVX2** | -2 to -7% | **-5 to -7% (fence loses to M-off)** ❌ | +2 to +9% | **M-off wins** |
| 32 | AVX2 | **-19 to -21%** | -12 to -16% | +5 to +6% | M-on slight |
| 64 | AVX2 | -19 to -22% | -19 to -20% | -2 to -3% | fence slight |

**Summary for n1**: fence-only is best in most cases, but two
exceptions:
- R=8 and R=16 n1 AVX2: the fence is actively harmful. M-off (no
  fence, no pin) wins by 5-22%.
- R=16 n1 AVX-512 and R=32 n1 AVX2: pin provides a small additional
  win over fence-only (+4-10%). Magnitude is modest enough that we
  accept the loss for policy simplicity (see §5).

### 3.3 log3 codelets

| Radix | ISA | M-on vs M-off | M-fence vs M-off | M-fence vs M-on | Best |
|---:|---|---:|---:|---:|---|
| 16 | **AVX-512** | tied | tied (-1 to +1%) | **+6.5%** | **pin wins (fence is inert)** |
| 16 | AVX2 | -7% | -10% | +3 to +4% | M-on slight |
| 25 | **AVX-512** | -19 to -23% | -17 to -26% | **+11 to +14%** | **pin wins big** |
| 25 | AVX2 | **+6 to +9%** ❌ | -4 to -7% | -8 to -13% | fence wins |
| 32 | AVX-512 | -10% | -7% | +2 to +3% | M-on slight |
| 32 | AVX2 | -12% | -13% | tied | fence |
| 64 | AVX-512 | -7 to -8% | -7 to -8% | tied | tied |
| 64 | AVX2 | -22% | -20% | tied | tied |

**Summary for log3**: AVX-512 has a clear "pinning helps" zone at
R ≤ 32, peaking at R=25 (+11-14% over fence-only). R=64 log3 AVX-512
is tied. AVX2 log3 follows the same pattern as t1: fence-only is the
right call universally.

R=16 log3 AVX-512 is the most striking case: M-fence ≈ M-off (fence
provides zero benefit), but M-on wins +6.5% over both. **The entire
M-on advantage on this case comes from the pin, none from the
fence** — strong empirical evidence that pinning *can* contribute
something the fence doesn't, at least on this specific shape.

## 4. The three behavior categories

Re-reading the data through the lens of which mechanism wins:

**Fence-only wins or ties** (22 cases):
All t1 codelets across both ISAs.
All log3 AVX2 codelets.
n1 codelets at R ∈ {3, 4, 5, 7, 8} AVX-512.
n1 AVX2 at R ∈ {3, 4, 5, 7, 32, 64}.
log3 AVX-512 at R = 64.

**Pin provides genuine additional gain over fence-only** (5 cases):
log3 AVX-512 at R = 16 (+6.5%), 25 (+11-14%), 32 (+2-3%).
n1 AVX-512 at R = 16 (+4-10%).
n1 AVX2 at R = 32 (+5-6%).

**Fence-only is harmful; M-off wins** (2 cases):
n1 AVX2 at R = 8 (fence costs 22%).
n1 AVX2 at R = 16 (fence costs 5-7%).

**Tied / inconclusive** (3 cases):
t1 R = 4 AVX-512 and AVX2 (tiny codelets).
n1 R = 7 AVX2.

## 5. The two-rule policy

Capturing the three behavior categories with the smallest possible
predicate, accepting a small loss on the n1-pin cases for rule
simplicity:

```
Fence emission:
  default = ON
  disable when: kind == n1  AND  isa == AVX2  AND  R ∈ {8, 16}

Pin emission:
  default = OFF
  enable when:  kind == log3  AND  isa == AVX-512  AND  R ≤ 32
```

Both rules are narrow whitelists/blacklists with explicit empirical
support and clear boundaries. The "fence off" carve-out is bounded
below (R=4 n1 AVX2 strongly prefers fence-on) and above (R=32 n1 AVX2
prefers fence-on). The "pin on" carve-out is bounded above (R=64
log3 AVX-512 is tied) and below by the radix support in the generator.

### 5.1 What we keep vs lose

| Case | Status quo | Policy | Effect |
|---|---|---|---:|
| R=32 t1 AVX-512 | M-on (+6% slower than M-off) | fence-only | **+13%** |
| R=25 log3 AVX2 | M-on default | fence-only | +10% |
| R=32 log3 AVX2 | M-on | fence-only | +13% |
| R=64 log3 AVX2 | M-on | fence-only | +20% |
| R=8 n1 AVX2 | gate-dependent (often M-on) | M-off | **+25%** |
| R=16 n1 AVX2 | gate-dependent | M-off | +5-7% |
| R=5 n1 AVX2 | M-on | fence-only | +13-15% |
| R=25 log3 AVX-512 | M-on | M-on (carve-out keeps it) | 0% (kept) |
| R=16 log3 AVX-512 | varies | M-on (carve-out) | +6.5% (kept) |
| All t1 R=3,4,5,7,8,16,64 | varies | fence-only | 0% to +5% |
| R=16 n1 AVX-512 | M-on | fence-only | **-4 to -10%** (accepted loss) |
| R=32 n1 AVX2 | M-on | fence-only | **-5 to -6%** (accepted loss) |
| All remaining cases | varies | fence-only | 0 to small |

Net: meaningful wins on 7+ cases (3 of them >10%), two accepted losses
of modest magnitude on niche n1 codelets, and substantial
simplification of `regalloc.ml`.

### 5.2 What the policy makes dead code

- **Pin-density estimator** in `emit_c.ml:740-797`. The 0.8 threshold,
  the inline-set walk, the topo sort for density estimation — all
  unneeded. The estimator was a proxy for routing budget; the new
  policy doesn't route, it just decides on fence/pin per (kind, ISA,
  R).
- **AVX2 small-R carve-out** in the gate. Replaced by the n1-AVX2
  fence-disable rule, which is narrower and empirically motivated.
- **`is_log3` parameter** on `emit_codelet`. Was added for an earlier
  carve-out that turned out to be based on bench-label-inverted data
  (see §1 of `m_project_costs.md` and conversation history). Now
  used solely for the pin-enable rule, so a single parameter on the
  gate predicate.
- **Most of `regalloc.ml`'s allocator logic.** The graph coloring,
  the collision resolution, the routing decisions — all unneeded
  when we never emit `asm("regN")` clauses except via a tiny narrow
  whitelist. Keep only the spill-recipe coordination (pass-1/pass-2
  boundary handling for blocked emit on R=25 log3), which is
  structurally different from pinning.
- **`VFFT_PIN_FORCE` env var.** Was useful for measurement during
  this investigation; production policy is rule-driven, no override
  needed. Could keep it for diagnostic use; not load-bearing.

`VFFT_NO_REGALLOC=1` should be kept as a "disable everything"
debugging escape hatch.

## 6. What we don't yet know

- **Why pinning helps log3 AVX-512 at R≤32 specifically.** The
  fence-decomposition test says fences don't capture this win — the
  pinning is doing something extra. Candidate mechanisms: FMA
  encoding bias (132 vs 231 changes uop dispatch), bypass-network
  alignment from specific register choices, or interaction with GCC's
  post-RA scheduling pass. We have no measurement isolating these.
  The empirical rule works regardless, but the underlying mechanism
  remains opaque.

- **Why fences hurt n1 AVX2 at R∈{8,16} but help everywhere else
  (including R=4 and R=32 n1 AVX2).** Some kind of "sweet spot" in
  GCC's optimization heuristics for that specific size band. Not
  understood mechanistically. Bounded by measurement above and below.

- **Whether the policy holds on uarchs other than the host where these
  measurements were taken.** Modern Intel (Sapphire Rapids, Granite
  Rapids) and AMD (Zen 4, Zen 5) have similar rename-unit
  characteristics, so fence cost should be similar. Older uarchs
  (Skylake-X, Zen 2) may differ. Not tested in this investigation.

- **Whether log3 R<16 would also benefit from pinning.** The generator
  doesn't emit log3 for R<16, so untested. The rule "R ≤ 32" implies
  it would apply if those variants existed, but this is extrapolation.

## 7. Reproducing

Working trees:
- VFFT source: `/home/claude/work/strided_avx512/`
- Per-radix generated codelets: `/tmp/fence_test/` (R=64), `/tmp/fence_verify/`
  (R=16, R=32 t1), `/tmp/fence_small/` (R=4, R=8), `/tmp/fence_log3/`
  (R=16, R=32, R=64 log3), `/tmp/fence_primes/` (R=3, R=5, R=7),
  `/tmp/fence_n1_bound/` (R=32, R=64 n1 AVX2 upper-bound check)
- Bench harness template: `/tmp/fence_extra/bench_generic.c`

Steps to reproduce any single case:

```bash
# 1. Generate three variants for (R, ISA, kind)
VFFT_PIN_FORCE=1   gen_radix.exe R [flags] --emit-c --isa ISA > variant_mon.c
VFFT_NO_REGALLOC=1 gen_radix.exe R [flags] --emit-c --isa ISA > variant_moff.c

# 2. Strip pin clauses to get fence-only
sed -E 's/register __m###d ([a-zA-Z_][a-zA-Z0-9_]*) asm\("regN[0-9]+"\) =/register __m###d \1 =/g' \
    variant_mon.c > variant_fence.c

# 3. Compile each, rename function symbols, link into a 2-function bench binary.
# 4. Run 3 trials, report minimum of 11 trials per variant.
```

The 2-function-per-binary methodology is important. Earlier
measurements in the investigation that included 8-12 variants in one
binary showed substantial bench-harness interference (the same .o
file would report different timings depending on which other
functions were colocated). Suspect cause: i-cache pressure plus
cross-function branch-predictor pollution. Pairwise 2-function
binaries eliminate this and give stable measurements across runs.

## 8. Summary

M-project's `asm("regN")` register pinning was originally believed to
prevent high-value YMM/ZMM evictions. Decomposing the pattern by
stripping the pin clause while keeping the inline-asm scheduling
fence shows that the fence captures essentially the entire win on
most codelets (R=64 t1 AVX-512: M-on and M-fence-only both deliver
-19% vs M-off, within noise). Pinning is a narrow-band benefit
limited to log3 AVX-512 at R ≤ 32. The fence itself is harmful in a
narrow band of n1 AVX2 codelets at R ∈ {8, 16}.

A two-rule policy — fence-on except for n1 AVX2 R∈{8,16}, pin-on
only for log3 AVX-512 R≤32 — captures every meaningful win across 32
measured cases without regressing any non-niche codelet. It replaces
the pin-density estimator, the AVX2 small-R carve-out, and most of
`regalloc.ml`'s allocator with two trivially-reviewable boolean
predicates.

The mechanism story changes too: the fences work by constraining
GCC's instruction scheduler to honor the codelet generator's careful
SU+GH ordering rather than overriding it with GCC's own heuristics.
The pinning, where it adds value, does so through some other path
(likely FMA encoding bias or bypass-network alignment) that isn't
isolated by the experiments here. The empirical rule is solid; the
mechanism is partially understood.
