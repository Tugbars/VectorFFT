# Cascade stage fusion — removing the per-stage call

**Status:** design. The mechanism is built and gated in a probe; nothing here is
in `src/` yet.

**Scope:** K=1, 1D, C2C, interleaved, sub-2048. The saving is a fixed cost per
stage, so it matters where the transform is short and fades where it is long:
7.3% at N=128, under 0.5% at N ≥ 2048. The exclusion above 2048 is because the
GAIN vanishes there, not because of corpus cost — see the decision below.

---

## 0. The decision

**Emit the full cross-product — every (N, chain, buffer mode, direction) —
not just the chains that currently win.**

Two reasons, and neither is a judgement call:

1. **Emitting only the winners is self-defeating.** Every banked chain verdict
   was measured on the unfused executor (§4.4). Emit a fused function only for
   the incumbent and the chain can never be re-raced under fusion — there is
   nothing to race it against — so the verdict stays frozen on the engine
   fusion replaces. Racing chains fused needs a fused function for every
   CANDIDATE.
2. **The instruction cache is nowhere near the limit.** A fused whole-transform
   function measures **1.6–3.7 KB**, and exactly ONE executes per transform,
   against a **32 KB L1i** — 5–12% of it. Measured, not estimated (§4.3).

So corpus size is not a performance argument here. It is a binary-size budget,
against a library that already ships 1190 codelets. The only ceiling that would
ever bite is a SINGLE function outgrowing L1i, which needs roughly ten times
the current stage count.

---

## 1. What we do today

`vfft_zturn2_execute_fwd` (`src/core/oop/zturn.h:1048`) dispatches one call per
stage, and the interior ones are **indirect**, through pointers the plan chose
at create time:

```c
/* ingest */  _vfft_zt_s0t_fwd_pick(p->lanes_u)(zin, 0, p->plane, ...)
/* mids   */  const _vfft_zt_msg_fn f = p->msg_f[s];
              f(p->plane, 0, p->plane, 0, p->twz[s], ...)
/* term   */  _vfft_zt_term8_fwd(p, zout)      /* picks from a 2x2 matrix */
```

So a transform with `nf` stages pays `nf` calls, each through the frozen
11-argument codelet ABI: four register arguments, seven on the stack, and a
prologue/epilogue that saves and restores `xmm6`–`xmm15` plus the eight
callee-saved GPRs, ending in `vzeroupper`.

**No compiler setting removes this.** The mid calls are indirect and the target
is selected from wisdom at plan time, so there is nothing for LTO or
devirtualisation to resolve. The cost is structural and only a structural
change removes it.

---

## 2. What it costs — measured

### 2.1 The ABI, in isolation

Stubs compiled in their own translation unit (so the caller pays a real call),
carrying exactly the shape `objdump` shows for the real stage kernels:

| stub | what it carries | ns/call |
|---|---|---|
| 11-arg codelet ABI | `xmm6`–`xmm15` + 8 GPR saves, reads all 7 stack args, `vzeroupper` | **2.75** |
| GPR-only | GPR saves + arg reads, no xmm, no `vzeroupper` | 1.77 |
| bare `ret` | nothing | 0.88 – 1.00 |

So roughly **1.8 ns per call is pure ABI**, dominated by the `xmm6`–`xmm15`
save/restore pair. A narrower 3-argument ABI measures 1.88 ns and a
7-argument one 2.18 ns, which brackets the same number.

### 2.2 End to end, fused vs unfused

Same kernels, same arithmetic, same chain — the only difference is whether the
stage bodies are separate functions or inlined into one. One cell per process,
21 rounds, alternating arm order, three rotations:

| N | stages | unfused | fused | saved | share |
|---|---|---|---|---|---|
| 128 | 3 | 78.5 ns | 72.8 ns | 5.7 ns | **7.3%** |
| 256 | 3 | 164.1 ns | 159.2 ns | 4.9 ns | 3.0% |
| 512 | 3 | 353.8 ns | 348.8 ns | 5.0 ns | 1.4% |
| 1024 | 4 | 759.8 ns | 746.0 ns | 13.8 ns | 1.8% |

That is **≈1.7–2.0 ns per stage removed**, which agrees with the stub table and
confirms the saving is the ABI and not something else.

Two consequences worth stating plainly:

- At **N=128 this is the largest single execute-side effect measured in the
  sub-2048 work** — larger than the chain choice (2.7% at 512) and larger than
  any kernel-body change.
- It scales with **stage count**, not with N, so it is worth more on longer
  chains. That couples it to the chain axis: a 4-stage chain that loses to a
  3-stage one unfused may win once both are fused. **The two must be raced
  together, never separately** — measuring either alone gets the answer wrong
  in both directions.

---

## 3. The target arrangement

One function per transform. No calls inside it. Stage boundaries become loop
boundaries in a single body, so the register allocator, the scheduler and the
twiddle cursor all see the whole pipeline at once.

Concretely, a fused function for a given (N, chain, buffer mode):

1. **Bodies inlined.** Each stage kind exposes its body as an
   `always_inline` form; the fused function instantiates the ingest, each mid,
   and the terminator in sequence.
2. **Literal trip counts.** The chain is known when the function is emitted, so
   every stage's `L`, group count and column count is a compile-time constant
   rather than a plan field read at run time.
3. **One twiddle cursor.** The per-stage streams are laid out contiguously in
   stage order and walked by a single cursor that is read from the plan once,
   before the first store, and thereafter carried from stage to stage in a
   register. This requires the streams to be one allocation, concatenated in
   stage order — the arrival point of each stage's walk must *be* the next
   stage's base, which is an assertable property of the geometry, not a
   convention.
4. **One buffer where possible.** With a distinct, sufficiently aligned
   destination the whole pipeline can run in the destination and the scratch
   plane goes untouched; otherwise everything runs in the plane and only the
   terminator writes out. No ping-pong either way.

Executing then costs **one** call, not `nf`.

---

## 4. How it enters the tree

### 4.1 The plan side

The plan holds a single function pointer to a whole-transform function,
selected at bind time from a table keyed by (N, chain, buffer mode). Execute
becomes a single indirect call. This is the existing law — bind at plan time,
execute is pure dispatch — with the binding moved up one level, from per-stage
to per-transform.

### 4.2 The emitter side

Today `cascade_z.ml` emits one standalone `.c` per (kind, radix, variant), each
containing an *external* function, compiled into the codelet library. Fusion
needs two things it does not currently produce:

- **An inlinable body form** of each stage kind — the arithmetic without the
  function wrapper, in a form a driver can include.
- **The fused drivers** themselves, one per (N, chain, mode), instantiating
  those bodies with literal sizes.

The structural precedent for a family that must link beside its ordinary form
is the existing per-kind variant stems: a variant gets its own C stem so both
forms coexist in one binary. A fused driver is a new family rather than a new
variant, because it is keyed by (N, chain) rather than by (kind, radix).

### 4.3 Emit every chain, not only the winners

A fused function is per (N, chain, mode, direction). Over sub-2048 pow2 that is
on the order of 36 (N, chain) pairs x 2 buffer modes x 2 directions, so roughly
140 functions.

**Emit all of them.** Two reasons, and the first is decisive.

**1. Emitting only the winners is self-defeating.** The winning chain per cell
was itself decided by a race on the UNFUSED executor (4.4). If only that chain
gets a fused function, the chain race can never be re-run fused -- there would
be nothing to race it against -- so the verdict would be frozen forever on the
engine fusion replaces. Racing chains under fusion requires a fused function
for every CANDIDATE chain, not just the incumbent. A two-phase
"race unfused, emit the winner" scheme bakes in exactly the defect 4.4
identifies.

**2. The I-cache ceiling is not close.** Measured on the probe objects, one
fused whole-transform function is:

`nm -S` over the built object, every fused forward/dest form present:

| fused function | stages | bytes | % of 32 KB L1i |
|---|---|---|---|
| `64_8_8` | 2 | **3712** | 11.3% |
| `512_8_8_8` | 3 | 3632 | 11.1% |
| `1024_8_8_4_4` | 4 | 3360 | 10.3% |
| `256_8_8_4` | 3 | 3008 | 9.2% |
| `2048_4_8_4_4_4` | 5 | 2448 | 7.5% |
| `512_8_4_4_4` | 4 | 2288 | 7.0% |
| `512_4_4_4_8` | 4 | 2224 | 6.8% |
| `512_4_8_4_4` | 4 | 2112 | 6.4% |
| `512_4_4_8_4` | 4 | 2112 | 6.4% |
| `128_8_4_4` | 3 | 1936 | 5.9% |
| `128_4_4_8` | 3 | 1888 | 5.8% |
| `32_8_4` | 2 | **1648** | 5.0% |

**1.6-3.7 KB each against a 32 KB L1i**, and exactly ONE of them executes per
transform, so the hot footprint is a single function at 5-12% of the
instruction cache. A workload sweeping five sizes touches ~15 KB. Size tracks
radix-8 stage count rather than N -- the largest is N=64, not N=1024.

The remaining ~350 KB of the corpus is COLD text: it costs binary size, not
instruction-cache pressure, against a library that already ships 1190
codelets. Binary size is a real but different budget, and it is not a
performance argument.

**The limit, if one is ever hit, is a single function outgrowing L1i** -- which
would need a chain roughly ten times the current stage count. Until a measured
cell approaches that, emit the full cross-product.

## 4.4 The banked verdicts were raced on the executor this replaces

All of the create-time calibrators time the SAME function — the unfused,
one-call-per-stage path (`src/core/planning/cascade_calibrate.h`):

```c
static void _zt_chain_arm(void *v) { ... vfft_zturn2_execute_fwd(c->p, c->zi, c->zd); }
static void _zt_tf_arm(void *v)    { ... vfft_zturn2_execute_fwd(c->p, c->zi, c->zd); }
```

so the **chain** verdict, the **tform/ntform** verdict and the **t2q**
terminator-twin verdict were each measured on an executor that fusion removes.
None of them transfers, and for the chain this is measured, not argued —
the crossing (every chain x {unfused, fused}, one process, one round) reorders
the ranking wherever chains of different stage COUNT compete:

| N | unfused ranking | fused ranking | runs differing |
|---|---|---|---|
| 128 | 4.4.8 > 4.8.4 > 8.4.4 | *same* | 1/4 (all chains are 3-stage — the mechanism cannot bite) |
| 256 | 4.8.8 > 8.8.4 > 8.4.8 > **4.4.4.4** | 4.8.8 > **4.4.4.4** > 8.4.8 > 8.8.4 | 4/4 |
| 512 | 4.8.4.4 > **8.8.8** > 4.4.8.4 | 4.8.4.4 > 4.4.8.4 > **8.8.8** | 3/4 |

The cause is arithmetic: the saving is per stage, so the ABSOLUTE gain tracks
stage count — 4-stage chains collect 12–18 ns, 3-stage chains 7–13 ns. A
3-stage chain that leads unfused can therefore fall behind a 4-stage one once
both are fused. `8.8.8` at N=512 goes from second to last; `4.4.4.4` at N=256
goes from last to second.

(Control was clean at 512 (+0.20%) but large at 256 (+2.85%) and 128 (+5.22%),
an arm-position artifact rather than drift. 512's reordering stands on its own;
256's is consistent with the mechanism and with 4/4 runs, but its control wants
fixing before the number is quoted as final.)

**What this does and does not imply.** It is NOT an extra cost item for
shipping fusion: banked verdicts are disposable during development, and a full
re-race is already the plan once the method stops changing (race once while
shipping, then save). What it does imply is a constraint on HOW that final race
is run:

1. it must be run **on the fused executor**, because a verdict from the unfused
   one answers a different question; and
2. **every candidate chain must exist in fused form** for it to be raced at all
   — which is the decision in §0, arrived at independently.

## 5. Gate

A fused function changes *where* the arithmetic lives, never *what* it is. So
the gate is exact:

- **Bitwise identity** to the unfused form, at every cell, every seed, and both
  buffer modes — `memcmp` over the whole output, not a tolerance.
- The unfused form itself checked against a scalar reference at ≤1e-12.
- **Mutation hooks** that break exactly one element (a trip count, the cursor
  handover, a stage base) and a gate run that fails if the damage lands
  anywhere other than the intended victim — otherwise the identity check is
  passing vacuously.
- **No non-finite output** anywhere, and destination and scratch poisoned
  before each run so a variant that fails to write part of its output is caught
  rather than inheriting the previous arm's correct answer.

The probe implementation passes this gate at 64/64 rows across 16 cells with
six mutation hooks, each hitting only its intended target.

---

## 6. Measuring it

The effect is single-digit percent, so it sits near the noise floor of this
machine and needs the full protocol:

- pin to one core, high priority, single thread;
- alternate arm order per round, ≥21 rounds, pace between arms;
- **a cooldown between runs**, and report **both** spreads — within-run
  (round p90−p10) and across-run (spread of the per-run medians);
- **a control arm**: the same code timed twice. Anything smaller than the
  control delta is not a result. This is not optional — an earlier
  back-to-back sweep without a control manufactured a 2.4% effect and a 17 ns
  variance difference that both vanished under pacing.
