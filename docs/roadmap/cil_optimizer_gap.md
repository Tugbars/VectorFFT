# The cil optimizer gap — `codelet_cil.ml` never enters the shared pipeline

**Status:** structural diagnosis stands; the PERFORMANCE thesis below was
partly REFUTED the same day — read §0 first. 2026-08-06.
**One line:** the pure-IL emitter builds its bodies directly instead of through
`Dft.dft_expand*`, so it runs **none** of the DAG optimization sequence —
no CSE, no factoring, no FMA lifting — while both sibling emitters do.

---

## 0. 🔴 CORRECTION (2026-08-06, same day) — read before acting

The N=512 RE campaign (`docs/research/mkl512_gap_campaign/CONCLUSIONS.md`)
measured the op mix directly and **refuted the FMA framing** this document
was originally built on. Per-transform vector-ALU classes at N=512:

| class | ports | ours | MKL | delta |
|---|---|---|---|---|
| FMA + MUL | {p0,p1} | 1264 | 1344 | **−80 — we are 6% BELOW MKL** |
| FADD/SUB/ADDSUB | {p1,p5} | 2144 | 1776 | +368 |
| **in-lane SHUFFLE** | {p1,p5} | **912** | **472** | **+440 — the largest class** |
| LOGIC (`vxorpd`) | {p0,p1,p5} | 440 | 232 | +208 |
| **{p1,p5} cluster** | | **3312** | **2504** | **+808 = 90% of the 79 ns deficit** |

**What survives:** the structural finding (§1–§2) — cil is genuinely off the
pipeline, uniquely among the three emitters. That is a fact about the code.

**What died:** "our FMA density is the problem." We already issue *fewer*
FMA+MUL ops than MKL; the campaign lists *"Our FMA scheme is expensive"* as
**sign-reversed**. So `fma_lift` is **not** the lever, and §5's original
ordering advice (CSE first because it feeds FMA) had the wrong destination
even though the ordering happened to be right.

**Where the excess actually is:** almost entirely the **twiddled mid**. Our
untwiddled leaf issues 1984 vALU ops and MKL's untwiddled pass issues
**1984 — an exact tie**. 100% of the +936 excess is one stage, and the
biggest single class in it is **in-lane shuffles, nearly 2× MKL's**.

**Revised lever ranking** — the passes that remove *adds* and *shared
sub-expressions* (`dedup_sub_pairs`, `share_subsums`, `collect_m`,
`deep_collect`) target the +368 FADD term; nothing in the pipeline obviously
removes the +440 shuffle term, which prior work
(`il_register_pressure.md`) called the AVX2 floor for interleaved-in/out
compute — yet MKL does it with half as many, so that claim needs re-testing.
**The shuffle question is now the highest-value open item, not FMA.**

⚠ **The campaign's own port-floor model is self-flagged as unreliable for
ranking candidates** (its §7 D9): it mispredicted two form races — one 8×
low, one backwards — while raw instruction count predicted both to 1–2%.
Use the class table above as evidence of *where* the excess is, not as a
predictor of what a given change will buy. VTune counters have not been run.

---

## 1. The finding

Three emitters produce our codelets. Only one is off the optimization pipeline.

| emitter | family | dir | lines | `Algsimp` | `fma_lift` | `Dft.dft_expand*` |
|---|---|---|---|---|---|---|
| `codelet_cil.ml` | **pure IL** (Bailey `n1t`/`t2`/`t2b48`/`n1tb44`) | `zil/avx2/pure_il/` | 1690 | 5 † | 1 † | **0** |
| `codelet_zsplit.ml` | boundary split (cascade `s0t_r4`/`msg`/`stf`) | `zil/avx2/boundary_split/` | 1722 | 35 | 6 | 5 |
| `codelet_oop.ml` | split OOP (`n1_oop`/`t1_oop`) | `codelets/oop/` | 2484 | 50 | 8 | 7 |

† **cil's counts are not real usage.** Its single `fma_lift` mention is a
*comment* (line 38) describing what the pass does; its `Algsimp` mention
(line 86) is a comment comparing cil's own global tag counter to
`Algsimp.reset`. cil never references the `Algsimp.t` type, never calls a
pass, and never calls `Dft.dft_expand*`.

`codelet_zsplit.ml` by contrast builds through `Dft.dft_expand_twiddled`
(line 603), threads `Algsimp.t` nodes throughout (lines 340–363), and reads
`VFFT_FORCE_FMA_LIFT` / `VFFT_DISABLE_FMA_LIFT` (579–584) into the shared
pipeline call — the same wiring `codelet_oop.ml` has at 1217–1234.

🔴 **The `zil/` directory name is historical.** Nothing in it is zil-emitted.
`pure_il/` is cil, `boundary_split/` is zsplit. Only the provenance header
at the top of a `.c` identifies the emitter — never the path.

---

## 2. What cil skips, in order

From `pipeline.ml`. The CSE/factoring half runs **first** and is what creates
the forms the FMA half can absorb.

```
      ── op-COUNT reduction (CSE / factoring) ──
149   dedup_sub_pairs           collapse duplicated sub-expressions
160   factor_common_muls        pull shared multiplies out
161   factor_by_atom            factor by common atom
165   dedup_sub_pairs           again, post-factoring
172   collect_m                 collect M-terms
183   deep_collect      ×loop   iterated deep collection
184   collect_m                 again, to fixpoint
201   share_subsums             share common partial sums
      ── op-COST reduction (FMA absorption) ──
246   fma_lift                  Add/Sub with single-use Mul -> FMA
262   factor_const_muls         creates Mul(K, sum) nodes
263   multi_use_fma_lift  ┐
264   fma_addend_factor   │
265   multi_use_fma_lift  │     4x mfl interleaved with 3x faf
266   fma_addend_factor   │
267   multi_use_fma_lift  │
268   fma_addend_factor   │
269   multi_use_fma_lift  ┘
270   flatten_fma_mul_addend
```

**The two halves are not independent.** `fma_passes.ml:1126` states that
`multi_use_fma_lift` exists to absorb "the factored `Mul(K, sum)` node"
produced by `factor_const_muls`. Without the upstream factoring there is
materially less for the FMA passes to eat — so adding FMA lifting alone
would under-deliver.

---

## 3. The measured consequence

### 3a. Instruction mix in the two shipped N=512 kernels

Counted from our own objdumps of the shipped PE objects:

| kernel | FMA | add/sub | standalone `vmulpd` | FMA share |
|---|---|---|---|---|
| `radix16_z_n1t` (leaf, untwiddled) | 32 | 116 | 8 | 20.5% |
| `radix32_z_t2b48` (mid, twiddled) | 67 | 152 | **51** | 24.8% |

The mid carries 51 standalone multiplies beside 152 add/subs. The leaf's low
FMA share is partly structural — an untwiddled butterfly is genuinely
mostly adds — so **the mid is where the recoverable mass sits**.

### 3b. The tier correlation (measured, front door, vs MKL)

| tier | emitter | on the pipeline? | ratio vs MKL |
|---|---|---|---|
| Bailey 256 / 512 / 1024 | `codelet_cil.ml` | **no** | 0.85 / **0.78** / 0.92 — we **lose** |
| cascade 2048 / 8192 / 16384 | `codelet_zsplit.ml` | **yes** | 1.09–1.16 / 1.00–1.03 / 1.02–1.03 — we **win** |

⚠ **This correlation confounds emitter with algorithm and N** — Bailey and
cascade differ in more than their emitter, so it cannot stand alone. It is
listed because it agrees with the independent instruction-mix evidence
above, not as proof.

### 3c. Where the 512 deficit lives

The N=512 RE campaign (`docs/research/mkl512_gap_campaign/`, gitignored —
its MKL-side numbers are disassembly-derived and stay there) found that on
the current (32,16) + blocked-`t2b48` plan our **static instruction count and
spill share are now at or better than MKL's**, while the measured gap is
still 1.28×, and that the residual tracks **vector-ALU issue-port pressure**
(~1.199×, ≈94% of the gap).

⚠ **Pre-verification.** Those figures come from the campaign's lens phase;
its adversarial verifiers had not reported when this document was written.
Treat 1.199× as the motivating hypothesis, not a settled number.

If it holds, the reading is: we are **not** losing on instruction count or
register spill any more — we are losing on how many vector-ALU ops per point
we issue. That is precisely what CSE (fewer ops) and FMA absorption (two ops
→ one) attack.

---

## 4. Why this is newly worth attempting

`fma_lift` has burned this project before. `gen_main.ml:1029` records it
being gated to primes only after an R=8 regression, later re-enabled
everywhere except `Split_radix`. `il_register_pressure.md` names the
mechanism: *"fusing changes value lifetimes, and lifetime pressure, not op
count, is what these bodies are dying of."* An FMA holds three live inputs
where a mul-then-add holds two at a time.

**That constraint was removed on 2026-08-06 by unrelated work.** The blocked
R≥32 kernels became the shipped default; blocking cut spill traffic ~69%,
and the campaign census puts us at **9.08%** pipeline-weighted ymm spill
against MKL's 13.1%. There is register headroom now that did not exist when
the trade was last evaluated.

Two further points in our favour:

- **The interaction is pre-designed.** `fma_passes.ml:198–203` exists
  specifically so `fma_lift` can coexist with the SU+spill recipe: spill
  markers reference tags from *before* the lift, and frozen nodes are
  returned identity-unchanged. Our blocked kernels are built on exactly that
  spill-marker machinery.
- **The CSE half carries no lifetime risk.** Collapsing a duplicated
  subexpression removes a computation; it does not extend a live range the
  way FMA absorption does. The historical regression was an *fma_lift*
  problem, not a CSE problem.

---

## 5. Scope of the fix

**Not** "add `fma_lift` to cil." Two reasons:

1. `algsimp.ml` exposes only five public symbols (`lift_spill_markers`,
   stats/print helpers); the passes `pipeline.ml` calls arrive through a
   facade re-exporting `fma_passes.ml`. There is no one-line hook.
2. cil has **zero** `Dft.dft_expand*` calls — it constructs bodies directly.
   There is no DAG for the passes to act on.

So the real work is **finishing the port**: get cil to emit a DAG the shared
pipeline can consume, the way `codelet_zsplit.ml` does via
`Dft.dft_expand_twiddled`, then run `Pipeline.run` over it.

This reframes a standing note. The record says cil is "zil re-hosted, ported
for REACH, explicitly NOT for i9 speed." That was an accurate description of
the port's *goal* — and the sub-2048 deficit may simply be its **unpaid
remainder**: zsplit was carried onto the shared machinery, cil was not.

### Suggested order

1. **CSE/factoring first** (`dedup_sub_pairs` … `share_subsums`). Lower
   risk, no lifetime hazard, and it creates the forms the FMA passes need.
2. **FMA half second**, behind `VFFT_FORCE_FMA_LIFT` so it is raceable
   rather than mandatory.
3. **Race, do not assume.** Emit one `t2b48` variant through the pipeline
   and race it against the shipped kernel, same discipline as the R16 leaf
   race (alternating arms, pinned core, control arm). Gate on **tolerance,
   not bit-identity** — reassociation changes the arithmetic.

### Falsification criteria

| observation | reading |
|---|---|
| FMA share ↑, vector-ALU ops/pt → 8.2, spill stays < 13% | thesis holds, finish the port |
| ops/pt ↓ but time flat | port-pressure hypothesis was wrong — go to VTune before more emitter work |
| spill rebounds past 13% | the R=8 regression rediscovered on a new body — stop, the headroom was illusory |

🔴 **Race before building.** This codebase is 3-for-3 on static
stack-traffic improvements failing to predict time (cursors, reord,
blocked-leaf). Every candidate here should be proven on a proxy first.

---

## 6. Open, related

- **R=64 blocked kernels do not exist.** R=32 and R=16 are both covered as
  of 2026-08-06; R=64 is the last structural gap in the blocked family.
- **3-stage chains beat every 2-stage pair at 512 by +7.6–8.8%**, measured
  twice, never banked. A plan-level lever independent of this one.
- **VTune** is the instrument that settles §3c directly: it measures port
  distribution rather than deriving it. The highest-value experiment is to
  profile 256/512/1024 and find the counter whose shape tracks the
  non-monotonic 0.85 / 0.78 / 0.92 curve — any mechanism that would hurt
  1024 equally is suspect.
