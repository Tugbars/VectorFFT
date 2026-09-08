# What the sub-2048 work transfers to the N ≥ 2048 cascade

**Status:** assessment. Each item is marked with what is actually measured and
what is not. Nothing here is in `src/`.

The sub-2048 campaign built and measured a run-contiguous executor and a stack
of optimisations on top of it. Some of that stack is size-specific and dies
above 2048; some is a property of the kernels, which are shared, and therefore
transfers by construction. This separates them.

---

## Summary

| # | item | transfers? | evidence |
|---|---|---|---|
| 1 | **Register-pressure relief on the radix-8 bodies** | **YES — and the ≥2048 kernels are worse** | measured, both sides |
| 2 | **Baked quarter-wave twiddle build** | **YES — create-time, size-independent** | measured |
| 3 | Measurement protocol (pacing, two spreads, controls, crossings) | **YES — method, not code** | measured the hard way |
| 4 | Stage fusion (one function, zero calls) | mechanism yes, **value no** | measured: <0.5% at ≥2048 |
| 5 | Carried twiddle cursor / contiguous twiddle stream | **NO — measured null** | measured |
| 6 | Chain-shape preference (radix-4 ingest) | **NO — size-specific, and contradicted at 2048** | measured both ways |
| 7 | Buffer rule (run in `dst`, no ping-pong) | **NO — blocked by orientation** | structural |
| 8 | Narrow stores / permute-count savings | **NO — refuted twice** | measured twice |

---

## 1. Register-pressure relief on the radix-8 bodies — the real transfer

The sub-2048 probe found its radix-8 stages spilling to the stack while its
radix-4 stages did not. **The shipped cascade kernels have the same defect and
carry it further.** Innermost loop, identical classifier both sides, `spill` =
a memory operand based on `rsp`/`rbp`:

| shipped kernel | insns | **spill ops** | ymm used | shuf |
|---|---|---|---|---|
| `radix8_z_msg_fwd_avx2` | 145 | **6** | 16 | 0 |
| `radix8_z_stf_r4_fwd_avx2` | 182 | **7** | 16 | 32 |
| **`radix8_z_sterm_fwd_avx2`** | **276** | **37** | 16 | 64 |
| `radix4_z_msg_fwd_avx2` | 56 | **0** | 11 | 0 |
| `radix4_z_stf_r4_fwd_avx2` | 74 | **0** | 15 | 16 |

The split is clean and it is the same split the probe showed: **every radix-8
body sits at all 16 ymm and spills; every radix-4 body has headroom and does
not.** `sterm` — the terminator the ≥2048 cascade actually runs — spills 37
times per innermost iteration, against the probe terminator's 10.

These are the kernels that serve N ≥ 2048, so this is not a port: it is the
same code, already shipping, with a measurable defect.

**What is NOT established.** That fixing it pays. On the probe the store-sunk
treatment measured −1.74% inside a fused pipeline at one chain, and **~0% once
the chain moved the radix-8 stage elsewhere**; in isolation it was a pure null
(−0.08% against a −0.04% control). So the mechanism is real and the defect is
real, but the *value* was small and context-dependent at sub-2048. `sterm`'s 37
spills are 3.7× the probe's worst case, so the headroom may be larger — that is
a hypothesis with a clear test, not a result.

**Test before building:** census `sterm2`, `stfn`, and the `_bwd` twins the same
way; then race a sunk `sterm` against the incumbent under the protocol in §3.
Note `sterm`/`sterm2` are already a raced pair on the `t2q` axis, so a third
form slots into machinery that exists.

---

## 2. Baked quarter-wave twiddle build — transfers as-is

The sub-2048 work replaced a per-record `cos`/`sin` table build with a
**quarter-wave table expanded by index shift, octant reflection and sign-bit
XOR** — integer work plus a table load, with no floating-point in the builder
beyond seeding the base table.

Measured: build cost **1.7–4.5 µs against our 6–11 µs**, and the result is
**4–6× closer to the exact twiddle** than the `cos`/`sin` reference (it is not
bit-identical to it — 35–40% of doubles differ — because it is *more* accurate,
not less).

This is create-time work whose mechanism does not depend on N, so it applies to
the ≥2048 cascade unchanged. It costs nothing at execute and improves accuracy,
which makes it the lowest-risk item on this list.

---

## 3. The measurement protocol — transfers, and should

Four protocol facts were learned expensively during the sub-2048 work, and they
apply to every future race in this tree:

1. **Pace between RUNS, not just between arms.** Launching processes
   back-to-back leaves each run starting on a core the previous run heated. A
   sweep without cooldowns manufactured a −2.4% result and a 17 ns variance
   difference that both vanished when 15 s cooldowns were added.
2. **Report TWO spreads.** Within-run (round p90−p10) and **across-run** (the
   spread of the per-run medians). A claim about stability lives on the second
   one; only reporting the first hides placement swings entirely.
3. **Always carry a control arm** — the same code timed twice. Several effects
   chased in this campaign were smaller than their own null. In one sweep the
   control read −3.33% while the effect under test read −3.64%.
4. **Cross the axes; do not A/B them one at a time.** Optimisations here are
   not additive. Store sinking measured −0.08% alone and −1.74% inside the
   fused pipeline — the isolated test would have retired it. Conversely chain
   and kernels together gave −2.87% where the parts summed to −4.34%
   (interaction +1.22%): adding the headline numbers oversells the pair.

---

## 4. Stage fusion — mechanism transfers, value does not

Removing the per-stage call is worth **7.3% at N=128 and under 0.5% at
N ≥ 2048** (see `cascade_stage_fusion.md`). The saving is a fixed ~2 ns per
stage, so it is diluted by any transform long enough to matter at ≥2048.

The mechanism would work there; it simply is not worth its emitter corpus for
a sub-0.5% return. **Do not transfer for speed.** The one reason it might come
back is if the fused form relieves register pressure across stage boundaries
(item 1) in a way the per-stage form cannot — that is untested and would need
to be measured as a register-pressure change, not as a call-count change.

---

## 5–8. What does not transfer, and why

**5. Carried twiddle cursor / one contiguous twiddle stream.** Built,
statically verified (twiddle-base reads per fused function went K−1 → 1), and
**timing-null** at every sub-2048 cell — ±0.5%, never significant. The
mechanism removes 1–4 instructions from a 324–629-instruction function. There
is no reason to expect a different answer at larger N, where the same
instructions are amortised over more work.

**6. Chain-shape preference.** Sub-2048 found radix-4-ingest chains winning
consistently — at N=512 every r4-first chain beat every r8-first one, and the
reference implementation's own chain was the worst of the five. This does **not** transfer: an earlier
measurement has r8-first winning at 1024/2048, and above 2048 the chain is
already a searched axis in the offline planner rather than a fixed preference.
Carry the *method* (search it) not the *answer*.

**7. The buffer rule.** The sub-2048 arrangement can run the whole pipeline in
the destination when it is distinct and aligned, leaving scratch untouched — no
ping-pong, one buffer instead of two. This is blocked at ≥2048 by orientation:
a decimation-in-frequency terminator cannot write into the plane it is reading.
Transferring it means changing the engine's orientation, which is not an
optimisation but a different engine.

**8. Narrow stores / permute-count savings on the edges.** Refuted twice, by
independent measurements. The unordered-lane edge (skipping the two
`permute4x64` per leg) was built as twins and raced: bit-identical, **no gain**,
because the terminator sweep is not port-bound. Narrow stores alone measured
*worse* (spill ops 10 → 12, ~0% median), and deleting all 16 p5-only `vpermpd`
moved nothing — the loop runs at 85% of its ALU roof. **Do not re-propose
either.** The relevant lever in that neighbourhood is live-range shortening
(item 1), which is a different mechanism and is judged on its own measurement.
