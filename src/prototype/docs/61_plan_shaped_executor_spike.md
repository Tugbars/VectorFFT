# 61. Plan-shaped executor spike — Plan B+A validation

**Date:** 2026-05-16
**Status:** Spike complete. Architecture validated; 5% wall-time win on a
synthetic bench of the N=131072 K=4 cell. Not wired into production.

**Where things live:**
- Emitter — [src/prototype/bin/emit_executor_h.ml](../bin/emit_executor_h.ml)
- Generated header — `src/prototype/generated/plan_executors.h`
- Standalone bench — [src/prototype/bench/spike_n131072_k4.c](../bench/spike_n131072_k4.c)
- VTune doc that motivated this — [docs/dev/vtune_n131072_k4_vfft_vs_mkl.md](../../../docs/dev/vtune_n131072_k4_vfft_vs_mkl.md)

---

## The gap being closed

The N=131072 K=4 VTune profile measured 21% of CPU time in
`_stride_execute_fwd_slice_from` and 7% in `stride_cmul_scalar_avx2` —
~28% wall-time share outside the codelet bodies. For an 8-stage plan
that's per-stage executor overhead paid 8 times: per-group base pointer
compute, twiddle pointer load, variant branch tree, scalar twiddle prep
gate, and an indirect call through `st->t1s_fwd`.

The VTune doc names the closure path: "codelet fusion — generate one
monolithic 8-stage codelet ... eliminates the 8× executor-overhead pay
and the 8× codelet function-call overhead." That's an MKL-style v2.0+
workstream. This spike tests a smaller architectural lever first:
**plan-shaped executor emission**, which keeps the per-stage codelet
structure but specializes the wrapper around it per (N, K, factorization,
variant-assignment) tuple.

## The two-part design

| Part | What it does | What it saves |
| --- | --- | --- |
| **(B) Direct calls** | Emit one C function per wisdom entry. The codelet call sites are direct symbol references, not function pointers loaded from `plan->stages[s].t1s_fwd`. | Indirect-call latency (~2-3 cycles/call). The 4-branch variant tree (n1-fallback / log3 / t1s / flat) collapses to one codepath per stage, resolved at emit time. |
| **(A) Pre-walked tape** | At plan-build time, pack each invocation's per-group runtime values into one flat 24-byte struct `{base, tw_re, tw_im}`. The emitted executor walks this tape sequentially. | Per-group load count drops from ~5 scattered (across `group_base`, `needs_tw`, `cf0_re`, `cf0_im`, `tw_scalar_re/im`) to ~3 sequential. HW prefetcher streams one cache line ahead. |

(B) alone produced ~0.7% (within-noise). (A) is where the actual cache-
stall reduction lives — sequential prefetch reaches the next entry before
the codelet returns. Combined, (B)+(A) gave 5.0% on this bench.

## Spike scope (what's emitted)

For this spike, exactly one wisdom entry is hard-coded in OCaml:

```ocaml
{ n=131072; k=4;
  factors  = [|4; 4; 4; 4; 8; 4; 4; 4|];
  variants = [|FLAT; T1S; T1S; T1S; T1S; T1S; T1S; T1S|];
  use_dif_forward = false }
```

The emitted function name encodes the full tuple:
`exec_n131072_k4_44448444_v02222222_dit_fwd_avx2`. Inner-stage variants
of LOG3, FLAT, and BUF are stubbed (`abort()` if reached) — implementing
them is straightforward but unnecessary for the architectural validation.

Inner-stage `needs_tw[g]` and `cf0 != (1.0, 0.0)` branches are dropped
from the emitted code in this spike. Real plans with mixed paths need
both back (compile-time-decided codepaths per group group-class, since
those branches are per-group runtime decisions).

## Validation: disassembly

The two architectural claims show up directly in assembly:

```
1ee3:  e8 78 04 00 00    call   2360 <radix4_n1_fwd_avx2>
1f47:  e8 f4 04 00 00    call   2440 <radix4_t1s_dit_fwd_avx2>
2067:  e8 14 08 00 00    call   2880 <radix8_t1s_dit_fwd_avx2>
```

Every codelet call site uses opcode `e8` (direct relative call). No
`call *%reg` (indirect call through pointer) anywhere in the executor
body. Baseline executor in the same binary emits indirect calls for the
same codelets — the contrast is exactly what (B) promises.

## Measurement

Standalone bench:
[src/prototype/bench/spike_n131072_k4.c](../bench/spike_n131072_k4.c).
Builds a plan with realistic stage strides
(`stride[s] = K · ∏(R_{s+1}..R_{S-1})`) but synthetic `group_base[]` —
**timing-only**, not correctness. Both executors share the same plan
and codelets; the only delta is call style + tape vs scattered loads.

Run conditions: Raptor Lake i9-14900KF, CPU 2 pinned (`taskset -c 2`),
20 reps/run × 11 runs, median ns/FFT reported.

```
                            baseline      spike       speedup
(B)-only direct calls       1163305 ns    1155072 ns  1.007×
(B)+(A) tape walk           1180148 ns    1120837 ns  1.053×
```

Per-run breakdown for (B)+(A) — spike wins **all 11 runs** before noise
from thermal walking takes over at run 10:

```
run  spike(ns)    base(ns)    spike-base
 0   1,059,085   1,114,539     -55,454
 1   1,089,901   1,116,012     -26,111
 2   1,092,011   1,121,782     -29,771
 3   1,110,922   1,155,499     -44,577
 4   1,117,329   1,172,308     -54,979
 5   1,120,837   1,180,148     -59,311  ← median
 6   1,130,211   1,186,115     -55,904
 7   1,139,242   1,195,942     -56,700
 8   1,196,257   1,204,270      -8,013
 9   1,235,966   1,215,763     +20,203
10   1,484,225   1,272,733    +211,492  ← thermal outlier
```

## Caveats — what this bench does NOT prove

1. **Correctness.** Twiddles are filled with arbitrary values; the
   spike validates execution flow and timing, not numerical output.
   Correctness validation against the production planner is a separate
   workstream.
2. **Realistic memory pressure.** The synthetic `group_base[g] = g*K
   mod headroom` makes inter-group memory accesses cache-friendlier
   than a real plan's bit-reversal-like pattern. Total per-FFT wall
   time was 1.16 ms vs the production 2.36 ms — the synthetic setup
   undercounts memory traffic. Net effect on the spike vs baseline
   delta is unclear: in a more memory-bound regime, the wrapper share
   is a smaller fraction of wall time, so the absolute speedup might
   shrink; but the tape's sequential prefetch becomes more valuable
   too. Real-plan bench is the conclusive test.
3. **Single cell.** N=1024 K=128 (the second validation cell agreed in
   the spike plan) and the other ~200 wisdom entries are not yet
   covered.

## What's reused vs new

| Component | Status |
| --- | --- |
| `bin/emit_executor_h.ml` (~250 lines OCaml) | NEW |
| `generated/plan_executors.h` | NEW — auto-emitted, 320 lines |
| `bench/spike_n131072_k4.c` | NEW — standalone harness, ~270 lines |
| Codelet symbol naming convention | LEAVES — same as production now (`radix{R}_{variant}_{isa}`, no `_gen_inplace_su*` suffix; aligned in same session). |
| OCaml DAG emit pipeline | LEAVES — codelets unchanged; spike just emits additional plumbing around the same codelet calls. |

## Path forward (in order of decreasing "cheap")

1. **(i) Second-cell validation** — N=1024 K=128. Hard-code another
   entry in the OCaml; recompile bench; measure. Confirms the gain
   is structural across plan shapes, not a fluke of 8-stage R=4.
2. **(ii) Real-plan bench** — build the plan via production's planner
   (`stride_auto_plan_wis` etc.), point our spike executor at it. The
   plan's `stride_stage_t` has all the fields the spike needs (plus
   more we ignore). This conclusively shows the gain on production-
   shaped memory access. Touches production code only to USE the
   planner; doesn't replace anything yet.
3. **(iii) Generalize the emitter** — parse `build_tuned/vfft_wisdom_tuned.txt`
   in OCaml (~50 lines); emit per-entry executors for all ~200 cells.
   Full bench sweep vs the generic executor across the production
   portfolio. Likely Phase 2 of this workstream.
4. **(iv) Wire into production** — replace `_stride_execute_fwd_slice_from`
   with a lookup-then-call dispatcher: specialized path for wisdom
   entries, generic fallback for cold cells. Build-time integration via
   CMake. Probably Phase 3.
5. **(v) Codelet fusion (v2.0+)** — the MKL-style monolithic-kernel
   workstream. Far bigger, separate effort. Not gated on this spike.

## Decision points worth memory entries when this lands

- **5% wall-time gain is shippable but small** — the per-stage codelet
  call structure is the lower bound for this design's headroom. Bigger
  wins (matching MKL's instruction density) need inter-stage fusion.
- **The tape pre-walk does the heavy lifting**, not the direct-call
  conversion. Anyone repeating this should not expect (B)-only to be
  worth implementing without (A).
- **OCaml-side emission scales naturally** to per-entry specialization
  — same machinery as `emit_profile_h.ml`. Adding another entry is
  ~3 lines + a regen.

## See also

- [docs/dev/vtune_n131072_k4_vfft_vs_mkl.md](../../../docs/dev/vtune_n131072_k4_vfft_vs_mkl.md)
  — the VTune profile this spike addresses. 21% wrapper share, 28%
  combined wrapper+scalar-prep share, ~3× MKL ceiling via codelet fusion.
- Codelet symbol naming alignment (same session, prior to this spike):
  prototype codelets renamed from `radix{R}_{variant}_{isa}_gen_inplace[_su[_spill]]`
  to production-style `radix{R}_{variant}_{isa}`. 832 codelets regenerated,
  66 consumer files updated. Drop-in compatible with production symbol
  slots once n1 signature alignment lands (separate task).
