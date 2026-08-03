# Register pressure in the radix-32-class interleaved kernels

*Benchmark- and disassembly-derived, this host (Raptor Lake, AVX2, 48 KB
L1d, 16 architectural ymm), 2026-08-03. All numbers from our own kernels
and probes; same-run ratios only.*

## Summary

The two-pass interleaved K=1 engine (`il2p`: leaf `n1t(R2)` → scratch →
mid `t2(R1)`) is kernel-bound, and within the kernels the binding
constraint at radix 32 is **register pressure, not arithmetic and not
redundancy**. The radix-16-class bodies spend 4.5–8.6 % of their
instruction stream on stack spill/reload; the radix-32-class bodies spend
**23.6–26.7 %**, plus 3.8–7.8 % on the scalar pointer reloads the spills
induce. The consequence is visible at every level:

- `t2(32)` runs **0.699 ns/point** where its r16 sibling `t2(16)` runs
  0.396 — **1.77× worse per point** for the same job at a bigger radix.
- The instruction count *and* the per-instruction rate degrade together,
  because they are the same phenomenon: a spill is extra instructions
  *and* extra L1 round-trips on the critical path. Subtracting spill
  traffic from the `t2(32)` body (683 → ~522 insns/iter) removes most of
  what makes it fat.
- There is **no seam**: full `il2p` execute equals leaf+mid within
  −2.4 %..+4.2 % (both N, two runs). Dispatcher, glue and aliasing are
  free; 100 % of any improvement must come from inside the two loop
  bodies.

Working-set arithmetic for context: at N=512 the engine uses 31.0 KB
(z 8 KB + scratch 8 KB + tables 15 KB) and fits L1d; at N=1024 it uses
63.0 KB and exceeds it. That overflow raises the floor at 1024 but is not
the differentiator between the r16 and r32 classes — the 1.77× per-point
gap exists with both kernels running on the same L1-resident data.

## The census (objdump of the generated kernels, production flags)

| kernel | insns/iter (main loop) | pts/iter | insns/pt | ymm spill st+ld | shuffles |
|---|---|---|---|---|---|
| `n1t(16)` leaf | 185 | 32 | 5.78 | 8.6 % | — |
| `t2(16)` mid | 220 | 32 | 6.88 | 4.5 % | — |
| `n1t(32)` leaf | 602 | 64 | 9.41 | **26.7 %** | 13.9 % (pipeline) |
| `t2(32)` mid | 683 | 64 | 10.67 | **23.6 %** | — |

The r32 bodies keep roughly **40 ymm values simultaneously live against
16 architectural registers**. The compiler does what it must: it spills,
and every spilled value that is reloaded is a store+load round-trip
through L1 competing with the kernel's real data traffic (~21 KB of stack
traffic per transform on an 8 KB problem at N=512).

A second, related census fact: the FMA fraction of these bodies is low
(10–11 %) against a high separate add/sub fraction (22–25 %). This is the
known fused-multiply-lift trade-off in the generator — fusing changes
value lifetimes, and lifetime pressure, not op count, is what these
bodies are dying of. The same mechanism, observed from the other side.

## Why this is not a CSE problem

The generator's algebraic simplifier already deduplicates the DAG; there
is no redundant arithmetic to eliminate. Classical CSE cannot help here
and would actively hurt: deduplicating a subexpression *extends* the
lifetime of its single computed value across all its uses, which widens
the live set — the exact quantity that is over budget. The gap is not
"compute less"; it is "hold less at once".

## Where the live-set width actually comes from

1. **Conjugate-pair construction.** The cil family computes the +k and
   −k columns as a coupled pair, which halves the twiddle table but
   *doubles the live state* — the two columns' intermediates coexist for
   the whole body. At radix 16 the coupled state (~20 values) fits the
   register file; at radix 32 (~40 values) it cannot. The r16 class is
   the control experiment: same construction, same generator, clean
   bodies. Conjugate-pair is a table-bytes-for-registers trade, and at
   radix 32 it sits on the wrong side of the budget.

2. **ILP-greedy scheduling.** The SU list scheduler orders operations by
   readiness, deliberately braiding independent chains for
   instruction-level parallelism. Braiding maximizes the live set. On a
   body that already spills a quarter of its stream, ILP is being bought
   with L1 round-trips — a bad trade that a register-minimizing
   (Sethi–Ullman-style, depth-first) schedule would decline. Serializing
   the two columns of the pair would halve peak live width at some ILP
   cost; on these bodies the exchange rate strongly favors registers.

3. **Twiddle residency.** Every twiddle vector loaded into a named
   register occupies an architectural register for its lifetime. AVX2
   permits arithmetic with full-width *memory operands* (note: embedded
   broadcast does not exist in AVX2 — that is AVX-512 — so a memory
   operand must be a pre-expanded 32-byte record). A twiddle consumed
   directly from memory as the source operand of its multiply/FMA
   occupies **zero** registers and eliminates the explicit load
   instruction. The runtime table for these kernels already stores
   pre-duplicated `[c c][−s s]` records, i.e. the layout is already
   memory-operand-consumable. The profit condition, stated precisely:
   this pays where twiddles are *streamed* (used once per fetch — true
   for the `t2` column stream), the body is *pressure-bound* (true for
   the r32 class), the layout is interleaved (records vectorize
   naturally), and the fatter table stays L1-resident. Where a twiddle
   is *reused* across iterations (group-invariant records held in a
   register), converting it would re-fetch the same value every
   iteration — a regression. Opt-in per kernel kind, never global.

4. **Lane surgery placement.** The pipeline carries 13.9 % shuffle
   traffic, and shuffles both add instructions and extend operand
   lifetimes. The corner-turn can instead be absorbed into the *leaf's
   stores* (the sectioned-store technique already proven in the cascade's
   `s0t` ingest), choosing the scratch layout such that the mid's loads
   and stores are straight contiguous moves. That removes the surgery
   from exactly the body that is over budget.

## Levers, ranked

1. **Memory-operand twiddle records** in the `t2` mids (mechanism 3) —
   attacks count and pressure simultaneously; a P0 probe (hand-modified
   generated C, bit-identity-gated by memcmp since the transform changes
   neither values nor order, codegen census before/after) is in flight.
   The open codegen question the probe answers: whether the compiler
   folds single-use load intrinsics into memory operands under
   production flags once the emitter stops naming the temporary.
2. **Permute-free pass 2 via scratch-layout choice** (mechanism 4).
3. **A non-conjugate-pair radix-32 twin** (mechanism 1): 2× twiddle
   table, decoupled column lifetimes. The table cost must be *raced*,
   not reasoned — at 1024 the working set is already past L1d, so the
   trade could go either way per cell.
4. **A register-minimizing schedule mode** for wide bodies (mechanism
   2): pure reordering, so bit-identity-gateable; raced against the SU
   schedule per kernel.
5. Plan shapes that keep every kernel in the r16 pressure class — noted
   for completeness; the current kernel registry cannot express one at
   1024 (every alternative pairing was raced and lost: 16×64 and 64×16
   both place a radix-32-or-64 body somewhere and measured 1.03–1.31×
   the 32×32 incumbent; at 512 the 3-stage chains DID win, +7.6–8.8 %
   over every 2-stage pair, two runs — the stage-count axis is live and
   belongs to the planner).

All levers enter through the established discipline: opt-in emitter
flags, bit-identity + speed gates per kernel (including the 4 KB-aliasing
check — fatter or relaid tables change stride patterns), per-cell races
through the planner, verdicts banked in wisdom. Capability everywhere,
application by measurement.

## Reproduction

- Kernel census: compile the generated `radix{16,32}_z_{n1t,t2}_*` with
  the production flags, `objdump -d`, count the main-loop body between
  the back-branch bounds; spill = `vmov` to/from `rsp/rbp` frames.
- Timing decomposition: `build_tuned/benches/smallre_split_probe.c`
  (leaf/mid/full arms, pinned core 2, paced, medians, same-run).
- Pair/stage-count race: `build_tuned/benches/smallre_pairs_probe.c`.
