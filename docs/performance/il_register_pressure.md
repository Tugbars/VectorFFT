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

1. **The radix-2 DIT even/odd seam** *(corrected 2026-08-03 — an earlier
   revision blamed conjugate-pair construction; recon showed CP serves
   odd radices only)*. The pow2 bodies couple outputs k and k+n/2
   through `butterfly_pair`: both outputs share the entire even and odd
   sub-DFT results plus one twiddled product, with only two
   column-private combine nodes. That sharing is genuine flop savings —
   decoupling it would roughly double the arithmetic, so it is the
   RIGHT construction. The pressure comes from the *straight-line
   R-wide body*: all E[·] and O[·] intermediates must coexist across
   the seam, and at R=32 that live set (~254 arithmetic nodes, ~40–64
   live values at peak) overwhelms 16 registers. At R=16 it fits — the
   r16 class is the control experiment. The generator's own BLOCKED
   comment documents the r32 spill numbers (158–197 stack ops) as the
   motivation for the in-tree `--cil-blocked` multi-pass form, which
   cuts lifetimes by staging through a spill plane deliberately —
   restructured math, tolerance-gated (NOT bit-identical), raceable.

2. **ILP-greedy scheduling.** The SU list scheduler orders operations by
   readiness, deliberately braiding independent chains for
   instruction-level parallelism. Braiding maximizes the live set. On a
   body that already spills a quarter of its stream, ILP is being bought
   with L1 round-trips — a bad trade that a register-minimizing
   (Sethi–Ullman-style, depth-first) schedule would decline. Serializing
   the two columns of the pair would halve peak live width at some ILP
   cost; on these bodies the exchange rate strongly favors registers.

3. **Twiddle residency — RESOLVED, already optimal (P0 probe, 2026-08-03).**
   The cil family's twiddle apply is a single fused IR node whose
   renderer splices the load intrinsics directly into the FMA argument
   slots; the compiler folds 100 % of twiddled legs into the 3-insn
   memory-operand shape (`vpermilpd $0x5` + `vmulpd disp(cursor)` +
   `vfmadd132pd disp(cursor)`) against the pre-duplicated
   `[c c c c][−s +s −s +s]` records, one cursor per table. Twiddles
   occupy ZERO vector registers in the shipped kernels — verified by a
   hand-transform probe whose "improved" variants disassembled to
   byte-identical instruction streams (md5-equal). Consequence: the
   spill pressure in the r32 bodies is **entirely data-DAG pressure**
   (~64 live values in the 32-leg straight-line body), which is why
   mechanisms 1, 2 and 4 are the levers and twiddle handling is not.
   ⚠ Scope: this holds for the cil (pure-IL) render path. The OTHER
   twiddle-render family (the `Load(Twiddle)`→named-temp path serving
   the oop/rfft/strided/boundary-split kinds, including the stfn
   terminators) has UNVERIFIED folding — worth one objdump check, as
   free wins may exist there.

4. **Lane surgery placement.** The pipeline carries 13.9 % shuffle
   traffic, and shuffles both add instructions and extend operand
   lifetimes. The corner-turn can instead be absorbed into the *leaf's
   stores* (the sectioned-store technique already proven in the cascade's
   `s0t` ingest), choosing the scratch layout such that the mid's loads
   and stores are straight contiguous moves. That removes the surgery
   from exactly the body that is over budget.

## Levers, ranked (revised after the P0 probe)

~~Memory-operand twiddle records~~ — **already shipped** (mechanism 3
above): the probe found the shipped kernels byte-identical to the
"improved" variants. The census numbers in this document were measured
WITH that optimization in effect. Struck through so nobody re-proposes it.

1. **Race the in-tree `--cil-blocked` multi-pass form for the r32 mid**
   (mechanism 1, corrected): decoupling the even/odd seam would double
   flops, but the blocked form cuts the live width the RIGHT way — it
   restages the same math through a deliberate spill plane with short
   lifetimes. It is restructured math (different summation order), so
   the gate is scalar-DFT tolerance + speed, not bit-identity. It
   exists today; it has never been raced in the il2p mid slot at 1024.
2. **A register-minimizing schedule order** (mechanism 2): the
   scheduler functor already exposes order-injection hooks
   (`VFFT_SCHED_ORDER`), so a pressure-minimizing order can be raced on
   the shipped kernels with ZERO emitter changes; if it wins, the
   permanent form is porting the existing top-level pressure mode
   (`~gh`, threshold 12) into the functor as a default-off knob. Pure
   reordering — bit-identity-gateable.
3. **Permute-free pass 2 via scratch-layout choice** (mechanism 4) —
   the remaining 12–14 % shuffle share.
4. **Verify folding in the OTHER twiddle-render family** (the
   named-temp path serving the oop/rfft/strided/boundary-split kinds,
   including the stfn terminators): one objdump census; free wins may
   exist where the compiler declined to fold named temporaries.
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

## Measured outcome of the levers (2026-08-03, strengthened protocol)

Raced under: 17 paced rounds × 1 sample/arm, alternated order, duplicated
shipped arm as CONTROL (its spread = the noise floor; deltas inside it
reported as noise), 3–4 fresh-process runs, shape-must-reproduce. The
machine was in a poor thermal state (controls 0.1–2.3 % on valid
sections; two sections self-invalidated at 9.5 %/23 % and were discarded)
— magnitudes are indicative, shapes are the verdict.

- **Blocked 4·8 split (`t2b48(32)`): CONFIRMED, both levels.** The only
  arm whose win reproduced in every valid section of every run: kernel
  −9..−21 %, full-pipeline −11..−21 %, always well above the same-run
  control. Static census: spill traffic −69 % (144→45 stack ops/iter),
  body 668→555. Restructured math — scalar-DFT tolerance gate
  (3.6e-15), never bit-identity. Contract: count % 2 == 0 (the il2p mid
  slot always satisfies it).
- **Blocked 2·16 (`t2b(32)`)**: pipeline win reproduces (−3..−15 %);
  the kernel-level win did NOT survive the cleanest-control run —
  2·16-vs-4·8 stays a per-cell measured pick. Bonus property: 2·16 is
  BIT-IDENTICAL to the shipped kernel (class-aware pairing), so it
  carries the strongest gate available.
- **Blocked r16 (`t2b(16)`) at 512: CONFIRMED (−4..−16 %)** — despite a
  WASH in the static census. A census wash falsified at timing: the
  staged structure pays even where spills were already low. (The reverse
  of the usual lesson, and worth remembering alongside it.)
- **Register-minimizing schedule order**: kernel-level inside noise;
  pipeline −1.6..−4.5 % — real but marginal, consistent with the
  simulation (peak live 35→33 against a 16-register budget: the SU
  schedule was already near the reachable floor; ordering cannot
  un-spill a 33-wide live set).

**PROMOTED (2026-08-04, commit `c60d84bb`).** Quiet-machine re-race
confirmed t2b48 promotion-grade (kernel −18..−20 %, pipeline −5..−14 %,
3/3 valid sections across three sessions) and t2b's kernel win
(−25..−27 %; an apparent earlier flip was traced to a per-process
4 KB-alias state that inflated the shipped/reordered arms +55 % while
**the blocked kernels sailed through unaffected** — placement-tail-risk
immunity as a side benefit). The reordered-schedule arm stayed inside
noise and was NOT promoted.

Shipped form: the three blocked kernels are first-class
(`radix{32,16}_z_t2b_avx2.c`, `radix32_z_t2b48_avx2.c`, forward-only —
the backward path never consumes `mid_f`), and `vfft_il2p_create` runs a
create-time mid race (t2q precedent: measured on the installed binary,
per cell, no wisdom change): correctness first (memcmp for the
bit-identical 2·16 class, 1e-12 for 4·8), then 7 paced rounds with a
>3 % hysteresis favoring the incumbent; kill switch `VFFT_NO_T2B`.
Both front-door natural gates re-ran ALL PASS from the promoted state;
the in-tree kernels are byte-identical to the raced campaign probes.

Measured through the front door (`--k1nat`, natural-vs-natural in-place,
same-day pre/post, 3 passes): **1024: −25 % median ns (1377→1033),
ratio 0.61→0.73, with the cross-engine error fingerprint changing —
numeric proof the tolerance-class t2b48 won the create-time race there;
512: −5.7 % (0.70→0.73), matching t2b16's prior quiet magnitude; 128
unchanged (the de-facto control cell).** The per-cell picks observed
live: 256→t2b16, 512→t2b16/t2b48 by pair, 1024→t2b48, 2048→t2b.

## Reproduction

- Kernel census: compile the generated `radix{16,32}_z_{n1t,t2}_*` with
  the production flags, `objdump -d`, count the main-loop body between
  the back-branch bounds; spill = `vmov` to/from `rsp/rbp` frames.
- Timing decomposition: `build_tuned/benches/smallre_split_probe.c`
  (leaf/mid/full arms, pinned core 2, paced, medians, same-run).
- Pair/stage-count race: `build_tuned/benches/smallre_pairs_probe.c`.
