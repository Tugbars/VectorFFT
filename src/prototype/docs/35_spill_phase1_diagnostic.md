# 35. Spill Controller Phase 1: Diagnostic

## Context

Doc 34 closed the real-shuffle measurement work and identified the spill
controller redesign as the highest-leverage next step. The empirical
question that prompted this:

- R=512 AVX-512 monolithic has 5216 stack ops per K iter
- R=512 AVX2 has 12066 stack ops per K iter
- These spill counts dominate memory traffic and explain why the AVX2
  R=512 crossover at B=256 happens; eliminating them could shift the
  boundary or eliminate it

Before designing a fix, we need to know what these spill ops actually are.
The current recipe (`dft_expand_twiddled_spill` in lib/dft.ml) places one
spill marker per Pass 1 output at the outermost CT level. For R=512 =
CT(16, 32) that's 16 * 32 = 512 slots × 2 (re+im) = 1024 stores plus 1024
reloads = 2048 recipe-mandated stack ops per call.

But R=512 AVX-512 has 5216 stack ops total. **GCC is adding 3168 extras
beyond what the recipe placed.** This doc characterizes those extras and
identifies where they come from.

## Methodology

For each codelet (R ∈ {16, 32, 64, 128, 256, 512} × ISA ∈ {AVX-512, AVX2}):

1. Generate the C source via `gen_radix --twiddled --in-place --isa <isa>`
2. Compile to asm with `gcc-13 -O3 -masm=intel`
3. Identify the K-loop body between the first two `.L<n>:` labels
4. Count stack ops by scanning for `vmovapd … [rsp]` in Intel syntax (offset
   appears before `[rsp]`, e.g., `vmovapd ZMMWORD PTR 8648[rsp], zmm3`)
5. Separately count recipe-mandated stores/reloads by searching the C
   source for `_mm*_storeu_pd(&spill_re[…])` and `_mm*_loadu_pd(&spill_re[…])`
6. Categorize:
   - Recipe-mandated: forced by `dft_expand_twiddled_spill` at the Pass 1/Pass 2 boundary
   - GCC-extra: added by GCC's register allocator on top of the recipe

To split GCC-extras by pass, detect the Pass 1 → Pass 2 boundary in the
asm body: the position where cumulative (stores − reloads) is maximized.
Recipe places all stores at end of Pass 1 and all reloads at start of
Pass 2, so this position is well-defined.

## Results: total stack ops broken down

```
                AVX-512                       AVX2
R     recipe   gcc-extra  total       recipe   gcc-extra  total
16    64       -52        12          64       46         110
32    128      -44        84          128      238        366
64    256      -1         255         256      505        761
128   512      147        659         512      1397       1909
256   1024     1016       2040        1024     4182       5206
512   2048     3168       5216        2048     10018      12066
```

(Total = recipe + gcc-extra. Recipe = 2 stores + 2 reloads per Pass 1
output × N outputs = 4N ops. Gcc-extra = (total measured in asm) − recipe.)

Three things stand out:

The first is that **at R ≤ 64 on AVX-512, gcc-extra is *negative***. GCC's
register allocator actively removes recipe-mandated spills it considers
unnecessary. With 32 ZMM names available, the recipe is over-eager at
small R — declaring slots GCC then keeps in registers. At R=16, the
recipe declares 64 spill ops but only 12 survive compilation.

The second is the **R=64 transition point on AVX-512**: gcc-extra ≈ 0.
This is where the recipe is calibrated to match actual register pressure.
Above R=64, peak live exceeds the recipe's plan and GCC adds extras.
Below, the recipe over-spills and GCC trims. R=64 is the structural
inflection point — the recipe was designed for it.

The third is that **AVX2 always has positive gcc-extras**, even at
R=16. The 16-register architectural file is exceeded by anything with
non-trivial live state. GCC has no slack to absorb the recipe's
over-eagerness; instead the recipe + GCC both pile spills on.

## Results: pairing and dead-spill analysis

For each distinct `[rsp+offset]` slot, count how many stores and reloads
hit it. Categorize:

- Paired: ≥ 1 store *and* ≥ 1 reload
- Write-only: store with no matching reload (dead spill)
- Read-only: reload with no matching store (cannot happen for stack slots in our pattern)

```
AVX-512                                   AVX2
R      slots  paired  dead  multi-write   slots  paired  dead  multi-write
16     7      5       2     0             23     21      2     5
32     44     40      4     0             63     62      1     11
64     122    118     4     4             144    144     0     25
128    292    289     3     10            361    360     1     47
256    588    583     5     54            572    572     0     231
512    1664   1660    4     92            1145   1143    2     469
```

Dead spills are essentially zero on both ISAs (0.1–0.4% of slots). The
"remove dead spills" hypothesis is closed — every spill is reloaded at
least once. No low-hanging fruit there.

**Multi-write slots are large on AVX2** (469 of 1145 = 41% at R=512) but
small on AVX-512 (92 of 1664 = 5.5%). On AVX2, GCC reuses the same stack
location for *different* live values across the body: spill value A,
reload A, use A, spill value B to the same slot, reload B, etc. This is
register-pressure thrashing — GCC literally cannot keep enough state in
16 YMM registers and treats the stack as scratch space.

On AVX-512 slots are almost exclusively accessed exactly twice (store +
reload). Clean Pass 1 → Pass 2 transit. Multi-write at AVX-512 R=512 is
only 92 slots out of 1664 — small minority.

## Results: store-to-reload distance

Average instructions in the K-loop body between a slot's store and its
first reload:

```
                AVX-512                  AVX2
R              avg-dist                 avg-dist
16             147                      174
32             488                      485
64             1263                     1227
128            2446                     2237
256            5716                     4777
512            9273                     10536
```

Distances grow linearly with body size. At R=512 the average spill is
held across roughly 9-10k instructions — about 30% of the body. This is
the macro Pass 1 / Pass 2 transit signature: most spills cross the
entire computation. There are no "short-distance convenience spills"
to attack.

## Results: per-pass GCC-extras

The boundary between Pass 1 and Pass 2 in the asm body is where
cumulative (stack stores − stack reloads) is maximized. The recipe
puts all 2N stores in Pass 1's tail and all 2N reloads in Pass 2's
head, so this position is well-defined.

```
AVX-512                                       AVX2
R      recipe   p1-gcc   p2-gcc   total      recipe   p1-gcc   p2-gcc   total
16     64       0        1        1          64       32       14       46
32     128      0        3        3          128      168      70       238
64     256      4        12       16         256      324      181      505
128    512      67       80       147        512      977      420      1397
256    1024     155      861      1016       1024     1992     2190     4182
512    2048     1585     1583     3168       2048     5731     4287     10018
```

The Pass-1/Pass-2 split is informative across sizes:

- **AVX-512 R=512** is exactly balanced (1585 vs 1583). Both passes
  contribute equally to GCC-extras. Sub-cluster spilling would
  symmetrically address both.
- **AVX-512 R=256** is lopsided toward Pass 2 (155 vs 861). The CT(16, 16)
  factorization makes Pass 2's inputs (Pass 1's twiddled outputs) more
  register-pressure-intensive than the Pass 1 sub-DFT-16s on raw inputs.
- **AVX2 at R ≤ 128** is consistently Pass-1-dominant (64-70%). Pass 1
  sub-DFTs at AVX2 are register-starved from the start; Pass 2 reads
  from spill arrays so its working set per sub-DFT is smaller.

## What the data says about Phase 2 design

Three findings converge on a single recommendation.

**Finding 1: GCC-extras are real spills, not waste.** Pairing analysis
shows 99.6%+ of spills are paired, distances are large (~30% of body),
no dead-store cleanup opportunity exists. These are values that genuinely
need to be in memory because peak live exceeds the register file.

**Finding 2: The recipe only addresses the macro boundary, not within-pass
pressure.** The recipe places markers at the Pass 1/Pass 2 boundary only.
But GCC-extras happen *inside* each pass because the within-pass live
count itself exceeds the register file. At R=512 AVX-512 the recipe
handles 2048 ops and GCC adds 3168 more inside the passes.

**Finding 3: Both passes contribute substantially.** Across most (R, ISA)
combinations, Pass 1 and Pass 2 each contribute 30-70% of GCC-extras.
Any fix must address both passes, not just one.

The Phase 2 recommendation: **add intra-pass sub-cluster spill markers**.
Currently the recipe says "spill all Pass 1 outputs at the boundary."
The extension says "additionally, within Pass 1, after each group of K
sub-DFT-N2s, spill that group's outputs before computing the next group."

For R=512 = CT(16, 32) with sub-cluster size = 4, Pass 1 becomes:

```
group 0: compute sub-DFT-32 #0, #1, #2, #3 → spill their 128 outputs
group 1: compute sub-DFT-32 #4, #5, #6, #7 → spill their 128 outputs
group 2: compute sub-DFT-32 #8, #9, #10, #11 → spill
group 3: compute sub-DFT-32 #12, #13, #14, #15 → spill
```

Peak live within Pass 1 drops from ~256-512 values (current) to ~128
values (sub-cluster of 4 × 32 outputs) plus working state of one
sub-DFT-32 (~30-50 values). Total peak live: ~150-180 values vs current
~300-500. The 32 ZMM register file should hold the active computation
without needing GCC-extras.

Same approach applied to Pass 2. The total recipe stack ops would grow
slightly (more boundary spills) but GCC-extras should shrink
substantially. Net: lower total.

## Predicted impact

Order-of-magnitude prediction for R=512 AVX-512 with sub-cluster size 4:

- Current: 2048 recipe + 3168 GCC = 5216 total stack ops
- Target: ~2500 recipe (boundary + 3 intra-pass × 256 each per pass) +
  small GCC-extras (peak live now in range)
- Net: ~2800-3200 total — roughly half the current count

If achieved, this drops mono's stack-op disadvantage vs multi-stage's
1728 down to near-parity, and runtime-wise should close the R=512
AVX-512 B=256 parity gap into a clear mono win. On AVX2 R=512, the
expected drop is smaller in percentage terms (the 16-register file is
the structural bottleneck regardless) but absolute counts should still
fall significantly.

## Phase 2 entry point

The recipe lives in `lib/dft.ml`'s `dft_expand_twiddled_spill`. The
spill_marker type currently has `(slot, re_expr, im_expr)`. To support
intra-pass clustering, we'd add a `cluster_id: int` field (or
equivalent ordering metadata) so emit_c knows which markers belong
together and can place stores immediately after computing a group's
outputs rather than batching all stores at the end.

The cluster size becomes a parameter, default size based on register
file size: sub-cluster ≈ vec_regs / (1 + N2/16) or similar heuristic
that keeps within-cluster live count below vec_regs.

Validation must include:
- Prime correctness 56/56 still passes
- R=16/R=32 ours-vs-hand no regression on either ISA
- Stack op counts measured and reported per (R, ISA) cell
- Real-shuffle bench re-run on R=512 to confirm boundary shift

The diagnostic infrastructure built for this doc (the spill-pattern
analyzer in `/tmp/spill_phase1`) becomes the validation harness.

## Open questions for Phase 2

The sub-cluster size is the key tunable. Too small: too many recipe
ops, also too many transitions GCC may not handle well. Too large:
within-cluster live count still exceeds register file, GCC adds extras
anyway. Empirical sweep across cluster sizes (1, 2, 4, 8, 16) at R=512
on both ISAs would find the optimum.

A secondary question: should the cluster size be ISA-dependent? AVX2's
16-register file likely wants smaller clusters than AVX-512's 32. The
heuristic `cluster_size ≈ vec_regs / 4` would give AVX-512 cluster=8
and AVX2 cluster=4. This needs measurement.

Third: does the sub-cluster help Pass 2 the same way it helps Pass 1?
Pass 2 reads inputs from spilled values, computes a sub-DFT-N1, writes
outputs to the rio buffer (not back to spill). Its live count profile
is different — likely lower peak per sub-DFT but more accumulated across
sub-DFTs. The fix may need different cluster sizes per pass.
