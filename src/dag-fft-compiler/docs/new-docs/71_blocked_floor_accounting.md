# 71: The blocked construction's floor accounting — seam scalarization, allocator tax, and the verdict on further register-side work

> One-sentence version: decomposing the blocked pow2 codelets shows (a)
> the doc-58 seam DOES NOT EXIST at the machine level — both compilers
> fully scalarize spill_re/im (zero symbol references, zero leas, arg-io
> counts confirm), so blocking wins by SHAPING the allocation problem,
> not by owning spills; (b) that shaping is worth 3.3–4.7× on the Belady
> order-floor (70→21 at R=16, 249→53 at R=32, like-for-like); (c) the
> remaining realized-vs-floor gap is ALLOCATOR TAX (+38/+130 under gcc)
> which the annealer's direct search already proved only ~10%
> order-payable (blk32: 183→168 of a −130 gap) — therefore items (2)
> blocked-comparator ablation and (3) cluster minimax are CLOSED as
> low-expected-value, and register-side pow2 work is complete; the
> remaining performance lives on the memory side (doc 70 §5.4).

Tool: `tools/c_traffic.py` — Belady traffic of an EMITTED codelet's
order parsed from the C, with seam accesses treated the way SRA treats
them (stores bind slot→producer, loads are aliases). Validated by
reproducing mono numbers under the same convention.

## 1. The seam is a fiction (measured, bit-exact-gated)

Static-scratch probe on blk16/blk32 under gcc-13.3 and clang-18:
`spill_` appears NOWHERE in the assembly — not as a direct rip
reference, not via lea'd base registers. vmov memory operands classify
exactly into rsp/rbp (allocator) + argument-pointer io (62–64 at R=16,
127–128 at R=32 = precisely the input/output traffic). Both compilers
scalarize the non-escaping arrays (SRA) and re-decide ALL spilling
themselves. Consequences: the historical "59 spills" at blk16 was
never seam+allocator — it is 100% allocator; and doc 58's "seam
through L1 scratch by design" is design INTENT that shapes the
dataflow but never becomes memory. This matches doc 70's probe (gcc
deleted our explicit scratch the same way): allocators refuse to
surrender spilling, under any construction.

## 2. Like-for-like floors (c_traffic convention: sinks included*)

| codelet | order Belady floor @16 | gcc realized | clang realized | allocator tax |
|---|---:|---:|---:|---|
| blk16 | **21** | 59 | 52 | +38 / +31 |
| mono16 | 70 | 78 | 82 | +8 / +12 |
| blk32 | **53** | 183 | 150 | +130 / +97 |
| mono32 | 249 | 289 | 289 | +40 / +40 |

*Convention note, applying to all earlier ABSOLUTE traffic numbers
(docs 68/69, dump-based): the dump excludes store sinks as users, so
output values were free; c_traffic includes them. Every RELATIVE
result (races, certifications, ablations) used a single convention
throughout and is unaffected; absolutes are convention-tagged from
here on.

## 3. Three findings

**F1 — Blocking's shaping, finally quantified.** The two-pass
cluster-contiguous order lowers the DAG's Belady floor 3.3× (R=16) and
4.7× (R=32) versus the monolithic SR order OF THE SAME DAG. This is
the construction's certificate: it is a traffic transformer, and doc
70's headline is hereby refined from "construction-level SPILLING
dominates" to "construction-level SHAPING dominates" — the mechanism
is a better allocation problem, not explicit spills (which never
survive compilation).

**F2 — The allocator tax anti-correlates with floor quality.**
Compilers pay +8/+12 above the floor on the mediocre mono orders but
+38..+130 on the excellent blocked orders: real allocators realize a
roughly structure-dependent absolute spill mass and cannot exploit
ultra-low-traffic orders. The gap concentrates where cluster-contiguity
creates long cross-boundary ranges that greedy/linear-scan splitting
handles poorly.

**F3 — The tax is not order-payable (empirical bound already held).**
The blk32 annealer campaign searched this exact gap directly on
realized gcc output: 183→168, i.e. −15 recovered of the −130 on paper.
That is the measured ceiling for what items (2) comparator-ablation
and (3) cluster-minimax could deliver: single-digit-percent static
gains, further diluted at runtime.

## 4. Verdict (gating decision of doc 70 §5)

Items (2) and (3): CLOSED. Existing wisdom entries stay banked; the
subset comparator stays un-ablated as accepted debt (documented here);
the annealer remains the certifier per doc 66. Register-side pow2 is
COMPLETE: construction certified (docs 70+71), order at/near floor by
design, residual = allocator tax bounded by direct search, agnosticism
already achieved (clang ≤ gcc on blocked: 52/150 vs 59/183).

Open, in priority order: the MEMORY side (doc 70 §5.4: A9 padding ×
MKL race at K=128, seam/aliasing placement, hugepage stack — the
phase-3 port-slot analysis located the real MKL gap there); MSVC
baseline for the Windows driver product (spill_inject.py's counting
harness reusable); and one build-system note, not a code item: clang
beats gcc on blocked pow2 by 12–18% spills — per-TU compiler choice
for hot codelets is a legitimate agnostic lever if runtime confirms.
