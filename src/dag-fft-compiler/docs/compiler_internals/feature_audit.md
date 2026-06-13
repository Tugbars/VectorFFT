# Feature audit: K-batched in-place codelet emission vs codelet_oop

What runs when you say `dune exec bin/gen_radix.exe -- --radix R --in-place --twiddled`
vs what runs in `codelet_oop.emit_codelet`.

Evidence: R=16 n1 strided codelet uses 144 vector ops (0 standalone muls,
38 FMAs); OOP UL/UL emits 166 ops (22 standalone muls, 0 fmadd/fnmadd).
Gap is the entire IR optimization pipeline, not just fence emission.

═══════════════════════════════════════════════════════════════════════
LAYER 1 — Math layer DAG construction
═══════════════════════════════════════════════════════════════════════

| Feature                             | gen_radix path             | codelet_oop |
|-------------------------------------|----------------------------|-------------|
| Dft.dft_expand (n1)                 | ✓                          | ✓           |
| Dft.dft_expand_twiddled (t1)        | ✓                          | ✓           |
| Dft.dft_expand_n1_blocked (R≥25 n1) | ✓ (auto-on via             | ✗ MISSING   |
|                                     |   should_block_n1)         |             |
| Dft.dft_expand_twiddled_spill (t1)  | ✓ (auto-on via             | ✗ MISSING   |
|                                     |   should_spill)            |             |
| spill_markers from math layer       | ✓                          | ✗ MISSING   |
| ct=(n1,n2) cluster info             | ✓                          | ✗ MISSING   |

Impact: at R≥25 n1 or any t1 above the should_spill threshold, the
codelet should use PASS-1/PASS-2 blocked structure with explicit spill
markers. Without blocking, monolithic emit at R=25 produces 1128 vector
instructions vs 67 ns/call blocked (47% AVX-512 speedup).

═══════════════════════════════════════════════════════════════════════
LAYER 2 — Algebraic simplification (Algsimp)
═══════════════════════════════════════════════════════════════════════

| Pass                            | gen_radix | codelet_oop |
|---------------------------------|-----------|-------------|
| of_assignments ~reassoc         | ✓         | ✓           |
| dedup_sub_pairs                 | ✓         | ✗ MISSING   |
| factor_common_muls              | ✓         | ✗ MISSING   |
| factor_by_atom                  | ✓         | ✗ MISSING   |
| dedup_sub_pairs (second pass)   | ✓         | ✗ MISSING   |
| collect_m  (VFFT_COLLECT_M)     | ✓ opt-in  | ✗ MISSING   |
| deep_collect (VFFT_DEEP_COLLECT)| ✓ opt-in  | ✗ MISSING   |
| share_subsums (aggressive)      | ✓ for prm | ✗ MISSING   |
| net_transpose (Frigo)           | ✓ for prm | ✗ MISSING   |
| **fma_lift**                    | ✓         | ✗ MISSING   |
| Multi-stage frozen-tag remaps   | ✓         | ✗ MISSING   |

Impact: this is where the 22 standalone muls come from. fma_lift turns
`Add(Mul(a,b), c)` → `Fma(a, b, c)` at the IR level; without it, the
emitter writes `_mm512_mul_pd` followed by `_mm512_add_pd` as separate
declarations. gcc -O3 can sometimes auto-fuse these but not always
(especially with the fences blocking inter-instruction visibility).

═══════════════════════════════════════════════════════════════════════
LAYER 3 — Spill / pass-structure handling
═══════════════════════════════════════════════════════════════════════

| Feature                                | emit_codelet | codelet_oop |
|----------------------------------------|--------------|-------------|
| make_spill_info ~ct ~fuse              | ✓            | ✗ MISSING   |
| classify_passes (PASS 1 / PASS 2 split)| ✓            | ✗ MISSING   |
| is_spilled / is_fused_tag predicates   | ✓            | ✗ MISSING   |
| Cluster-local scheduling (by sub-FFT)  | ✓            | ✗ MISSING   |
| spill_re[] / spill_im[] array decl     | ✓            | ✗ MISSING   |
| Per-position spill_sites emission      | ✓            | ✗ MISSING   |
| Per-position reload_sites emission     | ✓            | ✗ MISSING   |
| Cluster-boundary store flush           | ✓            | ✗ MISSING   |
| Fused-slot retention across passes     | ✓            | ✗ MISSING   |

Impact: for R≥32, peak_live at R=32 is 260 values, R=64 is 267 values.
The 32 zmm registers can't hold this — without spill structure, gcc
spills to the stack repeatedly. With spill markers, the math layer
breaks the computation into passes that each fit in registers, and
emit_codelet generates explicit stores/loads at the boundaries.

═══════════════════════════════════════════════════════════════════════
LAYER 4 — Scheduling
═══════════════════════════════════════════════════════════════════════

| Scheduler                       | emit_codelet | codelet_oop |
|---------------------------------|--------------|-------------|
| Topological (sort by tag)       | ✓            | ✓ (only)    |
| Bisection (Frigo's recursive)   | ✓            | ✗ MISSING   |
| **SU (Sethi-Ullman)** w/ uarch  | ✓            | ✗ MISSING   |
| BB (branch-and-bound lex cost)  | ✓            | ✗ MISSING   |
| Annotated_topological           | ✓            | ✗ MISSING   |
| Annotated_bisection             | ✓            | ✗ MISSING   |
| Annotated_SU                    | ✓            | ✗ MISSING   |
| Goodman-Hsu mode (AVX2 R≥32)    | ✓ auto       | ✗ MISSING   |
| su_schedule_subset (clustered)  | ✓            | ✗ MISSING   |
| compute_inline_set              | ✓            | ✗ MISSING   |
| compute_su_number               | ✓            | ✗ MISSING   |
| cp_dist computation             | ✓            | ✗ MISSING   |

Impact: SU schedules nodes by cp_dist (critical path distance) with
uarch latency table as tiebreaker. The schedule order affects register
pressure and instruction-level parallelism. Topological by tag is the
"worst" baseline; SU+GH is ~5-10% faster than Topological at moderate
R, more at high R.

═══════════════════════════════════════════════════════════════════════
LAYER 5 — Register allocation
═══════════════════════════════════════════════════════════════════════

| Feature                              | emit_codelet | codelet_oop |
|--------------------------------------|--------------|-------------|
| Regalloc.prepare_for_simple_codelet  | ✓            | ✗ MISSING   |
| Regalloc.allocate (SSA linear-scan)  | ✓            | ✗ MISSING   |
| Regalloc.allocate_with_spilling      | ✓            | ✗ MISSING   |
| force_last_use map                   | ✓            | ✗ MISSING   |
| spill_sites table                    | ✓            | ✗ MISSING   |
| reload_sites table                   | ✓            | ✗ MISSING   |
| name_overrides for reload variables  | ✓            | ✗ MISSING   |
| install_alloc_canonical              | ✓            | ✗ MISSING   |
| current_regalloc ref consumption     | ✓            | ✗ MISSING   |
| Regalloc overflow handling           | ✓            | ✗ MISSING   |

Impact: only activates under the two-rule policy (log3 AVX-512 R≤32).
For most n1/t1 codelets the regalloc path is OFF — but the spill_sites
machinery is what makes the spill recipe work. Without it, the spill
markers in layer 3 have nowhere to go.

═══════════════════════════════════════════════════════════════════════
LAYER 6 — Per-node emission style
═══════════════════════════════════════════════════════════════════════

| Decl style                         | emit_codelet         | codelet_oop  |
|------------------------------------|----------------------|--------------|
| `const __m512d tN = ...;`          | ✓ (fallback)         | ✓ (only)     |
| `register __m512d tN = ...;` +     | ✓ (default for       | ✗ MISSING    |
|   `asm volatile ("" : "+v"(tN));`  |   almost everything) |              |
| `register __m512d tN asm("zmmK")=` | ✓ (when regalloc ON  | ✗ MISSING    |
|   + asm volatile fence             |   AND not unpin)     |              |
| Pinned reload variable form        | ✓                    | ✗ MISSING    |
| Selective unpin (NK_Mul→consumer)  | ✓                    | ✗ MISSING    |
| Inline expression at consumer      | ✓ via inline_set     | ✗ MISSING    |

Impact: fence emission is THE main perf mechanism per the docs
("inline-asm scheduling fence — not the register-pin clause — is the
actual win mechanism in nearly all codelets"). Empirical: fence-only is
the default in production, applies to virtually all codelets except n1
AVX2 R∈{8,16}. We have it OFF for everything.

═══════════════════════════════════════════════════════════════════════
LAYER 7 — Scope/annotation
═══════════════════════════════════════════════════════════════════════

| Feature                          | emit_codelet | codelet_oop |
|----------------------------------|--------------|-------------|
| Annotate.annotate (scope tree)   | ✓            | ✗ MISSING   |
| Annotate.emit_scope              | ✓            | ✗ MISSING   |
| Nested-block lifetime hints      | ✓            | ✗ MISSING   |

Impact: opt-in via --annotate. Gives gcc lifetime hints via inner
`{ ... }` blocks; small additional perf when codelet has many
short-lived intermediates.

═══════════════════════════════════════════════════════════════════════
LAYER 8 — Two-rule policy + env hatches
═══════════════════════════════════════════════════════════════════════

| Mechanism                                | emit_codelet | codelet_oop |
|------------------------------------------|--------------|-------------|
| Two-rule pin/fence default policy        | ✓            | ✗           |
| VFFT_PEAK_LIVE diagnostics               | ✓            | ✗           |
| VFFT_USE_REGALLOC (legacy)               | ✓            | ✗           |
| VFFT_NO_REGALLOC                         | ✓            | ✗           |
| VFFT_PIN_FORCE                           | ✓            | ✗           |
| VFFT_DISABLE_SELECTIVE_PIN               | ✓            | ✗           |

═══════════════════════════════════════════════════════════════════════
THE GAP, CONCRETELY (R=16 n1, AVX-512)
═══════════════════════════════════════════════════════════════════════

```
strided codelet (full pipeline)       OOP codelet (raw)
─────────────────────────────────     ─────────────────────────────
total vector ops:   144               total vector ops:   166        (+15%)
_mm512_mul_pd:        0               _mm512_mul_pd:       22
_mm512_fmadd_pd:     18               _mm512_fmadd_pd:      0
_mm512_fnmadd_pd:    20               _mm512_fnmadd_pd:     0
_mm512_fmsub_pd:      0               _mm512_fmsub_pd:      0
_mm512_fnmsub_pd:     2               _mm512_fnmsub_pd:     4
_mm512_add_pd:       52               _mm512_add_pd:       68
_mm512_sub_pd:       52               _mm512_sub_pd:       72

decl style: register + asm volatile   decl style: const           
fence emission: ON                    fence emission: OFF
```

The OOP codelet has 22 mul + 0 fmadd/fnmadd vs strided's 0 mul + 38 FMA.
That's 22 unfused multiplications — without fma_lift at the IR level,
the emitter writes them as separate Mul nodes and gcc -O3 doesn't always
fold them across the (missing) fences.

═══════════════════════════════════════════════════════════════════════
PROPOSED WIRING ORDER (each independently shippable)
═══════════════════════════════════════════════════════════════════════

### Tier A — quick wins, almost no code (~30 lines)

A1. **Set current_fence_only := true before body emission**
    Result: every const_decl becomes fenced_decl
    Expected gain: 5-15% at all radices, more at higher R
    Risk: none — fenced_decl is byte-identical to const_decl from gcc's
    POV except for the asm volatile fence (which is the whole point)

A2. **Run the algebraic simplification chain on the raw DAG**
    Mirror gen_radix's pipeline:
      dedup_sub_pairs → factor_common_muls → factor_by_atom →
      dedup_sub_pairs → fma_lift
    Result: 22 muls → 0 muls + ~20 FMAs (saves 20 instructions)
    Expected gain: 10-15% at R=16, more at higher R
    Risk: low — the passes are well-tested and feed into the same
    fma_lift the strided codelets use

A3. **Compute and pass inline_set to render_node_def**
    Result: single-use intermediates inline at their consumer instead
    of getting their own `const __m512d tN = ...;` line
    Expected gain: 5-10% via reduced register pressure
    Risk: low

Combined A1+A2+A3: should close most of the R=16 gap (currently 9%,
expected ~0-3% after).

### Tier B — medium effort, big win at R≥25 (~150 lines)

B1. **Switch n1 codelet construction to dft_expand_n1_blocked when
     should_block_n1 (R ≥ 25)**
    Result: codelet has PASS 1 / PASS 2 structure with spill markers
    Expected gain: huge at R=32/64 (the existing R=25 blocked beats
    monolithic by 47% AVX-512)
    Risk: medium — need to thread spill_markers through to emit_codelet

B2. **Construct spill_info via make_spill_info ~ct ~fuse**
    Mirror gen_radix's tag-remap logic for spill markers post-algsimp

B3. **Wire classify_passes + Pass 1 / Pass 2 emission with spill
     stores/reloads at the boundary**
    The spill emission helpers (emit_node_spill_sites,
    emit_node_reload_sites, emit_regalloc_spill_decl) are already in
    emit_c.ml; just need to call them at the right positions

### Tier C — full optimization quality, ~300 lines

C1. **Replace topological emission with SU scheduler**
    Use Schedule.su_schedule (for non-blocked) or
    Schedule.su_schedule_subset per cluster (for blocked)
    Expected gain: 5-10% on top of A+B

C2. **Wire Regalloc.allocate via install_alloc_canonical**
    Sets current_regalloc for render_node_def to consume
    Only activates under two-rule (log3 AVX-512 R≤32 by default)

C3. **Wire selective unpin via compute_unpin_candidates**
    Lets gcc auto-fuse NK_Mul into FMA across what would be a fence
    barrier (the FMA gets a register, the Mul disappears)

═══════════════════════════════════════════════════════════════════════
RECOMMENDATION
═══════════════════════════════════════════════════════════════════════

Do Tier A first (~30 lines, very low risk). Re-bench. The Tier A
delta alone should bring us close to FFTW parity at R≤16 and partially
close the gap at R=32/64.

Then evaluate whether Tier B is worth it. For R=32/64 it almost
certainly is — the empirical numbers from R=25 blocked vs monolithic
(47% speedup) point at where the perf is buried.

Tier C is the last 5-10% on top.
