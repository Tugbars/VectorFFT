# M3a — SSA register allocation, working result

**Status: VALIDATED on R=64 t1_dit AVX-512.**
7.5% runtime speedup, 50% spill reduction, 13% smaller code.

## What this delivers

A new pass between Schedule and Emit_c that pre-allocates ZMM registers
for every materialized tag in pass 1 and pass 2 of the spill-flow path
(t1/t1s/log3 codelets). When `VFFT_USE_REGALLOC=1` is set, the codelet
generator emits `register __m512d tN asm("zmmK") = ...; asm volatile
("" : "+v"(tN));` declarations — the barrier forces gcc-11 to honor
our register choice instead of running its own RA.

Default behavior (env unset) is byte-identical to pre-M3a output.

## Measured results (R=64 t1_dit AVX-512, gcc-11 -O3)

| Metric | Default (gcc RA) | M3a | Δ |
|---|---|---|---|
| Stack spills | 493 | 247 | **-50%** |
| Code size | 25,072 B | 21,800 B | **-13%** |
| Total instructions | 2,116 | 2,029 | -4% |
| Runtime | 1,049 ns | 970 ns | **-7.5%** |

Correctness: bit-identical except 1 LSB at sample 784/1024 (gcc commutes
some FMA chains slightly differently between the two paths — expected
FP behavior).

## What's IN the M3a deliverable

### `lib/regalloc.ml` — the allocator
- `type assignment = Reg of string | Default`
- `allocate_linear_scan : isa → scheduled → budget → ?skip_tags → ?inline_set → ?force_last_use → alloc_result`
- Standard SSA linear-scan / chordal greedy coloring
- Three correctness fixes (see "Subtleties" below)

### `lib/isa.ml` — output mechanism
- `pinned_reg_decl : isa → name → reg → expr → string` emits the barrier-pinned form

### `lib/emit_c.ml` — integration
- Top-level `current_regalloc : Regalloc.allocation option ref` (single-threaded; documented)
- `install_alloc` called at `spill_pass1` and `spill_pass2` sites; computes per-pass `force_last_use` from `pass1_assigns` / `pass2_assigns` with cluster-flush awareness
- Decision in declaration emission point: pinned-reg form when allocation has `Reg`, else fall through to existing `const __m512d` form

## Three subtleties that made this hard

### Subtlety 1: barriers are required
`register __m512d t asm("zmm5") = ...;` without a barrier is treated by
gcc as a HINT. gcc's own RA runs and ignores the pin. The fix:

```c
register __m512d t asm("zmm5") = <expr>;
asm volatile ("" : "+v"(t));    /* commit point */
```

The barrier forces gcc to materialize t in zmm5 at that point. Without
this, our register choice is silently ignored. Confirmed via a probe
that tried 20+ pinned variables; without barriers, only 11/20 ZMMs
were honored, with barriers all 20 were honored.

### Subtlety 2: inlined nodes' transitive uses
emit_c's `should_inline` predicate folds single-use simple expressions
into consumer RHS. When tag X is inlined into Y's RHS, X's preds A,B
are referenced at Y's position, NOT at X's scheduled position.

The naive last_use computation walks immediate preds only. Result: A
and B get freed prematurely (at X's position), then later allocations
clobber A's and B's registers, and Y's emitted expression reads
stale values.

Fix: walk preds transitively whenever the immediate pred is in
`inline_set`. Bound by the inline depth limit (emit_c uses 32) but
in practice never goes that deep.

### Subtlety 3: store emission outside the scheduled walk
emit_c emits stores at TWO places that the IR pred relation doesn't
capture:

1. End-of-pass: `List.iter emit_store pass1_assigns` at end of pass 1.
2. Mid-pass via cluster flushing: `flush_cluster_stores prev` whenever
   pass 2's walk crosses a cluster boundary.

Both reference tags whose IR-level last_use is earlier. Without
extending lifetime to the store position, the output stores read
clobbered registers.

Fix: build a per-call `force_last_use : tag → position` map.
- For pass 1 outputs: position = end of pass (`n`)
- For pass 2 outputs: position = last position in the tag's cluster
  (= position just before that cluster's flush)
- For unclustered pass 2 outputs: position = end of pass

The pass 2 cluster-aware lifetimes are essential. Forcing all 128
pass 2 outputs alive to end-of-pass would need 128+ registers; with
cluster-aware lifetimes, peak fits in ~25 ZMM (within the 28-budget
discussed for M3a).

## Bug-hunting methodology that worked

Once initial output diverged at sample 16+, the productive move was
to write a small Python tool that parses the emitted C and detects
**use-after-clobber**: cases where a register is pinned to a new
variable while the previous binding is still being referenced.

First version flagged 1190 false positives (uses BEFORE clobber, which
are legitimate). Fixing to flag only uses AFTER clobber revealed pass 2
output stores as the issue. Adding scope-awareness (so pass 2's reload
of `t760` doesn't get confused with pass 1's `t760`) yielded clean
"bug = real lifetime miscount" reports.

The detector remains useful for validating future changes —
`/tmp/find_bad_reuse_v3.py` is in the work tree.

## What's NOT in M3a (deferred to M5+)

- **AVX2 support.** AVX2 has 16 YMM budget; pass 1 peak_live = 20
  already overflows. Requires pre-spilling first.
- **R=128, R=256.** Pass 1 peak_live exceeds 28 budget at these radixes
  (M2 measurements: 37, 37 respectively). Requires pre-spilling.
- **Cross-pass register sharing.** Currently passes 1 and 2 allocate
  independently; values crossing the pass boundary go through the
  existing scratchpad mechanism. Could be sharper.
- **The +122 unfused loads.** A measurable cost of the barrier
  approach: gcc can't fuse `vfmadd ..., [mem]` when the input is
  pinned via barrier. Possible mitigation: barrier only on values
  whose specific register matters (multi-use intermediates), let
  single-use immediate loads fold. This is an M5 optimization.
- **t1_dit_log3 borderline overflow.** Pass 1 peak_live = 33 vs
  budget 28. One register over. Could be fixed by rematerializing
  one Const broadcast, or by raising budget to 31 (still leaves
  zmm31 for gcc).

## How to apply

The three files in this folder are drop-in replacements:
- `lib/regalloc.ml` — replaces the M1/M2 version with M3a allocator
- `lib/emit_c.ml` — adds the install_alloc plumbing and `current_regalloc` ref
- `lib/isa.ml` — adds `pinned_reg_decl`

Usage:
```bash
# Default — byte-identical to pre-M3a
./_build/default/bin/gen_radix.exe 64 --twiddled --in-place --isa avx512 --emit-c

# M3a regalloc enabled
VFFT_USE_REGALLOC=1 ./_build/default/bin/gen_radix.exe 64 --twiddled --in-place --isa avx512 --emit-c

# Stderr will show: regalloc: NNN tags bound (per pass)
```

The opt-in env-var design means M3a can ship as an experimental flag
and be tested on selected codelets before becoming default.
