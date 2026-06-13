# SSA Register Allocator — M1 + M2 Progress Report

**Status: M1 ✓, M2 ✓.** Module in place, types coherent, peak-live
analysis works and produces meaningful data on the canary panel.

## What M1 delivered

`lib/regalloc.ml` (file went from 1-line placeholder → 226 lines):

```ocaml
type assignment = Reg of string | Default
type allocation = { isa : Isa.t; assign : (int, assignment) Hashtbl.t }
val allocate_stub : isa:Isa.t -> scheduled:Algsimp.t list -> allocation
val lookup        : allocation -> int -> assignment
val count_bindings: allocation -> int * int
```

Validation:
- Build clean
- `bin_test/m1_test.exe` exercises: build stub, inject Reg, lookup
  round-trips, count_bindings reports correctly
- Codelet generation byte-identical pre/post M1 (r16 strided AVX-512
  diff empty; r64 t1_dit reproducible)
- No changes to emit_c.ml, gen_radix.ml, or any other module

## What M2 delivered

### Algorithm

`peak_live_analysis : isa:Isa.t -> scheduled:Algsimp.t list -> live_info`

Two-pass:
1. Walk schedule forward computing `last_use[tag] = max position
   where tag is read (as pred) OR defined`.
2. Walk again maintaining `live` set. At each position i: add defined
   tag to `live`, snapshot `|live|` (potential peak), then kill any
   tag whose `last_use == i`.

The output is `{ peak_live; peak_at; n_nodes; budget; fits }`. The
naive analysis counts every scheduled tag — overestimate, since
emit_c's render_node_def inlines single-use operands and gcc DCEs the
resulting dead declarations. M3 will refine.

### emit_c wiring

Three additions to emit_c.ml, all gated on `VFFT_PEAK_LIVE=1` env var:
1. `record_peak_live` helper inside `emit_codelet` (lines 522-548)
2. Hooks at 6 body-scheduler sites:
   - Topological emit (line 1403, 1450)
   - bisection_schedule emit (line 1424, 1469)
   - su_schedule emit (line 1509, 1536)
3. Hooks at 2 spill-path per-pass sites:
   - pass1_blocked closure
   - pass2_ordered closure

Default (no env var): byte-identical output. With env var:
diagnostic-only stderr.

### Canary results (the real M2 deliverable)

**Two distinct regimes appeared in the data:**

#### Regime 1: Topological / OOP (no spill_info)
Peak grows as ~2× butterfly count. R=16 OOP already hits naive peak 64
— well beyond 32 ZMM. **But this is naive — emit_c inlines heavily,
so gcc sees much less actual register pressure.** M3 needs an
inline-aware refinement to get an honest reading here.

| ISA      | Radix | n_nodes | peak | budget | fits |
|----------|-------|---------|------|--------|------|
| AVX-512  | 4     | 24      | 13   | 32     | ✓   |
| AVX-512  | 8     | 73      | 30   | 32     | ✓   |
| AVX-512  | 16    | 202     | 64   | 32     | ✗ (naive) |
| AVX-512  | 32    | 529     | 132  | 32     | ✗ (naive) |
| AVX-512  | 64    | 1304    | 267  | 32     | ✗ (naive) |
| AVX2     | 4     | 24      | 13   | 16     | ✓   |
| AVX2     | 8     | 73      | 30   | 16     | ✗ (naive) |

#### Regime 2: t1_dit with spill_info (the doc-28-class case)
Per-pass peak is **bounded by the structural spill**, NOT by codelet
size. The pass1/pass2 split keeps each pass's live set in a regime
chordal coloring can handle.

| ISA      | Radix | Pass1 peak | Pass2 peak | Fits 32 ZMM? |
|----------|-------|------------|------------|--------------|
| AVX-512  | 32    | 20         | 13         | both ✓      |
| AVX-512  | 64    | **20**     | **25**     | both ✓ ★    |
| AVX-512  | 128   | 37         | 25         | pass1 short by 5 |
| AVX-512  | 256   | 37         | 51         | both overflow |
| AVX2     | 32    | 20         | 13         | pass1 needs help |
| AVX2     | 64    | 20         | 25         | both need help |

★ **R=64 t1_dit AVX-512: pass1=20, pass2=25.** Both per-pass peaks
fit in 32 ZMM with room to spare. This is the doc-28 target codelet
and the most-used t1_dit shape. **M3 can attack this directly with
chordal coloring, no pre-spilling needed.**

## What M2 tells us about M3 staging

Original plan: "R=4 through R=32." Data says we can be more
ambitious:

**Revised M3 scope: R=32 + R=64 t1_dit on AVX-512.**

Reasoning:
- R=32 t1_dit: pass1=20, pass2=13 — chordal coloring fits trivially
- R=64 t1_dit: pass1=20, pass2=25 — chordal coloring fits with room
- R=64 t1_dit is the doc-28 regression target, so M3 win is directly
  measurable against the existing scheduler
- Both use the spill-path which already produces bounded per-pass
  schedules — no need to fight the OOP path's enormous naive peaks

**Deferred to M5: pre-spilling for R=128+ and AVX2.** R=128 pass1
overflows by 5, R=256 overflows by 19. Standard pre-spill (evict
latest-next-use values to scratchpad slots) handles this.

**Deferred to M3.5: inline-aware analysis** for OOP/Topological
codelets. Without it the naive numbers for R≥16 OOP are misleading.
The fix: replicate emit_c's `compute_inline_set` decision and
subtract inlined tags from the live set.

## File deliverables in this folder

| File                  | Purpose                                       |
|-----------------------|-----------------------------------------------|
| `regalloc.ml`         | Full module (226 lines), M1+M2 code           |
| `regalloc.ml.diff`    | Diff from 1-line placeholder                  |
| `emit_c.ml.diff`      | Diff from strided-session emit_c (M2 hooks)   |
| `m1_test.ml`          | M1 sanity test (types, stub, lookup)          |
| `m2_test.ml`          | M2 sanity test (hand-built schedules)         |
| `canary_results.txt`  | Full panel of peak_live measurements          |

## How to use

```bash
# Build and run sanity tests
dune build
./_build/default/bin_test/m1_test.exe
./_build/default/bin_test/m2_test.exe

# Default codelet generation (byte-identical to pre-M1)
./_build/default/bin/gen_radix.exe 64 --twiddled --in-place --isa avx512 --emit-c

# With peak-live diagnostic
VFFT_PEAK_LIVE=1 ./_build/default/bin/gen_radix.exe \
    64 --twiddled --in-place --isa avx512 --emit-c 2>/tmp/peak.log

# Diagnostic appears on stderr; stdout is unchanged C output
```

## What's next (M3 sketch)

The M3 implementation, with concrete code shape:

1. **`peak_live_analysis_with_inline`** — extend M2's function with an
   optional `inline_tags : (int, unit) Hashtbl.t` parameter; tags in
   the set don't enter the live set (their value never gets a
   separate register). Replicates the gcc-DCE effect at analysis
   time.

2. **`build_interference_graph`** — given the schedule + inline set,
   build the chordal interference graph. Each tag is a node; tags
   that are simultaneously live interfere.

3. **`color_chordal`** — greedy coloring on a perfect elimination
   ordering. For chordal graphs this is provably optimal (uses
   ≤chromatic_number colors). Returns the tag → register name map.

4. **Refine `allocate`** — wrap the above into the M3
   `Regalloc.allocate : isa -> scheduled -> spill_info option ->
   allocation`. For R=64 t1_dit, this should produce ≤25 distinct
   ZMM bindings (one per simultaneously-live value at peak).

5. **emit_c consumes the allocation** — line 262 of emit_c.ml is
   `Isa.const_decl isa (v e) body`. M3 changes this to check
   `Regalloc.lookup` and emit `register %s tN asm("%s") = %s;` when
   a binding exists, else fall through to `const`.

6. **Validate**: generate R=64 t1_dit AVX-512 codelet with M3 RA.
   Confirm it compiles (gcc-11). Check objdump for the expected ZMM
   bindings. Bench vs current scheduler — should be at least equal,
   ideally faster because we eliminated gcc's RA from the
   compiler-agnostic variance.

Estimated effort for M3: 3-4 days (per the original 14-18 day total
estimate for M1-M6).
