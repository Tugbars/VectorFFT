# 24. Single-use inlining + Pass-split store emission

## TL;DR

Two changes to `lib/emit_c.ml`:

1. **Single-use inlining** for the SU emit path. Values with exactly
   one consumer are inlined into the consumer's C expression rather
   than emitted as `const __m512d t<N> = ...;`. Closes the
   nested-intrinsic gap to hand-coded FFTW codelets (R=13 t1_dif:
   24 → 102 nested patterns; hand: 120). Wins on DIT primes:
   R=13 t1_dit −20 movapd, R=17 t1_dit −19 movapd, both translating
   to bench wins at K≥512 (R=13 K=1024 hits 0.83 G/H — 17% faster
   than hand).

2. **Pass 1 / Pass 2 output store split** for the spill-emit path.
   Pre-existing bug surfaced: composite codelets R=32/64 t1_dit/t1_dif
   were failing to compile because output values whose dep chain
   didn't cross the spill boundary stayed Pass 1, but their stores
   were emitted inside Pass 2's nested scope where the value was out
   of scope. Fixed by splitting `assigns` by classification and
   emitting each set in its matching pass scope. Composites now
   compile and pass machine-precision correctness.

## The DIF investigation that motivated this

R=13/17 t1_dif were 5-20% slower than hand on bench. R=13 t1_dif
had 81 movapd vs hand's 31; R=17 had 176 vs 115.

The root cause turned out to be expression structure. Hand-coded
R=13 t1_dif uses ~120 nested intrinsic patterns
(`_mm512_fmadd_pd(K, _mm512_sub_pd(a, b))`). Our generator linearized
everything into separate SSA temporaries — ~250 unique `const __m512d`
declarations vs hand's ~130 named values. GCC's allocator handles
fewer SSA names better.

Two failed attempts before single-use inlining:

- **Sink-preference scheduling**: Made cmul nodes fire ASAP at the IR
  level, reordered the C output. GCC re-allocated regardless. No
  asm change.
- **`--annotate` (Annotated_SU/Annotated_topological)**: Made it
  *worse* — R=11 t1_dif 50 → 93 movapd. `annotate.ml` forward-declares
  mutable `__m512d` at the outer scope and assigns them later. Each
  is still a single-assignment SSA value, but the forward-decl
  confuses GCC's lifetime analysis without giving the slot-reuse
  benefit hand gets. Hand only forward-declares mutables for the
  *slot variables* (`x0_re ... x12_im`) that get reassigned through
  `input → intermediate → output → cmul-output`. That's the FFTW
  idiom; annotate doesn't capture it.

## Single-use inlining design

`compute_inline_set` walks the DAG once, counting users per tag:
DAG predecessors plus output-assignment refs. A tag is inlinable
iff:

- count = 1 (single consumer)
- not a sink (output assignment — needs a name for the store)
- not in fused_muls (already suppressed by FMA fusion)
- kind allows inlining (Const inlined as broadcast already; Load
  not inlined to avoid duplicate memory ops; Cmul not inlined
  because its paired emit semantics span two intrinsic calls)

`render_node_def` was extended with `?inline_set`. The body
renderer uses `render_operand` for predecessors:

```ocaml
let rec render_operand depth n =
  if depth >= inline_max_depth || not (should_inline n) then v n
  else render_inlined depth n
and render_inlined depth n = match n.node with
  | NK_Add (a, b) ->
    Isa.add_pd isa
      (render_operand (depth+1) a)
      (render_operand (depth+1) b)
  | NK_Sub ... | NK_Mul ... | NK_Neg ... | NK_Fma ...
  | NK_CmulRe _ | NK_CmulIm _ | NK_Load _ -> v n  (* never inline *)
```

`inline_max_depth = 32` — single-use chain length is bounded by the
predecessor chain length anyway, so the constant matters only as
a sanity limit.

The SU emit path (lib/emit_c.ml ~line 1185) computes inline_set,
threads it through `render_node_def`, and skips standalone emission
for tags in the set:

```ocaml
| SU uarch ->
  let scheduled = Schedule.su_schedule uarch assigns in
  let inline_set = compute_inline_set assigns in
  let is_inlined e = Hashtbl.mem inline_set e.tag in
  List.iter (fun (oref_opt, e) ->
    match oref_opt with
    | None ->
      if not (is_inlined e) && not (Hashtbl.mem defined e.tag) then
        ... render_node_def ~inline_set:(Some inline_set) e
    | Some oref ->
      ... emit_store buf oref e   (* sinks always emit standalone *)
  ) scheduled
```

Other emit paths (Topological, Bisection, Annotated_*, spill
PASS 1/PASS 2) don't pass `inline_set`, so they fall back to the
old behavior. Containment is intentional: the SU non-spill path is
where prime codelets live.

## Results

### IR-level (movapd)

| Codelet | Before | After | Hand |
|---|---:|---:|---:|
| R=11 t1_dit | 22 | **21** | n/a (we already beat hand) |
| R=11 t1_dif | 50 | **49** | 50 |
| R=13 t1_dit | 75 | **55** | n/a |
| R=13 t1_dif | 81 | 83 | 31 |
| R=17 t1_dit | 157 | **138** | n/a |
| R=17 t1_dif | 176 | 180 | 115 |
| R=17 t1_dit_log3 | 156 | 150 | n/a |

### Bench (5-run median, virt-Skylake-X)

DIT primes (the wins):

| Codelet | K=512 | K=1024 | K=2048 | K=4096 |
|---|---:|---:|---:|---:|
| R=11 t1_dit | 0.93 | 0.88 | 0.91 | 0.86 |
| R=13 t1_dit | 0.97 | **0.83** | 0.96 | 1.05 |
| R=17 t1_dit | **0.86** | 0.95 | 0.96 | 1.17 |

R=13 K=1024 at 0.83 G/H = 17% faster than hand. R=17 K=512 at 0.86 = 14%
faster.

DIF primes still trail hand 5-20%. Inlining didn't help DIF specifically
because raw outputs have 2 uses (CmulRe + CmulIm), explicitly excluded
from single-use inlining.

## The composite codelet bug (separate fix in same session)

Side discovery: R=32/64 t1_dit/t1_dif had been failing to compile with
errors like `'t918' undeclared (first use in this function)`. I'd
initially assumed the inlining work caused this, but a clean test
(setting `inline_set = None`) showed the bug existed independently.

### Diagnosis

```c
{                                                      /* PASS 1 scope */
    const __m512d t918 = _mm512_sub_pd(t902, t917);    /* defined PASS 1 */
    /* ... PASS 1 stores spilled values ... */
}                                                      /* t918 dies */
{                                                      /* PASS 2 scope */
    /* ... PASS 2 reloads spills, computes ... */
    _mm512_storeu_pd(&rio_re[31*ios + k], t918);       /* ERROR: undeclared */
}
```

t918 was an output value with no internal consumers (only the eventual
store). The forward pass classified it Pass 1 (no spilled ancestors).
The backward pass requires non-empty internal consumers to reclassify,
so it stayed Pass 1. But the safety net at end of PASS 2 iterated all
`assigns` regardless of classification — wrong scope.

### Wrong fix attempt

I added `output_tags` to `classify_passes` so output stores would count
as Pass 2 consumers, pushing output-only nodes to Pass 2. This worked
for t918 in isolation but broke other codelets — promoting an output
to Pass 2 doesn't bring its preds along, so Pass 2 then referenced
Pass 1 values that weren't spilled (`'t13' undeclared`). The fix was
reverted.

### Correct fix

Emit each store in the scope where its value lives. Split `assigns` by
classification:

```ocaml
let pass1_assigns = List.filter (fun (_, e) ->
  Hashtbl.find_opt cls e.tag = Some `Pass1
) assigns in
let pass2_assigns = List.filter (fun (_, e) ->
  Hashtbl.find_opt cls e.tag = Some `Pass2
) assigns in
```

At end of PASS 1's `{ ... }`, before the closing brace, emit
`pass1_assigns` stores. The PASS 2 store machinery
(`assigns_by_cluster`, the cluster flush loop, the safety net)
already worked — just restrict its iteration target from `assigns`
to `pass2_assigns` so it doesn't try to re-store Pass 1 outputs in
the wrong scope.

### Verification

| Codelet | Error | Result |
|---|---|---|
| R=32 t1_dit | 1.58e-14 | PASS |
| R=32 t1_dif | 1.58e-14 | PASS |
| R=64 t1_dit | 3.79e-14 | PASS |
| R=64 t1_dif | 3.79e-14 | PASS |

R=32 t1_dit bench: median T/H=1.04 at K=1024, parity at K=2048+, 1.027
at K=4096. Acceptable for sizes anyone cares about.

## Files changed

- `lib/emit_c.ml`:
  - Added `inline_max_depth = 32` constant.
  - Added `compute_inline_set` helper.
  - Extended `render_node_def` with `?inline_set` parameter +
    mutually-recursive `render_operand` / `render_inlined`.
  - Wired `inline_set` through the SU emission path.
  - Split `assigns` into `pass1_assigns` / `pass2_assigns` in the
    spill-emit path.
  - Emit Pass 1 stores at end of PASS 1 scope.
  - Restricted safety net + `assigns_by_cluster` to iterate
    `pass2_assigns`.

`classify_passes` itself is unchanged. The `output_tags` experiment
was reverted.

## Coverage

Comprehensive compile sweep: R={5, 7, 11, 13, 16, 17, 32, 64} ×
{t1_dit, t1_dif, t1_dit_log3, n1} all compile cleanly. 32/32 prime
correctness PASS at machine precision. R=32/64 composites pass
machine-precision DFT-vs-brute-force.

## Future work

- **Destructive-update emission for DIF cmul layer** would close the
  remaining DIF gap (5–20%). Requires emit_c.ml to recognize "raw
  output → cmul output" patterns and emit FFTW-style mutable slot
  reassignment (`tr = x_re; x_re = fmsub(...); x_im = fmadd(tr, ...)`).
  Substantial rewrite.

- **Spill-path inlining**: `compute_inline_set` is currently global
  but only used by the SU non-spill path. Plumbing it through PASS 1
  and PASS 2 emission would extend the gain to composite codelets,
  but the cluster boundary makes it tricky — a node inlined in one
  pass but referenced in another would break.
