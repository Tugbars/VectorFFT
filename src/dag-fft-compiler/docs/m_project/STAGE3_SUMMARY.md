# Stage 3 Complete: Canonical Regalloc Input Type

## What was added

Three things in `lib/regalloc.ml`:

1. **A canonical input type** `regalloc_input`:
   ```ocaml
   type regalloc_input = {
     scheduled : Algsimp.t list;
     inline_set : (int, unit) Hashtbl.t option;
     force_last_use : (int, int) Hashtbl.t option;
   }
   ```
   Bundles the three parameters that historically were passed as separate
   optional arguments to `allocate`.

2. **`prepare_for_simple_codelet`** — a prep function for straight-line
   codelets without the cluster-spill pass structure (primes, n1, strided).
   Dedupes the raw scheduled list (handling the duplicate-entry pattern
   that broke M7), builds `force_last_use` mapping output tags to
   end-of-schedule, and returns a `regalloc_input` guaranteed to satisfy
   I1 and I5.

3. **`prepare_for_simple_codelet_from_oref`** — convenience wrapper
   accepting raw `(Expr.elem_ref option * Algsimp.t) list` directly from
   SU/Bisection scheduler output. Flattens via `List.map snd` and dedupes.

Net change: 104 lines added to `regalloc.ml`. No code in `emit_c.ml`
touched. The cluster-spill recipe continues to construct its
`regalloc_input` inline (since its prep is deeply entwined with
`classify_passes` / cluster-flush positions / per-cluster SU subsetting,
and is already proven correct via the 9-case regression suite).

## Design decision: two prep paths, one output type

The original Stage 3 plan envisioned a single universal prep function.
Empirically that was the wrong abstraction:

- The cluster-spill recipe's prep is **structurally entangled** with
  `classify_passes` (which depends on `spill_info`), `min_slot` ordering
  (which depends on the spill array layout), and `flush_pos_for_cluster`
  (which depends on per-cluster output assignment placement). Extracting
  these as parameters of a universal prep function would have produced a
  function with ~10 mutually-dependent arguments — worse than the
  current inline construction.

- The simple-codelet path needs **none of that machinery**. Its prep is
  ~15 lines: dedupe scheduled, build force_last_use for output tags.

The right abstraction: keep two prep paths, share the output type. The
cluster-spill recipe and `prepare_for_simple_codelet` both produce
`regalloc_input` values that the allocator consumes uniformly.

This also means **Stage 3 is non-invasive**: no existing call site is
modified. The 9 regression cases produce the same emit output as before.

## Verification

**1. 9-case regression suite passes.** All composite codelets continue to
work; cluster-spill's inline prep unchanged.

**2. Stage 3 self-test (`bin_test/stage3_test.ml`) passes 6 checks:**

```
=== Stage 3 sanity check ===
raw length=6, dedup length=4 (expect 4)
I1 dedup: PASS (all 4 entries unique)
First occurrence preserved: PASS
Regalloc.allocate on prep output: PASS
I1 fires on raw duplicates: PASS
oref variant dedupes: PASS
=== All Stage 3 tests passed ===
```

The test constructs a synthetic IR with intentional duplicates (mimicking
the SU scheduler's `(None, e)` + `(Some oref, e)` pattern that broke M7),
runs it through `prepare_for_simple_codelet`, and verifies:
- Dedup correctness (6 entries → 4)
- I1 compliance (every tag unique)
- First-occurrence preservation (def position kept)
- Round-trip through `Regalloc.allocate` without firing assertions
- I1 assertion fires when raw duplicates are passed directly (confirming
  the prep is doing real work, not a no-op)
- The oref variant handles the SU-style `(oref_opt, e) list` shape

## What this unlocks (Stage 4)

The prime/n1 path in `emit_c.ml` can now be wired up correctly:

```ocaml
(* In each branch of the (match scheduler with ... | None -> ...) block: *)
| SU uarch ->
  let scheduled_raw = Schedule.su_schedule uarch assigns in
  let inline_set = compute_inline_set assigns in
  let input = Regalloc.prepare_for_simple_codelet_from_oref
    ~raw_scheduled:scheduled_raw ~assigns
    ~inline_set:(Some inline_set) () in
  (* Pass to allocate via install_alloc, which now uses the canonical input *)
  install_alloc_canonical "su_n1" input;
  (* Emit by walking input.scheduled (the deduped list).
     Stores happen AFTER the def loop, like Topological does. *)
  List.iteri (fun pos e ->
    current_emit_position := pos;
    emit_spill_sites buf pos;
    emit_reload_sites buf pos;
    Buffer.add_string buf (render_node_def ... e);
  ) input.scheduled;
  List.iter (fun (lhs, e) -> emit_store buf lhs e) assigns
```

Key property: **emitter and allocator walk the SAME list** (`input.scheduled`),
so the position-space invariant (I2/P1) is satisfied by construction.
That's the architectural fix that makes M7-class bugs structurally
impossible for new callers using the prep function.

## Why this approach is safe to ship

- **No behavior change for existing code.** Cluster-spill recipe untouched.
- **No new behavior enabled yet.** `prepare_for_simple_codelet` is library
  code; nothing calls it from `emit_c.ml` yet. Stage 4 wires it up.
- **The prep functions are pure and testable in isolation.** Stage 3 test
  exercises them with synthetic IR; doesn't depend on the full codelet
  generator.
- **Contract assertions remain active.** Stage 2's I1/I5 checks fire on
  any caller (cluster-spill OR future prime/n1) that violates them.

## Files

- `/mnt/user-data/outputs/regalloc_stage3.ml` — full updated regalloc.ml
- `/mnt/user-data/outputs/regalloc_stage3.diff` — full diff (175 lines: 71 from Stage 2 + 104 from Stage 3)
- `/mnt/user-data/outputs/stage3_test.ml` — self-test (84 lines)

## Tree state after Stage 3

| File | Status |
|---|---|
| `lib/regalloc.ml` | Stage 2 assertions + Stage 3 prep type/functions |
| `lib/emit_c.ml` | pre-M7 baseline (no prime support yet) |
| `bin_test/stage3_test.ml` | new self-test |
| `bin_test/dune` | added stage3_test to executables list |
| `/tmp/emit_c_m7_attempt.ml` | M7 attempt preserved for reference |
| `/tmp/regalloc_pre_m7.ml` | pre-M7 baseline (for revert if needed) |

## Stage 4 plan

Now that the prep function exists, Stage 4 is small and well-bounded:

1. In `emit_c.ml`, replace each `| None -> (match scheduler with ...)`
   branch's emit logic with: call prep → call allocate → walk
   `input.scheduled` for emission → walk `assigns` for stores.

2. Test on R=3 AVX2 (M3a fits — should match the working M7 case).

3. Test on R=3 AVX-512 (the M7 bug case — must now pass because prep
   dedupes the SU output that previously violated I1).

4. Test on R=5..R=19, both ISAs. M5 spill arena must be declared at the
   start of the prime scope (helpers `emit_regalloc_spill_decl` etc.
   from the M7 attempt can be reused).

5. Run correctness diffs and find_bad_reuse_v3.py on each generated
   codelet. Expect: bit-exact or LSB-only diffs, 0 use-after-clobber bugs.

6. Measure runtime to confirm the predicted +10-25% gains on primes.

The remaining risk surface is the spill array emission (M5/M6 helpers)
and the per-node spill_sites/reload_sites loop wiring. Both are
mechanical and don't touch the allocator's contract.

Estimated time for Stage 4: half a day.
