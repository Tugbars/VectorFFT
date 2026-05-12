# 46. What FFTW's algsimp Does That Ours Doesn't

## Method

Cloned FFTW3 sources (`fftw3/genfft/algsimp.ml`, 580 lines vs our 1854) and
walked through each FFTW pass to identify what we have parity on, what we
do differently, and what we lack entirely.

The user-visible function in FFTW is `AlgSimp.algsimp : expr list -> expr list`.
The driver does memoized recursive traversal applying simplification rules
bottom-up, with two parallel hash tables — one for syntactic structure
(simp table) and one for CSE (CSE table).

## What we have parity on

- **Constant folding**: zero/one/neg-one elimination on Mul, zero
  elimination on Add. ✓ (Our `mk_const`, `mk_mul`, `mk_add`)
- **Commutative Mul canonicalization**: `a*b` and `b*a` hash to same node.
  FFTW does this by checking both orderings in `structurallyEqualCSE`.
  We do it by canonicalizing operand order via tag: `if a.tag ≤ b.tag
  then (a, b) else (b, a)` before hashcons. Equivalent. ✓
- **Sum flattening**: `Add(Add(a, b), c)` → `Add(a, b, c)` as a list of
  signed terms. FFTW does this via `reduce_sumM` + n-ary `Plus`. We do
  it via `flatten_sum` returning `(sign, term)` list. ✓
- **Sign cancellation**: `a + (-a)` → 0. FFTW does this in `reduce_sumM`.
  We do it in `cancel_signs`. ✓
- **`-(-a)` → a**: Both have this trivially.
- **Const + Const collapsing**: FFTW's `reduce_sumM` and our
  `flatten_sum` + reconstruction both handle this.

## What FFTW does that we lack (in priority order)

### 1. Oracle-based randomized CSE — the big one

```ocaml
(* in algsimp.ml *)
let randomized_cse = ref true   (* default ON in Magic.ml *)

let hashCSE x = 
  if !Magic.randomized_cse then Oracle.hash x
  else Expr.hash x

let equalCSE a b = 
  if !Magic.randomized_cse then
    (structurallyEqualCSE a b || Oracle.likely_equal a b)
  else
    structurallyEqualCSE a b
```

The Oracle picks random values for all leaves (Load nodes), evaluates the
DAG numerically, and uses the result as the CSE hash key. Two expressions
that compute the same value get the same hash — even if they're
syntactically different.

**Why this matters for FFT**: mathematically equivalent expressions arise
constantly in butterflies. For instance, `(a - b) * cos(θ) + (a + b) * sin(θ)`
and `cos(θ)*a - cos(θ)*b + sin(θ)*a + sin(θ)*b` evaluate to the same value
but are structurally distinct. Syntactic CSE misses this; Oracle catches it.

**What we have instead**: pure structural CSE via `hashcons`. We catch
exact matches and commutative variants of Mul, but nothing semantic.

**Implementation cost**: write an `oracle.ml` that:
- Maps each Load to a random `Int64` value
- Walks the DAG recursively, computing a numerical hash via the
  same operations
- Uses the result `mod hash_table_size` as the CSE bucket

The complication is FP non-associativity: `(a+b)+c` and `a+(b+c)` would
give slightly different numerical results, defeating the purpose.
FFTW's `oracle.ml` works around this by using **exact rational
arithmetic** (a custom `Number` type) — slower per-evaluation but
exact equality.

**Expected impact**: 5-15% reduction in DAG node count after CSE for
the larger codelets (R≥32). Would translate to ~3-8% fewer instructions
since the deduplicated nodes are computed once instead of N times.

### 2. `collectM` factors any shared first factor, not just constants

FFTW's collect:
```ocaml
let collectible1 = function
  | NaN _ -> false
  | Uminus x -> collectible1 x
  | _ -> true   (* ANYTHING except NaN is a collectible coefficient *)
```

So `a*x + a*y` factors to `a*(x+y)` whether `a` is a constant or any
other expression. Our `factor_common_muls` only fires when one operand
is `NK_Const`:

```ocaml
(* in algsimp.ml *)
let const_mul_of (n : t) : (t * float) option =
  match n.node with
  | NK_Mul (a, b) ->
    (match a.node, b.node with
     | NK_Const c, _ -> Some (b, c)
     | _, NK_Const c -> Some (a, c)
     | _ -> None)
```

**Why we made this choice (rational)**: in FFT, the natural "coefficients"
are twiddles (constants). Non-constant factoring requires the shared
factor to be cheap enough to compute that re-computing isn't a win. For
FFT codelets, after twiddle CSE, most remaining shared factors aren't
single-expression — they're sub-DFT partial results that are already
handled by `share_subsums`.

**Could be wrong**: there may be patterns we miss. Worth trying a
non-constant variant of `factor_common_muls` and seeing if any radix
shrinks.

### 3. `deepCollectM` with depth=1

```ocaml
let mangleSumM x = returnM x
    >>= reduce_sumM 
    >>= collectM (fun (a, b) -> (a, b))
    >>= collectM (fun (a, b) -> (b, a))      (* two collect passes *)
    >>= reduce_sumM 
    >>= deepCollectM !Magic.deep_collect_depth
    >>= reduce_sumM
```

Where `deep_collect_depth = 1` by default. The pass looks ONE level into
nested Plus structures for shared subexpressions:

> "simplify patterns of the form ((c_1 * a + ...) + ...) + (c_2 * a + ...)
> A common case is the butterfly (a + b) + (a - b)."

Our `flatten_sum` flattens nested Add/Sub/Neg recursively to one big
flat list, then operates on that. FFTW's `deepCollect` preserves inner
Plus nodes as units and looks for matches at depth 1.

**Why the FFTW approach can catch things we miss**: if `(a + b)` appears
as an inner Plus inside a larger sum AND also appears separately
elsewhere in the DAG, our flattening destroys the inner-Plus identity
before `share_subsums` runs, so we can't share it.

**Why our approach can catch things FFTW misses**: with everything
flattened, `share_subsums` searches a larger space and can find sharing
that FFTW's depth-limited search misses.

These are different tradeoffs. Worth empirically comparing on real
codelets — generate the same R=64 with our pipeline vs. FFTW and
compare node counts after CSE.

### 4. `generateFusedMultAddM` — pre-FMA factorization

```ocaml
let separate l = (* split sum into terms with const coefficients vs not *)
  ...
in
if !Magic.enable_fma && count is_multiplication l >= 2 then
  let (w, wo, max) = separate l in
  snumM (Number.div Number.one max) >>= fun invmax' ->
    snumM max >>= fun max' ->
      mapM (fun x -> stimesM (invmax', x)) w >>= splusM >>= fun pw' ->
        stimesM (max', pw') >>= fun mw' ->
          splusM (wo @ [mw'])
```

When a sum has ≥2 multiplications by constants, FFTW factors the largest
coefficient out: `c1*a + c2*b + c3*c → c_max * (c1/c_max * a + c2/c_max
* b + c3/c_max * c)`. This produces FMA-friendly structures and reduces
the dynamic range of constants in the inner expression.

**Crucial caveat**: `enable_fma = false` by default in FFTW. They turn
this OFF and let the compiler handle FMA fusion. This matches doc 28's
finding for composites: explicit FMA atoms regress composite codelets
(R=32 t1_dit llvm-mca SKX 312→226 cycles when FMA is DISABLED).

So FFTW's `generateFusedMultAddM` isn't actually used in their default
pipeline. We have `fma_lift` gated to primes only (doc 28) — different
mechanism, same conclusion.

### 5. `Uminus` auto-memoization

```ocaml
let identityM x =
  let memo x = memoizing lookupCSEM insertCSEM returnM x in
  match x with
    Uminus _ -> memo x 
  | _ -> memo x >>= fun x' -> memo (Uminus x') >> returnM x'
```

When FFTW caches `x`, it also pre-emptively caches `Uminus x`. So the
first time someone asks for `-x`, the CSE table already has it.

**Impact**: minor — saves a hashcons round-trip on negation. We don't
have this; each `mk_neg` call hashcons separately.

### 6. `reorder_uminus` — canonical sum ordering

```ocaml
let rec reorder_uminus = function
  | [] -> []
  | ((Uminus _) as a' :: b) -> (reorder_uminus b) @ [a']
  | (a :: b) -> a :: (reorder_uminus b)
```

Pushes all `Uminus` terms to the end of a sum. So `a + (-b) + c + (-d)`
becomes `a + c + (-b) + (-d)`. Standardizes Plus list ordering for
syntactic CSE.

**Impact on us**: we sort sum terms by canonical key in `emit_pair_fold`
which gives a similar (but different) canonicalization. Whether it
catches all the same cases would need a direct comparison.

## What we have that FFTW doesn't

For completeness:

1. **Conjugate-pair direct DFT for primes** (doc 23). FFTW uses a
   different decomposition for direct primes — they go via Rader's
   algorithm for primes ≥ 13. Our conjugate-pair gives R=11 in 190 ops
   vs FFTW's Rader at ~250 ops for the equivalent. Win for us at small
   primes.

2. **`share_subsums` on flattened sums**. FFTW's `deepCollectM`
   limited to depth=1 doesn't search as broadly. Win for us on patterns
   that flattening exposes.

3. **Spill controller + SU+GH scheduler with recipe** (docs 09-13). FFTW
   has its own scheduler (`schedule.ml`) using a different
   register-aware algorithm. We've empirically shown the recipe + SU
   wins on R≥16 in our setting. Different choice, both reasonable.

4. **`fma_lift` gated to primes only** (doc 28). FFTW's
   `generateFusedMultAddM` is gated by `Magic.enable_fma` (default
   OFF). Same conclusion via different mechanisms.

5. **`-flive-range-shrinkage` integration** with gcc-11 (doc 38). FFTW's
   codelets are compiled with whatever compiler the user has; no
   compiler-specific tuning in the gen.

## What to do about it

Three tiers of investigation, in priority order:

**Tier 1 — randomized CSE / Oracle**: highest expected impact.
Implementation cost ~200-400 LOC OCaml (new `oracle.ml` module +
integration with hashcons). Hardest part is FP non-associativity —
either use rational arithmetic (slow but exact) or use a fixed
deterministic FP evaluation order (fast but may miss equivalences).
FFTW's choice: rational arithmetic. Reasonable for ~30-40K-line
generated codelets where Oracle eval happens once per node.

Diagnostic test: generate R=64 with and without Oracle, compare DAG
node count. If <5% reduction, not worth doing. If 10%+, valuable.

**Tier 2 — extend `factor_common_muls` to non-constant factors**.
Implementation cost ~50 LOC OCaml. The existing pass works; just
remove the `const_mul_of` constraint and let any shared first factor
fire.

Diagnostic: generate R=32, R=64 with the extension. Look for codelets
where node count drops. Risk: over-factoring may CREATE shared
subexpressions that aren't worth materializing (re-computation
cheaper than the extra store/load). Some heuristic on minimum
factor size needed.

**Tier 3 — deep collect at depth=1**. Implementation cost ~100 LOC.
Modify `flatten_sum` to preserve inner Plus nodes when share_subsums
follows. Trade off the all-flat approach's broader search for
FFTW's structure-preserving narrower search.

Diagnostic: same — generate, count nodes, compare.

## Honest assessment

The biggest potential gain is the Oracle / randomized CSE. The
other items are incremental. Whether the Oracle is worth implementing
depends on:

1. Does our generated R=64 codelet have many syntactic-different-but-
   semantically-equal expressions that our pure-syntactic CSE misses?
   Unknown — never measured.
2. Is the implementation cost worth the gain on production codelets?
   At ~3-8% expected fewer instructions, on an already-tight pipeline,
   probably yes if it doesn't take more than a few days.

A single diagnostic experiment would resolve this: implement a
minimal Oracle (random FP evaluation, accepting some false negatives)
in 50-100 lines and check if it finds new CSE opportunities on R=64.
If yes, invest in the rational-arithmetic version. If no, drop.

This is a research-effort question, not a "fix this bug" question.
Worth a small experiment but not urgent.
