(* ir.ml — the hash-consed DAG IR: node type, smart constructors, CSE.
 *
 * The bottom of the rewrite stack. Two responsibilities:
 *
 *   1. ALGEBRAIC SIMPLIFICATION AT CONSTRUCTION: the smart constructors
 *      (mk_add, mk_mul, ...) fold trivial operations like x*0 = 0,
 *      x*1 = x, x+0 = x, x-x = 0, etc. Also canonicalize floating-point
 *      noise like cos(pi/2) = 6e-17 (mathematically zero, computationally
 *      tiny) into exact zero.
 *
 *   2. COMMON SUBEXPRESSION ELIMINATION (CSE): identify subtrees that
 *      appear multiple times and share them. For DFT-N this finds the
 *      Cooley-Tukey butterfly structure mechanically, without it being
 *      programmed in.
 *
 * The CSE trick is hash-consing: smart constructors (mk_add, mk_mul, ...)
 * intern every newly-built expression, so structurally-equal subtrees
 * become physically the same value. Equality reduces to pointer/tag
 * comparison; CSE is automatic.
 *
 * Frigo's genfft does this. We do the same with one extension: stronger
 * canonicalization of floating-point constants, so the generator is
 * robust to numerical noise from cos/sin computations at radices that
 * aren't pure power-of-two.
 *
 * The rewrite passes that run OVER this IR live in simplify.ml (the
 * algebraic family), fma_passes.ml (the FMA family) and algsimp.ml
 * (spill lifting, butterfly_share_mul, statistics). Algsimp re-exports
 * this module in full, so Algsimp.mk_add etc. remain the canonical
 * spellings at every call site.
 * ------------------------------------------------------------------
 * MODULE CARD (ir.ml — grep "MODULE CARD" for the full set)
 * ROLE: node_kind / t, the hashcons table (the CSE mechanism itself),
 * tag counter, preds, topo_sort_reachable, the mk_* smart-constructor
 * recursion group, and the Expr lift of_expr / of_assignments + reset.
 * PIPELINE: Expr assignments -> of_assignments -> pass stack -> emit
 * PUBLIC SURFACE (measured): zero direct Ir.X references — every
 * consumer reaches these names through the Algsimp facade chain
 * (Ir < Simplify < Fma_passes < Algsimp).
 * DEPS (grep counts, comment mentions included): Expr(13).
 * STATE GOTCHA: hcons_table / next_tag / of_expr_memo are
 * process-global; Algsimp.reset clears them and MUST run before each
 * generation or stale tags from a prior codelet leak into the DAG.
 * ------------------------------------------------------------------
 *)

open Expr

(* === HASH-CONSED IR ===
 *
 * Every hash-consed expression carries a unique integer tag. Two
 * expressions are structurally equal iff their tags are equal — and
 * because of hash-consing, this is automatically the case.
 *)

type node_kind =
  | NK_Const of float
  | NK_Load of elem_ref
  | NK_Neg of t
  | NK_Add of t * t
  | NK_Sub of t * t
  | NK_Mul of t * t
  (* === N-ARY PLUS ===
   *
   * NK_Plus represents a sum of signed terms: [(s_1, t_1); ...; (s_n, t_n)]
   * means s_1*t_1 + ... + s_n*t_n where each s_i ∈ {+1, -1}.
   *
   * Inspired by FFTW's genfft, where `Plus` is a list. The n-ary form enables
   * `collectM`-style simplification ("ax + bx + cx → (a+b+c)x") in one pass,
   * which the binary NK_Add/NK_Sub form cannot express without recursive
   * tree-walking that misses cross-subtree sharing.
   *
   * Invariants (enforced by mk_plus):
   *   1. Length >= 2. Single-term sums collapse to t (or Neg t) at construction.
   *   2. Terms sorted by tag for canonical hash-consing. Sign attaches to the
   *      term, not the position.
   *   3. At most one NK_Const term (constants combined at construction).
   *   4. No nested NK_Plus terms (flattened at construction).
   *   5. NK_Neg terms have their sign absorbed (Neg(x) with sign +1 becomes
   *      x with sign -1).
   *
   * COEXISTENCE WITH NK_Add/NK_Sub:
   *
   * Commit 1 (this commit) introduces NK_Plus but doesn't yet generate it.
   * All existing of_expr / mk_add / mk_sub paths still produce binary
   * NK_Add/NK_Sub. NK_Plus is reachable only via explicit mk_plus calls,
   * which currently no production code makes. This is intentional — it lets
   * us land the type and helpers behind tests with zero risk to existing
   * codelets, then migrate consumers one at a time. *)
  | NK_Plus of (int * t) list
  (* Complex multiply outputs. Treated as opaque atoms by reassoc — the
   * sum-flattening pass does NOT recurse into them, preserving cmul
   * structure that reassoc would otherwise shred.
   *
   * NK_CmulRe(xr, xi, wr, wi) represents (xr*wr - xi*wi)
   * NK_CmulIm(xr, xi, wr, wi) represents (xr*wi + xi*wr)
   *
   * These are split into two single-output nodes so the IR stays
   * single-output throughout. Hash-consing dedups them independently:
   * if two Cmuls have identical operands, both their re and im outputs
   * share. *)
  | NK_CmulRe of t * t * t * t
  | NK_CmulIm of t * t * t * t
  (* Fused-multiply-add atom — represents one of the four FMA variants:
   *
   *   neg_mul=false, neg_add=false:  (a * b) + c    — fmadd
   *   neg_mul=false, neg_add=true :  (a * b) - c    — fmsub
   *   neg_mul=true,  neg_add=false: -(a * b) + c    — fnmadd
   *   neg_mul=true,  neg_add=true : -(a * b) - c    — fnmsub
   *
   * Lifted by the `fma_lift` pass from Add/Sub-of-Mul patterns where the
   * inner Mul has use_count = 1 (single consumer). After lifting, the
   * Mul is "claimed" by the Fma and other passes treat the Fma as opaque
   * — never recursing into it for factoring or subsum sharing.
   *
   * Codegen renders Fma as a single AVX-512 FMA intrinsic, which is one
   * machine instruction per FMA (vs 2 for separate mul + add). This is
   * the difference between our DAG-level "op count" metric and actual
   * post-fusion hardware instruction count. *)
  | NK_Fma of t * t * t * bool * bool

and t =
  { tag : int
  ; node : node_kind
  }

(* Immediate predecessors of a node — the IR sub-expressions referenced
 * by its constructor. Walking these reaches the full DAG.
 *
 * Centralized here because every layer (schedule, classify_passes,
 * cluster propagation, PASS 2 reload tracking, topological sort) needs
 * the same walk. Keep it in sync with `node_kind` above whenever a
 * constructor is added. *)
let preds (e : t) : t list =
  match e.node with
  | NK_Const _ | NK_Load _ -> []
  | NK_Neg a -> [ a ]
  | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> [ a; b ]
  | NK_Plus terms -> List.map snd terms
  | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) -> [ a; b; c; d ]
  | NK_Fma (a, b, c, _, _) -> [ a; b; c ]
;;

(* Reachable-set topological sort, via `preds`. Hash-cons tags are
 * assigned in construction order, so sorting reachable nodes by tag is a
 * valid topological order. Collects only nodes reachable from `roots`.
 *
 * This is the NK_Plus-tolerant traversal (preds handles NK_Plus), the
 * shared base for callers that may see post-migration IR. emit_c keeps
 * its own NK_Plus-fatal topo_sort_reachable for the 11 emission sites
 * that deliberately fail loud on unmigrated NK_Plus (house style); this
 * one is for codelet_oop's Tier-B/C body, which previously hand-copied
 * exactly this loop with a "Mirrors emit_c.topo_sort_reachable" comment. *)
let topo_sort_reachable (roots : t list) : t list =
  let seen : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag)
    then (
      Hashtbl.add seen e.tag e;
      List.iter visit (preds e))
  in
  List.iter visit roots;
  Hashtbl.fold (fun _ e acc -> e :: acc) seen []
  |> List.sort (fun (a : t) b -> compare a.tag b.tag)
;;

(* === NK_PLUS HELPERS ===
 *
 * Commit 1 introduces NK_Plus but does not yet wire it into existing passes.
 * Match sites that don't yet know how to handle NK_Plus call nk_plus_unreachable
 * to fail loudly if NK_Plus ever appears — which it won't until Commit 2 starts
 * migrating consumers, and we'll then replace these by-site as we go.
 *
 * We could instead silently lower NK_Plus to binary NK_Add/NK_Sub at every
 * consume site, but that would hide which consumers haven't been migrated yet.
 * Failing loud is the better default during the migration. *)
let nk_plus_unreachable (site : string) : 'a =
  failwith
    (Printf.sprintf
       "NK_Plus reached site %S which is not yet wired (Commit 2+).      If you see \
        this, a consumer is generating NK_Plus before its readers      are migrated; \
        check the call stack."
       site)
;;

(* === HASH-CONSING INFRASTRUCTURE === *)

let hcons_table : (node_kind, t) Hashtbl.t = Hashtbl.create 1024
let next_tag = ref 0

let hashcons (nk : node_kind) : t =
  match Hashtbl.find_opt hcons_table nk with
  | Some existing -> existing
  | None ->
    let tag = !next_tag in
    incr next_tag;
    let entry = { tag; node = nk } in
    Hashtbl.add hcons_table nk entry;
    entry
;;

(* Lookup-only — returns Some node if it exists in the hash-cons table,
 * None if not. Used by share_subsums to detect pre-existing shareable
 * subexpressions without creating them. *)
let lookup_node (nk : node_kind) : t option = Hashtbl.find_opt hcons_table nk

(* === OF_EXPR MEMOIZATION ===
 *
 * The math layer (Dft.dft_ct etc.) produces Expr trees with high textual
 * redundancy: a single OCaml-allocated Expr value gets referenced many
 * times across PASS 1 / PASS 2 outputs. At R=64 the textual node count
 * is ~95M while the unique post-hashcons count is ~7K — a 13,000×
 * redundancy ratio that grows ~6.5× per doubling of N.
 *
 * Without memoization, of_expr does work proportional to textual count:
 * each textual occurrence triggers a full recursive walk down to atomic
 * Const/Load nodes. This is the O(N⁴) scaling wall observed at R=128
 * (see docs/31_split_radix_research_arc.md and the profile_pipeline
 * diagnostic).
 *
 * The fix: memoize of_expr on physical Expr identity. Multiple references
 * to the same OCaml allocation get processed once. Physical equality
 * (==) catches the dft.ml pattern of `pass1_re.(n1_idx).(k2)` being
 * stored once and read many times — these reads return the same
 * allocation. Structurally-equal-but-different-allocation cases would
 * miss the memo (correct, just no speedup); since they don't happen in
 * dft.ml's construction style, the memo catches essentially all the
 * sharing.
 *
 * Worst case: memo misses → fall back to full re-walk for that subtree.
 * No correctness risk; the smart constructors and hashcons still produce
 * the same final t whether memoized or not. *)
module ExprPhysHash = struct
  type t = Expr.expr

  let equal = ( == ) (* physical equality on the immutable Expr value *)
  let hash = Hashtbl.hash (* bounded-depth structural hash; fast *)
end

module ExprMemo = Hashtbl.Make (ExprPhysHash)

let of_expr_memo : t ExprMemo.t = ExprMemo.create 1024

(* Constant identity map: quantize ONLY the dedup key (14 sig digits) so that
 * numerically-noisy recomputations of the same mathematical constant unify
 * into one node; STORE the first-seen full-precision value. The previous
 * behavior stored the quantized value itself, injecting up to ~4e-14
 * relative error (~22-30 ulp) into every emitted twiddle constant — the
 * accuracy harness measured exactly that against a long-double reference
 * (radix-16/8 chains at 28-76 eps L2 vs MKL's 1-3; radix-4 chains, whose
 * constants are exact, matched MKL). Keyed on the magnitude; sign is
 * canonicalized to a Neg wrapper as before. Cleared by reset(). *)
let const_ident : (string, t) Hashtbl.t = Hashtbl.create 256

let reset () =
  Hashtbl.clear hcons_table;
  ExprMemo.clear of_expr_memo;
  Hashtbl.clear const_ident;
  next_tag := 0
;;

(* === CANONICALIZATION HELPERS === *)

let zero_threshold = 1e-14

let is_zero (e : t) : bool =
  match e.node with
  | NK_Const c -> Float.abs c < zero_threshold
  | _ -> false
;;

let is_one (e : t) : bool =
  match e.node with
  | NK_Const c -> Float.abs (c -. 1.0) < zero_threshold
  | _ -> false
;;

let is_neg_one (e : t) : bool =
  match e.node with
  | NK_Const c -> Float.abs (c +. 1.0) < zero_threshold
  | _ -> false
;;

(* === SMART CONSTRUCTORS ===
 * Each does algebraic simplification first, then hash-consing.
 *)

let mk_const (c : float) : t =
  let rounded = if c = 0.0 then 0.0 else float_of_string (Printf.sprintf "%.13e" c) in
  if Float.abs rounded < zero_threshold
  then hashcons (NK_Const 0.0)
  else if Float.abs (rounded -. 1.0) < zero_threshold
  then hashcons (NK_Const 1.0)
  else if Float.abs (rounded +. 1.0) < zero_threshold
  then hashcons (NK_Const (-1.0))
  else (
    let mag = Float.abs c in
    let key = Printf.sprintf "%.13e" mag in
    let base =
      match Hashtbl.find_opt const_ident key with
      | Some t0 -> t0
      | None ->
        let t0 = hashcons (NK_Const mag) in
        Hashtbl.add const_ident key t0;
        t0
    in
    if c < 0.0
    then
      (* Canonicalize negative non-trivial constants to -|c|.
       * This unifies all multiplications-by-c with multiplications-by-(-c):
       *   Mul(x, -c) → Mul(x, Neg(c)) → Neg(Mul(x, c)) via Neg-hoisting.
       * The underlying Mul(x, c) is then shared by hash-consing.
       * Hand-coded codelets do this manually (e.g. vnc = -vc); we get
       * the same effect mechanically. *)
      hashcons (NK_Neg base)
    else base)
;;

let mk_load (r : elem_ref) : t = hashcons (NK_Load r)

(* === SIGNED-TERM SUM REPRESENTATION ===
 *
 * For reassociation, we view every Add/Sub/Neg chain as a SUM of signed
 * terms: a list of (sign, leaf_expr) pairs where sign is +1 or -1 and
 * leaf_expr is a non-Add/Sub/Neg expression.
 *
 * Example: ((a + b) - (c + d)) + (-e)   becomes
 *          [(+1, a); (+1, b); (-1, c); (-1, d); (-1, e)]
 *
 * The advantage: the structure is canonical regardless of how the user
 * wrote the expression. Two mathematically-equal sums produce identical
 * sorted term lists, and the pair-fold then produces identical hash-consed
 * trees — which means CSE catches shared subsums automatically.
 *
 * The pairing rule is "interleaved" (recursive half-split): for a sorted
 * list of 2k terms, recurse on the even-indexed and odd-indexed halves
 * separately. For radix-4 this exposes the Cooley-Tukey butterfly:
 * with 4 sorted inputs, halves are (input[0], input[2]) and
 * (input[1], input[3]) — exactly the even/odd butterfly structure.
 *)

(* Flatten an expression into a list of (sign, leaf) pairs.
 * Recursively descends through Add/Sub/Neg, accumulating signs. *)
let rec flatten_sum (sign : int) (e : t) : (int * t) list =
  match e.node with
  | NK_Add (a, b) -> flatten_sum sign a @ flatten_sum sign b
  | NK_Sub (a, b) -> flatten_sum sign a @ flatten_sum (-sign) b
  | NK_Neg inner -> flatten_sum (-sign) inner
  | _ -> [ sign, e ]
;;

(* Deeper flatten that ALSO sees through early-peephole NK_Fma nodes.
 *
 * The Sub(Neg(Mul(a,b)), c) → Fma(a, b, c, true, true) peephole in
 * mk_sub_binary fires during dedup_sub_pairs, creating opaque Fma leaves
 * that flatten_sum can't decompose. For deep_collect's distribute-then-
 * collect pipeline we want to look through these to expose the underlying
 * Mul and addend as separate terms — both candidates for collection.
 *
 * NK_Fma(a, b, c, nm, na) decomposes as:
 *   (nm ? -(a*b) : a*b) + (na ? -c : c)
 *
 * Decomposing creates two terms. When deep_collect later emits via
 * mk_sub_binary, the same peephole re-creates the Fma if the pattern
 * survives, so no FMA fusion is permanently lost — only delayed past
 * the collection step.
 *
 * ONLY used inside deep_collect. The existing flatten_sum stays
 * unchanged so other passes' invariants are preserved. *)
let rec flatten_sum_through_fma (sign : int) (e : t) : (int * t) list =
  match e.node with
  | NK_Add (a, b) -> flatten_sum_through_fma sign a @ flatten_sum_through_fma sign b
  | NK_Sub (a, b) -> flatten_sum_through_fma sign a @ flatten_sum_through_fma (-sign) b
  | NK_Neg inner -> flatten_sum_through_fma (-sign) inner
  | NK_Fma (a, b, c, nm, na) ->
    (* Reconstruct the multiplied term and addend as separate signed
     * leaves. The mul term is itself a leaf for flatten purposes
     * (we don't decompose Mul further). *)
    let mul_term = hashcons (NK_Mul (a, b)) in
    let mul_sign = if nm then -sign else sign in
    let add_sign = if na then -sign else sign in
    flatten_sum_through_fma mul_sign mul_term @ flatten_sum_through_fma add_sign c
  | _ -> [ sign, e ]
;;

(* Cancel pairs of (+1, x) and (-1, x) — they sum to 0 and are dropped.
 * Sort the result canonically by tag.
 *
 * Implementation: tally signed coefficients per tag in a hashtable, then
 * emit (coefficient, t) for nonzero coefficients in tag order. *)
let cancel_signs (terms : (int * t) list) : (int * t) list =
  let coeff = Hashtbl.create 16 in
  let tag_to_t = Hashtbl.create 16 in
  List.iter
    (fun (s, e) ->
       Hashtbl.replace tag_to_t e.tag e;
       let prev =
         try Hashtbl.find coeff e.tag with
         | Not_found -> 0
       in
       Hashtbl.replace coeff e.tag (prev + s))
    terms;
  let result =
    Hashtbl.fold
      (fun tag c acc -> if c = 0 then acc else (c, Hashtbl.find tag_to_t tag) :: acc)
      coeff
      []
  in
  List.sort (fun (_, a) (_, b) -> compare a.tag b.tag) result
;;

(* Split a list into (evens, odds) by index — used by interleaved
 * pair-folding to expose butterfly subsums. *)
let split_interleaved (lst : 'a list) : 'a list * 'a list =
  let evens = ref [] in
  let odds = ref [] in
  List.iteri
    (fun i x -> if i mod 2 = 0 then evens := x :: !evens else odds := x :: !odds)
    lst;
  List.rev !evens, List.rev !odds
;;

(* === SMART CONSTRUCTORS (mutually recursive) ===
 *
 * mk_neg, mk_sub, mk_add, mk_mul are user-facing.
 * mk_add_binary, mk_sub_binary are leaf operations used by the pair-fold
 * after flattening/sorting (they bypass reassociation to avoid infinite
 * recursion).
 * emit_pair_fold and combine_two rebuild a binary tree from a sorted
 * signed-term list using interleaved pairing.
 *)

let rec mk_neg (e : t) : t =
  match e.node with
  | NK_Const c -> mk_const (-.c)
  | NK_Neg inner -> inner
  | _ -> hashcons (NK_Neg e)
(* Note: we used to have `NK_Sub (a, b) -> mk_sub b a` as a Neg-of-Sub
 * rewrite, but that creates a cycle with the canonical-order
 * mk_sub_binary below: mk_sub_binary in the reversed branch calls
 * mk_neg on a Sub, which would call mk_sub_binary again, etc.
 * Just emit a Neg(Sub) directly; mk_add_binary's `Add(x, Neg(y)) =
 * Sub(x, y)` peephole picks it up at the next level. *)

(* User-facing add: flatten, cancel, sort, pair-fold. *)
and mk_add (a : t) (b : t) : t =
  let terms = flatten_sum 1 a @ flatten_sum 1 b in
  let canonical = cancel_signs terms in
  emit_pair_fold canonical

(* User-facing sub: same, with b's terms negated. *)
and mk_sub (a : t) (b : t) : t =
  let terms = flatten_sum 1 a @ flatten_sum (-1) b in
  let canonical = cancel_signs terms in
  emit_pair_fold canonical

and mk_mul (a : t) (b : t) : t =
  if is_zero a || is_zero b
  then mk_const 0.0
  else if is_one a
  then b
  else if is_one b
  then a
  else if is_neg_one a
  then mk_neg b
  else if is_neg_one b
  then mk_neg a
  else (
    match a.node, b.node with
    | NK_Const x, NK_Const y -> mk_const (x *. y)
    | NK_Neg a', _ -> mk_neg (mk_mul a' b)
    | _, NK_Neg b' -> mk_neg (mk_mul a b')
    | _ ->
      let a, b = if a.tag <= b.tag then a, b else b, a in
      hashcons (NK_Mul (a, b)))

(* Leaf binary Add — used post-reassoc by emit_pair_fold. Hash-conses,
 * applies trivial identities, and recognizes Add(x, Neg(y)) → Sub(x, y)
 * to avoid redundant Neg+Add pairs after the pair-fold rebuilds. *)
and mk_add_binary (a : t) (b : t) : t =
  if is_zero a
  then b
  else if is_zero b
  then a
  else (
    match a.node, b.node with
    | NK_Const x, NK_Const y -> mk_const (x +. y)
    | _, NK_Neg b' -> mk_sub_binary a b' (* x + (-y) = x - y *)
    | NK_Neg a', _ -> mk_sub_binary b a' (* (-x) + y = y - x *)
    | _ ->
      let a, b = if a.tag <= b.tag then a, b else b, a in
      hashcons (NK_Add (a, b)))

and mk_sub_binary (a : t) (b : t) : t =
  if is_zero b
  then a
  else if is_zero a
  then mk_neg b
  else if a.tag = b.tag
  then mk_const 0.0
  else (
    match b.node with
    | NK_Neg b' ->
      (* x - (-y) = x + y. Catches the case where const_cmul produced
       * a Neg in a twiddle output that then gets subtracted. *)
      mk_add_binary a b'
    | _ ->
      (match a.node with
       | NK_Neg inner ->
         (match inner.node with
          | NK_Mul (x, y) ->
            (* Sub(Neg(Mul(x, y)), b) = -(x*y) - b
             *                        = NK_Fma(x, y, b, neg_mul=true, neg_add=true)
             *                        = vfnmsub at emission.
             *
             * dedup_sub_pairs introduces Neg(winner) substitutions; when the
             * substitution lands as the LHS of another Sub and the original
             * was Mul, we get Sub(Neg(Mul), c) — which without this peephole
             * emits as 3-4 instructions including a vxorpd with a -0.0 mask
             * (see docs/30_sub_neg_mul_fnmsub.md). The peephole fires at
             * construction time (during dedup_sub_pairs' rebuild) so the
             * Fma replaces the bad pattern before spill markers, scheduling,
             * or register allocation see it.
             *
             * Implemented as a peephole here (rather than a standalone pass)
             * because a standalone pass would orphan nodes that downstream
             * code — including spill markers captured before the rewrite —
             * still references. Constructing the Fma during dedup means the
             * resulting DAG has consistent tags throughout. *)
            hashcons (NK_Fma (x, y, b, true, true))
          | _ -> hashcons (NK_Sub (a, b)))
       | _ -> hashcons (NK_Sub (a, b))))

(* === NK_PLUS SMART CONSTRUCTOR ===
 *
 * Build a canonical NK_Plus node from a list of signed terms.
 *
 * Invariants enforced (mirror NK_Plus comment in the type definition):
 *   1. Result is NK_Plus only if 2+ terms remain. Single-term collapses
 *      to the term itself (with sign applied via mk_neg if -1).
 *   2. Empty list → Const 0.0.
 *   3. Nested NK_Plus is flattened: Plus[(+1, Plus[(+1, a); (-1, b)]); (+1, c)]
 *      becomes Plus[(+1, a); (-1, b); (+1, c)].
 *   4. NK_Neg is absorbed into the sign: (+1, Neg x) → (-1, x).
 *   5. At most one NK_Const term — multiple constants are summed at
 *      construction.
 *   6. Terms sorted by tag (ascending) for canonical hash-cons keys.
 *      Sign attaches to the term in the list, NOT to the position.
 *   7. Zero terms are dropped: (+1, Const 0.0) is removed.
 *   8. Tag-identical terms with opposite signs cancel: (+1, x) and (-1, x)
 *      both removed. Tag-identical terms with same sign coalesce into
 *      coefficient 2 — but since we don't have a coefficient-aware term
 *      form, we keep them as duplicates for now and let collectM (Commit 3)
 *      catch this case.
 *
 * The single-term collapse means callers can blindly construct a Plus with
 * any number of terms; the constructor returns whatever shape best
 * represents the sum. *)
and mk_plus (terms : (int * t) list) : t =
  (* Step 1: flatten nested NK_Plus terms, absorb NK_Neg into sign. *)
  let rec flatten (sign : int) (term : t) : (int * t) list =
    match term.node with
    | NK_Plus inner_terms ->
      List.concat_map (fun (s, t) -> flatten (sign * s) t) inner_terms
    | NK_Neg inner -> flatten (-sign) inner
    | _ -> [ sign, term ]
  in
  let flat = List.concat_map (fun (s, t) -> flatten s t) terms in
  (* Step 2: separate constants from non-constants and sum them. *)
  let const_sum = ref 0.0 in
  let nonconst =
    List.filter
      (fun (s, t) ->
         match t.node with
         | NK_Const c ->
           const_sum := !const_sum +. (float_of_int s *. c);
           false
         | _ -> true)
      flat
  in
  (* Step 3: drop zero-coefficient duplicates (tag-identical with opposite
   * signs cancel). Group by tag; keep terms where the signs don't sum to 0. *)
  let by_tag : (int, int * t) Hashtbl.t = Hashtbl.create 32 in
  List.iter
    (fun (s, t) ->
       match Hashtbl.find_opt by_tag t.tag with
       | None -> Hashtbl.add by_tag t.tag (s, t)
       | Some (s', _) ->
         let s_new = s + s' in
         if s_new = 0
         then Hashtbl.remove by_tag t.tag
         else Hashtbl.replace by_tag t.tag (s_new, t))
    nonconst;
  let merged = Hashtbl.fold (fun _ v acc -> v :: acc) by_tag [] in
  (* Step 4: re-expand merged terms whose coefficient is not ±1.
   * In Commit 2, we don't have coefficient-aware Plus terms, so
   * a coefficient of ±2 means two copies of the same term. We
   * expand to duplicate entries; collectM (Commit 3) will reintroduce
   * coefficients properly. For coefficient ≥ 2, keep as duplicates. *)
  let expanded =
    List.concat_map
      (fun (s, t) ->
         let n = abs s in
         let sign = if s >= 0 then 1 else -1 in
         if n = 0
         then []
         else if n = 1
         then [ sign, t ]
         else
           (* Duplicate (sign, t) n times. *)
           List.init n (fun _ -> sign, t))
      merged
  in
  (* Step 5: sort by tag for canonical ordering. *)
  let sorted = List.sort (fun (_, a) (_, b) -> compare a.tag b.tag) expanded in
  (* Step 6: re-prepend the const term if non-zero. *)
  let with_const =
    if !const_sum = 0.0 then sorted else (1, mk_const !const_sum) :: sorted
  in
  (* Step 7: collapse to single-term forms when appropriate. *)
  match with_const with
  | [] -> mk_const 0.0
  | [ (1, t) ] -> t
  | [ (-1, t) ] -> mk_neg t
  | _ -> hashcons (NK_Plus with_const)

(* === NK_PLUS LOWERING ===
 *
 * Convert an NK_Plus back to a left-associated chain of NK_Add / NK_Sub.
 * Required before passes that don't understand NK_Plus (currently: everyone
 * except the future collectM pass).
 *
 * Lowering is one-shot: a Plus with N terms becomes N-1 binary operations.
 * The choice of which term to emit first matters for FMA fusion downstream:
 * if a term is `Mul(_, _)`, fma_lift can absorb it into an FMA when it sits
 * as the right operand of an Add or Sub. We currently lower terms in tag
 * order; further optimization could reorder for better FMA opportunities.
 *
 * Negative terms produce NK_Sub edges; positive terms produce NK_Add edges.
 * The first term carries its sign as Neg-wrap if negative. *)
and lower_plus (e : t) : t =
  match e.node with
  | NK_Plus terms -> lower_plus_terms terms
  | _ -> e

(* Lower an n-ary Plus back to a binary Add/Sub tree.
 *
 * Uses emit_pair_fold (the same balanced-tree constructor used by mk_add/
 * mk_sub) so the resulting tree shape matches what the existing pipeline
 * expects. A left-linear chain via fold_left mk_add_binary would compile
 * to the same arithmetic but lose:
 *   - share_subsums opportunities (which look for balanced sub-tree
 *     structure across outputs)
 *   - fma_lift opportunities (which pattern-match on local Add-of-Mul
 *     shapes; a linear chain only exposes the head term)
 *   - the butterfly structure that radix-2/4 codelets rely on
 *
 * The sort by tag inside emit_pair_fold gives canonical hash-consing,
 * matching what mk_add would produce for the same flat term list.
 *)
and lower_plus_terms (terms : (int * t) list) : t =
  (* Each term may itself contain NK_Plus; lower recursively first so
   * emit_pair_fold sees a fully binary sub-tree at each leaf. *)
  let recursively_lowered = List.map (fun (s, t) -> s, lower_plus t) terms in
  emit_pair_fold recursively_lowered

(* Tried: Common-multiplicand factoring peephole
 *   Add(Mul(a, k), Mul(b, k)) → Mul(Add(a, b), k)
 *   Sub(Mul(a, k), Mul(b, k)) → Mul(Sub(a, b), k)
 * for compile-time-constant k.
 *
 * Empirically MAKES OP COUNT WORSE. The reason: when Mul(xr, k) is shared
 * between the Re and Im parts of a complex multiply (each consumer pulls
 * out its own product), the factoring eliminates the sharing. Now Re uses
 * Mul(Sub(xr, xi), k) and Im needs Mul(xr, k) again — net +1 Mul.
 *
 * R=32 op count after peephole:
 *   - was 662 → became 817 scalar ops (+23%)
 *   - vec instructions 600 → 755 (+26%)
 *
 * Conclusion: reassociation needs use-count awareness. Reverted; would
 * need a post-pass that examines DAG use counts before deciding whether
 * to factor. Left as future work. *)

(* Smart constructor for the complex multiply. Given xr, xi (input
 * complex value as split-complex pair) and wr, wi (twiddle as split-
 * complex pair), produces (out_re, out_im) where:
 *
 *   out_re = xr * wr - xi * wi
 *   out_im = xr * wi + xi * wr
 *
 * The outputs are NK_CmulRe/NK_CmulIm nodes — opaque to reassoc. This
 * preserves cmul structure during simplification, matching how hand-
 * tuned codelets keep cmul as a unit.
 *
 * Special cases (constant-folding when twiddle is known at compile time):
 *
 *   wr=1, wi=0:  trivial twiddle, output = (xr, xi). No multiplies.
 *   wr=0, wi=1:  twiddle is +i, output = (-xi, xr).
 *   wr=0, wi=-1: twiddle is -i, output = (xi, -xr).
 *
 * For runtime-loaded twiddles (Load(Twiddle ...)), neither special case
 * fires and we emit Cmul nodes. *)
and mk_cmul (xr : t) (xi : t) (wr : t) (wi : t) : t * t =
  (* Trivial-twiddle cases (compile-time known): *)
  match wr.node, wi.node with
  | NK_Const c1, NK_Const c2 when is_zero wi && is_one wr ->
    let _ = c1 in
    let _ = c2 in
    xr, xi
  | NK_Const _, NK_Const _ when is_zero wr && is_one wi -> mk_neg xi, xr
  | NK_Const _, NK_Const _ when is_zero wr && is_neg_one wi -> xi, mk_neg xr
  | _ ->
    (* General case: emit opaque Cmul nodes. *)
    let re = hashcons (NK_CmulRe (xr, xi, wr, wi)) in
    let im = hashcons (NK_CmulIm (xr, xi, wr, wi)) in
    re, im

(* Build a single signed term: (-1, x) -> Neg x. *)
and emit_signed_term ((sign, e) : int * t) : t = if sign >= 0 then e else mk_neg e

(* Combine two signed terms into one expression. *)
and combine_two ((s1, e1) : int * t) ((s2, e2) : int * t) : t =
  match s1, s2 with
  | 1, 1 -> mk_add_binary e1 e2
  | 1, -1 -> mk_sub_binary e1 e2
  | -1, 1 -> mk_sub_binary e2 e1
  | -1, -1 -> mk_neg (mk_add_binary e1 e2)
  | _ ->
    (* Coefficients other than ±1: emit Mul(const, leaf). Rare in FFT. *)
    let lhs = if s1 = 0 then mk_const 0.0 else mk_mul (mk_const (float_of_int s1)) e1 in
    let rhs = if s2 = 0 then mk_const 0.0 else mk_mul (mk_const (float_of_int s2)) e2 in
    mk_add_binary lhs rhs

(* Pair-fold a sorted list of signed terms into a binary tree by
 * recursive interleaved splitting. This exposes butterfly subsums
 * because the half-split structure matches even/odd index pairing. *)
and emit_pair_fold (terms : (int * t) list) : t =
  match terms with
  | [] -> mk_const 0.0
  | [ t ] -> emit_signed_term t
  | [ t1; t2 ] -> combine_two t1 t2
  | _ ->
    let evens, odds = split_interleaved terms in
    let lhs = emit_pair_fold evens in
    let rhs = emit_pair_fold odds in
    (* lhs and rhs are now positive subsum expressions (signs were
     * absorbed during folding via combine_two). Just Add them. *)
    mk_add_binary lhs rhs
;;

(* === LIFT FROM Expr.expr TO HASH-CONSED t ===
 *
 * Pattern detection for cmul: the math-layer DFT builder emits
 *   re_part = Sub(Mul(xr, wr), Mul(xi, wi))
 *   im_part = Add(Mul(xr, wi), Mul(xi, wr))
 * for each twiddled leg's complex multiply. We detect this pattern at
 * lift time and emit Cmul nodes — opaque to reassoc, preserving cmul
 * structure during simplification.
 *
 * The reassoc flag controls whether mk_add/mk_sub flatten n-ary sums
 * and pair-fold them. With reassoc=true (default), reassoc finds
 * butterfly subsums in flat sums. With reassoc=false, only binary
 * hash-consing happens; the input tree's structure is preserved.
 *
 * Use reassoc=false when the input was produced by a structured
 * algorithm (e.g. Cooley-Tukey decomposition in Dft.ml) where the
 * tree shape IS the optimization. Use reassoc=true when the input
 * is a flat sum from direct DFT expansion that needs reassoc to
 * find shared subexpressions.
 *)

let rec of_expr ?(reassoc = true) (e : Expr.expr) : t =
  (* Physical-identity memo: subtrees referenced multiple times via the
   * same OCaml allocation get processed once. See ExprMemo block above
   * for rationale and correctness argument. *)
  match ExprMemo.find_opt of_expr_memo e with
  | Some t -> t
  | None ->
    let add_op = if reassoc then mk_add else mk_add_binary in
    let sub_op = if reassoc then mk_sub else mk_sub_binary in
    let result =
      match e with
      | Expr.Const c -> mk_const c
      | Expr.Load r -> mk_load r
      | Expr.Neg e1 -> mk_neg (of_expr ~reassoc e1)
      (* CMUL.RE PATTERN: Sub(Mul(xr, wr), Mul(xi, wi)) → cmul real output. *)
      | Expr.Sub (Expr.Mul (xr_e, wr_e), Expr.Mul (xi_e, wi_e)) ->
        let xr = of_expr ~reassoc xr_e in
        let wr = of_expr ~reassoc wr_e in
        let xi = of_expr ~reassoc xi_e in
        let wi = of_expr ~reassoc wi_e in
        let is_const e =
          match e.node with
          | NK_Const _ -> true
          | NK_Neg n ->
            (match n.node with
             | NK_Const _ -> true
             | _ -> false)
          | _ -> false
        in
        if is_const xr || is_const xi || is_const wr || is_const wi
        then sub_op (mk_mul xr wr) (mk_mul xi wi)
        else (
          let re, _im = mk_cmul xr xi wr wi in
          re)
      (* CMUL.IM PATTERN — needs reassoc flag threaded too. *)
      | Expr.Add (Expr.Mul (xr_e, wi_e), Expr.Mul (xi_e, wr_e)) ->
        let xr = of_expr ~reassoc xr_e in
        let wi = of_expr ~reassoc wi_e in
        let xi = of_expr ~reassoc xi_e in
        let wr = of_expr ~reassoc wr_e in
        let is_const e =
          match e.node with
          | NK_Const _ -> true
          | NK_Neg n ->
            (match n.node with
             | NK_Const _ -> true
             | _ -> false)
          | _ -> false
        in
        if is_const xr || is_const xi || is_const wr || is_const wi
        then add_op (mk_mul xr wi) (mk_mul xi wr)
        else (
          let _re, im = mk_cmul xr xi wr wi in
          im)
      | Expr.Add (a, b) -> add_op (of_expr ~reassoc a) (of_expr ~reassoc b)
      | Expr.Sub (a, b) -> sub_op (of_expr ~reassoc a) (of_expr ~reassoc b)
      | Expr.Mul (a, b) -> mk_mul (of_expr ~reassoc a) (of_expr ~reassoc b)
    in
    ExprMemo.add of_expr_memo e result;
    result
;;

let of_assignments ?(reassoc = true) (al : Expr.assignment list)
  : (Expr.elem_ref * t) list
  =
  List.map (fun (lhs, rhs) -> lhs, of_expr ~reassoc rhs) al
;;
