(* algsimp.ml — algebraic simplification and common subexpression elimination.
 *
 * This is the meat of the math layer. Two responsibilities:
 *
 *   1. ALGEBRAIC SIMPLIFICATION: fold trivial operations like x*0 = 0,
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

and t = { tag : int; node : node_kind }

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
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag e;
      List.iter visit (preds e)
    end
  in
  List.iter visit roots;
  Hashtbl.fold (fun _ e acc -> e :: acc) seen []
  |> List.sort (fun (a : t) b -> compare a.tag b.tag)

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
       "NK_Plus reached site %S which is not yet wired (Commit 2+).      If \
        you see this, a consumer is generating NK_Plus before its readers      \
        are migrated; check the call stack."
       site)

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

let reset () =
  Hashtbl.clear hcons_table;
  ExprMemo.clear of_expr_memo;
  next_tag := 0

(* === CANONICALIZATION HELPERS === *)

let zero_threshold = 1e-14

let is_zero (e : t) : bool =
  match e.node with NK_Const c -> Float.abs c < zero_threshold | _ -> false

let is_one (e : t) : bool =
  match e.node with
  | NK_Const c -> Float.abs (c -. 1.0) < zero_threshold
  | _ -> false

let is_neg_one (e : t) : bool =
  match e.node with
  | NK_Const c -> Float.abs (c +. 1.0) < zero_threshold
  | _ -> false

(* === SMART CONSTRUCTORS ===
 * Each does algebraic simplification first, then hash-consing.
 *)

let mk_const (c : float) : t =
  let rounded =
    if c = 0.0 then 0.0 else float_of_string (Printf.sprintf "%.13e" c)
  in
  if Float.abs rounded < zero_threshold then hashcons (NK_Const 0.0)
  else if Float.abs (rounded -. 1.0) < zero_threshold then
    hashcons (NK_Const 1.0)
  else if Float.abs (rounded +. 1.0) < zero_threshold then
    hashcons (NK_Const (-1.0))
  else if rounded < 0.0 then
    (* Canonicalize negative non-trivial constants to -|c|.
     * This unifies all multiplications-by-c with multiplications-by-(-c):
     *   Mul(x, -c) → Mul(x, Neg(c)) → Neg(Mul(x, c)) via Neg-hoisting.
     * The underlying Mul(x, c) is then shared by hash-consing.
     * Hand-coded codelets do this manually (e.g. vnc = -vc); we get
     * the same effect mechanically. *)
    hashcons (NK_Neg (hashcons (NK_Const (-.rounded))))
  else hashcons (NK_Const rounded)

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
  | _ -> [ (sign, e) ]

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
  | NK_Add (a, b) ->
      flatten_sum_through_fma sign a @ flatten_sum_through_fma sign b
  | NK_Sub (a, b) ->
      flatten_sum_through_fma sign a @ flatten_sum_through_fma (-sign) b
  | NK_Neg inner -> flatten_sum_through_fma (-sign) inner
  | NK_Fma (a, b, c, nm, na) ->
      (* Reconstruct the multiplied term and addend as separate signed
       * leaves. The mul term is itself a leaf for flatten purposes
       * (we don't decompose Mul further). *)
      let mul_term = hashcons (NK_Mul (a, b)) in
      let mul_sign = if nm then -sign else sign in
      let add_sign = if na then -sign else sign in
      flatten_sum_through_fma mul_sign mul_term
      @ flatten_sum_through_fma add_sign c
  | _ -> [ (sign, e) ]

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
      let prev = try Hashtbl.find coeff e.tag with Not_found -> 0 in
      Hashtbl.replace coeff e.tag (prev + s))
    terms;
  let result =
    Hashtbl.fold
      (fun tag c acc ->
        if c = 0 then acc else (c, Hashtbl.find tag_to_t tag) :: acc)
      coeff []
  in
  List.sort (fun (_, a) (_, b) -> compare a.tag b.tag) result

(* Split a list into (evens, odds) by index — used by interleaved
 * pair-folding to expose butterfly subsums. *)
let split_interleaved (lst : 'a list) : 'a list * 'a list =
  let evens = ref [] in
  let odds = ref [] in
  List.iteri
    (fun i x ->
      if i mod 2 = 0 then evens := x :: !evens else odds := x :: !odds)
    lst;
  (List.rev !evens, List.rev !odds)

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
  if is_zero a || is_zero b then mk_const 0.0
  else if is_one a then b
  else if is_one b then a
  else if is_neg_one a then mk_neg b
  else if is_neg_one b then mk_neg a
  else
    match (a.node, b.node) with
    | NK_Const x, NK_Const y -> mk_const (x *. y)
    | NK_Neg a', _ -> mk_neg (mk_mul a' b)
    | _, NK_Neg b' -> mk_neg (mk_mul a b')
    | _ ->
        let a, b = if a.tag <= b.tag then (a, b) else (b, a) in
        hashcons (NK_Mul (a, b))

(* Leaf binary Add — used post-reassoc by emit_pair_fold. Hash-conses,
 * applies trivial identities, and recognizes Add(x, Neg(y)) → Sub(x, y)
 * to avoid redundant Neg+Add pairs after the pair-fold rebuilds. *)
and mk_add_binary (a : t) (b : t) : t =
  if is_zero a then b
  else if is_zero b then a
  else
    match (a.node, b.node) with
    | NK_Const x, NK_Const y -> mk_const (x +. y)
    | _, NK_Neg b' -> mk_sub_binary a b' (* x + (-y) = x - y *)
    | NK_Neg a', _ -> mk_sub_binary b a' (* (-x) + y = y - x *)
    | _ ->
        let a, b = if a.tag <= b.tag then (a, b) else (b, a) in
        hashcons (NK_Add (a, b))

and mk_sub_binary (a : t) (b : t) : t =
  if is_zero b then a
  else if is_zero a then mk_neg b
  else if a.tag = b.tag then mk_const 0.0
  else
    match b.node with
    | NK_Neg b' ->
        (* x - (-y) = x + y. Catches the case where const_cmul produced
         * a Neg in a twiddle output that then gets subtracted. *)
        mk_add_binary a b'
    | _ -> (
        match a.node with
        | NK_Neg inner -> (
            match inner.node with
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
        | _ -> hashcons (NK_Sub (a, b)))

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
    | _ -> [ (sign, term) ]
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
          if s_new = 0 then Hashtbl.remove by_tag t.tag
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
        if n = 0 then []
        else if n = 1 then [ (sign, t) ]
        else
          (* Duplicate (sign, t) n times. *)
          List.init n (fun _ -> (sign, t)))
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
  match e.node with NK_Plus terms -> lower_plus_terms terms | _ -> e

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
  let recursively_lowered = List.map (fun (s, t) -> (s, lower_plus t)) terms in
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
  match (wr.node, wi.node) with
  | NK_Const c1, NK_Const c2 when is_zero wi && is_one wr ->
      let _ = c1 in
      let _ = c2 in
      (xr, xi)
  | NK_Const _, NK_Const _ when is_zero wr && is_one wi -> (mk_neg xi, xr)
  | NK_Const _, NK_Const _ when is_zero wr && is_neg_one wi -> (xi, mk_neg xr)
  | _ ->
      (* General case: emit opaque Cmul nodes. *)
      let re = hashcons (NK_CmulRe (xr, xi, wr, wi)) in
      let im = hashcons (NK_CmulIm (xr, xi, wr, wi)) in
      (re, im)

(* Build a single signed term: (-1, x) -> Neg x. *)
and emit_signed_term ((sign, e) : int * t) : t =
  if sign >= 0 then e else mk_neg e

(* Combine two signed terms into one expression. *)
and combine_two ((s1, e1) : int * t) ((s2, e2) : int * t) : t =
  match (s1, s2) with
  | 1, 1 -> mk_add_binary e1 e2
  | 1, -1 -> mk_sub_binary e1 e2
  | -1, 1 -> mk_sub_binary e2 e1
  | -1, -1 -> mk_neg (mk_add_binary e1 e2)
  | _ ->
      (* Coefficients other than ±1: emit Mul(const, leaf). Rare in FFT. *)
      let lhs =
        if s1 = 0 then mk_const 0.0 else mk_mul (mk_const (float_of_int s1)) e1
      in
      let rhs =
        if s2 = 0 then mk_const 0.0 else mk_mul (mk_const (float_of_int s2)) e2
      in
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
              | NK_Neg n -> (
                  match n.node with NK_Const _ -> true | _ -> false)
              | _ -> false
            in
            if is_const xr || is_const xi || is_const wr || is_const wi then
              sub_op (mk_mul xr wr) (mk_mul xi wi)
            else
              let re, _im = mk_cmul xr xi wr wi in
              re
        (* CMUL.IM PATTERN — needs reassoc flag threaded too. *)
        | Expr.Add (Expr.Mul (xr_e, wi_e), Expr.Mul (xi_e, wr_e)) ->
            let xr = of_expr ~reassoc xr_e in
            let wi = of_expr ~reassoc wi_e in
            let xi = of_expr ~reassoc xi_e in
            let wr = of_expr ~reassoc wr_e in
            let is_const e =
              match e.node with
              | NK_Const _ -> true
              | NK_Neg n -> (
                  match n.node with NK_Const _ -> true | _ -> false)
              | _ -> false
            in
            if is_const xr || is_const xi || is_const wr || is_const wi then
              add_op (mk_mul xr wi) (mk_mul xi wr)
            else
              let _re, im = mk_cmul xr xi wr wi in
              im
        | Expr.Add (a, b) -> add_op (of_expr ~reassoc a) (of_expr ~reassoc b)
        | Expr.Sub (a, b) -> sub_op (of_expr ~reassoc a) (of_expr ~reassoc b)
        | Expr.Mul (a, b) -> mk_mul (of_expr ~reassoc a) (of_expr ~reassoc b)
      in
      ExprMemo.add of_expr_memo e result;
      result

let of_assignments ?(reassoc = true) (al : Expr.assignment list) :
    (Expr.elem_ref * t) list =
  List.map (fun (lhs, rhs) -> (lhs, of_expr ~reassoc rhs)) al

(* === SPILL MARKER LIFTING ===
 *
 * Dft.dft_expand_twiddled_spill returns (assignments, spill_markers)
 * where each marker carries an Expr.expr for the PASS 1 output value.
 * After of_assignments lifts the assignment list, the same Expr.expr
 * values appear as (already-hash-consed) Algsimp.t subtrees. We can
 * walk them via of_expr to retrieve their tags — hash-consing
 * deduplicates so we get the SAME Algsimp.t back, with the same tag.
 *
 * Important: lift markers AFTER of_assignments. The order matters
 * because of_expr may apply CSE/peephole rewrites that change which
 * Algsimp.t represents a given Expr.expr. By lifting assignments
 * first, we lock in the same tags the assignment closure uses.
 *
 * The reassoc flag must match what of_assignments was called with —
 * otherwise marker exprs might be lifted differently than the
 * assignment-context counterparts. *)

type spill_tag_marker = { slot : int; re_tag : int; im_tag : int }

let lift_spill_markers ?(reassoc = true) (markers : Dft.spill_marker list) :
    spill_tag_marker list =
  let trace = Sys.getenv_opt "SPILL_MARKER_TRACE" <> None in
  let node_kind n =
    match n.node with
    | NK_Const _ -> "Const"
    | NK_Load _ -> "Load"
    | NK_Neg _ -> "Neg"
    | NK_Add _ -> "Add"
    | NK_Sub _ -> "Sub"
    | NK_Mul _ -> "Mul"
    | NK_Fma _ -> "Fma"
    | NK_CmulRe _ -> "CmulRe"
    | NK_CmulIm _ -> "CmulIm"
    | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:596"
  in
  List.map
    (fun m ->
      let re = of_expr ~reassoc m.Dft.re_expr in
      let im = of_expr ~reassoc m.Dft.im_expr in
      if trace then
        Printf.eprintf "spill_marker slot=%d: re=t%d(%s) im=t%d(%s)\n"
          m.Dft.slot re.tag (node_kind re) im.tag (node_kind im);
      { slot = m.slot; re_tag = re.tag; im_tag = im.tag })
    markers

(* === SUB-PAIR DEDUPLICATION PASS ===
 *
 * After reassociation, we may have both `Sub(a, b)` and `Sub(b, a)` in
 * the DAG, computed independently, even though they're negatives of
 * each other. This pass detects such pairs and rewrites uses of one
 * to be `Neg` of the other — and then the smart constructors' peephole
 * `Add(x, Neg(y)) → Sub(x, y)` collapses the result.
 *
 * This is a global pass: which Sub direction "wins" depends on which
 * gets used more often across all roots. We pick the winner by usage
 * count, breaking ties by lower tag for determinism.
 *
 * Algorithm:
 *   1. Walk the DAG from all assignment roots, building two indices:
 *      - sub_pairs: for each Sub(a,b), record (a.tag, b.tag) -> node
 *      - parents:   for each node, list of nodes that reference it
 *      - usage_count: how many times each Sub node is used (parent count)
 *   2. For each (a,b) pair, check if (b,a) also exists.
 *      If yes, pick winner = higher usage_count (lower tag if tied).
 *      Mark loser with substitution: loser_tag -> mk_neg(winner).
 *   3. Rebuild each root using a memoized substitution walk.
 *
 * The rebuild uses the existing smart constructors, so the
 * Add-of-Neg peephole fires naturally during reconstruction.
 *)

let dedup_sub_pairs (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list =
  (* Step 1: walk DAG, build indexes. *)
  let visited = Hashtbl.create 256 in
  let usage_count = Hashtbl.create 256 in
  (* tag -> count *)
  let sub_index = Hashtbl.create 64 in
  (* (small_tag, big_tag) -> [Sub nodes both directions] *)
  let bump_usage tag =
    let c = try Hashtbl.find usage_count tag with Not_found -> 0 in
    Hashtbl.replace usage_count tag (c + 1)
  in
  let rec visit (e : t) =
    if not (Hashtbl.mem visited e.tag) then begin
      Hashtbl.add visited e.tag ();
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg inner ->
          bump_usage inner.tag;
          visit inner
      | NK_Add (a, b) | NK_Mul (a, b) ->
          bump_usage a.tag;
          bump_usage b.tag;
          visit a;
          visit b
      | NK_Sub (a, b) ->
          bump_usage a.tag;
          bump_usage b.tag;
          (* Index the Sub by (small_tag, big_tag) regardless of direction.
           * The list will hold both Sub(a,b) and Sub(b,a) if both exist. *)
          let key = if a.tag < b.tag then (a.tag, b.tag) else (b.tag, a.tag) in
          let prev = try Hashtbl.find sub_index key with Not_found -> [] in
          Hashtbl.replace sub_index key (e :: prev);
          visit a;
          visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          (* Cmul outputs are opaque to dedup. Visit the four operands so
           * usage counts include them; don't index Cmul itself for Sub-pair
           * matching. *)
          bump_usage a.tag;
          bump_usage b.tag;
          bump_usage c.tag;
          bump_usage d.tag;
          visit a;
          visit b;
          visit c;
          visit d
      | NK_Fma (a, b, c, _, _) ->
          (* Fma is opaque to dedup — same treatment as Cmul. *)
          bump_usage a.tag;
          bump_usage b.tag;
          bump_usage c.tag;
          visit a;
          visit b;
          visit c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:648"
    end
  in
  List.iter (fun (_, e) -> visit e) assigns;

  (* Step 2: find Sub-pair conflicts and pick winners. *)
  let substitute : (int, t) Hashtbl.t = Hashtbl.create 16 in
  Hashtbl.iter
    (fun _key nodes ->
      match nodes with
      | [ _ ] -> () (* only one direction in the DAG, no conflict *)
      | nodes_list -> (
          (* Multiple Sub nodes share the same (small,big) key. Should be
           * exactly two: Sub(a,b) and Sub(b,a). Pick the winner by usage. *)
          let scored =
            List.map
              (fun n ->
                let c =
                  try Hashtbl.find usage_count n.tag with Not_found -> 0
                in
                (c, n))
              nodes_list
          in
          let scored =
            List.sort
              (fun (c1, n1) (c2, n2) ->
                (* Higher usage wins; tie-break by lower tag (deterministic). *)
                if c1 <> c2 then compare c2 c1 else compare n1.tag n2.tag)
              scored
          in
          match scored with
          | (_, winner) :: losers ->
              List.iter
                (fun (_, loser) ->
                  if loser.tag <> winner.tag then
                    Hashtbl.add substitute loser.tag (mk_neg winner))
                losers
          | [] -> ()))
    sub_index;

  (* Step 3: rebuild assignments with the substitution applied.
   * Uses memoization over tags so each shared subtree is rebuilt once. *)
  let rebuild_cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rebuild (e : t) : t =
    match Hashtbl.find_opt rebuild_cache e.tag with
    | Some result -> result
    | None ->
        let result =
          match Hashtbl.find_opt substitute e.tag with
          | Some replacement -> replacement
          | None -> (
              (* Recursively rebuild children. The smart constructors handle
               * any new peepholes that fire (e.g. Add of Neg → Sub). *)
              match e.node with
              | NK_Const _ | NK_Load _ -> e
              | NK_Neg inner -> mk_neg (rebuild inner)
              | NK_Add (a, b) -> mk_add_binary (rebuild a) (rebuild b)
              | NK_Sub (a, b) -> mk_sub_binary (rebuild a) (rebuild b)
              | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
              | NK_CmulRe (a, b, c, d) ->
                  let re, _im =
                    mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d)
                  in
                  re
              | NK_CmulIm (a, b, c, d) ->
                  let _re, im =
                    mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d)
                  in
                  im
              | NK_Fma (a, b, c, neg_mul, neg_add) ->
                  (* Fma is opaque — rebuild its operands but preserve the
                   * fused structure. *)
                  hashcons
                    (NK_Fma (rebuild a, rebuild b, rebuild c, neg_mul, neg_add))
              | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:718")
        in
        Hashtbl.add rebuild_cache e.tag result;
        result
  in
  List.map (fun (lhs, e) -> (lhs, rebuild e)) assigns

(* === COLLECT-M PASS ===
 *
 * Inspired by FFTW's collectM (genfft/algsimp.ml). Walks each Add/Sub
 * subtree in the DAG, flattens it into a list of signed terms, then groups
 * terms by their non-constant factor (the "atom") and sums their
 * coefficients. Tag identity from hash-consing tells us when two atoms are
 * the SAME node — that's the case we want to merge.
 *
 * Example transformations:
 *   ax + bx + cx     -> (a+b+c)·x            [3 muls + 2 adds -> 1 mul]
 *   ax - bx          -> (a-b)·x              [2 muls + 1 sub  -> 1 mul]
 *   ax + x           -> (a+1)·x              [1 mul + 1 add   -> 1 mul]
 *   x + y + x        -> 2·x + y              [2 adds          -> 1 mul + 1 add]
 *
 * The pass also accumulates Const terms in the sum:
 *   2 + 3 + x        -> x + 5                [1 add + 1 const -> 1 add]
 *
 * SHALLOW vs DEEP:
 *
 * This is the SHALLOW form: it only collects within ONE Add/Sub subtree.
 * It does NOT distribute Mul through nested Plus structures, so a pattern
 * like `Mul(c, Add(x, y)) + Mul(c, z)` won't see the shared `c` because
 * the inner Mul is opaque (its operand `Add(x, y)` is a different atom
 * than `z`).
 *
 * The DEEP variant (FFTW's deepCollectM, planned for a follow-up) would
 * recursively distribute Muls through Plus children to expose more
 * sharing. We start with shallow because it's the simpler case to
 * verify and bench, and the savings (if any) tell us whether the deep
 * variant is worth pursuing.
 *
 * GATING:
 *
 * Enabled by VFFT_COLLECT_M=1. Default off so existing codelets retain
 * exactly their current op counts until we've measured collect_m end to end.
 *)

(* Extract (coefficient, atom) from a term.
 *   Mul(Const c, x)     -> (c, x)
 *   Mul(x, Const c)     -> (c, x)
 *   Neg(t)              -> negate the coefficient of (extract t)
 *   anything else       -> (1.0, t)
 *
 * NOT recursive into nested Muls: Mul(Const c, Mul(Const k, x)) would
 * extract as (c, Mul(Const k, x)), not (c*k, x). This is a deliberate
 * limitation of the shallow variant — the deep variant would fold Const*Const
 * here as well.
 *)
let extract_coefficient (t : t) : float * t =
  let unsigned (t : t) : float * t =
    match t.node with
    | NK_Mul (a, b) -> (
        match (a.node, b.node) with
        | NK_Const c, _ -> (c, b)
        | _, NK_Const c -> (c, a)
        | _ -> (1.0, t))
    | _ -> (1.0, t)
  in
  match t.node with
  | NK_Neg inner ->
      let c, atom = unsigned inner in
      (-.c, atom)
  | _ -> unsigned t

let collect_m (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  if
    Sys.getenv_opt "VFFT_COLLECT_M" <> Some "1"
    && Sys.getenv_opt "VFFT_DEEP_COLLECT" <> Some "1"
  then assigns
  else begin
    let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in

    (* For each Add/Sub subtree: flatten, recursively rebuild leaves,
     * group by atom, emit collected Plus and lower to binary. *)
    let rec rebuild (e : t) : t =
      match Hashtbl.find_opt cache e.tag with
      | Some r -> r
      | None ->
          let r =
            match e.node with
            | NK_Const _ | NK_Load _ -> e
            | NK_Neg inner -> mk_neg (rebuild inner)
            | NK_Add (a, b) ->
                (* Decide whether to collect: pre-check if this subtree has any
                 * shared atom. If yes, flatten and collect. If no, preserve the
                 * original binary structure (it was built by mk_add's pair-fold,
                 * which is balanced and FMA-friendly; re-flattening would
                 * linearize it and hurt fma_lift downstream). *)
                if subtree_has_collectible e then collect_subtree e
                else mk_add_binary (rebuild a) (rebuild b)
            | NK_Sub (a, b) ->
                if subtree_has_collectible e then collect_subtree e
                else mk_sub_binary (rebuild a) (rebuild b)
            | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
            | NK_CmulRe (a, b, c, d) ->
                let re, _ =
                  mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d)
                in
                re
            | NK_CmulIm (a, b, c, d) ->
                let _, im =
                  mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d)
                in
                im
            | NK_Fma (a, b, c, nm, na) ->
                hashcons (NK_Fma (rebuild a, rebuild b, rebuild c, nm, na))
            | NK_Plus _ ->
                (* In normal operation collect_m's input is binary-form. If we
                 * see NK_Plus, lower it and recurse. *)
                rebuild (lower_plus e)
          in
          Hashtbl.add cache e.tag r;
          r
    (* Check whether an Add/Sub subtree has any collectible structure.
     * "Collectible" means at least two terms share the same atom tag, OR
     * multiple constants appear (could be folded). If neither, the subtree
     * has no opportunities and we should preserve its original tree shape.
     *
     * Looks through NK_Fma leaves (created by the Sub-Neg-Mul peephole)
     * to expose their internal Mul + addend as separate terms. *)
    and subtree_has_collectible (e : t) : bool =
      let terms = flatten_sum_through_fma 1 e in
      let seen_atoms : (int, unit) Hashtbl.t = Hashtbl.create 16 in
      let n_consts = ref 0 in
      let has_dup = ref false in
      List.iter
        (fun (_, term) ->
          match term.node with
          | NK_Const _ -> incr n_consts
          | _ ->
              let _, atom = extract_coefficient term in
              if Hashtbl.mem seen_atoms atom.tag then has_dup := true
              else Hashtbl.add seen_atoms atom.tag ())
        terms;
      !has_dup || !n_consts > 1
    (* Collect a subtree: flatten (through Fma), group by atom, emit. *)
    and collect_subtree (e : t) : t =
      let terms = flatten_sum_through_fma 1 e in
      let rebuilt = List.map (fun (s, t) -> (s, rebuild t)) terms in
      let by_atom : (int, float * t) Hashtbl.t = Hashtbl.create 16 in
      let constant_acc = ref 0.0 in
      List.iter
        (fun (sign, term) ->
          match term.node with
          | NK_Const c ->
              constant_acc := !constant_acc +. (float_of_int sign *. c)
          | _ -> (
              let coeff, atom = extract_coefficient term in
              let signed_coeff = float_of_int sign *. coeff in
              match Hashtbl.find_opt by_atom atom.tag with
              | None -> Hashtbl.add by_atom atom.tag (signed_coeff, atom)
              | Some (existing, _) ->
                  Hashtbl.replace by_atom atom.tag
                    (existing +. signed_coeff, atom)))
        rebuilt;
      let new_terms = ref [] in
      Hashtbl.iter
        (fun _ (c, atom) ->
          if c <> 0.0 then begin
            let term = mk_mul (mk_const c) atom in
            new_terms := (1, term) :: !new_terms
          end)
        by_atom;
      if !constant_acc <> 0.0 then
        new_terms := (1, mk_const !constant_acc) :: !new_terms;
      let plus_node = mk_plus !new_terms in
      lower_plus plus_node
    in
    List.map (fun (lhs, e) -> (lhs, rebuild e)) assigns
  end

(* === DEEP-COLLECT (deepCollectM) ===
 *
 * The deep variant of collectM. Where shallow collectM merges terms within
 * ONE Add/Sub subtree, deepCollectM also distributes Const*Sum patterns
 * through nested sums to EXPOSE inner atoms to the outer collection. This
 * is FFTW's `deepCollectM` (genfft/algsimp.ml) with their default
 * `deep_collect_depth = 5`.
 *
 * Example transformation:
 *   k * (a*x - b*y) + k * (c*x - d*y)
 * Shallow collect on this won't find merges — the inner Subs hide the
 * atoms. Deep collect distributes:
 *   = (k*a)*x - (k*b)*y + (k*c)*x - (k*d)*y       [after distribute]
 *   = ((k*a) + (k*c))*x + (-(k*b) - (k*d))*y      [after collect]
 *
 * The wins:
 *   1. Atoms x and y are now visible at the outer level.
 *   2. Constant folding (k*a, k*c, etc.) reduces constants to one per
 *      atom per outer term, often via hash-cons sharing across outputs.
 *   3. FMA fusion catches the (combined_const * atom) pairs naturally.
 *
 * The risks:
 *   1. Distribution adds ops upfront. `k * (x + y)` becomes `k*x + k*y`
 *      (one extra Mul). We need collectM to find shared atoms (or CSE
 *      via hash-cons across other outputs) to recoup.
 *   2. Unbounded recursion would explode the DAG. Bounded by depth limit
 *      (default 5, matching FFTW).
 *   3. Distribution destroys the original tree shape, which may have been
 *      FMA-friendly. We compare the IR node count of the distributed-
 *      collected result vs the original; keep whichever is smaller.
 *
 * GATING:
 *
 * Enabled by VFFT_DEEP_COLLECT=1. Independent of VFFT_COLLECT_M (deep
 * collect is a superset).
 *)

(* Distribute a single signed term, recursing up to depth. Returns the
 * resulting list of signed terms after pushing Const factors through
 * inner Add/Sub/Neg structure and folding nested Const*Mul rotations. *)
let rec distribute_term ~(depth : int) ((sign, t) : int * t) : (int * t) list =
  if depth <= 0 then [ (sign, t) ]
  else
    match t.node with
    | NK_Neg inner -> distribute_term ~depth (-sign, inner)
    | NK_Mul (a, b) -> (
        (* Identify which operand is Const (if any). *)
        let const_part, other_part =
          match (a.node, b.node) with
          | NK_Const _, _ -> (Some a, b)
          | _, NK_Const _ -> (Some b, a)
          | _ -> (None, t)
        in
        match const_part with
        | None -> [ (sign, t) ]
        | Some c -> (
            match other_part.node with
            | NK_Add (x, y) ->
                (* c * (x + y) = c*x + c*y *)
                distribute_term ~depth:(depth - 1) (sign, mk_mul c x)
                @ distribute_term ~depth:(depth - 1) (sign, mk_mul c y)
            | NK_Sub (x, y) ->
                (* c * (x - y) = c*x - c*y *)
                distribute_term ~depth:(depth - 1) (sign, mk_mul c x)
                @ distribute_term ~depth:(depth - 1) (-sign, mk_mul c y)
            | NK_Neg inner ->
                (* c * (-x) = -(c * x) *)
                distribute_term ~depth (-sign, mk_mul c inner)
            | NK_Mul (m1, m2) -> (
                (* c * Mul(...): if inner has Const, rotate; otherwise leave. *)
                let rotated_opt =
                  match (m1.node, m2.node) with
                  | NK_Const _, _ -> Some (mk_mul (mk_mul c m1) m2)
                  | _, NK_Const _ -> Some (mk_mul (mk_mul c m2) m1)
                  | _ -> None
                in
                match rotated_opt with
                | Some rotated when rotated != t ->
                    distribute_term ~depth (sign, rotated)
                | _ -> [ (sign, t) ])
            | _ -> [ (sign, t) ]))
    | _ -> [ (sign, t) ]

(* Group a flat list of signed terms by atom tag and emit collected form.
 * Returns the IR node count of the resulting binary tree, plus the tree
 * itself. We compute both so the caller can compare cost vs the original
 * binary tree and decide whether to use the distributed-collected result. *)
let collect_terms_to_tree (terms : (int * t) list) : t =
  let by_atom : (int, float * t) Hashtbl.t = Hashtbl.create 16 in
  let constant_acc = ref 0.0 in
  List.iter
    (fun (sign, term) ->
      match term.node with
      | NK_Const c -> constant_acc := !constant_acc +. (float_of_int sign *. c)
      | _ -> (
          let coeff, atom = extract_coefficient term in
          let signed_coeff = float_of_int sign *. coeff in
          match Hashtbl.find_opt by_atom atom.tag with
          | None -> Hashtbl.add by_atom atom.tag (signed_coeff, atom)
          | Some (existing, _) ->
              Hashtbl.replace by_atom atom.tag (existing +. signed_coeff, atom)))
    terms;
  let new_terms = ref [] in
  Hashtbl.iter
    (fun _ (c, atom) ->
      if c <> 0.0 then begin
        let term = mk_mul (mk_const c) atom in
        new_terms := (1, term) :: !new_terms
      end)
    by_atom;
  if !constant_acc <> 0.0 then
    new_terms := (1, mk_const !constant_acc) :: !new_terms;
  lower_plus (mk_plus !new_terms)

(* Count the IR nodes reachable from a node (treats hashcons as identity).
 * Used as a cost heuristic to compare distributed vs original forms. *)
let count_ir_nodes (root : t) : int =
  let seen = Hashtbl.create 64 in
  let n = ref 0 in
  let rec walk e =
    if Hashtbl.mem seen e.tag then ()
    else begin
      Hashtbl.add seen e.tag ();
      incr n;
      List.iter walk (preds e)
    end
  in
  walk root;
  !n

let deep_collect ?(depth_limit = 5) (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list =
  if Sys.getenv_opt "VFFT_DEEP_COLLECT" <> Some "1" then assigns
  else begin
    let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in

    let rec rebuild (e : t) : t =
      match Hashtbl.find_opt cache e.tag with
      | Some r -> r
      | None ->
          let r =
            match e.node with
            | NK_Const _ | NK_Load _ -> e
            | NK_Neg inner -> mk_neg (rebuild inner)
            | NK_Add (a, b) ->
                try_deep_collect e (mk_add_binary (rebuild a) (rebuild b))
            | NK_Sub (a, b) ->
                try_deep_collect e (mk_sub_binary (rebuild a) (rebuild b))
            | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
            | NK_CmulRe (a, b, c, d) ->
                let re, _ =
                  mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d)
                in
                re
            | NK_CmulIm (a, b, c, d) ->
                let _, im =
                  mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d)
                in
                im
            | NK_Fma (a, b, c, nm, na) ->
                hashcons (NK_Fma (rebuild a, rebuild b, rebuild c, nm, na))
            | NK_Plus _ -> rebuild (lower_plus e)
          in
          Hashtbl.add cache e.tag r;
          r
    (* Distribute when at least one resulting Mul of c with a child node
     * already exists in the hash-cons table. Even one hit means
     * distribution doesn't add net new nodes (it transforms a Mul-of-Add
     * into a reference + a new Mul). Combined with the strict post-collect
     * guard below, false positives get filtered out. *)
    and any_mul_exists (c : t) (x : t) (y : t) : bool =
      lookup_node
        (NK_Mul
           ((if c.tag <= x.tag then c else x), if c.tag <= x.tag then x else c))
      <> None
      || lookup_node
           (NK_Mul
              ( (if c.tag <= y.tag then c else y),
                if c.tag <= y.tag then y else c ))
         <> None
    and distribute_use_aware ~depth ((sign, t) : int * t) : (int * t) list =
      if depth <= 0 then [ (sign, t) ]
      else
        match t.node with
        | NK_Neg inner -> distribute_use_aware ~depth (-sign, inner)
        | NK_Mul (a, b) -> (
            let const_part, other_part =
              match (a.node, b.node) with
              | NK_Const _, _ -> (Some a, b)
              | _, NK_Const _ -> (Some b, a)
              | _ -> (None, t)
            in
            match const_part with
            | None -> [ (sign, t) ]
            | Some c -> (
                match other_part.node with
                | NK_Add (x, y) when any_mul_exists c x y ->
                    distribute_use_aware ~depth:(depth - 1) (sign, mk_mul c x)
                    @ distribute_use_aware ~depth:(depth - 1) (sign, mk_mul c y)
                | NK_Sub (x, y) when any_mul_exists c x y ->
                    distribute_use_aware ~depth:(depth - 1) (sign, mk_mul c x)
                    @ distribute_use_aware ~depth:(depth - 1) (-sign, mk_mul c y)
                | NK_Neg inner ->
                    distribute_use_aware ~depth (-sign, mk_mul c inner)
                | NK_Mul (m1, m2) -> (
                    let rotated_opt =
                      match (m1.node, m2.node) with
                      | NK_Const _, _ -> Some (mk_mul (mk_mul c m1) m2)
                      | _, NK_Const _ -> Some (mk_mul (mk_mul c m2) m1)
                      | _ -> None
                    in
                    match rotated_opt with
                    | Some rotated when rotated != t ->
                        distribute_use_aware ~depth (sign, rotated)
                    | _ -> [ (sign, t) ])
                | _ -> [ (sign, t) ]))
        | _ -> [ (sign, t) ]
    and try_deep_collect (original : t) (rebuilt_binary : t) : t =
      (* Use the FMA-aware flatten so we see through early-peephole
       * Fma nodes that block ordinary flatten_sum. *)
      let terms = flatten_sum_through_fma 1 original in
      let n_input_terms = List.length terms in
      let rebuilt_terms = List.map (fun (s, t) -> (s, rebuild t)) terms in
      let distributed =
        List.concat_map (distribute_use_aware ~depth:depth_limit) rebuilt_terms
      in
      if List.length distributed <= n_input_terms then rebuilt_binary
      else begin
        let atom_set : (int, unit) Hashtbl.t = Hashtbl.create 16 in
        let has_const = ref false in
        List.iter
          (fun (_, term) ->
            match term.node with
            | NK_Const _ -> has_const := true
            | _ ->
                let _, atom = extract_coefficient term in
                Hashtbl.replace atom_set atom.tag ())
          distributed;
        let n_groups = Hashtbl.length atom_set + if !has_const then 1 else 0 in
        (* Strict win condition: collected term count must be STRICTLY LESS
         * than the original term count. n_groups <= n_input_terms tolerates
         * pure expansion-without-merging, which is what caused R=20's
         * regression with the looser check. *)
        let win = n_groups < n_input_terms in
        if Sys.getenv_opt "VFFT_DEEP_COLLECT_TRACE" = Some "1" then
          Printf.eprintf "deep_collect: in=%d dist=%d groups=%d %s\n"
            n_input_terms (List.length distributed) n_groups
            (if win then "WIN" else "skip");
        if win then collect_terms_to_tree distributed else rebuilt_binary
      end
    in
    List.map (fun (lhs, e) -> (lhs, rebuild e)) assigns
  end

(* === SUB-NEG-MUL → FNMSUB LIFTING ===
 *
 *   Sub(Neg(Mul(a, b)), c)  →  NK_Fma(a, b, c, neg_mul=true, neg_add=true)
 *
 * Why this exists. dedup_sub_pairs introduces Neg nodes when the loser of
 * a Sub-pair conflict gets substituted as Neg(winner). When that Neg is
 * consumed as the LHS of another Sub, mk_sub_binary doesn't simplify (its
 * peephole only matches NK_Neg on the RHS, not LHS), so the pattern
 * survives to emission as `Sub(Neg(Mul), Mul)` → emit_c renders it as
 * `vsubpd(vxorpd(neg_zero, Mul), Mul)`, costing 4 instructions and
 * pinning a -0.0 mask in .rodata.
 *
 * The mathematical equivalence Sub(Neg(Mul(a,b)), c) = -(a*b) - c =
 * NK_Fma(a, b, c, true, true) maps directly to vfnmsub231pd at codegen.
 * One instruction instead of three, no -0.0 mask, no extra register
 * pressure for the mask broadcast.
 *
 * UNCONDITIONAL: unlike fma_lift (which we gate to primes because
 * explicit FMA atoms constrain GCC's RA on composite DAGs), this rewrite
 * is strictly better in all cases:
 *   - The Sub(Neg(Mul), c) pattern ALREADY emits as 3-4 instructions
 *     including an XOR-with-mask. Replacing with 1 fnmsub reduces both
 *     instruction count and register pressure (no mask register needed).
 *   - The variant choice (fnmsub231) is unambiguous — there's no doc-28
 *     "GCC could pick a better variant" concern because the alternative
 *     emission was already worse than a forced fnmsub.
 *
 * Pattern is uncommon (R=25 t1_dit AVX-512: 6 occurrences out of 678
 * total IR ops) but each occurrence is a 3:1 to 4:1 instruction reduction
 * in the hot loop body. *)

let lift_sub_neg_mul (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list =
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rebuild (e : t) : t =
    match Hashtbl.find_opt cache e.tag with
    | Some r -> r
    | None ->
        let result =
          match e.node with
          | NK_Sub (a, b) -> (
              let a' = rebuild a in
              let b' = rebuild b in
              (* Pattern match: Sub(Neg(Mul(x, y)), z) → NK_Fma(x, y, z, true, true) *)
              match a'.node with
              | NK_Neg inner -> (
                  match inner.node with
                  | NK_Mul (x, y) -> hashcons (NK_Fma (x, y, b', true, true))
                  | _ -> mk_sub_binary a' b')
              | _ -> mk_sub_binary a' b')
          | NK_Add (a, b) -> mk_add_binary (rebuild a) (rebuild b)
          | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
          | NK_Neg inner -> mk_neg (rebuild inner)
          | NK_Const _ | NK_Load _ -> e
          | NK_CmulRe (a, b, c, d) ->
              hashcons (NK_CmulRe (rebuild a, rebuild b, rebuild c, rebuild d))
          | NK_CmulIm (a, b, c, d) ->
              hashcons (NK_CmulIm (rebuild a, rebuild b, rebuild c, rebuild d))
          | NK_Fma (a, b, c, nm, na) ->
              hashcons (NK_Fma (rebuild a, rebuild b, rebuild c, nm, na))
          | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:777"
        in
        Hashtbl.add cache e.tag result;
        result
  in
  List.map (fun (lhs, e) -> (lhs, rebuild e)) assigns

(* === DISTRIBUTIVE FACTORING ===
 *
 *   Σ ± c · x_i  →  c · (Σ ± x_i)   when c is a constant and all input
 *                                    Muls have use_count = 1
 *
 * This is the key simplification for monolithic prime butterflies (R=3,
 * 5, 7, 11, ...) where dft_direct emits Σ x_j · cos(2πjk/N) ± x_j · sin(...)
 * and the Winograd structure (s = x_1+x_{N-1}, d = x_1-x_{N-1}, ...) emerges
 * from grouping like-coefficient terms.
 *
 * Operates on FLAT sums (not binary Add/Sub pairs) — the binary form
 * orders by tag, so same-constant Muls aren't adjacent siblings and a
 * peephole on Add(Mul(_,c), Mul(_,c)) never fires for primes ≥ 5.
 *
 * SAFETY: in CT-decomposed codelets the same Mul(xr, k) is shared
 * between Cmul Re and Im outputs (use_count ≥ 2). Factoring naively
 * would destroy that sharing — Re uses Mul(Sub(xr,xi), k) but Im still
 * needs Mul(xr, k) standalone, net +1 mul. We only factor groups of
 * Muls that ALL have use_count = 1 in the original DAG. Validated on
 * R=32: with use_count > 1 inside Cmul, no factoring fires. *)

(* === DISTRIBUTIVE FACTORING (monolithic-prime-only) ===
 *
 *   Σ ± c · x_i  →  c · (Σ ± x_i)   when c is constant and the source
 *                                    Muls have use_count = 1
 *
 * STRUCTURAL DISCRIMINATOR — why this is monolithic-prime-only:
 *
 *   CT-decomposed codelets are ALREADY in FMA-friendly form. A twiddle
 *   multiplication (xr,xi)·(cos,sin) produces 4 muls with DISTINCT
 *   constants — no factoring opportunity. Special twiddles where
 *   |cos|=|sin| (e.g. ω₈ = 1/√2·(1,-1) appearing in R=8/16/32/64) DO
 *   give same-const muls, but those muls are shared between Re and Im
 *   (use_count > 1) — the safety check rejects them, AND that sharing
 *   IS the FMA-friendly structure we want to preserve.
 *
 *   Stray same-const fires that DO pass safety (use_count = 1) in CT
 *   codelets produce factored terms that don't share globally; the
 *   factored mul is just dead weight. Empirically R=16 with full safety
 *   still regressed +94 ops because of these fires. Conclusion: the
 *   use_count=1 condition is necessary but not sufficient for CT.
 *
 *   Monolithic primes are the inverse case. The DFT matrix cyclic
 *   symmetry means c·x_j appears in MANY outputs (use_count >> 1), and
 *   the factored c·(x_j+x_{N-j}) IS the shared Winograd structure that
 *   emerges. The "shared mul" the safety check would protect doesn't
 *   actually exist — it's an illusion of pre-factoring; both outputs
 *   would migrate to the factored form. So we disable safety entirely
 *   in aggressive mode.
 *
 * INTERFACE:
 *   ~aggressive:false (default) — pass-through. Use for CT-decomposed N.
 *   ~aggressive:true            — full flat-sum factoring. Use for
 *                                 monolithic primes (R=3,5,7,11). *)

let factor_common_muls ?(aggressive = false)
    (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  if not aggressive then assigns
  else
    (* If n is Mul(x, Const c) or Mul(Const c, x), return Some (x, c). *)
    let const_mul_of (n : t) : (t * float) option =
      match n.node with
      | NK_Mul (a, b) -> (
          match (a.node, b.node) with
          | NK_Const c, _ -> Some (b, c)
          | _, NK_Const c -> Some (a, c)
          | _ -> None)
      | _ -> None
    in
    (* Flatten an Add/Sub/Neg chain into [(sign, term)] terms.
     * Same logic as flatten_sum (which is private to construction). *)
    let rec flatten (sign : int) (e : t) : (int * t) list =
      match e.node with
      | NK_Add (a, b) -> flatten sign a @ flatten sign b
      | NK_Sub (a, b) -> flatten sign a @ flatten (-sign) b
      | NK_Neg inner -> flatten (-sign) inner
      | _ -> [ (sign, e) ]
    in
    (* Reconstruct a sum from a (sign, term) list. Separates positive and
     * negative terms, builds each via mk_add (which flattens + sorts +
     * pair-folds deterministically), then combines via mk_sub or mk_neg.
     * This ensures hash-cons hits when the same semantic sum is constructed
     * elsewhere — e.g., Neg(Add(a, b)) is canonical, never Sub(Neg(a), b). *)
    let rebuild_sum (terms : (int * t) list) : t =
      let pos =
        List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms
      in
      let neg =
        List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms
      in
      let build_sum lst =
        match lst with
        | [] -> mk_const 0.0
        | [ x ] -> x
        | x :: rest -> List.fold_left mk_add x rest
      in
      match (pos, neg) with
      | [], [] -> mk_const 0.0
      | _, [] -> build_sum pos
      | [], _ -> mk_neg (build_sum neg)
      | _, _ -> mk_sub (build_sum pos) (build_sum neg)
    in
    let max_iter = 20 in
    let rec loop assigns iter =
      if iter >= max_iter then assigns
      else begin
        let changed = ref false in
        let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in

        (* Try to factor a flat term list. Returns new term list and whether
         * any factoring fired. Groups of same-constant Muls become a single
         * new term: (+1, Mul(inner_sum, Const c)). No use-count safety —
         * aggressive mode treats all cross-output mul-sharing as factor-eligible
         * because primes' Winograd structure emerges from precisely this. *)
        let factor_terms (terms : (int * t) list) : (int * t) list * bool =
          if Sys.getenv_opt "FACTOR_TRACE" <> None && List.length terms >= 3
          then begin
            Printf.eprintf "  factor_terms input (%d): " (List.length terms);
            List.iter
              (fun (s, t) ->
                match const_mul_of t with
                | Some (_, c) ->
                    Printf.eprintf "%sc=%g " (if s > 0 then "+" else "-") c
                | None ->
                    Printf.eprintf "%sleaf(t%d) "
                      (if s > 0 then "+" else "-")
                      t.tag)
              terms;
            Printf.eprintf "\n"
          end;
          (* Bucket by constant value of Mul-coefficient.
           * Use float-bit-equality on the constant. *)
          let by_const : (int64, (int * t * t) list) Hashtbl.t =
            Hashtbl.create 8
          in
          (* int64 = float bits; payload is (sign, x, original_mul) *)
          let leftover : (int * t) list ref = ref [] in
          List.iter
            (fun (sign, term) ->
              match const_mul_of term with
              | Some (x, c) ->
                  let key = Int64.bits_of_float c in
                  let cur =
                    try Hashtbl.find by_const key with Not_found -> []
                  in
                  Hashtbl.replace by_const key ((sign, x, term) :: cur)
              | _ -> leftover := (sign, term) :: !leftover)
            terms;
          let factored = ref [] in
          let any_fired = ref false in
          Hashtbl.iter
            (fun key entries ->
              match entries with
              | [] -> ()
              | [ (s, _, orig) ] ->
                  (* Single mul with this constant; not a factoring opportunity. *)
                  leftover := (s, orig) :: !leftover
              | _ ->
                  (* ≥2 muls share the same constant. Factor them. *)
                  any_fired := true;
                  changed := true;
                  let inner_terms =
                    List.map (fun (s, x, _) -> (s, x)) entries
                  in
                  let inner_sum = rebuild_sum inner_terms in
                  let c = Int64.float_of_bits key in
                  let factored_term = mk_mul inner_sum (mk_const c) in
                  factored := (1, factored_term) :: !factored)
            by_const;
          (!leftover @ !factored, !any_fired)
        in

        let rec rewrite (n : t) : t =
          match Hashtbl.find_opt cache n.tag with
          | Some r -> r
          | None ->
              let r =
                match n.node with
                | NK_Const _ | NK_Load _ -> n
                | NK_Neg a ->
                    let a' = rewrite a in
                    if a' == a then n else mk_neg a'
                | NK_Add (a, b) ->
                    (* Look for factoring across the full flat sum (recurses
                     * through nested Add/Sub/Neg). If found, restructure via
                     * rebuild_sum. If not, preserve binary structure with
                     * substituted children — re-flattening would destroy
                     * sharing of inner Adds with the rest of the DAG. *)
                    let raw_terms = flatten 1 n in
                    let rewritten_terms =
                      List.map (fun (s, t) -> (s, rewrite t)) raw_terms
                    in
                    let new_terms, fired = factor_terms rewritten_terms in
                    if fired then rebuild_sum new_terms
                    else
                      let a' = rewrite a in
                      let b' = rewrite b in
                      if a' == a && b' == b then n else mk_add_binary a' b'
                | NK_Sub (a, b) ->
                    let raw_terms = flatten 1 n in
                    let rewritten_terms =
                      List.map (fun (s, t) -> (s, rewrite t)) raw_terms
                    in
                    let new_terms, fired = factor_terms rewritten_terms in
                    if fired then rebuild_sum new_terms
                    else
                      let a' = rewrite a in
                      let b' = rewrite b in
                      if a' == a && b' == b then n else mk_sub_binary a' b'
                | NK_Mul (a, b) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    if a' == a && b' == b then n else mk_mul a' b'
                | NK_CmulRe (a, b, c, d) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    let c' = rewrite c in
                    let d' = rewrite d in
                    if a' == a && b' == b && c' == c && d' == d then n
                    else hashcons (NK_CmulRe (a', b', c', d'))
                | NK_CmulIm (a, b, c, d) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    let c' = rewrite c in
                    let d' = rewrite d in
                    if a' == a && b' == b && c' == c && d' == d then n
                    else hashcons (NK_CmulIm (a', b', c', d'))
                | NK_Fma (a, b, c, neg_mul, neg_add) ->
                    (* Fma is opaque to factoring — the muls inside are already
                     * claimed by the FMA fusion. Recurse into operands but
                     * never restructure. *)
                    let a' = rewrite a in
                    let b' = rewrite b in
                    let c' = rewrite c in
                    if a' == a && b' == b && c' == c then n
                    else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
                | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:964"
              in
              Hashtbl.add cache n.tag r;
              r
        in
        let new_assigns =
          List.map (fun (oref, e) -> (oref, rewrite e)) assigns
        in
        if !changed then loop new_assigns (iter + 1) else new_assigns
      end
    in
    loop assigns 0

(* === SUBSUM SHARING ===
 *
 * Recognize pre-existing 2-term sub-expressions inside larger flat sums
 * and reuse them. The motivating case is the X[0] output in monolithic
 * primes:
 *
 *   X[0].re = x[0] + x[1] + x[2] + x[3] + x[4]      (5 terms, 4 binary adds)
 *
 * After factoring fires, the DAG already contains pair sums:
 *   s14 = x[1] + x[4]    (built for 0.309·s14 inner sum)
 *   s23 = x[2] + x[3]    (built for 0.809·s23 inner sum)
 *
 * X[0].re could be expressed as `x[0] + s14 + s23` (3 terms, 2 binary adds),
 * saving 2 ops per X[0] output. Across the 2 X[0] outputs (.re/.im) and
 * scaling with N, this is meaningful.
 *
 * The savings:    pre-existing pair (use_count >= 1 from the factored mul)
 *                 → substitute Add(a, b) into the chain.
 *
 * Algorithm: for each Add chain, partition terms by sign, then within each
 * sign group greedily pick a pair (a, b) such that NK_Add(a, b) is already
 * hash-cons'd with use_count > 0. Replace the pair with the existing node.
 * Repeat until no more shareable pairs. *)

(* === FACTOR BY ATOM ===
 *
 * Complementary to factor_common_muls. Where that pass buckets by the
 * CONSTANT operand of Mul (factoring `c*a + c*b → c*(a+b)`), this one
 * buckets by the NON-CONSTANT operand (factoring `c1*a + c2*a → (c1+c2)*a`).
 *
 * The killer case: when c1 + c2 + ... + cN is a compile-time-foldable sum
 * of distinct constants. Each ci is a Const, so (c1 + c2 + ... + cN) folds
 * to ONE constant at DAG construction time. N muls collapse to 1 mul.
 *
 * This is FFTW's `collectM` with the second-operand-as-coeff path. The
 * pattern arises in DFT computations where multiple twiddle factors
 * multiply the same input element across outputs.
 *
 * IR-level extraction:
 *   Mul(Const c, x)        — atom = x, coeff = c
 *   Mul(x, Const c)        — atom = x, coeff = c   (canonical-tagged form)
 *   Neg(Mul(Const c, x))   — atom = x, coeff = -c
 *
 * For each atom seen, sum the coefficients (compile-time fold). Emit
 * `Mul(folded_const, atom)` if folded_const ≠ 0, else drop the term.
 *
 * FMA awareness: collapsing N muls to 1 saves at least (N-1) instructions
 * regardless of FMA fusion downstream. A standalone mul that loses
 * siblings can still fuse with at most one consumer, so the merge can
 * never lose.
 *
 * Fires only in aggressive mode (primes). Safe-mode CT codelets don't
 * have this pattern. *)

let factor_by_atom ?(aggressive = false) (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list =
  if not aggressive then assigns
  else
    let const_of (e : t) : float option =
      match e.node with NK_Const c -> Some c | _ -> None
    in
    let rec atom_view (sign : int) (e : t) : (float * t) option =
      match e.node with
      | NK_Mul (a, b) -> (
          match (const_of a, const_of b) with
          | Some c, None -> Some (float_of_int sign *. c, b)
          | None, Some c -> Some (float_of_int sign *. c, a)
          | _ -> None)
      | NK_Neg inner -> atom_view (-sign) inner
      | _ -> None
    in
    let rec flatten (sign : int) (e : t) : (int * t) list =
      match e.node with
      | NK_Add (a, b) -> flatten sign a @ flatten sign b
      | NK_Sub (a, b) -> flatten sign a @ flatten (-sign) b
      | NK_Neg inner -> flatten (-sign) inner
      | _ -> [ (sign, e) ]
    in
    let rebuild_sum (terms : (int * t) list) : t =
      let pos =
        List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms
      in
      let neg =
        List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms
      in
      let build lst =
        match lst with
        | [] -> mk_const 0.0
        | [ x ] -> x
        | x :: rest -> List.fold_left mk_add x rest
      in
      match (pos, neg) with
      | [], [] -> mk_const 0.0
      | _, [] -> build pos
      | [], _ -> mk_neg (build neg)
      | _, _ -> mk_sub (build pos) (build neg)
    in

    let max_iter = 8 in
    let rec loop assigns iter =
      if iter >= max_iter then assigns
      else begin
        let changed = ref false in
        let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in

        (* Bucket flat terms by atom-tag, sum coefficients (compile-time fold).
         * `fired` = at least one bucket had multiple entries OR a coefficient
         * summed to zero. *)
        let factor_terms (terms : (int * t) list) : (int * t) list * bool =
          let by_atom : (int, t * float ref * int ref) Hashtbl.t =
            Hashtbl.create 8
          in
          let leftover : (int * t) list ref = ref [] in
          List.iter
            (fun (sign, term) ->
              match atom_view sign term with
              | Some (c, atom) -> (
                  match Hashtbl.find_opt by_atom atom.tag with
                  | Some (_, acc, count) ->
                      acc := !acc +. c;
                      incr count
                  | None -> Hashtbl.add by_atom atom.tag (atom, ref c, ref 1))
              | None -> leftover := (sign, term) :: !leftover)
            terms;

          let new_factored : (int * t) list ref = ref [] in
          let any_collapse_or_zero = ref false in
          Hashtbl.iter
            (fun _ (atom, c_ref, count_ref) ->
              let c = !c_ref in
              let count = !count_ref in
              if c = 0.0 then begin
                any_collapse_or_zero := true
              end
              else if count >= 2 then begin
                (* Multiple originals collapsed into one. *)
                let new_term = mk_mul (mk_const c) atom in
                new_factored := (1, new_term) :: !new_factored;
                any_collapse_or_zero := true
              end
              else begin
                (* Single occurrence — preserve as Mul(c, atom). *)
                let new_term = mk_mul (mk_const c) atom in
                new_factored := (1, new_term) :: !new_factored
              end)
            by_atom;
          let final_terms = !new_factored @ !leftover in
          (final_terms, !any_collapse_or_zero)
        in

        let rec rewrite (n : t) : t =
          match Hashtbl.find_opt cache n.tag with
          | Some r -> r
          | None ->
              let r =
                match n.node with
                | NK_Const _ | NK_Load _ -> n
                | NK_Neg a ->
                    let a' = rewrite a in
                    if a' == a then n else mk_neg a'
                | NK_Add (a, b) ->
                    let raw_terms = flatten 1 n in
                    let rewritten_terms =
                      List.map (fun (s, t) -> (s, rewrite t)) raw_terms
                    in
                    let new_terms, fired = factor_terms rewritten_terms in
                    if fired then begin
                      changed := true;
                      rebuild_sum new_terms
                    end
                    else
                      let a' = rewrite a in
                      let b' = rewrite b in
                      if a' == a && b' == b then n else mk_add_binary a' b'
                | NK_Sub (a, b) ->
                    let raw_terms = flatten 1 n in
                    let rewritten_terms =
                      List.map (fun (s, t) -> (s, rewrite t)) raw_terms
                    in
                    let new_terms, fired = factor_terms rewritten_terms in
                    if fired then begin
                      changed := true;
                      rebuild_sum new_terms
                    end
                    else
                      let a' = rewrite a in
                      let b' = rewrite b in
                      if a' == a && b' == b then n else mk_sub_binary a' b'
                | NK_Mul (a, b) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    if a' == a && b' == b then n else mk_mul a' b'
                | NK_CmulRe (a, b, c, d) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    let c' = rewrite c in
                    let d' = rewrite d in
                    if a' == a && b' == b && c' == c && d' == d then n
                    else hashcons (NK_CmulRe (a', b', c', d'))
                | NK_CmulIm (a, b, c, d) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    let c' = rewrite c in
                    let d' = rewrite d in
                    if a' == a && b' == b && c' == c && d' == d then n
                    else hashcons (NK_CmulIm (a', b', c', d'))
                | NK_Fma (a, b, c, neg_mul, neg_add) ->
                    let a' = rewrite a in
                    let b' = rewrite b in
                    let c' = rewrite c in
                    if a' == a && b' == b && c' == c then n
                    else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
                | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1174"
              in
              Hashtbl.add cache n.tag r;
              r
        in
        let new_assigns =
          List.map (fun (oref, e) -> (oref, rewrite e)) assigns
        in
        if !changed then loop new_assigns (iter + 1) else new_assigns
      end
    in
    loop assigns 0

let share_subsums ?(aggressive = false) (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list =
  if not aggressive then assigns
  else
    (* Use-count over the whole DAG (excluding our reconstruction). *)
    let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    let visited : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    let bump tag =
      let c = try Hashtbl.find use_count tag with Not_found -> 0 in
      Hashtbl.replace use_count tag (c + 1)
    in
    let rec walk e =
      if not (Hashtbl.mem visited e.tag) then begin
        Hashtbl.add visited e.tag ();
        match e.node with
        | NK_Const _ | NK_Load _ -> ()
        | NK_Neg a ->
            bump a.tag;
            walk a
        | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
            bump a.tag;
            bump b.tag;
            walk a;
            walk b
        | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
            bump a.tag;
            bump b.tag;
            bump c.tag;
            bump d.tag;
            walk a;
            walk b;
            walk c;
            walk d
        | NK_Fma (a, b, c, _, _) ->
            bump a.tag;
            bump b.tag;
            bump c.tag;
            walk a;
            walk b;
            walk c
        | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1242"
      end
    in
    List.iter
      (fun (_, e) ->
        bump e.tag;
        walk e)
      assigns;

    let used_elsewhere n =
      (try Hashtbl.find use_count n.tag with Not_found -> 0) >= 1
    in

    let rec flatten (sign : int) (e : t) : (int * t) list =
      match e.node with
      | NK_Add (a, b) -> flatten sign a @ flatten sign b
      | NK_Sub (a, b) -> flatten sign a @ flatten (-sign) b
      | NK_Neg inner -> flatten (-sign) inner
      | _ -> [ (sign, e) ]
    in

    (* Try to find a pair (i, j) in `terms` with the same sign such that
     * NK_Add(a, b) (sorted by tag) already exists in the hash-cons table
     * with at least one external user. Returns (i, j, existing_node) or None. *)
    let find_shareable_pair (terms : (int * t) array) : (int * int * t) option =
      let n = Array.length terms in
      let result = ref None in
      let i = ref 0 in
      while !result = None && !i < n do
        let j = ref (!i + 1) in
        while !result = None && !j < n do
          let s1, t1 = terms.(!i) in
          let s2, t2 = terms.(!j) in
          if s1 = s2 && t1.tag <> t2.tag then begin
            let a, b = if t1.tag <= t2.tag then (t1, t2) else (t2, t1) in
            match lookup_node (NK_Add (a, b)) with
            | Some existing when used_elsewhere existing ->
                result := Some (!i, !j, existing)
            | _ -> ()
          end;
          incr j
        done;
        incr i
      done;
      !result
    in

    let rebuild_sum_binary (terms : (int * t) list) : t =
      let pos =
        List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms
      in
      let neg =
        List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms
      in
      let build_chain lst =
        match lst with
        | [] -> mk_const 0.0
        | [ x ] -> x
        | x :: rest -> List.fold_left mk_add_binary x rest
      in
      match (pos, neg) with
      | [], [] -> mk_const 0.0
      | _, [] -> build_chain pos
      | [], _ -> mk_neg (build_chain neg)
      | _, _ -> mk_sub_binary (build_chain pos) (build_chain neg)
    in

    let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
    let rec rewrite (n : t) : t =
      match Hashtbl.find_opt cache n.tag with
      | Some r -> r
      | None ->
          let r =
            match n.node with
            | NK_Const _ | NK_Load _ -> n
            | NK_Neg a ->
                let a' = rewrite a in
                if a' == a then n else mk_neg a'
            | NK_Add _ | NK_Sub _ ->
                (* Flatten this Add/Sub chain and try to share 2-term subsums. *)
                let raw_terms = flatten 1 n in
                let rewritten_terms =
                  List.map (fun (s, t) -> (s, rewrite t)) raw_terms
                in
                if List.length rewritten_terms < 3 then
                  (* Nothing to share at this level; preserve binary structure. *)
                  begin match n.node with
                  | NK_Add (a, b) ->
                      let a' = rewrite a in
                      let b' = rewrite b in
                      if a' == a && b' == b then n else mk_add_binary a' b'
                  | NK_Sub (a, b) ->
                      let a' = rewrite a in
                      let b' = rewrite b in
                      if a' == a && b' == b then n else mk_sub_binary a' b'
                  | _ -> n
                  end
                else begin
                  (* Greedy substitution of shareable pairs. *)
                  let arr = ref (Array.of_list rewritten_terms) in
                  let any_shared = ref false in
                  let continue_loop = ref true in
                  while !continue_loop do
                    match find_shareable_pair !arr with
                    | None -> continue_loop := false
                    | Some (i, j, existing) ->
                        any_shared := true;
                        let sign, _ = !arr.(i) in
                        (* Replace position i with (sign, existing); remove position j. *)
                        let n_arr = Array.length !arr in
                        let new_arr = Array.make (n_arr - 1) (1, n) in
                        Array.blit !arr 0 new_arr 0 i;
                        new_arr.(i) <- (sign, existing);
                        Array.blit !arr (i + 1) new_arr (i + 1) (j - i - 1);
                        if j < n_arr - 1 then
                          Array.blit !arr (j + 1) new_arr j (n_arr - 1 - j);
                        arr := new_arr
                  done;
                  if !any_shared then rebuild_sum_binary (Array.to_list !arr)
                  else
                    (* No pairs shareable; preserve original binary structure. *)
                    match n.node with
                    | NK_Add (a, b) ->
                        let a' = rewrite a in
                        let b' = rewrite b in
                        if a' == a && b' == b then n else mk_add_binary a' b'
                    | NK_Sub (a, b) ->
                        let a' = rewrite a in
                        let b' = rewrite b in
                        if a' == a && b' == b then n else mk_sub_binary a' b'
                    | _ -> n
                end
            | NK_Mul (a, b) ->
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_mul a' b'
            | NK_CmulRe (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d then n
                else hashcons (NK_CmulRe (a', b', c', d'))
            | NK_CmulIm (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d then n
                else hashcons (NK_CmulIm (a', b', c', d'))
            | NK_Fma (a, b, c, neg_mul, neg_add) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                if a' == a && b' == b && c' == c then n
                else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
            | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1315"
          in
          Hashtbl.add cache n.tag r;
          r
    in
    List.map (fun (oref, e) -> (oref, rewrite e)) assigns

(* === DAG TRANSPOSITION ===
 *
 * Linear-network transposition: for each node N in the DAG, compute T[N]
 * representing N's contribution if the network were run "in reverse" —
 * roots become inputs, leaves become outputs.
 *
 * Rule: T[N] = Σ over parents P (consumers of N): w · T[P]
 *   where w is N's coefficient in P's definition:
 *     Add(N, _) or Add(_, N): w = +1
 *     Sub(N, _):              w = +1   (left operand)
 *     Sub(_, N):              w = -1   (right operand)
 *     Mul(N, Const c) or Mul(Const c, N): w = c
 *     Mul(N, _) where _ not const: NOT linear — skip (primes don't have this)
 *     Neg(N):                 w = -1
 *
 * For roots (output assignments), T[root] = synthetic Load with the
 * original output's elem_ref. For leaves (Load nodes with input/twiddle
 * elem_refs), T[leaf] is the new "transposed output" expression.
 *
 * The output is a new assigns list: each original input load's elem_ref
 * is the output reference, with T[load] as the value. The simplifier
 * can then run on this transposed view, finding CSEs that aren't visible
 * in the forward direction. Transposing twice (with simplification in
 * between) gives back the original direction with new optimizations.
 *
 * Per Frigo PLDI'99 Table 7, transposition saves muls specifically on
 * sizes 5, 10, 13, 15.
 *
 * LIMITATION: Cmul nodes are not handled (they're nonlinear in some
 * uses). For monolithic primes (R=3,5,7,11) all twiddles are constants
 * so the DAG is pure linear, no Cmul nodes — this is fine. *)

let transpose (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  (* Step 1: Collect all reachable nodes from roots in topo order
   * (children before parents). *)
  let visited : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let topo_rev : t list ref = ref [] in
  let rec dfs n =
    if not (Hashtbl.mem visited n.tag) then begin
      Hashtbl.add visited n.tag ();
      (match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a -> dfs a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          dfs a;
          dfs b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          dfs a;
          dfs b;
          dfs c;
          dfs d
      | NK_Fma (a, b, c, _, _) ->
          dfs a;
          dfs b;
          dfs c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1433");
      topo_rev := n :: !topo_rev
    end
  in
  List.iter (fun (_, e) -> dfs e) assigns;

  (* Step 2: Build parent map. For each node, record list of contributions
   * from each parent: (sign, scale_const_option, parent_node). *)
  let contribs : (int, (int * t option * t) list) Hashtbl.t =
    Hashtbl.create 256
  in
  let add_contrib (child : t) (parent : t) (sign : int) (scale : t option) =
    let cur = try Hashtbl.find contribs child.tag with Not_found -> [] in
    Hashtbl.replace contribs child.tag ((sign, scale, parent) :: cur)
  in
  (* Process each node's structure to register contributions to its children. *)
  Hashtbl.iter (fun _ () -> ()) visited;
  List.iter
    (fun n ->
      match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a -> add_contrib a n (-1) None
      | NK_Add (a, b) ->
          add_contrib a n 1 None;
          add_contrib b n 1 None
      | NK_Sub (a, b) ->
          add_contrib a n 1 None;
          add_contrib b n (-1) None
      | NK_Mul (a, b) -> (
          (* Const · X form — the X operand has weight = const.
           * Const itself never has a useful T value (it's a leaf with no
           * input semantics in transposition), so skip its contrib. *)
          match (a.node, b.node) with
          | NK_Const _, _ -> add_contrib b n 1 (Some a)
          | _, NK_Const _ -> add_contrib a n 1 (Some b)
          | _ ->
              (* Non-linear Mul — can't transpose cleanly. Skip both
               * operands. The transposed DAG won't include this node's
               * contributions. *)
              ())
      | NK_CmulRe _ | NK_CmulIm _ ->
          (* Skip — primes don't produce these. *)
          ()
      | NK_Fma _ ->
          (* Fma is opaque to transposition. The transpose pass shouldn't
           * normally encounter Fma anyway since fma_lift runs LAST. *)
          ()
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1456")
    (List.rev !topo_rev);

  (* topo_rev is built by PREPENDING after DFS post-order recursion;
   * the root (added last) ends up at the front, leaves at the back.
   * So topo_rev itself iterates roots-first; List.rev topo_rev iterates
   * leaves-first. Contribs population (just above) iterates leaves-first
   * — order doesn't matter, every node visited once. *)

  (* Step 3: Compute T[N] for each node, in order parents-first
   * (so parents have T set before children look them up). topo_rev is
   * roots-first, which is parents-first. *)
  let t_value : (int, t) Hashtbl.t = Hashtbl.create 256 in

  (* Roots: T[root] = Load with the original output's elem_ref. *)
  List.iter
    (fun (oref, root) ->
      if not (Hashtbl.mem t_value root.tag) then
        Hashtbl.add t_value root.tag (mk_load oref))
    assigns;

  (* For nodes that have contribs (= internal nodes used by parents),
   * compute their T from their parents' T values. Process roots-first. *)
  List.iter
    (fun n ->
      match n.node with
      | NK_Const _ -> () (* constants have no transposed value *)
      | _ ->
          (* If this node already has a T (it's a root), skip. Otherwise
           * compute T from contribs. *)
          if not (Hashtbl.mem t_value n.tag) then begin
            let parent_contribs =
              try Hashtbl.find contribs n.tag with Not_found -> []
            in
            let terms =
              List.filter_map
                (fun (sign, scale, parent) ->
                  match Hashtbl.find_opt t_value parent.tag with
                  | None -> None (* parent's T not computed; skip *)
                  | Some t_parent ->
                      let scaled =
                        match scale with
                        | None -> t_parent
                        | Some c -> mk_mul c t_parent
                      in
                      Some (sign, scaled))
                parent_contribs
            in
            let pos =
              List.filter_map
                (fun (s, t) -> if s > 0 then Some t else None)
                terms
            in
            let neg =
              List.filter_map
                (fun (s, t) -> if s < 0 then Some t else None)
                terms
            in
            let build lst =
              match lst with
              | [] -> mk_const 0.0
              | [ x ] -> x
              | x :: rest -> List.fold_left mk_add x rest
            in
            let t_n =
              match (pos, neg) with
              | [], [] -> mk_const 0.0
              | _, [] -> build pos
              | [], _ -> mk_neg (build neg)
              | _, _ -> mk_sub (build pos) (build neg)
            in
            Hashtbl.add t_value n.tag t_n
          end)
    !topo_rev;

  (* topo_rev is roots-first (DFS prepends after recursion → root at front). *)

  (* Step 4: Build new assigns. For each input Load (leaf with elem_ref),
   * the new assignment is (input_elem_ref, T[load]). *)
  let new_assigns =
    List.filter_map
      (fun n ->
        match n.node with
        | NK_Load r -> (
            match Hashtbl.find_opt t_value n.tag with
            | None -> None (* No T computed (e.g., load not in any sum) *)
            | Some t_n -> Some (r, t_n))
        | _ -> None)
      !topo_rev
  in
  new_assigns

(* === FMA LIFT PASS ===
 *
 * Recognize Add/Sub-of-Mul patterns where the inner Mul has use_count = 1
 * and rewrite them as NK_Fma atoms. After this pass, the codegen emits
 * each Fma as a single AVX-512 FMA intrinsic (vfmadd / vfmsub /
 * vfnmadd / vfnmsub) — one machine instruction instead of two.
 *
 * Patterns lifted (where M = Mul(a, b) and use_count(M) = 1):
 *
 *   Add(M, c)              →  Fma(a, b, c, neg_mul=F, neg_add=F)   a*b + c
 *   Add(c, M)              →  Fma(a, b, c, F, F)                   a*b + c
 *   Sub(M, c)              →  Fma(a, b, c, F, T)                   a*b - c
 *   Sub(c, M)              →  Fma(a, b, c, T, F)                  -a*b + c
 *
 * And also the negated-mul forms, where N = Neg(Mul(a, b)) with use_count(N)=1
 * and use_count(Mul(a,b))=1:
 *
 *   Add(N, c)              →  Fma(a, b, c, T, F)                  -a*b + c
 *   Add(c, N)              →  Fma(a, b, c, T, F)                  -a*b + c
 *   Sub(N, c)              →  Fma(a, b, c, T, T)                  -a*b - c
 *   Sub(c, N)              →  Fma(a, b, c, F, F)                   a*b + c
 *
 * Constraints:
 * - The lifted Mul (or Neg(Mul)) must have use_count = 1 — it has only
 *   ONE consumer (the Add/Sub being rewritten). Otherwise lifting would
 *   either DUPLICATE the mul (worse) or break sharing.
 * - This pass should run LAST (after factor/share/transpose). All
 *   downstream passes treat Fma as opaque.
 * - The pass is "conservative" in that it never lifts when use_count > 1.
 *   It does NOT try to push factoring back to enable more fusion. *)

(* fma_lift with optional [frozen_tags] set.
 *
 * When [frozen_tags] is supplied, any node whose tag is in the set is
 * returned UNCHANGED by the rewrite, preserving both its tag identity and
 * its entire subtree. This is required when fma_lift runs alongside the
 * SU+spill recipe: spill_markers reference tags BEFORE fma_lift; if those
 * tags are rewritten into Fma atoms, the markers point to nodes orphaned
 * from the reachable DAG. emit_c walks reachable nodes only, so spill
 * stores never emit, PASS 2 reloads garbage, and stale operand pointers
 * (cached in non-rewritten subtrees) reference undeclared tags.
 *
 * Mirror of the doc 30 / doc 31 fix for lift_sub_neg_mul: keep the rewrite
 * self-consistent with respect to ALL DAG roots (assigns + spill_markers),
 * not just assigns.
 *
 * Frozen-tag handling: parents of frozen nodes may still be rewritten;
 * their operand pointers continue to reference the frozen child (unchanged).
 * Children of frozen nodes are NOT visited by the rewrite walk through this
 * path, but are reachable via other parents if shared, in which case those
 * other paths produce normal rewrites. *)
(* === FACTOR CONSTANT MULS — SAFE PEEPHOLE ===
 *
 * Recognize the pattern Add(Mul(K, X), Mul(K, Y)) → Mul(K, Add(X, Y))
 * and similarly Sub(Mul(K, X), Mul(K, Y)) → Mul(K, Sub(X, Y)) where K
 * is a Const node. This is the FFTW-style factoring that enables
 * downstream FMA absorption: the resulting Mul(K, sum) can be lifted
 * into its consumer Add/Sub via multi_use_fma_lift.
 *
 * Why this is needed (FMA-at-expansion-time gap vs FFTW):
 *
 *   For radix-{2^k} composites, half the twiddles have |cr| = |ci| =
 *   1/√2 (the W^1, W^3, W^5, W^7 family). const_cmul emits these as
 *   Sub(Mul(xr, K), Mul(xi, K)) and Add(Mul(xr, K), Mul(xi, K)) — two
 *   separate K-multiplications combined by Add/Sub. fma_lift can't
 *   fuse them (multi-use), and the downstream Adds/Subs that consume
 *   the K-multiplied values stay as plain Add/Sub instead of FMAs.
 *
 *   FFTW's genfft builds the AST already factored: it computes
 *   xr+xi and xr-xi first, then K*sum and K*diff. Each K-multiply is
 *   a single Mul whose consumer Add/Sub fuses into an FMA.
 *
 * Why the previous `factor_common_muls` broke on composites:
 *
 *   1. It used `flatten` over the entire Add/Sub chain, destroying
 *      shared intermediate sums. A pre-existing shared partial-sum
 *      `s1 = a + b + K*c` referenced by multiple outputs would be
 *      shredded into [a, b, K*c] terms during one output's flatten,
 *      then rebuilt as a different sum that no longer hashcons-matches
 *      s1. Other outputs keep referencing the old s1, so we end up
 *      with both the old s1 chain AND the new restructured chain.
 *   2. It had no use-count safety. If Mul(K, X) was referenced by
 *      both a factor pattern AND an unrelated sum, factoring the
 *      factor pattern produced Mul(K, Add(X, Y)) as a new node, but
 *      the original Mul(K, X) stayed alive for the other use. Net:
 *      extra Mul, no savings.
 *   3. It didn't respect spill markers — frozen subtrees got their
 *      shared muls factored, orphaning the marker.
 *
 * This new pass fixes all three:
 *
 *   1. Bottom-up DAG rewrite (no flatten). Only LOCAL patterns
 *      Add(Mul,Mul)/Sub(Mul,Mul) are rewritten; everything else stays
 *      structurally identical. Shared intermediate sums survive.
 *   2. Full use-count safety: only factor when EVERY use of both
 *      input Muls is itself a factor pattern with the same K. After
 *      factoring, both Muls become unreachable (DCE'd at emit time).
 *   3. Frozen-tag awareness: refuses to factor any node whose tag
 *      is in frozen_tags, AND credits factor-parent only when neither
 *      child Mul is frozen. *)
let factor_const_muls ?(frozen_tags : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list * (int, int) Hashtbl.t =
  let is_frozen tag =
    match frozen_tags with None -> false | Some tbl -> Hashtbl.mem tbl tag
  in

  (* Raw rewrite log accumulated across all rounds: maps every node that
   * got rewritten in any round (n.tag) to its replacement (r.tag). At
   * the end, for each frozen original tag we walk this chain to find
   * the final tag the spill marker should point at. *)
  let rewrite_log : (int, int) Hashtbl.t = Hashtbl.create 64 in

  (* One round of factoring. Returns (new_assigns, fired) where `fired`
   * is true iff any factor peephole fired. We iterate to a fixed point
   * because a single round can create new Muls whose own parents are
   * themselves factor patterns at a higher level — the recursive case
   * needs use_counts recomputed on the rewritten DAG. *)
  let one_round (assigns : (Expr.elem_ref * t) list) :
      (Expr.elem_ref * t) list * bool =
    let fired = ref false in
    (* Step 1: compute global use_count across all assignments. *)
    let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    let visited1 : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    let bump_use tag =
      let c = try Hashtbl.find use_count tag with Not_found -> 0 in
      Hashtbl.replace use_count tag (c + 1)
    in
    let rec walk1 e =
      if not (Hashtbl.mem visited1 e.tag) then begin
        Hashtbl.add visited1 e.tag ();
        match e.node with
        | NK_Const _ | NK_Load _ -> ()
        | NK_Neg a ->
            bump_use a.tag;
            walk1 a
        | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
            bump_use a.tag;
            bump_use b.tag;
            walk1 a;
            walk1 b
        | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
            bump_use a.tag;
            bump_use b.tag;
            bump_use c.tag;
            bump_use d.tag;
            walk1 a;
            walk1 b;
            walk1 c;
            walk1 d
        | NK_Fma (a, b, c, _, _) ->
            bump_use a.tag;
            bump_use b.tag;
            bump_use c.tag;
            walk1 a;
            walk1 b;
            walk1 c
        | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1689"
      end
    in
    List.iter
      (fun (_, e) ->
        bump_use e.tag;
        walk1 e)
      assigns;

    let const_mul_of (n : t) : (t * t) option =
      match n.node with
      | NK_Mul (a, b) -> (
          match a.node with
          | NK_Const _ -> Some (a, b)
          | _ -> ( match b.node with NK_Const _ -> Some (b, a) | _ -> None))
      | _ -> None
    in

    (* Step 2: identify factor-pattern parents for each Mul. *)
    let factor_parent_count : (int, int) Hashtbl.t = Hashtbl.create 64 in
    let credit_factor tag =
      let c = try Hashtbl.find factor_parent_count tag with Not_found -> 0 in
      Hashtbl.replace factor_parent_count tag (c + 1)
    in
    let visited2 : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    let rec scan n =
      if not (Hashtbl.mem visited2 n.tag) then begin
        Hashtbl.add visited2 n.tag ();
        (match n.node with
        | NK_Add (a, b) | NK_Sub (a, b) -> (
            if (not (is_frozen a.tag)) && not (is_frozen b.tag) then
              match (const_mul_of a, const_mul_of b) with
              | Some (ka, _), Some (kb, _) when ka.tag = kb.tag ->
                  credit_factor a.tag;
                  credit_factor b.tag
              | _ -> ())
        | _ -> ());
        match n.node with
        | NK_Const _ | NK_Load _ -> ()
        | NK_Neg a -> scan a
        | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
            scan a;
            scan b
        | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
            scan a;
            scan b;
            scan c;
            scan d
        | NK_Fma (a, b, c, _, _) ->
            scan a;
            scan b;
            scan c
        | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1735"
      end
    in
    List.iter (fun (_, e) -> scan e) assigns;

    let safe_to_factor (m : t) : bool =
      if is_frozen m.tag then false
      else
        let uses = try Hashtbl.find use_count m.tag with Not_found -> 0 in
        let fuses =
          try Hashtbl.find factor_parent_count m.tag with Not_found -> 0
        in
        uses > 0 && uses = fuses
    in

    let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
    let rec rewrite (n : t) : t =
      (* Frozen node policy:
       *   Add/Sub: rewriting is value-preserving (factor is an algebraic
       *     identity). Allow rewrite; the rewrite_log records the new
       *     tag so spill markers can be retargeted at the end.
       *   Mul/Cmul/Fma/Neg: short-circuit. Rewriting these would either
       *     change the value, or the rewrite is trivial (just rebuild
       *     with possibly-rewritten children, no algebraic gain). Cheaper
       *     to preserve identity.
       *   Const/Load: no children, no rewrite needed regardless. *)
      let frozen_short_circuit =
        is_frozen n.tag
        && match n.node with NK_Add _ | NK_Sub _ -> false | _ -> true
      in
      if frozen_short_circuit then n
      else
        match Hashtbl.find_opt cache n.tag with
        | Some r -> r
        | None ->
            let r =
              match n.node with
              | NK_Const _ | NK_Load _ -> n
              | NK_Neg a ->
                  let a' = rewrite a in
                  if a' == a then n else mk_neg a'
              | NK_Mul (a, b) ->
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_mul a' b'
              | NK_Add (a, b) -> (
                  let a' = rewrite a in
                  let b' = rewrite b in
                  match try_factor a' b' true with
                  | Some folded ->
                      fired := true;
                      folded
                  | None ->
                      if a' == a && b' == b then n else mk_add_binary a' b')
              | NK_Sub (a, b) -> (
                  let a' = rewrite a in
                  let b' = rewrite b in
                  match try_factor a' b' false with
                  | Some folded ->
                      fired := true;
                      folded
                  | None ->
                      if a' == a && b' == b then n else mk_sub_binary a' b')
              | NK_CmulRe (a, b, c, d) ->
                  let a' = rewrite a in
                  let b' = rewrite b in
                  let c' = rewrite c in
                  let d' = rewrite d in
                  if a' == a && b' == b && c' == c && d' == d then n
                  else hashcons (NK_CmulRe (a', b', c', d'))
              | NK_CmulIm (a, b, c, d) ->
                  let a' = rewrite a in
                  let b' = rewrite b in
                  let c' = rewrite c in
                  let d' = rewrite d in
                  if a' == a && b' == b && c' == c && d' == d then n
                  else hashcons (NK_CmulIm (a', b', c', d'))
              | NK_Fma (a, b, c, nm, na) ->
                  let a' = rewrite a in
                  let b' = rewrite b in
                  let c' = rewrite c in
                  if a' == a && b' == b && c' == c then n
                  else hashcons (NK_Fma (a', b', c', nm, na))
              | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1776"
            in
            Hashtbl.add cache n.tag r;
            (* Log raw rewrite so the outer loop can chain remaps across
             * rounds and produce a final frozen-original → final-tag map. *)
            if r != n then Hashtbl.replace rewrite_log n.tag r.tag;
            r
    and try_factor (a : t) (b : t) (is_add : bool) : t option =
      if is_frozen a.tag || is_frozen b.tag then None
      else
        match (const_mul_of a, const_mul_of b) with
        | Some (ka, xa), Some (kb, xb) when ka.tag = kb.tag ->
            let sa = safe_to_factor a in
            let sb = safe_to_factor b in
            if Sys.getenv_opt "FACTOR_TRACE" <> None then
              Printf.eprintf
                "  try_factor(%s, t%d, t%d): const_mul_match=true, safe_a=%b \
                 (uses=%d, fuses=%d), safe_b=%b (uses=%d, fuses=%d)\n"
                (if is_add then "Add" else "Sub")
                a.tag b.tag sa
                (try Hashtbl.find use_count a.tag with Not_found -> -1)
                (try Hashtbl.find factor_parent_count a.tag
                 with Not_found -> -1)
                sb
                (try Hashtbl.find use_count b.tag with Not_found -> -1)
                (try Hashtbl.find factor_parent_count b.tag
                 with Not_found -> -1);
            if sa && sb then begin
              let inner =
                if is_add then mk_add_binary xa xb else mk_sub_binary xa xb
              in
              Some (mk_mul ka inner)
            end
            else None
        | _ -> None
    in
    let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in
    (new_assigns, !fired)
  in

  (* Iterate to fixed point. Cap at 20 rounds for paranoia. *)
  let max_rounds = 20 in
  let trace = Sys.getenv_opt "FACTOR_TRACE" <> None in
  let rec loop assigns rounds =
    if rounds >= max_rounds then begin
      if trace then
        Printf.eprintf "factor_const_muls: hit max_rounds %d\n" max_rounds;
      assigns
    end
    else
      let next, fired = one_round assigns in
      if trace then
        Printf.eprintf "factor_const_muls: round %d, fired=%b\n" rounds fired;
      if fired then loop next (rounds + 1) else next
  in
  let final_assigns = loop assigns 0 in

  (* Build frozen-original → final-tag remap by chaining rewrite_log.
   * Only frozen tags that actually got rewritten end up in the output. *)
  let final_remap : (int, int) Hashtbl.t = Hashtbl.create 16 in
  (match frozen_tags with
  | None -> ()
  | Some tbl ->
      Hashtbl.iter
        (fun frozen_orig () ->
          let rec chase t seen =
            if List.mem t seen then t (* cycle guard; shouldn't happen *)
            else
              match Hashtbl.find_opt rewrite_log t with
              | Some t' when t' <> t -> chase t' (t :: seen)
              | _ -> t
          in
          let final_t = chase frozen_orig [] in
          if final_t <> frozen_orig then
            Hashtbl.add final_remap frozen_orig final_t)
        tbl);

  if Sys.getenv_opt "FACTOR_TRACE" <> None then
    Printf.eprintf "factor_const_muls: remapped %d frozen tags\n"
      (Hashtbl.length final_remap);

  (final_assigns, final_remap)

let fma_lift ?(frozen_tags : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  let is_frozen tag =
    match frozen_tags with None -> false | Some tbl -> Hashtbl.mem tbl tag
  in
  (* Step 1: Compute global use_count over the assigns DAG. *)
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let visited : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let bump tag =
    let c = try Hashtbl.find use_count tag with Not_found -> 0 in
    Hashtbl.replace use_count tag (c + 1)
  in
  let rec walk e =
    if not (Hashtbl.mem visited e.tag) then begin
      Hashtbl.add visited e.tag ();
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a ->
          bump a.tag;
          walk a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          bump a.tag;
          bump b.tag;
          walk a;
          walk b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          bump a.tag;
          bump b.tag;
          bump c.tag;
          bump d.tag;
          walk a;
          walk b;
          walk c;
          walk d
      | NK_Fma (a, b, c, _, _) ->
          bump a.tag;
          bump b.tag;
          bump c.tag;
          walk a;
          walk b;
          walk c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1901"
    end
  in
  List.iter
    (fun (_, e) ->
      bump e.tag;
      walk e)
    assigns;

  let single_use n =
    (try Hashtbl.find use_count n.tag with Not_found -> 0) = 1
  in

  (* Lifting policy (doc 56 restoration of single_use):
   *
   * Earlier this function used `liftable_mul = true` unconditionally, with
   * the argument that duplicating shared Muls is "free" at asm level
   * because (a) each Fma computes a*b internally and (b) the original Mul
   * becomes unreachable if all consumers absorb it. That argument breaks
   * on composite codelets where:
   *   1. Shared Muls (e.g. twiddle products consumed by multiple
   *      butterflies) have non-Add consumers that keep the Mul alive,
   *      AND the duplicated Fmas each redo the multiplication in parallel.
   *   2. Even when the Mul becomes dead, N parallel Fmas issue N
   *      independent muls competing for FMA port throughput, where the
   *      shared schedule issued 1 mul.
   *
   * doc 28 measured this as 33-48% regression on R=32 t1 (910 vs 717 FP
   * instructions, vs hand's 709). The single_use restriction restores
   * the invariant: every lift is op-count-preserving (1 mul + 1 add → 1
   * fma), never op-count-increasing.
   *
   * For shared Muls (use_count > 1), fma_lift now leaves them as
   * Mul + Add patterns. gcc's pattern matcher still contracts them at
   * the asm level via `-O3 -ffp-contract=fast`, so we don't lose FMAs
   * for un-lifted patterns — we just keep the operand-ordering
   * flexibility that lets gcc pick the right fmadd variant during RA. *)
  let liftable_mul (n : t) : bool = single_use n in

  (* Step 2: Walk the DAG, lifting patterns greedily. Each Add/Sub examines
   * its operands; if one is a single-use Mul (or single-use Neg(Mul)),
   * lift to Fma. *)
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rewrite (n : t) : t =
    if is_frozen n.tag then n (* preserve frozen tag identity and subtree *)
    else
      match Hashtbl.find_opt cache n.tag with
      | Some r -> r
      | None ->
          let r =
            match n.node with
            | NK_Const _ | NK_Load _ -> n
            | NK_Neg inner ->
                let inner' = rewrite inner in
                if inner' == inner then n else mk_neg inner'
            | NK_Mul (a, b) ->
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_mul a' b'
            | NK_CmulRe (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d then n
                else hashcons (NK_CmulRe (a', b', c', d'))
            | NK_CmulIm (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d then n
                else hashcons (NK_CmulIm (a', b', c', d'))
            | NK_Fma (a, b, c, neg_mul, neg_add) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                if a' == a && b' == b && c' == c then n
                else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
            | NK_Add (a, b) -> (
                let a' = rewrite a in
                let b' = rewrite b in
                (* Try to lift one operand into an FMA. Try LEFT first; if it
                 * doesn't fuse, try RIGHT. Only one Mul can fuse per Add. *)
                match try_lift_add_operand a' b' with
                | Some fma -> fma
                | None -> (
                    match try_lift_add_operand b' a' with
                    | Some fma -> fma
                    | None ->
                        if a' == a && b' == b then n else mk_add_binary a' b'))
            | NK_Sub (a, b) -> (
                let a' = rewrite a in
                let b' = rewrite b in
                (* For Sub(a, b) we have two patterns:
                 *   Sub(M, c)  → Fma(a_m, b_m, c, F, T)   (a_m*b_m - c)   — fmsub
                 *   Sub(c, M)  → Fma(a_m, b_m, c, T, F)   (-a_m*b_m + c)  — fnmadd
                 *   Sub(N, c)  → Fma(a_m, b_m, c, T, T)   (-a_m*b_m - c)  — fnmsub  (N = Neg(M))
                 *   Sub(c, N)  → Fma(a_m, b_m, c, F, F)   (a_m*b_m + c)   — fmadd   (N = Neg(M))
                 *)
                match try_lift_sub_left a' b' with
                | Some fma -> fma
                | None -> (
                    match try_lift_sub_right a' b' with
                    | Some fma -> fma
                    | None ->
                        if a' == a && b' == b then n else mk_sub_binary a' b'))
            | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1954"
          in
          Hashtbl.add cache n.tag r;
          r
  (* Try to fuse `m_node + other` as an FMA — m_node is the candidate
   * Mul/Neg(Mul) operand of an Add, other is the addend.
   * Refuses if m_node is frozen (a spill-marker target) — lifting would
   * orphan m_node from the rewritten DAG. *)
  and try_lift_add_operand (m_node : t) (other : t) : t option =
    if is_frozen m_node.tag then None
    else
      match m_node.node with
      | NK_Mul (a, b) when liftable_mul m_node ->
          (* Fma(a, b, other, F, F) = a*b + other *)
          Some (hashcons (NK_Fma (a, b, other, false, false)))
      | NK_Neg inner when liftable_mul m_node -> (
          match inner.node with
          | NK_Mul (a, b) when liftable_mul inner && not (is_frozen inner.tag)
            ->
              (* Fma(a, b, other, T, F) = -a*b + other *)
              Some (hashcons (NK_Fma (a, b, other, true, false)))
          | _ -> None)
      | _ -> None
  (* Sub(left, right) where left is Mul/Neg(Mul). *)
  and try_lift_sub_left (left : t) (right : t) : t option =
    if is_frozen left.tag then None
    else
      match left.node with
      | NK_Mul (a, b) when liftable_mul left ->
          (* Fma(a, b, right, F, T) = a*b - right *)
          Some (hashcons (NK_Fma (a, b, right, false, true)))
      | NK_Neg inner when liftable_mul left -> (
          match inner.node with
          | NK_Mul (a, b) when liftable_mul inner && not (is_frozen inner.tag)
            ->
              (* Fma(a, b, right, T, T) = -a*b - right *)
              Some (hashcons (NK_Fma (a, b, right, true, true)))
          | _ -> None)
      | _ -> None
  (* Sub(left, right) where right is Mul/Neg(Mul). *)
  and try_lift_sub_right (left : t) (right : t) : t option =
    if is_frozen right.tag then None
    else
      match right.node with
      | NK_Mul (a, b) when liftable_mul right ->
          (* Fma(a, b, left, T, F) = -a*b + left *)
          Some (hashcons (NK_Fma (a, b, left, true, false)))
      | NK_Neg inner when liftable_mul right -> (
          match inner.node with
          | NK_Mul (a, b) when liftable_mul inner && not (is_frozen inner.tag)
            ->
              (* Fma(a, b, left, F, F) = a*b + left *)
              Some (hashcons (NK_Fma (a, b, left, false, false)))
          | _ -> None)
      | _ -> None
  in
  List.map (fun (oref, e) -> (oref, rewrite e)) assigns

(* === MULTI-USE FMA LIFT ===
 *
 * fma_lift requires single_use for absorption. This pass relaxes that:
 * a Mul M with N>1 uses can be absorbed IF every consumer is an Add/Sub
 * where M is a direct operand. Each consumer gets its own FMA that
 * duplicates the multiplication inside its fused mul-add unit.
 *
 * Op-count accounting per absorbed Mul:
 *   Before:  M (1 op) + N consumers each being Add/Sub (N ops)
 *   After:   M dead + N consumers each being Fma (N ops)
 *   Δ = -1 (the Mul disappears)
 *
 * Plus the consumers change from plain Add/Sub to FMA, which is what
 * lets us close the FMA-count gap vs FFTW.
 *
 * Why no throughput cost: an FMA instruction fuses one mul + one add
 * into a single µ-op on every modern CPU. Encoding the same (a, b)
 * mul operands N times across N FMAs is identical throughput to
 * computing a*b once and adding to N values — the mul work happens
 * inside each FMA's pipeline anyway.
 *
 * Pairing with factor_const_muls: that pass converts
 * Add(Mul(K,X),Mul(K,Y)) → Mul(K,Add(X,Y)). The resulting Mul has
 * multiple consumers (Add/Sub of K*sum with various values). Without
 * multi_use_fma_lift those Adds/Subs stay plain. With it, they each
 * become FMAs absorbing the shared Mul. *)
let multi_use_fma_lift ?(frozen_tags : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list * (int, int) Hashtbl.t =
  let is_frozen tag =
    match frozen_tags with None -> false | Some tbl -> Hashtbl.mem tbl tag
  in
  (* Records (old_tag → new_tag) when a frozen Add/Sub is rewritten
   * into an Fma. The Fma is algebraically equivalent (X ± K*Y → fma),
   * so any spill marker pointing at the old tag can be updated to
   * point at the new tag without changing the spilled value. *)
  let tag_remap : (int, int) Hashtbl.t = Hashtbl.create 16 in
  (* Phase 1: classify each Mul as absorbable.
   *
   * A Mul is absorbable iff every use is either:
   *   - direct operand of Add (becomes fmadd)
   *   - direct operand of Sub (becomes fmsub or fnmadd)
   *   - operand of Neg whose parent is Add/Sub (becomes fnmadd/fnmsub)
   * Any other context (operand of Mul, Cmul, Fma's mul operand or
   * addend operand, root assignment) disqualifies.
   *
   * Single-pass: walk the DAG; at each non-leaf, observe how its child
   * Muls are being used. Use a Hashtbl keyed by Mul tag with value
   * `false` = disqualified, `true` = still a candidate. *)
  let mul_status : (int, bool) Hashtbl.t = Hashtbl.create 64 in
  (* Track WHY each mul was disqualified, for diagnostics. *)
  let disqualify_reason : (int, string) Hashtbl.t = Hashtbl.create 64 in
  let disqualify_for tag reason =
    if not (Hashtbl.mem disqualify_reason tag) then
      Hashtbl.add disqualify_reason tag reason;
    Hashtbl.replace mul_status tag false
  in
  let note_use (m : t) ~absorbable =
    if is_frozen m.tag then disqualify_for m.tag "frozen"
    else begin
      let cur = Hashtbl.find_opt mul_status m.tag in
      match (cur, absorbable) with
      | Some false, _ -> ()
      | None, true -> Hashtbl.add mul_status m.tag true
      | None, false -> Hashtbl.add mul_status m.tag false
      | Some true, false -> Hashtbl.replace mul_status m.tag false
      | Some true, true -> ()
    end
  in
  let visited : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let rec scan (n : t) =
    if Hashtbl.mem visited n.tag then ()
    else begin
      Hashtbl.add visited n.tag ();
      match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg inner ->
          (* Neg by itself doesn't make a child Mul absorbable; we'd need
           * to know the Neg's parent context (was it inside Add/Sub?).
           * To keep the analysis local we mark inner Mul "non-absorbable"
           * here. The Add/Sub branch separately checks Neg(Mul) patterns
           * and marks absorbable from there — but for safety, any Neg(Mul)
           * encountered outside an Add/Sub will disqualify. *)
          (match inner.node with
          | NK_Mul _ -> disqualify_for inner.tag "neg-outside-add-sub"
          | _ -> ());
          scan inner
      | NK_Add (a, b) ->
          (match a.node with
          | NK_Mul _ -> note_use a ~absorbable:true
          | _ -> ());
          (match b.node with
          | NK_Mul _ -> note_use b ~absorbable:true
          | _ -> ());
          scan a;
          scan b
      | NK_Sub (a, b) ->
          (match a.node with
          | NK_Mul _ -> note_use a ~absorbable:true
          | _ -> ());
          (match b.node with
          | NK_Mul _ -> note_use b ~absorbable:true
          | _ -> ());
          scan a;
          scan b
      | NK_Mul (a, b) ->
          (* Mul nested in Mul disqualifies the inner Muls. *)
          (match a.node with
          | NK_Mul _ -> disqualify_for a.tag "nested-in-mul"
          | _ -> ());
          (match b.node with
          | NK_Mul _ -> disqualify_for b.tag "nested-in-mul"
          | _ -> ());
          scan a;
          scan b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          List.iter
            (fun x ->
              match x.node with
              | NK_Mul _ -> disqualify_for x.tag "in-cmul"
              | _ -> ())
            [ a; b; c; d ];
          scan a;
          scan b;
          scan c;
          scan d
      | NK_Fma (a, b, c, _, _) ->
          (* Fma operands: any Mul here is in a slot that already has a
           * mul (a*b) or is the addend (c). Either way, NOT absorbable
           * into THIS Fma — we'd need a hypothetical "nested Fma" which
           * doesn't exist. Disqualify them. *)
          (match a.node with
          | NK_Mul _ -> disqualify_for a.tag "in-fma-mul-slot"
          | _ -> ());
          (match b.node with
          | NK_Mul _ -> disqualify_for b.tag "in-fma-mul-slot"
          | _ -> ());
          (match c.node with
          | NK_Mul _ -> disqualify_for c.tag "in-fma-addend"
          | _ -> ());
          scan a;
          scan b;
          scan c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2132"
    end
  in
  (* A Mul at a root (= a top-level assignment value) cannot be
   * absorbed because the assignment value IS the Mul; there's no
   * Add/Sub consumer. Mark root-Muls as disqualified. *)
  List.iter
    (fun (_, e) ->
      (match e.node with
      | NK_Mul _ -> disqualify_for e.tag "root-assignment"
      | _ -> ());
      scan e)
    assigns;
  let is_absorbable (m : t) : bool =
    match Hashtbl.find_opt mul_status m.tag with
    | Some true -> true
    | _ -> false
  in
  if Sys.getenv_opt "MULIFT_TRACE" <> None then begin
    let absorbed =
      Hashtbl.fold (fun _ v c -> if v then c + 1 else c) mul_status 0
    in
    let total = Hashtbl.length mul_status in
    Printf.eprintf
      "multi_use_fma_lift: %d Muls classified absorbable / %d total\n" absorbed
      total;
    (* Group disqualified muls by reason *)
    let by_reason : (string, int) Hashtbl.t = Hashtbl.create 8 in
    Hashtbl.iter
      (fun _ reason ->
        let c = try Hashtbl.find by_reason reason with Not_found -> 0 in
        Hashtbl.replace by_reason reason (c + 1))
      disqualify_reason;
    Hashtbl.iter
      (fun reason c ->
        Printf.eprintf "  disqualified: %d Muls due to '%s'\n" c reason)
      by_reason
  end;

  (* Phase 2: rewrite. Add/Sub nodes whose operand is an absorbable
   * Mul become the corresponding Fma.
   *
   * Frozen-tag handling: we DO rewrite frozen Add/Sub nodes (the
   * Fma we produce is algebraically identical, so spill markers
   * pointing at the old tag can be remapped to the new tag). We
   * do NOT absorb Muls that are themselves frozen — absorbing a
   * Mul folds K*X into a parent Fma whose value is K*X + Y, not
   * K*X. The Mul-frozen disqualification is enforced in phase 1
   * via note_use's is_frozen check. *)
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rewrite (n : t) : t =
    match Hashtbl.find_opt cache n.tag with
    | Some r -> r
    | None ->
        let r =
          match n.node with
          | NK_Const _ | NK_Load _ -> n
          | NK_Neg a ->
              let a' = rewrite a in
              if a' == a then n else mk_neg a'
          | NK_Mul (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_mul a' b'
          | NK_Add (a, b) -> (
              let a' = rewrite a in
              let b' = rewrite b in
              let trace = Sys.getenv_opt "MULFMA_TRACE" <> None in
              if trace then begin
                let is_m x =
                  match x.node with NK_Mul _ -> true | _ -> false
                in
                let abs_a = is_m a' && is_absorbable a' in
                let abs_b = is_m b' && is_absorbable b' in
                if is_m a' || is_m b' then
                  Printf.eprintf
                    "  mfl Add(t%d, t%d): a'=t%d (mul=%b absorb=%b) b'=t%d \
                     (mul=%b absorb=%b)\n"
                    a.tag b.tag a'.tag (is_m a') abs_a b'.tag (is_m b') abs_b
              end;
              (* Add(Mul, c) → Fma(_, _, c, F, F).
               * Add(c, Mul) → Fma(_, _, c, F, F). *)
              match a'.node with
              | NK_Mul (ma, mb) when is_absorbable a' ->
                  hashcons (NK_Fma (ma, mb, b', false, false))
              | _ -> (
                  match b'.node with
                  | NK_Mul (ma, mb) when is_absorbable b' ->
                      hashcons (NK_Fma (ma, mb, a', false, false))
                  | _ -> if a' == a && b' == b then n else mk_add_binary a' b'))
          | NK_Sub (a, b) -> (
              let a' = rewrite a in
              let b' = rewrite b in
              let trace = Sys.getenv_opt "MULFMA_TRACE" <> None in
              if trace then begin
                let is_m x =
                  match x.node with NK_Mul _ -> true | _ -> false
                in
                let abs_a = is_m a' && is_absorbable a' in
                let abs_b = is_m b' && is_absorbable b' in
                if is_m a' || is_m b' then
                  Printf.eprintf
                    "  mfl Sub(t%d, t%d): a'=t%d (mul=%b absorb=%b) b'=t%d \
                     (mul=%b absorb=%b)\n"
                    a.tag b.tag a'.tag (is_m a') abs_a b'.tag (is_m b') abs_b
              end;
              (* Sub(Mul, c) → Fma(_, _, c, F, T) = a*b - c (fmsub).
               * Sub(c, Mul) → Fma(_, _, c, T, F) = -a*b + c (fnmadd). *)
              match a'.node with
              | NK_Mul (ma, mb) when is_absorbable a' ->
                  hashcons (NK_Fma (ma, mb, b', false, true))
              | _ -> (
                  match b'.node with
                  | NK_Mul (ma, mb) when is_absorbable b' ->
                      hashcons (NK_Fma (ma, mb, a', true, false))
                  | _ -> if a' == a && b' == b then n else mk_sub_binary a' b'))
          | NK_CmulRe (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulRe (a', b', c', d'))
          | NK_CmulIm (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulIm (a', b', c', d'))
          | NK_Fma (a, b, c, nm, na) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              if a' == a && b' == b && c' == c then n
              else hashcons (NK_Fma (a', b', c', nm, na))
          | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2220"
        in
        Hashtbl.add cache n.tag r;
        (* If a frozen node was rewritten to a different node, record
         * the mapping so callers can update spill markers / other
         * external references that point at the old tag. *)
        if r != n && is_frozen n.tag then Hashtbl.replace tag_remap n.tag r.tag;
        r
  in
  let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in
  (new_assigns, tag_remap)

(* === FMA-ADDEND FACTOR PASS ===
 *
 * Recognizes the pattern Fma(K, X, Mul(K, Y), nm, na) where the FMA's
 * mul slot and the addend's Mul share the same constant K. This is a
 * factor opportunity:
 *
 *   nm=F, na=F:  K·X + K·Y = K·(X + Y)              → Mul(K, Add(X, Y))
 *   nm=F, na=T:  K·X − K·Y = K·(X − Y)              → Mul(K, Sub(X, Y))
 *   nm=T, na=F: −K·X + K·Y = K·(Y − X)              → Mul(K, Sub(Y, X))
 *   nm=T, na=T: −K·X − K·Y = −K·(X + Y)             → Neg(Mul(K, Add(X, Y)))
 *
 * WHY THIS MATTERS:
 * After fma_lift / multi_use_fma_lift run, the surviving Muls are
 * typically those whose uses include FMA-addend slots — multi_use's
 * absorbability check rejects them because not all uses are Add/Sub
 * direct-operand. But if those FMAs happen to use the SAME K constant
 * as the Mul, we can refactor: the K-multiplication folds out, and
 * the inner sum/diff becomes a single Add/Sub. The resulting outer
 * Mul(K, Sum) is then a NEW Mul whose uses (from the downstream
 * consumers of the original Fmas) are typically Add/Sub direct-
 * operand — so multi_use_fma_lift on a follow-up pass absorbs it.
 *
 * Net effect for the t290 / t311 / t358 case at R=16:
 *   Before: 1 Mul + 2 FMAs + 4 downstream Add/Sub = 7 ops
 *   After:  2 Add/Sub (sum, diff) + 4 FMAs (absorbed)  = 6 ops
 *   Savings: 1 op per such Fma-pair.
 *
 * SAFETY:
 * - We only fire when ALL uses of the addend Mul are such factor-
 *   pattern Fmas. Otherwise removing the Mul would change the value
 *   computed at some other consumer.
 * - We don't touch frozen Muls (the Mul value might be a spill target).
 * - Frozen Fmas that get rewritten are tracked in tag_remap so spill
 *   markers can be retargeted.
 *)
let fma_addend_factor ?(frozen_tags : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list * (int, int) Hashtbl.t =
  let is_frozen tag =
    match frozen_tags with None -> false | Some tbl -> Hashtbl.mem tbl tag
  in
  let tag_remap : (int, int) Hashtbl.t = Hashtbl.create 16 in

  (* Identify the shared K and the (X, Y) for an Fma(a, b, c, _, _)
   * where c = Mul(m1, m2). Returns Some (k, x, y) if K is a Const
   * appearing as one of (a, b) and as one of (m1, m2). Otherwise None. *)
  let identify_kxy (a : t) (b : t) (m1 : t) (m2 : t) : (t * t * t) option =
    let is_const n = match n.node with NK_Const _ -> true | _ -> false in
    if is_const a && a.tag = m1.tag then Some (a, b, m2)
    else if is_const a && a.tag = m2.tag then Some (a, b, m1)
    else if is_const b && b.tag = m1.tag then Some (b, a, m2)
    else if is_const b && b.tag = m2.tag then Some (b, a, m1)
    else None
  in

  (* Step 1: count uses of each node, AND count "factor-pattern" uses
   * for each Mul (i.e., uses where the Mul appears as the addend of
   * an Fma whose mul slot shares its constant). *)
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 64 in
  let factor_use_count : (int, int) Hashtbl.t = Hashtbl.create 64 in
  let bump tbl t =
    let c = try Hashtbl.find tbl t with Not_found -> 0 in
    Hashtbl.replace tbl t (c + 1)
  in
  let visited = Hashtbl.create 256 in
  let rec scan (n : t) =
    if not (Hashtbl.mem visited n.tag) then begin
      Hashtbl.add visited n.tag ();
      match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a ->
          bump use_count a.tag;
          scan a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          bump use_count a.tag;
          bump use_count b.tag;
          scan a;
          scan b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          bump use_count a.tag;
          bump use_count b.tag;
          bump use_count c.tag;
          bump use_count d.tag;
          scan a;
          scan b;
          scan c;
          scan d
      | NK_Fma (a, b, c, _, _) -> (
          bump use_count a.tag;
          bump use_count b.tag;
          bump use_count c.tag;
          scan a;
          scan b;
          scan c;
          (* Factor patterns:
           *   Type A: c = Mul(K, Y)        — direct addend Mul
           *   Type B: c = Neg(Mul(K, Y))   — Neg-wrapped addend Mul
           * In both cases, the FMA's mul slot must also use K. The
           * "factor target" is c (the addend whose tag we credit).
           * The Type B case appears when Path B's outer Mul gets
           * negated (e.g., from a `-cos*X` rotation where Path B
           * preserved the negative sign in the outer factor before
           * mk_const canonicalized to Neg(Const)). The rewrite is the
           * same as Type A but with `na` inverted. *)
          match c.node with
          | NK_Mul (m1, m2) -> (
              match identify_kxy a b m1 m2 with
              | Some _ -> bump factor_use_count c.tag
              | None -> ())
          | NK_Neg inner -> (
              match inner.node with
              | NK_Mul (m1, m2) -> (
                  match identify_kxy a b m1 m2 with
                  | Some _ -> bump factor_use_count c.tag
                  | None -> ())
              | _ -> ())
          | _ -> ())
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2387"
    end
  in
  List.iter (fun (_, e) -> scan e) assigns;

  let safe_to_factor (m : t) : bool =
    if is_frozen m.tag then false
    else
      let uses = try Hashtbl.find use_count m.tag with Not_found -> 0 in
      let fuses =
        try Hashtbl.find factor_use_count m.tag with Not_found -> 0
      in
      uses > 0 && uses = fuses
  in

  if Sys.getenv_opt "FMA_ADDEND_TRACE" <> None then begin
    let n_candidates = ref 0 in
    Hashtbl.iter
      (fun tag _ ->
        if
          safe_to_factor { tag; node = NK_Const 0.0 (* dummy; only tag used *) }
        then incr n_candidates)
      factor_use_count;
    Printf.eprintf "fma_addend_factor: %d candidate Muls\n" !n_candidates
  end;

  (* Step 2: rewrite. For each Fma whose addend is a factor-safe Mul
   * sharing the same K, fold to the equivalent Mul(K, Sum/Diff) or
   * Neg(Mul(K, Sum)) form. *)
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rewrite (n : t) : t =
    match Hashtbl.find_opt cache n.tag with
    | Some r -> r
    | None ->
        let r =
          match n.node with
          | NK_Const _ | NK_Load _ -> n
          | NK_Neg a ->
              let a' = rewrite a in
              if a' == a then n else mk_neg a'
          | NK_Mul (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_mul a' b'
          | NK_Add (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_add_binary a' b'
          | NK_Sub (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_sub_binary a' b'
          | NK_CmulRe (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulRe (a', b', c', d'))
          | NK_CmulIm (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulIm (a', b', c', d'))
          | NK_Fma (a, b, c, nm, na) -> (
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              (* Try factor pattern on the rewritten children. We use the
               * ORIGINAL c (not c') for the safe_to_factor check because
               * the use_count was built on the original DAG.
               *
               * Two factor patterns share the same identify_kxy machinery:
               *   Type A: c' = Mul(K, Y)         → fold with (nm, na)
               *   Type B: c' = Neg(Mul(K, Y))    → fold with (nm, !na)
               * because  Fma(K, X, -Mul(K,Y), nm, na)
               *        = Fma(K, X,  Mul(K,Y), nm, !na). *)
              let try_factor () =
                let try_with_mul m1 m2 na_eff =
                  match identify_kxy a' b' m1 m2 with
                  | Some (k, x, y) ->
                      let folded =
                        match (nm, na_eff) with
                        | false, false -> mk_mul k (mk_add_binary x y)
                        | false, true -> mk_mul k (mk_sub_binary x y)
                        | true, false -> mk_mul k (mk_sub_binary y x)
                        | true, true -> mk_neg (mk_mul k (mk_add_binary x y))
                      in
                      if Sys.getenv_opt "FMA_ADDEND_TRACE" <> None then
                        Printf.eprintf
                          "[fma_addend] rewrite t%d (Fma nm=%b na=%b \
                           na_eff=%b) → t%d  K=t%d X=t%d Y=t%d  c=t%d→t%d\n"
                          n.tag nm na na_eff folded.tag k.tag x.tag y.tag c.tag
                          c'.tag;
                      Some folded
                  | None -> None
                in
                match (c.node, c'.node) with
                | _, NK_Mul (m1, m2) when safe_to_factor c ->
                    try_with_mul m1 m2 na
                | _, NK_Neg inner when safe_to_factor c -> (
                    match inner.node with
                    | NK_Mul (m1, m2) -> try_with_mul m1 m2 (not na)
                    | _ -> None)
                | _ -> None
              in
              match try_factor () with
              | Some folded -> folded
              | None ->
                  if a' == a && b' == b && c' == c then n
                  else hashcons (NK_Fma (a', b', c', nm, na)))
          | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2454"
        in
        Hashtbl.add cache n.tag r;
        if r != n && is_frozen n.tag then Hashtbl.replace tag_remap n.tag r.tag;
        r
  in
  let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in

  (* Reachability sanity check: walk the new assigns and collect every
   * tag transitively referenced. Any tag this pass produced as an
   * operand of a node we emitted should be in this set. *)
  if Sys.getenv_opt "FMA_ADDEND_TRACE" <> None then begin
    let reach = Hashtbl.create 256 in
    let rec walk (n : t) =
      if not (Hashtbl.mem reach n.tag) then begin
        Hashtbl.add reach n.tag ();
        match n.node with
        | NK_Const _ | NK_Load _ -> ()
        | NK_Neg a -> walk a
        | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
            walk a;
            walk b
        | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
            walk a;
            walk b;
            walk c;
            walk d
        | NK_Fma (a, b, c, _, _) ->
            walk a;
            walk b;
            walk c
        | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2519"
      end
    in
    List.iter (fun (_, e) -> walk e) new_assigns;
    Printf.eprintf "[fma_addend] post-pass reachable tags: %d\n"
      (Hashtbl.length reach)
  end;

  (new_assigns, tag_remap)

(* === FLATTEN FMA-MUL ADDEND INTO 2-FMA CHAIN ===
 *
 * Recognizes the residual Cat-B pattern that survives multi_use_fma_lift
 * and fma_addend_factor:
 *
 *     Add(P, Fma(A, B, Mul(C, D), nm, na))
 *     Sub(P, Fma(A, B, Mul(C, D), nm, na))
 *     Sub(Fma(A, B, Mul(C, D), nm, na), P)
 *
 * The Fma's addend is a Mul whose constants don't match the Fma's
 * mul-slot constants, so fma_addend_factor doesn't fire. But the
 * outer Add/Sub gives us a third operand P that we can use as the
 * addend of a 2-FMA chain — eliminating both the standalone Mul and
 * the outer Add/Sub.
 *
 * Rewrite (with μ = -1 if nm else +1, ν = -1 if na else +1):
 *
 *   Add(P, Fma(a,b,Mul(c,d),nm,na))  =  P + μ·a·b + ν·c·d
 *     → Fma(c, d, Fma(a, b, P, nm,    false), na,    false)
 *
 *   Sub(P, Fma(a,b,Mul(c,d),nm,na))  =  P - μ·a·b - ν·c·d
 *     → Fma(c, d, Fma(a, b, P, !nm,   false), !na,   false)
 *
 *   Sub(Fma(a,b,Mul(c,d),nm,na), P)  =  μ·a·b + ν·c·d - P
 *     → Fma(c, d, Fma(a, b, P, nm,    true ), na,    false)
 *
 * In each case: 1 mul + 1 fma + 1 add/sub → 2 fma (save 1 op).
 *
 * Safety: outer Fma and inner Mul both must be single-use. Otherwise
 * the rewrite duplicates work in other consumers.
 *
 * Frozen handling: skip the rewrite if any of the participating tags
 * (the outer Add/Sub, the Fma, or the Mul) is frozen. Track tag_remap
 * for the outer node so downstream spill_info stays consistent.
 *)
let flatten_fma_mul_addend ?(frozen_tags : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list * (int, int) Hashtbl.t =
  let is_frozen tag =
    match frozen_tags with None -> false | Some tbl -> Hashtbl.mem tbl tag
  in
  let tag_remap : (int, int) Hashtbl.t = Hashtbl.create 16 in

  (* Step 1: count uses over the original DAG. *)
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let bump t =
    let c = try Hashtbl.find use_count t with Not_found -> 0 in
    Hashtbl.replace use_count t (c + 1)
  in
  let visited = Hashtbl.create 256 in
  let rec scan (n : t) =
    if not (Hashtbl.mem visited n.tag) then begin
      Hashtbl.add visited n.tag ();
      match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a ->
          bump a.tag;
          scan a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          bump a.tag;
          bump b.tag;
          scan a;
          scan b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          bump a.tag;
          bump b.tag;
          bump c.tag;
          bump d.tag;
          scan a;
          scan b;
          scan c;
          scan d
      | NK_Fma (a, b, c, _, _) ->
          bump a.tag;
          bump b.tag;
          bump c.tag;
          scan a;
          scan b;
          scan c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2591"
    end
  in
  List.iter
    (fun (_, e) ->
      bump e.tag;
      scan e)
    assigns;
  let single_use (n : t) =
    (try Hashtbl.find use_count n.tag with Not_found -> 0) = 1
  in

  (* Track parent node types for each Fma-with-Mul-addend. Used to decide
   * if relaxing single-use is safe (all consumers should be Add/Sub).
   * Always built (not just under TRACE) because the rewrite logic
   * consults this table to allow safe multi-use rewrites. *)
  let fma_parents : (int, string list) Hashtbl.t = Hashtbl.create 32 in
  let add_parent_kind addend_kind tag =
    let cur = try Hashtbl.find fma_parents tag with Not_found -> [] in
    Hashtbl.replace fma_parents tag (addend_kind :: cur)
  in
  let parent_kind = function
    | NK_Const _ -> "Const"
    | NK_Load _ -> "Load"
    | NK_Neg _ -> "Neg"
    | NK_Add _ -> "Add"
    | NK_Sub _ -> "Sub"
    | NK_Mul _ -> "Mul"
    | NK_Fma _ -> "Fma"
    | NK_CmulRe _ -> "CmulRe"
    | NK_CmulIm _ -> "CmulIm"
    | NK_Plus _ -> "Plus"
  in
  let parent_visited = Hashtbl.create 256 in
  let rec parent_scan (n : t) =
    if not (Hashtbl.mem parent_visited n.tag) then begin
      Hashtbl.add parent_visited n.tag ();
      let me_kind = parent_kind n.node in
      let note_child c =
        match c.node with
        | NK_Fma (_, _, addend, _, _) -> (
            match addend.node with
            | NK_Mul _ -> add_parent_kind me_kind c.tag
            | _ -> ())
        | _ -> ()
      in
      match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a ->
          note_child a;
          parent_scan a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          note_child a;
          note_child b;
          parent_scan a;
          parent_scan b
      | NK_Fma (a, b, c, _, _) ->
          note_child a;
          note_child b;
          note_child c;
          parent_scan a;
          parent_scan b;
          parent_scan c
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          note_child a;
          note_child b;
          note_child c;
          note_child d;
          parent_scan a;
          parent_scan b;
          parent_scan c;
          parent_scan d
      | NK_Plus _ -> ()
    end
  in
  List.iter
    (fun (oref, e) ->
      (* Treat the root assignment as a "parent" for the immediate root expr. *)
      (match e.node with
      | NK_Fma (_, _, addend, _, _) -> (
          match addend.node with
          | NK_Mul _ -> add_parent_kind "Root" e.tag
          | _ -> ())
      | _ -> ());
      parent_scan e;
      ignore oref)
    assigns;

  (* Predicate: is this Fma safely rewriteable, considering multi-use?
   * - Single-use → always safe (the original Fma becomes dead after rewrite).
   * - Multi-use → safe only if all parents are Add/Sub. The bottom-up
   *   walk visits each Add/Sub parent independently and rewrites locally;
   *   after all are rewritten, the original Fma+Mul are unreachable.
   *   If any parent is not Add/Sub (e.g., another Fma), that parent will
   *   still reference the original Fma, leaving a dangling chain.
   *
   * ============================================================
   * RUNTIME GATE — why this is env-gated default-OFF
   * ============================================================
   *
   * Empirically (Xeon, AVX-512, K=8, best-of-11 × 200k calls):
   *
   *   Radix  Δmuls  Δfmas  Δadd/sub  Δtotal   Runtime
   *   -----  -----  -----  --------  ------   -------
   *   R=25     -4    +12    -8         0      -3.12% (faster)
   *   R=32     -4    +12    -8         0      -2.81% (faster)
   *   R=64    -20    +60   -40         0      +6.75% (SLOWER)
   *
   * All three are op-count neutral. R=25/R=32 speed up modestly, but
   * R=64 regresses by ~6% — so this pass is opt-in by env flag, not
   * a default optimization. The cause is a subtle critical-path effect
   * worth understanding before re-enabling unconditionally.
   *
   * --- Per-chain analysis ---
   *
   * The rewrite turns
   *
   *   Add(P, Fma(a, b, Mul(c, d)))
   *
   * into
   *
   *   Fma(c, d, Fma(a, b, P))    [= c*d + (a*b + P)]
   *
   * On a single chain, the dependency graph favors the rewrite:
   *
   *   Baseline                         Relaxed
   *   --------                         -------
   *   m   = c*d         (vmulpd)       inner = a*b + P     (vfmadd)
   *   f   = a*b + m     (vfmadd)       result = c*d + inner (vfmadd)
   *   res = P + f       (vaddpd)
   *
   *   Critical path:                   Critical path:
   *     ready_cd ──→ mul ──→ fma ──→ add    ready_P ──→ fma ──→ fma
   *     3 ops × 4c = 12 cycles            2 ops × 4c = 8 cycles
   *
   * Relaxed wins per chain — IF the chain runs in isolation.
   *
   * --- Why R=64 nonetheless regresses ---
   *
   * In baseline, `Mul(c, d)` is a STANDALONE instruction. The OoO
   * scheduler can issue it as soon as c and d are ready, completely
   * independent of when P arrives. With 20 such Muls at R=64, the
   * scheduler spreads them across the FMA ports during whatever cycles
   * are otherwise idle — they're "free fill" for the execution units.
   *
   * In relaxed, `c*d` is BURIED inside the outer FMA `Fma(c, d, inner)`,
   * which cannot issue until `inner` is ready. The multiplication of c*d
   * is now bottlenecked by the inner FMA's 4-cycle latency. The
   * scheduler loses 20 free-floating muls that could fill bubbles.
   *
   * Additional cost from register pressure: each chain extends the
   * liveness of c, d, a, b across an extra FMA, requiring the compiler
   * to insert ~2 extra vmovapd per chain. Measured at R=64: +44 reg-to-
   * reg copies in relaxed vs baseline (440 → 484), even though stack
   * spill counts are unchanged. At 20 chains the cumulative frontend
   * pressure tips the balance against the rewrite.
   *
   * R=25/R=32 fire only 4 paired rewrites each → too few chains to
   * exhaust the OoO window's ability to overlap them, so the per-chain
   * critical-path win dominates.
   *
   * --- Why FFTW's genfft doesn't do this rewrite ---
   *
   * FFTW emits explicit standalone Muls and relies on the C compiler
   * (gcc/clang) to fuse `K * X` into FMA when it judges fusion is
   * beneficial. The standalone-Mul form preserves scheduling freedom;
   * compilers are conservative about over-chaining FMAs precisely
   * because of the issue analyzed above. VFFT emits FMA intrinsics
   * directly, which is why we have to make this tradeoff explicit at
   * the algsimp level rather than delegating to the C compiler.
   *
   * --- Operational summary ---
   *
   * Default: AUTO (density-gated, see below). Empirically tuned to
   *   enable at R=25/R=32 (low chain density → wins) and disable at
   *   R=64 (high chain density → loses). No env flag needed for
   *   correct default behavior.
   *
   * Env override:  VFFT_FMA_MULTIUSE=0  forces OFF (single_use only)
   *                VFFT_FMA_MULTIUSE=1  forces ON  (no density gate)
   *                unset                → AUTO (default)
   * ============================================================ *)

  (* Density gate threshold: when more than this many Fmas would
   * rewrite cleanly (multi_use_safe + single_use addend), the OoO
   * window can't overlap the resulting 2-FMA chains; disable.
   *
   * Empirical data (measured rewrite-eligible Fma counts):
   *   R=25:  4 candidates  → win  (-3.12%)
   *   R=32:  4 candidates  → win  (-2.81%)
   *   R=64: 20 candidates  → loss (+6.75%)
   *
   * Threshold of 12 sits comfortably between win/loss regimes with
   * margin. Future radices (R=49, R=121, etc.) that land in the
   * 5-11 candidate range are likely safe but unverified; the env
   * override lets users force on/off if measurement disagrees.
   *
   * Counted: Fmas where (a) all parents are Add/Sub AND (b) the Mul
   * addend has use_count=1. This matches the actual rewrite conditions
   * in match_fma_mul. Both-match-bailout Fmas are still counted (they
   * pass the structural test but get rejected at rewrite time), which
   * slightly overcounts at R=64 but doesn't change the gate decision. *)
  let multiuse_density_threshold = 12 in

  let is_rewriteable_consumer = function "Add" | "Sub" -> true | _ -> false in

  (* Count Fmas eligible for multi-use rewrite. Walks the DAG once;
   * O(N) in DAG size, runs even when env forces off so the diagnostic
   * count is always available. *)
  let multiuse_candidate_count =
    let visited = Hashtbl.create 256 in
    let count = ref 0 in
    let rec walk (n : t) =
      if Hashtbl.mem visited n.tag then ()
      else begin
        Hashtbl.add visited n.tag ();
        (match n.node with
        | NK_Fma (_, _, addend, _, _) when not (is_frozen n.tag) -> (
            match addend.node with
            | NK_Mul _
              when (not (is_frozen addend.tag))
                   && (try Hashtbl.find use_count addend.tag
                       with Not_found -> 0)
                      = 1 -> (
                (* Mul addend is single-use; check parent kinds. *)
                match Hashtbl.find_opt fma_parents n.tag with
                | Some kinds
                  when kinds <> [] && List.for_all is_rewriteable_consumer kinds
                  ->
                    (* Also exclude single-use Fmas — those already rewrite under
                     * the unrelaxed rule and don't count toward density pressure. *)
                    let use =
                      try Hashtbl.find use_count n.tag with Not_found -> 0
                    in
                    if use > 1 then incr count
                | _ -> ())
            | _ -> ())
        | _ -> ());
        match n.node with
        | NK_Const _ | NK_Load _ -> ()
        | NK_Neg a -> walk a
        | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
            walk a;
            walk b
        | NK_Fma (a, b, c, _, _) ->
            walk a;
            walk b;
            walk c
        | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
            walk a;
            walk b;
            walk c;
            walk d
        | NK_Plus _ -> ()
      end
    in
    List.iter (fun (_, e) -> walk e) assigns;
    !count
  in

  (* Resolve the gate decision: env override wins; otherwise apply density. *)
  let multiuse_enabled, multiuse_decision_reason =
    match Sys.getenv_opt "VFFT_FMA_MULTIUSE" with
    | Some "0" -> (false, "forced OFF by env")
    | Some "1" -> (true, "forced ON by env")
    | _ ->
        if multiuse_candidate_count <= multiuse_density_threshold then
          ( true,
            Printf.sprintf "AUTO ON (count=%d ≤ threshold=%d)"
              multiuse_candidate_count multiuse_density_threshold )
        else
          ( false,
            Printf.sprintf "AUTO OFF (count=%d > threshold=%d)"
              multiuse_candidate_count multiuse_density_threshold )
  in

  let multi_use_safe (n : t) =
    if not multiuse_enabled then false
    else
      match Hashtbl.find_opt fma_parents n.tag with
      | None -> false (* No parent info (shouldn't happen for valid Fma-Mul) *)
      | Some kinds -> kinds <> [] && List.for_all is_rewriteable_consumer kinds
  in
  if Sys.getenv_opt "FLATTEN_FMA_MUL_TRACE" <> None then begin
    Printf.eprintf "  [parent-scan complete] fma_parents table size = %d\n"
      (Hashtbl.length fma_parents);
    Printf.eprintf "  [multi-use gate] candidates=%d, decision=%s\n"
      multiuse_candidate_count multiuse_decision_reason
  end;

  (* Helper: if n is Fma(a, b, Mul(c, d), nm, na) with safe-to-rewrite
   * conditions, return Some (a,b,c,d,nm,na).
   *
   * Conditions:
   * - The Fma itself is either single-use, or multi-use with ALL parents
   *   being Add/Sub (so the bottom-up walk will rewrite every consumer
   *   independently, leaving the original Fma dead).
   * - The Mul addend is single-use (used only inside this Fma).
   * - Neither node is frozen. *)
  let match_fma_mul (n : t) =
    if is_frozen n.tag then None
    else
      match n.node with
      | NK_Fma (a, b, addend, nm, na)
        when (single_use n || multi_use_safe n)
             && single_use addend
             && not (is_frozen addend.tag) -> (
          match addend.node with
          | NK_Mul (c, d) -> Some (n.tag, a, b, c, d, nm, na)
          | _ -> None)
      | _ -> None
  in

  let n_rewrites = ref 0 in
  (* Counters for diagnostics *)
  let n_addsub_seen = ref 0 in
  let n_fma_mul_candidates = ref 0 in
  let n_blocked_fma_multiuse = ref 0 in
  let n_blocked_mul_multiuse = ref 0 in
  let n_blocked_fma_frozen = ref 0 in
  let n_blocked_mul_frozen = ref 0 in
  let inspect_candidate (n : t) =
    match n.node with
    | NK_Fma (_, _, addend, _, _) ->
        let fma_su = single_use n in
        let mul_su = single_use addend in
        let fma_frz = is_frozen n.tag in
        let mul_frz = is_frozen addend.tag in
        let is_mul_addend =
          match addend.node with NK_Mul _ -> true | _ -> false
        in
        if is_mul_addend then begin
          incr n_fma_mul_candidates;
          if fma_frz then incr n_blocked_fma_frozen
          else if mul_frz then incr n_blocked_mul_frozen
          else if not fma_su then begin
            incr n_blocked_fma_multiuse;
            if Sys.getenv_opt "FLATTEN_FMA_MUL_TRACE" <> None then begin
              let uc = try Hashtbl.find use_count n.tag with Not_found -> 0 in
              Printf.eprintf
                "    blocked-fma-multiuse: Fma tag=t%d use_count=%d\n" n.tag uc
            end
          end
          else if not mul_su then incr n_blocked_mul_multiuse
        end
    | _ -> ()
  in
  let do_flatten ~fma_tag ~p ~fa ~fb ~mc ~md ~inner_nm ~inner_na ~outer_nm =
    incr n_rewrites;
    let _ = fma_tag in
    let inner = hashcons (NK_Fma (fa, fb, p, inner_nm, inner_na)) in
    hashcons (NK_Fma (mc, md, inner, outer_nm, false))
  in

  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rewrite (n : t) : t =
    match Hashtbl.find_opt cache n.tag with
    | Some r -> r
    | None ->
        let r =
          match n.node with
          | NK_Const _ | NK_Load _ -> n
          | NK_Neg a ->
              let a' = rewrite a in
              if a' == a then n else mk_neg a'
          | NK_Mul (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_mul a' b'
          | NK_CmulRe (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulRe (a', b', c', d'))
          | NK_CmulIm (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulIm (a', b', c', d'))
          | NK_Fma (a, b, c, nm, na) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              if a' == a && b' == b && c' == c then n
              else hashcons (NK_Fma (a', b', c', nm, na))
          | NK_Add (a, b) when not (is_frozen n.tag) -> (
              incr n_addsub_seen;
              inspect_candidate a;
              inspect_candidate b;
              (* Check both orderings — Add is commutative *)
              match (match_fma_mul a, match_fma_mul b) with
              | Some _, Some _ ->
                  (* Both sides are Fma-with-Mul-addend. Picking one and skipping
                   * the other leaves the "skipped" Fma referenced by the new
                   * chain (as the addend), so it survives without its Mul being
                   * absorbed. When this pattern recurs as Add(F1,F2) + Sub(F1,F2)
                   * (the dominant case for conjugate-pair outputs), each consumer
                   * picks a different sibling — net result is BOTH chains duplicate
                   * work without killing either Mul: 2 Muls + 2 Fmas + Add + Sub
                   * (6 ops) becomes 2 Muls + 2 Fmas + 4 chain Fmas (8 ops, +2).
                   * Conservative bailout. *)
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_add_binary a' b'
              | Some (ft, fa, fb, mc, md, nm, na), None ->
                  let p = rewrite b in
                  do_flatten ~fma_tag:ft ~p ~fa:(rewrite fa) ~fb:(rewrite fb)
                    ~mc:(rewrite mc) ~md:(rewrite md) ~inner_nm:nm
                    ~inner_na:false ~outer_nm:na
              | None, Some (ft, fa, fb, mc, md, nm, na) ->
                  let p = rewrite a in
                  do_flatten ~fma_tag:ft ~p ~fa:(rewrite fa) ~fb:(rewrite fb)
                    ~mc:(rewrite mc) ~md:(rewrite md) ~inner_nm:nm
                    ~inner_na:false ~outer_nm:na
              | None, None ->
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_add_binary a' b')
          | NK_Add (a, b) ->
              (* Frozen — pass through with recursive rewrite of children only *)
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_add_binary a' b'
          | NK_Sub (a, b) when not (is_frozen n.tag) -> (
              incr n_addsub_seen;
              inspect_candidate a;
              inspect_candidate b;
              match (match_fma_mul a, match_fma_mul b) with
              | Some _, Some _ ->
                  (* Same conservative bailout as Add: both sides matching means
                   * each chain references the other's Fma, leaving both Muls
                   * alive while adding 4 chain Fmas. See comment on the Add
                   * branch above. *)
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_sub_binary a' b'
              | None, Some (ft, fa, fb, mc, md, nm, na) ->
                  (* Sub(p, F) — flip both signs *)
                  let p = rewrite a in
                  do_flatten ~fma_tag:ft ~p ~fa:(rewrite fa) ~fb:(rewrite fb)
                    ~mc:(rewrite mc) ~md:(rewrite md) ~inner_nm:(not nm)
                    ~inner_na:false ~outer_nm:(not na)
              | Some (ft, fa, fb, mc, md, nm, na), None ->
                  (* Sub(F, p) — inner gets neg_add=true to flip p *)
                  let p = rewrite b in
                  do_flatten ~fma_tag:ft ~p ~fa:(rewrite fa) ~fb:(rewrite fb)
                    ~mc:(rewrite mc) ~md:(rewrite md) ~inner_nm:nm
                    ~inner_na:true ~outer_nm:na
              | None, None ->
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_sub_binary a' b')
          | NK_Sub (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_sub_binary a' b'
          | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2634"
        in
        Hashtbl.add cache n.tag r;
        if r != n && is_frozen n.tag then Hashtbl.replace tag_remap n.tag r.tag;
        r
  in
  let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in

  if Sys.getenv_opt "FLATTEN_FMA_MUL_TRACE" <> None then begin
    Printf.eprintf "[flatten_fma_mul_addend] %d rewrites\n" !n_rewrites;
    Printf.eprintf "  %d Add/Sub seen, %d Fma(_,_,Mul,_,_) candidates found\n"
      !n_addsub_seen !n_fma_mul_candidates;
    Printf.eprintf
      "  blocked: fma_multiuse=%d mul_multiuse=%d fma_frozen=%d mul_frozen=%d\n"
      !n_blocked_fma_multiuse !n_blocked_mul_multiuse !n_blocked_fma_frozen
      !n_blocked_mul_frozen;
    (* Parent-type distribution across all multi-use Fma-with-Mul-addend nodes.
     * If all parents are Add/Sub, relaxing single_use is safe — each parent
     * will independently rewrite into a 2-FMA chain and the original Fma
     * becomes dead. *)
    (* Build sig counts directly via list accumulation to avoid any
     * Hashtbl iteration timing oddness with stderr buffering. *)
    let all_signatures = ref [] in
    let all_kinds = ref [] in
    Hashtbl.iter
      (fun fma_tag parent_kinds ->
        if Sys.getenv_opt "FLATTEN_FMA_MUL_TRACE_VERBOSE" <> None then
          Printf.eprintf "    fma_tag=t%d parent_kinds=[%s]\n" fma_tag
            (String.concat "," parent_kinds);
        let sig_ = String.concat "+" (List.sort String.compare parent_kinds) in
        all_signatures := sig_ :: !all_signatures;
        List.iter (fun k -> all_kinds := k :: !all_kinds) parent_kinds)
      fma_parents;
    let sig_list_sorted = List.sort String.compare !all_signatures in
    let kind_list_sorted = List.sort String.compare !all_kinds in
    let count_consecutive lst =
      let rec go acc = function
        | [] -> List.rev acc
        | x :: _ as l ->
            let same, rest = List.partition (fun y -> y = x) l in
            go ((x, List.length same) :: acc) rest
      in
      go [] lst
    in
    Printf.eprintf "  parent kinds (all Fma-w-Mul-addend, across instances):\n";
    List.iter
      (fun (k, c) -> Printf.eprintf "    %s: %d\n" k c)
      (count_consecutive kind_list_sorted);
    Printf.eprintf "  parent signatures (per Fma instance):\n";
    List.iter
      (fun (s, c) -> Printf.eprintf "    [%s]: %d Fmas\n" s c)
      (count_consecutive sig_list_sorted);
    (* Also scan the entire DAG for any Fma-with-Mul-addend nodes
     * to understand if they exist at all by this point. *)
    let global_count = ref 0 in
    let global_visited = Hashtbl.create 256 in
    let rec gscan (n : t) =
      if Hashtbl.mem global_visited n.tag then ()
      else begin
        Hashtbl.add global_visited n.tag ();
        (match n.node with
        | NK_Fma (_, _, c, _, _) -> (
            match c.node with NK_Mul _ -> incr global_count | _ -> ())
        | _ -> ());
        match n.node with
        | NK_Const _ | NK_Load _ -> ()
        | NK_Neg a -> gscan a
        | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
            gscan a;
            gscan b
        | NK_Fma (a, b, c, _, _) ->
            gscan a;
            gscan b;
            gscan c
        | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
            gscan a;
            gscan b;
            gscan c;
            gscan d
        | NK_Plus _ -> ()
      end
    in
    List.iter (fun (_, e) -> gscan e) assigns;
    Printf.eprintf "  GLOBAL: %d Fma-with-Mul-addend nodes in entire DAG\n"
      !global_count
  end;

  (new_assigns, tag_remap)

(* === BUTTERFLY-SHARE-MUL PASS ===
 *
 * Recognizes "swap-pair" butterfly patterns where two FMAs compute
 * sum/diff of the same two K-products but with the products in OPPOSITE
 * roles (mul-slot vs addend):
 *
 *   F  = Fma(a, b, Mul(p, q), nm,  na)   = ±a·b ± p·q
 *   F' = Fma(p, q, Mul(a, b), nm', na')  = ±p·q ± a·b
 *
 * The two products (a·b) and (p·q) are each currently inlined into one
 * FMA's addend slot — emit_c inlines because each Mul has use_count = 1
 * (only one consumer apiece). The result is 2 inlined Muls + 2 FMAs,
 * even though there are only 2 distinct K-products.
 *
 * Rewrite: change F's mul-slot from (a, b) to (p, q) and its addend
 * from Mul(p, q) to Mul(a, b), swapping sign-flags (nm, na) → (na, nm).
 * Value preserved (commutativity of + and the sign-flag swap):
 *   ±a·b ± p·q  =  ±p·q ± a·b
 *
 * After the rewrite, both F and F' use Mul(a, b) as addend. Its
 * use_count goes from 1 to 2, so emit_c declares it as a variable
 * instead of inlining. Mul(p, q) becomes orphaned (use_count = 0).
 *
 *   Before: 2 FMA + 2 inlined Mul        = 4 ops
 *   After:  2 FMA + 1 declared Mul       = 3 ops
 *   Savings: 1 op per swap-pair.
 *
 * At R=32 there are 2 such pairs (4 of the 10 residual Muls), saving
 * 2 ops. Larger radices have more.
 *
 * SAFETY:
 * - Only rewrites when the OTHER product Mul(a, b) already exists in
 *   the DAG (as F'.addend). This guarantees the rewrite doesn't
 *   introduce new Mul nodes.
 * - Does not touch frozen Fmas — sign-flag swap changes the Fma's
 *   tag but spill markers expect the OLD value. (Tag-remap is tracked
 *   for frozen Fmas so subsequent passes can still find them.)
 *)
let butterfly_share_mul ?(frozen_tags : (int, unit) Hashtbl.t option = None)
    (assigns : (Expr.elem_ref * t) list) :
    (Expr.elem_ref * t) list * (int, int) Hashtbl.t =
  let _ = frozen_tags in
  let tag_remap : (int, int) Hashtbl.t = Hashtbl.create 16 in

  (* Step 1: walk DAG, collect (Fma node, addend Mul node) pairs.
   * Also count uses of each Mul so we know its current multiplicity. *)
  let fma_with_mul_addend : (t * t) list ref = ref [] in
  let mul_uses : (int, int) Hashtbl.t = Hashtbl.create 64 in
  let bump_use t =
    let c = try Hashtbl.find mul_uses t with Not_found -> 0 in
    Hashtbl.replace mul_uses t (c + 1)
  in
  let visited = Hashtbl.create 256 in
  let rec scan (n : t) =
    if not (Hashtbl.mem visited n.tag) then begin
      Hashtbl.add visited n.tag ();
      match n.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg a -> scan a
      | NK_Add (a, b) | NK_Sub (a, b) ->
          (match a.node with NK_Mul _ -> bump_use a.tag | _ -> ());
          (match b.node with NK_Mul _ -> bump_use b.tag | _ -> ());
          scan a;
          scan b
      | NK_Mul (a, b) ->
          scan a;
          scan b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          scan a;
          scan b;
          scan c;
          scan d
      | NK_Fma (a, b, c, _, _) -> (
          scan a;
          scan b;
          scan c;
          match c.node with
          | NK_Mul _ ->
              bump_use c.tag;
              fma_with_mul_addend := (n, c) :: !fma_with_mul_addend
          | _ -> ())
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2789"
    end
  in
  List.iter (fun (_, e) -> scan e) assigns;

  (* Step 2: index FMAs by their addend Mul's tag. *)
  let by_addend : (int, t list) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun (f, m) ->
      let cur = try Hashtbl.find by_addend m.tag with Not_found -> [] in
      Hashtbl.replace by_addend m.tag (f :: cur))
    !fma_with_mul_addend;

  (* Step 3: scan FMAs for swap-pair partners. For each F = Fma(a, b, Mul(p, q), _, _):
   * - Compute the canonical Mul(a, b) via mk_mul (hashcons returns the
   *   existing node if any).
   * - Look up FMAs with that addend in by_addend.
   * - Each candidate F' must have mul-slot operands matching {p, q}
   *   (as a multiset).
   * - If found, mark F as "should be rewritten to share Mul(a, b)".
   * To avoid re-processing the same pair twice (F and F' would each
   * find each other), only rewrite when F's tag < F'.tag — picks a
   * canonical member of the pair. *)
  let rewrite_to_share : (int, t) Hashtbl.t = Hashtbl.create 16 in
  List.iter
    (fun (f, m_addend) ->
      match (f.node, m_addend.node) with
      | NK_Fma (a, b, _, _, _), NK_Mul (p, q) -> (
          (* Canonical Mul(a, b) — hashcons returns existing if present. *)
          let m_other = mk_mul a b in
          (* Did mk_mul return a Mul node? It could fold (e.g., 0, 1, Neg)
           * but we only proceed if it stayed a Mul. *)
          match m_other.node with
          | NK_Mul _ ->
              let cands =
                try Hashtbl.find by_addend m_other.tag with Not_found -> []
              in
              List.iter
                (fun f' ->
                  if f'.tag <> f.tag then
                    match f'.node with
                    | NK_Fma (a', b', _, _, _) ->
                        (* Check mul-slot of f' matches (p, q) as multiset. *)
                        let f_slot_matches_pq =
                          (a'.tag = p.tag && b'.tag = q.tag)
                          || (a'.tag = q.tag && b'.tag = p.tag)
                        in
                        if
                          f_slot_matches_pq && f.tag < f'.tag
                          && (not (Hashtbl.mem rewrite_to_share f.tag))
                          && not (Hashtbl.mem rewrite_to_share f'.tag)
                        then begin
                          Hashtbl.add rewrite_to_share f.tag m_other;
                          if Sys.getenv_opt "BSM_TRACE" <> None then
                            Printf.eprintf
                              "[bsm] swap pair: F=t%d (mul=t%d,t%d \
                               add=Mul(t%d,t%d)) ↔ F'=t%d → share \
                               Mul(t%d,t%d)=t%d\n"
                              f.tag a.tag b.tag p.tag q.tag f'.tag a.tag b.tag
                              m_other.tag
                        end
                    | _ -> ())
                cands
          | _ -> ())
      | _ -> ())
    !fma_with_mul_addend;

  (* Step 4: rewrite. The walker visits the DAG; for each Fma node
   * marked in rewrite_to_share, swap mul-slot and addend, swap (nm, na).
   * Value-preserving. *)
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let is_frozen tag =
    match frozen_tags with None -> false | Some tbl -> Hashtbl.mem tbl tag
  in
  let rec rewrite (n : t) : t =
    match Hashtbl.find_opt cache n.tag with
    | Some r -> r
    | None ->
        let r =
          match n.node with
          | NK_Const _ | NK_Load _ -> n
          | NK_Neg a ->
              let a' = rewrite a in
              if a' == a then n else mk_neg a'
          | NK_Mul (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_mul a' b'
          | NK_Add (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_add_binary a' b'
          | NK_Sub (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_sub_binary a' b'
          | NK_CmulRe (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulRe (a', b', c', d'))
          | NK_CmulIm (a, b, c, d) ->
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulIm (a', b', c', d'))
          | NK_Fma (a, b, c, nm, na) -> (
              let a' = rewrite a in
              let b' = rewrite b in
              let c' = rewrite c in
              (* Check if this Fma is marked for swap-rewrite. *)
              match (Hashtbl.find_opt rewrite_to_share n.tag, c'.node) with
              | Some m_shared, NK_Mul (p_orig, q_orig) ->
                  (* New mul-slot: (p_orig, q_orig) (rewritten children of c').
                   * New addend: m_shared.
                   * New flags: (na, nm). *)
                  let _ = p_orig in
                  let _ = q_orig in
                  hashcons (NK_Fma (p_orig, q_orig, m_shared, na, nm))
              | _ ->
                  if a' == a && b' == b && c' == c then n
                  else hashcons (NK_Fma (a', b', c', nm, na)))
          | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2899"
        in
        Hashtbl.add cache n.tag r;
        if r != n && is_frozen n.tag then Hashtbl.replace tag_remap n.tag r.tag;
        r
  in
  let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in

  if Sys.getenv_opt "BSM_TRACE" <> None then
    Printf.eprintf "[bsm] rewrites applied: %d\n"
      (Hashtbl.length rewrite_to_share);

  (new_assigns, tag_remap)

(* === DAG STATISTICS === *)

type dag_stats = {
  total_nodes : int;
  consts : int;
  loads : int;
  negs : int;
  adds : int;
  subs : int;
  muls : int;
  cmuls : int; (* number of distinct Cmul nodes (Re or Im) *)
  fmas : int; (* number of NK_Fma nodes (each = 1 instruction) *)
  arithmetic_ops : int; (* counts each Cmul-node as 2 muls + 1 add/sub *)
}

(* Stats restricted to nodes reachable from the given roots — this is
 * the meaningful count, since dead nodes from intermediate construction
 * pollute a raw `Hashtbl.length`. *)
let stats_reachable (roots : t list) : dag_stats =
  let seen = Hashtbl.create 256 in
  let consts = ref 0 in
  let loads = ref 0 in
  let negs = ref 0 in
  let adds = ref 0 in
  let subs = ref 0 in
  let muls = ref 0 in
  let cmuls = ref 0 in
  let fmas = ref 0 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag ();
      (match e.node with
      | NK_Const _ -> incr consts
      | NK_Load _ -> incr loads
      | NK_Neg inner -> (
          (* Neg(Const) is a compile-time constant — emits as a single
           * broadcast load with the negated literal, not a runtime negation.
           * Don't count it as an op. *)
          match inner.node with
          | NK_Const _ -> () (* compile-time constant, no runtime op *)
          | _ -> incr negs)
      | NK_Add _ -> incr adds
      | NK_Sub _ -> incr subs
      | NK_Mul _ -> incr muls
      | NK_CmulRe _ | NK_CmulIm _ -> incr cmuls
      | NK_Fma _ -> incr fmas
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2960 (counter)");
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg e1 -> visit e1
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          visit a;
          visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          visit a;
          visit b;
          visit c;
          visit d
      | NK_Fma (a, b, c, _, _) ->
          visit a;
          visit b;
          visit c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:2975 (visit)"
    end
  in
  List.iter visit roots;
  (* Each Cmul-node represents (a*c ± b*d), which is 2 muls + 1 add/sub.
   * We count both Re and Im outputs separately as Cmul nodes; their
   * arithmetic-cost is 3 ops each, so total is 3 * cmuls. *)
  {
    total_nodes = Hashtbl.length seen;
    consts = !consts;
    loads = !loads;
    negs = !negs;
    adds = !adds;
    subs = !subs;
    muls = !muls;
    cmuls = !cmuls;
    fmas = !fmas;
    (* Each Cmul node (Re or Im) represents (a*c ± b*d) = 2 muls + 1 add/sub
     * = 3 arithmetic ops. Each Fma is 1 mul + 1 add fused = 2 arithmetic ops
     * but 1 instruction. So contribution to arith ops: 3*cmuls + 2*fmas. *)
    arithmetic_ops = !adds + !subs + !muls + !negs + (3 * !cmuls) + (2 * !fmas);
  }

let string_of_stats (s : dag_stats) : string =
  (* Report at three levels of granularity:
   *
   * 1. DAG node breakdown — what's in the tree, ISA-agnostic.
   * 2. Vector instructions — what we actually emit. With FMA-fused Cmul,
   *    each Cmul-Re/Im becomes 1 mul + 1 fma = 2 instructions (NOT 3).
   *    This is the count that matches disassembly on AVX-512 / AVX-2.
   * 3. Scalar-equivalent ops — useful for FLOP counts. Each Cmul represents
   *    3 scalar ops (2 muls + 1 add/sub) per output. Multiply by SIMD lane
   *    width (8 for AVX-512, 4 for AVX-2) for actual scalar work per
   *    inner-loop iteration. *)
  let vec_arith = s.adds + s.subs + s.muls + s.negs + (2 * s.cmuls) + s.fmas in
  let scalar_ops =
    s.adds + s.subs + s.muls + s.negs + (3 * s.cmuls) + (2 * s.fmas)
  in
  let buf = Buffer.create 512 in
  Buffer.add_string buf (Printf.sprintf "DAG nodes: %d total\n" s.total_nodes);
  Buffer.add_string buf (Printf.sprintf "  Loads:  %d\n" s.loads);
  Buffer.add_string buf (Printf.sprintf "  Consts: %d\n" s.consts);
  Buffer.add_string buf (Printf.sprintf "  Negs:   %d\n" s.negs);
  Buffer.add_string buf (Printf.sprintf "  Adds:   %d\n" s.adds);
  Buffer.add_string buf (Printf.sprintf "  Subs:   %d\n" s.subs);
  Buffer.add_string buf (Printf.sprintf "  Muls:   %d\n" s.muls);
  Buffer.add_string buf
    (Printf.sprintf
       "  Cmuls:  %d   (each = 1 mul + 1 fmadd/fnmadd = 2 instructions)\n"
       s.cmuls);
  Buffer.add_string buf
    (Printf.sprintf
       "  Fmas:   %d   (each = 1 fmadd/fmsub/fnmadd/fnmsub = 1 instruction)\n"
       s.fmas);
  Buffer.add_string buf "\n";
  Buffer.add_string buf
    (Printf.sprintf "Vector instructions (FMA-fused, ISA-independent): %d\n"
       vec_arith);
  Buffer.add_string buf
    (Printf.sprintf
       "  Breakdown: %d add/sub/mul/neg + %d cmul-pair instructions + %d fma\n"
       (s.adds + s.subs + s.muls + s.negs)
       (2 * s.cmuls) s.fmas);
  Buffer.add_string buf "\n";
  Buffer.add_string buf
    (Printf.sprintf
       "Scalar-equivalent ops (each Cmul = 3 ops, each Fma = 2 ops): %d\n"
       scalar_ops);
  Buffer.add_string buf
    (Printf.sprintf "  AVX-512 work (×8 lanes): %d ops/iter\n" (scalar_ops * 8));
  Buffer.add_string buf
    (Printf.sprintf "  AVX-2   work (×4 lanes): %d ops/iter\n" (scalar_ops * 4));
  Buffer.contents buf

(* === DAG PRETTY-PRINTING ===
 * Prints each unique node once, with tag, then the assignment list. *)

let string_of_node_kind (nk : node_kind) : string =
  match nk with
  | NK_Const c ->
      if c < 0.0 then Printf.sprintf "(%g)" c else Printf.sprintf "%g" c
  | NK_Load r -> Expr.string_of_elem_ref r
  | NK_Neg e -> Printf.sprintf "-t%d" e.tag
  | NK_Add (a, b) -> Printf.sprintf "t%d + t%d" a.tag b.tag
  | NK_Sub (a, b) -> Printf.sprintf "t%d - t%d" a.tag b.tag
  | NK_Mul (a, b) -> Printf.sprintf "t%d * t%d" a.tag b.tag
  | NK_CmulRe (a, b, c, d) ->
      Printf.sprintf "cmul.re(t%d, t%d, t%d, t%d)" a.tag b.tag c.tag d.tag
  | NK_CmulIm (a, b, c, d) ->
      Printf.sprintf "cmul.im(t%d, t%d, t%d, t%d)" a.tag b.tag c.tag d.tag
  | NK_Fma (a, b, c, neg_mul, neg_add) ->
      let sign_mul = if neg_mul then "-" else "+" in
      let sign_add = if neg_add then "-" else "+" in
      Printf.sprintf "fma(%st%d*t%d, %st%d)" sign_mul a.tag b.tag sign_add c.tag
  | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:3019"

let print_dag (assigns : (Expr.elem_ref * t) list) : string =
  let roots = List.map snd assigns in
  let seen = Hashtbl.create 256 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag e;
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg e1 -> visit e1
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
          visit a;
          visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
          visit a;
          visit b;
          visit c;
          visit d
      | NK_Fma (a, b, c, _, _) ->
          visit a;
          visit b;
          visit c
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:3043"
    end
  in
  List.iter visit roots;
  let nodes = Hashtbl.fold (fun _ e acc -> e :: acc) seen [] in
  let nodes = List.sort (fun a b -> compare a.tag b.tag) nodes in
  let buf = Buffer.create 4096 in
  List.iter
    (fun e ->
      Buffer.add_string buf
        (Printf.sprintf "  t%-3d = %s\n" e.tag (string_of_node_kind e.node)))
    nodes;
  Buffer.add_string buf "\n";
  List.iter
    (fun (lhs, e) ->
      Buffer.add_string buf
        (Printf.sprintf "  %-12s = t%d\n" (Expr.string_of_elem_ref lhs) e.tag))
    assigns;
  Buffer.contents buf
