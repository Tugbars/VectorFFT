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
  | NK_Load  of elem_ref
  | NK_Neg   of t
  | NK_Add   of t * t
  | NK_Sub   of t * t
  | NK_Mul   of t * t
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

and t = {
  tag : int;
  node : node_kind;
}

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

let reset () =
  Hashtbl.clear hcons_table;
  next_tag := 0

(* === CANONICALIZATION HELPERS === *)

let zero_threshold = 1e-14

let is_zero (e : t) : bool =
  match e.node with
  | NK_Const c -> Float.abs c < zero_threshold
  | _ -> false

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
    if c = 0.0 then 0.0
    else float_of_string (Printf.sprintf "%.13e" c)
  in
  if Float.abs rounded < zero_threshold then hashcons (NK_Const 0.0)
  else if Float.abs (rounded -. 1.0) < zero_threshold then hashcons (NK_Const 1.0)
  else if Float.abs (rounded +. 1.0) < zero_threshold then hashcons (NK_Const (-1.0))
  else if rounded < 0.0 then
    (* Canonicalize negative non-trivial constants to -|c|.
     * This unifies all multiplications-by-c with multiplications-by-(-c):
     *   Mul(x, -c) → Mul(x, Neg(c)) → Neg(Mul(x, c)) via Neg-hoisting.
     * The underlying Mul(x, c) is then shared by hash-consing.
     * Hand-coded codelets do this manually (e.g. vnc = -vc); we get
     * the same effect mechanically. *)
    hashcons (NK_Neg (hashcons (NK_Const (-. rounded))))
  else
    hashcons (NK_Const rounded)

let mk_load (r : elem_ref) : t =
  hashcons (NK_Load r)

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
  | NK_Neg inner  -> flatten_sum (-sign) inner
  | _ -> [(sign, e)]

(* Cancel pairs of (+1, x) and (-1, x) — they sum to 0 and are dropped.
 * Sort the result canonically by tag.
 *
 * Implementation: tally signed coefficients per tag in a hashtable, then
 * emit (coefficient, t) for nonzero coefficients in tag order. *)
let cancel_signs (terms : (int * t) list) : (int * t) list =
  let coeff = Hashtbl.create 16 in
  let tag_to_t = Hashtbl.create 16 in
  List.iter (fun (s, e) ->
    Hashtbl.replace tag_to_t e.tag e;
    let prev = try Hashtbl.find coeff e.tag with Not_found -> 0 in
    Hashtbl.replace coeff e.tag (prev + s)
  ) terms;
  let result = Hashtbl.fold (fun tag c acc ->
    if c = 0 then acc
    else (c, Hashtbl.find tag_to_t tag) :: acc
  ) coeff [] in
  List.sort (fun (_, a) (_, b) -> compare a.tag b.tag) result

(* Split a list into (evens, odds) by index — used by interleaved
 * pair-folding to expose butterfly subsums. *)
let split_interleaved (lst : 'a list) : 'a list * 'a list =
  let evens = ref [] in
  let odds  = ref [] in
  List.iteri (fun i x ->
    if i mod 2 = 0 then evens := x :: !evens
    else odds := x :: !odds
  ) lst;
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
  | NK_Const c -> mk_const (-. c)
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
    match a.node, b.node with
    | NK_Const x, NK_Const y -> mk_const (x *. y)
    | NK_Neg a', _ -> mk_neg (mk_mul a' b)
    | _, NK_Neg b' -> mk_neg (mk_mul a b')
    | _ ->
      let (a, b) = if a.tag <= b.tag then (a, b) else (b, a) in
      hashcons (NK_Mul (a, b))

(* Leaf binary Add — used post-reassoc by emit_pair_fold. Hash-conses,
 * applies trivial identities, and recognizes Add(x, Neg(y)) → Sub(x, y)
 * to avoid redundant Neg+Add pairs after the pair-fold rebuilds. *)
and mk_add_binary (a : t) (b : t) : t =
  if is_zero a then b
  else if is_zero b then a
  else
    match a.node, b.node with
    | NK_Const x, NK_Const y -> mk_const (x +. y)
    | _, NK_Neg b' -> mk_sub_binary a b'      (* x + (-y) = x - y *)
    | NK_Neg a', _ -> mk_sub_binary b a'      (* (-x) + y = y - x *)
    | _ ->
      let (a, b) = if a.tag <= b.tag then (a, b) else (b, a) in
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
    | _ ->
      hashcons (NK_Sub (a, b))

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
    let _ = c1 in let _ = c2 in (xr, xi)
  | NK_Const _, NK_Const _ when is_zero wr && is_one wi ->
    (mk_neg xi, xr)
  | NK_Const _, NK_Const _ when is_zero wr && is_neg_one wi ->
    (xi, mk_neg xr)
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
  match s1, s2 with
  | 1,  1  -> mk_add_binary e1 e2
  | 1,  -1 -> mk_sub_binary e1 e2
  | -1, 1  -> mk_sub_binary e2 e1
  | -1, -1 -> mk_neg (mk_add_binary e1 e2)
  | _ ->
    (* Coefficients other than ±1: emit Mul(const, leaf). Rare in FFT. *)
    let lhs = if s1 = 0 then mk_const 0.0
              else mk_mul (mk_const (float_of_int s1)) e1 in
    let rhs = if s2 = 0 then mk_const 0.0
              else mk_mul (mk_const (float_of_int s2)) e2 in
    mk_add_binary lhs rhs

(* Pair-fold a sorted list of signed terms into a binary tree by
 * recursive interleaved splitting. This exposes butterfly subsums
 * because the half-split structure matches even/odd index pairing. *)
and emit_pair_fold (terms : (int * t) list) : t =
  match terms with
  | [] -> mk_const 0.0
  | [t] -> emit_signed_term t
  | [t1; t2] -> combine_two t1 t2
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
  let add_op = if reassoc then mk_add else mk_add_binary in
  let sub_op = if reassoc then mk_sub else mk_sub_binary in
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
    let is_const e = match e.node with
      | NK_Const _ -> true
      | NK_Neg n -> (match n.node with NK_Const _ -> true | _ -> false)
      | _ -> false in
    if is_const xr || is_const xi || is_const wr || is_const wi then
      sub_op (mk_mul xr wr) (mk_mul xi wi)
    else
      let (re, _im) = mk_cmul xr xi wr wi in
      re

  (* CMUL.IM PATTERN — needs reassoc flag threaded too. *)
  | Expr.Add (Expr.Mul (xr_e, wi_e), Expr.Mul (xi_e, wr_e)) ->
    let xr = of_expr ~reassoc xr_e in
    let wi = of_expr ~reassoc wi_e in
    let xi = of_expr ~reassoc xi_e in
    let wr = of_expr ~reassoc wr_e in
    let is_const e = match e.node with
      | NK_Const _ -> true
      | NK_Neg n -> (match n.node with NK_Const _ -> true | _ -> false)
      | _ -> false in
    if is_const xr || is_const xi || is_const wr || is_const wi then
      add_op (mk_mul xr wi) (mk_mul xi wr)
    else
      let (_re, im) = mk_cmul xr xi wr wi in
      im

  | Expr.Add (a, b) -> add_op (of_expr ~reassoc a) (of_expr ~reassoc b)
  | Expr.Sub (a, b) -> sub_op (of_expr ~reassoc a) (of_expr ~reassoc b)
  | Expr.Mul (a, b) -> mk_mul (of_expr ~reassoc a) (of_expr ~reassoc b)

let of_assignments ?(reassoc = true) (al : Expr.assignment list)
    : (Expr.elem_ref * t) list =
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

type spill_tag_marker = {
  slot: int;
  re_tag: int;
  im_tag: int;
}

let lift_spill_markers ?(reassoc = true) (markers : Dft.spill_marker list)
    : spill_tag_marker list =
  List.map (fun m ->
    let re = of_expr ~reassoc m.Dft.re_expr in
    let im = of_expr ~reassoc m.Dft.im_expr in
    { slot = m.slot; re_tag = re.tag; im_tag = im.tag }
  ) markers

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

let dedup_sub_pairs (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  (* Step 1: walk DAG, build indexes. *)
  let visited = Hashtbl.create 256 in
  let usage_count = Hashtbl.create 256 in   (* tag -> count *)
  let sub_index = Hashtbl.create 64 in       (* (small_tag, big_tag) -> [Sub nodes both directions] *)
  let bump_usage tag =
    let c = try Hashtbl.find usage_count tag with Not_found -> 0 in
    Hashtbl.replace usage_count tag (c + 1)
  in
  let rec visit (e : t) =
    if not (Hashtbl.mem visited e.tag) then begin
      Hashtbl.add visited e.tag ();
      (match e.node with
       | NK_Const _ | NK_Load _ -> ()
       | NK_Neg inner ->
         bump_usage inner.tag;
         visit inner
       | NK_Add (a, b) | NK_Mul (a, b) ->
         bump_usage a.tag; bump_usage b.tag;
         visit a; visit b
       | NK_Sub (a, b) ->
         bump_usage a.tag; bump_usage b.tag;
         (* Index the Sub by (small_tag, big_tag) regardless of direction.
          * The list will hold both Sub(a,b) and Sub(b,a) if both exist. *)
         let key = if a.tag < b.tag then (a.tag, b.tag) else (b.tag, a.tag) in
         let prev = try Hashtbl.find sub_index key with Not_found -> [] in
         Hashtbl.replace sub_index key (e :: prev);
         visit a; visit b
       | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
         (* Cmul outputs are opaque to dedup. Visit the four operands so
          * usage counts include them; don't index Cmul itself for Sub-pair
          * matching. *)
         bump_usage a.tag; bump_usage b.tag;
         bump_usage c.tag; bump_usage d.tag;
         visit a; visit b; visit c; visit d)
    end
  in
  List.iter (fun (_, e) -> visit e) assigns;

  (* Step 2: find Sub-pair conflicts and pick winners. *)
  let substitute : (int, t) Hashtbl.t = Hashtbl.create 16 in
  Hashtbl.iter (fun _key nodes ->
    match nodes with
    | [_] -> ()  (* only one direction in the DAG, no conflict *)
    | nodes_list ->
      (* Multiple Sub nodes share the same (small,big) key. Should be
       * exactly two: Sub(a,b) and Sub(b,a). Pick the winner by usage. *)
      let scored = List.map (fun n ->
        let c = try Hashtbl.find usage_count n.tag with Not_found -> 0 in
        (c, n)
      ) nodes_list in
      let scored = List.sort (fun (c1, n1) (c2, n2) ->
        (* Higher usage wins; tie-break by lower tag (deterministic). *)
        if c1 <> c2 then compare c2 c1
        else compare n1.tag n2.tag
      ) scored in
      (match scored with
       | (_, winner) :: losers ->
         List.iter (fun (_, loser) ->
           if loser.tag <> winner.tag then
             Hashtbl.add substitute loser.tag (mk_neg winner)
         ) losers
       | [] -> ())
  ) sub_index;

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
        | None ->
          (* Recursively rebuild children. The smart constructors handle
           * any new peepholes that fire (e.g. Add of Neg → Sub). *)
          (match e.node with
           | NK_Const _ | NK_Load _ -> e
           | NK_Neg inner -> mk_neg (rebuild inner)
           | NK_Add (a, b) -> mk_add_binary (rebuild a) (rebuild b)
           | NK_Sub (a, b) -> mk_sub_binary (rebuild a) (rebuild b)
           | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
           | NK_CmulRe (a, b, c, d) ->
             let (re, _im) = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
             re
           | NK_CmulIm (a, b, c, d) ->
             let (_re, im) = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
             im)
      in
      Hashtbl.add rebuild_cache e.tag result;
      result
  in
  List.map (fun (lhs, e) -> (lhs, rebuild e)) assigns

(* === DAG STATISTICS === *)


type dag_stats = {
  total_nodes : int;
  consts : int;
  loads : int;
  negs : int;
  adds : int;
  subs : int;
  muls : int;
  cmuls : int;            (* number of distinct Cmul nodes (Re or Im) *)
  arithmetic_ops : int;   (* counts each Cmul-node as 2 muls + 1 add/sub *)
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
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag ();
      (match e.node with
       | NK_Const _ -> incr consts
       | NK_Load _ -> incr loads
       | NK_Neg inner ->
         (* Neg(Const) is a compile-time constant — emits as a single
          * broadcast load with the negated literal, not a runtime negation.
          * Don't count it as an op. *)
         (match inner.node with
          | NK_Const _ -> ()  (* compile-time constant, no runtime op *)
          | _ -> incr negs)
       | NK_Add _ -> incr adds
       | NK_Sub _ -> incr subs
       | NK_Mul _ -> incr muls
       | NK_CmulRe _ | NK_CmulIm _ -> incr cmuls);
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg e1 -> visit e1
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
        visit a; visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        visit a; visit b; visit c; visit d
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
    (* Each Cmul node (Re or Im) represents (a*c ± b*d) = 2 muls + 1 add/sub
     * = 3 arithmetic ops. So contribution is 3 * cmuls. *)
    arithmetic_ops = !adds + !subs + !muls + !negs + (3 * !cmuls);
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
  let vec_arith = s.adds + s.subs + s.muls + s.negs + (2 * s.cmuls) in
  let scalar_ops = s.adds + s.subs + s.muls + s.negs + (3 * s.cmuls) in
  let buf = Buffer.create 512 in
  Buffer.add_string buf (Printf.sprintf "DAG nodes: %d total\n" s.total_nodes);
  Buffer.add_string buf (Printf.sprintf "  Loads:  %d\n" s.loads);
  Buffer.add_string buf (Printf.sprintf "  Consts: %d\n" s.consts);
  Buffer.add_string buf (Printf.sprintf "  Negs:   %d\n" s.negs);
  Buffer.add_string buf (Printf.sprintf "  Adds:   %d\n" s.adds);
  Buffer.add_string buf (Printf.sprintf "  Subs:   %d\n" s.subs);
  Buffer.add_string buf (Printf.sprintf "  Muls:   %d\n" s.muls);
  Buffer.add_string buf (Printf.sprintf "  Cmuls:  %d   (FMA-emitted: 2 instructions each — 1 mul + 1 fmadd/fnmadd)\n" s.cmuls);
  Buffer.add_string buf "\n";
  Buffer.add_string buf (Printf.sprintf "Vector instructions (FMA-fused, ISA-independent): %d\n" vec_arith);
  Buffer.add_string buf (Printf.sprintf "  Breakdown: %d add/sub/mul/neg + %d cmul-pair instructions\n"
    (s.adds + s.subs + s.muls + s.negs) (2 * s.cmuls));
  Buffer.add_string buf "\n";
  Buffer.add_string buf (Printf.sprintf "Scalar-equivalent ops (each Cmul = 3 ops):       %d\n" scalar_ops);
  Buffer.add_string buf (Printf.sprintf "  AVX-512 work (×8 lanes): %d ops/iter\n" (scalar_ops * 8));
  Buffer.add_string buf (Printf.sprintf "  AVX-2   work (×4 lanes): %d ops/iter\n" (scalar_ops * 4));
  Buffer.contents buf

(* === DAG PRETTY-PRINTING ===
 * Prints each unique node once, with tag, then the assignment list. *)

let string_of_node_kind (nk : node_kind) : string =
  match nk with
  | NK_Const c ->
    if c < 0.0 then Printf.sprintf "(%g)" c
    else Printf.sprintf "%g" c
  | NK_Load r -> Expr.string_of_elem_ref r
  | NK_Neg e -> Printf.sprintf "-t%d" e.tag
  | NK_Add (a, b) -> Printf.sprintf "t%d + t%d" a.tag b.tag
  | NK_Sub (a, b) -> Printf.sprintf "t%d - t%d" a.tag b.tag
  | NK_Mul (a, b) -> Printf.sprintf "t%d * t%d" a.tag b.tag
  | NK_CmulRe (a, b, c, d) ->
    Printf.sprintf "cmul.re(t%d, t%d, t%d, t%d)" a.tag b.tag c.tag d.tag
  | NK_CmulIm (a, b, c, d) ->
    Printf.sprintf "cmul.im(t%d, t%d, t%d, t%d)" a.tag b.tag c.tag d.tag

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
        visit a; visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        visit a; visit b; visit c; visit d
    end
  in
  List.iter visit roots;
  let nodes = Hashtbl.fold (fun _ e acc -> e :: acc) seen [] in
  let nodes = List.sort (fun a b -> compare a.tag b.tag) nodes in
  let buf = Buffer.create 4096 in
  List.iter (fun e ->
    Buffer.add_string buf
      (Printf.sprintf "  t%-3d = %s\n" e.tag (string_of_node_kind e.node))
  ) nodes;
  Buffer.add_string buf "\n";
  List.iter (fun (lhs, e) ->
    Buffer.add_string buf
      (Printf.sprintf "  %-12s = t%d\n" (Expr.string_of_elem_ref lhs) e.tag)
  ) assigns;
  Buffer.contents buf
