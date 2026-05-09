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
  | NK_Fma   of t * t * t * bool * bool

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

(* Lookup-only — returns Some node if it exists in the hash-cons table,
 * None if not. Used by share_subsums to detect pre-existing shareable
 * subexpressions without creating them. *)
let lookup_node (nk : node_kind) : t option =
  Hashtbl.find_opt hcons_table nk

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
         visit a; visit b; visit c; visit d
       | NK_Fma (a, b, c, _, _) ->
         (* Fma is opaque to dedup — same treatment as Cmul. *)
         bump_usage a.tag; bump_usage b.tag; bump_usage c.tag;
         visit a; visit b; visit c)
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
             im
           | NK_Fma (a, b, c, neg_mul, neg_add) ->
             (* Fma is opaque — rebuild its operands but preserve the
              * fused structure. *)
             hashcons (NK_Fma (rebuild a, rebuild b, rebuild c, neg_mul, neg_add)))
      in
      Hashtbl.add rebuild_cache e.tag result;
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
    (assigns : (Expr.elem_ref * t) list)
    : (Expr.elem_ref * t) list =
  if not aggressive then assigns
  else
  (* If n is Mul(x, Const c) or Mul(Const c, x), return Some (x, c). *)
  let const_mul_of (n : t) : (t * float) option =
    match n.node with
    | NK_Mul (a, b) ->
      (match a.node, b.node with
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
    | _ -> [(sign, e)]
  in
  (* Reconstruct a sum from a (sign, term) list. Separates positive and
   * negative terms, builds each via mk_add (which flattens + sorts +
   * pair-folds deterministically), then combines via mk_sub or mk_neg.
   * This ensures hash-cons hits when the same semantic sum is constructed
   * elsewhere — e.g., Neg(Add(a, b)) is canonical, never Sub(Neg(a), b). *)
  let rebuild_sum (terms : (int * t) list) : t =
    let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
    let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
    let build_sum lst = match lst with
      | [] -> mk_const 0.0
      | [x] -> x
      | x :: rest -> List.fold_left mk_add x rest
    in
    match pos, neg with
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
        if Sys.getenv_opt "FACTOR_TRACE" <> None && List.length terms >= 3 then begin
          Printf.eprintf "  factor_terms input (%d): " (List.length terms);
          List.iter (fun (s, t) ->
            match const_mul_of t with
            | Some (_, c) -> Printf.eprintf "%sc=%g " (if s>0 then "+" else "-") c
            | None -> Printf.eprintf "%sleaf(t%d) " (if s>0 then "+" else "-") t.tag
          ) terms;
          Printf.eprintf "\n"
        end;
        (* Bucket by constant value of Mul-coefficient.
         * Use float-bit-equality on the constant. *)
        let by_const : (int64, (int * t * t) list) Hashtbl.t = Hashtbl.create 8 in
        (* int64 = float bits; payload is (sign, x, original_mul) *)
        let leftover : (int * t) list ref = ref [] in
        List.iter (fun (sign, term) ->
          match const_mul_of term with
          | Some (x, c) ->
            let key = Int64.bits_of_float c in
            let cur = try Hashtbl.find by_const key with Not_found -> [] in
            Hashtbl.replace by_const key ((sign, x, term) :: cur)
          | _ ->
            leftover := (sign, term) :: !leftover
        ) terms;
        let factored = ref [] in
        let any_fired = ref false in
        Hashtbl.iter (fun key entries ->
          match entries with
          | [] -> ()
          | [(s, _, orig)] ->
            (* Single mul with this constant; not a factoring opportunity. *)
            leftover := (s, orig) :: !leftover
          | _ ->
            (* ≥2 muls share the same constant. Factor them. *)
            any_fired := true;
            changed := true;
            let inner_terms = List.map (fun (s, x, _) -> (s, x)) entries in
            let inner_sum = rebuild_sum inner_terms in
            let c = Int64.float_of_bits key in
            let factored_term = mk_mul inner_sum (mk_const c) in
            factored := (1, factored_term) :: !factored
        ) by_const;
        (!leftover @ !factored, !any_fired)
      in

      let rec rewrite (n : t) : t =
        match Hashtbl.find_opt cache n.tag with
        | Some r -> r
        | None ->
          let r = match n.node with
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
              let rewritten_terms = List.map (fun (s, t) -> (s, rewrite t)) raw_terms in
              let (new_terms, fired) = factor_terms rewritten_terms in
              if fired then rebuild_sum new_terms
              else
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_add_binary a' b'
            | NK_Sub (a, b) ->
              let raw_terms = flatten 1 n in
              let rewritten_terms = List.map (fun (s, t) -> (s, rewrite t)) raw_terms in
              let (new_terms, fired) = factor_terms rewritten_terms in
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
              let a' = rewrite a in let b' = rewrite b in
              let c' = rewrite c in let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulRe (a', b', c', d'))
            | NK_CmulIm (a, b, c, d) ->
              let a' = rewrite a in let b' = rewrite b in
              let c' = rewrite c in let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulIm (a', b', c', d'))
            | NK_Fma (a, b, c, neg_mul, neg_add) ->
              (* Fma is opaque to factoring — the muls inside are already
               * claimed by the FMA fusion. Recurse into operands but
               * never restructure. *)
              let a' = rewrite a in let b' = rewrite b in
              let c' = rewrite c in
              if a' == a && b' == b && c' == c then n
              else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
          in
          Hashtbl.add cache n.tag r;
          r
      in
      let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in
      if !changed then loop new_assigns (iter + 1)
      else new_assigns
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

let factor_by_atom ?(aggressive = false)
    (assigns : (Expr.elem_ref * t) list)
    : (Expr.elem_ref * t) list =
  if not aggressive then assigns
  else
  let const_of (e : t) : float option =
    match e.node with NK_Const c -> Some c | _ -> None
  in
  let rec atom_view (sign : int) (e : t) : (float * t) option =
    match e.node with
    | NK_Mul (a, b) ->
      (match const_of a, const_of b with
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
    | _ -> [(sign, e)]
  in
  let rebuild_sum (terms : (int * t) list) : t =
    let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
    let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
    let build lst = match lst with
      | [] -> mk_const 0.0
      | [x] -> x
      | x :: rest -> List.fold_left mk_add x rest
    in
    match pos, neg with
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
        let by_atom : (int, t * float ref * int ref) Hashtbl.t = Hashtbl.create 8 in
        let leftover : (int * t) list ref = ref [] in
        List.iter (fun (sign, term) ->
          match atom_view sign term with
          | Some (c, atom) ->
            (match Hashtbl.find_opt by_atom atom.tag with
             | Some (_, acc, count) ->
               acc := !acc +. c;
               incr count
             | None ->
               Hashtbl.add by_atom atom.tag (atom, ref c, ref 1))
          | None ->
            leftover := (sign, term) :: !leftover
        ) terms;

        let new_factored : (int * t) list ref = ref [] in
        let any_collapse_or_zero = ref false in
        Hashtbl.iter (fun _ (atom, c_ref, count_ref) ->
          let c = !c_ref in
          let count = !count_ref in
          if c = 0.0 then begin
            any_collapse_or_zero := true
          end else if count >= 2 then begin
            (* Multiple originals collapsed into one. *)
            let new_term = mk_mul (mk_const c) atom in
            new_factored := (1, new_term) :: !new_factored;
            any_collapse_or_zero := true
          end else begin
            (* Single occurrence — preserve as Mul(c, atom). *)
            let new_term = mk_mul (mk_const c) atom in
            new_factored := (1, new_term) :: !new_factored
          end
        ) by_atom;
        let final_terms = !new_factored @ !leftover in
        (final_terms, !any_collapse_or_zero)
      in

      let rec rewrite (n : t) : t =
        match Hashtbl.find_opt cache n.tag with
        | Some r -> r
        | None ->
          let r = match n.node with
            | NK_Const _ | NK_Load _ -> n
            | NK_Neg a ->
              let a' = rewrite a in
              if a' == a then n else mk_neg a'
            | NK_Add (a, b) ->
              let raw_terms = flatten 1 n in
              let rewritten_terms = List.map (fun (s, t) -> (s, rewrite t)) raw_terms in
              let (new_terms, fired) = factor_terms rewritten_terms in
              if fired then begin changed := true; rebuild_sum new_terms end
              else
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_add_binary a' b'
            | NK_Sub (a, b) ->
              let raw_terms = flatten 1 n in
              let rewritten_terms = List.map (fun (s, t) -> (s, rewrite t)) raw_terms in
              let (new_terms, fired) = factor_terms rewritten_terms in
              if fired then begin changed := true; rebuild_sum new_terms end
              else
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_sub_binary a' b'
            | NK_Mul (a, b) ->
              let a' = rewrite a in
              let b' = rewrite b in
              if a' == a && b' == b then n else mk_mul a' b'
            | NK_CmulRe (a, b, c, d) ->
              let a' = rewrite a in let b' = rewrite b in
              let c' = rewrite c in let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulRe (a', b', c', d'))
            | NK_CmulIm (a, b, c, d) ->
              let a' = rewrite a in let b' = rewrite b in
              let c' = rewrite c in let d' = rewrite d in
              if a' == a && b' == b && c' == c && d' == d then n
              else hashcons (NK_CmulIm (a', b', c', d'))
            | NK_Fma (a, b, c, neg_mul, neg_add) ->
              let a' = rewrite a in let b' = rewrite b in let c' = rewrite c in
              if a' == a && b' == b && c' == c then n
              else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
          in
          Hashtbl.add cache n.tag r;
          r
      in
      let new_assigns = List.map (fun (oref, e) -> (oref, rewrite e)) assigns in
      if !changed then loop new_assigns (iter + 1)
      else new_assigns
    end
  in
  loop assigns 0


let share_subsums ?(aggressive = false)
    (assigns : (Expr.elem_ref * t) list)
    : (Expr.elem_ref * t) list =
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
      | NK_Neg a -> bump a.tag; walk a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
        bump a.tag; bump b.tag; walk a; walk b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        bump a.tag; bump b.tag; bump c.tag; bump d.tag;
        walk a; walk b; walk c; walk d
      | NK_Fma (a, b, c, _, _) ->
        bump a.tag; bump b.tag; bump c.tag;
        walk a; walk b; walk c
    end
  in
  List.iter (fun (_, e) -> bump e.tag; walk e) assigns;

  let used_elsewhere n =
    (try Hashtbl.find use_count n.tag with Not_found -> 0) >= 1
  in

  let rec flatten (sign : int) (e : t) : (int * t) list =
    match e.node with
    | NK_Add (a, b) -> flatten sign a @ flatten sign b
    | NK_Sub (a, b) -> flatten sign a @ flatten (-sign) b
    | NK_Neg inner -> flatten (-sign) inner
    | _ -> [(sign, e)]
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
        let (s1, t1) = terms.(!i) in
        let (s2, t2) = terms.(!j) in
        if s1 = s2 && t1.tag <> t2.tag then begin
          let (a, b) = if t1.tag <= t2.tag then (t1, t2) else (t2, t1) in
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
    let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
    let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
    let build_chain lst = match lst with
      | [] -> mk_const 0.0
      | [x] -> x
      | x :: rest -> List.fold_left mk_add_binary x rest
    in
    match pos, neg with
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
      let r = match n.node with
        | NK_Const _ | NK_Load _ -> n
        | NK_Neg a ->
          let a' = rewrite a in
          if a' == a then n else mk_neg a'
        | NK_Add _ | NK_Sub _ ->
          (* Flatten this Add/Sub chain and try to share 2-term subsums. *)
          let raw_terms = flatten 1 n in
          let rewritten_terms = List.map (fun (s, t) -> (s, rewrite t)) raw_terms in
          if List.length rewritten_terms < 3 then begin
            (* Nothing to share at this level; preserve binary structure. *)
            match n.node with
            | NK_Add (a, b) ->
              let a' = rewrite a in let b' = rewrite b in
              if a' == a && b' == b then n else mk_add_binary a' b'
            | NK_Sub (a, b) ->
              let a' = rewrite a in let b' = rewrite b in
              if a' == a && b' == b then n else mk_sub_binary a' b'
            | _ -> n
          end else begin
            (* Greedy substitution of shareable pairs. *)
            let arr = ref (Array.of_list rewritten_terms) in
            let any_shared = ref false in
            let continue_loop = ref true in
            while !continue_loop do
              match find_shareable_pair !arr with
              | None -> continue_loop := false
              | Some (i, j, existing) ->
                any_shared := true;
                let (sign, _) = (!arr).(i) in
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
                let a' = rewrite a in let b' = rewrite b in
                if a' == a && b' == b then n else mk_add_binary a' b'
              | NK_Sub (a, b) ->
                let a' = rewrite a in let b' = rewrite b in
                if a' == a && b' == b then n else mk_sub_binary a' b'
              | _ -> n
          end
        | NK_Mul (a, b) ->
          let a' = rewrite a in
          let b' = rewrite b in
          if a' == a && b' == b then n else mk_mul a' b'
        | NK_CmulRe (a, b, c, d) ->
          let a' = rewrite a in let b' = rewrite b in
          let c' = rewrite c in let d' = rewrite d in
          if a' == a && b' == b && c' == c && d' == d then n
          else hashcons (NK_CmulRe (a', b', c', d'))
        | NK_CmulIm (a, b, c, d) ->
          let a' = rewrite a in let b' = rewrite b in
          let c' = rewrite c in let d' = rewrite d in
          if a' == a && b' == b && c' == c && d' == d then n
          else hashcons (NK_CmulIm (a', b', c', d'))
        | NK_Fma (a, b, c, neg_mul, neg_add) ->
          let a' = rewrite a in let b' = rewrite b in let c' = rewrite c in
          if a' == a && b' == b && c' == c then n
          else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
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

let transpose (assigns : (Expr.elem_ref * t) list)
    : (Expr.elem_ref * t) list =
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
       | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> dfs a; dfs b
       | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
         dfs a; dfs b; dfs c; dfs d
       | NK_Fma (a, b, c, _, _) ->
         dfs a; dfs b; dfs c);
      topo_rev := n :: !topo_rev
    end
  in
  List.iter (fun (_, e) -> dfs e) assigns;

  (* Step 2: Build parent map. For each node, record list of contributions
   * from each parent: (sign, scale_const_option, parent_node). *)
  let contribs : (int, (int * t option * t) list) Hashtbl.t = Hashtbl.create 256 in
  let add_contrib (child : t) (parent : t) (sign : int) (scale : t option) =
    let cur = try Hashtbl.find contribs child.tag with Not_found -> [] in
    Hashtbl.replace contribs child.tag ((sign, scale, parent) :: cur)
  in
  (* Process each node's structure to register contributions to its children. *)
  Hashtbl.iter (fun _ () -> ()) visited;
  List.iter (fun n ->
    match n.node with
    | NK_Const _ | NK_Load _ -> ()
    | NK_Neg a ->
      add_contrib a n (-1) None
    | NK_Add (a, b) ->
      add_contrib a n 1 None;
      add_contrib b n 1 None
    | NK_Sub (a, b) ->
      add_contrib a n 1 None;
      add_contrib b n (-1) None
    | NK_Mul (a, b) ->
      (* Const · X form — the X operand has weight = const.
       * Const itself never has a useful T value (it's a leaf with no
       * input semantics in transposition), so skip its contrib. *)
      (match a.node, b.node with
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
  ) (List.rev !topo_rev);
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
  List.iter (fun (oref, root) ->
    if not (Hashtbl.mem t_value root.tag) then
      Hashtbl.add t_value root.tag (mk_load oref)
  ) assigns;

  (* For nodes that have contribs (= internal nodes used by parents),
   * compute their T from their parents' T values. Process roots-first. *)
  List.iter (fun n ->
    match n.node with
    | NK_Const _ -> ()  (* constants have no transposed value *)
    | _ ->
      (* If this node already has a T (it's a root), skip. Otherwise
       * compute T from contribs. *)
      if not (Hashtbl.mem t_value n.tag) then begin
        let parent_contribs =
          try Hashtbl.find contribs n.tag with Not_found -> []
        in
        let terms = List.filter_map (fun (sign, scale, parent) ->
          match Hashtbl.find_opt t_value parent.tag with
          | None -> None  (* parent's T not computed; skip *)
          | Some t_parent ->
            let scaled = match scale with
              | None -> t_parent
              | Some c -> mk_mul c t_parent
            in
            Some (sign, scaled)
        ) parent_contribs in
        let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
        let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
        let build lst = match lst with
          | [] -> mk_const 0.0
          | [x] -> x
          | x :: rest -> List.fold_left mk_add x rest
        in
        let t_n = match pos, neg with
          | [], [] -> mk_const 0.0
          | _, [] -> build pos
          | [], _ -> mk_neg (build neg)
          | _, _ -> mk_sub (build pos) (build neg)
        in
        Hashtbl.add t_value n.tag t_n
      end
  ) !topo_rev;
  (* topo_rev is roots-first (DFS prepends after recursion → root at front). *)

  (* Step 4: Build new assigns. For each input Load (leaf with elem_ref),
   * the new assignment is (input_elem_ref, T[load]). *)
  let new_assigns = List.filter_map (fun n ->
    match n.node with
    | NK_Load r ->
      (match Hashtbl.find_opt t_value n.tag with
       | None -> None  (* No T computed (e.g., load not in any sum) *)
       | Some t_n -> Some (r, t_n))
    | _ -> None
  ) !topo_rev in
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

let fma_lift (assigns : (Expr.elem_ref * t) list)
    : (Expr.elem_ref * t) list =
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
      | NK_Neg a -> bump a.tag; walk a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
        bump a.tag; bump b.tag; walk a; walk b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        bump a.tag; bump b.tag; bump c.tag; bump d.tag;
        walk a; walk b; walk c; walk d
      | NK_Fma (a, b, c, _, _) ->
        bump a.tag; bump b.tag; bump c.tag;
        walk a; walk b; walk c
    end
  in
  List.iter (fun (_, e) -> bump e.tag; walk e) assigns;

  let single_use n = (try Hashtbl.find use_count n.tag with Not_found -> 0) = 1 in
  let _ = single_use in  (* now using the more permissive `liftable` below *)

  (* A Mul with use_count > 1 can still be lifted into FMAs IF all its
   * consumers will absorb it via FMA fusion. The cost analysis:
   *
   *   Shared (current behavior, single_use only): 1 mul + N adds = N+1 ops
   *   Duplicated into N fmas:                     N fmas             = N ops
   *
   * Duplication wins by 1 op per shared mul. The "duplication" is
   * implicit at the asm level: each Fma(a, b, c) computes a*b internally.
   * The original Mul node becomes unreachable from outputs IF all its
   * consumers are FMA-lifting Adds/Subs. emit_c won't emit it.
   *
   * If the Mul has consumers that AREN'T Add/Sub (e.g., another Mul, a
   * Cmul operand, or a direct output store), the standalone Mul stays
   * alive. In that case, lifting an Add consumer still works at the
   * source level, and at the asm level the Mul is computed once and the
   * FMA computes its own a*b in parallel — slight redundancy, but
   * harmless because mul throughput is high (2/cycle on modern x86).
   *
   * Net: we always lift Mul into Fma when the consumer is an Add/Sub,
   * regardless of use_count. *)
  let liftable_mul (_n : t) : bool = true in

  (* Step 2: Walk the DAG, lifting patterns greedily. Each Add/Sub examines
   * its operands; if one is a single-use Mul (or single-use Neg(Mul)),
   * lift to Fma. *)
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rewrite (n : t) : t =
    match Hashtbl.find_opt cache n.tag with
    | Some r -> r
    | None ->
      let r = match n.node with
        | NK_Const _ | NK_Load _ -> n
        | NK_Neg inner ->
          let inner' = rewrite inner in
          if inner' == inner then n else mk_neg inner'
        | NK_Mul (a, b) ->
          let a' = rewrite a in let b' = rewrite b in
          if a' == a && b' == b then n else mk_mul a' b'
        | NK_CmulRe (a, b, c, d) ->
          let a' = rewrite a in let b' = rewrite b in
          let c' = rewrite c in let d' = rewrite d in
          if a' == a && b' == b && c' == c && d' == d then n
          else hashcons (NK_CmulRe (a', b', c', d'))
        | NK_CmulIm (a, b, c, d) ->
          let a' = rewrite a in let b' = rewrite b in
          let c' = rewrite c in let d' = rewrite d in
          if a' == a && b' == b && c' == c && d' == d then n
          else hashcons (NK_CmulIm (a', b', c', d'))
        | NK_Fma (a, b, c, neg_mul, neg_add) ->
          let a' = rewrite a in let b' = rewrite b in let c' = rewrite c in
          if a' == a && b' == b && c' == c then n
          else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
        | NK_Add (a, b) ->
          let a' = rewrite a in let b' = rewrite b in
          (* Try to lift one operand into an FMA. Try LEFT first; if it
           * doesn't fuse, try RIGHT. Only one Mul can fuse per Add. *)
          (match try_lift_add_operand a' b' with
           | Some fma -> fma
           | None ->
             match try_lift_add_operand b' a' with
             | Some fma -> fma
             | None ->
               if a' == a && b' == b then n else mk_add_binary a' b')
        | NK_Sub (a, b) ->
          let a' = rewrite a in let b' = rewrite b in
          (* For Sub(a, b) we have two patterns:
           *   Sub(M, c)  → Fma(a_m, b_m, c, F, T)   (a_m*b_m - c)   — fmsub
           *   Sub(c, M)  → Fma(a_m, b_m, c, T, F)   (-a_m*b_m + c)  — fnmadd
           *   Sub(N, c)  → Fma(a_m, b_m, c, T, T)   (-a_m*b_m - c)  — fnmsub  (N = Neg(M))
           *   Sub(c, N)  → Fma(a_m, b_m, c, F, F)   (a_m*b_m + c)   — fmadd   (N = Neg(M))
           *)
          (match try_lift_sub_left a' b' with
           | Some fma -> fma
           | None ->
             match try_lift_sub_right a' b' with
             | Some fma -> fma
             | None ->
               if a' == a && b' == b then n else mk_sub_binary a' b')
      in
      Hashtbl.add cache n.tag r;
      r

  (* Try to fuse `m_node + other` as an FMA — m_node is the candidate
   * Mul/Neg(Mul) operand of an Add, other is the addend. *)
  and try_lift_add_operand (m_node : t) (other : t) : t option =
    match m_node.node with
    | NK_Mul (a, b) when liftable_mul m_node ->
      (* Fma(a, b, other, F, F) = a*b + other *)
      Some (hashcons (NK_Fma (a, b, other, false, false)))
    | NK_Neg inner when liftable_mul m_node ->
      (match inner.node with
       | NK_Mul (a, b) when liftable_mul inner ->
         (* Fma(a, b, other, T, F) = -a*b + other *)
         Some (hashcons (NK_Fma (a, b, other, true, false)))
       | _ -> None)
    | _ -> None

  (* Sub(left, right) where left is Mul/Neg(Mul). *)
  and try_lift_sub_left (left : t) (right : t) : t option =
    match left.node with
    | NK_Mul (a, b) when liftable_mul left ->
      (* Fma(a, b, right, F, T) = a*b - right *)
      Some (hashcons (NK_Fma (a, b, right, false, true)))
    | NK_Neg inner when liftable_mul left ->
      (match inner.node with
       | NK_Mul (a, b) when liftable_mul inner ->
         (* Fma(a, b, right, T, T) = -a*b - right *)
         Some (hashcons (NK_Fma (a, b, right, true, true)))
       | _ -> None)
    | _ -> None

  (* Sub(left, right) where right is Mul/Neg(Mul). *)
  and try_lift_sub_right (left : t) (right : t) : t option =
    match right.node with
    | NK_Mul (a, b) when liftable_mul right ->
      (* Fma(a, b, left, T, F) = -a*b + left *)
      Some (hashcons (NK_Fma (a, b, left, true, false)))
    | NK_Neg inner when liftable_mul right ->
      (match inner.node with
       | NK_Mul (a, b) when liftable_mul inner ->
         (* Fma(a, b, left, F, F) = a*b + left *)
         Some (hashcons (NK_Fma (a, b, left, false, false)))
       | _ -> None)
    | _ -> None
  in
  List.map (fun (oref, e) -> (oref, rewrite e)) assigns


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
  fmas : int;             (* number of NK_Fma nodes (each = 1 instruction) *)
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
  let fmas = ref 0 in
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
       | NK_CmulRe _ | NK_CmulIm _ -> incr cmuls
       | NK_Fma _ -> incr fmas);
      match e.node with
      | NK_Const _ | NK_Load _ -> ()
      | NK_Neg e1 -> visit e1
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
        visit a; visit b
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        visit a; visit b; visit c; visit d
      | NK_Fma (a, b, c, _, _) ->
        visit a; visit b; visit c
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
  let scalar_ops = s.adds + s.subs + s.muls + s.negs + (3 * s.cmuls) + (2 * s.fmas) in
  let buf = Buffer.create 512 in
  Buffer.add_string buf (Printf.sprintf "DAG nodes: %d total\n" s.total_nodes);
  Buffer.add_string buf (Printf.sprintf "  Loads:  %d\n" s.loads);
  Buffer.add_string buf (Printf.sprintf "  Consts: %d\n" s.consts);
  Buffer.add_string buf (Printf.sprintf "  Negs:   %d\n" s.negs);
  Buffer.add_string buf (Printf.sprintf "  Adds:   %d\n" s.adds);
  Buffer.add_string buf (Printf.sprintf "  Subs:   %d\n" s.subs);
  Buffer.add_string buf (Printf.sprintf "  Muls:   %d\n" s.muls);
  Buffer.add_string buf (Printf.sprintf "  Cmuls:  %d   (each = 1 mul + 1 fmadd/fnmadd = 2 instructions)\n" s.cmuls);
  Buffer.add_string buf (Printf.sprintf "  Fmas:   %d   (each = 1 fmadd/fmsub/fnmadd/fnmsub = 1 instruction)\n" s.fmas);
  Buffer.add_string buf "\n";
  Buffer.add_string buf (Printf.sprintf "Vector instructions (FMA-fused, ISA-independent): %d\n" vec_arith);
  Buffer.add_string buf (Printf.sprintf "  Breakdown: %d add/sub/mul/neg + %d cmul-pair instructions + %d fma\n"
    (s.adds + s.subs + s.muls + s.negs) (2 * s.cmuls) s.fmas);
  Buffer.add_string buf "\n";
  Buffer.add_string buf (Printf.sprintf "Scalar-equivalent ops (each Cmul = 3 ops, each Fma = 2 ops): %d\n" scalar_ops);
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
  | NK_Fma (a, b, c, neg_mul, neg_add) ->
    let sign_mul = if neg_mul then "-" else "+" in
    let sign_add = if neg_add then "-" else "+" in
    Printf.sprintf "fma(%st%d*t%d, %st%d)" sign_mul a.tag b.tag sign_add c.tag

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
      | NK_Fma (a, b, c, _, _) ->
        visit a; visit b; visit c
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
