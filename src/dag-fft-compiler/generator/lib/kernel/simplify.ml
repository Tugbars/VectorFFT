(* simplify.ml — algebraic rewrite passes over the hash-consed IR.
 *
 * The non-FMA passes: sub-pair dedup, collect-M / deep-collect,
 * sub-neg-mul lifting, distributive factoring (common-const and
 * by-atom), subsum sharing, and Frigo network transposition. Each
 * pass is a standalone assignment-list -> assignment-list function;
 * gen_main / pipeline choose the order and gating.
 *
 * Gating conventions worth knowing before touching anything here:
 *   - The aggressive-only passes (factor_common_muls, factor_by_atom,
 *     share_subsums) short-circuit to identity unless aggressive=true,
 *     which drivers derive from Dft_select.pick_algorithm = Direct (monolithic
 *     odd primes). Running them on CT-decomposed DAGs is documented
 *     unsafe: they shred the Cmul sharing the recursion built.
 *   - collect_m / deep_collect are env-gated opt-ins, identity when
 *     the corresponding VFFT variable is unset.
 *   - dedup_sub_pairs is on by default; VFFT_NO_SUBDEDUP=1 disables it
 *     at the drivers.
 * ------------------------------------------------------------------
 * MODULE CARD (simplify.ml — grep "MODULE CARD" for the full set)
 * ROLE: The algebraic (non-FMA) rewrite passes over the Ir DAG.
 * PIPELINE: Ir.of_assignments output -> these passes -> FMA family
 * PUBLIC SURFACE (measured): zero direct Simplify.X references —
 * callers use the Algsimp facade: dedup_sub_pairs, collect_m,
 * deep_collect, factor_common_muls, factor_by_atom, share_subsums
 * (gen_main, pipeline, bin/dbg_eval, bin/test_mk_plus).
 * DEPS: Ir via include (re-exported onward); Expr(13).
 * ENV: VFFT_COLLECT_M, VFFT_DEEP_COLLECT, VFFT_DEEP_COLLECT_TRACE.
 * ------------------------------------------------------------------
 *)

open Ir  (* M1: was `include` — Ir is no longer re-exported through Simplify *)
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
  let usage_count = Hashtbl.create 256 in
  (* tag -> count *)
  let sub_index = Hashtbl.create 64 in
  (* (small_tag, big_tag) -> [Sub nodes both directions] *)
  let bump_usage tag =
    let c =
      try Hashtbl.find usage_count tag with
      | Not_found -> 0
    in
    Hashtbl.replace usage_count tag (c + 1)
  in
  let rec visit (e : t) =
    if not (Hashtbl.mem visited e.tag)
    then (
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
        let key = if a.tag < b.tag then a.tag, b.tag else b.tag, a.tag in
        let prev =
          try Hashtbl.find sub_index key with
          | Not_found -> []
        in
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
      | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:648")
  in
  List.iter (fun (_, e) -> visit e) assigns;
  (* Step 2: find Sub-pair conflicts and pick winners. *)
  let substitute : (int, t) Hashtbl.t = Hashtbl.create 16 in
  Hashtbl.iter
    (fun _key nodes ->
       match nodes with
       | [ _ ] -> () (* only one direction in the DAG, no conflict *)
       | nodes_list ->
         (* Multiple Sub nodes share the same (small,big) key. Should be
          * exactly two: Sub(a,b) and Sub(b,a). Pick the winner by usage. *)
         let scored =
           List.map
             (fun n ->
                let c =
                  try Hashtbl.find usage_count n.tag with
                  | Not_found -> 0
                in
                c, n)
             nodes_list
         in
         let scored =
           List.sort
             (fun (c1, n1) (c2, n2) ->
                (* Higher usage wins; tie-break by lower tag (deterministic). *)
                if c1 <> c2 then compare c2 c1 else compare n1.tag n2.tag)
             scored
         in
         (match scored with
          | (_, winner) :: losers ->
            List.iter
              (fun (_, loser) ->
                 if loser.tag <> winner.tag
                 then Hashtbl.add substitute loser.tag (mk_neg winner))
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
             let re, _im = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
             re
           | NK_CmulIm (a, b, c, d) ->
             let _re, im = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
             im
           | NK_Fma (a, b, c, neg_mul, neg_add) ->
             (* Fma is opaque — rebuild its operands but preserve the
              * fused structure. *)
             hashcons (NK_Fma (rebuild a, rebuild b, rebuild c, neg_mul, neg_add))
           | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:718")
      in
      Hashtbl.add rebuild_cache e.tag result;
      result
  in
  List.map (fun (lhs, e) -> lhs, rebuild e) assigns
;;

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
    | NK_Mul (a, b) ->
      (match a.node, b.node with
       | NK_Const c, _ -> c, b
       | _, NK_Const c -> c, a
       | _ -> 1.0, t)
    | _ -> 1.0, t
  in
  match t.node with
  | NK_Neg inner ->
    let c, atom = unsigned inner in
    -.c, atom
  | _ -> unsigned t
;;

let collect_m (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  if
    Sys.getenv_opt "VFFT_COLLECT_M" <> Some "1"
    && Sys.getenv_opt "VFFT_DEEP_COLLECT" <> Some "1"
  then assigns
  else (
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
            if subtree_has_collectible e
            then collect_subtree e
            else mk_add_binary (rebuild a) (rebuild b)
          | NK_Sub (a, b) ->
            if subtree_has_collectible e
            then collect_subtree e
            else mk_sub_binary (rebuild a) (rebuild b)
          | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
          | NK_CmulRe (a, b, c, d) ->
            let re, _ = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
            re
          | NK_CmulIm (a, b, c, d) ->
            let _, im = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
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
             if Hashtbl.mem seen_atoms atom.tag
             then has_dup := true
             else Hashtbl.add seen_atoms atom.tag ())
        terms;
      !has_dup || !n_consts > 1
    (* Collect a subtree: flatten (through Fma), group by atom, emit. *)
    and collect_subtree (e : t) : t =
      let terms = flatten_sum_through_fma 1 e in
      let rebuilt = List.map (fun (s, t) -> s, rebuild t) terms in
      let by_atom : (int, float * t) Hashtbl.t = Hashtbl.create 16 in
      let constant_acc = ref 0.0 in
      List.iter
        (fun (sign, term) ->
           match term.node with
           | NK_Const c -> constant_acc := !constant_acc +. (float_of_int sign *. c)
           | _ ->
             let coeff, atom = extract_coefficient term in
             let signed_coeff = float_of_int sign *. coeff in
             (match Hashtbl.find_opt by_atom atom.tag with
              | None -> Hashtbl.add by_atom atom.tag (signed_coeff, atom)
              | Some (existing, _) ->
                Hashtbl.replace by_atom atom.tag (existing +. signed_coeff, atom)))
        rebuilt;
      let new_terms = ref [] in
      Hashtbl.iter
        (fun _ (c, atom) ->
           if c <> 0.0
           then (
             let term = mk_mul (mk_const c) atom in
             new_terms := (1, term) :: !new_terms))
        by_atom;
      if !constant_acc <> 0.0 then new_terms := (1, mk_const !constant_acc) :: !new_terms;
      let plus_node = mk_plus !new_terms in
      lower_plus plus_node
    in
    List.map (fun (lhs, e) -> lhs, rebuild e) assigns)
;;

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
  if depth <= 0
  then [ sign, t ]
  else (
    match t.node with
    | NK_Neg inner -> distribute_term ~depth (-sign, inner)
    | NK_Mul (a, b) ->
      (* Identify which operand is Const (if any). *)
      let const_part, other_part =
        match a.node, b.node with
        | NK_Const _, _ -> Some a, b
        | _, NK_Const _ -> Some b, a
        | _ -> None, t
      in
      (match const_part with
       | None -> [ sign, t ]
       | Some c ->
         (match other_part.node with
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
          | NK_Mul (m1, m2) ->
            (* c * Mul(...): if inner has Const, rotate; otherwise leave. *)
            let rotated_opt =
              match m1.node, m2.node with
              | NK_Const _, _ -> Some (mk_mul (mk_mul c m1) m2)
              | _, NK_Const _ -> Some (mk_mul (mk_mul c m2) m1)
              | _ -> None
            in
            (match rotated_opt with
             | Some rotated when rotated != t -> distribute_term ~depth (sign, rotated)
             | _ -> [ sign, t ])
          | _ -> [ sign, t ]))
    | _ -> [ sign, t ])
;;

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
       | _ ->
         let coeff, atom = extract_coefficient term in
         let signed_coeff = float_of_int sign *. coeff in
         (match Hashtbl.find_opt by_atom atom.tag with
          | None -> Hashtbl.add by_atom atom.tag (signed_coeff, atom)
          | Some (existing, _) ->
            Hashtbl.replace by_atom atom.tag (existing +. signed_coeff, atom)))
    terms;
  let new_terms = ref [] in
  Hashtbl.iter
    (fun _ (c, atom) ->
       if c <> 0.0
       then (
         let term = mk_mul (mk_const c) atom in
         new_terms := (1, term) :: !new_terms))
    by_atom;
  if !constant_acc <> 0.0 then new_terms := (1, mk_const !constant_acc) :: !new_terms;
  lower_plus (mk_plus !new_terms)
;;

(* Count the IR nodes reachable from a node (treats hashcons as identity).
 * Used as a cost heuristic to compare distributed vs original forms. *)
let count_ir_nodes (root : t) : int =
  let seen = Hashtbl.create 64 in
  let n = ref 0 in
  let rec walk e =
    if Hashtbl.mem seen e.tag
    then ()
    else (
      Hashtbl.add seen e.tag ();
      incr n;
      List.iter walk (preds e))
  in
  walk root;
  !n
;;

let deep_collect ?(depth_limit = 5) (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref * t) list
  =
  if Sys.getenv_opt "VFFT_DEEP_COLLECT" <> Some "1"
  then assigns
  else (
    let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
    let rec rebuild (e : t) : t =
      match Hashtbl.find_opt cache e.tag with
      | Some r -> r
      | None ->
        let r =
          match e.node with
          | NK_Const _ | NK_Load _ -> e
          | NK_Neg inner -> mk_neg (rebuild inner)
          | NK_Add (a, b) -> try_deep_collect e (mk_add_binary (rebuild a) (rebuild b))
          | NK_Sub (a, b) -> try_deep_collect e (mk_sub_binary (rebuild a) (rebuild b))
          | NK_Mul (a, b) -> mk_mul (rebuild a) (rebuild b)
          | NK_CmulRe (a, b, c, d) ->
            let re, _ = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
            re
          | NK_CmulIm (a, b, c, d) ->
            let _, im = mk_cmul (rebuild a) (rebuild b) (rebuild c) (rebuild d) in
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
        (NK_Mul ((if c.tag <= x.tag then c else x), if c.tag <= x.tag then x else c))
      <> None
      || lookup_node
           (NK_Mul ((if c.tag <= y.tag then c else y), if c.tag <= y.tag then y else c))
         <> None
    and distribute_use_aware ~depth ((sign, t) : int * t) : (int * t) list =
      if depth <= 0
      then [ sign, t ]
      else (
        match t.node with
        | NK_Neg inner -> distribute_use_aware ~depth (-sign, inner)
        | NK_Mul (a, b) ->
          let const_part, other_part =
            match a.node, b.node with
            | NK_Const _, _ -> Some a, b
            | _, NK_Const _ -> Some b, a
            | _ -> None, t
          in
          (match const_part with
           | None -> [ sign, t ]
           | Some c ->
             (match other_part.node with
              | NK_Add (x, y) when any_mul_exists c x y ->
                distribute_use_aware ~depth:(depth - 1) (sign, mk_mul c x)
                @ distribute_use_aware ~depth:(depth - 1) (sign, mk_mul c y)
              | NK_Sub (x, y) when any_mul_exists c x y ->
                distribute_use_aware ~depth:(depth - 1) (sign, mk_mul c x)
                @ distribute_use_aware ~depth:(depth - 1) (-sign, mk_mul c y)
              | NK_Neg inner -> distribute_use_aware ~depth (-sign, mk_mul c inner)
              | NK_Mul (m1, m2) ->
                let rotated_opt =
                  match m1.node, m2.node with
                  | NK_Const _, _ -> Some (mk_mul (mk_mul c m1) m2)
                  | _, NK_Const _ -> Some (mk_mul (mk_mul c m2) m1)
                  | _ -> None
                in
                (match rotated_opt with
                 | Some rotated when rotated != t ->
                   distribute_use_aware ~depth (sign, rotated)
                 | _ -> [ sign, t ])
              | _ -> [ sign, t ]))
        | _ -> [ sign, t ])
    and try_deep_collect (original : t) (rebuilt_binary : t) : t =
      (* Use the FMA-aware flatten so we see through early-peephole
       * Fma nodes that block ordinary flatten_sum. *)
      let terms = flatten_sum_through_fma 1 original in
      let n_input_terms = List.length terms in
      let rebuilt_terms = List.map (fun (s, t) -> s, rebuild t) terms in
      let distributed =
        List.concat_map (distribute_use_aware ~depth:depth_limit) rebuilt_terms
      in
      if List.length distributed <= n_input_terms
      then rebuilt_binary
      else (
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
        if Sys.getenv_opt "VFFT_DEEP_COLLECT_TRACE" = Some "1"
        then
          Printf.eprintf
            "deep_collect: in=%d dist=%d groups=%d %s\n"
            n_input_terms
            (List.length distributed)
            n_groups
            (if win then "WIN" else "skip");
        if win then collect_terms_to_tree distributed else rebuilt_binary)
    in
    List.map (fun (lhs, e) -> lhs, rebuild e) assigns)
;;

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

let lift_sub_neg_mul (assigns : (Expr.elem_ref * t) list) : (Expr.elem_ref * t) list =
  let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec rebuild (e : t) : t =
    match Hashtbl.find_opt cache e.tag with
    | Some r -> r
    | None ->
      let result =
        match e.node with
        | NK_Sub (a, b) ->
          let a' = rebuild a in
          let b' = rebuild b in
          (* Pattern match: Sub(Neg(Mul(x, y)), z) → NK_Fma(x, y, z, true, true) *)
          (match a'.node with
           | NK_Neg inner ->
             (match inner.node with
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
  List.map (fun (lhs, e) -> lhs, rebuild e) assigns
;;

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

let factor_common_muls ?(aggressive = false) (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref * t) list
  =
  if not aggressive
  then assigns
  else (
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
      | _ -> [ sign, e ]
    in
    (* Reconstruct a sum from a (sign, term) list. Separates positive and
     * negative terms, builds each via mk_add (which flattens + sorts +
     * pair-folds deterministically), then combines via mk_sub or mk_neg.
     * This ensures hash-cons hits when the same semantic sum is constructed
     * elsewhere — e.g., Neg(Add(a, b)) is canonical, never Sub(Neg(a), b). *)
    let rebuild_sum (terms : (int * t) list) : t =
      let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
      let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
      let build_sum lst =
        match lst with
        | [] -> mk_const 0.0
        | [ x ] -> x
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
      if iter >= max_iter
      then assigns
      else (
        let changed = ref false in
        let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
        (* Try to factor a flat term list. Returns new term list and whether
         * any factoring fired. Groups of same-constant Muls become a single
         * new term: (+1, Mul(inner_sum, Const c)). No use-count safety —
         * aggressive mode treats all cross-output mul-sharing as factor-eligible
         * because primes' Winograd structure emerges from precisely this. *)
        let factor_terms (terms : (int * t) list) : (int * t) list * bool =
          if Sys.getenv_opt "FACTOR_TRACE" <> None && List.length terms >= 3
          then (
            Printf.eprintf "  factor_terms input (%d): " (List.length terms);
            List.iter
              (fun (s, t) ->
                 match const_mul_of t with
                 | Some (_, c) -> Printf.eprintf "%sc=%g " (if s > 0 then "+" else "-") c
                 | None ->
                   Printf.eprintf "%sleaf(t%d) " (if s > 0 then "+" else "-") t.tag)
              terms;
            Printf.eprintf "\n");
          (* Bucket by constant value of Mul-coefficient.
           * Use float-bit-equality on the constant. *)
          let by_const : (int64, (int * t * t) list) Hashtbl.t = Hashtbl.create 8 in
          (* int64 = float bits; payload is (sign, x, original_mul) *)
          let leftover : (int * t) list ref = ref [] in
          List.iter
            (fun (sign, term) ->
               match const_mul_of term with
               | Some (x, c) ->
                 let key = Int64.bits_of_float c in
                 let cur =
                   try Hashtbl.find by_const key with
                   | Not_found -> []
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
                 let inner_terms = List.map (fun (s, x, _) -> s, x) entries in
                 let inner_sum = rebuild_sum inner_terms in
                 let c = Int64.float_of_bits key in
                 let factored_term = mk_mul inner_sum (mk_const c) in
                 factored := (1, factored_term) :: !factored)
            by_const;
          !leftover @ !factored, !any_fired
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
                let rewritten_terms = List.map (fun (s, t) -> s, rewrite t) raw_terms in
                let new_terms, fired = factor_terms rewritten_terms in
                if fired
                then rebuild_sum new_terms
                else (
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_add_binary a' b')
              | NK_Sub (a, b) ->
                let raw_terms = flatten 1 n in
                let rewritten_terms = List.map (fun (s, t) -> s, rewrite t) raw_terms in
                let new_terms, fired = factor_terms rewritten_terms in
                if fired
                then rebuild_sum new_terms
                else (
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_sub_binary a' b')
              | NK_Mul (a, b) ->
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_mul a' b'
              | NK_CmulRe (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d
                then n
                else hashcons (NK_CmulRe (a', b', c', d'))
              | NK_CmulIm (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d
                then n
                else hashcons (NK_CmulIm (a', b', c', d'))
              | NK_Fma (a, b, c, neg_mul, neg_add) ->
                (* Fma is opaque to factoring — the muls inside are already
                 * claimed by the FMA fusion. Recurse into operands but
                 * never restructure. *)
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                if a' == a && b' == b && c' == c
                then n
                else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
              | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:964"
            in
            Hashtbl.add cache n.tag r;
            r
        in
        let new_assigns = List.map (fun (oref, e) -> oref, rewrite e) assigns in
        if !changed then loop new_assigns (iter + 1) else new_assigns)
    in
    loop assigns 0)
;;

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

let factor_by_atom ?(aggressive = false) (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref * t) list
  =
  if not aggressive
  then assigns
  else (
    let const_of (e : t) : float option =
      match e.node with
      | NK_Const c -> Some c
      | _ -> None
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
      | _ -> [ sign, e ]
    in
    let rebuild_sum (terms : (int * t) list) : t =
      let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
      let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
      let build lst =
        match lst with
        | [] -> mk_const 0.0
        | [ x ] -> x
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
      if iter >= max_iter
      then assigns
      else (
        let changed = ref false in
        let cache : (int, t) Hashtbl.t = Hashtbl.create 256 in
        (* Bucket flat terms by atom-tag, sum coefficients (compile-time fold).
         * `fired` = at least one bucket had multiple entries OR a coefficient
         * summed to zero. *)
        let factor_terms (terms : (int * t) list) : (int * t) list * bool =
          let by_atom : (int, t * float ref * int ref) Hashtbl.t = Hashtbl.create 8 in
          let leftover : (int * t) list ref = ref [] in
          List.iter
            (fun (sign, term) ->
               match atom_view sign term with
               | Some (c, atom) ->
                 (match Hashtbl.find_opt by_atom atom.tag with
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
               if c = 0.0
               then any_collapse_or_zero := true
               else if count >= 2
               then (
                 (* Multiple originals collapsed into one. *)
                 let new_term = mk_mul (mk_const c) atom in
                 new_factored := (1, new_term) :: !new_factored;
                 any_collapse_or_zero := true)
               else (
                 (* Single occurrence — preserve as Mul(c, atom). *)
                 let new_term = mk_mul (mk_const c) atom in
                 new_factored := (1, new_term) :: !new_factored))
            by_atom;
          let final_terms = !new_factored @ !leftover in
          final_terms, !any_collapse_or_zero
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
                let rewritten_terms = List.map (fun (s, t) -> s, rewrite t) raw_terms in
                let new_terms, fired = factor_terms rewritten_terms in
                if fired
                then (
                  changed := true;
                  rebuild_sum new_terms)
                else (
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_add_binary a' b')
              | NK_Sub (a, b) ->
                let raw_terms = flatten 1 n in
                let rewritten_terms = List.map (fun (s, t) -> s, rewrite t) raw_terms in
                let new_terms, fired = factor_terms rewritten_terms in
                if fired
                then (
                  changed := true;
                  rebuild_sum new_terms)
                else (
                  let a' = rewrite a in
                  let b' = rewrite b in
                  if a' == a && b' == b then n else mk_sub_binary a' b')
              | NK_Mul (a, b) ->
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_mul a' b'
              | NK_CmulRe (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d
                then n
                else hashcons (NK_CmulRe (a', b', c', d'))
              | NK_CmulIm (a, b, c, d) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                let d' = rewrite d in
                if a' == a && b' == b && c' == c && d' == d
                then n
                else hashcons (NK_CmulIm (a', b', c', d'))
              | NK_Fma (a, b, c, neg_mul, neg_add) ->
                let a' = rewrite a in
                let b' = rewrite b in
                let c' = rewrite c in
                if a' == a && b' == b && c' == c
                then n
                else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
              | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1174"
            in
            Hashtbl.add cache n.tag r;
            r
        in
        let new_assigns = List.map (fun (oref, e) -> oref, rewrite e) assigns in
        if !changed then loop new_assigns (iter + 1) else new_assigns)
    in
    loop assigns 0)
;;

let share_subsums ?(aggressive = false) (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref * t) list
  =
  if not aggressive
  then assigns
  else (
    (* Use-count over the whole DAG (excluding our reconstruction). *)
    let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    let visited : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    let bump tag =
      let c =
        try Hashtbl.find use_count tag with
        | Not_found -> 0
      in
      Hashtbl.replace use_count tag (c + 1)
    in
    let rec walk e =
      if not (Hashtbl.mem visited e.tag)
      then (
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
        | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1242")
    in
    List.iter
      (fun (_, e) ->
         bump e.tag;
         walk e)
      assigns;
    let used_elsewhere n =
      (try Hashtbl.find use_count n.tag with
       | Not_found -> 0)
      >= 1
    in
    let rec flatten (sign : int) (e : t) : (int * t) list =
      match e.node with
      | NK_Add (a, b) -> flatten sign a @ flatten sign b
      | NK_Sub (a, b) -> flatten sign a @ flatten (-sign) b
      | NK_Neg inner -> flatten (-sign) inner
      | _ -> [ sign, e ]
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
          if s1 = s2 && t1.tag <> t2.tag
          then (
            let a, b = if t1.tag <= t2.tag then t1, t2 else t2, t1 in
            match lookup_node (NK_Add (a, b)) with
            | Some existing when used_elsewhere existing ->
              result := Some (!i, !j, existing)
            | _ -> ());
          incr j
        done;
        incr i
      done;
      !result
    in
    let rebuild_sum_binary (terms : (int * t) list) : t =
      let pos = List.filter_map (fun (s, t) -> if s > 0 then Some t else None) terms in
      let neg = List.filter_map (fun (s, t) -> if s < 0 then Some t else None) terms in
      let build_chain lst =
        match lst with
        | [] -> mk_const 0.0
        | [ x ] -> x
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
        let r =
          match n.node with
          | NK_Const _ | NK_Load _ -> n
          | NK_Neg a ->
            let a' = rewrite a in
            if a' == a then n else mk_neg a'
          | NK_Add _ | NK_Sub _ ->
            (* Flatten this Add/Sub chain and try to share 2-term subsums. *)
            let raw_terms = flatten 1 n in
            let rewritten_terms = List.map (fun (s, t) -> s, rewrite t) raw_terms in
            if List.length rewritten_terms < 3
            then (* Nothing to share at this level; preserve binary structure. *)
              (
              match n.node with
              | NK_Add (a, b) ->
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_add_binary a' b'
              | NK_Sub (a, b) ->
                let a' = rewrite a in
                let b' = rewrite b in
                if a' == a && b' == b then n else mk_sub_binary a' b'
              | _ -> n)
            else (
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
                  new_arr.(i) <- sign, existing;
                  Array.blit !arr (i + 1) new_arr (i + 1) (j - i - 1);
                  if j < n_arr - 1 then Array.blit !arr (j + 1) new_arr j (n_arr - 1 - j);
                  arr := new_arr
              done;
              if !any_shared
              then rebuild_sum_binary (Array.to_list !arr)
              else (
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
                | _ -> n))
          | NK_Mul (a, b) ->
            let a' = rewrite a in
            let b' = rewrite b in
            if a' == a && b' == b then n else mk_mul a' b'
          | NK_CmulRe (a, b, c, d) ->
            let a' = rewrite a in
            let b' = rewrite b in
            let c' = rewrite c in
            let d' = rewrite d in
            if a' == a && b' == b && c' == c && d' == d
            then n
            else hashcons (NK_CmulRe (a', b', c', d'))
          | NK_CmulIm (a, b, c, d) ->
            let a' = rewrite a in
            let b' = rewrite b in
            let c' = rewrite c in
            let d' = rewrite d in
            if a' == a && b' == b && c' == c && d' == d
            then n
            else hashcons (NK_CmulIm (a', b', c', d'))
          | NK_Fma (a, b, c, neg_mul, neg_add) ->
            let a' = rewrite a in
            let b' = rewrite b in
            let c' = rewrite c in
            if a' == a && b' == b && c' == c
            then n
            else hashcons (NK_Fma (a', b', c', neg_mul, neg_add))
          | NK_Plus _ -> nk_plus_unreachable "algsimp.ml:1315"
        in
        Hashtbl.add cache n.tag r;
        r
    in
    List.map (fun (oref, e) -> oref, rewrite e) assigns)
;;

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
    if not (Hashtbl.mem visited n.tag)
    then (
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
      topo_rev := n :: !topo_rev)
  in
  List.iter (fun (_, e) -> dfs e) assigns;
  (* Step 2: Build parent map. For each node, record list of contributions
   * from each parent: (sign, scale_const_option, parent_node). *)
  let contribs : (int, (int * t option * t) list) Hashtbl.t = Hashtbl.create 256 in
  let add_contrib (child : t) (parent : t) (sign : int) (scale : t option) =
    let cur =
      try Hashtbl.find contribs child.tag with
      | Not_found -> []
    in
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
       if not (Hashtbl.mem t_value root.tag)
       then Hashtbl.add t_value root.tag (mk_load oref))
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
         if not (Hashtbl.mem t_value n.tag)
         then (
           let parent_contribs =
             try Hashtbl.find contribs n.tag with
             | Not_found -> []
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
           let t_n =
             match pos, neg with
             | [], [] -> mk_const 0.0
             | _, [] -> build pos
             | [], _ -> mk_neg (build neg)
             | _, _ -> mk_sub (build pos) (build neg)
           in
           Hashtbl.add t_value n.tag t_n))
    !topo_rev;
  (* topo_rev is roots-first (DFS prepends after recursion → root at front). *)

  (* Step 4: Build new assigns. For each input Load (leaf with elem_ref),
   * the new assignment is (input_elem_ref, T[load]). *)
  let new_assigns =
    List.filter_map
      (fun n ->
         match n.node with
         | NK_Load r ->
           (match Hashtbl.find_opt t_value n.tag with
            | None -> None (* No T computed (e.g., load not in any sum) *)
            | Some t_n -> Some (r, t_n))
         | _ -> None)
      !topo_rev
  in
  new_assigns
;;
