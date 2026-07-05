(* algsimp.ml — facade of the IR + pass stack, plus the passes that
 * bridge to neighbouring layers.
 *
 * `include Fma_passes` re-exports the whole chain
 * (Ir < Simplify < Fma_passes), so every `Algsimp.X` reference and
 * `open Algsimp` site compiles unchanged and module-level mutable
 * state (hash-cons table, tag counter, memo) stays physically
 * single. Owned here: spill-marker lifting (the Dft -> tag bridge),
 * butterfly_share_mul, and DAG statistics / pretty-printing.
 * ------------------------------------------------------------------
 * MODULE CARD (algsimp.ml — grep "MODULE CARD" for the full set)
 * ROLE: Facade of the IR + pass stack, plus the layer-bridging passes
 * that need Dft types (lift_spill_markers) or serve diagnostics
 * (stats_reachable, print_dag) and butterfly_share_mul.
 * PIPELINE: the name every layer above the math level talks to.
 * PUBLIC SURFACE (measured; grep counts incl. comments): schedule(68),
 * codelet_oop(42), bb(36), regalloc(34), gen_main(28), pipeline(26),
 * emit_render(8), annotate(4), dft(3), dft_select(1) + bin tools
 * dbg_eval(25), test_mk_plus(14), dump_ir(13). Hot names: t, tag,
 * node, preds, of_assignments, reset, the pass entry points.
 * DEPS: Fma_passes chain via include; Dft(5) for spill_marker.
 * ENV: SPILL_MARKER_TRACE.
 * ------------------------------------------------------------------
 *)

include Fma_passes

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
