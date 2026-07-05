(* fma_passes.ml — the FMA rewrite family.
 *
 * fma_lift (single-use Add/Sub-of-Mul fusion), factor_const_muls,
 * multi_use_fma_lift, fma_addend_factor and the 2-FMA-chain
 * flattener. All frozen-tag aware: passes that rewrite operand
 * structure take the spill-marker frozen set and return a tag remap
 * (consumed by the pipeline's marker remap chain).
 *
 * The cascade discipline: drivers run factor_const_muls, then
 * alternate fma_addend_factor / multi_use_fma_lift a few rounds, then
 * flatten_fma_mul_addend last. Every pass must (a) leave frozen tags
 * untouched and (b) report where each rewritten tag moved, because
 * spill markers reference exact tags and the emitter walks only the
 * reachable-from-assignments subset — a missed remap means PASS 2
 * reloads a dead node. See pipeline.ml for the canonical ordering.
 * ------------------------------------------------------------------
 * MODULE CARD (fma_passes.ml — grep "MODULE CARD" for the full set)
 * ROLE: The FMA rewrite family; closes the FMA-count gap vs FFTW.
 * PIPELINE: simplify passes -> this cascade -> schedule -> emit
 * PUBLIC SURFACE (measured): zero direct Fma_passes.X references —
 * callers use the Algsimp facade: fma_lift, factor_const_muls,
 * multi_use_fma_lift, fma_addend_factor, flatten_fma_mul_addend
 * (gen_main runs the cascade inline; pipeline.ml is the shared copy).
 * DEPS: Simplify via include (chain re-exported onward); Expr(11).
 * ENV: VFFT_FMA_MULTIUSE.
 * ------------------------------------------------------------------
 *)

include Simplify
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

