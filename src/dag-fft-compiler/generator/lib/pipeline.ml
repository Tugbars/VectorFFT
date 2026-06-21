(* ═══════════════════════════════════════════════════════════════════════
 * PIPELINE.ML — Shared codelet preparation pipeline
 *
 * Single source of truth for the cascade + spill construction shared
 * between gen_radix.ml's --strided path and codelet_oop.ml's butterfly
 * body emission. Before this module existed, both callers had inline
 * copies of the same logic; drift was nearly impossible to avoid (the
 * 8-step remap_tag chain that wires spill markers to their post-cascade
 * tags has a silent failure mode — if you skip a remap, spill stores
 * point to dead nodes and PASS 2 reloads garbage).
 *
 * WHAT THIS MODULE COVERS
 * -----------------------
 * 1. Hash-cons the raw assignments via `of_assignments ~reassoc`
 * 2. Run the dedup_sub_pairs / factor_common_muls / factor_by_atom /
 *    share_subsums / collect_m / deep_collect / fma_lift / cascade
 *    in the same order gen_radix.ml does
 * 3. Capture frozen_tags from pre-cascade spill markers and thread them
 *    through every pass that can rewrite operand structure
 * 4. Capture per-pass remaps individually (NOT just for extend_frozen —
 *    they also drive the post-cascade marker remap_tag chain)
 * 5. Re-lift spill markers post-cascade, apply 8-remap chain to recover
 *    the final tag each marker should reference, build spill_info
 *
 * WHAT THIS MODULE DOES NOT COVER
 * -------------------------------
 * - Emission (per-caller; emit_c.ml for strided, codelet_oop for OOP)
 * - Scheduling (classify_passes, cluster_of_pass2_node, su_schedule_subset)
 *   — emission-time decisions per caller
 * - reachable_nodes / compute_inline_set — cheap to compute per-caller
 *   so we let each emission site do it. Could be lifted up later if a
 *   third caller appears.
 * - Annotate, Bisection, BB — opt-in alternates production handles in
 *   emit_c.ml
 * - Regalloc — log3 AVX-512 R≤32 only, gated in emit_c.ml
 *
 * AGGRESSIVE MODE
 * ---------------
 * The `aggressive` parameter mirrors gen_radix's flag: ON for Direct
 * primes (R=3/5/7/11/13/17 — Winograd structure), OFF for Cooley-Tukey
 * decomposed codelets (our Bailey R=8/16/32/64).
 *
 * For aggressive=true, factor_common_muls / factor_by_atom /
 * share_subsums fire and reveal Winograd structure. The transpose FP
 * loop ALSO has a `aggressive && not is_direct` guard, which means it
 * is unreachable in current gen_radix (aggressive ↔ is_direct). We
 * preserve that dead path for fidelity to the production flow but it
 * never executes.
 *
 * For aggressive=false (our case): factor_common_muls /
 * factor_by_atom / share_subsums short-circuit at their aggressive
 * guards and return their input unchanged. Calling them with
 * aggressive=false is a no-op but matches gen_radix's call sequence
 * byte-for-byte, so codelets generated through this pipeline are
 * identical to what gen_radix produces.
 *
 * FROZEN_TAGS POLICY
 * ------------------
 * frozen_tags is a (int → unit) Hashtbl of tags that must survive the
 * cascade unchanged. It's populated from the pre-cascade spill markers
 * (lift_spill_markers on the raw spill_markers list). Each pass that
 * accepts frozen_tags returns a (assigns, remap) pair; we call
 * extend_frozen on each remap so that values introduced BY a pass
 * (which previously-frozen markers now alias through the remap) are
 * also protected from subsequent passes.
 *
 * The 8 remaps are captured INDIVIDUALLY (not just composed into
 * frozen_tags) because the spill marker remap_tag chain post-cascade
 * needs them in order. extend_frozen alone is necessary but not
 * sufficient — see the comment in `prepare_codelet` below.
 * ═══════════════════════════════════════════════════════════════════════ *)

(* Result of pipeline preparation.
 *
 * Both gen_radix and codelet_oop consume this. emit_c then takes the
 * assigns + spill_info to build its scheduled emission. *)
type prepared = {
  assigns : (Expr.elem_ref * Algsimp.t) list;
  (* spill_info is None when monolithic, Some when the math layer
     produced spill markers (R≥25 CT for n1, or should_spill for t1). *)
  spill_info : Emit_c.spill_info option;
}

(* ─── prepare_codelet ────────────────────────────────────────────────
 * Single shared entry point. Mirrors gen_radix.ml lines 250-598 exactly,
 * with the aggressive path expressed via a parameter rather than a
 * boolean derived in-place.
 *
 * Inputs:
 *   - raw_assigns: output of dft_expand / dft_expand_n1_blocked /
 *     dft_expand_twiddled_spill from the math layer
 *   - spill_markers_raw: empty list when monolithic, populated when
 *     the math layer chose a blocked variant
 *   - spill_ct: Some (n1, n2) when CT-blocked, None otherwise
 *   - reassoc: whether to allow reassociation during hash-cons; from
 *     Dft.needs_reassoc n
 *   - aggressive: true for Direct primes (enables factor_common_muls
 *     etc.), false for CT-decomposed codelets
 *   - apply_fma_lift_override: per gen_radix's VFFT_FORCE_FMA_LIFT /
 *     VFFT_DISABLE_FMA_LIFT, only honored when the algorithm is not
 *     Split_radix
 *   - fuse: production default is 0 (every PASS 1→2 value spills and
 *     reloads, no register retention across pass boundary)
 *
 * Returns prepared { assigns; spill_info }. ─ *)
let prepare_codelet ~(raw_assigns : (Expr.elem_ref * Expr.expr) list)
    ~(spill_markers_raw : Dft.spill_marker list)
    ~(spill_ct : (int * int) option) ~(reassoc : bool) ~(aggressive : bool)
    ~(algorithm : Dft.algorithm) ~(force_fma_lift : bool)
    ~(disable_fma_lift : bool) ~(build_spill_info : bool) ~(fuse : int) :
    prepared =
  (* Hash-cons. CRITICAL: caller must have run Algsimp.reset () before
     this point. The reset clears the global hash-cons table; without
     it, prior generations leak tags into our DAG and the remap chain
     can resolve to dead nodes from a prior call. We don't reset here
     because gen_radix and codelet_oop differ on when they want the
     reset (gen_radix resets before its math-layer expansion; codelet_oop
     resets at the same point). Keeping the reset outside this function
     preserves both call patterns without behavior change. *)
  let simplified = Algsimp.of_assignments ~reassoc raw_assigns in
  let deduped_pre =
    (if Sys.getenv_opt "VFFT_NO_SUBDEDUP" = Some "1" then fun x -> x
     else Algsimp.dedup_sub_pairs)
      simplified
  in

  (* Aggressive prime-only passes. For Direct primes (aggressive=true),
     factor_common_muls / factor_by_atom recognize Winograd structure:
       c·x_a + c·x_b → c·(x_a + x_b)
       c1·x + c2·x + c3·x → (c1+c2+c3)·x
     For CT codelets (aggressive=false), these short-circuit and return
     input unchanged — they would destroy Cmul sharing if applied.

     Mirror of gen_radix.ml lines 281-287. *)
  let factored = Algsimp.factor_common_muls ~aggressive deduped_pre in
  let factored = Algsimp.factor_by_atom ~aggressive factored in
  let factored =
    (if Sys.getenv_opt "VFFT_NO_SUBDEDUP" = Some "1" then fun x -> x
     else Algsimp.dedup_sub_pairs)
      factored
  in

  (* collect_m: opt-in via VFFT_COLLECT_M=1. Default off in gen_radix.
     Falls through to identity when the env var is unset. Placed after
     dedup_sub_pairs (sees canonicalized form) and before fma_lift
     (Mul nodes it introduces are visible to FMA absorption). *)
  let factored = Algsimp.collect_m factored in

  (* deep_collect: opt-in via VFFT_DEEP_COLLECT=1. Default off.
     Fixpoint loop combining deep_collect + collect_m. *)
  let factored =
    if Sys.getenv_opt "VFFT_DEEP_COLLECT" = Some "1" then begin
      let max_iters = 5 in
      let rec loop n cur =
        if n = 0 then cur
        else
          let next = Algsimp.deep_collect cur in
          let next = Algsimp.collect_m next in
          let same =
            try
              List.for_all2
                (fun (_, a) (_, b) -> a.Algsimp.tag = b.Algsimp.tag)
                cur next
            with Invalid_argument _ -> false
          in
          if same then cur else loop (n - 1) next
      in
      loop max_iters factored
    end
    else factored
  in

  (* share_subsums: aggressive-only. For CT codelets (is_direct=false
     for our flow), production calls share_subsums with aggressive=false
     which is a no-op. We match exactly. *)
  let is_direct = aggressive in
  let shared =
    if is_direct then factored else Algsimp.share_subsums ~aggressive factored
  in

  (* Transpose FP loop. Production guards this with
     `aggressive && not has_cmul && not is_direct`. Since aggressive
     ↔ is_direct in the current pipeline, this conjunction is always
     false and the loop is dead code in gen_radix's current state. We
     preserve the structural fidelity to make any future change to the
     aggressive flag visible here too. *)
  (* Transpose fixed-point loop removed. It was gated on
     `aggressive && not is_direct`, which is always false in the current
     pipeline (aggressive is equivalent to is_direct), so the loop never
     ran. With it goes its only consumer of the legacy op-counter, so
     post_trans is just the shared DAG. *)
  let post_trans = shared in

  (* FMA lift gating per doc 56. *)
  let fma_lift_safe =
    match algorithm with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> true
    | Dft.Split_radix -> false
  in
  let apply_fma_lift =
    (fma_lift_safe || force_fma_lift) && not disable_fma_lift
  in

  (* Capture pre-cascade frozen_tags from spill markers. Must run
     lift_spill_markers BEFORE fma_lift so the marker tags reference
     nodes that fma_lift can leave unchanged via the frozen guard. *)
  let frozen_tags : (int, unit) Hashtbl.t option =
    if apply_fma_lift && spill_markers_raw <> [] then begin
      let pre_markers = Algsimp.lift_spill_markers ~reassoc spill_markers_raw in
      let tbl = Hashtbl.create 64 in
      List.iter
        (fun (m : Algsimp.spill_tag_marker) ->
          Hashtbl.replace tbl m.re_tag ();
          Hashtbl.replace tbl m.im_tag ())
        pre_markers;
      Some tbl
    end
    else None
  in
  let extend_frozen (remap : (int, int) Hashtbl.t) =
    match frozen_tags with
    | None -> ()
    | Some tbl ->
        Hashtbl.iter (fun _old_t new_t -> Hashtbl.replace tbl new_t ()) remap
  in

  let deduped =
    if apply_fma_lift then
      Algsimp.fma_lift ?frozen_tags:(Some frozen_tags) post_trans
    else post_trans
  in

  (* The 8-remap cascade. CAPTURE each remap individually — the spill
     marker post-cascade remap_tag chain needs them in order.
     extend_frozen alone keeps the cascade self-consistent but doesn't
     tell the marker where its tag MOVED TO. *)
  let empty_remap () : (int, int) Hashtbl.t = Hashtbl.create 0 in
  let step pass a =
    if apply_fma_lift then (
      let a', remap = pass ?frozen_tags:(Some frozen_tags) a in
      extend_frozen remap;
      (a', remap))
    else (a, empty_remap ())
  in
  let deduped, factor_tag_remap = step Algsimp.factor_const_muls deduped in
  let deduped, mfl_tag_remap = step Algsimp.multi_use_fma_lift deduped in
  let deduped, fma_addend_remap = step Algsimp.fma_addend_factor deduped in
  let deduped, mfl2_tag_remap = step Algsimp.multi_use_fma_lift deduped in
  let deduped, fma_addend_remap2 = step Algsimp.fma_addend_factor deduped in
  let deduped, mfl3_tag_remap = step Algsimp.multi_use_fma_lift deduped in
  let deduped, fma_addend_remap3 = step Algsimp.fma_addend_factor deduped in
  let deduped, mfl4_tag_remap = step Algsimp.multi_use_fma_lift deduped in
  let deduped, _flatten_tag_remap =
    step Algsimp.flatten_fma_mul_addend deduped
  in

  let assigns = deduped in

  (* Build spill_info post-cascade. The remap chain walks each marker
     tag through the 8 remaps in cascade order. flatten_tag_remap is
     intentionally EXCLUDED: flatten produces 2-FMA chains that absorb
     previously-standalone Muls, and a spill marker whose tag is one
     of those Muls should remain a separate spillable value, NOT get
     remapped to the post-flatten Fma (which would mean reloading the
     Mul means re-computing the chain). Production excludes it too —
     see gen_radix.ml line 588 (last walk is mfl4, not flatten). *)
  let spill_info =
    if build_spill_info && spill_markers_raw <> [] then
      let raw_markers = Algsimp.lift_spill_markers ~reassoc spill_markers_raw in
      let remap_tag t =
        let walk tbl t =
          match Hashtbl.find_opt tbl t with Some t' -> t' | None -> t
        in
        let t = walk factor_tag_remap t in
        let t = walk mfl_tag_remap t in
        let t = walk fma_addend_remap t in
        let t = walk mfl2_tag_remap t in
        let t = walk fma_addend_remap2 t in
        let t = walk mfl3_tag_remap t in
        let t = walk fma_addend_remap3 t in
        let t = walk mfl4_tag_remap t in
        t
      in
      let tag_markers =
        List.map
          (fun (m : Algsimp.spill_tag_marker) ->
            { m with re_tag = remap_tag m.re_tag; im_tag = remap_tag m.im_tag })
          raw_markers
      in
      Some (Emit_c.make_spill_info ?ct:spill_ct ~fuse tag_markers)
    else None
  in

  { assigns; spill_info }
