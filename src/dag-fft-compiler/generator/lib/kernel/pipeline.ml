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
 * - Annotate, BB — opt-in alternates production handles in
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
 *
 * ------------------------------------------------------------------
 * MODULE CARD (pipeline.ml — grep "MODULE CARD" for the full set)
 * ROLE: The shared hash-cons -> pass-cascade -> spill_info recipe as
 * one function, prepare_codelet.
 * PIPELINE: math-layer raw assignments -> prepare_codelet -> emit
 * PUBLIC SURFACE (measured): codelet_oop(2): prepare_codelet,
 * prepared.
 * DEPS: Algsimp(26), Dft(6), Emit_c(2), Expr(2).
 * ENV: VFFT_NO_SUBDEDUP, VFFT_DEEP_COLLECT (collect_m and the FMA
 * knobs are read inside the passes themselves).
 * GOTCHA 1: caller MUST run Ir.reset first — the reset is
 * deliberately outside this function (drivers differ on when).
 * GOTCHA 2: gen_main.run still carries its own INLINE copy of this
 * exact cascade; the two must be kept in lockstep until unified —
 * a pass added or reordered in one and not the other silently
 * diverges the oop family from the in-place families.
 * ------------------------------------------------------------------
 *)

(* Result of pipeline preparation.
 *
 * Both gen_radix and codelet_oop consume this. emit_c then takes the
 * assigns + spill_info to build its scheduled emission. *)
type prepared =
  { assigns : (Expr.elem_ref * Ir.t) list
  ; (* spill_info is None when monolithic, Some when the math layer
     produced spill markers (R≥25 CT for n1, or should_spill for t1). *)
    spill_info : Algsimp.spill_info option
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
 *     Dft_select.needs_reassoc n
 *   - aggressive: true for Direct primes (enables factor_common_muls
 *     etc.), false for CT-decomposed codelets
 *   - apply_fma_lift_override: per gen_radix's VFFT_FORCE_FMA_LIFT /
 *     VFFT_DISABLE_FMA_LIFT, only honored when the algorithm is not
 *     Split_radix
 *   - fuse: production default is 0 (every PASS 1→2 value spills and
 *     reloads, no register retention across pass boundary)
 *
 * Returns prepared { assigns; spill_info }. ─ *)
(* ── M11: the RECIPE — per-caller cascade arms, explicit ──
 * The two arms that historically existed ONLY in gen_main's inline copy
 * travel as recipe fields so the unification is byte-identical per
 * route: the main driver enables both; the family callers (c2c_split,
 * cascade_z) keep their historical cascade by using default_recipe.
 * Whether those routes SHOULD adopt the arms (and the VFFT_FORCE_REASSOC
 * env override) is M11b — an owner decision needing a race, not a diff. *)
type dup_ctx =
  { uarch : Uarch.t (* SR schedule for the dup probe placement *)
  ; barrier_sink : (int, unit) Hashtbl.t (* the driver's Scratch.dup_barrier_tags *)
  }

type recipe =
  { butterfly_share : bool
    (* enable the env-gated (VFFT_BUTTERFLY_SHARE=1) butterfly_share_mul arm *)
  ; dup : dup_ctx option
    (* enable the env-gated (VFFT_DUP=1) selective-duplication arm *)
  }

let default_recipe = { butterfly_share = false; dup = None }

let prepare_codelet
      ~(recipe : recipe)
      ~(raw_assigns : (Expr.elem_ref * Expr.expr) list)
      ~(spill_markers_raw : Dft.spill_marker list)
      ~(spill_ct : (int * int) option)
      ~(reassoc : bool)
      ~(aggressive : bool)
      ~(algorithm : Dft_select.algorithm)
      ~(force_fma_lift : bool)
      ~(disable_fma_lift : bool)
      ~(build_spill_info : bool)
      ~(fuse : int)
  : prepared
  =
  (* Hash-cons. CRITICAL: caller must have run Ir.reset () before
     this point. The reset clears the global hash-cons table; without
     it, prior generations leak tags into our DAG and the remap chain
     can resolve to dead nodes from a prior call. We don't reset here
     because gen_radix and codelet_oop differ on when they want the
     reset (gen_radix resets before its math-layer expansion; codelet_oop
     resets at the same point). Keeping the reset outside this function
     preserves both call patterns without behavior change. *)
  let simplified = Ir.of_assignments ~reassoc raw_assigns in
  let deduped_pre =
    (if Sys.getenv_opt "VFFT_NO_SUBDEDUP" = Some "1"
     then fun x -> x
     else Simplify.dedup_sub_pairs)
      simplified
  in
  (* Aggressive prime-only passes. For Direct primes (aggressive=true),
     factor_common_muls / factor_by_atom recognize Winograd structure:
       c·x_a + c·x_b → c·(x_a + x_b)
       c1·x + c2·x + c3·x → (c1+c2+c3)·x
     For CT codelets (aggressive=false), these short-circuit and return
     input unchanged — they would destroy Cmul sharing if applied.

     Mirror of gen_radix.ml lines 281-287. *)
  let factored = Simplify.factor_common_muls ~aggressive deduped_pre in
  let factored = Simplify.factor_by_atom ~aggressive factored in
  let factored =
    (if Sys.getenv_opt "VFFT_NO_SUBDEDUP" = Some "1"
     then fun x -> x
     else Simplify.dedup_sub_pairs)
      factored
  in
  (* collect_m: opt-in via VFFT_COLLECT_M=1. Default off in gen_radix.
     Falls through to identity when the env var is unset. Placed after
     dedup_sub_pairs (sees canonicalized form) and before fma_lift
     (Mul nodes it introduces are visible to FMA absorption). *)
  let factored = Simplify.collect_m factored in
  (* deep_collect: opt-in via VFFT_DEEP_COLLECT=1. Default off.
     Fixpoint loop combining deep_collect + collect_m. *)
  let factored =
    if Sys.getenv_opt "VFFT_DEEP_COLLECT" = Some "1"
    then (
      let max_iters = 5 in
      let rec loop n cur =
        if n = 0
        then cur
        else (
          let next = Simplify.deep_collect cur in
          let next = Simplify.collect_m next in
          let same =
            try
              List.for_all2 (fun (_, a) (_, b) -> a.Ir.tag = b.Ir.tag) cur next
            with
            | Invalid_argument _ -> false
          in
          if same then cur else loop (n - 1) next)
      in
      loop max_iters factored)
    else factored
  in
  (* share_subsums: aggressive-only. For CT codelets (is_direct=false
     for our flow), production calls share_subsums with aggressive=false
     which is a no-op. We match exactly. *)
  let is_direct = aggressive in
  let shared =
    if is_direct then factored else Simplify.share_subsums ~aggressive factored
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
    | Dft_select.Direct -> true
    | Dft_select.Cooley_Tukey _ -> true
    | Dft_select.Split_radix -> false
  in
  let apply_fma_lift = (fma_lift_safe || force_fma_lift) && not disable_fma_lift in
  (* Capture pre-cascade frozen_tags from spill markers. Must run
     lift_spill_markers BEFORE fma_lift so the marker tags reference
     nodes that fma_lift can leave unchanged via the frozen guard. *)
  let frozen_tags : (int, unit) Hashtbl.t option =
    if apply_fma_lift && spill_markers_raw <> []
    then (
      let pre_markers = Algsimp.lift_spill_markers ~reassoc spill_markers_raw in
      let tbl = Hashtbl.create 64 in
      List.iter
        (fun (m : Algsimp.spill_tag_marker) ->
           Hashtbl.replace tbl m.re_tag ();
           Hashtbl.replace tbl m.im_tag ())
        pre_markers;
      Some tbl)
    else None
  in
  let extend_frozen (remap : (int, int) Hashtbl.t) =
    match frozen_tags with
    | None -> ()
    | Some tbl -> Hashtbl.iter (fun _old_t new_t -> Hashtbl.replace tbl new_t ()) remap
  in
  let deduped =
    if apply_fma_lift
    then Fma_passes.fma_lift ?frozen_tags:(Some frozen_tags) post_trans
    else post_trans
  in
  (* The 8-remap cascade. CAPTURE each remap individually — the spill
     marker post-cascade remap_tag chain needs them in order.
     extend_frozen alone keeps the cascade self-consistent but doesn't
     tell the marker where its tag MOVED TO. *)
  let empty_remap () : (int, int) Hashtbl.t = Hashtbl.create 0 in
  let step pass a =
    if apply_fma_lift
    then (
      let a', remap = pass ?frozen_tags:(Some frozen_tags) a in
      extend_frozen remap;
      a', remap)
    else a, empty_remap ()
  in
  let deduped, factor_tag_remap = step Fma_passes.factor_const_muls deduped in
  let deduped, mfl_tag_remap = step Fma_passes.multi_use_fma_lift deduped in
  let deduped, fma_addend_remap = step Fma_passes.fma_addend_factor deduped in
  let deduped, mfl2_tag_remap = step Fma_passes.multi_use_fma_lift deduped in
  let deduped, fma_addend_remap2 = step Fma_passes.fma_addend_factor deduped in
  let deduped, mfl3_tag_remap = step Fma_passes.multi_use_fma_lift deduped in
  let deduped, fma_addend_remap3 = step Fma_passes.fma_addend_factor deduped in
  let deduped, mfl4_tag_remap = step Fma_passes.multi_use_fma_lift deduped in
  let deduped, bsm_tag_remap =
    if recipe.butterfly_share
       && apply_fma_lift
       && Sys.getenv_opt "VFFT_BUTTERFLY_SHARE" = Some "1"
    then (
      let a', remap = Algsimp.butterfly_share_mul ~frozen_tags deduped in
      extend_frozen remap;
      a', remap)
    else deduped, empty_remap ()
  in
  let deduped, _flatten_tag_remap = step Fma_passes.flatten_fma_mul_addend deduped in
  (* M11: the selective-duplication arm (doc 65 §8) — moved VERBATIM
     from gen_main's inline copy; fires only for callers whose recipe
     carries a dup_ctx AND VFFT_DUP=1.  MUST stay the final DAG
     transform (clones bypass hashcons); skipped when spill markers
     are present (dup carries no marker remap). *)
  let deduped =
    match recipe.dup with
    | None -> deduped
    | Some ctx ->
      (match Sys.getenv_opt "VFFT_DUP" with
       | Some "1" when spill_markers_raw = [] ->
      let geti k d =
        match Sys.getenv_opt k with
        | Some v ->
          (try int_of_string v with
           | _ -> d)
        | None -> d
      in
      let sched_of asg = List.map snd (Schedule.su_schedule ctx.uarch asg) in
      (* Freeze the PRE-dup SR schedule. The rebuild re-tags ancestor
       * cones (and hashcons can MERGE a rebuilt kind into an existing
       * node), so naive tag-chasing yields duplicate/incomplete order
       * files the injector refuses. Robust composition: Kahn-sort the
       * FINAL dag with priority = the node's pre-image position in
       * sched0 (min over merged pre-images; clones sit just before
       * their consumer; fresh no-preimage nodes inherit min-user
       * position). Always complete, always topologically legal,
       * equal to the frozen probe placement wherever possible. *)
      let sched0 = sched_of deduped in
      let a, btags, remap, inserts =
        Algsimp.duplicate_uncse
          ~span_s:(geti "VFFT_DUP_S" 30)
          ~cap:(geti "VFFT_DUP_CAP" 16)
          ~maxcost:(geti "VFFT_DUP_COST" 1)
          ~schedule:sched_of
          deduped
      in
      Hashtbl.iter (fun t () -> Hashtbl.replace ctx.barrier_sink t ()) btags;
      let chase t =
        let rec go t k =
          if k > 64
          then t
          else (
            match Hashtbl.find_opt remap t with
            | Some t' when t' <> t -> go t' (k + 1)
            | _ -> t)
        in
        go t 0
      in
      let pos : (int, float) Hashtbl.t = Hashtbl.create 1024 in
      List.iteri
        (fun i (n : Ir.t) ->
           let f = chase n.tag in
           let p = float_of_int i in
           match Hashtbl.find_opt pos f with
           | Some q when q <= p -> ()
           | _ -> Hashtbl.replace pos f p)
        sched0;
      let roots = List.map snd a in
      let nodes = Ir.topo_sort_reachable roots in
      let usersd : (int, Ir.t list) Hashtbl.t = Hashtbl.create 1024 in
      List.iter
        (fun (n : Ir.t) ->
           List.iter
             (fun (p : Ir.t) ->
                let l =
                  try Hashtbl.find usersd p.tag with
                  | Not_found -> []
                in
                Hashtbl.replace usersd p.tag (n :: l))
             (Ir.preds n))
        nodes;
      (* Pin each clone to its consumer's DECLARED ANCHOR slot: SU
       * interleaves sibling fma chains, so an inner chain node's own
       * sched0 slot is many positions before the line it inlines
       * into (measured: 224/217/210 for a line at 242) — pinning
       * there hoists the clone across other groups. The probe pins
       * to the LINE; the anchor (outermost declared node of the
       * single-use chain, in the FINAL dag) is the line. Tiny
       * ascending offsets keep application (span-desc) order within
       * a block, matching the probe's emission order. *)
      let roots_set : (int, unit) Hashtbl.t = Hashtbl.create 64 in
      List.iter
        (fun ((_, v) : Expr.elem_ref * Ir.t) -> Hashtbl.replace roots_set v.tag ())
        a;
      let node_by : (int, Ir.t) Hashtbl.t = Hashtbl.create 1024 in
      List.iter (fun (n : Ir.t) -> Hashtbl.replace node_by n.tag n) nodes;
      let declared_f (x : Ir.t) =
        List.length
          (try Hashtbl.find usersd x.tag with
           | Not_found -> [])
        >= 2
        || Hashtbl.mem roots_set x.tag
        || Hashtbl.mem btags x.tag
      in
      let rec anch (x : Ir.t) : Ir.t =
        if declared_f x
        then x
        else (
          match Hashtbl.find_opt usersd x.tag with
          | Some [ w ] -> anch w
          | _ -> x)
      in
      List.iteri
        (fun i (c, u) ->
           let target =
             match Hashtbl.find_opt node_by (chase u) with
             | Some nd -> (anch nd).tag
             | None -> chase u
           in
           if Sys.getenv_opt "VFFT_DUP_TRACE" <> None
           then
             Printf.eprintf
               "PIN clone t%d: u=t%d final=t%d anchor=t%d pos=%s\n"
               c
               u
               (chase u)
               target
               (match Hashtbl.find_opt pos target with
                | Some p -> string_of_float p
                | None -> "MISS");
           match Hashtbl.find_opt pos target with
           | Some p -> Hashtbl.replace pos c (p -. 0.5 +. (float_of_int i *. 1e-4))
           | None -> ())
        inserts;
      List.iter
        (fun (n : Ir.t) ->
           if not (Hashtbl.mem pos n.tag)
           then (
             let up =
               List.fold_left
                 (fun acc (u : Ir.t) ->
                    match Hashtbl.find_opt pos u.tag with
                    | Some p -> min acc p
                    | None -> acc)
                 infinity
                 (try Hashtbl.find usersd n.tag with
                  | Not_found -> [])
             in
             Hashtbl.replace pos n.tag (if up = infinity then 1e9 else up -. 0.25)))
        (List.rev nodes);
      let indeg : (int, int) Hashtbl.t = Hashtbl.create 1024 in
      List.iter
        (fun (n : Ir.t) ->
           Hashtbl.replace indeg n.tag (List.length (Ir.preds n)))
        nodes;
      let module PQ = Set.Make (struct
          type t = float * int

          let compare = compare
        end)
      in
      let node_of : (int, Ir.t) Hashtbl.t = Hashtbl.create 1024 in
      List.iter (fun (n : Ir.t) -> Hashtbl.replace node_of n.tag n) nodes;
      let ready = ref PQ.empty in
      List.iter
        (fun (n : Ir.t) ->
           if Hashtbl.find indeg n.tag = 0
           then ready := PQ.add (Hashtbl.find pos n.tag, n.tag) !ready)
        nodes;
      let buf = Buffer.create 4096 in
      let count = ref 0 in
      while not (PQ.is_empty !ready) do
        let ((_, t) as m) = PQ.min_elt !ready in
        ready := PQ.remove m !ready;
        Buffer.add_string buf (string_of_int t ^ "\n");
        incr count;
        let n = Hashtbl.find node_of t in
        List.iter
          (fun (u : Ir.t) ->
             let d = Hashtbl.find indeg u.tag - 1 in
             Hashtbl.replace indeg u.tag d;
             if d = 0 then ready := PQ.add (Hashtbl.find pos u.tag, u.tag) !ready)
          (try Hashtbl.find usersd n.tag with
           | Not_found -> [])
      done;
      let ord_file = Filename.temp_file "vfft_dup_order" ".txt" in
      let oc = open_out ord_file in
      output_string oc (Buffer.contents buf);
      close_out oc;
      Unix.putenv "VFFT_SCHED_ORDER" ord_file;
      Printf.eprintf
        "duplicate_uncse: %d clones, pinned %d/%d nodes\n%!"
        (Hashtbl.length btags)
        !count
        (List.length nodes);
      a
       | Some "1" ->
         prerr_endline "VFFT_DUP: skipped (spill markers present)";
         deduped
       | _ -> deduped)
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
    if build_spill_info && spill_markers_raw <> []
    then (
      let raw_markers = Algsimp.lift_spill_markers ~reassoc spill_markers_raw in
      let remap_tag t =
        let walk tbl t =
          match Hashtbl.find_opt tbl t with
          | Some t' -> t'
          | None -> t
        in
        let t = walk factor_tag_remap t in
        let t = walk mfl_tag_remap t in
        let t = walk fma_addend_remap t in
        let t = walk mfl2_tag_remap t in
        let t = walk fma_addend_remap2 t in
        let t = walk mfl3_tag_remap t in
        let t = walk fma_addend_remap3 t in
        let t = walk mfl4_tag_remap t in
        let t = walk bsm_tag_remap t in
        t
      in
      let tag_markers =
        List.map
          (fun (m : Algsimp.spill_tag_marker) ->
             { m with re_tag = remap_tag m.re_tag; im_tag = remap_tag m.im_tag })
          raw_markers
      in
      Some (Algsimp.make_spill_info ?ct:spill_ct ~fuse tag_markers))
    else None
  in
  { assigns; spill_info }
;;
