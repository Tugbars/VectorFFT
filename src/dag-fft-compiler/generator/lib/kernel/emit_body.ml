(* emit_body.ml — M8.2: renamed from emit_c.ml (§9 #23 Emit_body — the
 * schedule / spill / regalloc / render meeting point).  The M8 family
 * moves (Real, C2c_split) will thin this file toward the irreducible
 * core; until each family's arms move, emit_codelet remains the
 * C-emission driver for every in-place codelet family (n1, t1 dit /
 * dif / t1s / log3, twidsq, strided, the whole r2c / c2r / trig
 * signature zoo).
 *
 * Top of the emit chain (Emit_render < Emit_body). Given the
 * simplified assignment DAG plus a scheduler choice and the per-emission
 * cfg/sc the driver builds (M6), emit_codelet:
 *
 *   1. resolves the signature family (which pointer ABI the function
 *      gets) from cfg via Abi (M4);
 *   2. schedules the DAG — Topological or SU (optionally Goodman-Hsu
 *      pressure-switched, optionally Bb branch-and-bound refined),
 *      plus Annotated variants — monolithic or split per spill pass
 *      via classify_passes / cluster_split_schedule;
 *   3. optionally runs Regalloc (M-project pin / fence, opt-in via env)
 *      and the selective-pin / const-hoist refinements;
 *   4. renders: k-loop over the batch, per-node declarations through
 *      Emit_render, spill stores / reloads at pass seams, and the
 *      arbitrary-K tail (bulk vector loop + masked or scalar remainder
 *      via Isa.ls_mode) unless VFFT_NO_ANYK_TAIL disables it;
 *   5. stamps codelet_metadata + provenance so every emitted file
 *      records the exact command and env that produced it.
 *
 * Inputs / outputs are split-complex with K-stride layout: element j's
 * real component sits at in_re[j*K + k]; twiddles use the same layout
 * (or broadcast forms for t1s / t1p). The k-loop steps by the ISA
 * vector width.
 *
 * emit_codelet is deliberately one large function: it is the single
 * place where scheduling, spill structure, regalloc and rendering
 * decisions meet, and every codelet family shares that one meeting
 * point rather than duplicating it.
 * ------------------------------------------------------------------
 * MODULE CARD (emit_c.ml — grep "MODULE CARD" for the full set)
 * ROLE: Facade of the emit chain + emit_codelet + the strided
 * load/store preamble helpers.
 * PIPELINE: gen_main (or codelet_oop for the oop family) -> here ->
 * C text on stdout.
 * PUBLIC SURFACE (measured; grep counts incl. comments):
 * codelet_oop(46), gen_main(23), pipeline(2), dft_r2c(1), regalloc(1),
 * gen_set(1 per driver). Hot names: emit_codelet, Algsimp.spill_info,
 * make_spill_info, scheduler, the Emit_state mode refs re-exported
 * through the chain.
 * DEPS: Emit_render chain via include; Algsimp (open), Schedule(7),
 * Regalloc(21), Isa(20), Annotate(6), Bb(3), Expr(7).
 * ENV: VFFT_NO_REGALLOC, VFFT_PIN_FORCE, VFFT_FORCE_FENCE,
 * VFFT_PEAK_LIVE, VFFT_DISABLE_SELECTIVE_PIN, VFFT_NO_ANYK_TAIL.
 * ------------------------------------------------------------------
 *)

(* Layering: Emit_state (mode refs) < Emit_render (renderers) < this
 * file (emit_codelet, the driver). `include` re-exports both layers
 * so every external Emit_c.X reference compiles unchanged. The open
 * of Algsimp precedes the include so names defined in the render
 * chain (e.g. its topo_sort_reachable) keep shadowing Algsimp's,
 * exactly as the pre-split single file resolved them. *)
(* M1: `open Algsimp` removed — after requalification emit_c uses nothing of
   Algsimp bare; the historical open-order shadow dance (see the header note
   above) is therefore moot, and topo_sort_reachable is qualified at all 14
   sites anyway. *)
open Ir  (* M1: names formerly re-exported through the chain *)
(* M6.2: `open Emit_state` removed — emit_c reads NO globals: config arrives
   as ~cfg, scratch as ~sc, both per-emission from the driver. *)
open Emit_render  (* M1: was `include`; AFTER `open Algsimp` so the render chain's names keep shadowing Algsimp's (topo_sort_reachable also pre-qualified at all 14 sites) *)

(* M8.3: FAMILY HOOKS — the seam between the shared engine and the
   feature modules (§9: Emit_body serves C2c_split and Real).  Each field
   names a POSITION in the emission sequence; a family passes closures
   whose bodies are its moved arms, capturing cfg/isa/radix family-side.
   The kernel keeps the cfg dispatch tests and fails LOUDLY if a
   family-owned arm is reached without its hook — never silently emits
   wrong bytes. *)
type family_hooks =
  { strided_prologue : (Buffer.t -> unit) option
    (* after "(void)tw_*": the r2c/c2r rio alias block *)
  ; strided_locals : (Buffer.t -> unit) option
    (* after the b-loop opens: the c2r _hx lane decls *)
  ; strided_load : (Buffer.t -> unit) option
    (* the load-lattice family arm (c2r merge prologue) *)
  ; strided_store : (Buffer.t -> unit) option
    (* the store-lattice family arm (r2c fused conjugate split) *)
  ; trailer : (Buffer.t -> unit) option
    (* after the tail region: the hc_ranged pointer-advance closer *)
  }

let no_hooks =
  { strided_prologue = None
  ; strided_locals = None
  ; strided_load = None
  ; strided_store = None
  ; trailer = None
  }

let emit_codelet
      ?(hooks = no_hooks)
  ~(sc : Emit_render.Scratch.t)
  ~(cfg : Emit_render.Cfg.t)
      ?(in_place = false)
      ?(t1s = false)
      ?(twidsq = false)
      ?(twidsq_n = 0)
      ?(strided = false)
      ?(radix = 0)
      ?(scheduler = Topological)
      ?(isa = Isa.avx512)
      ?(gh = false)
      ?(bb_budget : float option = None)
      ?(spill : Algsimp.spill_info option = None)
      ?(is_log3 = false)
      (assigns : (Expr.elem_ref * t) list)
      ~(name : string)
  : string
  =
  (* === M2 peak-live diagnostic ===
   *
   * If VFFT_PEAK_LIVE=1, stderr-print peak_live measurements at each
   * scheduling site within this codelet. No effect on emitted C — the
   * output is purely diagnostic. Designed to be a no-op when the env
   * var is unset, so default builds are byte-identical to pre-M2.
   *
   * Gating choice: env var rather than a build flag so we can A/B
   * codelet generation runs from the same binary without rebuilding.
   * The labels include the codelet `name` and the scheduling site
   * identifier so we can correlate output across many codelets. *)
  let peak_live_enabled =
    try Sys.getenv "VFFT_PEAK_LIVE" = "1" with
    | Not_found -> false
  in
  (* === SCHEDULE WISDOM RESOLUTION (per-codelet) ===
   * Explicit VFFT_SCHED_ORDER (exact file for monolithic, prefix for
   * blocked subsets) wins; else VFFT_SCHED_WISDOM/<name> — the codelet
   * symbol is the wisdom key, encoding R, family, direction, and ISA.
   * The injectors in schedule.ml verify each file's #dagsig, refuse
   * stale/incomplete orders (falling back to su), and record every
   * accept/refuse in Schedule.injection_log, spliced into a trailer at
   * the end of this function. The log is reset here so a multi-codelet
   * driver (gen_set) stamps each codelet with only its own events.
   * Default path (neither env set): order_source = None, injectors
   * no-op, output byte-identical. *)
  Schedule.injection_log := [];
  (Schedule.order_source
   := match Sys.getenv_opt "VFFT_SCHED_ORDER" with
      | Some _ as s -> s
      | None ->
        (match Sys.getenv_opt "VFFT_SCHED_WISDOM" with
         | Some dir -> Some (Filename.concat dir name)
         | None -> None));
  (* Captures the maximum per-pass peak_live across all scheduling sites
   * in this codelet. For CT codelets the emitter schedules each pass
   * separately (with on-demand spill reloads), so the true register
   * pressure is the max over passes, NOT the whole-DAG peak. We capture
   * it here from the emitter's actual schedules rather than re-deriving
   * it, because the reload ordering can't be reconstructed from a plain
   * topological sort. Fed to codelet_metadata as the reported peak. *)
  let max_pass_peak = ref 0 in
  let record_peak_live (label : string) (scheduled : t list) =
    let info = Regalloc.peak_live_analysis ~isa ~scheduled in
    if info.peak_live > !max_pass_peak then max_pass_peak := info.peak_live;
    if peak_live_enabled
    then Printf.eprintf "[%s:%s] %s\n" name label (Regalloc.format_live_info info)
  in
  (* Reference both bindings to avoid "unused variable" warnings when
   * VFFT_PEAK_LIVE is not set at compile time (it's a runtime check,
   * so the compiler can't know — but ocaml may still warn). *)
  let _ = record_peak_live in
  (* === M3a register allocation ===
   *
   * If VFFT_USE_REGALLOC=1, run SSA-based linear-scan allocation on
   * each scheduled list and pass the result to render_node_def via
   * a top-level mutable ref. When allocation fits in budget,
   * render_node_def emits
   *   register __m512d tN asm("zmmK") = ...;
   *   asm volatile ("" : "+v"(tN));
   * instead of
   *   const __m512d tN = ...;
   * When allocation overflows, we fall back to default behavior for
   * this pass (stderr warn so the user knows which codelets exceeded
   * the M3a budget).
   *
   * Budget: isa.vec_regs - 4 (28 for AVX-512, 12 for AVX2). The
   * margin leaves room for gcc's ABI / temporary needs.
   *
   * The current_regalloc ref carries the active allocation across the
   * boundary between emit_codelet (which computes it) and
   * render_node_def (which consumes it). It's set per scheduling
   * site — each pass gets its own allocation. The ref is reset to
   * None at exit so leftover allocations don't leak into other call
   * sites.
   *
   * Threading: this is a top-level ref, so concurrent emit_codelet
   * calls in the same process would race. Our codelet generator is
   * single-threaded; documenting the limitation here. *)
  (* === Two-rule policy: pin emission and fence emission ===
   *
   * The previous pin-density gate (and the asm("regN") pinning it
   * controlled) is replaced by a measurement-driven two-rule policy.
   * Empirical basis: a 32-case sweep across radices, ISAs, and codelet
   * kinds (t1 / n1 / log3) showed that the inline-asm scheduling
   * fence — not the register-pin clause — is the actual win
   * mechanism in nearly all codelets. The pin is a narrow-band
   * benefit (log3 on AVX-512 at R≤32) and an active cost in most
   * other cases. See `docs/fence_pin_decomposition.md` for the full
   * decomposition and the data tables.
   *
   * The policy:
   *
   *   pin_enabled (controls `asm("zmmN")` clause emission and the
   *                Regalloc.allocate pass that decides register
   *                assignments):
   *     default = OFF
   *     enable when: kind = log3 AND isa = AVX-512 AND R ≤ 32
   *
   *   fence_enabled (controls `asm volatile("" : "+v"(t))` emission):
   *     default = ON
   *     disable when: kind = n1 AND isa = AVX2 AND R ∈ {8, 16}
   *
   * Env-var escape hatches:
   *   VFFT_NO_REGALLOC=1 — disable BOTH pin and fence (true M-off)
   *   VFFT_PIN_FORCE=1   — force pin on regardless of policy (diagnostic) *)

  (* Detect n1 by scanning the codelet name for "_n1_". The codelet
   * generator uses a stable naming convention: radix<R>_<kind>_..._<isa>
   * where <kind> is "t1" or "n1". Keeps the policy free of new
   * parameter plumbing. *)
  let contains_substring s pat =
    let plen = String.length pat in
    let slen = String.length s in
    let rec aux i =
      if i + plen > slen
      then false
      else if String.sub s i plen = pat
      then true
      else aux (i + 1)
    in
    aux 0
  in
  let is_n1 = contains_substring name "_n1_" in
  let is_avx2 = isa.Isa.vec_regs <= 16 in
  let opt_out =
    try Sys.getenv "VFFT_NO_REGALLOC" = "1" with
    | Not_found -> false
  in
  let force_pin =
    try Sys.getenv "VFFT_PIN_FORCE" = "1" with
    | Not_found -> false
  in
  let force_fence =
    try Sys.getenv "VFFT_FORCE_FENCE" = "1" with
    | Not_found -> false
  in
  (* M-PROJECT DEFAULT FLIPPED TO OFF (2026-06-09).
   * Measured across t1/t1s/log3/n1 x avx2/avx512 x R=4..128 on gcc-13:
   * M-on (pin and/or fence) is net-negative or a tie in every cell. The
   * pin's round-robin register assignment fights gcc-13's coalescing
   * (e.g. log3 R=4: 10 reg-reg copies on an 8-register working set, ~10%
   * runtime cost); the fence both fragments live ranges and DEFEATS
   * operand folding (e.g. t1s: the fence un-folds an embedded {1to8}
   * broadcast into a named register, +9%). The protective function M was
   * built for (forcing FMA contraction, blocking remat on gcc-11) is
   * obsolete: we IR-lift FMAs and gcc-13 fuses on its own.
   *
   * The machinery is kept, not deleted — pin and fence are now OPT-IN:
   *   VFFT_PIN_FORCE=1   → pin (+ fence) on, for deterministic
   *                        cross-compiler asm (the one surviving M
   *                        motivation: same output on gcc-11/13/clang,
   *                        at a measured runtime cost).
   *   VFFT_FORCE_FENCE=1 → fence-only (no pin), diagnostic / A-B.
   *   VFFT_NO_REGALLOC=1 → force everything off (now also the default).
   * The retired auto-pin policy (was: log3+avx512+R<=32; avx2+n1+R>=16)
   * is preserved in git history; do not re-add as a default without a
   * fresh measurement on the deployment compiler. *)
  let regalloc_enabled = (not opt_out) && force_pin in
  let fence_enabled = (not opt_out) && (force_pin || force_fence) in
  (* Set the fence-only emission flag for the non-pinned path.
   *   pin_enabled                  → render_node_def emits pinned form
   *   !pin_enabled && fence_enabled → render_node_def emits fenced form
   *   !pin_enabled && !fence_enabled → render_node_def emits const form *)
  sc.Scratch.fence_only <- fence_enabled && not regalloc_enabled;
  let install_alloc
        (label : string)
        (scheduled : t list)
        (inline_set : (int, unit) Hashtbl.t option)
        (force_last_use : (int, int) Hashtbl.t option)
    =
    if regalloc_enabled
    then (
      match Regalloc.allocate ~isa ~scheduled ~inline_set ~force_last_use () with
      | Regalloc.Allocated alloc ->
        sc.Scratch.regalloc <- Some alloc;
        sc.Scratch.unpin_candidates <- Some (compute_unpin_candidates scheduled);
        let regs, _ = Regalloc.count_bindings alloc in
        Printf.eprintf "[%s:%s] regalloc: %d tags bound\n" name label regs
      | Regalloc.Overflow budget ->
        sc.Scratch.regalloc <- None;
        sc.Scratch.unpin_candidates <- None;
        Printf.eprintf
          "[%s:%s] regalloc: OVERFLOW budget=%d, falling back to default\n"
          name
          label
          budget)
  in
  let clear_alloc () =
    sc.Scratch.regalloc <- None;
    sc.Scratch.unpin_candidates <- None;
    sc.Scratch.fence_only <- false
  in
  let _ = install_alloc in
  let _ = clear_alloc in
  (* === Stage 4 helpers ===
   *
   * `install_alloc_canonical` is the Stage-3-aware wrapper around
   * `Regalloc.allocate`. It takes a `Regalloc.regalloc_input` record
   * (the canonical shape) rather than three separate optional args.
   * Used by the prime/n1 path (Stage 4); the cluster-spill recipe
   * continues to use the older `install_alloc` for compatibility.
   *
   * The spill emission helpers (`emit_regalloc_spill_decl`,
   * `emit_node_spill_sites`, `emit_node_reload_sites`) factor out the
   * per-position M5/M6 emission patterns from the cluster-spill recipe.
   * In the spill recipe they remain inline (no need to refactor working
   * code); the prime/n1 path uses them through this factored form. *)
  let install_alloc_canonical (label : string) (input : Regalloc.regalloc_input) =
    if regalloc_enabled
    then (
      match
        Regalloc.allocate
          ~isa
          ~scheduled:input.scheduled
          ~inline_set:input.inline_set
          ~force_last_use:input.force_last_use
          ()
      with
      | Regalloc.Allocated alloc ->
        sc.Scratch.regalloc <- Some alloc;
        sc.Scratch.unpin_candidates <- Some (compute_unpin_candidates input.scheduled);
        let regs, _ = Regalloc.count_bindings alloc in
        Printf.eprintf "[%s:%s] regalloc: %d tags bound\n" name label regs
      | Regalloc.Overflow budget ->
        sc.Scratch.regalloc <- None;
        sc.Scratch.unpin_candidates <- None;
        Printf.eprintf
          "[%s:%s] regalloc: OVERFLOW budget=%d, falling back to default\n"
          name
          label
          budget)
  in
  let _ = install_alloc_canonical in
  let emit_regalloc_spill_decl (buf : Buffer.t) =
    match sc.Scratch.regalloc with
    | Some alloc when alloc.num_spill_slots > 0 ->
      Buffer.add_string
        buf
        (Printf.sprintf
           "        %s regalloc_spill[%d];\n"
           isa.vec_type
           alloc.num_spill_slots)
    | _ -> ()
  in
  let emit_node_spill_sites (buf : Buffer.t) (pos : int) =
    match sc.Scratch.regalloc with
    | Some alloc ->
      (match Hashtbl.find_opt alloc.spill_sites pos with
       | Some spills ->
         List.iter
           (fun (tag, slot) ->
              Buffer.add_string
                buf
                (Printf.sprintf
                   "        %s(&regalloc_spill[%d], t%d);\n"
                   isa.storeu_pd
                   slot
                   tag))
           spills
       | None -> ())
    | None -> ()
  in
  let emit_node_reload_sites (buf : Buffer.t) (pos : int) =
    match sc.Scratch.regalloc with
    | Some alloc ->
      (match Hashtbl.find_opt alloc.reload_sites pos with
       | Some reloads ->
         List.iter
           (fun (r : Regalloc.reload_decl) ->
              Buffer.add_string
                buf
                (Printf.sprintf
                   "        %s\n"
                   (Isa.pinned_reg_decl
                      isa
                      r.reload_name
                      r.reload_reg
                      (Printf.sprintf
                         "%s(&regalloc_spill[%d])"
                         isa.loadu_pd
                         r.reload_slot))))
           reloads
       | None -> ())
    | None -> ()
  in
  let _ = emit_regalloc_spill_decl in
  let _ = emit_node_spill_sites in
  let _ = emit_node_reload_sites in
  (* The twidsq flag selects the OOP-with-separate-strides signature.
   * Doc 43 introduced the twidsq math layer; this branch emits the
   * matching codelet calling convention with `is`, `os`, and `V` so the
   * codelet can be called with arbitrary input/output row strides — the
   * common case in multi-stage cascades where stage N's output stride
   * differs from stage N+1's input stride.
   *
   * Twidsq implies in_place=false (can't both transpose AND be in-place
   * with our current layout). We assert this rather than silently
   * recovering: a caller that sets both has a bug. *)
  if twidsq && in_place
  then failwith "emit_codelet: twidsq and in_place are mutually exclusive";
  if strided && (twidsq || in_place)
  then failwith "emit_codelet: strided not yet supported with twidsq or in_place";
  if strided && radix <= 0 then failwith "emit_codelet: strided requires --radix > 0";
  if strided && radix mod isa.vec_width <> 0
  then
    failwith
      (Printf.sprintf
         "emit_codelet: strided requires radix divisible by vec_width=%d (got %d)"
         isa.vec_width
         radix);
  let buf = Buffer.create 4096 in
  (* Arbitrary-K rem-aware hybrid tail (docs/roadmap/arbitrary_k_scalartail_
   * experiment.md = THE CONTRACT). Every in-place c2c batch codelet
   * (rio_re/rio_im/ios/me) — monolithic AND composite/CT-blocked (spill=Some) —
   * gets it; the strided two-pass codelets take a different signature branch and
   * are excluded. The bulk loop is byte-identical on aligned K (one never-taken
   * branch after it); the tail covers 1..VW-1 leftover lanes per the contract:
   * rem==1 -> ONE scalar single lane; rem>=2 -> ONE masked vector pass. The
   * scalar pass renders the DAG monolithically at width 1 (emit_body ~force_mono),
   * so composite codelets honour the same rem==1=scalar contract — a single lane
   * has no register pressure, so the CT spill scratch is simply not referenced
   * (no __m256d-vs-double clash). Kill switch VFFT_NO_ANYK_TAIL reverts to the
   * legacy K%VW==0-only loop.
   *
   * Enabled for the in-place c2c batch codelet (loop bound `me`) AND the r2r/trig
   * family (DCT/DST/DHT — signature `(in,out,K)`, loop bound `K`, batched over K,
   * same simple strided in[leg*K+k] -> out[leg*K+k] pattern, no twiddles). The
   * trig family hoists its trig-coefficient consts to function scope (__m256d), so
   * the tail resets sc.Scratch.hoisted_const_tags first → the scalar/masked passes re-emit
   * the consts inline at the right width (double / __m256d), shadowing the
   * function-scope ones (no-op for the hoist-off c2c tree). *)
  (* Real-FFT cascade families (rfft fwd + c2r bwd): r2cf/r2cb leaves, the hc2hc
   * stages, packed hc2c, and the hc2c-nat terminator/initiator — ALL set hc_strided
   * (gen_main: hc_strided := hc2hc || hc2c || hc2c_nat), loop over `v` with bound `vl`,
   * strided access in[leg*stride + v]. The 2D-transpose `strided` family, the
   * always-aligned `n1_oop_strided` decoupled-stride inner (executor always passes a
   * vec-width-multiple B), and r2c_term do NOT set hc_strided and are intentionally
   * EXCLUDED: transpose tail = phase-2 (masked transpose); stride inner never needs it;
   * r2c_term = separate fusion phase, wire later. Odd K must route to THIS cascade, not
   * the stride path (front-door selective-guard relax, src/core/transforms/real). *)
  let real_fft_sig = cfg.Cfg.r2cf || cfg.Cfg.r2cb || cfg.Cfg.hc_strided in
  let anyk_tail =
    (in_place || cfg.Cfg.r2r || real_fft_sig)
    &&
    match Sys.getenv_opt "VFFT_NO_ANYK_TAIL" with
    | Some _ -> false
    | None -> true
  in
  (* tail loop bound: in-place uses `me`; the r2r/trig signature uses `K`; the
   * real-FFT cascade uses `vl`. *)
  let tail_bound = if cfg.Cfg.r2r then "K" else if real_fft_sig then "vl" else "me" in
  (* tail loop var: must match render_load's loop_var — `v` for the real-FFT
   * cascade, `k` for in-place + r2r/trig. *)
  let tail_var = if real_fft_sig then "v" else "k" in
  (* Bulk batch-loop header for the real-FFT `v`-loop, with the conditional hoist so
   * `v` stays live for the remainder block (mirrors the in-place `k` hoist at the top
   * of the in-place signature branch). Used by every real-FFT signature branch. *)
  let emit_v_loop_header bound =
    if anyk_tail
    then (
      Buffer.add_string buf "    size_t v = 0;\n";
      Buffer.add_string
        buf
        (Printf.sprintf
           "    for (; v + %d <= %s; v += %d) {\n"
           isa.vec_width
           bound
           isa.vec_width))
    else
      Buffer.add_string
        buf
        (Printf.sprintf "    for (size_t v = 0; v < %s; v += %d) {\n" bound isa.vec_width)
  in
  let family =
    if strided
    then "strided-batch (Design C, 2D rows)"
    else if twidsq
    then "twidsq (FFTW-style intermediate)"
    else if is_n1
    then if in_place then "in-place n1 (no twiddles)" else "n1"
    else if t1s
    then "t1s (broadcast twiddles)"
    else if in_place
    then "in-place t1 (twiddled CT)"
    else "t1"
  in
  let sched_str =
    match scheduler with
    | Topological -> "topological (plain dependency order)"
    | SU _ ->
      "the list scheduler (flag name --su; lazy loads, sink-first, cp_dist, SU tiebreak; \
       section 30)"
    | _ -> "annotated/experimental"
  in
  Buffer.add_string
    buf
    (provenance_block
       ~family
       [ Printf.sprintf
           "ISA: %d-bit vectors, %d vector regs%s"
           (isa.Isa.vec_width * 64)
           isa.Isa.vec_regs
           (if is_avx2 then " (16-reg pressure rules apply)" else "")
       ; Printf.sprintf "Scheduler: %s" sched_str
       ; Printf.sprintf
           "GH pressure mode: %b (auto-rule: vec_regs<=16 && n>=32; +4-8%% documented)"
           gh
       ; (match spill with
          | Some sp when sp.ct_n1 > 0 ->
            Printf.sprintf
              "Construction: BLOCKED two-pass CT %dx%d; seam through L1 scratch by \
               design (doc 58); threshold n>=16 on <=16-reg ISAs else 25 (section 35)"
              sp.ct_n1
              sp.ct_n2
          | Some _ -> "Construction: blocked (spill recipe, non-CT marker set)"
          | None ->
            "Construction: MONOLITHIC (below blocking threshold, or prime/Direct: no CT \
             pass boundary)")
       ; Printf.sprintf
           "Regalloc+pinning: %b (gate: log3+avx512+R<=32 OR avx2 n1 R>=16, sections \
            32/34/35; kill switch VFFT_NO_REGALLOC)"
           regalloc_enabled
       ; Printf.sprintf
           "Value fences: %b (asm-volatile +v stops gcc rematerialization, doc 28; \
            exempt: n1 avx2 R in {8,16}, measured better unfenced)"
           fence_enabled
       ; (if is_log3 then "log3: DIT radix-decomposition ladder variant" else "log3: no")
       ]);
  Buffer.add_string buf "#include <immintrin.h>\n";
  Buffer.add_string buf "#include <stddef.h>\n\n";
  (* No _vfft_masklo table: the avx2 tail is all-SSE2 (no mask), and AVX-512 computes
   * its __mmask8 = (1<<rem)-1 inline. *)
  if isa.vec_width = 1
  then (
    Buffer.add_string
      buf
      "static inline double vfft_scalar_load(const double *p) { return *p; }\n";
    Buffer.add_string
      buf
      "static inline void vfft_scalar_store(double *p, double v) { *p = v; }\n\n");
  (* ── M4 phase 2: THE signature comes from Abi — one total constructor over
     the kind shape (proven byte-equivalent to the historical 13-arm ladder by
     the VFFT_ABI_XCHECK dual-emission pass over the full corpus before the
     ladder arms were deleted).  The chain below keeps only per-kind BODY
     content.  Shape derivation mirrors the historical arm ORDER; the M3
     legality guards live on here. *)
  let abi_shape : Abi.shape =
    if strided
    then (
      if cfg.Cfg.strided_il_in && cfg.Cfg.strided_il_out
      then
        failwith
          "emit_c(strided): --strided-il-in + --strided-il-out is the banned hybrid";
      if (cfg.Cfg.strided_il_in || cfg.Cfg.strided_il_out) && cfg.Cfg.strided_r2c
      then
        failwith
          "emit_c(strided): --strided-il-in/out cannot combine with --strided-r2c";
      Abi.Strided
        { il =
            (if cfg.Cfg.strided_il_in then `In else if cfg.Cfg.strided_il_out then `Out else `None)
        ; r2c =
            (if cfg.Cfg.strided_r2c_bwd then `Bwd else if cfg.Cfg.strided_r2c then `Fwd else `No)
        })
    else if in_place
    then
      Abi.In_place
        { il =
            (match Layout.ip_buffers_of_bools ~il_in:cfg.Cfg.ip_il_in ~il_out:cfg.Cfg.ip_il_out with
             | Layout.From_z -> `In
             | Layout.To_z -> `Out
             | _ -> `None)
        }
    else if twidsq
    then Abi.Twidsq
    else if cfg.Cfg.r2cb
    then Abi.R2cb
    else if cfg.Cfg.r2cf
    then Abi.R2cf
    else if cfg.Cfg.r2c_term_ls
    then Abi.R2c_term_ls
    else if cfg.Cfg.r2c_term
    then Abi.R2c_term { rt = cfg.Cfg.r2c_term_rt }
    else if cfg.Cfg.hc2c_natural
    then Abi.Hc2c_nat { ranged = cfg.Cfg.hc_ranged }
    else if cfg.Cfg.hc2c_natural_bwd
    then Abi.Hc2c_nat_bwd { ranged = cfg.Cfg.hc_ranged }
    else if cfg.Cfg.hc_strided
    then Abi.Hc_strided { ranged = cfg.Cfg.hc_ranged }
    else if cfg.Cfg.n1_oop_strided
    then Abi.N1_oop_strided
    else if cfg.Cfg.r2r
    then Abi.R2r
    else Abi.Oop_generic
  in
  Buffer.add_string
    buf
    (Abi.signature (Abi.make ~symbol:name ~target_attr:isa.target_attr abi_shape));
  if strided
  then (
    (* Strided-batch codelet (Design C for 2D row FFT).
     *
     * Signature:
     *   void NAME(rio_re, rio_im, tw_re, tw_im, row_stride, me)
     *
     * Reads B=vec_width rows from a matrix at stride row_stride, transposes
     * 4×4 (AVX2) / 8×8 (AVX-512) to get N lane vectors (each holding B batch
     * lanes at one FFT index), runs the codelet body, then inverse-transposes
     * and stores back to the matrix. No scratch buffer — matrix↔registers↔
     * matrix only.
     *
     * For v1 only n1 (no-twiddle) is supported; tw_re/tw_im are passed for
     * signature uniformity but unused. *)
    (* M4: signature emission deleted — Abi.signature is the one printer (strided). *)
    Buffer.add_string buf "    (void)tw_re; (void)tw_im;\n";
    (* M8.3: the r2c/c2r rio alias prologue moved to Real (family hook). *)
    (match hooks.strided_prologue with
     | Some f -> f buf
     | None ->
       if cfg.Cfg.strided_r2c_bwd || cfg.Cfg.strided_r2c
       then failwith "emit_codelet: strided r2c/c2r requires the Real route (M8.3)");
    (* AVX-512 only: pre-declare the two __m512i index vectors used by the
     * 8×8 transpose preamble AND postamble. Function-scope (outside the
     * b loop) so gcc treats them as loop-invariant constants. The
     * indices match transpose.h Kernel C — idx_lo gathers even-column
     * cross-lane elements, idx_hi gathers odd-column. *)
    if isa.vec_width = 8
    then (
      Buffer.add_string
        buf
        "    const __m512i _tp_idx_lo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);\n";
      Buffer.add_string
        buf
        "    const __m512i _tp_idx_hi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);\n";
      if cfg.Cfg.strided_il_out
      then (
        Buffer.add_string
          buf
          "    const __m512i _il_idx_e = _mm512_set_epi64(11, 3, 10, 2, 9, 1, 8, 0);\n";
        Buffer.add_string
          buf
          "    const __m512i _il_idx_o = _mm512_set_epi64(15, 7, 14, 6, 13, 5, 12, 4);\n");
      if cfg.Cfg.strided_il_in
      then (
        Buffer.add_string
          buf
          "    const __m512i _il_idx_de = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);\n";
        Buffer.add_string
          buf
          "    const __m512i _il_idx_do = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);\n"));
    Buffer.add_string
      buf
      (Printf.sprintf "    for (size_t b = 0; b < me; b += %d) {\n" isa.vec_width);
    (* Per-iteration locals: lane_re_0..radix-1 (inputs after transpose),
       out_lane_re_0..radix-1 (outputs before inverse transpose). Plus
       _im versions. *)
    (* M8.3: the c2r _hx lane decls moved to Real (family hook). *)
    (match hooks.strided_locals with
     | Some f -> f buf
     | None ->
       if cfg.Cfg.strided_r2c_bwd
       then failwith "emit_codelet: strided c2r requires the Real route (M8.3)");
    for j = 0 to radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf "        %s lane_re_%d, lane_im_%d;\n" isa.vec_type j j);
      Buffer.add_string
        buf
        (Printf.sprintf "        %s out_lane_re_%d, out_lane_im_%d;\n" isa.vec_type j j)
    done;
    Buffer.add_string buf "\n";
    (* AVX2 4×4 transpose preamble. For each group of 4 consecutive fft
     * indices (j0, j0+1, j0+2, j0+3), load 4 rows of 4 consecutive cols
     * starting at fft_idx=j0, then 4×4 transpose to get 4 lane vectors
     * each holding (row 0, row 1, row 2, row 3) at one fft_idx. *)
    if isa.vec_width = 4
    then (
      let groups = radix / 4 in
      if cfg.Cfg.strided_il_in
      then
        for g = 0 to groups - 1 do
          let j0 = g * 4 in
          Buffer.add_string
            buf
            (Printf.sprintf
               "        {  /* 4x4 transpose group (il_in): fft_idx %d..%d */\n"
               j0
               (j0 + 3));
          for r = 0 to 3 do
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m256d _zv0_%d = \
                  _mm256_loadu_pd(&in_z[2*((b+%d)*row_stride + %d) + 0]);\n"
                 r
                 r
                 j0);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m256d _zv1_%d = \
                  _mm256_loadu_pd(&in_z[2*((b+%d)*row_stride + %d) + 4]);\n"
                 r
                 r
                 j0);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m256d _row_re_%d = \
                  _mm256_permute4x64_pd(_mm256_unpacklo_pd(_zv0_%d, _zv1_%d), 0xD8);\n"
                 r
                 r
                 r);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m256d _row_im_%d = \
                  _mm256_permute4x64_pd(_mm256_unpackhi_pd(_zv0_%d, _zv1_%d), 0xD8);\n"
                 r
                 r
                 r)
          done;
          List.iter
            (fun suf ->
               for k = 0 to 3 do
                 let base = k / 2 * 2 in
                 let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m256d _t%d_%s = _mm256_%s_pd(_row_%s_%d, \
                       _row_%s_%d);\n"
                      k
                      suf
                      op
                      suf
                      base
                      suf
                      (base + 1))
               done;
               for i = 0 to 3 do
                 let ta = i mod 2 in
                 let tb = 2 + (i mod 2) in
                 let imm = if i < 2 then "0x20" else "0x31" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            lane_%s_%d = _mm256_permute2f128_pd(_t%d_%s, _t%d_%s, \
                       %s);\n"
                      suf
                      (j0 + i)
                      ta
                      suf
                      tb
                      suf
                      imm)
               done)
            [ "re"; "im" ];
          Buffer.add_string buf "        }\n"
        done
      else if cfg.Cfg.strided_r2c_bwd
      then (
        (* M8.3: the merge-load lattice moved to Real (family hook). *)
        match hooks.strided_load with
        | Some f -> f buf
        | None -> failwith "emit_codelet: strided c2r requires the Real route (M8.3)")
      else
        Simd.load_transpose_4x4 ~buf ~groups)
    else if isa.vec_width = 8
    then (
      (* AVX-512 8×8 transpose preamble. For each group of 8 consecutive
       * fft indices (j0..j0+7), load 8 rows of 8 cols starting at
       * fft_idx=j0, then 3-stage in-register transpose to produce 8
       * lane vectors each holding (row b..row b+7) at one fft_idx.
       *
       * Reference: transpose.h Kernel C. The 3-stage pipeline is:
       *   Stage 1: 8 unpacklo/unpackhi_pd pairs over row pairs (0,1) (2,3) (4,5) (6,7)
       *   Stage 2: 8 permutex2var_pd with _tp_idx_lo / _tp_idx_hi (declared at function scope)
       *   Stage 3: 8 shuffle_f64x2 with imm 0x44 (lo halves) / 0xEE (hi halves)
       *           assigned directly to lane_re_{j0..j0+7} / lane_im_{j0..j0+7}
       * Stages 1 and 2 use block-local `const __m512d _tk_re` / `_xk_re`
       * names so the same identifiers are reused across groups without
       * collision — the `{ ... }` block scope makes that safe. *)
      let groups = radix / 8 in
      if cfg.Cfg.strided_il_in
      then
        for g = 0 to groups - 1 do
          let j0 = g * 8 in
          Buffer.add_string
            buf
            (Printf.sprintf
               "        {  /* 8x8 transpose group (il_in): fft_idx %d..%d */\n"
               j0
               (j0 + 7));
          for r = 0 to 7 do
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m512d _zv0_%d = \
                  _mm512_loadu_pd(&in_z[2*((b+%d)*row_stride + %d) + 0]);\n"
                 r
                 r
                 j0);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m512d _zv1_%d = \
                  _mm512_loadu_pd(&in_z[2*((b+%d)*row_stride + %d) + 8]);\n"
                 r
                 r
                 j0);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m512d _row_re_%d = _mm512_permutex2var_pd(_zv0_%d, \
                  _il_idx_de, _zv1_%d);\n"
                 r
                 r
                 r);
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const __m512d _row_im_%d = _mm512_permutex2var_pd(_zv0_%d, \
                  _il_idx_do, _zv1_%d);\n"
                 r
                 r
                 r)
          done;
          List.iter
            (fun suf ->
               for k = 0 to 7 do
                 let base = k / 2 * 2 in
                 let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m512d _t%d_%s = _mm512_%s_pd(_row_%s_%d, \
                       _row_%s_%d);\n"
                      k
                      suf
                      op
                      suf
                      base
                      suf
                      (base + 1))
               done;
               for k = 0 to 7 do
                 let ua = (k mod 4 mod 2) + (k / 4 * 4) in
                 let ub = ua + 2 in
                 let idx = if k mod 4 < 2 then "_tp_idx_lo" else "_tp_idx_hi" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            const __m512d _x%d_%s = \
                       _mm512_permutex2var_pd(_t%d_%s, %s, _t%d_%s);\n"
                      k
                      suf
                      ua
                      suf
                      idx
                      ub
                      suf)
               done;
               for i = 0 to 7 do
                 let va = if i < 4 then i else i - 4 in
                 let vb = if i < 4 then i + 4 else i in
                 let imm = if i < 4 then "0x44" else "0xEE" in
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            lane_%s_%d = _mm512_shuffle_f64x2(_x%d_%s, _x%d_%s, \
                       %s);\n"
                      suf
                      (j0 + i)
                      va
                      suf
                      vb
                      suf
                      imm)
               done)
            [ "re"; "im" ];
          Buffer.add_string buf "        }\n"
        done
      else if cfg.Cfg.strided_r2c_bwd
      then (
        (* M8.3: the merge-load lattice moved to Real (family hook). *)
        match hooks.strided_load with
        | Some f -> f buf
        | None -> failwith "emit_codelet: strided c2r requires the Real route (M8.3)")
      else
        Simd.load_transpose_8x8 ~buf ~groups)
    else
      failwith
        (Printf.sprintf
           "emit_codelet: strided not supported for vec_width=%d"
           isa.vec_width);
    Buffer.add_string buf "\n")
  else if in_place
  then (
    (* M4: signature emission deleted — Abi.signature is the one printer (in_place). *)
    if isa.vec_width = 8 && (cfg.Cfg.ip_il_in || cfg.Cfg.ip_il_out)
    then (
      if cfg.Cfg.ip_il_in
      then (
        Buffer.add_string
          buf
          "    const __m512i _il_de = _mm512_setr_epi64(0,2,4,6,8,10,12,14);\n";
        Buffer.add_string
          buf
          "    const __m512i _il_do = _mm512_setr_epi64(1,3,5,7,9,11,13,15);\n";
        Buffer.add_string buf "    (void)_il_de; (void)_il_do;\n");
      if cfg.Cfg.ip_il_out
      then (
        Buffer.add_string
          buf
          "    const __m512i _il_pe = _mm512_setr_epi64(0,8,1,9,2,10,3,11);\n";
        Buffer.add_string
          buf
          "    const __m512i _il_po = _mm512_setr_epi64(4,12,5,13,6,14,7,15);\n";
        Buffer.add_string buf "    (void)_il_pe; (void)_il_po;\n"));
    (* Spill array decl, OUTSIDE the for loop so it's allocated once *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    if anyk_tail
    then (
      (* Hoist k so it stays live for the remainder block after the bulk loop. *)
      Buffer.add_string buf "    size_t k = 0;\n";
      Buffer.add_string
        buf
        (Printf.sprintf
           "    for (; k + %d <= me; k += %d) {\n"
           isa.vec_width
           isa.vec_width))
    else
      Buffer.add_string
        buf
        (Printf.sprintf "    for (size_t k = 0; k < me; k += %d) {\n" isa.vec_width))
  else if twidsq
  then (
    (* Twidsq OOP signature with separate input/output strides.
     *   in_re[slot * is + v] for input element at slot i*n+k
     *   out_re[slot * os + v] for output element at slot j*n+i (TRANSPOSED
     *     — the math layer already encodes the transpose via Output indices)
     *   Twiddles broadcast across V lanes (uniform across batches).
     *   V is the loop bound; vec_width lanes processed per iteration.
     *)
    (* M4: signature emission deleted — Abi.signature is the one printer (twidsq). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ());
    Buffer.add_string
      buf
      (Printf.sprintf "    for (size_t v = 0; v < V; v += %d) {\n" isa.vec_width))
  else if cfg.Cfg.r2cb
  then (
    (* r2cb backward real leaf (section 62): halfcomplex INPUT (in_re, in_im)
     * -> real OUTPUT (out_re). The body reconstructs the conjugate-symmetric
     * spectrum from the packed half and runs a backward DFT; the result is
     * purely real so there is no out_im. Same stride/loop shape as r2cf
     * (is input stride, os_re output stride, vl lanes). *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.r2cb). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    emit_v_loop_header "vl")
  else if cfg.Cfg.r2cf
  then (
    (* r2cf leaf v2 (section 62): the composition algebra forces a
     * REVERSED im output stream (packed im slot for k lands at
     * position n-k, walking DOWN as k walks up), so the strides are
     * signed and split per parity. The executor passes os_im < 0 with
     * out_im based one-past the region. P1's stride_n1_fn-shaped v1
     * is withdrawn — composition beats typedef aesthetics. *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.r2cf). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    emit_v_loop_header "vl")
  else if cfg.Cfg.r2c_term_ls
  then (
    (* model (b): fused last-stage terminator. Two columns of r complex legs +
     * packed twiddle table (3r slots) -> 2r outputs (Xp[s], Xm[s]).
     * Input(j) for j<r = col k leg j; Input(r+j) = col m-k leg j. Legs are
     * strided by is_leg within each column; the two columns are at separate
     * base pointers in_k / in_m (the executor passes the two physical rows). *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.r2c_term_ls). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    emit_v_loop_header "vl")
  else if cfg.Cfg.r2c_term
  then (
    (* r2c_term (step-2 fusion): 2 inputs (Z[k], Z[m]) -> 2 outputs (X[k], X[m]),
     * vectorized over vl lanes. Reads scratch rows for the column pair
     * sequentially (the executor supplies them in natural order, no scatter).
     * Output(0) -> Xp pair (X[k]); Output(1) -> Xm pair (X[m]). *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.r2c_term). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    emit_v_loop_header "vl")
  else if cfg.Cfg.hc2c_natural
  then (
    (* D2 natural terminator (section 69): four output pointers,
     * boundary baked at generation time. *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.hc2c_natural). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    if cfg.Cfg.hc_ranged then Buffer.add_string buf "    for (int kc = 0; kc < kcount; kc++) {\n";
    emit_v_loop_header "vl")
  else if cfg.Cfg.hc2c_natural_bwd
  then (
    (* c2r natural INITIATOR (inverse of hc2c_natural): four SPLIT const inputs
     * (Rp/Ip direct rows + Rm/Im conjugate-mirror rows, the same sstar map as
     * the forward but on the INPUT side) -> two PACKED cascade columns
     * (out_re/out_im). isp/ism = split input row strides; os = packed output
     * stride. The forward's 6-pointer ABI, flipped. *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.hc2c_natural_bwd). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    if cfg.Cfg.hc_ranged then Buffer.add_string buf "    for (int kc = 0; kc < kcount; kc++) {\n";
    emit_v_loop_header "vl")
  else if cfg.Cfg.hc_strided
  then (
    (* hc2hc / hc2c strided variant (section 62): the generic ABI's
     * hardcoded slot stride K cannot address middle cascade stages
     * (slot strides are Q*K-multiples and out != in stride). Twiddles
     * replicate per vl lanes, slot 0 never loaded. *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.hc_strided). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    if cfg.Cfg.hc_ranged then Buffer.add_string buf "    for (int kc = 0; kc < kcount; kc++) {\n";
    emit_v_loop_header "vl")
  else if cfg.Cfg.n1_oop_strided
  then (
    (* strided-OOP n1: ABI = vfft_proto_n1_fn (registry n1 slot shape).
     * size_t strides (not ptrdiff_t) to match the fn-pointer type
     * byte-for-byte; vl assumed a multiple of the vector width (the r2c
     * executor always passes B, a vec-width multiple). No tw params:
     * n1 DAGs carry no Twiddle refs. *)
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.n1_oop_strided). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    emit_v_loop_header "vl")
  else if cfg.Cfg.r2r
  then (
    (* M4: signature emission deleted — Abi.signature is the one printer (cfg.Cfg.r2r). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    if anyk_tail
    then (
      Buffer.add_string buf "    size_t k = 0;
";
      Buffer.add_string
        buf
        (Printf.sprintf
           "    for (; k + %d <= K; k += %d) {
"
           isa.vec_width
           isa.vec_width))
    else
      Buffer.add_string
        buf
        (Printf.sprintf "    for (size_t k = 0; k < K; k += %d) {
" isa.vec_width))
  else (
    (* M4: signature emission deleted — Abi.signature is the one printer (oop_generic). *)
    Buffer.add_string buf (Emit_render.body_preamble ~sc ~isa ~spill ~consts:assigns ());
    Buffer.add_string
      buf
      (Printf.sprintf "    for (size_t k = 0; k < K; k += %d) {\n" isa.vec_width));
  let out_buf is_re =
    match in_place, is_re with
    | true, true -> "rio_re"
    | true, false -> "rio_im"
    | false, true -> if cfg.Cfg.r2r then "out" else "out_re"
    | false, false -> "out_im"
  in
  (* Output stride and loop variable depend on the codelet kind:
   *   in_place : stride=ios, loop=k
   *   twidsq   : stride=os,  loop=v, AND decompose slot to (row, col)
   *   OOP      : stride=K,   loop=k
   *
   * For twidsq, the math layer's Output(j*n + i, _) encodes the transpose
   * via index choice (row j, col i of the OUTPUT block). The emitter
   * decomposes the linear slot s = j*n + i back into (s/n, s%n) so the
   * address is `(s/n)*os + (s%n)*V + v` — naturally row-major in the
   * output buffer with caller-supplied row stride `os`. *)
  let out_stride =
    if in_place
    then "ios"
    else if twidsq
    then "os"
    else if cfg.Cfg.hc_strided || cfg.Cfg.n1_oop_strided
    then "os"
    else "K"
  in
  let out_stride_for is_re =
    if cfg.Cfg.r2cf
    then if is_re then "os_re" else "os_im"
    else if cfg.Cfg.r2cb
    then "os_re"
    else out_stride
  in
  let loop_var =
    if
      twidsq
      || cfg.Cfg.r2cf
      || cfg.Cfg.r2cb
      || cfg.Cfg.hc_strided
      || cfg.Cfg.n1_oop_strided
      || cfg.Cfg.r2c_term
      || cfg.Cfg.r2c_term_ls
    then "v"
    else "k"
  in
  (* The codelet body, rendered off the SINGLE schedule. Called once for the
   * bulk loop (isa = outer, LS_vector) and — for the arbitrary-K tail — again
   * for the scalar single lane (isa = scalar) and the masked vector pass
   * (isa = outer, LS_masked). The `isa` parameter shadows the outer ISA so
   * render_node_def / emit_store pick up the per-pass width; the regalloc /
   * spill helpers above stay inert for the monolithic regalloc-off codelets
   * that take the tail (so re-rendering is safe and the schedule is computed
   * deterministically each pass — identical order, no re-scheduling hazard). *)
  let emit_body ?(force_mono = false) (isa : Isa.t) () =
    Scratch.il_reset sc;
    let render_output_addr k is_re =
      if cfg.Cfg.r2c_term_ls
      then (* Output(2s) = Xp slot s at Xp[s*osp+v]; Output(2s+1) = Xm slot s. *)
        if k land 1 = 0
        then
          Printf.sprintf
            "%s[%d*osp + %s]"
            (if is_re then "Xp_re" else "Xp_im")
            (k / 2)
            loop_var
        else
          Printf.sprintf
            "%s[%d*osm + %s]"
            (if is_re then "Xm_re" else "Xm_im")
            (k / 2)
            loop_var
      else if cfg.Cfg.r2c_term
      then (* Output(0)=X[k]->Xp pair; Output(1)=X[m]->Xm pair, over v lanes. *)
        if k = 0
        then Printf.sprintf "%s[%s]" (if is_re then "Xp_re" else "Xp_im") loop_var
        else Printf.sprintf "%s[%s]" (if is_re then "Xm_re" else "Xm_im") loop_var
      else if cfg.Cfg.hc2c_natural
      then
        if k <= cfg.Cfg.hc2c_nat_sstar
        then Printf.sprintf "%s[%d*osp + %s]" (if is_re then "Rp" else "Ip") k loop_var
        else
          Printf.sprintf
            "%s[%d*osm + %s]"
            (if is_re then "Rm" else "Im")
            (cfg.Cfg.hc2c_nat_r - 1 - k)
            loop_var
      else (
        let buf = out_buf is_re in
        if twidsq && twidsq_n > 0
        then (
          let row = k / twidsq_n in
          let col = k mod twidsq_n in
          Printf.sprintf "%s[%d*%s + %d*V + %s]" buf row out_stride col loop_var)
        else Printf.sprintf "%s[%d*%s + %s]" buf k (out_stride_for is_re) loop_var)
    in
    let emit_store buf oref e =
      (* M5: three cases for the store value source:
       *   1. name_override exists at current_emit_position: use the
       *      reload variable (e.g. tT_r0). Register-pinned, fastest.
       *   2. Tag is in spilled_of_tag but no override: emit inline
       *      load-from-memory. Used when no register was available for
       *      a reload at the flush position (peak register pressure).
       *      gcc handles the temp register.
       *   3. Otherwise: bare tT — the value is in its register.
       *
       * Returns (value_expr, is_inline_load). When is_inline_load is
       * true, the value_expr is a load intrinsic; we wrap it in the
       * store accordingly. *)
      let value_expr =
        match sc.Scratch.regalloc with
        | None -> Printf.sprintf "t%d" e.tag
        | Some alloc ->
          (match
             Hashtbl.find_opt alloc.name_overrides (sc.Scratch.emit_position, e.tag)
           with
           | Some n -> n
           | None ->
             (match Hashtbl.find_opt alloc.spilled_of_tag e.tag with
              | Some slot ->
                (* Inline load from regalloc_spill. gcc picks a temp. *)
                Printf.sprintf "%s(&regalloc_spill[%d])" isa.loadu_pd slot
              | None -> Printf.sprintf "t%d" e.tag))
      in
      match oref with
      | Expr.Output (k, true) when strided ->
        Buffer.add_string
          buf
          (Printf.sprintf "        out_lane_re_%d = %s;\n" k value_expr)
      | Expr.Output (k, false) when strided ->
        Buffer.add_string
          buf
          (Printf.sprintf "        out_lane_im_%d = %s;\n" k value_expr)
      | Expr.Output (k, true) when cfg.Cfg.ip_il_out ->
        (* defer: fused with the adjacent im-store (sink-first pairs) *)
        sc.Scratch.il_stash <- Some (k, value_expr)
      | Expr.Output (k, false) when cfg.Cfg.ip_il_out ->
        let vre =
          match sc.Scratch.il_stash with
          | Some (k2, vre) when k2 = k -> vre
          | _ -> failwith "ip_il_out: unpaired im store (scheduler contract)"
        in
        sc.Scratch.il_stash <- None;
        let ee = Printf.sprintf "%d*ios + k" k in
        (match isa.vec_width, sc.Scratch.ls_mode with
         | 1, _ ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        out_z[2*(%s)] = %s;\n        out_z[2*(%s) + 1] = %s;\n"
                ee
                vre
                ee
                value_expr)
         | 8, Isa.LS_vector ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        _mm512_storeu_pd(&out_z[2*(%s)], _mm512_permutex2var_pd(%s, \
                 _il_pe, %s));\n\
                \        _mm512_storeu_pd(&out_z[2*(%s) + 8], _mm512_permutex2var_pd(%s, \
                 _il_po, %s));\n"
                ee
                vre
                value_expr
                ee
                vre
                value_expr)
         | 8, Isa.LS_masked m ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { const unsigned _ild = _pdep_u32((unsigned)%s, 0x5555u) * 3u;\n\
                \          _mm512_mask_storeu_pd(&out_z[2*(%s)], (__mmask8)_ild, \
                 _mm512_permutex2var_pd(%s, _il_pe, %s));\n\
                \          _mm512_mask_storeu_pd(&out_z[2*(%s) + 8], (__mmask8)(_ild >> \
                 8), _mm512_permutex2var_pd(%s, _il_po, %s)); }\n"
                m
                ee
                vre
                value_expr
                ee
                vre
                value_expr)
         | 4, _ ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { const __m256d _ilp = _mm256_unpacklo_pd(%s, %s);\n\
                \          const __m256d _ilq = _mm256_unpackhi_pd(%s, %s);\n\
                \          _mm256_storeu_pd(&out_z[2*(%s)], _mm256_permute2f128_pd(_ilp, \
                 _ilq, 0x20));\n\
                \          _mm256_storeu_pd(&out_z[2*(%s) + 4], \
                 _mm256_permute2f128_pd(_ilp, _ilq, 0x31)); }\n"
                vre
                value_expr
                vre
                value_expr
                ee
                ee)
         | 2, _ ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        _mm_storeu_pd(&out_z[2*(%s)], _mm_unpacklo_pd(%s, %s));\n\
                \        _mm_storeu_pd(&out_z[2*(%s) + 2], _mm_unpackhi_pd(%s, %s));\n"
                ee
                vre
                value_expr
                ee
                vre
                value_expr)
         | _ -> failwith "ip_il_out: unsupported width")
      | Expr.Output (k, true) ->
        Buffer.add_string buf "        ";
        Buffer.add_string
          buf
          (Isa.storeu_pd
             ~mode:sc.Scratch.ls_mode
             isa
             (render_output_addr k true)
             value_expr);
        Buffer.add_string buf ";\n"
      | Expr.Output (k, false) ->
        Buffer.add_string buf "        ";
        Buffer.add_string
          buf
          (Isa.storeu_pd
             ~mode:sc.Scratch.ls_mode
             isa
             (render_output_addr k false)
             value_expr);
        Buffer.add_string buf ";\n"
      | _ -> failwith "emit_codelet: assignment LHS must be an Output"
    in
    (* Spill-aware emission path. When ?spill is provided, take this path
   * regardless of scheduler choice — the spill structure imposes a strict
   * pass boundary that supersedes whatever ordering the scheduler would
   * pick across passes. Within each pass, we still emit in topological
   * order (matching Topological scheduler behavior).
   *
   * Currently only Topological scheduling within passes is supported.
   * SU + spill is straightforward to add later: run SU per-pass on the
   * filtered node lists. For first validation, Topo+spill is the priority. *)
    (* force_mono: render the DAG monolithically (no PASS1/PASS2 spill split) even
   * for a spill=Some codelet. Used ONLY by the rem==1 SCALAR tail pass: a single
   * scalar lane has no vector-register pressure, so the CT-blocking that the spill
   * recipe exists to relieve is unnecessary — and rendering monolithically avoids
   * the __m256d scratch entirely (the scratch is still declared at function scope
   * for the bulk/masked passes; the scalar pass simply doesn't reference it). This
   * is what lets the contract's "rem==1 => scalar single lane" hold for composite
   * codelets, not just monolithic ones. *)
    match if force_mono then None else spill with
    | Some sp ->
      let roots = List.map snd assigns in
      let nodes = Emit_render.topo_sort_reachable roots in
      let cls = classify_passes sp nodes in
      (* Single-use inlining for the spill path. A tag is inlinable iff
       * it has exactly one consumer (compute_inline_set), is not a
       * Load/Const/Cmul, is NOT spilled, and has all consumers in the
       * SAME pass as the producer. Single source of truth shared with
       * codelet_oop: filter_inline_set_cross_pass ~sc (section 37). *)
      let inline_set = filter_inline_set_cross_pass ~sc assigns sp nodes in
      let is_inlined e = Hashtbl.mem inline_set e.tag in
      (* Constants are leaves (no predecessors) shared across passes via
       * hash-consing — a single NK_Const node may be referenced by both
       * PASS 1 and PASS 2 nodes (e.g. 1/√2 used in radix-8 internal
       * twiddles in PASS 1 and again in radix-4 internal twiddles in
       * PASS 2). To keep them in scope across both pass scopes, hoist
       * NK_Const declarations to the for-loop body top, BEFORE either
       * pass scope opens.
       *
       * Loads stay in their classified pass — they depend on the loop
       * variable `k` and are used only by their direct consumers. *)
      let is_const e =
        match e.node with
        | NK_Const _ -> true
        | _ -> false
      in
      let const_nodes = List.filter is_const nodes in
      let pass1_nodes =
        List.filter
          (fun e ->
             (not (is_const e))
             &&
             match Hashtbl.find_opt cls e.tag with
             | Some `Pass1 -> true
             | _ -> false)
          nodes
      in
      let pass2_nodes =
        List.filter
          (fun e ->
             (not (is_const e))
             &&
             match Hashtbl.find_opt cls e.tag with
             | Some `Pass2 -> true
             | _ -> false)
          nodes
      in
      (* Split output assigns by where their value is computed.
       *
       * Output stores must be emitted in the same C scope where the value
       * is in scope. Pass 1 outputs (value computed in PASS 1, no spilled
       * dependencies) get their stores at the end of PASS 1's `{ ... }`
       * block; Pass 2 outputs at the end of PASS 2's block. The original
       * design assumed all outputs were Pass 2 (everything depended on
       * spilled intermediates), but composite codelets like R=32 t1_dit
       * have outputs whose entire dep chain is twiddled-input cmuls →
       * inner-DFT chains that DON'T cross the spill boundary. Without
       * splitting, those outputs were emitted as `_mm512_storeu_pd(..., t<N>)`
       * inside PASS 2's scope, but t<N> was declared in PASS 1's scope
       * which had already closed — undefined-reference compile errors. *)
      let pass1_assigns =
        List.filter (fun (_, e) -> Hashtbl.find_opt cls e.tag = Some `Pass1) assigns
      in
      let pass2_assigns =
        List.filter (fun (_, e) -> Hashtbl.find_opt cls e.tag = Some `Pass2) assigns
      in
      (* Helper: list (slot, tag) pairs sorted by slot for deterministic output.
       * Currently unused — deferred-reload path emits reloads on demand
       * rather than in slot order, but keep helper available. *)
      let _sorted_by_slot (h : (int, int) Hashtbl.t) : (int * int) list =
        Hashtbl.fold (fun tag slot acc -> (slot, tag) :: acc) h []
        |> List.sort (fun (s1, _) (s2, _) -> compare s1 s2)
      in
      (* Hoisted constants — emitted at for-loop body top, in scope everywhere. *)
      List.iter
        (fun e ->
           Buffer.add_string
             buf
             (render_node_def ~sc ~cfg ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e);
           Buffer.add_char buf '\n')
        const_nodes;
      Buffer.add_char buf '\n';
      (* Tag → slot lookups (separate hashtables for re/im so we can tell
       * which spill array a tag belongs to). *)
      let lookup_re_slot tag = Hashtbl.find_opt sp.re_slot tag in
      let lookup_im_slot tag = Hashtbl.find_opt sp.im_slot tag in
      (* Fused tags: those whose slot is in fused_slots. These keep their
      * SSA values alive across the PASS 1 / PASS 2 boundary instead of
      * round-tripping through spill_re/spill_im. They need:
      *   1. Forward declaration at outer scope (before PASS 1 `{`)
      *   2. PASS 1 emission as assignment (no `__m512d` declarator)
      *   3. No spill store at end of PASS 1 emission for that tag
      *   4. No reload at start of PASS 2 for that slot
      * They remain accessible in PASS 2 by their original t<tag> name. *)
      (* Fused-tag predicate: the module-level is_fused_tag (single source
        shared with codelet_oop's OOP path). Closed over sp here so the
        call sites below stay `is_fused_tag tag`. *)
      let is_fused_tag tag = is_fused_tag sp tag in
      (* Forward-declare fused tags at outer scope. *)
      let fused_pass1_tags =
        List.filter_map
          (fun e -> if is_fused_tag e.tag then Some e.tag else None)
          pass1_nodes
      in
      if fused_pass1_tags <> []
      then (
        let names = List.map (fun t -> Printf.sprintf "t%d" t) fused_pass1_tags in
        Buffer.add_string buf "        ";
        Buffer.add_string buf (Isa.forward_decl isa names);
        Buffer.add_char buf '\n';
        Buffer.add_char buf '\n');
      (* Block-sequential PASS 1 ordering.
      *
      * Plain topological (tag) order interleaves sub-FFT computations,
      * causing peak live-set in PASS 1 to be the sum of all in-flight
      * sub-FFTs' state. With CT(N1, N2), all N1 sub-FFTs are
      * independent, so they CAN be ordered block-sequentially
      * (sub-FFT 0 fully complete, then sub-FFT 1, ...). This is what
      * hand-coded does, and it keeps peak live = O(N2) instead of
      * O(N1*N2).
      *
      * For each PASS 1 node, we compute min_descendant_slot — the
      * smallest spill slot reachable from this node. Spill targets
      * have their own slot. Intermediates inherit from their
      * (forward) successors. Sorting nodes by (min_descendant_slot,
      * tag) clusters sub-FFTs:
      *   - Sub-FFT 0 owns slots 0..N2-1
      *   - Sub-FFT 1 owns slots N2..2N2-1
      *   - etc.
      * so all sub-FFT-0 nodes sort before sub-FFT-1 nodes.
      *
      * Within a sub-FFT, tag order is preserved so dependencies are
      * respected. Across sub-FFTs, there are no dependencies (CT
      * independence) — except for shared constants, which we already
      * hoisted outside both pass scopes. *)
      (* min_slot + pre-cluster ordering via the shared helper (single
      * source with codelet_oop). The helper uses an explicit descending
      * sort, deleting the prior List.rev-on-pass1_nodes dependence on the
      * producer's order. *)
      let min_slot, pass1_blocked_topo = compute_min_slot_pass1 sp pass1_nodes in
      (* If scheduler is SU, replace tag-order WITHIN each sub-FFT cluster
       * with SU ordering. Cluster boundary = min_slot range corresponding
       * to one PASS 1 sub-FFT. CT(N1, N2): cluster k owns slots [k*N2, (k+1)*N2 - 1].
       * Sub-FFTs are mutually independent (CT property), so SU within a
       * cluster is safe — it cannot reorder across cluster boundaries.
       *
       * For non-CT cases (ct_n2 = 0), fall back to global tag order. *)
      let pass1_blocked =
        match scheduler with
        | SU uarch when sp.ct_n2 > 0 ->
          (* Cluster-split + per-cluster schedule via the shared helper
            (single source with codelet_oop). bb_budget selects su vs bb
            per cluster — that's the caller-specific closure. *)
          cluster_split_schedule
            sp
            ~pass1_blocked_topo
            ~min_slot
            ~schedule_cluster:(fun ~subset ~sinks ->
              match bb_budget with
              | None -> Schedule.su_schedule_subset uarch ~gh ~subset ~sinks
              | Some t -> Bb.bb_schedule_subset uarch ~time_budget_sec:t ~subset ~sinks)
        | _ -> pass1_blocked_topo
      in
      record_peak_live "spill_pass1" pass1_blocked;
      (* Build pass1 force_last_use: tags in pass1_assigns are stored at
       * end of pass 1 via a final List.iter. Force their last_use to
       * the end of the schedule. *)
      let pass1_force_last_use : (int, int) Hashtbl.t = Hashtbl.create 16 in
      let pass1_n = List.length pass1_blocked in
      List.iter
        (fun (_, e) -> Hashtbl.replace pass1_force_last_use e.tag pass1_n)
        pass1_assigns;
      install_alloc
        "spill_pass1"
        pass1_blocked
        (Some inline_set)
        (Some pass1_force_last_use);
      (* PASS 1 nested scope: emit block-sequentially with immediate spill.
       * For fused tags: emit as assignment (no declarator) to outer-scope
       * variable, and skip the spill store.
       * For inlined tags: skip standalone declaration; the consumer's
       * render will inline the expression directly. *)
      Buffer.add_string buf "        {\n";
      (* M5: declare the regalloc_spill[] scratch array if M5 spilling
       * is active for this pass. The array is pass-local — its slots
       * are only referenced between defs and uses within this pass. *)
      (match sc.Scratch.regalloc with
       | Some alloc when alloc.num_spill_slots > 0 ->
         Buffer.add_string
           buf
           (Printf.sprintf
              "            %s regalloc_spill[%d];\n"
              isa.vec_type
              alloc.num_spill_slots)
       | _ -> ());
      List.iteri
        (fun pos e ->
           sc.Scratch.emit_position <- pos;
           (* M5: emit spill stores for any tags evicted at this position.
            * The eviction was decided by the allocator (pool empty); the
            * store happens BEFORE the new def overwrites the register.
            *
            * ORDER: spill stores must precede reload loads. If a tag T is
            * spilled AND reloaded at the same position (because T was
            * evicted at p AND T has force_last_use[T] = p), the spill
            * writes T's value to slot S; the reload then reads S. If we
            * reloaded first, the slot would be uninitialized. *)
           (match sc.Scratch.regalloc with
            | Some alloc ->
              (match Hashtbl.find_opt alloc.spill_sites pos with
               | Some spills ->
                 List.iter
                   (fun (tag, slot) ->
                      Buffer.add_string
                        buf
                        (Printf.sprintf
                           "            %s(&regalloc_spill[%d], t%d);\n"
                           isa.storeu_pd
                           slot
                           tag))
                   spills
               | None -> ())
            | None -> ());
           (* M5: emit any reload declarations for this position before the
            * node's own def. Each reload is a fresh register-pinned load
            * from regalloc_spill[slot] into a shadow variable tT_rK. *)
           (match sc.Scratch.regalloc with
            | Some alloc ->
              (match Hashtbl.find_opt alloc.reload_sites pos with
               | Some reloads ->
                 List.iter
                   (fun (r : Regalloc.reload_decl) ->
                      Buffer.add_string
                        buf
                        (Printf.sprintf
                           "        %s\n"
                           (Isa.pinned_reg_decl
                              isa
                              r.reload_name
                              r.reload_reg
                              (Printf.sprintf
                                 "%s(&regalloc_spill[%d])"
                                 isa.loadu_pd
                                 r.reload_slot))))
                   reloads
               | None -> ())
            | None -> ());
           if is_inlined e
           then ()
           else (
             let no_declarator = is_fused_tag e.tag in
             Buffer.add_string
               buf
               (render_node_def
               ~sc
               ~cfg
                  ~no_declarator
                  ~t1s
                  ~isa
                  ~in_place
                  ~twidsq
                  ~twidsq_n
                  ~strided
                  ~inline_set:(Some inline_set)
                  e);
             Buffer.add_char buf '\n';
             if not no_declarator
             then (
               (match lookup_re_slot e.tag with
                | Some slot ->
                  Buffer.add_string
                    buf
                    (Printf.sprintf
                       "            %s(&spill_re[%d], t%d);\n"
                       isa.storeu_pd
                       slot
                       e.tag)
                | None -> ());
               match lookup_im_slot e.tag with
               | Some slot ->
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            %s(&spill_im[%d], t%d);\n"
                      isa.storeu_pd
                      slot
                      e.tag)
               | None -> ())))
        pass1_blocked;
      (* Emit stores for Pass 1 outputs at end of PASS 1 — values are still
       * in scope here. Pass 2 outputs are stored later, inside PASS 2's
       * scope (per-cluster flush + safety net).
       *
       * M5: pass 1's force_last_use put pass1_assigns tags at position
       * pass1_n (one past last). Reloads registered there. Set
       * current_emit_position so emit_store sees the right overrides. *)
      let pass1_n = List.length pass1_blocked in
      sc.Scratch.emit_position <- pass1_n;
      (* M5: spill stores BEFORE reload loads at end-of-pass. *)
      (match sc.Scratch.regalloc with
       | Some alloc ->
         (match Hashtbl.find_opt alloc.spill_sites pass1_n with
          | Some spills ->
            List.iter
              (fun (tag, slot) ->
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            %s(&regalloc_spill[%d], t%d);\n"
                      isa.storeu_pd
                      slot
                      tag))
              spills
          | None -> ())
       | None -> ());
      (match sc.Scratch.regalloc with
       | Some alloc ->
         (match Hashtbl.find_opt alloc.reload_sites pass1_n with
          | Some reloads ->
            List.iter
              (fun (r : Regalloc.reload_decl) ->
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "        %s\n"
                      (Isa.pinned_reg_decl
                         isa
                         r.reload_name
                         r.reload_reg
                         (Printf.sprintf
                            "%s(&regalloc_spill[%d])"
                            isa.loadu_pd
                            r.reload_slot))))
              reloads
          | None -> ())
       | None -> ());
      List.iter (fun (lhs, e) -> emit_store buf lhs e) pass1_assigns;
      Buffer.add_string buf "        }\n";
      (* PASS 2 nested scope: deferred-reload emission.
       *
       * KEY INSIGHT: bulk-loading all 32 spilled values at PASS 2 top
       * forces 32 live values plus PASS 2's working set, often exceeding
       * the 32 ZMM register budget and causing GCC to re-spill internally
       * (~148 extra stack ops measured). Hand avoids this by loading each
       * spilled value just-in-time at first use.
       *
       * We replicate that here: walk PASS 2 in scheduled order, and for
       * each node, emit reloads of any not-yet-reloaded spilled predecessors
       * immediately before emitting the node itself. Each reload still
       * fires exactly once. This keeps peak live at PASS 2 manageable. *)
      Buffer.add_string buf "        {\n";
      (* PASS 2 emission: cluster by sub-DFT (when CT-decomposed), then SU
       * within each cluster. This matches Hand's structure: process sub-DFT
       * #0 fully (load 4 inputs, compute, store 4 outputs), then sub-DFT #1,
       * etc. Keeps peak live within PASS 2 around N1+working-set instead
       * of N1*N2+working-set. Removes most of the GCC re-spilling.
       *
       * Each PASS 2 sub-DFT #k2 (for k2 in 0..N2-1) consumes spill slots
       * {n1*N2 + k2 : n1 in 0..N1-1} and produces N1 outputs. We assign
       * each PASS 2 node to a cluster by computing the minimum spill slot
       * it transitively reads, then taking that mod N2.
       *
       * For non-CT (or no spill targets visible), fall back to flat SU
       * over the whole PASS 2. *)
      let cluster_of_pass2_node : (int, int) Hashtbl.t = Hashtbl.create 256 in
      if sp.ct_n2 > 0
      then (
        let min_input_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
        List.iter
          (fun e ->
             let direct_slot =
               match
                 Hashtbl.find_opt sp.re_slot e.tag, Hashtbl.find_opt sp.im_slot e.tag
               with
               | Some s, _ | _, Some s -> Some s
               | None, None -> None
             in
             let pred_min =
               List.fold_left
                 (fun acc p ->
                    match Hashtbl.find_opt min_input_slot p.tag with
                    | Some s ->
                      (match acc with
                       | None -> Some s
                       | Some a -> Some (min a s))
                    | None -> acc)
                 None
                 (preds e)
             in
             let my_min =
               match direct_slot, pred_min with
               | Some a, Some b -> Some (min a b)
               | Some a, None | None, Some a -> Some a
               | None, None -> None
             in
             match my_min with
             | Some s -> Hashtbl.add min_input_slot e.tag s
             | None -> ())
          nodes;
        List.iter
          (fun e ->
             match Hashtbl.find_opt min_input_slot e.tag with
             | Some s -> Hashtbl.add cluster_of_pass2_node e.tag (s mod sp.ct_n2)
             | None -> ())
          pass2_nodes;
        (* DIF post-multiply Twiddle Loads have no spill-slot ancestors —
         * they're consumed by Cmuls on PASS 2 outputs. Assign each
         * unclustered Pass2 Load to the cluster of its (first) consumer. *)
        let consumers_p2 : (int, t list) Hashtbl.t = Hashtbl.create 256 in
        List.iter
          (fun e ->
             List.iter
               (fun p ->
                  let prev =
                    try Hashtbl.find consumers_p2 p.tag with
                    | Not_found -> []
                  in
                  Hashtbl.replace consumers_p2 p.tag (e :: prev))
               (preds e))
          pass2_nodes;
        (* Use MIN of all consumer clusters, not first. A shared load (e.g.
        * a log3-derived twiddle base) with consumers in multiple clusters
        * MUST be assigned to the earliest consumer's cluster — otherwise
        * the concatenation `cluster_0 ++ cluster_1 ++ ...` declares the
        * load AFTER consumers in earlier clusters reference it, causing
        * use-before-decl.
        *
        * Iterate to fixpoint: clusters propagate backward through unclustered
        * chains (twiddle Load → inner Mul/Cmul → outer Fma → spill-consumer).
        * A single pass only assigns nodes whose consumers ARE already
        * clustered; chains of unclustered intermediates need multiple
        * propagation rounds. Empirically this fixes the (DIF, Fwd) and
        * (DIT, Bwd) log3 combinations that were previously falling back
        * to monolithic emit. *)
        (* Track which nodes were assigned by the FIRST WALK (have their own
        * min_input_slot) — those clusters reflect actual data dependencies
        * and must not be reduced. Nodes assigned by THIS fix are eligible
        * for reduction if a smaller consumer cluster appears later. *)
        let first_walk_assigned : (int, unit) Hashtbl.t = Hashtbl.create 256 in
        Hashtbl.iter
          (fun tag _ -> Hashtbl.add first_walk_assigned tag ())
          cluster_of_pass2_node;
        (* Iterate to fixpoint: clusters propagate backward through unclustered
         * chains (twiddle Load → derived twiddle Cmul → output-side Cmul →
         * spill-consumer). A single pass only assigns nodes whose consumers
         * ARE already clustered; chains of unclustered intermediates need
         * multiple propagation rounds AND we must allow reducing a previously-
         * assigned cluster if a smaller consumer cluster becomes available
         * later in the iteration. This fixes the (DIF, Fwd) and (DIT, Bwd)
         * log3 combinations that previously fell back to monolithic emit. *)
        let changed = ref true in
        while !changed do
          changed := false;
          List.iter
            (fun e ->
               if not (Hashtbl.mem first_walk_assigned e.tag)
               then (
                 let cs =
                   try Hashtbl.find consumers_p2 e.tag with
                   | Not_found -> []
                 in
                 let consumer_cluster =
                   List.fold_left
                     (fun acc c ->
                        match acc, Hashtbl.find_opt cluster_of_pass2_node c.tag with
                        | None, Some k -> Some k
                        | Some a, Some k -> Some (min a k)
                        | _, None -> acc)
                     None
                     cs
                 in
                 match consumer_cluster, Hashtbl.find_opt cluster_of_pass2_node e.tag with
                 | Some k, None ->
                   Hashtbl.add cluster_of_pass2_node e.tag k;
                   changed := true
                 | Some new_k, Some old_k when new_k < old_k ->
                   Hashtbl.replace cluster_of_pass2_node e.tag new_k;
                   changed := true
                 | _ -> ()))
            pass2_nodes
        done);
      let pass2_ordered =
        match scheduler with
        | SU uarch when pass2_nodes <> [] && sp.ct_n2 > 0 ->
          (* Group by cluster (k2), then SU within. *)
          let groups = Array.make sp.ct_n2 [] in
          List.iter
            (fun e ->
               match Hashtbl.find_opt cluster_of_pass2_node e.tag with
               | Some k2 -> groups.(k2) <- e :: groups.(k2)
               | None -> () (* unreachable nodes — shouldn't happen *))
            pass2_nodes;
          (* Reverse each group to restore topo order, then SU per group. *)
          let assign_tags =
            List.fold_left
              (fun acc (_, e) ->
                 Hashtbl.replace acc e.tag ();
                 acc)
              (Hashtbl.create 32)
              assigns
          in
          let result = ref [] in
          (* Emit sub-DFTs in increasing k2 order. *)
          for k2 = 0 to sp.ct_n2 - 1 do
            let group_nodes = List.rev groups.(k2) in
            let group_sinks =
              List.filter (fun e -> Hashtbl.mem assign_tags e.tag) group_nodes
            in
            let scheduled =
              if group_nodes = []
              then []
              else if group_sinks = []
              then group_nodes
              else (
                match bb_budget with
                | None ->
                  Schedule.su_schedule_subset
                    uarch
                    ~gh
                    ~subset:group_nodes
                    ~sinks:group_sinks
                | Some t ->
                  Bb.bb_schedule_subset
                    uarch
                    ~time_budget_sec:t
                    ~subset:group_nodes
                    ~sinks:group_sinks)
            in
            result := scheduled :: !result
          done;
          List.concat (List.rev !result)
        | SU uarch when pass2_nodes <> [] ->
          let assign_tags =
            List.fold_left
              (fun acc (_, e) ->
                 Hashtbl.replace acc e.tag ();
                 acc)
              (Hashtbl.create 32)
              assigns
          in
          let pass2_sinks =
            List.filter (fun e -> Hashtbl.mem assign_tags e.tag) pass2_nodes
          in
          if pass2_sinks = []
          then pass2_nodes
          else (
            match bb_budget with
            | None ->
              Schedule.su_schedule_subset uarch ~gh ~subset:pass2_nodes ~sinks:pass2_sinks
            | Some t ->
              Bb.bb_schedule_subset
                uarch
                ~time_budget_sec:t
                ~subset:pass2_nodes
                ~sinks:pass2_sinks)
        | _ -> pass2_nodes
      in
      record_peak_live "spill_pass2" pass2_ordered;
      (* Build pass2 force_last_use: for each pass2_assigns tag, find when
       * its cluster flushes — that's its real last reference.
       *
       * Cluster flushing: emit_c walks pass2_ordered; when it crosses a
       * cluster boundary (from cluster prev to cluster cur != prev), it
       * calls flush_cluster_stores prev DURING the iter of cur's first
       * node. At that moment, current_emit_position == first-pos-of-cur.
       *
       * So for output tag T in cluster c, the flush of c happens at
       * position p_flush = first-pos-of-next-cluster.
       *
       * For the LAST cluster (or unclustered outputs), the flush happens
       * at end-of-pass-iter, after current_emit_position has cycled
       * through all pass2_ordered positions. We use pass2_n (one past
       * last) as the conceptual position for these. emit_c sets
       * current_emit_position to pass2_n just before the final flush. *)
      let pass2_force_last_use : (int, int) Hashtbl.t = Hashtbl.create 16 in
      let pass2_n = List.length pass2_ordered in
      (* Walk pass2_ordered tracking cluster transitions to find each
       * cluster's flush position (= first position of next cluster). *)
      let flush_pos_for_cluster : (int, int) Hashtbl.t = Hashtbl.create 16 in
      let prev_c = ref None in
      List.iteri
        (fun i (e : t) ->
           let cur_c = Hashtbl.find_opt cluster_of_pass2_node e.tag in
           (match !prev_c, cur_c with
            | Some pc, Some cc when pc <> cc ->
              (* Transition: previous cluster pc flushes at position i. *)
              Hashtbl.replace flush_pos_for_cluster pc i
            | _ -> ());
           match cur_c with
           | Some _ -> prev_c := cur_c
           | None -> ())
        pass2_ordered;
      (* The LAST seen cluster (if any) flushes at pass2_n (after iter ends). *)
      (match !prev_c with
       | Some c when not (Hashtbl.mem flush_pos_for_cluster c) ->
         Hashtbl.replace flush_pos_for_cluster c pass2_n
       | _ -> ());
      (* Position of each node in pass2_ordered (its emission/def position).
       * Used by store-on-compute to set an output's force_last_use to its
       * own def, so the register frees immediately after the inline store. *)
      let pass2_pos_of_tag : (int, int) Hashtbl.t = Hashtbl.create 64 in
      List.iteri (fun i (e : t) -> Hashtbl.replace pass2_pos_of_tag e.tag i) pass2_ordered;
      List.iter
        (fun (_, e) ->
           if cfg.Cfg.store_on_compute && Hashtbl.mem pass2_pos_of_tag e.tag
           then
             (* Store-on-compute: last use is the inline store at the def. *)
             Hashtbl.replace
               pass2_force_last_use
               e.tag
               (Hashtbl.find pass2_pos_of_tag e.tag)
           else (
             match Hashtbl.find_opt cluster_of_pass2_node e.tag with
             | Some c ->
               (match Hashtbl.find_opt flush_pos_for_cluster c with
                | Some pos -> Hashtbl.replace pass2_force_last_use e.tag pos
                | None -> Hashtbl.replace pass2_force_last_use e.tag pass2_n)
             | None ->
               (* Unclustered: stored in the safety-net loop at end-of-pass *)
               Hashtbl.replace pass2_force_last_use e.tag pass2_n))
        pass2_assigns;
      install_alloc
        "spill_pass2"
        pass2_ordered
        (Some inline_set)
        (Some pass2_force_last_use);
      (* M5: NOW that pass 2's allocator has run, emit the regalloc_spill
       * array decl with the correct size. (Earlier I tried emitting this
       * at the top of the pass 2 block, but at that point current_regalloc
       * still held pass 1's allocation, leading to a too-small array.) *)
      (match sc.Scratch.regalloc with
       | Some alloc when alloc.num_spill_slots > 0 ->
         Buffer.add_string
           buf
           (Printf.sprintf
              "            %s regalloc_spill[%d];\n"
              isa.vec_type
              alloc.num_spill_slots)
       | _ -> ());
      (* Track which spilled tags have been reloaded. Walk pass2_ordered
       * and for each node, emit any pending reloads of its predecessors
       * before emitting the node. *)
      let reloaded : (int, unit) Hashtbl.t = Hashtbl.create 32 in
      let emit_reload_if_needed (p : t) =
        if Hashtbl.mem reloaded p.tag
        then ()
        else (
          let do_reload arr_name slot =
            Buffer.add_string
              buf
              (Printf.sprintf
                 "            const %s t%d = %s(&%s[%d]);\n"
                 isa.vec_type
                 p.tag
                 isa.loadu_pd
                 arr_name
                 slot);
            Hashtbl.add reloaded p.tag ()
          in
          match Hashtbl.find_opt sp.re_slot p.tag with
          | Some slot when not (is_fused_slot sp slot) -> do_reload "spill_re" slot
          | _ ->
            (match Hashtbl.find_opt sp.im_slot p.tag with
             | Some slot when not (is_fused_slot sp slot) -> do_reload "spill_im" slot
             | _ -> ()))
      in
      (* Transitive reload walk: when emitting a node Z, ensure reloads
       * are emitted for every spilled tag reachable through Z's
       * predecessor chain WHILE THE CHAIN IS INLINED. If X is inlined
       * into Z and X references a spilled Y, then Z's rendered body
       * (with X inlined) references t<Y> directly, so Y must be
       * reloaded before Z emits. emit_reload_if_needed is idempotent
       * (memoized via the reloaded table), so re-visits are safe. *)
      let rec reload_through_inlines (e : t) =
        emit_reload_if_needed e;
        if Hashtbl.mem inline_set e.tag then List.iter reload_through_inlines (preds e)
      in
      (* Group assigns by their PASS 2 cluster so each sub-DFT's outputs
       * can be stored immediately after its computation. This frees the
       * registers holding the outputs and reduces peak live for the rest
       * of PASS 2. *)
      let assigns_by_cluster : (int, (Expr.elem_ref * t) list) Hashtbl.t =
        Hashtbl.create 16
      in
      (* Only Pass 2 assigns can have a cluster (cluster_of_pass2_node is
       * populated from pass2_nodes), so iterating pass2_assigns is exact;
       * Pass 1 assigns were stored at the end of PASS 1 and are skipped. *)
      List.iter
        (fun ((_, e) as a) ->
           match Hashtbl.find_opt cluster_of_pass2_node e.tag with
           | Some k2 ->
             let cur =
               try Hashtbl.find assigns_by_cluster k2 with
               | Not_found -> []
             in
             Hashtbl.replace assigns_by_cluster k2 (a :: cur)
           | None -> ())
        pass2_assigns;
      let last_pass2_cluster : int option ref = ref None in
      (* Store-on-compute: map value-node tag -> output assigns it feeds, and
       * a set of tags already stored inline so cluster flush / safety net
       * skip them. (Output value nodes are sinks, so a tag feeds at most the
       * stores listed here.) *)
      let soc_assigns_by_tag : (int, (Expr.elem_ref * t) list) Hashtbl.t =
        Hashtbl.create 64
      in
      if cfg.Cfg.store_on_compute
      then
        List.iter
          (fun ((_, e) as a) ->
             let cur =
               try Hashtbl.find soc_assigns_by_tag e.tag with
               | Not_found -> []
             in
             Hashtbl.replace soc_assigns_by_tag e.tag (a :: cur))
          pass2_assigns;
      let soc_stored : (int, unit) Hashtbl.t = Hashtbl.create 64 in
      let flush_cluster_stores k2 =
        match Hashtbl.find_opt assigns_by_cluster k2 with
        | Some clist ->
          List.iter
            (fun (lhs, e) ->
               if Hashtbl.mem soc_stored e.tag
               then () (* already stored inline *)
               else (
                 emit_reload_if_needed e;
                 emit_store buf lhs e))
            (List.rev clist)
        | None -> ()
      in
      List.iteri
        (fun pos e ->
           sc.Scratch.emit_position <- pos;
           (* M5: emit spill stores BEFORE reload loads. See pass 1 for
            * the reasoning (same-position spill+reload sequencing). *)
           (match sc.Scratch.regalloc with
            | Some alloc ->
              (match Hashtbl.find_opt alloc.spill_sites pos with
               | Some spills ->
                 List.iter
                   (fun (tag, slot) ->
                      Buffer.add_string
                        buf
                        (Printf.sprintf
                           "            %s(&regalloc_spill[%d], t%d);\n"
                           isa.storeu_pd
                           slot
                           tag))
                   spills
               | None -> ())
            | None -> ());
           (* M5: emit reload declarations for this position. *)
           (match sc.Scratch.regalloc with
            | Some alloc ->
              (match Hashtbl.find_opt alloc.reload_sites pos with
               | Some reloads ->
                 List.iter
                   (fun (r : Regalloc.reload_decl) ->
                      Buffer.add_string
                        buf
                        (Printf.sprintf
                           "        %s\n"
                           (Isa.pinned_reg_decl
                              isa
                              r.reload_name
                              r.reload_reg
                              (Printf.sprintf
                                 "%s(&regalloc_spill[%d])"
                                 isa.loadu_pd
                                 r.reload_slot))))
                   reloads
               | None -> ())
            | None -> ());
           if is_inlined e
           then ()
           else (
             (* Emit reloads of any spilled predecessors not yet reloaded.
              * Walk transitively through inlined preds since their bodies
              * inline into e's expression and may reference spilled tags. *)
             List.iter reload_through_inlines (preds e);
             Buffer.add_string
               buf
               (render_node_def
               ~sc
               ~cfg
                  ~isa
                  ~in_place
                  ~t1s
                  ~twidsq
                  ~twidsq_n
                  ~strided
                  ~inline_set:(Some inline_set)
                  e);
             Buffer.add_char buf '\n');
           (* Store-on-compute: emit the store(s) for any PASS 2 output whose
            * value node is e, right after its def. The value is in t%d here
            * (e is a sink, never inlined), and force_last_use was set to this
            * position so the register frees immediately. Marked in soc_stored
            * so cluster flush / safety net skip it. *)
           if cfg.Cfg.store_on_compute
           then (
             match Hashtbl.find_opt soc_assigns_by_tag e.tag with
             | Some alist ->
               List.iter
                 (fun (lhs, ae) ->
                    emit_store buf lhs ae;
                    Hashtbl.replace soc_stored ae.tag ())
                 (List.rev alist)
             | None -> ());
           (* Cluster-boundary detection: when this node finishes a cluster
            * (all of its cluster's nodes have been emitted), flush that
            * cluster's stores immediately. We track the LAST emitted node's
            * cluster and detect the transition.
            *
            * Two-arm match: only the prev≠cur transition does work (flushing
            * the previous cluster). The "first cluster" case (prev = None)
            * and the "same cluster" case (prev = cur) both fall into the
            * unconditional update below. *)
           let cur_cluster = Hashtbl.find_opt cluster_of_pass2_node e.tag in
           (match !last_pass2_cluster, cur_cluster with
            | Some prev, Some now when prev <> now -> flush_cluster_stores prev
            | _ -> ());
           match cur_cluster with
           | Some _ -> last_pass2_cluster := cur_cluster
           | None -> ())
        pass2_ordered;
      (* M5: before the final flush, set current_emit_position to pass2_n
       * (one past last). This is the "virtual position" where end-of-pass
       * reloads and stores happen. Also emit any reload decls registered
       * at this virtual position (for spilled output tags in the last
       * cluster or unclustered). *)
      let final_pos = List.length pass2_ordered in
      sc.Scratch.emit_position <- final_pos;
      (* M5: emit spill stores for any tags evicted AT the final
       * position (i.e., during Step 3's fixed-point or post-iter
       * cascade). These must precede the reload loads so the slot
       * is initialized first. *)
      (match sc.Scratch.regalloc with
       | Some alloc ->
         (match Hashtbl.find_opt alloc.spill_sites final_pos with
          | Some spills ->
            List.iter
              (fun (tag, slot) ->
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "            %s(&regalloc_spill[%d], t%d);\n"
                      isa.storeu_pd
                      slot
                      tag))
              spills
          | None -> ())
       | None -> ());
      (match sc.Scratch.regalloc with
       | Some alloc ->
         (match Hashtbl.find_opt alloc.reload_sites final_pos with
          | Some reloads ->
            List.iter
              (fun (r : Regalloc.reload_decl) ->
                 Buffer.add_string
                   buf
                   (Printf.sprintf
                      "        %s\n"
                      (Isa.pinned_reg_decl
                         isa
                         r.reload_name
                         r.reload_reg
                         (Printf.sprintf
                            "%s(&regalloc_spill[%d])"
                            isa.loadu_pd
                            r.reload_slot))))
              reloads
          | None -> ())
       | None -> ());
      (* Flush the final cluster's stores. *)
      (match !last_pass2_cluster with
       | Some c -> flush_cluster_stores c
       | None -> ());
      (* Safety net: emit stores for Pass 2 outputs not associated with
       * any cluster. Pass 1 outputs were already stored at the end of
       * PASS 1 above; we exclusively iterate pass2_assigns here. *)
      List.iter
        (fun ((_, e) as a) ->
           if Hashtbl.mem soc_stored e.tag
           then () (* already stored inline *)
           else (
             match Hashtbl.find_opt cluster_of_pass2_node e.tag with
             | Some _ -> () (* already emitted via cluster *)
             | None ->
               emit_reload_if_needed e;
               let lhs, e = a in
               emit_store buf lhs e))
        pass2_assigns;
      Buffer.add_string buf "        }\n";
      clear_alloc () (* M3a: reset allocation at end of spill flow *)
    | None ->
      (match scheduler with
       | Topological ->
         (* Existing path: emit all definitions in topo order, then stores.
          *
          * Stage 4: when VFFT_USE_REGALLOC is set, run M3a/M5/M6 on this
          * codelet. Topological has no inlining and no duplicate-entry
          * issue (nodes is topo-sorted with each tag exactly once), but
          * we still go through prepare_for_simple_codelet for uniformity
          * and to get force_last_use construction for free. *)
         let roots = List.map snd assigns in
         let nodes = Emit_render.topo_sort_reachable roots in
         record_peak_live "topological_s1" nodes;
         let input =
           Regalloc.prepare_for_simple_codelet ~raw_scheduled:nodes ~assigns ()
         in
         install_alloc_canonical "topo_n1" input;
         emit_regalloc_spill_decl buf;
         List.iteri
           (fun pos e ->
              sc.Scratch.emit_position <- pos;
              emit_node_spill_sites buf pos;
              emit_node_reload_sites buf pos;
              Buffer.add_string
                buf
                (render_node_def ~sc ~cfg ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e);
              Buffer.add_char buf '\n')
           input.scheduled;
         (* End-of-schedule spill/reload emission. force_last_use put
          * output tags at position n=List.length input.scheduled, so the
          * allocator may have spill/reload sites at that position.
          * Mirrors the cluster-spill recipe's pass1_n handling. *)
         let n = List.length input.scheduled in
         sc.Scratch.emit_position <- n;
         emit_node_spill_sites buf n;
         emit_node_reload_sites buf n;
         Buffer.add_char buf '\n';
         List.iter (fun (lhs, e) -> emit_store buf lhs e) assigns;
         clear_alloc ()
       | Annotated_topological ->
         (* Topological order, but emitted with nested-block scopes via annotate.ml.
          * Same instructions, same order — just nested `{ ... }` to communicate
          * variable lifetimes to GCC. *)
         let roots = List.map snd assigns in
         let nodes = Emit_render.topo_sort_reachable roots in
         record_peak_live "topological_s2" nodes;
         (* Build the entry list: intermediates first (in topo order), then stores. *)
         let entries =
           List.map (fun e -> None, e) nodes
           @ List.map (fun (lhs, e) -> Some lhs, e) assigns
         in
         let render_intermediate e =
           render_node_def ~sc ~cfg ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e
         in
         let render_store oref e =
           let buf2 = Buffer.create 128 in
           emit_store buf2 oref e;
           (* emit_store added its own \n; strip the trailing \n and indent. *)
           let s = Buffer.contents buf2 in
           String.trim s
         in
         let scope = Annotate.annotate entries in
         Annotate.emit_scope isa buf render_intermediate render_store scope
       | SU uarch ->
         (* SU list scheduler: priority = (cp_dist DESC, su_num ASC).
          * Output shape: list of (oref_opt, alg_node)
          * where None = intermediate, Some oref = store.
          *
          * Single-use inlining: any intermediate with exactly one consumer
          * (in the DAG OR via output assignment) is inlined at the consumer
          * rather than emitted as a standalone declaration. This matches
          * hand-coded FFTW codelet style and significantly reduces register
          * pressure for DIF prime codelets.
          *
          * Stage 4: when VFFT_USE_REGALLOC is set, route through the
          * canonical prep. The SU scheduler's output contains duplicate
          * entries when a tag appears as both intermediate (None, e) and
          * store sink (Some oref, e) — `prepare_for_simple_codelet_from_oref`
          * dedupes. Emission walks `input.scheduled` (deduped) for defs,
          * then `assigns` for stores at end-of-scope. This matches the
          * cluster-spill recipe's def-then-stores ordering and resolves
          * the position-space ambiguity that broke M7. *)
         let scheduled_raw = Schedule.su_schedule uarch assigns in
         record_peak_live "su_s1" (List.map snd scheduled_raw);
         let inline_set = compute_inline_set ~sc assigns in
         let input =
           Regalloc.prepare_for_simple_codelet_from_oref
             ~raw_scheduled:scheduled_raw
             ~assigns
             ~inline_set:(Some inline_set)
             ()
         in
         install_alloc_canonical "su_n1" input;
         emit_regalloc_spill_decl buf;
         let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
         let is_inlined e = Hashtbl.mem inline_set e.tag in
         List.iteri
           (fun pos (e : t) ->
              sc.Scratch.emit_position <- pos;
              emit_node_spill_sites buf pos;
              emit_node_reload_sites buf pos;
              (* Skip emission for inlined values — their consumer will inline. *)
              if (not (is_inlined e)) && not (Hashtbl.mem defined e.tag)
              then (
                Hashtbl.add defined e.tag ();
                Buffer.add_string
                  buf
                  (render_node_def
               ~sc
               ~cfg
                     ~isa
                     ~in_place
                     ~t1s
                     ~twidsq
                     ~twidsq_n
                     ~strided
                     ~inline_set:(Some inline_set)
                     e);
                Buffer.add_char buf '\n'))
           input.scheduled;
         (* End-of-schedule spill/reload emission. *)
         let n = List.length input.scheduled in
         sc.Scratch.emit_position <- n;
         emit_node_spill_sites buf n;
         emit_node_reload_sites buf n;
         Buffer.add_char buf '\n';
         (* Output stores happen after the def loop, like Topological.
          * The cluster-spill recipe does the same pattern (defs in pass,
          * stores via final List.iter assigns). *)
         List.iter
           (fun (lhs, e) ->
              (* Ensure the value is defined (it should be — SU's intermediate
               * was in input.scheduled — but the defined-check is harmless). *)
              if (not (Hashtbl.mem defined e.tag)) && not (is_inlined e)
              then (
                Hashtbl.add defined e.tag ();
                Buffer.add_string
                  buf
                  (render_node_def
               ~sc
               ~cfg
                     ~isa
                     ~in_place
                     ~t1s
                     ~twidsq
                     ~twidsq_n
                     ~strided
                     ~inline_set:(Some inline_set)
                     e);
                Buffer.add_char buf '\n');
              emit_store buf lhs e)
           assigns;
         clear_alloc ()
       | Annotated_SU uarch ->
         let scheduled = Schedule.su_schedule uarch assigns in
         record_peak_live "su_s2" (List.map snd scheduled);
         let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
         let entries =
           List.filter_map
             (fun (oref_opt, e) ->
                match oref_opt with
                | None ->
                  if Hashtbl.mem defined e.tag
                  then None
                  else (
                    Hashtbl.add defined e.tag ();
                    Some (None, e))
                | Some oref -> Some (Some oref, e))
             scheduled
         in
         let render_intermediate e =
           render_node_def ~sc ~cfg ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e
         in
         let render_store oref e =
           let buf2 = Buffer.create 128 in
           emit_store buf2 oref e;
           String.trim (Buffer.contents buf2)
         in
         let scope = Annotate.annotate entries in
         Annotate.emit_scope isa buf render_intermediate render_store scope)
  in
  emit_body isa ();
  (* Strided postamble: inverse 4×4 transpose + scatter back to matrix.
   * The body has populated out_lane_re_0..radix-1 / out_lane_im_0..radix-1
   * as plain assignments. Inverse-transpose them in groups of 4 and store
   * back to matrix at row_stride. *)
  if strided && isa.vec_width = 4
  then (
    Buffer.add_string buf "\n";
    let groups = radix / 4 in
    if cfg.Cfg.strided_il_out
    then
      for g = 0 to groups - 1 do
        let j0 = g * 4 in
        let stfn = if cfg.Cfg.strided_ilo_nt then "_mm256_stream_pd" else "_mm256_storeu_pd" in
        Buffer.add_string
          buf
          (Printf.sprintf
             "        {  /* inverse 4x4 transpose + interleave: fft_idx %d..%d */\n"
             j0
             (j0 + 3));
        List.iter
          (fun suf ->
             for k = 0 to 3 do
               let base = j0 + (k / 2 * 2) in
               let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
               Buffer.add_string
                 buf
                 (Printf.sprintf
                    "            const __m256d _u%d_%s = _mm256_%s_pd(out_lane_%s_%d, \
                     out_lane_%s_%d);\n"
                    k
                    suf
                    op
                    suf
                    base
                    suf
                    (base + 1))
             done)
          [ "re"; "im" ];
        for k = 0 to 3 do
          Buffer.add_string
            buf
            (Printf.sprintf
               "            const __m256d _p%d_lo = _mm256_unpacklo_pd(_u%d_re, _u%d_im);\n"
               k
               k
               k);
          Buffer.add_string
            buf
            (Printf.sprintf
               "            const __m256d _p%d_hi = _mm256_unpackhi_pd(_u%d_re, _u%d_im);\n"
               k
               k
               k)
        done;
        for i = 0 to 3 do
          let pa = i mod 2 in
          let pb = 2 + (i mod 2) in
          let imm = if i < 2 then "0x20" else "0x31" in
          Buffer.add_string
            buf
            (Printf.sprintf
               "            %s(&out_z[2*((b+%d)*row_stride + %d) + 0], \
                _mm256_permute2f128_pd(_p%d_lo, _p%d_hi, %s));\n"
               stfn
               i
               j0
               pa
               pa
               imm);
          Buffer.add_string
            buf
            (Printf.sprintf
               "            %s(&out_z[2*((b+%d)*row_stride + %d) + 4], \
                _mm256_permute2f128_pd(_p%d_lo, _p%d_hi, %s));\n"
               stfn
               i
               j0
               pb
               pb
               imm)
        done;
        Buffer.add_string buf "        }\n"
      done
    else if cfg.Cfg.strided_r2c && not cfg.Cfg.strided_r2c_bwd
    then (
      (* M8.3: the fused conjugate-split store moved to Real (family hook). *)
      match hooks.strided_store with
      | Some f -> f buf
      | None -> failwith "emit_codelet: strided r2c requires the Real route (M8.3)")
    else
      Simd.store_transpose_4x4 ~buf ~groups);
  (* Strided postamble: inverse 8×8 transpose + scatter back to matrix
   * (AVX-512 path).
   *
   * The 8×8 transpose is its own inverse, so the postamble uses the
   * same intrinsic sequence as the preamble — Kernel C's 3 stages.
   * Inputs are out_lane_re_{j0..j0+7} / out_lane_im_{j0..j0+7} (set
   * by the body), outputs are stored at (b+i)*row_stride + j0 for
   * i=0..7. We name stage-1 unpacks (_u0_re.._u7_re) and stage-2
   * permutex2var (_v0_re.._v7_re), then fuse stage-3 shuffle_f64x2
   * inline with the 8 storeu_pd calls. *)
  if strided && isa.vec_width = 8
  then (
    Buffer.add_string buf "\n";
    let groups = radix / 8 in
    if cfg.Cfg.strided_il_out
    then
      for g = 0 to groups - 1 do
        let j0 = g * 8 in
        let stfn = if cfg.Cfg.strided_ilo_nt then "_mm512_stream_pd" else "_mm512_storeu_pd" in
        Buffer.add_string
          buf
          (Printf.sprintf
             "        {  /* inverse 8x8 transpose + interleave: fft_idx %d..%d */\n"
             j0
             (j0 + 7));
        List.iter
          (fun suf ->
             for k = 0 to 7 do
               let base = j0 + (k / 2 * 2) in
               let op = if k mod 2 = 0 then "unpacklo" else "unpackhi" in
               Buffer.add_string
                 buf
                 (Printf.sprintf
                    "            const __m512d _u%d_%s = _mm512_%s_pd(out_lane_%s_%d, \
                     out_lane_%s_%d);\n"
                    k
                    suf
                    op
                    suf
                    base
                    suf
                    (base + 1))
             done;
             for k = 0 to 7 do
               let ua = (k mod 4 mod 2) + (k / 4 * 4) in
               let ub = ua + 2 in
               let idx = if k mod 4 < 2 then "_tp_idx_lo" else "_tp_idx_hi" in
               Buffer.add_string
                 buf
                 (Printf.sprintf
                    "            const __m512d _v%d_%s = _mm512_permutex2var_pd(_u%d_%s, \
                     %s, _u%d_%s);\n"
                    k
                    suf
                    ua
                    suf
                    idx
                    ub
                    suf)
             done)
          [ "re"; "im" ];
        for i = 0 to 7 do
          let va = if i < 4 then i else i - 4 in
          let vb = if i < 4 then i + 4 else i in
          let imm = if i < 4 then "0x44" else "0xEE" in
          Buffer.add_string
            buf
            (Printf.sprintf
               "            const __m512d _r%d_re = _mm512_shuffle_f64x2(_v%d_re, \
                _v%d_re, %s);\n"
               i
               va
               vb
               imm);
          Buffer.add_string
            buf
            (Printf.sprintf
               "            const __m512d _r%d_im = _mm512_shuffle_f64x2(_v%d_im, \
                _v%d_im, %s);\n"
               i
               va
               vb
               imm);
          Buffer.add_string
            buf
            (Printf.sprintf
               "            %s(&out_z[2*((b+%d)*row_stride + %d) + 0], \
                _mm512_permutex2var_pd(_r%d_re, _il_idx_e, _r%d_im));\n"
               stfn
               i
               j0
               i
               i);
          Buffer.add_string
            buf
            (Printf.sprintf
               "            %s(&out_z[2*((b+%d)*row_stride + %d) + 8], \
                _mm512_permutex2var_pd(_r%d_re, _il_idx_o, _r%d_im));\n"
               stfn
               i
               j0
               i
               i)
        done;
        Buffer.add_string buf "        }\n"
      done
    else if cfg.Cfg.strided_r2c && not cfg.Cfg.strided_r2c_bwd
    then (
      (* M8.3: the fused conjugate-split store moved to Real (family hook). *)
      match hooks.strided_store with
      | Some f -> f buf
      | None -> failwith "emit_codelet: strided r2c requires the Real route (M8.3)")
    else
      Simd.store_transpose_8x8 ~buf ~groups);
  Buffer.add_string buf "    }\n";
  if anyk_tail
  then (
    (* Rem-aware hybrid tail — THE CONTRACT (docs/roadmap/arbitrary_k_scalartail_
     * experiment.md): the bulk loop stopped at the last full vector; cover the
     * 1..VW-1 leftover lanes with
     *   rem == 1 -> ONE scalar single lane (SSE-1-wide, the measured-cheapest +1 case)
     *   rem >= 2 -> ONE masked vector pass (flat cost, holds the MKL margin).
     * This holds for EVERY codelet, monolithic AND composite: the scalar pass
     * renders the DAG monolithically at width 1 (force_mono) — a single lane has no
     * register pressure, so the CT spill scratch is simply not referenced (no
     * __m256d-vs-double clash). The masked pass keeps the codelet's normal spill
     * recipe (avx2, full-width scratch via the raw isa.storeu_pd field). Broadcast
     * twiddles / constants are lane-independent and stay unmasked. In-place safe:
     * the masked pass touches only lanes [var, var+rem) = [bound-rem, bound),
     * disjoint from the bulk; masked-off lanes never touch memory. `bound`/`var` are
     * `me`/`k` for in-place, `K`/`k` for r2r/trig, `vl`/`v` for the real-FFT cascade. *)
    (* Hoisted trig consts are at function scope (__m256d); clear the skip-set so the
     * scalar (double) and masked (__m256d) tail passes re-emit them inline at the
     * right width, shadowing the function-scope ones. No-op for the hoist-off c2c
     * tree (sc.Scratch.hoisted_const_tags already empty there). *)
    Hashtbl.reset sc.Scratch.hoisted_const_tags;
    Buffer.add_string buf (Printf.sprintf "    if (%s < %s) {\n" tail_var tail_bound);
    Buffer.add_string
      buf
      (Printf.sprintf "        const size_t rem = %s - %s;\n" tail_bound tail_var);
    Buffer.add_string buf "        if (rem == 1) {\n";
    sc.Scratch.ls_mode <- Isa.LS_vector;
    emit_body ~force_mono:true Isa.scalar ();
    Buffer.add_string buf "        } else {\n";
    if isa.vec_width = 8
    then (
      (* avx512: masked pass (vmaskz/mask_storeu full-rate on SKX+). AVX-512 keeps the
       * masked tail — its remainder is up to 7 lanes (SSE2's width 2 can't fit it) and
       * vmaskz is not the slow vmaskmov that the avx2 SSE2 tail works around. *)
      Buffer.add_string
        buf
        "            const __mmask8 _m = (__mmask8)((1u << rem) - 1u);\n";
      sc.Scratch.ls_mode <- Isa.LS_masked "_m";
      emit_body isa ();
      sc.Scratch.ls_mode <- Isa.LS_vector)
    else (
      (* avx2 SSE2 remainder: width-2 unmasked loop over the rem lanes + a scalar STORE
       * for an odd straggler. Robustly beats masked vmaskmov at BOTH rem=2 (~-35%) and
       * rem=3 (~-12%, K=7 ~95% win-rate, tight-interleaved) — even the 2-pass
       * SSE2+scalar is faster than one vmaskmov pass on Raptor Lake, so avx2 carries no
       * masked tail at all (no _vfft_masklo table). The width-2 body renders
       * monolithically (force_mono) so composite codelets don't reference the __m256d
       * spill at width 2. The straggler MUST be scalar (a 2-wide _mm_storeu_pd would
       * write one lane past `bound`). *)
      sc.Scratch.ls_mode <- Isa.LS_vector;
      Buffer.add_string
        buf
        (Printf.sprintf
           "            for (; %s + 2 <= %s; %s += 2) {\n"
           tail_var
           tail_bound
           tail_var);
      Hashtbl.reset sc.Scratch.hoisted_const_tags;
      emit_body ~force_mono:true Isa.sse2 ();
      Buffer.add_string buf "            }\n";
      Buffer.add_string
        buf
        (Printf.sprintf "            if (%s < %s) {\n" tail_var tail_bound);
      Hashtbl.reset sc.Scratch.hoisted_const_tags;
      emit_body ~force_mono:true Isa.scalar ();
      Buffer.add_string buf "            }\n");
    Buffer.add_string buf "        }\n";
    Buffer.add_string buf "    }\n");
  (* M8.3: the hc_ranged pointer-advance trailer moved to Real (family hook). *)
  (match hooks.trailer with
   | Some f -> f buf
   | None ->
     if cfg.Cfg.hc_ranged
     then failwith "emit_codelet: hc_ranged requires the Real route (M8.3)");
  if strided && cfg.Cfg.strided_ilo_nt then Buffer.add_string buf "    _mm_sfence();\n";
  Buffer.add_string buf "}\n";
  Buffer.add_string
    buf
    (codelet_metadata
       ~isa
       ~spill
       ~tw_broadcast:(t1s || twidsq)
       ~peak_live:!max_pass_peak
       assigns);
  (* === SCHEDULE WISDOM TRAILER ===
   * Spliced from Schedule.injection_log (written at the actual
   * injection points during scheduling above — single source of truth,
   * same rule as the provenance header). Emitted as a trailer rather
   * than in the header because the header is buffered BEFORE
   * scheduling runs; a header claim would be intent, not fact. Absent
   * when no injection machinery fired, so default output is
   * byte-identical. Comments only: object code unchanged. *)
  (match !Schedule.injection_log with
   | [] -> ()
   | l ->
     Buffer.add_string
       buf
       "/* ===================== SCHEDULE WISDOM =====================\n";
     List.iter (fun s -> Buffer.add_string buf (" * " ^ s ^ "\n")) (List.rev l);
     Buffer.add_string
       buf
       " * ====================================================== */\n");
  (* M4: VFFT_ABI_XCHECK=1 — retained as a permanent debug env.  During
     phase 1 this compared the LEGACY ladder's emission against Abi over the
     full corpus (clean, with a sabotage positive-control).  Post-ladder it
     self-checks the buffer's first signature against a fresh Abi render —
     guarding against any future non-Abi signature writer. *)
  (if Sys.getenv_opt "VFFT_ABI_XCHECK" = Some "1"
   then (
     let want =
       Abi.signature (Abi.make ~symbol:name ~target_attr:isa.target_attr abi_shape)
     in
     let text = Buffer.contents buf in
     let needle = "__attribute__((target" in
     let nl = String.length needle in
     let rec find i =
       if i + nl > String.length text
       then None
       else if String.sub text i nl = needle
       then Some i
       else find (i + 1)
     in
     let got =
       match find 0 with
       | None -> None
       | Some i ->
         let stop_pat = ")" ^ String.make 1 (Char.chr 10) ^ "{" in
         let rec fb j =
           if j + 3 > String.length text
           then None
           else if String.sub text j 3 = stop_pat
           then Some (j + 3)
           else fb (j + 1)
         in
         (match fb i with
          | None -> None
          | Some stop ->
            let stop =
              if stop < String.length text && text.[stop] = Char.chr 10
              then stop + 1
              else stop
            in
            Some (String.sub text i (stop - i)))
     in
     match got with
     | Some g when g = want -> ()
     | Some g ->
       failwith
         (Printf.sprintf
            "VFFT_ABI_XCHECK MISMATCH for %s (non-Abi signature writer?)\n--- buffer ---\n%s--- Abi ---\n%s"
            name
            g
            want)
     | None -> failwith ("VFFT_ABI_XCHECK: no signature found in output of " ^ name)));
  Buffer.contents buf
;;

(* ── M2 OOP UnitLeg edge helpers ──────────────────────────────────────
   codelet_oop.ml's UnitLeg load/store edges delegate here. The UnitLeg
   pattern reuses the AOS<->SOA transpose preamble that currently lives
   inlined inside emit_codelet (hardcoded rio_ prefix). Factoring that
   out parameterized over buffer/stride names is the unfinished M2
   phase-2 extraction. The UnitGroup edges (K-batched, the executor's
   registry ABI) are fully implemented inline in codelet_oop.ml and do
   NOT call these, so generating UnitGroup OOP codelets works today.
   These guard the UnitLeg (Bailey 2D strided) path until extracted. *)
(* M8.2: the two emit_strided_*_preamble/postamble `failwith` stubs that
   stood here were DELETED — zero callers repo-wide (the M7 row's corrected
   finding: 22 UnitLeg codelets ship and reproduce without them).  The no-op
   below is the one survivor: codelet_oop calls it on the UnitLeg path. *)
let emit_avx512_transpose_indices (_isa : Isa.t) (_buf : Buffer.t) : unit = ()
