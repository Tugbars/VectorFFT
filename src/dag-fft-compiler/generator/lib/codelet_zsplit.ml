(* codelet_zsplit.ml — pipeline-hosted zil BLOCK-SPLIT split family.
 *
 * The port of codelet_zil.ml's raw-template split kinds onto the production
 * DAG pipeline (docs/roadmap/zil_pipeline_port.md). Where the legacy module
 * emits hand-written C strings, this one derives the SAME kernels through:
 *
 *   Dft.dft_expand_twiddled (math layer, elem-ref space)
 *     -> Pipeline.prepare_codelet (hash-cons + algsimp/FMA cascade)
 *     -> Schedule.su_schedule (SR list scheduler over the DAG)
 *     -> Emit_c.render_node_def (Isa-parameterized rendering)
 *
 * with zsplit-specific EDGES (block-split plane loads/stores) and TWIDDLE
 * RENDERING (Emit_state.current_tw_zsplit: [c×VW][s×VW] records in tw_re;
 * the z ABI's tw_im slot is dead).
 *
 * ABI: the frozen 11-arg z ABI of zsplit.h — function names, parameter
 * names, and (void) silencers match codelet_zil byte-for-byte so emitted
 * files are drop-in replacements in codelets/zil/avx2/.
 *
 * TIER GATE: the split family is radix 4/8 ONLY and monolithic BY DESIGN
 * (16 planes fit the ymm file; "r16 split = 32 live planes, spills" —
 * codelet_zil.ml). We therefore do NOT consult Dft.should_spill here:
 * its n>=5 clause would put R=8 on the spill recipe and pay stack traffic
 * the legacy kernels don't have (R1 op-census parity would fail).
 *
 * P1 scope (zil_pipeline_port.md §8): kind "ms" (in-place split mid,
 * DIT pre-twiddle, splat-pair records) and its bwd twin "msb" (IDFT +
 * POST-twiddle; the runtime table twspb is ALREADY conjugated, so the
 * math layer runs with ~table_conj:true — see the double-conj trap,
 * zil_pipeline_port.md §6.1). Later phases add s0s / msg / sterm / etc.
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_zsplit.ml — grep "MODULE CARD" for the full set)
 * ROLE: DAG-pipeline emitter for the zil block-split kinds.
 * PIPELINE: Dft -> Pipeline.prepare_codelet -> Schedule.su_schedule ->
 * zsplit edges + Emit_c.render_node_def.
 * PUBLIC SURFACE: emit_codelet (gen_main --zp-* flags).
 * DEPS: Dft, Algsimp, Pipeline, Schedule, Emit_c (render + provenance +
 * compute_inline_set + current_tw_zsplit via the Emit_state chain), Isa,
 * Uarch.
 * GOTCHA: sets Emit_c.current_tw_zsplit for the duration of emission
 * (Fun.protect-reset); no other family may leave it non-None.
 * ------------------------------------------------------------------ *)

(* ─── kind table (P1) ─────────────────────────────────────────────── *)

type zs_kind =
  { base : string (* "ms" — the C-name stem *)
  ; bwd : bool
  }

let kind_of_string (s : string) : zs_kind =
  match s with
  | "ms" -> { base = "ms"; bwd = false }
  | "msb" -> { base = "ms"; bwd = true }
  | other ->
    failwith
      (Printf.sprintf
         "codelet_zsplit: unknown kind %s (P1 supports: ms msb)"
         other)
;;

(* ─── emission ────────────────────────────────────────────────────── *)

let emit_codelet ~(kind : string) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t)
  : string
  =
  let k = kind_of_string kind in
  if radix <> 4 && radix <> 8
  then failwith "codelet_zsplit: split family is radix 4/8 only (see TIER GATE)";
  let vw = isa.Isa.vec_width in
  if vw <> 4
  then
    (* The generator side is width-parameterized, but the RUNTIME block
       geometry ([re×VW][im×VW]) is baked into zsplit.h's plan builder at
       VW=4. Lift this gate together with the zsplit.h vw parameterization
       (zil_pipeline_port.md §5). *)
    failwith "codelet_zsplit: runtime block geometry is VW=4 until zsplit.h is parameterized";
  let dir_s = if k.bwd then "bwd" else "fwd" in
  let fname = Printf.sprintf "radix%d_z_%s_%s_%s" radix k.base dir_s isa.Isa.name in
  let sign : [ `Fwd | `Bwd ] = if k.bwd then `Bwd else `Fwd in
  (* ─── math layer ──────────────────────────────────────────────────
     ms fwd: DIT pre-twiddle t1 (legs 1..R-1 cmul'd on load) — exactly
     dft_expand_twiddled's (DIT, Fwd). msb: (DIT, Bwd) = IDFT butterfly +
     POST-twiddle, with ~table_conj because twspb carries the conjugation. *)
  let raw_assigns =
    Dft.dft_expand_twiddled
      ~policy:Dft.TP_Flat
      ~direction:Dft.DIT
      ~sign
      ~table_conj:k.bwd
      radix
  in
  (* ─── prepare (shared cascade) ──────────────────────────────────── *)
  Algsimp.reset ();
  let force_fma_lift =
    try Sys.getenv "VFFT_FORCE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  let disable_fma_lift =
    try Sys.getenv "VFFT_DISABLE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  let pipe : Pipeline.prepared =
    Pipeline.prepare_codelet
      ~raw_assigns
      ~spill_markers_raw:[]
      ~spill_ct:None
      ~reassoc:(Dft.needs_reassoc radix)
      ~aggressive:false
      ~algorithm:(Dft.pick_algorithm radix)
      ~force_fma_lift
      ~disable_fma_lift
      ~build_spill_info:false
      ~fuse:0
  in
  let assigns = pipe.Pipeline.assigns in
  (* ─── schedule (SU over the whole monolithic body) ──────────────── *)
  let scheduled = Schedule.su_schedule uarch assigns in
  let inline_set = Emit_c.compute_inline_set assigns in
  (* ─── emit ────────────────────────────────────────────────────────── *)
  let buf = Buffer.create 8192 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — BLOCK-SPLIT interior family, PIPELINE-HOSTED\n\
        \ * (codelet_zsplit.ml; docs/roadmap/zil_pipeline_port.md). Scratch = 64-B\n\
        \ * [re x%d][im x%d] blocks (z addressing +%d for im; one stream per leg row).\n\
        \ * %s\n\
        \ * CONTRACT: count %% %d == 0 (%d columns per iteration).\n\
        \ * tw_re = ONE per-group splat-pair set: legs 1..R-1, %d doubles/leg\n\
        \ * [c×%d][s×%d]; no cursor (group-constant). tw_im unused.%s */\n"
       vw
       vw
       vw
       (if k.bwd
        then "ms bwd twin (IDFT + POST-twiddle; table twspb pre-conjugated -> table_conj)."
        else "ms (split mid, IN-PLACE zin==zout, SHUFFLE-FREE, splat-pair tw), fwd.")
       vw
       vw
       (2 * vw)
       vw
       vw
       (if k.bwd then " Roundtrip = N*x (no 1/N in-kernel)." else ""));
  Buffer.add_string
    buf
    (Emit_c.provenance_block
       ~family:"zsplit-pipeline"
       [ Printf.sprintf "kind=%s radix=%d dir=%s isa=%s" k.base radix dir_s isa.Isa.name
       ; "math: Dft.dft_expand_twiddled TP_Flat DIT"
         ^ (if k.bwd then " sign=Bwd table_conj=true" else " sign=Fwd")
       ; "prepare: Pipeline.prepare_codelet (monolithic, fuse=0)"
       ; "schedule: Schedule.su_schedule (SR list scheduler)"
       ]);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  Buffer.add_string
    buf
    (Printf.sprintf
       "__attribute__((target(\"%s\")))\n\
        void %s(\n\
       \    const double * __restrict__ zin,\n\
       \    const double * __restrict__ zin_unused,\n\
       \    double       * __restrict__ zout,\n\
       \    double       * __restrict__ zout_unused,\n\
       \    const double * tw_re, const double * tw_im,\n\
       \    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)\n\
        {\n\
       \    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Gs; (void)OLs; \
        (void)OGs;\n"
       isa.Isa.target_attr
       fname);
  Buffer.add_string
    buf
    (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" vw vw);
  (* ── ZBlockSplit load edge: lane_{re,im}_l from the split planes.
        Leg l's re half at zin + 2*(l*Ls + k), im half +VW. ── *)
  Buffer.add_string buf "        /* ZBlockSplit load edge */\n";
  for l = 0 to radix - 1 do
    let re_addr =
      if l = 0
      then "zin[2*(size_t)k]"
      else Printf.sprintf "zin[2*((size_t)%d*Ls + k)]" l
    in
    let im_addr =
      if l = 0
      then Printf.sprintf "zin[2*(size_t)k + %d]" vw
      else Printf.sprintf "zin[2*((size_t)%d*Ls + k) + %d]" l vw
    in
    Buffer.add_string
      buf
      (Printf.sprintf
         "        %s\n        %s\n"
         (Isa.const_decl
            isa
            (Printf.sprintf "lane_re_%d" l)
            (Isa.loadu_pd isa re_addr))
         (Isa.const_decl
            isa
            (Printf.sprintf "lane_im_%d" l)
            (Isa.loadu_pd isa im_addr)))
  done;
  (* ── SU-scheduled body. Defs in schedule order (first occurrence per
        tag); single-use tags render inline at their consumer. Twiddle
        loads render via the zsplit record mode. ── *)
  Buffer.add_string buf "        /* SU-scheduled body (pipeline) */\n";
  Fun.protect
    ~finally:(fun () -> Emit_c.current_tw_zsplit := None)
    (fun () ->
       Emit_c.current_tw_zsplit := Some "";
       let seen : (int, unit) Hashtbl.t = Hashtbl.create 256 in
       List.iter
         (fun ((_ : Expr.elem_ref option), (e : Algsimp.t)) ->
            if
              (not (Hashtbl.mem seen e.Algsimp.tag))
              && not (Hashtbl.mem inline_set e.Algsimp.tag)
            then (
              Hashtbl.replace seen e.Algsimp.tag ();
              (* render_node_def embeds its own 8-space indent *)
              Buffer.add_string
                buf
                (Emit_c.render_node_def
                   ~isa
                   ~in_place:false
                   ~t1s:false
                   ~strided:true
                   ~inline_set:(Some inline_set)
                   e);
              Buffer.add_char buf '\n'))
         scheduled);
  (* ── ZBlockSplit store edge (in place: same plane addressing). ── *)
  Buffer.add_string buf "        /* ZBlockSplit store edge */\n";
  List.iter
    (fun (lhs, (e : Algsimp.t)) ->
       let tname = Printf.sprintf "t%d" e.Algsimp.tag in
       match lhs with
       | Expr.Output (l, true) ->
         let addr =
           if l = 0
           then "zout[2*(size_t)k]"
           else Printf.sprintf "zout[2*((size_t)%d*Ls + k)]" l
         in
         Buffer.add_string
           buf
           (Printf.sprintf "        %s;\n" (Isa.storeu_pd isa addr tname))
       | Expr.Output (l, false) ->
         let addr =
           if l = 0
           then Printf.sprintf "zout[2*(size_t)k + %d]" vw
           else Printf.sprintf "zout[2*((size_t)%d*Ls + k) + %d]" l vw
         in
         Buffer.add_string
           buf
           (Printf.sprintf "        %s;\n" (Isa.storeu_pd isa addr tname))
       | _ -> failwith "codelet_zsplit: assign LHS is not Output")
    assigns;
  Buffer.add_string buf "    }\n}\n";
  Buffer.contents buf
;;
