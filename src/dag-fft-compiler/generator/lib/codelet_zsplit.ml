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
 * Kinds (zil_pipeline_port.md §8, P1–P5): ms/msb (in-place split mid),
 * msg/msgb (group-looped wrapper), s0s/s0sb (z-boundary leaf), sterm/stermb
 * (terminator), sterm2 (2-quad unroll-and-jam terminator = TWO shifted DAG
 * instances concatenated il2-style and braided by the SU scheduler, plus a
 * baseline-shaped 4-column tail). Bwd kinds run with ~table_conj because
 * the runtime tables (twspb/twqb) are ALREADY conjugated — the double-conj
 * trap, zil_pipeline_port.md §6.1.
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_zsplit.ml — grep "MODULE CARD" for the full set)
 * ROLE: DAG-pipeline emitter for the zil block-split kinds.
 * PIPELINE: Dft -> Pipeline.prepare_codelet -> Schedule.su_schedule ->
 * zsplit edges + Emit_c.render_node_def.
 * PUBLIC SURFACE: emit_codelet (gen_main --zp-* flags).
 * DEPS: Dft, Algsimp, Pipeline, Schedule, Emit_c (render + provenance +
 * compute_inline_set + current_tw_zsplit via the Emit_state chain), Isa,
 * Uarch.
 * GOTCHA 1: sets Emit_c.current_tw_zsplit for the duration of emission
 * (Fun.protect-reset); no other family may leave it non-None.
 * GOTCHA 2: sterm2 prepares TWO DAGs sequentially (main 2-instance, then
 * the 1-instance tail); each prepare calls Algsimp.reset, so the main
 * body must be FULLY rendered to the buffer before the tail's prepare.
 * ------------------------------------------------------------------ *)

(* ─── kind table ──────────────────────────────────────────────────── *)

(* Edge shapes for the column loop's memory boundary:
     E_planes — block-split planes: re at 2*(l*S+k), im +VW. Plain vector
                loads/stores, shuffle-free (the interior contract).
     E_z      — natural interleaved z at the same leg addressing: two z
                vectors per leg, DEINT on load / REINT on store (the API
                boundary; shuffles paid once per cascade).
     E_blocks — terminator col-block edge: per column, R consecutive
                complexes as R/VW [re×VW][im×VW] blocks at 2·R·(k+c);
                TR4 register transposes swap column-lane ↔ leg-index
                (load: blocks → leg-major lanes; store: the inverse).
   Each stride-using edge carries the C stride NAME ("Ls" | "OLs") —
   the leaf/mids run on Ls, the terminator's z comb on OLs. *)
type zs_edge =
  | E_planes of string
  | E_z of string
  | E_blocks

type zs_kind =
  { base : string (* "ms" | "msg" | "s0s" | "sterm" | "sterm2" — C-name stem *)
  ; bwd : bool
  ; group_loop : bool
    (* msg: emit the ms column loop as a static always_inline _zsg body
       and export a thin wrapper that walks the Gs groups in-kernel
       (bp += 2·R·Ls doubles, twg += (R-1)·2·VW). One call per stage —
       kills the per-group call overhead + trip-count mispredicts
       (z_cascade_plan §4.9991/§4.9992). *)
  ; uj2 : bool
    (* sterm2: 2-quad unroll-and-jam. The math layer concatenates TWO
       shifted instances (Input/Output +R slots, Twiddle +1 slot — quad
       B's packed w¹ record is exactly the next 2·VW doubles of the
       stream, so the render mode composes for free) and the SU scheduler
       braids them by readiness — the pipeline's answer to the hand
       template's 5 phases (z_cascade_plan §4.9993). Main loop k += 2·VW
       plus a baseline-shaped VW-column tail. Their kernels are
       bit-identical by construction; which one the cascade runs is the
       MEASURED per-cell t2q pick (§4.9994). *)
  ; twiddled : bool (* false: n1 math + (void)tw_re (s0s leaf) *)
  ; policy : Dft.twiddle_policy
    (* TP_Flat: splat-pair records per leg (mids). TP_PowW1: ONE packed
       per-column w¹ record, higher powers derived in-DAG by the
       squaring tree (terminator). *)
  ; tw_off : string
    (* C base-offset expression for the twiddle record stream: "" for
       table-start records (mids), "2*(size_t)k" for the terminator's
       column-indexed packed-w¹ stream. *)
  ; in_edge : zs_edge
  ; out_edge : zs_edge
  }

let kind_of_string (s : string) : zs_kind =
  let mid = { base = "ms"; bwd = false; group_loop = false; uj2 = false
            ; twiddled = true; policy = Dft.TP_Flat; tw_off = ""
            ; in_edge = E_planes "Ls"; out_edge = E_planes "Ls" } in
  match s with
  | "ms" -> mid
  | "msb" -> { mid with bwd = true }
  | "msg" -> { mid with base = "msg"; group_loop = true }
  | "msgb" -> { mid with base = "msg"; group_loop = true; bwd = true }
  | "s0s" ->
    { mid with base = "s0s"; twiddled = false; in_edge = E_z "Ls" }
  | "s0sb" ->
    { mid with base = "s0s"; twiddled = false; bwd = true; out_edge = E_z "Ls" }
  | "sterm" ->
    { mid with base = "sterm"; policy = Dft.TP_PowW1; tw_off = "2*(size_t)k"
    ; in_edge = E_blocks; out_edge = E_z "OLs" }
  | "stermb" ->
    { mid with base = "sterm"; bwd = true; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"; in_edge = E_z "OLs"; out_edge = E_blocks }
  | "sterm2" ->
    (* fwd only: the bwd 2-quad was REFUTED (+29..36%% kernel, §4.9993). *)
    { mid with base = "sterm2"; uj2 = true; policy = Dft.TP_PowW1
    ; tw_off = "2*(size_t)k"; in_edge = E_blocks; out_edge = E_z "OLs" }
  | other ->
    failwith
      (Printf.sprintf
         "codelet_zsplit: unknown kind %s (supported: ms msb msg msgb s0s s0sb sterm \
          stermb sterm2)"
         other)
;;

(* ─── emission ────────────────────────────────────────────────────── *)

let emit_codelet ~(kind : string) ~(radix : int) ~(isa : Isa.t) ~(uarch : Uarch.t)
  : string
  =
  let k = kind_of_string kind in
  if radix <> 4 && radix <> 8
  then failwith "codelet_zsplit: split family is radix 4/8 only (see TIER GATE)";
  if (k.base = "sterm" || k.base = "sterm2") && radix <> 8
  then failwith "codelet_zsplit: sterm/sterm2 are radix-8 only (zsplit.h chain contract)";
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
  let force_fma_lift =
    try Sys.getenv "VFFT_FORCE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  let disable_fma_lift =
    try Sys.getenv "VFFT_DISABLE_FMA_LIFT" = "1" with
    | Not_found -> false
  in
  (* ─── DAG preparation (math layer + shared cascade + SU schedule) ──
     Called once per emitted body. sterm2 calls it twice — 2-instance
     main, then 1-instance tail — see MODULE CARD GOTCHA 2.
     ms fwd: DIT pre-twiddle t1 (legs 1..R-1 cmul'd on load) — exactly
     dft_expand_twiddled's (DIT, Fwd). msb: (DIT, Bwd) = IDFT butterfly +
     POST-twiddle, with ~table_conj because twspb carries the conjugation.
     s0s/s0sb: twiddle-free n1 (the leaf pays only the z boundary).
     sterm/sterm2: TP_PowW1 squaring tree from the packed w¹ record. *)
  let prepare ~(two_inst : bool)
    : (Expr.elem_ref * Algsimp.t) list
      * (Expr.elem_ref option * Algsimp.t) list
      * (int, unit) Hashtbl.t
    =
    let base_assigns =
      if k.twiddled
      then
        Dft.dft_expand_twiddled
          ~policy:k.policy
          ~direction:Dft.DIT
          ~sign
          ~table_conj:k.bwd
          radix
      else Dft.dft_expand ~sign radix
    in
    let raw_assigns =
      if not two_inst
      then base_assigns
      else (
        (* il2-style second instance: Input/Output slots +radix, Twiddle
           slots +1 (TP_PowW1 consults slot 0 only, so instance B reads
           slot 1 = the NEXT packed record at +2·VW doubles — matching
           quad B's w¹ in the streamed table). Mirror of dft.ml's
           dft_expand_twiddled_il2, with the PowW1 twiddle shift. *)
        let shift_ref = function
          | Expr.Input (i, p) -> Expr.Input (i + radix, p)
          | Expr.Output (i, p) -> Expr.Output (i + radix, p)
          | Expr.Twiddle (t, p) -> Expr.Twiddle (t + 1, p)
        in
        let rec shift = function
          | Expr.Const c -> Expr.Const c
          | Expr.Load r -> Expr.Load (shift_ref r)
          | Expr.Neg e -> Expr.Neg (shift e)
          | Expr.Add (a, b) -> Expr.Add (shift a, shift b)
          | Expr.Sub (a, b) -> Expr.Sub (shift a, shift b)
          | Expr.Mul (a, b) -> Expr.Mul (shift a, shift b)
        in
        base_assigns @ List.map (fun (lhs, e) -> shift_ref lhs, shift e) base_assigns)
    in
    Algsimp.reset ();
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
    let scheduled = Schedule.su_schedule uarch assigns in
    let inline_set = Emit_c.compute_inline_set assigns in
    assigns, scheduled, inline_set
  in
  (* ─── emit ────────────────────────────────────────────────────────── *)
  let buf = Buffer.create 8192 in
  Buffer.add_string
    buf
    (Printf.sprintf
       "/* Auto-generated by vfft_v2 — BLOCK-SPLIT interior family, PIPELINE-HOSTED\n\
        \ * (codelet_zsplit.ml; docs/roadmap/zil_pipeline_port.md). Scratch = 64-B\n\
        \ * [re x%d][im x%d] blocks (z addressing +%d for im; one stream per leg row).\n\
        \ * %s\n\
        \ * CONTRACT: count %% %d == 0 (%s).\n\
        \ * %s%s */\n"
       vw
       vw
       vw
       ((match k.base, k.bwd with
         | "sterm2", _ ->
           "sterm2: 2-quad unroll-and-jam terminator twin (SU-braided 2-instance DAG \
            + baseline-shaped tail; bit-identical pair with sterm, per-cell t2q pick)."
         | "sterm", false ->
           "sterm (SPLIT-INPUT terminator: TR4 loads, packed w^1 squaring tree, REINT \
            drev-comb stores), fwd."
         | "sterm", true ->
           "sterm bwd twin (drev comb DEINT in, IDFT + POST conj-w^1, TR4 block stores)."
         | "s0s", false ->
           "s0s (z-in -> split-out leaf, twiddle-free, DEINT loads), fwd."
         | "s0s", true ->
           "s0s bwd twin (split-in -> natural-z-out IDFT leaf, REINT stores)."
         | "msg", false ->
           "msg (GROUP-LOOPED split mid: one call/stage, in-kernel bp/twg bumps), fwd."
         | "msg", true ->
           "msg bwd twin (group loop over IDFT+POST-tw body; table twspb pre-conjugated)."
         | _, true ->
           "ms bwd twin (IDFT + POST-twiddle; table twspb pre-conjugated -> table_conj)."
         | _, false ->
           "ms (split mid, IN-PLACE zin==zout, SHUFFLE-FREE, splat-pair tw), fwd."))
       vw
       (if k.uj2
        then Printf.sprintf "%d columns per main trip, %d-column tail" (2 * vw) vw
        else Printf.sprintf "%d columns per iteration" vw)
       (if not k.twiddled
        then "tw_re/tw_im unused (twiddle-free leaf)."
        else if k.policy = Dft.TP_PowW1
        then
          Printf.sprintf
            "tw_re = packed per-column w^1 at tw_re + 2k: [c(k..k+%d)][s(k..k+%d)]; \
             powers w^2..w^%d derived in-register (squaring tree). tw_im unused."
            (vw - 1)
            (vw - 1)
            (radix - 1)
        else
          Printf.sprintf
            "tw_re = %s: legs 1..R-1, %d doubles/leg [c×%d][s×%d]. tw_im unused."
            (if k.group_loop
             then "Gs per-group splat-pair sets, in-kernel cursor (twg bump/group)"
             else "ONE per-group splat-pair set, no cursor (group-constant)")
            (2 * vw)
            vw
            vw)
       (if k.bwd then " Roundtrip = N*x (no 1/N in-kernel)." else ""));
  Buffer.add_string
    buf
    (Emit_c.provenance_block
       ~family:"zsplit-pipeline"
       [ Printf.sprintf "kind=%s radix=%d dir=%s isa=%s" k.base radix dir_s isa.Isa.name
       ; (if k.twiddled
          then
            Printf.sprintf
              "math: Dft.dft_expand_twiddled %s DIT%s%s"
              (match k.policy with
               | Dft.TP_PowW1 -> "TP_PowW1"
               | Dft.TP_Log3 -> "TP_Log3"
               | Dft.TP_Flat -> "TP_Flat")
              (if k.bwd then " sign=Bwd table_conj=true" else " sign=Fwd")
              (if k.uj2 then " (2-instance concat, il2-style)" else "")
          else "math: Dft.dft_expand (n1)" ^ if k.bwd then " sign=Bwd" else " sign=Fwd")
       ; "prepare: Pipeline.prepare_codelet (monolithic, fuse=0)"
       ; "schedule: Schedule.su_schedule (SR list scheduler)"
       ]);
  Buffer.add_string buf "#include <immintrin.h>\n#include <stddef.h>\n\n";
  (* TR4 emission helper (E_blocks): 4 unpacks + 4 permute2f128 turning
     four column vectors into four leg/index vectors (or back). srcs/dsts
     are C variable names; dsts are declared const. *)
  let emit_tr4 ~(qid : string) (srcs : string array) (dsts : string array) : unit =
    let unlo = Isa.intr isa "unpacklo_pd"
    and unhi = Isa.intr isa "unpackhi_pd"
    and p2f = Isa.intr isa "permute2f128_pd" in
    Buffer.add_string
      buf
      (Printf.sprintf
         "        %s\n        %s\n        %s\n        %s\n"
         (Isa.const_decl isa (Printf.sprintf "_u0_%s" qid)
            (Printf.sprintf "%s(%s, %s)" unlo srcs.(0) srcs.(1)))
         (Isa.const_decl isa (Printf.sprintf "_u1_%s" qid)
            (Printf.sprintf "%s(%s, %s)" unhi srcs.(0) srcs.(1)))
         (Isa.const_decl isa (Printf.sprintf "_u2_%s" qid)
            (Printf.sprintf "%s(%s, %s)" unlo srcs.(2) srcs.(3)))
         (Isa.const_decl isa (Printf.sprintf "_u3_%s" qid)
            (Printf.sprintf "%s(%s, %s)" unhi srcs.(2) srcs.(3))));
    Buffer.add_string
      buf
      (Printf.sprintf
         "        %s\n        %s\n        %s\n        %s\n"
         (Isa.const_decl isa dsts.(0)
            (Printf.sprintf "%s(_u0_%s, _u2_%s, 0x20)" p2f qid qid))
         (Isa.const_decl isa dsts.(1)
            (Printf.sprintf "%s(_u1_%s, _u3_%s, 0x20)" p2f qid qid))
         (Isa.const_decl isa dsts.(2)
            (Printf.sprintf "%s(_u0_%s, _u2_%s, 0x31)" p2f qid qid))
         (Isa.const_decl isa dsts.(3)
            (Printf.sprintf "%s(_u1_%s, _u3_%s, 0x31)" p2f qid qid)))
  in
  (* column-c block address: base 2·R·(k+c), halves at h·2·VW, im +VW *)
  let blk_addr (buf_name : string) (c : int) (off : int) : string =
    if c = 0
    then Printf.sprintf "%s[%d*(size_t)k + %d]" buf_name (2 * radix) off
    else Printf.sprintf "%s[%d*((size_t)k + %d) + %d]" buf_name (2 * radix) c off
  in
  (* leg-addressed edge address: 2*(leg*STRIDE + k + colo), im/hi +VW.
     colo is the instance's column offset (0 for instance A). *)
  let leg_addr (buf_name : string) (leg : int) (stride : string) (colo : int)
        (plus : int)
    : string
    =
    let base =
      match leg, colo with
      | 0, 0 -> "2*(size_t)k"
      | 0, o -> Printf.sprintf "2*((size_t)k + %d)" o
      | l, 0 -> Printf.sprintf "2*((size_t)%d*%s + k)" l stride
      | l, o -> Printf.sprintf "2*((size_t)%d*%s + k + %d)" l stride o
    in
    if plus = 0
    then Printf.sprintf "%s[%s]" buf_name base
    else Printf.sprintf "%s[%s + %d]" buf_name base plus
  in
  (* ── column loop: edges + SU-scheduled body + store edge for ONE
        prepared DAG. ninst = 1 normally, 2 for the sterm2 main loop
        (slots radix..2·radix-1 are instance B = columns k+VW..k+2·VW-1).
        open_line is the C for-statement (the uj2 shell shares one
        function-scope k across the main loop and the tail). ── *)
  let emit_col_loop
        ~(open_line : string)
        ~(ninst : int)
        ((assigns, scheduled, inline_set) :
          (Expr.elem_ref * Algsimp.t) list
          * (Expr.elem_ref option * Algsimp.t) list
          * (int, unit) Hashtbl.t)
    : unit
    =
    let nslots = ninst * radix in
    Buffer.add_string buf open_line;
    (match k.in_edge with
     | E_planes s ->
       (* ── ZBlockSplit load edge: lane_{re,im}_l from the split planes.
             Leg l's re half at zin + 2*(l*S + k), im half +VW. ── *)
       Buffer.add_string buf "        /* ZBlockSplit load edge */\n";
       for sl = 0 to nslots - 1 do
         let leg = sl mod radix
         and colo = sl / radix * vw in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" sl)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s colo 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" sl)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s colo vw))))
       done
     | E_z s ->
       (* ── Z load edge (DEINT): two z vectors per leg, deinterleaved into
             the lane planes — unpacklo/hi + permute4x64 0xD8, the shuffles
             the cascade pays once at its API boundary. ── *)
       Buffer.add_string buf "        /* Z load edge (DEINT) */\n";
       let unlo = Isa.intr isa "unpacklo_pd"
       and unhi = Isa.intr isa "unpackhi_pd"
       and p44 = Isa.intr isa "permute4x64_pd" in
       for sl = 0 to nslots - 1 do
         let leg = sl mod radix
         and colo = sl / radix * vw in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n        %s\n        %s\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zl_%d" sl)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s colo 0)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_zh_%d" sl)
                 (Isa.loadu_pd isa (leg_addr "zin" leg s colo vw)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_re_%d" sl)
                 (Printf.sprintf "%s(%s(_zl_%d, _zh_%d), 0xD8)" p44 unlo sl sl))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "lane_im_%d" sl)
                 (Printf.sprintf "%s(%s(_zl_%d, _zh_%d), 0xD8)" p44 unhi sl sl)))
       done
     | E_blocks ->
       (* ── Block load edge (TR4): per column, load the R/VW block halves
             and TR4 each quad into leg-major lanes (lane = column). The
             terminator's load-side transpose; ninst=2 covers 2·VW cols. ── *)
       Buffer.add_string buf "        /* Block load edge (TR4) */\n";
       let halves = radix / vw in
       for c = 0 to (ninst * vw) - 1 do
         for h = 0 to halves - 1 do
           Buffer.add_string
             buf
             (Printf.sprintf
                "        %s\n        %s\n"
                (Isa.const_decl
                   isa
                   (Printf.sprintf "_br%d_%d" h c)
                   (Isa.loadu_pd isa (blk_addr "zin" c (h * 2 * vw))))
                (Isa.const_decl
                   isa
                   (Printf.sprintf "_bi%d_%d" h c)
                   (Isa.loadu_pd isa (blk_addr "zin" c ((h * 2 * vw) + vw)))))
         done
       done;
       for inst = 0 to ninst - 1 do
         for h = 0 to halves - 1 do
           emit_tr4
             ~qid:(Printf.sprintf "lr%d_%d" h inst)
             (Array.init 4 (fun j -> Printf.sprintf "_br%d_%d" h ((inst * vw) + j)))
             (Array.init 4 (fun j ->
                Printf.sprintf "lane_re_%d" ((inst * radix) + (h * vw) + j)));
           emit_tr4
             ~qid:(Printf.sprintf "li%d_%d" h inst)
             (Array.init 4 (fun j -> Printf.sprintf "_bi%d_%d" h ((inst * vw) + j)))
             (Array.init 4 (fun j ->
                Printf.sprintf "lane_im_%d" ((inst * radix) + (h * vw) + j)))
         done
       done);
    (* ── SU-scheduled body. Defs in schedule order (first occurrence per
          tag); single-use tags render inline at their consumer. Twiddle
          loads render via the zsplit record mode. ── *)
    Buffer.add_string buf "        /* SU-scheduled body (pipeline) */\n";
    Fun.protect
      ~finally:(fun () -> Emit_c.current_tw_zsplit := None)
      (fun () ->
         Emit_c.current_tw_zsplit := Some k.tw_off;
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
    (* per-slot output tag arrays (all edge shapes consume pairs) *)
    let re_tag = Array.make nslots (-1)
    and im_tag = Array.make nslots (-1) in
    List.iter
      (fun (lhs, (e : Algsimp.t)) ->
         match lhs with
         | Expr.Output (l, true) -> re_tag.(l) <- e.Algsimp.tag
         | Expr.Output (l, false) -> im_tag.(l) <- e.Algsimp.tag
         | _ -> failwith "codelet_zsplit: assign LHS is not Output")
      assigns;
    (match k.out_edge with
     | E_planes s ->
       Buffer.add_string buf "        /* ZBlockSplit store edge */\n";
       for sl = 0 to nslots - 1 do
         let leg = sl mod radix
         and colo = sl / radix * vw in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s;\n        %s;\n"
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s colo 0)
                 (Printf.sprintf "t%d" re_tag.(sl)))
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s colo vw)
                 (Printf.sprintf "t%d" im_tag.(sl))))
       done
     | E_z s ->
       (* ── Z store edge (REINT): permute4x64 0xD8 each plane, then
             unpacklo/hi re-interleaves back to natural z. For the
             terminator this addressing (leg-major on OLs = N/R) IS the
             digit-reversed comb — the scramble itself is plan-side. ── *)
       Buffer.add_string buf "        /* Z store edge (REINT) */\n";
       let unlo = Isa.intr isa "unpacklo_pd"
       and unhi = Isa.intr isa "unpackhi_pd"
       and p44 = Isa.intr isa "permute4x64_pd" in
       for sl = 0 to nslots - 1 do
         let leg = sl mod radix
         and colo = sl / radix * vw in
         Buffer.add_string
           buf
           (Printf.sprintf
              "        %s\n        %s\n        %s;\n        %s;\n"
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_pr_%d" sl)
                 (Printf.sprintf "%s(t%d, 0xD8)" p44 re_tag.(sl)))
              (Isa.const_decl
                 isa
                 (Printf.sprintf "_qi_%d" sl)
                 (Printf.sprintf "%s(t%d, 0xD8)" p44 im_tag.(sl)))
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s colo 0)
                 (Printf.sprintf "%s(_pr_%d, _qi_%d)" unlo sl sl))
              (Isa.storeu_pd
                 isa
                 (leg_addr "zout" leg s colo vw)
                 (Printf.sprintf "%s(_pr_%d, _qi_%d)" unhi sl sl)))
       done
     | E_blocks ->
       (* ── Block store edge (TR4 back): leg-major result vectors transpose
             to column vectors, stored as each column's R/VW block halves
             (the stermb output side). ── *)
       Buffer.add_string buf "        /* Block store edge (TR4) */\n";
       let halves = radix / vw in
       for inst = 0 to ninst - 1 do
         for h = 0 to halves - 1 do
           emit_tr4
             ~qid:(Printf.sprintf "sr%d_%d" h inst)
             (Array.init 4 (fun j ->
                Printf.sprintf "t%d" re_tag.((inst * radix) + (h * vw) + j)))
             (Array.init 4 (fun j -> Printf.sprintf "_cr%d_%d" h ((inst * vw) + j)));
           emit_tr4
             ~qid:(Printf.sprintf "si%d_%d" h inst)
             (Array.init 4 (fun j ->
                Printf.sprintf "t%d" im_tag.((inst * radix) + (h * vw) + j)))
             (Array.init 4 (fun j -> Printf.sprintf "_ci%d_%d" h ((inst * vw) + j)));
           for j = 0 to vw - 1 do
             let c = (inst * vw) + j in
             Buffer.add_string
               buf
               (Printf.sprintf
                  "        %s;\n        %s;\n"
                  (Isa.storeu_pd
                     isa
                     (blk_addr "zout" c (h * 2 * vw))
                     (Printf.sprintf "_cr%d_%d" h c))
                  (Isa.storeu_pd
                     isa
                     (blk_addr "zout" c ((h * 2 * vw) + vw))
                     (Printf.sprintf "_ci%d_%d" h c)))
           done
         done
       done);
    Buffer.add_string buf "    }\n"
  in
  (* ── the shared 11-arg z ABI signature + computed (void) list ── *)
  let uses stride =
    let edge_uses = function
      | E_planes s | E_z s -> s = stride
      | E_blocks -> false
    in
    edge_uses k.in_edge || edge_uses k.out_edge
  in
  let plain_voids =
    String.concat
      " "
      (List.map
         (fun p -> Printf.sprintf "(void)%s;" p)
         (List.concat
            [ [ "zin_unused"; "zout_unused"; "tw_im" ]
            ; (if uses "Ls" then [] else [ "Ls" ])
            ; [ "Gs" ]
            ; (if uses "OLs" then [] else [ "OLs" ])
            ; [ "OGs" ]
            ; (if k.twiddled then [] else [ "tw_re" ])
            ]))
  in
  let emit_signature () =
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
         \    %s\n"
         isa.Isa.target_attr
         fname
         plain_voids)
  in
  (if k.uj2
   then (
     (* ── sterm2: one function, shared k cursor, 2-instance main loop
           (SU-braided) + baseline-shaped VW-column tail. The two DAGs are
           prepared sequentially — main is rendered before the tail's
           Algsimp.reset (GOTCHA 2). ── *)
     emit_signature ();
     Buffer.add_string buf "    size_t k = 0;\n";
     let main_dag = prepare ~two_inst:true in
     emit_col_loop
       ~open_line:
         (Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" (2 * vw) (2 * vw))
       ~ninst:2
       main_dag;
     Buffer.add_string
       buf
       (Printf.sprintf
          "    /* ---- baseline-shaped %d-column tail (count %% %d == %d) ---- */\n"
          vw
          (2 * vw)
          vw);
     let tail_dag = prepare ~two_inst:false in
     emit_col_loop
       ~open_line:(Printf.sprintf "    for (; k + %d <= count; k += %d) {\n" vw vw)
       ~ninst:1
       tail_dag;
     Buffer.add_string buf "}\n")
   else if not k.group_loop
   then (
     (* ── plain kind: exported function wraps the column loop directly ── *)
     emit_signature ();
     let dag = prepare ~two_inst:false in
     emit_col_loop
       ~open_line:
         (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" vw vw)
       ~ninst:1
       dag;
     Buffer.add_string buf "}\n")
   else (
     (* ── msg: static always_inline body + thin group-loop wrapper.
           The body carries NO target attribute (always_inline requires the
           callee's target ⊆ caller's; it inlines into the attributed
           wrapper). Wrapper shape mirrors legacy codelet_zil byte-for-byte:
           in-place on zout (zin voided), bp += 2·R·Ls, twg += (R-1)·2·VW. ── *)
     let body_name = Printf.sprintf "_zsg%d%s_body" radix (if k.bwd then "b" else "f") in
     Buffer.add_string
       buf
       (Printf.sprintf
          "static __attribute__((always_inline)) inline void %s(\n\
          \    const double * __restrict__ zin, double * __restrict__ zout,\n\
          \    const double *tw_re, size_t Ls, size_t count)\n\
           {\n"
          body_name);
     let dag = prepare ~two_inst:false in
     emit_col_loop
       ~open_line:
         (Printf.sprintf "    for (size_t k = 0; k + %d <= count; k += %d) {\n" vw vw)
       ~ninst:1
       dag;
     Buffer.add_string buf "}\n\n";
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
          \    (void)zin; (void)zin_unused; (void)zout_unused; (void)tw_im;\n\
          \    (void)OLs; (void)OGs;\n\
          \    double *bp = zout;\n\
          \    const double *twg = tw_re;\n\
          \    for (size_t g = 0; g < Gs; g++) {\n\
          \        %s(bp, bp, twg, Ls, count);\n\
          \        bp += 2 * (size_t)%d * Ls;\n\
          \        twg += %d;\n\
          \    }\n\
           }\n"
          isa.Isa.target_attr
          fname
          body_name
          radix
          ((radix - 1) * 2 * vw))));
  Buffer.contents buf
;;
