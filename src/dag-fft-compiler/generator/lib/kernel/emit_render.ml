(* emit_render.ml — node/DAG-to-C rendering layer of the emitter.
 *
 * Everything emit_codelet composes but does not itself define:
 *   - topo_sort_reachable (tag order IS topological order here — the
 *     hashcons assigns tags bottom-up) and render_load ~sc / render_node_def,
 *     the per-node C renderers with Cmul -> mul+fma lowering and the
 *     single-use inlining machinery;
 *   - selective pinning (which NK_Mul tags to unpin so gcc can
 *     auto-fuse) and constant hoisting for the loop-dominated trig
 *     family — each with its measured rationale in place;
 *   - the spill_info type + classify_passes / cluster_split_schedule /
 *     compute_min_slot_pass1 — the PASS 1 / PASS 2 blocked-emission
 *     bookkeeping consumed by both emit_c and codelet_oop;
 *   - codelet_metadata and the provenance block (argv + env overrides
 *     stamped into every emitted file).
 *
 * Feature-local mutable state (unpin candidates, hoisted-const table,
 * provenance argv) deliberately lives here beside its feature rather
 * than in emit_state: it is set and read within one emission, not a
 * driver-facing mode.
 * ------------------------------------------------------------------
 * MODULE CARD (emit_render.ml — grep "MODULE CARD" for the full set)
 * ROLE: The render toolbox between the mode refs and emit_codelet.
 * PIPELINE: Emit_state modes -> these renderers -> emit_codelet text
 * PUBLIC SURFACE (measured): zero direct Emit_render.X references —
 * reached as Emit_c.X (codelet_oop's render_node_def x5,
 * cluster_split_schedule x5, compute_min_slot_pass1 x4 are the
 * heaviest external uses).
 * DEPS: Emit_state via include; Algsimp (open, +8), Isa(44),
 * Regalloc(5), Expr(17), Uarch(2), Bb(1).
 * ------------------------------------------------------------------
 *)
(* M6.2: `open Emit_state` removed — this module reads NO globals: config
   arrives as ~cfg (Cfg.t), scratch as ~sc (Scratch.t), both per-emission. *)
open Algsimp
open Ir  (* M1: names formerly re-exported through the chain *)

(* ── M6.1: the per-emission SCRATCH record (§11.2) — the ~10 genuinely
   mutable short-lived cells, previously process globals with THREE coexisting
   reset disciplines and a hand-maintained 9-of-66 reset list.  A FRESH one is
   created per emission by each driver, so nothing can leak across codelets in
   a warm gen_set process: all eight recorded temporal defects (D-1..D-8)
   become unrepresentable.  NOT `private` (reviewed: field assignment must
   compile; the guarantee is create-per-emission, not immutability). *)
module Scratch = struct
  type t =
    { mutable ls_mode : Isa.ls_mode
    ; mutable regalloc : Regalloc.allocation option
    ; mutable emit_position : int
    ; mutable fence_only : bool
    ; il_seen : (int, unit) Hashtbl.t
    ; il_pending : Buffer.t
    ; mutable il_stash : (int * string) option
    ; dup_barrier_tags : (int, unit) Hashtbl.t
    ; mutable unpin_candidates : (int, unit) Hashtbl.t option
    ; hoisted_const_tags : (int, unit) Hashtbl.t
    }

  let create () =
    { ls_mode = Isa.LS_vector
    ; regalloc = None
    ; emit_position = 0
    ; fence_only = false
    ; il_seen = Hashtbl.create 64
    ; il_pending = Buffer.create 256
    ; il_stash = None
    ; dup_barrier_tags = Hashtbl.create 16
    ; unpin_candidates = None
    ; hoisted_const_tags = Hashtbl.create 64
    }

  let il_reset sc =
    Hashtbl.reset sc.il_seen;
    Buffer.clear sc.il_pending;
    sc.il_stash <- None

  let il_take_pending sc =
    let s = Buffer.contents sc.il_pending in
    Buffer.clear sc.il_pending;
    s
end

(* ── M6.2: the per-emission CONFIG VIEW — the last emit_state cells, now a
   VALUE that flows FORWARD from the driver/family into the renderers.  The
   three twiddle cells were the design's flagship BACK-EDGES (a family module
   writing upstream state so a renderer could read it); [tw] is now simply a
   field the CALLER sets — codelet_oop passes Tw_perpos/Tw_linear, zsplit
   passes Tw_zsplit, emit_c passes Tw_default.  Immutable: "this is a
   decision", per §11.2. *)
module Cfg = struct
  type tw_source =
    | Tw_default
    | Tw_perpos
    | Tw_linear of int (* nlegs — the streaming-cursor leg count *)
    | Tw_zsplit of string (* the [c x VW][s x VW] record offset expr, "" = none *)

  (* projections matching the historical int / string-option read shapes *)
  let tw_linear_legs = function Tw_linear n -> n | _ -> 0
  let tw_zsplit_off = function Tw_zsplit off -> Some off | _ -> None

  type t =
    { r2r : bool
    ; r2cf : bool
    ; r2cb : bool
    ; hc_strided : bool
    ; n1_oop_strided : bool
    ; strided_il_in : bool
    ; strided_il_out : bool
    ; strided_ilo_nt : bool
    ; strided_r2c : bool
    ; strided_r2c_bwd : bool
    ; ip_il_in : bool
    ; ip_il_out : bool
    ; hc2c_natural : bool
    ; hc2c_natural_bwd : bool
    ; r2c_term : bool
    ; r2c_term_rt : bool
    ; r2c_term_ls : bool
    ; r2c_term_ls_r : int
    ; hc_ranged : bool
    ; hc_ranged_r : int
    ; hc2c_nat_r : int
    ; hc2c_nat_sstar : int
    ; store_on_compute : bool
    ; tw : tw_source
    }

  let default =
    { r2r = false; r2cf = false; r2cb = false; hc_strided = false
    ; n1_oop_strided = false; strided_il_in = false; strided_il_out = false
    ; strided_ilo_nt = false; strided_r2c = false; strided_r2c_bwd = false
    ; ip_il_in = false; ip_il_out = false; hc2c_natural = false
    ; hc2c_natural_bwd = false; r2c_term = false; r2c_term_rt = false
    ; r2c_term_ls = false; r2c_term_ls_r = 0; hc_ranged = false
    ; hc_ranged_r = 0; hc2c_nat_r = 0; hc2c_nat_sstar = 0
    ; store_on_compute = false; tw = Tw_default
    }
end


(* === Topological sort of the DAG nodes ===
 *
 * We need to emit definitions in dependency order: a node's definition
 * must come AFTER the definitions of all its operands. Since hash-consing
 * assigns tags in construction order (bottom-up), sorting by tag gives
 * a valid topological order automatically. *)
let topo_sort_reachable (roots : t list) : t list =
  let seen = Hashtbl.create 256 in
  let rec visit (e : t) =
    if not (Hashtbl.mem seen e.tag)
    then (
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
      | NK_Plus _ -> Ir.nk_plus_unreachable "emit_c.ml:33")
  in
  List.iter visit roots;
  let nodes = Hashtbl.fold (fun _ e acc -> e :: acc) seen [] in
  List.sort (fun a b -> compare a.tag b.tag) nodes
;;

(* === Render a Load operation as C ===
 *
 * Inputs and twiddles use K-strided layout: the j-th element's vector
 * is at &arr[j*K + k]. We use _mm512_loadu_pd because we don't enforce
 * alignment in the emitted code (matches FFTW's safer default).
 *
 * The `in_place` flag changes the input buffer name from `in_re/in_im`
 * to `rio_re/rio_im` (matches user's t1_dit signature) and the stride
 * variable name from `K` to `ios` (matches user's signature).
 *
 * The `t1s` flag (scalar-broadcast twiddles) changes Twiddle loads from
 * vector strided loads (`_mm512_loadu_pd(&tw_re[j*me + k])`) to scalar
 * broadcasts (`_mm512_set1_pd(tw_re[j])`). t1s is for inner CT codelets
 * where all k iterations share the same twiddle set; the bench harness
 * passes a smaller twiddle array (n-1 scalars instead of (n-1)*me). *)

(* -- ip_il_in lattice: shared z loads + deinterleave, memoized per input
      index j, emitted lazily at the first scheduled consumer of either side
      (the pending buffer is flushed as a prefix of the next rendered node
      definition, so placement follows the scheduler's own ordering).
      Widths: 8 (permutex2var against function-scope _il_de/_il_do), 4
      (unpack + permute4x64 0xD8), 2 (unpack pair), 1 (direct pair indexing,
      no memo). Masked pass (LS_masked m): maskz z loads under the
      pdep-expanded column mask. *)
let il_in_name ~(sc : Scratch.t) (isa : Isa.t) (j : int) (is_re : bool) : string =
  let w = isa.vec_width in
  if w = 1
  then Printf.sprintf "in_z[2*(%d*ios + k)%s]" j (if is_re then "" else " + 1")
  else (
    if not (Hashtbl.mem sc.il_seen j)
    then (
      Hashtbl.add sc.il_seen j ();
      let e = Printf.sprintf "%d*ios + k" j in
      let b = sc.il_pending in
      (match w, sc.ls_mode with
       | 8, Isa.LS_vector ->
         Buffer.add_string
           b
           (Printf.sprintf
              "const __m512d _ilz0_%d = _mm512_loadu_pd(&in_z[2*(%s)]);\n\
              \        const __m512d _ilz1_%d = _mm512_loadu_pd(&in_z[2*(%s) + 8]);\n\
              \        "
              j
              e
              j
              e)
       | 8, Isa.LS_masked m ->
         Buffer.add_string
           b
           (Printf.sprintf
              "const unsigned _ilm_%d = _pdep_u32((unsigned)%s, 0x5555u) * 3u;\n\
              \        const __m512d _ilz0_%d = _mm512_maskz_loadu_pd((__mmask8)_ilm_%d, \
               &in_z[2*(%s)]);\n\
              \        const __m512d _ilz1_%d = _mm512_maskz_loadu_pd((__mmask8)(_ilm_%d \
               >> 8), &in_z[2*(%s) + 8]);\n\
              \        "
              j
              m
              j
              j
              e
              j
              j
              e)
       | 4, _ ->
         Buffer.add_string
           b
           (Printf.sprintf
              "const __m256d _ilz0_%d = _mm256_loadu_pd(&in_z[2*(%s)]);\n\
              \        const __m256d _ilz1_%d = _mm256_loadu_pd(&in_z[2*(%s) + 4]);\n\
              \        "
              j
              e
              j
              e)
       | 2, _ ->
         Buffer.add_string
           b
           (Printf.sprintf
              "const __m128d _ilz0_%d = _mm_loadu_pd(&in_z[2*(%s)]);\n\
              \        const __m128d _ilz1_%d = _mm_loadu_pd(&in_z[2*(%s) + 2]);\n\
              \        "
              j
              e
              j
              e)
       | _ -> failwith "il_in_name: unsupported width");
      match w with
      | 8 ->
        Buffer.add_string
          b
          (Printf.sprintf
             "const __m512d _ilde_%d = _mm512_permutex2var_pd(_ilz0_%d, _il_de, _ilz1_%d);\n\
             \        const __m512d _ildo_%d = _mm512_permutex2var_pd(_ilz0_%d, _il_do, \
              _ilz1_%d);\n\
             \        "
             j
             j
             j
             j
             j
             j)
      | 4 ->
        Buffer.add_string
          b
          (Printf.sprintf
             "const __m256d _ilde_%d = \
              _mm256_permute4x64_pd(_mm256_unpacklo_pd(_ilz0_%d, _ilz1_%d), 0xD8);\n\
             \        const __m256d _ildo_%d = \
              _mm256_permute4x64_pd(_mm256_unpackhi_pd(_ilz0_%d, _ilz1_%d), 0xD8);\n\
             \        "
             j
             j
             j
             j
             j
             j)
      | 2 ->
        Buffer.add_string
          b
          (Printf.sprintf
             "const __m128d _ilde_%d = _mm_unpacklo_pd(_ilz0_%d, _ilz1_%d);\n\
             \        const __m128d _ildo_%d = _mm_unpackhi_pd(_ilz0_%d, _ilz1_%d);\n\
             \        "
             j
             j
             j
             j
             j
             j)
      | _ -> ());
    if is_re then Printf.sprintf "_ilde_%d" j else Printf.sprintf "_ildo_%d" j)
;;

let render_load
  ~(sc : Scratch.t)
  ~(cfg : Cfg.t)
      ~(isa : Isa.t)
      ~(in_place : bool)
      ~(t1s : bool)
      ?(twidsq = false)
      ?(twidsq_n = 0)
      ?(strided = false)
      (r : Expr.elem_ref)
  : string
  =
  (* In strided mode, Input(j, _) refers to pre-computed lane locals
   * populated by the 4×4 transpose preamble at the top of each loop iter.
   * Twiddles still go through their normal load path (n1 codelets don't
   * have inter-stage twiddles anyway). *)
  if strided
  then (
    match r with
    | Expr.Input (j, true) -> Printf.sprintf "lane_re_%d" j
    | Expr.Input (j, false) -> Printf.sprintf "lane_im_%d" j
    (* OOP twiddles. PerGroupTwiddles (t1, t1s=false): per-group vector
        twiddles tw_re[(j-1)*me + b] for leg j in [1,R); the math index is
        0-based (=leg-1) and the OOP loop var is `b` with stride `me`, so
        the address is tw_re[j*me + b]. BroadcastTwiddles (t1s=true): the
        twiddle is constant across the K batches (Stockham/CT inner stage),
        so load (R-1) scalars with a single broadcast tw_re[j] — no per-batch
        twiddle bandwidth. *)
    | Expr.Twiddle (j, true) ->
      (match (Cfg.tw_zsplit_off cfg.Cfg.tw) with
       | Some off ->
         (* zsplit record [c×VW][s×VW] in tw_re (tw_im slot dead) — see
            Emit_state.current_tw_zsplit. *)
         let idx = j * 2 * isa.vec_width in
         Isa.loadu_pd
           isa
           (if off = ""
            then Printf.sprintf "tw_re[%d]" idx
            else Printf.sprintf "tw_re[%s + %d]" off idx)
       | None ->
      if (Cfg.tw_linear_legs cfg.Cfg.tw) > 0
      then
        (* LINEAR layout (§12.4 4a): consumption-order stream, one cursor.
             Per quad base = b*NLEGS (each quad consumes NLEGS 4-vectors). *)
        Isa.loadu_pd
          ~mode:sc.ls_mode
          isa
          (Printf.sprintf "tw_re[b*%d + %d]" (Cfg.tw_linear_legs cfg.Cfg.tw) (j * isa.vec_width))
      else if (cfg.Cfg.tw = Cfg.Tw_perpos)
      then
        Isa.set1_pd_str
          isa
          (Printf.sprintf "tw_re[%d*(me/%d) + b/%d]" j isa.vec_width isa.vec_width)
      else if t1s
      then Isa.set1_pd_str isa (Printf.sprintf "tw_re[%d]" j)
      else
        (* PerGroupTwiddles: per-lane, indexed by the group var b -> maskable
             in the arbitrary-K tail (current_ls_mode). The set1 broadcasts above
             are lane-independent and stay unmasked. *)
        Isa.loadu_pd ~mode:sc.ls_mode isa (Printf.sprintf "tw_re[%d*me + b]" j))
    | Expr.Twiddle (j, false) ->
      (match (Cfg.tw_zsplit_off cfg.Cfg.tw) with
       | Some off ->
         (* zsplit: the sin half lives at +VW inside the tw_re record. *)
         let idx = (j * 2 * isa.vec_width) + isa.vec_width in
         Isa.loadu_pd
           isa
           (if off = ""
            then Printf.sprintf "tw_re[%d]" idx
            else Printf.sprintf "tw_re[%s + %d]" off idx)
       | None ->
      if (Cfg.tw_linear_legs cfg.Cfg.tw) > 0
      then
        Isa.loadu_pd
          ~mode:sc.ls_mode
          isa
          (Printf.sprintf "tw_im[b*%d + %d]" (Cfg.tw_linear_legs cfg.Cfg.tw) (j * isa.vec_width))
      else if (cfg.Cfg.tw = Cfg.Tw_perpos)
      then
        Isa.set1_pd_str
          isa
          (Printf.sprintf "tw_im[%d*(me/%d) + b/%d]" j isa.vec_width isa.vec_width)
      else if t1s
      then Isa.set1_pd_str isa (Printf.sprintf "tw_im[%d]" j)
      else Isa.loadu_pd ~mode:sc.ls_mode isa (Printf.sprintf "tw_im[%d*me + b]" j))
    | Expr.Output _ ->
      failwith "render_load: Output ref shouldn't appear as a Load source")
  else (
    let in_buf is_re =
      match in_place, is_re with
      | true, true -> "rio_re"
      | true, false -> "rio_im"
      | false, true -> if cfg.Cfg.r2r then "in" else "in_re"
      | false, false -> "in_im"
    in
    (* For twidsq codelets the OOP path uses a separate input stride `is`
     * (vs. `K` for the standard OOP path). Twiddles in twidsq codelets are
     * always broadcast across V lanes — they depend only on the inter-stage
     * (i, k) decomposition, not on the batch dim — so we treat them like
     * t1s regardless of the t1s flag's value.
     *
     * Twidsq address arithmetic decomposes the linear slot index s into
     * (row, col) = (s/n, s%n). The natural OOP row-major layout addresses
     * element (row, col) of block-batch b as:
     *
     *   in_re[row * is + col * V + b]
     *
     * where `is` is the input row stride (=n in the simplest case),
     * `V` is the vector-batch dim, and `b` is the loop variable.
     *
     * For the existing standard OOP path (no twidsq), the math layer's
     * Input(j, _) has j ∈ [0, n) and the address is `j*K + k` (slot-major
     * K-interleaved). The twidsq path preserves this convention for the
     * inner-slot dim and adds the row dim multiplied by the row stride. *)
    let stride =
      if in_place
      then "ios"
      else if twidsq
      then "is"
      else if cfg.Cfg.r2cf || cfg.Cfg.r2cb || cfg.Cfg.hc_strided || cfg.Cfg.n1_oop_strided
      then "is"
      else "K"
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
    let tw_stride = if in_place then "me" else if cfg.Cfg.hc_strided then "vl" else "K" in
    let tw_broadcast = t1s || twidsq || cfg.Cfg.r2c_term_rt || cfg.Cfg.r2c_term_ls in
    let render_input_addr j is_re =
      let buf = in_buf is_re in
      if cfg.Cfg.r2c_term_ls
      then (
        (* Input(j) j<r = col k leg j at ink[j*is_leg+v]; Input(r+j) = col m-k leg j
         * at inm[j*is_leg+v]. r is r2c_term_ls_r. *)
        let r = cfg.Cfg.r2c_term_ls_r in
        let bk = if is_re then "ink_re" else "ink_im" in
        let bm = if is_re then "inm_re" else "inm_im" in
        if j < r
        then Printf.sprintf "%s[%d*is_leg + %s]" bk j loop_var
        else Printf.sprintf "%s[%d*is_leg + %s]" bm (j - r) loop_var)
      else if cfg.Cfg.r2c_term
      then
        (* r2c_term: Input(0)=Z[k] at in_re[v]; Input(1)=Z[m] at in_re[is+v].
         * Two scratch rows, row stride `is`, vectorized over v. *)
        if j = 0
        then Printf.sprintf "%s[%s]" buf loop_var
        else Printf.sprintf "%s[is + %s]" buf loop_var
      else if twidsq && twidsq_n > 0
      then (
        let row = j / twidsq_n in
        let col = j mod twidsq_n in
        Printf.sprintf "%s[%d*%s + %d*V + %s]" buf row stride col loop_var)
      else if cfg.Cfg.hc2c_natural_bwd
      then
        (* c2r natural INITIATOR: read the SPLIT half-spectrum. Slot j<=sstar is
         * a direct row (Rp/Ip + j*isp); j>sstar is a conjugate-mirror row
         * (Rm/Im + (r-1-j)*ism). Exactly the forward terminator's OUTPUT sstar
         * map, but on the INPUT side. *)
        if j <= cfg.Cfg.hc2c_nat_sstar
        then Printf.sprintf "%s[%d*isp + %s]" (if is_re then "Rp" else "Ip") j loop_var
        else
          Printf.sprintf
            "%s[%d*ism + %s]"
            (if is_re then "Rm" else "Im")
            (cfg.Cfg.hc2c_nat_r - 1 - j)
            loop_var
      else (
        (* r2cb (section 62 / c2r cascade): split input strides. The
         * backward leaf reads the layout r2cf WRITES: re at +is_re from
         * its base, im at NEGATIVE stride from a one-past (+NK) base.
         * A single shared `is` cannot express the sign split. *)
        let stride =
          if cfg.Cfg.r2cb then if is_re then "is_re" else "is_im" else stride
        in
        Printf.sprintf "%s[%d*%s + %s]" buf j stride loop_var)
    in
    match r with
    | Expr.Input (j, true) when cfg.Cfg.ip_il_in && in_place -> il_in_name ~sc isa j true
    | Expr.Input (j, false) when cfg.Cfg.ip_il_in && in_place -> il_in_name ~sc isa j false
    | Expr.Input (j, true) ->
      Isa.loadu_pd ~mode:sc.ls_mode isa (render_input_addr j true)
    | Expr.Input (j, false) ->
      Isa.loadu_pd ~mode:sc.ls_mode isa (render_input_addr j false)
    | Expr.Twiddle (j, true) ->
      if tw_broadcast
      then Isa.set1_pd_str isa (Printf.sprintf "tw_re[%d]" j)
      else
        Isa.loadu_pd
          ~mode:sc.ls_mode
          isa
          (Printf.sprintf "tw_re[%d*%s + %s]" j tw_stride loop_var)
    | Expr.Twiddle (j, false) ->
      if tw_broadcast
      then Isa.set1_pd_str isa (Printf.sprintf "tw_im[%d]" j)
      else
        Isa.loadu_pd
          ~mode:sc.ls_mode
          isa
          (Printf.sprintf "tw_im[%d*%s + %s]" j tw_stride loop_var)
    | Expr.Output _ ->
      failwith "render_load: Output ref shouldn't appear as a Load source")
;;

(* === Render a single node's definition as C ===
 *
 * Each node becomes a `const VECTYPE t<tag> = <expr>;` declaration.
 * For Cmul nodes we expand to the underlying FMA arithmetic at emit time
 * (the math layer kept Cmul opaque to protect it from reassoc, but the
 * emitter is the right place to lower it back to vector instructions).
 *
 * Cmul.re(xr, xi, wr, wi) = xr*wr - xi*wi
 *   FMA form: vfnmadd(xi, wi, mul(xr, wr))   -- one mul + one fnmadd
 *
 * Cmul.im(xr, xi, wr, wi) = xr*wi + xi*wr
 *   FMA form: vfmadd(xr, wi, mul(xi, wr))    -- one mul + one fmadd
 *
 * Both AVX-512 and AVX2 have FMA (target attr "avx2,fma"), so the
 * pattern is the same; only the intrinsic prefix differs.
 *)
(* Maximum recursion depth for single-use inlining.
 * Each level inlines one node into its consumer's expression. Single-use
 * values form a chain only as long as their predecessor chain (each node
 * is single-use to one consumer), so depth = N bounds the inlined chain
 * length to N nodes. We pick a value high enough to handle prime DFT
 * codelets (R=17 has 6-deep FMA chains + sums); the practical concern
 * is C source readability and compiler handling of long expressions,
 * not correctness. Multi-use nodes act as natural "stop" points. *)
let inline_max_depth = 32

(* === Selective pinning (doc 56 follow-up) ===
 *
 * When M-project's register pinning is active (current_regalloc is
 * Some), every scheduled value gets emitted as:
 *
 *   register __m512d t<tag> asm("zmmK") = <body>;
 *   asm volatile ("" : "+v"(t<tag>));
 *
 * The `asm volatile` is a side-effect barrier to gcc. It prevents
 * `Add(Mul(a,b), c) → vfmadd*` auto-contraction across the barrier,
 * which kills FMA fusion on multi-use Muls that single_use lifting
 * (in algsimp's fma_lift) correctly refuses to duplicate.
 *
 * Measured impact on R=64 n1 AVX-512 hot path: under M-project,
 * gcc adds ZERO auto-fusion (asm FMA count = source FMA count).
 * Unbarriered emission gets 160 asm FMAs; barriered gets 117 (with
 * fma_lift on) or 34 (without). 43-126 FMAs lost to the barriers.
 *
 * Mechanism this set targets: NK_Mul nodes whose consumers include
 * at least one Add/Sub. For these, dropping the pin lets gcc see
 * the Mul → Add pattern and emit `vfmadd*pd` directly to the Add's
 * pinned destination register. The Mul value disappears entirely;
 * no intermediate register is needed; M-project's RA choice for
 * the Mul (which would have placed it in a specific zmm slot)
 * becomes moot.
 *
 * Conservative scope: only Muls with at least one Add/Sub consumer
 * are unpinned. Muls consumed solely by Fma/Cmul/output/Neg/Mul
 * keep their pin — those don't gain from auto-fusion and may
 * benefit from M-project's RA placement. Add/Sub/Fma/Neg/Cmul
 * nodes themselves keep their pin: they're either the *result* of
 * an FMA (already an explicit intrinsic) or values M-project needs
 * to control directly.
 *
 * Gating: VFFT_DISABLE_SELECTIVE_PIN=1 reverts to pin-everything
 * for A/B testing and as a safety belt.
 *
 * Threading: same single-threaded assumption as current_regalloc. *)
(* M6.1: cell moved into Scratch *)

(* Walk the scheduled DAG and identify NK_Mul tags that have at least
 * one direct Add/Sub consumer. Single-pass; O(nodes + edges). *)
let compute_unpin_candidates (scheduled : t list) : (int, unit) Hashtbl.t =
  (* Step 1: find tags of all NK_Mul nodes (these are the candidates) *)
  let mul_tags : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun n ->
       match n.node with
       | NK_Mul _ -> Hashtbl.replace mul_tags n.tag ()
       | _ -> ())
    scheduled;
  (* Step 2: scan Add/Sub nodes, mark any Mul operand as unpin candidate *)
  let result : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun n ->
       match n.node with
       | NK_Add (a, b) | NK_Sub (a, b) ->
         if Hashtbl.mem mul_tags a.tag then Hashtbl.replace result a.tag ();
         if Hashtbl.mem mul_tags b.tag then Hashtbl.replace result b.tag ()
       | _ -> ())
    scheduled;
  result
;;

(* ── Constant hoisting (notebook section 51) ──────────────────────
 * NK_Const nodes are loop-invariant by definition, but emitting them
 * inside the k-loop as fenced register temps forces gcc to
 * re-materialize them every iteration (measured: 18 loads/iter vs
 * python's 5+6 hoisted at dct2 N=8 — the whole 9% race deficit).
 * Hoist: render every NK_Const ONCE, unfenced, at function scope
 * BEFORE the k-loop; record the tag here so the in-loop renderers
 * emit nothing for it. Names (tN) are unchanged, arithmetic order is
 * unchanged, so outputs stay bit-exact. *)
(* M6.1: cell moved into Scratch *)

(* Gate: hoisting helps loop-dominated r2r/trig codelets (won the N=8
 * race vs the hand codelet) but TAXES spill-bound DFT kernels ~2%
 * (hoisted consts are live across the whole body and steal registers
 * from data). gen_main sets this true for the trig family only;
 * default false keeps the DFT tree byte-identical. *)
let hoist_consts_enabled : bool ref = ref false

let render_hoisted_consts ~(sc : Scratch.t) ~(isa : Isa.t) (nodes : t list) : string =
  Hashtbl.reset sc.hoisted_const_tags;
  if not !hoist_consts_enabled
  then ""
  else (
    let b = Buffer.create 256 in
    List.iter
      (fun e ->
         match e.node with
         | NK_Const c ->
           Hashtbl.replace sc.hoisted_const_tags e.tag ();
           Buffer.add_string
             b
             (Printf.sprintf
                "    %s\n"
                (Isa.const_decl
                   isa
                   (Printf.sprintf "t%d" e.tag)
                   (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c))))
         | _ -> ())
      nodes;
    Buffer.contents b)
;;

let render_node_def_core
  ~(sc : Scratch.t)
  ~(cfg : Cfg.t)
      ?(no_declarator = false)
      ?(inline_set : (int, unit) Hashtbl.t option = None)
      ?(twidsq = false)
      ?(twidsq_n = 0)
      ?(strided = false)
      ~(isa : Isa.t)
      ~(in_place : bool)
      ~(t1s : bool)
      (e : t)
  : string
  =
  if Hashtbl.mem sc.hoisted_const_tags e.tag
  then ""
  else (
    (* Name renderer: usually returns "t<tag>", but if M5 has installed a
     * name override for (current_emit_position, t.tag) and t is not the
     * tag being defined (i.e., t is an operand reference, not the LHS),
     * return the override name instead — used to point at reload
     * variables. *)
    let v t =
      let default_name () = Printf.sprintf "t%d" t.tag in
      if t.tag = e.tag
      then default_name () (* LHS: never override *)
      else (
        match sc.regalloc with
        | None -> default_name ()
        | Some alloc ->
          (match
             Hashtbl.find_opt alloc.name_overrides (sc.emit_position, t.tag)
           with
           | Some n -> n
           | None -> default_name ()))
    in
    (* Should this node be inlined into its consumer's expression? *)
    let should_inline n =
      match inline_set with
      | None -> false
      | Some tbl -> Hashtbl.mem tbl n.tag
    in
    (* Render an operand. If single-use, inline its expression recursively
     * (up to depth limit). Otherwise, just emit `t<tag>` and rely on the
     * standalone declaration (which will be emitted elsewhere). *)
    let rec render_operand depth n =
      if depth >= inline_max_depth || not (should_inline n)
      then v n
      else render_inlined depth n
    and render_inlined depth n =
      (* Recursive case: inline this node's expression. Don't inline Loads
       * (their memory operand is fine, but inlining them duplicates loads)
       * or Cmul nodes (they'd require complex parenthesization for the
       * pseudo-FMA pair semantics).
       *
       * Note: source-level FMA fusion of Add(Mul(x,y), b) → fmadd(x,y,b)
       * is NOT done here. GCC -O3 -mfma fuses these patterns automatically
       * via instruction contraction. Source-level fusion was tried and
       * found to be a wash (sometimes slightly hurt by constraining GCC's
       * variant selection — see the nearby commit history). The IR-level
       * NK_Fma node (created by `fma_lift`) IS rendered as fmadd directly;
       * that path is the only one that explicitly emits FMA intrinsics. *)
      match n.node with
      | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)
      | NK_Load _ -> v n (* don't inline loads — keep named *)
      | NK_Neg inner ->
        (match inner.node with
         | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" (-.c))
         | _ ->
           Isa.xor_pd isa (render_operand (depth + 1) inner) (Isa.set1_pd_str isa "-0.0"))
      | NK_Add (a, b) ->
        Isa.add_pd isa (render_operand (depth + 1) a) (render_operand (depth + 1) b)
      | NK_Sub (a, b) ->
        Isa.sub_pd isa (render_operand (depth + 1) a) (render_operand (depth + 1) b)
      | NK_Mul (a, b) ->
        Isa.mul_pd isa (render_operand (depth + 1) a) (render_operand (depth + 1) b)
      | NK_CmulRe _ | NK_CmulIm _ -> v n (* don't inline cmul *)
      | NK_Fma (a, b, c, neg_mul, neg_add) ->
        let ra = render_operand (depth + 1) a in
        let rb = render_operand (depth + 1) b in
        let rc = render_operand (depth + 1) c in
        (match neg_mul, neg_add with
         | false, false -> Isa.fmadd_pd isa ra rb rc
         | false, true -> Isa.fmsub_pd isa ra rb rc
         | true, false -> Isa.fnmadd_pd isa ra rb rc
         | true, true -> Isa.fnmsub_pd isa ra rb rc)
      | NK_Plus _ ->
        (* NK_Plus must be lowered to binary NK_Add/NK_Sub before the emitter
         * runs. If it reaches here, it's a bug in the lowering pass. *)
        Ir.nk_plus_unreachable "emit_c.ml render_operand"
    in
    (* Operand renderer for THIS node's body — depth=0 meaning we're already
     * inside the body of `e`, so its operands start at depth=0 (and inline up
     * to inline_max_depth from there). *)
    let op = render_operand 0 in
    let body =
      match e.node with
      | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)
      | NK_Load r -> render_load ~sc ~cfg ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided r
      | NK_Neg inner ->
        (* Neg(Const c) is a compile-time constant — emit as a single
         * broadcast of -c rather than a runtime XOR. *)
        (match inner.node with
         | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" (-.c))
         | _ -> Isa.xor_pd isa (op inner) (Isa.set1_pd_str isa "-0.0"))
      | NK_Add (a, b) -> Isa.add_pd isa (op a) (op b)
      | NK_Sub (a, b) -> Isa.sub_pd isa (op a) (op b)
      | NK_Mul (a, b) -> Isa.mul_pd isa (op a) (op b)
      | NK_CmulRe (xr, xi, wr, wi) ->
        Isa.fnmadd_pd isa (op xi) (op wi) (Isa.mul_pd isa (op xr) (op wr))
      | NK_CmulIm (xr, xi, wr, wi) ->
        Isa.fmadd_pd isa (op xr) (op wi) (Isa.mul_pd isa (op xi) (op wr))
      | NK_Fma (a, b, c, neg_mul, neg_add) ->
        (* (neg_mul ? -a*b : a*b) + (neg_add ? -c : c)
         *
         *   neg_mul=F, neg_add=F:  a*b + c       → fmadd
         *   neg_mul=F, neg_add=T:  a*b - c       → fmsub
         *   neg_mul=T, neg_add=F:  -a*b + c      → fnmadd
         *   neg_mul=T, neg_add=T:  -a*b - c      → fnmsub *)
        (match neg_mul, neg_add with
         | false, false -> Isa.fmadd_pd isa (op a) (op b) (op c)
         | false, true -> Isa.fmsub_pd isa (op a) (op b) (op c)
         | true, false -> Isa.fnmadd_pd isa (op a) (op b) (op c)
         | true, true -> Isa.fnmsub_pd isa (op a) (op b) (op c))
      | NK_Plus _ ->
        (* NK_Plus must be lowered to binary NK_Add/NK_Sub before this point.
         * If it reaches the body renderer, the lowering pass missed it. *)
        Ir.nk_plus_unreachable "emit_c.ml render_body"
    in
    if no_declarator
    then
      (* Plain assignment to a variable forward-declared at outer scope.
       * Used for spill "fused slots" — values whose lifetime crosses the
       * PASS 1 / PASS 2 boundary as register-resident SSA, so they're
       * declared once before either pass opens and assigned in PASS 1.
       * Not eligible for M3a register pinning: the variable was already
       * declared without a pin, so we just assign. *)
      Printf.sprintf "        %s = %s;" (v e) body
    else (
      (* === M3a regalloc switch ===
       *
       * If the active allocation (current_regalloc) has a Reg binding
       * for this tag, emit the barrier-pinned variant:
       *   register __m512d t<tag> asm("zmmK") = <body>;
       *   asm volatile ("" : "+v"(t<tag>));
       * Otherwise (no allocation, tag not in table, or Default), fall
       * through to the existing const-decl behavior. The Reg path gives
       * us deterministic register choice; the Default path is
       * byte-identical to pre-M3a output, which is what we want when
       * VFFT_USE_REGALLOC is unset.
       *
       * Selective pinning (doc 56 follow-up): if this node is in
       * current_unpin_candidates (NK_Mul with at least one Add/Sub
       * consumer), fall through to const_decl even when regalloc has
       * a Reg binding for it. This lets gcc auto-fuse the Mul→Add
       * pattern across what would otherwise be an asm volatile barrier.
       * The Mul disappears into the consumer's vfmadd instruction;
       * M-project's chosen register for the Mul becomes moot.
       * Override: VFFT_DISABLE_SELECTIVE_PIN=1. *)
      let selective_pin_disabled =
        try Sys.getenv "VFFT_DISABLE_SELECTIVE_PIN" = "1" with
        | Not_found -> false
      in
      let is_unpin_candidate =
        if selective_pin_disabled
        then false
        else (
          match sc.unpin_candidates with
          | None -> false
          | Some tbl -> Hashtbl.mem tbl e.tag)
      in
      (* Helper: in fence-only mode emit `register ... = expr; asm volatile(...)`;
       * otherwise emit the plain `const ... = expr;` form. *)
      let non_pinned_decl name body =
        if Hashtbl.mem sc.dup_barrier_tags e.tag
        then
          (* duplication clone (doc 65 §8): non-const + "+x" barrier or
           * gcc re-CSEs the clone back into the original at -O3. *)
          Printf.sprintf
            "%s %s = %s; __asm__ volatile(\"\" : \"+x\"(%s));"
            isa.vec_type
            name
            body
            name
        else if sc.fence_only
        then Isa.fenced_decl isa name body
        else Isa.const_decl isa name body
      in
      match sc.regalloc with
      | Some alloc when not is_unpin_candidate ->
        (match Regalloc.lookup alloc e.tag with
         | Regalloc.Reg reg_name ->
           Printf.sprintf "        %s" (Isa.pinned_reg_decl isa (v e) reg_name body)
         | Regalloc.Spilled _ ->
           (* M5: the Spilled variant is reserved but no longer used by
            * the spilling allocator (it tracks spills via spill_sites
            * separately, keeping the tag's assignment as Reg). If we
            * see Spilled here it's a future extension; fall back to
            * Default emission. *)
           Printf.sprintf "        %s" (non_pinned_decl (v e) body)
         | Regalloc.Default -> Printf.sprintf "        %s" (non_pinned_decl (v e) body))
      | Some _ ->
        (* Selective unpin: drop the pin to enable gcc auto-fusion *)
        Printf.sprintf "        %s" (non_pinned_decl (v e) body)
      | None -> Printf.sprintf "        %s" (non_pinned_decl (v e) body)))
;;

(* === Emit a complete codelet ===
 *
 * Two signatures supported:
 *
 * Out-of-place (in_place=false, the default):
 *   void NAME(in_re, in_im, out_re, out_im, tw_re, tw_im, K)
 *
 * In-place (in_place=true, matches user's hand-coded t1_dit):
 *   void NAME(rio_re, rio_im, tw_re, tw_im, ios, me)
 *   — rio_* serves as both input and output (same buffer)
 *   — ios is the stride between legs in the rio buffer
 *   — me is the batch size (= K), used as twiddle stride
 *
 * In-place is safe here because the topological sort places all input
 * loads at the top of the function body and all output stores at the
 * bottom — by the time any store fires, all loads have completed and
 * their results are in registers.
 *)
type scheduler =
  | Topological (* sort reachable nodes by tag, flat emit *)
  | Annotated_topological (* topological order + nested-block scopes (annotate.ml) *)
  | SU of Uarch.t (* Sethi-Ullman list scheduler with µarch profile *)
  | Annotated_SU of Uarch.t (* SU + nested blocks *)

(* === SINGLE-USE INLINING SET ===
 *
 * Compute the set of node tags that should be inlined at their consumer
 * rather than emitted as separate `const __m512d t<tag> = ...;`
 * declarations. Inlining matches FFTW hand-coded codelet style:
 *
 *   const __m512d t1 = _mm512_sub_pd(a, b);
 *   const __m512d t2 = _mm512_mul_pd(K, t1);
 *
 * vs the inlined form:
 *
 *   const __m512d t2 = _mm512_mul_pd(K, _mm512_sub_pd(a, b));
 *
 * Both compute the same value, but the second gives GCC a tighter SSA
 * form: t1 has no name, no scope, and its lifetime is implicit in the
 * outer expression. Empirically, hand-coded R=13 t1_dif uses ~120 nested
 * intrinsic call patterns; our linearized output uses ~24. The gap is
 * register pressure: every named intermediate is one more SSA value
 * GCC's allocator has to track. Inlining single-use values closes the
 * gap to hand parity on R=11/13/17 t1_dif (and helps DIT too).
 *
 * Criteria for inlining:
 *   - Use count is exactly 1 (the value flows to one consumer)
 *   - Not a Load (loads have memory operands; inlining duplicates them)
 *   - Not a Cmul (Cmul.re/Cmul.im share state via 2-instruction sequence)
 *   - Not a sink (sinks are output assignments)
 *
 * "Use count" = number of distinct nodes that reference this tag as a
 * predecessor PLUS 1 if the tag also appears as an output assignment
 * (the store counts as a use).
 *)
let compute_inline_set ~(sc : Scratch.t) (assigns : (Expr.elem_ref * t) list) : (int, unit) Hashtbl.t =
  let roots = List.map snd assigns in
  let nodes = topo_sort_reachable roots in
  (* Use count = how many other nodes reference this tag. *)
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let bump tag =
    let cur =
      try Hashtbl.find use_count tag with
      | Not_found -> 0
    in
    Hashtbl.replace use_count tag (cur + 1)
  in
  List.iter
    (fun n ->
       match n.node with
       | NK_Const _ | NK_Load _ -> ()
       | NK_Neg a -> bump a.tag
       | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
         bump a.tag;
         bump b.tag
       | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
         bump a.tag;
         bump b.tag;
         bump c.tag;
         bump d.tag
       | NK_Fma (a, b, c, _, _) ->
         bump a.tag;
         bump b.tag;
         bump c.tag
       | NK_Plus _ -> Ir.nk_plus_unreachable "emit_c.ml use_count walker")
    nodes;
  (* Each output assignment also counts as a use. *)
  List.iter (fun (_, e) -> bump e.tag) assigns;
  (* Sinks: tags that are direct output assignments. Don't inline these
   * — they need a named t<tag> for the store to reference. *)
  let sink_tags : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  List.iter (fun (_, e) -> Hashtbl.replace sink_tags e.tag ()) assigns;
  let result = Hashtbl.create 256 in
  List.iter
    (fun n ->
       let count =
         try Hashtbl.find use_count n.tag with
         | Not_found -> 0
       in
       let is_sink = Hashtbl.mem sink_tags n.tag in
       let kind_inlinable =
         match n.node with
         | NK_Load _ -> false (* don't duplicate loads *)
         | NK_CmulRe _ | NK_CmulIm _ -> false (* paired emit *)
         | NK_Const _ -> false (* already inlined as set1 broadcast *)
         | _ -> true
       in
       if
         count = 1
         && (not is_sink)
         && kind_inlinable
         && not (Hashtbl.mem sc.dup_barrier_tags n.tag)
         (* duplication clones MUST be declared: the "+x" barrier that
          * stops gcc re-CSEing them attaches to the declaration
          * (doc 65 §8); inlining them makes the clone a no-op. *)
       then Hashtbl.add result n.tag ())
    nodes;
  result
;;

(* M2/G8: the spill_info TYPE + make_spill_info moved DOWN to algsimp.ml
   (beside spill_tag_marker, their input) — Pipeline (L1) consumed them from
   here (L3), a cross-layer inversion that surfaced when pipeline.mli was
   written.  The render-side ALGORITHMS over spill_info (is_spilled,
   classify_passes, cluster_split_schedule, ...) stay in this module. *)

let is_spilled (sp : spill_info) (tag : int) : bool =
  Hashtbl.mem sp.re_slot tag || Hashtbl.mem sp.im_slot tag
;;

let is_fused_slot (sp : spill_info) (slot : int) : bool = Hashtbl.mem sp.fused_slots slot

(* === Intrinsic codelet metadata (emitted as a header comment) ===
 *
 * Exact structural metrics computed from the post-algsimp DAG and the
 * spill plan at generation time, so register-pressure reasoning becomes a
 * self-reported number with a regression gate instead of something
 * reverse-engineered from disassembly.
 *
 * Roofline model (Golden Cove / Sapphire Rapids; 256b AVX2 or 512b
 * AVX-512): FP issue ~2/cyc, memory ~2 ops/cyc. A codelet is memory-bound
 * even at its structural floor (which NO register allocator can fix) iff
 *   memory_floor > fp_instr.
 * Otherwise it CAN be made compute-bound, provided peak_live stays within
 * the register budget.
 *
 * The floor terms are exact: distinct input loads + output stores + the
 * cross-pass cut the decomposition forces (in-place codelets cannot
 * rematerialize loads, so the cross-pass store/reload is structural).
 * peak_live is the topological-schedule value: exact for monolithic
 * codelets, a whole-codelet upper bound for CT-decomposed ones whose true
 * per-pass register floor is the cross-pass cut. The minimum-register
 * schedule itself is NP-hard, so this is a bound, not the optimum. *)
let codelet_metadata
      ~(isa : Isa.t)
      ~(spill : spill_info option)
      ~(tw_broadcast : bool)
      ~(peak_live : int)
      (assigns : (Expr.elem_ref * t) list)
  : string
  =
  let nodes = topo_sort_reachable (List.map snd assigns) in
  let peak_live =
    if peak_live > 0
    then peak_live
    else (Regalloc.peak_live_analysis ~isa ~scheduled:nodes).peak_live
  in
  let nadd = ref 0
  and nsub = ref 0
  and nmul = ref 0
  and nfma = ref 0
  and ncmul = ref 0
  and nneg = ref 0
  and n_vload =
    ref 0 (* per-iteration vector loads: input data + (non-broadcast) twiddles *)
  and n_swload =
    ref 0
    (* loop-invariant scalar twiddle broadcasts (hoisted, ~free) *)
  in
  List.iter
    (fun n ->
       match n.node with
       | NK_Add _ -> incr nadd
       | NK_Sub _ -> incr nsub
       | NK_Mul _ -> incr nmul
       | NK_Fma _ -> incr nfma
       | NK_Neg _ -> incr nneg
       | NK_CmulRe _ | NK_CmulIm _ -> incr ncmul
       | NK_Load r ->
         (match r with
          | Expr.Input _ -> incr n_vload
          | Expr.Twiddle _ -> if tw_broadcast then incr n_swload else incr n_vload
          | Expr.Output _ ->
            incr n_vload (* defensive: Output as load source is invalid *))
       | NK_Const _ | NK_Plus _ -> ())
    nodes;
  let fp_instr = !nadd + !nsub + !nmul + !nfma + (2 * !ncmul) + !nneg in
  let flops = !nadd + !nsub + !nmul + (2 * !nfma) + (2 * !ncmul) + !nneg in
  let ess_loads = !n_vload in
  (* per-iteration vector loads only *)
  let ess_stores = List.length assigns in
  let ess_io = ess_loads + ess_stores in
  let xslots =
    match spill with
    | Some sp -> sp.num_slots
    | None -> 0
  in
  let xpass_mem = 4 * xslots
  and xpass_live = 2 * xslots in
  let memory_floor = ess_io + xpass_mem in
  let budget = isa.vec_regs - if isa.vec_regs >= 32 then 4 else 2 in
  let membound = memory_floor > fp_instr in
  let fits = peak_live <= budget in
  Printf.sprintf
    "/* codelet-metrics [intrinsic, gen-time]:\n\
    \ *   fp_instr=%d  flops=%d  (add=%d sub=%d mul=%d fma=%d cmul=%d neg=%d)\n\
    \ *   essential_io=%d ops (vec_loads=%d + stores=%d)  [+%d hoisted scalar-twiddle \
     loads, not counted]\n\
    \ *   cross_pass_cut=%d slots => +%d mem ops, %d vectors live across pass boundary\n\
    \ *   memory_floor=%d mem ops   peak_live(max-per-pass)=%d   budget=%d regs\n\
    \ *   ROOFLINE: %s at floor (memory_floor %s fp_instr)\n\
    \ *   PRESSURE: %s (peak_live %s budget)%s\n\
    \ */\n"
    fp_instr
    flops
    !nadd
    !nsub
    !nmul
    !nfma
    !ncmul
    !nneg
    ess_io
    ess_loads
    ess_stores
    !n_swload
    xslots
    xpass_mem
    xpass_live
    memory_floor
    peak_live
    budget
    (if membound then "MEMORY-BOUND" else "compute-capable")
    (if membound then ">" else "<=")
    (if fits then "fits" else "SPILLS")
    (if fits then "<=" else ">")
    (if xslots > 0
     then
       "  [CT: peak_live is max over passes; cross_pass_cut is added explicit spill \
        traffic]"
     else "")
;;

(* Is this tag's spill slot fused (kept in register, not stored)? *)
let is_fused_tag (sp : spill_info) (tag : int) : bool =
  match Hashtbl.find_opt sp.re_slot tag, Hashtbl.find_opt sp.im_slot tag with
  | Some s, _ | _, Some s -> is_fused_slot sp s
  | None, None -> false
;;

(* ── compute_min_slot_pass1 ─────────────────────────────────────────
 * Assign each PASS-1 node a min_slot (its own spill slot if it is a
 * spill target, else the minimum min_slot of its PASS-1 successors), and
 * return both the table and pass1_nodes sorted by (min_slot, tag) — the
 * pre-cluster ordering consumed by cluster_split_schedule.
 *
 * Single source for the PASS-1 min_slot computation that emit_c and
 * codelet_oop previously hand-copied. NORMALIZATION: the reverse-topo
 * walk uses an EXPLICIT descending-tag sort rather than List.rev
 * pass1_nodes. The two prior copies were equal only under the invariant
 * that pass1_nodes is ascending-tag-ordered (it is — both feed from
 * Ir.topo_sort_reachable, now itself single-sourced). The explicit
 * sort DELETES that latent dependence: this helper is correct regardless
 * of the caller's input order. (Also uses Hashtbl.replace, not add;
 * equivalent here since each tag is visited once in the reverse walk.) ─ *)
let compute_min_slot_pass1 (sp : spill_info) (pass1_nodes : t list)
  : (int, int) Hashtbl.t * t list
  =
  let lookup_slot tag =
    match Hashtbl.find_opt sp.re_slot tag with
    | Some s -> Some s
    | None -> Hashtbl.find_opt sp.im_slot tag
  in
  let pass1_set = Hashtbl.create 256 in
  List.iter (fun (e : t) -> Hashtbl.replace pass1_set e.tag ()) pass1_nodes;
  let succs : (int, int list) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun (e : t) ->
       List.iter
         (fun (p : t) ->
            if Hashtbl.mem pass1_set p.tag
            then (
              let cur =
                try Hashtbl.find succs p.tag with
                | Not_found -> []
              in
              Hashtbl.replace succs p.tag (e.tag :: cur)))
         (preds e))
    pass1_nodes;
  let min_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
  (* Reverse topological order = descending tag (hash-cons tags are
     construction-ordered). Explicit sort, not List.rev — see header. *)
  let pass1_rev = List.sort (fun (a : t) b -> compare b.tag a.tag) pass1_nodes in
  List.iter
    (fun (e : t) ->
       let my =
         match lookup_slot e.tag with
         | Some s -> Some s
         | None ->
           let s_tags =
             try Hashtbl.find succs e.tag with
             | Not_found -> []
           in
           let s_mins = List.filter_map (fun t -> Hashtbl.find_opt min_slot t) s_tags in
           (match s_mins with
            | [] -> None
            | _ -> Some (List.fold_left min max_int s_mins))
       in
       match my with
       | Some s -> Hashtbl.replace min_slot e.tag s
       | None -> ())
    pass1_rev;
  let pass1_blocked_topo =
    List.sort
      (fun (a : t) b ->
         let ma =
           try Hashtbl.find min_slot a.tag with
           | Not_found -> max_int
         in
         let mb =
           try Hashtbl.find min_slot b.tag with
           | Not_found -> max_int
         in
         let c = compare ma mb in
         if c <> 0 then c else compare a.tag b.tag)
      pass1_nodes
  in
  min_slot, pass1_blocked_topo
;;

(* ── cluster_split_schedule ─────────────────────────────────────────
 * PASS-1 cluster-local scheduling: split a min_slot-ordered node list
 * into maximal same-cluster runs and schedule each run independently.
 *
 * Cluster boundary: for CT(N1,N2), cluster k owns spill slots
 * [k*N2, (k+1)*N2 - 1], so a node's cluster = min_slot / ct_n2.
 * Sub-FFTs are mutually independent (CT property: different n1_idx read
 * disjoint input cells), so reordering WITHIN a cluster is safe — it
 * cannot cross a cluster boundary (no dependency edges to cross), and
 * constants are pre-hoisted out of the node list.
 *
 * Single source of truth for the PASS-1 splitter that emit_c (in-place)
 * and codelet_oop (OOP) previously hand-copied verbatim. The one real
 * divergence between the two callers — which per-cluster scheduler to
 * run (su_schedule_subset vs Bb.bb_schedule_subset by time budget) — is
 * the `schedule_cluster` closure each caller supplies. cluster_of is
 * deliberately NOT parameterized (kept inline as min_slot/ct_n2); when
 * the blocked-newsplit/SR work needs non-uniform cluster ranges, that
 * becomes a one-place change made with the real requirement in hand.
 *
 * The ct_n2 <= 0 guard (non-CT — shouldn't fire for R>=25, all CT) lives
 * INSIDE the helper so no caller (present or future) can forget it. ─ *)
let cluster_split_schedule
      (sp : spill_info)
      ~(pass1_blocked_topo : t list)
      ~(min_slot : (int, int) Hashtbl.t)
      ~(schedule_cluster : subset:t list -> sinks:t list -> t list)
  : t list
  =
  if sp.ct_n2 <= 0
  then pass1_blocked_topo
  else (
    let cluster_of_node (e : t) =
      match Hashtbl.find_opt min_slot e.tag with
      | Some s -> s / sp.ct_n2
      | None -> sp.ct_n1 (* unreachable → fake last cluster *)
    in
    (* Walk pass1_blocked_topo, split into contiguous same-cluster runs. *)
    let groups : (int * t list) list =
      let rec go acc cur_cluster cur_acc = function
        | [] ->
          (match cur_acc with
           | [] -> List.rev acc
           | _ -> List.rev ((cur_cluster, List.rev cur_acc) :: acc))
        | n :: rest ->
          let k = cluster_of_node n in
          if k = cur_cluster
          then go acc cur_cluster (n :: cur_acc) rest
          else (
            let acc' =
              match cur_acc with
              | [] -> acc
              | _ -> (cur_cluster, List.rev cur_acc) :: acc
            in
            go acc' k [ n ] rest)
      in
      match pass1_blocked_topo with
      | [] -> []
      | n :: rest -> go [] (cluster_of_node n) [ n ] rest
    in
    List.concat_map
      (fun (_cluster_id, group_nodes) ->
         let cluster_sinks =
           List.filter
             (fun (e : t) -> Hashtbl.mem sp.re_slot e.tag || Hashtbl.mem sp.im_slot e.tag)
             group_nodes
         in
         if cluster_sinks = []
         then group_nodes
         else schedule_cluster ~subset:group_nodes ~sinks:cluster_sinks)
      groups)
;;

(* Split a topologically-ordered list of nodes into PASS 1 and PASS 2.
 *
 * A node is PASS 2 iff it transitively depends on a spilled tag (i.e.,
 * some pred or pred-of-pred etc. is in the spill_table). A node is
 * PASS 1 if it doesn't, INCLUDING the spilled tags themselves (they
 * are the boundary, computed in PASS 1, then spilled before PASS 2).
 *
 * Walk in topological order so each node's preds have already been
 * classified by the time we reach it. *)
let classify_passes (sp : spill_info) (nodes : t list)
  : (int, [ `Pass1 | `Pass2 ]) Hashtbl.t
  =
  let cls = Hashtbl.create 256 in
  List.iter
    (fun e ->
       if is_spilled sp e.tag
       then Hashtbl.add cls e.tag `Pass1
       else (
         let pred_in_pass2 =
           List.exists
             (fun p ->
                match Hashtbl.find_opt cls p.tag with
                | Some `Pass2 -> true
                | _ -> false)
             (preds e)
         in
         let pred_is_spilled = List.exists (fun p -> is_spilled sp p.tag) (preds e) in
         if pred_in_pass2 || pred_is_spilled
         then Hashtbl.add cls e.tag `Pass2
         else Hashtbl.add cls e.tag `Pass1))
    nodes;
  (* DIF post-multiply: Twiddle Loads (and log3 cmul derivations of them) have
   * no spill-slot ancestors, so the forward pass classifies them as Pass1.
   * But their CONSUMERS may be in Pass2 (cmul on PASS 2 outputs). C block
   * scoping means Pass1-emitted variables go out of scope before Pass2;
   * references would fail to compile.
   *
   * Backward pass: reclassify any Pass1 node whose consumers are exclusively
   * in Pass2 → push to Pass2. This handles DIF cleanly (Twiddle Loads, log3
   * cmul derivations) without changing DIT behavior (where consumers of
   * Loads/cmul derivations are pre-multiply ops in Pass1). *)
  let consumers : (int, t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun e ->
       List.iter
         (fun p ->
            let prev =
              try Hashtbl.find consumers p.tag with
              | Not_found -> []
            in
            Hashtbl.replace consumers p.tag (e :: prev))
         (preds e))
    nodes;
  (* Iterate to fixpoint: a node X may need reclassification once a node it
   * feeds (Y) gets reclassified, in case Y was the reason X stayed Pass1. *)
  let changed = ref true in
  while !changed do
    changed := false;
    List.iter
      (fun e ->
         match Hashtbl.find_opt cls e.tag with
         | Some `Pass1 when not (is_spilled sp e.tag) ->
           let cs =
             try Hashtbl.find consumers e.tag with
             | Not_found -> []
           in
           if
             cs <> []
             && List.for_all (fun c -> Hashtbl.find_opt cls c.tag = Some `Pass2) cs
           then (
             Hashtbl.replace cls e.tag `Pass2;
             changed := true)
         | _ -> ())
      nodes
  done;
  cls
;;

(* ── filter_inline_set_cross_pass ───────────────────────────────────
 * Single source of truth for the spill-path inline-set filter, shared
 * by the emit_c monolithic spill emitter and codelet_oop's Tier-B/C
 * OOP body (which previously hand-copied this with a "we replicate that
 * filter here" comment — the exact drift class section 37 warns about).
 *
 * Given the raw single-use inline candidates (compute_inline_set), the
 * spill_info, and the reachable node list, keep a tag inlinable iff:
 *   - it is NOT spilled (spilled values must be named to store/reload), and
 *   - it has at least one consumer, and
 *   - ALL its consumers are in the SAME pass-class as the producer
 *     (cross-pass inlining would emit the producer's expression in the
 *     wrong scope, where its operands are out of scope; cross-pass values
 *     must round-trip through the spill array).
 * `nodes` is taken as a parameter rather than recomputed so callers that
 * already hold the reachable set don't topo-sort twice. ─ *)
let filter_inline_set_cross_pass
  ~(sc : Scratch.t)
      (assigns : (Expr.elem_ref * t) list)
      (sp : spill_info)
      (nodes : t list)
  : (int, unit) Hashtbl.t
  =
  let cls = classify_passes sp nodes in
  let all = compute_inline_set ~sc assigns in
  let consumers : (int, t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun e ->
       List.iter
         (fun p ->
            let prev =
              try Hashtbl.find consumers p.tag with
              | Not_found -> []
            in
            Hashtbl.replace consumers p.tag (e :: prev))
         (preds e))
    nodes;
  let filtered = Hashtbl.create 64 in
  Hashtbl.iter
    (fun tag () ->
       if not (is_spilled sp tag)
       then (
         let producer_class = Hashtbl.find_opt cls tag in
         let consumer_classes =
           match Hashtbl.find_opt consumers tag with
           | None -> []
           | Some cs -> List.map (fun c -> Hashtbl.find_opt cls c.tag) cs
         in
         if
           consumer_classes <> []
           && List.for_all (fun cc -> cc = producer_class) consumer_classes
         then Hashtbl.add filtered tag ()))
    all;
  filtered
;;

(* ── PROVENANCE STAMP ────────────────────────────────────────────────
 * Every generated file records the exact command line, active env
 * overrides, and each decision the auto-rules took, with reasons.
 * Comments only: object code is byte-identical with or without the
 * stamp. Single source of truth = the actual booleans computed above,
 * so the header can never drift from behavior. (Tugbars, section 37.) *)
(* Set by Gen_main.run so in-process drivers (gen_set) stamp the
 * LOGICAL per-codelet command, not the driver's own argv. *)
let provenance_argv : string array option ref = ref None

let provenance_env_overrides () : string =
  let keys =
    [ "VFFT_N1_BLOCK_MIN"
    ; "VFFT_NO_REGALLOC"
    ; "VFFT_PIN_FORCE"
    ; "VFFT_CT_FACTOR"
    ; "VFFT_SPLIT_RADIX"
    ; "VFFT_COLLECT_M"
    ; "VFFT_DEEP_COLLECT"
    ; (* Schedule-search / wisdom knobs. These change the emitted C, so the
       * stamp must record them (previously an injected codelet stamped
       * "(none)" — indistinguishable from stock). VFFT_SCHED_DUMP is
       * excluded: it writes side files only, object code is unchanged. *)
      "VFFT_SCHED_ORDER"
    ; "VFFT_SCHED_WISDOM"
    ; "VFFT_GH_THRESHOLD"
    ; "VFFT_NO_ANYK_TAIL"
    ]
  in
  let act =
    List.filter_map
      (fun k ->
         match Sys.getenv_opt k with
         | Some v -> Some (k ^ "=" ^ v)
         | None -> None)
      keys
  in
  match act with
  | [] -> "(none)"
  | l -> String.concat " " l
;;

let provenance_block ~(family : string) (lines : string list) : string =
  let b = Buffer.create 1024 in
  Buffer.add_string b "/* ===================== PROVENANCE =====================\n";
  Buffer.add_string
    b
    (Printf.sprintf
       " * Generated by: %s\n"
       (String.concat
          " "
          (Array.to_list
             (match !provenance_argv with
              | Some a -> a
              | None -> Sys.argv))));
  Buffer.add_string
    b
    (Printf.sprintf " * Env overrides: %s\n" (provenance_env_overrides ()));
  Buffer.add_string b (Printf.sprintf " * Family: %s\n" family);
  List.iter (fun l -> Buffer.add_string b (" * " ^ l ^ "\n")) lines;
  Buffer.add_string b " * ====================================================== */\n";
  Buffer.contents b
;;

(* Pending-lattice flush: every rendered node definition carries any il_in
   lattice statements its expression triggered, placed by the scheduler's
   own ordering (lazy first-touch). *)
let render_node_def
  ~(sc : Scratch.t)
  ~(cfg : Cfg.t)
      ?(no_declarator = false)
      ?(inline_set : (int, unit) Hashtbl.t option = None)
      ?(twidsq = false)
      ?(twidsq_n = 0)
      ?(strided = false)
      ~(isa : Isa.t)
      ~(in_place : bool)
      ~(t1s : bool)
      (e : t)
  : string
  =
  let core =
    render_node_def_core
      ~sc
      ~cfg
      ~no_declarator
      ~inline_set
      ~twidsq
      ~twidsq_n
      ~strided
      ~isa
      ~in_place
      ~t1s
      e
  in
  let p = Scratch.il_take_pending sc in
  if p = "" then core else p ^ core
;;

(* ── M4 phase 3: THE body preamble — the spill-array declarations plus the
   hoisted constants, previously the SAME 10-line block copy-pasted 12 times
   across emit_c's arms (11 of them followed by the same hoisted-consts call;
   twidsq alone omits the consts).  One definition, called once per arm. *)
let body_preamble ~(sc : Scratch.t) ~isa ~spill ?consts () =
  let b = Buffer.create 256 in
  (match spill with
   | None -> ()
   | Some sp ->
     Buffer.add_string
       b
       (Printf.sprintf "    %s spill_re[%d];
" isa.Isa.vec_type sp.num_slots);
     Buffer.add_string
       b
       (Printf.sprintf "    %s spill_im[%d];
" isa.Isa.vec_type sp.num_slots));
  (match consts with
   | None -> ()
   | Some assigns ->
     Buffer.add_string
       b
       (render_hoisted_consts ~sc ~isa (topo_sort_reachable (List.map snd assigns))));
  Buffer.contents b
;;
