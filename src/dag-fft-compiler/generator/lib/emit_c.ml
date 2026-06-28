(* emit_c.ml — naive AVX-512 C emitter for radix-N t1_dit codelets.
 *
 * Walks the hash-consed DAG in topological order, emits one __m512d
 * variable per node. No scheduling, no register allocation — GCC handles
 * those when compiling the resulting C.
 *
 * The output signature matches user's gen_radix4.py t1_dit form:
 *
 *   void radix<N>_t1_dit_fwd_avx512(
 *       const double *in_re,  const double *in_im,
 *       double       *out_re, double       *out_im,
 *       const double *tw_re,  const double *tw_im,
 *       size_t K)
 *
 * Inputs/outputs are split-complex with K-stride layout: element j's real
 * component is in_re[j*K + k], imag in_im[j*K + k]. Twiddles same layout.
 * Loop iterates k by 8 (AVX-512 vector width for double).
 *)

open Algsimp

(* === Topological sort of the DAG nodes ===
 *
 * We need to emit definitions in dependency order: a node's definition
 * must come AFTER the definitions of all its operands. Since hash-consing
 * assigns tags in construction order (bottom-up), sorting by tag gives
 * a valid topological order automatically. *)
let topo_sort_reachable (roots : t list) : t list =
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
      | NK_Plus _ -> Algsimp.nk_plus_unreachable "emit_c.ml:33"
    end
  in
  List.iter visit roots;
  let nodes = Hashtbl.fold (fun _ e acc -> e :: acc) seen [] in
  List.sort (fun a b -> compare a.tag b.tag) nodes

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

(* t1p (per-position broadcast) twiddles: one scalar per (leg, position),
   broadcast across the vec_width batch lanes. Table is (R-1)*(me/VW) scalars,
   loaded with set1, vs t1's per-(position,lane) vector load (VW× larger).
   Set by emit_codelet for PerPositionTwiddles codelets, read below. *)
let current_tw_perpos : bool ref = ref false

(* Lean real-to-real ABI (notebook section 51, step 2): when true, the
 * OOP signature is `(const double *in, double *out, size_t K)` — the
 * convention dct.h's fast paths consume (python n8 vintage). Set by
 * gen_main for the real->real trig family {dct2, dct2_trigII, dct3,
 * dct4, dht}; rdft keeps the generic signature (complex output). The
 * body's array names map in_re->in, out_re->out; in_im/out_im/tw are
 * never referenced by these DAGs. *)
let r2r_signature : bool ref = ref false
let r2cf_signature : bool ref = ref false

(* r2cb (section 62): backward real leaf. Halfcomplex INPUT (in_re, in_im) ->
 * real OUTPUT (out_re only). Distinct from r2cf, which is real-in / split-out
 * with a reversed im stream. *)
let r2cb_signature : bool ref = ref false
let hc_strided : bool ref = ref false

(* n1_oop_strided (pack fix, section: r2c stage-0 copy deletion): true
 * strided-OOP n1 signature matching vfft_proto_n1_fn exactly —
 * (in_re, in_im, out_re, out_im, size_t is, size_t os, size_t vl),
 * legs j at in[j*is+v] / out[j*os+v]. Exists so the r2c fused first
 * stage (the one in!=out caller) doesn't pay the registry wrapper's
 * memcpy-then-in-place bridge. n1-only: no twiddles in the DAG. *)
let n1_oop_strided : bool ref = ref false

(* D2 (section 69): hc2c NATURAL-split terminator. Sub-mode of
 * hc_strided (inputs/twiddles identical); overrides the signature
 * (FFTW khc2c-shaped 4-pointer outputs) and the output slot map.
 * sstar = constant regime boundary (design doc lemma): slots
 * s <= sstar are direct rows (Rp/Ip + s*osp), slots s > sstar are
 * conjugate-mirror rows (Rm/Im + (r-1-s)*osm). *)
let hc2c_natural : bool ref = ref false

(* hc2c_natural_bwd (c2r initiator, the INVERSE of hc2c_natural): the backward
 * natural codelet reads the SPLIT half-spectrum (Rp/Ip direct rows, Rm/Im
 * conjugate-mirror rows — the same sstar map as the forward, but on the INPUT
 * side) and writes the 2 PACKED cascade columns (out_re/out_im). I.e. the
 * forward natural's 2-packed-in / 4-split-out ABI, flipped: 4 split const-in /
 * 2 packed-out. The butterfly is dft_hc2c_dif ~sign:`Bwd (c2r's DIF-backward).
 * This lets a split-input c2r run the fast packed cascade with NO repack — the
 * mirror of how rfft_execute_fwd_natural gives split output at packed speed. *)
let hc2c_natural_bwd : bool ref = ref false

(* r2c_term (step-2 fusion): the fused forward terminator codelet. Dual-output
 * 2-input shape: reads Z[k] (input 0) and Z[m] (input 1), writes X[k]
 * (Xp pair) and X[m] (Xm pair), vectorized over vl lanes. Simpler than
 * hc2c_natural's r-way sstar split — exactly 2 outputs. *)
let r2c_term_signature : bool ref = ref false

(* r2c_term runtime-twiddle variant (slice 3b): one codelet for all
 * frequencies, loading W^f from a tw pointer. Adds tw_re/tw_im to the
 * signature and addresses Twiddle(0) as tw_re[0]/tw_im[0] (the executor
 * passes tw_re+f per call, so slot 0 is the current frequency's twiddle). *)
let r2c_term_rt : bool ref = ref false

(* model (b): the last-stage-fused terminator. r-leg inputs for two columns
 * (col k legs 0..r-1 at in[s*is_leg], col m-k legs r..2r-1 at in[(r+j)*is_leg]
 * via the is_leg stride and the is column offset), packed twiddle table (3r
 * slots, broadcast), 2r outputs: even slot s -> Xp[s*osp], odd -> Xm[s*osm]. *)
let r2c_term_laststage : bool ref = ref false
let r2c_term_ls_r : int ref = ref 0 (* radix r, for output slot count 2r *)

(* T1 (section 70, FFTW-structural): RANGED variant — the column
 * loop lives INSIDE the codelet (one call walks kcount columns,
 * pointers and twiddles advancing), amortizing call overhead and
 * making the twiddle walk a linear stream. Wraps the v-loop;
 * render layer unchanged (parameter pointers are mutated). *)
let hc_ranged : bool ref = ref false
let hc_ranged_r : int ref = ref 0
let hc2c_nat_sstar : int ref = ref 0
let hc2c_nat_r : int ref = ref 0

(* Arbitrary-K tail (docs/roadmap/arbitrary_k_vectorization): the mode the
 * body emitter is currently rendering under. Defaults to LS_vector so every
 * existing path (and the bulk loop) is byte-identical to before. The in-place
 * c2c tail sets this to (LS_masked m) around the masked remainder pass; it is
 * read by render_load (rio + per-lane twiddle loads) and emit_store. Broadcast
 * twiddles / constants go through set1_pd_str and are never masked. *)
let current_ls_mode : Isa.ls_mode ref = ref Isa.LS_vector

let render_load ~(isa : Isa.t) ~(in_place : bool) ~(t1s : bool)
    ?(twidsq = false) ?(twidsq_n = 0) ?(strided = false) (r : Expr.elem_ref) :
    string =
  (* In strided mode, Input(j, _) refers to pre-computed lane locals
   * populated by the 4×4 transpose preamble at the top of each loop iter.
   * Twiddles still go through their normal load path (n1 codelets don't
   * have inter-stage twiddles anyway). *)
  if strided then
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
        if !current_tw_perpos then
          Isa.set1_pd_str isa
            (Printf.sprintf "tw_re[%d*(me/%d) + b/%d]" j isa.vec_width
               isa.vec_width)
        else if t1s then Isa.set1_pd_str isa (Printf.sprintf "tw_re[%d]" j)
        else
          (* PerGroupTwiddles: per-lane, indexed by the group var b -> maskable
             in the arbitrary-K tail (current_ls_mode). The set1 broadcasts above
             are lane-independent and stay unmasked. *)
          Isa.loadu_pd ~mode:!current_ls_mode isa
            (Printf.sprintf "tw_re[%d*me + b]" j)
    | Expr.Twiddle (j, false) ->
        if !current_tw_perpos then
          Isa.set1_pd_str isa
            (Printf.sprintf "tw_im[%d*(me/%d) + b/%d]" j isa.vec_width
               isa.vec_width)
        else if t1s then Isa.set1_pd_str isa (Printf.sprintf "tw_im[%d]" j)
        else
          Isa.loadu_pd ~mode:!current_ls_mode isa
            (Printf.sprintf "tw_im[%d*me + b]" j)
    | Expr.Output _ ->
        failwith "render_load: Output ref shouldn't appear as a Load source"
  else
    let in_buf is_re =
      match (in_place, is_re) with
      | true, true -> "rio_re"
      | true, false -> "rio_im"
      | false, true -> if !r2r_signature then "in" else "in_re"
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
      if in_place then "ios"
      else if twidsq then "is"
      else if
        !r2cf_signature || !r2cb_signature || !hc_strided || !n1_oop_strided
      then "is"
      else "K"
    in
    let loop_var =
      if
        twidsq || !r2cf_signature || !r2cb_signature || !hc_strided
        || !n1_oop_strided || !r2c_term_signature || !r2c_term_laststage
      then "v"
      else "k"
    in
    let tw_stride =
      if in_place then "me" else if !hc_strided then "vl" else "K"
    in
    let tw_broadcast = t1s || twidsq || !r2c_term_rt || !r2c_term_laststage in
    let render_input_addr j is_re =
      let buf = in_buf is_re in
      if !r2c_term_laststage then
        (* Input(j) j<r = col k leg j at ink[j*is_leg+v]; Input(r+j) = col m-k leg j
         * at inm[j*is_leg+v]. r is r2c_term_ls_r. *)
        let r = !r2c_term_ls_r in
        let bk = if is_re then "ink_re" else "ink_im" in
        let bm = if is_re then "inm_re" else "inm_im" in
        if j < r then Printf.sprintf "%s[%d*is_leg + %s]" bk j loop_var
        else Printf.sprintf "%s[%d*is_leg + %s]" bm (j - r) loop_var
      else if !r2c_term_signature then
        (* r2c_term: Input(0)=Z[k] at in_re[v]; Input(1)=Z[m] at in_re[is+v].
         * Two scratch rows, row stride `is`, vectorized over v. *)
        if j = 0 then Printf.sprintf "%s[%s]" buf loop_var
        else Printf.sprintf "%s[is + %s]" buf loop_var
      else if twidsq && twidsq_n > 0 then
        let row = j / twidsq_n in
        let col = j mod twidsq_n in
        Printf.sprintf "%s[%d*%s + %d*V + %s]" buf row stride col loop_var
      else if !hc2c_natural_bwd then
        (* c2r natural INITIATOR: read the SPLIT half-spectrum. Slot j<=sstar is
         * a direct row (Rp/Ip + j*isp); j>sstar is a conjugate-mirror row
         * (Rm/Im + (r-1-j)*ism). Exactly the forward terminator's OUTPUT sstar
         * map, but on the INPUT side. *)
        begin if j <= !hc2c_nat_sstar then
          Printf.sprintf "%s[%d*isp + %s]" (if is_re then "Rp" else "Ip") j loop_var
        else
          Printf.sprintf "%s[%d*ism + %s]" (if is_re then "Rm" else "Im")
            (!hc2c_nat_r - 1 - j) loop_var
        end
      else
        (* r2cb (section 62 / c2r cascade): split input strides. The
         * backward leaf reads the layout r2cf WRITES: re at +is_re from
         * its base, im at NEGATIVE stride from a one-past (+NK) base.
         * A single shared `is` cannot express the sign split. *)
        let stride =
          if !r2cb_signature then if is_re then "is_re" else "is_im" else stride
        in
        Printf.sprintf "%s[%d*%s + %s]" buf j stride loop_var
    in
    match r with
    | Expr.Input (j, true) ->
        Isa.loadu_pd ~mode:!current_ls_mode isa (render_input_addr j true)
    | Expr.Input (j, false) ->
        Isa.loadu_pd ~mode:!current_ls_mode isa (render_input_addr j false)
    | Expr.Twiddle (j, true) ->
        if tw_broadcast then Isa.set1_pd_str isa (Printf.sprintf "tw_re[%d]" j)
        else
          Isa.loadu_pd ~mode:!current_ls_mode isa
            (Printf.sprintf "tw_re[%d*%s + %s]" j tw_stride loop_var)
    | Expr.Twiddle (j, false) ->
        if tw_broadcast then Isa.set1_pd_str isa (Printf.sprintf "tw_im[%d]" j)
        else
          Isa.loadu_pd ~mode:!current_ls_mode isa
            (Printf.sprintf "tw_im[%d*%s + %s]" j tw_stride loop_var)
    | Expr.Output _ ->
        failwith "render_load: Output ref shouldn't appear as a Load source"

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

(* === M3a regalloc consumption point ===
 *
 * Top-level mutable ref that carries the active register allocation
 * from emit_codelet (which computes it per scheduling site) to
 * render_node_def (which consumes it when emitting declarations).
 *
 * Why a top-level ref instead of a parameter: render_node_def has
 * ~10 call sites scattered through emit_codelet (different scheduler
 * paths, spill structure variants). Threading a parameter through all
 * of them would touch a lot of code for an opt-in feature. The ref is
 * set by emit_codelet at each scheduling site and reset to None at
 * exit, so other code paths see None.
 *
 * Threading note: ocaml refs are not thread-safe. Codelet generation
 * is single-threaded (gen_radix emits one codelet at a time), so this
 * is fine in practice. Documented limitation. *)
let current_regalloc : Regalloc.allocation option ref = ref None

(* === Fence-only emission mode ===
 *
 * When true, the non-pinned emission path uses Isa.fenced_decl
 * (`register __m512d tN = expr; asm volatile("" : "+v"(tN))`) instead
 * of Isa.const_decl (`const __m512d tN = expr;`). The fenced form
 * preserves the inline-asm scheduling barrier that constrains GCC's
 * scheduler to honor the codelet generator's SU+GH ordering, while
 * letting GCC pick the register freely.
 *
 * Set per-codelet at the top of emit_codelet based on the fence/pin
 * policy. See `docs/fence_pin_decomposition.md` for the empirical
 * basis. *)
let current_fence_only : bool ref = ref false

(* Store-on-compute (in-place 2-pass path). When true, each PASS 2 output
 * is stored to the output buffer the moment its value node is emitted,
 * with force_last_use set to that same position, instead of being held
 * live until its sub-DFT cluster flushes. Output value nodes are sinks
 * (compute_inline_set never inlines them), so their only use is the
 * store; storing at def is the natural last use and frees the register
 * immediately, lowering peak live in PASS 2. Safe because PASS 2 only
 * writes the output buffer, never reads it (all input reads retired in
 * PASS 1). PASS 1 outputs are left at end-of-PASS-1 (PASS 1 still reads
 * the input buffer, so an early write could alias an unread input). *)
let current_store_on_compute : bool ref = ref false

(* M5: schedule position of the node currently being emitted by
 * render_node_def. Set by the pass walkers in emit_c just before each
 * render_node_def call. Used by `v` (operand name renderer) to look
 * up name_overrides for reloaded values. *)
let current_emit_position : int ref = ref 0

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
let current_unpin_candidates : (int, unit) Hashtbl.t option ref = ref None

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

(* ── Constant hoisting (notebook section 51) ──────────────────────
 * NK_Const nodes are loop-invariant by definition, but emitting them
 * inside the k-loop as fenced register temps forces gcc to
 * re-materialize them every iteration (measured: 18 loads/iter vs
 * python's 5+6 hoisted at dct2 N=8 — the whole 9% race deficit).
 * Hoist: render every NK_Const ONCE, unfenced, at function scope
 * BEFORE the k-loop; record the tag here so the in-loop renderers
 * emit nothing for it. Names (tN) are unchanged, arithmetic order is
 * unchanged, so outputs stay bit-exact. *)
let hoisted_const_tags : (int, unit) Hashtbl.t = Hashtbl.create 64

(* Gate: hoisting helps loop-dominated r2r/trig codelets (won the N=8
 * race vs the hand codelet) but TAXES spill-bound DFT kernels ~2%
 * (hoisted consts are live across the whole body and steal registers
 * from data). gen_main sets this true for the trig family only;
 * default false keeps the DFT tree byte-identical. *)
let hoist_consts_enabled : bool ref = ref false

let render_hoisted_consts ~(isa : Isa.t) (nodes : t list) : string =
  Hashtbl.reset hoisted_const_tags;
  if not !hoist_consts_enabled then ""
  else begin
    let b = Buffer.create 256 in
    List.iter
      (fun e ->
        match e.node with
        | NK_Const c ->
            Hashtbl.replace hoisted_const_tags e.tag ();
            Buffer.add_string b
              (Printf.sprintf "    %s\n"
                 (Isa.const_decl isa
                    (Printf.sprintf "t%d" e.tag)
                    (Isa.set1_pd_str isa (Printf.sprintf "%.17g" c))))
        | _ -> ())
      nodes;
    Buffer.contents b
  end

let render_node_def ?(no_declarator = false)
    ?(inline_set : (int, unit) Hashtbl.t option = None) ?(twidsq = false)
    ?(twidsq_n = 0) ?(strided = false) ~(isa : Isa.t) ~(in_place : bool)
    ~(t1s : bool) (e : t) : string =
  if Hashtbl.mem hoisted_const_tags e.tag then ""
  else
    (* Name renderer: usually returns "t<tag>", but if M5 has installed a
     * name override for (current_emit_position, t.tag) and t is not the
     * tag being defined (i.e., t is an operand reference, not the LHS),
     * return the override name instead — used to point at reload
     * variables. *)
    let v t =
      let default_name () = Printf.sprintf "t%d" t.tag in
      if t.tag = e.tag then default_name () (* LHS: never override *)
      else
        match !current_regalloc with
        | None -> default_name ()
        | Some alloc -> (
            match
              Hashtbl.find_opt alloc.name_overrides
                (!current_emit_position, t.tag)
            with
            | Some n -> n
            | None -> default_name ())
    in
    (* Should this node be inlined into its consumer's expression? *)
    let should_inline n =
      match inline_set with None -> false | Some tbl -> Hashtbl.mem tbl n.tag
    in
    (* Render an operand. If single-use, inline its expression recursively
     * (up to depth limit). Otherwise, just emit `t<tag>` and rely on the
     * standalone declaration (which will be emitted elsewhere). *)
    let rec render_operand depth n =
      if depth >= inline_max_depth || not (should_inline n) then v n
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
      | NK_Neg inner -> (
          match inner.node with
          | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" (-.c))
          | _ ->
              Isa.xor_pd isa
                (render_operand (depth + 1) inner)
                (Isa.set1_pd_str isa "-0.0"))
      | NK_Add (a, b) ->
          Isa.add_pd isa
            (render_operand (depth + 1) a)
            (render_operand (depth + 1) b)
      | NK_Sub (a, b) ->
          Isa.sub_pd isa
            (render_operand (depth + 1) a)
            (render_operand (depth + 1) b)
      | NK_Mul (a, b) ->
          Isa.mul_pd isa
            (render_operand (depth + 1) a)
            (render_operand (depth + 1) b)
      | NK_CmulRe _ | NK_CmulIm _ -> v n (* don't inline cmul *)
      | NK_Fma (a, b, c, neg_mul, neg_add) -> (
          let ra = render_operand (depth + 1) a in
          let rb = render_operand (depth + 1) b in
          let rc = render_operand (depth + 1) c in
          match (neg_mul, neg_add) with
          | false, false -> Isa.fmadd_pd isa ra rb rc
          | false, true -> Isa.fmsub_pd isa ra rb rc
          | true, false -> Isa.fnmadd_pd isa ra rb rc
          | true, true -> Isa.fnmsub_pd isa ra rb rc)
      | NK_Plus _ ->
          (* NK_Plus must be lowered to binary NK_Add/NK_Sub before the emitter
           * runs. If it reaches here, it's a bug in the lowering pass. *)
          Algsimp.nk_plus_unreachable "emit_c.ml render_operand"
    in
    (* Operand renderer for THIS node's body — depth=0 meaning we're already
     * inside the body of `e`, so its operands start at depth=0 (and inline up
     * to inline_max_depth from there). *)
    let op = render_operand 0 in
    let body =
      match e.node with
      | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" c)
      | NK_Load r ->
          render_load ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided r
      | NK_Neg inner -> (
          (* Neg(Const c) is a compile-time constant — emit as a single
           * broadcast of -c rather than a runtime XOR. *)
          match inner.node with
          | NK_Const c -> Isa.set1_pd_str isa (Printf.sprintf "%.17g" (-.c))
          | _ -> Isa.xor_pd isa (op inner) (Isa.set1_pd_str isa "-0.0"))
      | NK_Add (a, b) -> Isa.add_pd isa (op a) (op b)
      | NK_Sub (a, b) -> Isa.sub_pd isa (op a) (op b)
      | NK_Mul (a, b) -> Isa.mul_pd isa (op a) (op b)
      | NK_CmulRe (xr, xi, wr, wi) ->
          Isa.fnmadd_pd isa (op xi) (op wi) (Isa.mul_pd isa (op xr) (op wr))
      | NK_CmulIm (xr, xi, wr, wi) ->
          Isa.fmadd_pd isa (op xr) (op wi) (Isa.mul_pd isa (op xi) (op wr))
      | NK_Fma (a, b, c, neg_mul, neg_add) -> (
          (* (neg_mul ? -a*b : a*b) + (neg_add ? -c : c)
           *
           *   neg_mul=F, neg_add=F:  a*b + c       → fmadd
           *   neg_mul=F, neg_add=T:  a*b - c       → fmsub
           *   neg_mul=T, neg_add=F:  -a*b + c      → fnmadd
           *   neg_mul=T, neg_add=T:  -a*b - c      → fnmsub *)
          match (neg_mul, neg_add) with
          | false, false -> Isa.fmadd_pd isa (op a) (op b) (op c)
          | false, true -> Isa.fmsub_pd isa (op a) (op b) (op c)
          | true, false -> Isa.fnmadd_pd isa (op a) (op b) (op c)
          | true, true -> Isa.fnmsub_pd isa (op a) (op b) (op c))
      | NK_Plus _ ->
          (* NK_Plus must be lowered to binary NK_Add/NK_Sub before this point.
           * If it reaches the body renderer, the lowering pass missed it. *)
          Algsimp.nk_plus_unreachable "emit_c.ml render_body"
    in
    if no_declarator then
      (* Plain assignment to a variable forward-declared at outer scope.
       * Used for spill "fused slots" — values whose lifetime crosses the
       * PASS 1 / PASS 2 boundary as register-resident SSA, so they're
       * declared once before either pass opens and assigned in PASS 1.
       * Not eligible for M3a register pinning: the variable was already
       * declared without a pin, so we just assign. *)
      Printf.sprintf "        %s = %s;" (v e) body
    else
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
        try Sys.getenv "VFFT_DISABLE_SELECTIVE_PIN" = "1"
        with Not_found -> false
      in
      let is_unpin_candidate =
        if selective_pin_disabled then false
        else
          match !current_unpin_candidates with
          | None -> false
          | Some tbl -> Hashtbl.mem tbl e.tag
      in
      (* Helper: in fence-only mode emit `register ... = expr; asm volatile(...)`;
       * otherwise emit the plain `const ... = expr;` form. *)
      let non_pinned_decl name body =
        if !current_fence_only then Isa.fenced_decl isa name body
        else Isa.const_decl isa name body
      in
      match !current_regalloc with
      | Some alloc when not is_unpin_candidate -> (
          match Regalloc.lookup alloc e.tag with
          | Regalloc.Reg reg_name ->
              Printf.sprintf "        %s"
                (Isa.pinned_reg_decl isa (v e) reg_name body)
          | Regalloc.Spilled _ ->
              (* M5: the Spilled variant is reserved but no longer used by
               * the spilling allocator (it tracks spills via spill_sites
               * separately, keeping the tag's assignment as Reg). If we
               * see Spilled here it's a future extension; fall back to
               * Default emission. *)
              Printf.sprintf "        %s" (non_pinned_decl (v e) body)
          | Regalloc.Default ->
              Printf.sprintf "        %s" (non_pinned_decl (v e) body))
      | Some _ ->
          (* Selective unpin: drop the pin to enable gcc auto-fusion *)
          Printf.sprintf "        %s" (non_pinned_decl (v e) body)
      | None -> Printf.sprintf "        %s" (non_pinned_decl (v e) body)

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
  | Bisection (* Frigo's recursive bisection, flat emit *)
  | Annotated_topological
    (* topological order + nested-block scopes (annotate.ml) *)
  | Annotated_bisection (* bisection schedule + nested-block scopes *)
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
let compute_inline_set (assigns : (Expr.elem_ref * t) list) :
    (int, unit) Hashtbl.t =
  let roots = List.map snd assigns in
  let nodes = topo_sort_reachable roots in
  (* Use count = how many other nodes reference this tag. *)
  let use_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let bump tag =
    let cur = try Hashtbl.find use_count tag with Not_found -> 0 in
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
      | NK_Plus _ -> Algsimp.nk_plus_unreachable "emit_c.ml use_count walker")
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
      let count = try Hashtbl.find use_count n.tag with Not_found -> 0 in
      let is_sink = Hashtbl.mem sink_tags n.tag in
      let kind_inlinable =
        match n.node with
        | NK_Load _ -> false (* don't duplicate loads *)
        | NK_CmulRe _ | NK_CmulIm _ -> false (* paired emit *)
        | NK_Const _ -> false (* already inlined as set1 broadcast *)
        | _ -> true
      in
      if count = 1 && (not is_sink) && kind_inlinable then
        Hashtbl.add result n.tag ())
    nodes;
  result

(* === SPILL CONFIGURATION ===
 *
 * When ?spill is provided, emission produces a PASS-1/PASS-2 split:
 *   - PASS 1: compute everything up to and including the spill targets.
 *             Emit explicit stack-array stores for spill targets.
 *   - PASS 2: reload spill targets into a fresh nested scope; compute
 *             dependents; emit final output stores.
 *
 * The spill_table maps Algsimp tag → slot index. Two parallel arrays
 * are declared at function entry: spill_re[N] and spill_im[N] of vector
 * type. The re/im distinction comes from the tag being marked as either
 * a real or imaginary PASS 1 output.
 *
 * `num_slots` is the size of each array (max slot index + 1).
 *
 * `fused_slots` (if non-empty) marks spill slots whose values are kept
 * in registers across the PASS 1/PASS 2 boundary instead of being stored
 * to and reloaded from spill_re[] / spill_im[]. Set by make_spill_info
 * when ?ct=(n1,n2) and ?fuse=M are provided: fuses the M PASS 2 sub-DFTs
 * whose inputs are emitted LAST in PASS 1 (giving short lifetime extension).
 *
 * For CT(n1, n2): PASS 2 sub-DFT-n1 #k2 consumes slots {n1_idx*n2 + k2 :
 * n1_idx in 0..n1-1}. We fuse k2 in {n2-fuse..n2-1} since these correspond
 * to the LAST sub-DFT-n2 output positions in each PASS 1 sub-FFT — the
 * latest-emitted (and thus latest-stored) values, which are also the
 * first-consumed in PASS 2 emission order. *)
type spill_info = {
  re_slot : (int, int) Hashtbl.t; (* re tag → slot *)
  im_slot : (int, int) Hashtbl.t; (* im tag → slot *)
  num_slots : int;
  fused_slots : (int, unit) Hashtbl.t; (* slots NOT spilled — kept in regs *)
  ct_n1 : int; (* PASS 1 sub-FFT count, 0 if not CT-decomposed *)
  ct_n2 : int; (* PASS 1 sub-FFT size, 0 if not CT-decomposed *)
}

let make_spill_info ?ct ?(fuse = 0) (markers : Algsimp.spill_tag_marker list) :
    spill_info =
  let re_slot = Hashtbl.create 64 in
  let im_slot = Hashtbl.create 64 in
  let max_slot = ref (-1) in
  List.iter
    (fun m ->
      Hashtbl.replace re_slot m.Algsimp.re_tag m.slot;
      Hashtbl.replace im_slot m.Algsimp.im_tag m.slot;
      if m.slot > !max_slot then max_slot := m.slot)
    markers;
  let fused_slots = Hashtbl.create 16 in
  let ct_n1, ct_n2 =
    match ct with Some (n1, n2) -> (n1, n2) | None -> (0, 0)
  in
  (match ct with
  | Some (n1, n2) when fuse > 0 ->
      let m = min fuse n2 in
      for k2 = n2 - m to n2 - 1 do
        for n1_idx = 0 to n1 - 1 do
          Hashtbl.replace fused_slots ((n1_idx * n2) + k2) ()
        done
      done
  | _ -> ());
  { re_slot; im_slot; num_slots = !max_slot + 1; fused_slots; ct_n1; ct_n2 }

let is_spilled (sp : spill_info) (tag : int) : bool =
  Hashtbl.mem sp.re_slot tag || Hashtbl.mem sp.im_slot tag

let is_fused_slot (sp : spill_info) (slot : int) : bool =
  Hashtbl.mem sp.fused_slots slot

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
let codelet_metadata ~(isa : Isa.t) ~(spill : spill_info option)
    ~(tw_broadcast : bool) ~(peak_live : int)
    (assigns : (Expr.elem_ref * t) list) : string =
  let nodes = topo_sort_reachable (List.map snd assigns) in
  let peak_live =
    if peak_live > 0 then peak_live
    else (Regalloc.peak_live_analysis ~isa ~scheduled:nodes).peak_live
  in
  let nadd = ref 0
  and nsub = ref 0
  and nmul = ref 0
  and nfma = ref 0
  and ncmul = ref 0
  and nneg = ref 0
  and n_vload =
    ref
      0 (* per-iteration vector loads: input data + (non-broadcast) twiddles *)
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
      | NK_Load r -> (
          match r with
          | Expr.Input _ -> incr n_vload
          | Expr.Twiddle _ ->
              if tw_broadcast then incr n_swload else incr n_vload
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
  let xslots = match spill with Some sp -> sp.num_slots | None -> 0 in
  let xpass_mem = 4 * xslots and xpass_live = 2 * xslots in
  let memory_floor = ess_io + xpass_mem in
  let budget = isa.vec_regs - if isa.vec_regs >= 32 then 4 else 2 in
  let membound = memory_floor > fp_instr in
  let fits = peak_live <= budget in
  Printf.sprintf
    "/* codelet-metrics [intrinsic, gen-time]:\n\
    \ *   fp_instr=%d  flops=%d  (add=%d sub=%d mul=%d fma=%d cmul=%d neg=%d)\n\
    \ *   essential_io=%d ops (vec_loads=%d + stores=%d)  [+%d hoisted \
     scalar-twiddle loads, not counted]\n\
    \ *   cross_pass_cut=%d slots => +%d mem ops, %d vectors live across pass \
     boundary\n\
    \ *   memory_floor=%d mem ops   peak_live(max-per-pass)=%d   budget=%d regs\n\
    \ *   ROOFLINE: %s at floor (memory_floor %s fp_instr)\n\
    \ *   PRESSURE: %s (peak_live %s budget)%s\n\
    \ */\n"
    fp_instr flops !nadd !nsub !nmul !nfma !ncmul !nneg ess_io ess_loads
    ess_stores !n_swload xslots xpass_mem xpass_live memory_floor peak_live
    budget
    (if membound then "MEMORY-BOUND" else "compute-capable")
    (if membound then ">" else "<=")
    (if fits then "fits" else "SPILLS")
    (if fits then "<=" else ">")
    (if xslots > 0 then
       "  [CT: peak_live is max over passes; cross_pass_cut is added explicit \
        spill traffic]"
     else "")

(* Is this tag's spill slot fused (kept in register, not stored)? *)
let is_fused_tag (sp : spill_info) (tag : int) : bool =
  match (Hashtbl.find_opt sp.re_slot tag, Hashtbl.find_opt sp.im_slot tag) with
  | Some s, _ | _, Some s -> is_fused_slot sp s
  | None, None -> false

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
 * Algsimp.topo_sort_reachable, now itself single-sourced). The explicit
 * sort DELETES that latent dependence: this helper is correct regardless
 * of the caller's input order. (Also uses Hashtbl.replace, not add;
 * equivalent here since each tag is visited once in the reverse walk.) ─ *)
let compute_min_slot_pass1 (sp : spill_info) (pass1_nodes : t list) :
    (int, int) Hashtbl.t * t list =
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
          if Hashtbl.mem pass1_set p.tag then begin
            let cur = try Hashtbl.find succs p.tag with Not_found -> [] in
            Hashtbl.replace succs p.tag (e.tag :: cur)
          end)
        (preds e))
    pass1_nodes;
  let min_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
  (* Reverse topological order = descending tag (hash-cons tags are
     construction-ordered). Explicit sort, not List.rev — see header. *)
  let pass1_rev =
    List.sort (fun (a : t) b -> compare b.tag a.tag) pass1_nodes
  in
  List.iter
    (fun (e : t) ->
      let my =
        match lookup_slot e.tag with
        | Some s -> Some s
        | None -> (
            let s_tags = try Hashtbl.find succs e.tag with Not_found -> [] in
            let s_mins =
              List.filter_map (fun t -> Hashtbl.find_opt min_slot t) s_tags
            in
            match s_mins with
            | [] -> None
            | _ -> Some (List.fold_left min max_int s_mins))
      in
      match my with Some s -> Hashtbl.replace min_slot e.tag s | None -> ())
    pass1_rev;
  let pass1_blocked_topo =
    List.sort
      (fun (a : t) b ->
        let ma = try Hashtbl.find min_slot a.tag with Not_found -> max_int in
        let mb = try Hashtbl.find min_slot b.tag with Not_found -> max_int in
        let c = compare ma mb in
        if c <> 0 then c else compare a.tag b.tag)
      pass1_nodes
  in
  (min_slot, pass1_blocked_topo)

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
let cluster_split_schedule (sp : spill_info) ~(pass1_blocked_topo : t list)
    ~(min_slot : (int, int) Hashtbl.t)
    ~(schedule_cluster : subset:t list -> sinks:t list -> t list) : t list =
  if sp.ct_n2 <= 0 then pass1_blocked_topo
  else begin
    let cluster_of_node (e : t) =
      match Hashtbl.find_opt min_slot e.tag with
      | Some s -> s / sp.ct_n2
      | None -> sp.ct_n1 (* unreachable → fake last cluster *)
    in
    (* Walk pass1_blocked_topo, split into contiguous same-cluster runs. *)
    let groups : (int * t list) list =
      let rec go acc cur_cluster cur_acc = function
        | [] -> (
            match cur_acc with
            | [] -> List.rev acc
            | _ -> List.rev ((cur_cluster, List.rev cur_acc) :: acc))
        | n :: rest ->
            let k = cluster_of_node n in
            if k = cur_cluster then go acc cur_cluster (n :: cur_acc) rest
            else
              let acc' =
                match cur_acc with
                | [] -> acc
                | _ -> (cur_cluster, List.rev cur_acc) :: acc
              in
              go acc' k [ n ] rest
      in
      match pass1_blocked_topo with
      | [] -> []
      | n :: rest -> go [] (cluster_of_node n) [ n ] rest
    in
    List.concat_map
      (fun (_cluster_id, group_nodes) ->
        let cluster_sinks =
          List.filter
            (fun (e : t) ->
              Hashtbl.mem sp.re_slot e.tag || Hashtbl.mem sp.im_slot e.tag)
            group_nodes
        in
        if cluster_sinks = [] then group_nodes
        else schedule_cluster ~subset:group_nodes ~sinks:cluster_sinks)
      groups
  end

(* Split a topologically-ordered list of nodes into PASS 1 and PASS 2.
 *
 * A node is PASS 2 iff it transitively depends on a spilled tag (i.e.,
 * some pred or pred-of-pred etc. is in the spill_table). A node is
 * PASS 1 if it doesn't, INCLUDING the spilled tags themselves (they
 * are the boundary, computed in PASS 1, then spilled before PASS 2).
 *
 * Walk in topological order so each node's preds have already been
 * classified by the time we reach it. *)
let classify_passes (sp : spill_info) (nodes : t list) :
    (int, [ `Pass1 | `Pass2 ]) Hashtbl.t =
  let cls = Hashtbl.create 256 in
  List.iter
    (fun e ->
      if is_spilled sp e.tag then Hashtbl.add cls e.tag `Pass1
      else
        let pred_in_pass2 =
          List.exists
            (fun p ->
              match Hashtbl.find_opt cls p.tag with
              | Some `Pass2 -> true
              | _ -> false)
            (preds e)
        in
        let pred_is_spilled =
          List.exists (fun p -> is_spilled sp p.tag) (preds e)
        in
        if pred_in_pass2 || pred_is_spilled then Hashtbl.add cls e.tag `Pass2
        else Hashtbl.add cls e.tag `Pass1)
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
          let prev = try Hashtbl.find consumers p.tag with Not_found -> [] in
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
            let cs = try Hashtbl.find consumers e.tag with Not_found -> [] in
            if
              cs <> []
              && List.for_all
                   (fun c -> Hashtbl.find_opt cls c.tag = Some `Pass2)
                   cs
            then begin
              Hashtbl.replace cls e.tag `Pass2;
              changed := true
            end
        | _ -> ())
      nodes
  done;

  cls

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
let filter_inline_set_cross_pass (assigns : (Expr.elem_ref * t) list)
    (sp : spill_info) (nodes : t list) : (int, unit) Hashtbl.t =
  let cls = classify_passes sp nodes in
  let all = compute_inline_set assigns in
  let consumers : (int, t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun e ->
      List.iter
        (fun p ->
          let prev = try Hashtbl.find consumers p.tag with Not_found -> [] in
          Hashtbl.replace consumers p.tag (e :: prev))
        (preds e))
    nodes;
  let filtered = Hashtbl.create 64 in
  Hashtbl.iter
    (fun tag () ->
      if not (is_spilled sp tag) then begin
        let producer_class = Hashtbl.find_opt cls tag in
        let consumer_classes =
          match Hashtbl.find_opt consumers tag with
          | None -> []
          | Some cs -> List.map (fun c -> Hashtbl.find_opt cls c.tag) cs
        in
        if
          consumer_classes <> []
          && List.for_all (fun cc -> cc = producer_class) consumer_classes
        then Hashtbl.add filtered tag ()
      end)
    all;
  filtered

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
    [
      "VFFT_N1_BLOCK_MIN";
      "VFFT_NO_REGALLOC";
      "VFFT_PIN_FORCE";
      "VFFT_CT_FACTOR";
      "VFFT_SPLIT_RADIX";
      "VFFT_COLLECT_M";
      "VFFT_DEEP_COLLECT";
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
  match act with [] -> "(none)" | l -> String.concat " " l

let provenance_block ~(family : string) (lines : string list) : string =
  let b = Buffer.create 1024 in
  Buffer.add_string b
    "/* ===================== PROVENANCE =====================\n";
  Buffer.add_string b
    (Printf.sprintf " * Generated by: %s\n"
       (String.concat " "
          (Array.to_list
             (match !provenance_argv with Some a -> a | None -> Sys.argv))));
  Buffer.add_string b
    (Printf.sprintf " * Env overrides: %s\n" (provenance_env_overrides ()));
  Buffer.add_string b (Printf.sprintf " * Family: %s\n" family);
  List.iter (fun l -> Buffer.add_string b (" * " ^ l ^ "\n")) lines;
  Buffer.add_string b
    " * ====================================================== */\n";
  Buffer.contents b

let emit_codelet ?(in_place = false) ?(t1s = false) ?(twidsq = false)
    ?(twidsq_n = 0) ?(strided = false) ?(radix = 0) ?(scheduler = Topological)
    ?(isa = Isa.avx512) ?(gh = false) ?(bb_budget : float option = None)
    ?(spill : spill_info option = None) ?(is_log3 = false)
    (assigns : (Expr.elem_ref * t) list) ~(name : string) : string =
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
    try Sys.getenv "VFFT_PEAK_LIVE" = "1" with Not_found -> false
  in
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
    if peak_live_enabled then
      Printf.eprintf "[%s:%s] %s\n" name label (Regalloc.format_live_info info)
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
      if i + plen > slen then false
      else if String.sub s i plen = pat then true
      else aux (i + 1)
    in
    aux 0
  in
  let is_n1 = contains_substring name "_n1_" in
  let is_avx2 = isa.Isa.vec_regs <= 16 in
  let opt_out =
    try Sys.getenv "VFFT_NO_REGALLOC" = "1" with Not_found -> false
  in
  let force_pin =
    try Sys.getenv "VFFT_PIN_FORCE" = "1" with Not_found -> false
  in
  let force_fence =
    try Sys.getenv "VFFT_FORCE_FENCE" = "1" with Not_found -> false
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
  current_fence_only := fence_enabled && not regalloc_enabled;
  let install_alloc (label : string) (scheduled : t list)
      (inline_set : (int, unit) Hashtbl.t option)
      (force_last_use : (int, int) Hashtbl.t option) =
    if regalloc_enabled then
      begin match
        Regalloc.allocate ~isa ~scheduled ~inline_set ~force_last_use ()
      with
      | Regalloc.Allocated alloc ->
          current_regalloc := Some alloc;
          current_unpin_candidates := Some (compute_unpin_candidates scheduled);
          let regs, _ = Regalloc.count_bindings alloc in
          Printf.eprintf "[%s:%s] regalloc: %d tags bound\n" name label regs
      | Regalloc.Overflow budget ->
          current_regalloc := None;
          current_unpin_candidates := None;
          Printf.eprintf
            "[%s:%s] regalloc: OVERFLOW budget=%d, falling back to default\n"
            name label budget
      end
  in
  let clear_alloc () =
    current_regalloc := None;
    current_unpin_candidates := None;
    current_fence_only := false
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
  let install_alloc_canonical (label : string) (input : Regalloc.regalloc_input)
      =
    if regalloc_enabled then
      begin match
        Regalloc.allocate ~isa ~scheduled:input.scheduled
          ~inline_set:input.inline_set ~force_last_use:input.force_last_use ()
      with
      | Regalloc.Allocated alloc ->
          current_regalloc := Some alloc;
          current_unpin_candidates :=
            Some (compute_unpin_candidates input.scheduled);
          let regs, _ = Regalloc.count_bindings alloc in
          Printf.eprintf "[%s:%s] regalloc: %d tags bound\n" name label regs
      | Regalloc.Overflow budget ->
          current_regalloc := None;
          current_unpin_candidates := None;
          Printf.eprintf
            "[%s:%s] regalloc: OVERFLOW budget=%d, falling back to default\n"
            name label budget
      end
  in
  let _ = install_alloc_canonical in
  let emit_regalloc_spill_decl (buf : Buffer.t) =
    match !current_regalloc with
    | Some alloc when alloc.num_spill_slots > 0 ->
        Buffer.add_string buf
          (Printf.sprintf "        %s regalloc_spill[%d];\n" isa.vec_type
             alloc.num_spill_slots)
    | _ -> ()
  in
  let emit_node_spill_sites (buf : Buffer.t) (pos : int) =
    match !current_regalloc with
    | Some alloc -> (
        match Hashtbl.find_opt alloc.spill_sites pos with
        | Some spills ->
            List.iter
              (fun (tag, slot) ->
                Buffer.add_string buf
                  (Printf.sprintf "        %s(&regalloc_spill[%d], t%d);\n"
                     isa.storeu_pd slot tag))
              spills
        | None -> ())
    | None -> ()
  in
  let emit_node_reload_sites (buf : Buffer.t) (pos : int) =
    match !current_regalloc with
    | Some alloc -> (
        match Hashtbl.find_opt alloc.reload_sites pos with
        | Some reloads ->
            List.iter
              (fun (r : Regalloc.reload_decl) ->
                Buffer.add_string buf
                  (Printf.sprintf "        %s\n"
                     (Isa.pinned_reg_decl isa r.reload_name r.reload_reg
                        (Printf.sprintf "%s(&regalloc_spill[%d])" isa.loadu_pd
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
  if twidsq && in_place then
    failwith "emit_codelet: twidsq and in_place are mutually exclusive";
  if strided && (twidsq || in_place) then
    failwith "emit_codelet: strided not yet supported with twidsq or in_place";
  if strided && radix <= 0 then
    failwith "emit_codelet: strided requires --radix > 0";
  if strided && radix mod isa.vec_width <> 0 then
    failwith
      (Printf.sprintf
         "emit_codelet: strided requires radix divisible by vec_width=%d (got \
          %d)"
         isa.vec_width radix);
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
   * legacy K%VW==0-only loop. *)
  let anyk_tail =
    in_place
    && (match Sys.getenv_opt "VFFT_NO_ANYK_TAIL" with
       | Some _ -> false
       | None -> true)
  in
  let family =
    if strided then "strided-batch (Design C, 2D rows)"
    else if twidsq then "twidsq (FFTW-style intermediate)"
    else if is_n1 then if in_place then "in-place n1 (no twiddles)" else "n1"
    else if t1s then "t1s (broadcast twiddles)"
    else if in_place then "in-place t1 (twiddled CT)"
    else "t1"
  in
  let sched_str =
    match scheduler with
    | Topological -> "topological (plain dependency order)"
    | Bisection ->
        "Frigo recursive bisection (repaired, sections 27-28; not a production \
         default)"
    | SU _ ->
        "the list scheduler (flag name --su; lazy loads, sink-first, cp_dist, \
         SU tiebreak; section 30)"
    | _ -> "annotated/experimental"
  in
  Buffer.add_string buf
    (provenance_block ~family
       [
         Printf.sprintf "ISA: %d-bit vectors, %d vector regs%s"
           (isa.Isa.vec_width * 64) isa.Isa.vec_regs
           (if is_avx2 then " (16-reg pressure rules apply)" else "");
         Printf.sprintf "Scheduler: %s" sched_str;
         Printf.sprintf
           "GH pressure mode: %b (auto-rule: vec_regs<=16 && n>=32; +4-8%% \
            documented)"
           gh;
         (match spill with
         | Some sp when sp.ct_n1 > 0 ->
             Printf.sprintf
               "Construction: BLOCKED two-pass CT %dx%d; seam through L1 \
                scratch by design (doc 58); threshold n>=16 on <=16-reg ISAs \
                else 25 (section 35)"
               sp.ct_n1 sp.ct_n2
         | Some _ -> "Construction: blocked (spill recipe, non-CT marker set)"
         | None ->
             "Construction: MONOLITHIC (below blocking threshold, or \
              prime/Direct: no CT pass boundary)");
         Printf.sprintf
           "Regalloc+pinning: %b (gate: log3+avx512+R<=32 OR avx2 n1 R>=16, \
            sections 32/34/35; kill switch VFFT_NO_REGALLOC)"
           regalloc_enabled;
         Printf.sprintf
           "Value fences: %b (asm-volatile +v stops gcc rematerialization, doc \
            28; exempt: n1 avx2 R in {8,16}, measured better unfenced)"
           fence_enabled;
         (if is_log3 then "log3: DIT radix-decomposition ladder variant"
          else "log3: no");
       ]);
  Buffer.add_string buf "#include <immintrin.h>\n";
  Buffer.add_string buf "#include <stddef.h>\n\n";
  (* AVX2 masked tail needs an __m256i lane mask: row r has r leading all-ones
   * (active) lanes. AVX-512 uses an __mmask8 = (1<<rem)-1 computed inline, so
   * no table. File-scope `static const`, one per codelet .c (name is static =
   * file-local, no cross-file clash). *)
  if anyk_tail && isa.vec_width = 4 then begin
    let vw = isa.vec_width in
    Buffer.add_string buf
      (Printf.sprintf "static const long long _vfft_masklo[%d][%d] = {\n    "
         (vw + 1) vw);
    for r = 0 to vw do
      if r > 0 then Buffer.add_string buf ",";
      Buffer.add_char buf '{';
      for c = 0 to vw - 1 do
        if c > 0 then Buffer.add_char buf ',';
        Buffer.add_string buf (if c < r then "-1" else "0")
      done;
      Buffer.add_char buf '}'
    done;
    Buffer.add_string buf "};\n\n"
  end;
  if isa.vec_width = 1 then begin
    Buffer.add_string buf
      "static inline double vfft_scalar_load(const double *p) { return *p; }\n";
    Buffer.add_string buf
      "static inline void vfft_scalar_store(double *p, double v) { *p = v; }\n\n"
  end;
  Buffer.add_string buf
    (Printf.sprintf "__attribute__((target(\"%s\")))\n" isa.target_attr);
  Buffer.add_string buf (Printf.sprintf "void %s(\n" name);

  if strided then begin
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
    Buffer.add_string buf "    double       * __restrict__ rio_re,\n";
    Buffer.add_string buf "    double       * __restrict__ rio_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    size_t row_stride,\n";
    Buffer.add_string buf "    size_t me)\n";
    Buffer.add_string buf "{\n";
    Buffer.add_string buf "    (void)tw_re; (void)tw_im;\n";
    (* AVX-512 only: pre-declare the two __m512i index vectors used by the
     * 8×8 transpose preamble AND postamble. Function-scope (outside the
     * b loop) so gcc treats them as loop-invariant constants. The
     * indices match transpose.h Kernel C — idx_lo gathers even-column
     * cross-lane elements, idx_hi gathers odd-column. *)
    if isa.vec_width = 8 then begin
      Buffer.add_string buf
        "    const __m512i _tp_idx_lo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, \
         1, 0);\n";
      Buffer.add_string buf
        "    const __m512i _tp_idx_hi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, \
         3, 2);\n"
    end;
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t b = 0; b < me; b += %d) {\n"
         isa.vec_width);
    (* Per-iteration locals: lane_re_0..radix-1 (inputs after transpose),
       out_lane_re_0..radix-1 (outputs before inverse transpose). Plus
       _im versions. *)
    for j = 0 to radix - 1 do
      Buffer.add_string buf
        (Printf.sprintf "        %s lane_re_%d, lane_im_%d;\n" isa.vec_type j j);
      Buffer.add_string buf
        (Printf.sprintf "        %s out_lane_re_%d, out_lane_im_%d;\n"
           isa.vec_type j j)
    done;
    Buffer.add_string buf "\n";
    (* AVX2 4×4 transpose preamble. For each group of 4 consecutive fft
     * indices (j0, j0+1, j0+2, j0+3), load 4 rows of 4 consecutive cols
     * starting at fft_idx=j0, then 4×4 transpose to get 4 lane vectors
     * each holding (row 0, row 1, row 2, row 3) at one fft_idx. *)
    if isa.vec_width = 4 then begin
      let groups = radix / 4 in
      for which_side = 0 to 1 do
        let suf = if which_side = 0 then "re" else "im" in
        for g = 0 to groups - 1 do
          let j0 = g * 4 in
          Buffer.add_string buf
            (Printf.sprintf
               "        {  /* 4x4 transpose group: fft_idx %d..%d, %s */\n" j0
               (j0 + 3) suf);
          for r = 0 to 3 do
            Buffer.add_string buf
              (Printf.sprintf
                 "            const __m256d _row_%s_%d = \
                  _mm256_loadu_pd(&rio_%s[(b+%d)*row_stride + %d]);\n"
                 suf r suf r j0)
          done;
          (* 4×4 transpose: r0 r1 r2 r3 (each row = 4 cols) → c0 c1 c2 c3 *)
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m256d _t0_%s = \
                _mm256_unpacklo_pd(_row_%s_0, _row_%s_1);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m256d _t1_%s = \
                _mm256_unpackhi_pd(_row_%s_0, _row_%s_1);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m256d _t2_%s = \
                _mm256_unpacklo_pd(_row_%s_2, _row_%s_3);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m256d _t3_%s = \
                _mm256_unpackhi_pd(_row_%s_2, _row_%s_3);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm256_permute2f128_pd(_t0_%s, \
                _t2_%s, 0x20);\n"
               suf j0 suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm256_permute2f128_pd(_t1_%s, \
                _t3_%s, 0x20);\n"
               suf (j0 + 1) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm256_permute2f128_pd(_t0_%s, \
                _t2_%s, 0x31);\n"
               suf (j0 + 2) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm256_permute2f128_pd(_t1_%s, \
                _t3_%s, 0x31);\n"
               suf (j0 + 3) suf suf);
          Buffer.add_string buf "        }\n"
        done
      done
    end
    else if isa.vec_width = 8 then begin
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
      for which_side = 0 to 1 do
        let suf = if which_side = 0 then "re" else "im" in
        for g = 0 to groups - 1 do
          let j0 = g * 8 in
          Buffer.add_string buf
            (Printf.sprintf
               "        {  /* 8x8 transpose group: fft_idx %d..%d, %s */\n" j0
               (j0 + 7) suf);
          (* 8 row loads *)
          for r = 0 to 7 do
            Buffer.add_string buf
              (Printf.sprintf
                 "            const __m512d _row_%s_%d = \
                  _mm512_loadu_pd(&rio_%s[(b+%d)*row_stride + %d]);\n"
                 suf r suf r j0)
          done;
          (* Stage 1: 8 unpacklo/unpackhi_pd. *)
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t0_%s = \
                _mm512_unpacklo_pd(_row_%s_0, _row_%s_1);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t1_%s = \
                _mm512_unpackhi_pd(_row_%s_0, _row_%s_1);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t2_%s = \
                _mm512_unpacklo_pd(_row_%s_2, _row_%s_3);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t3_%s = \
                _mm512_unpackhi_pd(_row_%s_2, _row_%s_3);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t4_%s = \
                _mm512_unpacklo_pd(_row_%s_4, _row_%s_5);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t5_%s = \
                _mm512_unpackhi_pd(_row_%s_4, _row_%s_5);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t6_%s = \
                _mm512_unpacklo_pd(_row_%s_6, _row_%s_7);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _t7_%s = \
                _mm512_unpackhi_pd(_row_%s_6, _row_%s_7);\n"
               suf suf suf);
          (* Stage 2: 8 permutex2var_pd. *)
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x0_%s = \
                _mm512_permutex2var_pd(_t0_%s, _tp_idx_lo, _t2_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x1_%s = \
                _mm512_permutex2var_pd(_t1_%s, _tp_idx_lo, _t3_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x2_%s = \
                _mm512_permutex2var_pd(_t0_%s, _tp_idx_hi, _t2_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x3_%s = \
                _mm512_permutex2var_pd(_t1_%s, _tp_idx_hi, _t3_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x4_%s = \
                _mm512_permutex2var_pd(_t4_%s, _tp_idx_lo, _t6_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x5_%s = \
                _mm512_permutex2var_pd(_t5_%s, _tp_idx_lo, _t7_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x6_%s = \
                _mm512_permutex2var_pd(_t4_%s, _tp_idx_hi, _t6_%s);\n"
               suf suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            const __m512d _x7_%s = \
                _mm512_permutex2var_pd(_t5_%s, _tp_idx_hi, _t7_%s);\n"
               suf suf suf);
          (* Stage 3: 8 shuffle_f64x2, assign directly to lane_re_{j0..j0+7}. *)
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x0_%s, _x4_%s, \
                0x44);\n"
               suf j0 suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x1_%s, _x5_%s, \
                0x44);\n"
               suf (j0 + 1) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x2_%s, _x6_%s, \
                0x44);\n"
               suf (j0 + 2) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x3_%s, _x7_%s, \
                0x44);\n"
               suf (j0 + 3) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x0_%s, _x4_%s, \
                0xEE);\n"
               suf (j0 + 4) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x1_%s, _x5_%s, \
                0xEE);\n"
               suf (j0 + 5) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x2_%s, _x6_%s, \
                0xEE);\n"
               suf (j0 + 6) suf suf);
          Buffer.add_string buf
            (Printf.sprintf
               "            lane_%s_%d = _mm512_shuffle_f64x2(_x3_%s, _x7_%s, \
                0xEE);\n"
               suf (j0 + 7) suf suf);
          Buffer.add_string buf "        }\n"
        done
      done
    end
    else begin
      failwith
        (Printf.sprintf "emit_codelet: strided not supported for vec_width=%d"
           isa.vec_width)
    end;
    Buffer.add_string buf "\n"
  end
  else if in_place then begin
    Buffer.add_string buf "    double       * __restrict__ rio_re,\n";
    Buffer.add_string buf "    double       * __restrict__ rio_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    size_t ios,\n";
    Buffer.add_string buf "    size_t me)\n";
    Buffer.add_string buf "{\n";
    (* Spill array decl, OUTSIDE the for loop so it's allocated once *)
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    if anyk_tail then begin
      (* Hoist k so it stays live for the remainder block after the bulk loop. *)
      Buffer.add_string buf "    size_t k = 0;\n";
      Buffer.add_string buf
        (Printf.sprintf "    for (; k + %d <= me; k += %d) {\n" isa.vec_width
           isa.vec_width)
    end
    else
      Buffer.add_string buf
        (Printf.sprintf "    for (size_t k = 0; k < me; k += %d) {\n"
           isa.vec_width)
  end
  else if twidsq then begin
    (* Twidsq OOP signature with separate input/output strides.
     *   in_re[slot * is + v] for input element at slot i*n+k
     *   out_re[slot * os + v] for output element at slot j*n+i (TRANSPOSED
     *     — the math layer already encodes the transpose via Output indices)
     *   Twiddles broadcast across V lanes (uniform across batches).
     *   V is the loop bound; vec_width lanes processed per iteration.
     *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    size_t is,\n";
    Buffer.add_string buf "    size_t os,\n";
    Buffer.add_string buf "    size_t V)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < V; v += %d) {\n" isa.vec_width)
  end
  else if !r2cb_signature then begin
    (* r2cb backward real leaf (section 62): halfcomplex INPUT (in_re, in_im)
     * -> real OUTPUT (out_re). The body reconstructs the conjugate-symmetric
     * spectrum from the packed half and runs a backward DFT; the result is
     * purely real so there is no out_im. Same stride/loop shape as r2cf
     * (is input stride, os_re output stride, vl lanes). *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    ptrdiff_t is_re,\n";
    Buffer.add_string buf "    ptrdiff_t is_im,\n";
    Buffer.add_string buf "    ptrdiff_t os_re,\n";
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !r2cf_signature then begin
    (* r2cf leaf v2 (section 62): the composition algebra forces a
     * REVERSED im output stream (packed im slot for k lands at
     * position n-k, walking DOWN as k walks up), so the strides are
     * signed and split per parity. The executor passes os_im < 0 with
     * out_im based one-past the region. P1's stride_n1_fn-shaped v1
     * is withdrawn — composition beats typedef aesthetics. *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    ptrdiff_t is,\n";
    Buffer.add_string buf "    ptrdiff_t os_re,\n";
    Buffer.add_string buf "    ptrdiff_t os_im,\n";
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !r2c_term_laststage then begin
    (* model (b): fused last-stage terminator. Two columns of r complex legs +
     * packed twiddle table (3r slots) -> 2r outputs (Xp[s], Xm[s]).
     * Input(j) for j<r = col k leg j; Input(r+j) = col m-k leg j. Legs are
     * strided by is_leg within each column; the two columns are at separate
     * base pointers in_k / in_m (the executor passes the two physical rows). *)
    Buffer.add_string buf "    const double * __restrict__ ink_re,\n";
    Buffer.add_string buf "    const double * __restrict__ ink_im,\n";
    Buffer.add_string buf "    const double * __restrict__ inm_re,\n";
    Buffer.add_string buf "    const double * __restrict__ inm_im,\n";
    Buffer.add_string buf "    double       * __restrict__ Xp_re,\n";
    Buffer.add_string buf "    double       * __restrict__ Xp_im,\n";
    Buffer.add_string buf "    double       * __restrict__ Xm_re,\n";
    Buffer.add_string buf "    double       * __restrict__ Xm_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    ptrdiff_t is_leg,\n";
    (* stride between legs within a column *)
    Buffer.add_string buf "    ptrdiff_t osp,\n";
    (* stride between Xp output slots *)
    Buffer.add_string buf "    ptrdiff_t osm,\n";
    (* stride between Xm output slots *)
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !r2c_term_signature then begin
    (* r2c_term (step-2 fusion): 2 inputs (Z[k], Z[m]) -> 2 outputs (X[k], X[m]),
     * vectorized over vl lanes. Reads scratch rows for the column pair
     * sequentially (the executor supplies them in natural order, no scatter).
     * Output(0) -> Xp pair (X[k]); Output(1) -> Xm pair (X[m]). *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ Xp_re,\n";
    Buffer.add_string buf "    double       * __restrict__ Xp_im,\n";
    Buffer.add_string buf "    double       * __restrict__ Xm_re,\n";
    Buffer.add_string buf "    double       * __restrict__ Xm_im,\n";
    if !r2c_term_rt then begin
      Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
      Buffer.add_string buf "    const double * __restrict__ tw_im,\n"
    end;
    Buffer.add_string buf "    ptrdiff_t is,\n";
    (* row stride between Z[k] and Z[m] inputs *)
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !hc2c_natural then begin
    (* D2 natural terminator (section 69): four output pointers,
     * boundary baked at generation time. *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ Rp,\n";
    Buffer.add_string buf "    double       * __restrict__ Ip,\n";
    Buffer.add_string buf "    double       * __restrict__ Rm,\n";
    Buffer.add_string buf "    double       * __restrict__ Im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    ptrdiff_t is,\n";
    Buffer.add_string buf "    ptrdiff_t osp,\n";
    Buffer.add_string buf "    ptrdiff_t osm,\n";
    if !hc_ranged then begin
      Buffer.add_string buf "    ptrdiff_t cs_in,\n";
      Buffer.add_string buf "    ptrdiff_t cs_out,\n";
      Buffer.add_string buf "    int kcount,\n"
    end;
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    if !hc_ranged then
      Buffer.add_string buf "    for (int kc = 0; kc < kcount; kc++) {\n";
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !hc2c_natural_bwd then begin
    (* c2r natural INITIATOR (inverse of hc2c_natural): four SPLIT const inputs
     * (Rp/Ip direct rows + Rm/Im conjugate-mirror rows, the same sstar map as
     * the forward but on the INPUT side) -> two PACKED cascade columns
     * (out_re/out_im). isp/ism = split input row strides; os = packed output
     * stride. The forward's 6-pointer ABI, flipped. *)
    Buffer.add_string buf "    const double * __restrict__ Rp,\n";
    Buffer.add_string buf "    const double * __restrict__ Ip,\n";
    Buffer.add_string buf "    const double * __restrict__ Rm,\n";
    Buffer.add_string buf "    const double * __restrict__ Im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    ptrdiff_t isp,\n";
    Buffer.add_string buf "    ptrdiff_t ism,\n";
    Buffer.add_string buf "    ptrdiff_t os,\n";
    if !hc_ranged then begin
      Buffer.add_string buf "    ptrdiff_t cs_in,\n";
      Buffer.add_string buf "    ptrdiff_t cs_out,\n";
      Buffer.add_string buf "    int kcount,\n"
    end;
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    if !hc_ranged then
      Buffer.add_string buf "    for (int kc = 0; kc < kcount; kc++) {\n";
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !hc_strided then begin
    (* hc2hc / hc2c strided variant (section 62): the generic ABI's
     * hardcoded slot stride K cannot address middle cascade stages
     * (slot strides are Q*K-multiples and out != in stride). Twiddles
     * replicate per vl lanes, slot 0 never loaded. *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    ptrdiff_t is,\n";
    Buffer.add_string buf "    ptrdiff_t os,\n";
    if !hc_ranged then begin
      Buffer.add_string buf "    ptrdiff_t cs_in,\n";
      Buffer.add_string buf "    ptrdiff_t cs_out,\n";
      Buffer.add_string buf "    int kcount,\n"
    end;
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    if !hc_ranged then
      Buffer.add_string buf "    for (int kc = 0; kc < kcount; kc++) {\n";
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !n1_oop_strided then begin
    (* strided-OOP n1: ABI = vfft_proto_n1_fn (registry n1 slot shape).
     * size_t strides (not ptrdiff_t) to match the fn-pointer type
     * byte-for-byte; vl assumed a multiple of the vector width (the r2c
     * executor always passes B, a vec-width multiple). No tw params:
     * n1 DAGs carry no Twiddle refs. *)
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    size_t is,\n";
    Buffer.add_string buf "    size_t os,\n";
    Buffer.add_string buf "    size_t vl)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t v = 0; v < vl; v += %d) {\n"
         isa.vec_width)
  end
  else if !r2r_signature then begin
    Buffer.add_string buf "    const double * __restrict__ in,\n";
    Buffer.add_string buf "    double       * __restrict__ out,\n";
    Buffer.add_string buf "    size_t K)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t k = 0; k < K; k += %d) {\n" isa.vec_width)
  end
  else begin
    Buffer.add_string buf "    const double * __restrict__ in_re,\n";
    Buffer.add_string buf "    const double * __restrict__ in_im,\n";
    Buffer.add_string buf "    double       * __restrict__ out_re,\n";
    Buffer.add_string buf "    double       * __restrict__ out_im,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
    Buffer.add_string buf "    const double * __restrict__ tw_im,\n";
    Buffer.add_string buf "    size_t K)\n";
    Buffer.add_string buf "{\n";
    (match spill with
    | None -> ()
    | Some sp ->
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_re[%d];\n" isa.vec_type sp.num_slots);
        Buffer.add_string buf
          (Printf.sprintf "    %s spill_im[%d];\n" isa.vec_type sp.num_slots));
    Buffer.add_string buf
      (render_hoisted_consts ~isa (topo_sort_reachable (List.map snd assigns)));
    Buffer.add_string buf
      (Printf.sprintf "    for (size_t k = 0; k < K; k += %d) {\n" isa.vec_width)
  end;

  let out_buf is_re =
    match (in_place, is_re) with
    | true, true -> "rio_re"
    | true, false -> "rio_im"
    | false, true -> if !r2r_signature then "out" else "out_re"
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
    if in_place then "ios"
    else if twidsq then "os"
    else if !hc_strided || !n1_oop_strided then "os"
    else "K"
  in
  let out_stride_for is_re =
    if !r2cf_signature then if is_re then "os_re" else "os_im"
    else if !r2cb_signature then "os_re"
    else out_stride
  in
  let loop_var =
    if
      twidsq || !r2cf_signature || !r2cb_signature || !hc_strided
      || !n1_oop_strided || !r2c_term_signature || !r2c_term_laststage
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
  let render_output_addr k is_re =
    if !r2c_term_laststage then
      (* Output(2s) = Xp slot s at Xp[s*osp+v]; Output(2s+1) = Xm slot s. *)
      begin if k land 1 = 0 then
        Printf.sprintf "%s[%d*osp + %s]"
          (if is_re then "Xp_re" else "Xp_im")
          (k / 2) loop_var
      else
        Printf.sprintf "%s[%d*osm + %s]"
          (if is_re then "Xm_re" else "Xm_im")
          (k / 2) loop_var
      end
    else if !r2c_term_signature then
      (* Output(0)=X[k]->Xp pair; Output(1)=X[m]->Xm pair, over v lanes. *)
      begin if k = 0 then
        Printf.sprintf "%s[%s]" (if is_re then "Xp_re" else "Xp_im") loop_var
      else Printf.sprintf "%s[%s]" (if is_re then "Xm_re" else "Xm_im") loop_var
      end
    else if !hc2c_natural then
      begin if k <= !hc2c_nat_sstar then
        Printf.sprintf "%s[%d*osp + %s]"
          (if is_re then "Rp" else "Ip")
          k loop_var
      else
        Printf.sprintf "%s[%d*osm + %s]"
          (if is_re then "Rm" else "Im")
          (!hc2c_nat_r - 1 - k)
          loop_var
      end
    else
      let buf = out_buf is_re in
      if twidsq && twidsq_n > 0 then
        let row = k / twidsq_n in
        let col = k mod twidsq_n in
        Printf.sprintf "%s[%d*%s + %d*V + %s]" buf row out_stride col loop_var
      else Printf.sprintf "%s[%d*%s + %s]" buf k (out_stride_for is_re) loop_var
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
      match !current_regalloc with
      | None -> Printf.sprintf "t%d" e.tag
      | Some alloc -> (
          match
            Hashtbl.find_opt alloc.name_overrides (!current_emit_position, e.tag)
          with
          | Some n -> n
          | None -> (
              match Hashtbl.find_opt alloc.spilled_of_tag e.tag with
              | Some slot ->
                  (* Inline load from regalloc_spill. gcc picks a temp. *)
                  Printf.sprintf "%s(&regalloc_spill[%d])" isa.loadu_pd slot
              | None -> Printf.sprintf "t%d" e.tag))
    in
    match oref with
    | Expr.Output (k, true) when strided ->
        Buffer.add_string buf
          (Printf.sprintf "        out_lane_re_%d = %s;\n" k value_expr)
    | Expr.Output (k, false) when strided ->
        Buffer.add_string buf
          (Printf.sprintf "        out_lane_im_%d = %s;\n" k value_expr)
    | Expr.Output (k, true) ->
        Buffer.add_string buf "        ";
        Buffer.add_string buf
          (Isa.storeu_pd ~mode:!current_ls_mode isa (render_output_addr k true)
             value_expr);
        Buffer.add_string buf ";\n"
    | Expr.Output (k, false) ->
        Buffer.add_string buf "        ";
        Buffer.add_string buf
          (Isa.storeu_pd ~mode:!current_ls_mode isa (render_output_addr k false)
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
  (match (if force_mono then None else spill) with
  | Some sp ->
      let roots = List.map snd assigns in
      let nodes = topo_sort_reachable roots in
      let cls = classify_passes sp nodes in

      (* Single-use inlining for the spill path. A tag is inlinable iff
       * it has exactly one consumer (compute_inline_set), is not a
       * Load/Const/Cmul, is NOT spilled, and has all consumers in the
       * SAME pass as the producer. Single source of truth shared with
       * codelet_oop: filter_inline_set_cross_pass (section 37). *)
      let inline_set = filter_inline_set_cross_pass assigns sp nodes in
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
      let is_const e = match e.node with NK_Const _ -> true | _ -> false in
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
        List.filter
          (fun (_, e) -> Hashtbl.find_opt cls e.tag = Some `Pass1)
          assigns
      in
      let pass2_assigns =
        List.filter
          (fun (_, e) -> Hashtbl.find_opt cls e.tag = Some `Pass2)
          assigns
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
          Buffer.add_string buf
            (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e);
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
      if fused_pass1_tags <> [] then begin
        let names =
          List.map (fun t -> Printf.sprintf "t%d" t) fused_pass1_tags
        in
        Buffer.add_string buf "        ";
        Buffer.add_string buf (Isa.forward_decl isa names);
        Buffer.add_char buf '\n';
        Buffer.add_char buf '\n'
      end;

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
      let min_slot, pass1_blocked_topo =
        compute_min_slot_pass1 sp pass1_nodes
      in

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
            cluster_split_schedule sp ~pass1_blocked_topo ~min_slot
              ~schedule_cluster:(fun ~subset ~sinks ->
                match bb_budget with
                | None -> Schedule.su_schedule_subset uarch ~gh ~subset ~sinks
                | Some t ->
                    Bb.bb_schedule_subset uarch ~time_budget_sec:t ~subset
                      ~sinks)
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
      install_alloc "spill_pass1" pass1_blocked (Some inline_set)
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
      (match !current_regalloc with
      | Some alloc when alloc.num_spill_slots > 0 ->
          Buffer.add_string buf
            (Printf.sprintf "            %s regalloc_spill[%d];\n" isa.vec_type
               alloc.num_spill_slots)
      | _ -> ());
      List.iteri
        (fun pos e ->
          current_emit_position := pos;
          (* M5: emit spill stores for any tags evicted at this position.
           * The eviction was decided by the allocator (pool empty); the
           * store happens BEFORE the new def overwrites the register.
           *
           * ORDER: spill stores must precede reload loads. If a tag T is
           * spilled AND reloaded at the same position (because T was
           * evicted at p AND T has force_last_use[T] = p), the spill
           * writes T's value to slot S; the reload then reads S. If we
           * reloaded first, the slot would be uninitialized. *)
          (match !current_regalloc with
          | Some alloc -> (
              match Hashtbl.find_opt alloc.spill_sites pos with
              | Some spills ->
                  List.iter
                    (fun (tag, slot) ->
                      Buffer.add_string buf
                        (Printf.sprintf
                           "            %s(&regalloc_spill[%d], t%d);\n"
                           isa.storeu_pd slot tag))
                    spills
              | None -> ())
          | None -> ());
          (* M5: emit any reload declarations for this position before the
           * node's own def. Each reload is a fresh register-pinned load
           * from regalloc_spill[slot] into a shadow variable tT_rK. *)
          (match !current_regalloc with
          | Some alloc -> (
              match Hashtbl.find_opt alloc.reload_sites pos with
              | Some reloads ->
                  List.iter
                    (fun (r : Regalloc.reload_decl) ->
                      Buffer.add_string buf
                        (Printf.sprintf "        %s\n"
                           (Isa.pinned_reg_decl isa r.reload_name r.reload_reg
                              (Printf.sprintf "%s(&regalloc_spill[%d])"
                                 isa.loadu_pd r.reload_slot))))
                    reloads
              | None -> ())
          | None -> ());
          if is_inlined e then ()
          else begin
            let no_declarator = is_fused_tag e.tag in
            Buffer.add_string buf
              (render_node_def ~no_declarator ~t1s ~isa ~in_place ~twidsq
                 ~twidsq_n ~strided ~inline_set:(Some inline_set) e);
            Buffer.add_char buf '\n';
            if not no_declarator then begin
              (match lookup_re_slot e.tag with
              | Some slot ->
                  Buffer.add_string buf
                    (Printf.sprintf "            %s(&spill_re[%d], t%d);\n"
                       isa.storeu_pd slot e.tag)
              | None -> ());
              match lookup_im_slot e.tag with
              | Some slot ->
                  Buffer.add_string buf
                    (Printf.sprintf "            %s(&spill_im[%d], t%d);\n"
                       isa.storeu_pd slot e.tag)
              | None -> ()
            end
          end)
        pass1_blocked;
      (* Emit stores for Pass 1 outputs at end of PASS 1 — values are still
       * in scope here. Pass 2 outputs are stored later, inside PASS 2's
       * scope (per-cluster flush + safety net).
       *
       * M5: pass 1's force_last_use put pass1_assigns tags at position
       * pass1_n (one past last). Reloads registered there. Set
       * current_emit_position so emit_store sees the right overrides. *)
      let pass1_n = List.length pass1_blocked in
      current_emit_position := pass1_n;
      (* M5: spill stores BEFORE reload loads at end-of-pass. *)
      (match !current_regalloc with
      | Some alloc -> (
          match Hashtbl.find_opt alloc.spill_sites pass1_n with
          | Some spills ->
              List.iter
                (fun (tag, slot) ->
                  Buffer.add_string buf
                    (Printf.sprintf
                       "            %s(&regalloc_spill[%d], t%d);\n"
                       isa.storeu_pd slot tag))
                spills
          | None -> ())
      | None -> ());
      (match !current_regalloc with
      | Some alloc -> (
          match Hashtbl.find_opt alloc.reload_sites pass1_n with
          | Some reloads ->
              List.iter
                (fun (r : Regalloc.reload_decl) ->
                  Buffer.add_string buf
                    (Printf.sprintf "        %s\n"
                       (Isa.pinned_reg_decl isa r.reload_name r.reload_reg
                          (Printf.sprintf "%s(&regalloc_spill[%d])" isa.loadu_pd
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
      if sp.ct_n2 > 0 then begin
        let min_input_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
        List.iter
          (fun e ->
            let direct_slot =
              match
                ( Hashtbl.find_opt sp.re_slot e.tag,
                  Hashtbl.find_opt sp.im_slot e.tag )
              with
              | Some s, _ | _, Some s -> Some s
              | None, None -> None
            in
            let pred_min =
              List.fold_left
                (fun acc p ->
                  match Hashtbl.find_opt min_input_slot p.tag with
                  | Some s -> (
                      match acc with None -> Some s | Some a -> Some (min a s))
                  | None -> acc)
                None (preds e)
            in
            let my_min =
              match (direct_slot, pred_min) with
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
                  try Hashtbl.find consumers_p2 p.tag with Not_found -> []
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
              if not (Hashtbl.mem first_walk_assigned e.tag) then begin
                let cs =
                  try Hashtbl.find consumers_p2 e.tag with Not_found -> []
                in
                let consumer_cluster =
                  List.fold_left
                    (fun acc c ->
                      match
                        (acc, Hashtbl.find_opt cluster_of_pass2_node c.tag)
                      with
                      | None, Some k -> Some k
                      | Some a, Some k -> Some (min a k)
                      | _, None -> acc)
                    None cs
                in
                match
                  ( consumer_cluster,
                    Hashtbl.find_opt cluster_of_pass2_node e.tag )
                with
                | Some k, None ->
                    Hashtbl.add cluster_of_pass2_node e.tag k;
                    changed := true
                | Some new_k, Some old_k when new_k < old_k ->
                    Hashtbl.replace cluster_of_pass2_node e.tag new_k;
                    changed := true
                | _ -> ()
              end)
            pass2_nodes
        done
      end;
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
                (Hashtbl.create 32) assigns
            in
            let result = ref [] in
            (* Emit sub-DFTs in increasing k2 order. *)
            for k2 = 0 to sp.ct_n2 - 1 do
              let group_nodes = List.rev groups.(k2) in
              let group_sinks =
                List.filter (fun e -> Hashtbl.mem assign_tags e.tag) group_nodes
              in
              let scheduled =
                if group_nodes = [] then []
                else if group_sinks = [] then group_nodes
                else
                  match bb_budget with
                  | None ->
                      Schedule.su_schedule_subset uarch ~gh ~subset:group_nodes
                        ~sinks:group_sinks
                  | Some t ->
                      Bb.bb_schedule_subset uarch ~time_budget_sec:t
                        ~subset:group_nodes ~sinks:group_sinks
              in
              result := scheduled :: !result
            done;
            List.concat (List.rev !result)
        | SU uarch when pass2_nodes <> [] -> (
            let assign_tags =
              List.fold_left
                (fun acc (_, e) ->
                  Hashtbl.replace acc e.tag ();
                  acc)
                (Hashtbl.create 32) assigns
            in
            let pass2_sinks =
              List.filter (fun e -> Hashtbl.mem assign_tags e.tag) pass2_nodes
            in
            if pass2_sinks = [] then pass2_nodes
            else
              match bb_budget with
              | None ->
                  Schedule.su_schedule_subset uarch ~gh ~subset:pass2_nodes
                    ~sinks:pass2_sinks
              | Some t ->
                  Bb.bb_schedule_subset uarch ~time_budget_sec:t
                    ~subset:pass2_nodes ~sinks:pass2_sinks)
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
          (match (!prev_c, cur_c) with
          | Some pc, Some cc when pc <> cc ->
              (* Transition: previous cluster pc flushes at position i. *)
              Hashtbl.replace flush_pos_for_cluster pc i
          | _ -> ());
          match cur_c with Some _ -> prev_c := cur_c | None -> ())
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
      List.iteri
        (fun i (e : t) -> Hashtbl.replace pass2_pos_of_tag e.tag i)
        pass2_ordered;
      List.iter
        (fun (_, e) ->
          if !current_store_on_compute && Hashtbl.mem pass2_pos_of_tag e.tag
          then
            (* Store-on-compute: last use is the inline store at the def. *)
            Hashtbl.replace pass2_force_last_use e.tag
              (Hashtbl.find pass2_pos_of_tag e.tag)
          else
            match Hashtbl.find_opt cluster_of_pass2_node e.tag with
            | Some c -> (
                match Hashtbl.find_opt flush_pos_for_cluster c with
                | Some pos -> Hashtbl.replace pass2_force_last_use e.tag pos
                | None -> Hashtbl.replace pass2_force_last_use e.tag pass2_n)
            | None ->
                (* Unclustered: stored in the safety-net loop at end-of-pass *)
                Hashtbl.replace pass2_force_last_use e.tag pass2_n)
        pass2_assigns;
      install_alloc "spill_pass2" pass2_ordered (Some inline_set)
        (Some pass2_force_last_use);
      (* M5: NOW that pass 2's allocator has run, emit the regalloc_spill
       * array decl with the correct size. (Earlier I tried emitting this
       * at the top of the pass 2 block, but at that point current_regalloc
       * still held pass 1's allocation, leading to a too-small array.) *)
      (match !current_regalloc with
      | Some alloc when alloc.num_spill_slots > 0 ->
          Buffer.add_string buf
            (Printf.sprintf "            %s regalloc_spill[%d];\n" isa.vec_type
               alloc.num_spill_slots)
      | _ -> ());

      (* Track which spilled tags have been reloaded. Walk pass2_ordered
       * and for each node, emit any pending reloads of its predecessors
       * before emitting the node. *)
      let reloaded : (int, unit) Hashtbl.t = Hashtbl.create 32 in
      let emit_reload_if_needed (p : t) =
        if Hashtbl.mem reloaded p.tag then ()
        else begin
          let do_reload arr_name slot =
            Buffer.add_string buf
              (Printf.sprintf "            const %s t%d = %s(&%s[%d]);\n"
                 isa.vec_type p.tag isa.loadu_pd arr_name slot);
            Hashtbl.add reloaded p.tag ()
          in
          match Hashtbl.find_opt sp.re_slot p.tag with
          | Some slot when not (is_fused_slot sp slot) ->
              do_reload "spill_re" slot
          | _ -> (
              match Hashtbl.find_opt sp.im_slot p.tag with
              | Some slot when not (is_fused_slot sp slot) ->
                  do_reload "spill_im" slot
              | _ -> ())
        end
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
        if Hashtbl.mem inline_set e.tag then
          List.iter reload_through_inlines (preds e)
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
                try Hashtbl.find assigns_by_cluster k2 with Not_found -> []
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
      if !current_store_on_compute then
        List.iter
          (fun ((_, e) as a) ->
            let cur =
              try Hashtbl.find soc_assigns_by_tag e.tag with Not_found -> []
            in
            Hashtbl.replace soc_assigns_by_tag e.tag (a :: cur))
          pass2_assigns;
      let soc_stored : (int, unit) Hashtbl.t = Hashtbl.create 64 in
      let flush_cluster_stores k2 =
        match Hashtbl.find_opt assigns_by_cluster k2 with
        | Some clist ->
            List.iter
              (fun (lhs, e) ->
                if Hashtbl.mem soc_stored e.tag then ()
                  (* already stored inline *)
                else begin
                  emit_reload_if_needed e;
                  emit_store buf lhs e
                end)
              (List.rev clist)
        | None -> ()
      in
      List.iteri
        (fun pos e ->
          current_emit_position := pos;
          (* M5: emit spill stores BEFORE reload loads. See pass 1 for
           * the reasoning (same-position spill+reload sequencing). *)
          (match !current_regalloc with
          | Some alloc -> (
              match Hashtbl.find_opt alloc.spill_sites pos with
              | Some spills ->
                  List.iter
                    (fun (tag, slot) ->
                      Buffer.add_string buf
                        (Printf.sprintf
                           "            %s(&regalloc_spill[%d], t%d);\n"
                           isa.storeu_pd slot tag))
                    spills
              | None -> ())
          | None -> ());
          (* M5: emit reload declarations for this position. *)
          (match !current_regalloc with
          | Some alloc -> (
              match Hashtbl.find_opt alloc.reload_sites pos with
              | Some reloads ->
                  List.iter
                    (fun (r : Regalloc.reload_decl) ->
                      Buffer.add_string buf
                        (Printf.sprintf "        %s\n"
                           (Isa.pinned_reg_decl isa r.reload_name r.reload_reg
                              (Printf.sprintf "%s(&regalloc_spill[%d])"
                                 isa.loadu_pd r.reload_slot))))
                    reloads
              | None -> ())
          | None -> ());
          if is_inlined e then ()
          else begin
            (* Emit reloads of any spilled predecessors not yet reloaded.
             * Walk transitively through inlined preds since their bodies
             * inline into e's expression and may reference spilled tags. *)
            List.iter reload_through_inlines (preds e);
            Buffer.add_string buf
              (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided
                 ~inline_set:(Some inline_set) e);
            Buffer.add_char buf '\n'
          end;
          (* Store-on-compute: emit the store(s) for any PASS 2 output whose
           * value node is e, right after its def. The value is in t%d here
           * (e is a sink, never inlined), and force_last_use was set to this
           * position so the register frees immediately. Marked in soc_stored
           * so cluster flush / safety net skip it. *)
          (if !current_store_on_compute then
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
          (match (!last_pass2_cluster, cur_cluster) with
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
      current_emit_position := final_pos;
      (* M5: emit spill stores for any tags evicted AT the final
       * position (i.e., during Step 3's fixed-point or post-iter
       * cascade). These must precede the reload loads so the slot
       * is initialized first. *)
      (match !current_regalloc with
      | Some alloc -> (
          match Hashtbl.find_opt alloc.spill_sites final_pos with
          | Some spills ->
              List.iter
                (fun (tag, slot) ->
                  Buffer.add_string buf
                    (Printf.sprintf
                       "            %s(&regalloc_spill[%d], t%d);\n"
                       isa.storeu_pd slot tag))
                spills
          | None -> ())
      | None -> ());
      (match !current_regalloc with
      | Some alloc -> (
          match Hashtbl.find_opt alloc.reload_sites final_pos with
          | Some reloads ->
              List.iter
                (fun (r : Regalloc.reload_decl) ->
                  Buffer.add_string buf
                    (Printf.sprintf "        %s\n"
                       (Isa.pinned_reg_decl isa r.reload_name r.reload_reg
                          (Printf.sprintf "%s(&regalloc_spill[%d])" isa.loadu_pd
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
          if Hashtbl.mem soc_stored e.tag then () (* already stored inline *)
          else
            match Hashtbl.find_opt cluster_of_pass2_node e.tag with
            | Some _ -> () (* already emitted via cluster *)
            | None ->
                emit_reload_if_needed e;
                let lhs, e = a in
                emit_store buf lhs e)
        pass2_assigns;
      Buffer.add_string buf "        }\n";
      clear_alloc () (* M3a: reset allocation at end of spill flow *)
  | None -> (
      match scheduler with
      | Topological ->
          (* Existing path: emit all definitions in topo order, then stores.
           *
           * Stage 4: when VFFT_USE_REGALLOC is set, run M3a/M5/M6 on this
           * codelet. Topological has no inlining and no duplicate-entry
           * issue (nodes is topo-sorted with each tag exactly once), but
           * we still go through prepare_for_simple_codelet for uniformity
           * and to get force_last_use construction for free. *)
          let roots = List.map snd assigns in
          let nodes = topo_sort_reachable roots in
          record_peak_live "topological_s1" nodes;
          let input =
            Regalloc.prepare_for_simple_codelet ~raw_scheduled:nodes ~assigns ()
          in
          install_alloc_canonical "topo_n1" input;
          emit_regalloc_spill_decl buf;
          List.iteri
            (fun pos e ->
              current_emit_position := pos;
              emit_node_spill_sites buf pos;
              emit_node_reload_sites buf pos;
              Buffer.add_string buf
                (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided
                   e);
              Buffer.add_char buf '\n')
            input.scheduled;
          (* End-of-schedule spill/reload emission. force_last_use put
           * output tags at position n=List.length input.scheduled, so the
           * allocator may have spill/reload sites at that position.
           * Mirrors the cluster-spill recipe's pass1_n handling. *)
          let n = List.length input.scheduled in
          current_emit_position := n;
          emit_node_spill_sites buf n;
          emit_node_reload_sites buf n;
          Buffer.add_char buf '\n';
          List.iter (fun (lhs, e) -> emit_store buf lhs e) assigns;
          clear_alloc ()
      | Bisection ->
          (* Frigo's bisection schedule: each entry is either an intermediate
           * (None, e) → emit definition, or an output (Some oref, e) → emit store.
           *
           * One subtlety: leaves (NK_Const, NK_Load) appear in the schedule
           * but render as a "load" definition. This is fine — they're still
           * t<tag> = _mm512_loadu_pd(...) lines.
           *
           * Another subtlety: an intermediate node and its output sibling share
           * the same alg_node (same tag). The intermediate's definition emits
           * once; when we hit the output node afterwards, we just store the
           * existing t<tag>. We track which tags have been defined to avoid
           * duplicate definitions. *)
          let scheduled_raw = Schedule.bisection_schedule assigns in
          record_peak_live "bisection_s1" (List.map snd scheduled_raw);
          (* Stage 4: route through canonical prep. Bisection has the same
           * (oref_opt, e) duplicate-entry pattern as SU. *)
          let input =
            Regalloc.prepare_for_simple_codelet_from_oref
              ~raw_scheduled:scheduled_raw ~assigns ()
          in
          install_alloc_canonical "bisect_n1" input;
          emit_regalloc_spill_decl buf;
          let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
          List.iteri
            (fun pos (e : t) ->
              current_emit_position := pos;
              emit_node_spill_sites buf pos;
              emit_node_reload_sites buf pos;
              if not (Hashtbl.mem defined e.tag) then begin
                Hashtbl.add defined e.tag ();
                Buffer.add_string buf
                  (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n
                     ~strided e);
                Buffer.add_char buf '\n'
              end)
            input.scheduled;
          (* End-of-schedule spill/reload emission. *)
          let n = List.length input.scheduled in
          current_emit_position := n;
          emit_node_spill_sites buf n;
          emit_node_reload_sites buf n;
          Buffer.add_char buf '\n';
          (* Stores at end-of-scope, mirroring SU/Topological pattern. *)
          List.iter
            (fun (lhs, e) ->
              if not (Hashtbl.mem defined e.tag) then begin
                Hashtbl.add defined e.tag ();
                Buffer.add_string buf
                  (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n
                     ~strided e);
                Buffer.add_char buf '\n'
              end;
              emit_store buf lhs e)
            assigns;
          clear_alloc ()
      | Annotated_topological ->
          (* Topological order, but emitted with nested-block scopes via annotate.ml.
           * Same instructions, same order — just nested `{ ... }` to communicate
           * variable lifetimes to GCC. *)
          let roots = List.map snd assigns in
          let nodes = topo_sort_reachable roots in
          record_peak_live "topological_s2" nodes;
          (* Build the entry list: intermediates first (in topo order), then stores. *)
          let entries =
            List.map (fun e -> (None, e)) nodes
            @ List.map (fun (lhs, e) -> (Some lhs, e)) assigns
          in
          let render_intermediate e =
            render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e
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
      | Annotated_bisection ->
          (* Bisection schedule, emitted with nested-block scopes via annotate.ml. *)
          let scheduled = Schedule.bisection_schedule assigns in
          record_peak_live "bisection_s2" (List.map snd scheduled);
          (* Bisection's output may have an alg_node appearing twice (once as
           * intermediate, once for store). Dedupe so annotate sees a clean list. *)
          let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
          let entries =
            List.filter_map
              (fun (oref_opt, e) ->
                match oref_opt with
                | None ->
                    if Hashtbl.mem defined e.tag then None
                    else begin
                      Hashtbl.add defined e.tag ();
                      Some (None, e)
                    end
                | Some oref ->
                    if not (Hashtbl.mem defined e.tag) then begin
                      (* Need to emit the definition before the store *)
                      Hashtbl.add defined e.tag ();
                      (* Note: this case shouldn't happen in well-formed bisection output. *)
                      Some (Some oref, e)
                    end
                    else Some (Some oref, e))
              scheduled
          in
          let render_intermediate e =
            render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e
          in
          let render_store oref e =
            let buf2 = Buffer.create 128 in
            emit_store buf2 oref e;
            String.trim (Buffer.contents buf2)
          in
          let scope = Annotate.annotate entries in
          Annotate.emit_scope isa buf render_intermediate render_store scope
      | SU uarch ->
          (* SU list scheduler: priority = (cp_dist DESC, su_num ASC).
           * Output shape mirrors Bisection: list of (oref_opt, alg_node)
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
          let inline_set = compute_inline_set assigns in
          let input =
            Regalloc.prepare_for_simple_codelet_from_oref
              ~raw_scheduled:scheduled_raw ~assigns
              ~inline_set:(Some inline_set) ()
          in
          install_alloc_canonical "su_n1" input;
          emit_regalloc_spill_decl buf;
          let defined : (int, unit) Hashtbl.t = Hashtbl.create 256 in
          let is_inlined e = Hashtbl.mem inline_set e.tag in
          List.iteri
            (fun pos (e : t) ->
              current_emit_position := pos;
              emit_node_spill_sites buf pos;
              emit_node_reload_sites buf pos;
              (* Skip emission for inlined values — their consumer will inline. *)
              if (not (is_inlined e)) && not (Hashtbl.mem defined e.tag) then begin
                Hashtbl.add defined e.tag ();
                Buffer.add_string buf
                  (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n
                     ~strided ~inline_set:(Some inline_set) e);
                Buffer.add_char buf '\n'
              end)
            input.scheduled;
          (* End-of-schedule spill/reload emission. *)
          let n = List.length input.scheduled in
          current_emit_position := n;
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
              if (not (Hashtbl.mem defined e.tag)) && not (is_inlined e) then begin
                Hashtbl.add defined e.tag ();
                Buffer.add_string buf
                  (render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n
                     ~strided ~inline_set:(Some inline_set) e);
                Buffer.add_char buf '\n'
              end;
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
                    if Hashtbl.mem defined e.tag then None
                    else begin
                      Hashtbl.add defined e.tag ();
                      Some (None, e)
                    end
                | Some oref -> Some (Some oref, e))
              scheduled
          in
          let render_intermediate e =
            render_node_def ~isa ~in_place ~t1s ~twidsq ~twidsq_n ~strided e
          in
          let render_store oref e =
            let buf2 = Buffer.create 128 in
            emit_store buf2 oref e;
            String.trim (Buffer.contents buf2)
          in
          let scope = Annotate.annotate entries in
          Annotate.emit_scope isa buf render_intermediate render_store scope))
  in
  emit_body isa ();

  (* Strided postamble: inverse 4×4 transpose + scatter back to matrix.
   * The body has populated out_lane_re_0..radix-1 / out_lane_im_0..radix-1
   * as plain assignments. Inverse-transpose them in groups of 4 and store
   * back to matrix at row_stride. *)
  if strided && isa.vec_width = 4 then begin
    Buffer.add_string buf "\n";
    let groups = radix / 4 in
    for which_side = 0 to 1 do
      let suf = if which_side = 0 then "re" else "im" in
      for g = 0 to groups - 1 do
        let j0 = g * 4 in
        Buffer.add_string buf
          (Printf.sprintf
             "        {  /* inverse 4x4 transpose group: fft_idx %d..%d, %s */\n"
             j0 (j0 + 3) suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m256d _u0_%s = \
              _mm256_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf j0 suf (j0 + 1));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m256d _u1_%s = \
              _mm256_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf j0 suf (j0 + 1));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m256d _u2_%s = \
              _mm256_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 2) suf (j0 + 3));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m256d _u3_%s = \
              _mm256_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 2) suf (j0 + 3));
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm256_storeu_pd(&rio_%s[(b+0)*row_stride + %d], \
              _mm256_permute2f128_pd(_u0_%s, _u2_%s, 0x20));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm256_storeu_pd(&rio_%s[(b+1)*row_stride + %d], \
              _mm256_permute2f128_pd(_u1_%s, _u3_%s, 0x20));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm256_storeu_pd(&rio_%s[(b+2)*row_stride + %d], \
              _mm256_permute2f128_pd(_u0_%s, _u2_%s, 0x31));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm256_storeu_pd(&rio_%s[(b+3)*row_stride + %d], \
              _mm256_permute2f128_pd(_u1_%s, _u3_%s, 0x31));\n"
             suf j0 suf suf);
        Buffer.add_string buf "        }\n"
      done
    done
  end;

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
  if strided && isa.vec_width = 8 then begin
    Buffer.add_string buf "\n";
    let groups = radix / 8 in
    for which_side = 0 to 1 do
      let suf = if which_side = 0 then "re" else "im" in
      for g = 0 to groups - 1 do
        let j0 = g * 8 in
        Buffer.add_string buf
          (Printf.sprintf
             "        {  /* inverse 8x8 transpose group: fft_idx %d..%d, %s */\n"
             j0 (j0 + 7) suf);
        (* Stage 1: 8 unpacklo/unpackhi_pd on out_lane pairs. *)
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u0_%s = \
              _mm512_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf j0 suf (j0 + 1));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u1_%s = \
              _mm512_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf j0 suf (j0 + 1));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u2_%s = \
              _mm512_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 2) suf (j0 + 3));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u3_%s = \
              _mm512_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 2) suf (j0 + 3));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u4_%s = \
              _mm512_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 4) suf (j0 + 5));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u5_%s = \
              _mm512_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 4) suf (j0 + 5));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u6_%s = \
              _mm512_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 6) suf (j0 + 7));
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _u7_%s = \
              _mm512_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d);\n"
             suf suf (j0 + 6) suf (j0 + 7));
        (* Stage 2: 8 permutex2var_pd. *)
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v0_%s = \
              _mm512_permutex2var_pd(_u0_%s, _tp_idx_lo, _u2_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v1_%s = \
              _mm512_permutex2var_pd(_u1_%s, _tp_idx_lo, _u3_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v2_%s = \
              _mm512_permutex2var_pd(_u0_%s, _tp_idx_hi, _u2_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v3_%s = \
              _mm512_permutex2var_pd(_u1_%s, _tp_idx_hi, _u3_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v4_%s = \
              _mm512_permutex2var_pd(_u4_%s, _tp_idx_lo, _u6_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v5_%s = \
              _mm512_permutex2var_pd(_u5_%s, _tp_idx_lo, _u7_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v6_%s = \
              _mm512_permutex2var_pd(_u4_%s, _tp_idx_hi, _u6_%s);\n"
             suf suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            const __m512d _v7_%s = \
              _mm512_permutex2var_pd(_u5_%s, _tp_idx_hi, _u7_%s);\n"
             suf suf suf);
        (* Stage 3 + store: 8 storeu_pd, each fused with a shuffle_f64x2. *)
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+0)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v0_%s, _v4_%s, 0x44));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+1)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v1_%s, _v5_%s, 0x44));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+2)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v2_%s, _v6_%s, 0x44));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+3)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v3_%s, _v7_%s, 0x44));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+4)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v0_%s, _v4_%s, 0xEE));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+5)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v1_%s, _v5_%s, 0xEE));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+6)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v2_%s, _v6_%s, 0xEE));\n"
             suf j0 suf suf);
        Buffer.add_string buf
          (Printf.sprintf
             "            _mm512_storeu_pd(&rio_%s[(b+7)*row_stride + %d], \
              _mm512_shuffle_f64x2(_v3_%s, _v7_%s, 0xEE));\n"
             suf j0 suf suf);
        Buffer.add_string buf "        }\n"
      done
    done
  end;

  Buffer.add_string buf "    }\n";
  if anyk_tail then begin
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
     * the masked pass touches only rio lanes [k, k+rem) = [me-rem, me), disjoint
     * from the bulk; masked-off lanes never touch memory. *)
    Buffer.add_string buf "    if (k < me) {\n";
    Buffer.add_string buf "        const size_t rem = me - k;\n";
    Buffer.add_string buf "        if (rem == 1) {\n";
    current_ls_mode := Isa.LS_vector;
    emit_body ~force_mono:true Isa.scalar ();
    Buffer.add_string buf "        } else {\n";
    (if isa.vec_width = 8 then
       Buffer.add_string buf
         "            const __mmask8 _m = (__mmask8)((1u << rem) - 1u);\n"
     else
       Buffer.add_string buf
         "            const __m256i _m = _mm256_loadu_si256((const __m256i \
          *)_vfft_masklo[rem]);\n");
    current_ls_mode := Isa.LS_masked "_m";
    emit_body isa ();
    current_ls_mode := Isa.LS_vector;
    Buffer.add_string buf "        }\n";
    Buffer.add_string buf "    }\n"
  end;
  if !hc_ranged then begin
    let r = !hc_ranged_r in
    if !hc2c_natural_bwd then begin
      (* mirror of the forward natural advance: the 4 SPLIT inputs walk like the
       * forward's split outputs (direct +, mirror -), the 2 PACKED outputs walk
       * like the forward's packed inputs (re +, im -). *)
      Buffer.add_string buf "    Rp += cs_in; Ip += cs_in;\n";
      Buffer.add_string buf "    Rm -= cs_in; Im -= cs_in;\n";
      Buffer.add_string buf "    out_re += cs_out; out_im -= cs_out;\n";
      Buffer.add_string buf
        (Printf.sprintf "    tw_re += %d; tw_im += %d;\n" r r)
    end
    else if !hc2c_natural then begin
      Buffer.add_string buf "    in_re += cs_in; in_im -= cs_in;\n";
      Buffer.add_string buf "    Rp += cs_out; Ip += cs_out;\n";
      Buffer.add_string buf "    Rm -= cs_out; Im -= cs_out;\n";
      Buffer.add_string buf
        (Printf.sprintf "    tw_re += %d; tw_im += %d;\n" r r)
    end
    else begin
      Buffer.add_string buf "    in_re += cs_in; in_im -= cs_in;\n";
      Buffer.add_string buf "    out_re += cs_out; out_im -= cs_out;\n";
      Buffer.add_string buf
        (Printf.sprintf "    tw_re += %d; tw_im += %d;\n" r r)
    end;
    Buffer.add_string buf "    }\n"
  end;
  Buffer.add_string buf "}\n";
  Buffer.add_string buf
    (codelet_metadata ~isa ~spill ~tw_broadcast:(t1s || twidsq)
       ~peak_live:!max_pass_peak assigns);
  Buffer.contents buf

(* ── M2 OOP UnitLeg edge helpers ──────────────────────────────────────
   codelet_oop.ml's UnitLeg load/store edges delegate here. The UnitLeg
   pattern reuses the AOS<->SOA transpose preamble that currently lives
   inlined inside emit_codelet (hardcoded rio_ prefix). Factoring that
   out parameterized over buffer/stride names is the unfinished M2
   phase-2 extraction. The UnitGroup edges (K-batched, the executor's
   registry ABI) are fully implemented inline in codelet_oop.ml and do
   NOT call these, so generating UnitGroup OOP codelets works today.
   These guard the UnitLeg (Bailey 2D strided) path until extracted. *)
let emit_strided_load_preamble ~isa ~radix ~in_re_name ~in_im_name
    ~group_stride_name (_buf : Buffer.t) : unit =
  ignore isa;
  ignore radix;
  ignore in_re_name;
  ignore in_im_name;
  ignore group_stride_name;
  failwith
    "emit_strided_load_preamble: UnitLeg OOP path not yet wired (M2 phase-2 \
     transpose-helper extraction); use --oop-load UG"

let emit_strided_store_postamble ~isa ~radix ~out_re_name ~out_im_name
    ~group_stride_name (_buf : Buffer.t) : unit =
  ignore isa;
  ignore radix;
  ignore out_re_name;
  ignore out_im_name;
  ignore group_stride_name;
  failwith
    "emit_strided_store_postamble: UnitLeg OOP path not yet wired (M2 phase-2 \
     transpose-helper extraction); use --oop-store UG"

(* No-op for UnitGroup codelets; only reached on the UnitLeg path, which
   guards downstream via emit_strided_*_preamble. Real AVX-512 transpose
   index emission belongs with the M2 phase-2 UnitLeg extraction. *)
let emit_avx512_transpose_indices (_isa : Isa.t) (_buf : Buffer.t) : unit = ()
