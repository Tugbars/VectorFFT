(* codelet_oop.ml — M2 codelet family: out-of-place with heterogeneous stride patterns
 *
 * SCOPE
 * ─────
 * This module emits a new codelet family ("oop") that generalizes the existing
 * --strided path with:
 *   1. Optional twiddles (t1 variant — per-butterfly-group twiddle factors)
 *   2. Separately configurable load and store stride patterns
 *   3. Optional two-buffer (genuinely OOP) signature for Stockham first stages
 *
 * The result is one codelet family that subsumes:
 *   • Bailey column FFT in-place (UnitGroup load + UnitGroup store)
 *   • Bailey row FFT with fused output transpose (UnitLeg load + UnitGroup store)
 *   • Existing 2D row FFT (UnitLeg load + UnitLeg store, single buffer)
 *   • Stockham first stage (UnitLeg load + UnitLeg store, two buffers, no twiddles)
 *   • Stockham middle stage (UnitLeg load + UnitLeg store, two buffers, with twiddles)
 *
 * STRIDE PATTERN VOCABULARY
 * ─────────────────────────
 * An "edge" is the load or store side of the codelet. Each edge has two
 * stride dimensions:
 *
 *   leg_stride   — distance between butterfly legs within one transform
 *                  (e.g., for a radix-R butterfly, leg j is at offset j*leg_stride)
 *
 *   group_stride — distance between butterfly groups (= between consecutive
 *                  independent transforms in the batch the codelet processes)
 *
 * The codelet processes `me` groups in chunks of `vec_width` per iteration.
 *
 * EDGE PATTERN TAXONOMY
 * ─────────────────────
 *   UnitLeg     — leg_stride = 1 (consecutive doubles per butterfly leg).
 *                 The vec_width legs of one group fit in one SIMD register
 *                 after the AOS→SOA transpose preamble. Existing --strided
 *                 path matches this pattern.
 *
 *   UnitGroup   — group_stride = 1 (consecutive doubles per group).
 *                 The vec_width groups for one leg fit in one SIMD register
 *                 directly without transposing. R separate SIMD loads (at
 *                 stride leg_stride) populate the R lane registers.
 *
 *   StridedFallback — neither stride is 1. Scalar load + insert sequence.
 *                 Slowest path. Not emitted in M2 first cut (deferred until
 *                 a use case demands it; Bailey, 2D, Stockham all have at
 *                 least one Unit edge).
 *
 * The combination of load + store patterns determines the codelet variant.
 * For M2 we emit (UnitLeg, UnitLeg), (UnitLeg, UnitGroup), (UnitGroup, UnitLeg),
 * (UnitGroup, UnitGroup) — four variants per (radix, ISA, direction, twiddle).
 *
 * BUFFER PATTERN
 * ──────────────
 * Independent of the stride pattern, the codelet can be:
 *
 *   InPlace  — single buffer (rio_re, rio_im). Reads and writes overlap in
 *              memory. Safe iff the (load_pat, store_pat) combination does
 *              not alias in a way that corrupts unread input. For M2's
 *              Bailey use, this is verified per stage.
 *
 *   OutOfPlace — separate buffers (in_re, in_im) and (out_re, out_im).
 *                Always safe regardless of stride patterns. Required for
 *                Stockham first stage (writes natural-order from bit-reversed
 *                input) and for Bailey row stage with output going to
 *                a different scratch buffer.
 *
 * INVARIANTS AND CONSTRAINTS
 * ──────────────────────────
 *   • radix must be divisible by isa.vec_width (= 4 for AVX2, 8 for AVX-512)
 *     when load_pat = UnitLeg, because the AOS→SOA transpose preamble
 *     processes vec_width butterfly legs at a time.
 *   • Twiddles, when present, are stored per-group: tw_re[(j-1)*me + b]
 *     for leg j ∈ [1, R), group b ∈ [0, me). The j=0 leg has no twiddle.
 *     This matches FFTW's t1*v convention.
 *
 * NON-GOALS FOR M2
 * ────────────────
 *   • Real-to-complex (R2C) — separate codelet family, deferred to M3.
 *   • StridedFallback edge — neither stride is 1. Deferred.
 *   • Mixed-radix-aware twiddle layouts (FFTW's t2*v family) — defer.
 *   • Specialization for known-constant strides (FFTW's plan-time codelets)
 *     — defer; would 2x the codelet count for marginal gain.
 *
 * BUTTERFLY BODY INTEGRATION
 * ──────────────────────────
 * prepare_butterfly ~sc runs the shared Pipeline.prepare_codelet cascade
 * (hash-cons, pass stack, spill markers), and the body emitters below
 * drive Emit_c's render / spill / schedule helpers directly — one
 * codelet family, self-contained from config to C text.
 * ------------------------------------------------------------------
 * MODULE CARD (codelet_oop.ml — grep "MODULE CARD" for the full set)
 * ROLE: The OOP codelet family: config types -> validate -> edge
 * load/store primitives -> prepare_butterfly ~sc -> monolithic / spill /
 * butterfly bodies -> emit_codelet driver -> canonical_name. Long but
 * single-concern; deliberately NOT split (owner-ratified).
 * PIPELINE: gen_main --oop path -> Codelet_oop.emit_codelet -> C
 * PUBLIC SURFACE (measured): gen_main(17): emit_codelet,
 * canonical_name, the edge/twiddle constructors, the current_oop_*
 * mode refs.
 * DEPS: Emit_c(46), Algsimp(42), Isa(26), Dft(19), Expr(8),
 * Schedule(2), Uarch(2), Pipeline(2).
 * ENV: VFFT_FORCE_FMA_LIFT, VFFT_DISABLE_FMA_LIFT, VFFT_NO_REGALLOC,
 * VFFT_PIN_FORCE, VFFT_FORCE_FENCE, VFFT_NO_ANYK_TAIL.
 * GOTCHA: mirrors gen_main's recipe decisions (should_spill /
 * should_block_n1) for the oop shapes; a recipe change in gen_main
 * needs a matching look here.
 * ------------------------------------------------------------------
 *)

(* ═══════════════════════════════════════════════════════════════
 * IR TYPES
 * ═══════════════════════════════════════════════════════════════ *)

(** Pattern of an edge of the codelet (load side or store side). *)
type edge_pattern =
  | UnitLeg
  (** leg_stride = 1: vec_width legs per SIMD register after AOS→SOA
          transpose preamble. Reuses the 4×4/8×8 transpose machinery from the
          existing --strided path. *)
  | UnitGroup
  (** group_stride = 1: vec_width groups per SIMD register loaded directly
          (no transpose needed). R lanes populated by R separate strided SIMD
          loads. *)
  | StridedFallback
  (** Both strides non-unit. Scalar-load+insert sequence. Not emitted in M2
          first cut. *)

(** Buffer layout of the codelet. *)
type buffer_layout =
  | InPlace (** Single (rio_re, rio_im) buffer pair. *)
  | OutOfPlace (** Separate (in_re, in_im) and (out_re, out_im) pairs. *)

(** Twiddle presence. n1 = no twiddles. t1 = per-group vector twiddles, one
    value per (leg, batch): tw_re[(j-1)*me + b]. t1s = scalar-broadcast
    twiddles, one value per leg: tw_re[j-1], broadcast across the K batches. For
    Stockham/CT inner stages the twiddle is constant across the batch dim, so
    t1s stores (R-1) scalars instead of (R-1)*me and loads them with a single
    broadcast — killing the per-batch twiddle bandwidth. *)
type twiddle_kind =
  | NoTwiddles
  | PerGroupTwiddles
  | BroadcastTwiddles
  | PerPositionTwiddles
(* t1p: per-position twiddle, broadcast across batch lanes *)

(** Direction of the transform. *)
type direction =
  | Forward
  | Backward

(** Full configuration of one codelet variant. *)
type config =
  { radix : int
  ; isa : Isa.t
  ; direction : direction
  ; load_pat : edge_pattern
  ; store_pat : edge_pattern
  ; buffer : buffer_layout
  ; twiddles : twiddle_kind
  ; name : string
    (** Symbol name as emitted in the .c file. Caller-supplied to allow
          consistent naming with existing convention (radix_R_t1_oop_fwd_avx512
          etc.). *)
  }

(* ═══════════════════════════════════════════════════════════════
 * VALIDATION
 *
 * Enforce the structural constraints that make the rest of emission
 * sound. Errors here indicate a planner bug or an unsupported variant.
 * ═══════════════════════════════════════════════════════════════ *)

(** Raise [Failure] with a clear message if the config is malformed or
    unsupported by M2 first cut. *)
let validate (c : config) : unit =
  if c.radix <= 0
  then failwith (Printf.sprintf "codelet_oop: radix must be > 0 (got %d)" c.radix);
  (* UnitLeg requires the AOS→SOA transpose preamble to process
     vec_width legs per iteration, which requires radix divisible
     by vec_width. *)
  if c.load_pat = UnitLeg && c.radix mod c.isa.vec_width <> 0
  then
    failwith
      (Printf.sprintf
         "codelet_oop: UnitLeg load requires radix %% vec_width = 0 (got radix=%d, \
          vec_width=%d)"
         c.radix
         c.isa.vec_width);
  if c.store_pat = UnitLeg && c.radix mod c.isa.vec_width <> 0
  then
    failwith
      (Printf.sprintf
         "codelet_oop: UnitLeg store requires radix %% vec_width = 0 (got radix=%d, \
          vec_width=%d)"
         c.radix
         c.isa.vec_width);
  (* M2 first cut: defer StridedFallback. *)
  if c.load_pat = StridedFallback || c.store_pat = StridedFallback
  then failwith "codelet_oop: StridedFallback edge not yet supported in M2"
;;

(* ═══════════════════════════════════════════════════════════════
 * SIGNATURE EMISSION
 *
 * Emits the C function signature. Six possible signatures based on
 * (buffer, twiddles): each twiddle setting × each buffer layout.
 *
 * Stride parameter convention:
 *   in_leg_stride / in_group_stride — for the load side
 *   out_leg_stride / out_group_stride — for the store side
 *
 * When buffer = InPlace, the same buffer is read and written, but the
 * codelet still takes both stride pairs as parameters — they describe
 * the read pattern vs the write pattern, which can differ (this is
 * what fuses Bailey's output transpose into the codelet).
 * ═══════════════════════════════════════════════════════════════ *)

(* Stride specialization: when Some (in_leg, in_group, out_leg, out_group),
   those four strides are baked as compile-time constants in the body instead
   of taken as runtime size_t parameters. Folds the leg*stride address
   arithmetic to constant displacements and drops the four argument registers.
   me stays a parameter. Set per-codelet by the caller (gen_radix --oop-strides). *)

(** §6a53 / Gap-A: POST-twiddle mode. The body expands as a PURE DFT
    (NoTwiddles math) under the t1 ABI, and a cmul postamble multiplies
    output legs 1..R-1 by W[(j-1)*me + m] just before the UnitGroup store —
    out = tw (.) DFT(in), the OOP twin of radix{R}_t1_dif_fwd. Leg 0 stays
    untwiddled. Loads are ls-mode-aware, so the anyk tail masks them like
    every other group-indexed access. Set from gen_main --post-tw. *)
let current_post_tw = ref false

let current_oop_strides : (int * int * int * int) option ref = ref None

(* M-project fuse count for the OOP path: how many trailing PASS-2 sub-DFTs
   are kept register-resident across the PASS 1/PASS 2 boundary (capped at n2
   by make_spill_info). 0 = none (every pass-boundary value rounds through the
   spill arrays). Set per-codelet by the caller (gen_radix --fuse). *)
let current_oop_fuse : int ref = ref 0

(* Store-on-compute for the UnitGroup store path: when true, each FFT output is
   stored to memory the moment it is computed (out_re[b*out_grp + j*out_leg]),
   instead of being accumulated into an out_lane_* register and written by a
   separate store phase at the end. Eliminates the 2R out_lane accumulators,
   freeing registers and reducing spills. Safe because the load phase pulls all
   inputs into registers before the body runs, and (in the 2-pass codelets)
   PASS 2 reads only the spill arrays, never the output buffer. UnitLeg store is
   unaffected (its transpose operates on out_lane). Set by gen_radix
   --oop-store-fused. *)
let current_oop_store_on_compute : bool ref = ref false

(* §P2 (docs/roadmap/row_major_engine.md §11e): interleaved-boundary edges for
   the OOP family. il_in: the load edge reads an INTERLEAVED z buffer
   (in_z[2*(b*Gs + j*Ls)], pair-factor 2 hardwired) and deinterleaves
   in-register into the split lane registers. il_out: the store edge
   interleaves the out_lane pair in-register and writes INTERLEAVED out_z.
   UnitGroup-only (the vector body spans vec_width CONSECUTIVE groups — the UG
   group_stride=1 contract, same as the split UG edges). Twiddle loads are
   Expr.Twiddle nodes, structurally untouched. Because the SSE2/scalar tail
   passes re-run emit_load_edge/emit_store_edge ~sc at narrower widths, the tails
   are IL-correct for ANY me — the reason these are EMITTED, replacing the
   deprecated il_derive.py derived twins (whose tails read split memory).
   avx512 (masked-tail lattice) not yet emitted — gen_main gates it.
   Set from gen_main --oop-il-in / --oop-il-out. *)
let current_oop_il_in : bool ref = ref false
let current_oop_il_out : bool ref = ref false

(* _sw twins: same lattice with the (re,im) pairing SWAPPED — il_in_sw reads z
   pairs as (im,re), il_out_sw writes (im,re). These are the BACKWARD-direction
   enablers: the unnormalized-inverse swap identity IDFT = swap(DFT(swap(.)))
   swaps re/im POINTERS on split buffers, which an interleaved z buffer cannot
   do — so the swap folds into the boundary lattice instead. The butterfly DAG
   stays the FORWARD one (fwd-named symbols, per the established il_out_sw
   convention). *)
let current_oop_il_in_sw : bool ref = ref false
let current_oop_il_out_sw : bool ref = ref false

(* LINEAR twiddle layout (§12.4 4a): pack the t1 table in consumption order
   (per group-quad, all legs contiguous) — one streaming cursor like MKL's,
   instead of (R-1) parallel strided rows. UL-load configs only (no rem tail
   exists there; the tail passes would index the flat layout). Set from
   gen_main --oop-tw-linear; forwarded to Emit_state via Emit_c at emit. *)
let current_oop_tw_linear : bool ref = ref false
let il_in_active () = !current_oop_il_in || !current_oop_il_in_sw
let il_out_active () = !current_oop_il_out || !current_oop_il_out_sw

(** Emit the function signature into the buffer. Trailing newline before the
    opening brace of the function body. *)
let emit_signature (buf : Buffer.t) (c : config) : unit =
  Buffer.add_string
    buf
    (Printf.sprintf "__attribute__((target(\"%s\")))\n" c.isa.target_attr);
  Buffer.add_string buf (Printf.sprintf "void %s(\n" c.name);
  (* Buffer pointers. IL edges swap the split pair for (z, unused) on their
     side — argument-for-argument the same 11-arg shape, so the planner's
     vfft_oop11_fn call sites stay uniform (caller passes NULL for unused). *)
  (* M3: data-plane parameters come from Layout — the ONE constructor of
     pointer params; the anti-hybrid law and the sw-pair exclusivity are
     enforced inside it (previously two independent ifs over two independent
     global pairs, §12.1/P6). Bytes unchanged (corpus gate is the proof). *)
  (let emit_side ps = List.iter (fun p -> Buffer.add_string buf (Layout.render p)) ps in
   match c.buffer with
   | InPlace ->
     emit_side (Layout.pointers Layout.Split ~const:false ~prefix:"rio" ~twin:false ())
   | OutOfPlace ->
     (match
        Layout.buffers_of_oop_bools
          ~il_in:!current_oop_il_in
          ~il_in_sw:!current_oop_il_in_sw
          ~il_out:!current_oop_il_out
          ~il_out_sw:!current_oop_il_out_sw
      with
      | Layout.Oop { load; store } ->
        let com pl = if pl = Layout.Split then None else Some "/* interleaved pairs */" in
        emit_side
          (Layout.pointers load ~const:true ~prefix:"in" ~twin:(load <> Layout.Split)
             ?comment:(com load) ());
        emit_side
          (Layout.pointers store ~const:false ~prefix:"out" ~twin:(store <> Layout.Split)
             ?comment:(com store) ())
      | _ -> assert false));
  (match c.twiddles with
   | NoTwiddles ->
     (* For signature uniformity with the t1 variant (and to make the
        planner's job easier — same call site shape), the n1 variant
        still takes tw_re/tw_im pointers. Caller passes NULL. The body
        marks them (void) to silence -Wunused-parameter. *)
     Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
     Buffer.add_string buf "    const double * __restrict__ tw_im,\n"
   | PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles ->
     Buffer.add_string buf "    const double * __restrict__ tw_re,\n";
     Buffer.add_string buf "    const double * __restrict__ tw_im,\n");
  (* Stride parameters. Always four — even when InPlace, the load and
     store edges may use different strides (this is what enables the
     fused transpose). When current_oop_strides is set, these become
     compile-time constants inside the body (see after the brace) and
     are dropped from the parameter list. *)
  (match !current_oop_strides with
   | None ->
     Buffer.add_string buf "    size_t in_leg_stride,\n";
     Buffer.add_string buf "    size_t in_group_stride,\n";
     Buffer.add_string buf "    size_t out_leg_stride,\n";
     Buffer.add_string buf "    size_t out_group_stride,\n"
   | Some _ -> ());
  (* Multiplicity: number of butterfly groups to process. *)
  Buffer.add_string buf "    size_t me)\n";
  Buffer.add_string buf "{\n";
  (match !current_oop_strides with
   | Some (l, g, ol, og) ->
     Buffer.add_string
       buf
       (Printf.sprintf
          "    /* stride-specialized: strides baked, folds to constant displacements */\n\
          \    const size_t in_leg_stride    = %d;\n\
          \    const size_t in_group_stride  = %d;\n\
          \    const size_t out_leg_stride   = %d;\n\
          \    const size_t out_group_stride = %d;\n"
          l
          g
          ol
          og)
   | None -> ());
  (* Unused-parameter silencing for n1 and the IL unused slots. *)
  if il_in_active () then Buffer.add_string buf "    (void)in_unused;\n";
  if il_out_active () then Buffer.add_string buf "    (void)out_unused;\n";
  match c.twiddles with
  | NoTwiddles -> Buffer.add_string buf "    (void)tw_re; (void)tw_im;\n"
  | PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles -> ()
;;

(* ═══════════════════════════════════════════════════════════════
 * LANE REGISTER DECLARATIONS
 *
 * Per-iteration locals: lane_re_j, lane_im_j for j ∈ [0, radix),
 * holding the SoA values after the load transpose (or directly from
 * UnitGroup loads). Plus out_lane_re_j, out_lane_im_j for the
 * outputs before the store transpose.
 *
 * Same convention as the existing --strided path so the butterfly
 * body emission can reuse lane name lookups unchanged.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_lane_decls (buf : Buffer.t) (c : config) : unit =
  let need_out_lane = not (!current_oop_store_on_compute && c.store_pat = UnitGroup) in
  for j = 0 to c.radix - 1 do
    Buffer.add_string
      buf
      (Printf.sprintf "        %s lane_re_%d, lane_im_%d;\n" c.isa.vec_type j j);
    if need_out_lane
    then
      Buffer.add_string
        buf
        (Printf.sprintf "        %s out_lane_re_%d, out_lane_im_%d;\n" c.isa.vec_type j j)
  done;
  Buffer.add_string buf "\n"
;;

(* ═══════════════════════════════════════════════════════════════
 * LOOP STRUCTURE
 *
 * The outer loop iterates the group dimension `b` from 0 to me in
 * steps of vec_width. Per iteration: load → body → store.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_loop_open (buf : Buffer.t) (c : config) : unit =
  Buffer.add_string
    buf
    (Printf.sprintf "    for (size_t b = 0; b < me; b += %d) {\n" c.isa.vec_width)
;;

let emit_loop_close (buf : Buffer.t) : unit =
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n"
;;

(* ═══════════════════════════════════════════════════════════════
 * LOAD EDGE — UnitLeg pattern
 *
 * Existing AOS→SOA transpose preamble from emit_c.ml. Loads
 * vec_width legs as vec_width SIMD registers, transposes 4×4 (AVX2)
 * or 8×8 (AVX-512), assigns to lane_re_j / lane_im_j.
 *
 * For OOP, reads from in_re/in_im; for InPlace, from rio_re/rio_im.
 *
 * Stub: the actual transpose codegen lives in emit_c.ml and will
 * be wired in via a helper extraction during M2 phase 2.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_load_unitleg (buf : Buffer.t) (c : config) : unit =
  (* NATIVE UL load lattice (P2 two-pass restructure, row_major_engine.md §12.4
     item 1 route (a)): vec_width legs are CONTIGUOUS (leg_stride ~ 1), groups
     strided — load vw rows (one per group) per leg-quad and 4x4-transpose
     in-register so the lane axis = groups, as the body expects. This fuses
     the four-step's transpose into the t1's load edge: the t1 reads the
     column pass's UNtransposed output directly (Ls=1, Gs=R1), eliminating
     the separate transpose sweep + one scratch buffer + one L1 round-trip
     (MKL's two-pass shape, §12.1). Replaces the former Emit_c stub
     delegation (M2 phase-2) with a self-contained lattice, same approach as
     the IL edges. *)
  let base_re =
    match c.buffer with
    | InPlace -> "rio_re"
    | OutOfPlace -> "in_re"
  in
  let base_im =
    match c.buffer with
    | InPlace -> "rio_im"
    | OutOfPlace -> "in_im"
  in
  if il_in_active ()
  then failwith "codelet_oop: UnitLeg load does not compose with il_in yet";
  Buffer.add_string
    buf
    "        /* UnitLeg load: vw legs contiguous, vw groups strided; 4x4\n\
    \           in-register transpose puts groups on the lane axis. */\n";
  match c.isa.Isa.vec_width with
  | 4 ->
    for lq = 0 to (c.radix / 4) - 1 do
      let l0 = 4 * lq in
      List.iter
        (fun (comp, base) ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { const __m256d _ta = _mm256_loadu_pd(&%s[(b + 0) * \
                 in_group_stride + %d * in_leg_stride]);\n\
                \          const __m256d _tb = _mm256_loadu_pd(&%s[(b + 1) * \
                 in_group_stride + %d * in_leg_stride]);\n\
                \          const __m256d _tc = _mm256_loadu_pd(&%s[(b + 2) * \
                 in_group_stride + %d * in_leg_stride]);\n\
                \          const __m256d _td = _mm256_loadu_pd(&%s[(b + 3) * \
                 in_group_stride + %d * in_leg_stride]);\n\
                \          const __m256d _u0 = _mm256_unpacklo_pd(_ta, _tb);\n\
                \          const __m256d _u1 = _mm256_unpackhi_pd(_ta, _tb);\n\
                \          const __m256d _u2 = _mm256_unpacklo_pd(_tc, _td);\n\
                \          const __m256d _u3 = _mm256_unpackhi_pd(_tc, _td);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u0, _u2, 0x20);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u1, _u3, 0x20);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u0, _u2, 0x31);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u1, _u3, 0x31); }\n"
                base
                l0
                base
                l0
                base
                l0
                base
                l0
                comp
                l0
                comp
                (l0 + 1)
                comp
                (l0 + 2)
                comp
                (l0 + 3)))
        [ "re", base_re; "im", base_im ]
    done
  | 2 ->
    for lq = 0 to (c.radix / 2) - 1 do
      let l0 = 2 * lq in
      List.iter
        (fun (comp, base) ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { const __m128d _ta = _mm_loadu_pd(&%s[(b + 0) * \
                 in_group_stride + %d * in_leg_stride]);\n\
                \          const __m128d _tb = _mm_loadu_pd(&%s[(b + 1) * \
                 in_group_stride + %d * in_leg_stride]);\n\
                \          lane_%s_%d = _mm_unpacklo_pd(_ta, _tb);\n\
                \          lane_%s_%d = _mm_unpackhi_pd(_ta, _tb); }\n"
                base
                l0
                base
                l0
                comp
                l0
                comp
                (l0 + 1)))
        [ "re", base_re; "im", base_im ]
    done
  | 1 ->
    for l = 0 to c.radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        lane_re_%d = %s[b * in_group_stride + %d * in_leg_stride];\n\
           \        lane_im_%d = %s[b * in_group_stride + %d * in_leg_stride];\n"
           l
           base_re
           l
           l
           base_im
           l)
    done
  | w -> failwith (Printf.sprintf "codelet_oop UL load: vec_width %d not emitted" w)
;;

(* ═══════════════════════════════════════════════════════════════
 * LOAD EDGE — UnitGroup pattern
 *
 * NEW pattern not present in the existing --strided path.
 *
 * For each butterfly leg j ∈ [0, radix):
 *   lane_*_j = SIMD load of vec_width consecutive groups at
 *              base + b * in_group_stride + j * in_leg_stride
 *
 * Since group_stride = 1, the vec_width groups are CONSECUTIVE in
 * memory and load as one SIMD register directly — no transpose
 * needed. The R legs are at stride in_leg_stride from each other,
 * so R separate SIMD loads (one per leg) populate the R lane regs.
 *
 * This is faster than UnitLeg when the codelet is reading a
 * column of an N1×N2 row-major matrix (Bailey col FFT case):
 * in_leg_stride = N2, in_group_stride = 1.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_load_unitgroup ~(sc : Emit_render.Scratch.t) (buf : Buffer.t) (c : config) : unit =
  if il_in_active ()
  then (
    (* IL load: vec_width consecutive groups are interleaved (re,im) pairs in
       z. Two vector loads + unpack/permute deinterleave into the split lane
       registers — one load pair per leg (the deprecated derived twins loaded
       each pair twice). Width-parametric so the SSE2/scalar tail passes,
       which re-enter here with a narrower isa, stay IL-correct.
       _sw: the (im,re)-swapped read — unpackhi feeds lane_re (the bwd swap
       identity folded into the lattice). *)
    let sw = !current_oop_il_in_sw in
    (* which unpack feeds re / im *)
    let u_re = if sw then "hi" else "lo"
    and u_im = if sw then "lo" else "hi" in
    Buffer.add_string
      buf
      (if sw
       then
         "        /* UnitGroup IL_SW load: interleaved groups read as (im,re) — bwd \
          swap. */\n"
       else
         "        /* UnitGroup IL load: vec_width consecutive interleaved groups;\n\
         \           deinterleave in-register into the split lane registers. */\n");
    for j = 0 to c.radix - 1 do
      let addr = Printf.sprintf "2*(b * in_group_stride + %d * in_leg_stride)" j in
      match c.isa.Isa.vec_width with
      | 4 ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "        { const __m256d _ilza = _mm256_loadu_pd(&in_z[%s]);\n\
             \          const __m256d _ilzb = _mm256_loadu_pd(&in_z[%s + 4]);\n\
             \          lane_re_%d = _mm256_permute4x64_pd(_mm256_unpack%s_pd(_ilza, \
              _ilzb), 0xD8);\n\
             \          lane_im_%d = _mm256_permute4x64_pd(_mm256_unpack%s_pd(_ilza, \
              _ilzb), 0xD8); }\n"
             addr
             addr
             j
             u_re
             j
             u_im)
      | 2 ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "        { const __m128d _ilza = _mm_loadu_pd(&in_z[%s]);\n\
             \          const __m128d _ilzb = _mm_loadu_pd(&in_z[%s + 2]);\n\
             \          lane_re_%d = _mm_unpack%s_pd(_ilza, _ilzb);\n\
             \          lane_im_%d = _mm_unpack%s_pd(_ilza, _ilzb); }\n"
             addr
             addr
             j
             u_re
             j
             u_im)
      | 1 ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "        lane_re_%d = in_z[%s + %d];\n        lane_im_%d = in_z[%s + %d];\n"
             j
             addr
             (if sw then 1 else 0)
             j
             addr
             (if sw then 0 else 1))
      | w ->
        failwith
          (Printf.sprintf
             "codelet_oop il_in: vec_width %d not emitted (avx512 masked lattice pending)"
             w)
    done)
  else (
    let base_re =
      match c.buffer with
      | InPlace -> "rio_re"
      | OutOfPlace -> "in_re"
    in
    let base_im =
      match c.buffer with
      | InPlace -> "rio_im"
      | OutOfPlace -> "in_im"
    in
    Buffer.add_string
      buf
      "        /* UnitGroup load: vec_width groups are consecutive (stride 1)\n";
    Buffer.add_string
      buf
      "           so they load as one SIMD register per leg. R separate\n";
    Buffer.add_string
      buf
      "           strided loads populate the R lane registers — no transpose. */\n";
    for j = 0 to c.radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        lane_re_%d = %s;\n"
           j
           (Isa.loadu_pd
              ~mode:sc.Emit_render.Scratch.ls_mode
              c.isa
              (Printf.sprintf "%s[b * in_group_stride + %d * in_leg_stride]" base_re j)));
      Buffer.add_string
        buf
        (Printf.sprintf
           "        lane_im_%d = %s;\n"
           j
           (Isa.loadu_pd
              ~mode:sc.Emit_render.Scratch.ls_mode
              c.isa
              (Printf.sprintf "%s[b * in_group_stride + %d * in_leg_stride]" base_im j)))
    done)
;;

(* ═══════════════════════════════════════════════════════════════
 * LOAD EDGE — dispatch
 * ═══════════════════════════════════════════════════════════════ *)

let emit_load_edge ~(sc : Emit_render.Scratch.t) (buf : Buffer.t) (c : config) : unit =
  match c.load_pat with
  | UnitLeg -> emit_load_unitleg buf c
  | UnitGroup -> emit_load_unitgroup ~sc buf c
  | StridedFallback -> failwith "emit_load_edge: StridedFallback not yet supported"
;;

(* ═══════════════════════════════════════════════════════════════
 * STORE EDGE — UnitLeg pattern
 *
 * Inverse of the UnitLeg load: 4×4 / 8×8 SIMD transpose to put
 * vec_width groups' values into vec_width consecutive memory cells
 * per leg, then storeu_pd.
 *
 * Stub: same as load_unitleg — wired in during phase 2.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_store_unitleg (buf : Buffer.t) (c : config) : unit =
  (* NATIVE UL store lattice (P2 two-pass restructure, route (b)): inverse of
     the UL load — transpose vw out_lane registers (lane axis = groups) into
     vw rows (one per group, vw legs contiguous) and store with strided group
     addressing. Fuses the four-step's transpose into the LEAF's store edge
     (n1_oop UG_UL: out_leg_stride=1, out_group_stride=R2 writes the
     transposed intermediate directly). Replaces the former Emit_c stub. *)
  let base_re =
    match c.buffer with
    | InPlace -> "rio_re"
    | OutOfPlace -> "out_re"
  in
  let base_im =
    match c.buffer with
    | InPlace -> "rio_im"
    | OutOfPlace -> "out_im"
  in
  if il_out_active ()
  then failwith "codelet_oop: UnitLeg store does not compose with il_out yet";
  if !current_post_tw
  then failwith "codelet_oop: UnitLeg store does not compose with --post-tw yet";
  Buffer.add_string
    buf
    "        /* UnitLeg store: transpose vw out_lanes (lanes=groups) into vw\n\
    \           rows and store per group with contiguous legs. */\n";
  match c.isa.Isa.vec_width with
  | 4 ->
    for lq = 0 to (c.radix / 4) - 1 do
      let l0 = 4 * lq in
      List.iter
        (fun (comp, base) ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { const __m256d _u0 = _mm256_unpacklo_pd(out_lane_%s_%d, \
                 out_lane_%s_%d);\n\
                \          const __m256d _u1 = _mm256_unpackhi_pd(out_lane_%s_%d, \
                 out_lane_%s_%d);\n\
                \          const __m256d _u2 = _mm256_unpacklo_pd(out_lane_%s_%d, \
                 out_lane_%s_%d);\n\
                \          const __m256d _u3 = _mm256_unpackhi_pd(out_lane_%s_%d, \
                 out_lane_%s_%d);\n\
                \          _mm256_storeu_pd(&%s[(b + 0) * out_group_stride + %d * \
                 out_leg_stride], _mm256_permute2f128_pd(_u0, _u2, 0x20));\n\
                \          _mm256_storeu_pd(&%s[(b + 1) * out_group_stride + %d * \
                 out_leg_stride], _mm256_permute2f128_pd(_u1, _u3, 0x20));\n\
                \          _mm256_storeu_pd(&%s[(b + 2) * out_group_stride + %d * \
                 out_leg_stride], _mm256_permute2f128_pd(_u0, _u2, 0x31));\n\
                \          _mm256_storeu_pd(&%s[(b + 3) * out_group_stride + %d * \
                 out_leg_stride], _mm256_permute2f128_pd(_u1, _u3, 0x31)); }\n"
                comp
                l0
                comp
                (l0 + 1)
                comp
                l0
                comp
                (l0 + 1)
                comp
                (l0 + 2)
                comp
                (l0 + 3)
                comp
                (l0 + 2)
                comp
                (l0 + 3)
                base
                l0
                base
                l0
                base
                l0
                base
                l0))
        [ "re", base_re; "im", base_im ]
    done
  | 2 ->
    for lq = 0 to (c.radix / 2) - 1 do
      let l0 = 2 * lq in
      List.iter
        (fun (comp, base) ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { _mm_storeu_pd(&%s[(b + 0) * out_group_stride + %d * \
                 out_leg_stride], _mm_unpacklo_pd(out_lane_%s_%d, out_lane_%s_%d));\n\
                \          _mm_storeu_pd(&%s[(b + 1) * out_group_stride + %d * \
                 out_leg_stride], _mm_unpackhi_pd(out_lane_%s_%d, out_lane_%s_%d)); }\n"
                base
                l0
                comp
                l0
                comp
                (l0 + 1)
                base
                l0
                comp
                l0
                comp
                (l0 + 1)))
        [ "re", base_re; "im", base_im ]
    done
  | 1 ->
    for l = 0 to c.radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        %s[b * out_group_stride + %d * out_leg_stride] = out_lane_re_%d;\n\
           \        %s[b * out_group_stride + %d * out_leg_stride] = out_lane_im_%d;\n"
           base_re
           l
           l
           base_im
           l
           l)
    done
  | w -> failwith (Printf.sprintf "codelet_oop UL store: vec_width %d not emitted" w)
;;

(* ═══════════════════════════════════════════════════════════════
 * STORE EDGE — UnitGroup pattern
 *
 * For each butterfly leg j ∈ [0, radix):
 *   SIMD store out_lane_*_j to
 *     base + b * out_group_stride + j * out_leg_stride
 *
 * Mirror of UnitGroup load. Used for Bailey row stage with fused
 * output transpose: out_leg_stride = N1, out_group_stride = 1.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_store_unitgroup ~(sc : Emit_render.Scratch.t) (buf : Buffer.t) (c : config) : unit =
  if !current_post_tw
  then (
    let scalar = c.isa.Isa.vec_width = 1 in
    let pfx =
      match c.isa.Isa.vec_width with
      | 8 -> "_mm512"
      | 4 -> "_mm256"
      | 2 -> "_mm"
      | _ -> "_scalar_unused"
    in
    Buffer.add_string
      buf
      "        /* Gap-A post-twiddle: out_j = W[j-1] (.) DFT_j (leg 0 untwiddled). */\n";
    for j = 1 to c.radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        { const %s _ptr = %s;\n"
           c.isa.Isa.vec_type
           (Isa.loadu_pd
              ~mode:sc.Emit_render.Scratch.ls_mode
              c.isa
              (Printf.sprintf "tw_re[%d * me + b]" (j - 1))));
      Buffer.add_string
        buf
        (Printf.sprintf
           "          const %s _pti = %s;\n"
           c.isa.Isa.vec_type
           (Isa.loadu_pd
              ~mode:sc.Emit_render.Scratch.ls_mode
              c.isa
              (Printf.sprintf "tw_im[%d * me + b]" (j - 1))));
      Buffer.add_string
        buf
        (Printf.sprintf
           "          const %s _pvr = out_lane_re_%d;\n"
           c.isa.Isa.vec_type
           j);
      Buffer.add_string
        buf
        (Printf.sprintf
           "          const %s _pvi = out_lane_im_%d;\n"
           c.isa.Isa.vec_type
           j);
      if scalar
      then (
        Buffer.add_string
          buf
          (Printf.sprintf "          out_lane_re_%d = _ptr * _pvr - _pti * _pvi;\n" j);
        Buffer.add_string
          buf
          (Printf.sprintf
             "          out_lane_im_%d = _ptr * _pvi + _pti * _pvr;\n        }\n"
             j))
      else (
        Buffer.add_string
          buf
          (Printf.sprintf
             "          out_lane_re_%d = %s_fmsub_pd(_ptr, _pvr, %s_mul_pd(_pti, _pvi));\n"
             j
             pfx
             pfx);
        Buffer.add_string
          buf
          (Printf.sprintf
             "          out_lane_im_%d = %s_fmadd_pd(_ptr, _pvi, %s_mul_pd(_pti, _pvr));\n\
             \        }\n"
             j
             pfx
             pfx))
    done);
  if il_out_active ()
  then (
    (* IL store: interleave the split out_lane pair in-register and store
       vec_width consecutive groups as (re,im) pairs in out_z. Composes with
       the post-tw cmul above (which operates on out_lane registers). Width-
       parametric for the SSE2/scalar tail passes. _sw: (im,re)-swapped write
       (the bwd swap identity folded into the lattice). *)
    let sw = !current_oop_il_out_sw in
    Buffer.add_string
      buf
      (if sw
       then
         "        /* UnitGroup IL_SW store: interleave in-register as (im,re) — bwd \
          swap. */\n"
       else
         "        /* UnitGroup IL store: interleave in-register, store vec_width\n\
         \           consecutive groups as (re,im) pairs in out_z. */\n");
    for j = 0 to c.radix - 1 do
      let addr = Printf.sprintf "2*(b * out_group_stride + %d * out_leg_stride)" j in
      let re_n = Printf.sprintf "out_lane_re_%d" j
      and im_n = Printf.sprintf "out_lane_im_%d" j in
      let a = if sw then im_n else re_n
      and bq = if sw then re_n else im_n in
      match c.isa.Isa.vec_width with
      | 4 ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "        { const __m256d _illo = _mm256_unpacklo_pd(%s, %s);\n\
             \          const __m256d _ilhi = _mm256_unpackhi_pd(%s, %s);\n\
             \          _mm256_storeu_pd(&out_z[%s], _mm256_permute2f128_pd(_illo, \
              _ilhi, 0x20));\n\
             \          _mm256_storeu_pd(&out_z[%s + 4], _mm256_permute2f128_pd(_illo, \
              _ilhi, 0x31)); }\n"
             a
             bq
             a
             bq
             addr
             addr)
      | 2 ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "        { _mm_storeu_pd(&out_z[%s], _mm_unpacklo_pd(%s, %s));\n\
             \          _mm_storeu_pd(&out_z[%s + 2], _mm_unpackhi_pd(%s, %s)); }\n"
             addr
             a
             bq
             addr
             a
             bq)
      | 1 ->
        Buffer.add_string
          buf
          (Printf.sprintf
             "        out_z[%s] = %s;\n        out_z[%s + 1] = %s;\n"
             addr
             a
             addr
             bq)
      | w ->
        failwith
          (Printf.sprintf
             "codelet_oop il_out: vec_width %d not emitted (avx512 masked lattice \
              pending)"
             w)
    done)
  else (
    let base_re =
      match c.buffer with
      | InPlace -> "rio_re"
      | OutOfPlace -> "out_re"
    in
    let base_im =
      match c.buffer with
      | InPlace -> "rio_im"
      | OutOfPlace -> "out_im"
    in
    Buffer.add_string
      buf
      "        /* UnitGroup store: R separate strided SIMD stores, no transpose. */\n";
    for j = 0 to c.radix - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        %s;\n"
           (Isa.storeu_pd
              ~mode:sc.Emit_render.Scratch.ls_mode
              c.isa
              (Printf.sprintf "%s[b * out_group_stride + %d * out_leg_stride]" base_re j)
              (Printf.sprintf "out_lane_re_%d" j)));
      Buffer.add_string
        buf
        (Printf.sprintf
           "        %s;\n"
           (Isa.storeu_pd
              ~mode:sc.Emit_render.Scratch.ls_mode
              c.isa
              (Printf.sprintf "%s[b * out_group_stride + %d * out_leg_stride]" base_im j)
              (Printf.sprintf "out_lane_im_%d" j)))
    done)
;;

(* ═══════════════════════════════════════════════════════════════
 * STORE EDGE — dispatch
 * ═══════════════════════════════════════════════════════════════ *)

(* Write one FFT output. With store-on-compute + UnitGroup store, stores it
   directly to the output buffer; otherwise accumulates into out_lane_* (the
   default, and the path UnitLeg's transpose requires). `indent` matches the
   surrounding scope. *)
let emit_output_write ~(sc : Emit_render.Scratch.t)
      (buf : Buffer.t)
      (c : config)
      ~(indent : string)
      ~(re : bool)
      ~(j : int)
      ~(tag : int)
  : unit
  =
  if !current_oop_store_on_compute && c.store_pat = UnitGroup
  then (
    let base =
      match c.buffer, re with
      | InPlace, true -> "rio_re"
      | InPlace, false -> "rio_im"
      | OutOfPlace, true -> "out_re"
      | OutOfPlace, false -> "out_im"
    in
    Buffer.add_string
      buf
      (Printf.sprintf
         "%s%s;\n"
         indent
         (Isa.storeu_pd
            ~mode:sc.Emit_render.Scratch.ls_mode
            c.isa
            (Printf.sprintf "%s[b * out_group_stride + %d * out_leg_stride]" base j)
            (Printf.sprintf "t%d" tag))))
  else (
    let lane = if re then "out_lane_re" else "out_lane_im" in
    Buffer.add_string buf (Printf.sprintf "%s%s_%d = t%d;\n" indent lane j tag))
;;

let emit_store_edge ~(sc : Emit_render.Scratch.t) (buf : Buffer.t) (c : config) : unit =
  if !current_post_tw && (c.store_pat <> UnitGroup || !current_oop_store_on_compute)
  then failwith "post-tw requires UnitGroup store without store-on-compute";
  match c.store_pat with
  | UnitLeg -> emit_store_unitleg buf c
  | UnitGroup ->
    (* store-on-compute already wrote every output inline in the body *)
    if !current_oop_store_on_compute then () else emit_store_unitgroup ~sc buf c
  | StridedFallback -> failwith "emit_store_edge: StridedFallback not yet supported"
;;

(* ═══════════════════════════════════════════════════════════════
 * BUTTERFLY BODY (HOOK)
 *
 * M2 phase 1: emit a placeholder. The actual body emission lives
 * in emit_c.ml and needs to be extracted into a callable function
 * during phase 2. The hook signature is fixed now so phase 2 only
 * touches this one site.
 *
 * The body operates on lane_re_j / lane_im_j (set by the load edge)
 * and produces out_lane_re_j / out_lane_im_j (consumed by the store
 * edge). Twiddle access, when present, uses tw_re/tw_im with the
 * per-group layout: tw_re[(j-1)*me + b] for leg j ∈ [1, radix).
 * ═══════════════════════════════════════════════════════════════ *)

(* ═══════════════════════════════════════════════════════════════
 * BUTTERFLY BODY
 *
 * Driven by the same DAG construction as the existing --strided
 * path: Dft.dft_expand (n1) or Dft.dft_expand_twiddled (t1) →
 * Ir.of_assignments → topological sort → render_node_def in
 * order → final stores to out_lane_*_j.
 *
 * Fence emission is wired (prep.fence_enabled, two-rule policy). Tier-C
 * cluster-local SU scheduling is wired on the spill path (see
 * emit_body_spill ~sc ~cfg / Emit_render.cluster_split_schedule). Register allocation
 * (pinning) is NOT emitted on the OOP path — that is the render-convention
 * blocker, genuinely deferred; do not read this as "no scheduling". gcc
 * still does the final instruction scheduling and register allocation.
 *
 * render_node_def with ~strided:true emits `lane_re_j` / `lane_im_j`
 * for Input references, exactly the names populated by the load
 * preamble. The output stores write to `out_lane_re_j` / `out_lane_im_j`
 * which the store postamble then transposes (UnitLeg) or stores
 * directly (UnitGroup).
 *
 * Twiddle support: t1 codelets call dft_expand_twiddled. The math layer
 * emits Load(Twiddle(j-1, ·)); render_load reconciles that against the
 * per-group OOP layout tw_re[j*me + b] (and the t1s scalar / t1p
 * per-position variants). VERIFIED numerically correct: flat, log3, t1s,
 * and t1p (per-position OOP), forward (PRE-twiddle) and backward
 * (POST-twiddle + conj), R16/32/64, all match a naive DFT to ~1e-12. See
 * benchmarks/run_t1_twiddle_gate.sh (the 24-cell gate). The earlier
 * "addressing may need fixup" concern is resolved.
 * ═══════════════════════════════════════════════════════════════ *)

(* ───────────────────────────────────────────────────────────────────
 * PREPARED BODY
 *
 * Output of `prepare_butterfly`: the math layer + algsimp pipeline +
 * spill_info construction, computed BEFORE emission so emit_codelet
 * can use spill_info to declare spill arrays at the right scope
 * (outside the for-loop). Body emission then consumes this record.
 * ─────────────────────────────────────────────────────────────────── *)
type prepared_body =
  { assigns_post : (Expr.elem_ref * Ir.t) list
  ; reachable_nodes : Ir.t list
  ; inline_set : (int, unit) Hashtbl.t
  ; spill_info : Algsimp.spill_info option
  ; fence_enabled : bool
  }

(* ───────────────────────────────────────────────────────────────────
 * PREPARE_BUTTERFLY
 *
 * Math layer DAG construction + Tier-A/Tier-B algsimp pipeline +
 * spill_info build. Mirrors gen_radix.ml's CT-codelet pipeline at
 * lines ~190-600.
 *
 * Tier A (R ≤ 16, n1 always; t1 R ≤ 16): monolithic. spill_info=None.
 * Tier B (R ≥ 25 n1, all t1 size ≥ 5 on AVX-512): blocked construction
 * via dft_expand_n1_blocked / dft_expand_twiddled_spill, spill markers
 * threaded through algsimp as frozen_tags, spill_info built.
 *
 * Gating per Dft.should_block_n1 / Dft.should_spill, matching
 * gen_main.ml's recipe_applicable + construction branch. NOTE: this
 * dispatch is a known cross-file mirror of gen_main's construction
 * selector and is the designated step-3 extraction target
 * (Dft.select_expansion, consumed by both callers) — gen_main's branch
 * has already grown cases this copy lacks (IL2, hc-cascade routes), so it
 * WILL drift when SR-blocked construction gating lands in gen_main. Do
 * the chooser extraction before touching the SR seam. See
 * docs/large_n_pass_minimization_plan.md.
 * ─────────────────────────────────────────────────────────────────── *)

(** When true, OOP twiddled codelets derive twiddles via log3 (load base W^(2^k)
    twiddles, derive the rest by complex multiply) instead of loading all R-1
    directly. Set from gen_radix.ml's --log3 flag. The twiddle table layout is
    unchanged (log3 reads a sparse subset of the same slots). *)
let current_tw_log3 = ref false

let prepare_butterfly ~(sc : Emit_render.Scratch.t) (c : config) : prepared_body =
  let sign : [ `Fwd | `Bwd ] =
    match c.direction with
    | Forward -> `Fwd
    | Backward -> `Bwd
  in
  (* ─── Math layer ────────────────────────────────────────────────
   * Decide whether to use the blocked / spill variant. Cross-file mirror
   * of gen_main's construction selector (step-3 extraction target — see
   * the header note above):
   *   - t1 (PerGroupTwiddles) + should_spill → dft_expand_twiddled_spill
   *   - n1 (NoTwiddles) + should_block_n1     → dft_expand_n1_blocked
   *   - else monolithic.
   * Returns (assigns, spill_markers, ct_factors). The latter two are
   * empty / None when monolithic. ─ *)
  let use_spill_n1 =
    c.twiddles = NoTwiddles && Dft.should_block_n1 c.radix c.isa.Isa.vec_regs
  in
  let use_spill_t1 =
    c.twiddles <> NoTwiddles && Dft.should_spill c.radix c.isa.Isa.vec_regs
  in
  let raw_assigns, spill_markers_raw, spill_ct =
    if !current_post_tw
    then
      (* Gap-A: pure-DFT body; twiddles applied in the store postamble.
         Blocked construction composes: PASS-2 outputs land in out_lane
         accumulators (non-SoC emit_output_write), which is exactly what
         the postamble consumes. Gated by closed-form checks either way. *)
      if Dft.should_block_n1 c.radix c.isa.Isa.vec_regs
      then Dft.dft_expand_n1_blocked ~sign c.radix
      else Dft.dft_expand ~sign c.radix, [], None
    else (
      match c.twiddles with
      | NoTwiddles when use_spill_n1 -> Dft.dft_expand_n1_blocked ~sign c.radix
      | NoTwiddles -> Dft.dft_expand ~sign c.radix, [], None
      | (PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles) when use_spill_t1 ->
        Dft.dft_expand_twiddled_spill
          ~policy:(if !current_tw_log3 then Dft.TP_Log3 else Dft.TP_Flat)
          ~direction:Dft.DIT
          ~sign
          c.radix
      | PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles ->
        ( Dft.dft_expand_twiddled
            ~policy:(if !current_tw_log3 then Dft.TP_Log3 else Dft.TP_Flat)
            ~direction:Dft.DIT
            ~sign
            c.radix
        , []
        , None ))
  in
  let has_spill = spill_markers_raw <> [] in
  (* ─── Algsimp pipeline ──────────────────────────────────────────
   * Reset hash-cons table — mandatory before of_assignments. Without
   * this, prior codelet generations leak tags into our DAG (we'd see
   * tags higher than any local node, and topological sort by tag
   * would still work, but the spill marker → tag remap chain could
   * resolve to dead nodes from a prior call).
   *
   * After reset, delegate the full cascade + spill marker handling
   * to the shared Pipeline module. Single source of truth with
   * gen_radix.ml — see lib/pipeline.ml for the per-pass commentary
   * and the rationale for the 8-step remap_tag chain.
   *
   * For Bailey CT codelets:
   *   aggressive = false       (Direct primes only; CT must skip
   *                             factor_common_muls / share_subsums
   *                             or Cmul sharing dies)
   *   force/disable_fma_lift   honor env vars same as gen_radix
   *   fuse = 0                 matches gen_main's default fuse ref ─ *)
  Ir.reset ();
  let reassoc = Dft_select.needs_reassoc c.radix in
  let aggressive =
    match Dft_select.pick_algorithm c.radix with
    | Dft_select.Direct -> true
    | Dft_select.Cooley_Tukey _ | Dft_select.Split_radix -> false
  in
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
      ~spill_markers_raw
      ~spill_ct
      ~reassoc
      ~aggressive
      ~algorithm:(Dft_select.pick_algorithm c.radix)
      ~force_fma_lift
      ~disable_fma_lift
      ~build_spill_info:has_spill
      ~fuse:!current_oop_fuse
  in
  let assigns = pipe.assigns in
  let spill_info = pipe.spill_info in
  (* ─── Topological sort of reachable nodes ───────────────────────
   * Single source of truth: Ir.topo_sort_reachable (preds-based,
   * NK_Plus-tolerant). Collects reachable-from-assigns nodes only; spill
   * targets are in this set since they're predecessors of the outputs.
   * (Previously an inline copy with a "Mirrors emit_c" comment — but it
   * used Ir.preds, not emit_c's NK_Plus-fatal version; the shared
   * helper now lives at the Algsimp layer both depend on.) ─ *)
  let roots = List.map snd assigns in
  let nodes = Ir.topo_sort_reachable roots in
  let _ = has_spill in
  (* still used downstream via spill_info presence *)
  (* ─── compute_inline_set ────────────────────────────────────────
   * Tags with use_count=1 (excluding Load/Const/Cmul/sinks) get
   * inlined at their consumer. For the spill path the set is filtered
   * to exclude spilled tags and cross-pass consumers.
   *
   * Single source of truth: Emit_render.filter_inline_set_cross_pass ~sc (this
   * was previously a hand-copy with a "we replicate that filter here"
   * comment — see section 37 on mirror drift). The no-spill case is
   * the unfiltered compute_inline_set. ─ *)
  let inline_set =
    match spill_info with
    | None -> Emit_render.compute_inline_set ~sc assigns
    | Some sp -> Emit_render.filter_inline_set_cross_pass ~sc assigns sp nodes
  in
  (* ─── Fence policy ──────────────────────────────────────────────
   * M-PROJECT OFF BY DEFAULT (2026-06-09 flip; emit_c.ml:1364). The
   * scheduling fence is net-negative-or-tie on gcc-13 (it fragments live
   * ranges and defeats operand folding) and the protective role it played
   * on gcc-11 is obsolete (IR-lifted FMAs, gcc fuses on its own). This path
   * previously hardcoded fence ON — that ignored the flip and is corrected
   * here to mirror the in-place gate EXACTLY: opt-in via env only. Pin stays
   * off on this path (regalloc deferred to Tier C). ─ *)
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
  let fence_enabled = (not opt_out) && (force_pin || force_fence) in
  { assigns_post = assigns
  ; reachable_nodes = nodes
  ; inline_set
  ; spill_info
  ; fence_enabled
  }
;;

(* ───────────────────────────────────────────────────────────────────
 * EMIT_BODY_MONOLITHIC (Tier A path)
 *
 * No spill markers. All values stay in scope for the whole codelet;
 * gcc handles allocation. This is the existing Tier-A behavior,
 * unchanged from the previous wiring.
 * ─────────────────────────────────────────────────────────────────── *)
let emit_body_monolithic ~(sc : Emit_render.Scratch.t) ~(cfg : Emit_render.Cfg.t) (buf : Buffer.t) (c : config) (prep : prepared_body) : unit =
  Buffer.add_string buf "\n";
  Buffer.add_string buf "        /* === BUTTERFLY BODY (monolithic) ===\n";
  Buffer.add_string
    buf
    "           Tier A: algsimp cascade + inline + fence, single scope. */\n";
  let tw_broadcast = c.twiddles = BroadcastTwiddles in
  List.iter
    (fun (e : Ir.t) ->
       if not (Hashtbl.mem prep.inline_set e.tag)
       then (
         Buffer.add_string buf "        ";
         Buffer.add_string
           buf
           (Emit_render.render_node_def
         ~sc
         ~cfg
              ~isa:c.isa
              ~in_place:(c.buffer = InPlace)
              ~t1s:tw_broadcast
              ~strided:true
              ~inline_set:(Some prep.inline_set)
              e);
         Buffer.add_char buf '\n'))
    prep.reachable_nodes;
  Buffer.add_char buf '\n';
  List.iter
    (fun (lhs, (e : Ir.t)) ->
       match lhs with
       | Expr.Output (j, true) ->
         emit_output_write ~sc buf c ~indent:"        " ~re:true ~j ~tag:e.tag
       | Expr.Output (j, false) ->
         emit_output_write ~sc buf c ~indent:"        " ~re:false ~j ~tag:e.tag
       | _ ->
         failwith "codelet_oop: assign LHS is not Output (math-layer invariant violated)")
    prep.assigns_post;
  Buffer.add_char buf '\n'
;;

(* ───────────────────────────────────────────────────────────────────
 * EMIT_BODY_SPILL (Tier B/C path)
 *
 * PASS 1 / PASS 2 split per spill_info. This is the OOP-path orchestrator:
 * it composes the shared scheduling/classification helpers with
 * OOP-specific emission (render_node_def, spill-store/reload, output
 * stores via emit_output_write). The shared pieces — single-sourced with
 * the in-place emit_c path — are:
 *   - Emit_render.classify_passes           (PASS 1 vs PASS 2 membership)
 *   - Emit_render.filter_inline_set_cross_pass ~sc (single-use inlining)
 *   - Emit_render.compute_min_slot_pass1    (cluster-membership key + ordering)
 *   - Emit_render.cluster_split_schedule    (Tier-C cluster-local SU)
 *   - Emit_render.is_fused_tag / is_fused_slot (M-project fuse semantics)
 *
 * What is OOP-path-specific (and so legitimately NOT shared with emit_c):
 *   - no regalloc (no current_regalloc; spill stores reference
 *     spill_re/spill_im directly, not regalloc_spill[])
 *   - no store-on-compute (emit_c's soc_* machinery is absent)
 *   - OOP store/load patterns (UnitGroup/UnitLeg) and signatures
 * Fuse: engages when current_oop_fuse > 0 (fused tags stay register-
 * resident across the PASS boundary); fuse=0 is the default, not an
 * invariant.
 *
 * PASS 1 emits cluster-sequentially with spill stores right after each
 * producer (tight lifetimes); PASS 2 reloads on-demand before first use.
 *
 * Caller has already declared spill_re[N] / spill_im[N] at function
 * scope (outside the for-loop), visible across both pass scopes.
 * ─────────────────────────────────────────────────────────────────── *)
let emit_body_spill ~(sc : Emit_render.Scratch.t) ~(cfg : Emit_render.Cfg.t)
      (buf : Buffer.t)
      (c : config)
      (prep : prepared_body)
      (sp : Algsimp.spill_info)
  : unit
  =
  Buffer.add_string buf "\n";
  Buffer.add_string buf "        /* === BUTTERFLY BODY (spill recipe) ===\n";
  Buffer.add_string
    buf
    (Printf.sprintf
       "           Tier B: PASS 1 / PASS 2 split via %d spill slots, fuse=0.\n"
       sp.num_slots);
  Buffer.add_string
    buf
    "           PASS 1 emits cluster-sequentially (by min_descendant_slot)\n";
  Buffer.add_string buf "           with spill stores immediately after each producer.\n";
  Buffer.add_string buf "           PASS 2 reloads on-demand before each consumer.  */\n";
  let tw_broadcast = c.twiddles = BroadcastTwiddles in
  let cls = Emit_render.classify_passes sp prep.reachable_nodes in
  (* Const nodes are hoisted to outer scope (before either pass opens)
     so they're in scope from both. They're free of dependencies and
     each contributes O(1) to live set. *)
  let is_const (e : Ir.t) =
    match e.node with
    | Ir.NK_Const _ -> true
    | _ -> false
  in
  let const_nodes = List.filter is_const prep.reachable_nodes in
  let pass1_nodes =
    List.filter
      (fun (e : Ir.t) ->
         (not (is_const e)) && Hashtbl.find_opt cls e.tag = Some `Pass1)
      prep.reachable_nodes
  in
  let pass2_nodes =
    List.filter
      (fun (e : Ir.t) ->
         (not (is_const e)) && Hashtbl.find_opt cls e.tag = Some `Pass2)
      prep.reachable_nodes
  in
  let pass1_assigns =
    List.filter
      (fun (_, (e : Ir.t)) -> Hashtbl.find_opt cls e.tag = Some `Pass1)
      prep.assigns_post
  in
  let pass2_assigns =
    List.filter
      (fun (_, (e : Ir.t)) -> Hashtbl.find_opt cls e.tag = Some `Pass2)
      prep.assigns_post
  in
  (* Hoist constants. *)
  List.iter
    (fun (e : Ir.t) ->
       Buffer.add_string buf "        ";
       Buffer.add_string
         buf
         (Emit_render.render_node_def
         ~sc
         ~cfg
            ~isa:c.isa
            ~in_place:(c.buffer = InPlace)
            ~t1s:tw_broadcast
            ~strided:true
            ~inline_set:(Some prep.inline_set)
            e);
       Buffer.add_char buf '\n')
    const_nodes;
  Buffer.add_char buf '\n';
  (* ─── Cluster-sequential PASS 1 ordering ───────────────────────
   * Compute min_descendant_slot for each PASS 1 node: the smallest
   * spill slot reachable through its forward successors (within
   * PASS 1). Spill targets have my_slot = own slot. Intermediates
   * inherit min from successors. Used both as a fallback ordering and
   * as the cluster-membership key (cluster = min_slot / ct_n2).
   *
   * Computed by the shared Emit_render.compute_min_slot_pass1 (single source
   * with the in-place path). ─ *)
  let lookup_slot tag =
    match Hashtbl.find_opt sp.re_slot tag with
    | Some s -> Some s
    | None -> Hashtbl.find_opt sp.im_slot tag
  in
  (* min_slot + pre-cluster ordering via the shared Emit_c helper (single
     source with the in-place path; uses an explicit descending sort so it
     does not depend on pass1_nodes' input order). *)
  let min_slot, pass1_blocked_topo = Emit_render.compute_min_slot_pass1 sp pass1_nodes in
  (* ─── Tier C: cluster-local SU scheduling for PASS 1 ──────────────
   * Replace tag-order within each sub-FFT cluster with SU ordering.
   * Cluster boundary: min_slot range corresponding to one PASS 1
   * sub-FFT. For CT(N1, N2), cluster k owns slots [k*N2, (k+1)*N2 - 1].
   *
   * Sub-FFTs are mutually independent (CT property: different n1_idx
   * read disjoint input cells), so SU within a cluster is safe — it
   * cannot reorder across cluster boundaries (no dependency edges to
   * cross). Constants are pre-hoisted to outer scope so they're not
   * in pass1_nodes either.
   *
   * Fallback: if sp.ct_n2 = 0 (non-CT — shouldn't fire for our R≥25
   * which are all CT), or a cluster has no sinks, keep the topo order.
   *
   * The split + per-cluster schedule is the shared
   * Emit_render.cluster_split_schedule (single source with the in-place
   * path). uarch is selected per-ISA below; codelet_oop hardcodes the
   * default (no CLI surface, unlike gen_main's --uarch — intentionally
   * not unified). GH (Goodman-Hsu) auto-enables when AVX2 + R≥32. ─ *)
  (* Per-ISA uarch: the SU latency tables and the GH pressure threshold
   * (raptor_lake_avx2 = 12, avx512 profiles = 24) must match the target
   * register file, or GH engages far too late on 16-register builds. *)
  let uarch =
    if c.isa.Isa.vec_regs <= 16
    then Uarch.raptor_lake_avx2
    else Uarch.sapphire_rapids_avx512
  in
  let gh = c.isa.Isa.vec_regs <= 16 && c.radix >= 32 in
  (* Cluster-split + per-cluster SU via the shared Emit_render.cluster_split_schedule
     (single source with the in-place path). The one caller-specific policy —
     which scheduler to run per cluster — is the closure; the OOP path always
     uses su_schedule_subset (no bb_budget knob). The ct_n2<=0 guard lives
     inside the helper. *)
  let pass1_blocked =
    Emit_render.cluster_split_schedule
      sp
      ~pass1_blocked_topo
      ~min_slot
      ~schedule_cluster:(fun ~subset ~sinks ->
        Schedule.su_schedule_subset uarch ~gh ~subset ~sinks)
  in
  (* ─── PASS 1 emission ──────────────────────────────────────────
   * Open block, emit nodes in cluster-sequential order, emit spill
   * stores immediately after each spilled producer, emit PASS 1
   * outputs, close block. ─ *)
  (* M-project (fuse): a tag whose spill slot is fused stays register-
     resident across the PASS 1 / PASS 2 boundary instead of round-
     tripping through spill_re[]/spill_im[]. Forward-declare such tags at
     loop-body scope (before either pass opens, so both passes see them),
     assign them in PASS 1 with no declarator and no spill store, and skip
     their reload in PASS 2. Fused-tag predicate is the shared
     Emit_render.is_fused_tag (single source with the in-place emission path). *)
  let is_fused_tag tag = Emit_render.is_fused_tag sp tag in
  List.iter
    (fun (e : Ir.t) ->
       if (not (Hashtbl.mem prep.inline_set e.tag)) && is_fused_tag e.tag
       then
         Buffer.add_string
           buf
           (Printf.sprintf "        %s t%d;\n" c.isa.Isa.vec_type e.tag))
    pass1_blocked;
  Buffer.add_string buf "        {  /* PASS 1: sub-FFTs of size n2, store to spill */\n";
  List.iter
    (fun (e : Ir.t) ->
       if not (Hashtbl.mem prep.inline_set e.tag)
       then
         if is_fused_tag e.tag
         then (
           (* assignment to the forward-declared register; no spill store *)
           Buffer.add_string
             buf
             (Emit_render.render_node_def
         ~sc
         ~cfg
                ~no_declarator:true
                ~isa:c.isa
                ~in_place:(c.buffer = InPlace)
                ~t1s:tw_broadcast
                ~strided:true
                ~inline_set:(Some prep.inline_set)
                e);
           Buffer.add_char buf '\n')
         else (
           Buffer.add_string buf "            ";
           Buffer.add_string
             buf
             (Emit_render.render_node_def
         ~sc
         ~cfg
                ~isa:c.isa
                ~in_place:(c.buffer = InPlace)
                ~t1s:tw_broadcast
                ~strided:true
                ~inline_set:(Some prep.inline_set)
                e);
           Buffer.add_char buf '\n';
           (* Spill store(s) for this tag — re_slot and/or im_slot may match.
         The same tag never appears in both (re and im are distinct
         dft_expand_n1_blocked output bins).

         The `double *` cast is REQUIRED for AVX2 — _mm256_storeu_pd
         takes `double *` and rejects `__m256d *` from `&spill_re[N]`.
         For AVX-512 the cast is a no-op accepted via `void *`. Always
         emitting the cast keeps the emitter ISA-independent. *)
           (match Hashtbl.find_opt sp.re_slot e.tag with
            | Some slot ->
              Buffer.add_string
                buf
                (Printf.sprintf
                   "            %s((double *)&spill_re[%d], t%d);\n"
                   c.isa.Isa.storeu_pd
                   slot
                   e.tag)
            | None -> ());
           match Hashtbl.find_opt sp.im_slot e.tag with
           | Some slot ->
             Buffer.add_string
               buf
               (Printf.sprintf
                  "            %s((double *)&spill_im[%d], t%d);\n"
                  c.isa.Isa.storeu_pd
                  slot
                  e.tag)
           | None -> ()))
    pass1_blocked;
  (* PASS 1 output assigns: outputs whose value is computed entirely
     in PASS 1 (no spilled dependency). These exist because some
     output cells of an n1 codelet may bypass the spill boundary
     (e.g. when n2=2 and only one Pass-1 sub-DFT is needed). Emit
     them at end of PASS 1's scope so the value is still in scope. *)
  List.iter
    (fun (lhs, (e : Ir.t)) ->
       match lhs with
       | Expr.Output (j, true) ->
         emit_output_write ~sc buf c ~indent:"            " ~re:true ~j ~tag:e.tag
       | Expr.Output (j, false) ->
         emit_output_write ~sc buf c ~indent:"            " ~re:false ~j ~tag:e.tag
       | _ ->
         failwith "codelet_oop: assign LHS is not Output (math-layer invariant violated)")
    pass1_assigns;
  Buffer.add_string buf "        }\n\n";
  (* ─── PASS 2 emission ──────────────────────────────────────────
   * Open block. For each PASS 2 node, walk its predecessors; for
   * any spilled pred not yet reloaded, emit a reload from
   * spill_re/spill_im. Then emit the node itself. Finally emit
   * PASS 2 output assigns. Close block.
   *
   * Reload format: `const __m512d tN = _mm512_loadu_pd(&spill_re[slot]);`
   * — reusing the same tag name as the original PASS 1 producer (which
   * has gone out of scope when PASS 1's block closed). ─ *)
  Buffer.add_string
    buf
    "        {  /* PASS 2: reload spilled values, sub-FFTs of size n1 */\n";
  let reloaded : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let emit_reload_if_needed (p : Ir.t) =
    if Hashtbl.mem reloaded p.tag
    then ()
    else (
      let do_reload arr_name slot =
        (* Same `double *` cast as the spill stores: required for
           AVX2 — _mm256_loadu_pd takes `double const *`; harmless
           via `void const *` for AVX-512. *)
        Buffer.add_string
          buf
          (Printf.sprintf
             "            const %s t%d = %s((const double *)&%s[%d]);\n"
             c.isa.Isa.vec_type
             p.tag
             c.isa.Isa.loadu_pd
             arr_name
             slot);
        Hashtbl.add reloaded p.tag ()
      in
      match Hashtbl.find_opt sp.re_slot p.tag with
      | Some slot ->
        if Emit_render.is_fused_slot sp slot
        then Hashtbl.add reloaded p.tag ()
        else do_reload "spill_re" slot
      | None ->
        (match Hashtbl.find_opt sp.im_slot p.tag with
         | Some slot ->
           if Emit_render.is_fused_slot sp slot
           then Hashtbl.add reloaded p.tag ()
           else do_reload "spill_im" slot
         | None -> ()))
  in
  (* Transitive reload through inlined predecessors: if X is inlined
     into Z and X references a spilled Y, Z's rendered body
     (with X inlined) references t<Y> directly, so Y must be reloaded.
     emit_reload_if_needed is idempotent so re-visits are safe. *)
  let rec reload_through_inlines (e : Ir.t) =
    emit_reload_if_needed e;
    if Hashtbl.mem prep.inline_set e.tag
    then List.iter reload_through_inlines (Ir.preds e)
  in
  (* ─── Tier C: cluster-local SU scheduling for PASS 2 ──────────────
   * Build cluster_of_pass2_node via the same fixpoint as emit_c's PASS-2
   * path. STATUS: this PASS-2 mirror was assessed during the step-2 de-dup
   * (Q2) and deliberately left unshared for now — it is a SEPARATE
   * candidate pair from the PASS-1 cluster-SU (which IS shared via
   * Emit_render.compute_min_slot_pass1 / cluster_split_schedule). PASS 2 uses a
   * different mechanism (min_input_slot fixpoint + mod-ct_n2 + array-bucket
   * SU, vs PASS 1's contiguous-run split), so the PASS-1 helpers do not
   * serve it; whether emit_c's and codelet_oop's PASS-2 copies are
   * near-verbatim across files is its own question, to evaluate on its own
   * merits (likely alongside the step-3 construction chooser, before SR).
   *
   * Build cluster_of_pass2_node:
   *
   * Step 1: for each PASS 2 node, compute min_input_slot = minimum
   * spill slot it transitively reads. Walk in topo order (low tag
   * first), inheriting min from predecessors. Direct readers of spill
   * (loads from spill_re[slot]/spill_im[slot]) have the slot directly.
   *
   * Step 2: cluster = min_input_slot mod ct_n2. This works because in
   * CT(N1, N2), the PASS 2 sub-DFT-N1 indexed by k2 reads slots
   * {n1_idx * N2 + k2 : n1_idx in 0..N1-1}, and the MIN of those is
   * exactly 0*N2 + k2 = k2. So min_input_slot mod N2 = k2 identifies
   * the PASS 2 sub-DFT.
   *
   * Step 3: fixpoint propagation. Nodes with no spill-slot ancestors
   * (e.g. DIF post-multiply twiddle Loads — they're consumed by Cmuls
   * on outputs) aren't assigned by step 1. Walk pass2_nodes repeatedly,
   * assigning each unclustered node to the MIN of its consumers'
   * clusters. The MIN matters: a shared load consumed by clusters
   * (3, 1, 5) must go to cluster 1, else `concat cluster_0..cluster_N`
   * places its decl AFTER consumers in cluster 1 reference it
   * (use-before-decl).
   *
   * Allow first-walk-assigned nodes to be REDUCED if a smaller consumer
   * cluster appears via later propagation (matches emit_c.ml's fix for
   * (DIF, Fwd) and (DIT, Bwd) log3 cases).
   *
   * Once clustered: group by k2, run SU per group with pass2_assigns
   * sinks within that cluster. Concat in k2 order.
   *
   * The cluster_of_pass2_node table is BOTH the scheduling input AND
   * the cluster-boundary detection key for the per-cluster store flush
   * below. We hoist it to outer scope so the emission loop can read it
   * after pass2_ordered is built. ─ *)
  let cluster_of_pass2_node : (int, int) Hashtbl.t = Hashtbl.create 256 in
  if pass2_nodes <> [] && sp.ct_n2 > 0
  then (
    let min_input_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
    (* Walk in topo order so preds are already classified when we visit. *)
    List.iter
      (fun (e : Ir.t) ->
         let direct = lookup_slot e.tag in
         let pred_min =
           List.fold_left
             (fun acc (p : Ir.t) ->
                match Hashtbl.find_opt min_input_slot p.tag with
                | Some s ->
                  (match acc with
                   | None -> Some s
                   | Some a -> Some (min a s))
                | None -> acc)
             None
             (Ir.preds e)
         in
         let my =
           match direct, pred_min with
           | Some a, Some b -> Some (min a b)
           | Some a, None | None, Some a -> Some a
           | None, None -> None
         in
         match my with
         | Some s -> Hashtbl.replace min_input_slot e.tag s
         | None -> ())
      prep.reachable_nodes;
    List.iter
      (fun (e : Ir.t) ->
         match Hashtbl.find_opt min_input_slot e.tag with
         | Some s -> Hashtbl.replace cluster_of_pass2_node e.tag (s mod sp.ct_n2)
         | None -> ())
      pass2_nodes;
    (* Fixpoint propagation for unclustered intermediates. *)
    let consumers_p2 : (int, Ir.t list) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun (e : Ir.t) ->
         List.iter
           (fun (p : Ir.t) ->
              let prev =
                try Hashtbl.find consumers_p2 p.tag with
                | Not_found -> []
              in
              Hashtbl.replace consumers_p2 p.tag (e :: prev))
           (Ir.preds e))
      pass2_nodes;
    let first_walk : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    Hashtbl.iter (fun tag _ -> Hashtbl.add first_walk tag ()) cluster_of_pass2_node;
    let changed = ref true in
    while !changed do
      changed := false;
      List.iter
        (fun (e : Ir.t) ->
           if not (Hashtbl.mem first_walk e.tag)
           then (
             let cs =
               try Hashtbl.find consumers_p2 e.tag with
               | Not_found -> []
             in
             let consumer_cluster =
               List.fold_left
                 (fun acc (cn : Ir.t) ->
                    match acc, Hashtbl.find_opt cluster_of_pass2_node cn.tag with
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
    if pass2_nodes = [] || sp.ct_n2 <= 0
    then pass2_nodes
    else (
      (* Group pass2_nodes by cluster k2 (preserve relative order within
         a group by reversing twice). *)
      let groups = Array.make sp.ct_n2 [] in
      List.iter
        (fun (e : Ir.t) ->
           match Hashtbl.find_opt cluster_of_pass2_node e.tag with
           | Some k2 -> groups.(k2) <- e :: groups.(k2)
           | None -> ())
        pass2_nodes;
      let assign_tags =
        List.fold_left
          (fun acc (_, (e : Ir.t)) ->
             Hashtbl.replace acc e.tag ();
             acc)
          (Hashtbl.create 32)
          prep.assigns_post
      in
      let result = ref [] in
      for k2 = 0 to sp.ct_n2 - 1 do
        let group_nodes = List.rev groups.(k2) in
        let group_sinks =
          List.filter (fun (e : Ir.t) -> Hashtbl.mem assign_tags e.tag) group_nodes
        in
        let scheduled =
          if group_nodes = []
          then []
          else if group_sinks = []
          then group_nodes
          else
            Schedule.su_schedule_subset uarch ~gh ~subset:group_nodes ~sinks:group_sinks
        in
        result := scheduled :: !result
      done;
      List.concat (List.rev !result))
  in
  (* ─── Cluster-boundary store flush prep ───────────────────────────
   * Production groups pass2_assigns by cluster_of_pass2_node and
   * flushes each cluster's output stores immediately at the END of
   * its cluster.
   *
   * NOT shared with Emit_c's PASS-2 flush (its flush_cluster_stores):
   * the boundary-detection core is the same shape, but Emit_c's version
   * is entangled with two optimizations the OOP path does not have —
   * M5 regalloc (current_regalloc spill_sites/reload_sites emission) and
   * store-on-compute (soc_assigns_by_tag / soc_stored inline stores). The
   * genuinely-shared logic is ~10 lines (assigns_by_cluster grouping +
   * the `prev <> now -> flush prev` pattern); the flush bodies and tail
   * sweeps diverge by design (flushed_tags here vs soc_stored there). This
   * divergence is intentional, not drift: it should converge only when the
   * OOP path gains store-on-compute / regalloc (roadmap), at which point
   * the feature is built once. Do not unify before then — see the uarch
   * non-unification note in emit_codelet for the same reasoning.
   *
   * Why per-cluster flush matters: emitting all output stores at
   * end-of-PASS-2 keeps every output register live until then. With
   * CT(8,8) PASS 2 has 8 clusters of 8 outputs each — at end-of-cluster-0
   * those 8 registers can be freed if their stores fire. Without
   * per-cluster flush, all 64 stay live to the end, raising peak_live and
   * forcing extra gcc spills.
   *
   * Mechanism:
   *   - Group pass2_assigns into assigns_by_cluster[k2]
   *   - During pass2_ordered emission, track last_cluster
   *   - When cluster changes (cur != prev), flush prev's stores
   *   - After loop, flush the final cluster + any unclustered remnants
   *
   * Reload safety: each output value's `tN = ...` declaration must
   * have happened before the store. For PASS 2 outputs the value is
   * a PASS 2 node, so its decl fires when render_node_def emits in
   * the cluster's body. If the value is itself a reload (a PASS 1
   * spill loaded just for this output), emit_reload_if_needed is
   * idempotent — it fires when first referenced and is a no-op on
   * subsequent calls. ─ *)
  let assigns_by_cluster : (int, (Expr.elem_ref * Ir.t) list) Hashtbl.t =
    Hashtbl.create 16
  in
  List.iter
    (fun ((_, (e : Ir.t)) as a) ->
       match Hashtbl.find_opt cluster_of_pass2_node e.tag with
       | Some k2 ->
         let cur =
           try Hashtbl.find assigns_by_cluster k2 with
           | Not_found -> []
         in
         Hashtbl.replace assigns_by_cluster k2 (a :: cur)
       | None -> () (* unclustered → flushed in the tail sweep below *))
    pass2_assigns;
  let flushed_tags : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  let emit_output_store lhs (e : Ir.t) =
    match lhs with
    | Expr.Output (j, true) ->
      emit_output_write ~sc buf c ~indent:"            " ~re:true ~j ~tag:e.tag
    | Expr.Output (j, false) ->
      emit_output_write ~sc buf c ~indent:"            " ~re:false ~j ~tag:e.tag
    | _ ->
      failwith "codelet_oop: assign LHS is not Output (math-layer invariant violated)"
  in
  let flush_cluster_stores k2 =
    match Hashtbl.find_opt assigns_by_cluster k2 with
    | Some clist ->
      (* List was built with `e :: cur` so it's in reverse insertion
         order; List.rev restores the original pass2_assigns order. *)
      List.iter
        (fun (lhs, (e : Ir.t)) ->
           if not (Hashtbl.mem flushed_tags e.tag)
           then (
             (* Edge case: an output value whose only consumer is the
             store itself never gets reloaded during normal pass2_ordered
             emission. Force a reload here if needed. *)
             emit_reload_if_needed e;
             emit_output_store lhs e;
             Hashtbl.add flushed_tags e.tag ()))
        (List.rev clist)
    | None -> ()
  in
  let last_cluster : int option ref = ref None in
  List.iter
    (fun (e : Ir.t) ->
       (* Cluster-boundary detection. Only fire on cluster CHANGE; the
       first node in PASS 2 sets last_cluster without flushing. *)
       (match Hashtbl.find_opt cluster_of_pass2_node e.tag with
        | Some k2 ->
          (match !last_cluster with
           | Some prev when prev <> k2 ->
             flush_cluster_stores prev;
             last_cluster := Some k2
           | None -> last_cluster := Some k2
           | _ -> ())
        | None -> ());
       (* unclustered node — no boundary signal *)
       if not (Hashtbl.mem prep.inline_set e.tag)
       then (
         List.iter reload_through_inlines (Ir.preds e);
         Buffer.add_string buf "            ";
         Buffer.add_string
           buf
           (Emit_render.render_node_def
         ~sc
         ~cfg
              ~isa:c.isa
              ~in_place:(c.buffer = InPlace)
              ~t1s:tw_broadcast
              ~strided:true
              ~inline_set:(Some prep.inline_set)
              e);
         Buffer.add_char buf '\n'))
    pass2_ordered;
  (* Final flush: the last cluster's stores (its boundary never fires
     since there's no following cluster to trigger it). *)
  (match !last_cluster with
   | Some last -> flush_cluster_stores last
   | None -> ());
  (* Tail sweep for any pass2_assigns whose value wasn't in
     cluster_of_pass2_node (shouldn't happen for our CT codelets but
     handle defensively — production also has this safety net). *)
  List.iter
    (fun (lhs, (e : Ir.t)) ->
       if not (Hashtbl.mem flushed_tags e.tag)
       then (
         emit_reload_if_needed e;
         emit_output_store lhs e))
    pass2_assigns;
  Buffer.add_string buf "        }\n\n"
;;

(* ───────────────────────────────────────────────────────────────────
 * EMIT_BUTTERFLY_BODY — Tier A/B dispatch
 *
 * Sets fence policy, dispatches to monolithic or spill path based on
 * prep.spill_info. Uses Fun.protect to ensure the fence ref resets
 * even on exception.
 * ─────────────────────────────────────────────────────────────────── *)
let emit_butterfly_body ~(sc : Emit_render.Scratch.t) ~(cfg : Emit_render.Cfg.t) (buf : Buffer.t) (c : config) (prep : prepared_body) : unit =
  (* M6.1: the historical save/Fun.protect/restore dance around the fence ref
     is RETIRED — sc is created fresh per emission, so nothing can leak into a
     later codelet; a plain field write is the whole protocol now. *)
  sc.Emit_render.Scratch.fence_only <- prep.fence_enabled;
  match prep.spill_info with
  | None -> emit_body_monolithic ~sc ~cfg buf c prep
  | Some sp -> emit_body_spill ~sc ~cfg buf c prep sp
;;

(* ═══════════════════════════════════════════════════════════════
 * TOP-LEVEL CODELET EMISSION
 *
 * Compose: header, signature, lane decls, loop, load, body, store, close.
 * ═══════════════════════════════════════════════════════════════ *)

(** Emit a complete codelet to a fresh string. *)
let emit_codelet (c : config) : string =
  (* M6.1: per-emission scratch — this entry's own instance *)
  let sc = Emit_render.Scratch.create () in
  validate c;
  (* M6.2: the D-2 back-edge writes are DEAD — the twiddle source now flows
     FORWARD as a field of this family's own config view. *)
  let cfg =
    { Emit_render.Cfg.default with
      Emit_render.Cfg.tw =
        (if c.twiddles = PerPositionTwiddles
         then Emit_render.Cfg.Tw_perpos
         else if !current_oop_tw_linear
         then Emit_render.Cfg.Tw_linear (c.radix - 1)
         else Emit_render.Cfg.Tw_default)
    ; store_on_compute = !current_oop_store_on_compute
    }
  in
  let buf = Buffer.create 4096 in
  (* Arbitrary-K rem-aware tail (docs/performance/arbitrary_k_tail_handling.md).
     UnitGroup edges, loop bound me = group count. Masks the in/out group
     loads/stores; body + lane locals stay full-width. Two twiddle kinds:
       - NoTwiddles (n1_oop, LEAF / BAILEY2 stage 1): internal radix twiddles are
         set1 constants, no memory table -> tail masks rio only.
       - PerGroupTwiddles (t1_oop, BAILEY2 stage 2 PER-LANE variant): twiddle is
         loadu(tw[j*me+b]) indexed by the group var -> the tail masks it too
         (render_load is mode-aware). This is the odd-K-correct s2 codelet (vs
         t1p's per-block broadcast, which straddles k2 boundaries at odd K).
     PerPositionTwiddles (t1p, per-block broadcast) is still EXCLUDED — odd K uses
     the t1_oop variant instead. Masked-only remainder (the __m256d lane locals
     would type-clash a width-1 scalar pass). Kill switch VFFT_NO_ANYK_TAIL. *)
  let anyk_tail =
    (c.twiddles = NoTwiddles || c.twiddles = PerGroupTwiddles)
    && c.load_pat = UnitGroup
    && c.store_pat = UnitGroup
    &&
    match Sys.getenv_opt "VFFT_NO_ANYK_TAIL" with
    | Some _ -> false
    | None -> true
  in
  (* File header. *)
  Buffer.add_string
    buf
    "/* Auto-generated by vfft_v2 codelet generator — OOP family (M2). */\n";
  Buffer.add_string buf "#include <immintrin.h>\n";
  Buffer.add_string buf "#include <stddef.h>\n\n";
  (* No _vfft_masklo table: the avx2 tail is all-SSE2 (no mask); AVX-512 computes its
   * __mmask8 = (1<<rem)-1 inline. *)
  emit_signature buf c;
  (* AVX-512 transpose indices at function scope. Needed by UnitLeg
     load preamble and UnitLeg store postamble. Emitting unconditionally
     for AVX-512 keeps the codelet ABI uniform — a no-op when no UnitLeg
     edge is present (gcc will eliminate the unused decls). The
     emit_avx512_transpose_indices helper is itself a no-op for AVX2,
     so this call is safe in all cases. *)
  if c.load_pat = UnitLeg || c.store_pat = UnitLeg
  then Emit_c.emit_avx512_transpose_indices c.isa buf;
  (* Prepare the body: math layer + algsimp pipeline + spill_info
     construction. Done BEFORE the for-loop opens so we know whether
     to emit spill_re/spill_im array declarations at function scope. *)
  let prep = prepare_butterfly ~sc c in
  (* Spill array declarations — outside the for-loop so they're
     allocated once per codelet call, reused across k iterations. The
     in-place path (emit_c) emits the same spill_re[N]/spill_im[N] decls
     in each of its signature variants; not shared here because it's a
     4-line decl that differs trivially per signature (divergence too
     small to centralize — see the design/accident rule in
     docs/large_n_pass_minimization_plan.md). *)
  (match prep.spill_info with
   | None -> ()
   | Some sp ->
     Buffer.add_string
       buf
       (Printf.sprintf "    %s spill_re[%d];\n" c.isa.Isa.vec_type sp.num_slots);
     Buffer.add_string
       buf
       (Printf.sprintf "    %s spill_im[%d];\n" c.isa.Isa.vec_type sp.num_slots));
  let emit_inner () =
    emit_lane_decls buf c;
    emit_load_edge ~sc buf c;
    emit_butterfly_body ~sc ~cfg buf c prep;
    emit_store_edge ~sc buf c
  in
  if anyk_tail
  then (
    (* Rem-aware hybrid tail — THE CONTRACT (docs arbitrary_k_scalartail_experiment):
       bulk full-vector loop, then for the 1..VW-1 leftover batch lanes
         rem == 1 -> ONE scalar single lane (monolithic, width-1 ISA)
         rem >= 2 -> ONE masked vector pass (mask in/out group loads/stores).
       The scalar pass renders MONOLITHICALLY at width 1 (emit_body_monolithic ~sc ~cfg with
       a scalar config): a single lane has no ymm/zmm register pressure, so the CT
       spill split is unnecessary, and the lane locals come out `double` with no
       __m256d clash. me = group count. *)
    let vw = c.isa.Isa.vec_width in
    Buffer.add_string buf "    size_t b = 0;\n";
    Buffer.add_string buf (Printf.sprintf "    for (; b + %d <= me; b += %d) {\n" vw vw);
    sc.Emit_render.Scratch.ls_mode <- Isa.LS_vector;
    emit_inner ();
    Buffer.add_string buf "    }\n";
    Buffer.add_string buf "    if (b < me) {\n";
    Buffer.add_string buf "        const size_t rem = me - b;\n";
    Buffer.add_string buf "        if (rem == 1) {\n";
    (* scalar single lane, monolithic (no vector-register pressure at width 1). *)
    let c_scalar = { c with isa = Isa.scalar } in
    sc.Emit_render.Scratch.ls_mode <- Isa.LS_vector;
    emit_lane_decls buf c_scalar;
    emit_load_edge ~sc buf c_scalar;
    emit_body_monolithic ~sc ~cfg buf c_scalar prep;
    emit_store_edge ~sc buf c_scalar;
    Buffer.add_string buf "        } else {\n";
    if vw = 8
    then (
      (* avx512: masked pass (vmaskz/mask_storeu full-rate; remainder up to 7 lanes). *)
      Buffer.add_string
        buf
        "            const __mmask8 _m = (__mmask8)((1u << rem) - 1u);\n";
      sc.Emit_render.Scratch.ls_mode <- Isa.LS_masked "_m";
      emit_inner ();
      sc.Emit_render.Scratch.ls_mode <- Isa.LS_vector)
    else (
      (* avx2 SSE2 remainder (DEFAULT, mirrors emit_c): a width-2 unmasked loop over the
       * rem lanes + a scalar straggler — faster than vmaskmov on Raptor Lake, and the
       * UnitGroup OOP body is vertical so it narrows 1:1 to 128-bit. Both passes render
       * MONOLITHICALLY (emit_body_monolithic) so a composite codelet does not reference
       * the __m256d spill scratch at width 2 / width 1. *)
      let c_sse2 = { c with isa = Isa.sse2 } in
      sc.Emit_render.Scratch.ls_mode <- Isa.LS_vector;
      Buffer.add_string buf "            for (; b + 2 <= me; b += 2) {\n";
      emit_lane_decls buf c_sse2;
      emit_load_edge ~sc buf c_sse2;
      emit_body_monolithic ~sc ~cfg buf c_sse2 prep;
      emit_store_edge ~sc buf c_sse2;
      Buffer.add_string buf "            }\n";
      Buffer.add_string buf "            if (b < me) {\n";
      emit_lane_decls buf c_scalar;
      emit_load_edge ~sc buf c_scalar;
      emit_body_monolithic ~sc ~cfg buf c_scalar prep;
      emit_store_edge ~sc buf c_scalar;
      Buffer.add_string buf "            }\n");
    Buffer.add_string buf "        }\n";
    Buffer.add_string buf "    }\n";
    Buffer.add_string buf "}\n")
  else (
    emit_loop_open buf c;
    emit_inner ();
    emit_loop_close buf);
  let family =
    let ep = function
      | UnitGroup -> "UG (unit-stride across the transform group)"
      | _ -> "strided/other"
    in
    let tw =
      match c.twiddles with
      | NoTwiddles -> "n1 leaf (no twiddles)"
      | PerPositionTwiddles -> "t1p (per-position twiddles, second-stage)"
      | BroadcastTwiddles -> "t1s-style (broadcast twiddles)"
      | _ -> "twiddled (other kind)"
    in
    let buf_s =
      match c.buffer with
      | InPlace -> "InPlace"
      | OutOfPlace -> "OutOfPlace (Bailey v3_t1)"
    in
    Printf.sprintf
      "OOP %s; edges %s/%s; buffer %s"
      tw
      (ep c.load_pat)
      (ep c.store_pat)
      buf_s
  in
  let vec_regs = c.isa.Isa.vec_regs in
  let blocked = c.twiddles = NoTwiddles && Dft.should_block_n1 c.radix vec_regs in
  let gh = vec_regs <= 16 && c.radix >= 32 in
  let prov =
    Emit_render.provenance_block
      ~family
      [ Printf.sprintf
          "ISA: %d vector regs%s"
          vec_regs
          (if vec_regs <= 16 then " (16-reg pressure rules apply)" else "")
      ; "Scheduler: shared Pipeline cascade + cluster-sequential emission; Tier C \
         cluster-local SU on the spill path (section 24); monolithic path tag-ordered \
         (Tier 1 queue item, section 25)"
      ; Printf.sprintf
          "Tier C uarch: %s (per-ISA selection: GH threshold 12 vs 24; section 24)"
          (if vec_regs <= 16 then "raptor_lake_avx2" else "sapphire_rapids_avx512")
      ; Printf.sprintf "GH pressure mode: %b (auto-rule: vec_regs<=16 && radix>=32)" gh
      ; (if blocked
         then
           "Construction: BLOCKED two-pass (shared dft_expand_n1_blocked, doc 58); \
            threshold n>=16 on <=16-reg ISAs else 25 (section 35)"
         else
           "Construction: MONOLITHIC (below blocking threshold, or twiddled/prime path)")
      ; Printf.sprintf
          "Value fences: %b (Pipeline-computed prep.fence_enabled)"
          prep.fence_enabled
      ; "Regalloc+pinning: not wired on the OOP path (render-convention blocker, section \
         36)"
      ]
  in
  prov ^ Buffer.contents buf
;;

(* ═══════════════════════════════════════════════════════════════
 * NAMING CONVENTION
 *
 * Symbol name pattern: radix<R>_<twkind>_oop_<dir>_<isa>[_<lpat><spat>][_<buf>]
 *
 * Examples:
 *   radix16_n1_oop_fwd_avx512_UL_UG       (UnitLeg load, UnitGroup store)
 *   radix16_t1_oop_fwd_avx512_UL_UG       (with twiddles)
 *   radix8_t1_oop_bwd_avx2_UG_UG_inplace  (in-place Bailey col-FFT shape)
 *
 * Single source of truth for the name pattern so the generator CLI,
 * the linker registration, and the planner agree on lookup keys.
 * ═══════════════════════════════════════════════════════════════ *)

let edge_pattern_suffix = function
  | UnitLeg -> "UL"
  | UnitGroup -> "UG"
  | StridedFallback -> "SF"
;;

let twiddle_suffix = function
  | NoTwiddles -> "n1"
  | PerGroupTwiddles -> "t1"
  | BroadcastTwiddles -> "t1s"
  | PerPositionTwiddles -> "t1p"
;;

let direction_suffix = function
  | Forward -> "fwd"
  | Backward -> "bwd"
;;

let buffer_suffix = function
  | InPlace -> "inplace"
  | OutOfPlace -> "oop"
;;

(** Compose a canonical name from the variant fields. *)
let canonical_name ~radix ~isa ~direction ~load_pat ~store_pat ~buffer ~twiddles : string =
  Printf.sprintf
    "radix%d_%s_%s_%s_%s_%s_%s"
    radix
    (twiddle_suffix twiddles)
    (buffer_suffix buffer)
    (direction_suffix direction)
    Isa.(isa.name)
    (edge_pattern_suffix load_pat)
    (edge_pattern_suffix store_pat)
;;

(* ═══════════════════════════════════════════════════════════════
 * K1 MONO EMISSION (--k1-mono; row_major_engine.md §12.4 item 3)
 *
 * ONE emitted function = the whole K=1 four-step for N = R1*R2, both
 * stages register/L1-resident: per column-chunk h (4 columns at a time)
 * [UG load (leg stride R1) -> radix-R2 body -> four-step twiddle cmul
 * against EMIT-TIME rodata tables -> park in function-scope U vars] ->
 * per row-chunk mh [4x4 register transpose of U -> radix-R1 body -> UG
 * store, natural order]. Generalizes the hand mono-64 (30ns = MKL-IL
 * parity, k1_fourstep_spike.c) with FMA cmuls + scheduler-ordered
 * bodies. M1 scope: N=64 (8x8), split, fwd, avx2. The stage bodies are
 * the SAME prepared radix DAG the OOP family uses; each instantiation
 * is block-scoped so t%d/lane names cannot collide.
 * ═══════════════════════════════════════════════════════════════ *)

(* il=true: z->z interleaved boundaries (the driver's load/store edges emit
   the existing il lattices). sw=true (implies il): the (im,re)-swapped
   lattices — this IS the backward transform via the swap identity
   IDFT = swap(DFT(swap(.))): forward DAG, forward rodata tables, both swaps
   folded into the boundaries (same algebra the 2-pass bwd_il gates proved).
   Split backward needs no codelet at all (caller pointer-swaps re/im). *)
let emit_k1_mono
      ~(isa : Isa.t)
      ~(n : int)
      ~(r1_opt : int option)
      ~(il : bool)
      ~(sw : bool)
  : string
  =
  (* M6.1: per-emission scratch — this entry's own instance *)
  let sc = Emit_render.Scratch.create () in
  if isa.Isa.vec_width <> 4 then failwith "--k1-mono: avx2 only (vec_width 4)";
  if sw && not il then failwith "--k1-mono: --k1-sw requires --k1-il";
  (* default pair per N; --k1-r1 overrides R1 (r2 = n/r1) *)
  let r1 =
    match r1_opt with
    | Some r -> r
    | None ->
      (match n with
       | 64 -> 8
       | 128 -> 16
       | 256 -> 16
       | _ -> failwith "--k1-mono: N in {64,128,256} (M3)")
  in
  if n mod r1 <> 0 then failwith "--k1-mono: R1 must divide N";
  let r2 = n / r1 in
  if r1 mod 4 <> 0 || r2 mod 4 <> 0
  then failwith "--k1-mono: R1 and R2 must be multiples of 4";
  (* neutralize every mode ref that could leak from a prior emission *)
  sc.Emit_render.Scratch.ls_mode <- Isa.LS_vector;
  (* M6.2: the hand "neutralize" writes are DEAD — fresh per-emission cfg. *)
  let cfg = Emit_render.Cfg.default in
  current_post_tw := false;
  current_oop_store_on_compute := false;
  current_oop_strides := None;
  current_oop_il_in := false;
  current_oop_il_out := false;
  current_oop_il_in_sw := false;
  current_oop_il_out_sw := false;
  let fname =
    if il
    then
      Printf.sprintf
        "vfft_k1_mono%d_%dx%d_il_%s_avx2"
        n
        r1
        r2
        (if sw then "bwd" else "fwd")
    else if n = 64 && r1 = 8
    then "vfft_k1_mono64_fwd_avx2" (* M1 name kept *)
    else Printf.sprintf "vfft_k1_mono%d_%dx%d_fwd_avx2" n r1 r2
  in
  let mk_cfg r =
    { radix = r
    ; isa
    ; direction = Forward
    ; load_pat = UnitGroup
    ; store_pat = UnitGroup
    ; buffer = OutOfPlace
    ; twiddles = NoTwiddles
    ; name = fname
    }
  in
  (* two per-stage configs/preps: columns are radix-R2, rows radix-R1. Bodies
     render MONOLITHICALLY even when prepare_butterfly ~sc chose the blocked
     construction (radix 16): emit_body_monolithic ~sc ~cfg ignores spill markers —
     the same precedent the SSE2/scalar tails ship on. *)
  let cfg_col = mk_cfg r2
  and cfg_row = mk_cfg r1 in
  validate cfg_col;
  validate cfg_row;
  let prep_col = prepare_butterfly ~sc cfg_col in
  let prep_row = prepare_butterfly ~sc cfg_row in
  let buf = Buffer.create 65536 in
  Buffer.add_string
    buf
    "/* Auto-generated by vfft_v2 codelet generator — K1 MONO family\n\
    \ * (row_major_engine.md §12.4 item 3). Whole K=1 four-step in ONE\n\
    \ * function; four-step twiddles are EMIT-TIME rodata (no runtime\n\
    \ * table fill, no Qr/Qi). Natural order, split, fwd. */\n\
     #include <immintrin.h>\n\
     #include <stddef.h>\n\n";
  (* rodata four-step diagonal: [h][m-1][lane j] = W_n^{m*(4h+j)} *)
  let pi = 4.0 *. atan 1.0 in
  List.iter
    (fun (nm, f) ->
       Buffer.add_string
         buf
         (Printf.sprintf
            "static const double vfft_k1m%d_%s[%d][%d][4] = {\n"
            n
            nm
            (r1 / 4)
            (r2 - 1));
       for h = 0 to (r1 / 4) - 1 do
         Buffer.add_string buf "  {";
         for m = 1 to r2 - 1 do
           Buffer.add_string buf "{";
           for j = 0 to 3 do
             let a = -2.0 *. pi *. float_of_int (m * ((4 * h) + j)) /. float_of_int n in
             Buffer.add_string buf (Printf.sprintf "%.17g," (f a))
           done;
           Buffer.add_string buf "},"
         done;
         Buffer.add_string buf "},\n"
       done;
       Buffer.add_string buf "};\n")
    [ "twr", cos; "twi", sin ];
  (* uniform 11-arg ABI (vfft_oop11_fn-shaped; strides/tw/me ignored) *)
  if il
  then
    Buffer.add_string
      buf
      (Printf.sprintf
         "\n\
          __attribute__((target(\"avx2,fma\")))\n\
          void %s(\n\
         \    const double * __restrict__ k1_in_z,          /* interleaved pairs */\n\
         \    const double * __restrict__ k1_in_unused,\n\
         \    double       * __restrict__ k1_out_z,         /* interleaved pairs */\n\
         \    double       * __restrict__ k1_out_unused,\n\
         \    const double * tw_re, const double * tw_im,\n\
         \    size_t s0, size_t s1, size_t s2, size_t s3, size_t me)\n\
          {\n\
         \    (void)k1_in_unused; (void)k1_out_unused;\n\
         \    (void)tw_re; (void)tw_im; (void)s0; (void)s1; (void)s2; (void)s3; (void)me;\n"
         fname)
  else
    Buffer.add_string
      buf
      (Printf.sprintf
         "\n\
          __attribute__((target(\"avx2,fma\")))\n\
          void %s(\n\
         \    const double * __restrict__ k1_in_re,\n\
         \    const double * __restrict__ k1_in_im,\n\
         \    double       * __restrict__ k1_out_re,\n\
         \    double       * __restrict__ k1_out_im,\n\
         \    const double * tw_re, const double * tw_im,\n\
         \    size_t s0, size_t s1, size_t s2, size_t s3, size_t me)\n\
          {\n\
         \    (void)tw_re; (void)tw_im; (void)s0; (void)s1; (void)s2; (void)s3; (void)me;\n"
         fname);
  (* function-scope U vars: u_{re,im}_{h}_{m} *)
  for h = 0 to (r1 / 4) - 1 do
    for m = 0 to r2 - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf "    __m256d u_re_%d_%d, u_im_%d_%d;\n" h m h m)
    done
  done;
  (* ---- stage 1: column chunks ---- *)
  current_oop_il_in := il && not sw;
  current_oop_il_in_sw := il && sw;
  for h = 0 to (r1 / 4) - 1 do
    if il
    then
      Buffer.add_string
        buf
        (Printf.sprintf
           "    { /* stage-1 chunk h=%d: columns %d..%d (interleaved) */\n\
           \        const double *in_z = k1_in_z + %d;\n\
           \        const size_t b = 0;\n\
           \        const size_t in_leg_stride = %d;\n\
           \        const size_t in_group_stride = 1;\n\
           \        (void)b; (void)in_group_stride;\n"
           h
           (4 * h)
           ((4 * h) + 3)
           (2 * 4 * h)
           r1)
    else
      Buffer.add_string
        buf
        (Printf.sprintf
           "    { /* stage-1 chunk h=%d: columns %d..%d */\n\
           \        const double *in_re = k1_in_re + %d;\n\
           \        const double *in_im = k1_in_im + %d;\n\
           \        const size_t b = 0;\n\
           \        const size_t in_leg_stride = %d;\n\
           \        const size_t in_group_stride = 1;\n\
           \        (void)b; (void)in_group_stride;\n"
           h
           (4 * h)
           ((4 * h) + 3)
           (4 * h)
           (4 * h)
           r1);
    emit_lane_decls buf cfg_col;
    emit_load_edge ~sc buf cfg_col;
    emit_body_monolithic ~sc ~cfg buf cfg_col prep_col;
    Buffer.add_string
      buf
      (Printf.sprintf
         "        u_re_%d_0 = out_lane_re_0;\n        u_im_%d_0 = out_lane_im_0;\n"
         h
         h);
    for m = 1 to r2 - 1 do
      Buffer.add_string
        buf
        (Printf.sprintf
           "        { const __m256d _twr = _mm256_loadu_pd(vfft_k1m%d_twr[%d][%d]);\n\
           \          const __m256d _twi = _mm256_loadu_pd(vfft_k1m%d_twi[%d][%d]);\n\
           \          u_re_%d_%d = _mm256_fmsub_pd(_twr, out_lane_re_%d, \
            _mm256_mul_pd(_twi, out_lane_im_%d));\n\
           \          u_im_%d_%d = _mm256_fmadd_pd(_twr, out_lane_im_%d, \
            _mm256_mul_pd(_twi, out_lane_re_%d)); }\n"
           n
           h
           (m - 1)
           n
           h
           (m - 1)
           h
           m
           m
           m
           h
           m
           m
           m)
    done;
    Buffer.add_string buf "    }\n"
  done;
  (* ---- stage 2: row chunks (transpose from U + radix-R1 body + store) ---- *)
  for mh = 0 to (r2 / 4) - 1 do
    Buffer.add_string
      buf
      (Printf.sprintf
         "    { /* stage-2 chunk mh=%d: rows m=%d..%d */\n"
         mh
         (4 * mh)
         ((4 * mh) + 3));
    emit_lane_decls buf cfg_row;
    (* T4: leg t=4h+j register = column j across rows u_*_h_{4mh..4mh+3} *)
    for h = 0 to (r1 / 4) - 1 do
      List.iter
        (fun comp ->
           Buffer.add_string
             buf
             (Printf.sprintf
                "        { const __m256d _u0 = _mm256_unpacklo_pd(u_%s_%d_%d, u_%s_%d_%d);\n\
                \          const __m256d _u1 = _mm256_unpackhi_pd(u_%s_%d_%d, u_%s_%d_%d);\n\
                \          const __m256d _u2 = _mm256_unpacklo_pd(u_%s_%d_%d, u_%s_%d_%d);\n\
                \          const __m256d _u3 = _mm256_unpackhi_pd(u_%s_%d_%d, u_%s_%d_%d);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u0, _u2, 0x20);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u1, _u3, 0x20);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u0, _u2, 0x31);\n\
                \          lane_%s_%d = _mm256_permute2f128_pd(_u1, _u3, 0x31); }\n"
                comp
                h
                (4 * mh)
                comp
                h
                ((4 * mh) + 1)
                comp
                h
                (4 * mh)
                comp
                h
                ((4 * mh) + 1)
                comp
                h
                ((4 * mh) + 2)
                comp
                h
                ((4 * mh) + 3)
                comp
                h
                ((4 * mh) + 2)
                comp
                h
                ((4 * mh) + 3)
                comp
                (4 * h)
                comp
                ((4 * h) + 1)
                comp
                ((4 * h) + 2)
                comp
                ((4 * h) + 3)))
        [ "re"; "im" ]
    done;
    emit_body_monolithic ~sc ~cfg buf cfg_row prep_row;
    current_oop_il_in := false;
    current_oop_il_in_sw := false;
    current_oop_il_out := il && not sw;
    current_oop_il_out_sw := il && sw;
    if il
    then
      Buffer.add_string
        buf
        (Printf.sprintf
           "        double *out_z = k1_out_z + %d;\n\
           \        const size_t b = 0;\n\
           \        const size_t out_leg_stride = %d;\n\
           \        const size_t out_group_stride = 1;\n\
           \        (void)b; (void)out_group_stride;\n"
           (2 * 4 * mh)
           r2)
    else
      Buffer.add_string
        buf
        (Printf.sprintf
           "        double *out_re = k1_out_re + %d;\n\
           \        double *out_im = k1_out_im + %d;\n\
           \        const size_t b = 0;\n\
           \        const size_t out_leg_stride = %d;\n\
           \        const size_t out_group_stride = 1;\n\
           \        (void)b; (void)out_group_stride;\n"
           (4 * mh)
           (4 * mh)
           r2);
    emit_store_edge ~sc buf cfg_row;
    Buffer.add_string buf "    }\n"
  done;
  Buffer.add_string buf "}\n";
  current_oop_il_out := false;
  current_oop_il_out_sw := false;
  Buffer.contents buf
;;
