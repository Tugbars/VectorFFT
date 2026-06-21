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
 * The butterfly body emission (DAG construction, scheduling, regalloc,
 * intrinsic emission) currently lives in Emit_c. For M2 first cut this
 * module emits the signature and edge prologue/epilogue only; the body
 * is provided by an external hook. Once the structure is validated we
 * extract the body emission from Emit_c into a public function and wire
 * it here.
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
  | InPlace  (** Single (rio_re, rio_im) buffer pair. *)
  | OutOfPlace  (** Separate (in_re, in_im) and (out_re, out_im) pairs. *)

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
type direction = Forward | Backward

type config = {
  radix : int;
  isa : Isa.t;
  direction : direction;
  load_pat : edge_pattern;
  store_pat : edge_pattern;
  buffer : buffer_layout;
  twiddles : twiddle_kind;
  name : string;
      (** Symbol name as emitted in the .c file. Caller-supplied to allow
          consistent naming with existing convention (radix_R_t1_oop_fwd_avx512
          etc.). *)
}
(** Full configuration of one codelet variant. *)

(* ═══════════════════════════════════════════════════════════════
 * VALIDATION
 *
 * Enforce the structural constraints that make the rest of emission
 * sound. Errors here indicate a planner bug or an unsupported variant.
 * ═══════════════════════════════════════════════════════════════ *)

(** Raise [Failure] with a clear message if the config is malformed or
    unsupported by M2 first cut. *)
let validate (c : config) : unit =
  if c.radix <= 0 then
    failwith (Printf.sprintf "codelet_oop: radix must be > 0 (got %d)" c.radix);
  (* UnitLeg requires the AOS→SOA transpose preamble to process
     vec_width legs per iteration, which requires radix divisible
     by vec_width. *)
  if c.load_pat = UnitLeg && c.radix mod c.isa.vec_width <> 0 then
    failwith
      (Printf.sprintf
         "codelet_oop: UnitLeg load requires radix %% vec_width = 0 (got \
          radix=%d, vec_width=%d)"
         c.radix c.isa.vec_width);
  if c.store_pat = UnitLeg && c.radix mod c.isa.vec_width <> 0 then
    failwith
      (Printf.sprintf
         "codelet_oop: UnitLeg store requires radix %% vec_width = 0 (got \
          radix=%d, vec_width=%d)"
         c.radix c.isa.vec_width);
  (* M2 first cut: defer StridedFallback. *)
  if c.load_pat = StridedFallback || c.store_pat = StridedFallback then
    failwith "codelet_oop: StridedFallback edge not yet supported in M2"

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

(** Emit the function signature into the buffer. Trailing newline before the
    opening brace of the function body. *)
let emit_signature (buf : Buffer.t) (c : config) : unit =
  Buffer.add_string buf
    (Printf.sprintf "__attribute__((target(\"%s\")))\n" c.isa.target_attr);
  Buffer.add_string buf (Printf.sprintf "void %s(\n" c.name);
  (* Buffer pointers. *)
  (match c.buffer with
  | InPlace ->
      Buffer.add_string buf "    double       * __restrict__ rio_re,\n";
      Buffer.add_string buf "    double       * __restrict__ rio_im,\n"
  | OutOfPlace ->
      Buffer.add_string buf "    const double * __restrict__ in_re,\n";
      Buffer.add_string buf "    const double * __restrict__ in_im,\n";
      Buffer.add_string buf "    double       * __restrict__ out_re,\n";
      Buffer.add_string buf "    double       * __restrict__ out_im,\n");
  (* Twiddles. *)
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
      Buffer.add_string buf
        (Printf.sprintf
           "    /* stride-specialized: strides baked, folds to constant \
            displacements */\n\
           \    const size_t in_leg_stride    = %d;\n\
           \    const size_t in_group_stride  = %d;\n\
           \    const size_t out_leg_stride   = %d;\n\
           \    const size_t out_group_stride = %d;\n"
           l g ol og)
  | None -> ());
  (* Unused-parameter silencing for n1. *)
  match c.twiddles with
  | NoTwiddles -> Buffer.add_string buf "    (void)tw_re; (void)tw_im;\n"
  | PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles -> ()

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
  let need_out_lane =
    not (!current_oop_store_on_compute && c.store_pat = UnitGroup)
  in
  for j = 0 to c.radix - 1 do
    Buffer.add_string buf
      (Printf.sprintf "        %s lane_re_%d, lane_im_%d;\n" c.isa.vec_type j j);
    if need_out_lane then
      Buffer.add_string buf
        (Printf.sprintf "        %s out_lane_re_%d, out_lane_im_%d;\n"
           c.isa.vec_type j j)
  done;
  Buffer.add_string buf "\n"

(* ═══════════════════════════════════════════════════════════════
 * LOOP STRUCTURE
 *
 * The outer loop iterates the group dimension `b` from 0 to me in
 * steps of vec_width. Per iteration: load → body → store.
 * ═══════════════════════════════════════════════════════════════ *)

let emit_loop_open (buf : Buffer.t) (c : config) : unit =
  Buffer.add_string buf
    (Printf.sprintf "    for (size_t b = 0; b < me; b += %d) {\n"
       c.isa.vec_width)

let emit_loop_close (buf : Buffer.t) : unit =
  Buffer.add_string buf "    }\n";
  Buffer.add_string buf "}\n"

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
  let in_re_name =
    match c.buffer with InPlace -> "rio_re" | OutOfPlace -> "in_re"
  in
  let in_im_name =
    match c.buffer with InPlace -> "rio_im" | OutOfPlace -> "in_im"
  in
  (* Reuses the extracted helper from emit_c.ml — identical machinery to
     the existing --strided path's preamble, just parameterized over
     buffer names and stride name. This is exactly the codegen the
     existing 2D row codelets ship with, now driving the M2 OOP family. *)
  Emit_c.emit_strided_load_preamble ~isa:c.isa ~radix:c.radix ~in_re_name
    ~in_im_name ~group_stride_name:"in_group_stride" buf

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

let emit_load_unitgroup (buf : Buffer.t) (c : config) : unit =
  let base_re =
    match c.buffer with InPlace -> "rio_re" | OutOfPlace -> "in_re"
  in
  let base_im =
    match c.buffer with InPlace -> "rio_im" | OutOfPlace -> "in_im"
  in
  Buffer.add_string buf
    "        /* UnitGroup load: vec_width groups are consecutive (stride 1)\n";
  Buffer.add_string buf
    "           so they load as one SIMD register per leg. R separate\n";
  Buffer.add_string buf
    "           strided loads populate the R lane registers — no transpose. */\n";
  for j = 0 to c.radix - 1 do
    Buffer.add_string buf
      (Printf.sprintf
         "        lane_re_%d = %s(&%s[b * in_group_stride + %d * \
          in_leg_stride]);\n"
         j c.isa.loadu_pd base_re j);
    Buffer.add_string buf
      (Printf.sprintf
         "        lane_im_%d = %s(&%s[b * in_group_stride + %d * \
          in_leg_stride]);\n"
         j c.isa.loadu_pd base_im j)
  done

(* ═══════════════════════════════════════════════════════════════
 * LOAD EDGE — dispatch
 * ═══════════════════════════════════════════════════════════════ *)

let emit_load_edge (buf : Buffer.t) (c : config) : unit =
  match c.load_pat with
  | UnitLeg -> emit_load_unitleg buf c
  | UnitGroup -> emit_load_unitgroup buf c
  | StridedFallback ->
      failwith "emit_load_edge: StridedFallback not yet supported"

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
  let out_re_name =
    match c.buffer with InPlace -> "rio_re" | OutOfPlace -> "out_re"
  in
  let out_im_name =
    match c.buffer with InPlace -> "rio_im" | OutOfPlace -> "out_im"
  in
  Emit_c.emit_strided_store_postamble ~isa:c.isa ~radix:c.radix ~out_re_name
    ~out_im_name ~group_stride_name:"out_group_stride" buf

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

let emit_store_unitgroup (buf : Buffer.t) (c : config) : unit =
  let base_re =
    match c.buffer with InPlace -> "rio_re" | OutOfPlace -> "out_re"
  in
  let base_im =
    match c.buffer with InPlace -> "rio_im" | OutOfPlace -> "out_im"
  in
  Buffer.add_string buf
    "        /* UnitGroup store: R separate strided SIMD stores, no transpose. \
     */\n";
  for j = 0 to c.radix - 1 do
    Buffer.add_string buf
      (Printf.sprintf
         "        %s(&%s[b * out_group_stride + %d * out_leg_stride], \
          out_lane_re_%d);\n"
         c.isa.storeu_pd base_re j j);
    Buffer.add_string buf
      (Printf.sprintf
         "        %s(&%s[b * out_group_stride + %d * out_leg_stride], \
          out_lane_im_%d);\n"
         c.isa.storeu_pd base_im j j)
  done

(* ═══════════════════════════════════════════════════════════════
 * STORE EDGE — dispatch
 * ═══════════════════════════════════════════════════════════════ *)

(* Write one FFT output. With store-on-compute + UnitGroup store, stores it
   directly to the output buffer; otherwise accumulates into out_lane_* (the
   default, and the path UnitLeg's transpose requires). `indent` matches the
   surrounding scope. *)
let emit_output_write (buf : Buffer.t) (c : config) ~(indent : string)
    ~(re : bool) ~(j : int) ~(tag : int) : unit =
  if !current_oop_store_on_compute && c.store_pat = UnitGroup then begin
    let base =
      match (c.buffer, re) with
      | InPlace, true -> "rio_re"
      | InPlace, false -> "rio_im"
      | OutOfPlace, true -> "out_re"
      | OutOfPlace, false -> "out_im"
    in
    Buffer.add_string buf
      (Printf.sprintf
         "%s%s(&%s[b * out_group_stride + %d * out_leg_stride], t%d);\n" indent
         c.isa.Isa.storeu_pd base j tag)
  end
  else begin
    let lane = if re then "out_lane_re" else "out_lane_im" in
    Buffer.add_string buf (Printf.sprintf "%s%s_%d = t%d;\n" indent lane j tag)
  end

let emit_store_edge (buf : Buffer.t) (c : config) : unit =
  match c.store_pat with
  | UnitLeg -> emit_store_unitleg buf c
  | UnitGroup ->
      (* store-on-compute already wrote every output inline in the body *)
      if !current_oop_store_on_compute then () else emit_store_unitgroup buf c
  | StridedFallback ->
      failwith "emit_store_edge: StridedFallback not yet supported"

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
 * Algsimp.of_assignments → topological sort → render_node_def in
 * order → final stores to out_lane_*_j.
 *
 * Fence emission is wired (prep.fence_enabled, two-rule policy). Tier-C
 * cluster-local SU scheduling is wired on the spill path (see
 * emit_body_spill / Emit_c.cluster_split_schedule). Register allocation
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
type prepared_body = {
  assigns_post : (Expr.elem_ref * Algsimp.t) list;
  reachable_nodes : Algsimp.t list;
  inline_set : (int, unit) Hashtbl.t;
  spill_info : Emit_c.spill_info option;
  fence_enabled : bool;
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

let prepare_butterfly (c : config) : prepared_body =
  let sign : [ `Fwd | `Bwd ] =
    match c.direction with Forward -> `Fwd | Backward -> `Bwd
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
    match c.twiddles with
    | NoTwiddles when use_spill_n1 -> Dft.dft_expand_n1_blocked ~sign c.radix
    | NoTwiddles -> (Dft.dft_expand ~sign c.radix, [], None)
    | (PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles)
      when use_spill_t1 ->
        Dft.dft_expand_twiddled_spill
          ~policy:(if !current_tw_log3 then Dft.TP_Log3 else Dft.TP_Flat)
          ~direction:Dft.DIT ~sign c.radix
    | PerGroupTwiddles | BroadcastTwiddles | PerPositionTwiddles ->
        ( Dft.dft_expand_twiddled
            ~policy:(if !current_tw_log3 then Dft.TP_Log3 else Dft.TP_Flat)
            ~direction:Dft.DIT ~sign c.radix,
          [],
          None )
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
  Algsimp.reset ();
  let reassoc = Dft.needs_reassoc c.radix in
  let aggressive =
    match Dft.pick_algorithm c.radix with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ | Dft.Split_radix -> false
  in
  let force_fma_lift =
    try Sys.getenv "VFFT_FORCE_FMA_LIFT" = "1" with Not_found -> false
  in
  let disable_fma_lift =
    try Sys.getenv "VFFT_DISABLE_FMA_LIFT" = "1" with Not_found -> false
  in
  let pipe : Pipeline.prepared =
    Pipeline.prepare_codelet ~raw_assigns ~spill_markers_raw ~spill_ct ~reassoc
      ~aggressive
      ~algorithm:(Dft.pick_algorithm c.radix)
      ~force_fma_lift ~disable_fma_lift ~build_spill_info:has_spill
      ~fuse:!current_oop_fuse
  in
  let assigns = pipe.assigns in
  let spill_info = pipe.spill_info in

  (* ─── Topological sort of reachable nodes ───────────────────────
   * Single source of truth: Algsimp.topo_sort_reachable (preds-based,
   * NK_Plus-tolerant). Collects reachable-from-assigns nodes only; spill
   * targets are in this set since they're predecessors of the outputs.
   * (Previously an inline copy with a "Mirrors emit_c" comment — but it
   * used Algsimp.preds, not emit_c's NK_Plus-fatal version; the shared
   * helper now lives at the Algsimp layer both depend on.) ─ *)
  let roots = List.map snd assigns in
  let nodes = Algsimp.topo_sort_reachable roots in
  let _ = has_spill in
  (* still used downstream via spill_info presence *)

  (* ─── compute_inline_set ────────────────────────────────────────
   * Tags with use_count=1 (excluding Load/Const/Cmul/sinks) get
   * inlined at their consumer. For the spill path the set is filtered
   * to exclude spilled tags and cross-pass consumers.
   *
   * Single source of truth: Emit_c.filter_inline_set_cross_pass (this
   * was previously a hand-copy with a "we replicate that filter here"
   * comment — see section 37 on mirror drift). The no-spill case is
   * the unfiltered compute_inline_set. ─ *)
  let inline_set =
    match spill_info with
    | None -> Emit_c.compute_inline_set assigns
    | Some sp -> Emit_c.filter_inline_set_cross_pass assigns sp nodes
  in

  (* ─── Fence policy ──────────────────────────────────────────────
   * Two-rule, per docs/fence_pin_decomposition.md:
   *   fence ON by default; OFF when (n1 ∧ AVX2 ∧ R∈{8,16}).
   *   pin is always OFF for codelet_oop in Tier B (regalloc deferred
   *   to Tier C). ─ *)
  let is_n1 = c.twiddles = NoTwiddles in
  let is_avx2 = c.isa.Isa.vec_regs <= 16 in
  let fence_enabled = not (is_n1 && is_avx2 && (c.radix = 8 || c.radix = 16)) in

  {
    assigns_post = assigns;
    reachable_nodes = nodes;
    inline_set;
    spill_info;
    fence_enabled;
  }

(* ───────────────────────────────────────────────────────────────────
 * EMIT_BODY_MONOLITHIC (Tier A path)
 *
 * No spill markers. All values stay in scope for the whole codelet;
 * gcc handles allocation. This is the existing Tier-A behavior,
 * unchanged from the previous wiring.
 * ─────────────────────────────────────────────────────────────────── *)
let emit_body_monolithic (buf : Buffer.t) (c : config) (prep : prepared_body) :
    unit =
  Buffer.add_string buf "\n";
  Buffer.add_string buf "        /* === BUTTERFLY BODY (monolithic) ===\n";
  Buffer.add_string buf
    "           Tier A: algsimp cascade + inline + fence, single scope. */\n";
  let tw_broadcast = c.twiddles = BroadcastTwiddles in
  List.iter
    (fun (e : Algsimp.t) ->
      if not (Hashtbl.mem prep.inline_set e.tag) then begin
        Buffer.add_string buf "        ";
        Buffer.add_string buf
          (Emit_c.render_node_def ~isa:c.isa ~in_place:(c.buffer = InPlace)
             ~t1s:tw_broadcast ~strided:true ~inline_set:(Some prep.inline_set)
             e);
        Buffer.add_char buf '\n'
      end)
    prep.reachable_nodes;
  Buffer.add_char buf '\n';
  List.iter
    (fun (lhs, (e : Algsimp.t)) ->
      match lhs with
      | Expr.Output (j, true) ->
          emit_output_write buf c ~indent:"        " ~re:true ~j ~tag:e.tag
      | Expr.Output (j, false) ->
          emit_output_write buf c ~indent:"        " ~re:false ~j ~tag:e.tag
      | _ ->
          failwith
            "codelet_oop: assign LHS is not Output (math-layer invariant \
             violated)")
    prep.assigns_post;
  Buffer.add_char buf '\n'

(* ───────────────────────────────────────────────────────────────────
 * EMIT_BODY_SPILL (Tier B/C path)
 *
 * PASS 1 / PASS 2 split per spill_info. This is the OOP-path orchestrator:
 * it composes the shared scheduling/classification helpers with
 * OOP-specific emission (render_node_def, spill-store/reload, output
 * stores via emit_output_write). The shared pieces — single-sourced with
 * the in-place emit_c path — are:
 *   - Emit_c.classify_passes           (PASS 1 vs PASS 2 membership)
 *   - Emit_c.filter_inline_set_cross_pass (single-use inlining)
 *   - Emit_c.compute_min_slot_pass1    (cluster-membership key + ordering)
 *   - Emit_c.cluster_split_schedule    (Tier-C cluster-local SU)
 *   - Emit_c.is_fused_tag / is_fused_slot (M-project fuse semantics)
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
let emit_body_spill (buf : Buffer.t) (c : config) (prep : prepared_body)
    (sp : Emit_c.spill_info) : unit =
  Buffer.add_string buf "\n";
  Buffer.add_string buf "        /* === BUTTERFLY BODY (spill recipe) ===\n";
  Buffer.add_string buf
    (Printf.sprintf
       "           Tier B: PASS 1 / PASS 2 split via %d spill slots, fuse=0.\n"
       sp.num_slots);
  Buffer.add_string buf
    "           PASS 1 emits cluster-sequentially (by min_descendant_slot)\n";
  Buffer.add_string buf
    "           with spill stores immediately after each producer.\n";
  Buffer.add_string buf
    "           PASS 2 reloads on-demand before each consumer.  */\n";

  let tw_broadcast = c.twiddles = BroadcastTwiddles in
  let cls = Emit_c.classify_passes sp prep.reachable_nodes in

  (* Const nodes are hoisted to outer scope (before either pass opens)
     so they're in scope from both. They're free of dependencies and
     each contributes O(1) to live set. *)
  let is_const (e : Algsimp.t) =
    match e.node with Algsimp.NK_Const _ -> true | _ -> false
  in
  let const_nodes = List.filter is_const prep.reachable_nodes in
  let pass1_nodes =
    List.filter
      (fun (e : Algsimp.t) ->
        (not (is_const e)) && Hashtbl.find_opt cls e.tag = Some `Pass1)
      prep.reachable_nodes
  in
  let pass2_nodes =
    List.filter
      (fun (e : Algsimp.t) ->
        (not (is_const e)) && Hashtbl.find_opt cls e.tag = Some `Pass2)
      prep.reachable_nodes
  in

  let pass1_assigns =
    List.filter
      (fun (_, (e : Algsimp.t)) -> Hashtbl.find_opt cls e.tag = Some `Pass1)
      prep.assigns_post
  in
  let pass2_assigns =
    List.filter
      (fun (_, (e : Algsimp.t)) -> Hashtbl.find_opt cls e.tag = Some `Pass2)
      prep.assigns_post
  in

  (* Hoist constants. *)
  List.iter
    (fun (e : Algsimp.t) ->
      Buffer.add_string buf "        ";
      Buffer.add_string buf
        (Emit_c.render_node_def ~isa:c.isa ~in_place:(c.buffer = InPlace)
           ~t1s:tw_broadcast ~strided:true ~inline_set:(Some prep.inline_set) e);
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
   * Computed by the shared Emit_c.compute_min_slot_pass1 (single source
   * with the in-place path). ─ *)
  let lookup_slot tag =
    match Hashtbl.find_opt sp.re_slot tag with
    | Some s -> Some s
    | None -> Hashtbl.find_opt sp.im_slot tag
  in
  (* min_slot + pre-cluster ordering via the shared Emit_c helper (single
     source with the in-place path; uses an explicit descending sort so it
     does not depend on pass1_nodes' input order). *)
  let min_slot, pass1_blocked_topo =
    Emit_c.compute_min_slot_pass1 sp pass1_nodes
  in

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
   * Emit_c.cluster_split_schedule (single source with the in-place
   * path). uarch is selected per-ISA below; codelet_oop hardcodes the
   * default (no CLI surface, unlike gen_main's --uarch — intentionally
   * not unified). GH (Goodman-Hsu) auto-enables when AVX2 + R≥32. ─ *)
  (* Per-ISA uarch: the SU latency tables and the GH pressure threshold
   * (raptor_lake_avx2 = 12, avx512 profiles = 24) must match the target
   * register file, or GH engages far too late on 16-register builds. *)
  let uarch =
    if c.isa.Isa.vec_regs <= 16 then Uarch.raptor_lake_avx2
    else Uarch.sapphire_rapids_avx512
  in
  let gh = c.isa.Isa.vec_regs <= 16 && c.radix >= 32 in
  (* Cluster-split + per-cluster SU via the shared Emit_c.cluster_split_schedule
     (single source with the in-place path). The one caller-specific policy —
     which scheduler to run per cluster — is the closure; the OOP path always
     uses su_schedule_subset (no bb_budget knob). The ct_n2<=0 guard lives
     inside the helper. *)
  let pass1_blocked =
    Emit_c.cluster_split_schedule sp ~pass1_blocked_topo ~min_slot
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
     Emit_c.is_fused_tag (single source with the in-place emission path). *)
  let is_fused_tag tag = Emit_c.is_fused_tag sp tag in
  List.iter
    (fun (e : Algsimp.t) ->
      if (not (Hashtbl.mem prep.inline_set e.tag)) && is_fused_tag e.tag then
        Buffer.add_string buf
          (Printf.sprintf "        %s t%d;\n" c.isa.Isa.vec_type e.tag))
    pass1_blocked;
  Buffer.add_string buf
    "        {  /* PASS 1: sub-FFTs of size n2, store to spill */\n";
  List.iter
    (fun (e : Algsimp.t) ->
      if not (Hashtbl.mem prep.inline_set e.tag) then
        begin if is_fused_tag e.tag then begin
          (* assignment to the forward-declared register; no spill store *)
          Buffer.add_string buf
            (Emit_c.render_node_def ~no_declarator:true ~isa:c.isa
               ~in_place:(c.buffer = InPlace) ~t1s:tw_broadcast ~strided:true
               ~inline_set:(Some prep.inline_set) e);
          Buffer.add_char buf '\n'
        end
        else begin
          Buffer.add_string buf "            ";
          Buffer.add_string buf
            (Emit_c.render_node_def ~isa:c.isa ~in_place:(c.buffer = InPlace)
               ~t1s:tw_broadcast ~strided:true
               ~inline_set:(Some prep.inline_set) e);
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
              Buffer.add_string buf
                (Printf.sprintf
                   "            %s((double *)&spill_re[%d], t%d);\n"
                   c.isa.Isa.storeu_pd slot e.tag)
          | None -> ());
          match Hashtbl.find_opt sp.im_slot e.tag with
          | Some slot ->
              Buffer.add_string buf
                (Printf.sprintf
                   "            %s((double *)&spill_im[%d], t%d);\n"
                   c.isa.Isa.storeu_pd slot e.tag)
          | None -> ()
        end
        end)
    pass1_blocked;
  (* PASS 1 output assigns: outputs whose value is computed entirely
     in PASS 1 (no spilled dependency). These exist because some
     output cells of an n1 codelet may bypass the spill boundary
     (e.g. when n2=2 and only one Pass-1 sub-DFT is needed). Emit
     them at end of PASS 1's scope so the value is still in scope. *)
  List.iter
    (fun (lhs, (e : Algsimp.t)) ->
      match lhs with
      | Expr.Output (j, true) ->
          emit_output_write buf c ~indent:"            " ~re:true ~j ~tag:e.tag
      | Expr.Output (j, false) ->
          emit_output_write buf c ~indent:"            " ~re:false ~j ~tag:e.tag
      | _ ->
          failwith
            "codelet_oop: assign LHS is not Output (math-layer invariant \
             violated)")
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
  Buffer.add_string buf
    "        {  /* PASS 2: reload spilled values, sub-FFTs of size n1 */\n";
  let reloaded : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let emit_reload_if_needed (p : Algsimp.t) =
    if Hashtbl.mem reloaded p.tag then ()
    else begin
      let do_reload arr_name slot =
        (* Same `double *` cast as the spill stores: required for
           AVX2 — _mm256_loadu_pd takes `double const *`; harmless
           via `void const *` for AVX-512. *)
        Buffer.add_string buf
          (Printf.sprintf
             "            const %s t%d = %s((const double *)&%s[%d]);\n"
             c.isa.Isa.vec_type p.tag c.isa.Isa.loadu_pd arr_name slot);
        Hashtbl.add reloaded p.tag ()
      in
      match Hashtbl.find_opt sp.re_slot p.tag with
      | Some slot ->
          if Emit_c.is_fused_slot sp slot then Hashtbl.add reloaded p.tag ()
          else do_reload "spill_re" slot
      | None -> (
          match Hashtbl.find_opt sp.im_slot p.tag with
          | Some slot ->
              if Emit_c.is_fused_slot sp slot then Hashtbl.add reloaded p.tag ()
              else do_reload "spill_im" slot
          | None -> ())
    end
  in
  (* Transitive reload through inlined predecessors: if X is inlined
     into Z and X references a spilled Y, Z's rendered body
     (with X inlined) references t<Y> directly, so Y must be reloaded.
     emit_reload_if_needed is idempotent so re-visits are safe. *)
  let rec reload_through_inlines (e : Algsimp.t) =
    emit_reload_if_needed e;
    if Hashtbl.mem prep.inline_set e.tag then
      List.iter reload_through_inlines (Algsimp.preds e)
  in

  (* ─── Tier C: cluster-local SU scheduling for PASS 2 ──────────────
   * Build cluster_of_pass2_node via the same fixpoint as emit_c's PASS-2
   * path. STATUS: this PASS-2 mirror was assessed during the step-2 de-dup
   * (Q2) and deliberately left unshared for now — it is a SEPARATE
   * candidate pair from the PASS-1 cluster-SU (which IS shared via
   * Emit_c.compute_min_slot_pass1 / cluster_split_schedule). PASS 2 uses a
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
  if pass2_nodes <> [] && sp.ct_n2 > 0 then begin
    let min_input_slot : (int, int) Hashtbl.t = Hashtbl.create 256 in
    (* Walk in topo order so preds are already classified when we visit. *)
    List.iter
      (fun (e : Algsimp.t) ->
        let direct = lookup_slot e.tag in
        let pred_min =
          List.fold_left
            (fun acc (p : Algsimp.t) ->
              match Hashtbl.find_opt min_input_slot p.tag with
              | Some s -> (
                  match acc with None -> Some s | Some a -> Some (min a s))
              | None -> acc)
            None (Algsimp.preds e)
        in
        let my =
          match (direct, pred_min) with
          | Some a, Some b -> Some (min a b)
          | Some a, None | None, Some a -> Some a
          | None, None -> None
        in
        match my with
        | Some s -> Hashtbl.replace min_input_slot e.tag s
        | None -> ())
      prep.reachable_nodes;
    List.iter
      (fun (e : Algsimp.t) ->
        match Hashtbl.find_opt min_input_slot e.tag with
        | Some s -> Hashtbl.replace cluster_of_pass2_node e.tag (s mod sp.ct_n2)
        | None -> ())
      pass2_nodes;
    (* Fixpoint propagation for unclustered intermediates. *)
    let consumers_p2 : (int, Algsimp.t list) Hashtbl.t = Hashtbl.create 256 in
    List.iter
      (fun (e : Algsimp.t) ->
        List.iter
          (fun (p : Algsimp.t) ->
            let prev =
              try Hashtbl.find consumers_p2 p.tag with Not_found -> []
            in
            Hashtbl.replace consumers_p2 p.tag (e :: prev))
          (Algsimp.preds e))
      pass2_nodes;
    let first_walk : (int, unit) Hashtbl.t = Hashtbl.create 256 in
    Hashtbl.iter
      (fun tag _ -> Hashtbl.add first_walk tag ())
      cluster_of_pass2_node;
    let changed = ref true in
    while !changed do
      changed := false;
      List.iter
        (fun (e : Algsimp.t) ->
          if not (Hashtbl.mem first_walk e.tag) then begin
            let cs =
              try Hashtbl.find consumers_p2 e.tag with Not_found -> []
            in
            let consumer_cluster =
              List.fold_left
                (fun acc (cn : Algsimp.t) ->
                  match
                    (acc, Hashtbl.find_opt cluster_of_pass2_node cn.tag)
                  with
                  | None, Some k -> Some k
                  | Some a, Some k -> Some (min a k)
                  | _, None -> acc)
                None cs
            in
            match
              (consumer_cluster, Hashtbl.find_opt cluster_of_pass2_node e.tag)
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
    if pass2_nodes = [] || sp.ct_n2 <= 0 then pass2_nodes
    else begin
      (* Group pass2_nodes by cluster k2 (preserve relative order within
         a group by reversing twice). *)
      let groups = Array.make sp.ct_n2 [] in
      List.iter
        (fun (e : Algsimp.t) ->
          match Hashtbl.find_opt cluster_of_pass2_node e.tag with
          | Some k2 -> groups.(k2) <- e :: groups.(k2)
          | None -> ())
        pass2_nodes;
      let assign_tags =
        List.fold_left
          (fun acc (_, (e : Algsimp.t)) ->
            Hashtbl.replace acc e.tag ();
            acc)
          (Hashtbl.create 32) prep.assigns_post
      in
      let result = ref [] in
      for k2 = 0 to sp.ct_n2 - 1 do
        let group_nodes = List.rev groups.(k2) in
        let group_sinks =
          List.filter
            (fun (e : Algsimp.t) -> Hashtbl.mem assign_tags e.tag)
            group_nodes
        in
        let scheduled =
          if group_nodes = [] then []
          else if group_sinks = [] then group_nodes
          else
            Schedule.su_schedule_subset uarch ~gh ~subset:group_nodes
              ~sinks:group_sinks
        in
        result := scheduled :: !result
      done;
      List.concat (List.rev !result)
    end
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
  let assigns_by_cluster : (int, (Expr.elem_ref * Algsimp.t) list) Hashtbl.t =
    Hashtbl.create 16
  in
  List.iter
    (fun ((_, (e : Algsimp.t)) as a) ->
      match Hashtbl.find_opt cluster_of_pass2_node e.tag with
      | Some k2 ->
          let cur =
            try Hashtbl.find assigns_by_cluster k2 with Not_found -> []
          in
          Hashtbl.replace assigns_by_cluster k2 (a :: cur)
      | None -> () (* unclustered → flushed in the tail sweep below *))
    pass2_assigns;
  let flushed_tags : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  let emit_output_store lhs (e : Algsimp.t) =
    match lhs with
    | Expr.Output (j, true) ->
        emit_output_write buf c ~indent:"            " ~re:true ~j ~tag:e.tag
    | Expr.Output (j, false) ->
        emit_output_write buf c ~indent:"            " ~re:false ~j ~tag:e.tag
    | _ ->
        failwith
          "codelet_oop: assign LHS is not Output (math-layer invariant \
           violated)"
  in
  let flush_cluster_stores k2 =
    match Hashtbl.find_opt assigns_by_cluster k2 with
    | Some clist ->
        (* List was built with `e :: cur` so it's in reverse insertion
         order; List.rev restores the original pass2_assigns order. *)
        List.iter
          (fun (lhs, (e : Algsimp.t)) ->
            if not (Hashtbl.mem flushed_tags e.tag) then begin
              (* Edge case: an output value whose only consumer is the
             store itself never gets reloaded during normal pass2_ordered
             emission. Force a reload here if needed. *)
              emit_reload_if_needed e;
              emit_output_store lhs e;
              Hashtbl.add flushed_tags e.tag ()
            end)
          (List.rev clist)
    | None -> ()
  in

  let last_cluster : int option ref = ref None in
  List.iter
    (fun (e : Algsimp.t) ->
      (* Cluster-boundary detection. Only fire on cluster CHANGE; the
       first node in PASS 2 sets last_cluster without flushing. *)
      (match Hashtbl.find_opt cluster_of_pass2_node e.tag with
      | Some k2 -> (
          match !last_cluster with
          | Some prev when prev <> k2 ->
              flush_cluster_stores prev;
              last_cluster := Some k2
          | None -> last_cluster := Some k2
          | _ -> ())
      | None -> ());
      (* unclustered node — no boundary signal *)
      if not (Hashtbl.mem prep.inline_set e.tag) then begin
        List.iter reload_through_inlines (Algsimp.preds e);
        Buffer.add_string buf "            ";
        Buffer.add_string buf
          (Emit_c.render_node_def ~isa:c.isa ~in_place:(c.buffer = InPlace)
             ~t1s:tw_broadcast ~strided:true ~inline_set:(Some prep.inline_set)
             e);
        Buffer.add_char buf '\n'
      end)
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
    (fun (lhs, (e : Algsimp.t)) ->
      if not (Hashtbl.mem flushed_tags e.tag) then begin
        emit_reload_if_needed e;
        emit_output_store lhs e
      end)
    pass2_assigns;
  Buffer.add_string buf "        }\n\n"

(* ───────────────────────────────────────────────────────────────────
 * EMIT_BUTTERFLY_BODY — Tier A/B dispatch
 *
 * Sets fence policy, dispatches to monolithic or spill path based on
 * prep.spill_info. Uses Fun.protect to ensure the fence ref resets
 * even on exception.
 * ─────────────────────────────────────────────────────────────────── *)
let emit_butterfly_body (buf : Buffer.t) (c : config) (prep : prepared_body) :
    unit =
  let saved_fence = !Emit_c.current_fence_only in
  Emit_c.current_fence_only := prep.fence_enabled;
  Fun.protect
    ~finally:(fun () -> Emit_c.current_fence_only := saved_fence)
    (fun () ->
      match prep.spill_info with
      | None -> emit_body_monolithic buf c prep
      | Some sp -> emit_body_spill buf c prep sp)

(* ═══════════════════════════════════════════════════════════════
 * TOP-LEVEL CODELET EMISSION
 *
 * Compose: header, signature, lane decls, loop, load, body, store, close.
 * ═══════════════════════════════════════════════════════════════ *)

(** Emit a complete codelet to a fresh string. *)
let emit_codelet (c : config) : string =
  validate c;
  Emit_c.current_tw_perpos := c.twiddles = PerPositionTwiddles;
  let buf = Buffer.create 4096 in
  (* File header. *)
  Buffer.add_string buf
    "/* Auto-generated by vfft_v2 codelet generator — OOP family (M2). */\n";
  Buffer.add_string buf "#include <immintrin.h>\n";
  Buffer.add_string buf "#include <stddef.h>\n\n";
  emit_signature buf c;
  (* AVX-512 transpose indices at function scope. Needed by UnitLeg
     load preamble and UnitLeg store postamble. Emitting unconditionally
     for AVX-512 keeps the codelet ABI uniform — a no-op when no UnitLeg
     edge is present (gcc will eliminate the unused decls). The
     emit_avx512_transpose_indices helper is itself a no-op for AVX2,
     so this call is safe in all cases. *)
  if c.load_pat = UnitLeg || c.store_pat = UnitLeg then
    Emit_c.emit_avx512_transpose_indices c.isa buf;

  (* Prepare the body: math layer + algsimp pipeline + spill_info
     construction. Done BEFORE the for-loop opens so we know whether
     to emit spill_re/spill_im array declarations at function scope. *)
  let prep = prepare_butterfly c in

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
      Buffer.add_string buf
        (Printf.sprintf "    %s spill_re[%d];\n" c.isa.Isa.vec_type sp.num_slots);
      Buffer.add_string buf
        (Printf.sprintf "    %s spill_im[%d];\n" c.isa.Isa.vec_type sp.num_slots));

  emit_loop_open buf c;
  emit_lane_decls buf c;
  emit_load_edge buf c;
  emit_butterfly_body buf c prep;
  emit_store_edge buf c;
  emit_loop_close buf;
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
    Printf.sprintf "OOP %s; edges %s/%s; buffer %s" tw (ep c.load_pat)
      (ep c.store_pat) buf_s
  in
  let vec_regs = c.isa.Isa.vec_regs in
  let blocked =
    c.twiddles = NoTwiddles && Dft.should_block_n1 c.radix vec_regs
  in
  let gh = vec_regs <= 16 && c.radix >= 32 in
  let prov =
    Emit_c.provenance_block ~family
      [
        Printf.sprintf "ISA: %d vector regs%s" vec_regs
          (if vec_regs <= 16 then " (16-reg pressure rules apply)" else "");
        "Scheduler: shared Pipeline cascade + cluster-sequential emission; \
         Tier C cluster-local SU on the spill path (section 24); monolithic \
         path tag-ordered (Tier 1 queue item, section 25)";
        Printf.sprintf
          "Tier C uarch: %s (per-ISA selection: GH threshold 12 vs 24; section \
           24)"
          (if vec_regs <= 16 then "raptor_lake_avx2"
           else "sapphire_rapids_avx512");
        Printf.sprintf
          "GH pressure mode: %b (auto-rule: vec_regs<=16 && radix>=32)" gh;
        (if blocked then
           "Construction: BLOCKED two-pass (shared dft_expand_n1_blocked, doc \
            58); threshold n>=16 on <=16-reg ISAs else 25 (section 35)"
         else
           "Construction: MONOLITHIC (below blocking threshold, or \
            twiddled/prime path)");
        Printf.sprintf "Value fences: %b (Pipeline-computed prep.fence_enabled)"
          prep.fence_enabled;
        "Regalloc+pinning: not wired on the OOP path (render-convention \
         blocker, section 36)";
      ]
  in
  prov ^ Buffer.contents buf

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

let twiddle_suffix = function
  | NoTwiddles -> "n1"
  | PerGroupTwiddles -> "t1"
  | BroadcastTwiddles -> "t1s"
  | PerPositionTwiddles -> "t1p"

let direction_suffix = function Forward -> "fwd" | Backward -> "bwd"
let buffer_suffix = function InPlace -> "inplace" | OutOfPlace -> "oop"

(** Compose a canonical name from the variant fields. *)
let canonical_name ~radix ~isa ~direction ~load_pat ~store_pat ~buffer ~twiddles
    : string =
  Printf.sprintf "radix%d_%s_%s_%s_%s_%s_%s" radix (twiddle_suffix twiddles)
    (buffer_suffix buffer)
    (direction_suffix direction)
    Isa.(isa.name)
    (edge_pattern_suffix load_pat)
    (edge_pattern_suffix store_pat)
