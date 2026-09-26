(* isa.ml — minimal ISA abstraction for AVX-512 / AVX2.
 *
 * Both targets are modern x86 with FMA. The differences captured here:
 *   - Vector width (lanes per register for double): AVX-512 = 8, AVX2 = 4
 *   - Architectural register count: AVX-512 = 32 ZMM, AVX2 = 16 YMM
 *   - Intrinsic naming prefix
 *   - C attribute target string
 *
 * What's NOT here:
 *   - FMA pattern preferences (both ISAs are FMA-capable; identical decision)
 *   - Algorithm choices (math is ISA-agnostic; lives in dft.ml)
 *   - Algebraic rewrites (lives in algsimp.ml, deliberately ISA-agnostic)
 *
 * Design constraint: this module's consumers are the EMIT layer and
 * any heuristic that needs register-pressure info (the SU scheduler
 * reads vec_regs through Uarch). The DAG itself never sees an Isa.t.
 * ------------------------------------------------------------------
 * MODULE CARD (isa.ml — grep "MODULE CARD" for the full set)
 * ROLE: ISA record (avx512 / avx2 / sse2 / scalar profiles), the
 * ls_mode vector-vs-masked tail selector, and the intrinsic-name
 * builders (mul_pd .. fnmsub_pd, loadu / storeu / set1, const /
 * pinned / fenced declaration forms).
 * PIPELINE: everything that renders C consults this.
 * PUBLIC SURFACE (measured): emit_render(44), codelet_oop(26),
 * emit_c(20), uarch(8), regalloc(7), emit_state(4), annotate(3),
 * gen_main(1).
 * DEPS: none.
 * ------------------------------------------------------------------
 *)

type t =
  { name : string (* short identifier, "avx512" | "avx2" *)
  ; vec_type : string (* C type for one vector, "__m512d" | "__m256d" *)
  ; vec_width : int (* doubles per vector: 8 | 4 *)
  ; vec_regs : int (* architectural vector register count *)
  ; intrinsic_prefix : string (* "_mm512" | "_mm256" *)
  ; target_attr : string (* GCC __attribute__((target(...))) string *)
  ; loadu_pd : string (* full intrinsic name for unaligned load *)
  ; storeu_pd : string
  ; set1_pd : string
  ; maskload_pd : string
    (* masked unaligned load — avx2 "_mm256_maskload_pd" (addr, mask),
     * avx512 "_mm512_maskz_loadu_pd" (mask, addr). "" for scalar. *)
  ; maskstore_pd : string
    (* masked store — avx2 "_mm256_maskstore_pd" (addr, mask, val),
     * avx512 "_mm512_mask_storeu_pd" (addr, mask, val). "" for scalar. *)
  }

(* Load/store mode for the arbitrary-K tail (notebook section 53 / docs
 * arbitrary_k). The same scheduled DAG renders three ways off ONE schedule:
 *   - LS_vector: the bulk loop, full-width loadu/storeu (default; every
 *     existing codelet path is unchanged because mode defaults to this).
 *   - LS_masked m: ONE remainder pass, maskload/maskstore gated by the C
 *     mask variable `m` (an __m256i for avx2 or __mmask8 for avx512) that
 *     emit_c declares from `rem` at the top of the tail block. Masks only
 *     the k-indexed rio + per-lane twiddle accesses; broadcast twiddles
 *     (set1) and constants are lane-independent and bypass loadu entirely,
 *     so they need no masking.
 *   - the scalar rem==1 lane reuses the width-1 `scalar` ISA, which ignores
 *     mode (a single lane is always active). *)
type ls_mode =
  | LS_vector
  | LS_masked of string

(* === PROFILES ===
 *
 * We avoid per-µarch sub-profiles for now (Sapphire Rapids vs Ice Lake
 * AVX-512, etc.) because the differences are scheduler-relevant, not
 * emission-relevant. A future uarch.ml will add timing parameters on
 * top of the ISA record. *)

let avx512 =
  { name = "avx512"
  ; vec_type = "__m512d"
  ; vec_width = 8
  ; vec_regs = 32
  ; intrinsic_prefix = "_mm512"
  ; target_attr = "avx512f"
  ; loadu_pd = "_mm512_loadu_pd"
  ; storeu_pd = "_mm512_storeu_pd"
  ; set1_pd = "_mm512_set1_pd"
  ; maskload_pd = "_mm512_maskz_loadu_pd"
  ; maskstore_pd = "_mm512_mask_storeu_pd"
  }
;;

let avx2 =
  { name = "avx2"
  ; vec_type = "__m256d"
  ; vec_width = 4
  ; vec_regs = 16
  ; intrinsic_prefix = "_mm256"
  ; target_attr = "avx2,fma"
  ; loadu_pd = "_mm256_loadu_pd"
  ; storeu_pd = "_mm256_storeu_pd"
  ; set1_pd = "_mm256_set1_pd"
  ; maskload_pd = "_mm256_maskload_pd"
  ; maskstore_pd = "_mm256_maskstore_pd"
  }
;;

(* Scalar lane (notebook section 53): the cascade's last rung, serving
 * K values that don't fill a vector. vec_width=1; ops render as plain
 * C double arithmetic. FMA renders as __builtin_fma (single rounding,
 * bit-identical to the vector FMA on the same lane, no math.h
 * dependency). Batch lanes never interact, so a lane computed at
 * width 1 is bit-exact with the same lane computed in a zmm. *)
let scalar =
  { name = "scalar"
  ; vec_type = "double"
  ; vec_width = 1
  ; vec_regs = 16
  ; intrinsic_prefix = ""
  ; target_attr = "fma"
  ; (* Named shims so emit_c's raw `%s(&addr)` / `%s(&addr, v)` spill
     * sites render valid C; the shims are emitted into scalar codelets'
     * preamble by emit_c. The helper-path constructors above bypass
     * these for the hot main-body loads/stores. *)
    loadu_pd = "vfft_scalar_load"
  ; storeu_pd = "vfft_scalar_store"
  ; set1_pd = ""
  ; maskload_pd = ""
  ; maskstore_pd = ""
  }
;;

(* SSE2 + FMA3 (128-bit), used ONLY as the arbitrary-K remainder pass on the AVX2
 * path: 2 doubles/op, full-throughput loads/stores (no vmaskmov). vec_width=2; every
 * arithmetic helper takes the intrinsic path (width<>1) → _mm_add_pd / _mm_fmadd_pd /
 * _mm_mul_pd / _mm_xor_pd via `intr`. No mask intrinsics (the SSE pass is unmasked;
 * an odd straggler lane is mopped up by the width-1 `scalar` ISA). target_attr is
 * cosmetic here — the SSE ops are emitted INLINE inside the enclosing avx2,fma codelet
 * (VEX-128, no AVX↔SSE transition penalty); this record is never emitted standalone. *)
let sse2 =
  { name = "sse2"
  ; vec_type = "__m128d"
  ; vec_width = 2
  ; vec_regs = 16
  ; intrinsic_prefix = "_mm"
  ; target_attr = "sse2,fma"
  ; loadu_pd = "_mm_loadu_pd"
  ; storeu_pd = "_mm_storeu_pd"
  ; set1_pd = "_mm_set1_pd"
  ; maskload_pd = ""
  ; maskstore_pd = ""
  }
;;

(* SCRATCH: 256-bit ops under AVX-512VL k-masks (maskz_loadu / mask_storeu at ymm).
   Emitted INLINE inside the avx512 codelet; never standalone. *)
let avx512vl256 =
  { avx2 with name = "avx512vl256"
  ; maskload_pd = "_mm256_maskz_loadu_pd"
  ; maskstore_pd = "_mm256_mask_storeu_pd"
  }
;;

(* Look up by name, for CLI. *)
let of_name (s : string) : t =
  match s with
  | "avx512" | "AVX512" | "avx-512" -> avx512
  | "avx2" | "AVX2" -> avx2
  | "sse2" | "SSE2" -> sse2
  | "scalar" | "SCALAR" -> scalar
  | other ->
    failwith (Printf.sprintf "unknown ISA: %s (expected avx512, avx2, or scalar)" other)
;;

(* === INTRINSIC HELPERS ===
 *
 * Construct an intrinsic call string. The pattern is uniform: prefix +
 * underscore + op_pd. We split this into a single helper plus named
 * wrappers for the common cases that have specific call shapes.
 *)

let intr (isa : t) (op : string) : string = Printf.sprintf "%s_%s" isa.intrinsic_prefix op

let mul_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "(%s * %s)" a b
  else Printf.sprintf "%s(%s, %s)" (intr isa "mul_pd") a b
;;

let add_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "(%s + %s)" a b
  else Printf.sprintf "%s(%s, %s)" (intr isa "add_pd") a b
;;

let sub_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "(%s - %s)" a b
  else Printf.sprintf "%s(%s, %s)" (intr isa "sub_pd") a b
;;

let addsub_pd (isa : t) (a : string) (b : string) : string =
  (* [a0-b0, a1+b1, ...] — exists only at SSE2/AVX2 widths; wider ISAs have
     no vaddsubpd and the render composes the mask form instead. *)
  if isa.vec_width = 2 || isa.vec_width = 4
  then Printf.sprintf "%s(%s, %s)" (intr isa "addsub_pd") a b
  else failwith "Isa.addsub_pd: no addsub at this width (render must compose)"
;;

(* CONTRACT: emit_c uses xor_pd only for sign-flip against the -0.0
 * mask (verified: both call sites pair it with set1("-0.0")). The
 * scalar form is therefore plain negation; the mask operand is
 * ignored. If a future caller needs general bit-xor, extend this. *)
let xor_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "(-(%s))" a
  else Printf.sprintf "%s(%s, %s)" (intr isa "xor_pd") a b
;;

(* fmadd_pd(a, b, c) = a*b + c    -- standard FMA *)
let fmadd_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "__builtin_fma(%s, %s, %s)" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fmadd_pd") a b c
;;

(* fnmadd_pd(a, b, c) = -a*b + c  -- negated multiplicand, useful for cmul.re *)
let fnmadd_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "__builtin_fma(-(%s), %s, %s)" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fnmadd_pd") a b c
;;

(* fmsub_pd(a, b, c) = a*b - c    -- positive multiplicand, subtract *)
let fmsub_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "__builtin_fma(%s, %s, -(%s))" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fmsub_pd") a b c
;;

(* fnmsub_pd(a, b, c) = -a*b - c   -- negated multiplicand, subtract *)
let fnmsub_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "__builtin_fma(-(%s), %s, -(%s))" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fnmsub_pd") a b c
;;

let set1_pd_str (isa : t) (literal : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "(%s)" literal
  else Printf.sprintf "%s(%s)" isa.set1_pd literal
;;

(* === PACKED-COMPLEX (full-IL) PRIMITIVES ===
 *
 * The interleaved-complex backend (zil_pipeline_port.md §11) holds 2
 * complex per 256-bit vector as [re,im,re,im]. Its add/sub/mul/fma are the
 * SAME intrinsics as the real-lane path (a packed complex add IS a vector
 * add), so those reuse the helpers above unchanged. Only two primitives
 * have no real-lane equivalent:
 *
 *   cflip  — swap re<->im WITHIN each complex. AVX2: vpermilpd imm 0x5
 *            (per-128-bit-lane swap, so it acts on each complex
 *            independently). AVX-512 needs imm 0x55 = 0b01010101, the same
 *            per-lane pattern extended to 4 complex.
 *   xor_mask — general XOR against a named sign-mask vector. The existing
 *            xor_pd is contractually pinned to the -0.0 broadcast (see its
 *            comment); ×(-i) needs the ALTERNATING mask [0,-0,0,-0], which
 *            must be a named constant, hence this second entry point.
 *
 * Together they express the two IL ops:
 *   RotNI x  =  xor_mask(cflip x, _M_IM)         -- x * (-i)
 *   BYTW2    =  fmadd(cvec, x, mul(svec, cflip x))
 *)

(* Per-complex re<->im swap. The immediate differs by width because
 * vpermilpd's control is one bit per DOUBLE, and we want the same
 * "swap within each 128-bit lane" pattern at every width. *)
let cflip_pd (isa : t) (a : string) : string =
  match isa.vec_width with
  | 8 -> Printf.sprintf "_mm512_permute_pd(%s, 0x55)" a
  | 4 -> Printf.sprintf "_mm256_permute_pd(%s, 0x5)" a
  | 2 -> Printf.sprintf "_mm_permute_pd(%s, 0x1)" a
  | w -> failwith (Printf.sprintf "cflip_pd: no packed-complex swap at vec_width %d" w)
;;

(* XOR against a NAMED mask vector (as opposed to xor_pd's -0.0 contract).
 * Used for the ×(-i) sign flip, where the mask alternates [+0,-0,...]. *)
let xor_mask_pd (isa : t) (a : string) (mask : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "(-(%s))" a
  else Printf.sprintf "%s(%s, %s)" (intr isa "xor_pd") a mask
;;

(* The alternating sign mask that turns a cflip into ×(-i):
 *   (a+bi)*(-i) = b - ai  ->  swap to [b,a] then negate the imag lane.
 * Declared once per emitted file by the IL backend's preamble. *)
let im_mask_decl (isa : t) (name : string) : string =
  let lanes = isa.vec_width / 2 in
  let body = String.concat ", " (List.init lanes (fun _ -> "0.0, -0.0") |> fun l -> l) in
  Printf.sprintf "static const %s %s = { %s };" isa.vec_type name body
;;

(* The mirror mask for ×(+i) — the BACKWARD quarter-turn:
 *   (a+bi)*(+i) = -b + ai  ->  cflip to [b,a], then negate the RE lane.
 * Same shape as im_mask_decl with the two lanes swapped. *)
let re_mask_decl (isa : t) (name : string) : string =
  let lanes = isa.vec_width / 2 in
  let body = String.concat ", " (List.init lanes (fun _ -> "-0.0, 0.0")) in
  Printf.sprintf "static const %s %s = { %s };" isa.vec_type name body
;;

(* mode defaults to LS_vector, so all existing positional callers
 * (`loadu_pd isa addr`) render exactly as before. The arbitrary-K tail
 * passes ~mode:(LS_masked m) to switch the rio + per-lane twiddle accesses
 * to masked intrinsics. The width-1 scalar ISA ignores mode (a lone lane
 * is always active). *)
let loadu_pd ?(mode = LS_vector) (isa : t) (addr : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "%s" addr
  else (
    match mode with
    | LS_vector -> Printf.sprintf "%s(&%s)" isa.loadu_pd addr
    | LS_masked m ->
      if isa.vec_width = 8 || isa.name = "avx512vl256"
      then
        (* avx512: maskz_loadu(mask, addr) — zeroes inactive lanes *)
        Printf.sprintf "%s(%s, &%s)" isa.maskload_pd m addr
      else
        (* avx2: maskload(addr, mask) — __m256i sign-bit mask *)
        Printf.sprintf "%s(&%s, %s)" isa.maskload_pd addr m)
;;

let storeu_pd ?(mode = LS_vector) (isa : t) (addr : string) (value : string) : string =
  if isa.vec_width = 1
  then Printf.sprintf "%s = %s" addr value
  else (
    match mode with
    | LS_vector -> Printf.sprintf "%s(&%s, %s)" isa.storeu_pd addr value
    | LS_masked m ->
      (* avx2 maskstore and avx512 mask_storeu both take (addr, mask, val) *)
      Printf.sprintf "%s(&%s, %s, %s)" isa.maskstore_pd addr m value)
;;

(* Render `const __m512d t<tag> = expr;` or its AVX2 equivalent.
 * Used by emit_c's render_node_def. *)
let const_decl (isa : t) (name : string) (expr : string) : string =
  Printf.sprintf "const %s %s = %s;" isa.vec_type name expr
;;

(* Render the register-pinned variant for use by the SSA RA pass:
 *   register __m512d t<tag> asm("zmm5") = expr;
 *   asm volatile ("" : "+v"(t<tag>));
 *
 * The `asm volatile ("" : "+v"(t))` barrier is mandatory — without it
 * gcc-11 treats the `asm("zmmN")` clause as a hint and runs its own
 * RA, ignoring the pin (confirmed via probe). With the barrier, gcc
 * is forced to materialize the variable in the pinned register at
 * that exact point, giving us deterministic register choice. *)
let pinned_reg_decl (isa : t) (name : string) (reg : string) (expr : string) : string =
  Printf.sprintf
    "register %s %s asm(\"%s\") = %s; asm volatile (\"\" : \"+v\"(%s));"
    isa.vec_type
    name
    reg
    expr
    name
;;

(* Render the fence-only variant: same as pinned_reg_decl but without
 * the asm("regN") clause, letting GCC choose the register while
 * keeping the scheduling fence intact:
 *   register __m512d t<tag> = expr;
 *   asm volatile ("" : "+v"(t<tag>));
 *
 * Empirical finding (see docs/fence_pin_decomposition.md): the fence
 * is the actual win mechanism in nearly all codelets — it constrains
 * GCC's scheduler to honor the codelet generator's SU+GH ordering.
 * The asm("regN") pin adds cost (FMA encoding tax, collision
 * preservation, spill staging) without benefit in most cases. This
 * helper is the new default emission for non-pinned-but-fenced
 * variables. *)
let fenced_decl (isa : t) (name : string) (expr : string) : string =
  let cons = if isa.vec_width = 1 then "+x" else "+v" in
  Printf.sprintf
    "register %s %s = %s; asm volatile (\"\" : \"%s\"(%s));"
    isa.vec_type
    name
    expr
    cons
    name
;;

(* Render `__m512d t1, t2, t3;` for forward declarations from annotate. *)
let forward_decl (isa : t) (names : string list) : string =
  match names with
  | [] -> ""
  | _ -> Printf.sprintf "%s %s;" isa.vec_type (String.concat ", " names)
;;

(* === LANE-SHUFFLE OPERATIONS (wf1 zsplit proposal) ===
 * Semantic operations, rendered per width — NOT intrinsic names. The
 * width-4 renderings are byte-for-byte what cascade_z.ml emitted before, so
 * the avx2 corpus is unchanged; width 8 uses permutex2var against the
 * function-scope index constants that [shuffle_consts] declares.
 *   deint_ordered lo hi : two z vectors (VW/2 complex each) -> (re, im) planes,
 *                         lane c = column c.
 *   reint_ordered re im : the inverse, -> (lo, hi) z vectors.
 *   transpose           : a VW x VW transpose of VW vectors. *)
let deint_ordered (isa : t) (lo : string) (hi : string) : string * string =
  match isa.vec_width with
  | 8 ->
    ( Printf.sprintf "_mm512_permutex2var_pd(%s, _zs_de, %s)" lo hi
    , Printf.sprintf "_mm512_permutex2var_pd(%s, _zs_do, %s)" lo hi )
  | 4 ->
    ( Printf.sprintf "%s(%s(%s, %s), 0xD8)" (intr isa "permute4x64_pd") (intr isa "unpacklo_pd") lo hi
    , Printf.sprintf "%s(%s(%s, %s), 0xD8)" (intr isa "permute4x64_pd") (intr isa "unpackhi_pd") lo hi )
  | 2 ->
    ( Printf.sprintf "%s(%s, %s)" (intr isa "unpacklo_pd") lo hi
    , Printf.sprintf "%s(%s, %s)" (intr isa "unpackhi_pd") lo hi )
  | 1 -> lo, hi
  | w -> failwith (Printf.sprintf "Isa.deint_ordered: width %d" w)
;;

(* the plane-side operands of reint_ordered at width 4 are pre-permuted by the
   caller (cascade_z keeps its _pr_/_qi_ temporaries), so this returns the
   PRE-permute (width 4: permute4x64 0xD8) and the final pair separately *)
let reint_pre (isa : t) (v : string) : string =
  match isa.vec_width with
  | 4 -> Printf.sprintf "%s(%s, 0xD8)" (intr isa "permute4x64_pd") v
  | _ -> v
;;

let reint_ordered (isa : t) (re : string) (im : string) : string * string =
  match isa.vec_width with
  | 8 ->
    ( Printf.sprintf "_mm512_permutex2var_pd(%s, _zs_pe, %s)" re im
    , Printf.sprintf "_mm512_permutex2var_pd(%s, _zs_po, %s)" re im )
  | 4 | 2 ->
    ( Printf.sprintf "%s(%s, %s)" (intr isa "unpacklo_pd") re im
    , Printf.sprintf "%s(%s, %s)" (intr isa "unpackhi_pd") re im )
  | 1 -> re, im
  | w -> failwith (Printf.sprintf "Isa.reint_ordered: width %d" w)
;;

(* function-scope constants the width-8 renderings above name *)
let shuffle_consts (isa : t) ~(deint : bool) ~(reint : bool) ~(transpose : bool) : string =
  if isa.vec_width <> 8
  then ""
  else
    String.concat
      ""
      ((if deint
        then
          [ "    const __m512i _zs_de = _mm512_setr_epi64(0,2,4,6,8,10,12,14);\n"
          ; "    const __m512i _zs_do = _mm512_setr_epi64(1,3,5,7,9,11,13,15);\n"
          ]
        else [])
       @ (if reint
          then
            [ "    const __m512i _zs_pe = _mm512_setr_epi64(0,8,1,9,2,10,3,11);\n"
            ; "    const __m512i _zs_po = _mm512_setr_epi64(4,12,5,13,6,14,7,15);\n"
            ]
          else [])
       @
       if transpose
       then
         [ "    const __m512i _zs_tlo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);\n"
         ; "    const __m512i _zs_thi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);\n"
         ]
       else [])
;;

(* VW x VW transpose of srcs.(0..VW-1) into dsts (declared const). Width 4 =
   the TR4 network cascade_z always emitted (4 unpack + 4 permute2f128, the
   _u0_<qid>.. names); width 8 = Simd.load_transpose_8x8's 3-stage lattice
   (8 unpack + 8 permutex2var + 8 shuffle_f64x2). *)
let transpose (isa : t) ~(qid : string) (srcs : string array) (dsts : string array) : string =
  let cd n e = Printf.sprintf "        %s\n" (const_decl isa n e) in
  match isa.vec_width with
  | 4 ->
    let unlo = intr isa "unpacklo_pd"
    and unhi = intr isa "unpackhi_pd"
    and p2f = intr isa "permute2f128_pd" in
    cd (Printf.sprintf "_u0_%s" qid) (Printf.sprintf "%s(%s, %s)" unlo srcs.(0) srcs.(1))
    ^ cd (Printf.sprintf "_u1_%s" qid) (Printf.sprintf "%s(%s, %s)" unhi srcs.(0) srcs.(1))
    ^ cd (Printf.sprintf "_u2_%s" qid) (Printf.sprintf "%s(%s, %s)" unlo srcs.(2) srcs.(3))
    ^ cd (Printf.sprintf "_u3_%s" qid) (Printf.sprintf "%s(%s, %s)" unhi srcs.(2) srcs.(3))
    ^ cd dsts.(0) (Printf.sprintf "%s(_u0_%s, _u2_%s, 0x20)" p2f qid qid)
    ^ cd dsts.(1) (Printf.sprintf "%s(_u1_%s, _u3_%s, 0x20)" p2f qid qid)
    ^ cd dsts.(2) (Printf.sprintf "%s(_u0_%s, _u2_%s, 0x31)" p2f qid qid)
    ^ cd dsts.(3) (Printf.sprintf "%s(_u1_%s, _u3_%s, 0x31)" p2f qid qid)
  | 8 ->
    let b = Buffer.create 2048 in
    for p = 0 to 3 do
      Buffer.add_string b
        (cd (Printf.sprintf "_t%d_%s" (2 * p) qid)
           (Printf.sprintf "_mm512_unpacklo_pd(%s, %s)" srcs.(2 * p) srcs.((2 * p) + 1)));
      Buffer.add_string b
        (cd (Printf.sprintf "_t%d_%s" ((2 * p) + 1) qid)
           (Printf.sprintf "_mm512_unpackhi_pd(%s, %s)" srcs.(2 * p) srcs.((2 * p) + 1)))
    done;
    List.iter
      (fun (x, a, idx, c) ->
         Buffer.add_string b
           (cd (Printf.sprintf "_x%d_%s" x qid)
              (Printf.sprintf "_mm512_permutex2var_pd(_t%d_%s, %s, _t%d_%s)" a qid idx c qid)))
      [ 0, 0, "_zs_tlo", 2; 1, 1, "_zs_tlo", 3; 2, 0, "_zs_thi", 2; 3, 1, "_zs_thi", 3
      ; 4, 4, "_zs_tlo", 6; 5, 5, "_zs_tlo", 7; 6, 4, "_zs_thi", 6; 7, 5, "_zs_thi", 7 ];
    List.iteri
      (fun j (a, c, imm) ->
         Buffer.add_string b
           (cd dsts.(j) (Printf.sprintf "_mm512_shuffle_f64x2(_x%d_%s, _x%d_%s, %s)" a qid c qid imm)))
      [ 0, 4, "0x44"; 1, 5, "0x44"; 2, 6, "0x44"; 3, 7, "0x44"
      ; 0, 4, "0xEE"; 1, 5, "0xEE"; 2, 6, "0xEE"; 3, 7, "0xEE" ];
    Buffer.contents b
  | w -> failwith (Printf.sprintf "Isa.transpose: width %d" w)
;;

(* a vector-width constant splat as a file-scope initializer ({v, v, ...}) *)
let const_splat_decl (isa : t) (name : string) (lit : string) : string =
  Printf.sprintf
    "static const %s %s = { %s };"
    isa.vec_type
    name
    (String.concat ", " (List.init isa.vec_width (fun _ -> lit)))
;;

(* === CORNER-TURN PRIMITIVES (packed complex, width-parametric) ===
 *
 * A "complex lane" is 128 bits ([re,im]). The corner-turn store transposes a
 * per x per block of complex (per = vec_width/2 legs x per columns) with
 * log2(per) rounds of ONE two-source op, the complex-lane DEINTERLEAVE:
 *   even(a,b) = [a0,a2,...,b0,b2,...]     odd(a,b) = [a1,a3,...,b1,b3,...]
 * Round r pairs the current list (L[2i], L[2i+1]) and emits
 * [E_0..E_{n/2-1}, O_0..O_{n/2-1}]; after log2(per) rounds list index =
 * column index (Cx_math.turn_transpose). The op per width:
 *   2 (sse2, 1 complex) : even = a, odd = b           (no instruction)
 *   4 (avx2, 2 complex) : vperm2f128 imm 0x20 / 0x31  (today's bytes)
 *   8 (avx512, 4 cplx)  : vshuff64x2 imm 0x88 / 0xDD  (1 uop, p5, 3c)
 * NOT unpacklo/hi or permute_pd: those act inside each 128-bit lane. *)
let cx_deint_pd (isa : t) ~(odd : bool) (a : string) (b : string) : string =
  match isa.vec_width with
  | 2 -> if odd then b else a
  | 4 -> Printf.sprintf "_mm256_permute2f128_pd(%s, %s, 0x%x)" a b (if odd then 0x31 else 0x20)
  | 8 -> Printf.sprintf "_mm512_shuffle_f64x2(%s, %s, 0x%X)" a b (if odd then 0xDD else 0x88)
  | w -> failwith (Printf.sprintf "cx_deint_pd: no complex-lane deinterleave at vec_width %d" w)
;;

(* Complex lane c of a vector as an __m128d (the scatter quarter). The
 * register-to-memory form (_mm_storeu_pd(p, extract(v,c))) compiles to the
 * store-form vextractf128 / vextractf64x2 m128 — store uops only, no
 * shuffle port. castpd*_pd128 is free. extractf64x2 is AVX512DQ. *)
let cx_part_pd (isa : t) (v : string) (c : int) : string =
  match isa.vec_width, c with
  | 2, 0 -> v
  | 4, 0 -> Printf.sprintf "_mm256_castpd256_pd128(%s)" v
  | 4, 1 -> Printf.sprintf "_mm256_extractf128_pd(%s, 1)" v
  | 8, 0 -> Printf.sprintf "_mm512_castpd512_pd128(%s)" v
  | 8, (1 | 2 | 3) -> Printf.sprintf "_mm512_extractf64x2_pd(%s, %d)" v c
  | w, _ ->
    failwith (Printf.sprintf "cx_part_pd: no complex lane %d at vec_width %d" c w)
;;

(* Store only the first r complex lanes of v at addr (1 <= r <= per).
 * r = per is the plain store. The partial forms are ISA policy: a
 * half-width prefix is a narrower cast store (no mask), anything else a
 * k-masked store (AVX-512 fault-suppresses the masked-off lanes). *)
let storeu_cx_prefix (isa : t) (addr : string) (v : string) (r : int) : string =
  let per = isa.vec_width / 2 in
  if r = per
  then storeu_pd isa addr v
  else (
    match isa.vec_width, r with
    | 4, 1 | 8, 1 -> Printf.sprintf "_mm_storeu_pd(&%s, %s)" addr (cx_part_pd isa v 0)
    | 8, 2 -> Printf.sprintf "_mm256_storeu_pd(&%s, _mm512_castpd512_pd256(%s))" addr v
    | 8, _ ->
      Printf.sprintf "_mm512_mask_storeu_pd(&%s, (__mmask8)0x%X, %s)" addr ((1 lsl (2 * r)) - 1) v
    | w, _ -> failwith (Printf.sprintf "storeu_cx_prefix: r=%d at vec_width %d" r w))
;;

(* The target attribute the packed-complex (cil) family needs. avx2 is the
 * record's own string (byte-identity). avx512: the IL body uses
 * _mm512_xor_pd (DQ), extractf64x2 (DQ), 128-bit masked stores (VL) and
 * the VEX-128 odd-count tail's _mm_fmadd_pd (FMA — NOT implied by
 * target("avx512f") in gcc 13: measured). *)
let cx_target_attr (isa : t) : string =
  if isa.vec_width = 8 then "avx512f,avx512dq,avx512vl,fma" else isa.target_attr
;;
