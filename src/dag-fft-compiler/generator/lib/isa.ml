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
 * Design constraint: this module's only consumers are the EMIT layer
 * (emit_c.ml, annotate.ml, future stats reporting) and any heuristic
 * that needs register-pressure info (future SU scheduler). The DAG
 * itself never sees an Isa.t.
 *)

type t = {
  name : string; (* short identifier, "avx512" | "avx2" *)
  vec_type : string; (* C type for one vector, "__m512d" | "__m256d" *)
  vec_width : int; (* doubles per vector: 8 | 4 *)
  vec_regs : int; (* architectural vector register count *)
  intrinsic_prefix : string; (* "_mm512" | "_mm256" *)
  target_attr : string; (* GCC __attribute__((target(...))) string *)
  loadu_pd : string; (* full intrinsic name for unaligned load *)
  storeu_pd : string;
  set1_pd : string;
}

(* === PROFILES ===
 *
 * We avoid per-µarch sub-profiles for now (Sapphire Rapids vs Ice Lake
 * AVX-512, etc.) because the differences are scheduler-relevant, not
 * emission-relevant. A future uarch.ml will add timing parameters on
 * top of the ISA record. *)

let avx512 =
  {
    name = "avx512";
    vec_type = "__m512d";
    vec_width = 8;
    vec_regs = 32;
    intrinsic_prefix = "_mm512";
    target_attr = "avx512f";
    loadu_pd = "_mm512_loadu_pd";
    storeu_pd = "_mm512_storeu_pd";
    set1_pd = "_mm512_set1_pd";
  }

let avx2 =
  {
    name = "avx2";
    vec_type = "__m256d";
    vec_width = 4;
    vec_regs = 16;
    intrinsic_prefix = "_mm256";
    target_attr = "avx2,fma";
    loadu_pd = "_mm256_loadu_pd";
    storeu_pd = "_mm256_storeu_pd";
    set1_pd = "_mm256_set1_pd";
  }

(* Scalar lane (notebook section 53): the cascade's last rung, serving
 * K values that don't fill a vector. vec_width=1; ops render as plain
 * C double arithmetic. FMA renders as __builtin_fma (single rounding,
 * bit-identical to the vector FMA on the same lane, no math.h
 * dependency). Batch lanes never interact, so a lane computed at
 * width 1 is bit-exact with the same lane computed in a zmm. *)
let scalar =
  {
    name = "scalar";
    vec_type = "double";
    vec_width = 1;
    vec_regs = 16;
    intrinsic_prefix = "";
    target_attr = "fma";
    (* Named shims so emit_c's raw `%s(&addr)` / `%s(&addr, v)` spill
     * sites render valid C; the shims are emitted into scalar codelets'
     * preamble by emit_c. The helper-path constructors above bypass
     * these for the hot main-body loads/stores. *)
    loadu_pd = "vfft_scalar_load";
    storeu_pd = "vfft_scalar_store";
    set1_pd = "";
  }

(* Look up by name, for CLI. *)
let of_name (s : string) : t =
  match s with
  | "avx512" | "AVX512" | "avx-512" -> avx512
  | "avx2" | "AVX2" -> avx2
  | "scalar" | "SCALAR" -> scalar
  | other ->
      failwith
        (Printf.sprintf "unknown ISA: %s (expected avx512, avx2, or scalar)"
           other)

(* === INTRINSIC HELPERS ===
 *
 * Construct an intrinsic call string. The pattern is uniform: prefix +
 * underscore + op_pd. We split this into a single helper plus named
 * wrappers for the common cases that have specific call shapes.
 *)

let intr (isa : t) (op : string) : string =
  Printf.sprintf "%s_%s" isa.intrinsic_prefix op

let mul_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "(%s * %s)" a b
  else Printf.sprintf "%s(%s, %s)" (intr isa "mul_pd") a b

let add_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "(%s + %s)" a b
  else Printf.sprintf "%s(%s, %s)" (intr isa "add_pd") a b

let sub_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "(%s - %s)" a b
  else Printf.sprintf "%s(%s, %s)" (intr isa "sub_pd") a b

(* CONTRACT: emit_c uses xor_pd only for sign-flip against the -0.0
 * mask (verified: both call sites pair it with set1("-0.0")). The
 * scalar form is therefore plain negation; the mask operand is
 * ignored. If a future caller needs general bit-xor, extend this. *)
let xor_pd (isa : t) (a : string) (b : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "(-(%s))" a
  else Printf.sprintf "%s(%s, %s)" (intr isa "xor_pd") a b

(* fmadd_pd(a, b, c) = a*b + c    -- standard FMA *)
let fmadd_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "__builtin_fma(%s, %s, %s)" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fmadd_pd") a b c

(* fnmadd_pd(a, b, c) = -a*b + c  -- negated multiplicand, useful for cmul.re *)
let fnmadd_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "__builtin_fma(-(%s), %s, %s)" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fnmadd_pd") a b c

(* fmsub_pd(a, b, c) = a*b - c    -- positive multiplicand, subtract *)
let fmsub_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "__builtin_fma(%s, %s, -(%s))" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fmsub_pd") a b c

(* fnmsub_pd(a, b, c) = -a*b - c   -- negated multiplicand, subtract *)
let fnmsub_pd (isa : t) (a : string) (b : string) (c : string) : string =
  if isa.vec_width = 1 then
    Printf.sprintf "__builtin_fma(-(%s), %s, -(%s))" a b c
  else Printf.sprintf "%s(%s, %s, %s)" (intr isa "fnmsub_pd") a b c

let set1_pd_str (isa : t) (literal : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "(%s)" literal
  else Printf.sprintf "%s(%s)" isa.set1_pd literal

let loadu_pd (isa : t) (addr : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "%s" addr
  else Printf.sprintf "%s(&%s)" isa.loadu_pd addr

let storeu_pd (isa : t) (addr : string) (value : string) : string =
  if isa.vec_width = 1 then Printf.sprintf "%s = %s" addr value
  else Printf.sprintf "%s(&%s, %s)" isa.storeu_pd addr value

(* Render `const __m512d t<tag> = expr;` or its AVX2 equivalent.
 * Used by emit_c's render_node_def. *)
let const_decl (isa : t) (name : string) (expr : string) : string =
  Printf.sprintf "const %s %s = %s;" isa.vec_type name expr

(* Render the register-pinned variant for use by the SSA RA pass:
 *   register __m512d t<tag> asm("zmm5") = expr;
 *   asm volatile ("" : "+v"(t<tag>));
 *
 * The `asm volatile ("" : "+v"(t))` barrier is mandatory — without it
 * gcc-11 treats the `asm("zmmN")` clause as a hint and runs its own
 * RA, ignoring the pin (confirmed via probe). With the barrier, gcc
 * is forced to materialize the variable in the pinned register at
 * that exact point, giving us deterministic register choice. *)
let pinned_reg_decl (isa : t) (name : string) (reg : string) (expr : string) :
    string =
  Printf.sprintf
    "register %s %s asm(\"%s\") = %s; asm volatile (\"\" : \"+v\"(%s));"
    isa.vec_type name reg expr name

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
  Printf.sprintf "register %s %s = %s; asm volatile (\"\" : \"%s\"(%s));"
    isa.vec_type name expr cons name

(* Render `__m512d t1, t2, t3;` for forward declarations from annotate. *)
let forward_decl (isa : t) (names : string list) : string =
  match names with
  | [] -> ""
  | _ -> Printf.sprintf "%s %s;" isa.vec_type (String.concat ", " names)
