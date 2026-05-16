(* emit_executor_h.ml — emit plan-shaped specialized executors.
 *
 * Generates one C function per wisdom entry (currently one hard-coded entry
 * for the spike). The emitted function is a drop-in replacement for
 * src/core/executor.h's _stride_execute_fwd_slice_from when the runtime
 * plan matches the (N, K, factors, variants, orient, dir) tuple of the
 * specialization.
 *
 * Compared to the generic executor:
 *   - Codelet calls are direct symbols (no function-pointer indirection)
 *   - Per-group 4-branch variant tree (n1-fallback / log3 / t1s / flat) is
 *     resolved at emit time → one codepath per stage
 *   - needs_tw[g] branch retained (runtime per-group decision)
 *
 * Symbol naming aligned with bin/gen_radix.ml's new convention
 * (radix{R}_{variant}_{isa}, no _gen/_inplace/_su/_spill suffixes).
 *
 * Usage:
 *   dune exec bin/emit_executor_h.exe -- --isa avx2 \
 *     > generated/plan_executors.h
 *)

(* Per-stage variant codes match wisdom format (0=FLAT 1=LOG3 2=T1S 3=BUF).
 * LOG3 / BUF are declared but unused by the spike's hard-coded entry; they
 * exist so that future entries (and the parser landing later) can construct
 * them. Disable warning 37 (unused-constructor). *)
type variant = FLAT | LOG3 | T1S | BUF [@@warning "-37"]

let variant_code = function FLAT -> 0 | LOG3 -> 1 | T1S -> 2 | BUF -> 3

let variant_name = function
  | FLAT -> "FLAT"
  | LOG3 -> "LOG3"
  | T1S  -> "T1S"
  | BUF  -> "BUF"

type plan_entry = {
  n               : int;
  k               : int;
  factors         : int array;
  variants        : variant array;
  use_dif_forward : bool;
}

(* Hard-coded spike entries.
 *
 * Cell 1: N=131072 K=4. Factorization 4×4×4×4×8×4×4×4, variants FLAT,T1S×7.
 *         This is the cell the VTune doc
 *         (docs/dev/vtune_n131072_k4_vfft_vs_mkl.md) targets — the 21%
 *         wrapper share is documented for this plan exactly.
 *
 * Cell 2: N=1024 K=128. Factorization 4×4×4×4×4, variants FLAT,T1S×4.
 *         Validates the (B)+(A) gain on a smaller plan. Also serves as
 *         the comparison cell for Plan C (monolithic radix1024_n1_fwd_avx2),
 *         which exists from the xl_pow2 codelet family. *)
let spike_entries : plan_entry list = [
  { n = 131072; k = 4;
    factors  = [| 4; 4; 4; 4; 8; 4; 4; 4 |];
    variants = [| FLAT; T1S; T1S; T1S; T1S; T1S; T1S; T1S |];
    use_dif_forward = false };
  { n = 1024; k = 128;
    factors  = [| 4; 4; 4; 4; 4 |];
    variants = [| FLAT; T1S; T1S; T1S; T1S |];
    use_dif_forward = false };
]

(* Direction of the transform. Spike emits forward only. *)
type direction = [ `Fwd | `Bwd ]
let dir_str : direction -> string = function `Fwd -> "fwd" | `Bwd -> "bwd"

(* ── Codelet symbol naming (matches gen_radix.ml) ── *)

let n1_symbol ~r ~direction ~isa =
  Printf.sprintf "radix%d_n1_%s_%s" r (dir_str direction) isa

let t1_symbol ~r ~variant ~direction ~isa =
  match variant with
  | FLAT -> Printf.sprintf "radix%d_t1_dit_%s_%s"      r (dir_str direction) isa
  | LOG3 -> Printf.sprintf "radix%d_t1_dit_log3_%s_%s" r (dir_str direction) isa
  | T1S  -> Printf.sprintf "radix%d_t1s_dit_%s_%s"     r (dir_str direction) isa
  | BUF  -> failwith "BUF variant unsupported in spike"

(* Executor function name encodes the full specialization tuple so it's
 * unique across (N, K, factors, variants, orient, dir, isa). Long but
 * unambiguous and greppable. *)
let executor_name (e : plan_entry) (d : direction) ~isa =
  let join_int xs = String.concat "" (List.map string_of_int (Array.to_list xs)) in
  let join_var vs = String.concat ""
                      (List.map (fun v -> string_of_int (variant_code v))
                                (Array.to_list vs)) in
  let orient = if e.use_dif_forward then "dif" else "dit" in
  Printf.sprintf "exec_n%d_k%d_%s_v%s_%s_%s_%s"
    e.n e.k (join_int e.factors) (join_var e.variants) orient
    (dir_str d) isa

(* Collect every codelet symbol the executor will call. Used for emitting
 * extern declarations at the top of the file. *)
let collect_externs (e : plan_entry) (d : direction) ~isa : string list =
  let acc = ref [] in
  Array.iteri (fun s v ->
    let r = e.factors.(s) in
    if s = 0 then
      acc := n1_symbol ~r ~direction:d ~isa :: !acc
    else begin
      acc := t1_symbol ~r ~variant:v ~direction:d ~isa :: !acc;
      (* needs_tw[g]=0 fallback path uses the n1 codelet for the same R. *)
      acc := n1_symbol ~r ~direction:d ~isa :: !acc
    end
  ) e.variants;
  List.sort_uniq compare !acc

(* ── C emission ── *)

(* emit_externs is inlined into emit_header_file now that we union across
 * multiple entries — kept the collect_externs helper for that union. *)

(* Emit the per-stage body. Spike scope: stage 0 is always n1 (no twiddles
 * at first stage); inner stages emit the T1S codepath consuming the
 * pre-walked tape.
 *
 * The tape (st->tape) is a flat array of `num_groups` invocations, each
 * 24 bytes: { base, tw_re, tw_im }. Walking it sequentially gives the HW
 * prefetcher a single contiguous stream to work with, replacing the old
 * pattern of scattered loads across 4-5 separate per-group arrays. *)
let emit_stage (e : plan_entry) (stage_idx : int) (d : direction) ~isa =
  let r = e.factors.(stage_idx) in
  let v = e.variants.(stage_idx) in
  Printf.printf "    /* Stage %d: R=%d, variant=%s */\n" stage_idx r (variant_name v);
  Printf.printf "    if (start_stage <= %d) {\n" stage_idx;
  Printf.printf "        const stride_stage_t *st = &plan->stages[%d];\n" stage_idx;
  Printf.printf "        const stride_invocation_t * __restrict__ tape = st->tape;\n";
  Printf.printf "        const int    num_groups = st->num_groups;\n";
  Printf.printf "        const size_t stride     = st->stride;  /* hoisted */\n";
  if stage_idx = 0 then begin
    (* Stage 0: n1 codelet, no twiddle. Tape's tw_re/tw_im fields are
     * unused for stage 0 — we pass NULL to match the prototype n1
     * 6-arg signature. base is read from the tape; this is the same
     * indexed load as st->group_base[g] but on a sequentially-walked
     * 24-byte stride that HW prefetch handles cleanly. *)
    Printf.printf "        for (int g = 0; g < num_groups; g++) {\n";
    Printf.printf "            size_t base = tape[g].base;\n";
    Printf.printf "            %s(re + base, im + base, NULL, NULL, stride, slice_K);\n"
      (n1_symbol ~r ~direction:d ~isa);
    Printf.printf "        }\n"
  end else begin
    (* Inner stages: T1S spike scope. needs_tw/cf0 branches dropped — the
     * plan-build code asserts all inner-stage groups take the T1S path
     * with trivial cf0. Future iteration: re-introduce those as compile-
     * time-decided codepaths per variant class. *)
    Printf.printf "        for (int g = 0; g < num_groups; g++) {\n";
    Printf.printf "            const stride_invocation_t inv = tape[g];\n";
    (match v with
     | T1S  ->
       Printf.printf "            %s(re + inv.base, im + inv.base,\n"
         (t1_symbol ~r ~variant:T1S ~direction:d ~isa);
       Printf.printf "                                       inv.tw_re, inv.tw_im,\n";
       Printf.printf "                                       stride, slice_K);\n"
     | FLAT ->
       Printf.printf "            /* FLAT variant emission deferred — spike scope is T1S only. */\n";
       Printf.printf "            (void)inv; abort();\n"
     | LOG3 ->
       Printf.printf "            /* LOG3 variant emission deferred — spike scope is T1S only. */\n";
       Printf.printf "            (void)inv; abort();\n"
     | BUF  ->
       Printf.printf "            /* BUF variant emission deferred. */\n";
       Printf.printf "            (void)inv; abort();\n");
    Printf.printf "        }\n"
  end;
  Printf.printf "    }\n"

let emit_executor (e : plan_entry) (d : direction) ~isa =
  let fn_name = executor_name e d ~isa in
  let factors_str  = String.concat "," (List.map string_of_int (Array.to_list e.factors)) in
  let variants_str = String.concat "," (List.map variant_name (Array.to_list e.variants)) in
  let orient = if e.use_dif_forward then "DIF" else "DIT" in
  Printf.printf "/* Plan-shaped executor specialization\n";
  Printf.printf " *   N=%d K=%d\n" e.n e.k;
  Printf.printf " *   factors=%s\n" factors_str;
  Printf.printf " *   variants=%s\n" variants_str;
  Printf.printf " *   orient=%s dir=%s isa=%s\n" orient (String.uppercase_ascii (dir_str d)) isa;
  Printf.printf " *\n";
  Printf.printf " * Drop-in replacement for _stride_execute_fwd_slice_from when the\n";
  Printf.printf " * runtime plan matches the tuple above. */\n";
  Printf.printf "static void %s(const stride_plan_t *plan,\n" fn_name;
  let pad = String.make (String.length fn_name + 12) ' ' in
  Printf.printf "%sdouble *re, double *im,\n" pad;
  Printf.printf "%ssize_t slice_K, size_t full_K,\n" pad;
  Printf.printf "%sint start_stage)\n" pad;
  Printf.printf "{\n";
  Printf.printf "    (void)full_K;  /* unused for T1S-only plans */\n\n";
  for s = 0 to Array.length e.factors - 1 do
    emit_stage e s d ~isa;
    if s < Array.length e.factors - 1 then Printf.printf "\n"
  done;
  Printf.printf "}\n"

(* Lookup helper: given a runtime plan, return the matching specialized
 * executor or NULL. Caller falls back to the generic executor on NULL.
 *
 * For the spike there are a few hard-coded entries; the lookup is a
 * linear scan that returns the first match. Future versions (post-spike
 * with wisdom-parse) will use a hash table indexed by (N, K, factors). *)
let emit_lookup (entries : plan_entry list) ~isa =
  Printf.printf "\n";
  Printf.printf "/* Lookup: returns the specialized executor for this plan or NULL.\n";
  Printf.printf " * Caller is expected to fall back to _stride_execute_fwd_slice_from\n";
  Printf.printf " * when this returns NULL (cold cell, not in wisdom). */\n";
  Printf.printf "typedef void (*vfft_proto_exec_fn)(const stride_plan_t *, double *, double *,\n";
  Printf.printf "                                   size_t, size_t, int);\n\n";
  Printf.printf "static inline vfft_proto_exec_fn vfft_proto_lookup_fwd_%s(const stride_plan_t *plan)\n" isa;
  Printf.printf "{\n";
  List.iter (fun (e : plan_entry) ->
    Printf.printf "    /* Entry: N=%d K=%d factors=%s */\n" e.n e.k
      (String.concat "," (List.map string_of_int (Array.to_list e.factors)));
    Printf.printf "    if (plan->N == %d && plan->K == %d && plan->num_stages == %d\n"
      e.n e.k (Array.length e.factors);
    Printf.printf "        && plan->use_dif_forward == %d"
      (if e.use_dif_forward then 1 else 0);
    Array.iteri (fun s r ->
      Printf.printf "\n        && plan->factors[%d] == %d" s r
    ) e.factors;
    Printf.printf ")\n";
    Printf.printf "        return %s;\n\n" (executor_name e `Fwd ~isa)
  ) entries;
  Printf.printf "    return NULL;  /* cold cell — caller falls back to generic */\n";
  Printf.printf "}\n"

(* ── Header file shape ── *)

let emit_header_file ~isa =
  print_endline "/* plan_executors.h — auto-generated by bin/emit_executor_h.ml.";
  print_endline " * DO NOT EDIT BY HAND.";
  print_endline " *";
  print_endline " * Plan-shaped specialized executors. Each function below is a";
  print_endline " * drop-in replacement for _stride_execute_fwd_slice_from for a";
  print_endline " * specific (N, K, factorization, variant-assignment) tuple. Compared";
  print_endline " * to the generic executor:";
  print_endline " *   - Codelet calls direct (vs function-pointer indirection)";
  print_endline " *   - 4-branch per-group variant tree collapsed to one codepath";
  print_endline " *   - needs_tw[g] branch retained (per-group runtime decision)";
  print_endline " *";
  print_endline " * For cold cells (no specialization), caller falls back to the";
  print_endline " * generic executor. See vfft_proto_lookup_fwd_<isa>() at the bottom. */";
  print_endline "#ifndef VFFT_PROTO_PLAN_EXECUTORS_H";
  print_endline "#define VFFT_PROTO_PLAN_EXECUTORS_H";
  print_endline "";
  print_endline "#include <stddef.h>";
  print_endline "#include <stdlib.h>  /* abort() for unimplemented variants */";
  print_endline "";
  print_endline "/* ── Minimal plan/stage types (standalone, no production dep) ──";
  print_endline " *";
  print_endline " * Field names + order match the prefix of production's stride_stage_t /";
  print_endline " * stride_plan_t (src/core/executor.h) so the spike executor body that";
  print_endline " * accesses these fields will compile cleanly against either struct.";
  print_endline " * When wiring into production later, define VFFT_PROTO_USE_PRODUCTION_PLAN_T";
  print_endline " * before including this header to skip our local definitions. */";
  print_endline "#ifndef VFFT_PROTO_USE_PRODUCTION_PLAN_T";
  print_endline "";
  print_endline "#define STRIDE_MAX_STAGES 16";
  print_endline "";
  print_endline "/* Function pointer types for codelet slots. Used by baseline (function-";
  print_endline " * pointer) executor paths; the spike executor calls by symbol directly. */";
  print_endline "typedef void (*vfft_proto_codelet_fn)(double *, double *,";
  print_endline "                                      const double *, const double *,";
  print_endline "                                      size_t, size_t);";
  print_endline "";
  print_endline "/* Pre-walked invocation tape entry (Plan B+A). One entry per (stage,";
  print_endline " * group). Populated at plan-build time so the executor's inner loop is";
  print_endline " * a tight sequential walk — no per-group branches, no scattered loads";
  print_endline " * across multiple per-group arrays. 24 bytes = friendly for HW prefetch. */";
  print_endline "typedef struct {";
  print_endline "    size_t        base;    /* offset into re/im (doubles) */";
  print_endline "    const double *tw_re;   /* per-group scalar twiddle, NULL for n1 stages */";
  print_endline "    const double *tw_im;";
  print_endline "} stride_invocation_t;";
  print_endline "";
  print_endline "typedef struct {";
  print_endline "    int    radix;";
  print_endline "    size_t stride;";
  print_endline "    int    num_groups;";
  print_endline "    /* Legacy per-group arrays — used by baseline executor and as the";
  print_endline "     * source data when building the tape. Retained for compatibility";
  print_endline "     * with the apples-to-apples bench setup. */";
  print_endline "    size_t *group_base;";
  print_endline "    int    *needs_tw;";
  print_endline "    double *cf0_re;";
  print_endline "    double *cf0_im;";
  print_endline "    double **tw_scalar_re;";
  print_endline "    double **tw_scalar_im;";
  print_endline "    /* Pre-walked tape — what the (B)+(A) spike executor consumes. */";
  print_endline "    stride_invocation_t *tape;";
  print_endline "    /* Function-pointer slots for baseline (generic-style) executor.";
  print_endline "     * The spike executor ignores these and calls by direct symbol. */";
  print_endline "    vfft_proto_codelet_fn n1_fwd;";
  print_endline "    vfft_proto_codelet_fn t1s_fwd;";
  print_endline "} stride_stage_t;";
  print_endline "";
  print_endline "typedef struct {";
  print_endline "    int    N;";
  print_endline "    size_t K;";
  print_endline "    int    num_stages;";
  print_endline "    int    factors[STRIDE_MAX_STAGES];";
  print_endline "    stride_stage_t stages[STRIDE_MAX_STAGES];";
  print_endline "    int    use_dif_forward;";
  print_endline "} stride_plan_t;";
  print_endline "";
  print_endline "/* Stub for the scalar twiddle preprocessing call the executor makes when";
  print_endline " * cf0 is non-trivial. In the spike harness, cf0 is always (1.0, 0.0) and";
  print_endline " * this branch is never taken; provide a stub so plan_executors.h links. */";
  print_endline "static inline void _stride_cmul_scalar_inplace(double *re, double *im,";
  print_endline "                                                size_t n,";
  print_endline "                                                double cfr, double cfi) {";
  print_endline "    (void)re; (void)im; (void)n; (void)cfr; (void)cfi;";
  print_endline "    /* spike harness sets cf0=(1.0,0.0) so this is never called. */";
  print_endline "}";
  print_endline "";
  print_endline "#endif /* VFFT_PROTO_USE_PRODUCTION_PLAN_T */";
  print_endline "";
  (* Externs: union of all entries' codelet symbols, deduped. *)
  let all_externs =
    List.concat_map (fun e -> collect_externs e `Fwd ~isa) spike_entries
    |> List.sort_uniq compare
  in
  print_endline "/* Externs for the codelets called by the executors below. */";
  List.iter (fun sym ->
    Printf.printf
      "extern void %s(double *rio_re, double *rio_im,\n\
      \                                 const double *tw_re, const double *tw_im,\n\
      \                                 size_t ios, size_t me);\n" sym
  ) all_externs;
  print_endline "";
  List.iter (fun e -> emit_executor e `Fwd ~isa; print_endline "") spike_entries;
  emit_lookup    spike_entries ~isa;
  print_endline "";
  print_endline "#endif /* VFFT_PROTO_PLAN_EXECUTORS_H */"

let () =
  let isa = ref "avx2" in
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    let arg = Sys.argv.(!i) in
    if arg = "--isa" && !i + 1 < Array.length Sys.argv then begin
      isa := Sys.argv.(!i + 1);
      i := !i + 2
    end else begin
      Printf.eprintf "warning: unknown arg %s\n" arg;
      incr i
    end
  done;
  emit_header_file ~isa:!isa
