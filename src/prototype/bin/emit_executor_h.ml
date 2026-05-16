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
  (* Synthetic entry — same N=131072 K=4 shape as the wisdom entry above,
   * but with FLAT inner stages instead of T1S. Wisdom doesn't pick this
   * exact configuration (T1S beats FLAT at K=4 in production measurement),
   * but exercising the FLAT codepath at maximum invocation count is the
   * cleanest way to measure whether (B)+(A) helps FLAT. The FLAT path
   * has more per-group wrapper work (K-blocked tw_buf staging via
   * _stride_broadcast_2) than T1S, so the (B)+(A) gain on this synthetic
   * cell tells us whether the spike helps broadly across variants. *)
  { n = 131072; k = 4;
    factors  = [| 4; 4; 4; 4; 8; 4; 4; 4 |];
    variants = [| FLAT; FLAT; FLAT; FLAT; FLAT; FLAT; FLAT; FLAT |];
    use_dif_forward = false };
  (* Synthetic LOG3 entry — same shape again, all-LOG3 inner stages.
   * Production rarely picks LOG3 for many-inner-stage plans (twiddle-
   * bandwidth-bound innermost-stage use-case), so this is research-only.
   * LOG3 path in production applies cf0 to ALL R legs (vs T1S's leg 0
   * only) — slightly more per-group prep than T1S. (B)+(A) expected to
   * recover similar share. *)
  { n = 131072; k = 4;
    factors  = [| 4; 4; 4; 4; 8; 4; 4; 4 |];
    variants = [| FLAT; LOG3; LOG3; LOG3; LOG3; LOG3; LOG3; LOG3 |];
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
    (* Inner stages: branch per-group on tape[g].tw_re != NULL.
     *
     * Real plans (DP-built, wisdom-driven) have inner-stage groups with
     * k_prev=0 where needs_tw=0 and tw_scalar_re[g]=NULL. The earlier
     * spike harness assumed all inner groups had twiddles (its hand-
     * built plans were constructed that way); calling the t1s/t1/log3
     * codelet on a NULL twiddle pointer crashes. The needs_tw[g]==0
     * path falls back to n1 (same dispatch the generic executor uses). *)
    let n1_sym = n1_symbol ~r ~direction:d ~isa in
    Printf.printf "        for (int g = 0; g < num_groups; g++) {\n";
    Printf.printf "            const stride_invocation_t inv = tape[g];\n";
    Printf.printf "            if (inv.tw_re == NULL) {\n";
    Printf.printf "                %s(re + inv.base, im + inv.base, NULL, NULL, stride, slice_K);\n" n1_sym;
    Printf.printf "                continue;\n";
    Printf.printf "            }\n";
    (match v with
     | T1S  ->
       Printf.printf "            %s(re + inv.base, im + inv.base,\n"
         (t1_symbol ~r ~variant:T1S ~direction:d ~isa);
       Printf.printf "                                       inv.tw_re, inv.tw_im,\n";
       Printf.printf "                                       stride, slice_K);\n"
     | FLAT ->
       (* FLAT variant: K-blocked tw_buf staging via _stride_broadcast_2,
        * then call the t1 (flat-twiddle) codelet. Mirrors production's
        * else-if branch at executor.h:455-491. tw_buf size is compile-time
        * constant ((R-1) × VFFT_PROTO_TW_BLOCK_K) so it stack-allocates
        * cleanly and fits L1 for any R≤64. *)
       Printf.printf "            double *base_re = re + inv.base;\n";
       Printf.printf "            double *base_im = im + inv.base;\n";
       Printf.printf "            double tw_buf_re[%d * VFFT_PROTO_TW_BLOCK_K];\n" (r - 1);
       Printf.printf "            double tw_buf_im[%d * VFFT_PROTO_TW_BLOCK_K];\n" (r - 1);
       Printf.printf "            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {\n";
       Printf.printf "                size_t this_K = (slice_K - kb < VFFT_PROTO_TW_BLOCK_K)\n";
       Printf.printf "                                ? (slice_K - kb) : VFFT_PROTO_TW_BLOCK_K;\n";
       Printf.printf "                for (int j = 0; j < %d; j++) {\n" (r - 1);
       Printf.printf "                    _stride_broadcast_2(tw_buf_re + (size_t)j * this_K,\n";
       Printf.printf "                                        tw_buf_im + (size_t)j * this_K,\n";
       Printf.printf "                                        this_K, inv.tw_re[j], inv.tw_im[j]);\n";
       Printf.printf "                }\n";
       Printf.printf "                %s(base_re + kb, base_im + kb,\n"
         (t1_symbol ~r ~variant:FLAT ~direction:d ~isa);
       Printf.printf "                                           tw_buf_re, tw_buf_im,\n";
       Printf.printf "                                           stride, this_K);\n";
       Printf.printf "            }\n"
     | LOG3 ->
       (* LOG3 variant: in production this path also applies cf0 to ALL
        * R legs (not just leg 0 like T1S — see executor.h:420-437). The
        * spike bench uses cf0=(1.0,0.0) so that branch is skipped; the
        * emitted path collapses to one direct call per group, like T1S
        * but to a different codelet symbol. Real-plan integration needs
        * the cf branch re-added (with cf0 either pre-walked into the
        * tape or read from st->cf0_re[g] like the baseline). *)
       Printf.printf "            %s(re + inv.base, im + inv.base,\n"
         (t1_symbol ~r ~variant:LOG3 ~direction:d ~isa);
       Printf.printf "                                            inv.tw_re, inv.tw_im,\n";
       Printf.printf "                                            stride, slice_K);\n"
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
  print_endline " * pointer) executor paths; the spike executor calls by symbol directly.";
  print_endline " *";
  print_endline " *   vfft_proto_codelet_fn — t1/t1s/log3, 6-arg in-place (matches";
  print_endline " *                            production's stride_t1_fn).";
  print_endline " *   vfft_proto_n1_fn      — n1, 7-arg OOP (matches production's";
  print_endline " *                            stride_n1_fn). The registry wraps the";
  print_endline " *                            6-arg in-place codelet to expose this";
  print_endline " *                            signature; see registry_<isa>.h. */";
  print_endline "typedef void (*vfft_proto_codelet_fn)(double *, double *,";
  print_endline "                                      const double *, const double *,";
  print_endline "                                      size_t, size_t);";
  print_endline "typedef void (*vfft_proto_n1_fn)(const double *, const double *,";
  print_endline "                                  double *, double *,";
  print_endline "                                  size_t, size_t, size_t);";
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
  print_endline "    /* Per-group full K-replicated twiddle tables. FLAT/LOG3 codelets";
  print_endline "     * read these. Layout: grp_tw_re[g][(j-1)*K + k] for j=1..R-1.";
  print_endline "     *   FLAT  : combined cf × per_leg (cf already baked in)";
  print_endline "     *   LOG3  : raw per_leg (executor applies cf to ALL legs first) */";
  print_endline "    double **grp_tw_re;";
  print_endline "    double **grp_tw_im;";
  print_endline "    /* n1_fallback path: full per-element twiddle for all R legs.";
  print_endline "     * cf_all[g*R*K + j*K + k]. Used at R=64 large-K when (cmul+n1) beats t1. */";
  print_endline "    double *cf_all_re;";
  print_endline "    double *cf_all_im;";
  print_endline "    int    use_n1_fallback;  /* 1 = use cf_all + n1 instead of t1 */";
  print_endline "    int    use_log3;         /* 1 = grp_tw is raw per_leg, apply cf to ALL legs */";
  print_endline "    /* Pre-walked tape — what the (B)+(A) spike executor consumes. */";
  print_endline "    stride_invocation_t *tape;";
  print_endline "    /* Function-pointer slots for baseline (generic-style) executor.";
  print_endline "     * The spike executor ignores these and calls by direct symbol. */";
  print_endline "    vfft_proto_n1_fn      n1_fwd;   /* 7-arg OOP (in==out for in-place) */";
  print_endline "    vfft_proto_n1_fn      n1_bwd;   /* 7-arg OOP, inverse butterfly for bwd */";
  print_endline "    vfft_proto_codelet_fn t1_fwd;   /* FLAT or LOG3 codelet (per use_log3) */";
  print_endline "    vfft_proto_codelet_fn t1s_fwd;  /* T1S scalar-broadcast codelet */";
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
  print_endline "#include <immintrin.h>";
  print_endline "";
  print_endline "/* Scalar cmul: (re + i*im) *= (cfr + i*cfi), in-place over n lanes.";
  print_endline " * Mirrors production's _stride_cmul_scalar_avx2 in src/core/executor.h. */";
  print_endline "static inline void _stride_cmul_scalar_inplace(double *re, double *im,";
  print_endline "                                                size_t n,";
  print_endline "                                                double cfr, double cfi) {";
  print_endline "    __m256d vcfr = _mm256_set1_pd(cfr);";
  print_endline "    __m256d vcfi = _mm256_set1_pd(cfi);";
  print_endline "    size_t k = 0;";
  print_endline "    for (; k + 4 <= n; k += 4) {";
  print_endline "        __m256d vr = _mm256_loadu_pd(re + k);";
  print_endline "        __m256d vi = _mm256_loadu_pd(im + k);";
  print_endline "        __m256d nr = _mm256_fmsub_pd(vr, vcfr, _mm256_mul_pd(vi, vcfi));";
  print_endline "        __m256d ni = _mm256_fmadd_pd(vr, vcfi, _mm256_mul_pd(vi, vcfr));";
  print_endline "        _mm256_storeu_pd(re + k, nr);";
  print_endline "        _mm256_storeu_pd(im + k, ni);";
  print_endline "    }";
  print_endline "    for (; k < n; k++) {";
  print_endline "        double tr = re[k];";
  print_endline "        re[k] = tr * cfr - im[k] * cfi;";
  print_endline "        im[k] = tr * cfi + im[k] * cfr;";
  print_endline "    }";
  print_endline "}";
  print_endline "";
  print_endline "/* SIMD broadcast helper for the FLAT variant's K-blocked tw_buf staging.";
  print_endline " * Mirrors production's _stride_broadcast_avx2 in src/core/executor.h. */";
  print_endline "static inline void _stride_broadcast_2(double *out_re, double *out_im, size_t n,";
  print_endline "                                       double s_re, double s_im) {";
  print_endline "    __m256d vr = _mm256_set1_pd(s_re);";
  print_endline "    __m256d vi = _mm256_set1_pd(s_im);";
  print_endline "    size_t i = 0;";
  print_endline "    for (; i + 4 <= n; i += 4) {";
  print_endline "        _mm256_storeu_pd(out_re + i, vr);";
  print_endline "        _mm256_storeu_pd(out_im + i, vi);";
  print_endline "    }";
  print_endline "    for (; i < n; i++) {";
  print_endline "        out_re[i] = s_re; out_im[i] = s_im;";
  print_endline "    }";
  print_endline "}";
  print_endline "";
  print_endline "/* K-block size for FLAT variant tw_buf staging. Mirrors production's";
  print_endline " * STRIDE_TW_BLOCK_K=64. Keeps the staging buffer in L1 for any R≤64. */";
  print_endline "#ifndef VFFT_PROTO_TW_BLOCK_K";
  print_endline "#define VFFT_PROTO_TW_BLOCK_K 64";
  print_endline "#endif";
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
