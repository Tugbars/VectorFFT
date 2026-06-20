(* emit_executor_h.ml — emit plan-shaped specialized executors.
 *
 * Reads a wisdom file (production format, see vfft_wisdom_tuned.txt) and
 * emits one specialized C function per entry. Each emitted function is a
 * drop-in replacement for the generic executor when the runtime plan
 * matches the (N, K, factors, variants, orient, dir) tuple of the entry.
 *
 * Per-stage bodies are emitted as macro invocations
 * (VFFT_PROTO_STAGE_OUTER/T1S/LOG3/FLAT) — see the macro definitions
 * emitted at the top of plan_executors.h. The macros token-paste into
 * the same code the inline form expanded to; this is purely a source-
 * size optimization.
 *
 * Usage:
 *   dune exec bin/emit_executor_h.exe -- \
 *     --isa avx2 \
 *     --wisdom generated/spike_wisdom.txt \
 *   > generated/plan_executors.h
 *
 * Wisdom file format (production-compatible):
 *   N K nf factors... best_ns use_blocked split_stage block_groups use_dif_forward variant_codes...
 *   variant codes: 0=FLAT 1=LOG3 2=T1S 3=BUF
 * Lines starting with '@' or '#', and empty lines, are skipped.
 * best_ns / use_blocked / split_stage / block_groups are read but ignored
 * by this emitter — they're production-side fields. *)

type variant = FLAT | LOG3 | T1S | BUF [@@warning "-37"]

let variant_of_int = function
  | 0 -> FLAT | 1 -> LOG3 | 2 -> T1S | 3 -> BUF
  | n -> failwith (Printf.sprintf "unknown variant code %d" n)

let variant_code = function FLAT -> 0 | LOG3 -> 1 | T1S -> 2 | BUF -> 3

let variant_name = function
  | FLAT -> "FLAT" | LOG3 -> "LOG3" | T1S -> "T1S" | BUF -> "BUF"

type plan_entry = {
  n               : int;
  k               : int;
  factors         : int array;
  variants        : variant array;
  use_dif_forward : bool;
}

(* ── Wisdom-file parser ────────────────────────────────────────────── *)

let is_blank_or_comment line =
  let s = String.trim line in
  s = "" || (s.[0] = '#') || (s.[0] = '@')

let split_ws s =
  String.split_on_char ' ' s
  |> List.concat_map (fun t -> String.split_on_char '\t' t)
  |> List.filter (fun t -> t <> "")

let parse_wisdom_line (lineno : int) (line : string) : plan_entry option =
  if is_blank_or_comment line then None
  else begin
    let toks = Array.of_list (split_ws (String.trim line)) in
    let need k ctx =
      if Array.length toks < k then
        failwith (Printf.sprintf "wisdom line %d: %s (have %d tokens): %s"
                    lineno ctx (Array.length toks) line)
    in
    need 3 "need at least N K nf";
    let n  = int_of_string toks.(0) in
    let k  = int_of_string toks.(1) in
    let nf = int_of_string toks.(2) in
    let total_expected = 3 + nf + 5 + nf in
    need total_expected
      (Printf.sprintf "nf=%d needs %d tokens" nf total_expected);
    let factors = Array.init nf (fun i -> int_of_string toks.(3 + i)) in
    let use_dif_forward = int_of_string toks.(3 + nf + 4) <> 0 in
    let variants = Array.init nf (fun i ->
      variant_of_int (int_of_string toks.(3 + nf + 5 + i))) in
    Some { n; k; factors; variants; use_dif_forward }
  end

let read_wisdom_file (path : string) : plan_entry list =
  let ic = open_in path in
  let entries = ref [] in
  let lineno = ref 0 in
  (try
    while true do
      incr lineno;
      let line = input_line ic in
      match parse_wisdom_line !lineno line with
      | Some e -> entries := e :: !entries
      | None -> ()
    done
  with End_of_file -> ());
  close_in ic;
  List.rev !entries

(* Deduplicate entries by (N, K, factors, variants, use_dif_forward).
 * Wisdom files can carry the same tuple multiple times (re-bench rows);
 * the C compiler will reject duplicate static-fn names. First wins. *)
let dedup_entries (entries : plan_entry list) : plan_entry list =
  let seen = Hashtbl.create 32 in
  List.filter (fun e ->
    let key = (e.n, e.k,
               Array.to_list e.factors,
               List.map variant_code (Array.to_list e.variants),
               e.use_dif_forward) in
    if Hashtbl.mem seen key then false
    else begin Hashtbl.add seen key (); true end
  ) entries

(* ── Direction / symbol naming ─────────────────────────────────────── *)

type direction = [ `Fwd | `Bwd ]
let dir_str : direction -> string = function `Fwd -> "fwd" | `Bwd -> "bwd"

let n1_symbol ~r ~direction ~isa =
  Printf.sprintf "radix%d_n1_%s_%s" r (dir_str direction) isa

(* orient: `Dit or `Dif (codelet family). The codelet name encodes orientation
 * in the dit/dif infix; variant selects which codelet within the family. *)
let t1_symbol ~r ~variant ~direction ?(orient=`Dit) ~isa () =
  let o = match orient with `Dit -> "dit" | `Dif -> "dif" in
  match variant, orient with
  | FLAT, _ -> Printf.sprintf "radix%d_t1_%s_%s_%s"      r o (dir_str direction) isa
  | LOG3, _ -> Printf.sprintf "radix%d_t1_%s_log3_%s_%s" r o (dir_str direction) isa
  | T1S, `Dit -> Printf.sprintf "radix%d_t1s_dit_%s_%s" r (dir_str direction) isa
  | T1S, `Dif -> failwith "T1S not supported in DIF (planner falls back to FLAT)"
  | BUF, _  -> failwith "BUF variant unsupported"

(* Executor function name. For forward, encodes the full (factors, variants)
 * tuple. For backward, only (N, K, factors) — backward uses only n1 codelets
 * (which are variant-independent), so multiple variant rows of the same
 * factor list share one backward executor. *)
let executor_name (e : plan_entry) (d : direction) ~isa =
  let join_int xs = String.concat "" (List.map string_of_int (Array.to_list xs)) in
  let join_var vs = String.concat ""
                      (List.map (fun v -> string_of_int (variant_code v))
                                (Array.to_list vs)) in
  let orient = if e.use_dif_forward then "dif" else "dit" in
  match d with
  | `Fwd ->
    Printf.sprintf "exec_n%d_k%d_%s_v%s_%s_%s_%s"
      e.n e.k (join_int e.factors) (join_var e.variants) orient
      (dir_str d) isa
  | `Bwd ->
    Printf.sprintf "exec_n%d_k%d_%s_%s_%s_%s"
      e.n e.k (join_int e.factors) orient (dir_str d) isa

(* Collect codelet externs the executor will call. Picks DIT or DIF
 * codelet symbols based on use_dif_forward. *)
let collect_externs (e : plan_entry) (d : direction) ~isa : string list =
  let acc = ref [] in
  let orient = if e.use_dif_forward then `Dif else `Dit in
  let nf = Array.length e.factors in
  match d with
  | `Bwd ->
    (* DIT bwd: n1_bwd + t1s_dit_bwd (fused).
     * DIF bwd: n1_bwd + t1_dif_bwd (fused). *)
    Array.iter (fun r ->
      acc := n1_symbol ~r ~direction:`Bwd ~isa :: !acc;
      let bwd_variant = if e.use_dif_forward then FLAT else T1S in
      acc := t1_symbol ~r ~variant:bwd_variant ~direction:`Bwd ~orient ~isa () :: !acc
    ) e.factors;
    List.sort_uniq compare !acc
  | `Fwd ->
    Array.iteri (fun s v ->
      let r = e.factors.(s) in
      let is_outer =
        if e.use_dif_forward then s = nf - 1 else s = 0
      in
      if is_outer then
        acc := n1_symbol ~r ~direction:`Fwd ~isa :: !acc
      else begin
        (* DIF folds T1S → FLAT (DIF has no T1S codelet). *)
        let effective_v =
          if e.use_dif_forward && v = T1S then FLAT else v
        in
        acc := t1_symbol ~r ~variant:effective_v ~direction:`Fwd ~orient ~isa () :: !acc;
        acc := n1_symbol ~r ~direction:`Fwd ~isa :: !acc
      end
    ) e.variants;
    List.sort_uniq compare !acc

(* Dedupe entries for backward emission: same (N, K, factors, use_dif_forward)
 * collapses to a single backward executor. *)
let dedup_for_bwd (entries : plan_entry list) : plan_entry list =
  let seen = Hashtbl.create 32 in
  List.filter (fun e ->
    let key = (e.n, e.k, Array.to_list e.factors, e.use_dif_forward) in
    if Hashtbl.mem seen key then false
    else begin Hashtbl.add seen key (); true end
  ) entries

(* ── C emission ────────────────────────────────────────────────────── *)

(* Per-stage forward emission. Picks DIT or DIF macro family based on
 * use_dif_forward, picks variant macro within the family. For DIF, T1S
 * falls back to FLAT (production parity — DIF has no T1S codelet). *)
let emit_stage_fwd (e : plan_entry) (stage_idx : int) ~isa =
  let r = e.factors.(stage_idx) in
  let v = e.variants.(stage_idx) in
  let nf = Array.length e.factors in
  let macro =
    if e.use_dif_forward then begin
      (* DIF: no-twiddle stage is LAST (s = nf-1). *)
      if stage_idx = nf - 1 then "VFFT_PROTO_STAGE_DIF_OUTER"
      else match v with
        | LOG3 -> "VFFT_PROTO_STAGE_DIF_LOG3 "
        | FLAT | T1S | BUF -> "VFFT_PROTO_STAGE_DIF_FLAT "
    end else begin
      (* DIT: no-twiddle stage is FIRST (s = 0). *)
      if stage_idx = 0 then "VFFT_PROTO_STAGE_OUTER"
      else match v with
        | T1S  -> "VFFT_PROTO_STAGE_T1S  "
        | LOG3 -> "VFFT_PROTO_STAGE_LOG3 "
        | FLAT -> "VFFT_PROTO_STAGE_FLAT "
        | BUF  -> failwith "BUF variant unsupported"
    end
  in
  Printf.printf "    %s(%d, %2d, %s)\n" macro stage_idx r isa

(* Per-stage backward emission. DIT uses VFFT_PROTO_STAGE_BWD (t1s_dit_bwd
 * + executor-side leg-0 cf cmul). DIF uses VFFT_PROTO_STAGE_DIF_BWD (fused
 * t1_dif_bwd, no leg-0 cmul since cf0=1 in DIF). Both handle the needs_tw=0
 * fallback (n1_bwd) inside the macro, so all stages emit the same macro. *)
let emit_stage_bwd (e : plan_entry) (stage_idx : int) ~isa =
  let r = e.factors.(stage_idx) in
  let macro =
    if e.use_dif_forward then "VFFT_PROTO_STAGE_DIF_BWD"
    else "VFFT_PROTO_STAGE_BWD"
  in
  Printf.printf "    %s(%d, %2d, %s)\n" macro stage_idx r isa

let emit_executor (e : plan_entry) (d : direction) ~isa =
  let fn_name = executor_name e d ~isa in
  let factors_str  = String.concat "," (List.map string_of_int (Array.to_list e.factors)) in
  let orient = if e.use_dif_forward then "DIF" else "DIT" in
  Printf.printf "/* Plan-shaped executor specialization\n";
  Printf.printf " *   N=%d K=%d\n" e.n e.k;
  Printf.printf " *   factors=%s\n" factors_str;
  (match d with
   | `Fwd ->
     let variants_str = String.concat "," (List.map variant_name (Array.to_list e.variants)) in
     Printf.printf " *   variants=%s\n" variants_str
   | `Bwd ->
     Printf.printf " *   (backward: variant-independent — uses n1_bwd only)\n");
  Printf.printf " *   orient=%s dir=%s isa=%s */\n" orient (String.uppercase_ascii (dir_str d)) isa;
  Printf.printf "static void %s(const stride_plan_t *plan,\n" fn_name;
  let pad = String.make (String.length fn_name + 12) ' ' in
  Printf.printf "%sdouble *re, double *im,\n" pad;
  Printf.printf "%ssize_t slice_K, size_t full_K,\n" pad;
  Printf.printf "%sint start_stage)\n" pad;
  Printf.printf "{\n";
  (match d with
   | `Fwd ->
     Printf.printf "    (void)full_K;\n";
     for s = 0 to Array.length e.factors - 1 do
       emit_stage_fwd e s ~isa
     done
   | `Bwd ->
     (* Backward walks stages in REVERSE order. full_K used for cf_all indexing. *)
     for s = Array.length e.factors - 1 downto 0 do
       emit_stage_bwd e s ~isa
     done);
  Printf.printf "}\n"

(* ── Lookup ────────────────────────────────────────────────────────── *)

(* Order entries so MORE-SPECIFIC variant tuples match before less-specific
 * ones at the same (N, K, factors). All-T1S inner is the most permissive
 * pattern (matched via _MATCH_T1S_INNER), so we put it last among siblings.
 *
 * Sort key: (N, K, factors) ascending, then variants with all-T1S last. *)
let sort_for_lookup (entries : plan_entry list) : plan_entry list =
  let all_inner_t1s e =
    let ok = ref true in
    Array.iteri (fun i v ->
      if i >= 1 && v <> T1S then ok := false
    ) e.variants;
    !ok
  in
  let cmp a b =
    let cN = compare a.n b.n in if cN <> 0 then cN
    else let cK = compare a.k b.k in if cK <> 0 then cK
    else let cF = compare (Array.to_list a.factors) (Array.to_list b.factors) in
    if cF <> 0 then cF
    else compare (all_inner_t1s a) (all_inner_t1s b) (* false < true → mixed first *)
  in
  List.stable_sort cmp entries

(* Emit the per-entry match condition (variant-aware).
 *
 * For DIF entries, the planner folds T1S → FLAT (no T1S codelet in DIF),
 * so the lookup variant check for T1S stages in a DIF entry must match
 * the FLAT pattern. _MATCH_T1S_INNER never applies in DIF mode.
 *
 * For DIF, the "outer" (no-twiddle) stage is the LAST (nf-1), not stage 0,
 * so variant checks on the outer stage are skipped accordingly. *)
let emit_match_body (e : plan_entry) =
  let nf = Array.length e.factors in
  (* For DIF, T1S → FLAT folding; also _MATCH_T1S_INNER is DIT-only. *)
  let all_inner_t1s =
    if e.use_dif_forward then false
    else begin
      let ok = ref true in
      Array.iteri (fun i v -> if i >= 1 && v <> T1S then ok := false) e.variants;
      !ok
    end
  in
  Printf.printf "    if (plan->N == %d && plan->K == %d && plan->num_stages == %d\n"
    e.n e.k nf;
  Printf.printf "        && plan->use_dif_forward == %d"
    (if e.use_dif_forward then 1 else 0);
  Array.iteri (fun s r ->
    Printf.printf "\n        && plan->factors[%d] == %d" s r
  ) e.factors;
  if all_inner_t1s then
    Printf.printf "\n        && _MATCH_T1S_INNER(plan, %d)" nf
  else begin
    let outer_stage_idx = if e.use_dif_forward then nf - 1 else 0 in
    Array.iteri (fun s v ->
      if s = outer_stage_idx then ()
      else begin
        (* DIF: T1S falls back to FLAT in planner — check the FLAT pattern. *)
        let effective_v =
          if e.use_dif_forward && v = T1S then FLAT else v
        in
        match effective_v with
        | T1S  -> Printf.printf "\n        && plan->stages[%d].t1s_fwd != NULL" s
        | LOG3 -> Printf.printf "\n        && plan->stages[%d].t1_fwd  != NULL && plan->stages[%d].use_log3" s s
        | FLAT -> Printf.printf "\n        && plan->stages[%d].t1_fwd  != NULL && !plan->stages[%d].use_log3" s s
        | BUF  -> failwith "BUF unsupported"
      end
    ) e.variants
  end;
  Printf.printf ")\n"

let emit_lookup_fwd (entries : plan_entry list) ~isa =
  let entries = sort_for_lookup entries in
  Printf.printf "\n";
  Printf.printf "/* Forward lookup: returns the specialized executor for this plan or NULL.\n";
  Printf.printf " * Caller falls back to the generic executor when this returns NULL. */\n";
  Printf.printf "static inline vfft_proto_exec_fn vfft_proto_lookup_fwd_%s(const stride_plan_t *plan)\n" isa;
  Printf.printf "{\n";
  Printf.printf "    /* Variant-check helper: a stage is T1S iff t1s_fwd != NULL. */\n";
  Printf.printf "    #define _MATCH_T1S_INNER(plan, nstages) ({ int _ok = 1; \\\n";
  Printf.printf "        for (int _s = 1; _s < (nstages); _s++) \\\n";
  Printf.printf "            if ((plan)->stages[_s].t1s_fwd == NULL) { _ok = 0; break; } \\\n";
  Printf.printf "        _ok; })\n\n";
  List.iter (fun (e : plan_entry) ->
    let factors_str = String.concat "," (List.map string_of_int (Array.to_list e.factors)) in
    let variants_str = String.concat "" (List.map (fun v -> string_of_int (variant_code v))
                                             (Array.to_list e.variants)) in
    Printf.printf "    /* Entry: N=%d K=%d factors=%s variants=v%s */\n"
      e.n e.k factors_str variants_str;
    emit_match_body e;
    Printf.printf "        return %s;\n\n" (executor_name e `Fwd ~isa)
  ) entries;
  Printf.printf "    #undef _MATCH_T1S_INNER\n\n";
  Printf.printf "    return NULL;\n";
  Printf.printf "}\n"

(* Backward lookup matches by (N, K, factors, use_dif_forward) only.
 * Variant doesn't affect bwd (no codepath difference) so we dedupe. *)
let emit_lookup_bwd (entries : plan_entry list) ~isa =
  let entries = dedup_for_bwd entries in
  Printf.printf "\n";
  Printf.printf "/* Backward lookup: matches by factors only (variant-independent). */\n";
  Printf.printf "static inline vfft_proto_exec_fn vfft_proto_lookup_bwd_%s(const stride_plan_t *plan)\n" isa;
  Printf.printf "{\n";
  List.iter (fun (e : plan_entry) ->
    let nf = Array.length e.factors in
    let factors_str = String.concat "," (List.map string_of_int (Array.to_list e.factors)) in
    Printf.printf "    /* Entry: N=%d K=%d factors=%s */\n" e.n e.k factors_str;
    Printf.printf "    if (plan->N == %d && plan->K == %d && plan->num_stages == %d\n"
      e.n e.k nf;
    Printf.printf "        && plan->use_dif_forward == %d"
      (if e.use_dif_forward then 1 else 0);
    Array.iteri (fun s r ->
      Printf.printf "\n        && plan->factors[%d] == %d" s r
    ) e.factors;
    Printf.printf ")\n";
    Printf.printf "        return %s;\n\n" (executor_name e `Bwd ~isa)
  ) entries;
  Printf.printf "    return NULL;\n";
  Printf.printf "}\n"

(* ── Per-stage macro definitions ───────────────────────────────────── *)

let emit_stage_macros () =
  print_endline "/* ───────────────────────────────────────────────────────────────────";
  print_endline " * Stage macros: token-paste expansion of the per-stage body. Generated";
  print_endline " * code is identical to the hand-unrolled form (compiler sees literal";
  print_endline " * `radix##R##_t1s_dit_fwd_##ISA` after substitution), but the source is";
  print_endline " * ~10× smaller.";
  print_endline " *";
  print_endline " *   VFFT_PROTO_STAGE_OUTER(S, R, ISA) — stage 0; no twiddle; n1 codelet";
  print_endline " *   VFFT_PROTO_STAGE_T1S  (S, R, ISA) — T1S with per-group needs_tw branch";
  print_endline " *   VFFT_PROTO_STAGE_LOG3 (S, R, ISA) — LOG3 with per-group needs_tw branch";
  print_endline " *                                       (branch is essentially never taken";
  print_endline " *                                        for stages ≥ 1; LOG3 plan-build";
  print_endline " *                                        bakes cf into all legs so even";
  print_endline " *                                        g=0 has tw_re populated)";
  print_endline " *   VFFT_PROTO_STAGE_FLAT (S, R, ISA) — FLAT K-blocked tw_buf staging";
  print_endline " * ─────────────────────────────────────────────────────────────────── */";
  print_endline "";
  print_endline "#define VFFT_PROTO_STAGE_OUTER(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const stride_invocation_t * __restrict__ tape = st->tape; \\";
  print_endline "        const int    num_groups = st->num_groups; \\";
  print_endline "        const size_t stride     = st->stride; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            size_t base = tape[g].base; \\";
  print_endline "            radix##R##_n1_fwd_##ISA(re + base, im + base, NULL, NULL, stride, slice_K); \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "#define VFFT_PROTO_STAGE_T1S(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const stride_invocation_t * __restrict__ tape = st->tape; \\";
  print_endline "        const int    num_groups = st->num_groups; \\";
  print_endline "        const size_t stride     = st->stride; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            const stride_invocation_t inv = tape[g]; \\";
  print_endline "            if (inv.tw_re) { \\";
  print_endline "                /* leg-0 common factor (cf0): codelet only touches legs 1..R-1. */ \\";
  print_endline "                const double _cfr = st->cf0_re[g]; \\";
  print_endline "                const double _cfi = st->cf0_im[g]; \\";
  print_endline "                if (_cfr != 1.0 || _cfi != 0.0) \\";
  print_endline "                    _stride_cmul_scalar_inplace(re + inv.base, im + inv.base, \\";
  print_endline "                                                slice_K, _cfr, _cfi); \\";
  print_endline "                radix##R##_t1s_dit_fwd_##ISA(re + inv.base, im + inv.base, \\";
  print_endline "                                              inv.tw_re, inv.tw_im, \\";
  print_endline "                                              stride, slice_K); \\";
  print_endline "            } else { \\";
  print_endline "                radix##R##_n1_fwd_##ISA(re + inv.base, im + inv.base, \\";
  print_endline "                                         NULL, NULL, \\";
  print_endline "                                         stride, slice_K); \\";
  print_endline "            } \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "#define VFFT_PROTO_STAGE_LOG3(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const stride_invocation_t * __restrict__ tape = st->tape; \\";
  print_endline "        const int    num_groups = st->num_groups; \\";
  print_endline "        const size_t stride     = st->stride; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            const stride_invocation_t inv = tape[g]; \\";
  print_endline "            if (inv.tw_re) { \\";
  print_endline "                /* LOG3 applies cf0 to ALL R legs (codelet reads raw per-leg). \\";
  print_endline "                 * Twiddles come from grp_tw (per-leg LOG3 format), NOT the tape's \\";
  print_endline "                 * scalar inv.tw — same as the generic LOG3 path and DIF LOG3. */ \\";
  print_endline "                const double _cfr = st->cf0_re[g]; \\";
  print_endline "                const double _cfi = st->cf0_im[g]; \\";
  print_endline "                if (_cfr != 1.0 || _cfi != 0.0) \\";
  print_endline "                    for (int _j = 0; _j < (R); _j++) \\";
  print_endline "                        _stride_cmul_scalar_inplace(re + inv.base + (size_t)_j * stride, \\";
  print_endline "                                                    im + inv.base + (size_t)_j * stride, \\";
  print_endline "                                                    slice_K, _cfr, _cfi); \\";
  print_endline "                radix##R##_t1_dit_log3_fwd_##ISA(re + inv.base, im + inv.base, \\";
  print_endline "                                                  st->grp_tw_re[g], st->grp_tw_im[g], \\";
  print_endline "                                                  stride, slice_K); \\";
  print_endline "            } else { \\";
  print_endline "                radix##R##_n1_fwd_##ISA(re + inv.base, im + inv.base, \\";
  print_endline "                                         NULL, NULL, \\";
  print_endline "                                         stride, slice_K); \\";
  print_endline "            } \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "#define VFFT_PROTO_STAGE_FLAT(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const stride_invocation_t * __restrict__ tape = st->tape; \\";
  print_endline "        const int    num_groups = st->num_groups; \\";
  print_endline "        const size_t stride     = st->stride; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            const stride_invocation_t inv = tape[g]; \\";
  print_endline "            double *base_re = re + inv.base; \\";
  print_endline "            double *base_im = im + inv.base; \\";
  print_endline "            if (!inv.tw_re) { \\";
  print_endline "                /* needs_tw=0 group (k_prev=0 in DIT): no twiddle, use n1. */ \\";
  print_endline "                radix##R##_n1_fwd_##ISA(base_re, base_im, NULL, NULL, \\";
  print_endline "                                         stride, slice_K); \\";
  print_endline "                continue; \\";
  print_endline "            } \\";
  print_endline "            /* leg-0 common factor (cf0), same as T1S; FLAT stages legs 1..R-1. */ \\";
  print_endline "            const double _cfr = st->cf0_re[g]; \\";
  print_endline "            const double _cfi = st->cf0_im[g]; \\";
  print_endline "            if (_cfr != 1.0 || _cfi != 0.0) \\";
  print_endline "                _stride_cmul_scalar_inplace(base_re, base_im, slice_K, _cfr, _cfi); \\";
  print_endline "            double tw_buf_re[((R)-1) * VFFT_PROTO_TW_BLOCK_K]; \\";
  print_endline "            double tw_buf_im[((R)-1) * VFFT_PROTO_TW_BLOCK_K]; \\";
  print_endline "            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) { \\";
  print_endline "                size_t this_K = (slice_K - kb < VFFT_PROTO_TW_BLOCK_K) \\";
  print_endline "                                ? (slice_K - kb) : VFFT_PROTO_TW_BLOCK_K; \\";
  print_endline "                for (int j = 0; j < ((R)-1); j++) { \\";
  print_endline "                    _stride_broadcast_2(tw_buf_re + (size_t)j * this_K, \\";
  print_endline "                                        tw_buf_im + (size_t)j * this_K, \\";
  print_endline "                                        this_K, inv.tw_re[j], inv.tw_im[j]); \\";
  print_endline "                } \\";
  print_endline "                radix##R##_t1_dit_fwd_##ISA(base_re + kb, base_im + kb, \\";
  print_endline "                                              tw_buf_re, tw_buf_im, \\";
  print_endline "                                              stride, this_K); \\";
  print_endline "            } \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "/* ───────────────────────────────────────────────────────────────────";
  print_endline " * Backward stage macro. Structurally simpler than forward: only n1_bwd";
  print_endline " * (variant doesn't affect backward — twiddles are post-multiplied after";
  print_endline " * the butterfly via cf_all). Reads from legacy per-group arrays";
  print_endline " * (st->group_base, st->needs_tw, st->cf_all_re/im) — not the tape.";
  print_endline " *";
  print_endline " * Macro body assumes `full_K` is in scope (declared in the executor's";
  print_endline " * function signature). Uses st->stride hoisted once per stage.";
  print_endline " * ─────────────────────────────────────────────────────────────────── */";
  print_endline "";
  print_endline "/* Broadcast scalar conj-mul: leg[0] *= conj(cf0) using SIMD.";
  print_endline " * Used by the fused bwd path to apply the leg-0 common-factor that";
  print_endline " * the t1s_bwd codelet doesn't handle (codelet only touches legs 1..R-1).";
  print_endline " * Two ISA variants emitted; macro token-pastes ISA suffix at use site. */";
  print_endline "#define VFFT_PROTO_BWD_LEG0_CONJ_avx2(re_p, im_p, cfr, cfi, n) \\";
  print_endline "    do { \\";
  print_endline "        if ((cfr) != 1.0 || (cfi) != 0.0) { \\";
  print_endline "            __m256d _vcfr = _mm256_set1_pd(cfr); \\";
  print_endline "            __m256d _vcfi = _mm256_set1_pd(cfi); \\";
  print_endline "            size_t _kk = 0; \\";
  print_endline "            for (; _kk + 4 <= (n); _kk += 4) { \\";
  print_endline "                __m256d _vr = _mm256_loadu_pd((re_p) + _kk); \\";
  print_endline "                __m256d _vi = _mm256_loadu_pd((im_p) + _kk); \\";
  print_endline "                __m256d _nr = _mm256_fmadd_pd(_vi, _vcfi, _mm256_mul_pd(_vr, _vcfr)); \\";
  print_endline "                __m256d _ni = _mm256_fnmadd_pd(_vr, _vcfi, _mm256_mul_pd(_vi, _vcfr)); \\";
  print_endline "                _mm256_storeu_pd((re_p) + _kk, _nr); \\";
  print_endline "                _mm256_storeu_pd((im_p) + _kk, _ni); \\";
  print_endline "            } \\";
  print_endline "            for (; _kk < (n); _kk++) { \\";
  print_endline "                double _tr = (re_p)[_kk]; \\";
  print_endline "                (re_p)[_kk] = _tr * (cfr) + (im_p)[_kk] * (cfi); \\";
  print_endline "                (im_p)[_kk] = (im_p)[_kk] * (cfr) - _tr * (cfi); \\";
  print_endline "            } \\";
  print_endline "        } \\";
  print_endline "    } while (0)";
  print_endline "";
  print_endline "#if defined(__AVX512F__)";
  print_endline "#define VFFT_PROTO_BWD_LEG0_CONJ_avx512(re_p, im_p, cfr, cfi, n) \\";
  print_endline "    do { \\";
  print_endline "        if ((cfr) != 1.0 || (cfi) != 0.0) { \\";
  print_endline "            __m512d _vcfr = _mm512_set1_pd(cfr); \\";
  print_endline "            __m512d _vcfi = _mm512_set1_pd(cfi); \\";
  print_endline "            size_t _kk = 0; \\";
  print_endline "            for (; _kk + 8 <= (n); _kk += 8) { \\";
  print_endline "                __m512d _vr = _mm512_loadu_pd((re_p) + _kk); \\";
  print_endline "                __m512d _vi = _mm512_loadu_pd((im_p) + _kk); \\";
  print_endline "                __m512d _nr = _mm512_fmadd_pd(_vi, _vcfi, _mm512_mul_pd(_vr, _vcfr)); \\";
  print_endline "                __m512d _ni = _mm512_fnmadd_pd(_vr, _vcfi, _mm512_mul_pd(_vi, _vcfr)); \\";
  print_endline "                _mm512_storeu_pd((re_p) + _kk, _nr); \\";
  print_endline "                _mm512_storeu_pd((im_p) + _kk, _ni); \\";
  print_endline "            } \\";
  print_endline "            for (; _kk < (n); _kk++) { \\";
  print_endline "                double _tr = (re_p)[_kk]; \\";
  print_endline "                (re_p)[_kk] = _tr * (cfr) + (im_p)[_kk] * (cfi); \\";
  print_endline "                (im_p)[_kk] = (im_p)[_kk] * (cfr) - _tr * (cfi); \\";
  print_endline "            } \\";
  print_endline "        } \\";
  print_endline "    } while (0)";
  print_endline "#endif /* __AVX512F__ */";
  print_endline "";
  print_endline "/* Backward stage: FUSED inverse butterfly + post-twiddle via t1s_bwd codelet.";
  print_endline " *";
  print_endline " * The t1s_bwd codelet (after the dft.ml DAG fix) does:";
  print_endline " *   1. Inverse butterfly (B⁻¹ = +θ-kernel DFT) on raw inputs";
  print_endline " *   2. Post-multiply legs 1..R-1 by conj(tw_scalar)";
  print_endline " *";
  print_endline " * Executor adds the leg-0 conj(cf0) post-mul (codelet doesn't touch leg 0).";
  print_endline " *";
  print_endline " * Compared to the pre-fusion path (n1_bwd + R-leg manual cmul loop), this";
  print_endline " * eliminates one full load/store pass over R-1 legs — the inverse butterfly's";
  print_endline " * output already has the legs in registers ready for the post-twiddle. */";
  print_endline "#define VFFT_PROTO_STAGE_BWD(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const int    num_groups  = st->num_groups; \\";
  print_endline "        const size_t stride      = st->stride; \\";
  print_endline "        const int    *needs_tw    = st->needs_tw; \\";
  print_endline "        const size_t *group_base  = st->group_base; \\";
  print_endline "        const double *cf0_re      = st->cf0_re; \\";
  print_endline "        const double *cf0_im      = st->cf0_im; \\";
  print_endline "        double      **tw_scalar_re = st->tw_scalar_re; \\";
  print_endline "        double      **tw_scalar_im = st->tw_scalar_im; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            double *base_re = re + group_base[g]; \\";
  print_endline "            double *base_im = im + group_base[g]; \\";
  print_endline "            if (!needs_tw[g]) { \\";
  print_endline "                radix##R##_n1_bwd_##ISA(base_re, base_im, NULL, NULL, stride, slice_K); \\";
  print_endline "                continue; \\";
  print_endline "            } \\";
  print_endline "            /* Fused: codelet does B⁻¹ + post-twiddle for legs 1..R-1. */ \\";
  print_endline "            radix##R##_t1s_dit_bwd_##ISA(base_re, base_im, \\";
  print_endline "                                          tw_scalar_re[g], tw_scalar_im[g], \\";
  print_endline "                                          stride, slice_K); \\";
  print_endline "            /* Executor: apply conj(cf0) to leg 0 (codelet skipped it). */ \\";
  print_endline "            VFFT_PROTO_BWD_LEG0_CONJ_##ISA(base_re, base_im, \\";
  print_endline "                                            cf0_re[g], cf0_im[g], slice_K); \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "/* ── DIF orientation macros ─────────────────────────────────────────";
  print_endline " *";
  print_endline " * DIF differs from DIT in two structural ways:";
  print_endline " *   1. The no-twiddle stage is the LAST (s = nf-1), not the first.";
  print_endline " *   2. cf0 = 1 universally (every cross-stage exponent contains j),";
  print_endline " *      so no executor-side leg-0 cmul is needed.";
  print_endline " *";
  print_endline " * Codelet calling convention: t1_dif_{fwd,bwd}_log3 takes raw per-leg";
  print_endline " * twiddles (read from grp_tw, K-replicated); t1_dif_{fwd,bwd} takes the";
  print_endline " * same layout for prototype-core's plan build (no separate flat staging).";
  print_endline " *";
  print_endline " * DIF backward uses the FUSED t1_dif_bwd codelet — production couldn't";
  print_endline " * (the old DAG was wrong); the dft.ml fix made it inverse-of-fwd. */";
  print_endline "";
  print_endline "/* DIF no-twiddle outer stage (s = nf-1). Just n1_fwd, no group branch. */";
  print_endline "#define VFFT_PROTO_STAGE_DIF_OUTER(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const int    num_groups  = st->num_groups; \\";
  print_endline "        const size_t stride      = st->stride; \\";
  print_endline "        const size_t *group_base = st->group_base; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            radix##R##_n1_fwd_##ISA(re + group_base[g], im + group_base[g], \\";
  print_endline "                                      NULL, NULL, stride, slice_K); \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "/* DIF FLAT (also T1S→FLAT fallback): codelet does butterfly + post-mul */";
  print_endline "/* legs 1..R-1 by grp_tw. needs_tw=0 groups fall to n1_fwd. */";
  print_endline "#define VFFT_PROTO_STAGE_DIF_FLAT(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const int    num_groups  = st->num_groups; \\";
  print_endline "        const size_t stride      = st->stride; \\";
  print_endline "        const int    *needs_tw    = st->needs_tw; \\";
  print_endline "        const size_t *group_base  = st->group_base; \\";
  print_endline "        double      **grp_tw_re   = st->grp_tw_re; \\";
  print_endline "        double      **grp_tw_im   = st->grp_tw_im; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            double *base_re = re + group_base[g]; \\";
  print_endline "            double *base_im = im + group_base[g]; \\";
  print_endline "            if (!needs_tw[g]) { \\";
  print_endline "                radix##R##_n1_fwd_##ISA(base_re, base_im, NULL, NULL, \\";
  print_endline "                                          stride, slice_K); \\";
  print_endline "                continue; \\";
  print_endline "            } \\";
  print_endline "            radix##R##_t1_dif_fwd_##ISA(base_re, base_im, \\";
  print_endline "                                          grp_tw_re[g], grp_tw_im[g], \\";
  print_endline "                                          stride, slice_K); \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "/* DIF LOG3: same as DIF_FLAT but calls log3 codelet variant. */";
  print_endline "#define VFFT_PROTO_STAGE_DIF_LOG3(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const int    num_groups  = st->num_groups; \\";
  print_endline "        const size_t stride      = st->stride; \\";
  print_endline "        const int    *needs_tw    = st->needs_tw; \\";
  print_endline "        const size_t *group_base  = st->group_base; \\";
  print_endline "        double      **grp_tw_re   = st->grp_tw_re; \\";
  print_endline "        double      **grp_tw_im   = st->grp_tw_im; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            double *base_re = re + group_base[g]; \\";
  print_endline "            double *base_im = im + group_base[g]; \\";
  print_endline "            if (!needs_tw[g]) { \\";
  print_endline "                radix##R##_n1_fwd_##ISA(base_re, base_im, NULL, NULL, \\";
  print_endline "                                          stride, slice_K); \\";
  print_endline "                continue; \\";
  print_endline "            } \\";
  print_endline "            radix##R##_t1_dif_log3_fwd_##ISA(base_re, base_im, \\";
  print_endline "                                              grp_tw_re[g], grp_tw_im[g], \\";
  print_endline "                                              stride, slice_K); \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline "";
  print_endline "/* DIF backward: fused t1_dif_bwd codelet (T_conj then B⁻¹) for needs_tw=1";
  print_endline " * groups, n1_bwd for needs_tw=0. cf0=1 in DIF so no leg-0 cmul needed. */";
  print_endline "#define VFFT_PROTO_STAGE_DIF_BWD(S, R, ISA) \\";
  print_endline "    if (start_stage <= (S)) { \\";
  print_endline "        const stride_stage_t *st = &plan->stages[S]; \\";
  print_endline "        const int    num_groups  = st->num_groups; \\";
  print_endline "        const size_t stride      = st->stride; \\";
  print_endline "        const int    *needs_tw    = st->needs_tw; \\";
  print_endline "        const size_t *group_base  = st->group_base; \\";
  print_endline "        double      **grp_tw_re   = st->grp_tw_re; \\";
  print_endline "        double      **grp_tw_im   = st->grp_tw_im; \\";
  print_endline "        for (int g = 0; g < num_groups; g++) { \\";
  print_endline "            double *base_re = re + group_base[g]; \\";
  print_endline "            double *base_im = im + group_base[g]; \\";
  print_endline "            if (!needs_tw[g]) { \\";
  print_endline "                radix##R##_n1_bwd_##ISA(base_re, base_im, NULL, NULL, \\";
  print_endline "                                          stride, slice_K); \\";
  print_endline "                continue; \\";
  print_endline "            } \\";
  print_endline "            radix##R##_t1_dif_bwd_##ISA(base_re, base_im, \\";
  print_endline "                                          grp_tw_re[g], grp_tw_im[g], \\";
  print_endline "                                          stride, slice_K); \\";
  print_endline "        } \\";
  print_endline "    }";
  print_endline ""

(* ── Header file shape ─────────────────────────────────────────────── *)

let emit_header_file (entries : plan_entry list) ~isa =
  print_endline "/* plan_executors.h — auto-generated by bin/emit_executor_h.ml.";
  print_endline " * DO NOT EDIT BY HAND.";
  print_endline " *";
  print_endline " * Plan-shaped specialized executors. Each function below is a";
  print_endline " * drop-in replacement for the generic executor for a specific";
  print_endline " * (N, K, factorization, variant-assignment) tuple. Compared to the";
  print_endline " * generic executor:";
  print_endline " *   - Codelet calls direct (vs function-pointer indirection)";
  print_endline " *   - 4-branch per-group variant tree collapsed to one codepath";
  print_endline " *   - needs_tw[g] branch retained (per-group runtime decision)";
  print_endline " *";
  print_endline " * Entries are sourced from a wisdom file (production format). To add a";
  print_endline " * new specialization: append a line to the wisdom file and regen. */";
  print_endline "#ifndef VFFT_PROTO_PLAN_EXECUTORS_H";
  print_endline "#define VFFT_PROTO_PLAN_EXECUTORS_H";
  print_endline "";
  print_endline "#include <stddef.h>";
  print_endline "#include <stdlib.h>  /* abort() for unimplemented variants */";
  print_endline "";
  print_endline "#ifndef VFFT_PROTO_USE_PRODUCTION_PLAN_T";
  print_endline "";
  print_endline "#define STRIDE_MAX_STAGES 16";
  print_endline "";
  print_endline "typedef void (*vfft_proto_codelet_fn)(double *, double *,";
  print_endline "                                      const double *, const double *,";
  print_endline "                                      size_t, size_t);";
  print_endline "typedef void (*vfft_proto_n1_fn)(const double *, const double *,";
  print_endline "                                  double *, double *,";
  print_endline "                                  size_t, size_t, size_t);";
  print_endline "";
  print_endline "typedef struct {";
  print_endline "    size_t        base;";
  print_endline "    const double *tw_re;";
  print_endline "    const double *tw_im;";
  print_endline "} stride_invocation_t;";
  print_endline "";
  print_endline "typedef struct {";
  print_endline "    int    radix;";
  print_endline "    size_t stride;";
  print_endline "    int    num_groups;";
  print_endline "    size_t *group_base;";
  print_endline "    int    *needs_tw;";
  print_endline "    double *cf0_re;";
  print_endline "    double *cf0_im;";
  print_endline "    double **tw_scalar_re;";
  print_endline "    double **tw_scalar_im;";
  print_endline "    double **grp_tw_re;";
  print_endline "    double **grp_tw_im;";
  print_endline "    double *tw_scalar_pool_re;";
  print_endline "    double *tw_scalar_pool_im;";
  print_endline "    double *tw_pool_re;";
  print_endline "    double *tw_pool_im;";
  print_endline "    double *cf_all_re;";
  print_endline "    double *cf_all_im;";
  print_endline "    int    use_n1_fallback;";
  print_endline "    int    use_log3;";
  print_endline "    stride_invocation_t *tape;";
  print_endline "    vfft_proto_n1_fn      n1_fwd;";
  print_endline "    vfft_proto_n1_fn      n1_bwd;";
  print_endline "    vfft_proto_codelet_fn t1_fwd;";
  print_endline "    vfft_proto_codelet_fn t1_bwd;   /* used by DIF bwd executor (fused) */";
  print_endline "    vfft_proto_codelet_fn t1s_fwd;";
  print_endline "} stride_stage_t;";
  print_endline "";
  print_endline "typedef struct {";
  print_endline "    int    N;";
  print_endline "    size_t K;";
  print_endline "    int    num_stages;";
  print_endline "    int    factors[STRIDE_MAX_STAGES];";
  print_endline "    int    variants[STRIDE_MAX_STAGES]; /* per-stage twiddle variant";
  print_endline "                                        * (0=FLAT 1=LOG3 2=T1S 3=BUF),";
  print_endline "                                        * recorded by plan_create_ex so";
  print_endline "                                        * OOP wisdom can persist the mix */";
  print_endline "    stride_stage_t stages[STRIDE_MAX_STAGES];";
  print_endline "    int    use_dif_forward;";
  print_endline "} stride_plan_t;";
  print_endline "";
  print_endline "#include <immintrin.h>";
  print_endline "";
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
  print_endline "#ifndef VFFT_PROTO_TW_BLOCK_K";
  print_endline "#define VFFT_PROTO_TW_BLOCK_K 64";
  print_endline "#endif";
  print_endline "";
  print_endline "#endif /* VFFT_PROTO_USE_PRODUCTION_PLAN_T */";
  print_endline "";
  let bwd_entries = dedup_for_bwd entries in
  (* Per-ISA emission helper: externs + executors + lookups. *)
  let emit_for_isa isa =
    let all_externs =
      let fwd_syms = List.concat_map (fun e -> collect_externs e `Fwd ~isa) entries in
      let bwd_syms = List.concat_map (fun e -> collect_externs e `Bwd ~isa) bwd_entries in
      List.sort_uniq compare (fwd_syms @ bwd_syms)
    in
    Printf.printf "/* Externs for %s codelets called by the executors below. */\n" isa;
    List.iter (fun sym ->
      Printf.printf
        "extern void %s(double *rio_re, double *rio_im,\n\
        \                                 const double *tw_re, const double *tw_im,\n\
        \                                 size_t ios, size_t me);\n" sym
    ) all_externs;
    print_endline "";
    Printf.printf "/* ── Forward executors (%s) ────────────────────────────── */\n" isa;
    List.iter (fun e -> emit_executor e `Fwd ~isa; print_endline "") entries;
    Printf.printf "/* ── Backward executors (%s) ─────────────────────────────── */\n" isa;
    List.iter (fun e -> emit_executor e `Bwd ~isa; print_endline "") bwd_entries;
    emit_lookup_fwd entries ~isa;
    emit_lookup_bwd entries ~isa
  in
  emit_stage_macros ();
  print_endline "/* Function-pointer signature shared by forward and backward executors. */";
  print_endline "typedef void (*vfft_proto_exec_fn)(const stride_plan_t *, double *, double *,";
  print_endline "                                   size_t, size_t, int);";
  print_endline "";
  (* AVX-2 emission — always available on x86-64 with AVX2 (the project's
   * minimum baseline). *)
  emit_for_isa "avx2";
  print_endline "";
  (* AVX-512 emission — guarded by __AVX512F__ so non-AVX-512 builds don't
   * try to link against AVX-512 codelet symbols that aren't compiled in. *)
  print_endline "#if defined(__AVX512F__)";
  emit_for_isa "avx512";
  print_endline "#endif /* __AVX512F__ */";
  let _ = isa in  (* unused now; CLI param kept for backward-compat. *)
  print_endline "";
  print_endline "#endif /* VFFT_PROTO_PLAN_EXECUTORS_H */"

(* ── CLI ────────────────────────────────────────────────────────────── *)

let () =
  let isa = ref "avx2" in
  let wisdom_path = ref None in
  let i = ref 1 in
  while !i < Array.length Sys.argv do
    let arg = Sys.argv.(!i) in
    if arg = "--isa" && !i + 1 < Array.length Sys.argv then begin
      isa := Sys.argv.(!i + 1);
      i := !i + 2
    end
    else if arg = "--wisdom" && !i + 1 < Array.length Sys.argv then begin
      wisdom_path := Some Sys.argv.(!i + 1);
      i := !i + 2
    end
    else begin
      Printf.eprintf "warning: unknown arg %s\n" arg;
      incr i
    end
  done;
  let path = match !wisdom_path with
    | Some p -> p
    | None -> failwith "--wisdom <path> is required"
  in
  let entries = read_wisdom_file path |> dedup_entries in
  if entries = [] then
    failwith (Printf.sprintf "wisdom file %s contains no entries" path);
  emit_header_file entries ~isa:!isa
