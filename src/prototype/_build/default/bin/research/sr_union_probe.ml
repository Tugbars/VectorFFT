(* bin/sr_union_probe.ml — Construct BOTH CT and SR derivations of the same
 * codelet within a single DAG, run our pipeline, count post-algsimp ops.
 * Compare against running CT alone and SR alone separately.
 *
 * Hypothesis being tested: do CT and SR share intermediate structure that
 * our existing simplification passes (factor_common_muls, share_subsums,
 * dedup_sub_pairs, fp transpose loop) can exploit when they see both
 * constructions simultaneously?
 *
 * Mechanism: emit BOTH X_CT[k] and X_SR[k] as distinct output assignments
 * (with different elem_refs). Run the pipeline on the joined assignment
 * list. Count post-algsimp ops over the union vs the sum of individual
 * runs.
 *
 * Interpretation:
 *   union_ops ≈ ct_alone + sr_alone   → constructions are disjoint;
 *                                       no cross-pollination possible
 *                                       within current passes.
 *   union_ops < ct_alone + sr_alone   → some sharing the passes find when
 *                                       given both views; gap quantifies
 *                                       potential for cooperation.
 *   union_ops < min(ct_alone,         → STRONG cooperation: union is smaller
 *                   sr_alone)           than either individually. Would
 *                                       require extracting just one X[k]
 *                                       per output to be deployable.
 *)

open Vfft_v2

(* Run pipeline on a given assignment list and return total op count + stats. *)
let run_pipeline (raw : Expr.assignment list) (n : int) : int * Algsimp.dag_stats =
  Algsimp.reset ();
  let reassoc = Dft.needs_reassoc n in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped_pre = Algsimp.dedup_sub_pairs simplified in
  let aggressive = match Dft.pick_algorithm n with
    | Dft.Direct -> true
    | Dft.Cooley_Tukey _ -> false
    | Dft.Split_radix -> false in
  let is_direct = aggressive in
  let factored = Algsimp.factor_common_muls ~aggressive deduped_pre in
  let factored = Algsimp.factor_by_atom ~aggressive factored in
  let factored = Algsimp.dedup_sub_pairs factored in
  let shared =
    if is_direct then factored
    else Algsimp.share_subsums ~aggressive factored
  in
  let stats = Algsimp.stats_reachable (List.map snd shared) in
  let total =
    stats.Algsimp.adds + stats.subs + stats.muls + stats.negs
    + (2 * stats.cmuls) + stats.fmas
  in
  (total, stats)

(* Capture raw assignments under either CT (env unset) or SR (env set). *)
let raw_for_mode (mode : [`CT | `SR]) (n : int) : Expr.assignment list =
  (match mode with
   | `CT -> Unix.putenv "VFFT_SPLIT_RADIX" ""
   | `SR -> Unix.putenv "VFFT_SPLIT_RADIX" "1");
  Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIT ~sign:`Fwd n

(* Re-tag CT outputs onto a different elem_ref namespace so they don't
 * collide with SR's outputs. We pretend CT outputs are written to a
 * synthetic Output index range [N, 2N). The math layer doesn't care about
 * elem_ref values; algsimp uses them only as keys. *)
let retag_outputs (offset : int) (assigns : Expr.assignment list) : Expr.assignment list =
  List.map (fun (r, e) ->
    let r' = match r with
      | Expr.Output (k, re) -> Expr.Output (k + offset, re)
      | other -> other  (* twiddles/inputs are shared; don't retag *)
    in
    (r', e)
  ) assigns

let pretty_stats label total st =
  let open Algsimp in
  Printf.printf "  %-22s adds=%d subs=%d muls=%d cmuls=%d fmas=%d negs=%d  total_mul_class+add_class=%d\n"
    label st.adds st.subs st.muls st.cmuls st.fmas st.negs total

let () =
  let n = int_of_string Sys.argv.(1) in
  Printf.printf "=== R=%d : union probe ===\n" n;

  let raw_ct = raw_for_mode `CT n in
  let raw_sr = raw_for_mode `SR n in

  let ct_total, ct_stats = run_pipeline raw_ct n in
  pretty_stats "CT alone:" ct_total ct_stats;

  let sr_total, sr_stats = run_pipeline raw_sr n in
  pretty_stats "SR alone:" sr_total sr_stats;

  (* Union: CT outputs at indices [0, N), SR outputs at indices [N, 2N). *)
  let raw_union =
    let ct_retagged = retag_outputs 0 raw_ct in   (* CT keeps [0, N) *)
    let sr_retagged = retag_outputs n raw_sr in   (* SR shifts to [N, 2N) *)
    ct_retagged @ sr_retagged
  in
  let union_total, union_stats = run_pipeline raw_union n in
  pretty_stats "Union (CT+SR):" union_total union_stats;

  let sum_alone = ct_total + sr_total in
  let savings_vs_sum = sum_alone - union_total in
  let pct_vs_sum =
    100.0 *. float_of_int savings_vs_sum /. float_of_int sum_alone in
  Printf.printf "\n  Sum (CT_alone + SR_alone)              = %d\n" sum_alone;
  Printf.printf "  Union (single pipeline run on combined) = %d\n" union_total;
  Printf.printf "  Savings from joint construction         = %d (%.1f%%)\n"
    savings_vs_sum pct_vs_sum;
  let smaller_alone = min ct_total sr_total in
  Printf.printf "  Union vs smaller-alone (%d): union/alone = %.3f\n"
    smaller_alone
    (float_of_int union_total /. float_of_int smaller_alone);
  if union_total < smaller_alone then
    Printf.printf "  *** STRONG: union is smaller than EITHER alone ***\n"
  else if savings_vs_sum > sum_alone / 20 then
    Printf.printf "  -> Some cross-pollination (savings > 5%% of sum)\n"
  else
    Printf.printf "  -> Constructions are essentially disjoint at our pipeline's resolution\n"
