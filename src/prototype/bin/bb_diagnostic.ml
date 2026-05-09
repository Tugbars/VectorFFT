(* bb_diagnostic.ml — count cluster peak-live and cp_progress before/after B&B
 * for a given codelet config. Useful for understanding cluster shape and
 * checking whether B&B finds non-trivial schedule alternatives. *)

open Vfft_v2

let () =
  let n =
    if Array.length Sys.argv > 1 then int_of_string Sys.argv.(1)
    else 32
  in
  let isa_name =
    if Array.length Sys.argv > 2 then Sys.argv.(2)
    else "avx2"
  in
  let uarch_name =
    if Array.length Sys.argv > 3 then Sys.argv.(3)
    else "raptor_lake_avx2"
  in
  let time_budget =
    if Array.length Sys.argv > 4 then float_of_string Sys.argv.(4)
    else 1.0
  in
  let _isa = Isa.of_name isa_name in
  let uarch = Uarch.of_name uarch_name in

  Printf.printf "=== B&B diagnostic: R=%d %s, budget=%.2fs/cluster ===\n%!" n isa_name time_budget;

  Algsimp.reset ();
  let policy = Dft.TP_Flat in
  let direction = Dft.DIT in
  let sign = `Fwd in
  let raw, spill_markers, spill_ct =
    Dft.dft_expand_twiddled_spill ~direction ~sign ~policy n
  in
  let reassoc = Dft.needs_reassoc n in
  let simplified = Algsimp.of_assignments ~reassoc raw in
  let deduped = Algsimp.dedup_sub_pairs simplified in

  let tag_markers = Algsimp.lift_spill_markers ~reassoc spill_markers in
  let sp = Emit_c.make_spill_info ?ct:spill_ct ~fuse:0 tag_markers in

  let n2 = match spill_ct with Some (_, n2) -> n2 | None -> failwith "no CT" in

  let is_spill_target tag =
    Hashtbl.mem sp.Emit_c.re_slot tag || Hashtbl.mem sp.Emit_c.im_slot tag
  in

  let cluster_sinks = Array.make n2 [] in
  List.iter (fun (oref, e) ->
    match oref with
    | Expr.Output (k, _) ->
      let k2 = k mod n2 in
      cluster_sinks.(k2) <- e :: cluster_sinks.(k2)
    | _ -> ()
  ) deduped;

  Printf.printf "Cluster diagnostics for R=%d %s (CT %dx%d):\n%!" n isa_name (n / n2) n2;

  let total_subset = ref 0 in
  let total_baseline_peak = ref 0 in
  let total_bb_peak = ref 0 in
  let num_improvements = ref 0 in

  for k2 = 0 to n2 - 1 do
    let sinks_k = cluster_sinks.(k2) in
    if sinks_k <> [] then begin
      let visited = Hashtbl.create 64 in
      let subset = ref [] in
      let rec walk e =
        if not (Hashtbl.mem visited e.Algsimp.tag) then begin
          Hashtbl.add visited e.Algsimp.tag ();
          if not (is_spill_target e.Algsimp.tag) then begin
            subset := e :: !subset;
            List.iter walk (Vfft_v2.Bb.preds_of e)
          end
        end
      in
      List.iter walk sinks_k;
      let subset = !subset in

      let baseline = Schedule.su_schedule_subset uarch ~gh:true ~subset ~sinks:sinks_k in
      let baseline_peak = Vfft_v2.Bb.compute_peak_live ~subset ~sinks:sinks_k baseline in
      let (_, bb_peak, bb_progress) =
        Vfft_v2.Bb.bb_search ~uarch ~subset ~sinks:sinks_k
          ~initial_schedule:baseline
          ~initial_peak:baseline_peak
          ~time_budget_sec:time_budget
      in
      let baseline_progress = Vfft_v2.Bb.compute_progress baseline
        (Vfft_v2.Schedule.compute_cp_dist uarch sinks_k subset) in
      total_subset := !total_subset + List.length subset;
      total_baseline_peak := !total_baseline_peak + baseline_peak;
      total_bb_peak := !total_bb_peak + bb_peak;
      if bb_peak < baseline_peak then incr num_improvements;
      Printf.printf "  Cluster %d: subset=%3d  sinks=%2d  peak: SU+GH=%2d -> B&B=%2d  progress: %d -> %d  %s\n%!"
        k2 (List.length subset) (List.length sinks_k)
        baseline_peak bb_peak baseline_progress bb_progress
        (if bb_peak < baseline_peak then "PEAK_IMPROVED"
         else if bb_progress > baseline_progress then "PROG_IMPROVED"
         else "tied")
    end
  done;

  Printf.printf "\n=== Summary ===\n";
  Printf.printf "Total cluster subset size: %d nodes\n" !total_subset;
  Printf.printf "Sum SU+GH peak: %d\n" !total_baseline_peak;
  Printf.printf "Sum B&B peak: %d\n" !total_bb_peak;
  Printf.printf "Clusters improved: %d / %d\n" !num_improvements n2
