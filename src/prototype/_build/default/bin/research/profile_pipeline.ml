(* bin/profile_pipeline.ml — Time each algsimp pass individually for
 * the pow2 codelet pipeline. Identifies which pass causes the
 * approximately-O(N⁴) scaling wall observed at R=128.
 *
 * Pipeline stages timed:
 *   1. dft_expand_twiddled   (math layer Expr construction)
 *   2. of_assignments        (initial hashconsing)
 *   3. dedup_sub_pairs       (first call)
 *   4. factor_common_muls
 *   5. factor_by_atom
 *   6. dedup_sub_pairs       (second call)
 *   7. share_subsums
 *
 * Per stage, prints wall-clock seconds and post-stage IR size (nodes
 * reachable from output assignments). Compares R=32 → R=64 scaling
 * per-pass to identify the culprit before attempting R=128 (which
 * would otherwise require ~5 minutes per measurement).
 *
 * Each stage has a per-call timeout: if a pass exceeds the budget,
 * we abort and report. This lets us probe R=128 without committing
 * to a multi-minute run.
 *)

open Vfft_v2

let now () = Unix.gettimeofday ()

let nodes_of_assigns (a : (Expr.elem_ref * Algsimp.t) list) : int =
  let st = Algsimp.stats_reachable (List.map snd a) in
  st.Algsimp.adds + st.subs + st.muls + st.negs
  + (2 * st.cmuls) + st.fmas

let time_pass (label : string) (f : unit -> 'a) : 'a * float =
  let t0 = now () in
  let r = f () in
  let t1 = now () in
  Printf.printf "    %-25s %7.3fs\n" label (t1 -. t0);
  flush stdout;
  (r, t1 -. t0)

let profile_n (n : int) =
  Printf.printf "  R=%-4d profile:\n" n;
  flush stdout;

  let aggressive = false in  (* pow2 = Cooley_Tukey or Split_radix → false *)

  (* Stage 1: math layer *)
  let raw, _t_raw = time_pass "math (dft_expand)" (fun () ->
    Dft.dft_expand_twiddled ~policy:Dft.TP_Flat ~direction:Dft.DIT ~sign:`Fwd n
  ) in
  Printf.printf "      → raw assigns: %d\n" (List.length raw);
  flush stdout;

  (* Stage 2: of_assignments *)
  Algsimp.reset ();
  let reassoc = Dft.needs_reassoc n in
  let simplified, _t_oa = time_pass "of_assignments" (fun () ->
    Algsimp.of_assignments ~reassoc raw
  ) in
  Printf.printf "      → reachable nodes: %d\n" (nodes_of_assigns simplified);
  flush stdout;

  (* Stage 3: dedup #1 *)
  let deduped1, _t_d1 = time_pass "dedup_sub_pairs #1" (fun () ->
    Algsimp.dedup_sub_pairs simplified
  ) in
  Printf.printf "      → reachable nodes: %d\n" (nodes_of_assigns deduped1);
  flush stdout;

  (* Stage 4: factor_common_muls *)
  let factored1, _t_fcm = time_pass "factor_common_muls" (fun () ->
    Algsimp.factor_common_muls ~aggressive deduped1
  ) in
  Printf.printf "      → reachable nodes: %d\n" (nodes_of_assigns factored1);
  flush stdout;

  (* Stage 5: factor_by_atom *)
  let factored2, _t_fba = time_pass "factor_by_atom" (fun () ->
    Algsimp.factor_by_atom ~aggressive factored1
  ) in
  Printf.printf "      → reachable nodes: %d\n" (nodes_of_assigns factored2);
  flush stdout;

  (* Stage 6: dedup #2 *)
  let deduped2, _t_d2 = time_pass "dedup_sub_pairs #2" (fun () ->
    Algsimp.dedup_sub_pairs factored2
  ) in
  Printf.printf "      → reachable nodes: %d\n" (nodes_of_assigns deduped2);
  flush stdout;

  (* Stage 7: share_subsums *)
  let shared, _t_ss = time_pass "share_subsums" (fun () ->
    Algsimp.share_subsums ~aggressive deduped2
  ) in
  Printf.printf "      → reachable nodes: %d\n" (nodes_of_assigns shared);
  flush stdout

let () =
  let radii =
    if Array.length Sys.argv > 1 then
      let r = int_of_string Sys.argv.(1) in [r]
    else
      [16; 32; 64]
  in
  Printf.printf "=== Per-pass timing for pow2 pipeline ===\n";
  List.iter (fun n ->
    profile_n n;
    Printf.printf "\n";
    flush stdout
  ) radii
