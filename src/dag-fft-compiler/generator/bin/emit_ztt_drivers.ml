(* emit_ztt_drivers.ml — emit the ZTURN-T fused driver FILES (Ztt_drivers.emit_all).
 *
 * The drivers are DERIVED from the corpus cells (every admissible (N, chain):
 * natural {fwd, bwd} x {dest, plane}, plain {fwd, bwd}), not codelets: they
 * live in generated/ beside the registries, produced by the same promote-rule
 * mechanism, and the registry (emit_ztt_registry.exe) is derived from the
 * same cell list. ONE FILE PER (family, N) since 2026-09-14 — ztt_drivers_
 * <isa>_<N>.c (natural order) and zttp_drivers_<isa>_<N>.c (plain, the
 * scrambled order) — so a kind change recompiles only its family's files, in
 * parallel; the single TU compiled 47 minutes on one thread.
 *
 * Usage:
 *   dune exec bin/emit_ztt_drivers.exe -- --isa avx2 --uarch raptor_lake_avx2 --split generated
 *)

let () =
  let isa = ref "avx2"
  and uarch = ref "raptor_lake_avx2"
  and dir = ref "" in
  let rec parse = function
    | "--isa" :: v :: tl ->
      isa := v;
      parse tl
    | "--uarch" :: v :: tl ->
      uarch := v;
      parse tl
    | "--split" :: v :: tl ->
      dir := v;
      parse tl
    | [] -> ()
    | t :: _ -> failwith ("emit_ztt_drivers: unknown flag " ^ t)
  in
  parse (List.tl (Array.to_list Sys.argv));
  if !dir = "" then failwith "emit_ztt_drivers: --split <dir> is required (one file per family and N)";
  Ztt_drivers.emit_all ~isa:(Isa.of_name !isa) ~uarch:(Uarch.of_name !uarch) ~dir:!dir
;;
