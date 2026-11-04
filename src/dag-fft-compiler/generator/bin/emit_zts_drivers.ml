(* emit_zts_drivers.ml — emit the zturn-ts fused driver FILES (Zts_drivers.emit_all):
 * one TU per (family, N), zts_drivers_<isa>_<N>.c (K=1, split planes) and
 * ztv_drivers_<isa>_<N>.c (lane-major K=VW), into generated/fused_codelets/.
 * The registry (emit_zts_registry.exe) lists the same cells.
 *
 * Usage:
 *   dune exec bin/emit_zts_drivers.exe -- --isa avx2 --uarch raptor_lake_avx2 --split generated/fused_codelets
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
    | t :: _ -> failwith ("emit_zts_drivers: unknown flag " ^ t)
  in
  parse (List.tl (Array.to_list Sys.argv));
  if !dir = "" then failwith "emit_zts_drivers: --split <dir> is required (one file per family and N)";
  Zts_drivers.emit_all ~isa:(Isa.of_name !isa) ~uarch:(Uarch.of_name !uarch) ~dir:!dir
;;
