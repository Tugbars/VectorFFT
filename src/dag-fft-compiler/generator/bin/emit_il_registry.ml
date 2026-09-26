(* emit_il_registry.ml — auto-generate the INTERLEAVED-COMPLEX (IL) registry.
 *
 * The last hand-maintained dispatch surface. The IL family is the largest in
 * the corpus (253 cells in `zil-pure` + `zil-boundary`) and was the only one
 * with no generated registry: `src/core/oop/il2p.h` carried its extern blocks
 * and its radix lists by hand, as `VFFT_IL2P_DECL_LEAF(4) ... (27)` macro
 * invocations and `C(4) C(8) ... C(27)` switch-case runs. Adding a kernel
 * meant remembering to touch both, in two files — and a forgotten entry is
 * silent: the codelet ships and is simply never selected.
 *
 * This emitter derives both from the corpus, so "exists" and "reachable"
 * cannot diverge.
 *
 * WHAT IT EMITS
 *   1. `extern void` declarations for every IL cell, on the frozen 11-arg
 *      z ABI (the same signature `vfft_il2p_fn` points to).
 *   2. X-macro radix lists per (kind, direction):
 *        VFFT_IL_<KIND>_FWD_RADICES(X)   fwd exists
 *        VFFT_IL_<KIND>_BWD_RADICES(X)   bwd exists
 *        VFFT_IL_<KIND>_PAIR_RADICES(X)  BOTH exist
 *      The PAIR list is the one a `bwd ? sym_bwd : sym_fwd` resolver needs —
 *      using it makes the one-sided-kernel bug unrepresentable.
 *
 * FILENAME != SYMBOL (the family's standing trap, verified against the tree):
 *   radix{R}_z_{kind}_avx2.c      defines  radix{R}_z_{kind}_fwd_avx2
 *   radix{R}_z_{kind}_bwd_avx2.c  defines  radix{R}_z_{kind}_bwd_avx2
 * i.e. forward is IMPLICIT in the filename and EXPLICIT in the symbol.
 *
 * Usage:
 *   dune exec bin/emit_il_registry.exe > generated/il_registry_avx2.h
 *)

let starts_with p s =
  String.length s >= String.length p && String.sub s 0 (String.length p) = p
;;

let ends_with sfx s =
  let ls = String.length s
  and lf = String.length sfx in
  ls >= lf && String.sub s (ls - lf) lf = sfx
;;

let chop_suffix_opt sfx s =
  if ends_with sfx s then Some (String.sub s 0 (String.length s - String.length sfx)) else None
;;

(* "radix8_z_tmg_bwd_avx2" -> (8, "tmg", `Bwd)
   "radix8_z_tmg_avx2"     -> (8, "tmg", `Fwd)   (fwd implicit) *)
let parse_stem ~(isa : string) (stem : string) : (int * string * [ `Fwd | `Bwd ]) option =
  match chop_suffix_opt ("_" ^ isa) stem with
  | None -> None
  | Some s ->
    let s, dir =
      match chop_suffix_opt "_bwd" s with
      | Some s' -> s', `Bwd
      | None -> s, `Fwd
    in
    if not (starts_with "radix" s)
    then None
    else (
      let rest = String.sub s 5 (String.length s - 5) in
      match String.index_opt rest '_' with
      | None -> None
      | Some i ->
        let digits = String.sub rest 0 i in
        let tag = String.sub rest (i + 1) (String.length rest - i - 1) in
        (match int_of_string_opt digits, chop_suffix_opt "" tag with
         | Some r, _ when starts_with "z_" tag ->
           Some (r, String.sub tag 2 (String.length tag - 2), dir)
         | _ -> None))
;;

let macro_of_tag (tag : string) : string =
  String.uppercase_ascii (String.map (fun c -> if c = '.' then '_' else c) tag)
;;

let quadrants_of_isa isa =
  if isa = "avx2" then [ "zil-boundary"; "zil-pure" ]
  else [ "zil-boundary-" ^ isa; "zil-pure-" ^ isa ]
;;

(* (tag, dir) -> radices, for one ISA's quadrants *)
let table_of_isa isa =
  let parsed =
    List.filter_map
      (fun (fname_c, _argv) ->
         let stem = if Filename.check_suffix fname_c ".c" then Filename.chop_suffix fname_c ".c" else fname_c in
         parse_stem ~isa stem)
      (List.concat_map Corpus.files (quadrants_of_isa isa))
  in
  let tbl : (string * [ `Fwd | `Bwd ], int list) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun (r, tag, dir) ->
       let k = tag, dir in
       Hashtbl.replace tbl k (r :: Option.value ~default:[] (Hashtbl.find_opt tbl k)))
    parsed;
  tbl
;;

let () =
  let isa = ref "avx2" in
  (match List.tl (Array.to_list Sys.argv) with
   | [] -> ()
   | [ "--isa"; v ] -> isa := v
   | l -> failwith ("emit_il_registry: usage [--isa avx2|avx512], got " ^ String.concat " " l));
  let isa = !isa in
  let up = String.uppercase_ascii isa in
  let tables = List.map (fun i -> i, table_of_isa i) Corpus.zil_isas in
  let tbl = List.assoc isa tables in
  let radices k = match Hashtbl.find_opt tbl k with Some l -> List.sort_uniq compare l | None -> [] in
  (* THE KIND UNIVERSE: every (kind, list) non-empty at ANY isa is DEFINED at
     every isa (empty where absent), so the consumers' X-macro surface is
     identical whatever VFFT_ISA the user picked; an empty list = a resolver
     that returns 0 = the cell is refused, never a compile error. *)
  let univ name_of =
    List.sort_uniq compare
      (List.concat_map
         (fun (_, t) -> Hashtbl.fold (fun k l acc -> if l <> [] then name_of k :: acc else acc) t [])
         tables)
  in
  let tags = univ fst in
  let defined_somewhere tag lst =
    List.exists
      (fun (_, t) ->
         let g d = match Hashtbl.find_opt t (tag, d) with Some l -> List.sort_uniq compare l | None -> [] in
         match lst with
         | "FWD" -> g `Fwd <> []
         | "BWD" -> g `Bwd <> []
         | _ -> List.exists (fun r -> List.mem r (g `Bwd)) (g `Fwd))
      tables
  in
  Printf.printf "/* AUTO-GENERATED by bin/emit_il_registry.ml --isa %s from\n" isa;
  Printf.printf " * Corpus.files %s. DO NOT EDIT BY HAND.\n" (String.concat " + " (List.map (Printf.sprintf "\"%s\"") (quadrants_of_isa isa)));
  Printf.printf " * Regenerate: redirect the emitter (bin/emit_il_registry.exe --isa %s);\n" isa;
  Printf.printf " * never a bare dune build in this tree (it promotes tracked headers).\n";
  Printf.printf " *\n";
  Printf.printf " * The IL family's extern declarations and radix lists, derived from the\n";
  Printf.printf " * corpus so that \"the codelet exists\" and \"the resolver can reach it\"\n";
  Printf.printf " * cannot drift apart. Consumed by src/core/oop/il2p.h.\n";
  Printf.printf " *\n";
  Printf.printf " * Radix lists are X-macros: pass a one-argument macro.\n";
  Printf.printf " *   #define C(R) case R: return radix##R##_z_n1t_fwd_%s;\n" isa;
  Printf.printf " *   VFFT_IL_N1T_FWD_RADICES(C)\n";
  Printf.printf " *   #undef C\n";
  Printf.printf " * _PAIR_ lists carry only radices where BOTH directions exist — use them\n";
  Printf.printf " * for any `bwd ? x_bwd : x_fwd` resolver.\n";
  Printf.printf " *\n";
  Printf.printf " * ISA-neutral surface: VFFT_IL_SYM(stem) pastes this ISA's suffix; every\n";
  Printf.printf " * kind of the union over ISAs is defined (empty = absent at this ISA). */\n";
  Printf.printf "#ifndef VFFT_IL_REGISTRY_%s_H\n#define VFFT_IL_REGISTRY_%s_H\n\n#include <stddef.h>\n\n" up up;
  Printf.printf "#define VFFT_IL_ISA_NAME \"%s\"\n" isa;
  Printf.printf "#define VFFT_IL_VW %d          /* doubles per vector: the twiddle-record width */\n" (match isa with "avx2" -> 4 | _ -> 8);
  Printf.printf "#define VFFT_IL_SYM(stem) stem##_%s\n\n" isa;
  Printf.printf "/* the frozen 11-arg z ABI (vfft_il2p_fn's pointee) */\n";
  Printf.printf "#define VFFT_IL_DECL(SYM) \\\n";
  Printf.printf "  extern void SYM(const double *, const double *, double *, double *, \\\n";
  Printf.printf "                  const double *, const double *, \\\n                  size_t, size_t, size_t, size_t, size_t);\n\n";
  let total = ref 0 in
  List.iter
    (fun tag ->
       let fwd = radices (tag, `Fwd) and bwd = radices (tag, `Bwd) in
       let pair = List.filter (fun r -> List.mem r bwd) fwd in
       Printf.printf "/* ── %s ── fwd %d · bwd %d · pair %d */\n" tag (List.length fwd) (List.length bwd) (List.length pair);
       List.iter
         (fun (dirtag, l) ->
            List.iter (fun r -> incr total; Printf.printf "VFFT_IL_DECL(radix%d_z_%s_%s_%s)\n" r tag dirtag isa) l)
         [ "fwd", fwd; "bwd", bwd ];
       let emit_list name l =
         if l <> [] || defined_somewhere tag name
         then (
           Printf.printf "#define VFFT_IL_%s_%s_RADICES(X)" (macro_of_tag tag) name;
           List.iter (fun r -> Printf.printf " X(%d)" r) l;
           Printf.printf "\n")
       in
       emit_list "FWD" fwd;
       emit_list "BWD" bwd;
       emit_list "PAIR" pair;
       Printf.printf "\n")
    tags;
  List.iter
    (fun q ->
       match Corpus.zil_absent q with
       | [] -> ()
       | l ->
         let reasons = List.sort_uniq compare (List.map snd l) in
         Printf.printf "/* %s: %d cells DECLARED ABSENT at %s:\n" q (List.length l) isa;
         List.iter
           (fun why -> Printf.printf " *   %d  %s\n" (List.length (List.filter (fun (_, w) -> w = why) l)) why)
           reasons;
         Printf.printf " */\n")
    (quadrants_of_isa isa);
  Printf.printf "/* %d declarations over %d kinds */\n" !total (List.length tags);
  Printf.printf "\n#endif\n"
;;
