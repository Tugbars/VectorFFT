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
  else if isa = "avx2_f32" then [ "zil-f32-avx2" ]
  else if isa = "avx512_f32" then [ "zil-f32-avx512" ]
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
   | l -> failwith ("emit_il_registry: usage [--isa avx2|avx512|avx2_f32|avx512_f32], got " ^ String.concat " " l));
  let isa = !isa in
  if not (List.mem isa (Corpus.zil_isas @ Corpus.zil_f32_isas))
  then failwith ("emit_il_registry: unknown isa " ^ isa ^ " (avx2 | avx512 | avx2_f32 | avx512_f32)");
  (* FP32 (2026-11-10): the avx2_f32 / avx512_f32 registries carry float pointers, VW in
     floats, and every list macro of the FP64 universe (empty where FP32 has no such kernel)
     beside their own, so the core compiles against either header. The FP64 registries keep
     the FP64 universe alone: byte-identical. *)
  let f32 = List.mem isa Corpus.zil_f32_isas in
  let up = String.uppercase_ascii isa in
  let tables =
    List.map (fun i -> i, table_of_isa i) (if f32 then Corpus.zil_isas @ Corpus.zil_f32_isas else Corpus.zil_isas)
  in
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
  if f32
  then
    Printf.printf
      "#define VFFT_IL_VW %d          /* floats per vector: the twiddle-record width */\n"
      (match isa with "avx2_f32" -> 8 | _ -> 16)
  else
    Printf.printf "#define VFFT_IL_VW %d          /* doubles per vector: the twiddle-record width */\n" (match isa with "avx2" -> 4 | _ -> 8);
  Printf.printf "#define VFFT_IL_SYM(stem) stem##_%s\n\n" isa;
  Printf.printf "/* the frozen 11-arg z ABI (vfft_il2p_fn's pointee) */\n";
  Printf.printf "#define VFFT_IL_DECL(SYM) \\\n";
  let e = if f32 then "float" else "double" in
  Printf.printf "  extern void SYM(const %s *, const %s *, %s *, %s *, \\\n" e e e e;
  Printf.printf "                  const %s *, const %s *, \\\n                  size_t, size_t, size_t, size_t, size_t);\n\n" e e;
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
  (* THE ROW FAMILY'S KERNEL PER RADIX (2026-11-08): the batched row kernel the 2D tier's plain row
     passes call (vfft_il_row_fn). The corpus holds ONE winner of the 2026-11-08 race per radix besides
     n1ccs -- row / rowb<m1><p1> / rowswap -- and that pair is the choice; a radix with none keeps the
     n1ccs pair (the lane-load construction, which also serves the turn route's turned stores). Two
     winners at one radix is a corpus error, not a tie-break. *)
  let pair_of t = List.filter (fun r -> List.mem r (radices (t, `Bwd))) (radices (t, `Fwd)) in
  (* two choices: the PLAIN row kernels (row, rowb<m1><p1>, rowswap; stores row-major) and their TURNED twins
     (the tag ends in t: rowt, rowb44t, rowswapt; stores point-major for the 2D turn route). n1ccs serves both
     roles where no twin exists (its store honours the turned pitches). *)
  let turned t = t <> "n1ccs" && starts_with "row" t && t.[String.length t - 1] = 't' in
  let choice name keep =
    let row_tags = List.filter (fun t -> t = "n1ccs" || (starts_with "row" t && keep t)) tags in
    let row_radices = List.sort_uniq compare (List.concat_map pair_of row_tags) in
    Printf.printf "#define %s(X)" name;
    List.iter
      (fun r ->
         match List.filter (fun t -> t <> "n1ccs" && List.mem r (pair_of t)) row_tags with
         | [ t ] -> Printf.printf " X(%d, %s)" r t
         | [] -> if List.mem r (pair_of "n1ccs") then Printf.printf " X(%d, n1ccs)" r
         | l ->
           failwith
             (Printf.sprintf
                "emit_il_registry: radix %d has %d row kernels besides n1ccs for %s (%s) at %s: the corpus holds one winner per radix"
                r (List.length l) name (String.concat ", " l) isa))
      row_radices;
    Printf.printf "\n"
  in
  Printf.printf "/* ── the row family: X(radix, tag) = the batched row kernel of the plain row passes, and the\n";
  Printf.printf " * turned-store twin for the 2D turn route ── */\n";
  choice "VFFT_IL_ROW_CHOICE" (fun t -> not (turned t));
  choice "VFFT_IL_ROWT_CHOICE" turned;
  Printf.printf "\n";
  (* THE FUSED K=1 FORMS (FP32, 2026-11-11): X(N, form, tag) = the mono route's form `form` >= 1 at N is
     the kernel pair radix<N>_z_<tag> (Corpus.zil_f32_mono_shapes: the order there IS the form number the
     wisdom banks). Only the pairs this ISA has; an ISA with none defines the list empty. *)
  if f32
  then (
    Printf.printf "/* ── the fused K=1 forms: X(N, form, tag) = the mono route's form at N (form 0 = the solo n1) ── */\n";
    Printf.printf "#define VFFT_IL_MONO_FUSED(X)";
    List.iter
      (fun (n, shapes) ->
         List.iteri
           (fun i (a, m) ->
              let tag = Printf.sprintf "mono%dx%d" a m in
              if List.mem n (pair_of tag) then Printf.printf " X(%d, %d, %s)" n (i + 1) tag)
           shapes)
      Corpus.zil_f32_mono_shapes;
    Printf.printf "\n\n");
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
