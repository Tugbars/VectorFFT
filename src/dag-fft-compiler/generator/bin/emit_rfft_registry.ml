(* emit_rfft_registry.ml — auto-generate the rfft codelet registry.
 *
 * Sibling to emit_registry_h.ml (which does the c2c families). Walks the
 * SAME single source of truth — Vfft_v2.Coverage.files "rfft-<isa>" — so
 * the registry can never drift from what gen_set actually emits. Adding a
 * codelet to the rfft quadrant in coverage.ml automatically gives it an
 * extern + a populated registry slot here; nothing is hand-written.
 *
 * Produces a header defining:
 *   1. extern declarations for every rfft codelet symbol in coverage
 *   2. rfft_register_all_<isa>(rfft_codelets_t *reg) — wires every
 *      emitted codelet into its ABI-correct, radix-indexed slot.
 *      Slots for codelets not in coverage stay NULL (caller calloc's).
 *
 * Design norm (this is the point): the slot a codelet lands in is chosen
 * by its KIND, and each kind's slot has that kind's function-pointer ABI.
 * A 4-pointer packed hc2c and a 6-pointer natural hc2c are DIFFERENT
 * kinds -> different slots -> the C type system rejects a mis-wire. This
 * is what prevents the packed-into-natural mistake.
 *
 * Usage:
 *   dune exec bin/emit_rfft_registry.exe -- --isa avx512 > generated/rfft_registry_avx512.h
 *   dune exec bin/emit_rfft_registry.exe -- --isa avx2   > generated/rfft_registry_avx2.h
 *)

(* ── codelet identity, derived from the coverage filename ───────────────
 * Filenames are radix{R}_{kind...}_{dir}_{isa}.c. We classify each to its
 * ABI bucket. The symbol name is the filename minus ".c" (gen_main emits
 * symbol == filename stem for the rfft family). *)

type abi =
  | R2cf (* rfft_r2cf_fn         : leaf / interior k0          *)
  | Hc2hc (* rfft_hc_fn           : twiddle stage (dit/dif/log3)*)
  | Hc2hc_rng (* rfft_hc_rng_fn       : ranged twiddle stage        *)
  | Hc2c_nat (* rfft_hc2c_nat_fn     : natural terminator (+log3)  *)
  | Hc2c_nat_rng (* rfft_hc2c_nat_rng_fn : ranged natural terminator   *)
  | Hc2c_packed (* rfft_hc_fn-shaped packed last stage; no slot yet   *)
  | Unknown

(* Order matters: test the most specific tokens first (rng before plain,
 * nat before packed hc2c). *)
let classify (fname : string) : abi =
  let has s =
    (* substring test *)
    let ls = String.length s and lf = String.length fname in
    let rec go i = i + ls <= lf && (String.sub fname i ls = s || go (i + 1)) in
    ls > 0 && go 0
  in
  if has "_r2cf_" then R2cf
  else if has "_hc2c_nat_rng_" then Hc2c_nat_rng
  else if has "_hc2c_nat_" then Hc2c_nat
  else if has "_hc2c_" then Hc2c_packed
  else if has "_hc2hc_" && has "_rng_" then Hc2hc_rng
  else if has "_hc2hc_" then Hc2hc
  else Unknown

(* radix prefix: "radix%d_..." -> %d *)
let radix_of (fname : string) : int option =
  if String.length fname > 5 && String.sub fname 0 5 = "radix" then
    let rec digits i acc =
      if i < String.length fname && fname.[i] >= '0' && fname.[i] <= '9' then
        digits (i + 1) ((acc * 10) + (Char.code fname.[i] - Char.code '0'))
      else if i = 5 then None
      else Some acc
    in
    digits 5 0
  else None

(* Is this a log3 variant? (affects which slot, not which ABI) *)
let is_log3 (fname : string) : bool =
  let s = "_log3_" and lf = String.length fname in
  let ls = String.length s in
  let rec go i = i + ls <= lf && (String.sub fname i ls = s || go (i + 1)) in
  go 0

(* Is this a DIF (decimation-in-frequency) variant? Orientation is part
 * of the codelet identity, so DIT and DIF get DISTINCT slots even though
 * they share an ABI — otherwise DIF silently overwrites DIT. The executor
 * does not call DIF yet (the dif slots stay populated-but-unused until a
 * DIF-using executor or planner path wants them). *)
let is_dif (fname : string) : bool =
  let s = "_dif_" and lf = String.length fname in
  let ls = String.length s in
  let rec go i = i + ls <= lf && (String.sub fname i ls = s || go (i + 1)) in
  go 0

(* slot field name in rfft_codelets_t for (abi, log3, dif). Returns None
 * for kinds with no executor slot yet (packed hc2c), or Unknown. *)
let slot_of (abi : abi) (log3 : bool) (dif : bool) : string option =
  match (abi, log3, dif) with
  | R2cf, _, _ -> Some "r2cf"
  | Hc2hc, false, false -> Some "hc2hc"
  | Hc2hc, false, true -> Some "hc2hc_dif"
  | Hc2hc, true, false -> Some "hc2hc_log3"
  | Hc2hc, true, true -> Some "hc2hc_dif_log3"
  | Hc2hc_rng, _, _ -> Some "hc2hc_rng"
  | Hc2c_nat, false, _ -> Some "hc2c"
  | Hc2c_nat, true, _ -> Some "hc2c_log3"
  | Hc2c_nat_rng, _, _ -> Some "hc2c_rng"
  | Hc2c_packed, _, _ -> None (* no executor consumer yet *)
  | Unknown, _, _ -> None

let () =
  let isa = ref "avx512" in
  Arg.parse
    [ ("--isa", Arg.Set_string isa, "target ISA (avx2|avx512)") ]
    (fun _ -> ())
    "emit_rfft_registry --isa <avx2|avx512>";
  let isa = !isa in
  let quadrant = "rfft-" ^ isa in
  let files = Vfft_v2.Coverage.files quadrant in

  (* (symbol, slot, radix) for every classifiable, slotted codelet *)
  let entries =
    List.filter_map
      (fun (fname_c, _argv) ->
        let stem =
          if Filename.check_suffix fname_c ".c" then
            Filename.chop_suffix fname_c ".c"
          else fname_c
        in
        match
          (radix_of stem, slot_of (classify stem) (is_log3 stem) (is_dif stem))
        with
        | Some r, Some slot -> Some (stem, slot, r)
        | _ -> None)
      files
  in

  Printf.printf "/* AUTO-GENERATED by bin/emit_rfft_registry.ml from\n";
  Printf.printf " * Coverage.files \"%s\". DO NOT EDIT BY HAND.\n" quadrant;
  Printf.printf " * Regenerate: dune build (promote rule) or\n";
  Printf.printf " *   dune exec bin/emit_rfft_registry.exe -- --isa %s\n" isa;
  Printf.printf " *\n";
  Printf.printf
    " * Each codelet is wired into the slot chosen by its KIND; the slot's\n";
  Printf.printf
    " * function-pointer type IS that kind's ABI, so a wrong-ABI wire is a\n";
  Printf.printf
    " * compile error. Coverage-driven: a new rfft codelet auto-appears here. */\n";
  Printf.printf "#ifndef VFFT_RFFT_REGISTRY_%s_H\n" (String.uppercase_ascii isa);
  Printf.printf "#define VFFT_RFFT_REGISTRY_%s_H\n\n"
    (String.uppercase_ascii isa);
  Printf.printf "#include \"rfft.h\"\n\n";

  (* externs, in coverage order. Each declared with its ABI's real
   * parameter list (not bare ()), so the prototype itself enforces the
   * signature and the registrar assignment needs no unsafe cast. *)
  let proto_for slot sym =
    match slot with
    | "r2cf" ->
        Printf.sprintf
          "extern void %s(const double*, double*, double*, ptrdiff_t, \
           ptrdiff_t, ptrdiff_t, size_t);"
          sym
    | "hc2hc" | "hc2hc_log3" | "hc2hc_dif" | "hc2hc_dif_log3" ->
        Printf.sprintf
          "extern void %s(const double*, const double*, double*, double*, \
           const double*, const double*, ptrdiff_t, ptrdiff_t, size_t);"
          sym
    | "hc2hc_rng" ->
        Printf.sprintf
          "extern void %s(const double*, const double*, double*, double*, \
           const double*, const double*, ptrdiff_t, ptrdiff_t, ptrdiff_t, \
           ptrdiff_t, int, size_t);"
          sym
    | "hc2c" | "hc2c_log3" ->
        Printf.sprintf
          "extern void %s(const double*, const double*, double*, double*, \
           double*, double*, const double*, const double*, ptrdiff_t, \
           ptrdiff_t, ptrdiff_t, size_t);"
          sym
    | "hc2c_rng" ->
        Printf.sprintf
          "extern void %s(const double*, const double*, double*, double*, \
           double*, double*, const double*, const double*, ptrdiff_t, \
           ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, int, size_t);"
          sym
    | _ -> Printf.sprintf "extern void %s();" sym
  in
  Printf.printf "/* extern declarations (every rfft codelet in coverage) */\n";
  List.iter (fun (sym, slot, _r) -> print_endline (proto_for slot sym)) entries;
  Printf.printf "\n";

  (* the registrar: one assignment per codelet. The extern prototype above
   * already matches the slot's ABI, so the assignment is a plain, checked
   * function-pointer store — a wrong-ABI codelet would fail to compile. *)
  Printf.printf
    "static inline void rfft_register_all_%s(rfft_codelets_t *reg)\n{\n" isa;
  List.iter
    (fun (sym, slot, r) -> Printf.printf "    reg->%s[%d] = %s;\n" slot r sym)
    entries;
  Printf.printf "}\n\n";
  Printf.printf "#endif\n"
