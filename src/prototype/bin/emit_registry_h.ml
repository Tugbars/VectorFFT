(* emit_registry_h.ml — auto-generate the prototype codelet registry.
 *
 * Sibling to emit_profile_h.ml and emit_executor_h.ml. Walks the same
 * radix coverage as scripts/generate_codelets.sh emits, and produces:
 *
 *   1. extern declarations for every emitted codelet symbol
 *   2. A function-pointer typedef shared by all slots
 *   3. vfft_proto_registry_t — struct of arrays of function pointers,
 *      indexed by radix. One slot per (kind, orient, log3?, direction).
 *   4. vfft_proto_registry_init_<isa>(reg) — wires every emitted codelet
 *      into the corresponding slot at runtime. Slots for codelets that
 *      DON'T exist (e.g. R=1024 has no DIF or t1s) stay NULL.
 *   5. Query helpers (vfft_proto_has_n1, etc.)
 *
 * Same naming as gen_radix.ml's post-doc-56 production-aligned
 * convention: radix{R}_{variant}_{isa} (no _gen_inplace_su* suffix).
 *
 * Namespaced as vfft_proto_* so it can coexist with production's
 * src/core/registry.h during the side-by-side period. When prototype
 * replaces production, drop the namespace.
 *
 * Usage:
 *   dune exec bin/emit_registry_h.exe -- --isa avx2   > generated/registry.h
 *   dune exec bin/emit_registry_h.exe -- --isa avx512 > generated/registry_avx512.h
 *)

(* ────────────────────────────────────────────────────────────────────
 * Codelet coverage rules — mirrors scripts/generate_codelets.sh.
 * If the script's family logic changes, update these.
 * ──────────────────────────────────────────────────────────────────── *)

(* Radixes that get the FULL 16+2 variant set (2 n1 + 16 t1/t1s/dit/dif/
 * fwd/bwd × {flat,log3}). Covers primes + composites + small/mid/large
 * pow2 families. *)
let standard_radixes =
  [ 2;  3;  4;  5;  6;  7;  8; 10; 11; 12; 13;
   16; 17; 19; 20; 25; 32; 64; 128; 256; 512 ]

(* R=1024 (xl_pow2 family) emits only 4 codelets per ISA — research-only
 * path per the script's xl_pow2 branch. Not all variants are available. *)
let xl_radix = 1024

let all_radixes = standard_radixes @ [xl_radix]

(* ────────────────────────────────────────────────────────────────────
 * Symbol naming — matches gen_radix.ml's post-doc-56 convention.
 * ──────────────────────────────────────────────────────────────────── *)

type orient = DIT | DIF
type kind   = T1 | T1S
type dir    = Fwd | Bwd

let orient_str = function DIT -> "dit" | DIF -> "dif"
let kind_str   = function T1  -> "t1"  | T1S -> "t1s"
let dir_str    = function Fwd -> "fwd" | Bwd -> "bwd"

let n1_symbol ~r ~direction ~isa =
  Printf.sprintf "radix%d_n1_%s_%s" r (dir_str direction) isa

let t1_symbol ~r ~kind ~orient ~log3 ~direction ~isa =
  let l3 = if log3 then "log3_" else "" in
  Printf.sprintf "radix%d_%s_%s_%s%s_%s"
    r (kind_str kind) (orient_str orient) l3 (dir_str direction) isa

(* ────────────────────────────────────────────────────────────────────
 * Per-codelet existence predicates.
 * ──────────────────────────────────────────────────────────────────── *)

let has_n1 r = List.mem r all_radixes

(* xl_pow2 (R=1024) emits ONLY t1_dit_fwd (flat) and t1_dit_fwd_log3 —
 * research-only path per the script. All other (kind, orient, dir, log3)
 * combinations are absent. Standard radixes get the full matrix. *)
let codelet_exists_t1 ~r ~kind ~orient ~log3:_ ~direction =
  if r = xl_radix then
    kind = T1 && orient = DIT && direction = Fwd
  else
    List.mem r standard_radixes

(* ────────────────────────────────────────────────────────────────────
 * C emission helpers.
 * ──────────────────────────────────────────────────────────────────── *)

let codelet_param_sig =
  "(double *rio_re, double *rio_im,\n\
  \                                 const double *tw_re, const double *tw_im,\n\
  \                                 size_t ios, size_t me)"

let slot_name ~kind ~orient ~log3 ~direction =
  let l3 = if log3 then "_log3" else "" in
  Printf.sprintf "%s_%s%s_%s"
    (kind_str kind) (orient_str orient) l3 (dir_str direction)

let kinds = [T1; T1S]
let orients = [DIT; DIF]
let dirs = [Fwd; Bwd]
let log3s = [false; true]

(* Iterator over the 16 (kind × orient × log3 × dir) combos in stable order. *)
let foreach_t1_combo f =
  List.iter (fun kind ->
    List.iter (fun orient ->
      List.iter (fun log3 ->
        List.iter (fun direction ->
          f ~kind ~orient ~log3 ~direction
        ) dirs
      ) log3s
    ) orients
  ) kinds

(* ────────────────────────────────────────────────────────────────────
 * Section emitters.
 * ──────────────────────────────────────────────────────────────────── *)

let emit_externs ~isa =
  print_endline "/* ─────────────────────────────────────────────────────────────────";
  print_endline " * extern declarations for every emitted codelet symbol.";
  print_endline " * Source of truth: scripts/generate_codelets.sh + the OCaml emitter.";
  print_endline " * ───────────────────────────────────────────────────────────────── */";
  print_endline "";
  (* n1 *)
  List.iter (fun r ->
    if has_n1 r then
      List.iter (fun d ->
        Printf.printf "extern void %s%s;\n" (n1_symbol ~r ~direction:d ~isa) codelet_param_sig
      ) dirs
  ) all_radixes;
  print_endline "";
  (* t1/t1s variants *)
  List.iter (fun r ->
    foreach_t1_combo (fun ~kind ~orient ~log3 ~direction ->
      if codelet_exists_t1 ~r ~kind ~orient ~log3 ~direction then
        Printf.printf "extern void %s%s;\n"
          (t1_symbol ~r ~kind ~orient ~log3 ~direction ~isa)
          codelet_param_sig
    )
  ) all_radixes;
  print_endline ""

let emit_typedef () =
  (* Types are shared across ISAs — guard them with their own header so
   * both registry_avx2.h and registry_avx512.h can be included in the
   * same TU without typedef redefinition errors. *)
  print_endline "#ifndef VFFT_PROTO_REGISTRY_TYPES_H";
  print_endline "#define VFFT_PROTO_REGISTRY_TYPES_H";
  print_endline "";
  print_endline "/* Codelet function pointer type. Same signature for n1 (tw_re/im";
  print_endline " * ignored), t1 (twiddle buffer), t1s (scalar twiddles), and log3";
  print_endline " * (sparse base-twiddle buffer). The codelet body uses tw_re/im";
  print_endline " * according to its variant. */";
  print_endline "typedef void (*vfft_proto_codelet_fn)(double *rio_re, double *rio_im,";
  print_endline "                                      const double *tw_re, const double *tw_im,";
  print_endline "                                      size_t ios, size_t me);";
  print_endline "";
  print_endline "#define VFFT_PROTO_REG_MAX_RADIX 1025";
  print_endline ""

let emit_struct () =
  print_endline "typedef struct {";
  print_endline "    /* No-twiddle (first/last stage) codelets. */";
  print_endline "    vfft_proto_codelet_fn n1_fwd[VFFT_PROTO_REG_MAX_RADIX];";
  print_endline "    vfft_proto_codelet_fn n1_bwd[VFFT_PROTO_REG_MAX_RADIX];";
  print_endline "";
  print_endline "    /* Twiddled inner-stage codelets. 16 slots:";
  print_endline "     *   kind   ∈ { t1, t1s }";
  print_endline "     *   orient ∈ { dit, dif }";
  print_endline "     *   log3?";
  print_endline "     *   direction ∈ { fwd, bwd }";
  print_endline "     */";
  foreach_t1_combo (fun ~kind ~orient ~log3 ~direction ->
    Printf.printf "    vfft_proto_codelet_fn %s[VFFT_PROTO_REG_MAX_RADIX];\n"
      (slot_name ~kind ~orient ~log3 ~direction)
  );
  print_endline "} vfft_proto_registry_t;";
  print_endline ""
  (* Note: types guard #endif emitted later, after the helpers. *)

let emit_init ~isa =
  Printf.printf "/* Initialize the registry for ISA=%s. Codelets that aren't emitted\n" isa;
  Printf.printf " * for a given radix (e.g. R=1024 has no DIF or t1s) leave their slots\n";
  Printf.printf " * NULL. Callers use the query helpers below to test before dispatch. */\n";
  Printf.printf "static inline void vfft_proto_registry_init_%s(vfft_proto_registry_t *reg) {\n" isa;
  print_endline "    memset(reg, 0, sizeof(*reg));";
  print_endline "";
  (* n1 *)
  print_endline "    /* n1 codelets */";
  List.iter (fun r ->
    if has_n1 r then begin
      Printf.printf "    reg->n1_fwd[%d] = %s;\n" r (n1_symbol ~r ~direction:Fwd ~isa);
      Printf.printf "    reg->n1_bwd[%d] = %s;\n" r (n1_symbol ~r ~direction:Bwd ~isa)
    end
  ) all_radixes;
  print_endline "";
  (* t1 variants, one block per slot *)
  foreach_t1_combo (fun ~kind ~orient ~log3 ~direction ->
    let slot = slot_name ~kind ~orient ~log3 ~direction in
    let lines = List.filter_map (fun r ->
      if codelet_exists_t1 ~r ~kind ~orient ~log3 ~direction then
        Some (Printf.sprintf "    reg->%s[%d] = %s;"
                slot r (t1_symbol ~r ~kind ~orient ~log3 ~direction ~isa))
      else None
    ) all_radixes in
    if lines <> [] then begin
      Printf.printf "    /* %s */\n" slot;
      List.iter print_endline lines;
      print_endline ""
    end
  );
  print_endline "}"

let emit_helpers () =
  print_endline "";
  print_endline "/* ─────────────────────────────────────────────────────────────────";
  print_endline " * Query helpers — return non-zero if the codelet exists in the";
  print_endline " * registry for the given radix.";
  print_endline " * ───────────────────────────────────────────────────────────────── */";
  print_endline "";
  print_endline "static inline int vfft_proto_has_n1(const vfft_proto_registry_t *reg, int r) {";
  print_endline "    return r > 0 && r < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[r] != NULL;";
  print_endline "}";
  print_endline "";
  foreach_t1_combo (fun ~kind ~orient ~log3 ~direction ->
    let slot = slot_name ~kind ~orient ~log3 ~direction in
    if direction = Fwd then begin  (* one helper per (kind, orient, log3); checks fwd *)
      Printf.printf "static inline int vfft_proto_has_%s_%s%s(const vfft_proto_registry_t *reg, int r) {\n"
        (kind_str kind) (orient_str orient) (if log3 then "_log3" else "");
      Printf.printf "    return r > 0 && r < VFFT_PROTO_REG_MAX_RADIX && reg->%s[r] != NULL;\n" slot;
      print_endline "}"
    end
  )

let emit_header_file ~isa =
  let isa_upper = String.uppercase_ascii isa in
  Printf.printf "/* registry_%s.h — auto-generated by bin/emit_registry_h.ml.\n" isa;
  print_endline " * DO NOT EDIT BY HAND. Regenerate via:";
  Printf.printf " *   dune exec bin/emit_registry_h.exe -- --isa %s \\\n" isa;
  Printf.printf " *     > generated/registry_%s.h\n" isa;
  print_endline " *";
  Printf.printf " * Prototype codelet registry for ISA=%s. Function-pointer table\n" isa;
  print_endline " * indexed by radix, one slot per (kind × orient × log3? × direction).";
  print_endline " * Codelets that aren't emitted for a given radix leave their slots NULL.";
  print_endline " *";
  print_endline " * Consumers usually include the dispatcher `registry.h` rather than";
  print_endline " * this file directly — that picks the right per-ISA registry via";
  print_endline " * ifdef. Include this file directly only if you need to bypass the";
  print_endline " * dispatch (e.g. building a multi-ISA fat binary).";
  print_endline " *";
  print_endline " * Namespaced as vfft_proto_* so it coexists with production's";
  print_endline " * src/core/registry.h. */";
  Printf.printf "#ifndef VFFT_PROTO_REGISTRY_%s_H\n" isa_upper;
  Printf.printf "#define VFFT_PROTO_REGISTRY_%s_H\n" isa_upper;
  print_endline "";
  print_endline "#include <stddef.h>";
  print_endline "#include <string.h>";
  print_endline "";
  emit_externs ~isa;
  emit_typedef ();
  emit_struct ();
  emit_helpers ();
  (* Close the shared types guard. Below this, the init function is the
   * one ISA-specific bit of the file. *)
  print_endline "";
  print_endline "#endif /* VFFT_PROTO_REGISTRY_TYPES_H */";
  print_endline "";
  emit_init ~isa;
  print_endline "";
  Printf.printf "#endif /* VFFT_PROTO_REGISTRY_%s_H */\n" isa_upper

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
