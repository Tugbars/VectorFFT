(* corpus.mli — M10 (also the coverage interface M9 deferred here).
   THE corpus: typed cells over the Codelet descriptor; the family
   matrices are private scaffold.  Laws (round-trip verbatim, per-dir
   file + global canonical-argv uniqueness) fire lazily at first
   cells/files use — in gen_set and the registry emitters. *)

type cell =
  { file : string
  ; c : Codelet.t
  }

val quadrants : string list

(* the FP32 quadrants (2026-11-10): zil-f32-avx2 / zil-f32-avx512 -- the cil kernels and the
   ZTURN-T stage kernels -- emitted by `gen_set f32` (never part of `all`); their isas
   avx2_f32 / avx512_f32 *)
val quadrants_f32 : string list
val zil_f32_isas : string list
(* the FP32 fused K=1 kernels: per N its shapes (A, M) in FORM order (form f >= 1 = the f-th) *)
val zil_f32_mono_shapes : (int * (int * int) list) list
val dir_of_quadrant : string -> string
val zil_folder : string -> string option   (* a zil file's folder by its kind (2026-09-24) *)
val dir_of_file : string -> string -> string   (* quadrant -> file name -> the folder to write it in *)

(** Typed corpus of one quadrant; forces the laws on first use. *)
val cells : string -> cell list

(** (filename, argv_tail sans --emit-c) — DERIVED via Codelet.to_argv;
    byte-equal to the historical Coverage.files by the round-trip law. *)
val files : string -> (string * string list) list
val zil_absent : string -> (string * string) list   (* a derived zil quadrant: (file, why the generator cannot emit it at that ISA) *)
val zil_isas : string list
