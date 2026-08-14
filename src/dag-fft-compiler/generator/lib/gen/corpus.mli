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
val dir_of_quadrant : string -> string
val ip_radices : int list

(** Typed corpus of one quadrant; forces the laws on first use. *)
val cells : string -> cell list

(** (filename, argv_tail sans --emit-c) — DERIVED via Codelet.to_argv;
    byte-equal to the historical Coverage.files by the round-trip law. *)
val files : string -> (string * string list) list
