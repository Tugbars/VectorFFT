(* simd.mli -- M7.  See simd.ml: the feature-blind strided transpose
   lattices (plain re/im arms), moved byte-verbatim from emit_c. *)

val load_transpose_4x4 : buf:Buffer.t -> groups:int -> unit
val load_transpose_8x8 : buf:Buffer.t -> groups:int -> unit
val store_transpose_4x4 : buf:Buffer.t -> groups:int -> unit
val store_transpose_8x8 : buf:Buffer.t -> groups:int -> unit
