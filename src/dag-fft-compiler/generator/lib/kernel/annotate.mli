(* annotate.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. DAG annotation.
   Generated from the inferred signature; trim = later per-module work. *)

type entry =
  { output_for : Expr.elem_ref option
  ; alg_node : Ir.t
  }

type scope =
  | Leaf of entry
  | Block of
      { decls : int list
      ; body : scope list
      }

val compute_lifetimes : entry array -> (int, int) Hashtbl.t * (int, int) Hashtbl.t
val min_block_size : int

val scope_range
  :  entry array
  -> (int, int) Hashtbl.t
  -> (int, int) Hashtbl.t
  -> int
  -> int
  -> scope

val annotate : (Expr.elem_ref option * Ir.t) list -> scope
val strip_const_prefix : Isa.t -> string -> string

val emit_scope
  :  Isa.t
  -> Buffer.t
  -> (Ir.t -> string)
  -> (Expr.elem_ref -> Ir.t -> string)
  -> scope
  -> unit
