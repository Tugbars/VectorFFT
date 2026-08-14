(* simplify.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

val dedup_sub_pairs :
  (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val extract_coefficient : Ir.t -> float * Ir.t
val collect_m : (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val distribute_term : depth:int -> int * Ir.t -> (int * Ir.t) list
val collect_terms_to_tree : (int * Ir.t) list -> Ir.t
val count_ir_nodes : Ir.t -> int
val deep_collect :
  ?depth_limit:int ->
  (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val lift_sub_neg_mul :
  (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val factor_common_muls :
  ?aggressive:bool ->
  (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val factor_by_atom :
  ?aggressive:bool ->
  (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val share_subsums :
  ?aggressive:bool ->
  (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
val transpose : (Expr.elem_ref * Ir.t) list -> (Expr.elem_ref * Ir.t) list
