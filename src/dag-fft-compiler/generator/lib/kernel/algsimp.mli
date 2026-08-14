(* algsimp.mli — M9b: full-surface interface, derived from the inferred
   signature (ocamlc -i) at the M9 library split.  Freezes the public
   surface — additions are now deliberate.  Trim opportunistically. *)

type spill_tag_marker = { slot : int; re_tag : int; im_tag : int; }
val lift_spill_markers :
  ?reassoc:bool -> Dft.spill_marker list -> spill_tag_marker list
val butterfly_share_mul :
  ?frozen_tags:(int, unit) Hashtbl.t option ->
  (Expr.elem_ref * Ir.t) list ->
  (Expr.elem_ref * Ir.t) list * (int, int) Hashtbl.t
val duplicate_uncse :
  ?span_s:int ->
  ?cap:int ->
  ?maxcost:int ->
  schedule:((Expr.elem_ref * Ir.t) list -> Ir.t list) ->
  (Expr.elem_ref * Ir.t) list ->
  (Expr.elem_ref * Ir.t) list * (int, unit) Hashtbl.t *
  (int, int) Hashtbl.t * (int * int) list
type dag_stats = {
  total_nodes : int;
  consts : int;
  loads : int;
  negs : int;
  adds : int;
  subs : int;
  muls : int;
  cmuls : int;
  fmas : int;
  arithmetic_ops : int;
}
val stats_reachable : Ir.t list -> dag_stats
val string_of_stats : dag_stats -> string
val string_of_node_kind : Ir.node_kind -> string
val print_dag : (Expr.elem_ref * Ir.t) list -> string
type spill_info = {
  re_slot : (int, int) Hashtbl.t;
  im_slot : (int, int) Hashtbl.t;
  num_slots : int;
  fused_slots : (int, unit) Hashtbl.t;
  ct_n1 : int;
  ct_n2 : int;
}
val make_spill_info :
  ?ct:int * int -> ?fuse:int -> spill_tag_marker list -> spill_info
