(* regalloc.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. Linear-scan allocation + peak-live.
   Generated from the inferred signature; trim = later per-module work. *)

type assignment = Reg of string | Spilled of int | Default
type reload_decl = {
  reload_tag : int;
  reload_name : string;
  reload_reg : string;
  reload_slot : int;
}
type allocation = {
  isa : Isa.t;
  assign : (int, assignment) Hashtbl.t;
  num_spill_slots : int;
  reload_sites : (int, reload_decl list) Hashtbl.t;
  spill_sites : (int, (int * int) list) Hashtbl.t;
  spilled_of_tag : (int, int) Hashtbl.t;
  name_overrides : (int * int, string) Hashtbl.t;
}
val allocate_stub : isa:Isa.t -> scheduled:Ir.t list -> allocation
val lookup : allocation -> int -> assignment
val count_bindings : allocation -> int * int
val count_spilled : allocation -> int
type live_info = {
  peak_live : int;
  peak_at : int;
  n_nodes : int;
  budget : int;
  fits : bool;
}
val peak_live_analysis : isa:Isa.t -> scheduled:Ir.t list -> live_info
val format_live_info : live_info -> string
type alloc_result = Allocated of allocation | Overflow of int
val reg_name_of_isa : Isa.t -> int -> string
val allocate_linear_scan :
  isa:Isa.t ->
  scheduled:Ir.t list ->
  budget:int ->
  ?skip_tags:(int, unit) Hashtbl.t option ->
  ?inline_set:(int, unit) Hashtbl.t option ->
  ?force_last_use:(int, int) Hashtbl.t option -> unit -> alloc_result
val allocate_with_spilling :
  isa:Isa.t ->
  scheduled:Ir.t list ->
  budget:int ->
  ?skip_tags:(int, unit) Hashtbl.t option ->
  ?inline_set:(int, unit) Hashtbl.t option ->
  ?force_last_use:(int, int) Hashtbl.t option -> unit -> alloc_result
type regalloc_input = {
  scheduled : Ir.t list;
  inline_set : (int, unit) Hashtbl.t option;
  force_last_use : (int, int) Hashtbl.t option;
}
val prepare_for_simple_codelet :
  raw_scheduled:Ir.t list ->
  assigns:(Expr.elem_ref * Ir.t) list ->
  ?inline_set:(int, unit) Hashtbl.t option -> unit -> regalloc_input
val prepare_for_simple_codelet_from_oref :
  raw_scheduled:(Expr.elem_ref option * Ir.t) list ->
  assigns:(Expr.elem_ref * Ir.t) list ->
  ?inline_set:(int, unit) Hashtbl.t option -> unit -> regalloc_input
val allocate :
  isa:Isa.t ->
  scheduled:Ir.t list ->
  ?budget:int ->
  ?skip_tags:(int, unit) Hashtbl.t option ->
  ?inline_set:(int, unit) Hashtbl.t option ->
  ?force_last_use:(int, int) Hashtbl.t option -> unit -> alloc_result
