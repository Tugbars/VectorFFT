(* schedule.mli — M2 (generator_lib_architecture.md §10): the compiler-checked
   MODULE CARD. SCHED_NODE + Make — the library's ONE functor (3 instantiations, both stacks).
   Generated from the inferred signature; trim = later per-module work. *)

val order_source : string option ref
val injection_log : string list ref
val log_injection : string -> unit
val resolve_order_source : unit -> string option
val dag_signature : (int * int list) list -> string
val read_order_file : string -> int list * string option
val node_latency : Uarch.t -> Ir.t -> int

module type SCHED_NODE = sig
  type payload

  type t =
    { tag : int
    ; node : payload
    }

  val preds : t -> t list
  val latency : Uarch.t -> t -> int
  val is_load : t -> bool
  val is_store : t -> bool
  val is_const : t -> bool
  val kind_char : t -> char
end

module Ir_node : sig
  type payload = Ir.node_kind

  type t = Ir.t =
    { tag : int
    ; node : payload
    }

  val preds : t -> t list
  val latency : Uarch.t -> t -> int
  val is_load : t -> bool
  val is_store : t -> bool
  val is_const : t -> bool
  val kind_char : t -> char
end

module Make : functor (N : SCHED_NODE) -> sig
  val compute_cp_dist : Uarch.t -> N.t list -> N.t list -> (int, int) Hashtbl.t
  val compute_su_number : N.t list -> (int, int) Hashtbl.t

  val su_schedule
    :  Uarch.t
    -> (Expr.elem_ref * N.t) list
    -> (Expr.elem_ref option * N.t) list
end

val compute_cp_dist : Uarch.t -> Ir_node.t list -> Ir_node.t list -> (int, int) Hashtbl.t
val compute_su_number : Ir_node.t list -> (int, int) Hashtbl.t

val su_schedule
  :  Uarch.t
  -> (Expr.elem_ref * Ir_node.t) list
  -> (Expr.elem_ref option * Ir_node.t) list

val su_schedule_subset
  :  Uarch.t
  -> gh:bool
  -> subset:Ir_node.t list
  -> sinks:Ir_node.t list
  -> Ir_node.t list
