(* cx_sched.ml — the shared SR (Starve-Retire) scheduler over the cx IR.
 * Split out of codelet_cil.ml (Phase 0, 2026-08-09, byte-identity gated).
 * MODULE CARD
 * ROLE: Schedule.SCHED_NODE instance for cx_kind + Sched = Schedule.Make.
 * DEPS: Cx_ir, Schedule (functor), Uarch (latencies). *)

open Cx_ir

(* ═══════════════════════════════════════════════════════════════
 *  SCHEDULER INSTANTIATION — the shared SR scheduler over this IR
 * ═══════════════════════════════════════════════════════════════ *)

module Node : Schedule.SCHED_NODE with type payload = cx_kind and type t = t = struct
  type payload = cx_kind

  type nonrec t = t =
    { tag : int
    ; node : payload
    }

  let preds (e : t) : t list =
    match e.node with
    | CIn _ -> []
    | CNeg a | CRotNI a | CRotPI a -> [ a ]
    | CAdd (a, b) | CSub (a, b) -> [ a; b ]
    | CFmaC (_, x, e) | CFnmaC (_, x, e) -> [ x; e ]
    | CTwC (_, _, x) | CTwV (_, x) | CTwL (_, x) -> [ x ]
  ;;

  (* Cycle costs, same convention as schedule.ml's real-valued table.
     CRotNI is a shuffle + xor (both ~1c, dependent) — charged as add
     latency, matching how NK_Neg (a sign-flip xor) is charged. CTwC is a
     mul + fma chain, dominated by fma latency, exactly like NK_Cmul*. *)
  let latency (uarch : Uarch.t) (e : t) : int =
    match e.node with
    | CIn _ -> uarch.load_l1_latency
    | CAdd _ | CSub _ -> uarch.add_latency
    (* CNeg is one xor — charged like NK_Neg on the real side. *)
    | CNeg _ -> uarch.add_latency
    | CRotNI _ | CRotPI _ -> uarch.add_latency
    | CFmaC _ | CFnmaC _ -> uarch.fma_latency
    | CTwC _ | CTwV _ -> uarch.fma_latency
    | CTwL _ ->
      (* table load feeds the same mul+fma chain; the load is off the
         critical path in steady state, so charge the arithmetic. *)
      uarch.fma_latency
  ;;

  let is_load (e : t) =
    match e.node with
    | CIn _ -> true
    | _ -> false
  ;;

  (* No store node kind in the complex IR either: stores are the assigns
     list (B2's SCHED_NODE store accessor — trivially false here). *)
  let is_store (_ : t) = false

  (* No standalone const nodes: real coefficients ride inside CFmaC/CTwC as
     emit-time set1/VLIT operands, so there is nothing for the lookahead
     leaf policy to defer. *)
  let is_const (_ : t) = false

  let kind_char (e : t) =
    match e.node with
    | CIn _ -> 'L'
    | CAdd _ | CSub _ | CNeg _ -> 'A'
    | CRotNI _ | CRotPI _ -> 'R'
    | CFmaC _ | CFnmaC _ -> 'F'
    | CTwC _ | CTwV _ -> 'X'
    | CTwL _ -> 'T'
  ;;
end

module Sched = Schedule.Make (Node)
