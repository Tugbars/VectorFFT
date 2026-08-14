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
    | CIn _ | CLoad _ -> []
    | CNeg a | CRotNI a | CRotPI a | CLo a | CHi a -> [ a ]
    | CStore (_, v) -> [ v ]
    | CAdd (a, b) | CSub (a, b) | CRotAdd (a, b) -> [ a; b ]
    | CTurn (a, b, _) -> [ a; b ]
    | CFmaC (_, x, e) | CFnmaC (_, x, e) -> [ x; e ]
    | CTwC (_, _, x) | CTwV (_, x) | CTwL (_, x) -> [ x ]
  ;;

  (* Cycle costs, same convention as schedule.ml's real-valued table.
     CRotNI is a shuffle + xor (both ~1c, dependent) — charged as add
     latency, matching how NK_Neg (a sign-flip xor) is charged. CTwC is a
     mul + fma chain, dominated by fma latency, exactly like NK_Cmul*. *)
  let latency (uarch : Uarch.t) (e : t) : int =
    match e.node with
    | CIn _ | CLoad _ -> uarch.load_l1_latency
    | CStore _ -> uarch.store_latency
    | CAdd _ | CSub _ | CRotAdd _ -> uarch.add_latency
    (* CNeg is one xor — charged like NK_Neg on the real side. *)
    | CNeg _ -> uarch.add_latency
    (* CTurn = one permute2f128; CLo is a free cast, CHi one extract —
       charged like the other lane ops. *)
    | CTurn _ | CLo _ | CHi _ -> uarch.add_latency
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
    | CIn _ | CLoad _ -> true
    | _ -> false
  ;;

  (* CStore made this hook LIVE (2026-08-09, complete-IR arc) — the B2
     accessor is no longer trivially false. Whether stores are scheduled is
     the emitter's placement policy; the IR and scheduler now support it. *)
  let is_store (e : t) =
    match e.node with
    | CStore _ -> true
    | _ -> false
  ;;

  (* No standalone const nodes: real coefficients ride inside CFmaC/CTwC as
     emit-time set1/VLIT operands, so there is nothing for the lookahead
     leaf policy to defer. *)
  let is_const (_ : t) = false

  let kind_char (e : t) =
    match e.node with
    | CIn _ | CLoad _ -> 'L'
    | CStore _ -> 'S'
    | CAdd _ | CSub _ | CNeg _ | CRotAdd _ -> 'A'
    | CTurn _ | CLo _ | CHi _ -> 'R'
    | CRotNI _ | CRotPI _ -> 'R'
    | CFmaC _ | CFnmaC _ -> 'F'
    | CTwC _ | CTwV _ -> 'X'
    | CTwL _ -> 'T'
  ;;
end

module Sched = Schedule.Make (Node)
