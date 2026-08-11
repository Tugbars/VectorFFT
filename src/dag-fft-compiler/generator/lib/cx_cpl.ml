(* cx_cpl.ml — CPL: Critical-Path List scheduler for the cx IR, the
 * ILP-objective alternative to SR (cx_sched / Schedule.Make).
 *
 * WHY (2026-08-11, emit_race.c): SR's starve-retire objective minimizes
 * live ranges — the right goal when bodies spilled. Tangent-interior
 * bodies spill ~0, and SR's depth-first order leaves ~16%-of-runtime in
 * dispatch slack vs the hand-scheduled proof kernel (hand +8c above its
 * port bound, SR-emitted +17c). The hand kernel is an existence proof
 * that a latency-aware order is worth ~25% on these bodies.
 *
 * ALGORITHM: classic list scheduling by critical-path height, with a
 * LIVE-SET CAP so gcc's register allocator is not flooded (the braid
 * refutation: unbounded width-first interleave lost +11.9% through
 * spills). Deterministic.
 *   height(n) = latency(n) + max over users(height)   (0 users -> lat)
 *   ready     = all preds scheduled
 *   pick      = if live >= cap: max (freed, height, -ready_t, -tag)
 *               else:           max (height, freed, -ready_t, -tag)
 *   where freed(n) = #preds this node retires (it is their last
 *   unscheduled user), ready_t = max pred finish time (model clock).
 *
 * Selected via VFFT_CX_SCHED=cpl (codelet_cil dispatcher); cap via
 * VFFT_CX_CPL_CAP (default 14 of the 16 ymm — constants/masks ride in
 * rip-memory operands, not registers). Default path remains SR: every
 * shipped emission is byte-identical unless the knob is set. *)

open Cx_ir

let cap () =
  match Sys.getenv_opt "VFFT_CX_CPL_CAP" with
  | Some s -> (try int_of_string s with _ -> 14)
  | None -> 14
;;

let schedule (uarch : Uarch.t) (assigns : (Expr.elem_ref * t) list)
  : (Expr.elem_ref option * t) list
  =
  let lat = Cx_sched.Node.latency uarch in
  let preds = Cx_sched.Node.preds in
  (* reachable closure *)
  let seen : (int, t) Hashtbl.t = Hashtbl.create 256 in
  let rec visit e =
    if not (Hashtbl.mem seen e.tag)
    then (
      Hashtbl.add seen e.tag e;
      List.iter visit (preds e))
  in
  List.iter (fun (_, e) -> visit e) assigns;
  let all = Hashtbl.fold (fun _ n acc -> n :: acc) seen [] in
  (* users map + counts *)
  let users : (int, t list) Hashtbl.t = Hashtbl.create 256 in
  let dedup_preds n =
    let s = Hashtbl.create 4 in
    List.filter
      (fun p ->
         if Hashtbl.mem s p.tag
         then false
         else (
           Hashtbl.add s p.tag ();
           true))
      (preds n)
  in
  List.iter
    (fun n ->
       List.iter
         (fun p ->
            let cur = try Hashtbl.find users p.tag with Not_found -> [] in
            Hashtbl.replace users p.tag (n :: cur))
         (dedup_preds n))
    all;
  let users_of n = try Hashtbl.find users n.tag with Not_found -> [] in
  (* height by memoized DFS toward sinks *)
  let height : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let rec h n =
    match Hashtbl.find_opt height n.tag with
    | Some v -> v
    | None ->
      (* mark to catch cycles loudly rather than looping *)
      Hashtbl.add height n.tag min_int;
      let hu = List.fold_left (fun a u -> max a (h u)) 0 (users_of n) in
      let v = lat n + hu in
      Hashtbl.replace height n.tag v;
      v
  in
  List.iter (fun n -> ignore (h n)) all;
  (* sink refs; loud on duplicate sink nodes *)
  let refs : (int, Expr.elem_ref) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun (r, e) ->
       if Hashtbl.mem refs e.tag
       then failwith "cx_cpl: duplicate sink node across outputs — unsupported";
       Hashtbl.add refs e.tag r)
    assigns;
  (* simulation state *)
  let unsched : (int, int) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n -> Hashtbl.add unsched n.tag (List.length (dedup_preds n))) all;
  let remaining_users : (int, int) Hashtbl.t = Hashtbl.create 256 in
  List.iter
    (fun n -> Hashtbl.add remaining_users n.tag (List.length (users_of n)))
    all;
  let finish : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let ready : (int, t) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> if Hashtbl.find unsched n.tag = 0 then Hashtbl.add ready n.tag n) all;
  let live = ref 0 in
  let capn = cap () in
  let out = ref [] in
  let total = List.length all in
  let ready_t n =
    List.fold_left
      (fun a p -> max a (try Hashtbl.find finish p.tag with Not_found -> 0))
      0
      (preds n)
  in
  let freed n =
    List.fold_left
      (fun a p -> if Hashtbl.find remaining_users p.tag = 1 then a + 1 else a)
      0
      (dedup_preds n)
  in
  for _ = 1 to total do
    (* pick best ready node *)
    let best = ref None in
    Hashtbl.iter
      (fun _ n ->
         let key =
           if !live >= capn
           then (freed n, Hashtbl.find height n.tag, - ready_t n, - n.tag)
           else (Hashtbl.find height n.tag, freed n, - ready_t n, - n.tag)
         in
         match !best with
         | None -> best := Some (key, n)
         | Some (bk, _) -> if key > bk then best := Some (key, n))
      ready;
    match !best with
    | None -> failwith "cx_cpl: ready set empty before all nodes scheduled (cycle?)"
    | Some (_, n) ->
      Hashtbl.remove ready n.tag;
      Hashtbl.replace finish n.tag (ready_t n + lat n);
      (* liveness: n becomes live if anyone will read it *)
      if Hashtbl.find remaining_users n.tag > 0 then incr live;
      List.iter
        (fun p ->
           let r = Hashtbl.find remaining_users p.tag - 1 in
           Hashtbl.replace remaining_users p.tag r;
           if r = 0 then decr live)
        (dedup_preds n);
      List.iter
        (fun u ->
           let c = Hashtbl.find unsched u.tag - 1 in
           Hashtbl.replace unsched u.tag c;
           if c = 0 then Hashtbl.add ready u.tag u)
        (users_of n);
      out := (Hashtbl.find_opt refs n.tag, n) :: !out
  done;
  List.rev !out
;;
