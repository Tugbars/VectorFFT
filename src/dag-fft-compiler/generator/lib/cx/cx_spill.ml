(* cx_spill.ml — Belady spill planner for the cx mono emission.
 *
 * WHY (2026-08-11, docs/roadmap/tangent_emitter_plan.md): the tangent/wing
 * mono bodies peak at 16 live vectors on a 16-register file, so gcc loses its
 * one free register and can no longer hoist the 3 loop-invariant broadcast
 * constants — it folds each into a per-use rip-relative memory operand
 * (27 L1 loads/iter) that lands on the critical path (CP 83 vs the hand
 * kernel's 69). PROVEN root cause: at peak-15 gcc hoists everything (measured
 * via a 32-register AVX-512 build: rip-const 27->0, spills ->0). The hand
 * kernel reaches 15 by carrying the origin's ~10 DELIBERATE stack spills.
 *
 * This module reproduces that: a Belady offline spiller (furthest-next-use
 * eviction) over the ALREADY-SCHEDULED cx node list, producing spill/reload
 * sites the mono emitter turns into an S[] round-trip. It reuses regalloc.ml's
 * proven design (spill_sites + name_overrides + per-slot scratch arena) but on
 * the cx IR instead of the real-valued Algsimp path.
 *
 * The pass is ORDER-PRESERVING (it does not reschedule) and only ADDS a
 * store+reload for evicted values, so it is numerically identical to the input
 * DAG. Selected by VFFT_CX_SPILL=<budget> (default OFF => byte-identical). *)

open Cx_ir

type plan =
  { nslots : int
  ; (* after emitting the node at position p, store each (tag,slot) *)
    spill_after : (int, (int * int) list) Hashtbl.t
  ; (* before emitting the node at position p, reload each (tag,slot) into a
       fresh name; downstream references to `tag` use that name *)
    reload_before : (int, (int * int) list) Hashtbl.t
  ; peak_in : int (* peak simultaneous live BEFORE spilling *)
  ; peak_out : int (* peak AFTER (should be <= budget when feasible) *)
  }

let budget () =
  match Sys.getenv_opt "VFFT_CX_SPILL" with
  | Some s ->
    (try Some (int_of_string s) with
     | _ -> None)
  | None -> None
;;

(* Peak simultaneous live over a straight-line schedule, honouring an eviction
   set: a tag is "in registers" from its def position to its last use, MINUS
   any [spill_pos, reload_pos) window where it was evicted. Loads count. *)
let compute (budget : int) (scheduled : (_ * t) list) : plan =
  let nodes = Array.of_list (List.map snd scheduled) in
  let n = Array.length nodes in
  let pos_of : (int, int) Hashtbl.t = Hashtbl.create 256 in
  Array.iteri
    (fun i e -> if not (Hashtbl.mem pos_of e.tag) then Hashtbl.add pos_of e.tag i)
    nodes;
  (* uses[tag] = sorted positions where tag is read as a pred *)
  let uses : (int, int list) Hashtbl.t = Hashtbl.create 256 in
  let add_use t p =
    let cur =
      try Hashtbl.find uses t with
      | Not_found -> []
    in
    Hashtbl.replace uses t (p :: cur)
  in
  let dedup l =
    let s = Hashtbl.create 4 in
    List.filter
      (fun x ->
         if Hashtbl.mem s x
         then false
         else (
           Hashtbl.add s x ();
           true))
      l
  in
  Array.iteri
    (fun p e -> List.iter (fun pr -> add_use pr.tag p) (dedup (Cx_sched.Node.preds e)))
    nodes;
  let last_use t =
    match Hashtbl.find_opt uses t with
    | Some (u :: _) -> u (* list built by consing => head is the LARGEST pos *)
    | _ -> Hashtbl.find pos_of t
  in
  (* future uses of tag t strictly after position p (ascending) *)
  let uses_after t p =
    match Hashtbl.find_opt uses t with
    | None -> []
    | Some l -> List.sort compare (List.filter (fun u -> u > p) l)
  in
  let spill_after = Hashtbl.create 64 in
  let reload_before = Hashtbl.create 64 in
  let add tbl k v =
    let cur =
      try Hashtbl.find tbl k with
      | Not_found -> []
    in
    Hashtbl.replace tbl k (v :: cur)
  in
  (* live register set of tags; slot bookkeeping *)
  let live : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let evicted : (int, int) Hashtbl.t = Hashtbl.create 32 in
  (* tag -> slot while spilled *)
  let free_slots = ref [] in
  let next_slot = ref 0 in
  let alloc_slot () =
    match !free_slots with
    | s :: r ->
      free_slots := r;
      s
    | [] ->
      let s = !next_slot in
      incr next_slot;
      s
  in
  let peak_in = ref 0 in
  let peak_out = ref 0 in
  (* per position: first reload the evicted tags whose next use is here, then
     define this node's tag, then if over budget evict furthest-next-use. *)
  for p = 0 to n - 1 do
    let e = nodes.(p) in
    (* reloads scheduled for this position (recorded when we evicted) *)
    (match Hashtbl.find_opt reload_before p with
     | Some rs -> List.iter (fun (t, _slot) -> Hashtbl.replace live t ()) rs
     | None -> ());
    (* define this node's value if it has future uses OR is an output sink *)
    let has_future = Hashtbl.mem uses e.tag in
    if has_future then Hashtbl.replace live e.tag ();
    (* retire values whose last use was at p (they were read here for the last
       time — free their register AFTER this position) *)
    let raw_live = Hashtbl.length live in
    if raw_live > !peak_in then peak_in := raw_live;
    (* Belady eviction while over budget: evict the currently-live tag with the
       FURTHEST next use that (a) is not this node's own def with an immediate
       next-use, and (b) actually has a future use to reload for. *)
    while Hashtbl.length live > budget do
      let best = ref None in
      Hashtbl.iter
        (fun t () ->
           if t <> e.tag
           then (
             match uses_after t p with
             | nu :: _ ->
               (match !best with
                | Some (_, bnu) when bnu >= nu -> ()
                | _ -> best := Some (t, nu))
             | [] -> ()))
        live;
      match !best with
      | None -> raise Exit (* infeasible at this budget; caught below *)
      | Some (t, nu) ->
        let slot = alloc_slot () in
        Hashtbl.remove live t;
        Hashtbl.replace evicted t slot;
        add spill_after p (t, slot);
        add reload_before nu (t, slot);
        (* the slot is busy until the reload consumes it; free it at reload by
           recording nothing here — freed lazily when reload fires below is not
           possible in one pass, so we conservatively keep distinct slots. *)
        ignore slot
    done;
    let now_live = Hashtbl.length live in
    if now_live > !peak_out then peak_out := now_live;
    (* free registers of tags making their LAST use at p (post-node retire) *)
    List.iter
      (fun pr -> if last_use pr.tag = p then Hashtbl.remove live pr.tag)
      (dedup (Cx_sched.Node.preds e));
    (* a reloaded value that is used only at its reload position also retires *)
    match Hashtbl.find_opt reload_before p with
    | Some rs ->
      List.iter
        (fun (t, slot) ->
           if last_use t <= p
           then (
             Hashtbl.remove live t;
             free_slots := slot :: !free_slots))
        rs
    | None -> ()
  done;
  { nslots = !next_slot
  ; spill_after
  ; reload_before
  ; peak_in = !peak_in
  ; peak_out = !peak_out
  }
;;

(* Public entry: returns None when the knob is off or the budget is infeasible
   (so the caller falls back to the plain byte-identical emission). *)
let plan (scheduled : (_ * t) list) : plan option =
  match budget () with
  | None -> None
  | Some b ->
    (try
       let pl = compute b scheduled in
       Some pl
     with
     | Exit -> None)
;;
