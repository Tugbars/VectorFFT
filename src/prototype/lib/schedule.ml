(* schedule.ml — DAG schedulers: topological / bisection / Sethi-Ullman.
 *
 * Each scheduler produces a flat list of Algsimp.t nodes in execution
 * order; emit_c.ml walks the list to produce the C body. Scheduler
 * choice is per-codelet via the `scheduler` variant in emit_codelet.
 *
 *   Topological     by hash-cons tag order. Default, cheap, valid.
 *   Bisection       Frigo's recursive 4-color cut (RED/BLUE/YELLOW/BLACK).
 *                   Cache-oblivious: each level halves the working set.
 *   SU              Sethi-Ullman list scheduling with µarch profile
 *                   (uarch.ml). Tracks live-count and switches between
 *                   LATENCY MODE (cp_dist priority) and PRESSURE MODE
 *                   (delta priority) at a threshold to manage spills.
 *   Annotated_*     wrap any of the above with annotate.ml nested-scope
 *                   structure, giving emit_c an inner-block boundary
 *                   to declare short-lived variables in. *)

open Algsimp

type color = BLACK | RED | YELLOW | BLUE

type node = {
  id: int;                           (* unique id for this DAG (not Algsimp tag) *)
  alg_node: Algsimp.t;               (* the underlying hash-consed expression *)
  output_for: Expr.elem_ref option;  (* if this is an assignment root *)
  mutable preds: node list;
  mutable succs: node list;
  mutable color: color;
}

(* === DAG CONSTRUCTION === *)

(* Build a scheduling DAG from a list of (output_ref, expr) assignments.
 *
 * The DAG includes:
 *   - one node per reachable Algsimp.t (Loads, Consts, Adds, etc.)
 *   - one synthetic "output" node per assignment, with the assignment's
 *     RHS as its sole predecessor and no successors
 *
 * Output nodes give us no-successors leaves for the bisection to start
 * the BLUE wave from. Loads/Consts give us no-predecessors leaves for
 * the RED wave. *)
let build_dag (assigns : (Expr.elem_ref * Algsimp.t) list) : node list =
  let next_id = ref 0 in
  let fresh () = let i = !next_id in incr next_id; i in
  let by_tag : (int, node) Hashtbl.t = Hashtbl.create 256 in

  let rec node_of (e : Algsimp.t) : node =
    match Hashtbl.find_opt by_tag e.tag with
    | Some n -> n
    | None ->
      let pred_exprs = Algsimp.preds e in
      let pred_nodes = List.map node_of pred_exprs in
      let n = {
        id = fresh ();
        alg_node = e;
        output_for = None;
        preds = pred_nodes;
        succs = [];
        color = BLACK;
      } in
      Hashtbl.add by_tag e.tag n;
      List.iter (fun p -> p.succs <- n :: p.succs) pred_nodes;
      n
  in
  let expr_nodes = List.map (fun (_, e) -> node_of e) assigns in
  let output_nodes = List.map2 (fun (oref, _) e_node ->
    let n = {
      id = fresh ();
      alg_node = e_node.alg_node;
      output_for = Some oref;
      preds = [e_node];
      succs = [];
      color = BLACK;
    } in
    e_node.succs <- n :: e_node.succs;
    n
  ) assigns expr_nodes in
  let all = Hashtbl.fold (fun _ n acc -> n :: acc) by_tag [] in
  all @ output_nodes

(* === BISECTION === *)

let is_input n = n.preds = []
let is_output n = n.succs = []

(* "Special inputs": leaves loading a single value (constant or twiddle).
 * They float between partitions based on neighbor colors. *)
let is_special_input n =
  match n.alg_node.node with
  | NK_Const _ -> true
  | NK_Load (Expr.Twiddle _) -> true
  | NK_Load (Expr.Input _) -> false
  | _ -> false

let has_color c n = n.color = c
let has_either_color c1 c2 n = n.color = c1 || n.color = c2

(* Bisect a list of nodes into (red, blue) partitions.
 *
 * The algorithm works as a fixed-point of two alternating waves:
 *   - Forward wave: BLACK nodes whose preds are RED-or-YELLOW become RED
 *   - Backward wave: BLACK/YELLOW nodes whose succs are all BLUE become BLUE
 *
 * Termination: when neither wave colors any new nodes, both have
 * stabilized. The remaining BLACK nodes (if any — usually none) are
 * the cut frontier; they end up uncolored.
 *
 * Edge case: if the BLUE wave wins the entire DAG (all nodes blue, no red),
 * we manually re-color inputs to RED to avoid an empty partition. *)
let bisect (nodes : node list) : node list * node list =
  List.iter (fun n -> n.color <- BLACK) nodes;
  let inputs = List.filter is_input nodes in
  let outputs = List.filter is_output nodes in
  let special = List.filter is_special_input nodes in

  List.iter (fun n -> n.color <- RED) inputs;
  List.iter (fun n -> n.color <- YELLOW) special;
  List.iter (fun n -> n.color <- BLUE) outputs;

  let rec loopi donep =
    let frontier = List.filter (fun n ->
      has_color BLACK n &&
      List.for_all (has_either_color RED YELLOW) n.preds
    ) nodes in
    match frontier with
    | [] -> if donep then () else loopo true
    | _ ->
      List.iter (fun n ->
        n.color <- RED;
        List.iter (fun p -> p.color <- RED) n.preds
      ) frontier;
      loopo false
  and loopo donep =
    let frontier = List.filter (fun n ->
      has_either_color BLACK YELLOW n &&
      List.for_all (has_color BLUE) n.succs
    ) nodes in
    match frontier with
    | [] -> if donep then () else loopi true
    | _ ->
      List.iter (fun n -> n.color <- BLUE) frontier;
      loopi false
  in
  loopi false;

  if not (List.exists (has_color RED) nodes) then
    List.iter (fun n -> n.color <- RED) inputs;

  let red = List.filter (has_color RED) nodes in
  let blue = List.filter (has_color BLUE) nodes in
  (red, blue)

(* === RECURSIVE SCHEDULING ===
 *
 *   schedule [] = Done
 *   schedule [a] = Instr a
 *   schedule alist = let (red, blue) = bisect alist in
 *                    Seq (schedule red, schedule blue)
 *
 * Frigo's Seq/Instr/Done tree flattens to an in-order traversal,
 * which is what we return directly. *)

let rec schedule_nodes (nodes : node list) : node list =
  match nodes with
  | [] -> []
  | [n] -> [n]
  | _ ->
    let (red, blue) = bisect nodes in
    if red = [] || blue = [] then
      topological_order nodes
    else if List.length red = List.length nodes ||
            List.length blue = List.length nodes then
      (* Bisection didn't actually split — fall back. *)
      topological_order nodes
    else
      schedule_nodes red @ schedule_nodes blue

and topological_order (nodes : node list) : node list =
  let emitted = Hashtbl.create 64 in
  let result = ref [] in
  let in_set n = List.exists (fun x -> x.id = n.id) nodes in
  let ready_in_set n =
    not (Hashtbl.mem emitted n.id) &&
    in_set n &&
    List.for_all (fun p ->
      Hashtbl.mem emitted p.id || not (in_set p)
    ) n.preds
  in
  let rec loop () =
    let ready = List.filter ready_in_set nodes in
    match ready with
    | [] -> ()
    | rs ->
      List.iter (fun n ->
        Hashtbl.add emitted n.id ();
        result := n :: !result
      ) rs;
      loop ()
  in
  loop ();
  List.rev !result

(* === PUBLIC API === *)

(* Schedule the assignments using Frigo's bisection algorithm.
 * Returns ordered list of (optional output assignment, expression). *)
let bisection_schedule (assigns : (Expr.elem_ref * Algsimp.t) list)
    : (Expr.elem_ref option * Algsimp.t) list =
  let dag = build_dag assigns in
  let scheduled = schedule_nodes dag in
  List.map (fun n -> (n.output_for, n.alg_node)) scheduled

(* For introspection. *)
let top_level_bisection (assigns : (Expr.elem_ref * Algsimp.t) list)
    : node list * node list =
  let dag = build_dag assigns in
  bisect dag

(* ═══════════════════════════════════════════════════════════════
 *  SU (Sethi-Ullman) LIST SCHEDULER
 * ═══════════════════════════════════════════════════════════════
 *
 * List scheduling with priority = (critical-path-distance, su-number).
 * Operates directly on Algsimp.t nodes rather than the bisection DAG.
 *
 * Critical-path distance: longest path (in cycles, weighted by Uarch
 * latencies) from a node to a sink. High cp_dist → schedule early.
 *
 * SU number: rough estimate of registers needed to evaluate this
 * subtree. Low su_number → schedule first (less pressure).
 *
 * The classical SU algorithm is for trees; we have a DAG with shared
 * subexpressions. Our "su number" is therefore approximate — it
 * over-counts pressure for shared values that are only computed once.
 * Empirically it's still useful as a tie-breaker.
 *
 * Load source-order preservation: loads have the property that they
 * have no predecessors and are always "ready". We add an extra
 * constraint that load N can only be scheduled after load N-1 (where
 * N is the load's index in tag-ascending order, which matches DFT
 * construction order). This keeps the prefetcher-friendly leg-by-leg
 * structure intact while letting arithmetic flow whenever its
 * dependencies are satisfied.
 *)

(* Latency of producing this node, given a Uarch profile. Used by
 * critical-path computation. *)
let node_latency (uarch : Uarch.t) (e : Algsimp.t) : int =
  match e.node with
  | NK_Const _ -> 0                         (* set1 broadcast: free / inlined *)
  | NK_Load _ -> uarch.load_l1_latency
  | NK_Neg _ -> uarch.add_latency           (* xor with sign mask, ~ add latency *)
  | NK_Add _ | NK_Sub _ -> uarch.add_latency
  | NK_Mul _ -> uarch.mul_latency
  | NK_CmulRe _ | NK_CmulIm _ -> uarch.fma_latency
                                            (* Cmul emits as 1 mul + 1 fma;
                                             * latency dominated by fma_latency *)
  | NK_Fma _ -> uarch.fma_latency           (* single FMA instruction *)

(* Compute critical-path distance from each node to a sink, in cycles.
 * cp_dist[n] = node_latency(n) + max(cp_dist[user] for user in users(n))
 * Sinks have cp_dist = node_latency(n). *)
let compute_cp_dist (uarch : Uarch.t)
                    (sinks : Algsimp.t list)
                    (all_nodes : Algsimp.t list)
    : (int, int) Hashtbl.t =
  (* Build successor (user) map. *)
  let users : (int, Algsimp.t list) Hashtbl.t = Hashtbl.create 256 in
  let add_user prod use =
    let cur = try Hashtbl.find users prod.tag with Not_found -> [] in
    Hashtbl.replace users prod.tag (use :: cur)
  in
  List.iter (fun n ->
    List.iter (fun p -> add_user p n) (preds n)
  ) all_nodes;

  let cp_dist : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let sink_tags = List.fold_left (fun acc s ->
    Hashtbl.replace acc s.tag (); acc
  ) (Hashtbl.create 16) sinks in

  (* Process in tag-descending order. Tag-ascending = topological (children
   * before parents in our hash-cons), so tag-descending = reverse-topological,
   * which is what we want for backward DP. *)
  let sorted_desc = List.sort (fun a b -> compare b.tag a.tag) all_nodes in
  List.iter (fun n ->
    let lat = node_latency uarch n in
    let user_list = try Hashtbl.find users n.tag with Not_found -> [] in
    let max_user_cp = List.fold_left (fun acc u ->
      let u_cp = try Hashtbl.find cp_dist u.tag with Not_found -> 0 in
      max acc u_cp
    ) 0 user_list in
    let cp =
      if Hashtbl.mem sink_tags n.tag then lat
      else lat + max_user_cp
    in
    Hashtbl.replace cp_dist n.tag cp
  ) sorted_desc;
  cp_dist

(* Compute SU number — register-pressure approximation.
 * For trees, classical SU labels are exact. On DAGs (shared subexprs),
 * this over-estimates pressure for shared values, which is a
 * conservative bias toward scheduling them earlier. *)
let compute_su_number (all_nodes : Algsimp.t list) : (int, int) Hashtbl.t =
  let su : (int, int) Hashtbl.t = Hashtbl.create 256 in
  let get_su (e : Algsimp.t) : int =
    try Hashtbl.find su e.tag with Not_found -> 1
  in
  let sorted_asc = List.sort (fun a b -> compare a.tag b.tag) all_nodes in
  List.iter (fun n ->
    let s = match n.node with
      | NK_Const _ | NK_Load _ -> 1
      | NK_Neg a -> get_su a
      | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) ->
        let sa = get_su a and sb = get_su b in
        if sa = sb then sa + 1
        else max sa sb
      | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) ->
        (* k-ary SU label: sort children by su descending, label = max_i (su_i + i). *)
        let sus = List.sort (fun x y -> compare y x)
                    [get_su a; get_su b; get_su c; get_su d] in
        let rec compute idx = function
          | [] -> 0
          | s :: rest -> max (s + idx) (compute (idx + 1) rest)
        in
        compute 0 sus
      | NK_Fma (a, b, c, _, _) ->
        (* 3-ary SU label, same scheme as Cmul. *)
        let sus = List.sort (fun x y -> compare y x)
                    [get_su a; get_su b; get_su c] in
        let rec compute idx = function
          | [] -> 0
          | s :: rest -> max (s + idx) (compute (idx + 1) rest)
        in
        compute 0 sus
    in
    Hashtbl.add su n.tag s
  ) sorted_asc;
  su

(* The list scheduler proper. *)
let su_schedule (uarch : Uarch.t) (assigns : (Expr.elem_ref * Algsimp.t) list)
    : (Expr.elem_ref option * Algsimp.t) list =
  (* Collect all reachable nodes. *)
  let seen : (int, Algsimp.t) Hashtbl.t = Hashtbl.create 256 in
  let rec visit (e : Algsimp.t) =
    if not (Hashtbl.mem seen e.tag) then begin
      Hashtbl.add seen e.tag e;
      List.iter visit (preds e)
    end
  in
  List.iter (fun (_, e) -> visit e) assigns;
  let all_nodes = Hashtbl.fold (fun _ n acc -> n :: acc) seen [] in
  let sinks = List.map snd assigns in

  let cp_dist = compute_cp_dist uarch sinks all_nodes in
  let su_num = compute_su_number all_nodes in

  (* Successor map for forward propagation when a node becomes scheduled. *)
  let users : (int, Algsimp.t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n ->
    List.iter (fun p ->
      let cur = try Hashtbl.find users p.tag with Not_found -> [] in
      Hashtbl.replace users p.tag (n :: cur)
    ) (preds n)
  ) all_nodes;

  (* Unscheduled-pred counter per node. *)
  let unsched_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n ->
    Hashtbl.add unsched_count n.tag (List.length (preds n))
  ) all_nodes;

  (* Ready set: nodes whose predecessors have all been scheduled. *)
  let scheduled : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let in_ready : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let ready : Algsimp.t list ref = ref [] in
  List.iter (fun n ->
    if preds n = [] then begin
      ready := n :: !ready;
      Hashtbl.add in_ready n.tag ()
    end
  ) all_nodes;

  (* Load source-order tracking: only allow the next-required load to fire. *)
  let load_tags = List.filter_map (fun n ->
    match n.node with NK_Load _ -> Some n.tag | _ -> None
  ) all_nodes |> List.sort compare in
  let load_array = Array.of_list load_tags in
  let load_idx = ref 0 in
  let next_required_load () : int option =
    if !load_idx < Array.length load_array
    then Some load_array.(!load_idx)
    else None
  in

  (* Pick next node from ready set.
   *
   * Policy:
   *   1. If ANY non-load instruction is ready, pick from those. Loads stay
   *      queued.
   *   2. Within arith-ready, prefer SINK nodes (no in-DAG users) when ready.
   *      For DIF prime codelets, the cmul post-multiply nodes ARE the
   *      output assignments — empty user lists, low cp_dist (~4 cycles).
   *      Without this rule, cp_dist DESC schedules them LAST, after all
   *      inner DFT nodes. Their inputs (raw DFT outputs) stay alive in
   *      registers from when the inner DFT produced them until the very
   *      end, peaking at 22+ live values for R=11. Firing each cmul as
   *      soon as both raw inputs are scheduled lets the raw values die
   *      immediately (they have no other consumers), shrinking live set
   *      by 2 per cmul-pair. Empirically: R=17 t1_dif vmovapd 176→~115,
   *      matching hand. DIT is unaffected: DIT cmuls are SOURCES (high
   *      cp_dist, fire early naturally); the actual DIT sinks are post-DFT
   *      combine nodes that fire late under cp_dist anyway.
   *   3. Within the same sink/non-sink class, use cp_dist DESC, su_num ASC,
   *      tag ASC (the original SU heuristic).
   *   4. Only when no arithmetic is ready, fire the next required load
   *      (preserves source order so prefetcher sees sequential loads).
   *
   * Sink detection: a node with EMPTY user list in the DAG. Its only role
   * is to feed an output store. *)
  let is_load n = match n.node with NK_Load _ -> true | _ -> false in
  let is_sink n =
    let user_list = try Hashtbl.find users n.tag with Not_found -> [] in
    user_list = []
  in
  let pick_next () : Algsimp.t option =
    if !ready = [] then None
    else
      let arith_ready = List.filter (fun n -> not (is_load n)) !ready in
      let cmp a b =
        let sa = is_sink a in let sb = is_sink b in
        if sa <> sb then compare sb sa
        else
        let cpa = try Hashtbl.find cp_dist a.tag with Not_found -> 0 in
        let cpb = try Hashtbl.find cp_dist b.tag with Not_found -> 0 in
        if cpa <> cpb then compare cpb cpa  (* higher cp_dist first *)
        else
          let sua = try Hashtbl.find su_num a.tag with Not_found -> 0 in
          let sub = try Hashtbl.find su_num b.tag with Not_found -> 0 in
          if sua <> sub then compare sua sub  (* lower su_num breaks ties *)
          else compare a.tag b.tag             (* stable: tag-ascending *)
      in
      match arith_ready with
      | _ :: _ ->
        (* At least one arithmetic op ready — pick the best one, defer loads. *)
        Some (List.hd (List.sort cmp arith_ready))
      | [] ->
        (* No arithmetic ready — must fire a load to unblock more work.
         * Fire the next required load (preserves source order). *)
        let req_load = next_required_load () in
        let load_candidates = List.filter (fun n ->
          is_load n && Some n.tag = req_load
        ) !ready in
        (match load_candidates with
         | [] -> None  (* shouldn't happen if scheduling is making progress *)
         | _ -> Some (List.hd load_candidates))
  in

  let result : (Expr.elem_ref option * Algsimp.t) list ref = ref [] in
  let rec loop () =
    match pick_next () with
    | None -> ()
    | Some n ->
      Hashtbl.add scheduled n.tag ();
      ready := List.filter (fun x -> x.tag <> n.tag) !ready;
      Hashtbl.remove in_ready n.tag;
      (match n.node with NK_Load _ -> incr load_idx | _ -> ());
      result := (None, n) :: !result;
      let user_list = try Hashtbl.find users n.tag with Not_found -> [] in
      List.iter (fun u ->
        let cur = try Hashtbl.find unsched_count u.tag with Not_found -> 0 in
        let new_count = cur - 1 in
        Hashtbl.replace unsched_count u.tag new_count;
        if new_count = 0
           && not (Hashtbl.mem in_ready u.tag)
           && not (Hashtbl.mem scheduled u.tag) then begin
          ready := u :: !ready;
          Hashtbl.add in_ready u.tag ()
        end
      ) user_list;
      loop ()
  in
  loop ();

  let intermediates = List.rev !result in
  let stores = List.map (fun (oref, e) -> (Some oref, e)) assigns in
  intermediates @ stores

(* === SU SCHEDULING ON A FIXED SUBSET ===
 *
 * Like su_schedule but operates on a pre-selected subset of nodes
 * (e.g. PASS 1 or PASS 2 alone). Predecessors not in the subset are
 * treated as "already available" (e.g. inputs to PASS 1, reloaded
 * spill values to PASS 2, hoisted constants).
 *
 * Inputs:
 *   - uarch: latency profile
 *   - subset: nodes to schedule
 *   - sinks: nodes whose cp_dist contributes to scheduling priority
 *            (typically PASS 1 spill targets, or PASS 2 output sinks)
 *
 * Returns: subset nodes in scheduled order.
 *
 * Note: load source-order tracking is intentionally simplified here —
 * within a pass, loads are typically twiddle/data loads that the
 * scheduler should issue early. For PASS 1 we still want loads to be
 * deferred behind ready arithmetic; for PASS 2 there are no loads
 * (reloads happen before the pass). *)
(* === GOODMAN-HSU MODE SWITCH ===
 *
 * Optional pressure-aware extension. When ~gh:true is passed, the picker
 * tracks the live-count of scheduled-but-not-yet-killed nodes and switches
 * priority functions when live-count exceeds a threshold:
 *
 *   - LATENCY MODE (live_count <= threshold): pick by (cp_dist DESC, su_num ASC)
 *     — same as base SU.
 *   - PRESSURE MODE (live_count > threshold): pick by (delta ASC, cp_dist DESC),
 *     where delta = births - kills for the candidate node:
 *       kills(n) = #(preds in subset whose remaining_users == 1 and n is one of them)
 *       births(n) = 1 if n has remaining users in subset OR n is a sink, else 0
 *     Pick the most-negative delta — i.e., the choice that frees the most live
 *     values relative to what it consumes.
 *
 * Threshold is `uarch.vec_regs - 4` (the 4-slot slack reserves room for cmul
 * scratch + FMA scratch). The live count is tracked over the SUBSET only;
 * out-of-subset predecessors are treated as "external slots" not counted here.
 *
 * Loads stay deferred behind arithmetic regardless of mode (the load-deferral
 * rule is orthogonal to pressure tracking). *)
let su_schedule_subset (uarch : Uarch.t)
    ~(gh : bool)
    ~(subset : Algsimp.t list)
    ~(sinks : Algsimp.t list)
    : Algsimp.t list =
  (* Subset membership lookup. *)
  let in_subset : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n -> Hashtbl.add in_subset n.Algsimp.tag ()) subset;
  (* Sinks lookup for births() — sinks stay live even when remaining_users hits 0. *)
  let sink_set : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun s -> Hashtbl.replace sink_set s.Algsimp.tag ()) sinks;

  (* cp_dist over the full reachable graph from sinks. This includes
   * predecessors outside the subset (they contribute to depth).
   * We use compute_cp_dist's existing logic which walks transitively. *)
  let all_reachable : (int, Algsimp.t) Hashtbl.t = Hashtbl.create 512 in
  let rec visit (e : Algsimp.t) =
    if not (Hashtbl.mem all_reachable e.tag) then begin
      Hashtbl.add all_reachable e.tag e;
      List.iter visit (preds e)
    end
  in
  List.iter visit sinks;
  let all_nodes = Hashtbl.fold (fun _ n acc -> n :: acc) all_reachable [] in
  let cp_dist = compute_cp_dist uarch sinks all_nodes in
  let su_num = compute_su_number all_nodes in

  (* Successor map RESTRICTED to subset edges. *)
  let users : (int, Algsimp.t list) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n ->
    List.iter (fun p ->
      if Hashtbl.mem in_subset p.Algsimp.tag then begin
        let cur = try Hashtbl.find users p.tag with Not_found -> [] in
        Hashtbl.replace users p.tag (n :: cur)
      end
    ) (preds n)
  ) subset;

  (* Unscheduled-pred counter — only count predecessors IN the subset.
   * Out-of-subset preds are treated as already satisfied. *)
  let unsched_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
  List.iter (fun n ->
    let in_subset_preds = List.filter (fun p ->
      Hashtbl.mem in_subset p.Algsimp.tag
    ) (preds n) in
    Hashtbl.add unsched_count n.tag (List.length in_subset_preds)
  ) subset;

  (* Remaining-user counter (for Goodman-Hsu): how many in-subset successors
   * of node X are still unscheduled. Initially = total in-subset users.
   * Decremented when a user is scheduled. When hits 0, X is no longer live
   * (unless it's a sink). *)
  let remaining_users : (int, int) Hashtbl.t = Hashtbl.create 256 in
  if gh then begin
    (* Pre-build successor map restricted to subset edges.
     * Same logic as the `users` table built below; we count entries here. *)
    let user_count : (int, int) Hashtbl.t = Hashtbl.create 256 in
    List.iter (fun n -> Hashtbl.add user_count n.Algsimp.tag 0) subset;
    List.iter (fun n ->
      List.iter (fun p ->
        if Hashtbl.mem in_subset p.Algsimp.tag then begin
          let cur = try Hashtbl.find user_count p.tag with Not_found -> 0 in
          Hashtbl.replace user_count p.tag (cur + 1)
        end
      ) (preds n)
    ) subset;
    Hashtbl.iter (fun tag c -> Hashtbl.add remaining_users tag c) user_count
  end;

  (* Live set tracker (for Goodman-Hsu only). Threshold uses the
   * uarch-specific pressure_threshold (24 for AVX-512, 12 for AVX2),
   * which already accounts for cmul/FMA scratch slack. *)
  let live : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let live_count () = Hashtbl.length live in
  let threshold = uarch.Uarch.pressure_threshold in

  (* Ready set: nodes with no unscheduled subset preds. *)
  let scheduled : (int, unit) Hashtbl.t = Hashtbl.create 256 in
  let in_ready : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let ready : Algsimp.t list ref = ref [] in
  List.iter (fun n ->
    if Hashtbl.find unsched_count n.tag = 0 then begin
      ready := n :: !ready;
      Hashtbl.add in_ready n.tag ()
    end
  ) subset;

  let is_load n = match n.Algsimp.node with NK_Load _ -> true | _ -> false in

  (* Load source order — only relevant if subset contains loads. *)
  let load_tags = List.filter_map (fun n ->
    if is_load n then Some n.Algsimp.tag else None
  ) subset |> List.sort compare in
  let load_array = Array.of_list load_tags in
  let load_idx = ref 0 in
  let next_required_load () =
    if !load_idx < Array.length load_array
    then Some load_array.(!load_idx)
    else None
  in

  let pick_next () : Algsimp.t option =
    if !ready = [] then None
    else
      let arith_ready = List.filter (fun n -> not (is_load n)) !ready in
      (* Latency-mode comparator: cp_dist DESC, su_num ASC, tag ASC.
       *
       * SINK PREFERENCE: when a sink (output assignment node) is ready
       * AND its only role is to consume preds and write to memory, fire
       * it ASAP. This is critical for DIF prime codelets where the cmul
       * post-multiply nodes ARE the outputs — they have cp_dist =
       * node_latency (~4 cycles), losing to inner DFT nodes (cp_dist
       * 8+) under raw cp_dist DESC. Firing the cmul as soon as both
       * raw DFT inputs are scheduled lets the raw inputs be killed
       * immediately (they have no other users), reducing live set by
       * 2 per cmul-pair. Empirically this drops R=17 t1_dif vmovapd
       * from 176 to ~115 (matches hand) and closes the DIF perf gap.
       *
       * DIT is unaffected: DIT's cmul nodes are sources (high cp_dist),
       * fire early naturally; the actual DIT sinks are the post-DFT
       * combine ops (Add/Sub), which already fire late under cp_dist
       * because there's nothing else ready by then. *)
      let cmp_latency a b =
        let sa = Hashtbl.mem sink_set a.Algsimp.tag in
        let sb = Hashtbl.mem sink_set b.Algsimp.tag in
        if sa <> sb then compare sb sa  (* true (sink) before false *)
        else
        let cpa = try Hashtbl.find cp_dist a.Algsimp.tag with Not_found -> 0 in
        let cpb = try Hashtbl.find cp_dist b.Algsimp.tag with Not_found -> 0 in
        if cpa <> cpb then compare cpb cpa
        else
          let sua = try Hashtbl.find su_num a.tag with Not_found -> 0 in
          let sub = try Hashtbl.find su_num b.tag with Not_found -> 0 in
          if sua <> sub then compare sua sub
          else compare a.tag b.tag
      in
      (* Goodman-Hsu pressure-mode comparator: minimize live-delta.
       *   delta(n) = births(n) - kills(n)
       *   kills(n) = preds in subset with remaining_users == 1 (n kills them)
       *   births(n) = 1 if n has remaining users in subset OR is a sink
       * Pick the most-negative delta (frees most relative to creates).
       * Tiebreak with cp_dist DESC (some latency awareness within pressure mode)
       * then tag ASC for stability. *)
      let pressure_delta n =
        let preds_in_sub = List.filter (fun p ->
          Hashtbl.mem in_subset p.Algsimp.tag
        ) (preds n) in
        let kills = List.fold_left (fun acc p ->
          let ru = try Hashtbl.find remaining_users p.Algsimp.tag with Not_found -> 0 in
          if ru = 1 then acc + 1 else acc
        ) 0 preds_in_sub in
        let n_users = try Hashtbl.find remaining_users n.Algsimp.tag with Not_found -> 0 in
        let is_sink = Hashtbl.mem sink_set n.tag in
        let births = if n_users > 0 || is_sink then 1 else 0 in
        births - kills
      in
      let cmp_pressure a b =
        let da = pressure_delta a in
        let db = pressure_delta b in
        if da <> db then compare da db  (* most-negative first *)
        else
          let cpa = try Hashtbl.find cp_dist a.Algsimp.tag with Not_found -> 0 in
          let cpb = try Hashtbl.find cp_dist b.Algsimp.tag with Not_found -> 0 in
          if cpa <> cpb then compare cpb cpa
          else compare a.tag b.tag
      in
      let cmp =
        if gh && live_count () > threshold then cmp_pressure
        else cmp_latency
      in
      match arith_ready with
      | _ :: _ -> Some (List.hd (List.sort cmp arith_ready))
      | [] ->
        let req_load = next_required_load () in
        let load_candidates = List.filter (fun n ->
          is_load n && Some n.Algsimp.tag = req_load
        ) !ready in
        (match load_candidates with
         | [] -> None
         | _ -> Some (List.hd load_candidates))
  in

  let result : Algsimp.t list ref = ref [] in
  let rec loop () =
    match pick_next () with
    | None -> ()
    | Some n ->
      Hashtbl.add scheduled n.Algsimp.tag ();
      ready := List.filter (fun x -> x.Algsimp.tag <> n.tag) !ready;
      Hashtbl.remove in_ready n.tag;
      (if is_load n then incr load_idx);
      result := n :: !result;

      (* Goodman-Hsu live-set update.
       * 1. For each pred P in subset, decrement P's remaining_users.
       *    If P hits 0 and isn't a sink, P dies (remove from live).
       * 2. n itself becomes live if it has remaining users OR is a sink. *)
      if gh then begin
        let preds_in_sub = List.filter (fun p ->
          Hashtbl.mem in_subset p.Algsimp.tag
        ) (preds n) in
        List.iter (fun p ->
          let cur = try Hashtbl.find remaining_users p.Algsimp.tag with Not_found -> 0 in
          let new_count = cur - 1 in
          Hashtbl.replace remaining_users p.tag new_count;
          if new_count = 0 && not (Hashtbl.mem sink_set p.tag) then
            Hashtbl.remove live p.tag
        ) preds_in_sub;
        let n_users = try Hashtbl.find remaining_users n.tag with Not_found -> 0 in
        let is_sink = Hashtbl.mem sink_set n.tag in
        if n_users > 0 || is_sink then
          Hashtbl.replace live n.tag ()
      end;

      let user_list = try Hashtbl.find users n.tag with Not_found -> [] in
      List.iter (fun u ->
        let cur = try Hashtbl.find unsched_count u.Algsimp.tag with Not_found -> 0 in
        let new_count = cur - 1 in
        Hashtbl.replace unsched_count u.tag new_count;
        if new_count = 0
           && not (Hashtbl.mem in_ready u.tag)
           && not (Hashtbl.mem scheduled u.tag) then begin
          ready := u :: !ready;
          Hashtbl.add in_ready u.tag ()
        end
      ) user_list;
      loop ()
  in
  loop ();
  List.rev !result
