(* schedule.ml — port of Frigo's recursive bisection scheduler.
 *
 * Algorithm (from genfft/schedule.ml lines 85-191):
 *   1. Build DAG with explicit predecessor and successor lists.
 *   2. Bisect: alternating waves color nodes RED (input-side) and
 *      BLUE (output-side) until both stabilize. The cut between them
 *      defines the partition.
 *   3. Recursively schedule each half.
 *   4. Concatenate: red_schedule ++ blue_schedule.
 *
 * Frigo's claim: this is cache-oblivious. At each recursion level, the
 * working set is divided. As we recurse deeper, cut sizes shrink and
 * fit smaller cache levels — without ever knowing the cache hierarchy.
 *
 * Special inputs: single-load nodes (Loads of constants or twiddles in
 * our representation) are colored YELLOW initially. They float to
 * whichever side claims them first — closer to use, reducing live range.
 *
 * Implementation note: Frigo's representation has explicit DAG nodes
 * with mutable color/predecessor/successor fields. We mirror that
 * structure in `node` below. Our hash-consed Algsimp.t is the input;
 * we build a parallel DAG for scheduling. *)

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
  (* SUBSET-RELATIVE dag view. genfft rebuilds the dag (makedag) at
   * every recursion level, so inputs/outputs and both wave conditions
   * are relative to the sublist being partitioned. With global
   * preds/succs, any pure-compute subset (no store sinks) has no
   * outputs, the BLUE wave never seeds, every node drains RED, the cut
   * degenerates, and schedule_nodes falls back to topological order,
   * which front-loads every ready load. Measured before this fix:
   * 245/1047/2997 spill movs at R=16/32/64 avx2; loads 12/12 at the
   * top of the emission. *)
  let member : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.replace member n.id ()) nodes;
  let in_set n = Hashtbl.mem member n.id in
  let preds_in n = List.filter in_set n.preds in
  let succs_in n = List.filter in_set n.succs in
  List.iter (fun n -> n.color <- BLACK) nodes;
  let inputs = List.filter (fun n -> preds_in n = []) nodes in
  let outputs = List.filter (fun n -> succs_in n = []) nodes in
  let special = List.filter is_special_input nodes in

  List.iter (fun n -> n.color <- RED) inputs;
  List.iter (fun n -> n.color <- YELLOW) special;
  List.iter (fun n -> n.color <- BLUE) outputs;

  let rec loopi donep =
    let frontier = List.filter (fun n ->
      has_color BLACK n &&
      List.for_all (has_either_color RED YELLOW) (preds_in n)
    ) nodes in
    match frontier with
    | [] -> if donep then () else loopo true
    | _ ->
      List.iter (fun n ->
        n.color <- RED;
        List.iter (fun p -> p.color <- RED) (preds_in n)
      ) frontier;
      loopo false
  and loopo donep =
    let frontier = List.filter (fun n ->
      has_either_color BLACK YELLOW n &&
      List.for_all (has_color BLUE) (succs_in n)
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

(* Connected components within a node subset (undirected reachability
 * over preds/succs restricted to the subset). genfft's schedule_alist
 * decomposes into components at EVERY recursion level before
 * partitioning (Par of independently scheduled components). This is
 * load-interleaving's actual mechanism: after a cut, the red half
 * disconnects into per-sub-FFT clusters, each carrying its own loads,
 * and each cluster schedules contiguously. Without this step the load
 * layer never separates and bisection emits all loads first (measured:
 * 245/1047/2997 spill movs at R=16/32/64 avx2 vs SU's 113/289/817). *)
let connected_components_of (nodes : node list) : node list list =
  (* CRITICAL FIDELITY POINT vs genfft: in genfft's IR, constants are
   * inline literals inside expressions, NOT dag nodes, so they create
   * no edges and the CT sub-DFTs of an FFT dag are genuinely
   * disconnected below the combine layer. In our Algsimp IR, NK_Const
   * (and twiddle loads) are hash-consed shared NODES whose edges weld
   * every sub-DFT into one blob, which silently disables this entire
   * decomposition. Fix: do not traverse THROUGH special inputs
   * (constants/twiddles). After components form over the non-special
   * subgraph, attach each special node to the component of its
   * minimum-id in-subset successor; specials with no in-subset
   * successor become singletons. *)
  let in_set : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.replace in_set n.id ()) nodes;
  let special, regular = List.partition is_special_input nodes in
  let seen : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let comp_of seed =
    let acc = ref [] in
    let q = Queue.create () in
    Queue.add seed q; Hashtbl.replace seen seed.id ();
    while not (Queue.is_empty q) do
      let n = Queue.pop q in
      acc := n :: !acc;
      List.iter (fun m ->
        if Hashtbl.mem in_set m.id && not (Hashtbl.mem seen m.id)
           && not (is_special_input m) then begin
          Hashtbl.replace seen m.id (); Queue.add m q
        end) (n.preds @ n.succs)
    done;
    !acc
  in
  let comps =
    List.filter_map
      (fun n -> if Hashtbl.mem seen n.id then None else Some (comp_of n))
      regular
  in
  match comps with
  | [] -> (match special with [] -> [] | _ -> [special])
  | _ ->
    (* Specials consumed by exactly ONE component attach to it (so its
     * cone stays self-contained). Specials consumed by SEVERAL
     * components are hoisted ahead of all components, the moral
     * equivalent of genfft's inline literals being available
     * everywhere. Attaching a shared constant to its min-id consumer
     * was a LEGALITY BUG: any earlier-scheduled consumer component
     * read it before definition (caught by gcc, use-before-def at
     * R=32). Specials with no in-subset consumer hoist too. *)
    let comp_arr = Array.of_list (List.map ref comps) in
    let owner : (int, int) Hashtbl.t = Hashtbl.create 64 in
    Array.iteri (fun i c -> List.iter (fun n -> Hashtbl.replace owner n.id i) !c) comp_arr;
    let hoisted = ref [] in
    List.iter (fun sp ->
      let cands = List.filter_map (fun m ->
        if Hashtbl.mem in_set m.id then Hashtbl.find_opt owner m.id else None
      ) sp.succs in
      let uniq = List.sort_uniq compare cands in
      match uniq with
      | [i] -> comp_arr.(i) := sp :: !(comp_arr.(i))
      | _ -> hoisted := sp :: !hoisted
    ) special;
    let attached = Array.to_list (Array.map (!) comp_arr) in
    (match !hoisted with [] -> attached | l -> l :: attached)

(* Port of genfft annotate.ml's `reorder`: greedy ordering of sibling
 * components to maximize variable overlap between adjacent blocks.
 * A block's variable set is every node id it contains plus every
 * operand (pred) id it references; after a cut, sibling components
 * frequently consume the same producer values from earlier regions,
 * and placing such siblings adjacently ends those producers' live
 * ranges early. genfft chains greedily: start from the smallest block
 * (their comment: "start with smallest block --- does this matter?"),
 * then repeatedly take the remaining block with maximum overlap
 * against the previous one. Ties break on min node id for
 * deterministic output. *)
let reorder_components (comps : node list list) : node list list =
  let vars_of comp =
    let t = Hashtbl.create 32 in
    List.iter (fun n ->
      Hashtbl.replace t n.id ();
      List.iter (fun p -> Hashtbl.replace t p.id ()) n.preds
    ) comp;
    t
  in
  let min_id comp = List.fold_left (fun a n -> min a n.id) max_int comp in
  let tagged = List.map (fun c -> (c, vars_of c, min_id c)) comps in
  let overlap va (_, vb, _) =
    Hashtbl.fold (fun k () acc -> if Hashtbl.mem vb k then acc + 1 else acc) va 0
  in
  match tagged with
  | [] -> []
  | _ ->
    (* smallest variable set first *)
    let start =
      List.fold_left (fun best c ->
        let (_, vb, ib) = best and (_, vc, ic) = c in
        let nb = Hashtbl.length vb and nc = Hashtbl.length vc in
        if nc < nb || (nc = nb && ic < ib) then c else best
      ) (List.hd tagged) (List.tl tagged)
    in
    let rest = List.filter (fun (_, _, i) -> i <> (let (_,_,i0)=start in i0)) tagged in
    let rec chain prev_vars remaining acc =
      match remaining with
      | [] -> List.rev acc
      | _ ->
        let best =
          List.fold_left (fun b c ->
            let ob = overlap prev_vars b and oc = overlap prev_vars c in
            let (_,_,ib) = b and (_,_,ic) = c in
            if oc > ob || (oc = ob && ic < ib) then c else b
          ) (List.hd remaining) (List.tl remaining)
        in
        let (_, bv, bi) = best in
        let remaining' = List.filter (fun (_,_,i) -> i <> bi) remaining in
        chain bv remaining' (best :: acc)
    in
    let (c0, v0, _) = start in
    c0 :: List.map (fun (c,_,_) -> c) (chain v0 rest [])

let rec schedule_nodes (nodes : node list) : node list =
  match nodes with
  | [] -> []
  | [n] -> [n]
  | _ ->
    match connected_components_of nodes with
    | comps when List.length comps > 1 ->
      (* Specials-only components (hoisted shared constants/twiddles)
       * MUST precede everything that reads them; exempt them from
       * reorder so legality never depends on the greedy chain. *)
      let specials, rest =
        List.partition (fun c -> List.for_all is_special_input c) comps in
      List.concat_map schedule_nodes specials
      @ List.concat_map schedule_nodes (reorder_components rest)
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
  | NK_Plus _ -> Algsimp.nk_plus_unreachable "schedule.ml:255"

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
      | NK_Plus _ -> Algsimp.nk_plus_unreachable "schedule.ml:319"
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

  let su_intermediates = List.rev !result in
  (* === SCHEDULE-SEARCH KNOBS (experiment; default = unchanged SU output) ===
   * VFFT_SCHED_DUMP=<file>: write the SU order as "tag:pred_tags..." per line,
   *   so the annealer knows the DAG and can permute legally.
   * VFFT_SCHED_ORDER=<file>: emit the intermediates in the explicit tag order
   *   from the file (one tag per line) instead of SU's order. The order MUST be
   *   a complete, legal topological order of su_intermediates — the caller
   *   (annealer) guarantees this; an illegal order yields a use-before-def C
   *   compile error, which the search treats as an invalid candidate. *)
  (match Sys.getenv_opt "VFFT_SCHED_DUMP" with
   | None -> ()
   | Some file ->
     let oc = open_out file in
     List.iter (fun (_, (n : Algsimp.t)) ->
       Printf.fprintf oc "%d:%s\n" n.tag
         (String.concat " "
            (List.map (fun (p : Algsimp.t) -> string_of_int p.tag) (preds n)))
     ) su_intermediates;
     close_out oc);
  let intermediates =
    match Sys.getenv_opt "VFFT_SCHED_ORDER" with
    | None -> su_intermediates
    | Some file ->
      let by_tag : (int, Algsimp.t) Hashtbl.t = Hashtbl.create 256 in
      List.iter (fun (n : Algsimp.t) -> Hashtbl.replace by_tag n.tag n) all_nodes;
      let ic = open_in file in
      let tags = ref [] in
      (try while true do
         match int_of_string_opt (String.trim (input_line ic)) with
         | Some t -> tags := t :: !tags
         | None -> ()
       done with End_of_file -> ());
      close_in ic;
      List.filter_map (fun t ->
        match Hashtbl.find_opt by_tag t with Some n -> Some (None, n) | None -> None)
        (List.rev !tags)
  in
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
  (* VFFT_GH_THRESHOLD env override (experiment knob; default = unchanged).
   * Lowering it makes GH pressure-mode engage earlier -> frees registers
   * sooner -> fewer stack spills, at a possible latency cost. *)
  let threshold =
    match Sys.getenv_opt "VFFT_GH_THRESHOLD" with
    | Some s -> (try int_of_string s with _ -> uarch.Uarch.pressure_threshold)
    | None -> uarch.Uarch.pressure_threshold
  in

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

  (* Port-class machinery (Option B: minimal P0/P1 vs P5 balancing).
   *
   * Background: on Ice Lake server, Mul/FMA can only dispatch to P0 or P1.
   * Add/Sub/Neg can dispatch to P0, P1, or P5. When the scheduler emits a
   * long mul/fma-heavy stretch followed by adds/subs, the P0/P1 ports get
   * over-pressured while P5 sits idle. gcc-11's post-RA scheduler reorders
   * to interleave add/sub work into mul/fma-heavy stretches, keeping P5
   * busy. llvm-mca measures the resulting cycle gap at ~2.5-3.4% across
   * R=32, R=64, R=128, R=256 c2c codelets on icelake-server.
   *
   * Option B: track a moving balance between P0/P1-only ops fired and
   * P5-capable ops fired. When the ready set contains both classes, prefer
   * the one whose pressure is "behind." This is a tiebreaker, not an
   * override — cp_dist DESC still wins for ops with significantly different
   * critical-path distance (gated by cp_window_for_port_balance below).
   *
   * Port classes:
   *   P01 (Mul/FMA-class): NK_Mul, NK_CmulRe, NK_CmulIm, NK_Fma
   *   P5  (Add/Sub-class): NK_Add, NK_Sub, NK_Neg
   *   Other: NK_Load, NK_Const (handled by existing load-credit/source-order logic) *)
  let port_class n =
    match n.Algsimp.node with
    | NK_Mul _ | NK_CmulRe _ | NK_CmulIm _ | NK_Fma _ -> `P01
    | NK_Add _ | NK_Sub _ | NK_Neg _ -> `P5
    | _ -> `Other
  in
  let p01_fired = ref 0 in
  let p5_fired = ref 0 in

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
          (* Port-balance tier (Option B): when cp_dist is tied, prefer the
           * op whose port class is "behind." This interleaves P0/P1-only
           * mul/fma ops with P5-capable add/sub ops, matching the pattern
           * gcc-11's post-RA scheduler produces. Tier comes AFTER cp_dist
           * (so critical path is preserved) but BEFORE su_num (port balance
           * matters more than the SU number tiebreaker for performance).
           *
           * Score: how much we want to fire this op based on current
           * imbalance. Higher = more desired.
           *   P01 op + p5 ahead  → score positive (catch up p01)
           *   P5  op + p01 ahead → score positive (catch up p5)
           *   matched → score 0  *)
          let port_balance_score n =
            match port_class n with
            | `P01 -> !p5_fired - !p01_fired
            | `P5  -> !p01_fired - !p5_fired
            | `Other -> 0
          in
          let pba = port_balance_score a in
          let pbb = port_balance_score b in
          if pba <> pbb then compare pbb pba  (* higher score first *)
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
      (* Update port-class counters for port-balance tiebreaker. *)
      (match port_class n with
       | `P01 -> incr p01_fired
       | `P5  -> incr p5_fired
       | `Other -> ());
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
