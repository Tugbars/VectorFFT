(* Bb.ml — Branch-and-bound cluster-local scheduler.
 *
 * For each cluster passed in, runs a time-budgeted DFS to find a schedule
 * with lower peak live count than what SU+GH produces. If none is found
 * within the budget, returns the SU+GH schedule unchanged.
 *
 * Cost function: peak live count (the count of unkilled, scheduled nodes
 * at the moment of maximum register pressure). This is the same quantity
 * GH approximates greedily; here we minimize it exactly within the budget.
 *
 * Strategy:
 *   - Start from SU+GH baseline → gives initial best-known peak
 *   - DFS: at each level, enumerate ready nodes in cmp_pressure order so
 *     the first leaf is "good"; backtracking finds alternatives.
 *   - Prune: current peak >= best_known_peak.
 *   - Time check: every N branches or per-leaf.
 *
 * On clusters where SU+GH is already optimal, this returns the same schedule.
 * On clusters where SU+GH was suboptimal, we get a better one.
 *
 * Limitation: the search space is N! pre-pruning. For clusters > ~50 ops
 * the time budget will typically be exhausted before exhaustive search
 * completes, but the best-found-so-far may still beat SU+GH. *)

let preds_of (e : Algsimp.t) : Algsimp.t list =
  match e.node with
  | NK_Const _ | NK_Load _ -> []
  | NK_Neg a -> [a]
  | NK_Add (a, b) | NK_Sub (a, b) | NK_Mul (a, b) -> [a; b]
  | NK_CmulRe (a, b, c, d) | NK_CmulIm (a, b, c, d) -> [a; b; c; d]

(* Compute peak-live for an existing schedule (subset and sinks).
 * This is the cost we're trying to minimize. *)
let compute_peak_live
    ~(subset : Algsimp.t list)
    ~(sinks : Algsimp.t list)
    (schedule : Algsimp.t list)
    : int =
  let in_subset : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add in_subset n.Algsimp.tag ()) subset;
  let sink_set : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  List.iter (fun s -> Hashtbl.replace sink_set s.Algsimp.tag ()) sinks;
  (* Initialize remaining_users from in-subset users. *)
  let remaining_users : (int, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add remaining_users n.Algsimp.tag 0) subset;
  List.iter (fun n ->
    List.iter (fun p ->
      if Hashtbl.mem in_subset p.Algsimp.tag then begin
        let cur = try Hashtbl.find remaining_users p.tag with Not_found -> 0 in
        Hashtbl.replace remaining_users p.tag (cur + 1)
      end
    ) (preds_of n)
  ) subset;
  (* Walk schedule, tracking live set. *)
  let live : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  let peak = ref 0 in
  List.iter (fun n ->
    let preds_in_sub = List.filter (fun p ->
      Hashtbl.mem in_subset p.Algsimp.tag
    ) (preds_of n) in
    List.iter (fun p ->
      let cur = try Hashtbl.find remaining_users p.Algsimp.tag with Not_found -> 0 in
      let new_count = cur - 1 in
      Hashtbl.replace remaining_users p.tag new_count;
      if new_count = 0 && not (Hashtbl.mem sink_set p.tag) then
        Hashtbl.remove live p.tag
    ) preds_in_sub;
    let n_users = try Hashtbl.find remaining_users n.Algsimp.tag with Not_found -> 0 in
    let is_sink = Hashtbl.mem sink_set n.tag in
    if n_users > 0 || is_sink then
      Hashtbl.replace live n.tag ();
    let l = Hashtbl.length live in
    if l > !peak then peak := l
  ) schedule;
  !peak

(* B&B core. Returns the best schedule found within the time budget,
 * along with its peak. *)
let bb_search
    ~(subset : Algsimp.t list)
    ~(sinks : Algsimp.t list)
    ~(initial_schedule : Algsimp.t list)
    ~(initial_peak : int)
    ~(time_budget_sec : float)
    : Algsimp.t list * int =
  let in_subset : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add in_subset n.Algsimp.tag ()) subset;
  let sink_set : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  List.iter (fun s -> Hashtbl.replace sink_set s.Algsimp.tag ()) sinks;

  (* Initialize remaining_users (will be mutated during DFS). *)
  let initial_users : (int, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add initial_users n.Algsimp.tag 0) subset;
  List.iter (fun n ->
    List.iter (fun p ->
      if Hashtbl.mem in_subset p.Algsimp.tag then begin
        let cur = try Hashtbl.find initial_users p.tag with Not_found -> 0 in
        Hashtbl.replace initial_users p.tag (cur + 1)
      end
    ) (preds_of n)
  ) subset;

  (* Initial unscheduled-pred counts (in-subset preds only). *)
  let initial_unsched : (int, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n ->
    let in_subset_preds = List.filter (fun p ->
      Hashtbl.mem in_subset p.Algsimp.tag
    ) (preds_of n) in
    Hashtbl.add initial_unsched n.tag (List.length in_subset_preds)
  ) subset;

  (* Successor map for ready-set updates. *)
  let users : (int, Algsimp.t list) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n ->
    List.iter (fun p ->
      if Hashtbl.mem in_subset p.Algsimp.tag then begin
        let cur = try Hashtbl.find users p.tag with Not_found -> [] in
        Hashtbl.replace users p.tag (n :: cur)
      end
    ) (preds_of n)
  ) subset;

  let best_schedule = ref initial_schedule in
  let best_peak = ref initial_peak in
  let start_time = Unix.gettimeofday () in
  let timed_out = ref false in
  let check_budget () =
    if not !timed_out
       && Unix.gettimeofday () -. start_time > time_budget_sec then
      timed_out := true;
    !timed_out
  in

  (* DFS state is held in mutable hashtables; we save/restore on each branch.
   * Specifically:
   *   - remaining_users: copied from initial; mutated by scheduling
   *   - unsched_count: copied from initial; mutated by scheduling
   *   - live: hashset, mutated
   *   - ready: list, passed by value
   *   - partial: schedule prefix in reverse order
   *   - peak: max live seen on this path
   *
   * To avoid copy overhead, we do undo-style: save (key,value) modifications
   * on a stack, restore on backtrack. *)
  let remaining_users = Hashtbl.copy initial_users in
  let unsched_count = Hashtbl.copy initial_unsched in
  let live : (int, unit) Hashtbl.t = Hashtbl.create 32 in

  (* Initial ready set. *)
  let initial_ready =
    List.filter (fun n ->
      Hashtbl.find initial_unsched n.Algsimp.tag = 0
    ) subset
  in

  (* DFS body. `partial` is reversed; `peak` is max live so far. *)
  let rec dfs (partial : Algsimp.t list) (ready : Algsimp.t list) (peak : int)
      : unit =
    if check_budget () then ()
    else if peak >= !best_peak then ()  (* prune *)
    else if ready = [] then begin
      (* Complete schedule. Beats best? *)
      if peak < !best_peak then begin
        best_peak := peak;
        best_schedule := List.rev partial
      end
    end else begin
      (* Order ready by (delta ASC, tag ASC) — pressure-aware exploration. *)
      let pressure_delta n =
        let preds_in_sub = List.filter (fun p ->
          Hashtbl.mem in_subset p.Algsimp.tag
        ) (preds_of n) in
        let kills = List.fold_left (fun acc p ->
          let ru = try Hashtbl.find remaining_users p.Algsimp.tag with Not_found -> 0 in
          if ru = 1 then acc + 1 else acc
        ) 0 preds_in_sub in
        let n_users = try Hashtbl.find remaining_users n.Algsimp.tag with Not_found -> 0 in
        let is_sink = Hashtbl.mem sink_set n.tag in
        let births = if n_users > 0 || is_sink then 1 else 0 in
        births - kills
      in
      let sorted_ready = List.sort (fun a b ->
        let da = pressure_delta a in
        let db = pressure_delta b in
        if da <> db then compare da db
        else compare a.Algsimp.tag b.tag
      ) ready in
      List.iter (fun n ->
        if check_budget () || peak >= !best_peak then ()
        else begin
          (* Schedule n: save state for undo. *)
          let preds_in_sub = List.filter (fun p ->
            Hashtbl.mem in_subset p.Algsimp.tag
          ) (preds_of n) in
          let saves = ref [] in
          let died = ref [] in   (* preds that died this step *)
          List.iter (fun p ->
            let cur = Hashtbl.find remaining_users p.Algsimp.tag in
            saves := (`User, p.tag, cur) :: !saves;
            let new_count = cur - 1 in
            Hashtbl.replace remaining_users p.tag new_count;
            if new_count = 0 && not (Hashtbl.mem sink_set p.tag)
               && Hashtbl.mem live p.tag then begin
              Hashtbl.remove live p.tag;
              died := p.tag :: !died
            end
          ) preds_in_sub;
          let n_users = try Hashtbl.find remaining_users n.tag with Not_found -> 0 in
          let is_sink = Hashtbl.mem sink_set n.tag in
          let n_alive = (n_users > 0 || is_sink) in
          if n_alive then Hashtbl.replace live n.tag ();
          let new_peak = max peak (Hashtbl.length live) in

          (* Update ready set: drop n, add successors that just became ready. *)
          let new_ready_extra = ref [] in
          let user_list = try Hashtbl.find users n.tag with Not_found -> [] in
          List.iter (fun u ->
            let cur = Hashtbl.find unsched_count u.Algsimp.tag in
            saves := (`Unsched, u.tag, cur) :: !saves;
            let new_count = cur - 1 in
            Hashtbl.replace unsched_count u.tag new_count;
            if new_count = 0 then new_ready_extra := u :: !new_ready_extra
          ) user_list;
          let ready' =
            List.filter (fun x -> x.Algsimp.tag <> n.tag) ready
            @ !new_ready_extra
          in

          dfs (n :: partial) ready' new_peak;

          (* Undo. *)
          if n_alive then Hashtbl.remove live n.tag;
          List.iter (fun tag -> Hashtbl.replace live tag ()) !died;
          List.iter (fun (kind, tag, old_val) ->
            match kind with
            | `User -> Hashtbl.replace remaining_users tag old_val
            | `Unsched -> Hashtbl.replace unsched_count tag old_val
          ) !saves
        end
      ) sorted_ready
    end
  in

  dfs [] initial_ready 0;
  (!best_schedule, !best_peak)

(* Public entry: drop-in replacement for su_schedule_subset.
 * Runs SU+GH first, then B&B with time budget. *)
let bb_schedule_subset (uarch : Uarch.t)
    ~(time_budget_sec : float)
    ~(subset : Algsimp.t list)
    ~(sinks : Algsimp.t list)
    : Algsimp.t list =
  if subset = [] then []
  else begin
    let baseline = Schedule.su_schedule_subset uarch ~gh:true ~subset ~sinks in
    let baseline_peak = compute_peak_live ~subset ~sinks baseline in
    let (best, _best_peak) =
      bb_search ~subset ~sinks
        ~initial_schedule:baseline
        ~initial_peak:baseline_peak
        ~time_budget_sec
    in
    best
  end
