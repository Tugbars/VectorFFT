(* Bb.ml — Branch-and-bound cluster-local scheduler with lexicographic cost.
 *
 * Cost function: lexicographic (saturated_peak ASC, -progress ASC) where:
 *
 *   saturated_peak = max(peak_live, uarch.vec_regs)
 *
 *     Treats peak counts that already fit in the architectural register file
 *     as equivalent. Below the register count, peak is irrelevant — GCC has
 *     enough registers to allocate without spilling. Above the register count,
 *     each unit of peak corresponds to a real spill, so we want to minimize.
 *
 *   progress = sum over schedule of cp_dist[n_i] × (N - 1 - i)
 *
 *     Higher when high-cp_dist nodes are scheduled early. This is the
 *     latency-aware tiebreaker: among schedules with the same saturated
 *     peak, prefer the one that schedules critical-path-heavy nodes first.
 *     Maximizing progress matches what SU's primary key (cp_dist DESC) does
 *     at each step.
 *
 * Status: opt-in alternative to SU+GH. Empirically the schedules differ
 * structurally (~50% of lines reordered) from SU+GH but the cost-tied result
 * means runtime is roughly equivalent on Raptor Lake. R=64 AVX2 shows a real
 * K-regime crossover: GH wins at small K, BB wins at K=512-1024 (+5.8%).
 * Other uarchs may favor BB-lex for reasons we haven't measured. See doc 22.
 *
 * Strategy:
 *   - SU+GH baseline → initial (peak, progress)
 *   - DFS: enumerate ready nodes in pressure order (lowest delta first) so
 *     the first leaf is good; backtrack for alternatives.
 *   - Prune: saturated_peak strictly worse, or saturated_peak tied and
 *     remaining-progress upper bound can't beat best.
 *   - Time check on each branch.
 *
 * Limitation: the search space is N! pre-pruning. For clusters > ~50 ops
 * the time budget will typically expire before exhaustive search completes;
 * best-found-so-far is returned.  *
 * ------------------------------------------------------------------
 * MODULE CARD (bb.ml — grep "MODULE CARD" for the full set)
 * ROLE: Time-boxed branch-and-bound cluster scheduler with lexicographic
 * cost; drop-in alternative to su_schedule_subset when a bb budget
 * is granted.
 * PIPELINE: Schedule.subset call sites -> [Bb] (budgeted)
 * PUBLIC SURFACE (measured; external callers in parens): bb_schedule_subset(4)
 * DEPS: Algsimp(36), Schedule(3), Uarch(3).
 * USED-BY: emit_c(3), emit_render(1).
 * ------------------------------------------------------------------
 *)

(* Re-export of Ir.preds — kept under the historical name for the
 * bb_diagnostic CLI. New call sites should use Ir.preds directly. *)
let preds_of = Ir.preds

(* Compute peak-live for an existing schedule. *)
let compute_peak_live ~(subset : Ir.t list) ~(sinks : Ir.t list) (schedule : Ir.t list)
  : int
  =
  let in_subset : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add in_subset n.Ir.tag ()) subset;
  let sink_set : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  List.iter (fun s -> Hashtbl.replace sink_set s.Ir.tag ()) sinks;
  let remaining_users : (int, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add remaining_users n.Ir.tag 0) subset;
  List.iter
    (fun n ->
       List.iter
         (fun p ->
            if Hashtbl.mem in_subset p.Ir.tag
            then (
              let cur =
                try Hashtbl.find remaining_users p.tag with
                | Not_found -> 0
              in
              Hashtbl.replace remaining_users p.tag (cur + 1)))
         (preds_of n))
    subset;
  let live : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  let peak = ref 0 in
  List.iter
    (fun n ->
       let preds_in_sub =
         List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds_of n)
       in
       List.iter
         (fun p ->
            let cur =
              try Hashtbl.find remaining_users p.Ir.tag with
              | Not_found -> 0
            in
            let new_count = cur - 1 in
            Hashtbl.replace remaining_users p.tag new_count;
            if new_count = 0 && not (Hashtbl.mem sink_set p.tag)
            then Hashtbl.remove live p.tag)
         preds_in_sub;
       let n_users =
         try Hashtbl.find remaining_users n.Ir.tag with
         | Not_found -> 0
       in
       let is_sink = Hashtbl.mem sink_set n.tag in
       if n_users > 0 || is_sink then Hashtbl.replace live n.tag ();
       let l = Hashtbl.length live in
       if l > !peak then peak := l)
    schedule;
  !peak
;;

(* Compute total cp_progress for an existing schedule, given cp_dist table.
 * progress = sum_i cp_dist[node_i] * (N - 1 - i). *)
let compute_progress (schedule : Ir.t list) (cp_dist : (int, int) Hashtbl.t) : int =
  let n = List.length schedule in
  let total = ref 0 in
  List.iteri
    (fun i node ->
       let cp =
         try Hashtbl.find cp_dist node.Ir.tag with
         | Not_found -> 0
       in
       total := !total + (cp * (n - 1 - i)))
    schedule;
  !total
;;

(* B&B core with lexicographic cost. Returns best schedule + (peak, progress). *)
let bb_search
      ~(uarch : Uarch.t)
      ~(subset : Ir.t list)
      ~(sinks : Ir.t list)
      ~(initial_schedule : Ir.t list)
      ~(initial_peak : int)
      ~(time_budget_sec : float)
  : Ir.t list * int * int
  =
  let vec_regs = uarch.Uarch.vec_regs in
  let saturate p = max p vec_regs in
  let in_subset : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add in_subset n.Ir.tag ()) subset;
  let sink_set : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  List.iter (fun s -> Hashtbl.replace sink_set s.Ir.tag ()) sinks;
  let cp_dist = Schedule.compute_cp_dist uarch sinks subset in
  let cp_of n =
    try Hashtbl.find cp_dist n.Ir.tag with
    | Not_found -> 0
  in
  let total_cp = List.fold_left (fun acc n -> acc + cp_of n) 0 subset in
  let initial_progress = compute_progress initial_schedule cp_dist in
  let n_total = List.length subset in
  let initial_users : (int, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter (fun n -> Hashtbl.add initial_users n.Ir.tag 0) subset;
  List.iter
    (fun n ->
       List.iter
         (fun p ->
            if Hashtbl.mem in_subset p.Ir.tag
            then (
              let cur =
                try Hashtbl.find initial_users p.tag with
                | Not_found -> 0
              in
              Hashtbl.replace initial_users p.tag (cur + 1)))
         (preds_of n))
    subset;
  let initial_unsched : (int, int) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun n ->
       let in_subset_preds =
         List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds_of n)
       in
       Hashtbl.add initial_unsched n.tag (List.length in_subset_preds))
    subset;
  let users : (int, Ir.t list) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun n ->
       List.iter
         (fun p ->
            if Hashtbl.mem in_subset p.Ir.tag
            then (
              let cur =
                try Hashtbl.find users p.tag with
                | Not_found -> []
              in
              Hashtbl.replace users p.tag (n :: cur)))
         (preds_of n))
    subset;
  let best_schedule = ref initial_schedule in
  let best_peak = ref initial_peak in
  let best_sat_peak = ref (saturate initial_peak) in
  let best_progress = ref initial_progress in
  let start_time = Unix.gettimeofday () in
  let timed_out = ref false in
  let check_budget () =
    if (not !timed_out) && Unix.gettimeofday () -. start_time > time_budget_sec
    then timed_out := true;
    !timed_out
  in
  let remaining_users = Hashtbl.copy initial_users in
  let unsched_count = Hashtbl.copy initial_unsched in
  let live : (int, unit) Hashtbl.t = Hashtbl.create 32 in
  let initial_ready =
    List.filter (fun n -> Hashtbl.find initial_unsched n.Ir.tag = 0) subset
  in
  let scheduled_cp_sum = ref 0 in
  let rec dfs
            (partial : Ir.t list)
            (ready : Ir.t list)
            (peak : int)
            (progress : int)
            (step : int)
    : unit
    =
    if check_budget ()
    then ()
    else (
      let sat_peak = saturate peak in
      if sat_peak > !best_sat_peak
      then ()
      else if ready = []
      then (
        if
          sat_peak < !best_sat_peak
          || (sat_peak = !best_sat_peak && progress > !best_progress)
        then (
          best_peak := peak;
          best_sat_peak := sat_peak;
          best_progress := progress;
          best_schedule := List.rev partial))
      else (
        let prune_secondary =
          sat_peak = !best_sat_peak
          &&
          let remaining_cp_sum = total_cp - !scheduled_cp_sum in
          let max_weight = n_total - 1 - step in
          let progress_ub = progress + (remaining_cp_sum * max_weight) in
          progress_ub <= !best_progress
        in
        if prune_secondary
        then ()
        else (
          let pressure_delta n =
            let preds_in_sub =
              List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds_of n)
            in
            let kills =
              List.fold_left
                (fun acc p ->
                   let ru =
                     try Hashtbl.find remaining_users p.Ir.tag with
                     | Not_found -> 0
                   in
                   if ru = 1 then acc + 1 else acc)
                0
                preds_in_sub
            in
            let n_users =
              try Hashtbl.find remaining_users n.Ir.tag with
              | Not_found -> 0
            in
            let is_sink = Hashtbl.mem sink_set n.tag in
            let births = if n_users > 0 || is_sink then 1 else 0 in
            births - kills
          in
          let sorted_ready =
            List.sort
              (fun a b ->
                 let da = pressure_delta a in
                 let db = pressure_delta b in
                 if da <> db
                 then compare da db
                 else (
                   let ca = cp_of a in
                   let cb = cp_of b in
                   if ca <> cb then compare cb ca else compare a.Ir.tag b.tag))
              ready
          in
          List.iter
            (fun n ->
               if check_budget ()
               then ()
               else (
                 let preds_in_sub =
                   List.filter (fun p -> Hashtbl.mem in_subset p.Ir.tag) (preds_of n)
                 in
                 let saves = ref [] in
                 let died = ref [] in
                 List.iter
                   (fun p ->
                      let cur = Hashtbl.find remaining_users p.Ir.tag in
                      saves := (`User, p.tag, cur) :: !saves;
                      let new_count = cur - 1 in
                      Hashtbl.replace remaining_users p.tag new_count;
                      if
                        new_count = 0
                        && (not (Hashtbl.mem sink_set p.tag))
                        && Hashtbl.mem live p.tag
                      then (
                        Hashtbl.remove live p.tag;
                        died := p.tag :: !died))
                   preds_in_sub;
                 let n_users =
                   try Hashtbl.find remaining_users n.tag with
                   | Not_found -> 0
                 in
                 let is_sink = Hashtbl.mem sink_set n.tag in
                 let n_alive = n_users > 0 || is_sink in
                 if n_alive then Hashtbl.replace live n.tag ();
                 let new_peak = max peak (Hashtbl.length live) in
                 let weight = n_total - 1 - step in
                 let new_progress = progress + (cp_of n * weight) in
                 let cp_n = cp_of n in
                 scheduled_cp_sum := !scheduled_cp_sum + cp_n;
                 let new_ready_extra = ref [] in
                 let user_list =
                   try Hashtbl.find users n.tag with
                   | Not_found -> []
                 in
                 List.iter
                   (fun u ->
                      let cur = Hashtbl.find unsched_count u.Ir.tag in
                      saves := (`Unsched, u.tag, cur) :: !saves;
                      let new_count = cur - 1 in
                      Hashtbl.replace unsched_count u.tag new_count;
                      if new_count = 0 then new_ready_extra := u :: !new_ready_extra)
                   user_list;
                 let ready' =
                   List.filter (fun x -> x.Ir.tag <> n.tag) ready @ !new_ready_extra
                 in
                 dfs (n :: partial) ready' new_peak new_progress (step + 1);
                 scheduled_cp_sum := !scheduled_cp_sum - cp_n;
                 if n_alive then Hashtbl.remove live n.tag;
                 List.iter (fun tag -> Hashtbl.replace live tag ()) !died;
                 List.iter
                   (fun (kind, tag, old_val) ->
                      match kind with
                      | `User -> Hashtbl.replace remaining_users tag old_val
                      | `Unsched -> Hashtbl.replace unsched_count tag old_val)
                   !saves))
            sorted_ready)))
  in
  dfs [] initial_ready 0 0 0;
  !best_schedule, !best_peak, !best_progress
;;

(* Public entry: SU+GH then time-boxed B&B; replaces su_schedule_subset. *)
let bb_schedule_subset
      (uarch : Uarch.t)
      ~(time_budget_sec : float)
      ~(subset : Ir.t list)
      ~(sinks : Ir.t list)
  : Ir.t list
  =
  if subset = []
  then []
  else (
    let baseline = Schedule.su_schedule_subset uarch ~gh:true ~subset ~sinks in
    let baseline_peak = compute_peak_live ~subset ~sinks baseline in
    let best, _peak, _progress =
      bb_search
        ~uarch
        ~subset
        ~sinks
        ~initial_schedule:baseline
        ~initial_peak:baseline_peak
        ~time_budget_sec
    in
    best)
;;
