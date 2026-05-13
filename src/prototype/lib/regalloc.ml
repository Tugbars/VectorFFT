(* regalloc.ml — SSA-based register allocator (M1: types + stub).
 *
 * === ROLE IN THE PIPELINE ===
 *
 * Sits between Schedule (which produces an ordered list of Algsimp.t)
 * and Emit_c's declaration emission. Maps each tag to either a
 * concrete physical register name OR Default (fall through to the
 * existing emit_c behavior, where gcc picks the register).
 *
 * The eventual goal: emit `register __m512d tN asm("zmm5") = ...;`
 * for every tag, bypassing gcc's register allocator entirely. That
 * gives compiler-agnostic output — gcc-11, gcc-13, and clang all see
 * the same register assignment instead of each running their own RA
 * pass and producing different code.
 *
 * === STAGING ===
 *
 * M1 (this file): just the types + a stub `allocate_stub` that
 *   returns an empty allocation. Every tag falls through to Default,
 *   so output is byte-identical to a build without regalloc.
 *
 * M2: live-range analysis. Compute peak_live per codelet; report it
 *   for sanity-checking against the existing pass-1/pass-2 split.
 *
 * M3: real chordal greedy coloring for within-budget codelets (R<=32
 *   on AVX-512). Emit_c starts consuming the allocation.
 *
 * M4: validation cells. R=4 through R=32 across AVX-512 and AVX2 with
 *   gcc-11.
 *
 * M5: pre-spilling for codelets that exceed per-pass register budget
 *   (R=64 on AVX2, R=128/256 anywhere).
 *
 * M6: extend to R=128, R=256.
 *
 * === DESIGN NOTES ===
 *
 * 1. SSA -> chordal interference graph (Hack & Goos 2006). Since our
 *    IR is SSA-equivalent via hashconsing (each tag is single
 *    assignment), the live-range interference graph is chordal.
 *    Chordal graphs are optimally colorable in polynomial time via
 *    greedy coloring along a perfect elimination ordering. M3
 *    implements this; for now we only need the type machinery.
 *
 * 2. We do NOT decide spilling here. The pass-1/pass-2 spill markers
 *    from Algsimp.lift_spill_markers (consumed by Emit_c.make_spill_info)
 *    pre-decide which values cross the pass boundary. Within each
 *    pass, chordal coloring tells us what fits. If a pass exceeds
 *    register budget after that, M5's pre-spilling adds finer-grained
 *    scratchpad evictions.
 *
 * 3. The `Default` assignment is the escape hatch: any tag we don't
 *    explicitly bind gets emit_c's existing `const __m512d tN = ...;`
 *    treatment, which leaves register choice to gcc. Useful for:
 *      - tags that get inlined by `should_inline` and never named
 *      - the "fused slot" forward-declared values (see emit_c.ml ~256)
 *      - any case we haven't yet handled
 *    This is what makes M1 a no-op: every tag is Default. *)

(* What an allocator decides per tag. *)
type assignment =
  | Reg of string
    (* Concrete physical register name, e.g. "zmm5" for AVX-512 or
     * "ymm3" for AVX2. Will trigger emit_c to emit
     *   register %s tN asm("%s") = ...;
     * instead of
     *   const %s tN = ...;  *)
  | Default
    (* No specific binding -- fall through to emit_c's existing
     * declaration (const + gcc-allocated). This is the M1 stub state
     * for every tag. *)

(* Allocation map: tag -> assignment.
 *
 * We carry the Isa.t alongside so emit_c doesn't need to thread it
 * through separately when consuming the allocation. (Emit_c already
 * has its own isa; this is for the regalloc's own internal use when
 * future passes need to know the vec_type / register count.) *)
type allocation = {
  isa : Isa.t;
  assign : (int, assignment) Hashtbl.t;
}

(* === M1 stub allocator ===
 *
 * Returns an allocation with no tag bindings -- every lookup falls
 * through to Default. This is the baseline state we'll build on in
 * M2/M3. Called by emit_c at M3+ when no other allocator is
 * specified, or by tests that want a controlled no-op.
 *
 * `scheduled` is the scheduled node order from Schedule; M1 ignores
 * it but takes it as a parameter to keep the signature stable for
 * M2/M3 which will use it. *)
let allocate_stub
    ~(isa : Isa.t)
    ~(scheduled : Algsimp.t list)
    : allocation =
  let _ = scheduled in
  { isa; assign = Hashtbl.create 16 }

(* === Lookup ===
 *
 * Get the assignment for a tag, defaulting to `Default` if the tag
 * isn't in the table. Cheap and idempotent. Emit_c will call this
 * once per `render_node_def` invocation at M3+. *)
let lookup (alloc : allocation) (tag : int) : assignment =
  match Hashtbl.find_opt alloc.assign tag with
  | Some a -> a
  | None -> Default

(* === Convenience: count bindings ===
 *
 * Diagnostic helper. Returns how many tags are bound to a concrete
 * register vs left at Default. M3+ tests use this to confirm "we
 * actually allocated something." *)
let count_bindings (alloc : allocation) : int * int =
  let regs = ref 0 in
  let defaults = ref 0 in
  Hashtbl.iter (fun _ a ->
    match a with
    | Reg _ -> incr regs
    | Default -> incr defaults
  ) alloc.assign;
  (!regs, !defaults)

(* === M2: Live-range / peak-live analysis ===
 *
 * Walks a scheduled list of Algsimp.t nodes and computes the maximum
 * number of simultaneously live values. This is the upper bound on
 * register usage if every scheduled node materializes as a named
 * temporary.
 *
 * Output is conservative — we count ALL scheduled tags as needing a
 * register, even ones that emit_c's `should_inline` would fold into a
 * consumer's RHS. The real RA pass at M3 will refine this. For now:
 * if peak_live (naive) <= isa.vec_regs, we're definitely safe; if it
 * exceeds, we need to consider inlining and/or spilling.
 *
 * Algorithm (standard):
 *   1. First pass: compute last_use[tag] = latest schedule position
 *      where tag is consumed (as a pred). For a tag never used as a
 *      pred (e.g., a sink output), last_use defaults to its own
 *      definition position — it's live for exactly one instruction.
 *
 *   2. Second pass: walk schedule. At each position i with node N:
 *        - Add N to `live` (N's value just got defined).
 *        - Snapshot |live| — this is the peak for this instruction.
 *        - Kill any tag whose last_use == i (its last consumer is here).
 *
 *   3. Peak = max snapshot across all positions.
 *
 * SSA cleanliness: since hashconsing means each tag has exactly one
 * definition site, `last_use` is well-defined. Pred references don't
 * introduce extra definitions. *)

type live_info = {
  peak_live : int;     (* maximum simultaneously live values *)
  peak_at   : int;     (* schedule position where peak occurred *)
  n_nodes   : int;     (* total scheduled node count *)
  budget    : int;     (* isa.vec_regs *)
  fits      : bool;    (* peak_live <= budget *)
}

let peak_live_analysis
    ~(isa : Isa.t)
    ~(scheduled : Algsimp.t list)
    : live_info =
  let budget = isa.vec_regs in
  let n_nodes = List.length scheduled in

  (* Pass 1: build last_use[tag] = latest position consuming tag.
   *
   * We process schedule in order. For each node at position i:
   *   - Tentatively set last_use[N.tag] = i (in case N is never
   *     consumed downstream — it's live exactly through position i).
   *   - For each pred P of N: set last_use[P.tag] = i (P is consumed
   *     here; later iterations may overwrite to a larger i). *)
  let last_use : (int, int) Hashtbl.t = Hashtbl.create n_nodes in
  List.iteri (fun i (e : Algsimp.t) ->
    Hashtbl.replace last_use e.tag i;
    List.iter (fun (p : Algsimp.t) ->
      Hashtbl.replace last_use p.tag i
    ) (Algsimp.preds e)
  ) scheduled;

  (* Pass 2: walk schedule forward, track live set. *)
  let live : (int, unit) Hashtbl.t = Hashtbl.create budget in
  let peak = ref 0 in
  let peak_at = ref 0 in
  List.iteri (fun i (e : Algsimp.t) ->
    (* The new value N is defined at i. It's now live. *)
    Hashtbl.replace live e.tag ();
    let cur = Hashtbl.length live in
    if cur > !peak then begin
      peak := cur;
      peak_at := i
    end;
    (* Kill anything whose last_use is this position. Includes N
     * itself if N has no consumer downstream (sink case). Also kills
     * any pred whose last consumer is here. We use preds-only +
     * self-check for efficiency rather than scanning all of `live`. *)
    let kill (tag : int) =
      match Hashtbl.find_opt last_use tag with
      | Some lu when lu = i -> Hashtbl.remove live tag
      | _ -> ()
    in
    kill e.tag;
    List.iter (fun (p : Algsimp.t) -> kill p.tag) (Algsimp.preds e)
  ) scheduled;

  { peak_live = !peak;
    peak_at = !peak_at;
    n_nodes;
    budget;
    fits = !peak <= budget;
  }

(* === Diagnostic print ===
 *
 * Emit a stderr-style summary. Useful during M2-M5 to track how
 * peak-live evolves as we add pre-spilling, coalescing, etc. *)
let format_live_info (info : live_info) : string =
  Printf.sprintf
    "peak_live=%d/%d at pos %d/%d %s"
    info.peak_live info.budget info.peak_at info.n_nodes
    (if info.fits then "[FITS]" else "[OVERFLOW]")

(* === M3a: linear-scan allocator (chordal-coloring for SSA) ===
 *
 * For SSA programs (which ours are via hashconsing), the live-range
 * interference graph is chordal (Hack & Goos 2006). Chordal graphs
 * admit optimal greedy coloring in linear time, which on the
 * scheduled order reduces to: linear scan, allocate at def, free at
 * last use.
 *
 * The "register" here is a *color* drawn from a finite budget. We
 * use AVX-512's ZMM0-ZMM27 (28 colors, leaving ZMM28-ZMM31 for gcc's
 * ABI / temporary needs as discussed) or AVX2's YMM0-YMM15 (less
 * than peak_live for everything we care about, so AVX2 needs M5).
 *
 * Algorithm:
 *   free_pool  : sorted list of available color indices, [0..budget-1]
 *   live_table : tag -> color (currently allocated)
 *   last_use   : tag -> last position in scheduled order (from peak_live)
 *
 *   For i, node in enumerate(scheduled):
 *     - Free colors of tags whose last_use < i (strictly before).
 *     - Pop the smallest free color, assign to node.tag.
 *     - If pool empty -> Overflow (pre-spilling needed; M5).
 *
 * Implementation notes:
 *
 * 1. The "free dead" step happens BEFORE allocating the new def.
 *    Tags with last_use = i are still alive during instruction i
 *    (they're consumed by it); they get freed at the start of i+1.
 *
 * 2. The smallest-color-first pop is conventional. Not strictly
 *    necessary for correctness — any free color works — but reduces
 *    fragmentation and keeps low-numbered ZMMs preferred (which gcc's
 *    own RA also tends to prefer; reduces moves at codelet boundary).
 *
 * 3. We skip allocation for:
 *      - Nodes in skip_tags: passed in by caller (inlined, fused,
 *        spilled — emit_c handles these specially)
 *    Skipped nodes do NOT consume a color but DO advance the scheduled
 *    position counter so last_use lookups remain consistent.
 *
 * 4. On overflow we return an Overflow result. The caller decides
 *    whether to retry with a larger budget (M3a doesn't apply for
 *    that codelet), fall back to current behavior (Default
 *    everywhere), or escalate to pre-spilling.
 *
 * Naming: zmm/ymm followed by the color index. AVX-512: 0..31 valid
 * names. AVX2: 0..15 valid. We do not return higher color names than
 * budget-1; the budget caller-supplied controls this. *)

type alloc_result =
  | Allocated of allocation
  | Overflow of int    (* color budget; tells caller how short we were *)

let reg_name_of_isa (isa : Isa.t) (color : int) : string =
  match isa.vec_width with
  | 8 -> Printf.sprintf "zmm%d" color    (* AVX-512 *)
  | 4 -> Printf.sprintf "ymm%d" color    (* AVX2 *)
  | _ -> failwith (Printf.sprintf
           "Regalloc.reg_name_of_isa: unsupported vec_width=%d"
           isa.vec_width)

let allocate_linear_scan
    ~(isa : Isa.t)
    ~(scheduled : Algsimp.t list)
    ~(budget : int)
    ?(skip_tags : (int, unit) Hashtbl.t option = None)
    ?(inline_set : (int, unit) Hashtbl.t option = None)
    ?(force_last_use : (int, int) Hashtbl.t option = None)
    ()
    : alloc_result =
  let n = List.length scheduled in
  let is_inlined tag =
    match inline_set with
    | None -> false
    | Some tbl -> Hashtbl.mem tbl tag
  in

  (* Compute last_use[tag] for each tag.
   *
   * SUBTLE 1: emit_c inlines single-use nodes (those in inline_set) into
   * their consumer's RHS expression. So when consumer Y at position p
   * references an inlined node X, X's preds (call them A,B) are
   * ACTUALLY referenced at p in the emitted code — not at X's
   * scheduled position. Walk transitively through inlined preds.
   *
   * SUBTLE 2: some tags are referenced by emit_c's store emission at
   * positions LATER than any IR pred relation captures. Specifically:
   * pass2 output tags get stored via cluster flush at a position
   * determined by emit_c, not the IR DAG. The caller passes in
   * force_last_use, a map of tag -> position. After the standard
   * last_use computation, we apply force_last_use as a lower bound:
   * last_use[t] = max(computed last_use, force_last_use[t]).
   *
   * Without this, output stores read stale registers (the register's
   * value was clobbered by a later allocation before the store). *)
  let last_use : (int, int) Hashtbl.t = Hashtbl.create n in
  let rec walk_pred (e : Algsimp.t) (pos : int) =
    Hashtbl.replace last_use e.tag pos;
    if is_inlined e.tag then
      List.iter (fun p -> walk_pred p pos) (Algsimp.preds e)
  in
  List.iteri (fun i (e : Algsimp.t) ->
    Hashtbl.replace last_use e.tag i;
    List.iter (fun (p : Algsimp.t) -> walk_pred p i) (Algsimp.preds e)
  ) scheduled;
  (* Apply force_last_use as a lower bound. *)
  (match force_last_use with
   | None -> ()
   | Some tbl ->
     Hashtbl.iter (fun tag forced_pos ->
       match Hashtbl.find_opt last_use tag with
       | Some cur when cur >= forced_pos -> ()  (* already later *)
       | _ -> Hashtbl.replace last_use tag forced_pos
     ) tbl);

  (* Sorted available color pool. Use a sorted list of ints for the
   * smallest-color-first pop; for budget=28 this is fine without
   * needing a priority queue. *)
  let free_pool = ref (List.init budget (fun c -> c)) in

  (* tag -> color *)
  let allocated : (int, int) Hashtbl.t = Hashtbl.create budget in
  let result : (int, assignment) Hashtbl.t = Hashtbl.create n in

  let is_skipped tag =
    match skip_tags with
    | None -> false
    | Some tbl -> Hashtbl.mem tbl tag
  in

  let release_dead i =
    (* Walk currently-allocated tags; if last_use < i, free their color. *)
    let to_free = Hashtbl.fold (fun tag color acc ->
      match Hashtbl.find_opt last_use tag with
      | Some lu when lu < i -> (tag, color) :: acc
      | _ -> acc
    ) allocated [] in
    List.iter (fun (tag, color) ->
      Hashtbl.remove allocated tag;
      free_pool := List.sort compare (color :: !free_pool)
    ) to_free
  in

  let overflow_at = ref (-1) in

  (try
    List.iteri (fun i (e : Algsimp.t) ->
      release_dead i;
      (* Skip allocation for inlined nodes — they don't get their own
       * variable declaration. The value is materialized inside the
       * consumer's RHS expression, which uses the CONSUMER's
       * allocated register as its destination. *)
      if not (is_skipped e.tag) && not (is_inlined e.tag) then begin
        match !free_pool with
        | [] ->
          overflow_at := i;
          raise Exit
        | color :: rest ->
          free_pool := rest;
          Hashtbl.replace allocated e.tag color;
          Hashtbl.replace result e.tag (Reg (reg_name_of_isa isa color))
      end
    ) scheduled
   with Exit -> ());

  if !overflow_at >= 0 then
    Overflow budget
  else
    Allocated { isa; assign = result }

(* === Convenience wrapper ===
 *
 * Pulls budget from a caller-specified value (defaults to
 * isa.vec_regs - 4 to leave headroom for gcc's ABI / temporary
 * needs, as discussed for M3a). *)
let allocate
    ~(isa : Isa.t)
    ~(scheduled : Algsimp.t list)
    ?(budget : int option)
    ?(skip_tags : (int, unit) Hashtbl.t option = None)
    ?(inline_set : (int, unit) Hashtbl.t option = None)
    ?(force_last_use : (int, int) Hashtbl.t option = None)
    ()
    : alloc_result =
  let b = match budget with
    | Some b -> b
    | None   -> isa.vec_regs - 4    (* AVX-512: 32-4=28; AVX2: 16-4=12 *)
  in
  allocate_linear_scan ~isa ~scheduled ~budget:b
    ~skip_tags ~inline_set ~force_last_use ()