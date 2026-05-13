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
  | Spilled of int
    (* M5: this tag's value is computed once at its definition site
     * (into a register), then immediately stored to regalloc_spill[N]
     * where N is the slot index. The original tT identifier is NOT
     * used after the def — emit_c instead emits reload variables
     * tT_rK at each use site (a fresh K per reload). The reload
     * variables are register-pinned and short-lived.
     *
     * The slot is a single-vector scratch slot in a regalloc_spill[]
     * array independent from the structural spill_re/spill_im
     * (which serve cross-pass values; M5 spills are intra-pass). *)
  | Default
    (* No specific binding -- fall through to emit_c's existing
     * declaration (const + gcc-allocated). This is the M1 stub state
     * for every tag. *)

(* Allocation map: tag -> assignment, plus M5 spill/reload metadata.
 *
 * M3a fields:
 *   isa, assign
 *
 * M5 fields:
 *   spilled_tags     : map from tag -> slot. Duplicates info in `assign`
 *                       but separated for easy iteration ("emit a
 *                       declaration of regalloc_spill[N]").
 *   num_spill_slots  : count of distinct slots used. emit_c declares
 *                       regalloc_spill[num_spill_slots] at function
 *                       scope. Zero means no scratchpad needed.
 *   reload_sites     : map from schedule position -> list of (tag, name,
 *                       reg). Just before emitting node at position p,
 *                       emit_c emits a reload-declaration for each
 *                       entry. The `name` is the fresh shadow name
 *                       (e.g. "t42_r0", "t42_r1") to use INSTEAD of
 *                       "t42" in the next node's RHS. *)
type reload_decl = {
  reload_tag  : int;       (* tag being reloaded *)
  reload_name : string;    (* fresh shadow name, e.g. "t42_r0" *)
  reload_reg  : string;    (* destination register, e.g. "zmm5" *)
  reload_slot : int;       (* source slot in regalloc_spill[] *)
}

type allocation = {
  isa : Isa.t;
  assign : (int, assignment) Hashtbl.t;
  (* M5 spilling state. Empty for M3a-style allocations. *)
  num_spill_slots : int;
  reload_sites    : (int, reload_decl list) Hashtbl.t;
  (* M5: which positions trigger a spill store, and for which tag.
   * spill_sites[pos] = list of (tag, slot) — AFTER emitting node at
   * pos, emit a storeu_pd(&regalloc_spill[slot], tT) for each entry.
   * This is how a previously-allocated Reg tag transitions to "has
   * been spilled, must be reloaded for future uses." *)
  spill_sites     : (int, (int * int) list) Hashtbl.t;
  (* M5: tag -> spill slot for tags that were ever spilled. emit_c
   * uses this to (a) detect that a tag is spilled even without a
   * name_override entry, and (b) emit an inline-from-memory store
   * pattern as a fallback when no reload register was registered.
   *
   * `spilled_of_tag[t] = slot` iff tag t is spilled (at some point).
   * Inverse: spill_sites maps positions; this maps tags directly. *)
  spilled_of_tag  : (int, int) Hashtbl.t;
  (* name_overrides[(pos, tag)] = name to use for `tag` in the node
   * emitted at scheduled position `pos`. If absent, use the default
   * "t<tag>" name. Updated after each reload — the reload at position
   * p installs an override that's used by all positions p, p+1, ...
   * until the next reload of the same tag. *)
  name_overrides  : (int * int, string) Hashtbl.t;
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
  { isa;
    assign = Hashtbl.create 16;
    num_spill_slots = 0;
    reload_sites = Hashtbl.create 0;
    spill_sites = Hashtbl.create 0;
    spilled_of_tag = Hashtbl.create 0;
    name_overrides = Hashtbl.create 0;
  }

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
    | Spilled _ -> incr regs  (* spilled tags still occupy a register at def *)
    | Default -> incr defaults
  ) alloc.assign;
  (!regs, !defaults)

(* M5: count distinct spilled tags. Useful for diagnostics. *)
let count_spilled (alloc : allocation) : int =
  let n = ref 0 in
  Hashtbl.iter (fun _ a ->
    match a with
    | Spilled _ -> incr n
    | _ -> ()
  ) alloc.assign;
  !n

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
    Allocated { isa; assign = result;
                num_spill_slots = 0;
                reload_sites = Hashtbl.create 0;
                spill_sites = Hashtbl.create 0;
                spilled_of_tag = Hashtbl.create 0;
                name_overrides = Hashtbl.create 0;
              }

(* === Convenience wrapper ===
 *
 * Tries M3a's linear scan first. If it overflows, falls back to M5's
 * spilling allocator (which never overflows — Belady ensures we
 * always make progress).
 *
 * Gated by env var VFFT_USE_REGALLOC_M5=1. When unset, overflow
 * causes a fallback to the Default (gcc-allocated) path.
 *
 * Defined after allocate_with_spilling below. *)

(* ============================================================ *)
(* M5: SPILLING ALLOCATOR                                       *)
(* ============================================================ *)
(*
 * When the linear scan would overflow the color budget, the M5
 * allocator picks a currently-live tag to evict (Belady-style: the
 * tag whose NEXT USE is furthest in the future), assigns it a spill
 * slot in a separate regalloc_spill[] scratchpad, and continues.
 *
 * At each subsequent use of a spilled tag, the allocator records a
 * reload at that position: emit_c must emit a reload declaration
 * (loading from regalloc_spill[slot] into a fresh register-pinned
 * variable tT_rK) just before emitting that position's node, and use
 * the fresh variable name in the node's RHS instead of the original
 * tT.
 *
 * Key data structures:
 *   uses[tag]       : sorted list of positions where tag is used.
 *                     (Computed via the same transitive walk as last_use,
 *                     but kept as a list to find next_use after any pos.)
 *   active_use_idx[tag]  : index into uses[tag] of the next-future use.
 *                          Advanced as we walk the schedule.
 *
 * The Belady choice: when free_pool is empty at position p, pick the
 * tag t in `allocated` that maximizes uses[t].(active_use_idx[t]) —
 * the tag used furthest in the future. Spill that tag. Its register
 * goes back to the free pool.
 *
 * Reload sites: walking forward, the FIRST use of a spilled tag at
 * a position p AFTER the spill is the reload site. We record
 * (position, tag, fresh_name, reg, slot) in reload_sites.
 *
 * Subsequent uses of the same spilled tag — if the reload variable
 * is still alive in its register at that point — can keep using the
 * same reload name. But if THAT reload variable's register also gets
 * clobbered (by further allocations), we need ANOTHER reload site.
 *
 * For M5a we take a simpler model: SPILL ONCE, RELOAD EACH USE.
 * Every use of a spilled tag emits a fresh reload. Suboptimal
 * (extra loads when uses are clustered) but always correct and
 * vastly simpler than tracking reload-variable lifetimes recursively.
 * Optimization deferred to M6+. *)

let allocate_with_spilling
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
  let is_skipped tag =
    match skip_tags with
    | None -> false
    | Some tbl -> Hashtbl.mem tbl tag
  in

  (* === Pass 1: compute uses[tag] = list of positions referencing tag ===
   *
   * Same transitive walk through inlined preds as the linear scan,
   * but instead of just tracking last_use (max position), we collect
   * ALL positions where the tag is used. This lets us answer
   * "what's the NEXT use of tag t after position p" — needed for
   * Belady eviction. *)
  let uses : (int, int list ref) Hashtbl.t = Hashtbl.create n in
  let record_use tag pos =
    let lst = try Hashtbl.find uses tag
              with Not_found ->
                let r = ref [] in
                Hashtbl.add uses tag r;
                r in
    (* Avoid duplicates from same-position uses (e.g. same pred listed twice) *)
    match !lst with
    | p :: _ when p = pos -> ()
    | _ -> lst := pos :: !lst
  in
  let rec walk_pred (e : Algsimp.t) (pos : int) =
    record_use e.tag pos;
    if is_inlined e.tag then
      List.iter (fun p -> walk_pred p pos) (Algsimp.preds e)
  in
  List.iteri (fun i (e : Algsimp.t) ->
    record_use e.tag i;
    List.iter (fun (p : Algsimp.t) -> walk_pred p i) (Algsimp.preds e)
  ) scheduled;
  (* Apply force_last_use: add the forced position to uses[tag]. *)
  (match force_last_use with
   | None -> ()
   | Some tbl ->
     Hashtbl.iter (fun tag forced_pos ->
       record_use tag forced_pos
     ) tbl);
  (* Reverse lists so they're in ascending position order. *)
  let uses_sorted : (int, int array) Hashtbl.t = Hashtbl.create n in
  Hashtbl.iter (fun tag lst ->
    let arr = Array.of_list (List.rev !lst) in
    Hashtbl.replace uses_sorted tag arr
  ) uses;

  (* === Helper: next_use_after tag pos ===
   *
   * Returns Some p where p > pos is the next use of tag, or None if
   * no more uses. Linear scan over uses[tag] — could binary-search
   * if we cared. *)
  let next_use_after tag pos =
    match Hashtbl.find_opt uses_sorted tag with
    | None -> None
    | Some arr ->
      let len = Array.length arr in
      let rec find i =
        if i >= len then None
        else if arr.(i) > pos then Some arr.(i)
        else find (i + 1)
      in find 0
  in

  (* === Pass 2: linear scan with Belady eviction ===
   *
   * free_pool : sorted list of free color indices
   * allocated : tag -> color currently held (for non-spilled tags)
   * spilled   : tag -> slot (committed spill assignments)
   *
   * When we hit overflow, we pick the live tag with the largest
   * next_use as the eviction victim. Its slot is a fresh
   * regalloc_spill[N] entry. *)
  let free_pool = ref (List.init budget (fun c -> c)) in
  let allocated : (int, int) Hashtbl.t = Hashtbl.create budget in
  let result : (int, assignment) Hashtbl.t = Hashtbl.create n in
  let spilled : (int, int) Hashtbl.t = Hashtbl.create 16 in
  let reload_sites : (int, reload_decl list) Hashtbl.t = Hashtbl.create 16 in
  let spill_sites : (int, (int * int) list) Hashtbl.t = Hashtbl.create 16 in
  let name_overrides : (int * int, string) Hashtbl.t = Hashtbl.create 16 in
  let next_spill_slot = ref 0 in
  let reload_counter : (int, int) Hashtbl.t = Hashtbl.create 16 in

  (* For Belady: at the current position p, return the live tag (one
   * in `allocated`) with the LARGEST next_use_after p.
   *
   * Excludes sentinels (negative tags). Sentinels represent reload
   * scratch registers needed at the current instruction; they can't
   * be spilled (no value to save). Real tags with future uses are
   * the only legitimate eviction candidates. *)
  let pick_belady_victim p =
    let best = ref None in
    Hashtbl.iter (fun tag _color ->
      if tag >= 0 then begin
        let nu = match next_use_after tag p with
          | Some k -> k
          | None -> max_int    (* no more uses → ideal victim *)
        in
        match !best with
        | None -> best := Some (tag, nu)
        | Some (_, bnu) when nu > bnu -> best := Some (tag, nu)
        | _ -> ()
      end
    ) allocated;
    !best
  in

  (* Release tags whose last (i.e., largest) use was < current pos. *)
  let release_dead i =
    let to_free = Hashtbl.fold (fun tag color acc ->
      match next_use_after tag (i - 1) with
      | None -> (tag, color) :: acc   (* no use at i or later → dead *)
      | _ -> acc
    ) allocated [] in
    List.iter (fun (tag, color) ->
      Hashtbl.remove allocated tag;
      free_pool := List.sort compare (color :: !free_pool)
    ) to_free
  in

  let alloc_color () : int option =
    match !free_pool with
    | [] -> None
    | c :: rest -> free_pool := rest; Some c
  in

  (* Spill a tag at the CURRENT POSITION. Picks a fresh slot, removes
   * the tag from `allocated`, records a spill_site at position `pos`
   * (which tells emit_c to emit `storeu_pd(&regalloc_spill[slot], tT)`
   * AFTER emitting the node at pos). Returns the freed color.
   *
   * CRITICAL: result[victim_tag] is NOT changed to Spilled. The tag's
   * assignment remains Reg (the register it was allocated to at its
   * def). emit_c uses that Reg name in the def and in the spill store.
   * The spill is a separate post-node emission step.
   *
   * Subsequent uses of victim_tag are handled by reload sites — at
   * each later use position, a reload decl loads from regalloc_spill
   * into a fresh tT_rK variable, and name_overrides substitutes the
   * reload name in the RHS. *)
  let do_spill (victim_tag : int) (pos : int) : int =
    let color = Hashtbl.find allocated victim_tag in
    Hashtbl.remove allocated victim_tag;
    let slot = !next_spill_slot in
    incr next_spill_slot;
    Hashtbl.add spilled victim_tag slot;
    (* DON'T touch result[victim_tag] — it stays as the original Reg. *)
    let prev = try Hashtbl.find spill_sites pos with Not_found -> [] in
    Hashtbl.replace spill_sites pos ((victim_tag, slot) :: prev);
    color
  in

  (* Record a reload for a spilled tag at position p.
   *
   * The reload variable's lifetime is the SAME POSITION p (alive
   * during the instruction at p, dead after). We model this by
   * inserting a sentinel "tag" (negative integer) into `allocated`
   * with no entry in uses_sorted. release_dead at position p+1 will
   * find next_use_after returns None for the sentinel and free it.
   *
   * Without this lifetime tracking, multiple reloads at the same
   * position would all pull color 0 (or whatever was freed), all
   * mapping to the same register and clobbering each other. *)
  let next_reload_sentinel = ref (-1) in
  (* Try to register a reload for tag at position pos. Returns Some name
   * if successful (a register was available, or a non-sentinel victim
   * could be spilled). Returns None if no victim was available —
   * caller must handle the fallback (e.g., emit_store can emit an
   * inline load-from-memory).
   *
   * This is the "may-fail" variant used by Step 3 and the post-iter
   * forced-reload handler, where failure is recoverable. *)
  let try_reload (tag : int) (pos : int) : string option =
    let slot = Hashtbl.find spilled tag in
    let color_opt = match alloc_color () with
      | Some c -> Some c
      | None ->
        (match pick_belady_victim pos with
         | None -> None
         | Some (victim_tag, _) ->
           if victim_tag < 0 then None    (* only sentinels available *)
           else Some (do_spill victim_tag pos))
    in
    match color_opt with
    | None -> None
    | Some color ->
      let counter = try Hashtbl.find reload_counter tag with Not_found -> 0 in
      Hashtbl.replace reload_counter tag (counter + 1);
      let name = Printf.sprintf "t%d_r%d" tag counter in
      let reg = reg_name_of_isa isa color in
      let decl = { reload_tag = tag; reload_name = name;
                   reload_reg = reg; reload_slot = slot } in
      let prev = try Hashtbl.find reload_sites pos with Not_found -> [] in
      Hashtbl.replace reload_sites pos (decl :: prev);
      Hashtbl.replace name_overrides (pos, tag) name;
      let sentinel = !next_reload_sentinel in
      decr next_reload_sentinel;
      Hashtbl.replace allocated sentinel color;
      Some name
  in
  (* The "must-succeed" variant — used by Step 1 (reloads for IR-level
   * preds, which MUST have a register since the node's RHS references
   * the reload variable name). If this fails, it's a bug or a
   * fundamental register-pressure issue we can't recover from. *)
  let do_reload (tag : int) (pos : int) : string =
    match try_reload tag pos with
    | Some name -> name
    | None ->
      failwith (Printf.sprintf
        "regalloc: reload at pos %d needs reg but no victim" pos)
  in

  let do_alloc tag color =
    Hashtbl.replace allocated tag color;
    Hashtbl.replace result tag (Reg (reg_name_of_isa isa color))
  in

  (* Build a reverse index: forced_at[p] = list of tags whose
   * force_last_use position is p. Used in the main iter to emit
   * reloads at the position where an output store will reference
   * the tag (output stores aren't IR-level uses, so they're not
   * captured by the regular walk_uses_of mechanism). *)
  let forced_at : (int, int list) Hashtbl.t = Hashtbl.create 16 in
  (match force_last_use with
   | None -> ()
   | Some tbl ->
     Hashtbl.iter (fun tag pos ->
       let prev = try Hashtbl.find forced_at pos with Not_found -> [] in
       Hashtbl.replace forced_at pos (tag :: prev)
     ) tbl);

  (* Walk the schedule. *)
  List.iteri (fun i (e : Algsimp.t) ->
    release_dead i;
    (* Step 1: emit reloads for all spilled preds used at position i.
     *
     * This is a fixed-point loop: do_reload may cascade-spill another
     * pred (Belady picks a victim from `allocated`, which may include
     * a pred we haven't visited yet). After such a cascade, the
     * newly-spilled pred ALSO needs a reload — but our list iteration
     * may have passed it already. Loop until no further changes.
     *
     * Collect the full set of pred tags (transitively through inlined
     * preds) up front, then loop checking each. The transitive walk
     * is itself deterministic (same on every iteration). *)
    let pred_tags = ref [] in
    let rec collect_preds_of (p : Algsimp.t) =
      pred_tags := p.tag :: !pred_tags;
      if is_inlined p.tag then
        List.iter collect_preds_of (Algsimp.preds p)
    in
    List.iter collect_preds_of (Algsimp.preds e);
    let pred_tags_list = !pred_tags in
    let changed = ref true in
    while !changed do
      changed := false;
      List.iter (fun tag ->
        if Hashtbl.mem spilled tag
           && not (Hashtbl.mem name_overrides (i, tag)) then begin
          let _ = do_reload tag i in ();
          changed := true
        end
      ) pred_tags_list
    done;
    (* Step 1b: also emit reloads for any spilled tags whose
     * force_last_use position is i. These are output tags whose
     * cluster-flush position is i — the output store at i needs
     * a reload variable. Uses try_reload because failure here is
     * recoverable: emit_store falls back to inline load-from-memory. *)
    (match Hashtbl.find_opt forced_at i with
     | Some tags ->
       List.iter (fun t ->
         if Hashtbl.mem spilled t then
           let _ = try_reload t i in ()
       ) tags
     | None -> ());
    (* Step 2: allocate a register for the current node's def
     * (unless inlined/skipped). *)
    if not (is_skipped e.tag) && not (is_inlined e.tag) then begin
      match alloc_color () with
      | Some c -> do_alloc e.tag c
      | None ->
        (* Belady eviction: spill the tag with farthest next_use, then
         * reuse its color for the current node. *)
        match pick_belady_victim i with
        | None ->
          (* No live tags to evict — pathological, shouldn't happen if
           * peak_live >= 1. Bail. *)
          failwith (Printf.sprintf
            "regalloc: no eviction candidate at position %d, tag %d" i e.tag)
        | Some (victim_tag, _) ->
          let freed_color = do_spill victim_tag i in
          do_alloc e.tag freed_color
    end;
    (* Step 3: re-check forced_at[i] until no new reloads. Uses
     * try_reload — if registration fails (no register and no
     * non-sentinel victim available), emit_store will fall back to
     * inline load-from-memory for that tag. The fixed-point loop
     * still terminates because we only count a "change" when a
     * reload was actually registered (name_overrides got an entry). *)
    (match Hashtbl.find_opt forced_at i with
     | Some tags ->
       let changed = ref true in
       while !changed do
         changed := false;
         List.iter (fun t ->
           if Hashtbl.mem spilled t
              && not (Hashtbl.mem name_overrides (i, t)) then
             match try_reload t i with
             | Some _ -> changed := true
             | None -> ()    (* fallback: emit_store does inline load *)
         ) tags
       done
     | None -> ())
  ) scheduled;

  (* Post-iter: handle force_last_use positions BEYOND the schedule
   * (i.e., position == n). These are tags whose output stores happen
   * at end-of-pass via the safety-net loop or final cluster flush.
   * emit_c sets current_emit_position = n before emitting these.
   * Fixed-point loop for the same reason as Step 3 in the iter. *)
  (match Hashtbl.find_opt forced_at n with
   | Some tags ->
     let changed = ref true in
     while !changed do
       changed := false;
       List.iter (fun t ->
         if Hashtbl.mem spilled t
            && not (Hashtbl.mem name_overrides (n, t)) then
           match try_reload t n with
           | Some _ -> changed := true
           | None -> ()
       ) tags
     done
   | None -> ());

  Allocated { isa;
              assign = result;
              num_spill_slots = !next_spill_slot;
              reload_sites;
              spill_sites;
              spilled_of_tag = spilled;
              name_overrides;
            }

(* === Top-level allocate wrapper (M3a + M5 fallback) ===
 *
 * Tries M3a's linear scan first. If it overflows AND
 * VFFT_USE_REGALLOC_M5=1, falls back to M5's spilling allocator.
 * Otherwise propagates the overflow upward so emit_c can fall back
 * to default (gcc-allocated) behavior. *)
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
    | None   -> isa.vec_regs - 4
  in
  let m5_enabled =
    try Sys.getenv "VFFT_USE_REGALLOC_M5" = "1" with Not_found -> false
  in
  match allocate_linear_scan ~isa ~scheduled ~budget:b
          ~skip_tags ~inline_set ~force_last_use () with
  | Allocated _ as a -> a
  | Overflow _ when m5_enabled ->
    allocate_with_spilling ~isa ~scheduled ~budget:b
      ~skip_tags ~inline_set ~force_last_use ()
  | Overflow _ as o -> o